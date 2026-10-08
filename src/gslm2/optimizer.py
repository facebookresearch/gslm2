"""Optimization utilities."""

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, LRScheduler, SequentialLR

from gslm2.config import OptimizerConfig


class CyclicLR(LRScheduler):
    def __init__(self, opt: Optimizer, cfg: OptimizerConfig, task_interval: int, last_epoch: int = -1) -> None:
        self.opt = opt
        self.cfg = cfg
        assert self.cfg.within_cycle_warmup_steps, "within_cycle_warmup_steps is required for CyclicLR"
        self.task_interval = task_interval
        self.init_lr_scale = cfg.init_lr_scale if cfg.warmup_steps > 0 else 1
        super().__init__(opt, last_epoch)

    def upper_envelope(self, envelope_index: int, upper_envelope_shape: str) -> list[float]:
        if upper_envelope_shape == "tristage":
            end_of_hold_steps = self.cfg.hold_steps + self.cfg.warmup_steps
            if self.last_epoch <= self.cfg.warmup_steps:
                start_factor = self.init_lr_scale
                end_factor = 1.0
                total_iters = self.cfg.warmup_steps
                envelope_value = self.cfg.lr * (
                    start_factor + (end_factor - start_factor) * envelope_index / total_iters
                )
            elif self.last_epoch <= end_of_hold_steps:
                envelope_value = self.cfg.lr
            else:
                envelope_value = self.cfg.lr * math.exp(
                    math.log(self.cfg.final_lr_scale) * (envelope_index - end_of_hold_steps) / self.cfg.decay_steps
                )
        elif upper_envelope_shape == "warmupconstant":
            if self.last_epoch <= self.cfg.warmup_steps:
                start_factor = self.init_lr_scale
                end_factor = 1.0
                total_iters = self.cfg.warmup_steps
                envelope_value = self.cfg.lr * (
                    start_factor + (end_factor - start_factor) * envelope_index / total_iters
                )
            else:
                envelope_value = self.cfg.lr
        elif upper_envelope_shape == "constant":
            envelope_value = self.cfg.lr
        return envelope_value

    def get_lr(self) -> list[float]:
        envelope_index = (self.last_epoch // self.task_interval + 1) * self.task_interval
        max_lr = self.upper_envelope(envelope_index, self.cfg.upper_envelope_shape)
        min_lr = max_lr * self.init_lr_scale
        if self.last_epoch % self.task_interval < self.cfg.within_cycle_warmup_steps:  # Warmup phase
            return [
                min_lr
                + (max_lr - min_lr) * (self.last_epoch % self.task_interval) / self.cfg.within_cycle_warmup_steps
                for _ in self.opt.param_groups
            ]
        return [max_lr for _ in self.opt.param_groups]


class CyclicDualLR(CyclicLR):
    def __init__(
        self,
        opt: Optimizer,
        cfg: OptimizerConfig,
        task_interval: int,
        inner_step: int,
        last_epoch: int = -1,
    ) -> None:
        self.inner_step = inner_step
        super().__init__(opt, cfg, task_interval, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch % self.task_interval < self.inner_step:
            envelope_index = (self.last_epoch // self.task_interval + 1) * self.task_interval
            max_lr_ssl = self.upper_envelope(envelope_index, self.cfg.upper_envelope_shape)
            min_lr_ssl = max_lr_ssl * self.init_lr_scale
            if self.last_epoch % self.task_interval < self.cfg.within_cycle_warmup_steps:  # Warmup phase
                return [
                    min_lr_ssl
                    + (max_lr_ssl - min_lr_ssl)
                    * (self.last_epoch % self.task_interval)
                    / self.cfg.within_cycle_warmup_steps
                    for _ in self.opt.param_groups
                ]
            return [max_lr_ssl for _ in self.opt.param_groups]
        lr_sl = self.upper_envelope(self.last_epoch, "tristage")
        return [lr_sl for _ in self.opt.param_groups]


def build_scheduler(  # noqa: PLR0911 -- one return per supported scheduler kind (flat dispatch)
    opt: Optimizer,
    cfg: OptimizerConfig,
    task_interval: int | None,
    inner_step: int | None,
) -> LRScheduler:
    init_lr_scale = cfg.init_lr_scale if cfg.warmup_steps > 0 else 1
    decay: LRScheduler
    if cfg.scheduler == "tristage":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: math.exp(math.log(cfg.final_lr_scale) * step / cfg.decay_steps))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "cosine":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        decay = CosineAnnealingLR(opt, cfg.max_steps - cfg.warmup_steps, cfg.final_lr_scale * cfg.lr)
        return SequentialLR(opt, [warmup, decay], [cfg.warmup_steps])
    if cfg.scheduler == "linear":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        decay = LinearLR(opt, start_factor=1.0, end_factor=cfg.final_lr_scale, total_iters=cfg.decay_steps)
        return SequentialLR(opt, [warmup, decay], [cfg.warmup_steps])
    if cfg.scheduler == "rsqrt":
        warmup = LinearLR(
            opt,
            start_factor=init_lr_scale,
            end_factor=1 / math.sqrt(1 + cfg.rsqrt_shift / cfg.rsqrt_timescale),
            total_iters=cfg.warmup_steps,
        )
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: 1 / math.sqrt(1 + (step + cfg.rsqrt_shift) / cfg.rsqrt_timescale))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "constant":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.max_steps)
        return SequentialLR(opt, [warmup, hold], [cfg.warmup_steps])
    if cfg.scheduler == "cyclic":
        assert task_interval, "Task interval is required for CyclicLR scheduler"
        return CyclicLR(opt, cfg, task_interval)
    if cfg.scheduler == "cyclic_dual_loss":
        assert task_interval, "Task interval is required for CyclicLR scheduler"
        assert inner_step, "FOBLO with inner_step is required for CyclicDualLR scheduler"
        return CyclicDualLR(opt, cfg, task_interval, inner_step)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def build_optimizer(
    model: torch.nn.Module,
    cfg: OptimizerConfig,
    task_interval: int | None = None,
    inner_step: int | None = None,
) -> tuple[AdamW, LRScheduler]:
    if cfg.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(cfg.dtype)
    if cfg.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        raise ValueError("Cannot use bfloat16 on this GPU (V100?), try again with float16")

    # Split params into trainable vs to-freeze. When cfg.no_decay_on_norm_bias is
    # set, biases and 1-D params (LayerNorm) are put in weight_decay=0 groups
    # (V-JEPA / modern LM recipe); otherwise all params share cfg.weight_decay.
    # Groups holding to-freeze params are tagged ``is_frozen`` so they can be
    # dropped wholesale at freeze_step regardless of the weight-decay split.
    trainable_wd: list = []
    trainable_no_wd: list = []
    frozen_wd: list = []
    frozen_no_wd: list = []
    for name, param in model.named_parameters():
        if any(name.startswith(exclude) for exclude in cfg.exclude_from_optimizer):
            continue
        is_frozen = any(name.startswith(freeze) for freeze in cfg.to_freeze)
        no_wd = cfg.no_decay_on_norm_bias and (("bias" in name) or param.ndim == 1)
        if is_frozen:
            (frozen_no_wd if no_wd else frozen_wd).append(param)
        else:
            (trainable_no_wd if no_wd else trainable_wd).append(param)

    param_groups: list[dict] = [{"params": trainable_wd}]
    if trainable_no_wd:
        param_groups.append({"params": trainable_no_wd, "weight_decay": 0.0})
    if frozen_wd:
        param_groups.append({"params": frozen_wd, "is_frozen": True})
    if frozen_no_wd:
        param_groups.append({"params": frozen_no_wd, "weight_decay": 0.0, "is_frozen": True})

    optimizer = AdamW(
        param_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=torch.cuda.is_available(),
    )
    scheduler = build_scheduler(optimizer, cfg, task_interval, inner_step)
    return optimizer, scheduler
