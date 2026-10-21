"""Optimization utilities."""

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, LRScheduler, SequentialLR

from gslm2.encoder.config import OptimizerConfig


def drop_scheduler_base_lrs(scheduler: LRScheduler, num_groups: int) -> None:
    """Shrink per-group scheduler state to ``num_groups`` entries.

    Called after frozen optimizer param groups are dropped: the schedulers
    captured per-group state at construction (``base_lrs``, ``_last_lr``,
    and ``lr_lambdas`` on LambdaLR-style schedulers), which must stay
    aligned with the live groups. Frozen groups are always a suffix
    (see ``build_optimizer``), so truncation preserves alignment; already
    short state (e.g. resumed post-drop checkpoints) is left untouched.
    Recurses into ``SequentialLR`` sub-schedulers.
    """
    for attr in ("base_lrs", "_last_lr", "lr_lambdas"):
        values = getattr(scheduler, attr, None)
        if isinstance(values, list) and len(values) > num_groups:
            del values[num_groups:]
    for sub in getattr(scheduler, "_schedulers", []):
        drop_scheduler_base_lrs(sub, num_groups)


def build_scheduler(
    opt: Optimizer,
    cfg: OptimizerConfig,
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
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def build_optimizer(
    model: torch.nn.Module,
    cfg: OptimizerConfig,
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
    scheduler = build_scheduler(optimizer, cfg)
    return optimizer, scheduler
