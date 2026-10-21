"""Distributed adversarial training for the supported unit HiFi-GAN models."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from gslm2.vocoder.data import AudioDataLoader, AudioItem, build_dataloader
from gslm2.vocoder.loss import discriminator_loss, generator_loss
from gslm2.vocoder.model import UnitDiscriminator, UnitVocoder
from gslm2.vocoder.utils import (
    AverageMeters,
    MetricsLogger,
    init_distributed,
    latest_checkpoint,
    set_pytorch_flags,
    set_seed,
    training_dtype,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from gslm2.vocoder.config import UnitVocoderConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Mutable loop counters restored from training checkpoints."""

    step: int = 1
    epoch: int = 1


@dataclass
class TrainingComponents:
    """Models, optimizers, schedulers, scaler, and running metrics."""

    vocoder: UnitVocoder
    discriminator: UnitDiscriminator
    optimizer_g: AdamW
    optimizer_d: AdamW
    scheduler_g: ExponentialLR
    scheduler_d: ExponentialLR
    scaler: torch.GradScaler
    meters: AverageMeters


@dataclass
class TrainingRuntime:
    """Objects shared by training, validation, logging, and checkpointing."""

    cfg: UnitVocoderConfig
    device: torch.device
    dtype: torch.dtype
    is_main: bool
    components: TrainingComponents
    state: TrainingState
    train_loader: AudioDataLoader
    train_sampler: StatefulDistributedSampler[AudioItem]
    val_loader: AudioDataLoader
    vocoder_ddp: DistributedDataParallel
    discriminator_ddp: DistributedDataParallel
    metrics: MetricsLogger

    @property
    def with_autocast(self) -> bool:
        """Return whether mixed-precision autocast is enabled."""
        return self.dtype != torch.float32


def _build_components(cfg: UnitVocoderConfig, device: torch.device) -> TrainingComponents:
    vocoder = UnitVocoder(cfg.n_units, agg_type=cfg.agg_type).train().to(device)
    discriminator = UnitDiscriminator().train().to(device)
    optimizer_g = AdamW(vocoder.parameters(), lr=cfg.lr, betas=cfg.betas, fused=True)
    optimizer_d = AdamW(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas, fused=True)
    return TrainingComponents(
        vocoder=vocoder,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=ExponentialLR(optimizer_g, cfg.gamma),
        scheduler_d=ExponentialLR(optimizer_d, cfg.gamma),
        scaler=torch.GradScaler(enabled=cfg.dtype == "float16"),
        meters=AverageMeters(),
    )


def _checkpoint_payload(runtime: TrainingRuntime) -> dict[str, object]:
    components = runtime.components
    return {
        "vocoder": components.vocoder.state_dict(),
        "discriminator": components.discriminator.state_dict(),
        "optim_g": components.optimizer_g.state_dict(),
        "optim_d": components.optimizer_d.state_dict(),
        "scheduler_g": components.scheduler_g.state_dict(),
        "scheduler_d": components.scheduler_d.state_dict(),
        "loader": runtime.train_loader.state_dict(),
        "scaler": components.scaler.state_dict(),
        "step": torch.tensor(runtime.state.step),
        "epoch": torch.tensor(runtime.state.epoch),
        "meters": components.meters.state_dict(),
    }


def _save_checkpoint(runtime: TrainingRuntime) -> None:
    if not runtime.is_main:
        return
    step = runtime.state.step
    torch.save(_checkpoint_payload(runtime), runtime.cfg.dir / f"checkpoint-{step}.pt")
    runtime.components.vocoder.save_pretrained(runtime.cfg.dir / f"vocoder-{step}")


def _load_mapping(checkpoint: dict[str, object], name: str) -> dict[str, Any]:
    value = checkpoint.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint field {name!r} must be a mapping")
    return cast("dict[str, Any]", value)


def _restore_checkpoint(
    components: TrainingComponents,
    train_loader: AudioDataLoader,
    checkpoint_path: Path,
    device: torch.device,
) -> TrainingState:
    checkpoint = cast(
        "dict[str, object]",
        torch.load(checkpoint_path, map_location=device, weights_only=True),
    )
    components.vocoder.load_state_dict(_load_mapping(checkpoint, "vocoder"))
    components.discriminator.load_state_dict(_load_mapping(checkpoint, "discriminator"))
    components.optimizer_g.load_state_dict(_load_mapping(checkpoint, "optim_g"))
    components.optimizer_d.load_state_dict(_load_mapping(checkpoint, "optim_d"))
    components.scheduler_g.load_state_dict(_load_mapping(checkpoint, "scheduler_g"))
    components.scheduler_d.load_state_dict(_load_mapping(checkpoint, "scheduler_d"))
    loader_state = checkpoint.get("loader")
    if loader_state is None:
        logger.warning("Checkpoint %s has no loader state; resuming from the start of its epoch", checkpoint_path)
    elif isinstance(loader_state, dict):
        train_loader.load_state_dict(loader_state)
    else:
        raise TypeError("checkpoint field 'loader' must be a mapping")
    components.scaler.load_state_dict(_load_mapping(checkpoint, "scaler"))
    components.meters.load_state_dict(cast("dict[str, dict[str, Tensor]]", _load_mapping(checkpoint, "meters")))
    step = checkpoint.get("step")
    epoch = checkpoint.get("epoch")
    if not isinstance(step, Tensor) or not isinstance(epoch, Tensor):
        raise TypeError("checkpoint step and epoch must be tensors")
    return TrainingState(step=int(step.item()), epoch=int(epoch.item()))


def _train_batch(runtime: TrainingRuntime, batch: AudioItem) -> dict[str, Tensor]:
    components = runtime.components
    batch = batch.to(runtime.device)
    components.optimizer_d.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=runtime.dtype, enabled=runtime.with_autocast):
        generated_audio = runtime.vocoder_ddp(batch.units)
        real_output = runtime.discriminator_ddp(batch.audio)
        generated_detached_output = runtime.discriminator_ddp(generated_audio.detach())
        loss_d, _ = discriminator_loss(real_output, generated_detached_output)
    components.scaler.scale(loss_d).backward()
    components.scaler.step(components.optimizer_d)

    components.optimizer_g.zero_grad(set_to_none=True)
    with runtime.discriminator_ddp.no_sync():
        with torch.autocast("cuda", dtype=runtime.dtype, enabled=runtime.with_autocast):
            with torch.no_grad():
                real_output = runtime.discriminator_ddp(batch.audio)
            generated_output = runtime.discriminator_ddp(generated_audio)
            loss_g, generator_parts = generator_loss(real_output, generated_output)
        components.scaler.scale(loss_g).backward()
    components.scaler.step(components.optimizer_g)
    components.scaler.update()
    return {"loss_mel": generator_parts["mel"], "loss_g": loss_g, "loss_d": loss_d}


@torch.no_grad()
def _validate(runtime: TrainingRuntime) -> dict[str, float]:
    runtime.vocoder_ddp.eval()
    runtime.discriminator_ddp.eval()
    totals = torch.zeros(4, device=runtime.device)
    for batch_item in runtime.val_loader:
        batch = batch_item.to(runtime.device)
        with torch.autocast("cuda", dtype=runtime.dtype, enabled=runtime.with_autocast):
            real = runtime.discriminator_ddp(batch.audio)
            generated = runtime.discriminator_ddp(runtime.vocoder_ddp(batch.units))
            loss_g, generator_parts = generator_loss(real, generated)
            loss_d = discriminator_loss(real, generated)[0]
        totals += torch.stack([generator_parts["mel"], loss_g, loss_d, totals.new_ones(())])
    dist.all_reduce(totals)
    runtime.vocoder_ddp.train()
    runtime.discriminator_ddp.train()
    mel, generator, discriminator, batches = totals.tolist()
    count = max(batches, 1.0)
    return {
        "val/loss_mel": mel / count,
        "val/loss_g": generator / count,
        "val/loss_d": discriminator / count,
    }


def _log_training(runtime: TrainingRuntime, progress: tqdm[Any]) -> None:
    metrics: dict[str, object] = {
        "step": runtime.state.step,
        "epoch": runtime.state.epoch,
        "train/lr": runtime.components.scheduler_d.get_last_lr()[0],
    }
    metrics.update({f"train/{name}": value for name, value in runtime.components.meters.pop().items()})
    runtime.metrics.log(metrics)
    progress.set_postfix(
        loss_mel=metrics["train/loss_mel"],
        loss_g=metrics["train/loss_g"],
        loss_d=metrics["train/loss_d"],
        epoch=runtime.state.epoch,
    )


def _run_training(runtime: TrainingRuntime) -> None:
    cfg = runtime.cfg
    with tqdm(
        initial=runtime.state.step,
        total=cfg.max_steps,
        disable=not runtime.is_main,
    ) as progress:
        while runtime.state.step < cfg.max_steps and runtime.state.epoch < cfg.max_epochs:
            runtime.train_sampler.set_epoch(runtime.state.epoch)
            for batch in runtime.train_loader:
                if runtime.state.step >= cfg.max_steps:
                    break
                losses = _train_batch(runtime, batch)
                runtime.components.meters.update(**losses)
                progress.update()

                if runtime.state.step % cfg.log_interval == 0:
                    _log_training(runtime, progress)
                if runtime.state.step % cfg.val_interval == 0:
                    _save_checkpoint(runtime)
                    dist.barrier()
                    runtime.metrics.log(
                        {"step": runtime.state.step, "epoch": runtime.state.epoch, **_validate(runtime)}
                    )
                runtime.state.step += 1
            runtime.components.scheduler_d.step()
            runtime.components.scheduler_g.step()
            runtime.state.epoch += 1


def train(cfg: UnitVocoderConfig) -> None:
    """Run distributed CUDA adversarial training described by ``cfg``."""
    cfg.validate_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("unit HiFi-GAN training requires CUDA")
    set_seed(cfg.seed)
    set_pytorch_flags()
    device = init_distributed()
    try:
        cfg.dir.mkdir(parents=True, exist_ok=True)
        is_main = dist.get_rank() == 0
        if is_main:
            (cfg.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=4), encoding="utf-8")

        train_loader, train_sampler = build_dataloader(
            cfg.train_manifest,
            cfg,
        )
        val_loader, _ = build_dataloader(
            cfg.val_manifest,
            cfg,
            is_train=False,
        )
        if train_sampler is None:
            raise RuntimeError("distributed sampler was not initialized")

        components = _build_components(cfg, device)
        checkpoint_path = latest_checkpoint(cfg.dir)
        state = (
            _restore_checkpoint(components, train_loader, checkpoint_path, device)
            if checkpoint_path is not None
            else TrainingState()
        )
        if checkpoint_path is not None and is_main:
            logger.info("Resuming from %s at step %d, epoch %d", checkpoint_path, state.step, state.epoch)
        if is_main:
            logger.info(
                "Vocoder parameters: %d", sum(parameter.numel() for parameter in components.vocoder.parameters())
            )
            logger.info(
                "Discriminator parameters: %d",
                sum(parameter.numel() for parameter in components.discriminator.parameters()),
            )
            logger.info(
                "Steps per epoch: %d (batch_size %d x %d ranks)",
                len(train_loader),
                cfg.batch_size,
                dist.get_world_size(),
            )

        vocoder_ddp = DistributedDataParallel(components.vocoder, device_ids=[device.index])
        discriminator_ddp = DistributedDataParallel(components.discriminator, device_ids=[device.index])
        with MetricsLogger(cfg.dir / f"metrics-{dist.get_rank()}.jsonl") as metrics:
            runtime = TrainingRuntime(
                cfg=cfg,
                device=device,
                dtype=training_dtype(cfg.dtype, device),
                is_main=is_main,
                components=components,
                state=state,
                train_loader=train_loader,
                train_sampler=train_sampler,
                val_loader=val_loader,
                vocoder_ddp=vocoder_ddp,
                discriminator_ddp=discriminator_ddp,
                metrics=metrics,
            )
            try:
                _run_training(runtime)
            finally:
                _save_checkpoint(runtime)
    finally:
        dist.destroy_process_group()
