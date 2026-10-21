"""Training and validation loops for duration predictors."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from gslm2.duration.dataset import DurationDataset, collate_fn
from gslm2.duration.model import (
    DurationPredictor,
    DurationPredictorModelConfig,
    decode_duration_predictions,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TextIO

    from gslm2.duration.config import DurationLossConfig, DurationPredictorConfig
    from gslm2.duration.dataset import DurationSample

logger = logging.getLogger(__name__)


def compute_loss(
    prediction: Tensor,
    target: Tensor,
    lengths: Tensor,
    config: DurationLossConfig,
) -> Tensor:
    """Compute a padding-masked classification or regression duration loss."""
    mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) < lengths.unsqueeze(1)

    if config.num_classes > 0:
        class_target = target.long().clamp(min=1, max=config.num_classes) - 1
        losses = F.cross_entropy(
            prediction.reshape(-1, config.num_classes),
            class_target.reshape(-1),
            reduction="none",
        ).reshape(target.shape)
        return (losses * mask).sum() / mask.sum()

    target_frames = target
    transformed_target = (target + 1).log() if config.log_duration else target
    if config.loss_type in {"mse", "mse_totlen"}:
        losses = F.mse_loss(prediction, transformed_target, reduction="none")
    elif config.loss_type == "huber":
        losses = F.smooth_l1_loss(prediction, transformed_target, reduction="none", beta=config.huber_beta)
    elif config.loss_type == "l1":
        losses = F.l1_loss(prediction, transformed_target, reduction="none")
    else:
        raise ValueError(f"unknown regression loss_type: {config.loss_type}")

    base_loss = (losses * mask).sum() / mask.sum()
    if config.loss_type != "mse_totlen" or config.totlen_weight == 0.0:
        return base_loss

    predicted_frames = prediction.exp() - 1 if config.log_duration else prediction
    predicted_totals = (predicted_frames.clamp(min=0) * mask).sum(dim=1)
    target_totals = (target_frames * mask).sum(dim=1)
    return base_loss + config.totlen_weight * (predicted_totals - target_totals).abs().mean()


@dataclass(frozen=True)
class ValMetrics:
    """Aggregate validation metrics in frame-duration space."""

    loss: float
    mae: float
    length_error: float
    length_ratio: float
    correlation: float

    def log_str(self) -> str:
        """Format metrics for a compact training log line."""
        return (
            f"val_loss={self.loss:.4f} val_mae={self.mae:.2f} "
            f"val_len_err={self.length_error:.4f} val_len_ratio={self.length_ratio:.4f} "
            f"val_corr={self.correlation:.4f}"
        )


def validate(
    model: DurationPredictor,
    loader: DataLoader[DurationSample],
    device: torch.device,
    loss_config: DurationLossConfig,
) -> ValMetrics:
    """Evaluate loss, duration MAE, total-length error, and correlation."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_frames = 0
    total_predicted_duration = 0.0
    total_target_duration = 0.0
    length_ratio_sum = 0.0
    num_sequences = 0
    correlation_sums = torch.zeros(5, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch_tokens, batch_durations, batch_lengths, batch_raw_duration_totals in loader:
            tokens = batch_tokens.to(device)
            durations = batch_durations.to(device)
            lengths = batch_lengths.to(device)
            raw_duration_totals = batch_raw_duration_totals.to(device)
            prediction = model(tokens, lengths)
            mask = torch.arange(durations.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            frame_count = int(mask.sum())
            loss = compute_loss(
                prediction,
                durations,
                lengths,
                loss_config,
            )
            total_loss += loss.item() * frame_count

            predicted_durations = decode_duration_predictions(
                prediction,
                log_duration=loss_config.log_duration,
                num_classes=loss_config.num_classes,
            )
            valid_prediction = predicted_durations[mask].double()
            valid_target = durations[mask].double()
            total_absolute_error += (valid_prediction - valid_target).abs().sum().item()
            total_frames += frame_count
            correlation_sums += torch.stack(
                [
                    valid_prediction.sum(),
                    valid_target.sum(),
                    valid_prediction.square().sum(),
                    valid_target.square().sum(),
                    (valid_prediction * valid_target).sum(),
                ]
            )

            for batch_idx, length in enumerate(lengths.tolist()):
                predicted_total = predicted_durations[batch_idx, :length].sum().item()
                target_total = raw_duration_totals[batch_idx].item()
                total_predicted_duration += predicted_total
                total_target_duration += target_total
                length_ratio_sum += predicted_total / target_total
                num_sequences += 1

    model.train(was_training)
    if total_frames == 0:
        raise ValueError("validation loader contains no duration frames")

    sum_prediction, sum_target, sum_prediction_sq, sum_target_sq, sum_product = correlation_sums.tolist()
    numerator = total_frames * sum_product - sum_prediction * sum_target
    denominator = math.sqrt(
        max(total_frames * sum_prediction_sq - sum_prediction**2, 0.0)
        * max(total_frames * sum_target_sq - sum_target**2, 0.0)
    )
    correlation = numerator / denominator if denominator else 0.0
    return ValMetrics(
        loss=total_loss / total_frames,
        mae=total_absolute_error / total_frames,
        length_error=abs(total_predicted_duration - total_target_duration) / total_target_duration,
        length_ratio=length_ratio_sum / num_sequences,
        correlation=correlation,
    )


def _make_loader(
    dataset: DurationDataset, cfg: DurationPredictorConfig, *, shuffle: bool
) -> DataLoader[DurationSample]:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _save_checkpoint(model: DurationPredictor, cfg: DurationPredictorConfig, step: int, path: Path) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, path)


def _update_link(link: Path, target_name: str) -> None:
    link.unlink(missing_ok=True)
    link.symlink_to(target_name)


def _validation_callback(
    model: DurationPredictor,
    loader: DataLoader[DurationSample],
    device: torch.device,
    cfg: DurationPredictorConfig,
    log_json: Callable[[dict[str, object]], None],
) -> Callable[[int, float], float]:
    def run(step: int, best_loss: float) -> float:
        metrics = validate(
            model,
            loader,
            device,
            cfg.loss_config(),
        )
        logger.info("step=%d %s", step, metrics.log_str())
        log_json({"type": "val", "step": step, **asdict(metrics)})
        if metrics.loss >= best_loss:
            return best_loss

        checkpoint_name = f"step_{step}.pt"
        _save_checkpoint(model, cfg, step, cfg.dir / checkpoint_name)
        _update_link(cfg.dir / "best.pt", checkpoint_name)
        logger.info("Saved best model at step %d (%s)", step, metrics.log_str())
        return metrics.loss

    return run


def _train_steps(
    model: DurationPredictor,
    loaders: tuple[DataLoader[DurationSample], DataLoader[DurationSample]],
    cfg: DurationPredictorConfig,
    device: torch.device,
    log_file: TextIO,
) -> float:
    train_loader, valid_loader = loaders
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=cfg.warmup_steps),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_steps - cfg.warmup_steps),
        ],
        [cfg.warmup_steps],
    )

    def log_json(entry: dict[str, object]) -> None:
        print(json.dumps(entry), file=log_file, flush=True)

    run_validation = _validation_callback(model, valid_loader, device, cfg, log_json)
    progress = tqdm(total=cfg.max_steps, desc="Training", mininterval=30)
    step = 0
    best_validation_loss = float("inf")
    running_loss = 0.0
    running_count = 0
    try:
        while step < cfg.max_steps:
            for batch_tokens, batch_durations, batch_lengths, _ in train_loader:
                tokens = batch_tokens.to(device)
                durations = batch_durations.to(device)
                lengths = batch_lengths.to(device)
                prediction = model(tokens, lengths)
                loss = compute_loss(
                    prediction,
                    durations,
                    lengths,
                    cfg.loss_config(),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                running_count += 1
                step += 1
                progress.update()

                if step % cfg.log_interval == 0:
                    average_loss = running_loss / running_count
                    learning_rate = scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=f"{average_loss:.4f}", lr=f"{learning_rate:.2e}")
                    logger.info("step=%d loss=%.4f lr=%.2e", step, average_loss, learning_rate)
                    log_json({"type": "train", "step": step, "loss": average_loss, "lr": learning_rate})
                    running_loss = 0.0
                    running_count = 0

                if step % cfg.val_interval == 0:
                    best_validation_loss = run_validation(step, best_validation_loss)
                if step >= cfg.max_steps:
                    break
        if cfg.max_steps % cfg.val_interval:
            best_validation_loss = run_validation(cfg.max_steps, best_validation_loss)
    finally:
        progress.close()
    return best_validation_loss


def train_duration_predictor(cfg: DurationPredictorConfig) -> None:
    """Train a duration predictor and write checkpoints and JSONL metrics."""
    if not cfg.train_dirs or not cfg.valid_dirs:
        raise ValueError("train_dirs and valid_dirs must each contain at least one directory")
    if not cfg.output_dir:
        raise ValueError("output_dir must be configured")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.dir.mkdir(parents=True, exist_ok=True)
    (cfg.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    logger.info("Loading training data from %s", cfg.train_dirs)
    train_dataset = DurationDataset(
        cfg.train_dirs,
        cfg.max_seq_len,
        cfg.max_duration,
        cfg.train_file_pattern,
        cfg.num_streams,
    )
    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Loading validation data from %s", cfg.valid_dirs)
    valid_dataset = DurationDataset(
        cfg.valid_dirs,
        cfg.max_seq_len,
        cfg.max_duration,
        cfg.valid_file_pattern,
        cfg.num_streams,
    )
    logger.info("Validation samples: %d", len(valid_dataset))
    if not train_dataset or not valid_dataset:
        raise ValueError("training and validation inputs must each yield at least one duration sample")

    train_loader = _make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = _make_loader(valid_dataset, cfg, shuffle=False)
    model_config = DurationPredictorModelConfig(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_streams=cfg.num_streams,
        num_classes=cfg.num_classes,
        encoder_type=cfg.encoder_type,
        attention_heads=cfg.attention_heads,
        conv_kernel=cfg.conv_kernel,
        ffn_expansion=cfg.ffn_expansion,
        max_seq_len=max(cfg.max_seq_len, 4096),
    )
    model = DurationPredictor(model_config).to(device)
    model.train()
    logger.info("Model parameters: %d", sum(parameter.numel() for parameter in model.parameters()))

    with (cfg.dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        best_validation_loss = _train_steps(model, (train_loader, valid_loader), cfg, device, log_file)

    checkpoint_name = f"step_{cfg.max_steps}.pt"
    _save_checkpoint(model, cfg, cfg.max_steps, cfg.dir / checkpoint_name)
    _update_link(cfg.dir / "last.pt", checkpoint_name)
    logger.info("Training complete. Best val_loss=%.4f", best_validation_loss)
