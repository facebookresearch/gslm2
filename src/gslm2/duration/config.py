"""Configuration for duration-predictor training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EncoderType = Literal["conformer", "transformer"]
LossType = Literal["classification", "huber", "l1", "mse", "mse_totlen"]


@dataclass(frozen=True)
class DurationLossConfig:
    """Options controlling the duration training objective."""

    log_duration: bool
    loss_type: LossType
    totlen_weight: float
    huber_beta: float
    num_classes: int


def _validate_positive_values(config: DurationPredictorConfig) -> None:
    positive_values = {
        "vocab_size": config.vocab_size,
        "num_streams": config.num_streams,
        "max_duration": config.max_duration,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "attention_heads": config.attention_heads,
        "conv_kernel": config.conv_kernel,
        "ffn_expansion": config.ffn_expansion,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "log_interval": config.log_interval,
        "val_interval": config.val_interval,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")


def _validate_ranges(config: DurationPredictorConfig) -> None:
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.weight_decay < 0.0 or config.totlen_weight < 0.0:
        raise ValueError("weight_decay and totlen_weight must be non-negative")
    if config.huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 0 < config.warmup_steps < config.max_steps:
        raise ValueError("warmup_steps must be greater than zero and less than max_steps")


def _validate_model(config: DurationPredictorConfig) -> None:
    if config.encoder_type not in {"conformer", "transformer"}:
        raise ValueError(f"unknown encoder_type: {config.encoder_type}")
    if config.hidden_dim % config.attention_heads:
        raise ValueError("hidden_dim must be divisible by attention_heads")
    if config.conv_kernel % 2 == 0:
        raise ValueError("conv_kernel must be odd")


def _validate_loss(config: DurationPredictorConfig) -> None:
    if config.loss_type not in {"classification", "huber", "l1", "mse", "mse_totlen"}:
        raise ValueError(f"unknown loss_type: {config.loss_type}")
    if config.num_classes < 0:
        raise ValueError("num_classes must be non-negative")
    if config.loss_type == "classification" and config.num_classes <= 0:
        raise ValueError("classification loss requires num_classes > 0")
    if config.loss_type != "classification" and config.num_classes > 0:
        raise ValueError("num_classes must be zero for regression losses")


@dataclass
class DurationPredictorConfig:
    """Hyperparameters and input paths for a duration-predictor run."""

    train_dirs: list[str] = field(default_factory=list)
    valid_dirs: list[str] = field(default_factory=list)
    train_file_pattern: str = "*.txt"
    valid_file_pattern: str = "*.txt"
    output_dir: str = ""

    vocab_size: int = 256
    num_streams: int = 1
    max_duration: int = 50
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    log_duration: bool = True

    loss_type: LossType = "mse"
    totlen_weight: float = 0.0
    huber_beta: float = 1.0
    num_classes: int = 0

    encoder_type: EncoderType = "conformer"
    attention_heads: int = 4
    conv_kernel: int = 17
    ffn_expansion: int = 4

    max_seq_len: int = 2048
    batch_size: int = 64
    num_workers: int = 4
    max_steps: int = 50_000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    log_interval: int = 100
    val_interval: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        """Normalize YAML numeric values and reject invalid hyperparameters."""
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.dropout = float(self.dropout)
        self.totlen_weight = float(self.totlen_weight)
        self.huber_beta = float(self.huber_beta)

        _validate_positive_values(self)
        _validate_ranges(self)
        _validate_model(self)
        _validate_loss(self)

    @property
    def dir(self) -> Path:
        """Return the absolute training-output directory."""
        return Path(self.output_dir).expanduser().resolve()

    def loss_config(self) -> DurationLossConfig:
        """Return the subset of options used by the training objective."""
        return DurationLossConfig(
            log_duration=self.log_duration,
            loss_type=self.loss_type,
            totlen_weight=self.totlen_weight,
            huber_beta=self.huber_beta,
            num_classes=self.num_classes,
        )
