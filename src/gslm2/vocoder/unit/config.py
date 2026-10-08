"""Configuration for the supported unit HiFi-GAN training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

VocoderDtype = Literal["bfloat16", "float16", "float32"]
AggregationType = Literal["average", "concat_lin"]
_SUPPORTED_HOP_SIZE = 320
_SUPPORTED_NUM_STREAMS = 6
_SUPPORTED_VOCAB_SIZE = 256
_NUM_BETAS = 2


@dataclass
class UnitVocoderConfig:
    """Training options shared by the supported single- and six-stream vocoders."""

    workdir: str
    train_manifest: str
    val_manifest: str
    n_units: int | list[int]

    max_steps: int = 500_000
    max_epochs: int = 2_000
    seed: int = 1234
    lr: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    gamma: float = 0.999
    batch_size: int = 64
    segment_size: int = 8_960
    units_hop_size: int = 320
    dtype: VocoderDtype = "bfloat16"
    log_interval: int = 100
    val_interval: int = 5_000
    agg_type: AggregationType = "average"
    num_workers: int | None = None

    def __post_init__(self) -> None:
        """Normalize YAML numerics and validate the supported architecture surface."""
        self.lr = float(self.lr)
        self.gamma = float(self.gamma)
        if len(self.betas) != _NUM_BETAS:
            raise ValueError("betas must contain two values")
        beta1, beta2 = self.betas
        self.betas = (float(beta1), float(beta2))

        positive_values = {
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "segment_size": self.segment_size,
            "units_hop_size": self.units_hop_size,
            "log_interval": self.log_interval,
            "val_interval": self.val_interval,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if self.units_hop_size != _SUPPORTED_HOP_SIZE:
            raise ValueError("only units_hop_size=320 is supported")
        if self.segment_size % self.units_hop_size:
            raise ValueError("segment_size must be divisible by units_hop_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self._validate_units()

    def _validate_units(self) -> None:
        if isinstance(self.n_units, list):
            if len(self.n_units) != _SUPPORTED_NUM_STREAMS or any(
                size != _SUPPORTED_VOCAB_SIZE for size in self.n_units
            ):
                raise ValueError("multi-stream configs require six 256-unit vocabularies")
            if self.agg_type != "concat_lin":
                raise ValueError("multi-stream configs require agg_type='concat_lin'")
        elif self.n_units != _SUPPORTED_VOCAB_SIZE or self.agg_type != "average":
            raise ValueError("single-stream configs require n_units=256 and agg_type='average'")

    @property
    def dir(self) -> Path:
        """Return the absolute training output directory."""
        return Path(self.workdir).expanduser().resolve()

    def validate_paths(self) -> None:
        """Require both manifests immediately before a training launch."""
        for label, value in (("training", self.train_manifest), ("validation", self.val_manifest)):
            if not Path(value).is_file():
                raise FileNotFoundError(f"{label} manifest not found: {value}")
