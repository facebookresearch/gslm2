"""Run, optimizer, masking, and slurm configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from gslm2.shared.config import SlurmConfig
from gslm2.shared.constants import AUDIO_SAMPLE_RATE as AUDIO_SAMPLE_RATE

MaskingStrategy = Literal["static", "uniform", "normal", "poisson"]
ModelType = Literal["spidr"]


@dataclass(frozen=True)
class RunConfig:
    """Training run configuration."""

    workdir: str
    wandb_project: str
    wandb_name: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    random_seed: int = 0
    use_deterministic: bool = False
    log_interval: int = 200
    eval_interval: int = 10_000
    save_interval: int = 2_500
    metrics_sync_interval: int = 300
    init_ckpt: str | None = None
    compile: bool = False
    keep_latest: int = -1
    save_latest: bool = False  # also write a rolling latest.pt
    keep_best_n: int = 0  # keep top-N evaluated checkpoints by keep_best_metric
    keep_best_metric: str = "loss"  # metrics.jsonl eval key to rank by
    keep_best_mode: Literal["min", "max"] = "min"
    model_type: ModelType = "spidr"
    slurm: SlurmConfig | None = None
    slurm_validation: SlurmConfig | None = None

    @property
    def dir(self) -> Path:
        return Path(self.workdir).resolve()

    def __post_init__(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW and learning rate scheduler configuration."""

    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)  # WARNING: Changed from the original (0.9, 0.98)
    weight_decay: float = 0.01
    eps: float = 1e-6
    max_norm: float = 10.0  # WARNING: Not set originally
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.01
    max_steps: int = 400_000

    warmup_steps: int = 12_000
    hold_steps: int = 188_000
    decay_steps: int = 200_000
    rsqrt_timescale: int = 10_000
    rsqrt_shift: int = 0

    scheduler: Literal["tristage", "cosine", "linear", "rsqrt", "constant"] = "tristage"

    exclude_from_optimizer: list[str] = field(default_factory=lambda: ["teacher"])
    to_freeze: list[str] = field(default_factory=lambda: ["feature_extractor", "feature_projection"])
    # False: apply weight_decay uniformly to all params. True: exclude biases and 1-D
    # params (LayerNorm) from weight decay (the recipe used by V-JEPA and HF LMs).
    no_decay_on_norm_bias: bool = False

    @property
    def mixed_precision(self) -> bool:
        return self.dtype != "float32"


@dataclass(frozen=True)
class MaskingConfig:
    mask_prob: float = 0.8  # model.mask_prob
    mask_selection: MaskingStrategy = "static"  # model.mask_selection
    mask_other: float = 0.0  # model.mask_other
    mask_length: int = 10  # model.mask_length
    no_mask_overlap: bool = False  # model.no_mask_overlap
    mask_min_space: int = 1  # model.mask_min_space
    mask_channel_prob: float = 0.0  # model.mask_channel_prob
    mask_channel_selection: MaskingStrategy = "static"  # model.mask_channel_selection
    mask_channel_other: float = 0.0  # model.mask_channel_other
    mask_channel_length: int = 10  # model.mask_channel_length
    no_mask_channel_overlap: bool = False  # model.no_mask_channel_overlap
    mask_channel_min_space: int = 1  # model.mask_channel_min_space
    legacy_mask_drop: bool = True  # use legacy [:min_len] shatter
    mask_drop_prob: float = 0.11  # per masked idx drop (no shatter)
