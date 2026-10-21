"""Native encoder training and resume configuration."""

import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from gslm2.encoder.config.common import (
    MaskingConfig,
    OptimizerConfig,
    RunConfig,
)
from gslm2.encoder.config.data import (
    DataConfig,
)
from gslm2.encoder.config.eval import (
    EvalSpec,
    build_eval_spec,
)
from gslm2.encoder.config.models import model_type_to_config_type
from gslm2.encoder.config.models.speech import SpidRConfig


@dataclass(frozen=True)
class BaseConfig:
    """Fields and methods shared by native encoder training configs."""

    run: RunConfig
    optimizer: OptimizerConfig

    def __post_init__(self) -> None:
        save_interval = self.run.save_interval
        for spec in self.eval_specs():
            if spec.eval_interval is None:
                continue
            if spec.eval_interval <= 0 or spec.eval_interval % save_interval != 0:
                raise ValueError(
                    f"eval_interval ({spec.eval_interval}) for eval_type {spec.eval_type!r} "
                    f"must be a positive multiple of run.save_interval ({save_interval}); "
                    "otherwise the eval would never have a checkpoint to evaluate."
                )

    def dump(self) -> None:
        with (self.run.dir / "config.json").open("w") as f:
            json.dump(asdict(self), f, indent=4, default=str)

    def eval_specs(self) -> list[EvalSpec]:
        """Return the canonical list of eval specs for validation scheduling."""
        return list(self.validation)

    @abstractmethod
    def eval_group_prefixes(self) -> list[str]:
        """Return wandb metric-namespace prefixes for each eval."""


@dataclass(frozen=True)
class SpeechConfig(BaseConfig):
    """Configuration for speech-encoder pretraining (SpidR 2.0)."""

    data: DataConfig | dict[str, DataConfig]
    model: SpidRConfig
    masking: MaskingConfig
    validation: list[EvalSpec] = field(default_factory=list)

    def eval_group_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        for spec in self.validation:
            cfg = spec.config
            group = cfg.get("group") if isinstance(cfg, dict) else None
            prefixes.append(group or spec.eval_type)
        return prefixes

    @classmethod
    def from_file(cls, path: str | Path) -> "SpeechConfig":
        path = Path(path)
        if path.suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        with path.open() as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        run = RunConfig(**data["run"])
        parsed_data = (
            DataConfig.from_dict(data["data"])
            if "manifest" in data["data"]
            else {k: DataConfig.from_dict(v) for k, v in data["data"].items()}
        )
        return cls(
            run=run,
            data=parsed_data,
            model=model_type_to_config_type(run.model_type)(**data.get("model", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            masking=MaskingConfig(**data.get("masking", {})),
            validation=[build_eval_spec(v) for v in (data.get("validation") or [])],
        )


def load_config(path: str | Path) -> SpeechConfig:
    """Load a native encoder training configuration."""
    return SpeechConfig.from_file(path)


@dataclass(frozen=True)
class ResumeConfig:
    step: int
    checkpoint: Path | None
    results: Path
