"""Evaluation configuration: ABX, eval specs, and per-eval Slurm overrides."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from gslm2.encoder.config.common import SlurmConfig


class ABXSubset(TypedDict):
    manifest: Path
    item_file: Path


@dataclass(frozen=True)
class ABXTrainConfig:
    online_abx: bool = False
    online_abx_layer: int = 8
    online_abx_bilevel_type: Literal["equal", "hard_positive", "hard_middle", "soft"] = "equal"
    online_abx_bilevel_temp: float = 0.01
    subsets: list[ABXSubset] = field(default_factory=list)
    codebook_proba: list[bool] = field(default_factory=lambda: [True, False])

    @property
    def launch_during_training(self) -> bool:
        return bool(self.subsets)


@dataclass(frozen=True)
class EvalSpec:
    """One eval launched during training.

    - ``eval_type``: key into ``gslm2.encoder.eval.EVAL_REGISTRY`` (e.g. ``"abx"``,
      ``"loss"``, ``"video_classification"``).
    - ``config``: per-eval payload (path to YAML or inline dict).
    - ``eval_interval``: launch when ``step % eval_interval == 0``;
      ``None`` falls back to ``cfg.run.save_interval``.
    - ``slurm``: optional per-eval Slurm overrides.
    - ``eval_name``: display name for job/dir/wandb; defaults to eval_type.
    """

    eval_type: str
    config: Path | dict[str, Any] = field(default_factory=dict)
    eval_interval: int | None = None
    slurm: SlurmConfig | None = None
    eval_name: str | None = None

    @property
    def resolved_name(self) -> str:
        """Eval name for metric groups, job names, and output dirs.

        Priority: explicit ``eval_name`` → config file stem → ``eval_type``.
        """
        if self.eval_name:
            return self.eval_name
        if isinstance(self.config, Path):
            return self.config.stem
        return self.eval_type


def build_eval_spec(data: dict[str, Any]) -> EvalSpec:
    """Build an ``EvalSpec`` from a raw YAML dict.

    Converts string ``config`` values to ``Path`` objects. Slurm overrides
    from the eval spec are stored as-is; the merge with the base
    ``run.slurm_validation`` happens at launch time in
    :class:`~gslm2.encoder.training.validation.ValidationLauncher`.
    """
    config_val = data.get("config", {})
    return EvalSpec(
        eval_type=data["eval_type"],
        config=Path(config_val) if isinstance(config_val, str) else config_val,
        eval_interval=data.get("eval_interval"),
        slurm=data.get("slurm"),
        eval_name=data.get("eval_name"),
    )
