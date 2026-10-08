"""Guards on checkpoint retention (``keep_latest`` / ``keep_best_n``).

Best-by-metric retention must work with unlimited ``keep_latest``, nothing
unscored may ever be deleted (a pending validation job still needs the file),
and ``keep_best_n`` must fail fast when no eval can produce its metric.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gslm2.checkpoint import Checkpointer
from gslm2.config.common import OptimizerConfig, RunConfig
from gslm2.config.config import BaseConfig
from gslm2.config.eval import EvalSpec
from gslm2.training.validation import ValidationLauncher


def _write_metrics(folder: Path, rows: list[tuple[int, float]]) -> None:
    with (folder / "metrics.jsonl").open("w") as f:
        for step, value in rows:
            f.write(json.dumps({"step": step, "group": "evals_lm_loss", "dev_loss": value}) + "\n")


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).touch()


def _retain(folder: Path, **kwargs: object) -> set[str]:
    Checkpointer(folder, interval=1, **kwargs).apply_retention()  # type: ignore[arg-type]
    return {p.name for p in folder.glob("*.pt")}


@dataclass(frozen=True)
class _Cfg(BaseConfig):
    validation: list[EvalSpec] = field(default_factory=list)

    def eval_group_prefixes(self) -> list[str]:
        return []


def _cfg(tmp_path: Path, validation: list[EvalSpec], **run_kwargs: object) -> _Cfg:
    run = RunConfig(workdir=str(tmp_path), wandb_project="p", **run_kwargs)  # type: ignore[arg-type]
    return _Cfg(run=run, optimizer=OptimizerConfig(), validation=validation)


def test_keep_best_applies_with_unlimited_latest(tmp_path: Path) -> None:
    """keep_best_n must work when keep_latest=-1 (no latest-based sparing)."""
    _touch(tmp_path, "step_1.pt", "step_2.pt", "step_3.pt", "final.pt")
    _write_metrics(tmp_path, [(1, 0.3), (2, 0.1), (3, 0.2)])
    remaining = _retain(tmp_path, keep_latest=-1, keep_best_n=1)
    assert remaining == {"step_2.pt", "final.pt"}


def test_unscored_older_than_newest_scored_survives(tmp_path: Path) -> None:
    """An unscored checkpoint is kept even when older than scored ones."""
    _touch(tmp_path, "step_1.pt", "step_2.pt", "step_3.pt")
    _write_metrics(tmp_path, [(3, 0.1)])
    remaining = _retain(tmp_path, keep_latest=1, keep_best_n=1)
    assert remaining == {"step_1.pt", "step_2.pt", "step_3.pt"}


def test_latest_only_deletes_only_scored(tmp_path: Path) -> None:
    """Without keep_best, keep_latest still spares unscored checkpoints."""
    _touch(tmp_path, "step_1.pt", "step_2.pt", "step_3.pt")
    _write_metrics(tmp_path, [(1, 0.5)])
    remaining = _retain(tmp_path, keep_latest=1, keep_best_n=0)
    assert remaining == {"step_2.pt", "step_3.pt"}


def test_final_pt_never_deleted(tmp_path: Path) -> None:
    _touch(tmp_path, "step_1.pt", "final.pt")
    _write_metrics(tmp_path, [(1, 0.5)])
    remaining = _retain(tmp_path, keep_latest=-1, keep_best_n=1)
    assert "final.pt" in remaining


def test_keep_best_requires_eval_specs(tmp_path: Path) -> None:
    """Launcher construction fails fast when no eval can log the metric."""
    cfg = _cfg(tmp_path, [], keep_best_n=2)
    with pytest.raises(ValueError, match="keep_best_n"):
        ValidationLauncher(cfg, tmp_path / "metrics.jsonl")


def test_keep_best_ok_without_eval_specs_when_disabled(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [], keep_best_n=0)
    ValidationLauncher(cfg, tmp_path / "metrics.jsonl")  # must not raise


def test_launcher_rejects_unknown_eval_type(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [EvalSpec(eval_type="typo_eval")], keep_best_n=1)
    with pytest.raises(KeyError, match="typo_eval"):
        ValidationLauncher(cfg, tmp_path / "metrics.jsonl")


def test_launcher_accepts_registered_eval_type(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [EvalSpec(eval_type="lm_loss")], keep_best_n=1)
    ValidationLauncher(cfg, tmp_path / "metrics.jsonl")  # must not raise


def test_launcher_rejects_unlogged_metric(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [EvalSpec(eval_type="lm_loss")], keep_best_n=1, keep_best_metric="dev_los")
    with pytest.raises(ValueError, match="never logged by any eval"):
        ValidationLauncher(cfg, tmp_path / "metrics.jsonl")


def test_launcher_accepts_known_metric(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [EvalSpec(eval_type="slm")], keep_best_n=1, keep_best_metric="accuracy")
    ValidationLauncher(cfg, tmp_path / "metrics.jsonl")  # must not raise
