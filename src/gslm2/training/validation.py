"""Validation/eval dispatch.

``ValidationLauncher.maybe_launch`` iterates ``cfg.eval_specs()`` and submits
one submitit job per spec. Each job runs ``_run_and_retain``: the spec's
runner from ``gslm2.eval.EVAL_REGISTRY`` (args from ``_build_job_args``),
then checkpoint retention — so trimming happens only in training-launched
validation jobs, never in manual runner invocations.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from gslm2.config import BaseConfig, EvalSpec, ResumeConfig, SpeechConfig
from gslm2.config.common import SlurmConfig
from gslm2.eval import EVAL_METRIC_KEYS, EVAL_REGISTRY
from gslm2.slurm import launch_with_submitit, validation_job_config

logger = logging.getLogger()


def _eval_yaml_slurm_time(config: Path | dict[str, Any]) -> dict:
    """Return ``{"time": ...}`` from an eval YAML's ``slurm`` block, else ``{}``.

    Only the wall-clock ``time`` is honored as a default for online launches; other
    resource fields stay governed by ``validation_job_config`` and the training
    config's ``spec.slurm``. Non-path configs (the inline dicts used by the
    speech/SLM evals) and YAMLs with no ``slurm.time`` return ``{}``.
    """
    if not isinstance(config, Path) or config.suffix not in {".yaml", ".yml"} or not config.exists():
        return {}
    with config.open() as f:
        data = yaml.safe_load(f) or {}
    slurm = data.get("slurm") or {}
    return {"time": slurm["time"]} if "time" in slurm else {}


def launch_validation(cfg: SpeechConfig, resume: ResumeConfig) -> None:
    """Launch validation for the given config and resume state."""
    ValidationLauncher(cfg, resume.results).maybe_launch(resume.step, resume.checkpoint)


def _build_job_args(spec: EvalSpec, cfg: BaseConfig, resume: ResumeConfig) -> tuple:
    """Map an EvalSpec into positional args for its registered runner."""
    if spec.eval_type == "loss":
        return (cfg, resume, spec.config["data"], spec.config["group"])
    if spec.eval_type == "abx":
        return (cfg, resume, spec.config["subset"], spec.config["codebook_proba"])
    if spec.eval_type in {"lm_loss", "slm"}:
        # Native-SLM runners take (eval config, resume, run_dir, eval_name).
        return (spec.config, resume, cfg.run.dir, spec.resolved_name)
    raise KeyError(f"No _build_job_args entry for eval_type {spec.eval_type!r}")


def _run_and_retain(  # noqa: PLR0913
    runner: Callable[..., Any],
    runner_args: tuple,
    folder: Path,
    keep_latest: int,
    keep_best_n: int,
    keep_best_metric: str,
    keep_best_mode: str,
) -> None:
    """Job-side entry: score, then enforce checkpoint retention.

    Submitted by ``ValidationLauncher.maybe_launch`` instead of the raw
    runner, so each landing score re-converges the latest-plus-best set — the
    last step is covered whenever its job runs, with no drain wait. Module
    level so submitit pickles it by reference like the runners themselves.
    """
    from gslm2.checkpoint import Checkpointer

    runner(*runner_args)
    Checkpointer(
        Path(folder),
        interval=1,
        keep_latest=keep_latest,
        keep_best_n=keep_best_n,
        keep_best_metric=keep_best_metric,
        keep_best_mode=keep_best_mode,
    ).apply_retention()


class ValidationLauncher:
    """Submit one submitit job per due eval, dispatched via ``EVAL_REGISTRY``."""

    def __init__(self, cfg: BaseConfig, metrics_file: Path) -> None:
        self.cfg = cfg
        self.metrics_file = metrics_file
        if cfg.run.keep_best_n > 0:
            self._validate_keep_best(cfg)

    @staticmethod
    def _validate_keep_best(cfg: BaseConfig) -> None:
        """Single fail-fast point for keep-best retention preconditions."""
        specs = cfg.eval_specs()
        if not specs:
            raise ValueError(
                f"keep_best_n={cfg.run.keep_best_n} requires at least one validation eval spec, "
                f"otherwise keep_best_metric={cfg.run.keep_best_metric!r} is never logged and "
                "retention keeps every checkpoint."
            )
        for spec in specs:
            ValidationLauncher._lookup_runner(spec.eval_type)
        if cfg.run.keep_best_metric not in EVAL_METRIC_KEYS:
            raise ValueError(
                f"keep_best_metric={cfg.run.keep_best_metric!r} is never logged by any eval "
                f"(known keys: {sorted(EVAL_METRIC_KEYS)})."
            )

    def due_specs(self, step: int) -> list[EvalSpec]:
        """Eval specs with a checkpoint due at ``step``."""
        save_interval = self.cfg.run.save_interval
        return [s for s in self.cfg.eval_specs() if step % (s.eval_interval or save_interval) == 0]

    @staticmethod
    def _lookup_runner(eval_type: str) -> Callable[..., Any]:
        try:
            return EVAL_REGISTRY[eval_type]
        except KeyError as e:
            raise KeyError(f"eval_type {eval_type!r} not in EVAL_REGISTRY; registered: {sorted(EVAL_REGISTRY)}") from e

    def maybe_launch(self, step: int, checkpoint: Path | None) -> None:
        if checkpoint is None:
            logger.info("No checkpoint found, skipping validation")
            return
        if self.cfg.run.slurm_validation is None and "SLURM_JOB_ID" not in os.environ:
            logger.info("Not training in a Slurm job and no slurm_validation config provided, skipping validation")
            return

        specs_due = self.due_specs(step)
        if not specs_due:
            logger.info("No eval specs due at step %s, skipping validation", step)
            return

        logger.info("Launching %s eval(s) at step %s", len(specs_due), step)
        resume = ResumeConfig(step=step, checkpoint=checkpoint, results=self.metrics_file)
        output_dir = self.cfg.run.dir / "validation"
        run = self.cfg.run
        for spec in specs_due:
            runner = self._lookup_runner(spec.eval_type)
            args = _build_job_args(spec, self.cfg, resume)
            job = (
                _run_and_retain,
                (
                    runner,
                    args,
                    run.dir,
                    run.keep_latest,
                    run.keep_best_n,
                    run.keep_best_metric,
                    run.keep_best_mode,
                ),
            )
            base = self.cfg.run.slurm_validation or validation_job_config()
            slurm = SlurmConfig(**{**base, **_eval_yaml_slurm_time(spec.config), **(spec.slurm or {})})
            launch_with_submitit(
                f"val-{step}-{spec.resolved_name}",
                [job],
                output_dir,
                slurm,
                copy_code=False,
            )
