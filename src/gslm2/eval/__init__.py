"""Eval registry. Maps ``eval_type`` -> callable invoked by the launcher.

Registered runners are called as ``runner(*args)`` from
``ValidationLauncher.maybe_launch``, with ``args`` constructed by
``_build_job_args`` in ``gslm2.training.validation``.
"""

import importlib
import os
from collections.abc import Callable
from typing import Any

from gslm2.environment import restrict_cuda_to_local_rank

EVAL_REGISTRY: dict[str, Callable[..., Any]] = {}

#: ``eval_type`` -> ``(module, attribute)`` for every lazily-registered runner.
EVAL_TARGETS: dict[str, tuple[str, str]] = {}


def register_eval(name: str, runner: Callable[..., Any]) -> None:
    EVAL_REGISTRY[name] = runner


def _try_register(name: str, module: str, attr: str) -> None:
    """Register a lazy resolver so the runner module isn't imported at registry init time."""

    def _lazy_runner(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        restrict_cuda_to_local_rank()
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_FILE", "/tmp/nccl_debug_%h_%p.log")  # noqa: S108
        return getattr(importlib.import_module(module), attr)(*args, **kwargs)

    EVAL_TARGETS[name] = (module, attr)
    register_eval(name, _lazy_runner)


# speech-encoder evals
_try_register("loss", "gslm2.validate", "validate_existing_checkpoint")
_try_register("abx", "gslm2.validate", "wrapper_abx")

# native SLM evals (online)
_try_register("slm", "gslm2.eval.slm.validation_runners", "run_slm_eval")
_try_register("lm_loss", "gslm2.eval.slm.validation_runners", "run_lm_loss_eval")

#: Metric keys runners append to ``metrics.jsonl`` (the only entries retention scores).
EVAL_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "dev_loss",
        "loss",
        "ssl_loss",
        "supervised_loss",
        "accuracy",
        "accuracy_norm",
        "npairs",
    }
)
