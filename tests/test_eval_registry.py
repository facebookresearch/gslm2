"""Guards on the ``gslm2.eval`` registry.

Runners are registered lazily so the registry stays cheap to import, which
means a stale module path or renamed function only fails when a job actually
launches — potentially hours into a training run. These tests resolve every
target eagerly instead.
"""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gslm2.eval import EVAL_REGISTRY, EVAL_TARGETS

EXPECTED_EVAL_TYPES = {"loss", "abx", "slm", "lm_loss"}


def test_registry_covers_expected_eval_types() -> None:
    assert set(EVAL_REGISTRY) == EXPECTED_EVAL_TYPES
    assert set(EVAL_REGISTRY) == set(EVAL_TARGETS)


@pytest.mark.parametrize("eval_type", sorted(EVAL_TARGETS))
def test_target_resolves(eval_type: str) -> None:
    """The registered ``(module, attr)`` must import and be callable."""
    module_path, attr = EVAL_TARGETS[eval_type]
    module = importlib.import_module(module_path)
    runner = getattr(module, attr, None)
    assert runner is not None, f"{module_path} has no attribute {attr!r}"
    assert callable(runner), f"{module_path}.{attr} is not callable"


@pytest.mark.parametrize("eval_type", sorted(EVAL_TARGETS))
def test_registry_entry_is_routable_by_build_job_args(eval_type: str) -> None:
    """A registered runner is useless if ``_build_job_args`` cannot route to it."""
    from pathlib import Path

    from gslm2.config import ResumeConfig
    from gslm2.config.eval import EvalSpec
    from gslm2.training.validation import _build_job_args

    spec = EvalSpec(
        eval_type=eval_type,
        config={"data": {}, "group": "g", "subset": "dev", "codebook_proba": 0.0},
    )
    resume = ResumeConfig(step=0, checkpoint=Path("ckpt.pt"), results=Path("metrics.jsonl"))
    cfg = SimpleNamespace(run=SimpleNamespace(dir=Path("/run")))

    args = _build_job_args(spec, cfg, resume)  # type: ignore[arg-type]
    assert isinstance(args, tuple), f"{eval_type} routed to a non-tuple: {type(args).__name__}"
    assert args, f"{eval_type} routed to an empty arg tuple"


def test_registry_import_does_not_initialise_cuda() -> None:
    """Importing the registry must leave CUDA uninitialised."""
    code = "import gslm2.eval, torch, sys; sys.exit(1 if torch.cuda.is_initialized() else 0)"
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"importing gslm2.eval initialised CUDA\n{result.stderr}"
