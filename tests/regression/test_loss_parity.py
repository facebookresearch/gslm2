"""N-step loss parity: per-step training-loss trajectory must match a committed baseline.

Three models, all on CPU in standard CI:
- test_spidr_loss_parity     — SpidR encoder SSL training (subprocess runner).
- test_spidr_v2_loss_parity  — SpidR-v2 (sjepa) encoder SSL training: same trainer
                               with entropy regularisation + teacher heads.
- test_slm_loss_parity       — SLM (OPT LM) training on a frozen SpidR encoder, run
                               in-process. Requires fairseq2; always skipped here.

    pytest tests/regression/test_loss_parity.py -v -m "not gpu"
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .helpers import (
    BASELINES_DIR,
    REPO_ROOT,
    assert_loss_trajectory_close,
    load_json_baseline,
    run_lm_loss_trajectory,
    run_slm_loss_trajectory,
    torchrun_env,
)

CONFIGS_DIR = REPO_ROOT / "configs" / "regression"
_SPIDR_BASELINE = BASELINES_DIR / "spidr_loss_trajectory.json"
_SPIDR_V2_BASELINE = BASELINES_DIR / "spidr_v2_loss_trajectory.json"
_SLM_BASELINE = BASELINES_DIR / "slm_loss_trajectory.json"
_LM_CAUSAL_BASELINE = BASELINES_DIR / "lm_causal_loss_trajectory.json"
_FINAL_TOL = 0.02
_SLM_STEPS = 50
_LM_STEPS = 50


def _run_encoder_loss_trajectory(config_name: str, tmp_path: Path, manifest: Path) -> list[float]:
    """Run the SSL encoder trainer for N steps via the loss-trajectory runner."""
    out_json = tmp_path / "loss.json"
    env = torchrun_env(tmp_path, manifest)
    cmd = [
        sys.executable,
        "-m",
        "tests.regression.run_spidr_loss_trajectory",
        str(CONFIGS_DIR / config_name),
        str(out_json),
    ]
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0:
        raise AssertionError(f"{config_name} loss runner exited {result.returncode}\nSTDERR:\n{result.stderr[-4000:]}")
    return json.loads(out_json.read_text())


def test_spidr_loss_parity(tmp_path: Path, tiny_manifest: Path) -> None:
    if not _SPIDR_BASELINE.exists():
        pytest.skip(f"baseline {_SPIDR_BASELINE.name} missing; regen via baselines/regen_baselines.py")
    actual = _run_encoder_loss_trajectory("spidr_loss_parity.yaml", tmp_path, tiny_manifest)
    assert_loss_trajectory_close(actual, load_json_baseline(_SPIDR_BASELINE), final_tol=_FINAL_TOL, name="spidr loss")


def test_spidr_v2_loss_parity(tmp_path: Path, tiny_manifest: Path) -> None:
    if not _SPIDR_V2_BASELINE.exists():
        pytest.skip(f"baseline {_SPIDR_V2_BASELINE.name} missing; regen via baselines/regen_baselines.py")
    actual = _run_encoder_loss_trajectory("spidr_v2_loss_parity.yaml", tmp_path, tiny_manifest)
    assert_loss_trajectory_close(
        actual, load_json_baseline(_SPIDR_V2_BASELINE), final_tol=_FINAL_TOL, name="spidr_v2 loss"
    )


def test_slm_loss_parity(tiny_manifest: Path) -> None:
    pytest.importorskip("fairseq2")
    if not _SLM_BASELINE.exists():
        pytest.skip(f"baseline {_SLM_BASELINE.name} missing; regen via baselines/regen_baselines.py")
    actual = run_slm_loss_trajectory(tiny_manifest, steps=_SLM_STEPS)
    assert_loss_trajectory_close(actual, load_json_baseline(_SLM_BASELINE), final_tol=_FINAL_TOL, name="slm loss")


def test_lm_causal_loss_parity(tiny_text_manifest: Path) -> None:
    if not _LM_CAUSAL_BASELINE.exists():
        pytest.skip(f"baseline {_LM_CAUSAL_BASELINE.name} missing; regen via baselines/regen_baselines.py")
    actual = run_lm_loss_trajectory(tiny_text_manifest, steps=_LM_STEPS)
    assert_loss_trajectory_close(
        actual, load_json_baseline(_LM_CAUSAL_BASELINE), final_tol=_FINAL_TOL, name="lm_causal loss"
    )
