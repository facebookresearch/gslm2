"""Regenerate committed regression baselines (all CPU).

    python tests/regression/baselines/regen_baselines.py inference
    python tests/regression/baselines/regen_baselines.py loss
    python tests/regression/baselines/regen_baselines.py all

Run after an intentional change to model arch/init/forward that you've
verified is correct, then commit the updated .pt / .json blobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Re-use the same forward-pass logic + manifest builder as the tests so the
# baselines stay aligned with what the tests load. Importing avoids drift.
from tests.regression.helpers import (  # noqa: E402
    build_tiny_manifest,
    build_tiny_text_manifest,
    run_lm_loss_trajectory,
    run_slm_loss_trajectory,
    torchrun_env,
)
from tests.regression.test_inference_parity import _opt_forward, _slm_forward, _spidr_forward  # noqa: E402

_SLM_STEPS = 50
_LM_STEPS = 50


def regen_inference_baselines() -> None:
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    spidr_out = _spidr_forward()
    torch.save(spidr_out, THIS_DIR / "spidr_inference_output.pt")
    print(f"wrote {THIS_DIR / 'spidr_inference_output.pt'} shape={tuple(spidr_out.shape)}")

    slm_out = _slm_forward()
    torch.save(slm_out, THIS_DIR / "slm_inference_output.pt")
    print(f"wrote {THIS_DIR / 'slm_inference_output.pt'} shape={tuple(slm_out.shape)}")

    opt_out = _opt_forward()
    torch.save(opt_out, THIS_DIR / "opt_inference_output.pt")
    print(f"wrote {THIS_DIR / 'opt_inference_output.pt'} shape={tuple(opt_out.shape)}")


def _make_tiny_manifest(workdir: Path) -> Path:
    return build_tiny_manifest(workdir / "audio", workdir / "tiny.csv")


def regen_spidr_loss_baseline() -> None:
    _regen_encoder_loss_baseline("spidr_loss_parity.yaml", "spidr_loss_trajectory.json")


def regen_spidr_v2_loss_baseline() -> None:
    _regen_encoder_loss_baseline("spidr_v2_loss_parity.yaml", "spidr_v2_loss_trajectory.json")


def _regen_encoder_loss_baseline(config_name: str, out_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="regression_baselines_") as tmp:
        tmp_path = Path(tmp)
        manifest = _make_tiny_manifest(tmp_path)
        out_json = tmp_path / out_name
        cmd = [
            sys.executable,
            "-m",
            "tests.regression.run_spidr_loss_trajectory",
            str(REPO_ROOT / "configs" / "regression" / config_name),
            str(out_json),
        ]
        subprocess.run(cmd, env=torchrun_env(tmp_path, manifest), cwd=str(REPO_ROOT), check=True)
        baseline = json.loads(out_json.read_text())
        (THIS_DIR / out_name).write_text(json.dumps(baseline, indent=2))
        print(f"wrote {out_name} ({len(baseline)} steps)")


def regen_slm_loss_baseline() -> None:
    with tempfile.TemporaryDirectory(prefix="regression_baselines_") as tmp:
        manifest = _make_tiny_manifest(Path(tmp))
        baseline = run_slm_loss_trajectory(manifest, steps=_SLM_STEPS)
        (THIS_DIR / "slm_loss_trajectory.json").write_text(json.dumps(baseline, indent=2))
        print(f"wrote slm_loss_trajectory.json ({len(baseline)} steps)")


def regen_lm_causal_loss_baseline() -> None:
    with tempfile.TemporaryDirectory(prefix="regression_baselines_") as tmp:
        manifest = build_tiny_text_manifest(Path(tmp) / "tiny_text.txt")
        baseline = run_lm_loss_trajectory(manifest, steps=_LM_STEPS)
        (THIS_DIR / "lm_causal_loss_trajectory.json").write_text(json.dumps(baseline, indent=2))
        print(f"wrote lm_causal_loss_trajectory.json ({len(baseline)} steps)")


def regen_loss_baselines() -> None:
    regen_spidr_loss_baseline()
    regen_spidr_v2_loss_baseline()
    regen_slm_loss_baseline()
    regen_lm_causal_loss_baseline()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["inference", "loss", "all"])
    args = parser.parse_args()
    if args.which in {"inference", "all"}:
        regen_inference_baselines()
    if args.which in {"loss", "all"}:
        regen_loss_baselines()


if __name__ == "__main__":
    main()
