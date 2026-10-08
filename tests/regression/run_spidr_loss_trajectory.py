"""Run the SSL speech trainer for N steps and dump the per-step loss trajectory to JSON.

Monkey-patches wandb.log to intercept training-loss values without needing wandb
credentials. The config must set run.wandb_mode=disabled, run.log_interval=1,
and optimizer.max_steps to a small N.

${VAR} references in the config are expanded from the environment before loading
(SpeechConfig.from_file does not interpolate), so callers just set METASR_REGRESSION_*.

Usage:
    python -m tests.regression.run_spidr_loss_trajectory <config.yaml> <out.json>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from string import Template

import wandb

from gslm2.config import SpeechConfig
from gslm2.training.speech.encoder import train


def _capture_wandb_log() -> list[dict[str, float]]:
    """Intercept every wandb.log payload into a list.

    wandb.init(mode="disabled") rebinds wandb.log to a no-op, so we also wrap
    wandb.init to re-apply our patch after the trainer initialises its run.
    """
    captured: list[dict[str, float]] = []

    def _patched_log(data, *args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        if isinstance(data, dict):
            captured.append({k: float(v) for k, v in data.items() if isinstance(v, (int, float))})

    original_init = wandb.init

    def _patched_init(*args, **kwargs):  # noqa: ANN002, ANN003
        run = original_init(*args, **kwargs)
        wandb.log = _patched_log  # type: ignore[assignment]
        return run

    wandb.init = _patched_init  # type: ignore[assignment]
    wandb.log = _patched_log  # type: ignore[assignment]
    return captured


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: run_spidr_loss_trajectory <config.yaml> <out.json>", file=sys.stderr)
        sys.exit(2)
    config_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = out_path.parent / f"_resolved_{config_path.name}"
    resolved.write_text(Template(config_path.read_text()).safe_substitute(os.environ))

    captured = _capture_wandb_log()
    train(SpeechConfig.from_file(resolved))

    losses = [entry["train/loss"] for entry in captured if "train/loss" in entry]
    with out_path.open("w") as f:
        json.dump(losses, f, indent=2)


if __name__ == "__main__":
    main()
