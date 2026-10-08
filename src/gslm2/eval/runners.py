"""Shared helpers for online validation runners."""

import json
import os
from pathlib import Path


def _append_metrics(metrics_file: Path, step: int, eval_type: str, results: dict) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if rank != 0:
        return
    entry = {"step": step, "group": f"evals_{eval_type}"} | results
    with metrics_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")
