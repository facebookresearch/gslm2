# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plan temperature/seed task manifests and execute one Slurm array task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gslm2.eval.generative.cli import STAGES, run_stage
from gslm2.eval.generative.config import load_config
from gslm2.eval.generative.io import read_jsonl, write_jsonl


def plan_tasks(config_paths: list[Path], mode: str | None = None) -> list[dict[str, object]]:
    """Expand configured temperatures and seeds without submitting any jobs."""
    tasks: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in config_paths:
        source = path.expanduser().resolve()
        if source in seen:
            raise ValueError(f"duplicate sweep config: {source}")
        seen.add(source)
        config = load_config(source)
        if mode is not None and config.mode != mode:
            continue
        if config.oracle:
            pairs = [(config.sampling.temperature, config.sampling.seed)]
        else:
            pairs = [
                (temperature, seed)
                for temperature in (config.sweep.temperatures or [config.sampling.temperature])
                for seed in config.sweep.seeds
            ]
        tasks.extend({"config": str(source), "temperature": temperature, "seed": seed} for temperature, seed in pairs)
    return tasks


def main(argv: list[str] | None = None) -> int:
    """Write a sweep plan, or run one previously planned array task."""
    parser = argparse.ArgumentParser(description="Plan generative evaluation temperature/seed sweeps")
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--task-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--mode", choices=("speech", "text"), help="Filter configurations while planning")
    args = parser.parse_args(argv)
    if args.task_manifest is None:
        if not args.configs or args.output is None:
            parser.error("planning requires at least one config and --output")
        if args.stage is not None or args.task_index is not None:
            parser.error("--stage and --task-index require --task-manifest")
        tasks = plan_tasks(args.configs, args.mode)
        write_jsonl(args.output, tasks)
        # Used by launchers to compute the array range.
        import sys

        sys.stdout.write(f"{len(tasks)}\n")
        return 0
    if (
        args.configs
        or args.output is not None
        or args.mode is not None
        or args.task_index is None
        or args.stage is None
    ):
        parser.error("execution requires --task-manifest, --task-index, and --stage")
    if args.stage == "calibrate":
        parser.error("calibration runs once per config after the sweep")
    tasks = read_jsonl(args.task_manifest)
    if not 0 <= args.task_index < len(tasks):
        parser.error("task index is outside the task manifest")
    task = tasks[args.task_index]
    return run_stage(
        args.stage, [str(task["config"]), "--temperature", str(task["temperature"]), "--seed", str(task["seed"])]
    )


if __name__ == "__main__":
    raise SystemExit(main())
