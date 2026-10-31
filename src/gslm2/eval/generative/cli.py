# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stage commands sharing one replica configuration and artifact layout."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from gslm2.eval.generative.config import load_config, override_replica
from gslm2.eval.generative.io import write_json

STAGES = ("decode", "synthesize", "score", "judge", "calibrate")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gslm2.eval.generative.config import RunConfig
    from gslm2.shared.distributed import DistributedContext


def _execute(config: RunConfig, args: argparse.Namespace, context: DistributedContext) -> None:
    if args.stage == "decode":
        from gslm2.lm.generation.runner import generate

        logger.info("Wrote %s", generate(config, context))
    elif args.stage == "synthesize":
        from gslm2.synthesis_continuations import synthesize_continuations

        logger.info("Wrote %s", synthesize_continuations(config, context))
    elif args.stage == "score":
        from gslm2.eval.generative.scoring.evaluator import evaluate

        manifest = args.candidate_manifest.expanduser().resolve() if args.candidate_manifest else None
        summary = evaluate(config.scoring_config(manifest), context=context)
        if context.is_leader:
            logger.info("Scoring summary: %s", summary)
    else:
        from gslm2.eval.generative.judge.evaluator import evaluate as judge

        manifest = args.input_manifest.expanduser().resolve() if args.input_manifest else None
        summary = judge(config.judge_config(manifest), context=context)
        if context.is_leader:
            logger.info("Judge summary: %s", summary)


def run_stage(stage: str | None, argv: list[str] | None = None) -> int:
    """Execute one stage in the environment selected by its caller."""
    parser = argparse.ArgumentParser(description="Generate and evaluate speech or text continuations")
    parser.add_argument("config", type=Path, help="Shared generative experiment YAML")
    parser.add_argument("--stage", choices=STAGES, default=stage, required=stage is None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-directory", type=Path, help="Override the experiment output root")
    parser.add_argument("--candidate-manifest", type=Path, help="Score an existing speech/text candidate manifest")
    parser.add_argument("--input-manifest", type=Path, help="Judge an existing post-ASR results manifest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = override_replica(load_config(args.config), args.temperature, args.seed)
    if args.output_directory is not None:
        config = replace(config, output_directory=args.output_directory.expanduser().resolve())
    if args.candidate_manifest is not None and args.stage != "score":
        parser.error("--candidate-manifest is only valid for scoring")
    if args.input_manifest is not None and args.stage != "judge":
        parser.error("--input-manifest is only valid for judging")
    if args.stage == "calibrate":
        from gslm2.eval.generative.calibration import calibrate_run

        calibrate_run(config)
        return 0
    from gslm2.shared.distributed import init_distributed_context

    context = init_distributed_context(config.distributed_port)
    try:
        if context.is_leader:
            write_json(config.run_directory / "config.json", asdict(config))
        context.barrier()
        _execute(config, args, context)
    finally:
        context.close()
    return 0
