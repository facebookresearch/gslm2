# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed orchestration for LLM-judge evaluation."""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol

from gslm2.eval.generative.judge.io import read_examples, write_json, write_results
from gslm2.eval.generative.judge.normalization import normalize_for_judge
from gslm2.eval.generative.judge.prompt import PROMPT_TEMPLATE_SHA256, build_prompt
from gslm2.eval.generative.judge.schema import JudgeExample, JudgePrediction, JudgeRecord
from gslm2.shared.distributed import DistributedContext, shard

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gslm2.eval.generative.judge.config import EvaluationConfig

logger = logging.getLogger(__name__)


class Judge(Protocol):
    def score(self, prompts: Sequence[str]) -> list[JudgePrediction]: ...


def _judge_examples(examples: Sequence[JudgeExample], judge: Judge) -> list[JudgeRecord]:
    normalized = [
        (normalize_for_judge(example.prompt_text), normalize_for_judge(example.continuation_text))
        for example in examples
    ]
    predictions = judge.score([build_prompt(prompt, continuation) for prompt, continuation in normalized])
    if len(predictions) != len(examples):
        raise ValueError("judge returned the wrong number of predictions")
    return [
        JudgeRecord(
            id=example.id,
            prompt_text=example.prompt_text,
            continuation_text=example.continuation_text,
            prompt_text_normalized=prompt,
            continuation_text_normalized=continuation,
            judge_feedback=prediction.feedback,
            judge_score=prediction.score,
        )
        for example, (prompt, continuation), prediction in zip(
            examples,
            normalized,
            predictions,
            strict=True,
        )
    ]


def _merge_record_shards(
    shards: Sequence[Sequence[JudgeRecord]],
    expected_ids: Sequence[str],
) -> list[JudgeRecord]:
    records = [record for rank_records in shards for record in rank_records]
    if [record.id for record in records] != list(expected_ids):
        raise ValueError("distributed judge records do not preserve input order")
    return records


def summarize(records: Sequence[JudgeRecord]) -> dict[str, object]:
    """Aggregate valid scores while reporting unparsable model responses."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    scores = [record.judge_score for record in records if record.judge_score is not None]
    counts = {str(score): scores.count(score) for score in range(1, 6)}
    return {
        "num_examples": len(records),
        "num_scored": len(scores),
        "num_unscored": len(records) - len(scores),
        "judge_score_mean": statistics.fmean(scores) if scores else None,
        "judge_score_median": statistics.median(scores) if scores else None,
        "score_counts": counts,
    }


def evaluate(
    config: EvaluationConfig,
    *,
    judge: Judge | None = None,
    context: DistributedContext | None = None,
) -> dict[str, object]:
    """Judge a post-ASR manifest locally or across data-parallel ranks."""
    context = context or DistributedContext()
    examples = read_examples(config.input_manifest)
    local_examples = [example for _, example in shard(examples, context)]
    if context.is_leader:
        config.output_directory.mkdir(parents=True, exist_ok=True)
    context.barrier()

    if judge is None and local_examples:
        from gslm2.eval.generative.judge.model import PrometheusJudge

        judge = PrometheusJudge(config.judge, context.device)
    local_records = _judge_examples(local_examples, judge) if judge is not None else []
    records = _merge_record_shards(
        context.all_gather_object(local_records),
        [example.id for example in examples],
    )
    summary = summarize(records)
    if context.is_leader:
        write_results(config.output_directory / "results.jsonl", records)
        write_json(config.output_directory / "summary.json", summary)
        write_json(
            config.output_directory / "config.json",
            {
                **asdict(config),
                "input_manifest": str(config.input_manifest),
                "output_directory": str(config.output_directory),
                "prompt_template_version": "v3.1-short",
                "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            },
        )
        logger.info("Wrote LLM-judge results to %s", config.output_directory)
    context.barrier()
    return summary
