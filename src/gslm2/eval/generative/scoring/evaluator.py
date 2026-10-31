# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestration for speech and text continuation evaluation."""

from __future__ import annotations

import gc
import logging
import math
import statistics
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol

import torch

from gslm2.eval.generative.io import file_digest
from gslm2.eval.generative.scoring.cache import cache_matches, transcript_input_digest, write_cache_metadata
from gslm2.eval.generative.scoring.diversity import RepetitionRates, repetition_rates, s12
from gslm2.eval.generative.scoring.io import (
    read_references,
    read_speech_candidates,
    read_text_candidates,
    read_transcripts,
    write_json,
    write_results,
    write_transcripts,
)
from gslm2.eval.generative.scoring.normalization import normalize_text
from gslm2.eval.generative.scoring.schema import (
    EvaluationRecord,
    PerplexityResult,
    Reference,
    ScoringExample,
    SpeechCandidate,
    SpeechTranscription,
)
from gslm2.shared.distributed import DistributedContext, shard

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gslm2.eval.generative.scoring.config import EvaluationConfig

_DEGENERATE_MAX_WORDS = 5
logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def transcribe(
        self,
        candidates: Sequence[SpeechCandidate],
        references: dict[str, Reference],
    ) -> dict[str, SpeechTranscription]: ...


class PerplexityScorer(Protocol):
    def score(self, prompts: Sequence[str], continuations: Sequence[str]) -> list[PerplexityResult]: ...


def _require_matching_ids(candidate_ids: set[str], references: Mapping[str, Reference]) -> None:
    reference_ids = set(references)
    if candidate_ids != reference_ids:
        missing = sorted(reference_ids - candidate_ids)
        extra = sorted(candidate_ids - reference_ids)
        raise ValueError(
            "candidate and reference manifests must contain identical ids: "
            f"missing candidates for {missing[:5]}, candidates without references {extra[:5]}"
        )


def _speech_examples(
    candidates: Sequence[SpeechCandidate],
    references: dict[str, Reference],
    transcripts: Mapping[str, SpeechTranscription],
) -> list[ScoringExample]:
    _require_matching_ids({candidate.id for candidate in candidates}, references)
    _require_matching_ids(set(transcripts), references)
    return [
        ScoringExample(
            id=candidate.id,
            prompt_text=transcripts[candidate.id].prompt_text,
            continuation_text=transcripts[candidate.id].continuation_text,
            reference_continuation=references[candidate.id].continuation_text,
        )
        for candidate in candidates
    ]


def _text_examples(config: EvaluationConfig, references: dict[str, Reference]) -> list[ScoringExample]:
    candidates = read_text_candidates(config.candidate_manifest)
    _require_matching_ids({candidate.id for candidate in candidates}, references)
    return [
        ScoringExample(
            id=candidate.id,
            prompt_text=candidate.prompt_text,
            continuation_text=candidate.continuation_text,
            reference_continuation=references[candidate.id].continuation_text,
            candidate_text_normalized=candidate.normalized,
        )
        for candidate in candidates
    ]


def _merge_transcript_shards(
    shards: Sequence[Mapping[str, SpeechTranscription]],
    references: Mapping[str, Reference],
) -> dict[str, SpeechTranscription]:
    transcripts: dict[str, SpeechTranscription] = {}
    for rank_transcripts in shards:
        duplicate_ids = transcripts.keys() & rank_transcripts.keys()
        if duplicate_ids:
            raise ValueError(f"duplicate transcript ids across ranks: {sorted(duplicate_ids)[:5]}")
        transcripts.update(rank_transcripts)
    _require_matching_ids(set(transcripts), references)
    return transcripts


def _merge_record_shards(
    shards: Sequence[Sequence[EvaluationRecord]],
    expected_ids: Sequence[str],
) -> list[EvaluationRecord]:
    records = [record for rank_records in shards for record in rank_records]
    record_ids = [record.id for record in records]
    if record_ids != list(expected_ids):
        raise ValueError("distributed evaluation records do not preserve candidate order")
    return records


def _evaluate_examples(
    examples: Sequence[ScoringExample],
    scorer: PerplexityScorer,
    window_words: int,
) -> list[EvaluationRecord]:
    normalized = [
        (
            example.prompt_text if example.candidate_text_normalized else normalize_text(example.prompt_text),
            example.continuation_text
            if example.candidate_text_normalized
            else normalize_text(example.continuation_text),
            normalize_text(example.reference_continuation),
        )
        for example in examples
    ]
    perplexities = scorer.score(
        [prompt for prompt, _, _ in normalized],
        [continuation for _, continuation, _ in normalized],
    )
    if len(perplexities) != len(examples):
        raise ValueError("perplexity scorer returned the wrong number of results")

    records = []
    for example, texts, perplexity in zip(examples, normalized, perplexities, strict=True):
        prompt, continuation, reference_continuation = texts
        candidate_rates = repetition_rates(continuation, window_words)
        reference_rates = repetition_rates(reference_continuation, window_words)
        records.append(
            EvaluationRecord(
                id=example.id,
                prompt_text=example.prompt_text,
                continuation_text=example.continuation_text,
                reference_continuation=example.reference_continuation,
                prompt_text_normalized=prompt,
                continuation_text_normalized=continuation,
                reference_continuation_normalized=reference_continuation,
                cond_genppl=perplexity.perplexity,
                cond_genppl_num_tokens=perplexity.num_tokens,
                repetition_1=candidate_rates.unigram,
                repetition_2=candidate_rates.bigram,
                reference_repetition_1=reference_rates.unigram,
                reference_repetition_2=reference_rates.bigram,
                continuation_words=len(continuation.split()),
            )
        )
    return records


def summarize(
    records: Sequence[EvaluationRecord],
    window_words: int,
    reference_target: RepetitionRates | None = None,
) -> dict[str, object]:
    """Aggregate one stochastic run using the corrected evaluation statistics."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    perplexities = [
        record.cond_genppl
        for record in records
        if record.cond_genppl is not None and math.isfinite(record.cond_genppl)
    ]
    if not perplexities:
        raise ValueError("evaluation contains no finite conditional perplexities")
    candidate_rates = [RepetitionRates(record.repetition_1, record.repetition_2) for record in records]
    measured_reference_rates = [
        RepetitionRates(record.reference_repetition_1, record.reference_repetition_2) for record in records
    ]
    reference_rates = [reference_target] if reference_target is not None else measured_reference_rates
    return {
        "num_examples": len(records),
        "window_words": window_words,
        "cond_genppl_mean": statistics.fmean(perplexities),
        "cond_genppl_median": statistics.median(perplexities),
        "cond_genppl_finite": len(perplexities),
        "candidate_repetition_1_mean": statistics.fmean(rate.unigram for rate in candidate_rates),
        "candidate_repetition_2_mean": statistics.fmean(rate.bigram for rate in candidate_rates),
        "reference_repetition_1_mean": statistics.fmean(rate.unigram for rate in reference_rates),
        "reference_repetition_2_mean": statistics.fmean(rate.bigram for rate in reference_rates),
        "measured_reference_repetition_1_mean": statistics.fmean(rate.unigram for rate in measured_reference_rates),
        "measured_reference_repetition_2_mean": statistics.fmean(rate.bigram for rate in measured_reference_rates),
        "s12_reference_source": "config" if reference_target is not None else "manifest",
        "s12": s12(candidate_rates, reference_rates),
        "degenerate_fraction": sum(record.continuation_words < _DEGENERATE_MAX_WORDS for record in records)
        / len(records),
    }


def _transcribe_candidates(
    config: EvaluationConfig,
    candidates: Sequence[SpeechCandidate],
    references: dict[str, Reference],
    context: DistributedContext,
    transcriber: Transcriber | None,
) -> tuple[list[SpeechCandidate], dict[str, Reference], dict[str, SpeechTranscription]]:
    """Transcribe this rank's candidates or reuse a validated shared cache."""
    local_candidates = [candidate for _, candidate in shard(candidates, context)]
    local_references = {candidate.id: references[candidate.id] for candidate in local_candidates}
    transcript_path = config.output_directory / "transcripts.jsonl"
    leader_digest = transcript_input_digest(config, candidates) if context.is_leader else None
    input_digest = context.all_gather_object(leader_digest)[0]
    if input_digest is None:
        raise RuntimeError("leader did not provide the transcript fingerprint")
    leader_reuse = (
        config.reuse_transcripts and cache_matches(config.output_directory, input_digest)
        if context.is_leader
        else False
    )
    reuse = context.all_gather_object(leader_reuse)[0]
    if reuse:
        logger.info("Reusing Qwen3 transcripts from %s", transcript_path)
        transcripts = read_transcripts(transcript_path)
        _require_matching_ids(set(transcripts), references)
        local_transcripts = {candidate.id: transcripts[candidate.id] for candidate in local_candidates}
    else:
        if transcriber is None and local_candidates:
            from gslm2.eval.generative.scoring.qwen3 import Qwen3Transcriber

            transcriber = Qwen3Transcriber(config.qwen3, device=context.device)
        local_transcripts = (
            transcriber.transcribe(local_candidates, local_references) if transcriber is not None else {}
        )
        transcripts = _merge_transcript_shards(
            context.all_gather_object(local_transcripts),
            references,
        )
        if context.is_leader:
            write_transcripts(transcript_path, transcripts)
            write_cache_metadata(config.output_directory, input_digest)
            logger.info("Wrote Qwen3 transcripts to %s", transcript_path)
        context.barrier()
    del transcriber
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return local_candidates, local_references, local_transcripts


def evaluate(
    config: EvaluationConfig,
    *,
    transcriber: Transcriber | None = None,
    scorer: PerplexityScorer | None = None,
    context: DistributedContext | None = None,
) -> dict[str, object]:
    """Evaluate one candidate manifest locally or across data-parallel ranks."""
    context = context or DistributedContext()
    references = read_references(config.reference_manifest)
    if context.is_leader:
        config.output_directory.mkdir(parents=True, exist_ok=True)
    context.barrier()

    if config.mode == "speech":
        candidates = read_speech_candidates(config.candidate_manifest, config.audio_root)
        _require_matching_ids({candidate.id for candidate in candidates}, references)
        local_candidates, local_references, local_transcripts = _transcribe_candidates(
            config, candidates, references, context, transcriber
        )
        examples = _speech_examples(local_candidates, local_references, local_transcripts)
        expected_ids = [candidate.id for candidate in candidates]
        del transcriber
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        all_examples = _text_examples(config, references)
        examples = [example for _, example in shard(all_examples, context)]
        expected_ids = [example.id for example in all_examples]

    if scorer is None and examples:
        from gslm2.eval.generative.scoring.perplexity import ConditionalPerplexityScorer

        scorer = ConditionalPerplexityScorer(config.perplexity, device=context.device)
    local_records = _evaluate_examples(examples, scorer, config.s12.window_words) if scorer is not None else []
    records = _merge_record_shards(context.all_gather_object(local_records), expected_ids)
    reference_target = (
        RepetitionRates(config.s12.reference_r1, config.s12.reference_r2)
        if config.s12.reference_r1 is not None and config.s12.reference_r2 is not None
        else None
    )
    summary = summarize(records, config.s12.window_words, reference_target)
    summary["reference_manifest_sha256"] = file_digest(config.reference_manifest)
    if context.is_leader:
        write_results(config.output_directory / "results.jsonl", records)
        write_json(config.output_directory / "summary.json", summary)
        write_json(
            config.output_directory / "config.json",
            {
                **asdict(config),
                "reference_manifest": str(config.reference_manifest),
                "candidate_manifest": str(config.candidate_manifest),
                "output_directory": str(config.output_directory),
                "audio_root": str(config.audio_root) if config.audio_root is not None else None,
            },
        )
        logger.info("Wrote evaluation results to %s", config.output_directory)
    context.barrier()
    return summary
