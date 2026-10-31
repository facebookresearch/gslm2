# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Expand decoded continuations while preserving full-rate speech prompts."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import soundfile as sf
import torch
from tqdm import tqdm

from gslm2.duration.predict import expand_streams, load_duration_predictor, predict_durations
from gslm2.eval.generative.artifacts import collect_records
from gslm2.eval.generative.io import read_jsonl, write_csv, write_json, write_jsonl
from gslm2.eval.generative.manifest import index_by_file_id, record_id
from gslm2.shared.distributed import shard
from gslm2.shared.units import cross_stream_run_length_encode, format_streams, parse_units, raw_units
from gslm2.vocoder.model import SAMPLE_RATE, UnitVocoder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from gslm2.duration.model import DurationPredictor
    from gslm2.eval.generative.config import RunConfig
    from gslm2.shared.distributed import DistributedContext


def _unit_value(row: Mapping[str, object], name: str) -> str | list[int] | list[list[int]]:
    value = row.get(name)
    if not isinstance(value, str | list):
        raise TypeError(f"decoded artifact needs {name!r} units")
    return cast("str | list[int] | list[list[int]]", value)


def expand_continuation(
    row: Mapping[str, object], predictor: DurationPredictor, vocab_sizes: Sequence[int], *, log_duration: bool
) -> torch.Tensor:
    """Predict with the full context, then prepend unchanged prompt frames."""
    if row.get("unit_offset") != 0 or row.get("deduplicated") is not True:
        raise ValueError("decoding artifacts must contain deduplicated raw code IDs")
    units = raw_units(parse_units(_unit_value(row, "audio_repr")), vocab_sizes)
    prompt = raw_units(parse_units(_unit_value(row, "prompt_units")), vocab_sizes)
    units = units.unsqueeze(0) if units.ndim == 1 else units
    prompt = prompt.unsqueeze(0) if prompt.ndim == 1 else prompt
    prompt_frames = row.get("prompt_frames")
    prompt_length = row.get("prompt_deduplicated_frames")
    continuation_frames = row.get("continuation_frames")
    if type(prompt_frames) is not int or prompt_frames != prompt.shape[-1]:
        raise ValueError("prompt frame count does not match the full-rate prompt")
    if type(prompt_length) is not int or type(continuation_frames) is not int or continuation_frames <= 0:
        raise ValueError("decoding artifact has invalid prompt/continuation lengths")
    prompt_deduplicated, _ = cross_stream_run_length_encode(prompt.tolist())
    if len(prompt_deduplicated[0]) != prompt_length or units[:, :prompt_length].tolist() != prompt_deduplicated:
        raise ValueError("decoded context must begin with the deduplicated prompt")
    predictor_input = units.tolist() if units.shape[0] > 1 else units[0].tolist()
    durations = predict_durations(predictor, predictor_input, log_duration=log_duration)
    continuation = expand_streams(units[:, prompt_length:].tolist(), durations[prompt_length:])
    continuation = [stream[:continuation_frames] for stream in continuation]
    if not continuation[0]:
        raise ValueError("duration expansion produced an empty continuation")
    return torch.cat([prompt, torch.tensor(continuation, dtype=torch.long)], dim=-1)


@torch.inference_mode()
def synthesize_continuations(config: RunConfig, context: DistributedContext) -> Path:
    """Synthesize a decoding artifact using native duration and vocoder models."""
    if config.mode != "speech" or config.speech is None:
        raise ValueError("continuation synthesis requires speech mode")
    speech = config.speech
    rows = read_jsonl(config.run_directory / "decode" / "decoded_units.jsonl")
    indexed = index_by_file_id(rows, "decoded units")
    references = index_by_file_id(read_jsonl(config.reference_manifest), config.reference_manifest)
    if set(indexed) != set(references):
        raise ValueError("decoded artifacts and references must have identical IDs")
    predictor, predictor_config = load_duration_predictor(speech.duration_predictor_checkpoint, context.device)
    vocoder = UnitVocoder.from_pretrained(speech.vocoder_checkpoint).eval().to(context.device)
    sizes = vocoder.n_units if isinstance(vocoder.n_units, list) else [vocoder.n_units]
    predictor_size = cast("torch.nn.Embedding", predictor.embeddings[0]).num_embeddings
    if predictor.num_streams != len(sizes) or any(size != predictor_size for size in sizes):
        raise ValueError("duration and vocoder checkpoints must have matching streams and vocabularies")
    directory = config.run_directory / "generation"
    records = []
    for index, row in tqdm(shard(rows, context), desc="Continuation synthesis"):
        if row["continuation_frames"] != int(speech.continuation_seconds / speech.feature_seconds):
            raise ValueError("continuation duration differs between decoding and synthesis configs")
        units = expand_continuation(
            row, predictor, sizes, log_duration=bool(predictor_config.get("log_duration", True))
        )
        row_id = record_id(row)
        relative_path = f"audio/{row_id}.flac"
        destination = directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frames = units.squeeze(0) if units.shape[0] == 1 else units
        audio = vocoder.generate(frames.unsqueeze(0).to(context.device)).squeeze(0).cpu().numpy()
        sf.write(destination, audio, SAMPLE_RATE)
        records.append(
            {
                "index": index,
                "fileid": row_id,
                "file_name": row["file_name"],
                "path": relative_path,
                "audio_repr": format_streams(units.tolist()),
                "prompt_frames": row["prompt_frames"],
                "deduplicated": False,
                "unit_offset": 0,
            }
        )
    merged = collect_records(directory, records, context, [record_id(row) for row in rows])
    if context.is_leader:
        write_json(directory / "config.json", asdict(config))
        write_jsonl(directory / "units.jsonl", merged)
        write_csv(config.candidate_manifest, merged, ("fileid", "path"))
    context.barrier()
    return config.candidate_manifest


def main(argv: list[str] | None = None) -> int:
    """Run the native continuation synthesis stage."""
    from gslm2.eval.generative.cli import run_stage

    return run_stage("synthesize", argv)


if __name__ == "__main__":
    raise SystemExit(main())
