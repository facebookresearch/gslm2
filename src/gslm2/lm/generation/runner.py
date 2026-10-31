# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed speech-unit and text candidate generation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from tqdm import tqdm

from gslm2.eval.generative.artifacts import collect_records
from gslm2.eval.generative.io import read_jsonl, write_csv, write_json, write_jsonl
from gslm2.eval.generative.manifest import index_by_file_id, record_id
from gslm2.lm.generation.speech import SpeechUnitGenerator
from gslm2.lm.generation.text import TextGenerator
from gslm2.shared.distributed import shard

if TYPE_CHECKING:
    from pathlib import Path

    from gslm2.eval.generative.config import RunConfig
    from gslm2.shared.distributed import DistributedContext


def generate(config: RunConfig, context: DistributedContext) -> Path:
    """Write decoding units or decoded text without loading synthesis models."""
    generation = config.speech if config.mode == "speech" else config.text
    if generation is None:
        raise ValueError(f"{config.mode} generation settings are required")
    rows = read_jsonl(generation.input_manifest)
    indexed = index_by_file_id(rows, generation.input_manifest)
    boundaries = index_by_file_id(read_jsonl(config.reference_manifest), config.reference_manifest)
    if set(indexed) != set(boundaries):
        raise ValueError("generation and reference manifests must have identical IDs")
    generator: SpeechUnitGenerator | TextGenerator
    if config.speech is not None:
        generator = SpeechUnitGenerator(
            config.speech, config.sampling, context, checkpoint_cache=config.run_directory / "decode/.checkpoint"
        )
        if set(generator.prompt_rows) != set(indexed):
            raise ValueError("prompt-unit and full-unit manifests must have identical IDs")
    elif config.text is not None:
        generator = TextGenerator(
            config.text,
            config.sampling,
            context.device,
            context=context,
            checkpoint_cache=config.run_directory / "generation/.checkpoint",
        )
    else:
        raise ValueError("generation settings are required")
    directory = config.run_directory / ("decode" if config.mode == "speech" else "generation")
    records = [generator.generate(row, index) for index, row in tqdm(shard(rows, context), desc="LM decoding")]
    merged = collect_records(directory, records, context, [record_id(row) for row in rows])
    artifact = directory / ("decoded_units.jsonl" if config.mode == "speech" else "candidates.csv")
    if context.is_leader:
        write_json(directory / "config.json", asdict(config))
        if config.mode == "speech":
            write_jsonl(artifact, merged)
        else:
            write_csv(artifact, merged, ("fileid", "file_name", "prompt_text", "continuation_text", "generated_text"))
    context.barrier()
    return artifact
