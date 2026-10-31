# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Decode aligned speech units while retaining full-rate prompt metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from gslm2.eval.generative.io import read_jsonl
from gslm2.eval.generative.manifest import PromptBoundaries, index_by_file_id, record_id
from gslm2.lm.inference import DecodingOptions, decode_tokens, load_opt_model
from gslm2.shared.units import cross_stream_run_length_encode, format_streams, parse_units

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from gslm2.eval.generative.config import SamplingConfig, SpeechGenerationConfig
    from gslm2.shared.distributed import DistributedContext


def parse_streams(value: object) -> list[list[int]]:
    """Parse an aligned single- or multi-stream manifest value."""
    if not isinstance(value, str | list):
        raise TypeError("units must be a string or integer lists")
    units = parse_units(value)
    return cast("list[list[int]]", (units.unsqueeze(0) if units.ndim == 1 else units).tolist())


def oracle_continuation(
    full_streams: list[list[int]], prompt: list[list[int]], prompt_frames: int, continuation_frames: int
) -> list[list[int]]:
    """Append ground-truth continuation tuples using the reference boundary rule."""
    continuation = [stream[prompt_frames : prompt_frames + continuation_frames] for stream in full_streams]
    if not continuation[0]:
        return prompt
    continuation, _ = cross_stream_run_length_encode(continuation)
    if len(continuation[0]) > 1 and all(
        stream[0] == prompt_stream[-1] for stream, prompt_stream in zip(continuation, prompt, strict=True)
    ):
        continuation = [stream[1:] for stream in continuation]
    return [[*prefix, *suffix] for prefix, suffix in zip(prompt, continuation, strict=True)]


class SpeechUnitGenerator:
    """Load one speech LM and produce deduplicated decoding artifacts."""

    def __init__(
        self,
        config: SpeechGenerationConfig,
        sampling: SamplingConfig,
        context: DistributedContext,
        *,
        checkpoint_cache: Path | None = None,
    ) -> None:
        self.config = config
        self.sampling = sampling
        self.device = context.device
        self.boundaries = PromptBoundaries(config.prompt_boundary_manifest)
        self.prompt_rows = index_by_file_id(read_jsonl(config.prompt_unit_manifest), config.prompt_unit_manifest)
        if config.model_checkpoint is None and not config.oracle:
            raise ValueError("speech.model_checkpoint is required for decoding")
        self.loaded_model = (
            None
            if config.oracle
            else load_opt_model(
                cast("Path", config.model_checkpoint),
                self.device,
                model_config=config.model_config,
                context=context,
                checkpoint_cache=checkpoint_cache,
            )
        )
        self.continuation_frames = int(config.continuation_seconds / config.feature_seconds)

    @torch.inference_mode()
    def generate(self, row: Mapping[str, object], example_index: int) -> dict[str, object]:
        """Return raw deduplicated IDs plus the original full-rate prompt units."""
        row_id = record_id(row)
        full_streams = parse_streams(row.get("audio_repr", row.get("units")))
        prompt_row = self.prompt_rows.get(row_id)
        if prompt_row is None:
            raise KeyError(f"missing prompt units for {row_id!r}")
        prompt_streams = parse_streams(prompt_row.get("audio_repr", prompt_row.get("units")))
        if len(prompt_streams) != len(full_streams):
            raise ValueError("prompt and full manifests have different stream counts")
        prompt_frames = self.boundaries.frames(row_id, self.config.feature_seconds)
        if len(prompt_streams[0]) != prompt_frames or len(full_streams[0]) <= prompt_frames:
            raise ValueError(f"{row_id} has inconsistent prompt or continuation frame counts")
        prompt, _ = cross_stream_run_length_encode(prompt_streams)
        if self.config.oracle:
            generated = oracle_continuation(full_streams, prompt, prompt_frames, self.continuation_frames)
        else:
            if self.loaded_model is None:
                raise RuntimeError("speech model was not initialized")
            if len(prompt_streams) != len(self.loaded_model.vocabulary_sizes):
                raise ValueError("manifest streams do not match the LM checkpoint")
            ratio = len(prompt[0]) / prompt_frames
            steps = min(int(self.continuation_frames * ratio * 1.5), self.continuation_frames)
            sequences = torch.tensor(prompt, dtype=torch.long, device=self.device) + self.loaded_model.unit_offset
            torch.manual_seed(self.sampling.seed + example_index)
            generated = cast(
                "list[list[int]]",
                (
                    decode_tokens(
                        self.loaded_model,
                        sequences,
                        steps,
                        options=DecodingOptions(
                            self.sampling.temperature,
                            self.sampling.top_p,
                            self.config.depformer_temperature_annealing,
                        ),
                    )
                    - self.loaded_model.unit_offset
                ).tolist(),
            )
        return {
            "index": example_index,
            "fileid": row_id,
            "file_name": str(row.get("file_name", row.get("audio", row_id))),
            "audio_repr": format_streams(generated),
            "prompt_units": format_streams(prompt_streams),
            "prompt_frames": prompt_frames,
            "prompt_deduplicated_frames": len(prompt[0]),
            "continuation_frames": self.continuation_frames,
            "unit_offset": 0,
            "deduplicated": True,
        }
