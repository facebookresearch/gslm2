# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Continuation transcription and prompt fallback with Qwen3-Omni."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from tqdm import tqdm

from gslm2.eval.generative.scoring.schema import SpeechTranscription

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import numpy as np

    from gslm2.eval.generative.scoring.config import Qwen3Config
    from gslm2.eval.generative.scoring.schema import Reference, SpeechCandidate


def _generated_sequences(generated: object) -> Tensor:
    if isinstance(generated, tuple):
        generated = generated[0]
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    if not isinstance(generated, Tensor):
        raise TypeError(f"unexpected output from Qwen3-Omni generate(): {type(generated)}")
    return generated


def _audio_segment(
    candidate: SpeechCandidate,
    reference: Reference,
    sample_rate: int,
    *,
    prompt: bool,
) -> np.ndarray:
    if reference.prompt_end_seconds is None:
        raise ValueError(f"speech reference {reference.id!r} has no prompt boundary")
    if not candidate.audio_path.is_file():
        raise FileNotFoundError(f"candidate audio does not exist: {candidate.audio_path}")
    soundfile = import_module("soundfile")
    samples, actual_sample_rate = soundfile.read(candidate.audio_path, dtype="float32", always_2d=True)
    if actual_sample_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz audio, got {actual_sample_rate}: {candidate.audio_path}")
    start = int(reference.prompt_end_seconds * sample_rate)
    if start >= samples.shape[0]:
        raise ValueError(f"prompt boundary leaves no continuation audio: {candidate.audio_path}")
    segment = samples[:start] if prompt else samples[start:]
    if segment.size == 0:
        raise ValueError(f"prompt boundary selects empty audio: {candidate.audio_path}")
    return segment.mean(axis=1)


class Qwen3Transcriber:
    """Batch deterministic Qwen3-Omni transcription of required audio segments."""

    def __init__(self, config: Qwen3Config, device: torch.device | None = None) -> None:
        from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.config = config
        self.processor = cast(
            "Any",
            Qwen3OmniMoeProcessor.from_pretrained(
                config.model_path,
                local_files_only=config.local_files_only,
            ),
        )
        self.processor.tokenizer.padding_side = "left"
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        model_config = Qwen3OmniMoeConfig.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
        )
        model_config.enable_audio_output = False
        device_map: str | dict[str, str] = config.device_map if device is None else {"": str(device)}
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            config.model_path,
            config=model_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=config.attention_implementation,
            local_files_only=config.local_files_only,
        ).eval()

    def transcribe(
        self,
        candidates: Sequence[SpeechCandidate],
        references: dict[str, Reference],
    ) -> dict[str, SpeechTranscription]:
        """Transcribe continuations and prompts whose reference alignment is unusable."""
        continuations = self._transcribe_segments(candidates, references, prompt=False)
        fallback_candidates = [
            candidate
            for candidate in candidates
            if not references[candidate.id].aligned or "<unk>" in references[candidate.id].prompt_text
        ]
        fallback_prompts = self._transcribe_segments(fallback_candidates, references, prompt=True)
        return {
            candidate.id: SpeechTranscription(
                id=candidate.id,
                prompt_text=fallback_prompts.get(candidate.id, references[candidate.id].prompt_text),
                continuation_text=continuations[candidate.id],
                prompt_source="asr" if candidate.id in fallback_prompts else "reference",
            )
            for candidate in candidates
        }

    def _transcribe_segments(
        self,
        candidates: Sequence[SpeechCandidate],
        references: dict[str, Reference],
        *,
        prompt: bool,
    ) -> dict[str, str]:
        transcripts: dict[str, str] = {}
        starts = range(0, len(candidates), self.config.batch_size)
        description = "Qwen3 prompt ASR" if prompt else "Qwen3 continuation ASR"
        for start in tqdm(starts, desc=description):
            batch = candidates[start : start + self.config.batch_size]
            waveforms = [
                _audio_segment(candidate, references[candidate.id], self.sample_rate, prompt=prompt)
                for candidate in batch
            ]
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "audio"},
                            {"type": "text", "text": self.config.prompt},
                        ],
                    }
                ]
                for _ in batch
            ]
            prompts = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(text=prompts, audio=waveforms, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_do_sample=False,
                    thinker_max_new_tokens=self.config.max_new_tokens,
                )
            sequences = _generated_sequences(generated)
            decoded = self.processor.batch_decode(
                sequences[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for candidate, transcript in zip(batch, decoded, strict=True):
                transcripts[candidate.id] = " ".join(transcript.strip().split())
        return transcripts
