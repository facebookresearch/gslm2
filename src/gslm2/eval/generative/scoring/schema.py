# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Typed records shared by the speech and text evaluation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Reference:
    """Reference prompt, continuation, and optional speech boundary for one item."""

    id: str
    prompt_text: str
    continuation_text: str
    prompt_end_seconds: float | None
    aligned: bool = True


@dataclass(frozen=True)
class SpeechCandidate:
    """Synthesized audio to evaluate."""

    id: str
    audio_path: Path


@dataclass(frozen=True)
class SpeechTranscription:
    """Qwen3 transcription of a speech candidate's prompt and continuation."""

    id: str
    prompt_text: str
    continuation_text: str
    prompt_source: str


@dataclass(frozen=True)
class TextCandidate:
    """Generated text and the exact prompt used to produce it."""

    id: str
    prompt_text: str
    continuation_text: str
    normalized: bool = False


@dataclass(frozen=True)
class ScoringExample:
    """A modality-independent prompt and candidate continuation."""

    id: str
    prompt_text: str
    continuation_text: str
    reference_continuation: str
    candidate_text_normalized: bool = False


@dataclass(frozen=True)
class PerplexityResult:
    """Conditional perplexity and the number of continuation tokens scored."""

    perplexity: float | None
    num_tokens: int


@dataclass(frozen=True)
class EvaluationRecord:
    """Per-example normalized text, conditional perplexity, and repetition rates."""

    id: str
    prompt_text: str
    continuation_text: str
    reference_continuation: str
    prompt_text_normalized: str
    continuation_text_normalized: str
    reference_continuation_normalized: str
    cond_genppl: float | None
    cond_genppl_num_tokens: int
    repetition_1: float
    repetition_2: float
    reference_repetition_1: float
    reference_repetition_2: float
    continuation_words: int
