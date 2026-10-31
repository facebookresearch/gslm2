# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Typed records used by LLM-judge evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeExample:
    """One raw prompt and generated continuation."""

    id: str
    prompt_text: str
    continuation_text: str


@dataclass(frozen=True)
class JudgePrediction:
    """Generated judge feedback and its parsed score."""

    feedback: str
    score: int | None


@dataclass(frozen=True)
class JudgeRecord:
    """One auditable per-example judge result."""

    id: str
    prompt_text: str
    continuation_text: str
    prompt_text_normalized: str
    continuation_text_normalized: str
    judge_feedback: str
    judge_score: int | None
