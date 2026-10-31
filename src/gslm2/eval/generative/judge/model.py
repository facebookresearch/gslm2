# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prometheus-2 inference for continuation judging."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from gslm2.eval.generative.judge.schema import JudgePrediction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gslm2.eval.generative.judge.config import JudgeModelConfig

_RESULT = re.compile(r"\[RESULT\]\s*(\d+)")
_MAX_SCORE = 5


def parse_score(feedback: str) -> int | None:
    """Parse the first result marker when it contains a valid 1-5 score."""
    match = _RESULT.search(feedback)
    if match is None:
        return None
    score = int(match.group(1))
    return score if 1 <= score <= _MAX_SCORE else None


class PrometheusJudge:
    """Run greedy Prometheus-2 generation on one device."""

    def __init__(self, config: JudgeModelConfig, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=getattr(torch, config.dtype),
            device_map={"": str(device)},
            local_files_only=config.local_files_only,
        ).eval()

    def score(self, prompts: Sequence[str]) -> list[JudgePrediction]:
        """Greedily generate feedback and parse one score per prompt."""
        return [self._score_one(prompt) for prompt in tqdm(prompts, desc="LLM judge")]

    @torch.inference_mode()
    def _score_one(self, prompt: str) -> JudgePrediction:
        inputs: Any = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        feedback = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not isinstance(feedback, str):
            raise TypeError("judge tokenizer must decode one feedback string")
        feedback = feedback.strip()
        return JudgePrediction(feedback=feedback, score=parse_score(feedback))
