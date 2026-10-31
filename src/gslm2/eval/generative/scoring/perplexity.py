# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batched conditional perplexity for generated continuations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F
from tqdm import tqdm

from gslm2.eval.generative.scoring.schema import PerplexityResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gslm2.eval.generative.scoring.config import PerplexityConfig

_IGNORE_INDEX = -100


def _dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def conditional_perplexities_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> list[PerplexityResult]:
    """Calculate per-row perplexities from causal logits and prompt-masked labels."""
    shifted_logits = logits[:, :-1].float().transpose(1, 2)
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits,
        shifted_labels,
        reduction="none",
        ignore_index=_IGNORE_INDEX,
    )
    valid = shifted_labels != _IGNORE_INDEX
    counts = valid.sum(dim=1)
    summed_losses = (token_losses * valid).sum(dim=1)
    results = []
    for loss, count in zip(summed_losses.tolist(), counts.tolist(), strict=True):
        results.append(
            PerplexityResult(
                perplexity=math.exp(loss / count) if count else None,
                num_tokens=count,
            )
        )
    return results


class ConditionalPerplexityScorer:
    """Score continuation tokens while using prompt tokens only as context."""

    def __init__(self, config: PerplexityConfig, device: torch.device | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            use_fast=True,
            local_files_only=config.local_files_only,
        )
        device_map: str | dict[str, str] = config.device_map if device is None else {"": str(device)}
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=_dtype(config.dtype),
            device_map=device_map,
            local_files_only=config.local_files_only,
        ).eval()

    def score(self, prompts: Sequence[str], continuations: Sequence[str]) -> list[PerplexityResult]:
        """Return conditional perplexity for aligned prompt and continuation strings."""
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must have equal length")
        results = []
        starts = range(0, len(prompts), self.config.batch_size)
        for start in tqdm(starts, desc="Conditional GenPPL"):
            results.extend(
                self._score_batch(
                    prompts[start : start + self.config.batch_size],
                    continuations[start : start + self.config.batch_size],
                )
            )
        return results

    @torch.inference_mode()
    def _score_batch(self, prompts: Sequence[str], continuations: Sequence[str]) -> list[PerplexityResult]:
        sequences = []
        labels = []
        for prompt, continuation in zip(prompts, continuations, strict=True):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            continuation_prefix = " " if prompt and continuation else ""
            continuation_ids = self.tokenizer.encode(
                f"{continuation_prefix}{continuation}",
                add_special_tokens=False,
            )
            token_ids = [*prompt_ids, *continuation_ids]
            if len(token_ids) > self.config.max_length:
                raise ValueError(
                    f"prompt and continuation encode to {len(token_ids)} tokens, "
                    f"exceeding max_length={self.config.max_length}"
                )
            sequences.append(token_ids)
            labels.append([_IGNORE_INDEX] * len(prompt_ids) + continuation_ids)

        if not sequences:
            return []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("perplexity tokenizer has neither a pad token nor an EOS token")

        max_length = max(map(len, sequences))
        input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        label_tensor = torch.full_like(input_ids, _IGNORE_INDEX)
        for row_index, (token_ids, token_labels) in enumerate(zip(sequences, labels, strict=True)):
            length = len(token_ids)
            input_ids[row_index, :length] = torch.tensor(token_ids)
            attention_mask[row_index, :length] = 1
            label_tensor[row_index, :length] = torch.tensor(token_labels)

        device = self.model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_tensor = label_tensor.to(device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return conditional_perplexities_from_logits(logits, label_tensor)
