# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TextLM continuation decoding for generative evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from gslm2.eval.generative.manifest import PromptBoundaries, record_id
from gslm2.lm.inference import DecodingOptions, LoadedOptModel, decode_tokens, load_opt_model

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from gslm2.eval.generative.config import SamplingConfig, TextGenerationConfig
    from gslm2.shared.distributed import DistributedContext


class TextGenerator:
    """Generate text continuations from a fairseq2 OPT language model."""

    def __init__(
        self,
        config: TextGenerationConfig,
        sampling: SamplingConfig,
        device: torch.device,
        *,
        context: DistributedContext | None = None,
        checkpoint_cache: Path | None = None,
    ) -> None:
        from tokenizers import decoders, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        self.config = config
        self.sampling = sampling
        self.device = device
        self.boundaries = PromptBoundaries(config.prompt_boundary_manifest)
        if config.model_checkpoint is None and not config.oracle:
            raise ValueError("text.model_checkpoint is required for decoding")
        self.loaded_model: LoadedOptModel | None = (
            None
            if config.oracle
            else load_opt_model(
                cast("Path", config.model_checkpoint),
                device,
                model_config=config.model_config,
                context=context,
                checkpoint_cache=checkpoint_cache,
            )
        )
        if self.loaded_model is not None and self.loaded_model.is_multi_stream:
            raise ValueError("text generation requires a single-stream checkpoint")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(config.tokenizer_path))
        backend = self.tokenizer.backend_tokenizer
        if backend.decoder is None:
            if not isinstance(backend.pre_tokenizer, pre_tokenizers.ByteLevel):
                raise ValueError("tokenizer has no decoder and does not use ByteLevel pre-tokenization")
            backend.decoder = decoders.ByteLevel()

    def _decode(self, token_ids: list[int], *, drop_first: bool = False) -> str:
        if drop_first:
            token_ids = token_ids[1:]
        text = self.tokenizer.backend_tokenizer.decode(token_ids, skip_special_tokens=False)
        return " ".join(text.replace("</s>", " ").split())

    def _encode(self, text: str) -> list[int]:
        return cast(
            "list[int]",
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            ),
        )

    def _prompt_tokens(self, row: Mapping[str, object], text: str, tokens: list[int], row_id: str) -> list[int]:
        prompt_field = self.config.prompt_text_field
        if prompt_field is not None:
            prompt_value = row.get(prompt_field)
            if not isinstance(prompt_value, str) or not prompt_value:
                raise ValueError(f"{row_id} has no non-empty {prompt_field!r} prompt")
            prompt_tokens = self._encode(prompt_value)
            if tokens[: len(prompt_tokens)] != prompt_tokens:
                raise ValueError(f"{row_id} explicit prompt is not a token prefix of {self.config.text_field!r}")
            return prompt_tokens

        prompt_word_count = self.boundaries.prompt_word_count(row_id)
        if prompt_word_count is None:
            return tokens[: self.config.prompt_tokens]
        prefix_text = " ".join(text.split()[:prompt_word_count])
        prefix_ids = self._encode(prefix_text)
        common_prefix_length = 0
        for full_id, prefix_id in zip(tokens, prefix_ids, strict=False):
            if full_id != prefix_id:
                break
            common_prefix_length += 1
        return tokens[: (common_prefix_length or self.config.prompt_tokens)]

    @torch.inference_mode()
    def generate(self, row: Mapping[str, object], example_index: int) -> dict[str, object]:
        """Generate one decoded prompt/continuation pair."""
        row_id = record_id(row)
        text_value = row.get(self.config.text_field)
        if not isinstance(text_value, str) or not text_value:
            raise ValueError(f"{row_id} has no non-empty {self.config.text_field!r} text")
        tokens = self._encode(text_value)
        prompt_tokens = self._prompt_tokens(row, text_value, tokens, row_id)
        prompt_text = self._decode(prompt_tokens, drop_first=True)

        if self.config.oracle:
            generated_ids = tokens[: len(prompt_tokens) + self.config.continuation_tokens]
        else:
            if self.loaded_model is None:
                raise RuntimeError("text model was not initialized")
            sequences = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            sequences += self.loaded_model.unit_offset
            torch.manual_seed(self.sampling.seed + example_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.sampling.seed + example_index)
            sequences = decode_tokens(
                self.loaded_model,
                sequences,
                self.config.continuation_tokens,
                options=DecodingOptions(self.sampling.temperature, self.sampling.top_p),
            )
            generated_ids = cast("list[int]", (sequences[0] - self.loaded_model.unit_offset).tolist())

        continuation_ids = generated_ids[len(prompt_tokens) :]
        return {
            "index": example_index,
            "fileid": row_id,
            "file_name": str(row.get("audio", row.get("file_name"))),
            "prompt_text": prompt_text,
            "continuation_text": self._decode(continuation_ids),
            "generated_text": self._decode(generated_ids, drop_first=True),
        }
