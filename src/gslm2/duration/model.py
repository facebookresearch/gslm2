# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Conformer and Transformer duration-predictor models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

_MULTISTREAM_INPUT_NDIM = 3
_SINGLE_STREAM_INPUT_NDIM = 2


@dataclass(frozen=True)
class DurationPredictorModelConfig:
    """Architecture options for a duration predictor."""

    vocab_size: int = 256
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_streams: int = 1
    num_classes: int = 0
    encoder_type: str = "conformer"
    attention_heads: int = 4
    conv_kernel: int = 17
    ffn_expansion: int = 4
    max_seq_len: int = 4096


def _resolve_model_config(
    config: DurationPredictorModelConfig | None,
    overrides: dict[str, object],
) -> DurationPredictorModelConfig:
    if config is not None and overrides:
        raise ValueError("pass either a model config or keyword overrides, not both")
    if not overrides:
        return config or DurationPredictorModelConfig()

    values = asdict(DurationPredictorModelConfig())
    unknown = overrides.keys() - values.keys()
    if unknown:
        raise TypeError(f"unknown duration-predictor options: {', '.join(sorted(unknown))}")
    values.update(overrides)
    return DurationPredictorModelConfig(**cast("dict[str, Any]", values))


def _make_key_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Return a ``(batch, time)`` mask that is true at padded positions."""
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def decode_duration_predictions(prediction: Tensor, *, log_duration: bool, num_classes: int) -> Tensor:
    """Decode regression values or class logits into integer frame durations."""
    if num_classes > 0:
        probabilities = prediction.softmax(dim=-1)
        bins = torch.arange(1, num_classes + 1, device=prediction.device, dtype=probabilities.dtype)
        durations = (probabilities * bins).sum(dim=-1).round()
    else:
        durations = (prediction.exp() - 1).round() if log_duration else prediction.round()
    return durations.long().clamp(min=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position embeddings to batch-first sequences."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        frequencies = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[: encoding[:, 1::2].size(1)])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Add encodings to a ``(batch, time, dim)`` tensor."""
        encoding = self.get_buffer("encoding")
        if inputs.size(1) > encoding.size(0):
            raise ValueError(f"sequence length {inputs.size(1)} exceeds configured maximum {encoding.size(0)}")
        return inputs + encoding[: inputs.size(1)].unsqueeze(0)


class ConformerConvModule(nn.Module):
    """Depthwise convolution module used by a Conformer block."""

    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel size must be odd")
        self.ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_ln = nn.LayerNorm(dim)
        self.swish = nn.SiLU()
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Transform a ``(batch, time, dim)`` sequence while masking padding."""
        residual = inputs
        outputs = self.ln(inputs).transpose(1, 2)
        outputs = self.pw1(outputs)
        outputs = self.glu(outputs)
        outputs = outputs.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        outputs = self.dw(outputs)
        outputs = self.conv_ln(outputs.transpose(1, 2)).transpose(1, 2)
        outputs = self.swish(outputs)
        outputs = self.pw2(outputs)
        outputs = self.dropout(outputs)
        return residual + outputs.transpose(1, 2)


class FeedForwardModule(nn.Module):
    """Pre-normalized feed-forward network used by encoder blocks."""

    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.swish = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim * expansion, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform a ``(batch, time, dim)`` tensor."""
        outputs = self.ln(inputs)
        outputs = self.lin1(outputs)
        outputs = self.swish(outputs)
        outputs = self.drop1(outputs)
        outputs = self.lin2(outputs)
        return self.drop2(outputs)


class MultiHeadSelfAttentionModule(nn.Module):
    """Pre-normalized multi-head self-attention with a residual connection."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Attend over a ``(batch, time, dim)`` tensor without attending to padding."""
        normalized = self.ln(inputs)
        attended, _ = self.mha(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return inputs + self.dropout(attended)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer encoder block."""

    def __init__(self, dim: int, heads: int, conv_kernel: int, dropout: float, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, heads, dropout)
        self.conv = ConformerConvModule(dim, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.post_ln = nn.LayerNorm(dim)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Encode a ``(batch, time, dim)`` tensor."""
        outputs = inputs + 0.5 * self.ffn1(inputs)
        outputs = self.mhsa(outputs, key_padding_mask)
        outputs = self.conv(outputs, key_padding_mask)
        outputs = outputs + 0.5 * self.ffn2(outputs)
        return self.post_ln(outputs)


class TransformerBlock(nn.Module):
    """Macaron-style Transformer encoder block."""

    def __init__(self, dim: int, heads: int, dropout: float, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, heads, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.post_ln = nn.LayerNorm(dim)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Encode a ``(batch, time, dim)`` tensor."""
        outputs = inputs + 0.5 * self.ffn1(inputs)
        outputs = self.mhsa(outputs, key_padding_mask)
        outputs = outputs + 0.5 * self.ffn2(outputs)
        return self.post_ln(outputs)


class DurationPredictor(nn.Module):
    """Predict one duration per deduplicated single- or multi-stream unit frame."""

    def __init__(
        self,
        config: DurationPredictorModelConfig | None = None,
        **overrides: object,
    ) -> None:
        super().__init__()
        config = _resolve_model_config(config, overrides)
        if config.encoder_type not in {"conformer", "transformer"}:
            raise ValueError(f"unknown encoder_type: {config.encoder_type}")
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.num_streams = config.num_streams
        self.num_classes = config.num_classes
        self.encoder_type = config.encoder_type
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.embed_dim) for _ in range(config.num_streams)]
        )

        input_dim = config.embed_dim * config.num_streams
        self.input_proj = nn.Linear(input_dim, config.hidden_dim) if input_dim != config.hidden_dim else nn.Identity()
        self.pos_enc = SinusoidalPositionalEncoding(config.hidden_dim, max_len=config.max_seq_len)
        self.blocks = (
            nn.ModuleList(
                [
                    ConformerBlock(
                        config.hidden_dim,
                        config.attention_heads,
                        config.conv_kernel,
                        config.dropout,
                        config.ffn_expansion,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            if config.encoder_type == "conformer"
            else nn.ModuleList(
                [
                    TransformerBlock(
                        config.hidden_dim,
                        config.attention_heads,
                        config.dropout,
                        config.ffn_expansion,
                    )
                    for _ in range(config.num_layers)
                ]
            )
        )

        output_dim = config.num_classes if config.num_classes > 0 else 1
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_dim),
        )

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        """Predict duration logits from ``(B,T)`` or ``(B,S,T)`` unit IDs."""
        if self.num_streams > 1:
            if tokens.ndim != _MULTISTREAM_INPUT_NDIM or tokens.size(1) != self.num_streams:
                raise ValueError(f"expected tokens with shape (batch, {self.num_streams}, time)")
            embedded = [embedding(tokens[:, index]) for index, embedding in enumerate(self.embeddings)]
            outputs = torch.cat(embedded, dim=-1)
        else:
            if tokens.ndim != _SINGLE_STREAM_INPUT_NDIM:
                raise ValueError("expected tokens with shape (batch, time)")
            outputs = self.embeddings[0](tokens)

        outputs = self.input_proj(outputs)
        outputs = self.pos_enc(outputs)
        key_padding_mask = _make_key_padding_mask(lengths, outputs.size(1))
        for block in self.blocks:
            outputs = block(outputs, key_padding_mask)

        predictions = self.proj(outputs)
        return predictions if self.num_classes > 0 else predictions.squeeze(-1)

    @torch.no_grad()
    def predict(self, tokens: Tensor, lengths: Tensor, *, log_duration: bool = True) -> Tensor:
        """Return integer frame durations for a padded batch of unit sequences."""
        self.eval()
        output = self.forward(tokens, lengths)
        return decode_duration_predictions(output, log_duration=log_duration, num_classes=self.num_classes)
