"""Learned absolute position embeddings for language models."""

from __future__ import annotations

import torch
from torch import nn


class LearnedPositionEmbedding(nn.Module):
    """Learned absolute position embedding with an OPT-style pad offset.

    Adds ``pad_offset`` to all position indices before lookup so the first
    ``pad_offset`` rows of the embedding table can be reserved for padding /
    special tokens. Set ``pad_offset=0`` for vanilla learned positions.
    """

    def __init__(self, max_seq_len: int, dim: int, pad_offset: int = 0) -> None:
        super().__init__()
        self.pad_offset = pad_offset
        self.weight = nn.Embedding(max_seq_len + pad_offset, dim)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is not None:
            return self.weight(position_ids + self.pad_offset)
        positions = torch.arange(seq_len, device=device) + self.pad_offset
        return self.weight(positions)
