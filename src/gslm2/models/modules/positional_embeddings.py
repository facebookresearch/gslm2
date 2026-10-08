"""Sinusoidal positional embeddings (1D, 2D, 3D).

Shared across vision and world modeling models.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int],
    grid_depth: int,
    *,
    uniform_power: bool = False,
) -> np.ndarray:
    """3D sinusoidal positional embedding for video (depth, height, width)."""
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    return pos_embed[:, :embed_dim]


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> np.ndarray:
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    return np.concatenate([emb_h, emb_w], axis=1)


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


class FactorizedPositionalEmbedding(nn.Module):
    """Per-axis positional embedding with concat-split feature layout.

    Splits ``embed_dim`` evenly across the tokenizer's axes, builds one
    ``[grid_shape[i], embed_dim // axes]`` table per axis, looks each one up
    with its coordinate tensor, and concatenates along the feature dim. This
    matches the convention of :func:`get_2d_sincos_pos_embed` /
    :func:`get_3d_sincos_pos_embed` (each axis owns disjoint feature
    channels), rather than the sum-over-axes / shared-channel scheme.

    For 1D (waveform) this degenerates to a single full-dim table.

    Modes:
        - ``"sinusoidal"``: fixed per-axis sin/cos table stored as a buffer
          (not in ``model.parameters()``, no grad).
        - ``"absolute_learned"``: per-axis ``nn.Parameter`` initialised
          ``trunc_normal_(std=0.02)``.
    """

    def __init__(
        self,
        embed_dim: int,
        grid_shape: tuple[int, ...],
        mode: Literal["sinusoidal", "absolute_learned"],
    ) -> None:
        super().__init__()
        if not grid_shape:
            raise ValueError("grid_shape must be non-empty")
        axes = len(grid_shape)
        if embed_dim % axes != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by axes ({axes})")
        per_axis_dim = embed_dim // axes
        if per_axis_dim % 2 != 0:
            raise ValueError(f"per-axis dim (embed_dim/axes = {per_axis_dim}) must be even for sincos init")
        if mode not in ("sinusoidal", "absolute_learned"):
            raise ValueError(f"mode must be 'sinusoidal' or 'absolute_learned'; got {mode!r}")

        self.embed_dim = embed_dim
        self.grid_shape = tuple(grid_shape)
        self.per_axis_dim = per_axis_dim
        self.mode = mode
        self.num_axes = axes

        if mode == "sinusoidal":
            # Buffers: move with .to(device) / state_dict, but stay out of
            # model.parameters() so optimizer/weight-decay logic doesn't see
            # them.
            for i, axis_len in enumerate(self.grid_shape):
                arr = get_1d_sincos_pos_embed_from_grid(per_axis_dim, np.arange(axis_len, dtype=float))
                self.register_buffer(f"table_{i}", torch.from_numpy(arr).float(), persistent=True)
            self.tables: nn.ParameterList | None = None
        else:
            params: list[nn.Parameter] = []
            for axis_len in self.grid_shape:
                p = nn.Parameter(torch.zeros(axis_len, per_axis_dim))
                nn.init.trunc_normal_(p, std=0.02)
                params.append(p)
            self.tables = nn.ParameterList(params)

    def _axis_table(self, i: int) -> torch.Tensor:
        if self.tables is not None:
            return self.tables[i]
        return getattr(self, f"table_{i}")

    def forward(self, positions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Concatenate per-axis lookups along the feature dim.

        Args:
            positions: tuple of ``[B, K]`` long tensors, one per axis. Each
                index must be in ``[0, grid_shape[i])``.

        Returns:
            ``[B, K, embed_dim]`` positional embedding.
        """
        if len(positions) != self.num_axes:
            raise ValueError(f"positions must have {self.num_axes} entries; got {len(positions)}")
        parts = [self._axis_table(i)[idx.long()] for i, idx in enumerate(positions)]
        return torch.cat(parts, dim=-1)
