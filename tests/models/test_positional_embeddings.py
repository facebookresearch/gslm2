"""Unit tests for sinusoidal positional embeddings (1D, 2D, 3D)."""

import numpy as np
import pytest

from gslm2.models.modules.positional_embeddings import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)


class TestGet1dSincosPosEmbed:
    def test_output_shape(self) -> None:
        pos = np.arange(10, dtype=float)
        out = get_1d_sincos_pos_embed_from_grid(32, pos)
        assert out.shape == (10, 32)

    def test_values_bounded(self) -> None:
        pos = np.arange(50, dtype=float)
        out = get_1d_sincos_pos_embed_from_grid(64, pos)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_different_positions_differ(self) -> None:
        pos = np.arange(5, dtype=float)
        out = get_1d_sincos_pos_embed_from_grid(16, pos)
        # Each row should be unique
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                assert not np.allclose(out[i], out[j])

    def test_requires_even_dim(self) -> None:
        with pytest.raises(AssertionError):
            get_1d_sincos_pos_embed_from_grid(15, np.arange(3, dtype=float))


class TestGet3dSincosPosEmbed:
    def test_shape_uniform_power(self) -> None:
        emb = get_3d_sincos_pos_embed(48, (4, 4), 2, uniform_power=True)
        n_tokens = 2 * 4 * 4
        assert emb.shape == (n_tokens, 48)

    def test_shape_non_uniform(self) -> None:
        emb = get_3d_sincos_pos_embed(48, (4, 4), 2, uniform_power=False)
        n_tokens = 2 * 4 * 4
        assert emb.shape == (n_tokens, 48)

    def test_all_rows_unique(self) -> None:
        emb = get_3d_sincos_pos_embed(48, (3, 3), 2, uniform_power=True)
        n = emb.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(emb[i], emb[j]), f"Rows {i} and {j} are identical"

    def test_truncation_to_embed_dim(self) -> None:
        # With uniform_power the raw concat may exceed embed_dim; output is truncated.
        emb = get_3d_sincos_pos_embed(48, (4, 4), 2, uniform_power=True)
        assert emb.shape[1] == 48


class TestGet2dSincosPosEmbed:
    def test_shape(self) -> None:
        emb = get_2d_sincos_pos_embed(64, (4, 6))
        assert emb.shape == (4 * 6, 64)

    def test_all_rows_unique(self) -> None:
        emb = get_2d_sincos_pos_embed(32, (3, 4))
        n = emb.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(emb[i], emb[j]), f"Rows {i} and {j} are identical"

    def test_square_grid(self) -> None:
        emb = get_2d_sincos_pos_embed(16, (5, 5))
        assert emb.shape == (25, 16)
