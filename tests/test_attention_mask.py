"""Regression tests for ``build_attention_mask``.

These pin down the two guarantees the function makes:

  1. It returns ``None`` when the batch has no padding, so that
     ``F.scaled_dot_product_attention`` can dispatch to the Flash Attention backend
     (an explicit, all-True mask would force the slower math backend).
  2. When the batch *does* have padding it returns the expected boolean mask
     (shape ``(batch, 1, max_len, max_len)``, ``True`` for valid positions), and this
     is numerically equivalent to feeding the network an explicit all-True mask on the
     unpadded case.
"""
# ruff: noqa: INP001  Test module; tests/ intentionally has no __init__.py.

import unittest

import torch
from torch.nn import functional as F

from gslm2.data.common.masking.speech import build_attention_mask, build_padding_mask
from gslm2.models.speech.components import SelfAttention


def _explicit_full_mask(lengths: torch.Tensor) -> torch.Tensor:
    """The pre-optimization behaviour: always materialise the boolean mask."""
    batch_size, max_len = lengths.size(0), int(lengths.max())
    padding_mask = build_padding_mask(lengths)
    return ~padding_mask[:, None, None, :].expand(batch_size, 1, max_len, max_len)


class TestBuildAttentionMask(unittest.TestCase):
    def test_returns_none_without_padding(self) -> None:
        # All sequences already span max_len -> nothing to mask -> None (enables Flash).
        self.assertIsNone(build_attention_mask(torch.tensor([100, 100, 100])))

    def test_single_sequence_returns_none(self) -> None:
        # A batch of one is never padded.
        self.assertIsNone(build_attention_mask(torch.tensor([42])))

    def test_returns_mask_with_padding(self) -> None:
        lengths = torch.tensor([100, 95, 90])
        mask = build_attention_mask(lengths)
        assert mask is not None
        self.assertEqual(mask.shape, (3, 1, 100, 100))
        self.assertEqual(mask.dtype, torch.bool)
        # True exactly on valid (non-padding) key positions.
        expected = ~build_padding_mask(lengths)  # (batch, max_len)
        torch.testing.assert_close(mask[:, 0, 0, :], expected)

    def test_matches_explicit_mask_when_padded(self) -> None:
        # With padding, the optimized function must return the exact same tensor
        # the old unconditional implementation produced.
        lengths = torch.tensor([100, 95])
        torch.testing.assert_close(build_attention_mask(lengths), _explicit_full_mask(lengths))

    @torch.no_grad()
    def test_selfattention_none_equivalent_to_full_mask(self) -> None:
        # For an unpadded batch, passing None (Flash-friendly) must give the same
        # attention output as passing the redundant all-True mask (math backend).
        torch.manual_seed(0)
        attention = SelfAttention(embed_dim=64, num_heads=8, qkv_bias=True).eval()
        x = torch.randn(3, 20, 64)
        lengths = torch.tensor([20, 20, 20])

        self.assertIsNone(build_attention_mask(lengths))
        out_none = attention(x, attention_mask=None)
        out_full = attention(x, attention_mask=_explicit_full_mask(lengths))
        torch.testing.assert_close(out_none, out_full)

    @torch.no_grad()
    def test_sdpa_none_equivalent_to_full_mask(self) -> None:
        # Same equivalence at the raw scaled_dot_product_attention level.
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
        full = _explicit_full_mask(torch.tensor([16, 16]))
        out_none = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out_full = F.scaled_dot_product_attention(q, k, v, attn_mask=full)
        torch.testing.assert_close(out_none, out_full)


if __name__ == "__main__":
    unittest.main()
