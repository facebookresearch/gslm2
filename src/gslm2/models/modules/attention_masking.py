"""Attention mask construction utilities.

Provides ``build_block_causal_mask`` for block-diagonal causal masks consumed
by ``scaled_dot_product_attention``.  ``True`` means the position is *allowed*
to attend.

For token-selection masks (JEPA context/target splits) and multi-clip/multi-mask
wrappers, see :mod:`gslm2.models.modules.masking`.
"""

from __future__ import annotations

import torch


def build_block_causal_mask(num_frames: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    """Block-diagonal causal attention mask over flattened (T, S) tokens.

    Within a frame: full attention (all S tokens attend to each other). Across
    frames: tokens can attend to all tokens in past or current frames, not
    future ones.

    Args:
        num_frames: Number of temporal frames (T).
        tokens_per_frame: Number of spatial tokens per frame (S).
        device: Target device.

    Returns:
        Boolean mask of shape ``(1, 1, T*S, T*S)`` broadcastable over batch
        and heads. ``True`` means the position is *allowed* to attend.
    """
    total = num_frames * tokens_per_frame
    frame_ids = torch.arange(total, device=device) // tokens_per_frame  # (T*S,)
    mask = frame_ids.unsqueeze(1) >= frame_ids.unsqueeze(0)  # (T*S, T*S)
    return mask.unsqueeze(0).unsqueeze(0)


__all__ = [
    "build_block_causal_mask",
]
