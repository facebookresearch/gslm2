# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling helpers for autoregressive OPT decoding.

Provides a single :func:`sample_next_token` implementation shared by both the
multi-stream training/eval code (:mod:`gslm2.fs2.models.opt.multi_stream_model`)
and the offline decoding feature
(``scripts/dist_feature_processor/features/slm_decode/slm_decode.py``).
"""

from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, temperature: float, topp: float, topk: int = 0) -> torch.Tensor:
    """Sample the next token from a (possibly batched) logits tensor.

    Args:
        logits: Tensor of shape ``[..., V]`` with raw, un-normalised logits.
        temperature: Sampling temperature. ``0.0`` selects the argmax greedily.
        topp: Nucleus sampling threshold. Values ``>= 1.0`` sample from the full
            distribution. Ignored when ``topk > 0``.
        topk: Top-k sampling threshold. Values ``<= 0`` disable top-k sampling.
            When enabled, sampling is restricted to the ``topk`` most probable
            tokens instead of using ``topp``.

    Returns:
        Tensor with the same leading shape as ``logits`` (i.e., ``logits.shape[:-1]``)
        containing sampled token ids.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if topk > 0:
        k = min(topk, probs.size(-1))
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_topk_probs = topk_probs.reshape(-1, k)  # [N, K]
        flat_topk_indices = topk_indices.reshape(-1, k)  # [N, K]

        sampled_in_topk = torch.multinomial(flat_topk_probs, num_samples=1)  # [N, 1]
        sampled = torch.gather(flat_topk_indices, dim=-1, index=sampled_in_topk)  # [N, 1]

        return sampled.reshape(*logits.shape[:-1])

    if topp >= 1.0:
        flat_probs = probs.reshape(-1, probs.size(-1))  # [N, V]
        sampled = torch.multinomial(flat_probs, num_samples=1)  # [N, 1]
        return sampled.reshape(*probs.shape[:-1])

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens up to and including the first one whose cumulative probability exceeds topp
    keep_mask = cumulative_probs <= topp
    keep_mask[..., 0] = True
    keep_mask[..., 1:] |= keep_mask[..., :-1] ^ keep_mask[..., 1:]

    filtered_probs = sorted_probs * keep_mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    flat_filtered_probs = filtered_probs.reshape(-1, filtered_probs.size(-1))  # [N, V]
    flat_sorted_indices = sorted_indices.reshape(-1, sorted_indices.size(-1))  # [N, V]

    sampled_in_sorted = torch.multinomial(flat_filtered_probs, num_samples=1)  # [N, 1]
    sampled = torch.gather(flat_sorted_indices, dim=-1, index=sampled_in_sorted)  # [N, 1]

    return sampled.reshape(*logits.shape[:-1])
