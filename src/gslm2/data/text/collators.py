"""Text collators for causal-LM training: plain padding and causal shift."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TextSample:
    input_ids: Tensor


class TextCollator:
    """Pads variable-length samples into ``(input_ids, attention_mask)``."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[TextSample]) -> tuple[Tensor, Tensor]:
        lengths = [s.input_ids.size(0) for s in samples]
        max_len = max(lengths)
        batch = torch.full((len(samples), max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, s in enumerate(samples):
            n = s.input_ids.size(0)
            batch[i, :n] = s.input_ids
            attn[i, :n] = 1
        return batch, attn


_IGNORE_INDEX = -100


class CausalCollator:
    """Decoder-only collator: builds ``(input_ids, attention_mask, targets)``.

    ``targets`` is ``input_ids`` shifted left by one with ``pad_token_id`` at
    the last position.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[TextSample]) -> tuple[Tensor, Tensor, Tensor]:
        ids, attn = TextCollator(self.pad_token_id)(samples)
        # Next-token targets. A target position is supervised only if its target
        # token (ids[:, i+1]) is real; padding targets and the final position are
        # set to _IGNORE_INDEX so the loss ignores them (matches the packed path
        # and HF, and avoids training the model to predict pad/eos on padding).
        targets = torch.full_like(ids, _IGNORE_INDEX)
        targets[:, :-1] = ids[:, 1:].masked_fill(attn[:, 1:] == 0, _IGNORE_INDEX)
        return ids, attn, targets


def _greedy_pack(samples: list[TextSample], max_length: int) -> list[list[Tensor]]:
    """Greedy sequential bin packing. Returns list of bins, each a list of token tensors."""
    bins: list[list[Tensor]] = [[]]
    bin_lengths: list[int] = [0]
    for s in samples:
        ids = s.input_ids
        length = ids.size(0)
        if length > max_length:
            ids = ids[:max_length]
            length = max_length
        if bin_lengths[-1] + length > max_length:
            bins.append([])
            bin_lengths.append(0)
        bins[-1].append(ids)
        bin_lengths[-1] += length
    return bins


def _build_packed_tensors(
    bins: list[list[Tensor]],
    max_length: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build packed_ids, doc_ids, and position_ids from packed bins."""
    B = len(bins)  # noqa: N806
    packed_ids = torch.full((B, max_length), pad_token_id, dtype=torch.long)
    doc_ids = torch.full((B, max_length), -1, dtype=torch.long)
    position_ids = torch.zeros((B, max_length), dtype=torch.long)
    for b, docs in enumerate(bins):
        offset = 0
        for d, doc in enumerate(docs):
            length = doc.size(0)
            packed_ids[b, offset : offset + length] = doc
            doc_ids[b, offset : offset + length] = d
            position_ids[b, offset : offset + length] = torch.arange(length)
            offset += length
    return packed_ids, doc_ids, position_ids


class PackedCausalCollator:
    """Packs multiple samples into sequences with document boundaries.

    Returns ``(packed_ids, doc_ids, position_ids, targets)`` — a 4-tuple
    that signals packed mode to the trainer. ``doc_ids`` encodes per-token
    document membership for ``flex_attention`` ``BlockMask`` construction.
    """

    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, samples: list[TextSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bins = _greedy_pack(samples, self.max_length)
        packed_ids, doc_ids, position_ids = _build_packed_tensors(
            bins,
            self.max_length,
            self.pad_token_id,
        )
        targets = torch.full_like(packed_ids, _IGNORE_INDEX)
        for b, docs in enumerate(bins):
            offset = 0
            for doc in docs:
                length = doc.size(0)
                if length > 1:
                    targets[b, offset : offset + length - 1] = doc[1:]
                offset += length
        return packed_ids, doc_ids, position_ids, targets
