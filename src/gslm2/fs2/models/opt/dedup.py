# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deduplicate token sequences while retaining full-resolution frame indices."""

from __future__ import annotations

import torch


@torch.no_grad()
def build_dedup_view(  # noqa: C901
    input_seqs: torch.Tensor,
    seq_lens: torch.Tensor,
    pad_idx: int | None,
    *,
    doc_seq_lens: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]] | None]:
    """Return deduplicated rows, their lengths, and each input frame's dedup index.

    ``doc_seq_lens`` gives per-document frame lengths for packed rows.
    """
    bsz, seq_len = input_seqs.shape
    device = input_seqs.device

    first_token_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    first_token_mask[:, 0] = True
    if seq_len > 1:
        first_token_mask[:, 1:] = input_seqs[:, 1:] != input_seqs[:, :-1]

    # Packed: ensure a dedup run never spans a document boundary
    if doc_seq_lens is not None:
        for b, lens in enumerate(doc_seq_lens):
            if sum(lens) != seq_len:
                raise ValueError(f"doc_seq_lens[{b}] sums to {sum(lens)}, expected {seq_len}.")
            offset = 0
            for length in lens[:-1]:
                offset += length
                first_token_mask[b, offset] = True

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    non_pad_mask = positions < seq_lens.unsqueeze(1)
    is_change_valid = first_token_mask & non_pad_mask

    dedup_idx_per_t = torch.cumsum(is_change_valid.long(), dim=1) - 1
    dedup_idx_per_t = dedup_idx_per_t.clamp(min=0)

    dedup_lens = is_change_valid.sum(dim=1)
    max_k = int(dedup_lens.max().item()) if bsz > 0 else 0
    max_k = max(max_k, 1)

    dedup_seqs = torch.full((bsz, max_k), pad_idx, dtype=input_seqs.dtype, device=device)

    for b in range(bsz):
        length = int(seq_lens[b].item())
        if length == 0:
            continue
        row = input_seqs[b, :length]
        row_change = is_change_valid[b, :length]
        dedup_row = row[row_change]
        dedup_seqs[b, : dedup_row.shape[0]] = dedup_row

    # Packed only: per-document dedup lengths
    dedup_doc_lens: list[list[int]] | None = None
    if doc_seq_lens is not None:
        dedup_doc_lens = []
        for b, lens in enumerate(doc_seq_lens):
            row_lens = []
            offset = 0
            for length in lens:
                row_lens.append(int(is_change_valid[b, offset : offset + length].sum().item()))
                offset += length
            dedup_doc_lens.append(row_lens)

    return dedup_seqs, dedup_lens, dedup_idx_per_t, dedup_doc_lens
