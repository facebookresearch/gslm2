"""Deduplicate token sequences while retaining full-resolution frame indices.

Used by the multi-stream OPT to build the deduplicated single-stream "teacher" view that
the multi-stream student is distilled towards. Ported from the fairseq2 unit-LM stack
; pure PyTorch, no fairseq2 dependency.
"""

from __future__ import annotations

import torch


@torch._dynamo.disable  # data-dependent .item()/loop: run eager so it doesn't fragment a compiled graph  # noqa: SLF001
@torch.no_grad()
def build_dedup_view(
    input_seqs: torch.Tensor,
    seq_lens: torch.Tensor,
    pad_idx: int | None,
    *,
    doc_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return deduplicated rows, their lengths, each input frame's dedup index, and (packed only)
    the per-dedup-position document ids.

    ``dedup_idx_per_t[b, t]`` is the position in the deduplicated row that input frame ``(b, t)`` maps
    to (i.e. how many run-changes preceded it), which lets a per-frame teacher signal be gathered back
    onto the full-resolution timeline. When ``doc_ids`` is given (packed batches), a dedup run never
    spans a document boundary, and ``dedup_doc_ids`` carries each dedup position's document id (``-1``
    padding) so the teacher pass can build a block-diagonal attention layout.
    """
    bsz, seq_len = input_seqs.shape
    device = input_seqs.device

    first_token_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    first_token_mask[:, 0] = True
    if seq_len > 1:
        first_token_mask[:, 1:] = input_seqs[:, 1:] != input_seqs[:, :-1]
        if doc_ids is not None:  # never let a run span a document boundary
            first_token_mask[:, 1:] |= doc_ids[:, 1:] != doc_ids[:, :-1]

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    non_pad_mask = positions < seq_lens.unsqueeze(1)
    is_change_valid = first_token_mask & non_pad_mask

    dedup_idx_per_t = torch.cumsum(is_change_valid.long(), dim=1) - 1
    dedup_idx_per_t = dedup_idx_per_t.clamp(min=0)

    dedup_lens = is_change_valid.sum(dim=1)
    max_k = int(dedup_lens.max().item()) if bsz > 0 else 0
    max_k = max(max_k, 1)

    dedup_seqs = torch.full((bsz, max_k), pad_idx, dtype=input_seqs.dtype, device=device)
    dedup_doc_ids = torch.full((bsz, max_k), -1, dtype=torch.long, device=device) if doc_ids is not None else None

    for b in range(bsz):
        length = int(seq_lens[b].item())
        if length == 0:
            continue
        row_change = is_change_valid[b, :length]
        dedup_row = input_seqs[b, :length][row_change]
        dedup_seqs[b, : dedup_row.shape[0]] = dedup_row
        if dedup_doc_ids is not None:
            dedup_doc_ids[b, : dedup_row.shape[0]] = doc_ids[b, :length][row_change]

    return dedup_seqs, dedup_lens, dedup_idx_per_t, dedup_doc_ids
