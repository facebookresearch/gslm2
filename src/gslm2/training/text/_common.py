"""Shared plumbing for the text language-model trainers (causal + masked)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def prepare_text_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Move a text batch to ``device`` and unpack it into
    ``(ids, attention_mask, supervision, doc_ids, position_ids)``.

    The collators emit a 4-tuple ``(ids, doc_ids, position_ids, supervision)`` on the packed
    path (document-block attention, no padding mask) and a 3-tuple ``(ids, attention_mask,
    supervision)`` otherwise.
    """
    tensors = tuple(t.to(device, non_blocking=True) for t in batch)
    if len(tensors) == 4:  # noqa: PLR2004
        ids, doc_ids, position_ids, supervision = tensors
        attention_mask = None
    else:
        ids, attention_mask, supervision = tensors
        doc_ids = position_ids = None
    return ids, attention_mask, supervision, doc_ids, position_ids
