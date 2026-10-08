"""Causal-LM sequence scoring: per-sequence raw-sum and length-normalized (pseudo-)log-likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def _calc_nll(logits: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Per-sequence NLL (summed over ``mask`` positions) and its length-normalized form."""
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
    masked = losses.view(target.shape) * mask
    nll = masked.sum(dim=-1)
    # clamp the denominator so a length-<=1 sequence (no scored positions) yields 0, not NaN
    return nll, nll / mask.sum(dim=-1).clamp(min=1)


@torch.inference_mode()
def sequence_log_likelihood(model: Module, token_seqs: list[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """Log-likelihood (raw sum and per-token) of each token sequence under the causal LM.

    Right-pads with PAD=0 and masks the pad positions out of the loss; causal attention makes
    trailing pad inert for the real positions, so no attention mask is needed. ``model(tokens)``
    must return logits ``[B, T, vocab]`` (the no-targets forward of the in-house causal LMs).
    """
    seq_lens = torch.tensor([seq.numel() for seq in token_seqs], device=device)
    padded = pad_sequence([seq.to(device) for seq in token_seqs], batch_first=True, padding_value=0)
    logits = model(padded)
    shifted_logits, shifted_target = logits[:, :-1, :], padded[:, 1:]
    max_len = shifted_target.size(1)
    mask = (torch.arange(max_len, device=device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)).float()
    nll, nll_norm = _calc_nll(shifted_logits, shifted_target, mask)
    return -nll, -nll_norm


@torch.inference_mode()
def multistream_sequence_log_likelihood(
    model: Module, stream_seqs: list[Tensor], device: torch.device, *, dedup_aware: bool = True
) -> tuple[Tensor, Tensor]:
    """Cross-stream (pseudo-)log-likelihood of aligned multi-stream unit sequences under MultiStreamOPT.

    Each element of ``stream_seqs`` is an ``[S, L]`` tensor of ``S`` aligned streams; ``model(streams)``
    with ``streams`` of shape ``[B, S, T]`` returns a list of per-stream logits ``[B, T, V_s]``. The
    per-stream next-token NLLs are summed and averaged over streams (raw-sum and per-token forms).

    ``dedup_aware`` mirrors the fairseq2 multi-stream eval: only frames where a stream's token changes
    are scored (with a fallback to all valid frames when a row has no change), matching the change-frame
    objective the multi-stream student is trained on.
    """
    num_streams = stream_seqs[0].size(0)
    seq_lens = torch.tensor([s.size(1) for s in stream_seqs], device=device)
    max_t = int(seq_lens.max())
    padded = torch.zeros(len(stream_seqs), num_streams, max_t, dtype=torch.long, device=device)
    for i, seq in enumerate(stream_seqs):
        padded[i, :, : seq.size(1)] = seq.to(device)

    per_stream_logits = model(padded)
    length_mask = torch.arange(max_t - 1, device=device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)
    total_ll = torch.zeros(len(stream_seqs), device=device)
    total_norm = torch.zeros(len(stream_seqs), device=device)
    for stream in range(num_streams):
        logits = per_stream_logits[stream][:, :-1, :]
        target = padded[:, stream, 1:]
        if dedup_aware:
            change = (target != padded[:, stream, :-1]) & length_mask
            has_change = change.any(dim=1, keepdim=True)
            mask = torch.where(has_change, change, length_mask).float()
        else:
            mask = length_mask.float()
        nll, nll_norm = _calc_nll(logits, target, mask)
        total_ll += -nll
        total_norm += -nll_norm
    return total_ll / num_streams, total_norm / num_streams
