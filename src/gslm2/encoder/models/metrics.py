"""Perplexity and entropy metrics for speech encoder training."""

import torch
import torch.compiler
import torch.distributed as dist
from torch import Tensor

from gslm2.shared.environment import pg_ddp


@torch.no_grad()
@torch.compiler.disable
def compute_ppl(y: Tensor, *, input_onehot: bool = False, tokenwise: bool = False) -> Tensor:
    if not input_onehot:
        y = y.softmax(dim=-1)
    if tokenwise:
        y = torch.exp2((-y * (y + 1e-8).log2()).sum(-1))
    if dist.is_initialized():
        n = torch.tensor([y.size(0)], device=y.device)
        y = y.sum(0)
        dist.all_reduce(n, group=pg_ddp())
        dist.all_reduce(y, group=pg_ddp())
        y = y / n
    else:
        y = y.mean(0)
    if not tokenwise:
        y = torch.exp2((-y * (y + 1e-8).log2()).sum())
    return y


def compute_entropy(probs: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
    """Compute predictive and codebook entropy from probability distributions.

    Args:
        probs: Probability tensor (..., vocab_size)
        log_probs: Log-probability tensor (..., vocab_size)

    Returns:
        (predictive_entropy, codebook_entropy) tuple where:
        - predictive_entropy: H(P(Z|Y)) = -E[sum(p * log(p))], mean entropy per sample
        - codebook_entropy: H(E[P]) = -sum(avg_p * log(avg_p)), entropy of average distribution
    """
    # Predictive: average entropy per sample
    predictive = -torch.sum(probs * log_probs, dim=-1).mean()
    # Codebook: entropy of the mean distribution (encourages uniform usage)
    avg_probs = probs.mean(dim=0)
    codebook = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
    return predictive, codebook
