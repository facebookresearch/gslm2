"""Attention and initialization utilities for language models."""

import math
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Callable[..., torch.Tensor]:
    # flex_attention runs slowly in eager mode (rebuilds kernels per call; PyTorch warns
    # to use torch.compile(flex_attention)). Compile once and cache the callable.
    return torch.compile(flex_attention)


def _dispatch_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: BlockMask,
    enable_gqa: bool,
) -> torch.Tensor:
    # Compiled flex_attention on CUDA; eager on CPU (keeps unit tests fast).
    fa = _compiled_flex_attention() if q.is_cuda else flex_attention
    return fa(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


def flex_backward_unsupported(device: torch.device) -> bool:
    """Whether flex_attention is unusable on device because a backward pass is possible.

    flex_attention has no CPU backward kernel, and it raises as soon as grad is enabled rather
    than at .backward(), so a packed path must build a dense mask for SDPA in that case.
    """
    return device.type == "cpu" and torch.is_grad_enabled()


def trunc_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Truncated normal initialization (from timm)."""

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class MultiheadSelfAttention(nn.Module):
    """Multihead self-attention with split q/k/v/out projections.

    Supports both bidirectional (``causal=False``) and causal attention,
    Grouped-Query Attention (``num_kv_heads < num_heads``), and optional
    local sliding-window causal masking.
    """

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
        attn_window_len: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        num_kv_heads = num_kv_heads or num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        if attn_window_len is not None and not causal:
            raise ValueError("attn_window_len is only meaningful with causal=True.")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_kv_heads
        self.causal = causal
        self.attn_window_len = attn_window_len
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _build_local_causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(n, device=device).unsqueeze(1)
        j = torch.arange(n, device=device).unsqueeze(0)
        return (j <= i) & ((i - j) < self.attn_window_len)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | BlockMask | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if isinstance(attention_mask, BlockMask):
            y = _dispatch_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=self.group_size > 1,
            )
        else:
            if attention_mask is None and self.causal and self.attn_window_len is not None:
                attention_mask = self._build_local_causal_mask(seq, x.device)
            is_causal = self.causal and attention_mask is None and self.attn_window_len is None
            dropout = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=is_causal,
                enable_gqa=self.group_size > 1,
            )

        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out_proj(y)
