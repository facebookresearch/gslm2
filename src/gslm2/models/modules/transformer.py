"""Shared transformer building blocks.

Provides the core transformer components (Block, RoPEAttention, MLP, DropPath,
trunc_normal_) used by both the vision models (VJEPA) and the world model.
"""

import math
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from gslm2.models.modules.mlp import MLP, SwiGLUFFN
from gslm2.models.speech.components import SelfAttention


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


class DropPath(nn.Module):
    """Stochastic depth (drop paths) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def _rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to queries or keys."""
    d = x.size(-1)
    omega = torch.arange(d // 2, dtype=x.dtype, device=x.device)
    omega /= d / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class RoPEAttention(nn.Module):
    """Multi-head attention with 3D Rotary Positional Embeddings (depth, height, width)."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        grid_size: int = 16,
        grid_depth: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # RoPE dimension splits: depth, height, width.
        # Each gets ~1/3 of head_dim, rounded down to even. Any leftover dims
        # (e.g. 4 out of 64) pass through without RoPE (see forward).
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.grid_depth = grid_depth

    def _separate_positions(
        self,
        ids: torch.Tensor,
        h_patches: int | None,
        w_patches: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = h_patches if h_patches is not None else self.grid_size
        w = w_patches if w_patches is not None else self.grid_size
        tokens_per_frame = h * w
        frame_ids = ids // tokens_per_frame
        spatial_ids = ids - tokens_per_frame * frame_ids
        height_ids = spatial_ids // w
        width_ids = spatial_ids - w * height_ids
        return frame_ids, height_ids, width_ids

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        T: int | None = None,  # noqa: N803
        H_patches: int | None = None,  # noqa: N803
        W_patches: int | None = None,  # noqa: N803
    ) -> torch.Tensor:
        b, n, c = x.size()

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build position indices.
        # When mask is provided, pos is (B, num_heads, N) with per-sample indices.
        # When mask is None (unmasked), pos is (N,) and broadcasts — all samples
        # share the same sequential positions, which is correct for full inputs.
        if mask is not None:
            pos = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        else:
            total = T * (H_patches or self.grid_size) * (W_patches or self.grid_size) if T else n
            pos = torch.arange(total, device=x.device)
        d_pos, h_pos, w_pos = self._separate_positions(pos, H_patches, W_patches)

        # Apply RoPE per spatial dimension
        s = 0
        qd = _rotate_queries_or_keys(q[..., s : s + self.d_dim], d_pos)
        kd = _rotate_queries_or_keys(k[..., s : s + self.d_dim], d_pos)
        s += self.d_dim
        qh = _rotate_queries_or_keys(q[..., s : s + self.h_dim], h_pos)
        kh = _rotate_queries_or_keys(k[..., s : s + self.h_dim], h_pos)
        s += self.h_dim
        qw = _rotate_queries_or_keys(q[..., s : s + self.w_dim], w_pos)
        kw = _rotate_queries_or_keys(k[..., s : s + self.w_dim], w_pos)
        s += self.w_dim

        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_prob if self.training else 0.0, attn_mask=attn_mask
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


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


class Block(nn.Module):
    """Pre-norm transformer block with optional N-D RoPE attention.

    Dispatches on ``axes``:

    - ``axes ∈ {1, 2}`` → uses :class:`RoPEAttentionND` (textbook RoPE).
      Forward: ``block(x, pos=(t_pos,))`` or ``block(x, pos=(t_pos, f_pos))``.
      Used by AJEPAWaveform and AJEPASpectrogram.
    - ``axes == 3`` (default) → uses :class:`RoPEAttention` (3D space-time RoPE
      with VJEPA's non-textbook rotation pattern; preserved for VJEPA / world
      model checkpoints). Forward: ``block(x, mask=..., attn_mask=..., T=...,
      H_patches=..., W_patches=...)``. Passes ``**rope_kwargs`` through.

    When ``use_rope=False``, falls back to plain :class:`SelfAttention` and
    accepts ``attn_mask=`` only.

    Defaults match the historical 3D ``Block`` (``axes=3``, ``use_rope=False``,
    ``qkv_bias=False``) so existing VJEPA / world-model call sites work
    unchanged.
    """

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int,
        axes: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        *,
        wide_silu: bool = True,
        use_sdpa: bool = True,
        is_causal: bool = False,
        use_rope: bool = False,
        # 3D-RoPE-specific kwargs (ignored unless axes == 3 and use_rope).
        grid_size: int = 16,
        grid_depth: int = 8,
    ) -> None:
        super().__init__()
        if axes not in (1, 2, 3):
            raise ValueError(f"axes must be 1, 2, or 3; got {axes}")
        if is_causal and axes != 1:
            raise ValueError("is_causal is only meaningful for axes=1 (temporal)")
        self.axes = axes
        self.norm1 = norm_layer(dim)
        if use_rope:
            if axes == 3:  # noqa: PLR2004
                self.attn: nn.Module = RoPEAttention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    grid_size=grid_size,
                    grid_depth=grid_depth,
                )
            else:
                self.attn = RoPEAttentionND(
                    dim,
                    axes=axes,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                )
        else:
            self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        pos: tuple[torch.Tensor, ...] | None = None,
        attn_mask: torch.Tensor | None = None,
        **rope_kwargs: object,
    ) -> torch.Tensor:
        # Dispatch by attention type. Each branch consumes a different subset
        # of the kwargs above:
        #   RoPEAttentionND (axes ∈ {1, 2}) → pos
        #   RoPEAttention   (axes == 3)     → mask, attn_mask, T, H_patches, W_patches  (via rope_kwargs)
        #   SelfAttention   (use_rope=False) → attention_mask
        if isinstance(self.attn, RoPEAttentionND):
            if pos is None:
                raise ValueError(f"Block(axes={self.axes}) with RoPE requires a pos tuple of length {self.axes}")
            y = self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask)
        elif isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), attn_mask=attn_mask, **rope_kwargs)
        else:
            y = self.attn(self.norm1(x), attention_mask=attn_mask)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x)))


#
# These share a single implementation parametrised by ``axes`` (1 or 2): the
# head dim is split evenly into ``axes`` slices, each receiving a textbook-RoPE
# rotation against its own position tensor. ``pos`` is always a tuple of
# ``axes`` index tensors (each broadcastable with ``[B, N]`` or ``[N]``).
#
# The 3D ``RoPEAttention`` above is intentionally *not* unified: it uses a
# different (non-textbook) rotation pattern that VJEPA depends on. Keep them
# separate so VJEPA's behavior is preserved.


def init_transformer_weights_(module: nn.Module, std: float = 0.02) -> None:
    """In-place trunc-normal init for Linear / ones+zeros for LayerNorm."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _rotate_queries_or_keys_1d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE to ``x`` of shape ``[B, num_heads, N, D]`` with even ``D``.

    ``pos`` is broadcastable with ``[B, num_heads, N]`` and supplies per-token
    positions. Standard RoPE frequencies ``omega_k = 1 / 10000^(2k/D)``.
    """
    _, _, _, d = x.size()
    if d % 2 != 0:
        raise ValueError(f"RoPE requires even feature dim; got {d}")
    omega = torch.arange(d // 2, dtype=x.dtype, device=x.device)
    omega /= d / 2.0
    omega = 1.0 / 10000**omega  # [D/2]

    freq = torch.einsum("..., f -> ... f", pos, omega)
    sin = freq.sin().repeat_interleave(2, dim=-1)
    cos = freq.cos().repeat_interleave(2, dim=-1)
    if cos.dim() == 3:  # noqa: PLR2004 — [B, N, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * cos + y * sin


class RoPEAttentionND(nn.Module):
    """Multi-head self-attention with N-D textbook RoPE on Q and K.

    ``axes`` controls how many position axes the RoPE rotation spans. The
    head dim is split evenly across the axes (rounded down to the nearest
    even slice); any leftover dims pass through without rotation. Inputs:

    - ``x``: ``[B, N, dim]``.
    - ``pos``: tuple of length ``axes``, each entry an index tensor
      broadcastable with ``[B, N]`` or ``[N]``.
    """

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        axes: int = 1,
        num_heads: int = 8,
        qkv_bias: bool = True,  # noqa: FBT001, FBT002
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        use_sdpa: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if axes < 1:
            raise ValueError(f"axes must be >= 1; got {axes}")
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.axes = axes
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal
        # Split head_dim evenly across axes, each slice even-sized.
        per_axis = 2 * ((head_dim // axes) // 2)
        self.per_axis_dim = per_axis
        self.rope_dim = per_axis * axes  # may be < head_dim; remainder is identity

    def forward(
        self,
        x: torch.Tensor,
        pos: tuple[torch.Tensor, ...],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(pos) != self.axes:
            raise ValueError(f"pos must have {self.axes} entries; got {len(pos)}")
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.per_axis_dim > 0:
            q_parts: list[torch.Tensor] = []
            k_parts: list[torch.Tensor] = []
            s = 0
            for axis_pos in pos:
                ap = axis_pos.to(device=x.device, dtype=q.dtype)
                q_parts.append(_rotate_queries_or_keys_1d(q[..., s : s + self.per_axis_dim], pos=ap))
                k_parts.append(_rotate_queries_or_keys_1d(k[..., s : s + self.per_axis_dim], pos=ap))
                s += self.per_axis_dim
            if s < self.head_dim:
                q_parts.append(q[..., s:])
                k_parts.append(k[..., s:])
            q = torch.cat(q_parts, dim=-1)
            k = torch.cat(k_parts, dim=-1)

        if self.use_sdpa:
            # SDPA can't combine an explicit attn_mask with is_causal=True; fold
            # causality into the mask when both are requested.
            sdpa_mask = attn_mask
            sdpa_is_causal = self.is_causal
            if attn_mask is not None and self.is_causal:
                causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
                sdpa_mask = (
                    attn_mask & causal
                    if attn_mask.dtype == torch.bool
                    else attn_mask.masked_fill(~causal, float("-inf"))
                )
                sdpa_is_causal = False
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=sdpa_is_causal,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.is_causal:
                causal_mask = torch.ones(n, n, dtype=torch.bool, device=attn.device).triu(diagonal=1)
                attn = attn.masked_fill(causal_mask, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))
