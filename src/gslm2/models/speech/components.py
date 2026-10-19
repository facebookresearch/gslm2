"""Adapted from torchaudio.

See https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py
"""

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from flash_attn.cute import flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func

        HAS_FLASH_ATTN = True
    except ImportError:
        HAS_FLASH_ATTN = False

if HAS_FLASH_ATTN:
    import inspect

    _FLASH_ATTN_HAS_DROPOUT = "dropout_p" in inspect.signature(flash_attn_varlen_func).parameters
else:
    _FLASH_ATTN_HAS_DROPOUT = False


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose."""

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        x = input.transpose(-2, -1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.transpose(-2, -1)


class ConvLayerBlock(nn.Module):
    """Convolution unit of FeatureExtractor."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        layer_norm: nn.Module | None,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)

    def forward_channels_last(self, x: Tensor) -> Tensor:
        """The same block over a [B, C, 1, T] channels-last tensor, without transpose copies.

        Args:
            x: [B, C_in, 1, T] in channels_last memory format.

        Returns:
            [B, C_out, 1, T'] in channels_last memory format.
        """
        weight = self.conv.weight.unsqueeze(2).to(memory_format=torch.channels_last)
        x = F.conv2d(x, weight, self.conv.bias, stride=(1, self.stride))
        if isinstance(self.layer_norm, LayerNorm):
            # Channels are innermost here, so the permute is a view.
            norm = self.layer_norm
            y = F.layer_norm(x.permute(0, 2, 3, 1), norm.normalized_shape, norm.weight, norm.bias, norm.eps)
            x = y.permute(0, 3, 1, 2)
        elif self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    """Extract features from audio.

    Args:
        conv_layers: the ConvLayerBlock stack.
        channels_last: hold activations with channels innermost. LayerNorm becomes a view
            rather than transposes, and cuDNN skips conversions.
    """

    def __init__(self, conv_layers: nn.ModuleList, *, channels_last: bool = True) -> None:
        super().__init__()
        self.conv_layers = conv_layers
        self.channels_last = channels_last

    def forward(self, x: Tensor) -> Tensor:
        if self.channels_last:
            x = x[:, None, None, :].to(memory_format=torch.channels_last)  # (batch, 1, 1, sample)
            for layer in self.conv_layers:
                x = layer.forward_channels_last(x)  # (batch, feature, 1, frame)
            return x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[3], x.shape[1])

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x = layer(x)  # (batch, feature, frame)
        return x.transpose(1, 2)  # (batch, frame, feature)


class FeatureProjection(nn.Module):
    """Projects features to encoder dimension."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x)
        return self.projection(x)


class ConvPositionalEmbedding(nn.Module):
    """Positional embedding which is placed at the beginning of Transformer."""

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.depth = depth
        self.layer_kernel_size = max(3, kernel_size // depth)
        self.layer_norm = LayerNorm(embed_dim, elementwise_affine=False)

        k = self.layer_kernel_size
        self.left_pad = k // 2
        self.right_pad = k // 2 - (1 if k % 2 == 0 else 0)

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, k, padding=0, groups=groups) for _ in range(depth)]
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        if cu_seqlens is not None:
            # Per-segment conv to avoid cross-segment contamination across the ~91-frame RF.
            assert x.shape[0] == 1, f"packed pos-conv expects a batch of 1, got {x.shape[0]}"
            x_2d = x.squeeze(0)
            num_segments = cu_seqlens.shape[0] - 1
            segments_out = []
            for s in range(num_segments):
                start = int(cu_seqlens[s].item())
                end = int(cu_seqlens[s + 1].item())
                seg = x_2d[start:end, :].unsqueeze(0).transpose(-2, -1)
                for conv in self.convs:
                    seg = F.pad(seg, (self.left_pad, self.right_pad))
                    seg = conv(seg)
                    seg = self.layer_norm(seg)
                    seg = F.gelu(seg)
                segments_out.append(seg.transpose(-2, -1).squeeze(0))
            return torch.cat(segments_out, dim=0).unsqueeze(0)

        mask = (
            torch.ones_like(x, dtype=torch.bool).transpose(-2, -1)
            if attention_mask is None
            else attention_mask[:, :, 0]
        )
        x = x.transpose(-2, -1)
        for conv in self.convs:
            x = x * mask
            x = F.pad(x, (self.left_pad, self.right_pad))
            x = conv(x)
            x = self.layer_norm(x)
            x = F.gelu(x)
        x = x * mask
        return x.transpose(-2, -1)


class SelfAttention(nn.Module):
    """Multihead Self Attention module."""

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward_packed(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None,
        dropout: float,
    ) -> Tensor:
        """Attend within each segment of a packed (1, total_tokens, embed_dim) sequence."""
        assert x.shape[0] == 1, f"packed attention expects a batch of 1, got {x.shape[0]}"

        use_flash = HAS_FLASH_ATTN and x.is_cuda
        if use_flash and dropout > 0.0 and not _FLASH_ATTN_HAS_DROPOUT:
            # Subbing output dropout changes regularizer, dense fallback ~48x slower than fa4, have to refuse
            raise RuntimeError(
                f"Packed attention with attention dropout (p={dropout}) requires a flash-attn "
                "build exposing dropout_p; this build does not. Set encoder_attention_dropout=0.0 "
                "for packed runs, or install a flash-attn build with dropout support."
            )

        if use_flash:
            total_tokens = x.shape[1]
            x_2d = x.squeeze(0)

            head_dim = self.embed_dim // self.num_heads
            qkv = self.qkv(x_2d).reshape(total_tokens, 3, self.num_heads, head_dim)
            q, k, v = qkv.unbind(1)

            orig_dtype = q.dtype
            if q.dtype == torch.float32:
                q, k, v = q.half(), k.half(), v.half()
            flash_kwargs = {
                "cu_seqlens_q": cu_seqlens.to(q.device),
                "cu_seqlens_k": cu_seqlens.to(q.device),
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
                "causal": False,
            }
            if _FLASH_ATTN_HAS_DROPOUT:
                flash_kwargs["dropout_p"] = dropout
            out = flash_attn_varlen_func(q, k, v, **flash_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.to(orig_dtype).reshape(total_tokens, self.embed_dim)
            return self.proj(out.unsqueeze(0))

        batch, seq, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        seg_ids = torch.zeros(seq, dtype=torch.long, device=x.device)
        for s in range(cu_seqlens.shape[0] - 1):
            start = int(cu_seqlens[s].item())
            end = int(cu_seqlens[s + 1].item())
            seg_ids[start:end] = s
        block_mask = (seg_ids.unsqueeze(0) == seg_ids.unsqueeze(1))[None, None, :, :]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, dropout_p=dropout)
        x = x.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(x)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> Tensor:
        dropout = self.dropout if self.training else 0.0

        if cu_seqlens is not None:
            return self.forward_packed(x, cu_seqlens, max_seqlen, dropout)

        batch, seq, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        head_dim = dim // self.num_heads
        k = k.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout)
        x = x.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(x)


class FeedForward(nn.Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(self, embed_dim: int, interm_features: int, interm_dropout: float) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(embed_dim, interm_features)
        self.intermediate_dropout = nn.Dropout(interm_dropout)
        self.output_dense = nn.Linear(interm_features, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.intermediate_dense(x)
        x = F.gelu(x)
        x = self.intermediate_dropout(x)
        return self.output_dense(x)


class TransformerLayer(nn.Module):
    """Combines multihead self attention and feed forward."""

    def __init__(
        self, attention: SelfAttention, dropout: float, feed_forward: FeedForward, *, layer_norm_first: bool
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(x, attention_mask=attention_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            residual = x
            layer_result = self.feed_forward(self.final_layer_norm(x))
            x = self.dropout(layer_result)
            x = residual + x
        else:
            x = self.layer_norm(x)
            residual = x
            layer_result = self.feed_forward(x)
            x = self.dropout(layer_result)
            x = self.final_layer_norm(residual + x)
        return x, layer_result


class Transformer(nn.Module):
    """Transformer module with positional convolutional embeddings."""

    def __init__(  # noqa: PLR0913
        self,
        layers: nn.ModuleList,
        pos_conv_embed: ConvPositionalEmbedding,
        dropout: float,
        layer_drop: float,
        *,
        layer_norm_first: bool,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, SelfAttention):
            nn.init.xavier_uniform_(module.qkv.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(module.proj.weight)
            nn.init.constant_(module.proj.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> Tensor:
        x = x + self.pos_conv_embed(x, attention_mask, cu_seqlens=cu_seqlens)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1) < self.layer_drop):
                x, _ = layer(x, attention_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        num_layers: int | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        if num_layers is not None and not 0 < num_layers <= len(self.layers):
            exception = f"`num_layers` must be between [1, {len(self.layers)}]"
            raise ValueError(exception)

        ret: list[Tensor] = []
        x = x + self.pos_conv_embed(x, attention_mask, cu_seqlens=cu_seqlens)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            x_output, layer_result = layer(x, attention_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if not (self.training and torch.rand(1) < self.layer_drop):
                x = x_output.clone()
            ret.append(layer_result)
            if num_layers is not None and len(ret) >= num_layers:
                return ret, x
        return ret, x


def get_feature_extractor(
    norm_mode: Literal["group_norm", "layer_norm"],
    shapes: list[tuple[int, int, int]],
    *,
    bias: bool,
    channels_last: bool = True,
) -> FeatureExtractor:
    """Get convolutional feature extractor."""
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError(norm_mode)
    blocks: list[nn.Module] = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization: nn.Module | None = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(normalized_shape=out_channels, elementwise_affine=True)
        blocks.append(ConvLayerBlock(in_channels, out_channels, kernel_size, stride, normalization, bias=bias))
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks), channels_last=channels_last)


def get_encoder(  # noqa: PLR0913
    in_features: int,
    embed_dim: int,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    pos_conv_depth: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_drop: float,
    *,
    layer_norm_first: bool,
    qkv_bias: bool,
) -> tuple[FeatureProjection, Transformer]:
    """Build Transformer encoder."""

    feature_projection = FeatureProjection(in_features, embed_dim)
    pos_conv = ConvPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups, pos_conv_depth)
    layers = nn.ModuleList()
    for _ in range(num_layers):
        attention = SelfAttention(
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            dropout=attention_dropout,
        )
        feed_forward = FeedForward(embed_dim, ff_interm_features, ff_interm_dropout)
        layers.append(TransformerLayer(attention, dropout, feed_forward, layer_norm_first=layer_norm_first))
    transformer = Transformer(
        layers,
        pos_conv,
        dropout,
        layer_drop,
        layer_norm_first=not layer_norm_first,
        embed_dim=embed_dim,
    )
    return feature_projection, transformer


class _UnitWeightNorm(nn.Module):
    """Weight normalization with fixed g=1 (unit norm rows)."""

    def forward(self, weight: Tensor) -> Tensor:
        return F.normalize(weight, p=2, dim=1)


def get_heads(  # noqa: PLR0913
    embed_dim: int,
    codebook_size: int,
    num_codebooks: int,
    *,
    layer_norm: Literal["none", "center", "affine"] = "none",
    use_weight_norm: bool = False,
    init: Literal["default", "orthogonal", "sphere"] = "default",
) -> nn.ModuleList:
    """Build classification head modules.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension
    codebook_size : int
        Output vocabulary size (number of codebook entries)
    num_codebooks : int
        Number of head modules to create (one per codebook)
    layer_norm : {"none", "center", "affine"}, default="none"
        LayerNorm mode before the linear layer:
        - "none": No LayerNorm
        - "center": Center-reduced LayerNorm (elementwise_affine=False)
        - "affine": Full LayerNorm with learnable affine transform
    use_weight_norm : bool, default=False
        If True, apply weight normalization to the rows of the linear layer.
    init : {"default", "orthogonal", "sphere"}, default="default"
        Weight initialization strategy for the linear layer:
        - "default": PyTorch default (Kaiming uniform)
        - "orthogonal": Orthogonal initialization (90° apart rows)
        - "sphere": Uniform on unit sphere (normalized Gaussian)

    Returns
    -------
    nn.ModuleList
        List of head modules (each is an nn.Sequential)
    """
    heads = []
    for _ in range(num_codebooks):
        layers = []

        # Optional LayerNorm before the linear layer
        if layer_norm != "none":
            elementwise_affine = layer_norm == "affine"
            layers.append(nn.LayerNorm(embed_dim, elementwise_affine=elementwise_affine))

        linear = nn.Linear(embed_dim, codebook_size, bias=True)
        if init == "orthogonal":
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
        elif init == "sphere":
            nn.init.normal_(linear.weight)
            linear.weight.data = F.normalize(linear.weight.data, p=2, dim=1)
            nn.init.zeros_(linear.bias)
        if use_weight_norm:
            nn.utils.parametrize.register_parametrization(linear, "weight", _UnitWeightNorm())
        layers.append(linear)

        layers.append(nn.LogSoftmax(dim=-1))

        heads.append(nn.Sequential(*layers))
    return nn.ModuleList(heads)
