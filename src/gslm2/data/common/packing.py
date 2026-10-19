"""Sequence packing primitives for FlashAttention-varlen block-diagonal attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_cu_seqlens(lengths: Tensor) -> tuple[Tensor, int]:
    """Build cumulative sequence lengths for FlashAttention varlen API.

    Args:
        lengths: int tensor of shape [N] with per-segment frame counts.

    Returns:
        cu_seqlens: int32 tensor of shape [N+1], cu_seqlens[0]=0,
                    cu_seqlens[i] = sum(lengths[:i]).
        max_seqlen: int, max(lengths).
    """
    cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    max_seqlen = int(lengths.max().item())
    return cu_seqlens, max_seqlen


def pack_features(
    features: Tensor,
    frame_lengths: Tensor,
) -> tuple[Tensor, Tensor, int, Tensor]:
    """Concatenate the valid frames of a padded batch into one packed sequence.

    Args:
        features: [B, T_max, D] padded feature tensor.
        frame_lengths: [B] int tensor of valid frame counts per utterance.

    Returns:
        packed: [1, sum(frame_lengths), D] concatenated features.
        cu_seqlens: [B+1] int32 cumulative lengths.
        max_seqlen: int, max segment length.
        segment_ids: [1, sum(frame_lengths)] int64 mapping each frame
                     to its source utterance index (0..B-1).
    """
    B, T_max, D = features.shape  # noqa: N806
    device = features.device
    frame_lengths_long = frame_lengths.to(torch.long)

    arange = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    valid_mask = arange < frame_lengths_long.unsqueeze(1)
    packed = features[valid_mask].unsqueeze(0)

    cu_seqlens, max_seqlen = build_cu_seqlens(frame_lengths_long)
    segment_ids = torch.repeat_interleave(torch.arange(B, device=device), frame_lengths_long).unsqueeze(0)

    return packed, cu_seqlens, max_seqlen, segment_ids


def pack_mask(mask: Tensor, frame_lengths: Tensor) -> Tensor:
    """Pack a padded per-batch mask into a single concatenated sequence.

    Args:
        mask: [B, T_max] bool tensor produced by baseline mask generation.
        frame_lengths: [B] int tensor of valid (unpadded) frame counts.
    Returns:
        [1, T_total] bool tensor where T_total = sum(frame_lengths).
    """
    frame_lengths_long = frame_lengths.long().to(mask.device)
    B = mask.shape[0]  # noqa: N806
    T_max = mask.shape[1]  # noqa: N806
    pos_idx = torch.arange(T_max, device=mask.device).unsqueeze(0).expand(B, -1)
    valid = pos_idx < frame_lengths_long.unsqueeze(1)
    return mask[valid].unsqueeze(0)


def packed_instance_norm(x: Tensor, cu_seqlens: Tensor, segment_ids: Tensor | None = None) -> Tensor:
    """Instance normalization computed independently per segment.

    Args:
        x: [B, D, T] tensor (instance_norm convention: channels=D, spatial=T).
           For packed sequences B=1 and T=sum(segment_lengths).
        cu_seqlens: [num_segments+1] int32 cumulative segment lengths.
        segment_ids: optional [T] int64 mapping each frame to its segment.
                     If None, computed from cu_seqlens.

    Returns:
        [B, D, T] tensor normalized per segment.
    """
    B, D, T = x.shape  # noqa: N806
    num_segments = cu_seqlens.shape[0] - 1
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)

    if segment_ids is None:
        seg_ids = torch.repeat_interleave(torch.arange(num_segments, device=x.device), lengths)
    else:
        seg_ids = segment_ids.view(-1).to(torch.long)

    # Accumulate in fp32 to preserve variance precision for long segments.
    x_t = x.transpose(1, 2).to(torch.float32)
    index = seg_ids.view(1, T, 1).expand(B, T, D)

    seg_sum = torch.zeros(B, num_segments, D, device=x.device, dtype=torch.float32)
    seg_sum.scatter_add_(1, index, x_t)
    seg_sq = torch.zeros(B, num_segments, D, device=x.device, dtype=torch.float32)
    seg_sq.scatter_add_(1, index, x_t * x_t)

    counts = lengths.to(torch.float32).clamp(min=1).view(1, num_segments, 1)
    mean = seg_sum / counts
    var = ((seg_sq / counts) - mean * mean).clamp(min=0.0)

    mean_t = mean.gather(1, index)
    var_t = var.gather(1, index)
    out_t = (x_t - mean_t) / torch.sqrt(var_t + 1e-5)

    return out_t.transpose(1, 2).to(x.dtype)


def conv_output_length(
    input_lengths: Tensor,
    conv_layer_config: list[tuple[int, int, int]],
) -> Tensor:
    """Compute output frame lengths after the convolutional feature extractor.

    Applies the valid-conv length formula for each layer:
        out_len = floor((in_len - kernel_size) / stride) + 1

    Args:
        input_lengths: [B] int tensor of input waveform sample counts.
        conv_layer_config: list of (out_channels, kernel_size, stride) tuples
                           defining the feature extractor architecture.

    Returns:
        [B] int tensor of output frame counts.
    """
    lengths = input_lengths.clone()
    for _out_ch, kernel_size, stride in conv_layer_config:
        lengths = (lengths - kernel_size) // stride + 1
        lengths = lengths.clamp(min=0)
    return lengths
