# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Parsing and validating aligned speech-unit sequences."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

type UnitSequence = str | list[int] | list[list[int]]
_ALIGNED_UNITS_NDIM = 2


def parse_units(units: UnitSequence) -> Tensor:
    """Parse raw IDs into ``(T,)`` or aligned ``(S,T)`` integer tensors."""
    if isinstance(units, str):
        streams = [[int(token) for token in stream.split()] for stream in units.split("|")]
    elif units and isinstance(units[0], list):
        streams = cast("list[list[int]]", units)
    else:
        streams = [cast("list[int]", units)]
    if not streams or any(not stream for stream in streams):
        raise ValueError("unit sequences must not be empty")
    if len({len(stream) for stream in streams}) != 1:
        raise ValueError("all unit streams must have the same length")
    if any(not isinstance(token, int) or isinstance(token, bool) for stream in streams for token in stream):
        raise TypeError("unit IDs must be integers")
    result = torch.tensor(streams, dtype=torch.long)
    return result.squeeze(0) if len(streams) == 1 else result


def raw_units(
    units: Tensor,
    vocab_size: int | Sequence[int],
    *,
    unit_offset: int = 0,
    special_token_ids: Sequence[int] = (),
) -> Tensor:
    """Strip aligned special-token frames, undo offsets, and validate code IDs."""
    if unit_offset < 0:
        raise ValueError("unit_offset must be non-negative")
    streams = units.unsqueeze(0) if units.ndim == 1 else units
    if streams.ndim != _ALIGNED_UNITS_NDIM:
        raise ValueError("units must have shape (T,) or (S,T)")
    if bool((streams < 0).any()):
        raise ValueError("unit IDs must be non-negative")
    sizes = [vocab_size] if isinstance(vocab_size, int) else list(vocab_size)
    if streams.size(0) != len(sizes):
        raise ValueError(f"expected {len(sizes)} streams, received {streams.size(0)}")
    special = streams < unit_offset
    for token in special_token_ids:
        special |= streams == token
    if not torch.equal(special.any(dim=0), special.all(dim=0)):
        raise ValueError("special tokens must occur at the same frames in all streams")
    result = streams[:, ~special.all(dim=0)] - unit_offset
    if result.size(-1) == 0:
        raise ValueError("no speech units remain after removing special tokens")
    for stream, size in zip(result, sizes, strict=True):
        if size <= 0 or bool(((stream < 0) | (stream >= size)).any()):
            raise ValueError(f"raw unit IDs must be in [0, {size})")
    return result.squeeze(0) if units.ndim == 1 else result


def run_length_encode(tokens: Sequence[int]) -> tuple[list[int], list[int]]:
    """Collapse consecutive equal tokens and return tokens plus frame durations."""
    if not tokens:
        return [], []

    deduplicated = [tokens[0]]
    durations = [1]
    for token in tokens[1:]:
        if token == deduplicated[-1]:
            durations[-1] += 1
        else:
            deduplicated.append(token)
            durations.append(1)
    return deduplicated, durations


def cross_stream_run_length_encode(streams: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    """Collapse frames only when all aligned streams remain unchanged."""
    if not streams:
        raise ValueError("at least one stream is required")

    length = len(streams[0])
    if any(len(stream) != length for stream in streams):
        raise ValueError("all streams must have the same length")
    if length == 0:
        return [[] for _ in streams], []

    deduplicated = [[stream[0]] for stream in streams]
    durations = [1]
    for frame_idx in range(1, length):
        if any(stream[frame_idx] != stream[frame_idx - 1] for stream in streams):
            for output_stream, stream in zip(deduplicated, streams, strict=True):
                output_stream.append(stream[frame_idx])
            durations.append(1)
        else:
            durations[-1] += 1
    return deduplicated, durations


def format_streams(streams: Sequence[Sequence[int]]) -> str:
    """Serialize aligned streams using the canonical separator."""
    return "|".join(" ".join(map(str, stream)) for stream in streams)
