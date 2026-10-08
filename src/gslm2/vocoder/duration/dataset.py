"""Datasets and run-length encoding for duration prediction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

DurationSample = tuple[Tensor, Tensor, Tensor]
StoredTokens = list[int] | list[list[int]]
_MIN_SEQUENCE_LENGTH = 2
_MULTISTREAM_TOKEN_NDIM = 2


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


def _split_streams(tokens: list[int], num_streams: int, source: Path, line_number: int) -> list[list[int]]:
    tokens_per_stream, remainder = divmod(len(tokens), num_streams)
    if remainder:
        raise ValueError(
            f"{source}:{line_number} has {len(tokens)} tokens, which is not divisible by {num_streams} streams"
        )
    return [tokens[index * tokens_per_stream : (index + 1) * tokens_per_stream] for index in range(num_streams)]


class DurationDataset(Dataset[DurationSample]):
    """Load full-rate unit rows and derive deduplicated tokens with durations."""

    def __init__(
        self,
        data_dirs: Sequence[str | Path],
        max_seq_len: int = 2048,
        max_duration: int = 50,
        file_pattern: str = "*.txt",
        num_streams: int = 1,
    ) -> None:
        if num_streams <= 0:
            raise ValueError("num_streams must be positive")

        self.num_streams = num_streams
        self.max_duration = max_duration
        self.samples: list[tuple[StoredTokens, list[int]]] = []
        for data_dir in data_dirs:
            for text_path in sorted(Path(data_dir).glob(file_pattern)):
                self._load_file(text_path, max_seq_len)

    def _load_file(self, text_path: Path, max_seq_len: int) -> None:
        with text_path.open(encoding="utf-8") as text_file:
            for line_number, line in enumerate(text_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                tokens = [int(value) for value in stripped.split()]
                self._add_tokens(tokens, max_seq_len, text_path, line_number)

    def _add_tokens(
        self,
        tokens: list[int],
        max_seq_len: int,
        source: Path,
        line_number: int,
    ) -> None:
        if self.num_streams > 1:
            streams = _split_streams(tokens, self.num_streams, source, line_number)
            deduplicated_streams, durations = cross_stream_run_length_encode(streams)
            stored_tokens: StoredTokens = [stream[:max_seq_len] for stream in deduplicated_streams]
        else:
            deduplicated_tokens, durations = run_length_encode(tokens)
            stored_tokens = deduplicated_tokens[:max_seq_len]

        if len(durations) < _MIN_SEQUENCE_LENGTH:
            return
        self.samples.append((stored_tokens, durations[:max_seq_len]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DurationSample:
        tokens, durations = self.samples[index]
        raw_durations = torch.tensor(durations, dtype=torch.float32)
        return (
            torch.tensor(tokens, dtype=torch.long),
            raw_durations.clamp(max=self.max_duration),
            raw_durations.sum(),
        )


def collate_fn(batch: list[DurationSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad token and duration tensors, retaining raw duration totals."""
    if not batch:
        raise ValueError("cannot collate an empty batch")

    token_tensors = [sample[0] for sample in batch]
    duration_tensors = [sample[1] for sample in batch]
    raw_duration_totals = torch.stack([sample[2] for sample in batch])
    if token_tensors[0].ndim == _MULTISTREAM_TOKEN_NDIM:
        num_streams = token_tensors[0].size(0)
        lengths = torch.tensor([tensor.size(1) for tensor in token_tensors], dtype=torch.long)
        tokens = torch.zeros(len(batch), num_streams, int(lengths.max()), dtype=torch.long)
        for batch_idx, tensor in enumerate(token_tensors):
            tokens[batch_idx, :, : tensor.size(1)] = tensor
    else:
        lengths = torch.tensor([tensor.size(0) for tensor in token_tensors], dtype=torch.long)
        tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)

    durations = pad_sequence(duration_tensors, batch_first=True, padding_value=0.0)
    return tokens, durations, lengths, raw_duration_totals
