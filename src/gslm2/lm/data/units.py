# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, override

import torch
from fairseq2.data import (
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.text import read_text
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    SequenceBatch,
)
from fairseq2.error import NotSupportedError
from torch import Tensor

from gslm2.lm.data.base import (
    DataReadOptions,
    LengthBatching,
    LMDataset,
    LMDatasetConfig,
    StaticBatching,
    SyncMode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fairseq2.gang import Gangs

    from gslm2.shared.utils.typing import PathLike


PAD_VALUE = 1


def _to_sequence_batch(example: SequenceData, num_streams: int) -> SequenceBatch:
    seqs, seq_lens = example["seqs"], example["seq_lens"]
    seqs = seqs.to(torch.int64)

    if num_streams == 1:
        return SequenceBatch(seqs, seq_lens, example=example)

    batch_size = seqs.size(0)
    device = seqs.device
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.long, device=device)
    stream_lens = seq_lens // num_streams
    stream_len = int(stream_lens.max().item())

    time_indices = torch.arange(stream_len, device=device)
    stream_offsets = torch.arange(num_streams, device=device).unsqueeze(1) * stream_lens.unsqueeze(0)
    indices = stream_offsets.T.unsqueeze(2) + time_indices

    gathered = torch.gather(seqs, 1, indices.reshape(batch_size, -1))
    gathered = gathered.reshape(batch_size, num_streams, stream_len)
    mask = time_indices < stream_lens[:, None, None]
    seqs = gathered.masked_fill(~mask, PAD_VALUE).reshape(batch_size * num_streams, stream_len)
    seq_lens = stream_lens[:, None].expand(-1, num_streams).reshape(-1).tolist()

    return SequenceBatch(seqs, seq_lens, example=example)


# A very basic dataset implementation that treats each directory under a base
# path as a split. Each split directory is expected to contain one or more text
# files ending with a ".txt" suffix. Each text file is expected to contain
# lines with space separated lists of numbers to be parsed as integers.
#
# Check gslm2/fs2/register.py to see how we register this dataset with fairseq2.
#
def _to_packed_sequence_batch(example: SequenceData) -> SequenceBatch:
    """Convert one fixed-length packed row into a packed ``SequenceBatch``."""
    seqs = example["seqs"].to(torch.int64)
    if seqs.ndim > 1:
        seqs = seqs.reshape(-1)

    seq_lens = example["seq_lens"]
    if isinstance(seq_lens, Tensor):
        seq_lens = seq_lens.tolist()

    width = seqs.size(0)
    total = int(sum(seq_lens))

    target_mask = torch.ones(width, dtype=torch.bool)
    if total < width:
        target_mask[total:] = False
    offset = 0
    for length in seq_lens[:-1]:
        offset += length
        target_mask[offset] = False

    return SequenceBatch(seqs, seq_lens, packed=True, target_mask=target_mask, example=example)


def _to_packed_ms_sequence_batch(example: SequenceData, num_streams: int) -> SequenceBatch:
    """Convert a collated multi-stream row into an ``(S, W)`` packed ``SequenceBatch``."""
    seqs = example["seqs"].to(torch.int64)
    seq_lens = example["seq_lens"]
    if isinstance(seq_lens, Tensor):
        seq_lens = seq_lens.tolist()

    doc_frame_lens = [int(dlen) // num_streams for dlen in seq_lens]  # T_doc per doc

    stream_rows = [
        torch.cat([seqs[b, s * t : (s + 1) * t] for b, t in enumerate(doc_frame_lens)]) for s in range(num_streams)
    ]
    packed = torch.stack(stream_rows, dim=0)  # (S, W_ps)
    width = packed.size(1)

    # (S, W_ps) target_mask. False at each document's first frame.
    row_mask = torch.ones(width, dtype=torch.bool)
    offset = 0
    for t in doc_frame_lens[:-1]:
        offset += t
        row_mask[offset] = False
    target_mask = row_mask.unsqueeze(0).expand(num_streams, -1).contiguous()

    packed_example = dict(example)
    packed_example["doc_frame_lens"] = doc_frame_lens
    return SequenceBatch(packed, [width] * num_streams, target_mask=target_mask, example=packed_example)


class _FreshStateDataPipelineReader(DataPipelineReader[SequenceBatch]):
    """A reader that ignores the data-reader state stored in a checkpoint."""

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        pass


def finalize_unit_pipeline(  # noqa: PLR0913
    builder: DataPipelineBuilder,
    gangs: Gangs,
    *,
    min_seq_len: int,
    max_seq_len: int,
    num_io_streams: int,
    options: DataReadOptions,
    seed: int,
    reset_state: bool = False,
) -> DataReader[SequenceBatch]:
    """Apply the shared length filter, unit offset, batching and collation stages."""

    def check_len(seq: Tensor) -> bool:
        numel = seq.numel()

        return numel >= min_seq_len * num_io_streams and numel <= max_seq_len * num_io_streams

    builder.filter(check_len)

    def offset_eos_bos(seq: Tensor) -> Tensor:
        return seq + 2  # offset indices by 2.

    builder.map(offset_eos_bos)

    seed += 1

    # Packing gives fixed-length rows, so torch.compile sees constant shapes.
    batching = options.batching

    if options.pack and not isinstance(batching, LengthBatching):
        raise NotSupportedError("Sequence packing requires LengthBatching (max_num_tokens).")

    if options.pack and num_io_streams == 1:
        builder.pack(
            num_elements=batching.max_num_elements,
            max_seq_len=max_seq_len,
            pad_value=PAD_VALUE,
            truncate=True,
            drop_remainder=True,
        )
    elif isinstance(batching, LengthBatching):
        # Also the multi-stream packing path, which batches here and packs downstream.
        bucket_sizes = create_bucket_sizes(
            min_seq_len=min_seq_len * num_io_streams,
            max_seq_len=max_seq_len * num_io_streams,
            max_num_elements=batching.max_num_elements * num_io_streams,
        )

        builder.bucket_by_length(
            bucket_sizes,
            skip_above_max_examples=True,
            drop_remainder=options.drop_remainder,
        )
        # Collate bucketed lines into a single padded batch: a dict with keys
        # "seqs" (stacked, right-padded) and "seq_lens" (per-row lengths).
        builder.collate(pad_value=PAD_VALUE)
    elif isinstance(batching, StaticBatching):
        builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        builder.collate(pad_value=PAD_VALUE)
    else:
        raise NotSupportedError(f"`{batching}` is not supported.")

    if options.batch_shuffle_window != 1:
        builder.shuffle(options.batch_shuffle_window, seed=seed)

    seed += 1

    if options.max_num_batches is not None:
        builder.take(options.max_num_batches)

    builder.prefetch(options.prefetch)

    if options.pack:
        to_batch = (
            _to_packed_sequence_batch
            if num_io_streams == 1
            else partial(_to_packed_ms_sequence_batch, num_streams=num_io_streams)
        )
    else:
        to_batch = partial(_to_sequence_batch, num_streams=num_io_streams)
    pipeline = builder.map(to_batch).and_return()

    reader_cls = _FreshStateDataPipelineReader if reset_state else DataPipelineReader[SequenceBatch]
    return reader_cls(
        pipeline,
        gangs,
        num_accumulate=options.num_accumulate,
        drop_remainder=options.drop_remainder,
        sync_mode=options.sync_mode,
        sync=options.sync_mode == SyncMode.UNTIL_LAST,
        strict_state=options.strict_state,
    )


class UnitDataset(LMDataset):
    """Unit dataset over a directory tree of sharded ``.txt`` files."""

    @staticmethod
    @override
    def from_path(path: Path) -> UnitDataset:
        splits = {p.name for p in path.iterdir() if p.is_dir()}

        return UnitDataset(path, splits)

    def create_reader(  # noqa: PLR0913
        self,
        split: str,
        gangs: Gangs,
        min_seq_len: int,
        max_seq_len: int,
        options: DataReadOptions | None = None,
        dynamic_chunking_batch_size: int = 1,  # noqa: ARG002
        num_io_streams: int = 1,
    ) -> DataReader[SequenceBatch]:
        # Raise well-known errors instead of assert statements for better
        # usability. Those are specially handled by the fairseq2 CLI.
        if split not in self._splits:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self._base_path}'. Available splits: {self._splits}"
            )

        if options is None:
            options = DataReadOptions()

        seed = options.seed

        gang = gangs.dp

        # Assume each directory represents a split.
        split_dir = self._base_path.joinpath(split)

        # List files under the split directory and cluster them by stem.
        subset2files = cluster_sharded_files(split_dir, suffix=".txt")

        def list_subset_files(files: list[str]) -> DataPipeline:
            assert len(files) > 0, "Expected at least one file in the subset."
            assert len(files) % gang.size == 0, (
                f"Expected number of files ({len(files)}) to be divisible by gang size ({gang.size})."
            )
            builder = read_sequence(files)
            builder.shuffle(shuffle_window=0, seed=seed)
            builder.shard(gang.rank, gang.size, allow_uneven=True)
            return builder.and_return()

        file_pipelines = [list_subset_files(files) for files in subset2files.values()]

        # Concatenate files from all sharded subsets.
        builder = DataPipeline.concat(file_pipelines)

        # Shuffle the files (if memory is ever a problem here, modify shuffle window)
        builder.shuffle(shuffle_window=0, seed=seed)

        # Read each line from the list of files consecutively.
        def read_file(file: Path) -> DataPipeline:
            return read_text(file, rtrim=True, memory_map=True).and_return()

        builder.yield_from(read_file)

        # Shuffle lines for better entropy.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        # This is one of the "magic" mappers available in fairseq2. When the
        # specified map function is `torch.tensor()`, it assumes the input is
        # a space separated list of numbers in text format and returns an
        # integer PyTorch tensor.
        builder.map(torch.tensor, num_parallel_calls=options.npc)

        return finalize_unit_pipeline(
            builder,
            gangs,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            num_io_streams=num_io_streams,
            options=options,
            seed=seed,
        )


def cluster_sharded_files(directory: PathLike, suffix: str = ".txt") -> dict[str, list[str]]:  # noqa: C901
    """
    Cluster sharded text files in `directory`.

    A "sharded file" is any file matching:  <stem>_<index>.txt
    - <stem>  : any string (may contain underscores)
    - <index> : one or more decimal digits (0-based shard id)

    Returns:
        Dict[str, List[str]]
            A mapping from the minimal distinguishing prefix of each stem
            to the list of file *paths* in that cluster, sorted by shard index.

    Notes:
        - Files that do not match the pattern are ignored.
        - The minimal distinguishing prefix is computed across all detected stems
          as 1 + the longest common prefix length to any other stem (capped at
          the full stem length).
        - If two stems are identical (i.e., true duplicates), the full stem is
          used as the key for that cluster.
    """
    # Match "<anything>_<digits>.txt" (case-insensitive for .txt)
    suffix_pattern = re.escape(suffix)
    rx = re.compile(rf"^(?P<stem>.+)_(?P<idx>\d+){suffix_pattern}$", re.IGNORECASE)

    # Collect all matching files as tuples: (stem, index, full_path)
    matches = []
    for name in os.listdir(directory):  # noqa: PTH208
        path = os.path.join(directory, name)  # noqa: PTH118
        if not os.path.isfile(path):  # noqa: PTH113
            continue
        m = rx.match(name)
        if not m:
            continue
        stem = m.group("stem")
        idx = int(m.group("idx"))
        matches.append((stem, idx, path))

    if not matches:
        return {}

    # Group by stem and keep (idx, path); sort each cluster by idx
    by_stem: dict[str, list[tuple]] = defaultdict(list)
    for stem, idx, path in matches:
        by_stem[stem].append((idx, path))
    for value in by_stem.values():
        value.sort(key=lambda t: t[0])

    stems = list(by_stem.keys())

    # Helper: length of the longest common prefix between two strings
    def _lcp_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # Compute minimal distinguishing prefix length per stem:
    # 1 + max LCP with any other stem (capped at len(stem))
    min_len_for: dict[str, int] = {}
    for s in stems:
        max_lcp = 0
        for t in stems:
            if s == t:
                continue
            max_lcp = max(max_lcp, _lcp_len(s, t))
        min_len_for[s] = min(len(s), max_lcp + 1)

    # Build final clusters keyed by minimal distinguishing prefix
    clusters: dict[str, list[str]] = {}
    used_keys = set()
    for stem, items in by_stem.items():
        key = stem[: min_len_for[stem]]
        # Safety: if a collision somehow remains (e.g., identical stems),
        # fall back to the full stem.
        if key in used_keys and key != stem:
            key = stem
        used_keys.add(key)
        clusters[key] = [path for _, path in items]

    return clusters


def open_speech_lm_dataset(config: LMDatasetConfig) -> UnitDataset:
    path = config.path

    path = path.expanduser().resolve()

    return UnitDataset.from_path(path)
