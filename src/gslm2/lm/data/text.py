# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import torch
from fairseq2.data import (
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.data_pipeline import (
    DataPipeline,
    DataPipelineBuilder,
    list_files,
    read_sequence,
)
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    SequenceBatch,
    SyncMode,
)
from fairseq2.error import NotSupportedError
from torch import Tensor

from gslm2.lm.data.base import (
    DataReadOptions,
    LengthBatching,
    LMDataset,
    LMDatasetConfig,
    StaticBatching,
)
from gslm2.lm.data.storage import resolve_manifest_root
from gslm2.lm.data.units import (
    PAD_VALUE,
    _to_packed_sequence_batch,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fairseq2.gang import Gangs


# The BPE-4096 tokenizer prepends `</s>` to every record, making it the only record
# delimiter in a shard's flat id stream, and each document's BOS once packed.
EOS_ID = 2


@dataclass
class TextUnitDatasetConfig(LMDatasetConfig):
    pass


class TextUnitDataset(LMDataset):
    def __init__(
        self,
        base_path: Path,
        splits: set[str],
    ) -> None:
        super().__init__(base_path, splits)

    @staticmethod
    @override
    def from_path(path: Path) -> TextUnitDataset:
        splits = {p.name for p in path.iterdir() if p.is_dir()}

        return TextUnitDataset(path, splits)

    @override
    def create_reader(  # noqa: C901, PLR0915
        self,
        split: str,
        gangs: Gangs,
        min_seq_len: int,
        max_seq_len: int,
        options: DataReadOptions | None = None,
        dynamic_chunking_batch_size: int = 1,
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

        builder = _list_text_files(split_dir)

        _check_shard_count(split_dir, gang.size)

        # Shuffle the files
        builder.shuffle(shuffle_window=0, seed=seed)

        # Shard the list of files among ranks.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        # Read list of tokens
        def read_file(text_file: str) -> list[int]:
            with open(text_file, encoding="utf-8") as f:  # noqa: PTH123
                return list(map(int, f.read().split()))

        builder.map(read_file, num_parallel_calls=10)

        if options.pack:
            # Packing needs one document per element to record per-document lengths for
            # the block-diagonal attention layout, so recover records before chunking.
            def split_documents(token_list: list[int]) -> DataPipeline:
                docs = split_on_eos(token_list)
                chunks = [chunk for doc in docs for chunk in split_into_chunks(doc, max_seq_len)]
                return read_sequence(chunks).and_return()

            builder.yield_from(split_documents)
        else:
            # Concatenate a few files before chunking so chunk boundaries are randomized.
            builder.bucket(dynamic_chunking_batch_size, drop_remainder=False)

            def concat_tokens(token_lists: list[list[int]]) -> list[int]:
                return [token for sublist in token_lists for token in sublist]

            builder.map(concat_tokens, num_parallel_calls=10)

            def chunk_tokens(token_list: list[int]) -> DataPipeline:
                chunks = split_into_chunks(token_list, max_seq_len)
                return read_sequence(chunks).and_return()

            builder.yield_from(chunk_tokens)

        # Shuffle lines for better entropy.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        # From list[int] back to str to prepare for tensorization
        def list_to_str(token_list: list[int]) -> str:
            return " ".join(map(str, token_list))

        builder.map(list_to_str, num_parallel_calls=10)

        # This is one of the "magic" mappers available in fairseq2. When the
        # specified map function is `torch.tensor()`, it assumes the input is
        # a space separated list of numbers in text format and returns an
        # integer PyTorch tensor.
        builder.map(torch.tensor, num_parallel_calls=10)

        # Filter out tensorized lines that do not fall within the accepted
        # minimum and maximum sequence lengths.
        def check_len(seq: Tensor) -> bool:
            numel = seq.numel()

            return numel >= min_seq_len and numel <= max_seq_len

        builder.filter(check_len)

        seed += 1

        # Bucket lines either statically (i.e. every N lines form a bucket of
        # size N) or by length (i.e. bucket lines with similar lengths and
        # return once a bucket reaches `max_num_elements` tokens).
        batching = options.batching

        if options.pack:
            if not isinstance(batching, LengthBatching):
                raise NotSupportedError("Sequence packing requires LengthBatching (max_num_tokens).")

            builder.pack(
                num_elements=batching.max_num_elements,
                max_seq_len=max_seq_len,
                pad_value=PAD_VALUE,
                truncate=True,
                drop_remainder=True,
            )
        elif isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            builder.bucket_by_length(
                bucket_sizes,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )

            # Collate bucketed lines into a padded batch: a dict with keys
            # "seqs" (stacked, right-padded) and "seq_lens" (per-row lengths).
            builder.collate(pad_value=PAD_VALUE)
        elif isinstance(batching, StaticBatching):
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)

            builder.collate(pad_value=PAD_VALUE)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle batches.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed=seed)

        seed += 1

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.prefetch)

        # Convert the dict-based batch returned by `collate()` op into a proper
        # strongly-typed batch
        def to_seq_batch(example: SequenceData) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]
            seqs = seqs.to(torch.int64)

            return SequenceBatch(seqs, seq_lens, example=example)

        to_batch = _to_packed_sequence_batch if options.pack else to_seq_batch

        pipeline = builder.map(to_batch).and_return()

        # This class handles all the tricky details of reading a data iterator
        # correctly in a multi-gpu/multi-node setting.
        return DataPipelineReader[SequenceBatch](
            pipeline,
            gangs,
            num_accumulate=options.num_accumulate,
            drop_remainder=options.drop_remainder,
            # Validation readers sync so every rank runs the same number of iterations,
            # keeping ranks in lockstep during async checkpoint saves. Training readers
            # do not, so an exhausted rank raises StopIteration rather than yielding an
            # empty batch and hanging the DDP all-reduce at an epoch rollover.
            sync_mode=options.sync_mode,
            sync=options.sync_mode == SyncMode.UNTIL_LAST,
            strict_state=False,
        )


def _check_shard_count(split_dir: Path, world_size: int) -> None:
    """Fail fast when a split has fewer ``.txt`` shard files than ranks.

    Files are the unit of sharding, so surplus ranks would get no data and wedge the
    end-of-pass collective until NCCL's watchdog aborts the job hours later.
    """
    num_files = sum(1 for _ in split_dir.glob("**/*.txt"))

    if num_files < world_size:
        raise ValueError(
            f"Split '{split_dir.name}' has {num_files} shard file(s) but is read by {world_size} "
            f"ranks, so {world_size - num_files} rank(s) would receive no data and hang the "
            f"end-of-pass collective. Republish the split with at least {world_size} shard files."
        )


def _list_text_files(split_dir: Path) -> DataPipelineBuilder:
    return list_files(split_dir, pattern="**/*.txt")


def split_into_chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    Split a list into chunks of a given size.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def split_on_eos(tokens: list[int], eos_id: int = EOS_ID) -> list[list[int]]:
    """Split a shard's flat id stream into records at each prepended ``eos_id``.

    Each returned record keeps its leading ``eos_id``, which the model consumes as that
    document's BOS.
    """
    if not tokens:
        return []

    starts = [index for index, token in enumerate(tokens) if token == eos_id]
    if not starts:
        return [tokens]

    # The publisher writes whole records, so a shard normally opens on a boundary.
    # Keep any leading fragment rather than dropping tokens without saying so.
    records = [] if starts[0] == 0 else [tokens[: starts[0]]]

    records.extend(tokens[start:end] for start, end in zip(starts, [*starts[1:], len(tokens)], strict=True))

    return records


def open_text_lm_dataset(config: LMDatasetConfig) -> TextUnitDataset:
    # Prefer relative paths in configs: the bbfs profile is mounted at a different
    # absolute location on compute (/s3/profiles/...) and login
    # (/bbfs/s3/profiles/...) nodes, so an absolute path only works on one of them.
    # Absolute paths outside the profile (a local dir, a fixture) pass through.
    path = resolve_manifest_root(config.path)

    path = path.expanduser().resolve()
    splits = {p.name for p in path.iterdir() if p.is_dir()}

    return TextUnitDataset(path, splits)
