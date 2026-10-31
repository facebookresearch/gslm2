# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from fairseq2.data.data_pipeline import (
    read_sequence,
)
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch, SyncMode

if TYPE_CHECKING:
    from fairseq2.gang import Gangs


class LMDataset(ABC):
    """A base class for LM fs2 datasets,
    containing shared logic between different modalities."""

    _base_path: Path
    _splits: set[str]

    def __init__(self, base_path: Path, splits: set[str]) -> None:
        self._base_path = base_path
        self._splits = splits

    @staticmethod
    @abstractmethod
    def from_path(path: Path) -> LMDataset: ...

    @abstractmethod
    def create_reader(  # noqa: PLR0913
        self,
        split: str,
        gangs: Gangs,
        min_seq_len: int,
        max_seq_len: int,
        options: DataReadOptions | None = None,
        dynamic_chunking_batch_size: int = 1,
        num_io_streams: int = 1,
    ) -> DataReader[SequenceBatch]:
        # Move common logic between speech and text here
        ...


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""


type Batching = StaticBatching | LengthBatching


@dataclass(kw_only=True)
class DataReadOptions:
    batching: Batching = field(default_factory=lambda: StaticBatching(1))
    """The batching strategy for returned examples."""

    example_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling examples. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    batch_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling batches. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    drop_remainder: bool = False
    """Whether to drop the last incomplete batch."""

    strict_state: bool = False
    """Whether to persist the pipeline's in-flight buffers in the checkpoint."""

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """
    The data synchronization mode among processes in the gang. Only effective if
    :attr:`sync_batches` is ``True``.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 1
    """
    The number of batches to accumulate in each iteration. Typically used with
    gradient accumulation during training.
    """

    prefetch: int = 1
    """The number of batches to prefetch in background."""

    npc: int = 10
    """The reference number of parallel calls that data reader can do."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

    pack: bool = False
    """If ``True``, use sequence packing."""


@dataclass
class LMDatasetConfig:
    path: Path = field(default_factory=Path)


def single_step_dummy_reader(gangs: Gangs) -> DataReader[SequenceBatch]:
    """
    Returns a dummy data reader that reads a single step with a single batch.
    This is useful for testing purposes or when you need a placeholder reader.
    """
    dummy_batch = SequenceBatch(torch.tensor([[0]]), [1], example=None)
    return DataPipelineReader[SequenceBatch](
        read_sequence([dummy_batch]).and_return(),
        gangs,
        num_accumulate=1,
        drop_remainder=False,
        sync=True,
        sync_mode=SyncMode.UNTIL_LAST,
    )
