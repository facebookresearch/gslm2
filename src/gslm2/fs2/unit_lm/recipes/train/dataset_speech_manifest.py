# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit-LM dataset family that streams raw N-stream tokens from a shard manifest tree."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pyarrow.parquet as pq
import torch
from fairseq2.data.data_pipeline import DataPipeline, read_iterator, read_sequence
from fairseq2.logging import get_log_writer

from gslm2.fs2.unit_lm.recipes.train.dataset import DataReadOptions, LMDataset
from gslm2.fs2.unit_lm.recipes.train.dataset_speech import finalize_unit_pipeline
from gslm2.fs2.unit_lm.recipes.train.storage_s3_eoz import read_manifest_s3

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import BinaryIO

    from fairseq2.datasets import DataReader, SequenceBatch
    from fairseq2.gang import Gang, Gangs

logger = get_log_writer(__name__)

MANIFEST_FILE = "shards.parquet"
START_OF_SHARD = "\x00start-of-shard:"
END_OF_SHARD = "\x00end-of-shard:"


class _ShardLines:
    """Line iterator over one shard that survives a transient storage failure."""

    def __init__(self, path: str, limit: int) -> None:
        self._path = path
        self._limit = limit
        self._offset = 0
        self._emitted = 0
        self._handle: BinaryIO | None = None

    def __iter__(self) -> _ShardLines:
        return self

    @property
    def path(self) -> str:
        return self._path

    @property
    def limit(self) -> int:
        return self._limit

    def __next__(self) -> str:
        # A manifest may select a prefix of the shard rather than the whole file.
        if self._emitted >= self._limit:
            raise StopIteration
        line = read_manifest_s3(read_fn=self._read_one, path=self._path)
        if not line:
            raise StopIteration
        self._offset += len(line)
        self._emitted += 1
        return line.decode("utf-8").rstrip()

    def _read_one(self) -> bytes:
        """One raw read: open + seek on first call, then ``readline`` on the shared handle."""
        if self._handle is None:
            self._handle = Path(self._path).open("rb")  # noqa: SIM115 (outlives this call)
            self._handle.seek(self._offset)
        try:
            return self._handle.readline()
        except OSError:
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None
            raise

    def __getstate__(self) -> dict[str, object]:
        return {
            "path": self._path,
            "limit": self._limit,
            "offset": self._offset,
            "emitted": self._emitted,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._path = str(state["path"])
        self._limit = int(state["limit"])  # type: ignore[call-overload]
        self._offset = int(state["offset"])  # type: ignore[call-overload]
        self._emitted = int(state["emitted"])  # type: ignore[call-overload]
        self._handle = None


def _restart_shard(lines: _ShardLines) -> _ShardLines:
    """Rebuild a shard iterator from scratch, for ``read_iterator``'s reset."""
    return _ShardLines(lines.path, lines.limit)


@dataclass
class ManifestUnitDatasetConfig:
    """Configuration for the manifest-backed unit token family."""

    path: Path = field(default_factory=Path)
    layers: list[int] | None = None

    reset_data_reader: bool = False  # discard the data-reader position stored in a restored checkpoint.


@dataclass(frozen=True)
class _Shard:
    dataset: str
    tokens_key: str
    records: int
    read_records: int
    audio_hours: float


def _read_manifest(manifest_dir: Path) -> list[_Shard]:
    """Read a published shard manifest.

    Takes the split directory; returns one ``_Shard`` per token file. ``tokens_key``
    is the raw string from the parquet, relative to the manifest tree root; the
    caller joins it against that root to open the file.
    """
    if not (manifest_dir / MANIFEST_FILE).is_file():
        raise FileNotFoundError(f"{manifest_dir} is not a manifest (missing {MANIFEST_FILE})")
    path = manifest_dir / MANIFEST_FILE
    columns = ["dataset", "tokens_key", "records", "audio_hours"]
    sliced = "read_records" in pq.read_schema(path).names
    table = pq.read_table(path, columns=[*columns, "read_records"] if sliced else columns)
    shards = []
    for row in table.to_pylist():
        records = int(row["records"])
        limit = row["read_records"] if sliced else None
        read_records = records if limit is None else int(limit)
        if not 0 < read_records <= records:
            raise ValueError(
                f"{manifest_dir.name} selects {read_records} of {records} records from "
                f"{row['tokens_key']}; the selection must be a non-empty prefix of the file"
            )
        shards.append(
            _Shard(
                dataset=row["dataset"],
                tokens_key=row["tokens_key"],
                records=records,
                read_records=read_records,
                audio_hours=float(row["audio_hours"]),
            )
        )
    return shards


def _dedup_layers(tokens: dict[str, list[int]], layers: Sequence[int]) -> np.ndarray:
    """Drop frames in which every requested layer repeats its previous value."""
    matrix = np.asarray([tokens[f"layer_{layer}"] for layer in layers], dtype=np.int16)
    if matrix.shape[1] == 0:
        return matrix
    keep = np.empty(matrix.shape[1], dtype=bool)
    keep[0] = True
    np.any(matrix[:, 1:] != matrix[:, :-1], axis=0, out=keep[1:])
    return matrix[:, keep]


def _chunk(matrix: np.ndarray, max_seq_len: int) -> list[torch.Tensor]:
    """Split an ``(S, T)`` token matrix into examples of at most ``max_seq_len`` frames."""
    frames = matrix.shape[1]
    return [
        torch.from_numpy(np.ascontiguousarray(matrix[:, start : start + max_seq_len]).reshape(-1).astype(np.int64))
        for start in range(0, frames, max_seq_len)
    ]


class ManifestUnitDataset(LMDataset):
    """Streams raw N-stream unit tokens from a shard-manifest tree."""

    def __init__(
        self,
        base_path: Path,
        splits: set[str],
        layers: Sequence[int] | None,
        *,
        reset_data_reader: bool = False,
    ) -> None:
        super().__init__(base_path, splits)
        self._layers = tuple(layers) if layers else None
        self._reset_data_reader = reset_data_reader

    @staticmethod
    @override
    def from_path(path: Path) -> ManifestUnitDataset:
        splits = {p.name for p in path.iterdir() if (p / MANIFEST_FILE).is_file()}
        return ManifestUnitDataset(path, splits, None)

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
        """Build a reader over one manifest.

        ``split`` names a manifest under the dataset root. ``num_io_streams`` must
        equal the number of configured layers. Returns a reader yielding
        ``SequenceBatch`` objects.
        """
        if split not in self._splits:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self._base_path}'. Available splits: {sorted(self._splits)}"
            )
        if options is None:
            options = DataReadOptions()
        layers = self._resolve_layers(num_io_streams)

        seed = options.seed
        gang = gangs.dp
        shards = _read_manifest(self._base_path / split)

        by_dataset: dict[str, list[_Shard]] = {}
        for shard in shards:
            by_dataset.setdefault(shard.dataset, []).append(shard)

        names = sorted(by_dataset)
        pipelines = [
            self._dataset_pipeline(
                by_dataset[name],
                gang=gang,
                seed=seed,
                options=options,
                max_seq_len=max_seq_len,
                layers=layers,
            )
            for name in names
        ]

        if len(pipelines) == 1:
            builder = DataPipeline.concat(pipelines)
        else:
            # `sample` draws one example at a time, so weighting by hours would
            # over-sample long-utterance sources; record count gives the intended mix.
            total_records = {n: sum(s.read_records for s in by_dataset[n]) for n in names}
            grand = sum(total_records.values())
            weights = [total_records[n] / grand for n in names]
            builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

        return finalize_unit_pipeline(
            builder,
            gangs,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            num_io_streams=num_io_streams,
            options=options,
            seed=seed,
            reset_state=self._reset_data_reader,
        )

    def _resolve_layers(self, num_io_streams: int) -> tuple[int, ...]:
        """Validate the configured layers against the model's stream count.

        Returns the layers to read, in stream order. Which layer ids are actually
        stored is a property of the on-disk shard files; a mismatch surfaces as a
        ``KeyError`` on the first record.
        """
        if self._layers is None:
            raise ValueError(
                "layers must be configured; they are taken from the extraction layers "
                "used to select which of the stored SSL layers to train on."
            )
        if len(self._layers) != num_io_streams:
            raise ValueError(
                f"model expects {num_io_streams} stream(s) but {len(self._layers)} "
                f"layer(s) are configured ({list(self._layers)}). num_io_streams is "
                "derived from the length of model.vocab_size."
            )
        return self._layers

    def _dataset_pipeline(  # noqa: PLR0913
        self,
        shards: list[_Shard],
        *,
        gang: Gang,
        seed: int,
        options: DataReadOptions,
        max_seq_len: int,
        layers: tuple[int, ...],
    ) -> DataPipeline:
        """Build the per-source-dataset pipeline feeding the mixer.

        Takes that dataset's shards and the read options; returns a pipeline
        yielding 1-D int64 tensors, one per training example, in on-disk unit ids.
        """
        keys = [str(self._base_path / shard.tokens_key) for shard in shards]
        expected_records = {k: shard.read_records for k, shard in zip(keys, shards, strict=True)}

        builder = read_sequence(keys)
        builder.shuffle(shuffle_window=0, seed=seed)

        # Shard whole files when there are enough to go round, otherwise shard the
        # example stream so small splits do not starve ranks.
        shard_by_file = len(keys) >= gang.size
        if shard_by_file:
            builder.shard(gang.rank, gang.size, allow_uneven=True)

        # Network-backed storage can serve a stale short body as a clean EOF, silently
        # dropping a shard's tail; the START/END markers below let us detect that. Only
        # checkable when whole files go to one rank.
        verify_records = shard_by_file

        def read_shard(key: str) -> DataPipeline:
            # `_ShardLines` stops at the manifest's record count, so a sliced manifest
            # never reads the tail. A no-op when the selection is the whole file.
            lines = read_iterator(
                _ShardLines(key, expected_records[key]),
                reset_fn=_restart_shard,
                infinite=False,
            ).and_return()
            if not verify_records:
                return lines
            start = read_sequence([START_OF_SHARD + key]).and_return()
            marker = read_sequence([END_OF_SHARD + key]).and_return()
            return DataPipeline.concat([start, lines, marker]).and_return()

        builder.yield_from(read_shard)

        if not shard_by_file:
            builder.shard(gang.rank, gang.size, allow_uneven=True)

        # Zeroed by the shard's own START, not the previous shard's END, which `sample`
        # may skip; `started` then limits verification to shards seen from the beginning.
        seen = {"records": 0, "started": False}

        def to_chunks(line: str) -> DataPipeline:
            if verify_records and line.startswith(START_OF_SHARD):
                seen["records"] = 0
                seen["started"] = True
                return read_sequence([]).and_return()
            if verify_records and line.startswith(END_OF_SHARD):
                key = line[len(END_OF_SHARD) :]
                got, want = seen["records"], expected_records[key]
                if seen["started"] and got != want:
                    raise RuntimeError(
                        f"{key} yielded {got} records but the manifest selects {want}. "
                        "The storage layer can serve a truncated read as a clean EOF; "
                        "retry the job rather than training on a partial shard."
                    )
                seen["started"] = False
                return read_sequence([]).and_return()
            seen["records"] += 1
            record = json.loads(line)
            chunks = _chunk(_dedup_layers(record["tokens"], layers), max_seq_len)
            return read_sequence(chunks).and_return()

        builder.yield_from(to_chunks)

        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        return builder.and_return()


def open_manifest_unit_dataset(config: ManifestUnitDatasetConfig) -> ManifestUnitDataset:
    """Open the dataset rooted at ``config.path`` (absolute), discovering its splits.

    Callers that use bbfs should resolve ``config.path`` upstream via
    ``storage_s3_eoz.resolve_manifest_root`` before invoking this opener.
    """
    path = Path(config.path).expanduser().resolve()
    splits = {p.name for p in path.iterdir() if (p / MANIFEST_FILE).is_file()}
    if not splits:
        raise FileNotFoundError(f"No manifests under {path}")
    return ManifestUnitDataset(path, splits, config.layers, reset_data_reader=config.reset_data_reader)
