"""Manifest parsing, audio decoding, and aligned segment sampling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os.path import commonpath
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from gslm2.shared.audio_io import decode_audio
from gslm2.shared.units import parse_units, raw_units
from gslm2.vocoder.model import SAMPLE_RATE, UNITS_HOP_SIZE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gslm2.vocoder.config import UnitVocoderConfig

_AUDIO_NDIM = 2


@dataclass(frozen=True)
class ManifestEntry:
    """One audio path and its frame-aligned unit sequence."""

    audio: str
    units: str | list[int] | list[list[int]]


def _parse_manifest_row(row: object, path: Path, line_number: int, *, require_full_rate: bool) -> ManifestEntry:
    if not isinstance(row, dict):
        raise TypeError(f"{path}:{line_number} must contain a JSON object")
    values = cast("dict[str, object]", row)
    if require_full_rate:
        if values.get("deduplicated", False) or values.get("unit_offset", 0) != 0:
            raise ValueError(f"{path}:{line_number} requires full-rate raw code IDs")
        if (
            values.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE
            or values.get("units_hop_size", UNITS_HOP_SIZE) != UNITS_HOP_SIZE
        ):
            raise ValueError(f"{path}:{line_number} requires 16 kHz audio and 320 samples per unit")
    audio = values.get("audio", values.get("file_name"))
    units = values.get("units", values.get("audio_repr"))
    if not isinstance(audio, str):
        raise TypeError(f"{path}:{line_number} needs an audio or file_name string")
    if not isinstance(units, str | list):
        raise TypeError(f"{path}:{line_number} needs units or audio_repr")
    return ManifestEntry(audio, cast("str | list[int] | list[list[int]]", units))


def iter_manifest(path: str | Path, *, require_full_rate: bool = False) -> Iterator[ManifestEntry]:
    """Stream audio paths and units from canonical or encoder JSONL records."""
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError(f"only JSONL unit-vocoder manifests are supported: {manifest_path}")
    count = 0
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if line.strip():
                yield _parse_manifest_row(
                    json.loads(line), manifest_path, line_number, require_full_rate=require_full_rate
                )
                count += 1
    if count == 0:
        raise ValueError(f"manifest is empty: {manifest_path}")


def read_manifest(path: str | Path, *, require_full_rate: bool = False) -> list[ManifestEntry]:
    """Read audio paths and units from canonical or encoder JSONL records."""
    return list(iter_manifest(path, require_full_rate=require_full_rate))


def relative_output_paths(entries: list[ManifestEntry]) -> list[Path]:
    """Return the paths used when synthesizing a unit manifest."""
    if not entries:
        raise ValueError("cannot derive output paths from an empty manifest")
    common_root = Path(commonpath([entry.audio for entry in entries]))
    if len(entries) == 1 or common_root.suffix:
        common_root = common_root.parent
    return [Path(entry.audio).relative_to(common_root) for entry in entries]


def load_audio(source: str | Path) -> Tensor:
    """Decode and peak-normalize one mono 16 kHz waveform to 0.95."""
    audio, sample_rate = decode_audio(source)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}: {source}")
    if audio.ndim != _AUDIO_NDIM or audio.size(0) != 1:
        raise ValueError(f"expected mono audio, got shape {tuple(audio.shape)}: {source}")
    return 0.95 * audio / audio.abs().max().clamp(min=1e-8)


def align_full_utterance(units: Tensor, audio: Tensor, units_hop_size: int) -> tuple[Tensor, Tensor]:
    """Trim units and audio to their common full-utterance duration."""
    units_length = min(audio.size(-1) // units_hop_size, units.size(-1))
    if units_length == 0:
        raise ValueError("audio and units must contain at least one aligned frame")
    return units[..., :units_length], audio[..., : units_length * units_hop_size]


def crop_segment(
    units: Tensor,
    audio: Tensor,
    units_hop_size: int,
    audio_segment_size: int,
    *,
    random_crop: bool = True,
) -> tuple[Tensor, Tensor]:
    """Draw a random aligned unit/audio crop, repeating short examples first."""
    if audio_segment_size % units_hop_size:
        raise ValueError("audio_segment_size must be divisible by units_hop_size")
    units_segment_size = audio_segment_size // units_hop_size
    units, audio = align_full_utterance(units, audio, units_hop_size)
    while audio.size(-1) < audio_segment_size:
        audio = torch.hstack([audio, audio])
        units = torch.hstack([units, units])
    n_positions = audio.size(-1) // units_hop_size - units_segment_size + 1
    start_unit = int(torch.randint(0, n_positions, (1,))) if random_crop else 0
    start_audio = start_unit * units_hop_size
    return (
        units[..., start_unit : start_unit + units_segment_size],
        audio[..., start_audio : start_audio + audio_segment_size],
    )


class AudioItem(NamedTuple):
    """One aligned unit/audio pair."""

    units: Tensor
    audio: Tensor

    def to(self, device: torch.device | str) -> AudioItem:
        """Move both tensors to ``device``."""
        return AudioItem(self.units.to(device), self.audio.to(device))


type AudioDataLoader = StatefulDataLoader[AudioItem]


class AudioDataset(Dataset[AudioItem]):
    """Load cropped or full aligned unit/audio pairs from a manifest."""

    def __init__(
        self,
        manifest: str | Path,
        segment_size: int | None,
        units_hop_size: int,
    ) -> None:
        self.entries = read_manifest(manifest, require_full_rate=True)
        self.segment_size = segment_size
        self.units_hop_size = units_hop_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> AudioItem:
        entry = self.entries[index]
        units = parse_units(entry.units)
        streams = 1 if units.ndim == 1 else units.size(0)
        units = raw_units(units, [256] * streams)
        audio = load_audio(entry.audio)
        if self.segment_size is None:
            units, audio = align_full_utterance(units, audio, self.units_hop_size)
        else:
            units, audio = crop_segment(units, audio, self.units_hop_size, self.segment_size)
        return AudioItem(units, audio)


def default_num_workers() -> int:
    """Return workers per rank from SLURM or the process affinity mask."""
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    return int(cpus_per_task) if cpus_per_task is not None else len(os.sched_getaffinity(0))


def build_dataloader(
    manifest: str | Path,
    config: UnitVocoderConfig,
    *,
    is_train: bool = True,
) -> tuple[AudioDataLoader, StatefulDistributedSampler[AudioItem] | None]:
    """Build cropped training batches or full-utterance validation batches."""
    dataset = AudioDataset(
        manifest,
        config.segment_size if is_train else None,
        config.units_hop_size,
    )
    sampler: StatefulDistributedSampler[AudioItem] | None = (
        StatefulDistributedSampler(dataset, shuffle=is_train, seed=config.seed, drop_last=is_train)
        if dist.is_initialized()
        else None
    )
    workers = default_num_workers() if config.num_workers is None else config.num_workers
    loader = StatefulDataLoader(
        dataset,
        batch_size=config.batch_size if is_train else 1,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=is_train,
        generator=torch.Generator().manual_seed(config.seed + (dist.get_rank() if dist.is_initialized() else 0)),
        persistent_workers=is_train and workers > 0,
    )
    return loader, sampler
