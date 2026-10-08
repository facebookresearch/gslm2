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
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchcodec.decoders import AudioDecoder

from gslm2.vocoder.unit.model import SAMPLE_RATE

if TYPE_CHECKING:
    from gslm2.vocoder.unit.config import UnitVocoderConfig

STREAM_SEPARATOR = "|"
_AUDIO_NDIM = 2


@dataclass(frozen=True)
class ManifestEntry:
    """One audio path and its frame-aligned unit sequence."""

    audio: str
    units: str | list[int] | list[list[int]]


def _parse_manifest_row(row: object, path: Path, line_number: int) -> ManifestEntry:
    if not isinstance(row, dict):
        raise TypeError(f"{path}:{line_number} must contain a JSON object")
    values = cast("dict[str, object]", row)
    audio = values.get("audio", values.get("file_name"))
    units = values.get("units", values.get("audio_repr"))
    if not isinstance(audio, str):
        raise TypeError(f"{path}:{line_number} needs an audio or file_name string")
    if not isinstance(units, str | list):
        raise TypeError(f"{path}:{line_number} needs units or audio_repr")
    return ManifestEntry(audio, cast("str | list[int] | list[list[int]]", units))


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read the JSONL schema emitted by SLM unit-vocoder feature extraction."""
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError(f"only JSONL unit-vocoder manifests are supported: {manifest_path}")
    entries = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if line.strip():
                entries.append(_parse_manifest_row(json.loads(line), manifest_path, line_number))
    if not entries:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return entries


def relative_output_paths(entries: list[ManifestEntry]) -> list[Path]:
    """Return the paths used when synthesizing a unit manifest."""
    if not entries:
        raise ValueError("cannot derive output paths from an empty manifest")
    common_root = Path(commonpath([entry.audio for entry in entries]))
    if len(entries) == 1 or common_root.suffix:
        common_root = common_root.parent
    return [Path(entry.audio).relative_to(common_root) for entry in entries]


def parse_units(units: str | list[int] | list[list[int]]) -> Tensor:
    """Parse a single stream or equal-length ``|``-separated streams."""
    if isinstance(units, str):
        stream_strings = units.split(STREAM_SEPARATOR)
        streams = [[int(token) for token in stream.split()] for stream in stream_strings]
    elif units and isinstance(units[0], list):
        streams = cast("list[list[int]]", units)
    else:
        return torch.tensor(cast("list[int]", units), dtype=torch.long)

    lengths = {len(stream) for stream in streams}
    if len(lengths) != 1:
        raise ValueError("all unit streams must have the same length")
    tensors = [torch.tensor(stream, dtype=torch.long) for stream in streams]
    return torch.stack(tensors) if len(tensors) > 1 else tensors[0]


def load_audio(source: str | Path) -> Tensor:
    """Decode and peak-normalize one mono 16 kHz waveform to 0.95."""
    samples = AudioDecoder(str(source)).get_all_samples()
    if samples.sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {samples.sample_rate}: {source}")
    audio = samples.data
    if audio.ndim != _AUDIO_NDIM or audio.size(0) != 1:
        raise ValueError(f"expected mono audio, got shape {tuple(audio.shape)}: {source}")
    peak = audio.abs().max()
    if peak == 0:
        raise ValueError(f"cannot normalize silent audio: {source}")
    return 0.95 * audio / peak


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
    units_length = min(audio.size(-1) // units_hop_size, units.size(-1))
    if units_length == 0:
        raise ValueError("audio and units must contain at least one aligned frame")
    units = units[..., :units_length]
    audio = audio[..., : units_length * units_hop_size]
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
    """One fixed-size unit/audio training pair."""

    units: Tensor
    audio: Tensor

    def to(self, device: torch.device | str) -> AudioItem:
        """Move both tensors to ``device``."""
        return AudioItem(self.units.to(device), self.audio.to(device))


class AudioDataset(Dataset[AudioItem]):
    """Load aligned unit/audio segments from an extraction manifest."""

    def __init__(
        self,
        manifest: str | Path,
        segment_size: int,
        units_hop_size: int,
        *,
        random_crop: bool = True,
    ) -> None:
        self.entries = read_manifest(manifest)
        self.segment_size = segment_size
        self.units_hop_size = units_hop_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> AudioItem:
        entry = self.entries[index]
        units = parse_units(entry.units)
        audio = load_audio(entry.audio)
        units, audio = crop_segment(
            units,
            audio,
            self.units_hop_size,
            self.segment_size,
            random_crop=self.random_crop,
        )
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
) -> tuple[DataLoader[AudioItem], DistributedSampler[AudioItem] | None]:
    """Build a deterministic local or distributed unit/audio loader."""
    dataset = AudioDataset(
        manifest,
        config.segment_size,
        config.units_hop_size,
        random_crop=is_train,
    )
    sampler: DistributedSampler[AudioItem] | None = (
        DistributedSampler(dataset, shuffle=is_train, seed=config.seed, drop_last=is_train)
        if dist.is_initialized()
        else None
    )
    workers = default_num_workers() if config.num_workers is None else config.num_workers
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=is_train,
        generator=torch.Generator().manual_seed(config.seed + (dist.get_rank() if dist.is_initialized() else 0)),
        persistent_workers=is_train and workers > 0,
    )
    return loader, sampler
