"""Equivalence guard: ``audio_io.load_audio`` yields the same waveform as ``SpeechDataset`` (training == eval)."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf
import torch

from gslm2.data.common.audio_io import (
    decode_audio,
    load_audio,
    open_decoder,
    reject_voxpopuli_ogg_paths,
    require_voxpopuli_flac,
)
from gslm2.data.speech.dataset import speech_dataset
from tests.regression.helpers import tiny_data_config

if TYPE_CHECKING:
    from pathlib import Path


VOXPOPULI_ROOT = "/datasets/metaspeechssl/shared/data/voxpopuli/raw_audios"


def test_voxpopuli_raw_ogg_is_rejected_before_decode() -> None:
    with pytest.raises(ValueError, match="OGG input is disabled"):
        open_decoder(f"{VOXPOPULI_ROOT}/en/2010/20100101-0000_en.ogg")


def test_voxpopuli_vad_source_requires_flac() -> None:
    with pytest.raises(ValueError, match="repaired FLAC"):
        require_voxpopuli_flac(VOXPOPULI_ROOT, ".ogg")
    require_voxpopuli_flac(VOXPOPULI_ROOT, ".flac")


def test_voxpopuli_manifest_hint_rejects_relative_ogg_path() -> None:
    with pytest.raises(ValueError, match="OGG input is disabled"):
        reject_voxpopuli_ogg_paths(["en/2010/20100101-0000_en.ogg"], "VoxPopuli-eng-train.csv")


def test_non_voxpopuli_ogg_remains_supported() -> None:
    require_voxpopuli_flac("/datasets/another_corpus/raw_audios", ".ogg")
    reject_voxpopuli_ogg_paths(["/datasets/another_corpus/example.ogg"])


def test_speech_dataset_rejects_voxpopuli_ogg_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "VoxPopuli-eng-train.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fileid", "path", "num_frames", "language"])
        writer.writeheader()
        writer.writerow(
            {
                "fileid": "20100101-0000_en",
                "path": "en/2010/20100101-0000_en.ogg",
                "num_frames": 16_000,
                "language": "en",
            }
        )

    with pytest.raises(ValueError, match="OGG input is disabled"):
        speech_dataset(tiny_data_config(manifest))


def test_load_audio_matches_speech_dataset(tiny_manifest: Path) -> None:
    data_cfg = tiny_data_config(tiny_manifest)
    dataset = speech_dataset(data_cfg)
    with tiny_manifest.open() as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows):
        sample = dataset[index]
        from_dataset = sample.waveform if hasattr(sample, "waveform") else sample[0]
        from_loader = load_audio(row["path"], normalize=data_cfg.normalize)
        assert torch.equal(from_dataset, from_loader), f"row {index} ({row['path']}) diverges"


def test_decode_audio_matches_soundfile_on_flac() -> None:
    """``decode_audio`` (torchcodec) matches a ``soundfile`` full read on a mono FLAC blob.

    File and archive datasets use the shared decoder. This checks it against a
    ``soundfile`` full read of the same ``PCM_24`` mono blob.
    """
    sample_rate = 16_000
    num_samples = int(1.5 * sample_rate)
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    # sum of tones so the waveform is not trivially symmetric under a decode swap
    wave = (0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave.reshape(-1, 1), sample_rate, format="FLAC", subtype="PCM_24")
    flac = buffer.getvalue()

    with sf.SoundFile(io.BytesIO(flac)) as handle:  # reference decode (soundfile)
        reference = torch.from_numpy(handle.read(dtype="float32", always_2d=True).T)  # (C, T)
        reference_sample_rate = handle.samplerate

    waveform, decoded_sample_rate = decode_audio(flac)

    assert decoded_sample_rate == reference_sample_rate == sample_rate
    assert waveform.dtype == torch.float32
    assert waveform.shape == reference.shape
    assert waveform.shape[0] == 1  # mono
    assert torch.allclose(waveform, reference, atol=1e-6)
