"""Shared waveform decode + wav2vec-style ``layer_norm`` normalization, keeping training and evaluation consistent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder

try:
    from torchcodec.decoders import WavDecoder
except ImportError:  # torchcodec < 0.10 (needed for fs2) doesn't ship WavDecoder
    WavDecoder = None

from gslm2.config import AUDIO_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from torch import Tensor

logger = logging.getLogger(__name__)


def is_voxpopuli_raw_audio_path(source: str | Path) -> bool:
    """Return whether a path refers to the VoxPopuli raw-audio tree."""
    normalized = str(source).casefold().replace("\\", "/")
    return "voxpopuli" in normalized and "raw_audios" in normalized


def require_voxpopuli_flac(audio_root: str | Path, audio_extension: str) -> None:
    """Require repaired FLAC sidecars when loading VoxPopuli raw recordings."""
    if is_voxpopuli_raw_audio_path(audio_root) and audio_extension.casefold() != ".flac":
        raise ValueError(
            "VoxPopuli raw audio must use the repaired FLAC sidecars. "
            "Set audio_extension to '.flac'. Loading the source OGG files is disabled."
        )


def reject_voxpopuli_ogg_paths(paths: Iterable[str | Path], source_hint: str | Path | None = None) -> None:
    """Reject VoxPopuli manifests that resolve to source OGG recordings on disk."""
    hinted_voxpopuli = source_hint is not None and "voxpopuli" in str(source_hint).casefold()
    for path in paths:
        if not str(path).casefold().endswith(".ogg"):
            continue
        if hinted_voxpopuli or is_voxpopuli_raw_audio_path(path):
            raise ValueError(
                "VoxPopuli OGG input is disabled because its seek timestamps are unreliable. "
                "Use the repaired sibling FLAC path instead."
            )


def open_decoder(source: str | Path | bytes) -> AudioDecoder | WavDecoder:
    """Open ``source`` with torchcodec's WAV-only decoder when it applies, else the general one.

    ``WavDecoder`` is bit-identical to ``AudioDecoder`` on PCM and float WAV and much cheaper to
    construct, but rejects mu-law/A-law and the RF64/Wave64 extensions, so a rejected header falls
    through to ``AudioDecoder``.
    """
    if isinstance(source, bytes):
        is_wav = source[:4] == b"RIFF" and source[8:12] == b"WAVE"
    else:
        reject_voxpopuli_ogg_paths([source])
        is_wav = str(source).lower().endswith(".wav")

    if is_wav and WavDecoder is not None:
        try:
            return WavDecoder(source)
        except (RuntimeError, ValueError):
            logger.debug("WavDecoder rejected %r; using AudioDecoder", source, exc_info=True)
    return AudioDecoder(source)


def decode_audio(source: str | Path | bytes) -> tuple[Tensor, int]:
    """Decode audio to ``(waveform [channels, time], sample_rate)`` via torchcodec."""
    samples = open_decoder(source).get_all_samples()
    return samples.data, samples.sample_rate


def normalize_waveform(waveform: Tensor) -> Tensor:
    """Zero-mean, unit-variance normalization over the whole waveform (wav2vec convention)."""
    return F.layer_norm(waveform, waveform.shape)


def load_audio(
    source: str | Path | bytes,
    *,
    normalize: bool = True,
    expected_sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Tensor:
    """Decode a single mono waveform to a 1-D tensor, normalized identically to ``SpeechDataset``.

    Shares the training decode + normalization so eval-time units match training-time units.
    Raises if the audio is not mono at ``expected_sample_rate`` (matching the dataset).
    """
    waveform, sample_rate = decode_audio(source)
    if sample_rate != expected_sample_rate or waveform.shape[0] != 1:
        raise ValueError(
            f"Expected mono {expected_sample_rate} Hz audio, got {waveform.shape[0]} ch @ {sample_rate} Hz: {source!r}"
        )
    if normalize:
        waveform = normalize_waveform(waveform)
    return waveform.squeeze(0)
