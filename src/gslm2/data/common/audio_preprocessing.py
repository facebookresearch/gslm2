"""Audio → log-mel spectrogram preprocessing.

Kaldi-fbank pipeline: resample → DC remove → kaldi.fbank (Hanning, no dither,
no energy, htk_compat=True) → zero-pad / truncate to a fixed time length →
normalize with ``(mel - mean) / (2 * std)``.

Output shape is ``[1, T, F]`` (float32), ready to be unsqueezed to
``[B, 1, T, F]`` for spectrogram JEPA models.
"""

from __future__ import annotations

import torch

AUDIO_MEAN: float = -4.268  # AudioSet stats (AudioMAE / AST convention)
AUDIO_STD: float = 4.569  # AudioSet stats (AudioMAE / AST convention)


def waveform_to_log_mel(  # noqa: PLR0913
    wav: torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int = 16_000,
    num_mel_bins: int = 128,
    target_length: int = 1024,
    frame_shift_ms: int = 10,
    mean: float = AUDIO_MEAN,
    std: float = AUDIO_STD,
) -> torch.Tensor:
    """Convert a waveform to a normalized log-mel spectrogram.

    Args:
        wav: ``[C, S]`` or ``[S]`` waveform tensor (any dtype convertible to
            float32). Multi-channel input is downmixed by averaging.
        sample_rate: Source sample rate of ``wav``.
        target_sample_rate: Sample rate fed to the mel-filterbank.
        num_mel_bins: Number of mel bins (frequency axis).
        target_length: Fixed time length in frames; pad with zeros or truncate
            as needed so output is always ``[1, target_length, num_mel_bins]``.
        frame_shift_ms: Hop size in milliseconds for ``kaldi.fbank``.
        mean, std: Normalization stats; output is ``(mel - mean) / (2 * std)``.

    Returns:
        ``[1, target_length, num_mel_bins]`` float32 tensor.
    """
    # ``torchaudio`` is an optional-feeling dep elsewhere in this repo — keep
    # the import local so importing the preprocessing module doesn't break
    # environments that don't need spectrograms.
    import torchaudio

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() != 2:  # noqa: PLR2004
        raise ValueError(f"wav must be [S] or [C, S]; got shape {tuple(wav.shape)}")

    wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
    wav = wav - wav.mean()

    mel = torchaudio.compliance.kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=target_sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=frame_shift_ms,
    ).unsqueeze(0)  # [1, T, F]

    n_frames = mel.shape[1]
    if n_frames < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - n_frames))
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    return (mel - mean) / (std * 2)


__all__ = ["AUDIO_MEAN", "AUDIO_STD", "waveform_to_log_mel"]
