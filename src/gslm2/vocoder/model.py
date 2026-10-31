# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit HiFi-GAN generator and discriminator architectures.

Adapted from ``unit-hifigan`` by Maxime Poli under the MIT license.
See https://github.com/bootphon/unit-hifigan/blob/main/src/unit_hifigan/model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torchaudio.transforms import MelSpectrogram

if TYPE_CHECKING:
    from collections.abc import Sequence

SAMPLE_RATE = 16_000
LRELU_SLOPE = 0.1
UNIT_EMBEDDING_DIM = 128
MULTISTREAM_EMBEDDING_DIM = 64
UNITS_HOP_SIZE = 320

_UPSAMPLE_STRIDES = (5, 4, 4, 2, 2)
_UPSAMPLE_KERNEL_SIZES = (11, 8, 8, 4, 4)
_MRF_KERNEL_SIZES = (3, 7, 11)
_MRF_DILATIONS = (1, 3, 5)
_MULTISTREAM_INPUT_NDIM = 3
_SINGLE_STREAM_INPUT_NDIM = 2


def one_dim_to_two_dim(waveform: Tensor, period: int) -> Tensor:
    """Fold a waveform into period-sized rows for a period discriminator."""
    batch_size, channels, length = waveform.size()
    if length % period:
        pad_length = period - length % period
        waveform = F.pad(waveform, (0, pad_length), "reflect")
        length += pad_length
    return waveform.view(batch_size, channels, length // period, period)


class PeriodDiscriminator(nn.Module):
    """Discriminate a waveform after folding it at one fixed period."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1_024, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(1_024, 1_024, (5, 1), 1, (2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1_024, 1, (3, 1), 1, (1, 0)))

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return discriminator scores and intermediate feature maps."""
        feature_maps = []
        output = one_dim_to_two_dim(waveform, self.period)
        for conv in self.convs:
            output = F.leaky_relu(conv(output), LRELU_SLOPE)
            feature_maps.append(output)
        output = self.conv_post(output)
        feature_maps.append(output)
        return output.view(output.size(0), -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Apply period discriminators for the canonical HiFi-GAN periods."""

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in (2, 3, 5, 7, 11)])

    def forward(self, waveform: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Return scores and flattened feature maps from every period."""
        outputs = [discriminator(waveform) for discriminator in self.discriminators]
        scores = tuple(output[0] for output in outputs)
        features = tuple(feature for output in outputs for feature in output[1])
        return scores, features


class ScaleDiscriminator(nn.Module):
    """Discriminate a waveform at one temporal scale."""

    def __init__(self, *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, 7)),
                norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
                norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
                norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
                norm(nn.Conv1d(512, 1_024, 41, 4, 20, groups=16)),
                norm(nn.Conv1d(1_024, 1_024, 41, 1, 20, groups=16)),
                norm(nn.Conv1d(1_024, 1_024, 5, 1, 2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1_024, 1, 3, 1, 1))

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return discriminator scores and intermediate feature maps."""
        output = waveform
        feature_maps = []
        for conv in self.convs:
            output = F.leaky_relu(conv(output), LRELU_SLOPE)
            feature_maps.append(output)
        output = self.conv_post(output)
        feature_maps.append(output)
        return output.view(output.size(0), -1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Apply three discriminators at progressively pooled scales."""

    def __init__(self) -> None:
        super().__init__()
        self.poolings = nn.ModuleList([nn.Identity(), nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=True), ScaleDiscriminator(), ScaleDiscriminator()]
        )

    def forward(self, waveform: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Return scores and flattened feature maps from every scale."""
        scores: list[Tensor] = []
        feature_maps: list[Tensor] = []
        output = waveform
        for discriminator, pooling in zip(self.discriminators, self.poolings, strict=True):
            output = pooling(output)
            score, features = discriminator(output)
            scores.append(score)
            feature_maps.extend(features)
        return tuple(scores), tuple(feature_maps)


class UnitDiscriminatorOutput(NamedTuple):
    """Discriminator outputs used by adversarial and reconstruction losses."""

    log_mel_spectrogram: Tensor
    mpd: tuple[Tensor, ...]
    mpd_features: tuple[Tensor, ...]
    msd: tuple[Tensor, ...]
    msd_features: tuple[Tensor, ...]


class UnitDiscriminator(nn.Module):
    """Combined mel, multi-period, and multi-scale discriminator."""

    def __init__(self) -> None:
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1_024,
            win_length=1_024,
            hop_length=256,
            f_min=0,
            f_max=8_000,
            n_mels=80,
            window_fn=torch.hann_window,
            power=1,
            normalized=False,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )

    @torch.compiler.disable
    def log_mel(self, waveform: Tensor) -> Tensor:
        """Compute the log-mel representation used for reconstruction loss."""
        with torch.autocast(waveform.device.type, enabled=False):
            padding = (1_024 - 256) // 2
            padded = F.pad(waveform.float().unsqueeze(1), (padding, padding), "reflect").squeeze(1)
            return torch.clamp(self.mel_spectrogram(padded), min=1e-5).log()

    def forward(self, audio: Tensor) -> UnitDiscriminatorOutput:
        """Score ``(batch, 1, samples)`` audio and return all loss features."""
        log_mel_spectrogram = self.log_mel(audio.squeeze(1))
        mpd, mpd_features = self.mpd(audio)
        msd, msd_features = self.msd(audio)
        return UnitDiscriminatorOutput(log_mel_spectrogram, mpd, mpd_features, msd, msd_features)


class ConvLayer(nn.Module):
    """Two-convolution residual layer used in each receptive-field block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding_conv1 = (kernel_size * dilation - dilation) // 2
        padding_conv2 = (kernel_size - 1) // 2
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding_conv1, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding_conv2))

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the residual convolution layer."""
        output = F.leaky_relu(inputs, LRELU_SLOPE)
        output = self.conv1(output)
        output = F.leaky_relu(output, LRELU_SLOPE)
        output = self.conv2(output)
        return inputs + output


class MultiReceptiveFieldFusion(nn.Module):
    """Average residual blocks with several temporal receptive fields."""

    def __init__(self, channels: int, kernel_sizes: Sequence[int], dilations: Sequence[int]) -> None:
        super().__init__()
        self.channels = channels
        self.residual_blocks = nn.ModuleList(
            nn.Sequential(*[ConvLayer(channels, kernel_size, dilation) for dilation in dilations])
            for kernel_size in kernel_sizes
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Fuse all receptive-field branches."""
        output = torch.zeros_like(inputs)
        for block in self.residual_blocks:
            output += block(inputs)
        return output / len(self.residual_blocks)


def _upsamplers(input_channels: int) -> nn.ModuleList:
    upsampler_layers = nn.ModuleList()
    for index, (kernel_size, stride) in enumerate(zip(_UPSAMPLE_KERNEL_SIZES, _UPSAMPLE_STRIDES, strict=True)):
        in_channels = input_channels // 2**index
        out_channels = input_channels // 2 ** (index + 1)
        padding = (kernel_size - stride) // 2
        upsampler_layers.append(
            weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding))
        )
    return upsampler_layers


def _mrfs(input_channels: int) -> nn.ModuleList:
    return nn.ModuleList(
        MultiReceptiveFieldFusion(input_channels // 2 ** (index + 1), _MRF_KERNEL_SIZES, _MRF_DILATIONS)
        for index in range(len(_UPSAMPLE_STRIDES))
    )


class Generator(nn.Module):
    """Canonical unit HiFi-GAN waveform generator for a 320-sample hop."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        input_channels = 512
        self.n_upsamples = len(_UPSAMPLE_STRIDES)
        self.conv_pre = weight_norm(nn.Conv1d(input_dim, input_channels, 7, padding=3))
        self.upsamplers = _upsamplers(input_channels)
        self.mrfs = _mrfs(input_channels)
        last_mrf = cast("MultiReceptiveFieldFusion", self.mrfs[-1])
        self.conv_post = weight_norm(nn.Conv1d(last_mrf.channels, 1, 7, padding=3))

    def forward(self, inputs: Tensor) -> Tensor:
        """Upsample ``(batch, channels, frames)`` features into audio."""
        output = self.conv_pre(inputs)
        for index in range(self.n_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.upsamplers[index](output)
            output = self.mrfs[index](output)
        output = F.leaky_relu(output)
        return self.conv_post(output).tanh()


class UnitVocoder(nn.Module):
    """Generate 16 kHz audio from one 256-unit stream or six aligned streams."""

    def __init__(self, n_units: int | list[int], *, agg_type: str = "average") -> None:
        super().__init__()
        self.agg_type = agg_type
        self.unit_embedding: nn.Embedding | nn.ModuleList
        self.agg: nn.Sequential | None = None
        if isinstance(n_units, list):
            if not n_units or any(size <= 0 for size in n_units):
                raise ValueError("all unit vocabularies must be positive")
            if agg_type != "concat_lin":
                raise ValueError("multi-stream unit vocoders require agg_type='concat_lin'")
            concatenated_dim = MULTISTREAM_EMBEDDING_DIM * len(n_units)
            self.unit_embedding = nn.ModuleList(nn.Embedding(size, MULTISTREAM_EMBEDDING_DIM) for size in n_units)
            self.agg = nn.Sequential(
                nn.LayerNorm(concatenated_dim),
                nn.Linear(concatenated_dim, UNIT_EMBEDDING_DIM),
                nn.Dropout(0.1),
            )
        else:
            if n_units <= 0:
                raise ValueError("unit vocabulary must be positive")
            if agg_type != "average":
                raise ValueError("single-stream unit vocoders require agg_type='average'")
            self.unit_embedding = nn.Embedding(n_units, UNIT_EMBEDDING_DIM)
        self.generator = Generator(UNIT_EMBEDDING_DIM)

    def forward(self, units: Tensor) -> Tensor:
        """Map ``(B,T)`` or ``(B,S,T)`` unit IDs to ``(B,1,T*320)`` audio."""
        if isinstance(self.unit_embedding, nn.ModuleList):
            if units.ndim != _MULTISTREAM_INPUT_NDIM or units.size(1) != len(self.unit_embedding):
                raise ValueError(f"expected units with shape (batch, {len(self.unit_embedding)}, frames)")
            embeddings = [cast("nn.Embedding", embedding) for embedding in self.unit_embedding]
            embedded = [embedding(units[:, index]) for index, embedding in enumerate(embeddings)]
            if self.agg is None:
                raise RuntimeError("multi-stream aggregation layer is not initialized")
            features = self.agg(torch.cat(embedded, dim=-1)).transpose(1, 2)
        else:
            if units.ndim != _SINGLE_STREAM_INPUT_NDIM:
                raise ValueError("expected units with shape (batch, frames)")
            features = self.unit_embedding(units).transpose(1, 2)
        return self.generator(features)

    @torch.inference_mode()
    def generate(self, units: Tensor) -> Tensor:
        """Generate peak-normalized ``(batch, samples)`` audio."""
        audio = self(units).squeeze(1)
        peak = audio.abs().max(dim=-1, keepdim=True).values
        return audio / peak.clamp_min(torch.finfo(audio.dtype).tiny)

    @property
    def n_units(self) -> int | list[int]:
        """Return the configured vocabulary size for each unit stream."""
        if isinstance(self.unit_embedding, nn.ModuleList):
            embeddings = [cast("nn.Embedding", embedding) for embedding in self.unit_embedding]
            return [embedding.weight.size(0) for embedding in embeddings]
        return self.unit_embedding.weight.size(0)

    def save_pretrained(self, path: str | Path) -> None:
        """Save a local checkpoint compatible with standalone unit-hifigan."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "agg_type": self.agg_type,
            "embedding_dim": UNIT_EMBEDDING_DIM,
            "f0_bins": None,
            "n_units": self.n_units,
            "speakers": None,
            "styles": None,
            "units_hop_size": UNITS_HOP_SIZE,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        state_dict = {name: tensor.detach().cpu().contiguous() for name, tensor in self.state_dict().items()}
        save_file(state_dict, output_dir / "model.safetensors", metadata={"format": "pt"})

    @classmethod
    def from_pretrained(cls, path: str | Path) -> UnitVocoder:
        """Load a local standalone-compatible unit-hifigan checkpoint."""
        checkpoint_dir = Path(path)
        raw_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise TypeError("vocoder config must be a JSON object")
        config = cast("dict[str, object]", raw_config)
        if config.get("embedding_dim", UNIT_EMBEDDING_DIM) != UNIT_EMBEDDING_DIM:
            raise ValueError("only embedding_dim=128 checkpoints are supported")
        if config.get("units_hop_size", UNITS_HOP_SIZE) != UNITS_HOP_SIZE:
            raise ValueError("only units_hop_size=320 checkpoints are supported")
        if any(config.get(name) is not None for name in ("speakers", "styles", "f0_bins")):
            raise ValueError("speaker, style, and F0-conditioned checkpoints are not supported")
        n_units = config.get("n_units")
        if not isinstance(n_units, int | list):
            raise TypeError("vocoder config n_units must be an integer or list")
        if isinstance(n_units, list) and not all(isinstance(size, int) for size in n_units):
            raise TypeError("all n_units entries must be integers")
        model = cls(cast("int | list[int]", n_units), agg_type=str(config.get("agg_type", "average")))
        model.load_state_dict(load_file(checkpoint_dir / "model.safetensors", device="cpu"))
        return model
