"""Speech collation and length utilities (padding, cropping, masking)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from gslm2.data.common.masking.speech import MaskGenerator, build_attention_mask, build_padding_mask

if TYPE_CHECKING:
    from gslm2.data.speech.dataset import Sample, SampleBatch


def conv_length(conv_layer_config: list[tuple[int, int, int]], length: Tensor) -> Tensor:
    """Frame count out of the conv feature extractor."""
    for _, kernel_size, stride in conv_layer_config:
        length = torch.div(length - kernel_size, stride, rounding_mode="floor") + 1
        length = torch.max(torch.zeros_like(length), length)
    return length


class SLMAudioCollator:
    """Batch raw waveforms for on-the-fly SLM unit extraction (no SSL masking).

    Caps each waveform at ``max_sample_size`` (random crop when longer), right-pads
    the batch, and builds the frame-level attention mask the encoder expects. Consumes
    ``(index, Sample)`` pairs and yields ``(waveforms, attention_mask, indices)`` — the
    dataset indices key the trainer's per-file unit cache.
    """

    def __init__(
        self,
        max_sample_size: int,
        conv_layer_config: list[tuple[int, int, int]],
        *,
        rand_crop: bool = True,
    ) -> None:
        self.max_sample_size = max_sample_size
        self.conv_layer_config = conv_layer_config
        self.rand_crop = rand_crop

    def __call__(self, indexed_samples: list[tuple[int, Sample]]) -> tuple[Tensor, Tensor, Tensor]:
        indices = torch.tensor([index for index, _ in indexed_samples])
        wavs: list[Tensor] = []
        for _, sample in indexed_samples:
            wav = sample.waveform
            if wav.size(0) > self.max_sample_size:
                offset = int(torch.randint(wav.size(0) - self.max_sample_size + 1, size=(1,))) if self.rand_crop else 0
                wav = wav[offset : offset + self.max_sample_size]
            wavs.append(wav)
        lengths = torch.tensor([wav.size(0) for wav in wavs])
        waveforms = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        frame_lens = conv_length(self.conv_layer_config, lengths)
        return waveforms, build_attention_mask(frame_lens), indices


class SpeechCollatorWithMasking:
    def __init__(  # noqa: PLR0913
        self,
        mask_generator: MaskGenerator,
        max_sample_size: int,
        conv_layer_config: list[tuple[int, int, int]],
        *,
        enable_padding: bool = False,
        rand_crop: bool = True,
    ) -> None:
        self.mask_generator = mask_generator
        self.max_sample_size = max_sample_size
        self.conv_layer_config = conv_layer_config
        self.enable_padding = enable_padding
        self.rand_crop = rand_crop

    def crop_samples(self, sample: Sample, num_frames: int) -> tuple[Tensor, int]:
        frame_offset = 0
        waveform = sample.waveform

        wav_length = waveform.size(0)
        num_frames = min(num_frames, self.max_sample_size)
        if wav_length > num_frames and self.rand_crop:
            frame_offset = int(torch.randint(wav_length - num_frames, size=(1,)))
        elif wav_length < num_frames:
            num_frames = wav_length

        waveform = waveform[frame_offset : frame_offset + num_frames]

        return waveform, num_frames

    def __call__(self, samples: list[Sample]) -> SampleBatch:
        from gslm2.data.speech.dataset import SampleBatch

        wavs = [s.waveform for s in samples]
        languages = [s.language for s in samples]
        num_frames = max([len(wav) for wav in wavs]) if self.enable_padding else min([len(wav) for wav in wavs])
        wav_list, wav_lengths = zip(*[self.crop_samples(sample, num_frames) for sample in samples], strict=True)
        waveforms = torch.nn.utils.rnn.pad_sequence(wav_list, batch_first=True)  # type: ignore[arg-type]
        lengths = conv_length(self.conv_layer_config, torch.tensor(wav_lengths))
        padding_mask, attention_mask = build_padding_mask(lengths), build_attention_mask(lengths)
        mask_indices: Tensor | None = self.mask_generator(padding_mask)[0]
        return SampleBatch(
            waveforms=waveforms,
            languages=languages,
            attention_mask=attention_mask,
            mask=mask_indices,
        )
