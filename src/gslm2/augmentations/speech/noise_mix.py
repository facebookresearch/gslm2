import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import AudioDecoder

from gslm2.augmentations.speech.base import AudioAugmentation

logger = logging.getLogger()

MAX_INTERNOISE_SILENCE_RATIO = 0.2
MAX_CLIPPING_AMPLITUDE = 0.9


class NoiseMixAugmentation(AudioAugmentation):
    """
    Class to mix noise into audio signals at a specified SNR.
    """

    def __init__(
        self, sample_rate: int, noise_database: str | Path, snr_min: float, snr_max: float, seed: int = 0
    ) -> None:
        logger.info("Initializing noise_mix augmentation...")
        super().__init__(sample_rate=sample_rate, seed=seed)
        self.noise_database = pd.read_csv(noise_database)
        assert all(col in self.noise_database.columns for col in ["path", "num_frames"]), (
            "Noise database must contain 'path' and 'num_frames' columns"
        )
        self.snr_min = snr_min
        self.snr_max = snr_max

    def sample_num_frames(self, num_frames: int, min_num_frames: int) -> tuple[int, int]:
        if num_frames < min_num_frames:
            raise ValueError(f"num_frames must be at least {min_num_frames}, got {num_frames}")
        sampled_num_frames = np.random.randint(min_num_frames, num_frames + 1)  # noqa: NPY002
        sampled_start_frame = np.random.randint(0, num_frames - sampled_num_frames + 1)  # noqa: NPY002
        return sampled_num_frames, sampled_start_frame

    def load_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Loads noise sample from noise database with size, dtype, and device matching the input waveform.
        Args:
            waveform (torch.Tensor): Input audio signal (T).
        Returns:
            torch.Tensor: Noise waveform with the same shape as the input waveform.
        """
        num_audio_frames = waveform.shape[-1]
        noise_audios = []
        num_noise_frames = 0
        max_internoise_silence = int(num_audio_frames * MAX_INTERNOISE_SILENCE_RATIO)
        while num_noise_frames < num_audio_frames:
            # add some silence in between noise samples
            silence_frames = np.random.randint(0, max_internoise_silence + 1)  # noqa: NPY002
            silence_audio = torch.zeros((silence_frames), dtype=waveform.dtype, device=waveform.device)
            num_noise_frames += silence_frames
            noise_audios.append(silence_audio)

            # add noise
            noise_dat = self.noise_database.iloc[np.random.randint(0, len(self.noise_database))]  # noqa: NPY002
            # noise will be a random length from L//2 to Lsn
            sampled_num_frames, sampled_start_frame = self.sample_num_frames(
                noise_dat["num_frames"], min_num_frames=noise_dat["num_frames"] // 2
            )
            num_noise_frames += sampled_num_frames
            noise_samples = AudioDecoder(noise_dat["path"]).get_all_samples()
            assert noise_samples.data.shape[0] == 1, "Noise audio must be mono (single channel)"
            assert noise_samples.sample_rate == self.sample_rate, (
                f"Sample rate mismatch for noise data: expected {self.sample_rate}, got {noise_samples.sample_rate}"
            )
            noise_audio = noise_samples.data.squeeze(0)[
                sampled_start_frame : sampled_start_frame + sampled_num_frames
            ].to(dtype=waveform.dtype, device=waveform.device)
            noise_audios.append(noise_audio)
        noise_waveform = torch.cat(noise_audios)
        return noise_waveform[:num_audio_frames]

    def solve_clipping(self, waveform: torch.Tensor) -> torch.Tensor:
        amplitude_max = torch.max(torch.abs(waveform))
        if amplitude_max < MAX_CLIPPING_AMPLITUDE:
            return waveform
        scale = MAX_CLIPPING_AMPLITUDE / amplitude_max
        return waveform * scale

    def mix_waveform(self, waveform: torch.Tensor, noise_waveform: torch.Tensor, snr: float) -> torch.Tensor:
        signal_power = torch.mean(waveform**2, axis=-1, keepdims=True)
        noise_power = torch.mean(noise_waveform**2, axis=-1, keepdims=True)

        if noise_power == 0:
            logger.warning("Noise power is zero, returning original waveform")
            return waveform

        snr_linear = 10 ** (snr / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
        mixed_waveform = waveform + scaling_factor * noise_waveform
        return self.solve_clipping(mixed_waveform)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        noise_waveform = self.load_noise(waveform)
        snr = np.random.uniform(self.snr_min, self.snr_max)  # noqa: NPY002
        return self.mix_waveform(waveform, noise_waveform, snr)

    def augment_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        mixed_waveforms = [self(wav) for wav in waveforms]
        return torch.stack(mixed_waveforms, dim=0)
