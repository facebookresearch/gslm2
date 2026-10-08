import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from librosa.effects import pitch_shift

from gslm2.augmentations.speech.base import AudioAugmentation

logger = logging.getLogger()

type Audio = npt.NDArray[np.float32]


class PitchShiftAugmentation(AudioAugmentation):
    def __init__(self, sample_rate: int, min_steps: int, max_steps: int, seed: int = 0) -> None:
        logger.info("Initializing pitch-shift augmentation...")
        super().__init__(sample_rate, seed)
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, waveform: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        n_steps = self.rng.uniform(self.min_steps, self.max_steps)
        wav_out = pitch_shift(waveform.numpy(), sr=self.sample_rate, n_steps=n_steps, **kwargs)
        return torch.tensor(wav_out).to(dtype=waveform.dtype)

    def augment_batch(self, waveforms: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        dtype = waveforms.dtype
        pitch_shifted_waveforms = [self(wav, **kwargs) for wav in waveforms]
        return torch.stack(pitch_shifted_waveforms, dim=0).to(dtype=dtype)
