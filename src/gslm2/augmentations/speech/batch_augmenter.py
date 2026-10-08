from typing import ClassVar

import torch

from gslm2.augmentations.speech.base import AudioAugmentation
from gslm2.augmentations.speech.noise_mix import NoiseMixAugmentation
from gslm2.augmentations.speech.pitch_shift import PitchShiftAugmentation
from gslm2.config import AUDIO_SAMPLE_RATE, AugmentConfig


class BatchAugmenter:
    AVAILABLE_AUGMENTATIONS: ClassVar[dict[str, AudioAugmentation]] = {
        "pitch_shift": PitchShiftAugmentation,
        "noise_mix": NoiseMixAugmentation,
    }

    def __init__(self, augmentations: list[AugmentConfig], seed: int = 0) -> None:
        self.seed = seed
        self.augmentations = sorted(
            augmentations, key=lambda a: list(self.AVAILABLE_AUGMENTATIONS.keys()).index(a.name)
        )
        augmentation_names = [aug.name for aug in self.augmentations]
        assert len(augmentation_names) == len(set(augmentation_names)), "Duplicate augmentations found"
        self.load_augmentations()
        self.generator = torch.Generator().manual_seed(self.seed)

    def load_augmentations(self) -> None:
        for aug in self.augmentations:
            if aug.name not in self.AVAILABLE_AUGMENTATIONS:
                raise ValueError(f"Unknown augmentation: {aug.name}")
            setattr(
                self,
                aug.name,
                self.AVAILABLE_AUGMENTATIONS[aug.name](
                    sample_rate=AUDIO_SAMPLE_RATE, seed=self.seed, **aug.augmentation_kwargs
                ),
            )

    def apply_batch_augmentation(self, waveforms: torch.Tensor, aug: AugmentConfig) -> torch.Tensor:
        augment_indices = (
            torch.bernoulli(torch.full((waveforms.shape[0],), aug.prob), generator=self.generator).bool().tolist()
        )
        if sum(augment_indices) == 0:
            return waveforms
        augment_waveforms = waveforms[augment_indices]
        augmentation_cls = getattr(self, aug.name)
        waveforms[augment_indices, ...] = augmentation_cls.augment_batch(augment_waveforms)
        return waveforms

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            waveforms = self.apply_batch_augmentation(waveforms, aug)
        return waveforms
