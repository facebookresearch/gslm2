from abc import ABC, abstractmethod
from typing import Any

import torch


class AudioAugmentation(ABC):
    """
    Base class for audio augmentations.
    Subclasses should implement the `__call__` method to apply the augmentation.
    """

    @abstractmethod
    def __init__(self, sample_rate: int, seed: int = 0, **kwargs: dict[str, Any]) -> None:
        self.sample_rate = sample_rate
        self.seed = seed

    def augment_batch(self, waveforms: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """
        Apply the augmentation to a batch of audio waveforms.
        Args:
            waveforms (torch.Tensor): Input audio signal (B, T).
            **kwargs: Additional keyword arguments for the augmentation.
        """
        raise NotImplementedError("Subclasses should implement this method.")
