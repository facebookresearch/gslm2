"""Trainer abstractions: ``BaseTrainer`` + speech leaves."""

from gslm2.training.base import BaseTrainer
from gslm2.training.speech import SpeechSSLTrainer

__all__ = ["BaseTrainer", "SpeechSSLTrainer"]
