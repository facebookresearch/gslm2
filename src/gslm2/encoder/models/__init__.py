"""SpidR speech encoder and checkpoint loading."""

from gslm2.encoder.models.factory import build_model
from gslm2.encoder.models.spidr import SpidR

__all__ = ["SpidR", "build_model"]
