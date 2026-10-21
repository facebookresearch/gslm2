"""Encoder model configuration dispatch."""

from gslm2.encoder.config.common import ModelType
from gslm2.encoder.config.models.base import BaseModelConfig
from gslm2.encoder.config.models.speech import SpidRConfig


def model_type_to_config_type(model_type: ModelType) -> type[BaseModelConfig]:
    if model_type != "spidr":
        raise ValueError(f"Unknown encoder model type: {model_type}")
    return SpidRConfig
