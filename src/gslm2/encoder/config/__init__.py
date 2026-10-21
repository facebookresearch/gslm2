"""Native encoder configuration dataclasses."""

from gslm2.encoder.config.common import (
    AUDIO_SAMPLE_RATE,
    MaskingConfig,
    MaskingStrategy,
    ModelType,
    OptimizerConfig,
    RunConfig,
    SlurmConfig,
)
from gslm2.encoder.config.config import BaseConfig, ResumeConfig, SpeechConfig, load_config
from gslm2.encoder.config.data import DataConfig
from gslm2.encoder.config.eval import ABXSubset, ABXTrainConfig, EvalSpec, build_eval_spec
from gslm2.encoder.config.models import model_type_to_config_type
from gslm2.encoder.config.models.base import BaseModelConfig
from gslm2.encoder.config.models.speech import RegularisationConfig, SpidRConfig

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "ABXSubset",
    "ABXTrainConfig",
    "BaseConfig",
    "BaseModelConfig",
    "DataConfig",
    "EvalSpec",
    "MaskingConfig",
    "MaskingStrategy",
    "ModelType",
    "OptimizerConfig",
    "RegularisationConfig",
    "ResumeConfig",
    "RunConfig",
    "SlurmConfig",
    "SpeechConfig",
    "SpidRConfig",
    "build_eval_spec",
    "load_config",
    "model_type_to_config_type",
]
