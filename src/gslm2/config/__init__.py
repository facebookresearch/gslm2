"""Configuration dataclasses."""

from gslm2.config.common import (
    AUDIO_SAMPLE_RATE,
    MaskingConfig,
    MaskingStrategy,
    ModelType,
    OptimizerConfig,
    RunConfig,
    SlurmConfig,
)
from gslm2.config.config import BaseConfig, LMConfig, ResumeConfig, SpeechConfig, load_config
from gslm2.config.data import (
    AugmentConfig,
    DataConfig,
    OnTheFlyUnitSource,
    PreExtractedUnitSource,
    SLMUnitDataConfig,
    SpeechUnitConfig,
    TextDataConfig,
)
from gslm2.config.eval import ABXSubset, ABXTrainConfig, EvalSpec, build_eval_spec
from gslm2.config.models import LM_MODEL_TYPES, model_type_to_config_type
from gslm2.config.models.base import BaseModelConfig
from gslm2.config.models.core import CausalLMConfig, MultiStreamCausalLMConfig
from gslm2.config.models.speech import RegularisationConfig, SpidRConfig

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "LM_MODEL_TYPES",
    "ABXSubset",
    "ABXTrainConfig",
    "AugmentConfig",
    "BaseConfig",
    "BaseModelConfig",
    "CausalLMConfig",
    "DataConfig",
    "EvalSpec",
    "LMConfig",
    "MaskingConfig",
    "MaskingStrategy",
    "ModelType",
    "MultiStreamCausalLMConfig",
    "OnTheFlyUnitSource",
    "OptimizerConfig",
    "PreExtractedUnitSource",
    "RegularisationConfig",
    "ResumeConfig",
    "RunConfig",
    "SLMUnitDataConfig",
    "SlurmConfig",
    "SpeechConfig",
    "SpeechUnitConfig",
    "SpidRConfig",
    "TextDataConfig",
    "build_eval_spec",
    "load_config",
    "model_type_to_config_type",
]
