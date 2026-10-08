"""Model-type → config-class dispatch."""

from gslm2.config.common import ModelType
from gslm2.config.models.base import BaseModelConfig
from gslm2.config.models.core import CausalLMConfig, MultiStreamCausalLMConfig
from gslm2.config.models.speech import DinoSRConfig, SpidRConfig


def model_type_to_config_type(model_type: ModelType) -> type[BaseModelConfig]:
    match model_type:
        # speech encoders
        case "dinosr":
            return DinoSRConfig
        case "spidr":
            return SpidRConfig
        # language models
        case "opt" | "gpt2":
            return CausalLMConfig
        case "msopt":
            return MultiStreamCausalLMConfig
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


LM_MODEL_TYPES: set[ModelType] = {"opt", "gpt2", "msopt"}
