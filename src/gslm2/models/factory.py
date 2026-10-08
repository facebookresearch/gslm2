import logging
import typing as tp
from pathlib import Path

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from gslm2.checkpoint import (
    compare_model_configs,
    config_from_checkpoint,
)
from gslm2.config import (
    BaseModelConfig,
    CausalLMConfig,
    DinoSRConfig,
    ModelType,
    MultiStreamCausalLMConfig,
    SpidRConfig,
)
from gslm2.models.core.multi_stream_opt import MultiStreamOPT
from gslm2.models.core.opt import OPT
from gslm2.models.speech.dinosr import DinoSR
from gslm2.models.speech.spidr import SpidR
from gslm2.utils import state_dict_from_dinosr_fairseq_checkpoint

logger = logging.getLogger()

_MODEL_REGISTRY: dict[ModelType, tuple[type[nn.Module], type[BaseModelConfig]]] = {
    # speech encoders
    "dinosr": (DinoSR, DinoSRConfig),
    "spidr": (SpidR, SpidRConfig),
    # language models
    "opt": (OPT, CausalLMConfig),
    "gpt2": (OPT, CausalLMConfig),
    "msopt": (MultiStreamOPT, MultiStreamCausalLMConfig),
}


def build_model(
    *,
    cfg: BaseModelConfig | None = None,
    model_type: ModelType,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    """Build a model from a config, a checkpoint, or both.

    ``model_type`` selects the model/config pair from the registry.
    When *checkpoint* is given the weights (and optionally the config)
    are loaded from disk; otherwise *cfg* must be provided.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}. Registered: {sorted(_MODEL_REGISTRY)}")
    model_cls, config_cls = _MODEL_REGISTRY[model_type]
    if checkpoint is not None:
        return _build_from_checkpoint(model_cls, config_cls, checkpoint, cfg)
    if cfg is None:
        raise ValueError("cfg is required when no checkpoint is provided.")
    return _instantiate_model(model_cls, cfg)


def _instantiate_model(model_cls: type[nn.Module], cfg: BaseModelConfig) -> nn.Module:
    """Instantiate a model from its config."""
    return model_cls(cfg)


def _build_from_checkpoint(
    model_cls: type[nn.Module],
    config_cls: type[BaseModelConfig],
    ckpt: str | Path,
    current_model_cfg: BaseModelConfig | None = None,
) -> nn.Module:
    """Load a model from a checkpoint file (``*.pt`` / ``*.pth``).

    If a ``config.json`` sits next to the checkpoint the config is
    reconstructed from it; otherwise defaults are used.  When
    *current_model_cfg* is provided (training-time resume) its values take
    precedence over the saved config.
    """
    path = Path(ckpt)
    if path.suffix not in (".pt", ".pth"):
        raise ValueError("Only .pt and .pth files are supported.")
    cfg = config_from_checkpoint(config_cls, path)

    if current_model_cfg is None:  # condition=True, when launch_validation or ABX computing.
        instance = _instantiate_model(model_cls, cfg)
    else:
        compare_model_configs(cfg, current_model_cfg)
        instance = _instantiate_model(model_cls, current_model_cfg)

    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    consume_prefix_in_state_dict_if_present(state_dict, "module.")

    missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        logger.warning("Missing keys when loading state_dict %s", missing_keys)
    if len(unexpected_keys) > 0:
        logger.warning("Unexpected keys when loading state_dict %s", unexpected_keys)
    return instance


def load_frozen_component(
    checkpoint: str | Path,
    component: str,
    model_type: ModelType,
) -> nn.Module:
    """Load a frozen sub-component from a checkpoint.

    Builds the full model via :func:`build_model`, extracts the named
    component, freezes it, and returns it.

    Args:
        checkpoint: Path to a checkpoint (``.pt`` / ``.pth``).
        component: Attribute name on the model, e.g. ``"target_encoder"``,
            ``"encoder"``, ``"predictor"``.  For wrapped components
            (:class:`EncoderWrapper` / :class:`PredictorWrapper`), the inner
            ``.backbone`` is returned automatically.
        model_type: Registry key for the full model.

    Returns:
        Frozen :class:`nn.Module` (eval mode, ``requires_grad=False``).
    """
    model = build_model(model_type=model_type, checkpoint=checkpoint)
    part = getattr(model, component)
    # Unwrap EncoderWrapper / PredictorWrapper → bare ViT / ViT-Predictor
    if hasattr(part, "backbone"):
        part = part.backbone
    part.eval()
    for p in part.parameters():
        p.requires_grad = False
    return part


PretrainedModel = tp.Literal["dinosr-original"]


def load_pretrained(name: PretrainedModel, **kwargs: str | bool) -> nn.Module:
    match name:
        case "dinosr-original":
            url = "https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt"
            state_dict = state_dict_from_dinosr_fairseq_checkpoint(load_state_dict_from_url(url, **kwargs))
            model_type, cfg = "dinosr", DinoSRConfig(encoder_qkv_bias=True)
        case _:
            raise ValueError(f"Model '{name}' not recognized. Available models: {tp.get_args(PretrainedModel)}.")

    model = build_model(cfg=cfg, model_type=model_type)
    model.load_state_dict(state_dict)
    model.eval()
    return model
