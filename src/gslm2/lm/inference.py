# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction, checkpoint resolution, and autoregressive OPT inference."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import yaml
from fairseq2 import init_fairseq2
from fairseq2.error import InvalidOperationError
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.module import to_device
from fairseq2.runtime.dependency import get_dependency_resolver

from gslm2.lm.models.opt import create_opt_model, get_opt_config_preset
from gslm2.lm.models.opt.config import MultiStreamOPTConfig, OPTConfig
from gslm2.lm.models.opt.multi_stream_model import TransformerLMMultiStream
from gslm2.lm.models.opt.sampling import sample_next_token
from gslm2.lm.recipes.train.config import load_unit_lm_train_config
from gslm2.lm.runtime.utils.checkpoint import unify_fsdp_checkpoint

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from fairseq2.models.transformer_lm import TransformerLM

    from gslm2.shared.distributed import DistributedContext

logger = logging.getLogger(__name__)
_PROMPT_NDIM = 2


@dataclass(frozen=True)
class LoadedOptModel:
    """An inference model and its raw vocabulary layout."""

    model: TransformerLM
    unit_offset: int
    vocabulary_sizes: tuple[int, ...]

    @property
    def is_multi_stream(self) -> bool:
        """Return whether the model predicts aligned streams."""
        return isinstance(self.model, TransformerLMMultiStream)


@dataclass(frozen=True)
class DecodingOptions:
    """Nucleus sampling and Depformer annealing settings."""

    temperature: float = 1.0
    top_p: float = 0.9
    temp_anneal: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("decoding temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1 or not math.isfinite(self.temp_anneal) or self.temp_anneal <= 0:
            raise ValueError("invalid nucleus threshold or Depformer annealing")


def build_opt_config(
    model_preset: str,
    vocab_size: int | list[int],
    offset: int,
    *,
    model_config_overrides: dict[str, Any] | None = None,
) -> OPTConfig:
    """Construct the same architecture for benchmark scoring and generation."""
    config = get_opt_config_preset(model_preset)
    if isinstance(config, MultiStreamOPTConfig):
        if not isinstance(vocab_size, list):
            raise TypeError("multi-stream presets require per-stream vocabulary sizes")
        config.vocab_size = [size + offset for size in vocab_size]
    else:
        if not isinstance(vocab_size, int):
            raise TypeError("single-stream presets require one vocabulary size")
        config.vocab_size = vocab_size + offset
    return replace(config, **model_config_overrides) if model_config_overrides else config


def parse_model_overrides(values: object) -> dict[str, Any]:
    """Validate the unstructured overrides stored by fairseq2 recipes."""
    if values is None:
        return {}
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise TypeError("model.config_overrides must be a mapping with string keys")
    return dict(values)


def checkpoint_config_path(checkpoint: Path) -> Path:
    """Locate the nearest saved recipe beside a checkpoint tree."""
    for parent in checkpoint.parents:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("saved recipe not found; supply model_config explicitly")


def load_opt_model(
    checkpoint: Path,
    device: torch.device,
    *,
    model_config: Path | None = None,
    context: DistributedContext | None = None,
    checkpoint_cache: Path | None = None,
) -> LoadedOptModel:
    """Load a recipe and a file or fairseq2 step directory on one rank."""
    try:
        get_dependency_resolver()
    except InvalidOperationError:
        init_fairseq2()
    recipe_path = model_config or checkpoint_config_path(checkpoint)
    with recipe_path.open(encoding="utf-8") as source:
        values = yaml.safe_load(source)
    if not isinstance(values, dict):
        raise TypeError(f"{recipe_path} must contain a YAML mapping")
    if "slm" in values:
        values = values["slm"]["train"]
    recipe = load_unit_lm_train_config(values)
    if recipe.model.vocab_size is None:
        raise ValueError("the saved recipe must specify model.vocab_size")
    overrides: dict[str, Any] = {}
    if isinstance(recipe.model.vocab_size, list):
        overrides = {
            "distill_temperature": recipe.distill_temperature,
            "depformer_generation_order": recipe.depformer_generation_order,
            "multi_pred_loss_weight": recipe.multi_pred_loss_weight,
            "self_distil_loss_weight": recipe.self_distil_loss_weight,
            "gen_loss_weight": recipe.gen_loss_weight,
        }
    overrides.update(parse_model_overrides(recipe.model.config_overrides))
    config = build_opt_config(
        recipe.model_preset,
        recipe.model.vocab_size,
        recipe.dataset.unit_offset,
        model_config_overrides=overrides,
    )
    checkpoint_path = (
        get_single_ckpt_path(checkpoint, context, unified_directory=checkpoint_cache)
        if checkpoint.is_dir()
        else checkpoint
    )
    model = create_opt_model(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    model.load_state_dict(state.get("model", state))
    to_device(model, device)
    sizes = config.vocab_size if isinstance(config.vocab_size, list) else [config.vocab_size]
    return LoadedOptModel(
        model.eval(), recipe.dataset.unit_offset, tuple(size - recipe.dataset.unit_offset for size in sizes)
    )


@torch.inference_mode()
def decode_tokens(
    loaded: LoadedOptModel,
    sequences: torch.Tensor,
    steps: int,
    *,
    options: DecodingOptions | None = None,
) -> torch.Tensor:
    """Append sampled tokens to one single- or multi-stream prompt."""
    if sequences.ndim != _PROMPT_NDIM or sequences.shape[0] != len(loaded.vocabulary_sizes) or not sequences.shape[1]:
        raise ValueError("prompt must have non-empty shape (streams, time)")
    if steps < 0:
        raise ValueError("decoding steps must be non-negative")
    options = options or DecodingOptions()
    for _ in range(steps):
        layout = BatchLayout(
            shape=sequences.shape,
            seq_lens=[sequences.shape[-1]] * sequences.shape[0],
            packed=False,
            device=sequences.device,
        )
        if isinstance(loaded.model, TransformerLMMultiStream):
            decoder_outputs = loaded.model.forward(sequences, layout, return_decoder_outputs=True)
            if not isinstance(decoder_outputs, torch.Tensor):
                raise TypeError("multi-stream model must return decoder states")
            candidates, _ = loaded.model.forward_depformer_inference(
                decoder_outputs, options.temperature, options.top_p, temp_anneal=options.temp_anneal
            )
            next_token = torch.stack(candidates)[:, 0, 0, :]
        else:
            logits = loaded.model(sequences, layout)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("single-stream model must return logits")
            logits[:, -1, : loaded.unit_offset] = float("-inf")
            next_token = sample_next_token(logits[:, -1], options.temperature, options.top_p).unsqueeze(0)
        sequences = torch.cat([sequences, next_token], dim=-1)
    return sequences


def get_single_ckpt_path(  # noqa: C901
    step_dir: Path,
    dist_context: DistributedContext | None = None,
    *,
    unified_directory: Path | None = None,
) -> Path:
    """Resolve the model checkpoint in a step dir.

    DDP writes one model file and is returned as-is; FSDP writes one sdp_* shard per
    rank and is unified first.
    """
    # Find all .pt files recursively
    all_pt_files = list(step_dir.rglob("*.pt"))

    # Filter to only include model-related .pt files:
    # 1. Files directly named "model.pt"
    # 2. Files in paths containing "model", essentially sdp_XX.pt sharded files
    model_pt_files = [
        pt_file
        for pt_file in all_pt_files
        if pt_file.name == "model.pt" or "model" in str(pt_file.relative_to(step_dir)).lower()
    ]

    if len(model_pt_files) == 0:
        raise ValueError(f"No model .pt files found in {step_dir}")

    if len(model_pt_files) == 1:
        return model_pt_files[0]

    # Check if there's exactly one "model.pt" file, and prioritize this one
    model_pt_candidates = [pt_file for pt_file in model_pt_files if pt_file.name == "model.pt"]
    if len(model_pt_candidates) == 1:
        return model_pt_candidates[0]
    if len(model_pt_candidates) > 1:
        raise ValueError(f"Multiple 'model.pt' files found in {step_dir}: {model_pt_candidates}")

    # Check for all sharded checkpoints
    if not all(pt_file.name.startswith("sdp_") for pt_file in model_pt_files):
        raise ValueError(f"Not all .pt files in {step_dir} start with 'sdp_' (sharded params): {model_pt_files}")

    # Sync here so that a single unification is performed, and all processes wait for it to complete
    if dist_context is not None:
        dist_context.barrier()

    # Unify the checkpoint
    unified_ckpt_dir = unified_directory or step_dir / "unified_ckpt"
    if dist_context is None or dist_context.rank == 0:
        logger.info("Unifying FSDP checkpoint in %s to %s ...", step_dir, unified_ckpt_dir)

        # Find the longest parent path ending with "model/"
        sharded_ckpt_dir = model_pt_files[0]
        while sharded_ckpt_dir != sharded_ckpt_dir.parent:
            if sharded_ckpt_dir.name == "model":
                break
            sharded_ckpt_dir = sharded_ckpt_dir.parent

        unify_fsdp_checkpoint(
            sharded_ckpt_dir=sharded_ckpt_dir,
            unified_ckpt_dir=unified_ckpt_dir,
            pp_size=1,
            tp_size=1,
        )

    if dist_context is not None:
        dist_context.barrier()

    # Unified checkpoints use the same layout DDP writes: pp_00/tp_00/sdp_00.pt.
    model_pt_file = unified_ckpt_dir / "pp_00" / "tp_00" / "sdp_00.pt"
    if not model_pt_file.exists():
        raise ValueError(
            f"FSDP-sharded checkpoint in {step_dir} could not be unified. "
            f"Expected unified file at {model_pt_file} but it was not found."
        )

    return model_pt_file
