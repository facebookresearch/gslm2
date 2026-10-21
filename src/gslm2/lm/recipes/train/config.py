# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used by the fairseq2 config structurer
from typing import TYPE_CHECKING, Any, cast

# from typing import override
import torch
from fairseq2.composition import register_model_family
from fairseq2.early_stopper import EarlyStopper
from fairseq2.metrics import format_as_float
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.models.transformer_lm import (
    TransformerLM,
    apply_ac_to_transformer_lm,
    apply_fsdp_to_transformer_lm,
    compile_transformer_lm,
)
from fairseq2.recipe.base import RecipeContext
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    COSINE_ANNEALING_LR,
    AdamWConfig,
    CommonSection,
    CosineAnnealingLRConfig,
    DatasetSection,
    FSDPConfig,
    GangSection,
    LRSchedulerSection,
    MetricRecordersConfig,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TensorBoardConfig,
    TorchConfig,
    TrainerSection,
)
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.utils.config import StandardConfigMerger
from fairseq2.utils.structured import StandardValueConverter
from fairseq2.utils.validation import Validatable, ValidationResult

from gslm2.lm.early_stopping import StandardEarlyStopper
from gslm2.lm.models.opt import (
    OPT_FAMILY,
    OPTConfig,
    convert_opt_state_dict,
    create_opt_model,
    register_opt_configs,
)

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


logger = logging.getLogger(__name__)
MIN_TEXT_VOCAB_SIZE = 3


@dataclass(kw_only=True)
class UnitLMTrainConfig:
    model: ModelSectionWithVocabSize = field(
        default_factory=lambda: ModelSectionWithVocabSize(
            family="opt",
            arch="metasr_unit_opt",
            dtype=torch.bfloat16,
            compile=False,
        )
    )

    dataset: UnitDatasetSection = field(default_factory=lambda: UnitDatasetSection())

    gang: GangSectionExtended = field(default_factory=lambda: GangSectionExtended())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            data_parallelism="ddp",
            fsdp=FSDPConfig(version="v2", fp32_reduce=True),
            max_grad_norm=1.0,
            gc_every_n_steps=1000,
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=1e-2, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.1, differentiable=False, impl="fused"
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR,
            config=CosineAnnealingLRConfig(
                num_warmup_steps=1000,
                cycle_mul=1.0,
                lr_mul=1.0,
                start_lr=1e-30,
                final_lr_scale=0.01,
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=48_000,
            validate_at_start=True,
            validate_every_n_steps=4000,
            score_metric="nll_loss",
            checkpoint_every_n_steps=1000,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=10,
        )
    )

    keep_checkpoints_at_step_nr: list[int] = field(default_factory=list)

    early_stopping: EarlyStoppingSection = field(default_factory=lambda: EarlyStoppingSection())

    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchConfig(default_sdpa="torch"),
            metric_recorders=MetricRecordersConfig(
                # Disable TensorBoard by default, since it's not installed
                tensorboard=TensorBoardConfig(enabled=False),
            ),
            seed=0,
        )
    )

    is_train_deterministic: bool = True

    model_preset: str = "opt125m"

    # Multi-stream model parameters
    multi_pred_loss_weight: float = 0.0
    self_distil_loss_weight: float = 0.0
    gen_loss_weight: float = 0.0
    distill_temperature: float = 1.0
    depformer_generation_order: list[int] | None = None


@dataclass(kw_only=True)
class ModelSectionWithVocabSize(ModelSection):
    vocab_size: int | list[int] | None = None
    """The vocabulary size of the model."""


@dataclass(kw_only=True)
class GangSectionExtended(GangSection):
    do_not_close: bool = True
    """If true, do not close the gangs (and do not destroy the underlying process groups)
    at the end of training."""


@dataclass(kw_only=True)
class UnitDatasetSection(DatasetSection):
    path: Path | str | None = None

    family: str | None = "metasr_unit"

    train_split: str = "train"

    valid_split: str | None = "valid"
    """The split whose NLL is the checkpoint score metric."""

    aux_valid_splits: list[str] = field(default_factory=list)
    """Extra splits to validate on."""

    valid_max_num_batches: int | None = None
    """If set, limits the number of batches evaluated during validation to this number."""

    valid_metrics: ValidMetricsSection = field(default_factory=lambda: ValidMetricsSection())

    unit_offset: int = 2
    """Adds this offset to the unit IDs.
    Useful when the tokenizer has no notion of special tokens
    (e.g. speech tokenizer), but you still need a padding ID, for example."""

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    batch_size: int | None = None
    """If set, overrides the `max_num_tokens` setting to use this static batch size instead."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    strict_data_state: bool = False
    """If True, persist the data pipeline's in-flight buffers in the checkpoint."""

    dynamic_chunking_batch_size: int = 1
    """Number of files to concatenate before chunking.
    Since files are shuffled, this means that statistically, the fixed-size chunks
    will always be different (since they will change at file junctions).
    Defaults to 1, i.e. no dynamic chunking.
    """

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    pack: bool = False
    """If True, use sequence packing instead of length-bucketed padded batches."""


@dataclass(kw_only=True)
class ValidMetricsSection:
    base_dir: Path | None = None
    """The base directory for the metrics."""

    names: list[str] = field(default_factory=list)
    """The names of the metrics to evaluate. If empty, disables this validation step."""

    max_pairs: int | None = None
    """The maximum number of pairs to evaluate. If None, evaluates all pairs."""

    batch_size: int = 16
    """The batch size for validation."""

    batch_sizes: dict[str, int] | None = None
    """Per-metric batch size overrides. Required for on-the-fly configs that use this field;
    without it, runs fail at startup with 'extra keys batch_sizes' config error."""


@dataclass(kw_only=True)
class EarlyStoppingSection(Validatable):
    enabled: bool = False
    """Whether to enable early stopping."""

    every_n_validations: int = 1
    """Evaluate for early stopping every N validation steps."""

    patience: int = 10
    """Number of evaluation periods to wait without improvement before stopping."""

    def validate(self) -> ValidationResult:
        res = ValidationResult()
        if not self.enabled:
            return res

        if self.every_n_validations <= 0:
            res.add_error("every_n_validations must be greater than 0.")

        if self.patience <= 0:
            res.add_error("patience must be greater than 0.")

        return res


def _load_unit_lm_train_config(config: object) -> UnitLMTrainConfig:
    converter = StandardValueConverter()
    config_merger = StandardConfigMerger()
    unstructured_default = converter.unstructure(UnitLMTrainConfig())
    unstructured_merged = config_merger.merge(unstructured_default, config)
    config_structured = converter.structure(unstructured_merged, UnitLMTrainConfig)
    return cast("UnitLMTrainConfig", config_structured)


def load_unit_lm_train_config(config: object) -> UnitLMTrainConfig:  # noqa: C901
    """
    If loading the config first fails,
    try to convert an old-style unstructured config to the new-style one.
    This is a temporary function to help with migration while we
    update all config .yaml files.
    """
    warnings.warn(
        "This config loading function is deprecated, the old config format is still supported for now, but please make sure to migrate your .yaml to the new format.",
        DeprecationWarning,
        stacklevel=2,
    )
    # First try the up-to-date loader
    try:
        return _load_unit_lm_train_config(config)
    except Exception as e:  # noqa: BLE001
        # these are fs2 structure errors
        warnings.warn(
            f"Failed to load config with new format: {e}. Trying to load with old format.",
            DeprecationWarning,
            stacklevel=2,
        )
    # Second, try to convert the keys of the old format
    key_rewrite_map = {
        "trainer.dtype": None,  # removed, bfloat16 is used by default
        "trainer.gradient_accumulation": "trainer.grad_accumulation.num_batches",
        "trainer.max_gradient_norm": "trainer.max_grad_norm",
        "trainer.gradient_check": "trainer.grad_check",
        "common.metric_recorders.wandb.resume": "common.metric_recorders.wandb.resume_mode",
        "keep_checkpoints_at_step_nr": None,  # not supported anymore
    }

    def get_from_path(d: dict, path: str) -> tuple[dict, str]:
        keys = path.split(".")
        for k in keys[:-1]:
            d = d.get(k, {})
        return d, keys[-1]

    def set_in_path(d: dict, path: str, value: object) -> None:
        keys = path.split(".")
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def del_in_path(d: dict, path: str) -> None:
        keys = path.split(".")
        for k in keys[:-1]:
            d = d.get(k, {})
        d.pop(keys[-1], None)

    # Make a deep copy to avoid mutating the original config
    config = cast("dict", copy.deepcopy(config))

    for old_key, new_key in key_rewrite_map.items():
        parent, leaf = get_from_path(config, old_key)
        if leaf in parent:
            value = parent[leaf]
            if new_key is None:
                del_in_path(config, old_key)
            else:
                set_in_path(config, new_key, value)
                del_in_path(config, old_key)

    return _load_unit_lm_train_config(config)


def register_unit_lm_metrics(container: DependencyContainer) -> None:
    def register(name: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        descriptor = MetricDescriptor(name, *args, **kwargs)
        container.collection.register_instance(MetricDescriptor, descriptor)

    # We don't need to register all the metrics here,
    # only the ones that could be used as a criterion in early stopping.
    # Non-registered metrics computed in validation will still be logged in to WandB.
    metrics = ["tSC", "syntactic_test", "lexical_test", "sSC"]

    for metric in metrics:
        register(f"{metric}_accuracy", f"{metric} Accuracy", 0, format_as_float, higher_better=True)
        register(
            f"{metric}_accuracy_per_token", f"{metric} Accuracy per Token", 0, format_as_float, higher_better=True
        )


def register_opt(container: DependencyContainer) -> None:
    # Register the family first
    register_model_family(
        container,
        OPT_FAMILY,
        kls=TransformerLM,
        config_kls=OPTConfig,
        factory=create_opt_model,
        state_dict_converter=convert_opt_state_dict,
        compiler=compile_transformer_lm,
        fsdp_applier=apply_fsdp_to_transformer_lm,
        layerwise_ac_applier=apply_ac_to_transformer_lm,
    )

    # Register base configs
    register_opt_configs(container)

    # Register Metasr-specific OPT architectures
    # 1. the unit LM, with the provided vocab size
    # 2. the text LM, with vocab size 50272 (original OPT tokenizer)
    arch = ConfigRegistrar(container, OPTConfig)
    text_vocab_size = 50272

    @arch("metasr_unit_opt", advanced=True)
    def unit_opt(resolver: DependencyResolver) -> OPTConfig:
        context = RecipeContext(resolver)
        recipe_config = context.config.as_(UnitLMTrainConfig)
        config = get_config(resolver, OPTConfig, recipe_config.model_preset)
        # Check for consistency here:
        # The vocab size for text is always expected to be the same
        vocab_size_maybe = recipe_config.model.vocab_size
        offset = recipe_config.dataset.unit_offset
        is_text_lm = recipe_config.model.arch is not None and "_text" in recipe_config.model.arch
        if is_text_lm and vocab_size_maybe is not None and vocab_size_maybe != text_vocab_size:
            raise ValueError(
                f"Text LM is used (arch: {recipe_config.model.arch}), but vocab_size is set to {vocab_size_maybe}, different from the expected tokenizer vocab size: {text_vocab_size}."
            )
        vocab_size = vocab_size_maybe or 256  # 256 by default
        if not is_text_lm and vocab_size_maybe is None and offset == 0:
            logger.warning("Using unit LM with no offset and default vocab size of 256. Make sure this is intended.")
        vocab_size += offset  # Adjust for special tokens
        config.vocab_size = vocab_size
        return config

    @arch("metasr_unit_multistream_opt", advanced=True)
    def unit_multistream(resolver: DependencyResolver) -> OPTConfig:
        """Multi-stream variant of ``metasr_unit_opt``.

        Expects ``model.vocab_size`` to be a list of per-stream vocabulary sizes,
        and propagates the multi-stream loss weights from the recipe config to
        the OPT config.
        """
        context = RecipeContext(resolver)
        recipe_config = context.config.as_(UnitLMTrainConfig)
        config = get_config(resolver, OPTConfig, recipe_config.model_preset)

        vocab_size = recipe_config.model.vocab_size
        if not isinstance(vocab_size, list):
            raise TypeError("Multiple vocab sizes are required for multistream models.")
        offset = recipe_config.dataset.unit_offset
        config.vocab_size = [
            v + offset for v in vocab_size
        ]  # Adjust for special tokens and build a new list rather than mutating

        # Training-related
        config.multi_pred_loss_weight = recipe_config.multi_pred_loss_weight
        config.self_distil_loss_weight = recipe_config.self_distil_loss_weight
        config.gen_loss_weight = recipe_config.gen_loss_weight
        config.distill_temperature = recipe_config.distill_temperature
        config.depformer_generation_order = recipe_config.depformer_generation_order

        return config

    @arch("metasr_text_opt", advanced=True)
    def text_opt(resolver: DependencyResolver) -> OPTConfig:
        context = RecipeContext(resolver)
        recipe_config = context.config.as_(UnitLMTrainConfig)
        config = get_config(resolver, OPTConfig, recipe_config.model_preset)
        vocab_size = recipe_config.model.vocab_size
        if vocab_size is not None and (not isinstance(vocab_size, int) or vocab_size < MIN_TEXT_VOCAB_SIZE):
            raise ValueError("Text LM vocab_size must be an integer larger than the reserved token IDs.")
        config.vocab_size = vocab_size if vocab_size is not None else text_vocab_size
        return config


def register_early_stopper(container: DependencyContainer) -> None:
    def create_early_stopper(resolver: DependencyResolver) -> EarlyStopper:
        # Grab the recipe config
        context = RecipeContext(resolver)
        config = context.config.as_(UnitLMTrainConfig)

        # Find the score metric descriptor and find out if lower is better
        # "score_metric" is configured in the RegimeSection
        lower_is_better = True  # default for loss functions
        metric_descriptor = resolver.resolve_optional(MetricDescriptor)
        if metric_descriptor is None and config.early_stopping.enabled:
            raise ValueError("No MetricDescriptor found for early stopping, none is registered.")
        if metric_descriptor is not None:
            if config.early_stopping.enabled:
                logger.info("Using metric '%s' for early stopping.", metric_descriptor.name)
            lower_is_better = not metric_descriptor.higher_better

        # every_n_steps == 0 disables early stopping
        every_n_steps = config.early_stopping.every_n_validations if config.early_stopping.enabled else 0
        return StandardEarlyStopper(
            patience=config.early_stopping.patience,
            every_n_steps=every_n_steps,
            lower_is_better=not lower_is_better,
            # we also pass the disabled flag explicitly, for clarity
            disabled=not config.early_stopping.enabled,
        )

    container.register(EarlyStopper, create_early_stopper)
