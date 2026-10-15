# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import TYPE_CHECKING, final, override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch, SyncMode
from fairseq2.early_stopper import EarlyStopper
from fairseq2.gang import Gangs
from fairseq2.logging import get_log_writer
from fairseq2.metrics.aggregation import Mean
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
)
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.evaluator import EvalUnit
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.utils.device_stat import DeviceStatTracker, _NoopDeviceStatTracker
from fairseq2.utils.rng import use_deterministic

from gslm2.fs2.unit_lm.eval import SLMEvalMetrics, eval_lm
from gslm2.fs2.unit_lm.recipes.train.config import (
    UnitLMTrainConfig,
    ValidMetricsSection,
    register_early_stopper,
    register_opt,
    register_unit_lm_metrics,
)
from gslm2.fs2.unit_lm.recipes.train.dataset import (
    DataReadOptions,
    LengthBatching,
    LMDataset,
    LMDatasetConfig,
    StaticBatching,
    single_step_dummy_reader,
)
from gslm2.fs2.unit_lm.recipes.train.dataset_speech import (
    UnitDataset,
    open_speech_lm_dataset,
)
from gslm2.fs2.unit_lm.recipes.train.dataset_speech_manifest import (
    ManifestUnitDataset,
    ManifestUnitDatasetConfig,
    open_manifest_unit_dataset,
)
from gslm2.fs2.unit_lm.recipes.train.dataset_text import (
    TextUnitDataset,
    TextUnitDatasetConfig,
    open_text_lm_dataset,
)
from gslm2.fs2.unit_lm.recipes.train.lr_schedulers import register_tri_stage_cosine_lr
from gslm2.fs2.utils.gang import disable_destroy_on_close

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from fairseq2.gang import Gangs
    from fairseq2.metrics import MetricBag
    from fairseq2.recipe.model import RecipeModel
    from fairseq2.runtime.dependency import DependencyContainer
    from torch import Tensor

# Configure logging
logger = get_log_writer(__name__)


@final
class UnitLMTrainRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        # Datasets
        register_dataset_family(
            container,
            "metasr_unit",
            UnitDataset,
            LMDatasetConfig,
            opener=open_speech_lm_dataset,
        )
        register_dataset_family(
            container,
            "metasr_text",
            TextUnitDataset,
            TextUnitDatasetConfig,
            opener=open_text_lm_dataset,
        )
        register_dataset_family(
            container,
            "metasr_manifest_unit",
            ManifestUnitDataset,
            ManifestUnitDatasetConfig,
            opener=open_manifest_unit_dataset,
        )

        # Metrics
        register_unit_lm_metrics(container)

        # Register the OPT 125m family and architectures,
        # e.g. with the desired vocab size
        register_opt(container)

        # Makes lr_scheduler.name: tri_stage_cosine resolvable.
        register_tri_stage_cosine_lr(container)

        # Add a no-op device stat tracker for CPU, for debugging
        try:
            container.register_type(DeviceStatTracker, _NoopDeviceStatTracker, key="cpu")
        except Exception as e:  # noqa: BLE001
            logger.warning(e)

        # Register early stopper
        register_early_stopper(container)

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config.as_(UnitLMTrainConfig)

        if config.is_train_deterministic:
            logger.info("Setting deterministic training parameters")
            # Needed, see https://docs.nvidia.com/cuda/cublas/#results-reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            use_deterministic(value=True)
        else:
            logger.info(
                "SLM training is not deterministic. If needed, set slm.train.is_train_deterministic: True in the config file"
            )

        seed = config.common.seed

        seed += 1

        unit = LMTrainUnit(context.model)

        dataset = context.default_dataset.as_(LMDataset)

        if config.dataset.batch_size is not None:
            batching = StaticBatching(config.dataset.batch_size)
        else:
            batching = LengthBatching(config.dataset.max_num_tokens)

        pack = config.dataset.pack

        read_options = DataReadOptions(
            batching=batching,
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            prefetch=config.dataset.num_prefetch,
            seed=seed,
            pack=pack,
            strict_state=config.dataset.strict_data_state,
        )

        data_reader = dataset.create_reader(
            config.dataset.train_split,
            context.gangs,
            min_seq_len=config.dataset.min_seq_len,
            max_seq_len=config.dataset.max_seq_len,
            options=read_options,
            dynamic_chunking_batch_size=config.dataset.dynamic_chunking_batch_size,
            num_io_streams=1 if isinstance(config.model.vocab_size, int) else len(config.model.vocab_size),
        )

        seed += 1

        # Initialize the validation units.
        valid_units = []
        valid_data_readers = []

        if config.dataset.valid_split:
            valid_unit = LMEvalUnit(context.model)

            read_options = DataReadOptions(
                batching=batching,
                sync_mode=SyncMode.UNTIL_LAST,
                max_num_batches=config.dataset.valid_max_num_batches,
                prefetch=config.dataset.num_prefetch,
                seed=seed,
                pack=pack,
            )

            valid_data_reader = dataset.create_reader(
                config.dataset.valid_split,
                context.gangs,
                config.dataset.min_seq_len,
                config.dataset.max_seq_len,
                read_options,
                num_io_streams=1 if isinstance(config.model.vocab_size, int) else len(config.model.vocab_size),
            )

            valid_units.append(valid_unit)
            valid_data_readers.append(valid_data_reader)

        seed += 1

        for aux_split in config.dataset.aux_valid_splits:
            aux_read_options = DataReadOptions(
                batching=batching,
                sync_mode=SyncMode.UNTIL_LAST,
                max_num_batches=config.dataset.valid_max_num_batches,
                prefetch=config.dataset.num_prefetch,
                seed=seed,
                pack=pack,
            )

            aux_data_reader = dataset.create_reader(
                aux_split,
                context.gangs,
                config.dataset.min_seq_len,
                config.dataset.max_seq_len,
                aux_read_options,
                num_io_streams=1 if isinstance(config.model.vocab_size, int) else len(config.model.vocab_size),
            )

            valid_units.append(LMEvalUnit(context.model, name=aux_split))
            valid_data_readers.append(aux_data_reader)

            seed += 1

        if config.dataset.valid_metrics and config.dataset.valid_metrics.names:
            if (
                config.dataset.family is not None
                and "text" in config.dataset.family
                and config.dataset.unit_offset > 0
            ):
                logger.warning(
                    "Using non-zero unit_offset=%d for dataset family %s. Usually text should already be correctly tokenized, so this is suspicious.",
                    config.dataset.unit_offset,
                    config.dataset.family,
                )

            end_to_end_eval_unit = EndToEndEvalUnit(
                context.model,
                config.dataset.valid_metrics,
                context.gangs,
                seed=2,  # Constant seed here makes it so validation curves are always comparable
                unit_offset=config.dataset.unit_offset,
            )

            end_to_end_dummy_data_reader = single_step_dummy_reader(context.gangs)

            valid_units.append(end_to_end_eval_unit)
            valid_data_readers.append(end_to_end_dummy_data_reader)

        trainer = context.create_trainer(
            unit,
            data_reader,
            valid_units=valid_units,
            valid_data_readers=valid_data_readers,
        )

        # There is no auto-wiring of EarlyStopper in the Trainer yet,
        # so we plug it in manually
        early_stopper = context.resolver.resolve_optional(EarlyStopper)
        trainer._early_stopper = early_stopper  # noqa: SLF001

        # Optionally override the closing behavior of gangs
        # to avoid destroying the underlying process groups, if needed later.
        if config.gang.do_not_close:
            disable_destroy_on_close(context.gangs)
            logger.info("Disabled destroy-on-close for gangs as per configuration.")

        return trainer

    @property
    @override
    def config_kls(self) -> type[object]:
        return UnitLMTrainConfig


class _LMBaseUnit:
    def __init__(self, model: RecipeModel) -> None:
        self._model = model

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_nll_loss_metric(metric_bag)
        add_seq_batch_metrics(metric_bag)
        metric_bag.add("stream_specific_loss", Mean())
        metric_bag.add("gen_loss", Mean())

    def _process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> tuple[Tensor, None]:
        input_batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = input_batch.as_input()

        # Multi-stream packing stashes per-document frame lengths in `example` so the
        # model can rebuild a packed (block-diagonal) attention layout.
        extra: dict[str, object] = {}
        example = input_batch.example
        if isinstance(example, dict) and example.get("doc_frame_lens") is not None:
            extra["doc_seq_lens"] = example["doc_frame_lens"]

        # target_mask is set for packed batches.
        nll_loss = self._model.module(
            seqs,
            seqs_layout,
            targets=target_batch.seqs,
            target_mask=target_batch.target_mask,
            reduction="mean",
            **extra,
        )

        update_nll_loss_metric(metric_bag, nll_loss)
        update_seq_batch_metrics(metric_bag, batch)

        model = self._model.base_module
        stream_specific_loss = getattr(model, "last_stream_specific_loss", None)
        gen_loss = getattr(model, "last_gen_loss", None)
        if stream_specific_loss is not None and gen_loss is not None:
            metric_bag.get("stream_specific_loss", Mean).update(stream_specific_loss)
            metric_bag.get("gen_loss", Mean).update(gen_loss)

        return nll_loss, None

    @property
    def model(self) -> RecipeModel:
        return self._model


@final
class LMTrainUnit(_LMBaseUnit, TrainUnit[SequenceBatch]):
    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> tuple[Tensor, int | None]:
        return self._process_batch(batch, metric_bag)


@final
class LMEvalUnit(_LMBaseUnit, EvalUnit[SequenceBatch]):
    def __init__(self, model: RecipeModel, name: str | None = None) -> None:
        super().__init__(model)
        self._name = name

    @property
    @override
    def name(self) -> str | None:
        return self._name

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        self._process_batch(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        # fairseq2 sums `score_metric` across units, so named splits republish NLL
        # under a distinct key to leave checkpoint scoring to the primary split.
        if self._name is not None and "nll_loss" in values:
            values[f"{self._name}_nll_loss"] = values.pop("nll_loss")


class EndToEndEvalUnit(EvalUnit[SequenceBatch]):
    def __init__(
        self,
        model: RecipeModel,
        valid_metrics_section: ValidMetricsSection,
        gangs: Gangs,
        seed: int,
        unit_offset: int,
    ) -> None:
        self._model = model
        self._batch_size = valid_metrics_section.batch_size
        assert valid_metrics_section.base_dir is not None, "Valid metrics base directory must be set."
        metric_paths = [
            valid_metrics_section.base_dir / "test" / "pairs" / f"{metric_name}.jsonl"
            for metric_name in valid_metrics_section.names
        ]
        self._metric_paths = metric_paths
        self._max_pairs = valid_metrics_section.max_pairs
        self._metrics = SLMEvalMetrics(metric_paths, gangs.dp)
        self._gang = gangs.dp
        self._seed = seed
        self._unit_offset = unit_offset

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        SLMEvalMetrics.prepare_metric_bag(self._metric_paths, metric_bag)

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        # this is slightly hacky, this can be redesigned:
        # we should get rid of the SLMEvalMetrics class and update
        # the metric bag with functions instead of methods.
        self._metrics.bag = metric_bag
        # to call only once
        eval_lm(
            self._model.base_module,
            self._metrics,
            self._gang,
            max_pairs=self._max_pairs,
            batch_size=self._batch_size,
            dataloader_num_workers=0,
            seed=self._seed,
            offset=self._unit_offset,
        )

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model
