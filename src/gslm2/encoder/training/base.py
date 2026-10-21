"""Native encoder training lifecycle."""

import contextlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
import wandb
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from gslm2.encoder.checkpoint import Checkpointer
from gslm2.encoder.config import BaseConfig
from gslm2.encoder.models import build_model
from gslm2.encoder.optimizer import build_optimizer as _build_optimizer
from gslm2.encoder.training.hooks import PostStepHook, PreStepHook
from gslm2.encoder.training.validation import ValidationLauncher
from gslm2.encoder.training.wandb_setup import init_wandb
from gslm2.shared.environment import setup_training
from gslm2.shared.tools import Records
from gslm2.shared.tools.logger import log_info_box, log_model_params

logger = logging.getLogger()

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_AVERAGED_KEYS: list[str] = [
    "loss",
    "grad_norm",
    "data_time",
    "batch_time",
    "batch_size",
]


class BaseTrainer(ABC):
    """Native encoder training loop.

    Leaf overrides: ``build_dataloader`` (required), ``training_step``
    (required), ``build_optimizer``, ``iterate_batches``,
    ``after_checkpoint_load``, ``default_compile``, ``_extra_checkpoint_kwargs``,
    ``_extra_log_fields``. Composition slots passed via ``__init__``:
    ``pre_step_hooks``, ``post_step_hooks``, ``records_extra_keys``,
    ``pbar_keys``, ``compile_strategy``, ``num_meta_workers``.
    """

    def __init__(  # noqa: PLR0913
        self,
        cfg: BaseConfig,
        *,
        validation_launcher_cls: type[ValidationLauncher] = ValidationLauncher,
        post_step_hooks: list[PostStepHook] | None = None,
        pre_step_hooks: list[PreStepHook] | None = None,
        records_extra_keys: list[str] | None = None,
        pbar_keys: list[str] | None = None,
        compile_strategy: Callable[[nn.Module], nn.Module] | None = None,
        num_meta_workers: int = 1,
        find_unused_parameters: bool = False,
    ) -> None:
        self.cfg = cfg
        self._validation_launcher_cls = validation_launcher_cls
        self.post_step_hooks: list[PostStepHook] = list(post_step_hooks or [])
        self.pre_step_hooks: list[PreStepHook] = list(pre_step_hooks or [])
        self.averaged_record_keys = list(set(DEFAULT_AVERAGED_KEYS + (records_extra_keys or [])))
        self.pbar_keys = pbar_keys or ["loss"]  # subset shown on tqdm progress bar
        self._compile_strategy = compile_strategy or self.default_compile
        self._num_meta_workers = num_meta_workers
        self._find_unused_parameters = find_unused_parameters

        # Populated by setup
        self.is_main: bool = False
        self.global_rank: int = 0
        self.world_size: int = 1
        self.device: torch.device
        self.dtype: torch.dtype
        self.model: nn.Module
        self.ddp_model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler
        self.loader: Iterable
        self.ckpt: Checkpointer
        self.validation_launcher: ValidationLauncher
        self.records: Records
        self.step: int = 0
        self.epoch: int = 0
        self._data_tick: float = 0.0
        self._batch_tick: float = 0.0
        self._pbar: tqdm | None = None
        self._last_validated_step: int = -1

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Run the full training lifecycle."""
        self._setup_environment()
        self._setup_distributed()
        if self.is_main:
            init_wandb(self.cfg, self.cfg.eval_group_prefixes())
        self.cfg.dump()
        self._build_components()
        self._build_checkpointer()
        self.validation_launcher = self._validation_launcher_cls(self.cfg, self.ckpt.metrics)

        dist.barrier(device_ids=[self.device.index] if self.device.type == "cuda" else None)
        self._initial_save_and_validate()
        self._wrap_distributed()

        self.records = Records(self.averaged_record_keys)
        self._pbar = tqdm(
            total=self.cfg.optimizer.max_steps,
            initial=self.step,
            disable=not self.is_main,
            desc="Training",
            miniters=10,  # only updating pbar every 10 iters to minimize dump
        )
        logger.info("Starting training loop")
        self._run_training_loop()
        self._finalize()

    # ------------------------------------------------------------------
    # Setup stages
    # ------------------------------------------------------------------
    def _setup_environment(self) -> None:
        setup_training(
            self.cfg.run.random_seed,
            use_deterministic=self.cfg.run.use_deterministic,
            num_workers=self._num_meta_workers,
        )

    def _setup_distributed(self) -> None:
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = self.global_rank == 0
        self.device = (
            torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info("Using device %s", self.device)
        self.dtype = _DTYPE_MAP[self.cfg.optimizer.dtype]

    def _build_components(self) -> None:
        logger.info("Building model, optimizer, and dataloaders")
        self.model = (
            build_model(
                cfg=self.cfg.model,
                model_type=self.cfg.run.model_type,
                checkpoint=self.cfg.run.init_ckpt,
            )
            .to(self.device)
            .train()
        )
        if self.is_main:
            logger.info("Model architecture:\n%s", self.model)
            log_model_params(self.model)
        self.optimizer, self.scheduler = self.build_optimizer(self.model, self.cfg.optimizer)
        self.loader = self.build_dataloader(self.cfg)
        data_cfg = self.cfg.data if not isinstance(self.cfg.data, dict) else next(iter(self.cfg.data.values()))
        self.model.enable_packing = getattr(data_cfg, "enable_packing", False)
        if self.is_main:
            self._log_dataset_summary()

    def _log_dataset_summary(self) -> None:
        """Log dataset name, size, and manifests. Leaf trainers may override."""
        dataset = getattr(self.loader, "dataset", None)
        if dataset is None:
            return
        fields: dict[str, object] = {"Dataset": type(dataset).__name__}
        with contextlib.suppress(TypeError):
            fields["Total samples"] = f"{len(dataset):,}"
        manifests = getattr(getattr(self.cfg, "data", None), "manifests", None)
        sizes = getattr(dataset, "dataset_sizes", None)
        if manifests:
            for i, manifest_path in enumerate(manifests):
                if sizes is not None and i < len(sizes):
                    fields[f"Manifest [{i}]"] = f"{manifest_path}  ({sizes[i]:,} samples)"
                else:
                    fields[f"Manifest [{i}]"] = str(manifest_path)
        log_info_box("Dataset", fields)

    def _build_checkpointer(self) -> None:
        self.ckpt = Checkpointer(
            self.cfg.run.dir,
            self.cfg.run.save_interval,
            self.cfg.run.keep_latest,
            keep_best_n=self.cfg.run.keep_best_n,
            keep_best_metric=self.cfg.run.keep_best_metric,
            keep_best_mode=self.cfg.run.keep_best_mode,
        )
        self.ckpt.init_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.ckpt.load_existing_run()
        self.step = int(self.ckpt.step)
        self.epoch = int(self.ckpt.epoch)
        self.model.current_step.fill_(self.step)
        self.after_checkpoint_load()

    def _initial_save_and_validate(self) -> None:
        # step > 0 iff a prior checkpoint was loaded
        if self.step > 0 or not self.is_main:
            return
        if self.ckpt.save(self.step, self.epoch, **self._extra_checkpoint_kwargs()):
            self.validation_launcher.maybe_launch(self.step, self.ckpt.last)
            self._last_validated_step = self.step

    def _wrap_distributed(self) -> None:
        if self.cfg.run.compile and torch.cuda.is_available():
            self.model = self._compile_strategy(self.model)
        ddp_kwargs: dict[str, Any] = {"device_ids": [self.device.index]} if self.device.type == "cuda" else {}
        self.ddp_model = DistributedDataParallel(
            self.model,
            find_unused_parameters=self._find_unused_parameters,
            **ddp_kwargs,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _run_training_loop(self) -> None:
        self._batch_tick = time.perf_counter()
        self._data_tick = self._batch_tick
        self._log_training_start_banner()
        for step_input in self.iterate_batches():
            if self.step >= self.cfg.optimizer.max_steps:
                break
            for hook in self.pre_step_hooks:
                hook(self, self.step, step_input)
            metrics = self._train_step(step_input, self.step)
            self._accumulate_records(metrics)
            self._pbar.update()
            if self.is_main and self.step % self.cfg.run.log_interval == 0:
                self._maybe_log(metrics)
            if self.is_main and self.step % self.cfg.run.metrics_sync_interval == 0:
                for val_metric in self.ckpt.find_new_metrics():
                    wandb.log(val_metric)
            if self.is_main:
                saved = self.ckpt.save(self.step, self.epoch, **self._extra_checkpoint_kwargs())
                if self.cfg.run.save_latest:
                    self.ckpt.save_latest(self.step, self.epoch, **self._extra_checkpoint_kwargs())
                if saved and self.step != self._last_validated_step:  # avoid double-eval on final step
                    self.validation_launcher.maybe_launch(self.step, self.ckpt.last)
                    self._last_validated_step = self.step
            self._batch_tick = time.perf_counter()
            self._data_tick = self._batch_tick

    def _log_training_start_banner(self) -> None:
        """Logging a box with some job meta-information"""
        fields: dict[str, object] = {
            "Resuming from step": self.step,
            "Epoch": self.epoch,
            "Max steps": self.cfg.optimizer.max_steps,
            "Num GPUs": self.world_size,
            "Precision": self.cfg.optimizer.dtype,
        }
        fields.update(self._extra_banner_fields())
        log_info_box("Training Start", fields)

    def _extra_banner_fields(self) -> dict[str, object]:
        """Leaf-specific banner fields (e.g. per-GPU batch size). Default empty."""
        return {}

    def _train_step(self, step_input: Any, step: int) -> dict[str, Any]:  # noqa: ANN401
        # ``training_step`` owns its autocast region.
        loss, metrics = self.training_step(step_input, step)
        loss.backward()
        grad_norm = clip_grad_norm_(self.ddp_model.parameters(), self.cfg.optimizer.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        self.step += 1
        for hook in self.post_step_hooks:
            hook(self, self.step, step_input, metrics)
        return {**metrics, "loss": loss.detach(), "grad_norm": grad_norm, "lr": lr}

    def _accumulate_records(self, metrics: dict[str, Any]) -> None:
        if "data_time" in self.records.meters:
            self.records["data_time"].update(self._data_tick - self._batch_tick)
        if "batch_time" in self.records.meters:
            self.records["batch_time"].update(time.perf_counter() - self._batch_tick)
        for k, v in metrics.items():
            if k in self.records.meters:
                self.records[k].update(v)

    def _maybe_log(self, metrics: dict[str, Any]) -> None:
        infos = self.records.log() | {
            "lr": metrics.get("lr"),
            "step": self.step,
            "epoch": self.epoch,
        }
        infos.update(self._extra_log_fields())
        # Forward leaf/hook-emitted keys not in Records as their last value.
        for k, v in metrics.items():
            if k in infos or k in self.records.meters:
                continue
            infos[k] = v.item() if torch.is_tensor(v) else v
        wandb.log({f"train/{key}": value for key, value in infos.items() if value is not None})
        assert self._pbar is not None
        postfix = {k: infos[k] for k in self.pbar_keys if k in infos}
        if postfix:
            self._pbar.set_postfix(**postfix)

    def _finalize(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        if self.is_main:
            if (
                self.ckpt.save_final(self.step, self.epoch, **self._extra_checkpoint_kwargs())
                and self.step != self._last_validated_step
            ):
                self.validation_launcher.maybe_launch(self.step, self.ckpt.last)
                self._last_validated_step = self.step
            # Converge retention with whatever scores have landed; every
            # validation job re-converges on its own completion, so stragglers
            # (including the last step) trim whenever they report.
            self.ckpt.apply_retention()
            wandb.finish()
        dist.destroy_process_group()
        logger.info("Training finished")

    # ------------------------------------------------------------------
    # Iteration / batch hooks
    # ------------------------------------------------------------------
    def iterate_batches(self) -> Iterator[Any]:
        """Yield batches across epochs until the final training step."""
        while self.step < self.cfg.optimizer.max_steps:
            self.epoch += 1
            if hasattr(self.loader, "batch_sampler") and hasattr(self.loader.batch_sampler, "set_epoch"):
                self.loader.batch_sampler.set_epoch(self.epoch)  # type: ignore[union-attr]
            for batch in self.loader:
                self._data_tick = time.perf_counter()
                yield batch
                if self.step >= self.cfg.optimizer.max_steps:
                    break

    # ------------------------------------------------------------------
    # Leaf overrides
    # ------------------------------------------------------------------
    @abstractmethod
    def build_dataloader(self, cfg: BaseConfig) -> Iterable: ...

    @abstractmethod
    def training_step(self, step_input: Any, step: int) -> tuple[torch.Tensor, dict[str, Any]]:  # noqa: ANN401
        """Forward + loss inside autocast. Returns ``(loss_for_backward, metrics)``."""

    def build_optimizer(
        self,
        model: nn.Module,
        opt_cfg: Any,  # noqa: ANN401
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Default: ``gslm2.encoder.optimizer.build_optimizer(model, opt_cfg)``."""
        return _build_optimizer(model, opt_cfg)

    def after_checkpoint_load(self) -> None:  # noqa: B027
        """Hook fired after step/epoch are restored. Default: no-op."""

    def default_compile(self, model: nn.Module) -> nn.Module:
        """Default ``torch.compile`` strategy when ``cfg.run.compile`` is True."""
        model.compile(dynamic=True)
        return model

    def _extra_checkpoint_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for ``Checkpointer.save`` / ``save_final``. Default: ``{}``."""
        return {}

    def _extra_log_fields(self) -> dict[str, Any]:
        """Extra fields merged into the wandb log payload. Default: ``{}``."""
        return {}
