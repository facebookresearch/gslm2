import json
import logging
import warnings
from collections.abc import Generator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from gslm2.config import BaseModelConfig, ModelType
from gslm2.tools.metrics_queue import MetricsQueue

logger = logging.getLogger()


def remove_param_group(optim: Optimizer, index: int) -> None:
    for param in optim.param_groups[index]["params"]:
        if param in optim.state:
            del optim.state[param]
    del optim.param_groups[index]


def remove_frozen_param_groups(optim: Optimizer) -> None:
    """Drop every param group tagged ``is_frozen`` (feature-extractor freeze at
    ``freeze_step``). No-op when the optimizer has no such groups."""
    for index in reversed(range(len(optim.param_groups))):
        if optim.param_groups[index].get("is_frozen"):
            remove_param_group(optim, index)


def find_checkpoints(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("step_*.pt"), key=lambda x: int(x.stem.removeprefix("step_")))
    if (folder / "final.pt").is_file():
        paths.append(folder / "final.pt")
    return paths


def model_type_from_checkpoint(checkpoint: str | Path) -> ModelType:
    """Read ``run.model_type`` from the ``config.json`` next to a checkpoint."""
    config_path = Path(checkpoint).parent / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Cannot infer model_type: no config.json next to checkpoint {checkpoint}.")
    with config_path.open() as f:
        model_type = json.load(f).get("run", {}).get("model_type")
    if model_type is None:
        raise ValueError(f"config.json next to {checkpoint} has no run.model_type.")
    return model_type


def config_from_checkpoint(config_cls: type[BaseModelConfig], checkpoint: str | Path) -> BaseModelConfig:
    """Reconstruct a model config from the ``config.json`` next to a checkpoint.

    Falls back to defaults (with a warning) when no ``config.json`` is present.
    Unknown fields are dropped so checkpoints from other branches still load.
    """
    config_path = Path(checkpoint).parent / "config.json"
    if not config_path.is_file():
        warnings.warn("Config file not found when loading checkpoint. Using default config.", stacklevel=2)
        return config_cls()
    with config_path.open() as f:
        json_cfg = json.load(f)
    if "model" in json_cfg:
        json_cfg = json_cfg["model"]
    valid_fields = {f.name for f in fields(config_cls)}
    unknown_fields = set(json_cfg.keys()) - valid_fields
    if unknown_fields:
        logger.warning("Ignoring unknown config fields: %s", unknown_fields)
        json_cfg = {k: v for k, v in json_cfg.items() if k in valid_fields}
    return config_cls(**json_cfg)


def compare_model_configs(ckpt_model_cfg: BaseModelConfig, current_model_cfg: BaseModelConfig) -> None:
    """Log differences between a checkpoint's config and the current one."""
    differences = {}
    for attr in vars(current_model_cfg):
        ckpt_value = getattr(ckpt_model_cfg, attr)
        current_value = getattr(current_model_cfg, attr)
        if ckpt_value != current_value:
            differences[attr] = (ckpt_value, current_value)
    if differences:
        logger.info("Differences found between the provided checkpoint's config and the existing config:")
        for attr, (ckpt_value, current_value) in differences.items():
            logger.info("{%s}:  %s != %s", attr, ckpt_value, current_value)


@dataclass
class Checkpoint:
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler
    epoch: torch.Tensor
    supervised_epoch: torch.Tensor
    step: torch.Tensor


class Checkpointer:
    def __init__(  # noqa: PLR0913
        self,
        folder: Path,
        interval: int,
        keep_latest: int = -1,
        *,
        disable: bool = False,
        start: int = 0,
        keep_best_n: int = 0,
        keep_best_metric: str = "dev_loss",
        keep_best_mode: str = "min",
    ) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.keep_latest = keep_latest
        self.keep_best_n = keep_best_n
        self.keep_best_metric = keep_best_metric
        self.keep_best_mode = keep_best_mode
        self.disable = disable or interval <= 0
        self._path: Path | None = None
        self._state: Checkpoint
        self._metrics_queue = MetricsQueue(self.folder / "metrics.jsonl", interval, start)

    @property
    def step(self) -> torch.Tensor:
        return self._state.step

    @property
    def epoch(self) -> torch.Tensor:
        return self._state.epoch

    @property
    def supervised_epoch(self) -> torch.Tensor:
        return self._state.supervised_epoch

    @property
    def last(self) -> Path | None:
        if self._path is not None:
            return self._path
        if len(checkpoints := find_checkpoints(self.folder)) > 0:
            self._path = checkpoints[-1]
            return self._path
        return None

    @property
    def metrics(self) -> Path:
        return self.folder / "metrics.jsonl"

    def init_state(self, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler) -> None:
        epoch, supervised_epoch, step = (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
        ckpt = Checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            supervised_epoch=supervised_epoch,
            step=step,
        )
        self._state = ckpt

    def _metric_values(self) -> dict[int, list[float]]:
        """Per-step score values from ``metrics.jsonl`` eval entries (async jobs append)."""
        per_step: dict[int, list[float]] = {}
        try:
            lines = (self.folder / "metrics.jsonl").read_text().splitlines()
        except OSError:
            return {}
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not str(entry.get("group", "")).startswith("evals_"):
                continue
            value = entry.get(self.keep_best_metric)
            if not isinstance(value, (int, float)):
                continue
            per_step.setdefault(int(entry["step"]), []).append(float(value))
        return per_step

    def _ranked_steps(self) -> list[int]:
        aggregate = min if self.keep_best_mode == "min" else max
        per_step = self._metric_values()
        return sorted(per_step, key=lambda s: (aggregate(per_step[s]), s))

    def best_steps(self) -> set[int]:
        """Top-N evaluated steps by the score metric (spared from retention)."""
        if self.keep_best_n <= 0:
            return set()
        ranked = self._ranked_steps()
        if self.keep_best_mode == "min":
            return set(ranked[: self.keep_best_n])
        return set(ranked[-self.keep_best_n :])

    def apply_retention(self) -> None:
        """Re-run retention now (e.g. end of training, once late scores landed)."""
        self._delete_old_checkpoints()

    def _delete_old_checkpoints(self) -> None:
        if self.keep_latest <= 0 and self.keep_best_n <= 0:
            return
        checkpoints = find_checkpoints(self.folder)
        if self.keep_latest <= 0:
            candidates = list(checkpoints)
        else:
            if len(checkpoints) <= self.keep_latest:
                return
            candidates = checkpoints[: -self.keep_latest]
        scored = set(self._metric_values())
        best = self.best_steps() if self.keep_best_n > 0 else set()
        stepped = [(p, self._step_nr(p)) for p in candidates]
        if skipped := sum(1 for _, s in stepped if s is not None and s not in best and s not in scored):
            logger.info("Retention keeping %d unscored checkpoint(s) until their scores land", skipped)
        to_delete = [p for p, s in stepped if s is not None and s not in best and s in scored]
        for path in to_delete:
            logger.info("Deleting old checkpoint %s", path)
            path.unlink()

    @staticmethod
    def _step_nr(path: Path) -> int | None:
        stem = path.stem
        return int(stem.removeprefix("step_")) if stem.startswith("step_") else None

    def _save_to_path(self, path: Path, step: int, epoch: int, supervised_epoch: int) -> bool:
        if self.disable:
            return False
        self._state.step.fill_(step)
        self._state.epoch.fill_(epoch)
        self._state.supervised_epoch.fill_(supervised_epoch)
        torch.save(
            {
                "model": self._state.model.state_dict(),
                "optimizer": self._state.optimizer.state_dict(),
                "scheduler": self._state.scheduler.state_dict(),
                "epoch": self.epoch,
                "supervised_epoch": self.supervised_epoch,
                "step": self.step,
            },
            path,
        )
        self._path = path
        self._delete_old_checkpoints()
        return True

    def save(self, step: int, epoch: int, *, supervised_epoch: int = 0, force: bool = False) -> bool:
        should_save = (not self.disable) and (force or step % self.interval == 0)
        if not should_save:
            return False
        return self._save_to_path(self.folder / f"step_{step}.pt", step, epoch, supervised_epoch)

    def save_latest(self, step: int, epoch: int, *, supervised_epoch: int = 0, interval: int = 300) -> bool:
        if self.disable or step % interval != 0:
            return False
        return self._save_to_path(self.folder / "latest.pt", step, epoch, supervised_epoch)

    def save_final(self, step: int, epoch: int, *, supervised_epoch: int = 0) -> bool:
        return self._save_to_path(self.folder / "final.pt", step, epoch, supervised_epoch)

    def _load_from_path(self, path: Path) -> bool:
        if self.disable or not path.is_file():
            return False
        if path.name == "final.pt":
            logger.warning("Loading final checkpoint")
        ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        self._state.step = ckpt["step"]
        self._state.epoch = ckpt["epoch"]
        self._state.supervised_epoch = ckpt["supervised_epoch"]
        self._state.scheduler.load_state_dict(ckpt["scheduler"])

        consume_prefix_in_state_dict_if_present(ckpt["model"], "module.")
        self._state.model.load_state_dict(ckpt["model"])

        freeze_step = getattr(self._state.model, "freeze_step", None)
        if freeze_step is not None and self._state.model.current_step >= freeze_step:
            self._state.model.freeze_extractor()
            remove_frozen_param_groups(self._state.optimizer)
        assert len(self._state.optimizer.param_groups) == len(ckpt["optimizer"]["param_groups"]), (
            "inconsistency across current and ckpt's param_groups."
        )
        self._state.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Resuming from checkpoint %s (epoch %d, step %d)", path.name, ckpt["epoch"], ckpt["step"])
        return True

    def load_existing_run(self) -> bool:
        # Prefer latest.pt if it exists and is newer than the last numbered checkpoint
        latest = self.folder / "latest.pt"
        if latest.is_file():
            return self._load_from_path(latest)
        if (path := self.last) is None:
            return False
        return self._load_from_path(path)

    def load(self, step: int) -> bool:
        return self._load_from_path(self.folder / f"step_{step}.pt")

    def find_new_metrics(self) -> Generator[dict[str, Any], None, None]:
        yield from self._metrics_queue()
