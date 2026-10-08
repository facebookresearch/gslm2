"""Wandb run init + per-group metric namespace setup."""

import dataclasses
import logging
from collections.abc import Iterable
from pathlib import Path

import wandb

from gslm2.config import BaseConfig

logger = logging.getLogger()


def init_wandb(cfg: BaseConfig, group_prefixes: Iterable[str] | None = None) -> str:
    """Init the wandb run; declare ``train/*`` + one namespace per group prefix.

    Uses the run directory name as a deterministic wandb ID so that resumed
    jobs continue the same wandb run instead of creating a duplicate.
    """
    logger.debug("Initializing wandb")
    run = wandb.init(
        project=cfg.run.wandb_project,
        name=cfg.run.wandb_name or cfg.run.dir.name,
        mode=cfg.run.wandb_mode,
        dir=Path(cfg.run.workdir).resolve(),
        config=dataclasses.asdict(cfg),
        id=cfg.run.dir.name,
        resume="allow",
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    prefixes = list(group_prefixes) if group_prefixes is not None else cfg.eval_group_prefixes()
    for prefix in prefixes:
        wandb.define_metric(f"{prefix}/step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
    return run.id
