# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from fairseq2.device import CPU
from fairseq2.gang import FakeGang, Gang, Gangs, ProcessGroupGang
from fairseq2.logging import get_log_writer
from fairseq2.recipe.base import RecipeContext
from fairseq2.recipe.internal.gang import (
    _log_ranks,
    _RecipeFSDPGangsFactory,
    _RecipeGangsFactory,
    _warmup_gangs,
)

if TYPE_CHECKING:
    from fairseq2.runtime.closable import Closable
    from fairseq2.runtime.dependency import DependencyResolver

# NOTE(mgleize): Right now the "create_gangs" is dependent on this single recipe config
# With multiple configs, we should make it depend on config sections
# like "GangSection" and "TrainerSection" instead.
from gslm2.fs2.unit_lm.recipes.train.config import UnitLMTrainConfig

logger = get_log_writer(__name__)


def _create_gangs(resolver: DependencyResolver) -> Gangs:
    """This is a copy-paste of a fairseq2-internal nested function."""
    gangs_factory = resolver.resolve(_RecipeGangsFactory)

    gangs = gangs_factory.create()

    fsdp_gangs_factory = resolver.resolve(_RecipeFSDPGangsFactory)

    gangs = fsdp_gangs_factory.create(gangs)

    _warmup_gangs(gangs)

    _log_ranks(gangs)

    return gangs


def maybe_setup_gangs(resolver: DependencyResolver) -> Gangs:
    context = RecipeContext(resolver)
    config = context.config.as_(UnitLMTrainConfig)

    if not dist.is_initialized():
        logger.info("Distributed not initialized, initializing process groups through fairseq2 gangs now.")
        # Let fs2 handle dist and process group initializations
        return _create_gangs(resolver)

    logger.info("Distributed already initialized, setting up gangs based on existing process groups.")
    # Build gangs on top of the existing process group.
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", rank)))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank!s}")
        torch.cuda.set_device(device)
    else:
        device = CPU
    # set up a single gang
    gang = (
        ProcessGroupGang(dist.group.WORLD, device)
        if dist.get_world_size() > 1  # possibly not needed if dist is initialized?
        else FakeGang(device)
    )

    # Handle replication vs sharding DP here, based on config
    fake_gang = FakeGang(device)

    if config.trainer.data_parallelism == "fsdp":
        sdp = gang
        rdp = fake_gang
    else:
        # These might not even be relevant for DDP, but better safe than sorry
        sdp = fake_gang
        rdp = gang

    return Gangs(
        root=gang,
        dp=gang,
        rdp=rdp,
        sdp=sdp,
        tp=fake_gang,
        pp=fake_gang,
    )


def disable_destroy_on_close(gangs: Gangs) -> None:
    """Disable the destruction of process groups when closing gangs.
    This is useful in scenarios where the process groups are managed externally
    or will be re-used afterwards and should not be destroyed when the gang is closed.
    :param gangs: The Gangs instance to modify.
    """
    gang_list: list[Gang] = [
        getattr(gangs, field.name)
        for field in gangs.__dataclass_fields__.values()
        if isinstance(getattr(gangs, field.name), Gang)
    ]

    def disable_close(closable: Closable) -> None:
        if not hasattr(closable, "close"):
            raise AttributeError(f"{closable!r} has no attribute 'close'")

        def noop_close(self):  # noqa: ANN001, ANN202
            pass

        closable.close = noop_close.__get__(closable, closable.__class__)

    for gang in gang_list:
        disable_close(gang)

    # Do not attempt to disable close on the top-level gangs object if it is frozen
    # (dataclasses.FrozenInstanceError). Only patch mutable objects.
    if not getattr(gangs, "__dataclass_params__", None) or not getattr(gangs.__dataclass_params__, "frozen", False):
        disable_close(gangs)
