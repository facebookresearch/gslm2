# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/facebookresearch/textlesslib/blob/main/tools/distributed_transcribe/distributed.py

import datetime
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist


@dataclass(frozen=True, repr=True, eq=True, unsafe_hash=True)
class DistributedContext:
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    mode: str = "none"

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        """Return the device assigned to this rank."""
        return torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

    def barrier(self) -> None:
        """Synchronize ranks when a process group is initialized."""
        if dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)

    def all_gather_object[T](self, value: T) -> list[T]:
        """Gather a Python value from each rank in rank order."""
        if not dist.is_initialized():
            return [value]
        gathered: list[object | None] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return cast("list[T]", gathered)

    def close(self) -> None:
        """Close an initialized process group."""
        if dist.is_initialized():
            dist.destroy_process_group()


def shard[T](items: Sequence[T], context: DistributedContext) -> list[tuple[int, T]]:
    """Return a contiguous shard retaining the original example indices."""
    quotient, remainder = divmod(len(items), context.world_size)
    start = context.rank * quotient + min(context.rank, remainder)
    length = quotient + int(context.rank < remainder)
    return list(enumerate(items[start : start + length], start=start))


def init_distributed_context(port: int) -> DistributedContext:
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # default, non-distributed context
    context = DistributedContext(is_distributed=False, rank=0, local_rank=0, world_size=1, mode="none")

    launch_keys = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]
    slurm_keys = [
        "SLURM_LOCALID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_NODEID",
        "SLURM_JOB_NODELIST",
    ]

    # is it torch.distributed.launch?
    if all(key in os.environ for key in launch_keys):
        init_method = "env://"
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        context = DistributedContext(
            is_distributed=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            mode="launch",
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=datetime.timedelta(minutes=720),
            world_size=world_size,
            rank=rank,
        )
    # is it slurm?
    elif all(key in os.environ for key in slurm_keys):
        init_method = "env://"
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        hostnames = subprocess.check_output(
            [
                "/usr/bin/scontrol",
                "show",
                "hostnames",
                os.environ["SLURM_JOB_NODELIST"],
            ]
        )
        leader_addr = hostnames.split()[0].decode("utf-8")

        os.environ["MASTER_ADDR"] = leader_addr
        port_span = min(1000, 65535 - port + 1)
        os.environ["MASTER_PORT"] = str(port + int(os.environ.get("SLURM_JOB_ID", "0")) % port_span)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        context = DistributedContext(
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            mode="slurm",
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=datetime.timedelta(minutes=720),
            world_size=world_size,
            rank=rank,
        )

    return context
