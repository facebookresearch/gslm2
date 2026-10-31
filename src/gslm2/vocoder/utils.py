# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed setup, checkpoint discovery, and metric accumulation."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist

if TYPE_CHECKING:
    from types import TracebackType
    from typing import TextIO

_MIN_BFLOAT16_COMPUTE_CAPABILITY = 8


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def set_pytorch_flags() -> None:
    """Enable the CUDA math settings used by the reference runs."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.autograd.set_detect_anomaly(mode=False, check_nan=True)


def init_distributed() -> torch.device:
    """Initialize one NCCL process per local GPU under torchrun, SLURM, or locally."""
    if os.getenv("LOCAL_RANK") is not None:
        global_rank = os.environ["RANK"]
        local_rank = os.environ["LOCAL_RANK"]
        world_size = os.environ["WORLD_SIZE"]
        master_port = os.environ["MASTER_PORT"]
        master_addr = os.environ["MASTER_ADDR"]
    elif os.getenv("SLURM_JOB_ID") is not None:
        global_rank = os.environ["SLURM_PROCID"]
        local_rank = os.environ["SLURM_LOCALID"]
        world_size = os.environ["SLURM_NTASKS"]
        scontrol = shutil.which("scontrol")
        if scontrol is None:
            raise RuntimeError("scontrol is required for a SLURM training launch")
        master_addr = subprocess.check_output(
            [scontrol, "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            text=True,
        ).split()[0]
        master_port = str(random.Random(int(os.environ["SLURM_JOB_ID"])).randint(20_000, 60_000))
    else:
        global_rank = local_rank = "0"
        world_size = "1"
        master_port = str(random.Random(-1).randint(20_000, 60_000))
        master_addr = "127.0.0.1"

    os.environ.update(
        RANK=global_rank,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        MASTER_PORT=master_port,
        MASTER_ADDR=master_addr,
    )
    device = torch.device("cuda", int(local_rank))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    return device


def training_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve and validate a configured CUDA training dtype."""
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if dtype == torch.bfloat16 and torch.cuda.get_device_properties(device).major < _MIN_BFLOAT16_COMPUTE_CAPABILITY:
        raise ValueError("bfloat16 requires an Ampere-or-newer GPU")
    return dtype


class AverageMeter:
    """Device-side running average for one scalar metric."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated sum and count."""
        self.sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.count = torch.zeros((), device=self.device, dtype=torch.long)
        self.avg = torch.zeros((), device=self.device, dtype=self.dtype)

    def update(self, value: Tensor) -> None:
        """Add one scalar observation."""
        self.sum += value.detach()
        self.count += 1
        self.avg = self.sum / self.count

    def pop(self) -> float:
        """Return the current average and reset the meter."""
        value = self.avg.item()
        self.reset()
        return value

    def state_dict(self) -> dict[str, Tensor]:
        """Return serializable meter state."""
        return {"sum": self.sum, "count": self.count, "avg": self.avg}

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Restore meter state on the configured device."""
        self.sum = state_dict["sum"].to(device=self.device, dtype=self.dtype)
        self.count = state_dict["count"].to(device=self.device, dtype=torch.long)
        self.avg = state_dict["avg"].to(device=self.device, dtype=self.dtype)


class AverageMeters:
    """Lazily constructed collection of scalar running averages."""

    def __init__(self) -> None:
        self.meters: dict[str, AverageMeter] = {}

    def update(self, **values: Tensor) -> None:
        """Add one observation for each named metric."""
        for name, value in values.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(value.device, value.dtype)
            self.meters[name].update(value)

    def pop(self) -> dict[str, float]:
        """Return and reset every named average."""
        return {name: meter.pop() for name, meter in self.meters.items()}

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        """Return serializable state for every meter."""
        return {name: meter.state_dict() for name, meter in self.meters.items()}

    def load_state_dict(self, state_dict: dict[str, dict[str, Tensor]]) -> None:
        """Restore every meter from a checkpoint."""
        self.meters = {}
        for name, meter_state in state_dict.items():
            meter = AverageMeter(meter_state["sum"].device, meter_state["sum"].dtype)
            meter.load_state_dict(meter_state)
            self.meters[name] = meter


class MetricsLogger:
    """Append JSON metric records to one rank-local file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.writer: TextIO | None = None

    def log(self, metrics: dict[str, object]) -> None:
        """Write and flush one JSON record."""
        if self.writer is None:
            raise RuntimeError("metrics logger is not open")
        print(json.dumps(metrics), file=self.writer, flush=True)

    def __enter__(self) -> Self:
        self.writer = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def latest_checkpoint(folder: Path) -> Path | None:
    """Return the numerically latest training-state checkpoint."""
    checkpoints = sorted(folder.glob("checkpoint-*.pt"), key=lambda path: int(path.stem.removeprefix("checkpoint-")))
    return checkpoints[-1] if checkpoints else None
