# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional CUDA memory profiling for SLM training."""

from __future__ import annotations

import logging
from collections.abc import Iterator  # noqa: TC003
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 100_000


def _is_leader() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


@contextmanager
def cuda_memory_snapshot(output_dir: Path, *, enabled: bool) -> Iterator[None]:
    """Record CUDA memory history over the wrapped block and dump it on exit.

    Writes ``memory_snapshot.pickle`` (view at https://pytorch.org/memory_viz) and
    ``memory_summary.txt`` under ``output_dir``. No-op unless ``enabled`` and running
    on CUDA as the leader rank.
    """
    if not (enabled and torch.cuda.is_available() and _is_leader()):
        yield
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.memory._record_memory_history(max_entries=_MAX_ENTRIES)  # noqa: SLF001
    logger.info("Recording CUDA memory history -> %s", output_dir)
    try:
        yield
    finally:
        try:
            (output_dir / "memory_summary.txt").write_text(torch.cuda.memory_summary())
            torch.cuda.memory._dump_snapshot(str(output_dir / "memory_snapshot.pickle"))  # noqa: SLF001
            logger.info("Dumped CUDA memory snapshot + summary -> %s", output_dir)
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001
