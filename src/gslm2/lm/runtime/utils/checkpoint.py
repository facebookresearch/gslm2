# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fairseq2.device import CPU
from fairseq2.file_system import LocalFileSystem
from fairseq2.gang import FakeGang, create_parallel_gangs
from fairseq2.io import TorchTensorDumper, TorchTensorLoader
from fairseq2.model_checkpoint import NativeModelCheckpointLoader

if TYPE_CHECKING:
    from gslm2.shared.utils.typing import PathLike


def unify_fsdp_checkpoint(
    sharded_ckpt_dir: PathLike,
    unified_ckpt_dir: PathLike,
    pp_size: int,
    tp_size: int,
) -> None:
    """
    Consolidate a sharded FSDP checkpoint into a unified checkpoint.
    This function takes a distributed checkpoint that has been sharded across multiple
    processes (using pipeline parallelism, tensor parallelism, and data parallelism) and
    consolidates it into a single unified checkpoint directory without any dp sharding.
    Args:
        sharded_ckpt_dir: Path to the directory containing the sharded checkpoint files.
        unified_ckpt_dir: Path to the directory where the unified checkpoint will be saved.
        pp_size: Pipeline parallelism size.
        tp_size: Tensor parallelism size.
    Returns:
        None
    Raises:
        FileNotFoundError: If the source checkpoint directory does not exist.
        IOError: If there are issues reading or writing checkpoint files.
    """
    if pp_size != 1:
        raise ValueError(f"Only pp=1 is currently supported. Got pp={pp_size}")

    for tp_rank in range(tp_size):
        root_gang = FakeGang(CPU, rank=tp_rank, size=tp_size)
        gangs = create_parallel_gangs(root_gang, tp_size=tp_size)

        # use the native checkpoint loader
        file_system = LocalFileSystem()
        loader = NativeModelCheckpointLoader(
            file_system,
            TorchTensorLoader(file_system),
        )
        state_dict_it = loader.lazy_load(
            Path(sharded_ckpt_dir),
            gangs,
        )

        state_dict = dict(list(state_dict_it))

        tensor_dumper = TorchTensorDumper(file_system)
        unified_ckpt_dir_path = Path(unified_ckpt_dir)
        unified_ckpt_dir_path.mkdir(parents=True, exist_ok=True)

        target_path = unified_ckpt_dir_path / "pp_00" / f"tp_{tp_rank:02d}" / "sdp_00.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tensor_dumper.dump(state_dict, target_path)
