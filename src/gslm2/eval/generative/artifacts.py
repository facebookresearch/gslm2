# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Merge distributed stage records in the original manifest order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gslm2.eval.generative.io import read_jsonl, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    from gslm2.shared.distributed import DistributedContext


def collect_records(
    directory: Path, records: list[dict[str, object]], context: DistributedContext, expected_ids: list[str]
) -> list[dict[str, object]]:
    """Persist each rank's shard, then verify and merge it on the leader."""
    shards = directory / ".shards"
    write_jsonl(shards / f"{context.rank}.jsonl", records, allow_empty=True)
    context.barrier()
    if not context.is_leader:
        return []
    merged = [
        row for rank in range(context.world_size) for row in read_jsonl(shards / f"{rank}.jsonl", allow_empty=True)
    ]
    if [row["fileid"] for row in merged] != expected_ids:
        raise ValueError("stage records do not preserve the complete input ID order")
    if [row["index"] for row in merged] != list(range(len(expected_ids))):
        raise ValueError("stage records do not cover every input index exactly once")
    return merged
