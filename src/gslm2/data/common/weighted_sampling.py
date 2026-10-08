"""Weighted batch samplers (weighted sampling with distributed sharding)."""

from __future__ import annotations

import logging
from collections.abc import Iterator  # noqa: TC003

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class WeightedDistributedBatchSampler:
    """Batch sampler with per-sample weights and distributed sharding.

    Draws ``num_samples`` indices (with replacement) according to ``weights``,
    groups them into batches, and shards across distributed ranks.
    """

    def __init__(
        self,
        weights: list[float],
        num_samples: int,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()

        batches: list[list[int]] = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)

        remainder = len(batches) % self.world_size
        if remainder:
            pad = self.world_size - remainder
            if self.rank == 0:
                logger.info(
                    "WeightedDistributedBatchSampler: padding %d batches by duplicating "
                    "the first %d to align with world_size=%d (epoch=%d)",
                    len(batches),
                    pad,
                    self.world_size,
                    self.epoch,
                )
            batches.extend(batches[:pad])

        # Shard across ranks
        for i in range(self.rank, len(batches), self.world_size):
            yield batches[i]

    def __len__(self) -> int:
        total_batches = self.num_samples // self.batch_size
        remainder = total_batches % self.world_size
        if remainder:
            total_batches += self.world_size - remainder
        return total_batches // self.world_size
