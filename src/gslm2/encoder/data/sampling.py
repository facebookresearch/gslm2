"""Speech batch samplers and collation.

Provides :class:`BucketizeBatchSampler` (length-bucketed batching),
:class:`DistributedBatchSampler` (distributed wrapper with per-language
task support), and :class:`SpeechCollatorWithMasking` (padding, cropping,
and masking).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator  # noqa: TC003
from typing import TYPE_CHECKING, Literal

import polars as pl
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import BatchSampler, DistributedSampler

from gslm2.shared.environment import pg_ddp, pg_metalearning

if TYPE_CHECKING:
    from gslm2.encoder.data.dataset import SpeechDataset

logger = logging.getLogger()


def _verify_lengths(min_len: int, max_len: int, batch_size: int | None, max_token_count: int | None) -> None:
    exception = ""
    if not 0 <= min_len <= max_len:
        exception += "min_len must be less than or equal to max_len \n"
    if max_token_count is not None and batch_size is not None:
        exception += "max_token_count and batch_size cannot be set simultaneously \n"
    if max_token_count is None and batch_size is None:
        exception += "max_token_count or batch_size must be set \n"
    if max_token_count is not None and max_len > max_token_count:
        exception += "max_len must be less than or equal to max_token_count \n"
    if exception:
        raise ValueError(exception)


class BucketizeBatchSampler(BatchSampler):
    """Batch sampler that groups samples of similar length into buckets."""

    def __init__(  # noqa: PLR0913
        self,
        lengths: dict[int, int],
        num_buckets: int,
        min_len: int = 0,
        max_len: int | None = None,
        max_token_count: int | None = None,
        batch_size: int | None = None,
        seed: int = 0,
        *,
        bucket_method: Literal["uniform", "percentile"] = "uniform",
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max(lengths.values())
        _verify_lengths(min_len, max_len, batch_size, max_token_count)
        filtered_length_idx = [(min(length, max_len), i) for i, length in lengths.items() if min_len <= length]
        if not filtered_length_idx:
            exception = "No samples with length in the range"
            raise ValueError(exception)

        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size  # type: ignore[assignment]
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
        self.drop_last = drop_last
        if bucket_method == "uniform":
            uniform_limits = (min_len, max_len)
        elif bucket_method == "percentile":
            uniform_limits = None
        else:
            raise ValueError("bucket_method must be either 'uniform' or 'percentile'")
        self.buckets = self._get_buckets(self.lengths, num_buckets, uniform_limits)
        self._update_iter_list()

    def _get_buckets(
        self, lengths: list[int], num_buckets: int, uniform_limits: tuple[int, int] | None
    ) -> dict[int, Tensor]:
        buckets: dict[int, list[int]] = {}
        if uniform_limits is not None:
            boundaries = torch.linspace(uniform_limits[0] - 1, uniform_limits[1] + 1, num_buckets + 1)
        else:
            boundaries = torch.quantile(
                torch.tensor(lengths, dtype=torch.float32),
                torch.linspace(0, 1, num_buckets + 1),
                interpolation="lower",
            )[1:]
        bucket_ids = torch.bucketize(torch.tensor(lengths), boundaries)
        for i in range(bucket_ids.size(0)):
            bucket_id = int(bucket_ids[i])
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        return dict(sorted([(k, torch.as_tensor(v, dtype=torch.int)) for k, v in buckets.items()]))

    def _update_iter_list(self) -> None:
        if self.shuffle:
            for k in self.buckets:
                new_idx = torch.randperm(self.buckets[k].size(0), generator=self.g)
                self.buckets[k] = self.buckets[k][new_idx]
        self.iter_list = []
        total_len, batch = 0, []
        max_batch_size = self.max_token_count or self.batch_size
        for k in self.buckets:
            for i in range(self.buckets[k].size(0)):
                index = int(self.buckets[k][i])
                sample_length = self.lengths[index] if self.max_token_count else 1
                if total_len + sample_length <= max_batch_size:
                    batch.append(self.indices[index])
                    total_len += sample_length
                else:
                    self.iter_list.append(batch)
                    batch = [self.indices[index]]
                    total_len = sample_length
        if len(batch) > 0 and (self.max_token_count or not self.drop_last):
            self.iter_list.append(batch)

    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch
        self.g.manual_seed(self.seed)
        self._update_iter_list()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.iter_list)

    def __len__(self) -> int:
        return len(self.iter_list)


class DistributedBatchSampler(DistributedSampler):
    """Distributed sampler for BucketizeBatchSampler."""

    def __init__(
        self,
        batch_samplers: dict[str, BucketizeBatchSampler],
        seed: int = 0,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.batch_samplers = batch_samplers
        self.num_replicas = dist.get_world_size(pg_ddp()) if dist.is_initialized() else 1
        self.rank = dist.get_rank(pg_ddp()) if dist.is_initialized() else 0
        self.metalearning_rank = dist.get_rank(pg_metalearning()) if dist.is_initialized() else 0
        self.metalearning_world_size = dist.get_world_size(pg_metalearning()) if dist.is_initialized() else 1
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_language = None
        self.batch_language_list = None

    def set_epoch(self, epoch: int) -> None:
        for key in self.batch_samplers:
            self.batch_samplers[key].set_epoch(epoch)
        return super().set_epoch(epoch)

    def set_batch_language(
        self,
        step: int,
        reset_interval: int,
        reset_batch_language_method: str | None = None,
        fixed_language: str | None = None,
    ) -> bool:
        if fixed_language:
            assert not reset_batch_language_method, (
                "Cannot provide both fixed_language and reset_batch_language_method"
            )
            self.batch_language = fixed_language
        elif reset_batch_language_method == "with_replacement":
            step = step - (step % reset_interval)  # ensure to use last reset if training stopped midway
            g = torch.Generator()
            g.manual_seed(self.seed + step + self.metalearning_rank)
            languages = list(self.batch_samplers.keys())
            self.batch_language = languages[torch.randint(0, len(languages), (1,), generator=g)]
        elif reset_batch_language_method == "without_replacement":
            num_reset = step // reset_interval
            if not self.batch_language_list:
                g = torch.Generator()
                g.manual_seed(self.seed)
                languages = list(self.batch_samplers.keys())
                shuffled_indices = torch.randperm(len(languages), generator=g)
                self.batch_language_list = [languages[i] for i in shuffled_indices]
            batch_language_index = (num_reset * self.metalearning_world_size + self.metalearning_rank) % len(
                self.batch_language_list
            )  # each worker on each reset should have a unique index
            self.batch_language = self.batch_language_list[batch_language_index]
        else:
            raise ValueError("Did not recognize method for replacing batch language.")
        logger.info("Setting batch language to %s", self.batch_language)

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.metalearning_rank)
        subsets = []
        language_batches = (
            [self.batch_samplers[self.batch_language]] if self.batch_language else self.batch_samplers.values()
        )
        for batch_sampler in language_batches:
            if self.shuffle:  # First, shuffle each sampler independently
                perm = torch.randperm(len(batch_sampler.iter_list), generator=g).tolist()
                indices = [batch_sampler.iter_list[i] for i in perm]
            else:
                indices = batch_sampler.iter_list
            if self.drop_last:
                total_size = len(indices) - len(indices) % self.num_replicas
            else:
                padding_size = self.num_replicas - len(indices) % self.num_replicas
                indices += indices[:padding_size]
                total_size = len(indices)
            # Now split across ranks
            num_samples = total_size // self.num_replicas
            subset = indices[self.rank : total_size : self.num_replicas]
            if len(subset) != num_samples:
                # Every rank should have the same number of batches
                # This Exception should never be raised thanks to the padding step above
                exception = f"Rank {self.rank} has subset of length {len(subset)} but expected {num_samples}"
                raise ValueError(exception)
            subsets += subset

        self.num_samples = len(subsets)
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.metalearning_rank)
        if len(self.batch_samplers) > 1 and self.shuffle:  # Now shuffle across samplers
            subsets = [subsets[i] for i in torch.randperm(self.num_samples, generator=g).tolist()]
        self.subset = subsets
        return iter(self.subset)

    def __len__(self) -> int:
        return self.num_samples


class PackingBatchSampler:
    """First-fit bin-packing sampler for sequence packing.

    Yields batches whose total waveform length fits within max_token_count.
    Samples shorter than min_len are dropped; samples longer than max_len are
    clipped to max_len, matching BucketizeBatchSampler.
    """

    def __init__(  # noqa: PLR0913
        self,
        lengths: dict[int, int],
        max_token_count: int,
        min_len: int = 0,
        max_len: int | None = None,
        seed: int = 0,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max(lengths.values()) if lengths else 0
        self.max_token_count = max_token_count
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        self.items = [(idx, min(length, max_len)) for idx, length in lengths.items() if min_len <= length]
        self.g = torch.Generator()
        self.iter_list: list[list[int]] = []
        self._build_packing_plan()

    def _build_packing_plan(self) -> None:
        """Build the current epoch's packing plan by first-fit bin packing."""
        self.g.manual_seed(self.seed + self.epoch)

        perm = torch.randperm(len(self.items), generator=self.g).tolist()
        ordered_items = [self.items[i] for i in perm]

        rows: list[list[int]] = []
        row_lengths: list[int] = []
        for idx, length in ordered_items:
            placed = False
            for r in range(len(rows)):
                if row_lengths[r] + length <= self.max_token_count:
                    rows[r].append(idx)
                    row_lengths[r] += length
                    placed = True
                    break
            if not placed:
                rows.append([idx])
                row_lengths.append(length)

        # Drop before shuffling: first-fit appends the under-filled bin last.
        if self.drop_last and rows:
            rows = rows[:-1]

        if self.shuffle:
            row_perm = torch.randperm(len(rows), generator=self.g).tolist()
            rows = [rows[i] for i in row_perm]

        self.iter_list = rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build_packing_plan()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.iter_list)

    def __len__(self) -> int:
        return len(self.iter_list)


def lengths_by_lang(dataset: SpeechDataset) -> dict[str, dict[int, int]]:
    return {
        str(lang): dict(zip(df["index"].to_list(), df["num_frames"].to_list(), strict=True))
        for (lang,), df in (
            pl.from_dataframe(dataset.manifest)
            .select("language", "num_frames")
            .with_row_index()
            .group_by("language", maintain_order=True)
        )
    }
