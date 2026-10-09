# ruff: noqa: TC001
"""Speech dataloader factory."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from gslm2.config import AugmentConfig, DataConfig, MaskingConfig
from gslm2.data.common.masking.speech import MaskGenerator
from gslm2.data.speech.collators import SLMAudioCollator, SpeechCollatorWithMasking
from gslm2.data.speech.dataset import speech_dataset
from gslm2.data.speech.sampling import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    PackingBatchSampler,
    lengths_by_lang,
)


class _IndexedDataset:
    """Wrap a dataset so ``__getitem__`` returns ``(index, sample)`` — lets the SLM
    collator key the on-the-fly unit cache by dataset index. Proxies ``lengths`` and
    ``__len__`` for the bucketized sampler."""

    def __init__(self, dataset: object) -> None:
        self._dataset = dataset

    def __getitem__(self, index: int) -> tuple[int, object]:
        return index, self._dataset[index]  # type: ignore[index]

    def __getitems__(self, indices: list[int]) -> list[tuple[int, object]]:
        # Preserve the wrapped dataset's concurrent batch fetch (Lance prefetch) while still
        # keying each sample by its dataset index for the SLM unit cache.
        getter = getattr(self._dataset, "__getitems__", None)
        samples = getter(indices) if getter else [self._dataset[i] for i in indices]  # type: ignore[index]
        return list(zip(indices, samples, strict=True))

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    @property
    def lengths(self) -> object:
        return self._dataset.lengths  # type: ignore[attr-defined]


def build_dataloader(
    data_cfg: DataConfig,
    mask_cfg: MaskingConfig | None,
    conv_layer_config: list[tuple[int, int, int]],
    augmentations: list[AugmentConfig] | None = None,
) -> DataLoader:
    # mask_cfg=None: on-the-fly SLM audio (raw waveforms + attention mask, no SSL masking).
    collate_fn = (
        SLMAudioCollator(
            data_cfg.max_sample_size,
            conv_layer_config,
            rand_crop=data_cfg.rand_crop,
        )
        if mask_cfg is None
        else SpeechCollatorWithMasking(
            MaskGenerator(mask_cfg),
            max_sample_size=data_cfg.max_sample_size,
            conv_layer_config=conv_layer_config,
            enable_padding=data_cfg.enable_padding,
            rand_crop=data_cfg.rand_crop,
            collate_augmentations=bool(augmentations),
            collate_phonemes=bool(data_cfg.alignments_path),
        )
    )
    dataset = speech_dataset(data_cfg, augmentations)
    if mask_cfg is None:  # SLM: yield (index, sample) so the trainer can cache units by index
        dataset = _IndexedDataset(dataset)
    if data_cfg.sampler_type == "packing":
        if data_cfg.by_lang:
            batch_samplers = {
                lang: PackingBatchSampler(
                    lengths,
                    max_token_count=data_cfg.max_batch_length,
                    min_len=data_cfg.min_sample_size,
                    max_len=data_cfg.max_sample_size,
                    seed=data_cfg.random_seed,
                    shuffle=True,
                    drop_last=data_cfg.drop_last,
                )
                for lang, lengths in lengths_by_lang(dataset).items()
            }
        else:
            batch_samplers = {
                "any": PackingBatchSampler(
                    dict(zip(range(len(dataset.lengths)), dataset.lengths, strict=True)),
                    max_token_count=data_cfg.max_batch_length,
                    min_len=data_cfg.min_sample_size,
                    max_len=data_cfg.max_sample_size,
                    seed=data_cfg.random_seed,
                    shuffle=True,
                    drop_last=data_cfg.drop_last,
                )
            }
    elif data_cfg.by_lang:
        batch_samplers = {
            lang: BucketizeBatchSampler(
                lengths,
                num_buckets=data_cfg.num_buckets,
                min_len=data_cfg.min_sample_size,
                max_len=data_cfg.max_sample_size,
                max_token_count=data_cfg.max_batch_length,
                seed=data_cfg.random_seed,
                bucket_method=data_cfg.bucket_method,
                shuffle=True,
                drop_last=data_cfg.drop_last,
            )
            for lang, lengths in lengths_by_lang(dataset).items()
        }
    else:
        batch_samplers = {
            "any": BucketizeBatchSampler(
                dict(zip(range(len(dataset.lengths)), dataset.lengths, strict=True)),
                num_buckets=data_cfg.num_buckets,
                min_len=data_cfg.min_sample_size,
                max_len=data_cfg.max_sample_size,
                max_token_count=data_cfg.max_batch_length,
                seed=data_cfg.random_seed,
                shuffle=True,
                drop_last=data_cfg.drop_last,
            )
        }
    distributed_sampler = DistributedBatchSampler(
        batch_samplers,
        seed=data_cfg.random_seed,
        shuffle=True,
        drop_last=data_cfg.drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=distributed_sampler,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
        generator=torch.Generator().manual_seed(
            data_cfg.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )
