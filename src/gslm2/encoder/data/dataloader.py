# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ruff: noqa: TC001
"""Speech dataloader factory."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from gslm2.encoder.config import DataConfig, MaskingConfig
from gslm2.encoder.data.collators import SpeechCollatorWithMasking
from gslm2.encoder.data.dataset import speech_dataset
from gslm2.encoder.data.masking.speech import MaskGenerator
from gslm2.encoder.data.sampling import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    PackingBatchSampler,
    lengths_by_lang,
)


def build_dataloader(
    data_cfg: DataConfig,
    mask_cfg: MaskingConfig,
    conv_layer_config: list[tuple[int, int, int]],
) -> DataLoader:
    collate_fn = SpeechCollatorWithMasking(
        MaskGenerator(mask_cfg),
        max_sample_size=data_cfg.max_sample_size,
        conv_layer_config=conv_layer_config,
        enable_padding=data_cfg.enable_padding,
        rand_crop=data_cfg.rand_crop,
    )
    dataset = speech_dataset(data_cfg)
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
