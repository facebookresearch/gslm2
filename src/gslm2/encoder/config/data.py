# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading configuration."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DataConfig:
    """Dataloading and data sampling configuration."""

    manifest: str | None = None
    enable_padding: bool = False
    rand_crop: bool = True
    normalize: bool = True
    num_buckets: int = 1000
    min_sample_size: int = 2_000
    max_sample_size: int = 320_000
    max_batch_length: int = 3_800_000
    drop_last: bool = False
    num_workers: int = 10
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    random_seed: int = 0
    bucket_method: Literal["uniform", "percentile"] = "uniform"
    by_lang: bool = False
    lang_chunk_size: int | None = None
    sort_by_length: bool = False
    enable_packing: bool = False
    sampler_type: Literal["bucketize", "packing"] = "bucketize"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        """Build from a plain dict (YAML / ``asdict`` round-trip)."""
        return cls(**dict(d))
