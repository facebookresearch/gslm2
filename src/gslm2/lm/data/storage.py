# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""S3-EOZ (bbfs) path resolution and shard-read retry for manifest datasets.

Bbfs mounts an S3 profile at ``/s3/profiles/<profile>`` on compute nodes and
``/bbfs/s3/profiles/<profile>`` on login nodes. This module centralises those
paths and the S3-EOZ-tuned retry policy so the manifest reader itself stays
storage-agnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from fairseq2.logging import get_log_writer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_log_writer(__name__)

RW_PROFILE = "licensed-speech-data--fair-aws-3--use2-az3--x-s3-rw"
COMPUTE_ROOT = Path(f"/s3/profiles/{RW_PROFILE}")
LOGIN_ROOT = Path(f"/bbfs/s3/profiles/{RW_PROFILE}")

# S3-EOZ shard reads are subject to transient bbfs failures; retry with exponential
# backoff up to the attempt count before giving up.
SHARD_READ_ATTEMPTS = 6
SHARD_READ_BACKOFF_SEC = 2.0
SHARD_READ_BACKOFF_CAP_SEC = 60.0


def bbfs_storage_root() -> Path:
    """Return the bbfs profile root visible on this host."""
    for root in (COMPUTE_ROOT, LOGIN_ROOT):
        if root.is_dir():
            return root
    raise RuntimeError(
        f"Neither {COMPUTE_ROOT} nor {LOGIN_ROOT} is available; the bbfs mount is "
        "missing on this node. Retry the job rather than proceeding."
    )


def resolve_manifest_root(path: Path) -> Path:
    """Resolve a manifest root, treating relative paths as keys under the bbfs profile."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else bbfs_storage_root() / path


def read_manifest_s3(
    read_fn: Callable[[], bytes],
    path: str,
) -> bytes:
    """Perform one S3-EOZ shard read with exponential-backoff retry on transient failure."""
    for attempt in range(SHARD_READ_ATTEMPTS):
        try:
            return read_fn()
        except OSError as ex:
            if attempt == SHARD_READ_ATTEMPTS - 1:
                raise RuntimeError(
                    f"{path} could not be read after {attempt + 1} attempts; the S3-EOZ storage is not recovering."
                ) from ex
            delay = min(SHARD_READ_BACKOFF_SEC * 2**attempt, SHARD_READ_BACKOFF_CAP_SEC)
            # fairseq2's LogWriter formats with str.format, not %-style.
            logger.warning(  # noqa: PLE1205
                "shard read failed (attempt {}/{}) at {}; retrying in {:.0f}s: {}",
                attempt + 1,
                SHARD_READ_ATTEMPTS,
                path,
                delay,
                ex,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
