# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data utilities."""

import math
import mmap
import os
from io import BytesIO
from pathlib import Path

import polars as pl
import soundfile as sf
from polars.dataframe.frame import DataFrame
from torchcodec.decoders import AudioDecoder


def num_samples(source: str | Path | bytes, *, verify: bool = False) -> int:
    try:
        return sf.info(BytesIO(source) if isinstance(source, bytes) else source).frames
    except sf.LibsndfileError:
        # Keep support for formats that FFmpeg can read but libsndfile cannot.
        pass
    metadata = AudioDecoder(source).metadata
    samples = metadata.duration_seconds_from_header * metadata.sample_rate
    count = round(samples)
    # FFmpeg header durations have microsecond precision, in addition to
    # floating point rounding when converting the duration to samples.
    tolerance = metadata.sample_rate * 1e-6 + 2 * math.ulp(samples)
    if verify and not math.isclose(samples, count, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"Number of samples {samples} is not an integer"
            + (f" in {source}" if isinstance(source, (str, Path)) else "")
        )
    return count


def bytes_from_archive(archive: Path | str, offset: int, file_size: int) -> bytes:
    with Path(archive).open("rb") as path, mmap.mmap(path.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
        return mmap_o[offset : offset + file_size]


def read_manifest(path: Path | str) -> DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix != ".tsv":
        raise ValueError("Only .csv and .tsv files are supported")
    with path.open("r") as file:
        root = Path(file.readline().strip())
    if not root.is_dir():
        raise ValueError("First line must be the root directory of the dataset")
    return (
        pl.scan_csv(path, separator="\t", skip_rows=1, has_header=False, new_columns=["fileid", "num_frames"])
        .with_columns((f"{root}{os.sep}" + pl.col("fileid")).alias("path"))
        .select("fileid", "path", "num_frames")
        .collect()
    )
