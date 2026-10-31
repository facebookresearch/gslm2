# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Write manifest files."""

import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TypedDict

import polars as pl
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from gslm2.shared.audio import bytes_from_archive, num_samples


class CompressedArchiveError(Exception):
    """Archive must be uncompressed to be read."""


class NoAudioFileError(Exception):
    """No audio file found in the archive."""

    def __init__(self, archive: Path | str, file_extension: str) -> None:
        super().__init__(f"No file found in {archive} with extension `{file_extension}`.")


class _FileInfo(TypedDict):
    path: str
    num_frames: int


def _file_info(path: Path) -> _FileInfo:
    frames = num_samples(path)
    # libsndfile reports INT64_MAX when a FLAC has no total sample count.
    if frames == (1 << 63) - 1:
        frames = AudioDecoder(path).get_all_samples().data.shape[-1]
    return {"path": str(path), "num_frames": frames}


def _files_info(dataset: Path | str, file_extension: str, num_workers: int) -> list[_FileInfo]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    paths = list(Path(dataset).resolve().rglob(f"*{file_extension}"))
    if num_workers == 1:
        return list(tqdm(map(_file_info, paths), total=len(paths)))
    # Spawn avoids inheriting native thread pools initialized by Polars/PyTorch.
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as pool:
        return list(tqdm(pool.map(_file_info, paths, chunksize=64), total=len(paths)))


def write_manifest(
    dataset: Path | str, output: Path | str, file_extension: str = ".wav", *, num_workers: int = 1
) -> None:
    """Write a manifest file containing the file paths and their number of samples.

    Each line contains the absolute path of the file and its number of samples.
    """
    files_info = _files_info(dataset, file_extension, num_workers)
    (
        pl.from_dicts(files_info)
        .lazy()
        .with_columns(pl.col("path").str.strip_suffix(file_extension).str.split(os.sep).list.get(-1).alias("fileid"))
        .select("fileid", "path", "num_frames")
        .sink_csv(output)
    )


def write_manifest_fairseq(
    dataset: Path | str, output: Path | str, file_extension: str = ".wav", *, num_workers: int = 1
) -> None:
    """Write a manifest file containing the file paths and their number of samples.

    First line is the root directory of the dataset.
    Each line contains the relative path of the file and its number of samples.
    """
    root = Path(dataset).resolve()
    lines = [str(root)]
    lines += [
        f"{Path(info['path']).relative_to(root)}\t{info['num_frames']}"
        for info in _files_info(root, file_extension, num_workers)
    ]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest_tar(dataset: Path | str, output: Path | str, file_extension: str = ".wav") -> None:
    if not tarfile.is_tarfile(dataset):
        raise ValueError(dataset)
    with tarfile.open(dataset, mode="r") as tar_file:
        if tar_file.mode != "r":
            raise CompressedArchiveError
        infolist = tar_file.getmembers()
    bytes_info = [
        {
            "path": info.name,
            "num_frames": num_samples(bytes_from_archive(dataset, info.offset_data, info.size)),
            "byte_offset": info.offset_data,
            "byte_size": info.size,
        }
        for info in tqdm(infolist)
        if not info.isdir() and info.name.endswith(file_extension)
    ]
    if not bytes_info:
        raise NoAudioFileError(dataset, file_extension)
    (
        pl.from_dicts(bytes_info)
        .lazy()
        .with_columns(
            pl.col("path").str.strip_suffix(file_extension).str.split(os.sep).list.get(-1).alias("fileid"),
            pl.lit(dataset).alias("archive"),
        )
        .select("fileid", "path", "num_frames", "archive", "byte_offset", "byte_size")
        .sink_csv(output)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write manifest files.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset directory or uncompressed tar file.")
    parser.add_argument("output", type=Path, help="Path to the output manifest file.")
    parser.add_argument("--ext", type=str, default=".wav", help="Extension of audio files. (default: %(default)s)")
    parser.add_argument("--fairseq", action="store_true", help="Write a Fairseq-style TSV manifest, instead of CSV.")
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Processes for directory audio metadata. (default: %(default)s)"
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    root = args.dataset.resolve()
    if root.is_dir():
        (write_manifest_fairseq if args.fairseq else write_manifest)(
            root, args.output, args.ext, num_workers=args.num_workers
        )
    elif root.suffix == ".tar":
        if args.num_workers != 1:
            parser.error("--num-workers is supported for audio directories")
        write_manifest_tar(root, args.ext)
    else:
        raise parser.error(f"{root} is not a directory or a tar file.")
