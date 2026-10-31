# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Resample LJSpeech to 16 kHz and write manifests using fixed split IDs."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

SPLITS = ("train", "valid", "test")
DEFAULT_SPLITS_DIR = Path(__file__).resolve().parents[1] / "manifests" / "ljspeech"
SAMPLE_RATE = 16000


def read_split_ids(splits_dir: Path) -> dict[str, list[str]]:
    """Read the fixed utterance IDs and reject overlapping splits."""
    splits: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        path = splits_dir / f"ljspeech_{split}.txt"
        fileids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not fileids or any(re.fullmatch(r"LJ\d{3}-\d{4}", fileid) is None for fileid in fileids):
            raise ValueError(f"Missing or invalid LJSpeech file IDs in {path}")
        if len(fileids) != len(set(fileids)) or seen.intersection(fileids):
            raise ValueError(f"Duplicate or overlapping LJSpeech file IDs in {path}")
        seen.update(fileids)
        splits[split] = fileids
    return splits


def sample_count(path: Path) -> int:
    """Check the prepared audio format and return its waveform sample count."""
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"Expected nonempty mono 16 kHz audio: {path}")
    return int(info.frames)


def resample_sample(fileid: str, data_dir: Path, audio_dir: Path, ffmpeg: str) -> tuple[str, str, int]:
    """Prepare one WAV, reusing an existing valid output when present."""
    source = data_dir / "wavs" / f"{fileid}.wav"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = audio_dir / f"{fileid}.wav"
    if destination.is_file():
        return fileid, str(destination), sample_count(destination)

    with tempfile.NamedTemporaryFile(dir=audio_dir, suffix=".wav", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-threads",
                "1",
                str(staged),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        count = sample_count(staged)
        staged.replace(destination)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"FFmpeg failed for {fileid}: {error.stderr.strip()}") from error
    finally:
        staged.unlink(missing_ok=True)
    return fileid, str(destination), count


def prepare_manifests(data_dir: Path, output_dir: Path, splits_dir: Path, ffmpeg: str, workers: int) -> None:
    """Prepare audio and write train, validation, and test CSVs."""
    if workers < 1:
        raise ValueError("workers must be positive")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    splits = read_split_ids(splits_dir)
    audio_dir = output_dir / "wavs_16k"
    manifest_dir = output_dir / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fileids = [fileid for ids in splits.values() for fileid in ids]
    worker = partial(resample_sample, data_dir=data_dir, audio_dir=audio_dir, ffmpeg=executable)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        samples = {
            sample[0]: sample
            for sample in tqdm(executor.map(worker, fileids), total=len(fileids), desc="Preparing LJSpeech")
        }
    for split, ids in splits.items():
        path = manifest_dir / f"ljspeech_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(["fileid", "path", "num_frames"])
            writer.writerows(samples[fileid] for fileid in ids)
        print(f"{path}: {len(ids)} utterances")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Extracted LJSpeech-1.1 directory containing wavs/."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Write wavs_16k/ and manifests/ here.")
    parser.add_argument(
        "--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Fixed ljspeech_{train,valid,test}.txt IDs."
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (default: found on PATH).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent audio conversions (default: 4).")
    args = parser.parse_args()
    prepare_manifests(args.data_dir, args.output_dir, args.splits_dir, args.ffmpeg, args.workers)


if __name__ == "__main__":
    main()
