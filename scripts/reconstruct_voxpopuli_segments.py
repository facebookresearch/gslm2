# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reconstruct mono 16 kHz FLAC segments and a CSV manifest from VoxPopuli RTTMs.

Clip ends to the decoded OGG length and skip empty intervals. IDs retain the
requested RTTM boundaries; CSV sample counts describe the actual audio. Write
a sibling .reconstruction.json report with clipping details and decoder versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO

SAMPLE_RATE = 16000
RTTM_COLUMNS = 10
ZSTD = shutil.which("zstd") or "/usr/bin/zstd"


@contextmanager
def open_rttm(path: Path) -> Iterator[TextIO]:
    """Read a plain RTTM or stream its zstd decompression."""
    if path.suffix != ".zst":
        with path.open(encoding="utf-8") as source:
            yield source
        return
    with subprocess.Popen([ZSTD, "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE, text=True) as decoder:
        try:
            yield decoder.stdout
        except BaseException:
            decoder.kill()
            raise
        finally:
            decoder.stdout.close()
        if decoder.wait():
            raise RuntimeError(f"RTTM decompression failed: {path}")


def read_segments(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Convert RTTM seconds into exact sample boundaries, grouped by recording."""
    recordings: dict[str, list[tuple[int, int]]] = {}
    with open_rttm(path) as source:
        for number, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != RTTM_COLUMNS or fields[0] != "SPEAKER" or fields[7] != "SPEECH":
                raise ValueError(f"Expected a 10-column SPEECH RTTM row at {path}:{number}")
            recording = fields[1]
            if re.fullmatch(r"\d{4}[A-Za-z0-9_-]*_(?:en|fr|de)", recording) is None:
                raise ValueError(f"Unexpected recording ID: {recording}")
            onset, duration = Decimal(fields[3]), Decimal(fields[4])
            if not onset.is_finite() or not duration.is_finite() or onset < 0 or duration <= 0:
                raise ValueError(f"Invalid segment boundaries at {path}:{number}")
            start = int((onset * SAMPLE_RATE).to_integral_value(rounding=ROUND_HALF_UP))
            stop = int(((onset + duration) * SAMPLE_RATE).to_integral_value(rounding=ROUND_HALF_UP))
            if stop <= start:
                raise ValueError(f"Empty segment at {path}:{number}")
            recordings.setdefault(recording, []).append((start, stop))
    if not recordings:
        raise ValueError(f"No speech segments in {path}")
    for recording, segments in recordings.items():
        if len(segments) != len(set(segments)):
            raise ValueError(f"Duplicate segment boundaries for {recording}")
    return recordings


def find_recording(audio_root: Path, recording: str) -> Path:
    """Resolve the original OGG in the official raw_audios layout."""
    language = recording.rsplit("_", 1)[-1]
    path = audio_root / language / recording[:4] / f"{recording}.ogg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def reconstruct_recording(
    recording: str,
    segments: list[tuple[int, int]],
    audio_root: Path,
    output: Path,
    *,
    clippings: list[dict[str, str | int]] | None = None,
) -> list[tuple[str, str, int, str]]:
    """Decode once and clip segment ends to the source length, preserving RTTM IDs."""
    source = find_recording(audio_root, recording)
    waveform, sample_rate = sf.read(source, dtype="int16", always_2d=True)
    if sample_rate != SAMPLE_RATE or waveform.shape[1] != 1:
        raise ValueError(f"Expected mono 16 kHz source audio: {source}")
    language = recording.rsplit("_", 1)[-1]
    directory = output / language / recording[:4]
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for start, requested_stop in segments:
        fileid = f"{recording}_{start}_{requested_stop}"
        stop = min(requested_stop, len(waveform))
        segment = waveform[start:stop]
        num_frames = len(segment)
        if clippings is not None and (stop != requested_stop or num_frames == 0):
            clippings.append(
                {
                    "fileid": fileid,
                    "source": str(source),
                    "source_num_frames": len(waveform),
                    "requested_start_sample": start,
                    "requested_stop_sample": requested_stop,
                    "actual_start_sample": min(start, len(waveform)),
                    "actual_stop_sample": stop,
                    "num_frames": num_frames,
                    "status": "clipped" if num_frames else "skipped_empty",
                }
            )
        if num_frames == 0:
            continue
        target = directory / f"{fileid}.flac"
        if target.exists():
            info = sf.info(target)
            if (info.samplerate, info.channels, info.frames, info.format, info.subtype) != (
                SAMPLE_RATE,
                1,
                num_frames,
                "FLAC",
                "PCM_16",
            ):
                raise ValueError(f"Existing segment has an incompatible format or length: {target}")
        else:
            temporary = target.with_suffix(".flac.partial")
            try:
                sf.write(temporary, segment, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        rows.append((fileid, str(target), num_frames, language))
    return rows


def write_report(report: dict[str, object], destination: Path) -> None:
    """Write the clipping and decoder report atomically."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reconstruct(rttm: Path, audio_root: Path, output_dir: Path, manifest: Path) -> None:
    """Write an actual-length CSV and a clipping/version report; reuse compatible FLACs."""
    recordings = read_segments(rttm)
    audio_root, output_dir, manifest = audio_root.resolve(), output_dir.resolve(), manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    clippings: list[dict[str, str | int]] = []
    report_path = manifest.with_suffix(".reconstruction.json")
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            suffix=".partial",
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            writer = csv.writer(destination)
            writer.writerow(["fileid", "path", "num_frames", "language"])
            for number, (recording, segments) in enumerate(recordings.items(), 1):
                rows = reconstruct_recording(recording, segments, audio_root, output_dir, clippings=clippings)
                writer.writerows(rows)
                count += len(rows)
                if number % 100 == 0 or number == len(recordings):
                    print(
                        f"Decoded {number:,}/{len(recordings):,} recordings; wrote {count:,} manifest rows", flush=True
                    )
        clipped = sum(event["status"] == "clipped" for event in clippings)
        skipped = sum(event["status"] == "skipped_empty" for event in clippings)
        with temporary.open("rb") as handle:
            manifest_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        write_report(
            {
                "rttm": str(rttm.resolve()),
                "audio_root": str(audio_root),
                "output_dir": str(output_dir),
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha256,
                "sample_rate": SAMPLE_RATE,
                "soundfile_version": sf.__version__,
                "libsndfile_version": sf.__libsndfile_version__,
                "fileid_boundaries": "requested RTTM sample boundaries",
                "num_frames_basis": "actual extracted samples",
                "requested_segments": sum(map(len, recordings.values())),
                "written_segments": count,
                "clipped_segments": clipped,
                "skipped_empty_segments": skipped,
                "clippings": clippings,
            },
            report_path,
        )
        temporary.replace(manifest)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Wrote {count:,} segments and {manifest}", flush=True)
    print(f"Clipped {clipped:,} segment ends; skipped {skipped:,} empty segments; report: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rttm", type=Path, required=True, help="Plain .rttm or released .rttm.zst file.")
    parser.add_argument("--audio-root", type=Path, required=True, help="Downloaded VoxPopuli raw_audios/ directory.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Write FLAC files under language/year directories."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Write the audio CSV and a sibling .reconstruction.json report."
    )
    args = parser.parse_args()
    reconstruct(args.rttm, args.audio_root, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()
