# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve bundled LibriSpeech evaluation manifests against a local download."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

SPLITS = ("dev-clean", "test-clean")
DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests" / "librispeech"
SAMPLE_RATE = 16000


def local_audio_rows(path: Path, data_dir: Path) -> list[dict[str, str]]:
    """Resolve relative audio paths while preserving the fixed selection."""
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    seen: set[str] = set()
    for row in rows:
        fileid = row["fileid"]
        relative = Path(row["path"])
        audio = (data_dir / relative).resolve()
        if relative.is_absolute() or not audio.is_relative_to(data_dir):
            raise ValueError(f"Expected an audio path relative to LibriSpeech/: {row['path']}")
        if audio.stem != fileid or fileid in seen:
            raise ValueError(f"Invalid or duplicate LibriSpeech file ID: {fileid}")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if int(row["num_frames"]) <= 0:
            raise ValueError(f"Invalid sample count for {fileid}")
        seen.add(fileid)
        row["path"] = str(audio)
    if not rows:
        raise ValueError(f"Empty audio manifest: {path}")
    return rows


def check_boundaries(path: Path, rows: list[dict[str, str]]) -> None:
    """Require one valid, fixed prompt boundary for each selected utterance."""
    lengths = {row["fileid"]: int(row["num_frames"]) for row in rows}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            fileid = record["fileid"]
            if fileid not in lengths or fileid in seen:
                raise ValueError(f"Unexpected or duplicate prompt file ID: {fileid}")
            seconds = float(record["prompt_secs"])
            if not math.isfinite(seconds) or not 0 < seconds <= lengths[fileid] / SAMPLE_RATE:
                raise ValueError(f"Invalid prompt duration for {fileid}: {seconds}")
            if type(record["prompt_frames"]) is not int or record["prompt_frames"] <= 0:
                raise ValueError(f"Invalid prompt frame count for {fileid}")
            seen.add(fileid)
    if seen != lengths.keys():
        raise ValueError(f"Missing prompt boundaries in {path}")


def prepare_transcripts(source: Path, output: Path, rows: list[dict[str, str]]) -> None:
    """Preserve full ASR transcripts while replacing their audio paths locally."""
    audio_by_id = {row["fileid"]: row["path"] for row in rows}
    transcripts: dict[str, str] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fileid = row.get("fileid") or Path(row.get("audio", row.get("file_name", ""))).stem
            transcript = row.get("transcript")
            if fileid not in audio_by_id or fileid in transcripts:
                raise ValueError(f"Unexpected or duplicate transcript ID: {fileid}")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"Missing full transcript for {fileid}")
            transcripts[fileid] = transcript
    if transcripts.keys() != audio_by_id.keys():
        raise ValueError(f"Transcript IDs must match the fixed selection: {source}")
    with output.open("w", encoding="utf-8") as destination:
        for row in rows:
            fileid = row["fileid"]
            destination.write(
                json.dumps({"fileid": fileid, "audio": audio_by_id[fileid], "transcript": transcripts[fileid]}) + "\n"
            )


def prepare_split(
    data_dir: Path, manifests_dir: Path, output_dir: Path, split: str, transcript_manifests_dir: Path | None = None
) -> None:
    """Write a local CSV and copy its matching prompt metadata unchanged."""
    stem = f"librispeech_{split}_1k_min6s"
    rows = local_audio_rows(manifests_dir / f"{stem}.csv", data_dir)
    boundaries = manifests_dir / f"{stem}_prompt_boundaries_3s.jsonl"
    check_boundaries(boundaries, rows)
    output = output_dir / f"{stem}.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["fileid", "path", "num_frames", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(boundaries, output_dir / boundaries.name)
    if transcript_manifests_dir is not None:
        prepare_transcripts(transcript_manifests_dir / f"{stem}_asr.jsonl", output_dir / f"{stem}_asr.jsonl", rows)
    print(f"{output}: {len(rows)} utterances and matching prompt boundaries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Local LibriSpeech/ root with dev-clean/ and test-clean/."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Write local audio CSVs and prompt JSONLs here."
    )
    parser.add_argument(
        "--manifests-dir", type=Path, default=DEFAULT_MANIFESTS_DIR, help="Bundled relative CSVs and prompt JSONLs."
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=SPLITS, help="Splits to prepare (default: both)."
    )
    parser.add_argument(
        "--transcript-manifests-dir",
        type=Path,
        help="Full *_asr.jsonl transcripts for TextLM generation; their IDs must match the fixed selections.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    manifests_dir = args.manifests_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == manifests_dir:
        parser.error("Choose an output directory separate from the bundled manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        prepare_split(data_dir, manifests_dir, output_dir, split, args.transcript_manifests_dir)


if __name__ == "__main__":
    main()
