# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build LibriSpeech encoder-evaluation labels from forced alignments.

Two steps, run in order:

1. ``corpus``: collect a LibriSpeech split into a flat Montreal Forced
   Aligner (MFA) corpus directory (``<fileid>.wav`` + ``<fileid>.lab``).
2. Align with MFA, e.g.
   ``mfa align -j 8 --clean <corpus> english_us_arpa english_us_arpa <grids>``.
3. ``labels``: convert the MFA ``.TextGrid`` output into the JSONL label
   files consumed by the encoder evals: word segments for MAP
   (``words.jsonl``) and 10 ms frame-level phones for codebook quality
   (``frames.jsonl``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

FRAME_SHIFT = 0.01
SILENCE_MARKS = {"", "sil", "sp", "spn", "<unk>"}

# Reference word-duration window (seconds): keep words within [MIN, MAX).
MIN_WORD_DURATION = 0.07
MAX_WORD_DURATION = 0.96
_SHORT_FORMAT_TIMES = 2


def _parse_textgrid_long(lines: list[str]) -> dict[str, list[tuple[float, float, str]]]:
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    current: list[tuple[float, float, str]] | None = None
    in_intervals = False
    xmin = xmax = 0.0
    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
            current = tiers.setdefault(name, [])
            in_intervals = False
        elif line.startswith("intervals: size ="):
            in_intervals = True
        elif in_intervals and current is not None:
            if line.startswith("xmin = "):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax = "):
                xmax = float(line.split("=", 1)[1])
            elif line.startswith("text = "):
                mark = line.split("=", 1)[1].strip().strip('"').replace('""', '"')
                current.append((xmin, xmax, mark))
    return tiers


def _parse_textgrid_short(lines: list[str]) -> dict[str, list[tuple[float, float, str]]]:
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    # Line 2 holds the tier count ("TextGrid" xmin xmax ntier); each tier
    # block is a header line, an interval-count line, then one row per interval.
    n_tiers = int(lines[1].split()[3])
    pos = 2
    for _ in range(n_tiers):
        header = lines[pos].strip().split()
        name = header[1].strip('"')
        pos += 1
        n_intervals = int(lines[pos].strip().split()[0])
        pos += 1
        intervals = []
        for _ in range(n_intervals):
            parts = lines[pos].strip().split(maxsplit=_SHORT_FORMAT_TIMES)
            xmin, xmax = float(parts[0]), float(parts[1])
            mark = (
                parts[_SHORT_FORMAT_TIMES].strip().strip('"').replace('""', '"')
                if len(parts) > _SHORT_FORMAT_TIMES
                else ""
            )
            intervals.append((xmin, xmax, mark))
            pos += 1
        tiers[name] = intervals
    return tiers


def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Parse a Praat TextGrid into ``{tier_name: [(xmin, xmax, mark)]}``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty TextGrid: {path}")
    if lines[0].startswith("File type ="):
        return _parse_textgrid_long(lines)
    return _parse_textgrid_short(lines)


def _find_tier(tiers: dict[str, list], *names: str) -> list:
    lowered = {name.lower(): intervals for name, intervals in tiers.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"No tier {names} in tiers {sorted(tiers)}")


def _strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone).upper()


def _word_phones(word: tuple[float, float, str], phones: list[tuple[float, float, str]]) -> list[str]:
    xmin, xmax, _ = word
    return [mark for pmin, pmax, mark in phones if mark.lower() not in SILENCE_MARKS and xmin <= pmin and pmax <= xmax]


def _frame_labels(phones: list[tuple[float, float, str]], duration: float) -> list[str]:
    n_frames = round(duration / FRAME_SHIFT)
    labels = []
    for i in range(n_frames):
        center = (i + 0.5) * FRAME_SHIFT
        mark = next((m for xmin, xmax, m in phones if xmin <= center < xmax), "SIL")
        labels.append("SIL" if mark.lower() in SILENCE_MARKS else _strip_stress(mark))
    return labels


def load_manifest_paths(manifest: Path) -> dict[str, str]:
    """Map fileid to audio path for CSV (fileid,path,...) or TSV (root + fileid\\tnum) manifests."""
    if manifest.suffix == ".csv":
        with manifest.open(newline="") as f:
            return {row["fileid"]: row["path"] for row in csv.DictReader(f)}
    if manifest.suffix != ".tsv":
        raise ValueError(f"Only .csv and .tsv manifests are supported: {manifest}")
    lines = manifest.read_text().splitlines()
    root = Path(lines[0].strip())
    rows = (line.split("\t") for line in lines[1:] if line.strip())
    return {Path(relpath).stem: str(root / relpath) for relpath, _ in rows}


def _has_oov(words: list[tuple[float, float, str]], phones: list[tuple[float, float, str]]) -> bool:
    """Whether the utterance contains an out-of-vocabulary word.

    Older MFA marks OOV words ``<unk>``; MFA 3.x emits ``spn`` phones for
    them instead. Either way the alignment is unreliable.
    """
    return any(mark == "<unk>" for _, _, mark in words) or any(mark.lower() == "spn" for _, _, mark in phones)


def convert_grid(grid: Path, audio_path: str) -> tuple[list[dict], dict, bool]:
    """Convert one TextGrid into word records, one frame-label record, and an OOV flag."""
    tiers = parse_textgrid(grid)
    words = _find_tier(tiers, "words", "word")
    phones = _find_tier(tiers, "phones", "phone")
    fileid = grid.stem
    speaker = fileid.split("-")[0] if "-" in fileid else fileid
    word_records = []
    for word in words:
        if word[2].lower() in SILENCE_MARKS:
            continue
        transcription = ",".join(_word_phones(word, phones))
        if not transcription:
            continue
        word_records.append(
            {
                "fileid": fileid,
                "speaker": speaker,
                "onset": round(word[0], 4),
                "offset": round(word[1], 4),
                "transcription": transcription,
                "spelling": word[2],
            }
        )
    duration = max(xmax for tier in tiers.values() for _, xmax, _ in tier)
    frame_record = {"audio": audio_path, "phones": " ".join(_frame_labels(phones, duration))}
    return word_records, frame_record, _has_oov(words, phones)


def cmd_labels(args: argparse.Namespace) -> None:
    grid_dir = Path(args.textgrid_dir)
    paths = load_manifest_paths(Path(args.manifest))
    out_words = Path(args.out_words)
    out_frames = Path(args.out_frames)
    out_words.parent.mkdir(parents=True, exist_ok=True)
    out_frames.parent.mkdir(parents=True, exist_ok=True)
    grids = sorted(grid_dir.glob("*.TextGrid"))
    if not grids:
        raise ValueError(f"No .TextGrid files in {grid_dir}")
    all_words: list[dict] = []
    all_frames: list[dict] = []
    for grid in tqdm(grids, desc="Converting TextGrids"):
        if grid.stem not in paths:
            tqdm.write(f"Skipping {grid.name}: fileid not in manifest")
            continue
        word_records, frame_record, has_oov = convert_grid(grid, paths[grid.stem])
        if not has_oov:
            all_words.extend(word_records)
        all_frames.append(frame_record)
    kept_spellings = {s for s, n in Counter(r["spelling"] for r in all_words).items() if n > 1}
    all_words = [r for r in all_words if r["spelling"] in kept_spellings]
    all_words = [r for r in all_words if MIN_WORD_DURATION <= r["offset"] - r["onset"] < MAX_WORD_DURATION]
    kept_trans = {t for t, n in Counter(r["transcription"] for r in all_words).items() if n > 1}
    all_words = [r for r in all_words if r["transcription"] in kept_trans]
    with out_words.open("w") as f_words, out_frames.open("w") as f_frames:
        for record in all_words:
            written = {k: v for k, v in record.items() if k != "spelling"}
            f_words.write(json.dumps(written) + "\n")
        for record in all_frames:
            f_frames.write(json.dumps(record) + "\n")
    print(f"Wrote {len(all_words)} word records to {out_words} and frame labels to {out_frames}")


def _prepare_one(transcript: tuple[str, str], audio: Path, out_dir: Path, ffmpeg: str, *, convert: bool) -> str | None:
    fileid, text = transcript
    if convert:
        target = out_dir / f"{fileid}.wav"
        if not target.exists() or target.stat().st_size == 0:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                check=True,
            )
    else:
        target = out_dir / audio.name
        if not target.exists():
            target.symlink_to(audio.resolve())
    (out_dir / f"{fileid}.lab").write_text(text + "\n")
    return None


def cmd_corpus(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if args.convert and ffmpeg is None:
        raise ValueError("ffmpeg not found; pass --ffmpeg or use --no-convert")
    pairs = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            fileid, _, text = line.partition(" ")
            audio = trans.parent / f"{fileid}.flac"
            if not audio.exists():
                print(f"Skipping {fileid}: missing {audio}")
                continue
            pairs.append(((fileid, text.strip()), audio))
    if not pairs:
        raise ValueError(f"No transcripts found under {split_dir}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            tqdm(
                pool.map(lambda p: _prepare_one(p[0], p[1], out_dir, ffmpeg, convert=args.convert), pairs),
                total=len(pairs),
                desc="Preparing MFA corpus",
            )
        )
    print(f"Prepared {len(pairs)} utterances in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p_corpus = sub.add_parser("corpus", help="Collect a LibriSpeech split into an MFA corpus directory")
    p_corpus.add_argument("--split-dir", type=Path, required=True, help="LibriSpeech split dir (e.g. dev-clean/)")
    p_corpus.add_argument("--out-dir", type=Path, required=True, help="Output MFA corpus directory")
    p_corpus.add_argument("--workers", type=int, default=8, help="Concurrent ffmpeg conversions")
    p_corpus.add_argument("--ffmpeg", type=str, default=None, help="ffmpeg executable")
    p_corpus.add_argument(
        "--no-convert", dest="convert", action="store_false", help="Symlink audio instead of WAV conversion"
    )
    p_corpus.set_defaults(func=cmd_corpus)
    p_labels = sub.add_parser("labels", help="Convert MFA TextGrids to MAP/codebook JSONL labels")
    p_labels.add_argument("--textgrid-dir", type=Path, required=True, help="Directory of MFA .TextGrid output")
    p_labels.add_argument("--manifest", type=Path, required=True, help="Audio manifest CSV/TSV mapping fileid to path")
    p_labels.add_argument("--out-words", type=Path, required=True, help="Output words.jsonl for MAP")
    p_labels.add_argument("--out-frames", type=Path, required=True, help="Output frame-phones JSONL for codebook")
    p_labels.set_defaults(func=cmd_labels)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
