# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Create local speech CSVs and text pair JSONLs from an LTswap release."""

from __future__ import annotations

import argparse
import csv
import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from typing import TextIO

TASKS = ("agrswap", "inflswap", "wordswap")


class PairSide(TypedDict):
    fileid: str
    text: str
    audio: str
    num_frames: int


class PairRecord(TypedDict):
    pair_id: str
    task: str
    freq_bin: str
    correct: PairSide
    incorrect: PairSide


@dataclass(frozen=True)
class AudioSample:
    fileid: str
    path: Path
    num_frames: int
    text: str


def read_sample(side: PairSide, *, pair_id: str, label: str, data_dir: Path) -> AudioSample:
    """Resolve a released sample and validate the fields used by LM readers."""
    fileid = side["fileid"]
    relative_audio = Path(side["audio"])
    audio = (data_dir / relative_audio).resolve()
    if relative_audio.is_absolute() or not audio.is_relative_to(data_dir):
        raise ValueError(f"Audio path must be relative to the dataset: {side['audio']}")
    if fileid != f"{pair_id}_{label}" or audio.stem != fileid or audio.suffix != ".wav":
        raise ValueError(f"Unexpected file ID or WAV filename: {fileid}")
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if type(side["num_frames"]) is not int or side["num_frames"] <= 0:
        raise ValueError(f"num_frames must be a positive integer: {fileid}")
    if not isinstance(side["text"], str):
        raise TypeError(f"text must be a string: {fileid}")
    return AudioSample(fileid, audio, side["num_frames"], side["text"])


def convert_task(data_dir: Path, output_dir: Path, task: str) -> int:
    """Write both manifest formats, grouped by the released frequency labels."""
    outputs: dict[str, tuple[TextIO, TextIO]] = {}
    seen_fileids: set[str] = set()
    with ExitStack() as stack, (data_dir / "pairs" / f"{task}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            pair = cast("PairRecord", json.loads(line))
            if pair["task"] != task:
                raise ValueError(f"Unexpected task in {task}.jsonl: {pair['task']}")
            freq_bin = pair["freq_bin"]
            if re.fullmatch(r"\d+-(?:\d+|inf)", freq_bin) is None:
                raise ValueError(f"Unexpected frequency bin: {freq_bin!r}")
            samples = [
                read_sample(side, pair_id=pair["pair_id"], label=label, data_dir=data_dir)
                for label, side in (("correct", pair["correct"]), ("incorrect", pair["incorrect"]))
            ]
            for sample in samples:
                if sample.fileid in seen_fileids:
                    raise ValueError(f"Duplicate file ID: {sample.fileid}")
                seen_fileids.add(sample.fileid)

            if freq_bin not in outputs:
                name = f"{task}_freq_{freq_bin}"
                raw = stack.enter_context((output_dir / "raw" / f"{name}.csv").open("w", newline="", encoding="utf-8"))
                text = stack.enter_context((output_dir / "text" / f"{name}.jsonl").open("w", encoding="utf-8"))
                csv.writer(raw).writerow(["fileid", "path", "num_frames"])
                outputs[freq_bin] = (raw, text)
            raw, text = outputs[freq_bin]
            csv.writer(raw).writerows((sample.fileid, str(sample.path), sample.num_frames) for sample in samples)
            text_pair = {
                "pair_id": pair["pair_id"],
                "task": task,
                "freq_bin": freq_bin,
                "positive": {"file_name": samples[0].fileid, "text": samples[0].text},
                "negative": {"file_name": samples[1].fileid, "text": samples[1].text},
            }
            text.write(json.dumps(text_pair, ensure_ascii=False) + "\n")
    if not outputs:
        raise ValueError(f"No pairs found in {task}.jsonl")
    return len(outputs)


def create_manifests(data_dir: Path, output_dir: Path, tasks: tuple[str, ...] = TASKS) -> None:
    """Build manifests for the selected tasks in one downloaded dataset variant."""
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown LTswap task: {task}")
        if not (data_dir / "pairs" / f"{task}.jsonl").is_file():
            raise FileNotFoundError(data_dir / "pairs" / f"{task}.jsonl")
    for kind in ("raw", "text"):
        (output_dir / kind).mkdir(parents=True, exist_ok=True)
    for task in tasks:
        count = convert_task(data_dir, output_dir, task)
        print(f"{task}: wrote {count} bins under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Dataset variant directory containing pairs/ and audio/."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Write speech CSVs under raw/ and text pair JSONLs under text/."
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=TASKS, help="Tasks to convert (default: all three)."
    )
    args = parser.parse_args()
    create_manifests(args.data_dir, args.output_dir, tuple(args.tasks))


if __name__ == "__main__":
    main()
