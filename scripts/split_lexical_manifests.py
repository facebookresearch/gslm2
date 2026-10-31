# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Split lexical audio manifests into invocab and OOV using sLM21 gold labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SPLITS = ("dev", "test")
CATEGORIES = ("invocab", "oov")
COLUMNS = ("fileid", "path", "num_frames")


def split_manifest(slm21_root: Path, manifests_dir: Path, output_dir: Path, split: str) -> None:
    """Classify pairs by the correct word's frequency and retain both members."""
    gold_path = slm21_root / "lexical" / split / "gold.csv"
    with gold_path.open(encoding="utf-8") as source:
        gold = list(csv.DictReader(source))

    pair_classes = {
        (row["id"], row["voice"]): "oov" if float(row["frequency"]) == 0 else "invocab"
        for row in gold
        if row["correct"] == "1"
    }
    with (manifests_dir / f"lexical_{split}.csv").open(encoding="utf-8") as source:
        manifest = list(csv.DictReader(source))

    for category in CATEGORIES:
        fileids = {row["filename"] for row in gold if pair_classes[(row["id"], row["voice"])] == category}
        selected = [row for row in manifest if row["fileid"] in fileids]
        if {row["fileid"] for row in selected} != fileids:
            raise ValueError(f"Missing lexical {split} {category} audio files")
        if len(selected) != len(fileids):
            raise ValueError(f"Duplicate lexical {split} {category} file IDs")

        output = output_dir / f"lexical_{split}_{category}.csv"
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
        print(f"{output}: {len(selected)} audio files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slm21-root",
        type=Path,
        required=True,
        help="sLM21-dataset directory containing lexical/{dev,test}/gold.csv.",
    )
    parser.add_argument(
        "--manifests-dir", type=Path, required=True, help="Directory containing lexical_dev.csv and lexical_test.csv."
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: --manifests-dir).")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=SPLITS, help="Splits to process (default: dev test)."
    )
    args = parser.parse_args()
    output_dir = (args.output_dir or args.manifests_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        split_manifest(args.slm21_root.resolve(), args.manifests_dir.resolve(), output_dir, split)


if __name__ == "__main__":
    main()
