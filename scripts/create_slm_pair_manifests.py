# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate SLM pair manifests from existing ZeroSpeech / StoryCloze benchmark data.

Reads the original per-file manifests and gold labels, then outputs one CSV per
benchmark with columns ``fileid_correct,fileid_incorrect,path_correct,path_incorrect``
— the format expected by the SpidrOpt recipe for cached SLM evaluation.

Usage:
    python scripts/create_slm_pair_manifests.py \\
        --manifests-dir /datasets/metaspeechssl/shared/assets/manifests/sLM21-dataset \\
        --audio-root   /datasets/metaspeechssl/shared/data/zr-data/datasets \\
        --output-dir   /path/to/pair_manifests \\
        --benchmarks   tSC sSC lexical_dev_invocab syntactic_dev

See PAIR_MANIFEST_FORMAT.md for the output schema.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from gslm2.lm.eval.pairing import (
    STORYCLOZE_BENCHMARKS,
    ZR21_BENCHMARKS,
    ZR21_GOLD_SUBPATH,
    ZR21_GROUP_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Subdir under --audio-root that contains the per-benchmark gold dirs.
SLM21_DATASET_DIR = "sLM21-dataset"


# ---------------------------------------------------------------------------
# StoryCloze (tSC / sSC)
# ---------------------------------------------------------------------------


def _pairs_storycloze(manifest_path: Path) -> pl.DataFrame:
    """Build pairs for StoryCloze benchmarks (tSC / sSC).

    Fileid pattern: ``{story_id}_{uuid}_{correct|incorrect}``
    Pairs share the same UUID.
    """
    df = pl.read_csv(manifest_path)

    # Parse UUID and label from fileid
    df = df.with_columns(
        pl.col("fileid").str.split("_").list.last().alias("label"),
        pl.col("fileid").str.replace(r"^[^_]+_", "").str.replace(r"_(correct|incorrect)$", "").alias("uuid"),
    )

    correct = df.filter(pl.col("label") == "correct").select(
        pl.col("uuid"),
        pl.col("fileid").alias("fileid_correct"),
        pl.col("path").alias("path_correct"),
    )
    incorrect = df.filter(pl.col("label") == "incorrect").select(
        pl.col("uuid"),
        pl.col("fileid").alias("fileid_incorrect"),
        pl.col("path").alias("path_incorrect"),
    )

    pairs = correct.join(incorrect, on="uuid", how="inner").select(
        "fileid_correct", "fileid_incorrect", "path_correct", "path_incorrect"
    )
    return pairs


# ---------------------------------------------------------------------------
# ZR21 (lexical / syntactic)
# ---------------------------------------------------------------------------


def _pairs_zr21(
    manifest_path: Path,
    gold_csv_path: Path,
    *,
    group_cols: list[str],
) -> pl.DataFrame:
    """Build pairs for ZR21 benchmarks (lexical / syntactic).

    Uses ``gold.csv`` to determine correct/incorrect within each group.
    ``group_cols`` is ``["id", "voice"]`` for lexical, or
    ``["id", "voice", "subtype"]`` for syntactic.
    """
    manifest = pl.read_csv(manifest_path)
    gold = pl.read_csv(gold_csv_path)

    # Keep only filenames present in the manifest (handles invocab/oov splits)
    manifest_fileids = set(manifest["fileid"].to_list())
    gold = gold.filter(pl.col("filename").is_in(manifest_fileids))

    # Build a filename → path lookup from the manifest
    path_lookup = dict(zip(manifest["fileid"].to_list(), manifest["path"].to_list(), strict=True))
    gold = gold.with_columns(pl.col("filename").replace_strict(path_lookup).alias("path"))

    correct = gold.filter(pl.col("correct") == 1).select(
        [*group_cols, pl.col("filename").alias("fileid_correct"), pl.col("path").alias("path_correct")]
    )
    incorrect = gold.filter(pl.col("correct") == 0).select(
        [*group_cols, pl.col("filename").alias("fileid_incorrect"), pl.col("path").alias("path_incorrect")]
    )

    pairs = correct.join(incorrect, on=group_cols, how="inner").select(
        "fileid_correct", "fileid_incorrect", "path_correct", "path_incorrect"
    )
    return pairs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_pairs(
    benchmark: str,
    manifests_dir: Path,
    audio_root: Path,
) -> pl.DataFrame:
    """Route to the correct pairing function based on benchmark name."""
    manifest_path = manifests_dir / f"{benchmark}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if benchmark in STORYCLOZE_BENCHMARKS:
        return _pairs_storycloze(manifest_path)

    if benchmark not in ZR21_GROUP_COLS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Known: {sorted(STORYCLOZE_BENCHMARKS | set(ZR21_BENCHMARKS))}"
        )

    gold_csv_path = audio_root / SLM21_DATASET_DIR / ZR21_GOLD_SUBPATH[benchmark] / "gold.csv"
    if not gold_csv_path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {gold_csv_path}")

    return _pairs_zr21(manifest_path, gold_csv_path, group_cols=list(ZR21_GROUP_COLS[benchmark]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SLM pair manifests from SLM benchmark data.")
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        required=True,
        help="Directory with per-file manifests (e.g. tSC.csv, lexical_dev_invocab.csv).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root of ZR benchmark audio data (contains sLM21-dataset/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write pair manifests.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="Benchmark names to process (e.g. tSC sSC lexical_dev_invocab syntactic_dev).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="If set, write only this many randomly sampled pairs per benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --max-pairs sampling.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in args.benchmarks:
        logger.info("Processing benchmark: %s", benchmark)
        pairs = build_pairs(benchmark, args.manifests_dir, args.audio_root)

        if args.max_pairs is not None and len(pairs) > args.max_pairs:
            pairs = pairs.sample(n=args.max_pairs, seed=args.seed)

        output_path = args.output_dir / f"{benchmark}.csv"
        pairs.write_csv(output_path)
        logger.info("  Wrote %d pairs → %s", len(pairs), output_path)


if __name__ == "__main__":
    main()
