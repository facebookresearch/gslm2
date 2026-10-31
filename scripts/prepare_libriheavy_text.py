# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce the LibriHeavy-long cases_and_punc text and tokenizer layout."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LOGGER = logging.getLogger(__name__)
SUBSETS = ("small", "medium", "large")
SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>")
# Match libriheavy-long/scripts/extract_and_normalize_transcript.py.
CLEAN_TABLE = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")  # noqa: RUF001
SEGMENT_PATTERN = re.compile(r"^([^/]+)/([^/]+)/([^/]+)/([^/]+_\d+)$")


def clean_text(text: str) -> str:
    """Keep case and punctuation with the historical character substitutions."""
    return text.translate(CLEAN_TABLE).strip()


def extract_subset(manifest: Path, output_dir: Path, subset: str) -> None:
    """Sort on disk, matching the original Kaldi export and extract_text.py."""
    destination = output_dir / f"libriheavy_text_{subset}"
    csv_path = output_dir / f"libriheavy_text_{subset}.csv"
    if destination.exists() or csv_path.exists():
        raise FileExistsError(f"Output already exists: {destination} or {csv_path}")

    # Stage outputs and the sorting database on the output filesystem.
    with tempfile.TemporaryDirectory(prefix=f".libriheavy-{subset}-", dir=output_dir) as temporary:
        staging = Path(temporary)
        staged_text = staging / destination.name
        staged_text.mkdir()
        connection = sqlite3.connect(staging / "segments.sqlite")
        try:
            connection.execute(
                "CREATE TABLE segments (ordinal INTEGER PRIMARY KEY, segment_id TEXT UNIQUE, "
                "collection TEXT, recording TEXT, start REAL, text TEXT)"
            )
            with gzip.open(manifest, "rt", encoding="utf-8") as source:
                for count, line in enumerate(source, 1):
                    cut = json.loads(line)
                    segment_id = cut["id"]
                    match = SEGMENT_PATTERN.fullmatch(segment_id)
                    if match is None or match[1] != subset or match[3] in {".", ".."}:
                        raise ValueError(f"Invalid {subset} segment ID in {manifest}: {segment_id}")
                    text = clean_text(cut["supervisions"][0]["custom"]["texts"][0])
                    collection = match[3]
                    recording = match[4].rsplit("_", 1)[0]
                    # The Kaldi export truncates start times to milliseconds.
                    start = math.floor(1000 * cut["start"]) / 1000
                    # A duplicate ID replaces its contents but keeps its first position,
                    # as in the original export's Python dictionaries.
                    connection.execute(
                        "INSERT INTO segments (segment_id, collection, recording, start, text) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(segment_id) DO UPDATE SET "
                        "start=excluded.start, text=excluded.text",
                        (segment_id, collection, recording, start, text),
                    )
                    if count % 10_000 == 0:
                        connection.commit()
                    if count % 100_000 == 0:
                        LOGGER.info("%s: read %s cuts", subset, count)
            connection.commit()
            connection.execute("CREATE INDEX collection_order ON segments (collection, recording, start, ordinal)")

            staged_csv = staging / csv_path.name
            written = 0
            collections = connection.execute(
                "SELECT collection FROM segments WHERE text != '' GROUP BY collection ORDER BY MIN(ordinal)"
            )
            with staged_csv.open("w", encoding="utf-8", newline="") as manifest_file:
                writer = csv.writer(manifest_file, lineterminator="\n")
                writer.writerow(["file"])
                for (collection,) in collections:
                    rows = connection.execute(
                        "SELECT text FROM segments WHERE collection=? AND text != '' "
                        "ORDER BY recording, start, ordinal",
                        (collection,),
                    )
                    first = next(rows, None)
                    # Empty Kaldi transcripts did not match the original extraction regex.
                    if first is None:
                        continue
                    text_dir = staged_text / collection
                    text_dir.mkdir()
                    with (text_dir / "text.txt").open("w", encoding="utf-8") as text_file:
                        text_file.write(first[0])
                        for (text,) in rows:
                            text_file.write(" " + text)
                    writer.writerow([str(destination / collection / "text.txt")])
                    written += 1
        finally:
            connection.close()
        staged_text.rename(destination)
        staged_csv.rename(csv_path)
    LOGGER.info("%s: wrote %s collection files and %s", subset, written, csv_path)


def extract_text(manifest_dir: Path, output_dir: Path, subsets: Sequence[str]) -> None:
    """Prepare one text directory and CSV per selected Libri-Light subset."""
    output_dir = output_dir.resolve()
    for subset in subsets:
        manifest = manifest_dir / f"libriheavy_cuts_{subset}.jsonl.gz"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for name in (f"libriheavy_text_{subset}", f"libriheavy_text_{subset}.csv"):
            if (output_dir / name).exists():
                raise FileExistsError(output_dir / name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        extract_subset(manifest_dir / f"libriheavy_cuts_{subset}.jsonl.gz", output_dir, subset)


def read_texts(manifests: Sequence[Path]) -> Iterator[str]:
    """Read whole collection texts in CSV order for BPE training."""
    for manifest in manifests:
        with manifest.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ["file"]:
                raise ValueError(f"Expected a single 'file' column in {manifest}")
            for row in reader:
                yield Path(row["file"]).read_text(encoding="utf-8")


def train_tokenizer(text_dir: Path, train_data: str, vocab_size: int) -> Path:
    """Use the historical custom BPE recipe, including its special token IDs."""
    # Extraction itself only needs the Python standard library.
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing

    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    subsets = SUBSETS[: SUBSETS.index(train_data.removeprefix("lh-")) + 1]
    manifests = [text_dir / f"libriheavy_text_{subset}.csv" for subset in subsets]
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    tokenizers_dir = text_dir.resolve() / "tokenizers"
    requested_dir = tokenizers_dir / f"{train_data}_bpe{vocab_size}"
    if requested_dir.exists():
        raise FileExistsError(requested_dir)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKENS))
    LOGGER.info("Training %s BPE on %s", train_data, ", ".join(subsets))
    tokenizer.train_from_iterator(read_texts(manifests), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(single="</s> $A", special_tokens=[("</s>", 2)])
    # A target of 1 retains the observed byte alphabet and learns no merges.
    # Name these artifacts by their actual size (174/178/191 for this release).
    output_dir = tokenizers_dir / f"{train_data}_bpe{tokenizer.get_vocab_size()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    LOGGER.info("Saved %s", output_dir / "tokenizer.json")
    return output_dir


def main() -> None:
    """Extract transcripts or train a tokenizer without hardcoded data paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Write collection texts and CSVs from long-cut manifests")
    extract.add_argument("--manifest-dir", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    tokenizer = commands.add_parser("train-tokenizer", help="Train the original custom byte-level BPE recipe")
    tokenizer.add_argument("--text-dir", type=Path, required=True)
    tokenizer.add_argument("--train-data", choices=[f"lh-{subset}" for subset in SUBSETS], default="lh-large")
    tokenizer.add_argument("--vocab-size", type=int, default=4096, help="Use 1 for a byte alphabet with no merges")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "extract":
        extract_text(args.manifest_dir, args.output_dir, list(dict.fromkeys(args.subsets)))
    else:
        train_tokenizer(args.text_dir, args.train_data, args.vocab_size)


if __name__ == "__main__":
    main()
