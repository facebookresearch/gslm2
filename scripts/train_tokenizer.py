#!/usr/bin/env python
"""Train a custom tokenizer on a text corpus for LM pretraining.

Supports two tokenizer types:
  - wordpiece: Retrain from bert-base-cased (for BERT MLM).
  - bpe: Retrain from gpt2 byte-level BPE (for GPT-2 causal LM).

The corpus can be a plain-text file (one line per utterance) or a JSON
manifest with a configurable text key.

Usage:
    python scripts/train_tokenizer.py \\
        --type wordpiece \\
        --train-file data/ego4d_train_metadata.json \\
        --text-key text \\
        --output-dir /checkpoint/.../tokenizers/bert_ego4d

    python scripts/train_tokenizer.py \\
        --type bpe \\
        --train-file data/ego4d_train_metadata.json \\
        --text-key text \\
        --vocab-size 52000 \\
        --output-dir /checkpoint/.../tokenizers/gpt2_ego4d
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000
_WORDPIECE_DEFAULTS = {"base": "bert-base-cased", "vocab_size": 30522}
_BPE_DEFAULTS = {"base": "gpt2", "vocab_size": 52000}


def _read_texts(paths: list[Path], text_key: str | None) -> list[str]:
    """Read text lines from plain-text or JSON manifest files."""
    texts: list[str] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open() as f:
                data = json.load(f)
            key = text_key or "text"
            if isinstance(data, list):
                texts.extend(entry[key] for entry in data if key in entry)
            elif isinstance(data, dict) and "data" in data:
                texts.extend(entry[key] for entry in data["data"] if key in entry)
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
            logger.info("  %s: %d texts (JSON, key=%r)", path.name, len(texts), key)
        elif suffix == ".jsonl":
            with path.open() as f:
                key = text_key or "text"
                for line in f:
                    entry = json.loads(line)
                    if key in entry:
                        texts.append(entry[key])
            logger.info("  %s: %d texts (JSONL, key=%r)", path.name, len(texts), key)
        else:
            with path.open() as f:
                new_lines = [stripped for line in f if (stripped := line.strip())]
            texts.extend(new_lines)
            logger.info("  %s: %d lines (plain text)", path.name, len(new_lines))
    return texts


def _batched_iterator(texts: list[str]) -> Iterator[list[str]]:
    for i in range(0, len(texts), _BATCH_SIZE):
        yield texts[i : i + _BATCH_SIZE]


def train_tokenizer(
    tokenizer_type: str,
    train_files: list[Path],
    output_dir: Path,
    vocab_size: int | None = None,
    text_key: str | None = None,
    val_files: list[Path] | None = None,
) -> None:
    defaults = _WORDPIECE_DEFAULTS if tokenizer_type == "wordpiece" else _BPE_DEFAULTS
    vocab_size = vocab_size or defaults["vocab_size"]
    base_name = defaults["base"]

    all_files = list(train_files)
    if val_files:
        all_files.extend(val_files)

    logger.info("Training %s tokenizer (base=%s, vocab_size=%d)", tokenizer_type, base_name, vocab_size)
    texts = _read_texts(all_files, text_key)
    logger.info("Total corpus: %d texts", len(texts))

    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        _batched_iterator(texts),
        vocab_size=vocab_size,
    )

    if tokenizer_type == "bpe":
        new_tokenizer.pad_token = new_tokenizer.eos_token

    output_dir.mkdir(parents=True, exist_ok=True)
    new_tokenizer.save_pretrained(output_dir)

    logger.info("Saved to %s", output_dir)
    logger.info("Vocab size: %d", len(new_tokenizer))
    logger.info("Special tokens: %s", new_tokenizer.all_special_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for LM pretraining")
    parser.add_argument("--type", choices=["wordpiece", "bpe"], required=True, help="Tokenizer type")
    parser.add_argument("--train-file", type=Path, required=True, action="append", dest="train_files",
                        help="Training corpus (plain text or JSON manifest). Can be repeated.")
    parser.add_argument("--val-file", type=Path, action="append", dest="val_files", default=None,
                        help="Optional validation corpus. Can be repeated.")
    parser.add_argument("--text-key", type=str, default=None,
                        help="JSON key for text field (default: 'text')")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Target vocab size (default: 30522 for wordpiece, 52000 for bpe)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save tokenizer")
    args = parser.parse_args()

    train_tokenizer(
        tokenizer_type=args.type,
        train_files=args.train_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        text_key=args.text_key,
        val_files=args.val_files,
    )


if __name__ == "__main__":
    main()
