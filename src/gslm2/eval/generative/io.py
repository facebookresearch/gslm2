# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Atomic artifact writers and CSV/JSONL readers for generative evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence


def read_rows(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Read object rows with their source line numbers."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            for line_number, row in enumerate(csv.DictReader(source), start=2):
                yield line_number, dict(row)
        return
    if path.suffix != ".jsonl":
        raise ValueError(f"manifest must be .csv or .jsonl: {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, row


def read_jsonl(path: str | Path, *, allow_empty: bool = False) -> list[dict[str, object]]:
    """Read a JSONL artifact, rejecting empty inputs by default."""
    source = Path(path)
    if source.suffix != ".jsonl":
        raise ValueError(f"expected a JSONL artifact: {source}")
    rows = [row for _, row in read_rows(source)]
    if not rows and not allow_empty:
        raise ValueError(f"manifest is empty: {source}")
    return rows


def write_json(path: str | Path, value: Mapping[str, object]) -> Path:
    """Atomically write a JSON mapping, serializing Paths as strings."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    payload = json.dumps(value, default=str, allow_nan=False, indent=2, sort_keys=True)
    temporary.write_text(f"{payload}\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]], *, allow_empty: bool = False) -> Path:
    """Atomically write JSONL, allowing empty distributed shards explicitly."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        count = 0
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(f"{json.dumps(row, allow_nan=False)}\n")
                count += 1
        if not count and not allow_empty:
            raise ValueError("cannot write an empty manifest")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: tuple[str, ...]) -> Path:
    """Atomically write selected columns from candidate records."""
    if not rows:
        raise ValueError("cannot write an empty candidate manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({name: row[name] for name in fieldnames} for row in rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def file_digest(path: Path) -> str:
    """Hash file contents for input and transcript cache validation."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
