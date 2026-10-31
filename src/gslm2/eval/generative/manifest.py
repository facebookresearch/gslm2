# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Manifest and prompt-boundary readers for candidate generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gslm2.eval.generative.io import read_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_REFERENCE_FEATURE_SECONDS = 0.02


def file_id(value: object) -> str:
    """Return the extension-free identifier from a manifest path field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest row needs a non-empty audio or file_name field")
    return Path(value).stem


def record_id(row: Mapping[str, object]) -> str:
    """Return a unique, filename-safe explicit ID or audio filename stem."""
    value = row.get("id", row.get("fileid"))
    row_id = value.strip() if isinstance(value, str) else file_id(row.get("file_name", row.get("audio")))
    if not row_id or row_id in {".", ".."} or "/" in row_id or "\\" in row_id:
        raise ValueError("manifest IDs must be non-empty filename components")
    return row_id


def index_by_file_id(rows: Iterable[Mapping[str, object]], source: str | Path) -> dict[str, Mapping[str, object]]:
    """Index rows by explicit ID or audio filename, rejecting duplicates."""
    indexed = {}
    for row in rows:
        row_id = record_id(row)
        if row_id in indexed:
            raise ValueError(f"duplicate id {row_id!r} in {source}")
        indexed[row_id] = row
    return indexed


class PromptBoundaries:
    """Word-aligned prompt metadata indexed by utterance id."""

    def __init__(self, path: str | Path) -> None:
        self.records = index_by_file_id(read_jsonl(path), path)

    def frames(self, row_id: str, feature_seconds: float) -> int:
        """Return the prompt boundary in feature frames."""
        record = self.records.get(row_id)
        if record is None:
            raise KeyError(f"missing prompt boundary for {row_id!r}")
        prompt_frames = record.get("prompt_frames")
        if prompt_frames is not None and feature_seconds == _REFERENCE_FEATURE_SECONDS:
            if not isinstance(prompt_frames, str | int | float):
                raise TypeError(f"{row_id} prompt_frames must be numeric")
            return int(prompt_frames)
        prompt_seconds = record["prompt_secs"]
        if not isinstance(prompt_seconds, str | int | float):
            raise TypeError(f"{row_id} prompt_secs must be numeric")
        return round(float(prompt_seconds) / feature_seconds)

    def prompt_word_count(self, row_id: str) -> int | None:
        """Return the number of words preceding the continuation."""
        record = self.records.get(row_id)
        if record is None:
            return None
        count = record.get("prompt_n_words")
        if count is None:
            return len(str(record.get("prompt_text", "")).split())
        if not isinstance(count, str | int):
            raise TypeError(f"{row_id} prompt_n_words must be an integer")
        return int(count)
