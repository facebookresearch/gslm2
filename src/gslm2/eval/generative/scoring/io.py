# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Manifest readers and result writers for generative evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from gslm2.eval.generative.io import read_rows as _rows
from gslm2.eval.generative.io import write_json as write_json
from gslm2.eval.generative.io import write_jsonl as _write_jsonl
from gslm2.eval.generative.scoring.schema import (
    EvaluationRecord,
    Reference,
    SpeechCandidate,
    SpeechTranscription,
    TextCandidate,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence


def _required_string(row: Mapping[str, object], names: Sequence[str], location: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{location} needs one of these non-empty string fields: {', '.join(names)}")


def _string(row: Mapping[str, object], names: Sequence[str], location: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str):
            return value.strip()
    raise ValueError(f"{location} needs one of these string fields: {', '.join(names)}")


def _id(row: Mapping[str, object], location: str) -> str:
    for name in ("id", "fileid"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _required_string(row, ("file_name", "audio"), location)
    return Path(value).stem


def _index_unique[T](records: Iterable[T], *, source: Path) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for record in records:
        record_id = getattr(record, "id", None)
        if not isinstance(record_id, str):
            raise TypeError(f"record from {source} has no string id")
        if record_id in indexed:
            raise ValueError(f"duplicate id {record_id!r} in {source}")
        indexed[record_id] = record
    if not indexed:
        raise ValueError(f"manifest is empty: {source}")
    return indexed


def _boolean(value: object, *, location: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise TypeError(f"{location} has a non-boolean aligned field")


def read_references(path: str | Path) -> dict[str, Reference]:
    """Read references, accepting the field names used by the corrected dev evaluation."""
    manifest_path = Path(path)

    def records() -> Iterator[Reference]:
        for line_number, row in _rows(manifest_path):
            location = f"{manifest_path}:{line_number}"
            boundary = row.get("prompt_end_seconds", row.get("prompt_secs"))
            if boundary in (None, ""):
                prompt_end_seconds = None
            elif isinstance(boundary, str | int | float):
                prompt_end_seconds = float(boundary)
            else:
                raise TypeError(f"{location} has a non-numeric prompt boundary")
            if prompt_end_seconds is not None and (not math.isfinite(prompt_end_seconds) or prompt_end_seconds <= 0.0):
                raise ValueError(f"{location} has an invalid prompt boundary")
            continuation = row.get("reference_continuation", row.get("continuation_text"))
            if continuation is None:
                words = row.get("words")
                prompt_word_count_value = row.get("prompt_n_words")
                if not isinstance(words, list) or not isinstance(prompt_word_count_value, str | int):
                    raise ValueError(f"{location} needs reference_continuation or words plus prompt_n_words")
                prompt_word_count = int(prompt_word_count_value)
                if not 0 < prompt_word_count < len(words):
                    raise ValueError(f"{location} has an invalid prompt word count")
                continuation = " ".join(
                    str(word["text"])
                    for word in words[prompt_word_count:]
                    if isinstance(word, dict) and word.get("text")
                )
            if not isinstance(continuation, str) or not continuation.strip():
                raise ValueError(f"{location} has an empty reference continuation")
            yield Reference(
                id=_id(row, location),
                prompt_text=_required_string(row, ("prompt_text",), location),
                continuation_text=continuation.strip(),
                prompt_end_seconds=prompt_end_seconds,
                aligned=_boolean(row.get("aligned", True), location=location),
            )

    return _index_unique(records(), source=manifest_path)


def read_speech_candidates(path: str | Path, audio_root: str | Path | None = None) -> list[SpeechCandidate]:
    """Read synthesized-audio candidates in JSONL or CSV format."""
    manifest_path = Path(path)
    root = Path(audio_root) if audio_root is not None else manifest_path.parent
    candidates = []
    for line_number, row in _rows(manifest_path):
        location = f"{manifest_path}:{line_number}"
        raw_path = _required_string(row, ("audio_path", "path", "audio"), location)
        audio_path = Path(raw_path)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        candidates.append(SpeechCandidate(id=_id(row, location), audio_path=audio_path.resolve()))
    return list(_index_unique(candidates, source=manifest_path).values())


def read_text_candidates(path: str | Path) -> list[TextCandidate]:
    """Read generated prompt/continuation pairs in JSONL or CSV format."""
    manifest_path = Path(path)
    candidates = []
    for line_number, row in _rows(manifest_path):
        location = f"{manifest_path}:{line_number}"
        prompt_text = _required_string(row, ("prompt_text",), location)
        continuation = row.get("continuation_text")
        normalized = False
        if not isinstance(continuation, str):
            normalized_prompt = row.get("prompt_text_norm")
            normalized_continuation = row.get("continuation_text_norm")
            if isinstance(normalized_prompt, str) and isinstance(normalized_continuation, str):
                prompt_text = normalized_prompt
                continuation = normalized_continuation
                normalized = True
            else:
                generated_text = row.get("generated_text")
                if not isinstance(generated_text, str) or not generated_text.startswith(prompt_text):
                    raise ValueError(f"{location} needs continuation text or a generated-text prompt prefix")
                continuation = generated_text[len(prompt_text) :]
        candidates.append(
            TextCandidate(
                id=_id(row, location),
                prompt_text=prompt_text,
                continuation_text=continuation.strip(),
                normalized=normalized,
            )
        )
    return list(_index_unique(candidates, source=manifest_path).values())


def read_transcripts(path: str | Path) -> dict[str, SpeechTranscription]:
    """Read a cached Qwen3 continuation transcript manifest."""
    manifest_path = Path(path)
    transcripts: dict[str, SpeechTranscription] = {}
    for line_number, row in _rows(manifest_path):
        location = f"{manifest_path}:{line_number}"
        record_id = _id(row, location)
        if record_id in transcripts:
            raise ValueError(f"duplicate id {record_id!r} in {manifest_path}")
        transcripts[record_id] = SpeechTranscription(
            id=record_id,
            prompt_text=_string(row, ("prompt_text",), location),
            continuation_text=_string(row, ("continuation_text",), location),
            prompt_source=_required_string(row, ("prompt_source",), location),
        )
    if not transcripts:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return transcripts


def write_transcripts(path: str | Path, transcripts: Mapping[str, SpeechTranscription]) -> Path:
    """Atomically write cached continuation transcripts."""
    rows = (asdict(transcript) for transcript in transcripts.values())
    return _write_jsonl(path, rows)


def write_results(path: str | Path, records: Sequence[EvaluationRecord]) -> Path:
    """Atomically write per-example evaluation records."""
    return _write_jsonl(path, (asdict(record) for record in records))
