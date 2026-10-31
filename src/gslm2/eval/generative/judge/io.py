# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Manifest IO for LLM-judge evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from gslm2.eval.generative.io import read_rows as _rows
from gslm2.eval.generative.io import write_json as write_json
from gslm2.eval.generative.io import write_jsonl as _write_jsonl
from gslm2.eval.generative.judge.schema import JudgeExample, JudgeRecord

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _required_text(row: Mapping[str, object], field: str, location: str, *, allow_empty: bool = False) -> str:
    value = row.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{location} needs {field!r} as a {qualifier}")
    return value.strip()


def read_examples(path: str | Path) -> list[JudgeExample]:
    """Read the uniform post-ASR prompt/continuation contract."""
    manifest_path = Path(path)
    examples = []
    seen_ids: set[str] = set()
    for line_number, row in _rows(manifest_path):
        location = f"{manifest_path}:{line_number}"
        example_id = _required_text(row, "id", location)
        if example_id in seen_ids:
            raise ValueError(f"duplicate id {example_id!r} in {manifest_path}")
        seen_ids.add(example_id)
        examples.append(
            JudgeExample(
                id=example_id,
                prompt_text=_required_text(row, "prompt_text", location),
                continuation_text=_required_text(row, "continuation_text", location, allow_empty=True),
            )
        )
    if not examples:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return examples


def write_results(path: str | Path, records: Sequence[JudgeRecord]) -> Path:
    """Atomically write per-example results as JSON Lines."""
    return _write_jsonl(path, (asdict(record) for record in records))
