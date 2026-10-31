# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validate transcript caches against audio, boundaries, and ASR settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from gslm2.eval.generative.io import file_digest, write_json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gslm2.eval.generative.scoring.config import EvaluationConfig
    from gslm2.eval.generative.scoring.schema import SpeechCandidate


def transcript_input_digest(config: EvaluationConfig, candidates: Sequence[SpeechCandidate]) -> str:
    """Hash candidate audio contents and every input affecting transcription."""
    payload = {
        "version": 1,
        "reference": file_digest(config.reference_manifest),
        "candidate_manifest": file_digest(config.candidate_manifest),
        "qwen3": asdict(config.qwen3),
        "audio": [
            (candidate.id, str(candidate.audio_path), file_digest(candidate.audio_path)) for candidate in candidates
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_matches(directory: Path, input_digest: str) -> bool:
    """Reuse a cache only when both its inputs and its transcript file match."""
    metadata = directory / "transcripts.metadata.json"
    transcripts = directory / "transcripts.jsonl"
    if not metadata.is_file() or not transcripts.is_file():
        return False
    try:
        values = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(values, dict)
        and values.get("input_sha256") == input_digest
        and values.get("transcripts_sha256") == file_digest(transcripts)
    )


def write_cache_metadata(directory: Path, input_digest: str) -> None:
    """Commit metadata after the transcript artifact has been written."""
    write_json(
        directory / "transcripts.metadata.json",
        {"input_sha256": input_digest, "transcripts_sha256": file_digest(directory / "transcripts.jsonl")},
    )
