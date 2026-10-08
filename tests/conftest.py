# ruff: noqa: INP001  Top-level pytest conftest, intentionally no tests/__init__.py.
"""Session-scoped fixtures for the regression test suite."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

# The in-process trajectory tests build trainers directly; never let wandb
# touch the network from the test suite.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from gslm2.config import AUDIO_SAMPLE_RATE
from tests.regression.helpers import ALIGNMENT_HZ, build_tiny_manifest, build_tiny_text_manifest

_PHONEME_FRAMES_PER_SAMPLE = AUDIO_SAMPLE_RATE // ALIGNMENT_HZ


@pytest.fixture(scope="session")
def tiny_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("regression_data")
    return build_tiny_manifest(d / "audio", d / "tiny.csv")


@pytest.fixture(scope="session")
def tiny_text_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("regression_text_data")
    return build_tiny_text_manifest(d / "tiny_text.txt")


@pytest.fixture(scope="session")
def tiny_alignments(tiny_manifest: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Phoneme alignments CSV (fileid, phones, language). One token per 10ms (ALIGNMENT_FREQ=100Hz)."""
    path = tmp_path_factory.mktemp("regression_alignments") / "tiny.csv"
    with tiny_manifest.open() as f:
        rows = list(csv.DictReader(f))
    cycle = ["AA", "AE", "AH", "AO"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fileid", "phones", "language"])
        writer.writeheader()
        for row in rows:
            num_phoneme_frames = int(row["num_frames"]) // _PHONEME_FRAMES_PER_SAMPLE
            phones = " ".join(cycle[i % len(cycle)] for i in range(num_phoneme_frames))
            writer.writerow({"fileid": row["fileid"], "phones": phones, "language": "en_arpa"})
    return path


@pytest.fixture(scope="session")
def tiny_phoneme_dict(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny phoneme dictionary mirroring src/metasr/tokenizer.py's expected schema."""
    path = tmp_path_factory.mktemp("regression_phoneme_dict") / "phoneme_dict.json"
    path.write_text(json.dumps({"en_arpa": "AA AE AH AO"}))
    return path
