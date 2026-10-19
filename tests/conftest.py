# ruff: noqa: INP001  Top-level pytest conftest, intentionally no tests/__init__.py.
"""Session-scoped fixtures for the regression test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The in-process trajectory tests build trainers directly; never let wandb
# touch the network from the test suite.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tests.regression.helpers import build_tiny_manifest, build_tiny_text_manifest


@pytest.fixture(scope="session")
def tiny_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("regression_data")
    return build_tiny_manifest(d / "audio", d / "tiny.csv")


@pytest.fixture(scope="session")
def tiny_text_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("regression_text_data")
    return build_tiny_text_manifest(d / "tiny_text.txt")
