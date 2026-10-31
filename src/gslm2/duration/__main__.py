# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line entry point for duration-predictor training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import yaml

from gslm2.duration.config import DurationPredictorConfig
from gslm2.duration.train import train_duration_predictor


def load_config(path: Path) -> DurationPredictorConfig:
    """Load a duration-predictor configuration from ``path``."""
    with path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)

    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a duration-predictor configuration mapping")

    values = document.get("duration_predictor", document)
    if not isinstance(values, dict):
        raise TypeError("duration_predictor must be a configuration mapping")
    return DurationPredictorConfig(**cast("dict[str, Any]", values))


def main(argv: list[str] | None = None) -> int:
    """Train a duration predictor from a YAML configuration."""
    parser = argparse.ArgumentParser(description="Train a duration predictor for deduplicated speech units")
    parser.add_argument("config", type=Path, nargs="?", help="DurationPredictorConfig YAML")
    parser.add_argument("--config", dest="config_option", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.config is not None and args.config_option is not None:
        parser.error("pass the configuration either positionally or with --config, not both")
    config_path = args.config or args.config_option
    if config_path is None:
        parser.error("a configuration path is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train_duration_predictor(load_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
