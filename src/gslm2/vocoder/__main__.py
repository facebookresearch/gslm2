"""Command-line entry point for unit HiFi-GAN training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import yaml

from gslm2.vocoder.config import UnitVocoderConfig
from gslm2.vocoder.train import train


def load_config(path: Path) -> UnitVocoderConfig:
    """Load flat or wrapped training configuration from YAML or JSON."""
    if path.suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
        if isinstance(document, dict):
            document = document.get("unit_vocoder", document)
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a unit-vocoder configuration mapping")
    return UnitVocoderConfig(**cast("dict[str, Any]", document))


def main(argv: list[str] | None = None) -> int:
    """Train a unit vocoder from YAML or legacy JSON configuration."""
    parser = argparse.ArgumentParser(description="Train a unit-conditioned HiFi-GAN vocoder")
    parser.add_argument("config", type=Path, nargs="?", help="unit-vocoder YAML or JSON configuration")
    parser.add_argument("--config", dest="config_option", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.config is not None and args.config_option is not None:
        parser.error("pass the configuration either positionally or with --config, not both")
    config_path = args.config or args.config_option
    if config_path is None:
        parser.error("a configuration path is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(load_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
