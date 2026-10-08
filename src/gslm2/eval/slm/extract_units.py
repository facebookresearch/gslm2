"""CLI: extract discrete units (codebook codes or k-means) from an encoder.

Reads a ``FeatExtractConfig`` YAML (see ``configs/extract/``), extracts units
for the configured manifests, and writes pre-extracted ``.txt`` / ``.jsonl``
manifests for GSLM2 training, the vocoder, and the duration predictor.
Single-process and torchrun/SLURM launches both work.
"""

import argparse
import logging
from pathlib import Path

import yaml

from gslm2.eval.slm.config import FeatExtractConfig
from gslm2.eval.slm.distributed import init_distributed_context
from gslm2.eval.slm.extract_features import run_feature_extraction_speech

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract discrete speech units from an encoder.")
    parser.add_argument("--config", type=Path, required=True, help="FeatExtractConfig YAML (see configs/extract/).")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg_dict = yaml.safe_load(f)
    cfg = FeatExtractConfig(**cfg_dict)
    context = init_distributed_context(cfg.distributed_port)
    run_feature_extraction_speech(cfg, context)


if __name__ == "__main__":
    main()
