# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for UnitLMTrainRecipe (fairseq2 recipe.py).

Usage with torchrun:
    torchrun --standalone --nproc_per_node=1 \
        -m gslm2.lm.recipes.train.run_recipe \
        --config-file path/to/config.yaml \
        -- /path/to/output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gslm2.lm.recipes.train.config import load_unit_lm_train_config
from gslm2.lm.recipes.train.recipe import UnitLMTrainRecipe
from gslm2.lm.runtime.recipe.run import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a UnitLM recipe.")
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save checkpoints and outputs.",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:  # noqa: PTH123
        raw_config = yaml.safe_load(f)

    config = load_unit_lm_train_config(raw_config)

    # Ensure the dataset path is also set in the config_overrides
    # so fairseq2's register_dataset_family can resolve it.
    if config.dataset.path and not config.dataset.config_overrides:
        config.dataset.config_overrides = {"path": str(config.dataset.path)}
    elif config.dataset.path and isinstance(config.dataset.config_overrides, dict):
        config.dataset.config_overrides.setdefault("path", str(config.dataset.path))

    recipe = UnitLMTrainRecipe()
    train(recipe, config, args.output_dir)


if __name__ == "__main__":
    main()
