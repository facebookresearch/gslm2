# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

# ruff: noqa: TC002
from fairseq2.composition import _register_library
from fairseq2.recipe.base import TrainRecipe
from fairseq2.recipe.run import _register_run, _run_recipe, _swap_default_resolver
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.warn import enable_deprecation_warnings

from gslm2.lm.runtime.recipe.composition.root import _register_train_recipe

if TYPE_CHECKING:
    from pathlib import Path


def train(recipe: TrainRecipe, config: object, output_dir: Path) -> None:
    enable_deprecation_warnings()

    configure_rich_logging()

    container = DependencyContainer()

    with _swap_default_resolver(container):
        _register_library(container)

        _register_train_recipe(container, recipe)

        _register_run(container, recipe, config, output_dir)

        _run_recipe(container)
