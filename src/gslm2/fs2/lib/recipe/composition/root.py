# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# ruff: noqa: TC002
from fairseq2.checkpoint import CheckpointManager, StandardCheckpointManager
from fairseq2.error import raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.recipe.base import (
    Recipe,
    RecipeContext,
    TrainRecipe,
)
from fairseq2.recipe.composition.data_parallel import _register_data_parallel_wrappers
from fairseq2.recipe.composition.lr_schedulers import _register_lr_schedulers
from fairseq2.recipe.composition.model import _register_train_model
from fairseq2.recipe.composition.optim import _register_optim
from fairseq2.recipe.composition.root import _register_recipe_common
from fairseq2.recipe.composition.trainer import _register_trainer_factory
from fairseq2.recipe.internal.gang import (
    _RecipeFSDPGangsFactory,
)
from fairseq2.recipe.task import Task
from fairseq2.recipe.trainer import Trainer
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

from gslm2.fs2.utils.gang import maybe_setup_gangs


def _register_train_recipe(container: DependencyContainer, recipe: TrainRecipe) -> None:
    _register_recipe_common(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    container.register_instance(TrainRecipe, recipe)

    # Trainer
    def create_trainer(resolver: DependencyResolver) -> Trainer:
        context = RecipeContext(resolver)

        try:
            return recipe.create_trainer(context)
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)

    container.register(Task, create_trainer)

    _register_data_parallel_wrappers(container)

    _register_lr_schedulers(container)

    _register_optim(container)

    _register_trainer_factory(container)

    _register_train_model(container)

    container.register_type(CheckpointManager, StandardCheckpointManager, singleton=True)

    # Gangs
    container.register(Gangs, maybe_setup_gangs, singleton=True)

    container.register_type(_RecipeFSDPGangsFactory)

    # Custom Objects
    recipe.register(container)
