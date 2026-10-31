# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom LR scheduler: tri-stage schedule with a cosine (instead of exponential) decay."""

from __future__ import annotations

import math
from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, final, override

from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.optim.lr_schedulers.lr_scheduler import (
    AbstractLRScheduler,
    get_per_param_group,
)
from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import RegimeSection, TriStageLRConfig
from fairseq2.utils.validation import ValidationError
from torch.optim import Optimizer  # noqa: TC002

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

TRI_STAGE_COSINE_LR: str = "tri_stage_cosine"


@final
class TriStageCosineLR(AbstractLRScheduler):
    """Tri-stage learning rate schedule with a cosine decay stage.

    Warm-up linearly to ``base_lr``, hold at it, then decay on a half-cosine down to
    ``final_lr_scale * base_lr``. Same warmup and hold as fairseq2's ``TriStageLR``,
    which decays exponentially instead. Not chainable.
    """

    def __init__(  # noqa: PLR0913  (mirrors fairseq2's TriStageLR signature)
        self,
        optimizer: Optimizer,
        num_steps: int,
        stage_ratio: tuple[float, float, float],
        *,
        start_lr_scale: float | Sequence[float] = 0.01,
        final_lr_scale: float | Sequence[float] = 0.01,
        last_epoch: int = -1,
    ) -> None:
        """Set up the schedule against ``optimizer``.

        ``num_steps`` is the total step budget, split by ``stage_ratio`` (warmup, hold,
        decay; must sum to 1). ``start_lr_scale`` and ``final_lr_scale`` set the warm-up
        and floor learning rates as fractions of ``base_lr``, per parameter group.
        """
        ratio_sum = sum(stage_ratio)
        if not math.isclose(ratio_sum, 1.0):
            raise ValueError(f"The sum of `stage_ratio` values must be 1.0, but is {ratio_sum} instead.")

        start_lr_scales = get_per_param_group(optimizer, "start_lr", start_lr_scale)
        final_lr_scales = get_per_param_group(optimizer, "final_lr", final_lr_scale)

        num_stage_steps = [int(r * num_steps) for r in stage_ratio]

        self.num_steps = num_steps
        self.start_lrs: Sequence[float] | None = None
        self.final_lrs: Sequence[float] | None = None
        self.start_lr_scales = start_lr_scales
        self.final_lr_scales = final_lr_scales
        self.num_stage_steps = num_stage_steps

        super().__init__(optimizer, last_epoch)

    # torch's LRScheduler persists every attribute, so drop the ones the config owns:
    # schedule shape comes from config, only the position within it from the checkpoint.
    _CONFIG_OWNED = (
        "num_steps",
        "num_stage_steps",
        "base_lrs",
        "start_lrs",
        "final_lrs",
        "start_lr_scales",
        "final_lr_scales",
    )

    @override
    def state_dict(self) -> dict:
        return {k: v for k, v in super().state_dict().items() if k not in self._CONFIG_OWNED}

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in self._CONFIG_OWNED})

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # Due to `LRScheduler`'s constructor quirks, we delay the initialization
        # of `start_lrs` and `final_lrs` to here.
        if self.start_lrs is None:
            self.start_lrs = [s * b for s, b in zip(self.start_lr_scales, base_lrs, strict=False)]

        if self.final_lrs is None:
            self.final_lrs = [s * b for s, b in zip(self.final_lr_scales, base_lrs, strict=False)]

        num_steps = self.last_epoch

        # Linearly increase the learning rate to its base value during warmup.
        if num_steps < self.num_stage_steps[0]:
            c = num_steps / self.num_stage_steps[0]

            return [s + (b - s) * c for b, s in zip(base_lrs, self.start_lrs, strict=False)]

        num_steps -= self.num_stage_steps[0]

        # Keep the learning rate constant during second stage.
        if num_steps < self.num_stage_steps[1]:
            return list(base_lrs)

        num_steps -= self.num_stage_steps[1]

        # Cosine-decay from `base_lr` to `final_lr` (= final_lr_scale * base_lr)
        # over the decay stage (progress c in [0, 1)).
        if num_steps < self.num_stage_steps[2]:
            c = num_steps / self.num_stage_steps[2]

            return [
                f_lr + 0.5 * (b - f_lr) * (1.0 + math.cos(math.pi * c))
                for b, f_lr in zip(base_lrs, self.final_lrs, strict=False)
            ]

        return list(self.final_lrs)


@final
class _TriStageCosineLRFactory:
    """Builds a :class:`TriStageCosineLR` from a ``TriStageLRConfig`` (mirrors
    fairseq2's ``_TriStageLRFactory``). Auto-wired with the optimizer and the
    regime section so it can read ``num_steps``."""

    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: TriStageLRConfig) -> TriStageCosineLR:
        num_steps = self._regime_section.num_steps
        if num_steps is None:
            raise ValidationError(
                f"`regime.num_steps` must be specified when `lr_scheduler` is '{TRI_STAGE_COSINE_LR}'."
            )

        # `TriStageCosineLR` accepts float-or-sequence and validates per-group
        # lengths internally via `get_per_param_group`.
        return TriStageCosineLR(
            self._optimizer,
            num_steps,
            config.stage_ratio,
            start_lr_scale=config.start_lr_scale,
            final_lr_scale=config.final_lr_scale,
        )


def register_tri_stage_cosine_lr(container: DependencyContainer) -> None:
    """Register the ``tri_stage_cosine`` LR scheduler on the recipe container.

    Call this from ``TrainRecipe.register`` (runs before the config is
    structured), so ``lr_scheduler.name: tri_stage_cosine`` resolves.
    """

    def create(resolver: DependencyResolver, config: TriStageLRConfig) -> LRScheduler:
        return resolver.resolve(_TriStageCosineLRFactory).create(config)

    register_component(
        container,
        LRScheduler,
        TRI_STAGE_COSINE_LR,
        config_kls=TriStageLRConfig,
        factory=create,
    )
    container.register_type(_TriStageCosineLRFactory)
