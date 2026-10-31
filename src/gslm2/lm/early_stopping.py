# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from fairseq2.early_stopper import EarlyStopper

logger = logging.getLogger(__name__)


class StandardEarlyStopper(EarlyStopper):
    """Standard early stopping implementation with patience and configurable evaluation frequency."""

    def __init__(
        self,
        patience: int,
        every_n_steps: int,
        *,
        lower_is_better: bool = True,
        disabled: bool = False,
    ) -> None:
        """
        Initialize the StandardEarlyStopper.
        :param patience: Number of evaluation periods to wait without improvement before stopping.
        :param every_n_steps: Evaluate for early stopping every N training steps.
        :param lower_is_better: If True, lower scores are considered better (e.g., for loss).
                               If False, higher scores are considered better (e.g., for accuracy).
        """
        self.patience = patience
        self.every_n_steps = every_n_steps
        self.lower_is_better = lower_is_better

        self.best_score: float | None = None
        self.wait_count = 0
        self.last_eval_step = -1

        self.disabled = disabled

    def should_stop(self, step_nr: int, score: float) -> bool:
        """
        Determine if training should be stopped based on early stopping criteria.
        :param step_nr: The number of the current training step.
        :param score: The validation score of the current training step.
        :returns: ``True`` if the training should be stopped; otherwise, ``False``.
        """
        if self.disabled:
            return False

        # Only evaluate for early stopping every n steps
        if step_nr - max(0, self.last_eval_step) < self.every_n_steps:
            return False

        self.last_eval_step = step_nr

        # Initialize best score on first evaluation
        if self.best_score is None:
            self.best_score = score
            self.wait_count = 0
            return False

        # Check if current score is better than best score
        is_better = score < self.best_score if self.lower_is_better else score > self.best_score

        if is_better:
            self.best_score = score
            self.wait_count = 0
            return False

        remaining_patience = self.patience - self.wait_count
        logger.info("Will early stop in %s validation steps if no improvement.", remaining_patience)

        self.wait_count += 1
        return self.wait_count >= self.patience
