"""Pre- and post-step hooks for ``BaseTrainer``.

``PostStepHook`` receives ``(trainer, step, batch, metrics)`` and may mutate
``metrics`` in place to surface fields into the wandb log.
``PreStepHook`` receives ``(trainer, step, batch)``.
"""

from typing import TYPE_CHECKING, Any, Protocol

from gslm2.checkpoint import remove_frozen_param_groups
from gslm2.optimizer import drop_scheduler_base_lrs

if TYPE_CHECKING:
    from gslm2.training.base import BaseTrainer


class PostStepHook(Protocol):
    def __call__(self, trainer: "BaseTrainer", step: int, batch: Any, metrics: dict[str, Any]) -> None: ...  # noqa: ANN401


class PreStepHook(Protocol):
    def __call__(self, trainer: "BaseTrainer", step: int, batch: Any) -> None: ...  # noqa: ANN401


class FreezeParamGroupHook:
    """At ``freeze_step``, drop the optimizer's frozen (feature-extractor) param groups."""

    def __init__(self, freeze_step: int) -> None:
        self.freeze_step = freeze_step

    def __call__(self, trainer: "BaseTrainer", step: int, batch: Any) -> None:  # noqa: ANN401, ARG002
        if step == self.freeze_step:
            n_before = len(trainer.optimizer.param_groups)
            removed = remove_frozen_param_groups(trainer.optimizer)
            if removed:
                assert removed == list(range(n_before - len(removed), n_before)), (
                    "frozen groups must be a suffix for scheduler truncation"
                )
                drop_scheduler_base_lrs(trainer.scheduler, len(trainer.optimizer.param_groups))


class UpdateEmaHook:
    """Call ``model.update_ema(step)`` and write decay to ``metrics["ema_decay"]``."""

    def __call__(self, trainer: "BaseTrainer", step: int, batch: Any, metrics: dict[str, Any]) -> None:  # noqa: ANN401, ARG002
        metrics["ema_decay"] = float(trainer.model.update_ema(step))
