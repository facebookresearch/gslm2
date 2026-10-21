"""Scheduling utilities for training."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gslm2.encoder.config import RegularisationConfig


def entropy_scheduler(  # noqa: PLR0911, C901
    step: int,
    reg_cfg: "RegularisationConfig",
    entropy_type: str,
) -> float:
    """Compute entropy regularization weight based on schedule.

    Args:
        step: Current training step
        reg_cfg: RegularisationConfig with scheduling parameters
        entropy_type: Either "predictive" or "codebook"

    Returns:
        Scheduled weight value
    """
    base_weight = getattr(reg_cfg, f"{entropy_type}_entropy_weight")
    schedule = getattr(reg_cfg, f"{entropy_type}_entropy_schedule")
    constant_steps = getattr(reg_cfg, f"{entropy_type}_entropy_constant_steps")
    max_steps = getattr(reg_cfg, f"{entropy_type}_entropy_max_steps")
    timescale = getattr(reg_cfg, f"{entropy_type}_entropy_timescale")
    final_weight = getattr(reg_cfg, f"{entropy_type}_entropy_final_weight")

    if base_weight == 0.0:
        return 0.0

    if schedule == "constant":
        return base_weight

    if "exponential" in schedule and timescale <= 0:
        raise ValueError(f"timescale must be positive for exponential schedules, but got {timescale}")

    if schedule == "constant_then_exponential_decrease":
        if final_weight >= base_weight:
            raise ValueError(
                f"For decrease schedule, {entropy_type}_entropy_final_weight ({final_weight}) "
                f"must be less than {entropy_type}_entropy_weight ({base_weight})"
            )
        if step < constant_steps:
            return base_weight
        if step < max_steps:
            weight = base_weight * math.exp(-(step - constant_steps) / timescale)
            return max(weight, final_weight)
        return final_weight

    if schedule == "constant_then_exponential_increase":
        if final_weight <= base_weight:
            raise ValueError(
                f"For increase schedule, {entropy_type}_entropy_final_weight ({final_weight}) "
                f"must be greater than {entropy_type}_entropy_weight ({base_weight})"
            )
        if step < constant_steps:
            return base_weight
        if step < max_steps:
            weight = final_weight - (final_weight - base_weight) * math.exp(-(step - constant_steps) / timescale)
            return min(weight, final_weight)
        return final_weight

    raise ValueError(f"Unknown entropy regularization schedule: {schedule}")
