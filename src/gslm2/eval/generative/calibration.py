# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate replicas and select an S12-matched decoding temperature."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from gslm2.eval.generative.config import RunConfig

from gslm2.eval.generative.io import write_json

_MIN_CALIBRATION_POINTS = 2


@dataclass(frozen=True)
class TemperaturePoint:
    """Metrics aggregated over every replica at one decoding temperature."""

    temperature: float
    window_words: int
    s12: float
    conditional_genppl: float
    replicas: int


@dataclass(frozen=True)
class CalibrationResult:
    """Interpolated S12-matched temperature and conditional GenPPL."""

    temperature: float
    window_words: int
    conditional_genppl: float
    lower_temperature: float
    upper_temperature: float


def aggregate_replicas(temperature: float, summaries: Sequence[Mapping[str, object]]) -> TemperaturePoint:
    """Aggregate per-replica summaries according to the corrected evaluation protocol."""
    if not summaries:
        raise ValueError("at least one replica summary is required")
    reference_digests = {summary.get("reference_manifest_sha256") for summary in summaries}
    if len(reference_digests) != 1:
        raise ValueError("replica summaries use different reference manifests")

    def values(name: str) -> list[float]:
        result = []
        for summary in summaries:
            value = summary.get(name)
            if not isinstance(value, int | float):
                raise TypeError(f"replica summary has no numeric {name!r}")
            result.append(float(value))
        return result

    candidate_r1 = statistics.fmean(values("candidate_repetition_1_mean"))
    candidate_r2 = statistics.fmean(values("candidate_repetition_2_mean"))
    reference_r1 = statistics.fmean(values("reference_repetition_1_mean"))
    reference_r2 = statistics.fmean(values("reference_repetition_2_mean"))
    if reference_r1 == 0.0 or reference_r2 == 0.0:
        raise ValueError("reference repetition rates must be non-zero to compute S12")
    windows = {int(value) for value in values("window_words")}
    if len(windows) != 1:
        raise ValueError(f"replica summaries use different word windows: {sorted(windows)}")
    return TemperaturePoint(
        temperature=temperature,
        window_words=windows.pop(),
        s12=0.5 * (candidate_r1 / reference_r1 + candidate_r2 / reference_r2),
        conditional_genppl=statistics.fmean(values("cond_genppl_median")),
        replicas=len(summaries),
    )


def calibrate_temperature(
    points: Sequence[TemperaturePoint],
    *,
    target_s12: float = 1.0,
    temperature_digits: int = 2,
) -> CalibrationResult:
    """Interpolate between neighboring temperatures that bracket the target S12."""
    if len(points) < _MIN_CALIBRATION_POINTS:
        raise ValueError("temperature calibration requires at least two points")
    ordered = sorted(points, key=lambda point: point.temperature)
    windows = {point.window_words for point in ordered}
    if len(windows) != 1:
        raise ValueError(f"temperature points use different word windows: {sorted(windows)}")
    window_words = windows.pop()
    exact = [point for point in ordered if point.s12 == target_s12]
    if exact:
        point = exact[0]
        return CalibrationResult(
            temperature=round(point.temperature, temperature_digits),
            window_words=window_words,
            conditional_genppl=point.conditional_genppl,
            lower_temperature=point.temperature,
            upper_temperature=point.temperature,
        )
    brackets = [
        (left, right) for left, right in pairwise(ordered) if (left.s12 - target_s12) * (right.s12 - target_s12) <= 0.0
    ]
    if not brackets:
        observed = (min(point.s12 for point in ordered), max(point.s12 for point in ordered))
        raise ValueError(f"target S12={target_s12} is not bracketed by observed range {observed}")
    left, right = min(
        brackets,
        key=lambda pair: abs(pair[0].s12 - target_s12) + abs(pair[1].s12 - target_s12),
    )
    if left.s12 == right.s12:
        raise ValueError("cannot interpolate between equal S12 values")
    if left.conditional_genppl <= 0.0 or right.conditional_genppl <= 0.0:
        raise ValueError("conditional GenPPL must be positive for log-linear interpolation")

    fraction = (target_s12 - left.s12) / (right.s12 - left.s12)
    temperature = left.temperature + fraction * (right.temperature - left.temperature)
    log_perplexity = math.log(left.conditional_genppl) + fraction * (
        math.log(right.conditional_genppl) - math.log(left.conditional_genppl)
    )
    return CalibrationResult(
        temperature=round(temperature, temperature_digits),
        window_words=window_words,
        conditional_genppl=math.exp(log_perplexity),
        lower_temperature=left.temperature,
        upper_temperature=right.temperature,
    )


def calibrate_run(config: RunConfig) -> Path:
    """Read every configured dev replica and write the selected temperature."""
    if config.oracle or not config.sweep.temperatures:
        raise ValueError("calibration requires a non-oracle temperature sweep")
    points = []
    digests: set[object] = set()
    for temperature in config.sweep.temperatures:
        summaries = []
        for seed in config.sweep.seeds:
            replica = replace(config, sampling=replace(config.sampling, temperature=temperature, seed=seed))
            path = replica.run_directory / "scoring" / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise TypeError(f"{path} must contain a summary mapping")
            digests.add(summary.get("reference_manifest_sha256"))
            summaries.append(summary)
        points.append(aggregate_replicas(temperature, summaries))
    if len(digests) != 1:
        raise ValueError("temperature points use different reference manifests")
    result = calibrate_temperature(points)
    return write_json(
        config.output_directory / "calibration.json",
        {"selected": asdict(result), "points": [asdict(point) for point in points], "config": asdict(config)},
    )


def main(argv: list[str] | None = None) -> int:
    """Select an S12-matched temperature from completed seed summaries."""
    from gslm2.eval.generative.cli import run_stage

    return run_stage("calibrate", argv)


if __name__ == "__main__":
    raise SystemExit(main())
