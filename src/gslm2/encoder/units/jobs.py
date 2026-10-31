# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Route unit extraction manifests to downstream output directories."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gslm2.encoder.units.config import FeatExtractConfig, FeatureExtractionTargets


@dataclass(frozen=True)
class ExtractionJob:
    manifest: Path
    output_subdir: str
    output_format: Literal["txt", "jsonl"]
    prompt_boundary_manifest: Path | None = None


def build_extraction_jobs(cfg: FeatExtractConfig) -> list[ExtractionJob]:
    """Expand component targets or train/valid/test manifests into jobs."""
    jobs: list[ExtractionJob] = []
    if cfg.targets is not None:
        assert isinstance(cfg.targets, FeatureExtractionTargets)
        if cfg.targets.duration_predictor is not None:
            jobs.extend(
                ExtractionJob(manifest, "duration_predictor", "txt")
                for manifest in cfg.targets.duration_predictor.manifests
            )
        if cfg.targets.unit_vocoder is not None:
            jobs.extend(
                ExtractionJob(manifest, "unit_vocoder", "jsonl") for manifest in cfg.targets.unit_vocoder.manifests
            )
        if cfg.targets.generative_eval is not None:
            for manifest in cfg.targets.generative_eval.manifests:
                jobs.append(ExtractionJob(manifest.audio_manifest, "generative_eval/full", "jsonl"))
                jobs.append(
                    ExtractionJob(
                        manifest.audio_manifest,
                        "generative_eval/prompt",
                        "jsonl",
                        prompt_boundary_manifest=manifest.prompt_boundary_manifest,
                    )
                )
        return jobs

    for split in ("train", "valid", "test"):
        output_format = cfg.output_format
        if output_format == "auto":
            output_format = "jsonl" if split == "test" else "txt"
        jobs.extend(ExtractionJob(manifest, split, output_format) for manifest in getattr(cfg, f"manifests_{split}"))
    return jobs
