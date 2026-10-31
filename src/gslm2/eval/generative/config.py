# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One experiment configuration shared by all generative evaluation stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from gslm2.eval.generative.judge.config import EvaluationConfig as JudgeEvaluationConfig
from gslm2.eval.generative.judge.config import JudgeModelConfig
from gslm2.eval.generative.scoring.config import EvaluationConfig as ScoringConfig
from gslm2.eval.generative.scoring.config import PerplexityConfig, Qwen3Config, S12Config

_FEATURE_SECONDS = 0.02
_MAX_TCP_PORT = 65535


@dataclass(frozen=True)
class SamplingConfig:
    """Nucleus sampling settings for one stochastic replica."""

    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class SpeechGenerationConfig:
    """Speech prompts, LM metadata, and native synthesis settings."""

    input_manifest: Path
    prompt_unit_manifest: Path
    prompt_boundary_manifest: Path
    duration_predictor_checkpoint: Path
    vocoder_checkpoint: Path
    model_checkpoint: Path | None = None
    model_config: Path | None = None
    oracle: bool = False
    feature_seconds: float = 0.02
    prompt_max_seconds: float = 3.0
    continuation_seconds: float = 10.0
    depformer_temperature_annealing: float = 1.0

    def __post_init__(self) -> None:
        # The supported vocoder produces exactly 320 samples per frame at 16 kHz.
        if self.feature_seconds != _FEATURE_SECONDS:
            raise ValueError("speech synthesis requires feature_seconds=0.02")
        for value in (self.prompt_max_seconds, self.continuation_seconds, self.depformer_temperature_annealing):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("speech durations and temperature annealing must be finite and positive")


@dataclass(frozen=True)
class TextGenerationConfig:
    """Text transcripts, tokenizer, and continuation length."""

    input_manifest: Path
    tokenizer_path: Path
    prompt_boundary_manifest: Path
    model_checkpoint: Path | None = None
    model_config: Path | None = None
    text_field: str = "transcript"
    prompt_text_field: str | None = None
    prompt_tokens: int = 8
    continuation_tokens: int = 48
    oracle: bool = False

    def __post_init__(self) -> None:
        if not self.text_field or self.prompt_tokens < 1 or self.continuation_tokens < 1:
            raise ValueError("text field and prompt/continuation lengths must be non-empty and positive")


@dataclass(frozen=True)
class EvaluationOptions:
    """Scorers and transcript reuse settings shared across replicas."""

    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    s12: S12Config = field(default_factory=S12Config)
    reuse_transcripts: bool = True
    audio_root: Path | None = None


@dataclass(frozen=True)
class SweepConfig:
    """Explicit temperature and seed values used to plan a sweep."""

    temperatures: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: list(range(10)))

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("sweep seeds must be non-empty and unique")
        if len(self.temperatures) != len(set(self.temperatures)):
            raise ValueError("sweep temperatures must be unique")
        for seed in self.seeds:
            SamplingConfig(seed=seed)
        for temperature in self.temperatures:
            SamplingConfig(temperature=temperature)


@dataclass(frozen=True)
class RunConfig:
    """Resolved settings and artifact locations for one evaluation experiment."""

    mode: Literal["speech", "text"]
    output_directory: Path
    reference_manifest: Path
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    speech: SpeechGenerationConfig | None = None
    text: TextGenerationConfig | None = None
    evaluation: EvaluationOptions = field(default_factory=EvaluationOptions)
    judge: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    distributed_port: int = 29571

    def __post_init__(self) -> None:
        if self.mode not in {"speech", "text"}:
            raise ValueError("mode must be speech or text")
        if (self.mode == "speech" and self.text is not None) or (self.mode == "text" and self.speech is not None):
            raise ValueError("the generation section must match mode")
        if not 1 <= self.distributed_port <= _MAX_TCP_PORT:
            raise ValueError("distributed_port must be a valid TCP port")
        generation = self.speech if self.mode == "speech" else self.text
        if generation is not None and generation.prompt_boundary_manifest != self.reference_manifest:
            raise ValueError("generation and scoring must use the same prompt-boundary manifest")

    @property
    def oracle(self) -> bool:
        """Return whether this run uses ground-truth continuations."""
        generation = self.speech if self.mode == "speech" else self.text
        return generation is not None and generation.oracle

    @property
    def run_directory(self) -> Path:
        """Derive one canonical replica path for every pipeline stage."""
        if self.oracle:
            return self.output_directory / "oracle"
        sampling = self.sampling
        return self.output_directory / f"t{float(sampling.temperature)}-p{float(sampling.top_p)}" / str(sampling.seed)

    @property
    def candidate_manifest(self) -> Path:
        """Return the generated candidate manifest consumed by scoring."""
        name = "synthesized.csv" if self.mode == "speech" else "candidates.csv"
        return self.run_directory / "generation" / name

    def scoring_config(self, candidate_manifest: Path | None = None) -> ScoringConfig:
        """Build a scoring configuration for this exact replica."""
        options = self.evaluation
        return ScoringConfig(
            mode=self.mode,
            reference_manifest=self.reference_manifest,
            candidate_manifest=candidate_manifest or self.candidate_manifest,
            output_directory=self.run_directory / "scoring",
            audio_root=options.audio_root,
            reuse_transcripts=options.reuse_transcripts,
            distributed_port=self.distributed_port,
            qwen3=options.qwen3,
            perplexity=options.perplexity,
            s12=options.s12,
        )

    def judge_config(self, input_manifest: Path | None = None) -> JudgeEvaluationConfig:
        """Build the judge configuration from this replica's scoring artifact."""
        return JudgeEvaluationConfig(
            input_manifest=input_manifest or self.run_directory / "scoring" / "results.jsonl",
            output_directory=self.run_directory / "judge",
            distributed_port=self.distributed_port,
            judge=self.judge,
        )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return dict(value)


def _resolve_paths(values: dict[str, Any], parent: Path) -> None:
    for name, value in list(values.items()):
        if value is not None and (
            name.endswith(("manifest", "checkpoint"))
            or name in {"model_config", "tokenizer_path", "output_directory", "audio_root"}
        ):
            path = Path(value).expanduser()
            values[name] = path.resolve() if path.is_absolute() else (parent / path).resolve()
        if name == "model_path" and isinstance(value, str) and value.startswith(("/", ".", "~")):
            path = Path(value).expanduser()
            values[name] = str(path.resolve() if path.is_absolute() else (parent / path).resolve())


def load_config(path: str | Path) -> RunConfig:
    """Load one YAML used by decoding, synthesis, scoring, judging, and calibration."""
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        values = _mapping(yaml.safe_load(handle), str(source))
    sampling = SamplingConfig(**_mapping(values.pop("sampling", {}), "sampling"))
    speech_values = values.pop("speech", None)
    text_values = values.pop("text", None)
    speech = None
    text = None
    if speech_values is not None:
        speech_values = _mapping(speech_values, "speech")
        _resolve_paths(speech_values, source.parent)
        speech = SpeechGenerationConfig(**speech_values)
    if text_values is not None:
        text_values = _mapping(text_values, "text")
        _resolve_paths(text_values, source.parent)
        text = TextGenerationConfig(**text_values)
    options = _mapping(values.pop("evaluation", {}), "evaluation")
    model_options = [_mapping(options.pop(name, {}), name) for name in ("qwen3", "perplexity", "s12")]
    for model_values in model_options:
        _resolve_paths(model_values, source.parent)
    _resolve_paths(options, source.parent)
    evaluation = EvaluationOptions(
        **options,
        qwen3=Qwen3Config(**model_options[0]),
        perplexity=PerplexityConfig(**model_options[1]),
        s12=S12Config(**model_options[2]),
    )
    judge_values = _mapping(values.pop("judge", {}), "judge")
    _resolve_paths(judge_values, source.parent)
    judge = JudgeModelConfig(**judge_values)
    sweep = SweepConfig(**_mapping(values.pop("sweep", {}), "sweep"))
    _resolve_paths(values, source.parent)
    generation = speech or text
    if "reference_manifest" not in values:
        if generation is None:
            raise ValueError("reference_manifest is required when no generation section is supplied")
        values["reference_manifest"] = generation.prompt_boundary_manifest
    return RunConfig(
        **values, sampling=sampling, speech=speech, text=text, evaluation=evaluation, judge=judge, sweep=sweep
    )


def override_replica(config: RunConfig, temperature: float | None, seed: int | None) -> RunConfig:
    """Apply sampling overrides without duplicating any input configuration."""
    sampling = config.sampling
    return replace(
        config,
        sampling=replace(
            sampling,
            temperature=sampling.temperature if temperature is None else temperature,
            seed=sampling.seed if seed is None else seed,
        ),
    )
