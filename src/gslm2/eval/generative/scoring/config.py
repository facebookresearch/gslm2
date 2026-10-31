# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for generative continuation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_PERPLEXITY_MODEL = "meta-llama/Llama-3.2-1B"
_MIN_MODEL_SEQUENCE_LENGTH = 2
_MAX_TCP_PORT = 65_535


@dataclass(frozen=True)
class Qwen3Config:
    """Settings for deterministic continuation transcription with Qwen3-Omni."""

    model_path: str = DEFAULT_QWEN3_MODEL
    batch_size: int = 1
    max_new_tokens: int = 512
    prompt: str = "Transcribe the English audio into text. Return only the transcription."
    device_map: str = "auto"
    attention_implementation: str = "sdpa"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("qwen3.batch_size must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("qwen3.max_new_tokens must be positive")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("qwen3.attention_implementation must be eager, sdpa, or flash_attention_2")


@dataclass(frozen=True)
class PerplexityConfig:
    """Settings for the causal LM used to score generated continuations."""

    model_path: str = DEFAULT_PERPLEXITY_MODEL
    batch_size: int = 16
    max_length: int = 2048
    device_map: str = "auto"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "float16"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("perplexity.batch_size must be positive")
        if self.max_length < _MIN_MODEL_SEQUENCE_LENGTH:
            raise ValueError("perplexity.max_length must be at least 2")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("perplexity.dtype must be auto, float16, bfloat16, or float32")


@dataclass(frozen=True)
class S12Config:
    """Word window and optional fixed reference rates for S12."""

    window_words: int = 15
    reference_r1: float | None = None
    reference_r2: float | None = None

    def __post_init__(self) -> None:
        if self.window_words < 1:
            raise ValueError("s12.window_words must be positive")
        if (self.reference_r1 is None) != (self.reference_r2 is None):
            raise ValueError("s12.reference_r1 and s12.reference_r2 must be set together")
        if (
            self.reference_r1 is not None
            and self.reference_r2 is not None
            and (self.reference_r1 <= 0.0 or self.reference_r2 <= 0.0)
        ):
            raise ValueError("S12 reference rates must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    """Inputs and outputs for one speech or text generation run."""

    mode: Literal["speech", "text"]
    reference_manifest: Path
    candidate_manifest: Path
    output_directory: Path
    audio_root: Path | None = None
    reuse_transcripts: bool = True
    distributed_port: int = 30_571
    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    s12: S12Config = field(default_factory=S12Config)

    def __post_init__(self) -> None:
        if self.mode not in {"speech", "text"}:
            raise ValueError("mode must be speech or text")
        if not 1 <= self.distributed_port <= _MAX_TCP_PORT:
            raise ValueError("distributed_port must be a valid TCP port")
        object.__setattr__(self, "reference_manifest", Path(self.reference_manifest).expanduser().resolve())
        object.__setattr__(self, "candidate_manifest", Path(self.candidate_manifest).expanduser().resolve())
        object.__setattr__(self, "output_directory", Path(self.output_directory).expanduser().resolve())
        if self.audio_root is not None:
            object.__setattr__(self, "audio_root", Path(self.audio_root).expanduser().resolve())


def load_config(path: str | Path) -> EvaluationConfig:
    """Load an evaluation configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)
    if not isinstance(document, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")

    values = cast("dict[str, Any]", document)
    qwen3_values = values.pop("qwen3", {})
    perplexity_values = values.pop("perplexity", {})
    s12_values = values.pop("s12", {})
    if (
        not isinstance(qwen3_values, dict)
        or not isinstance(perplexity_values, dict)
        or not isinstance(s12_values, dict)
    ):
        raise TypeError("qwen3, perplexity, and s12 configuration values must be mappings")

    for name in ("reference_manifest", "candidate_manifest", "output_directory", "audio_root"):
        value = values.get(name)
        if value is not None:
            value_path = Path(value).expanduser()
            values[name] = value_path if value_path.is_absolute() else config_path.parent / value_path

    return EvaluationConfig(
        **values,
        qwen3=Qwen3Config(**qwen3_values),
        perplexity=PerplexityConfig(**perplexity_values),
        s12=S12Config(**s12_values),
    )
