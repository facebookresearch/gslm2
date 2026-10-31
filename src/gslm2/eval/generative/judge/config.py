# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for Prometheus continuation judging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DEFAULT_JUDGE_MODEL = "prometheus-eval/prometheus-8x7b-v2.0"
_MAX_TCP_PORT = 65_535


@dataclass(frozen=True)
class JudgeModelConfig:
    """Model settings matching the target Prometheus-2 evaluation."""

    model_path: str = DEFAULT_JUDGE_MODEL
    max_new_tokens: int = 512
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("judge.max_new_tokens must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("judge.dtype must be float16, bfloat16, or float32")


@dataclass(frozen=True)
class EvaluationConfig:
    """Inputs and outputs for one post-ASR LLM-judge run."""

    input_manifest: Path
    output_directory: Path
    distributed_port: int = 31_571
    judge: JudgeModelConfig = field(default_factory=JudgeModelConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.distributed_port <= _MAX_TCP_PORT:
            raise ValueError("distributed_port must be a valid TCP port")
        object.__setattr__(self, "input_manifest", Path(self.input_manifest).expanduser().resolve())
        object.__setattr__(self, "output_directory", Path(self.output_directory).expanduser().resolve())


def load_config(path: str | Path) -> EvaluationConfig:
    """Load an LLM-judge configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)
    if not isinstance(document, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")

    values = cast("dict[str, Any]", document)
    judge_values = values.pop("judge", {})
    if not isinstance(judge_values, dict):
        raise TypeError("judge configuration values must be a mapping")

    for name in ("input_manifest", "output_directory"):
        value = values.get(name)
        if value is not None:
            value_path = Path(value).expanduser()
            values[name] = value_path if value_path.is_absolute() else config_path.parent / value_path

    return EvaluationConfig(**values, judge=JudgeModelConfig(**judge_values))
