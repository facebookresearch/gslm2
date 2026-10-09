"""Top-level configs (``BaseConfig``, ``SpeechConfig``, ``LMConfig``) and ``load_config``."""

import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from gslm2.config.common import (
    MaskingConfig,
    OptimizerConfig,
    RunConfig,
)
from gslm2.config.data import (
    DataConfig,
    OnTheFlyUnitSource,
    PreExtractedUnitSource,
    SLMUnitDataConfig,
    SpeechUnitConfig,
    TextDataConfig,
)
from gslm2.config.eval import (
    EvalSpec,
    build_eval_spec,
)
from gslm2.config.models import LM_MODEL_TYPES, model_type_to_config_type
from gslm2.config.models.core import CausalLMConfig, MultiStreamCausalLMConfig
from gslm2.config.models.speech import SpidRConfig


@dataclass(frozen=True)
class BaseConfig:
    """Shared fields and methods for all training configs."""

    run: RunConfig
    optimizer: OptimizerConfig

    def __post_init__(self) -> None:
        save_interval = self.run.save_interval
        for spec in self.eval_specs():
            if spec.eval_interval is None:
                continue
            if spec.eval_interval <= 0 or spec.eval_interval % save_interval != 0:
                raise ValueError(
                    f"eval_interval ({spec.eval_interval}) for eval_type {spec.eval_type!r} "
                    f"must be a positive multiple of run.save_interval ({save_interval}); "
                    "otherwise the eval would never have a checkpoint to evaluate."
                )

    def dump(self) -> None:
        with (self.run.dir / "config.json").open("w") as f:
            json.dump(asdict(self), f, indent=4, default=str)

    def eval_specs(self) -> list[EvalSpec]:
        """Return the canonical list of eval specs for validation scheduling."""
        return list(self.validation)

    @abstractmethod
    def eval_group_prefixes(self) -> list[str]:
        """Return wandb metric-namespace prefixes for each eval."""


@dataclass(frozen=True)
class SpeechConfig(BaseConfig):
    """Configuration for speech-encoder pretraining (SpidR 2.0)."""

    data: DataConfig | dict[str, DataConfig]
    model: SpidRConfig
    masking: MaskingConfig
    validation: list[EvalSpec] = field(default_factory=list)

    def eval_group_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        for spec in self.validation:
            cfg = spec.config
            group = cfg.get("group") if isinstance(cfg, dict) else None
            prefixes.append(group or spec.eval_type)
        return prefixes

    @classmethod
    def from_file(cls, path: str | Path) -> "SpeechConfig":
        path = Path(path)
        if path.suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        with path.open() as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        run = RunConfig(**data["run"])
        parsed_data = (
            DataConfig.from_dict(data["data"])
            if "manifest" in data["data"]
            else {k: DataConfig.from_dict(v) for k, v in data["data"].items()}
        )
        return cls(
            run=run,
            data=parsed_data,
            model=model_type_to_config_type(run.model_type)(**data.get("model", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            masking=MaskingConfig(**data.get("masking", {})),
            validation=[build_eval_spec(v) for v in (data.get("validation") or [])],
        )


def _parse_slm_unit_data(data_section: dict) -> SLMUnitDataConfig:
    """Build an :class:`SLMUnitDataConfig` from a raw ``data`` section, constructing the nested
    ``source`` dataclass: on-the-fly when it carries an ``audio`` block, else pre-extracted."""
    section = dict(data_section)
    source_section = dict(section.pop("source", {}))
    if "audio" in source_section:
        source: OnTheFlyUnitSource | PreExtractedUnitSource = OnTheFlyUnitSource(
            audio=DataConfig(**source_section["audio"]),
            feature_extraction=SpeechUnitConfig(**source_section["feature_extraction"]),
        )
    else:
        source = PreExtractedUnitSource(**source_section)
    return SLMUnitDataConfig(source=source, **section)


@dataclass(frozen=True)
class LMConfig(BaseConfig):
    """Configuration for causal language models (OPT / GPT-2 text, and speech-unit SLM)."""

    data: TextDataConfig | SLMUnitDataConfig
    model: CausalLMConfig | MultiStreamCausalLMConfig
    validation: list[EvalSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        # A multi-stream model predicts one stream per vocab (len(vocab_sizes)); the data must supply
        # exactly that many aligned streams, else MultiStreamOPT.forward rejects the batch at step 0.
        if (
            isinstance(self.data, SLMUnitDataConfig)
            and isinstance(self.model, MultiStreamCausalLMConfig)
            and self.data.num_streams != self.model.num_streams
        ):
            raise ValueError(
                f"data.num_streams ({self.data.num_streams}) must equal the model's stream count "
                f"len(vocab_sizes) = {self.model.num_streams}."
            )

    def eval_group_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        for spec in self.validation:
            base = f"evals_{spec.eval_name or spec.eval_type}"
            benchmarks = spec.config.get("benchmarks") if isinstance(spec.config, dict) else None
            if benchmarks:
                prefixes.extend(f"{base}/{b}" for b in benchmarks)
            else:
                prefixes.append(base)
        return prefixes

    @classmethod
    def from_file(cls, path: str | Path) -> "LMConfig":
        path = Path(path)
        if path.suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        with path.open() as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        run = RunConfig(**data["run"])
        model_cfg = model_type_to_config_type(run.model_type)(**data.get("model", {}))
        data_section = data.get("data", {})
        # msopt is always multi-stream SLM; opt/gpt2 carry a speech-unit ``source`` block for SLM
        # (otherwise they are plain text causal LMs). See SLMUnitDataConfig.
        parsed_data: TextDataConfig | SLMUnitDataConfig
        if run.model_type == "msopt" or "source" in data_section:
            parsed_data = _parse_slm_unit_data(data_section)
        else:
            parsed_data = TextDataConfig(**data_section)
        return cls(
            run=run,
            data=parsed_data,
            model=model_cfg,
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            validation=[build_eval_spec(v) for v in (data.get("validation") or [])],
        )


def load_config(path: str | Path) -> SpeechConfig | LMConfig:
    """Load a config dispatching to ``SpeechConfig`` or ``LMConfig`` by ``run.model_type``."""
    path = Path(path)
    if path.suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    with path.open() as f:
        data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    model_type = data.get("run", {}).get("model_type", "spidr")
    if model_type in LM_MODEL_TYPES:
        return LMConfig.from_file(path)
    return SpeechConfig.from_file(path)


@dataclass(frozen=True)
class ResumeConfig:
    step: int
    checkpoint: Path | None
    results: Path
