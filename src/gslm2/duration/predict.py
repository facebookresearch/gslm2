"""Checkpoint loading and inference helpers for duration predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from gslm2.duration.model import DurationPredictor, DurationPredictorModelConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

CheckpointConfig = dict[str, object]


def _integer_option(config: CheckpointConfig, name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int | float | str):
        raise TypeError(f"checkpoint option {name!r} must be numeric")
    return int(value)


def _float_option(config: CheckpointConfig, name: str, default: float) -> float:
    value = config.get(name, default)
    if not isinstance(value, int | float | str):
        raise TypeError(f"checkpoint option {name!r} must be numeric")
    return float(value)


def _model_from_config(config: CheckpointConfig) -> DurationPredictor:
    model_config = DurationPredictorModelConfig(
        vocab_size=_integer_option(config, "vocab_size", 256),
        embed_dim=_integer_option(config, "embed_dim", 256),
        hidden_dim=_integer_option(config, "hidden_dim", 256),
        num_layers=_integer_option(config, "num_layers", 2),
        dropout=_float_option(config, "dropout", 0.1),
        num_streams=_integer_option(config, "num_streams", 1),
        num_classes=_integer_option(config, "num_classes", 0),
        encoder_type=str(config.get("encoder_type", "conformer")),
        attention_heads=_integer_option(config, "attention_heads", 4),
        conv_kernel=_integer_option(config, "conv_kernel", 17),
        ffn_expansion=_integer_option(config, "ffn_expansion", 4),
        max_seq_len=max(_integer_option(config, "max_seq_len", 4096), 4096),
    )
    return DurationPredictor(model_config)


def load_duration_predictor(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[DurationPredictor, CheckpointConfig]:
    """Load a training checkpoint and reconstruct its duration predictor."""
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = cast(
        "dict[str, object]", torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    )
    config = cast("CheckpointConfig", checkpoint.get("config"))
    state_dict = cast("dict[str, Tensor]", checkpoint.get("model"))
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise TypeError("duration-predictor checkpoint must contain 'config' and 'model' mappings")

    model = _model_from_config(config)
    model.load_state_dict(state_dict)
    model.to(resolved_device).eval()
    return model, config


def _model_device(model: DurationPredictor, device: str | torch.device | None) -> torch.device:
    return torch.device(device) if device is not None else next(model.parameters()).device


def predict_durations(
    model: DurationPredictor,
    deduplicated_tokens: Sequence[int] | Sequence[Sequence[int]],
    *,
    log_duration: bool,
    device: str | torch.device | None = None,
) -> list[int]:
    """Predict durations for one unpadded single- or multi-stream sequence."""
    resolved_device = _model_device(model, device)
    if model.num_streams > 1:
        streams = cast("Sequence[Sequence[int]]", deduplicated_tokens)
        if len(streams) != model.num_streams:
            raise ValueError(f"expected {model.num_streams} streams, received {len(streams)}")
        sequence_length = len(streams[0])
        if any(len(stream) != sequence_length for stream in streams):
            raise ValueError("all streams must have the same length")
        tokens = torch.tensor(streams, dtype=torch.long, device=resolved_device).unsqueeze(0)
    else:
        sequence = cast("Sequence[int]", deduplicated_tokens)
        sequence_length = len(sequence)
        tokens = torch.tensor(sequence, dtype=torch.long, device=resolved_device).unsqueeze(0)

    if sequence_length == 0:
        return []
    lengths = torch.tensor([sequence_length], dtype=torch.long, device=resolved_device)
    durations = model.predict(tokens, lengths, log_duration=log_duration)
    return cast("list[int]", durations[0, :sequence_length].tolist())


def expand_tokens(deduplicated_tokens: Sequence[int], durations: Sequence[int]) -> list[int]:
    """Repeat each token by its corresponding predicted duration."""
    return [
        token for token, duration in zip(deduplicated_tokens, durations, strict=True) for _ in range(max(0, duration))
    ]


def expand_streams(deduplicated_streams: Sequence[Sequence[int]], durations: Sequence[int]) -> list[list[int]]:
    """Expand aligned streams using a common duration sequence."""
    return [expand_tokens(stream, durations) for stream in deduplicated_streams]


def dedup_to_full(
    model: DurationPredictor,
    deduplicated_tokens: Sequence[int] | Sequence[Sequence[int]],
    *,
    log_duration: bool,
    device: str | torch.device | None = None,
) -> list[int] | list[list[int]]:
    """Predict durations and expand one deduplicated unit sequence."""
    durations = predict_durations(model, deduplicated_tokens, log_duration=log_duration, device=device)
    if model.num_streams > 1:
        return expand_streams(cast("Sequence[Sequence[int]]", deduplicated_tokens), durations)
    return expand_tokens(cast("Sequence[int]", deduplicated_tokens), durations)
