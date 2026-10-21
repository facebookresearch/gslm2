"""Base class for all model configuration dataclasses."""

import logging
from dataclasses import dataclass, fields
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class BaseModelConfig:
    """Empty base for model configs.

    Every model configuration (SpidR, OPT, MultiStreamOPT) inherits
    from this so that functions accepting *any*
    model config can be typed as ``BaseModelConfig`` instead of
    spelling out the full union.
    """


def build_nested_config(config_cls: type[T], data: dict[str, Any]) -> T:
    """Build a nested config dataclass from *data*, dropping unknown fields.

    Nested blocks (``prototype_clustering``, ``regularisation``, …) arrive as raw
    dicts from YAML or from a checkpoint's ``config.json``. Passing them straight
    to the dataclass makes a single unfamiliar key fatal, which breaks loading any
    checkpoint trained on a branch that has since added a field — so unknown keys
    are dropped with a warning instead.

    Mirrors the top-level filtering in
    :func:`gslm2.encoder.models.factory._build_from_checkpoint`; that one guards the outer
    config, this one the nested blocks it contains.
    """
    valid_fields = {f.name for f in fields(config_cls)}  # type: ignore[arg-type]
    unknown_fields = set(data) - valid_fields
    if unknown_fields:
        logger.warning("Ignoring unknown %s fields: %s", config_cls.__name__, sorted(unknown_fields))
        data = {k: v for k, v in data.items() if k in valid_fields}
    return config_cls(**data)
