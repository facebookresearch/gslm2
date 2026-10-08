"""Optimizer param-group construction and freeze-time removal.

Guards the fix for the freeze bug: at ``freeze_step`` the frozen (feature-extractor)
groups must be dropped, not a hardcoded index. Also covers the weight-decay split
toggle (``no_decay_on_norm_bias``).
"""

from __future__ import annotations

from torch import nn

from gslm2.checkpoint import remove_frozen_param_groups
from gslm2.config import OptimizerConfig
from gslm2.optimizer import build_optimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(4, 4)  # matches default to_freeze
        self.encoder = nn.Linear(4, 4)  # trainable (weight + bias)
        self.norm = nn.LayerNorm(4)  # trainable 1-D params


def _frozen_ids(model: nn.Module) -> set[int]:
    return {id(p) for n, p in model.named_parameters() if n.startswith("feature_extractor")}


def test_frozen_groups_tagged_and_removed_by_tag() -> None:
    model = _TinyModel()
    frozen = _frozen_ids(model)
    opt, _ = build_optimizer(model, OptimizerConfig())

    tagged = [g for g in opt.param_groups if g.get("is_frozen")]
    assert tagged, "expected frozen-tagged param groups"
    tagged_ids = {id(p) for g in tagged for p in g["params"]}
    assert tagged_ids == frozen, "tagged groups must hold exactly the to_freeze params"

    trainable_ids = {id(p) for g in opt.param_groups if not g.get("is_frozen") for p in g["params"]}
    assert trainable_ids.isdisjoint(frozen)

    remove_frozen_param_groups(opt)
    remaining = {id(p) for g in opt.param_groups for p in g["params"]}
    assert remaining.isdisjoint(frozen), "frozen params must be dropped"
    # The old `remove_param_group(optimizer, 1)` wrongly dropped the trainable
    # bias/LayerNorm group; the tag-based removal must keep every trainable param.
    assert trainable_ids <= remaining


def test_no_decay_split_when_enabled() -> None:
    opt, _ = build_optimizer(_TinyModel(), OptimizerConfig(no_decay_on_norm_bias=True))
    no_wd = [g for g in opt.param_groups if not g.get("is_frozen") and g.get("weight_decay") == 0.0]
    assert no_wd, "expected a weight_decay=0 group for biases / LayerNorm"


def test_uniform_weight_decay_is_default() -> None:
    opt, _ = build_optimizer(_TinyModel(), OptimizerConfig())  # default now uniform (parity)
    trainable = [g for g in opt.param_groups if not g.get("is_frozen")]
    assert len(trainable) == 1, "default should be uniform weight decay (single trainable group)"


def test_uniform_weight_decay_when_disabled() -> None:
    opt, _ = build_optimizer(_TinyModel(), OptimizerConfig(no_decay_on_norm_bias=False))
    trainable = [g for g in opt.param_groups if not g.get("is_frozen")]
    assert len(trainable) == 1, "uniform weight decay should yield a single trainable group"
    assert trainable[0].get("weight_decay", None) != 0.0
