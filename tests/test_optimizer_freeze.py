"""Optimizer param-group construction and freeze-time removal.

Guards the fix for the freeze bug: at ``freeze_step`` the frozen (feature-extractor)
groups must be dropped, not a hardcoded index. Also covers the weight-decay split
toggle (``no_decay_on_norm_bias``).
"""

from __future__ import annotations

import pytest
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


def _walk_schedulers(sched):
    yield sched
    for sub in getattr(sched, "_schedulers", []):
        yield from _walk_schedulers(sub)


def test_scheduler_survives_frozen_group_drop() -> None:
    """Scheduler state must shrink with the optimizer at freeze_step.

    Regression test: dropping frozen param groups without shrinking the
    schedulers' base_lrs crashes every step after freeze_step on torch>=2.13
    (strict zip in _update_lr) — the encoder run died exactly this way at
    step 200000. The base_lrs invariant below catches it on any version.
    """
    import types

    from gslm2.training.hooks import FreezeParamGroupHook

    model = _TinyModel()
    opt, sched = build_optimizer(model, OptimizerConfig())
    assert len(opt.param_groups) > 1

    stub = types.SimpleNamespace(optimizer=opt, scheduler=sched)
    FreezeParamGroupHook(freeze_step=200_000)(stub, 200_000, None)

    n_groups = len(opt.param_groups)
    assert n_groups >= 1
    for s in _walk_schedulers(sched):
        if hasattr(s, "base_lrs"):  # SequentialLR itself holds none; subs do
            assert len(s.base_lrs) == n_groups, f"{type(s).__name__}.base_lrs not shrunk"
    sched.step()  # must not raise
    assert len(sched.get_last_lr()) == n_groups


def test_scheduler_prune_is_idempotent() -> None:
    """Pruning already-short scheduler state (resumed post-drop checkpoint)
    must be a no-op; remaining groups keep their exact LR schedule."""
    from gslm2.optimizer import drop_scheduler_base_lrs

    model = _TinyModel()
    opt, sched = build_optimizer(model, OptimizerConfig())
    _, ref_sched = build_optimizer(_TinyModel(), OptimizerConfig())
    removed = remove_frozen_param_groups(opt)
    assert removed
    n_groups = len(opt.param_groups)
    drop_scheduler_base_lrs(sched, n_groups)
    drop_scheduler_base_lrs(sched, n_groups)  # second call: no-op
    sched.step()
    ref_sched.step()
    assert sched.get_last_lr() == ref_sched.get_last_lr()[:n_groups]


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_boundary_resume_defers_drop_to_hook(tmp_path, legacy_metadata) -> None:
    """Resuming from a boundary checkpoint (full groups, step == freeze_step)
    must NOT drop at load: the hook drops + prunes the scheduler when the
    step runs. Dropping at load breaks the group-count assert and the
    optimizer state restore."""
    import types

    import torch

    from gslm2.checkpoint import Checkpointer
    from gslm2.training.hooks import FreezeParamGroupHook

    model = _TinyModel()
    model.freeze_step = 200_000
    model.register_buffer("current_step", torch.tensor([200_000]))
    opt, sched = build_optimizer(model, OptimizerConfig())
    assert len(opt.param_groups) == 2
    ckpt_path = tmp_path / "step_200000.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": torch.tensor([0]),
        "step": torch.tensor([200_000]),
    }
    if legacy_metadata:
        payload["supervised_epoch"] = torch.tensor([0])  # Obsolete metadata in older checkpoints is ignored.
    torch.save(payload, ckpt_path)

    fresh = _TinyModel()
    fresh.freeze_step = 200_000
    fresh.register_buffer("current_step", torch.tensor([0]))
    fresh_opt, fresh_sched = build_optimizer(fresh, OptimizerConfig())
    ckptr = Checkpointer(tmp_path, interval=1)
    ckptr.init_state(fresh, fresh_opt, fresh_sched)
    assert ckptr._load_from_path(ckpt_path)
    assert len(fresh_opt.param_groups) == 2, "boundary ckpt must keep full groups at load"

    stub = types.SimpleNamespace(optimizer=fresh_opt, scheduler=fresh_sched)
    FreezeParamGroupHook(freeze_step=200_000)(stub, 200_000, None)
    assert len(fresh_opt.param_groups) == 1
    fresh_sched.step()  # must not raise
    assert len(fresh_sched.get_last_lr()) == 1


def test_lambda_lr_survives_group_drop() -> None:
    """LambdaLR (the decay-phase scheduler) must also prune lr_lambdas.

    Regression test: pruning base_lrs alone still crashed inside
    LambdaLR.get_lr (strict zip over lr_lambdas) on the resumed run.
    """
    from torch.optim.lr_scheduler import LambdaLR

    from gslm2.optimizer import drop_scheduler_base_lrs

    model = _TinyModel()
    opt, _ = build_optimizer(model, OptimizerConfig())
    sched = LambdaLR(opt, lr_lambda=lambda epoch: 0.99**epoch)
    removed = remove_frozen_param_groups(opt)
    assert removed
    drop_scheduler_base_lrs(sched, len(opt.param_groups))
    sched.step()  # must not raise
    assert len(sched.get_last_lr()) == len(opt.param_groups)
