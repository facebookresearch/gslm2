"""Canonical SpidR positional convolution and SSL metric guards."""

from pathlib import Path

import torch

from gslm2.config import SpidRConfig
from gslm2.config.config import SpeechConfig
from gslm2.models.speech.components import get_encoder
from gslm2.models.speech.spidr import SpidR

CANONICAL_ENCODER_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "encoder" / "spidr2.0-ls960.yaml"


def _tiny_config(**overrides: bool | str | float) -> SpidRConfig:
    kwargs = {
        "encoder_embed_dim": 64,
        "encoder_num_layers": 2,
        "encoder_num_heads": 4,
        "encoder_ff_interm_features": 128,
        "encoder_pos_conv_kernel": 7,
        "encoder_pos_conv_groups": 8,
        "encoder_pos_conv_depth": 1,
        "encoder_dropout": 0.0,
        "encoder_attention_dropout": 0.0,
        "encoder_layer_drop": 0.0,
        "num_codebooks": 2,
        "codebook_size": 16,
        "teacher_use_heads": True,
        "use_soft_targets": True,
    }
    kwargs.update(overrides)
    return SpidRConfig(**kwargs)


def _encoder_kwargs(**overrides: bool | str | float | None) -> dict:
    kwargs: dict = {
        "in_features": 32,
        "embed_dim": 64,
        "pos_conv_kernel": 7,
        "pos_conv_groups": 8,
        "pos_conv_depth": 1,
        "num_layers": 2,
        "num_heads": 4,
        "attention_dropout": 0.0,
        "ff_interm_features": 128,
        "ff_interm_dropout": 0.0,
        "dropout": 0.0,
        "layer_drop": 0.0,
        "layer_norm_first": True,
        "qkv_bias": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_get_encoder_uses_pos_conv() -> None:
    _, with_conv = get_encoder(**_encoder_kwargs())
    assert with_conv.pos_conv_embed is not None

    ret, final = with_conv.get_intermediate_outputs(torch.randn(2, 12, 64))
    assert len(ret) == 2
    assert final.shape == (2, 12, 64)


def test_canonical_encoder_config_loads(tmp_path: Path) -> None:
    patched = tmp_path / "spidr2.0-ls960.yaml"
    patched.write_text(CANONICAL_ENCODER_CONFIG.read_text().replace("/path/to/runs", str(tmp_path)))
    cfg = SpeechConfig.from_file(patched)
    assert cfg.run.model_type == "spidr"
    assert cfg.run.random_seed == 0
    assert cfg.optimizer.max_steps == 400000
    assert cfg.masking.mask_prob == 0.8
    assert cfg.model.regularisation.reg_teacher is True
    assert cfg.model.regularisation.reg_student is False
    assert cfg.model.teacher_use_heads is True
    assert cfg.model.use_soft_targets is True


def test_ssl_loss_reports_stability_keys() -> None:
    torch.manual_seed(0)
    model = SpidR(_tiny_config())
    model.eval()
    batch, frames, dim = 2, 16, 64
    student_outputs = [torch.randn(batch, frames, dim) for _ in range(2)]
    mask = torch.zeros(batch, frames, dtype=torch.bool)
    mask[:, ::4] = True
    mask_indices = torch.nonzero(mask, as_tuple=True)
    targets = [torch.randn(int(mask.sum()), dim) for _ in range(2)]
    loss, loss_dict = model.get_ssl_loss(student_outputs, torch.randn(batch, frames, dim), targets, mask_indices)
    assert torch.isfinite(loss).all()
    for key in (
        "target_ppl",
        "pred_ppl",
        "ssl_loss_no_reg",
        "student_predictive_entropy",
        "student_codebook_entropy",
        "teacher_predictive_entropy",
        "teacher_codebook_entropy",
    ):
        assert key in loss_dict, f"missing stability key {key!r}"
        assert torch.isfinite(loss_dict[key]).all()
