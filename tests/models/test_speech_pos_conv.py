"""PosConv-study guards: disable flag, study configs, front-end modules, stability keys.

R0-R7 all load and run. Covers RoPE/ALiBi helpers, streamable fronts (EMA,
local window attention, RMSNorm) including chunked-vs-full streaming
equivalence, and the study config matrix.
"""

from pathlib import Path

import pytest
import torch

from gslm2.config import SpidRConfig
from gslm2.config.config import SpeechConfig
from gslm2.models.speech.components import (
    FrontEMA,
    FrontLocalAttn,
    RMSNorm,
    SelfAttention,
    SpeechFrontEnd,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    get_encoder,
    rope_cos_sin,
)
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


def test_get_encoder_disable_pos_conv() -> None:
    _, with_conv = get_encoder(**_encoder_kwargs())
    assert with_conv.pos_conv_embed is not None

    _, without_conv = get_encoder(**_encoder_kwargs(disable_pos_conv=True))
    assert without_conv.pos_conv_embed is None
    ret, final = without_conv.get_intermediate_outputs(torch.randn(2, 12, 64))
    assert len(ret) == 2
    assert final.shape == (2, 12, 64)


def test_spidr_disable_pos_conv_matches_teacher() -> None:
    model = SpidR(_tiny_config(encoder_disable_pos_conv=True))
    assert model.student.pos_conv_embed is None
    assert model.teacher.pos_conv_embed is None
    ret, final = model.student.get_intermediate_outputs(torch.randn(2, 12, 64))
    assert len(ret) == 2
    assert final.shape == (2, 12, 64)


def test_conv_and_front_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_encoder(**_encoder_kwargs(front_ema=True))


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


def test_rope_relative_shift_invariance() -> None:
    torch.manual_seed(0)
    dim, frames = 8, 12
    q = torch.randn(1, 1, 1, dim).expand(1, 1, frames, dim).contiguous()
    k = torch.randn(1, 1, 1, dim).expand(1, 1, frames, dim).contiguous()
    cos, sin = rope_cos_sin(frames, dim, 10000.0, torch.device("cpu"), torch.float32)
    dots = torch.einsum("ahtd,ahsd->ahts", apply_rope(q, cos, sin), apply_rope(k, cos, sin))
    for m, n, s in [(0, 0, 5), (1, 4, 3), (0, 7, 2)]:
        assert dots[0, 0, m, n] == pytest.approx(dots[0, 0, m + s, n + s].item())
    assert dots[0, 0, 0, 0] != pytest.approx(dots[0, 0, 0, 5].item())


def test_rope_changes_attention_output() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 12, 32)
    torch.manual_seed(0)
    with_rope = SelfAttention(32, 4, qkv_bias=True, rope=True)
    torch.manual_seed(0)
    without_rope = SelfAttention(32, 4, qkv_bias=True)
    out_rope, out_plain = with_rope(x), without_rope(x)
    assert torch.isfinite(out_rope).all()
    assert not torch.allclose(out_rope, out_plain)


def test_rope_model_smoke() -> None:
    model = SpidR(_tiny_config(encoder_disable_pos_conv=True, encoder_rope=True))
    _, final = model.student.get_intermediate_outputs(torch.randn(2, 12, 64))
    assert torch.isfinite(final).all()


def test_alibi_bias_values() -> None:
    slopes = alibi_slopes(12)
    assert len(slopes) == 12
    assert all(s > 0 for s in slopes)
    device = torch.device("cpu")
    bias = alibi_bias(4, 10, device, torch.float32, causal=False)
    assert bias.shape == (1, 4, 10, 10)
    head_slopes = alibi_slopes(4)
    for h in (0, 3):
        for i, j in [(0, 0), (2, 7), (9, 1)]:
            assert bias[0, h, i, j].item() == pytest.approx(-head_slopes[h] * abs(i - j))
    causal = alibi_bias(4, 10, device, torch.float32, causal=True)
    assert causal.shape == (1, 4, 10, 10)
    assert torch.isinf(causal[0, :, 2, 5]).all()
    assert causal[0, 1, 5, 2].item() == pytest.approx(-head_slopes[1] * 3)


def test_alibi_attention_runs_with_padding() -> None:
    torch.manual_seed(0)
    attn = SelfAttention(32, 4, qkv_bias=True, attention_bias="alibi")
    x = torch.randn(2, 12, 32)
    assert torch.isfinite(attn(x)).all()
    mask = torch.ones(2, 1, 12, 12, dtype=torch.bool)
    mask[:, :, :, 8:] = False
    assert torch.isfinite(attn(x, attention_mask=mask)).all()


def test_packed_path_rejects_rope_and_alibi() -> None:
    x = torch.randn(1, 8, 32)
    cu_seqlens = torch.tensor([0, 8])
    with pytest.raises(RuntimeError, match="packed"):
        SelfAttention(32, 4, qkv_bias=True, rope=True).forward_packed(x, cu_seqlens, 8, 0.0)
    with pytest.raises(RuntimeError, match="packed"):
        SelfAttention(32, 4, qkv_bias=True, attention_bias="alibi").forward_packed(x, cu_seqlens, 8, 0.0)


def test_front_ema_impulse_and_streaming() -> None:
    ema = FrontEMA(4)  # init_alpha=0.5
    x = torch.zeros(1, 5, 4)
    x[0, 0, 0] = 1.0
    full, _ = ema(x)
    assert full[0, :, 0].tolist() == pytest.approx([1.0, 0.5, 0.25, 0.125, 0.0625])
    assert (full[0, :, 1:] == 0).all()
    first, state = ema(x[:, :2])
    second, _ = ema(x[:, 2:], state=state)
    assert torch.allclose(torch.cat([first, second], dim=1), full)


def test_front_local_attn_prefix_and_window() -> None:
    torch.manual_seed(0)
    front = FrontLocalAttn(32, 4, window=4, qkv_bias=True)
    front.eval()
    x = torch.randn(1, 10, 32)
    full = front(x)
    assert torch.allclose(full[:, :6], front(x[:, :6]))
    perturbed = x.clone()
    perturbed[0, 0, :] += 5.0
    assert torch.allclose(front(perturbed)[:, 4:], full[:, 4:])


def test_rms_norm_unit_rows() -> None:
    out = RMSNorm(16)(torch.randn(2, 8, 16))
    assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_front_end_chain_and_r6_smoke() -> None:
    torch.manual_seed(0)
    front = SpeechFrontEnd(
        32, norm="rmsnorm", ema=True, local_attn=True, local_attn_window=4, num_heads=4, qkv_bias=True
    )
    front.eval()
    x = torch.randn(1, 10, 32)
    out = front(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    mask = torch.ones(1, 1, 10, 10, dtype=torch.bool)
    mask[:, :, :, 7:] = False
    assert torch.isfinite(front(x, mask)).all()

    model = SpidR(
        _tiny_config(
            encoder_disable_pos_conv=True,
            encoder_rope=True,
            encoder_front_ema=True,
            encoder_front_norm="rmsnorm",
        )
    )
    assert model.student.front_end is not None
    _, final = model.student.get_intermediate_outputs(torch.randn(2, 12, 64))
    assert torch.isfinite(final).all()


def test_ssl_loss_reports_stability_keys() -> None:
    torch.manual_seed(0)
    model = SpidR(_tiny_config(encoder_disable_pos_conv=True))
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
