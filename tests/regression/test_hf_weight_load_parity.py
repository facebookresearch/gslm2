"""HF-weight-load parity: converting a HuggingFace checkpoint through ``interop``
and loading it into the in-house model must reproduce the HF model's logits.

Validates the OPT / GPT-2 weight converters (name mapping, Conv1D
transposes, q/k/v split) against the reference implementations. CPU, fp32.
"""

from __future__ import annotations

import torch

from gslm2.config.models.core import CausalLMConfig
from gslm2.models.core.interop import convert_opt_state_dict
from gslm2.models.core.opt import OPT

_V, _D, _L, _H, _P = 128, 64, 2, 4, 40  # vocab, dim, layers, heads, max positions
# tied models list lm_head.weight and token_embed.weight as separate state_dict keys
# for the same tensor; a converter that supplies only the embedding leaves the head
# "missing" but correctly tied. current_step is a non-weight buffer.
_ALLOWED_MISSING = {"current_step", "lm_head.weight"}


def _assert_logits_match(hf_model: torch.nn.Module, ours: torch.nn.Module, tokens: torch.Tensor) -> None:
    with torch.no_grad():
        hf_logits = hf_model(tokens).logits
        our_logits = ours(tokens)
    torch.testing.assert_close(our_logits, hf_logits, atol=1e-4, rtol=1e-3)


def _check_load(ours: torch.nn.Module, renamed: dict, unmatched: list[str]) -> None:
    result = ours.load_state_dict(renamed, strict=False)
    assert not unmatched, f"unmapped HF keys: {unmatched}"
    assert set(result.missing_keys) <= _ALLOWED_MISSING, f"missing: {result.missing_keys}"
    assert not result.unexpected_keys, f"unexpected: {result.unexpected_keys}"


def test_gpt2_hf_load_parity() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    hf = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=_V,
            n_positions=_P,
            n_embd=_D,
            n_layer=_L,
            n_head=_H,
            n_inner=4 * _D,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    ours = OPT(
        CausalLMConfig(
            vocab_size=_V,
            max_seq_len=_P,
            model_dim=_D,
            num_layers=_L,
            num_attn_heads=_H,
            num_key_value_heads=_H,
            ffn_inner_dim=4 * _D,
            dropout_p=0.0,
            ffn_activation="gelu_new",
            tie_word_embeddings=True,
            pos_embed_pad_offset=0,
            attn_window_len=None,
        )
    )
    renamed, unmatched = convert_opt_state_dict(hf.state_dict(), source="gpt2_hf")
    _check_load(ours, renamed, unmatched)
    _assert_logits_match(hf, ours, torch.randint(0, _V, (2, 16)))


def test_opt_hf_load_parity() -> None:
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(0)
    hf = OPTForCausalLM(
        OPTConfig(
            vocab_size=_V,
            max_position_embeddings=_P,
            hidden_size=_D,
            word_embed_proj_dim=_D,
            num_hidden_layers=_L,
            num_attention_heads=_H,
            ffn_dim=4 * _D,
            dropout=0.0,
            do_layer_norm_before=True,
        )
    )
    ours = OPT(
        CausalLMConfig(
            vocab_size=_V,
            max_seq_len=_P,
            model_dim=_D,
            num_layers=_L,
            num_attn_heads=_H,
            num_key_value_heads=_H,
            ffn_inner_dim=4 * _D,
            dropout_p=0.0,
            ffn_activation="relu",
            tie_word_embeddings=True,
            pos_embed_pad_offset=2,
            attn_window_len=None,
        )
    )
    renamed, unmatched = convert_opt_state_dict(hf.state_dict(), source="hf")
    _check_load(ours, renamed, unmatched)
    _assert_logits_match(hf, ours, torch.randint(0, _V, (2, 16)))
