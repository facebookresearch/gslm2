"""Language model configs (causal decoder-only, single- and multi-stream)."""

from dataclasses import dataclass, replace
from typing import Literal

from gslm2.config.models.base import BaseModelConfig


@dataclass(frozen=True)
class CausalLMConfig(BaseModelConfig):
    """Decoder-only causal language model config (OPT / GPT-2 style)."""

    vocab_size: int = 50272
    model_dim: int = 768
    # HF OPT: embed_positions table = max_position_embeddings (2048) + pad_offset (2) = 2050.
    # LearnedPositionEmbedding builds max_seq_len + pos_embed_pad_offset rows, so 2048 + 2 = 2050.
    max_seq_len: int = 2048
    pad_idx: int | None = 1

    num_layers: int = 12
    num_attn_heads: int = 12
    num_key_value_heads: int = 12
    ffn_inner_dim: int = 3072

    attn_window_len: int | None = None
    dropout_p: float = 0.1

    ffn_activation: str = "relu"
    ffn_post_dropout: bool = False

    tie_word_embeddings: bool = False
    init_std: float = 0.02
    pos_embed_pad_offset: int = 2
    # "standard": GPT-NeoX/GPT-2 init (trunc_normal(init_std), per-layer residual rescaling,
    # tied head when tie_word_embeddings). "fairseq2": replicate the fairseq2 OPT init
    # (kaiming-uniform linears, N(0,1) embeddings, untied N(0, dim^-0.5) head, no rescaling)
    # for strict from-scratch parity with the fairseq2 reference — pair with
    # tie_word_embeddings=False.
    init_scheme: Literal["standard", "fairseq2"] = "standard"


# HF OPT/GPT-2 tie the input embedding and output head; set it on the presets so
# from-scratch training matches (the CausalLMConfig default stays False so custom
# configs — e.g. the SLM's LM — are unaffected).
_CAUSAL_PRESETS: dict[str, CausalLMConfig] = {
    "opt-40m": CausalLMConfig(
        model_dim=512,
        num_layers=6,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=2048,
        tie_word_embeddings=True,
    ),
    "opt-125m": CausalLMConfig(tie_word_embeddings=True),
    "opt-350m": CausalLMConfig(
        model_dim=1024,
        num_layers=24,
        num_attn_heads=16,
        num_key_value_heads=16,
        ffn_inner_dim=4096,
        tie_word_embeddings=True,
    ),
    "opt-1b": CausalLMConfig(
        model_dim=2048,
        num_layers=24,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=8192,
        tie_word_embeddings=True,
    ),
    "gpt2": CausalLMConfig(
        vocab_size=50257,
        max_seq_len=1024,
        pad_idx=None,
        ffn_activation="gelu_new",
        tie_word_embeddings=True,
        pos_embed_pad_offset=0,
    ),
    "gpt2-medium": CausalLMConfig(
        vocab_size=50257,
        model_dim=1024,
        max_seq_len=1024,
        pad_idx=None,
        num_layers=24,
        num_attn_heads=16,
        num_key_value_heads=16,
        ffn_inner_dim=4096,
        ffn_activation="gelu_new",
        tie_word_embeddings=True,
        pos_embed_pad_offset=0,
    ),
    "gpt2-large": CausalLMConfig(
        vocab_size=50257,
        model_dim=1280,
        max_seq_len=1024,
        pad_idx=None,
        num_layers=36,
        num_attn_heads=20,
        num_key_value_heads=20,
        ffn_inner_dim=5120,
        ffn_activation="gelu_new",
        tie_word_embeddings=True,
        pos_embed_pad_offset=0,
    ),
}


def get_causal_lm_preset(name: str) -> CausalLMConfig:
    if name not in _CAUSAL_PRESETS:
        raise ValueError(f"Unknown causal LM preset: {name!r}. Available: {list(_CAUSAL_PRESETS)}")
    return replace(_CAUSAL_PRESETS[name])


@dataclass(frozen=True)
class MultiStreamCausalLMConfig(BaseModelConfig):
    """Multi-stream causal LM (OPT-style).

    Predicts ``len(vocab_sizes)`` aligned token streams in parallel through one shared
    decoder + per-stream input embeddings and output projections, plus a small
    autoregressive "depformer" that generates the per-timestep streams conditioned on the
    shared decoder context. Ports the fairseq2 ``MultiStreamOPTConfig`` to the pure-PyTorch
    stack; ``vocab_sizes`` replaces the fairseq2 ``vocab_size: int | list`` overloading.
    """

    # Per-stream vocab sizes (one entry per stream); its length sets the stream count. The default
    # is the SpidR 5-codebook setup (256 units per codebook layer, matching the fairseq2 msopt recipe).
    vocab_sizes: tuple[int, ...] = (256, 256, 256, 256, 256)
    model_dim: int = 768
    max_seq_len: int = 2048
    pad_idx: int | None = 1

    num_layers: int = 12
    num_attn_heads: int = 12
    num_key_value_heads: int = 12
    ffn_inner_dim: int = 3072

    attn_window_len: int | None = None
    dropout_p: float = 0.1
    ffn_activation: str = "relu"
    ffn_post_dropout: bool = False
    init_std: float = 0.02
    pos_embed_pad_offset: int = 2
    # Weight initialisation scheme. "fairseq2": N(0, 1) embeddings, xavier-uniform attention projections
    # and kaiming-uniform feed-forward projections, with no residual rescaling. "standard": the GPT-2
    # style trunc_normal(init_std) throughout, with per-layer residual rescaling on both the backbone and
    # the depformer. The output heads scale with their input width under either scheme.
    init_scheme: Literal["standard", "fairseq2"] = "fairseq2"

    # Loss blend (mirrors the fairseq2 multi-stream OPT).
    multi_pred_loss_weight: float = 0.1
    self_distil_loss_weight: float = 1.0
    gen_loss_weight: float = 0.5
    distill_temperature: float = 1.0

    # Depformer (generative head).
    depformer_model_dim: int = 256
    depformer_num_layers: int = 6
    depformer_num_attn_heads: int = 4
    depformer_num_key_value_heads: int = 4
    depformer_ffn_inner_dim: int = 1024
    # A permutation of the stream indices, or None for natural order [0, ..., num_streams-1].
    depformer_generation_order: tuple[int, ...] | None = None

    @property
    def num_streams(self) -> int:
        return len(self.vocab_sizes)

    def backbone_config(self) -> CausalLMConfig:
        """Shared-decoder shape as a :class:`CausalLMConfig` (for reusing the OPT decoder
        blocks). ``vocab_size`` is a placeholder — the multi-stream model owns the per-stream
        embeddings and heads, not the backbone's."""
        return CausalLMConfig(
            vocab_size=max(self.vocab_sizes),
            model_dim=self.model_dim,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_idx,
            num_layers=self.num_layers,
            num_attn_heads=self.num_attn_heads,
            num_key_value_heads=self.num_key_value_heads,
            ffn_inner_dim=self.ffn_inner_dim,
            attn_window_len=self.attn_window_len,
            dropout_p=self.dropout_p,
            ffn_activation=self.ffn_activation,
            ffn_post_dropout=self.ffn_post_dropout,
            init_std=self.init_std,
            pos_embed_pad_offset=self.pos_embed_pad_offset,
        )


# Shared decoder shapes reuse the causal presets; depformer defaults follow the fairseq2 msopt*.
_MULTI_STREAM_PRESETS: dict[str, MultiStreamCausalLMConfig] = {
    "msopt-125m": MultiStreamCausalLMConfig(),
    "msopt-350m": MultiStreamCausalLMConfig(
        model_dim=1024, num_layers=24, num_attn_heads=16, num_key_value_heads=16, ffn_inner_dim=4096
    ),
    "msopt-1b": MultiStreamCausalLMConfig(
        model_dim=2048, num_layers=24, num_attn_heads=32, num_key_value_heads=32, ffn_inner_dim=8192
    ),
    "msopt-125m-4k": MultiStreamCausalLMConfig(max_seq_len=4096, attn_window_len=4096),
    "msopt-350m-4k": MultiStreamCausalLMConfig(
        model_dim=1024,
        num_layers=24,
        num_attn_heads=16,
        num_key_value_heads=16,
        ffn_inner_dim=4096,
        max_seq_len=4096,
        attn_window_len=4096,
    ),
    "msopt-1b-4k": MultiStreamCausalLMConfig(
        model_dim=2048,
        num_layers=24,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=8192,
        max_seq_len=4096,
        attn_window_len=4096,
    ),
}


def get_multi_stream_lm_preset(name: str) -> MultiStreamCausalLMConfig:
    if name not in _MULTI_STREAM_PRESETS:
        raise ValueError(f"Unknown multi-stream LM preset: {name!r}. Available: {list(_MULTI_STREAM_PRESETS)}")
    return replace(_MULTI_STREAM_PRESETS[name])
