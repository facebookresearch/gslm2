# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from fairseq2.runtime.config_registry import ConfigRegistrar

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer


OPT_FAMILY: Final = "opt"


@dataclass(kw_only=True)
class OPTConfig:
    """Holds the configuration of a OPT model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://arxiv.org/abs/2205.01068`.
    """

    model_dim: int = 768
    """The dimensionality of the model."""

    max_seq_len: int = 2048 + 1
    """The maximum sequence length."""

    vocab_size: int = 50272
    """The size of the vocabulary."""

    pad_idx: int | None = 1
    """The index of the PAD symbol in the vocabulary."""

    attn_window_len: int = 2048
    """The local attention window length."""

    num_layers: int = 12
    """The number of decoder layers."""

    num_attn_heads: int = 12
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 12
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 3072
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


@dataclass(kw_only=True)
class OPTConfig40MShape:
    model_dim: int = 512
    num_layers: int = 6
    num_attn_heads: int = 8
    num_key_value_heads: int = 8
    ffn_inner_dim: int = 2048


@dataclass(kw_only=True)
class OPTConfig350MShape:
    model_dim: int = 1024
    num_layers: int = 24
    num_attn_heads: int = 16
    num_key_value_heads: int = 16
    ffn_inner_dim: int = 4096


@dataclass(kw_only=True)
class OPTConfig1BShape:
    model_dim: int = 2048
    num_layers: int = 24
    num_attn_heads: int = 32
    num_key_value_heads: int = 32
    ffn_inner_dim: int = 8192


@dataclass(kw_only=True)
class OPTConfig3BShape:
    model_dim: int = 2560
    num_layers: int = 32
    num_attn_heads: int = 32
    num_key_value_heads: int = 32
    ffn_inner_dim: int = 10240


@dataclass(kw_only=True)
class OPTConfig7BShape:
    model_dim: int = 4096
    num_layers: int = 32
    num_attn_heads: int = 32
    num_key_value_heads: int = 32
    ffn_inner_dim: int = 16384


@dataclass(kw_only=True)
class OPTConfig40M(OPTConfig40MShape, OPTConfig):
    pass


@dataclass(kw_only=True)
class OPTConfig350M(OPTConfig350MShape, OPTConfig):
    pass


@dataclass(kw_only=True)
class OPTConfig1B(OPTConfig1BShape, OPTConfig):
    pass


@dataclass(kw_only=True)
class OPTConfig3B(OPTConfig3BShape, OPTConfig):
    pass


@dataclass(kw_only=True)
class OPTConfig7B(OPTConfig7BShape, OPTConfig):
    pass


@dataclass(kw_only=True)
class MultiStreamOPTConfig(OPTConfig):
    """Holds the configuration of a multi-stream OPT model.

    Multi-stream OPT predicts ``len(vocab_size)`` independent token streams in parallel
    via a shared decoder + per-stream final projections, plus a small "depformer"
    decoder that autoregressively generates each stream conditioned on the shared
    decoder context.
    """

    vocab_size: list[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    """Per-stream vocabulary sizes (one entry per stream)."""

    multi_pred_loss_weight: float = 0.1
    """Weight of the multi-stream prediction loss against the single-stream loss."""

    self_distil_loss_weight: float = 1.0
    """Weight of the self-distillation KL loss within the multi-stream loss."""

    gen_loss_weight: float = 0.5
    """Weight of the generative depformer loss against the stream-specific loss."""

    distill_temperature: float = 1.0
    """Softmax temperature for the self-distillation KL loss."""

    depformer_model_dim: int = 256
    depformer_num_layers: int = 6
    depformer_num_attn_heads: int = 4
    depformer_num_key_value_heads: int = 4
    depformer_ffn_inner_dim: int = 1024

    depformer_generation_order: list[int] | None = None
    """Custom generation order for the depformer streams. A permutation of
    ``[0, ..., num_streams-1]``.  ``None`` (default) uses the natural order.
    Example: ``[5, 4, 3, 2, 1, 0]`` generates the last stream first."""


@dataclass(kw_only=True)
class MultiStreamOPTConfig350M(OPTConfig350MShape, MultiStreamOPTConfig):
    pass


@dataclass(kw_only=True)
class MultiStreamOPTConfig1B(OPTConfig1BShape, MultiStreamOPTConfig):
    pass


@dataclass(kw_only=True)
class MultiStreamOPTConfig3B(OPTConfig3BShape, MultiStreamOPTConfig):
    pass


@dataclass(kw_only=True)
class MultiStreamOPTConfig7B(OPTConfig7BShape, MultiStreamOPTConfig):
    pass


@dataclass(kw_only=True)
class MultiStreamOPTConfig4k(MultiStreamOPTConfig):
    max_seq_len: int = 4096 + 1
    attn_window_len: int = 4096


@dataclass(kw_only=True)
class MultiStreamOPTConfig350M4k(MultiStreamOPTConfig350M):
    max_seq_len: int = 4096 + 1
    attn_window_len: int = 4096


@dataclass(kw_only=True)
class MultiStreamOPTConfig1B4k(MultiStreamOPTConfig1B):
    max_seq_len: int = 4096 + 1
    attn_window_len: int = 4096


@dataclass(kw_only=True)
class MultiStreamOPTConfig3B4k(MultiStreamOPTConfig3B):
    max_seq_len: int = 4096 + 1
    attn_window_len: int = 4096


@dataclass(kw_only=True)
class MultiStreamOPTConfig7B4k(MultiStreamOPTConfig7B):
    max_seq_len: int = 4096 + 1
    attn_window_len: int = 4096


_OPT_PRESETS: dict[str, OPTConfig] = {
    "opt40m": OPTConfig40M(),
    "opt125m": OPTConfig(),
    "opt350m": OPTConfig350M(),
    "opt1b": OPTConfig1B(),
    "opt3b": OPTConfig3B(),
    "opt7b": OPTConfig7B(),
    "msopt125m": MultiStreamOPTConfig(),
    "msopt125m_4k": MultiStreamOPTConfig4k(),
    "msopt350m": MultiStreamOPTConfig350M(),
    "msopt350m_4k": MultiStreamOPTConfig350M4k(),
    "msopt1b": MultiStreamOPTConfig1B(),
    "msopt1b_4k": MultiStreamOPTConfig1B4k(),
    "msopt3b": MultiStreamOPTConfig3B(),
    "msopt3b_4k": MultiStreamOPTConfig3B4k(),
    "msopt7b": MultiStreamOPTConfig7B(),
    "msopt7b_4k": MultiStreamOPTConfig7B4k(),
}


def get_opt_config_preset(arch: str) -> OPTConfig:
    if arch not in _OPT_PRESETS:
        raise ValueError(f"Unknown OPT preset: '{arch}'. Available presets: {list(_OPT_PRESETS.keys())}")
    return replace(_OPT_PRESETS[arch])


def register_opt_configs(container: DependencyContainer) -> None:  # noqa: C901
    arch = ConfigRegistrar(container, OPTConfig)

    @arch("opt40m")
    def _opt40m() -> OPTConfig:
        return OPTConfig40M()

    @arch("opt125m")
    def _opt125m() -> OPTConfig:
        return OPTConfig()

    @arch("opt350m")
    def _opt350m() -> OPTConfig:
        return OPTConfig350M()

    @arch("opt1b")
    def _opt1b() -> OPTConfig:
        return OPTConfig1B()

    @arch("opt3b")
    def _opt3b() -> OPTConfig:
        return OPTConfig3B()

    @arch("opt7b")
    def _opt7b() -> OPTConfig:
        return OPTConfig7B()

    @arch("msopt125m")
    def _msopt125m() -> OPTConfig:
        return MultiStreamOPTConfig()

    @arch("msopt125m_4k")
    def _msopt125m_4k() -> OPTConfig:
        return MultiStreamOPTConfig4k()

    @arch("msopt350m")
    def _msopt350m() -> OPTConfig:
        return MultiStreamOPTConfig350M()

    @arch("msopt350m_4k")
    def _msopt350m_4k() -> OPTConfig:
        return MultiStreamOPTConfig350M4k()

    @arch("msopt1b")
    def _msopt1b() -> OPTConfig:
        return MultiStreamOPTConfig1B()

    @arch("msopt1b_4k")
    def _msopt1b_4k() -> OPTConfig:
        return MultiStreamOPTConfig1B4k()

    @arch("msopt3b")
    def _msopt3b() -> OPTConfig:
        return MultiStreamOPTConfig3B()

    @arch("msopt3b_4k")
    def _msopt3b_4k() -> OPTConfig:
        return MultiStreamOPTConfig3B4k()

    @arch("msopt7b")
    def _msopt7b() -> OPTConfig:
        return MultiStreamOPTConfig7B()

    @arch("msopt7b_4k")
    def _msopt7b_4k() -> OPTConfig:
        return MultiStreamOPTConfig7B4k()
