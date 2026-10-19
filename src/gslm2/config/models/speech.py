"""SpidR 2.0 encoder configuration."""

from dataclasses import dataclass, field
from typing import Literal

from gslm2.config.models.base import BaseModelConfig, build_nested_config


@dataclass(frozen=True)
class RegularisationConfig:
    """Entropy regularisation configuration for SpidR models."""

    # Which components to regularise (add entropy term to loss)
    reg_student: bool = False
    reg_teacher: bool = False

    # Predictive Entropy: H(P(Z|Y)) - entropy of individual predictions
    predictive_entropy_weight: float = 0.0
    predictive_entropy_schedule: Literal[
        "constant", "constant_then_exponential_decrease", "constant_then_exponential_increase"
    ] = "constant"
    predictive_entropy_constant_steps: int = 20_000
    predictive_entropy_max_steps: int = 350_000
    predictive_entropy_timescale: float = 0.5
    predictive_entropy_final_weight: float = 0.0

    # Codebook Entropy: H(E[Z|Y]) - entropy of mean probabilities (encourages uniform codebook usage)
    codebook_entropy_weight: float = 0.0
    codebook_entropy_schedule: Literal[
        "constant", "constant_then_exponential_decrease", "constant_then_exponential_increase"
    ] = "constant"
    codebook_entropy_constant_steps: int = 20_000
    codebook_entropy_max_steps: int = 350_000
    codebook_entropy_timescale: float = 0.1
    codebook_entropy_final_weight: float = 0.001


@dataclass(frozen=True)
class SpidRConfig(BaseModelConfig):
    """SpidR 2.0 encoder configuration. Corresponding names in `fairseq` in comments."""

    extractor_mode: Literal["group_norm", "layer_norm"] = "layer_norm"  # model.extractor_mode
    extractor_conv_bias: bool = False  # model.conv_bias
    extractor_conv_layer_config: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    )  # model.conv_feature_layers
    extractor_channels_last: bool = True  # NHWC extractor, LayerNorm transposes become views
    encoder_embed_dim: int = 768  # model.encoder_embed_dim
    encoder_projection_dropout: float = 0  # model.dropout_input
    encoder_pos_conv_kernel: int = 95  # model.conv_pos
    encoder_pos_conv_groups: int = 16  # model.conv_pos_groups
    encoder_pos_conv_depth: int = 5  # model.pos_conv_depth
    encoder_num_layers: int = 12  # model.encoder_layers
    encoder_num_heads: int = 12  # model.encoder_attention_heads
    encoder_attention_dropout: float = 0.1  # model.attention_dropout
    encoder_ff_interm_features: int = 3072  # model.encoder_ffn_embed_dim
    encoder_ff_interm_dropout: float = 0.0  # model.activation_dropout
    encoder_dropout: float = 0.1  # model.dropout
    encoder_layer_norm_first: bool = False  # model.layer_norm_first
    encoder_layer_drop: float = 0.0  # model.encoder_layerdrop
    encoder_qkv_bias: bool = False  # WARNING: not in `fairseq`, always True there.
    feature_grad_mult: float = 1.0  # model.feature_grad_mult

    codebook_size: int = 256  # model.codebook_size
    codebook_decay: float = 0.9  # model.codebook_decay
    num_codebooks: int = 8  # model.average_top_k_layers
    ema_start_decay: float = 0.999  # model.ema_decay
    ema_final_decay: float = 0.9999  # model.ema_end_decay
    ema_final_step: int = 30_000  # model.ema_anneal_end_step
    ema_exclude_layers: list[str] = field(default_factory=lambda: ["pos_conv_embed"])
    ema_timescale: float = 20_000
    ema_threshold: float = 1e-7
    ema_final: float = 1.0
    freeze_step: int = 200_000  # model.freeze_teacher_step
    unfreeze_encoder_step: int = 0

    # Teacher target mode
    teacher_use_heads: bool = False  # If True, use heads for teacher targets instead of codebooks
    use_soft_targets: bool = (
        False  # If True (requires teacher_use_heads=True), use soft teacher probs instead of one-hot
    )

    # Heads architecture
    heads_layer_norm: Literal["none", "center", "affine"] = "none"  # LayerNorm before linear layer
    heads_use_weight_norm: bool = False  # If True, L2-normalize linear layer weights
    heads_init: Literal["default", "orthogonal", "sphere"] = "default"  # Weight initialization strategy

    # Entropy regularisation
    regularisation: RegularisationConfig | None = None

    def __post_init__(self) -> None:
        if self.feature_grad_mult != 1.0:
            raise ValueError("Feature gradient multiplier is not supported.")
        if self.use_soft_targets and not self.teacher_use_heads:
            raise ValueError("Soft targets heads on the teacher side is not supported.")
        if isinstance(self.regularisation, dict):
            object.__setattr__(self, "regularisation", build_nested_config(RegularisationConfig, self.regularisation))
        elif self.regularisation is None:
            object.__setattr__(self, "regularisation", RegularisationConfig())
