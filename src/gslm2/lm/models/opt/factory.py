# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    LocalAttentionStateFactory,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerNormOrder,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    StandardTransformerLMDecoderLayer,
    TransformerLM,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    StandardEmbedding,
    StandardLayerNorm,
)
from torch import nn

from gslm2.lm.models.opt.multi_stream_model import TransformerLMMultiStream
from gslm2.lm.models.opt.packed_sdpa import create_packed_aware_sdpa
from gslm2.lm.runtime.nn.position_encoder import LearnedPositionEncoder

if TYPE_CHECKING:
    from gslm2.lm.models.opt.config import OPTConfig


def create_opt_model(config: OPTConfig) -> TransformerLM:
    return OPTFactory(config).create_model()


class OPTFactory:
    def __init__(self, config: OPTConfig) -> None:
        self._config = config

    def create_model(self) -> TransformerLM:
        config = self._config

        if isinstance(config.vocab_size, list):
            return self.create_multi_stream_model()

        decoder_frontend = self.create_decoder_frontend()

        decoder = self.create_decoder()

        final_proj = self.create_final_projection()

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_seq_len,
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        config = self._config

        embed = self.create_embedding()

        pos_encoder = self.create_position_encoder()

        return TransformerEmbeddingFrontend(
            config.model_dim,
            embed,
            pos_encoder=pos_encoder,
            no_scale=True,
            # dropout_p=config.dropout_p,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(config.vocab_size, config.model_dim, config.pad_idx)

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        layers = []

        for _ in range(config.num_layers):
            layer = self.create_decoder_layer()

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        return LearnedPositionEncoder(config.model_dim, config.max_seq_len, _legacy_pad_idx=1)

    def create_decoder_layer(self) -> TransformerLMDecoderLayer:
        config = self._config

        self_attn = self.create_self_attention()

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn()

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias(attn_window_len=config.attn_window_len)

        sdpa = create_packed_aware_sdpa(attn_bias)

        state_factory = LocalAttentionStateFactory(config.attn_window_len)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            bias=True,
            state_factory=state_factory,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        return StandardFeedForwardNetwork(config.model_dim, config.ffn_inner_dim, bias=True)

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        return StandardLayerNorm(config.model_dim, bias=True)

    def create_final_projection(self) -> Projection:
        config = self._config

        return Linear(
            config.model_dim,
            config.vocab_size,
            bias=False,
            init_fn=_init_final_projection,
        )

    def create_multi_stream_model(self) -> TransformerLMMultiStream:
        config = self._config

        # All stream frontends share a single position encoder (positions are shared
        # across streams since they operate on aligned timesteps).
        pos_encoder = LearnedPositionEncoder(config.model_dim, config.max_seq_len, _legacy_pad_idx=1)
        decoder_frontend = nn.ModuleList(
            [
                TransformerEmbeddingFrontend(
                    config.model_dim,
                    StandardEmbedding(vocab_size, config.model_dim, config.pad_idx),
                    pos_encoder=pos_encoder,
                    no_scale=True,
                )
                for vocab_size in config.vocab_size
            ]
        )

        decoder = self.create_decoder()

        final_proj = nn.ModuleList(
            [
                Linear(
                    config.model_dim,
                    vocab_size,
                    bias=False,
                    init_fn=_init_final_projection,
                )
                for vocab_size in config.vocab_size
            ]
        )

        # Depformer (generative head): no position encoder.
        # With default generation order the last stream is never used as input,
        # so we only need S-1 frontends. With a custom order any stream could be
        # non-terminal, so we create all S frontends.
        depformer_decoder_frontend = nn.ModuleList(
            [
                TransformerEmbeddingFrontend(
                    config.depformer_model_dim,
                    StandardEmbedding(vocab_size, config.depformer_model_dim, config.pad_idx),
                    pos_encoder=None,
                    no_scale=True,
                )
                for vocab_size in config.vocab_size
            ]
        )
        if config.depformer_generation_order is None:
            depformer_decoder_frontend = depformer_decoder_frontend[:-1]
        depformer_decoder = self.create_depformer_decoder()
        depformer_final_proj = nn.ModuleList(
            [
                Linear(
                    config.depformer_model_dim,
                    vocab_size,
                    bias=False,
                    init_fn=_init_final_projection,
                )
                for vocab_size in config.vocab_size
            ]
        )

        return TransformerLMMultiStream(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_seq_len,
            multi_pred_loss_weight=config.multi_pred_loss_weight,
            self_distil_loss_weight=config.self_distil_loss_weight,
            gen_loss_weight=config.gen_loss_weight,
            distill_temperature=config.distill_temperature,
            depformer_decoder_frontend=depformer_decoder_frontend,
            depformer_decoder=depformer_decoder,
            depformer_final_proj=depformer_final_proj,
            depformer_generation_order=config.depformer_generation_order,
        )

    def _build_decoder_layer(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout_p: float,
    ) -> StandardTransformerLMDecoderLayer:
        """Construct a causal decoder layer for the depformer."""
        sdpa = create_packed_aware_sdpa(CausalAttentionBias())
        self_attn = StandardMultiheadAttention(
            model_dim,
            num_heads,
            sdpa,
            num_key_value_heads=num_kv_heads,
            bias=True,
        )
        self_attn_layer_norm = StandardLayerNorm(model_dim, bias=True)

        ffn = StandardFeedForwardNetwork(model_dim, ffn_dim, bias=True)
        ffn_layer_norm = StandardLayerNorm(model_dim, bias=True)

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            dropout_p=dropout_p,
        )

    def create_depformer_decoder(self) -> TransformerLMDecoder:
        config = self._config

        layers = [
            self._build_decoder_layer(
                model_dim=config.depformer_model_dim,
                num_heads=config.depformer_num_attn_heads,
                num_kv_heads=config.depformer_num_key_value_heads,
                ffn_dim=config.depformer_ffn_inner_dim,
                dropout_p=config.dropout_p,
            )
            for _ in range(config.depformer_num_layers)
        ]

        layer_norm = StandardLayerNorm(config.depformer_model_dim, bias=True)

        return StandardTransformerLMDecoder(layers, layer_norm)


def _init_final_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, std=proj.input_dim**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)
