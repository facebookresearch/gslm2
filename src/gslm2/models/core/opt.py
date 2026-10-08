"""Decoder-only causal language model (OPT-style): learned absolute positions, pre-norm blocks, GQA + optional local-window attention, configurable FFN."""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from gslm2.config.models.core import CausalLMConfig
from gslm2.models.core.interop import convert_opt_state_dict
from gslm2.models.modules.mlp import MLP
from gslm2.models.modules.positional_embeddings import LearnedPositionEmbedding
from gslm2.models.modules.transformer import MultiheadSelfAttention, flex_backward_unsupported, trunc_normal_

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


_IGNORE_INDEX = -100  # target value ignored by cross-entropy (padding / doc boundaries)

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,  # exact erf GELU
    "gelu_new": partial(nn.GELU, approximate="tanh"),  # HF GPT-2 "gelu_new" (tanh approx)
}


class DecoderLayer(nn.Module):
    def __init__(self, cfg: CausalLMConfig) -> None:
        super().__init__()
        if cfg.ffn_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown ffn_activation {cfg.ffn_activation!r}. Options: {list(_ACTIVATIONS)}")
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.attn = MultiheadSelfAttention(
            cfg.model_dim,
            cfg.num_attn_heads,
            cfg.num_key_value_heads,
            dropout=cfg.dropout_p,
            causal=True,
            attn_window_len=cfg.attn_window_len,
        )
        self.norm2 = nn.LayerNorm(cfg.model_dim)
        # The shared timm-style MLP applies dropout after the activation; the residual
        # `self.dropout` in forward() adds another on the FFN output. HF GPT-2 applies a
        # single dropout, on the FFN output only.
        self.ffn = MLP(
            cfg.model_dim,
            hidden_features=cfg.ffn_inner_dim,
            act_layer=_ACTIVATIONS[cfg.ffn_activation],
            drop=cfg.dropout_p,
            post_drop=cfg.ffn_post_dropout,
        )
        self.dropout = nn.Dropout(cfg.dropout_p)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | BlockMask | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask=attention_mask))
        return x + self.dropout(self.ffn(self.norm2(x)))


class OPT(nn.Module):
    """Decoder-only causal LM (OPT-style)."""

    def __init__(self, cfg: CausalLMConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CausalLMConfig()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.model_dim, padding_idx=cfg.pad_idx)
        self.pos_embed = LearnedPositionEmbedding(cfg.max_seq_len, cfg.model_dim, cfg.pos_embed_pad_offset)
        self.embed_dropout = nn.Dropout(cfg.dropout_p)

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)

        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.token_embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=cfg.model_dim**-0.5)
        if cfg.pad_idx is not None:
            with torch.no_grad():
                self.token_embed.weight[cfg.pad_idx].zero_()
        if cfg.init_scheme == "standard":
            self._rescale_blocks()
        else:  # "fairseq2": overwrite with a fresh fairseq2 OPT's init for strict parity runs
            self._init_from_fairseq2()

        self.register_buffer("current_step", torch.zeros(1, dtype=torch.int64))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=self.cfg.init_std)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _rescale_blocks(self) -> None:
        # Scale each block's output projections by 1/sqrt(2*(layer_id+1)) (GPT-NeoX /
        # Megatron residual init). HF GPT-2 uses a constant 1/sqrt(2*n_layer). Applies at
        # init only; a loaded checkpoint overwrites these weights.
        for layer_id, layer in enumerate(self.layers):
            layer.attn.out_proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.ffn.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_from_fairseq2(self) -> None:
        """Re-initialise the weights under the ``"fairseq2"`` scheme (no fairseq2 dependency).

        Token and position embeddings are N(0, 1). Attention projections are xavier-uniform,
        with gain ``2**-0.5`` on query, key and value and ``1.0`` on the output. The
        feed-forward projections keep PyTorch's default kaiming-uniform bound (via
        ``Linear.reset_parameters``). No residual rescaling is applied; the untied output head
        keeps the ``dim**-0.5`` scaling set in ``__init__``.
        """
        if self.cfg.tie_word_embeddings:
            raise ValueError("init_scheme='fairseq2' requires tie_word_embeddings=False (the head is untied)")
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=1.0)
        if self.cfg.pad_idx is not None:
            with torch.no_grad():
                self.token_embed.weight[self.cfg.pad_idx].zero_()
        nn.init.normal_(self.pos_embed.weight.weight, mean=0.0, std=1.0)
        for layer in self.layers:
            for proj in (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj):
                nn.init.xavier_uniform_(proj.weight, gain=2**-0.5)
            nn.init.xavier_uniform_(layer.attn.out_proj.weight)
            for proj in (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj, layer.attn.out_proj):
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            layer.ffn.fc1.reset_parameters()
            layer.ffn.fc2.reset_parameters()

    def load_pretrained(
        self,
        state_dict: dict[str, torch.Tensor],
        source: str = "auto",
        *,
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        renamed, unmatched = convert_opt_state_dict(state_dict, source=source)  # type: ignore[arg-type]
        if unmatched:
            logger.warning("OPT.load_pretrained: %d unmatched key(s) dropped: %s", len(unmatched), unmatched)
        return self.load_state_dict(renamed, strict=strict)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        t = tokens.size(1)
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}.")

        # With position_ids=None, positions are arange(t) (+ pad offset). HF OPT instead
        # derives them from the attention mask (cumsum(mask)-1), which differs for
        # left-padded inputs; pass explicit position_ids for that case.
        x = self.token_embed(tokens) + self.pos_embed(t, tokens.device, position_ids=position_ids)
        x = self.embed_dropout(x)

        if doc_ids is not None:
            # Packed path. Training uses a dense block mask through SDPA so attention dropout
            # (dropout_p) is applied exactly as on the unpacked path; flex_attention takes no
            # dropout argument. Eval (dropout off) uses the flex BlockMask, which is faster,
            # except where flex has no backward and would refuse to run at all.
            use_dense = self.training or flex_backward_unsupported(tokens.device)
            attn_mask: torch.Tensor | BlockMask | None = (
                self._build_block_mask_dense(doc_ids) if use_dense else self._build_block_mask(doc_ids)
            )
        elif attention_mask is not None:
            attn_mask = self._build_attn_mask(attention_mask, t, tokens.device)
        else:
            attn_mask = None

        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        target_flat = targets.reshape(-1)
        if (target_flat != _IGNORE_INDEX).any():
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), target_flat, ignore_index=_IGNORE_INDEX)
        else:
            # No supervised targets (e.g. a packed bin of only single-token docs):
            # cross_entropy would divide by zero → NaN. Return a grad-carrying zero.
            loss = logits.sum() * 0.0
        return loss, {"nll_loss": loss.detach(), "perplexity": loss.detach().clamp(max=20).exp()}

    @staticmethod
    def _build_block_mask(doc_ids: torch.Tensor) -> BlockMask:
        """Build a flex_attention BlockMask from per-token document IDs."""

        def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001, ANN202, ARG001
            same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
            not_pad = doc_ids[b, q_idx] >= 0
            return same_doc & not_pad & (q_idx >= kv_idx)

        B, T = doc_ids.shape  # noqa: N806
        # ``_compile=True`` compiles the mask construction (much faster than the
        # eager grid build); gated to CUDA so CPU tests stay eager and quick.
        return create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=doc_ids.device, _compile=doc_ids.is_cuda
        )

    @staticmethod
    def _build_block_mask_dense(doc_ids: torch.Tensor) -> torch.Tensor:
        """Dense boolean block mask ``[B, 1, T, T]`` equivalent to :meth:`_build_block_mask`.

        SDPA (unlike ``flex_attention``) applies attention dropout via ``dropout_p``, so the packed
        path routes through SDPA with this dense mask during training to be regularised identically
        to the unpacked path. A padding-query row (``doc_id < 0``) is fully masked, which makes SDPA
        emit NaN (``flex_attention`` returns 0 for an empty row); the identity term keeps every such
        row attending to itself so no row is empty. Real queries never attend to padding and the
        identity is already set for real queries, so the supervised logits are unchanged; padding
        outputs are discarded (their targets are ``-100``).
        """
        _, t = doc_ids.shape
        same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
        not_pad = (doc_ids >= 0)[:, :, None]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=doc_ids.device))
        eye = torch.eye(t, dtype=torch.bool, device=doc_ids.device)
        return ((same_doc & not_pad & causal) | eye)[:, None, :, :]

    @staticmethod
    def _build_attn_mask(attention_mask: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
        causal = torch.ones(t, t, device=device, dtype=torch.bool).tril()
        key_keep = attention_mask.to(torch.bool)[:, None, None, :]
        return causal[None, None, :, :] & key_keep
