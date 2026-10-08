"""Multi-stream OPT: parallel per-stream causal LM with a shared decoder + autoregressive depformer.

Pure-PyTorch port of the fairseq2 ``gslm2.fs2.models.opt.multi_stream_model.TransformerLMMultiStream``
(landed on main via #273). ``num_streams`` aligned token streams are embedded per-stream, mean-pooled
into one shared OPT decoder, and read out by per-stream heads; a small "depformer" then generates the
per-timestep streams autoregressively conditioned on the shared context. Training blends a deduplicated
single-stream teacher CE, a multi-stream student CE on change frames, a self-distillation KL, and the
depformer generative loss — matching the fairseq2 reference so it can be forward/loss-parity checked.

The interface is ``[B, S, T]`` (streams stacked on dim 1) rather than the fairseq2 ``[B*S, T]`` strided
layout; the parity test converts between the two. The generation-time (autoregressive-decode) depformer
sampler is intentionally omitted — training uses the teacher-forced depformer and sLM21 scoring uses only
the per-stream shared-context logits.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from gslm2.config.models.core import MultiStreamCausalLMConfig
from gslm2.models.core.dedup import build_dedup_view
from gslm2.models.core.opt import DecoderLayer
from gslm2.models.modules.positional_embeddings import LearnedPositionEmbedding
from gslm2.models.modules.transformer import flex_backward_unsupported, trunc_normal_

if TYPE_CHECKING:
    from gslm2.config.models.core import CausalLMConfig

logger = logging.getLogger(__name__)


def _ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, reduction: str) -> torch.Tensor:
    """Flattened cross-entropy over the vocab dim, ignoring ``ignore_index`` targets."""
    flat = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index, reduction=reduction
    )
    return flat.reshape(targets.shape) if reduction == "none" else flat


def _doc_block_mask(doc_ids: torch.Tensor) -> BlockMask:
    """flex_attention block mask: attend causally within the same document (packed eval path)."""

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001, ANN202, ARG001
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        not_pad = doc_ids[b, q_idx] >= 0
        return same_doc & not_pad & (q_idx >= kv_idx)

    bsz, seq_len = doc_ids.shape
    return create_block_mask(
        mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=doc_ids.device, _compile=doc_ids.is_cuda
    )


def _doc_block_mask_dense(doc_ids: torch.Tensor) -> torch.Tensor:
    """Dense ``[B, 1, T, T]`` block-diagonal causal mask (packed training path; SDPA applies dropout).

    A padding-query row (``doc_id < 0``) would be fully masked and make SDPA emit NaN, so the identity
    keeps every such row attending to itself. Real queries never attend to padding, so their outputs are
    unchanged and padding outputs are discarded by the loss mask (mirrors the single-stream OPT).
    """
    _, t = doc_ids.shape
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    not_pad = (doc_ids >= 0)[:, :, None]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=doc_ids.device))
    eye = torch.eye(t, dtype=torch.bool, device=doc_ids.device)
    return ((same_doc & not_pad & causal) | eye)[:, None, :, :]


def _positions_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """Per-document position ids restarting at 0 at each document boundary (packed batches)."""
    bsz, t = doc_ids.shape
    idx = torch.arange(t, device=doc_ids.device).unsqueeze(0).expand(bsz, t)
    boundary = torch.ones_like(doc_ids, dtype=torch.bool)
    boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    doc_start = torch.where(boundary, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return (idx - doc_start).clamp(min=0)


class _Backbone(nn.Module):
    """Shared OPT decoder stack (positions + layers + final norm) over a pre-embedded ``[B, T, D]`` input."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        super().__init__()
        self.pos_embed = LearnedPositionEmbedding(cfg.max_seq_len, cfg.model_dim, cfg.pos_embed_pad_offset)
        self.embed_dropout = nn.Dropout(cfg.dropout_p)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | BlockMask | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_dropout(x + self.pos_embed(x.size(1), x.device, position_ids))
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)
        return self.final_norm(x)


class _Depformer(nn.Module):
    """Decoder stack over the stream axis, with no position encoding (positions are the streams)."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, S, dep_D]; causal attention over the S stream positions (attn_mask=None -> is_causal).
        for layer in self.layers:
            x = layer(x, attention_mask=None)
        return self.final_norm(x)


class MultiStreamOPT(nn.Module):
    """Multi-stream causal LM with a shared decoder, per-stream heads, and a depformer."""

    def __init__(self, cfg: MultiStreamCausalLMConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = MultiStreamCausalLMConfig()
        self.cfg = cfg
        self.num_streams = cfg.num_streams
        self.pad_idx = cfg.pad_idx
        self._ignore_index = cfg.pad_idx if cfg.pad_idx is not None else -100
        vocab_sizes = cfg.vocab_sizes

        backbone_cfg = cfg.backbone_config()
        self.stream_embed = nn.ModuleList(
            [nn.Embedding(v, cfg.model_dim, padding_idx=cfg.pad_idx) for v in vocab_sizes]
        )
        self.backbone = _Backbone(backbone_cfg)
        self.final_proj = nn.ModuleList([nn.Linear(cfg.model_dim, v, bias=False) for v in vocab_sizes])

        # Depformer: its own (smaller) decoder shape, per-stream embeddings (no positions), per-stream heads.
        dep_cfg = replace(
            backbone_cfg,
            model_dim=cfg.depformer_model_dim,
            num_layers=cfg.depformer_num_layers,
            num_attn_heads=cfg.depformer_num_attn_heads,
            num_key_value_heads=cfg.depformer_num_key_value_heads,
            ffn_inner_dim=cfg.depformer_ffn_inner_dim,
            attn_window_len=None,
        )
        self.depformer_embed = nn.ModuleList(
            [nn.Embedding(v, cfg.depformer_model_dim, padding_idx=cfg.pad_idx) for v in vocab_sizes]
        )
        self.depformer_context_proj = nn.Linear(cfg.model_dim, cfg.depformer_model_dim, bias=False)
        self.depformer_bos = nn.Parameter(torch.empty(cfg.depformer_model_dim))
        self.depformer = _Depformer(dep_cfg)
        self.depformer_final_proj = nn.ModuleList(
            [nn.Linear(cfg.depformer_model_dim, v, bias=False) for v in vocab_sizes]
        )

        order = cfg.depformer_generation_order
        self.generation_order = list(range(self.num_streams)) if order is None else list(order)
        if sorted(self.generation_order) != list(range(self.num_streams)):
            raise ValueError("depformer_generation_order must be a permutation of the stream indices.")

        self.apply(self._init_weights)
        nn.init.normal_(self.depformer_bos, mean=0.0, std=cfg.depformer_model_dim**-0.5)
        # Output heads scale with their input width rather than the generic init_std, which keeps the
        # initial logits at unit variance however wide the per-stream vocabularies are.
        for proj in (*self.final_proj, *self.depformer_final_proj):
            nn.init.normal_(proj.weight, std=proj.in_features**-0.5)
        if cfg.init_scheme == "standard":
            self._rescale_blocks()
        else:  # "fairseq2" applies no residual rescaling
            self._init_fairseq2_scheme()
        if cfg.pad_idx is not None:
            with torch.no_grad():
                for emb in (*self.stream_embed, *self.depformer_embed):
                    emb.weight[cfg.pad_idx].zero_()

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
        # Scale each block's output projections by 1/sqrt(2*(layer_id+1)) (GPT-NeoX / Megatron residual
        # init), so the residual stream's variance does not grow with depth. The backbone and the
        # depformer are independent residual stacks, so each is scaled by its own depth. Applies at
        # init only; a loaded checkpoint overwrites these weights.
        for stack in (self.backbone, self.depformer):
            for layer_id, layer in enumerate(stack.layers):
                layer.attn.out_proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
                layer.ffn.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_fairseq2_scheme(self) -> None:
        """Re-initialise the weights under the ``"fairseq2"`` scheme.

        Token, depformer and position embeddings are N(0, 1), since none of the lookups are scaled by
        the model width. Attention projections are xavier-uniform, with gain ``2**-0.5`` on query, key
        and value and ``1.0`` on the output. The feed-forward and depformer context projections keep
        PyTorch's default kaiming-uniform bound. No residual rescaling is applied, and the output heads
        keep the input-width scaling they were built with.
        """
        for embed in (*self.stream_embed, *self.depformer_embed):
            nn.init.normal_(embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.backbone.pos_embed.weight.weight, mean=0.0, std=1.0)
        for stack in (self.backbone, self.depformer):
            for layer in stack.layers:
                for proj in (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj):
                    nn.init.xavier_uniform_(proj.weight, gain=2**-0.5)
                nn.init.xavier_uniform_(layer.attn.out_proj.weight)
                for proj in (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj, layer.attn.out_proj):
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)
                layer.ffn.fc1.reset_parameters()
                layer.ffn.fc2.reset_parameters()
        self.depformer_context_proj.reset_parameters()

    def _backbone_mask(self, doc_ids: torch.Tensor | None) -> torch.Tensor | BlockMask | None:
        """Attention mask for the shared decoder.

        Padded batches are right-padded causal LM, so ``None`` is correct and lets SDPA pick its fused
        (flash) causal kernel: trailing pad is in the future (masked by causality) and pad-query outputs
        are dropped by the loss. Passing a dense ``[B, 1, T, T]`` mask instead would force SDPA off the
        flash path — the ~20% throughput + memory regression vs fairseq2 at scale. Packed batches
        (``doc_ids`` given) need a block-diagonal within-document mask (dense in training so SDPA applies
        dropout, and wherever flex_attention has no backward; flex_attention in eval otherwise).
        """
        if doc_ids is None:
            return None
        if self.training or flex_backward_unsupported(doc_ids.device):
            return _doc_block_mask_dense(doc_ids)
        return _doc_block_mask(doc_ids)

    def _shared_context(self, streams: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pool the per-stream embeddings, then run the shared decoder. ``streams``: ``[B, S, T]``."""
        embedded = torch.stack([self.stream_embed[s](streams[:, s]) for s in range(self.num_streams)], dim=0).mean(0)
        attn = self._backbone_mask(doc_ids)
        positions = _positions_from_doc_ids(doc_ids) if doc_ids is not None else None
        return self.backbone(embedded, attn, positions)

    def forward(
        self,
        streams: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        doc_ids: torch.Tensor | None = None,
        stream_idx: int | None = None,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Inference (``targets=None``) -> list of per-stream logits ``[B, T, V_s]``.

        Training -> ``(loss, log_dict)``, where ``log_dict`` reports the two terms the loss blends,
        ``stream_specific_loss`` and ``gen_loss``. ``doc_ids`` (``[B, T]``, ``-1`` padding) enables
        sequence packing: several documents per row with block-diagonal attention. ``stream_idx`` fixes the
        sampled stream for the stream-specific loss term (used by the parity test); ``None`` samples it.
        """
        if streams.size(1) != self.num_streams:
            raise ValueError(f"expected {self.num_streams} streams, got {streams.size(1)}")
        context = self._shared_context(streams, doc_ids)
        if targets is None:
            return [proj(context) for proj in self.final_proj]
        return self._loss(streams, targets, attention_mask, doc_ids, context, stream_idx)

    def _loss(  # noqa: PLR0913
        self,
        streams: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor | None,
        doc_ids: torch.Tensor | None,
        context: torch.Tensor,
        stream_idx: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        b, _, t = streams.shape
        device = streams.device
        if attention_mask is None:
            seq_lens = torch.full((b,), t, device=device, dtype=torch.long)
        else:
            seq_lens = attention_mask.to(torch.long).sum(dim=1)

        k = torch.randint(self.num_streams, (1,)).item() if stream_idx is None else stream_idx
        seqs_k, targets_k = streams[:, k], targets[:, k]

        # Deduplicated single-stream "teacher": stream k's run-length-collapsed sequence through the decoder.
        dedup_seqs, _, dedup_idx, dedup_doc_ids = build_dedup_view(seqs_k, seq_lens, self.pad_idx, doc_ids=doc_ids)
        if doc_ids is not None:  # packed: block-diagonal teacher attention + per-document positions
            teacher_attn = self._backbone_mask(dedup_doc_ids)
            teacher_pos = _positions_from_doc_ids(dedup_doc_ids)
        else:  # right-padded causal -> is_causal (fused SDPA); dedup pad is future, teacher CE ignores it
            teacher_attn = None
            teacher_pos = None
        context_single = self.backbone(self.stream_embed[k](dedup_seqs), teacher_attn, teacher_pos)
        logits_single = self.final_proj[k](context_single)
        logits_multi = self.final_proj[k](context)

        stream_loss = self._stream_specific_loss(
            logits_single, logits_multi, seqs_k, targets_k, dedup_seqs, dedup_idx, seq_lens, dedup_doc_ids
        )

        targets_gen = targets.permute(1, 0, 2)  # [S, B, T]
        gen_loss = self._generative_loss(self._depformer_logits(context, targets_gen), targets_gen)

        loss = (1.0 - cfg.gen_loss_weight) * stream_loss + cfg.gen_loss_weight * gen_loss
        log_dict = {"stream_specific_loss": stream_loss.detach(), "gen_loss": gen_loss.detach()}
        return loss + self._unused_param_touch(k), log_dict

    @staticmethod
    def _length_mask(lens: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
        return torch.arange(t, device=device).unsqueeze(0) < lens.unsqueeze(1)

    def _stream_specific_loss(  # noqa: PLR0913
        self,
        logits_single: torch.Tensor,
        logits_multi: torch.Tensor,
        seqs_k: torch.Tensor,
        targets_k: torch.Tensor,
        dedup_seqs: torch.Tensor,
        dedup_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        dedup_doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        loss_single = logits_multi.new_zeros(())
        if cfg.multi_pred_loss_weight < 1.0 and dedup_seqs.size(1) > 1:
            dedup_target = dedup_seqs[:, 1:]
            if dedup_doc_ids is not None:  # packed: never predict across a document boundary
                dedup_target = dedup_target.masked_fill(
                    dedup_doc_ids[:, 1:] != dedup_doc_ids[:, :-1], self._ignore_index
                )
            loss_single = _ce(logits_single[:, :-1, :], dedup_target, self._ignore_index, "mean")

        loss_multi = logits_multi.new_zeros(())
        if cfg.multi_pred_loss_weight > 0.0:
            change_mask = self._change_mask(seqs_k, targets_k, seq_lens)
            multi_ce = logits_multi.new_zeros(())
            if cfg.self_distil_loss_weight < 1.0:
                per_frame = _ce(logits_multi, targets_k, self._ignore_index, "none")
                multi_ce = self._masked_mean(per_frame, change_mask)
            loss_distil = logits_multi.new_zeros(())
            if cfg.self_distil_loss_weight > 0.0:
                loss_distil = self._distillation_loss(logits_multi, logits_single, dedup_idx, change_mask)
            loss_multi = (1.0 - cfg.self_distil_loss_weight) * multi_ce + cfg.self_distil_loss_weight * loss_distil

        return (1.0 - cfg.multi_pred_loss_weight) * loss_single + cfg.multi_pred_loss_weight * loss_multi

    def _change_mask(self, seqs: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        non_pad = self._length_mask(seq_lens, seqs.size(1), seqs.device)
        return ((targets != seqs) & non_pad).to(torch.float32)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    def _distillation_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, dedup_idx: torch.Tensor, change_mask: torch.Tensor
    ) -> torch.Tensor:
        temp = float(self.cfg.distill_temperature)
        student_logp = F.log_softmax(student.to(torch.float32) / temp, dim=-1)
        with torch.no_grad():
            b, _, v = teacher.shape
            gather = dedup_idx.unsqueeze(-1).expand(b, student.size(1), v)
            teacher_gathered = teacher.gather(dim=1, index=gather)
            teacher_prob = F.softmax(teacher_gathered.to(torch.float32) / temp, dim=-1)
        kl = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=-1)
        return self._masked_mean(kl * temp**2, change_mask)

    def _depformer_logits(self, context: torch.Tensor, targets_gen: torch.Tensor) -> list[torch.Tensor]:
        """Teacher-forced depformer: per-stream logits ``[B, T, V_s]`` from the shared context."""
        b, t, _ = context.shape
        context_emb = self.depformer_context_proj(context)
        step_inputs = []
        for step in range(self.num_streams):
            if step == 0:
                token_emb = self.depformer_bos.view(1, 1, -1).expand(b, t, -1)
            else:
                prev = self.generation_order[step - 1]
                token_emb = self.depformer_embed[prev](targets_gen[prev])
            step_inputs.append(context_emb + token_emb)

        # [B, T, S, dep_D] -> [B*T, S, dep_D]; causal decode over the S stream positions.
        stacked = torch.stack(step_inputs, dim=2).reshape(b * t, self.num_streams, -1)
        out = self.depformer(stacked).view(b, t, self.num_streams, -1)

        logits_by_stream: dict[int, torch.Tensor] = {}
        for step, stream_idx in enumerate(self.generation_order):
            logits_by_stream[stream_idx] = self.depformer_final_proj[stream_idx](out[:, :, step, :])
        return [logits_by_stream[i] for i in range(self.num_streams)]

    def _generative_loss(self, depformer_logits: list[torch.Tensor], targets_gen: torch.Tensor) -> torch.Tensor:
        total = depformer_logits[0].new_zeros(())
        for i, logits in enumerate(depformer_logits):
            total = total + _ce(logits, targets_gen[i], self._ignore_index, "mean")
        return total / self.num_streams

    def _unused_param_touch(self, k: int) -> torch.Tensor:
        """Keep every parameter in the autograd graph so DDP doesn't flag unused ones.

        Per step only ``final_proj[k]`` and a subset of depformer frontends contribute; touch the
        rest with a 0-weighted term (matches the fairseq2 reference, which avoids find_unused_parameters).
        """
        touch = next(self.parameters()).new_zeros(())
        for i, proj in enumerate(self.final_proj):
            if i == k:
                continue
            for p in proj.parameters():
                touch = touch + 0.0 * p.sum()
        if self.generation_order:
            for p in self.depformer_embed[self.generation_order[-1]].parameters():
                touch = touch + 0.0 * p.sum()
        return touch
