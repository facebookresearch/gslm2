# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-stream OPT with stream-specific projections and an autoregressive depformer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
)
from fairseq2.nn.functional import cross_entropy
from torch import nn

from gslm2.lm.models.opt.dedup import build_dedup_view
from gslm2.lm.models.opt.sampling import sample_next_token

if TYPE_CHECKING:
    from fairseq2.data_type import DataType
    from fairseq2.device import Device
    from fairseq2.models.transformer_lm import TransformerLMDecoder


class TransformerLMMultiStream(TransformerLM):
    def __init__(  # noqa: PLR0913
        self,
        model_dim: int,
        decoder_frontend: nn.ModuleList,
        decoder: TransformerLMDecoder,
        final_proj: nn.ModuleList,
        pad_idx: int | None,
        max_seq_len: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        multi_pred_loss_weight: float = 0.1,
        self_distil_loss_weight: float = 1.0,
        gen_loss_weight: float = 0.1,
        distill_temperature: float = 1.0,
        depformer_decoder_frontend: nn.ModuleList,
        depformer_decoder: TransformerLMDecoder,
        depformer_final_proj: nn.ModuleList,
        depformer_generation_order: list[int] | None = None,
    ) -> None:
        super().__init__(
            model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            pad_idx,
            max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.num_streams = len(decoder_frontend)
        self.multi_pred_loss_weight = multi_pred_loss_weight
        self.self_distil_loss_weight = self_distil_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.distill_temperature = distill_temperature

        self.depformer_decoder_frontend = depformer_decoder_frontend
        self.depformer_decoder = depformer_decoder
        self.depformer_final_proj = depformer_final_proj
        self.generation_order = (
            list(range(self.num_streams)) if depformer_generation_order is None else list(depformer_generation_order)
        )
        if sorted(self.generation_order) != list(range(self.num_streams)):
            raise ValueError("depformer_generation_order must be a permutation of the stream indices.")

        depformer_model_dim = self.depformer_decoder_frontend[0].embed.embed_dim
        self.depformer_context_proj = torch.nn.Linear(model_dim, depformer_model_dim, bias=False)
        self.depformer_bos = torch.nn.Parameter(torch.empty(depformer_model_dim))
        torch.nn.init.normal_(self.depformer_bos, mean=0.0, std=depformer_model_dim**-0.5)
        self._last_stream_specific_loss: torch.Tensor | None = None
        self._last_gen_loss: torch.Tensor | None = None

    @property
    def last_stream_specific_loss(self) -> torch.Tensor | None:
        return self._last_stream_specific_loss

    @property
    def last_gen_loss(self) -> torch.Tensor | None:
        return self._last_gen_loss

    def _per_stream_layout(
        self,
        seqs: torch.Tensor,
        seqs_layout: BatchLayout,
        doc_seq_lens: list[int] | None = None,
    ) -> BatchLayout:
        """Build a layout for interleaved ``(B*S, T)`` stream sequences --> output ``(B, T)``.

        With ``doc_seq_lens`` (multi-stream packing) build a packed 1-D layout ``(sum(T_doc),)``.
        """
        bsz, seq_len = seqs.size()
        if bsz % self.num_streams != 0:
            raise ValueError(f"Batch size {bsz} is not divisible by {self.num_streams} streams.")
        bsz_per_stream = bsz // self.num_streams
        if doc_seq_lens is not None:
            lens = list(doc_seq_lens)
            if sum(lens) != seq_len + 1:
                raise ValueError(f"doc_seq_lens sums to {sum(lens)}, expected {seq_len + 1}.")
            if lens[-1] == 1:
                lens.pop()
            else:
                lens[-1] -= 1
            return BatchLayout(shape=(seq_len,), seq_lens=lens, packed=True, device=seqs.device)
        return BatchLayout(
            shape=(bsz_per_stream, seq_len),
            seq_lens=seqs_layout.seq_lens[:: self.num_streams],
            packed=False,
            device=seqs.device,
        )

    def forward(  # noqa: PLR0913, PLR0912, C901, PLR0915
        self,
        seqs: torch.Tensor,
        seqs_layout: BatchLayout,
        targets: torch.Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: torch.Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
        return_decoder_outputs: bool = False,
        doc_seq_lens: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return decoder states, per-stream logits, or the training loss."""
        packed = doc_seq_lens is not None
        if target_mask is not None and not packed:
            raise NotImplementedError("target_mask is only supported for multi-stream packing; pass doc_seq_lens.")
        seqs_layout = self._per_stream_layout(seqs, seqs_layout, doc_seq_lens)

        single_inputs_full = [
            self.decoder_frontend[i](
                seqs[i :: self.num_streams].squeeze(0) if packed else seqs[i :: self.num_streams],
                seqs_layout,
                state_bag=state_bag,
            )[0]
            for i in range(self.num_streams)
        ]
        multi_input = torch.stack(single_inputs_full, dim=0).mean(0)
        decoder_outputs_multi = self.decoder(multi_input, seqs_layout, state_bag=None)
        if packed:
            decoder_outputs_multi = decoder_outputs_multi.unsqueeze(0)

        if return_decoder_outputs:
            return decoder_outputs_multi

        if targets is None:
            return [proj(decoder_outputs_multi) for proj in self.final_proj]

        k = torch.randint(self.num_streams, (1,)).item()
        seqs_k = seqs[k :: self.num_streams]
        targets_k = targets[k :: self.num_streams]

        seq_lens_k = torch.as_tensor(seqs_layout.seq_lens, device=seqs.device, dtype=torch.long)
        if packed:
            # Force dedup changes at document boundaries and build a packed block-diagonal dedup layout
            doc_lens = list(seqs_layout.seq_lens)
            seq_lens_k = torch.full((seqs_k.size(0),), sum(doc_lens), device=seqs.device, dtype=torch.long)
            dedup_seqs_k, _, dedup_idx_per_t, dedup_doc_lens = build_dedup_view(
                seqs_k, seq_lens_k, self.pad_idx, doc_seq_lens=[doc_lens]
            )
            dedup_layout = BatchLayout(
                shape=(dedup_seqs_k.size(1),),
                seq_lens=dedup_doc_lens[0],
                packed=True,
                device=seqs.device,
            )
            # False at each document's first token so loss never predicts across document boundary.
            dedup_target_mask = torch.ones(dedup_seqs_k.size(1), dtype=torch.bool, device=seqs.device)
            off = 0
            for length in dedup_doc_lens[0][:-1]:
                off += length
                dedup_target_mask[off] = False
            dedup_target_mask = dedup_target_mask.unsqueeze(0)
        else:
            dedup_seqs_k, dedup_lens_k, dedup_idx_per_t, _ = build_dedup_view(seqs_k, seq_lens_k, self.pad_idx)
            dedup_layout = BatchLayout(
                shape=tuple(dedup_seqs_k.shape),
                seq_lens=dedup_lens_k.tolist(),
                packed=False,
                device=seqs.device,
            )
            dedup_target_mask = None
        single_emb_k, _ = self.decoder_frontend[k](
            dedup_seqs_k.squeeze(0) if packed else dedup_seqs_k, dedup_layout, state_bag=None
        )
        decoder_outputs_single = self.decoder(single_emb_k, dedup_layout, state_bag=None)
        if packed:
            decoder_outputs_single = decoder_outputs_single.unsqueeze(0)

        decoder_logits_single = self.final_proj[k](decoder_outputs_single)
        decoder_logits_multi = self.final_proj[k](decoder_outputs_multi)

        # Only ``final_proj[k]`` contributes to the loss this step.  Touch the other
        # projections' parameters directly, instead of running their (B*T, V)
        # projections just to scale the result by zero, so DDP still sees a gradient
        # for them and doesn't complain about unused parameters.
        unused_proj_loss = decoder_outputs_single.new_tensor(0.0, dtype=torch.float32)
        for i, proj in enumerate(self.final_proj):
            if i == k:
                continue

            for param in proj.parameters():
                unused_proj_loss = unused_proj_loss + 0.0 * param.sum()

        targets_gen = torch.stack([targets[i :: self.num_streams] for i in range(self.num_streams)], dim=0)
        # Packed: mask document-boundary frames
        gen_mask = (
            torch.stack([target_mask[i :: self.num_streams] for i in range(self.num_streams)], dim=0)
            if target_mask is not None
            else None
        )
        depformer_logits = self.forward_depformer(decoder_outputs_multi, targets_gen, seqs_layout)

        # With a custom generation order we have S depformer frontends but the
        # last-generated stream's frontend is never used.  Keep it in the
        # autograd graph so DDP doesn't complain about unused parameters.
        unused_depformer_loss = targets_gen.new_tensor(0.0, dtype=torch.float32)
        if len(self.depformer_decoder_frontend) == self.num_streams:
            last_stream = self.generation_order[-1]
            dummy_tok = targets_gen[last_stream]
            dummy_emb, _ = self.depformer_decoder_frontend[last_stream](
                dummy_tok.squeeze(0) if packed else dummy_tok, seqs_layout, state_bag=None
            )
            unused_depformer_loss = 0.0 * dummy_emb.sum()

        if return_logits:
            raise NotImplementedError("return_logits is not supported for multi-stream model")

        stream_specific_loss = self.compute_stream_specific_loss(
            decoder_logits_single=decoder_logits_single,
            decoder_logits_multi=decoder_logits_multi,
            targets_full_res=targets_k,
            seqs_full_res=seqs_k,
            dedup_seqs=dedup_seqs_k,
            dedup_idx_per_t=dedup_idx_per_t,
            seqs_layout_full_res=seqs_layout,
            label_smoothing=label_smoothing,
            target_mask=(target_mask[k :: self.num_streams] if target_mask is not None else None),
            dedup_target_mask=dedup_target_mask,
            reduction=reduction,
        )

        gen_loss = self.compute_generative_loss(
            depformer_logits,
            targets_gen,
            label_smoothing=label_smoothing,
            reduction=reduction,
            mask=gen_mask,
        )

        loss = (1.0 - self.gen_loss_weight) * stream_specific_loss + self.gen_loss_weight * gen_loss

        self._last_stream_specific_loss = stream_specific_loss.detach()
        self._last_gen_loss = gen_loss.detach()

        return loss + unused_proj_loss + unused_depformer_loss

    def forward_depformer(
        self, context: torch.Tensor, targets_by_stream: torch.Tensor, seqs_layout: BatchLayout
    ) -> list[torch.Tensor]:
        bsz, seq_len, _ = context.shape

        context_emb = self.depformer_context_proj(context)
        depformer_inputs = []
        for step in range(self.num_streams):
            if step == 0:
                token_emb = self.depformer_bos.view(1, 1, -1).expand(bsz, seq_len, -1)
            else:
                prev_stream = self.generation_order[step - 1]
                tok = targets_by_stream[prev_stream]
                token_emb, _ = self.depformer_decoder_frontend[prev_stream](
                    tok.squeeze(0) if seqs_layout.packed else tok,
                    seqs_layout,
                    state_bag=None,
                )
                if seqs_layout.packed:
                    token_emb = token_emb.unsqueeze(0)
            depformer_inputs.append(context_emb + token_emb)

        depformer_inputs = torch.stack(depformer_inputs, dim=2).reshape(bsz * seq_len, self.num_streams, -1)
        depformer_inputs_layout = BatchLayout(
            shape=(bsz * seq_len, self.num_streams),
            seq_lens=[self.num_streams] * (bsz * seq_len),
            packed=False,
            device=context.device,
        )
        depformer_outputs = self.depformer_decoder(depformer_inputs, depformer_inputs_layout, state_bag=None)
        depformer_outputs = depformer_outputs.view(bsz, seq_len, self.num_streams, -1)

        logits_by_stream: dict[int, torch.Tensor] = {}
        for step, stream_idx in enumerate(self.generation_order):
            logits_by_stream[stream_idx] = self.depformer_final_proj[stream_idx](depformer_outputs[:, :, step, :])

        return [logits_by_stream[i] for i in range(self.num_streams)]

    def _mask_special_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask reserved token IDs before decode-time sampling.

        Unit tokens are offset above the reserved IDs. In current unit-LM configs
        ``pad_idx == 1`` and real units start at 2; deriving the cutoff from
        ``pad_idx`` keeps this aligned with the model config.
        """
        if self.pad_idx is None:
            return logits

        cutoff = min(int(self.pad_idx) + 1, logits.size(-1))
        if cutoff <= 0:
            return logits

        logits = logits.clone()
        logits[..., :cutoff] = float("-inf")
        return logits

    @torch.no_grad()
    def forward_depformer_inference(  # noqa: PLR0913
        self,
        context: torch.Tensor,
        temperature: float,
        topp: float,
        topk: int = 0,
        *,
        num_candidates: int = 1,
        sample_first_stream_only: bool = False,
        temp_anneal: float = 1.0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        bsz, _, dim = context.shape

        num_streams = self.num_streams
        step_temps = [
            temperature * (1.0 - (1.0 - temp_anneal) * i / max(num_streams - 1, 1)) for i in range(num_streams)
        ]

        if num_candidates > 1:
            context = (
                context[:, None, :, :]
                .expand(bsz, num_candidates, context.size(1), dim)
                .reshape(bsz * num_candidates, context.size(1), dim)
            )
        bsz_exp = context.size(0)
        context_emb = self.depformer_context_proj(context[:, -1:, :])

        running_inputs = []
        logits_list = []
        pred_tokens: list[torch.Tensor] = []
        pred_logprobs: list[torch.Tensor] = []

        for step, stream_idx in enumerate(self.generation_order):
            if step == 0:
                token_emb = self.depformer_bos.view(1, 1, -1).expand(bsz_exp, 1, -1)
            else:
                prev_logits = self._mask_special_logits(logits_list[-1])
                prev_log_probs = torch.log_softmax(prev_logits, dim=-1)

                prev_ids = self._select_depformer_token(
                    prev_logits,
                    step_temps[step - 1],
                    topp,
                    topk,
                    sample=not sample_first_stream_only or step == 1,
                )
                prev_token_logprob = torch.gather(
                    prev_log_probs.squeeze(1),
                    dim=-1,
                    index=prev_ids,
                ).squeeze(-1)

                pred_tokens.append(prev_ids)
                pred_logprobs.append(prev_token_logprob)

                prev_stream = self.generation_order[step - 1]
                step_layout = BatchLayout(
                    shape=(bsz_exp, 1),
                    seq_lens=[1] * bsz_exp,
                    packed=False,
                    device=context.device,
                )
                token_emb, _ = self.depformer_decoder_frontend[prev_stream](
                    prev_ids,
                    step_layout,
                    state_bag=None,
                )

            running_inputs.append(context_emb + token_emb)

            depformer_inputs = torch.cat(running_inputs, dim=1)
            depformer_inputs_layout = BatchLayout(
                shape=(bsz_exp, step + 1),
                seq_lens=[step + 1] * bsz_exp,
                packed=False,
                device=context.device,
            )
            depformer_outputs = self.depformer_decoder(depformer_inputs, depformer_inputs_layout, state_bag=None)

            logits = self.depformer_final_proj[stream_idx](depformer_outputs[:, -1, :]).unsqueeze(1)
            logits_list.append(logits)

        final_logits = self._mask_special_logits(logits_list[-1])
        final_log_probs = torch.log_softmax(final_logits, dim=-1)
        prev_ids = self._select_depformer_token(
            final_logits,
            step_temps[num_streams - 1],
            topp,
            topk,
            sample=not sample_first_stream_only or self.num_streams == 1,
        )
        prev_token_logprob = torch.gather(
            final_log_probs.squeeze(1),
            dim=-1,
            index=prev_ids,
        ).squeeze(-1)

        pred_tokens.append(prev_ids)
        pred_logprobs.append(prev_token_logprob)

        pred_tokens = [x.view(bsz, num_candidates, *x.shape[1:]) for x in pred_tokens]
        pred_logprobs = [x.view(bsz, num_candidates) for x in pred_logprobs]

        tokens_by_stream = {stream_idx: pred_tokens[step] for step, stream_idx in enumerate(self.generation_order)}
        logprobs_by_stream = {stream_idx: pred_logprobs[step] for step, stream_idx in enumerate(self.generation_order)}

        return (
            [tokens_by_stream[i] for i in range(self.num_streams)],
            [logprobs_by_stream[i] for i in range(self.num_streams)],
        )

    @staticmethod
    def _select_depformer_token(
        logits: torch.Tensor,
        temperature: float,
        topp: float,
        topk: int,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if sample:
            return sample_next_token(logits, temperature, topp, topk)

        return logits.argmax(dim=-1)

    def compute_stream_specific_loss(  # noqa: PLR0913
        self,
        *,
        decoder_logits_single: torch.Tensor,
        decoder_logits_multi: torch.Tensor,
        targets_full_res: torch.Tensor,
        seqs_full_res: torch.Tensor,
        dedup_seqs: torch.Tensor,
        dedup_idx_per_t: torch.Tensor,
        seqs_layout_full_res: BatchLayout,
        label_smoothing: float = 0.0,
        target_mask: torch.Tensor | None = None,
        dedup_target_mask: torch.Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> torch.Tensor:
        """Blend the deduplicated teacher loss with the multi-stream student loss.

        For packed batches ``target_mask`` marks valid student frames and
        ``dedup_target_mask`` marks valid teacher (dedup) frames, both excluding
        document boundaries.
        """
        loss_single = decoder_logits_multi.new_tensor(0.0, dtype=torch.float32)
        if self.multi_pred_loss_weight < 1.0 and decoder_logits_single.size(1) > 1:
            loss_single = cross_entropy(
                decoder_logits_single[:, :-1, :],
                dedup_seqs[:, 1:],
                self.pad_idx,
                label_smoothing=label_smoothing,
                target_mask=(None if dedup_target_mask is None else dedup_target_mask[:, 1:]),
                reduction=reduction,
            )

        loss_multi = decoder_logits_multi.new_tensor(0.0, dtype=torch.float32)
        if self.multi_pred_loss_weight > 0.0:
            change_mask = self._change_mask(seqs_full_res, targets_full_res, seqs_layout_full_res, target_mask)

            multi_ce = decoder_logits_multi.new_tensor(0.0, dtype=torch.float32)
            if self.self_distil_loss_weight < 1.0:
                ce_per_frame = cross_entropy(
                    decoder_logits_multi,
                    targets_full_res,
                    self.pad_idx,
                    label_smoothing=label_smoothing,
                    target_mask=None,
                    reduction="none",
                )
                multi_ce = self._masked_reduce(ce_per_frame, change_mask, reduction)

            loss_distil = decoder_logits_multi.new_tensor(0.0, dtype=torch.float32)
            if self.self_distil_loss_weight > 0.0:
                loss_distil = self._distillation_loss(
                    decoder_logits_multi,
                    decoder_logits_single,
                    dedup_idx_per_t,
                    change_mask,
                    reduction,
                )

            loss_multi = (1.0 - self.self_distil_loss_weight) * multi_ce + self.self_distil_loss_weight * loss_distil

        return (1.0 - self.multi_pred_loss_weight) * loss_single + self.multi_pred_loss_weight * loss_multi

    @staticmethod
    def _change_mask(
        seqs: torch.Tensor,
        targets: torch.Tensor,
        seqs_layout: BatchLayout,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_mask is not None:
            valid = target_mask.to(torch.bool)
        else:
            seq_lens = torch.as_tensor(seqs_layout.seq_lens, device=seqs.device, dtype=torch.long)
            valid = torch.arange(seqs.shape[1], device=seqs.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        return ((targets != seqs) & valid).to(torch.float32)

    @staticmethod
    def _masked_reduce(values: torch.Tensor, mask: torch.Tensor, reduction: Literal["sum", "mean"]) -> torch.Tensor:
        total = (values * mask).sum()
        return total if reduction == "sum" else total / mask.sum().clamp(min=1.0)

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        dedup_idx_per_t: torch.Tensor,
        change_mask: torch.Tensor,
        reduction: Literal["sum", "mean"],
    ) -> torch.Tensor:
        temperature = float(self.distill_temperature)
        student_log_probs = F.log_softmax(student_logits.to(torch.float32) / temperature, dim=-1)

        with torch.no_grad():
            batch_size, _, vocab_size = teacher_logits.shape
            gather_idx = dedup_idx_per_t.unsqueeze(-1).expand(batch_size, student_logits.shape[1], vocab_size)
            teacher_logits = teacher_logits.gather(dim=1, index=gather_idx)
            teacher_probs = F.softmax(teacher_logits.to(torch.float32) / temperature, dim=-1)

        kl_per_frame = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        return self._masked_reduce(kl_per_frame * temperature**2, change_mask, reduction)

    def compute_generative_loss(
        self,
        depformer_logits: list[torch.Tensor],
        targets_by_stream: torch.Tensor,
        *,
        label_smoothing: float = 0.0,
        reduction: Literal["sum", "mean"] = "sum",
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_gen = targets_by_stream.new_tensor(0.0, dtype=torch.float32)
        denom = targets_by_stream.new_tensor(0.0, dtype=torch.float32)

        for i, logits in enumerate(depformer_logits):
            ce = cross_entropy(
                logits,
                targets_by_stream[i],
                self.pad_idx,
                label_smoothing=label_smoothing,
                target_mask=None,
                reduction="none",
            )
            frame_mask = (mask[i] if mask is not None else targets_by_stream[i] != self.pad_idx).to(ce.dtype)
            loss_gen = loss_gen + (ce * frame_mask).sum()
            denom = denom + frame_mask.sum()

        if reduction == "mean":
            return loss_gen / denom.clamp(min=1.0)

        return loss_gen / self.num_streams
