"""SpidR 2.0 self-supervised speech encoder."""

import copy
import logging
import math
from collections.abc import Iterable
from dataclasses import replace
from functools import partial

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import functional as F

from gslm2.config import SpidRConfig
from gslm2.data.common.packing import pack_features, pack_mask, packed_instance_norm
from gslm2.data.speech.dataset import SampleBatch
from gslm2.environment import pg_ddp
from gslm2.models.speech.components import (
    FeatureExtractor,
    FeatureProjection,
    Transformer,
    get_encoder,
    get_feature_extractor,
    get_heads,
)
from gslm2.models.speech.metrics import compute_entropy, compute_ppl
from gslm2.tools.scheduler import entropy_scheduler

logger = logging.getLogger()


class Codebook(nn.Module):
    def __init__(self, encoder_embed_dim: int, codebook_size: int, codebook_decay: float) -> None:
        super().__init__()
        self.codebook_decay = codebook_decay
        self.codebook_size = codebook_size
        codebook = torch.randn(encoder_embed_dim, codebook_size).unsqueeze(0)
        codebook = F.instance_norm(codebook).transpose(1, 2).squeeze().float().contiguous()
        self.codebook: Tensor
        self.counts: Tensor
        self.register_buffer("codebook", codebook)
        self.register_buffer("counts", torch.ones(codebook_size, dtype=torch.float32).contiguous())

    @torch.no_grad()
    def forward(self, target: Tensor) -> Tensor:
        codebook = self.codebook / self.counts.unsqueeze(1)
        labels = torch.cdist(target, codebook, p=2).argmin(1)
        onehot_target = F.one_hot(labels, self.codebook_size).float()
        if self.training:
            self.step(onehot_target, target)
        return onehot_target

    @torch.no_grad()
    def step(self, onehot_target: Tensor, target: Tensor) -> None:
        if self.codebook_decay >= 1.0:
            return
        count = onehot_target.sum(0)
        memory = torch.matmul(onehot_target.t(), target)
        if dist.is_initialized():
            dist.all_reduce(memory, group=pg_ddp())  # Sum of embeddings
            dist.all_reduce(count, group=pg_ddp())  # Total counts
        alpha = torch.ones_like(count).unsqueeze(1)
        alpha[count != 0] = self.codebook_decay
        self.counts = alpha.squeeze(1) * self.counts + (1 - alpha).squeeze(1) * count
        self.codebook = alpha * self.codebook + (1 - alpha) * memory


def exp_ema_scheduler(
    step: int,
    start_decay: float,
    timescale: float,
    threshold: float,
    ema_final: float,
) -> float:
    decay = 1 - (1 - start_decay) * math.exp(-step / timescale)
    return decay if 1 - decay > threshold else ema_final


def _init_teacher(student: Transformer, exclude_layers: Iterable[str]) -> tuple[Transformer, set[str]]:
    teacher = copy.deepcopy(student).float()
    teacher.apply(teacher.init_weights)
    teacher.eval()
    teacher.requires_grad_(requires_grad=False)
    teacher_exclude_layers: set[str] = set()
    for name, param in teacher.named_parameters():
        param.detach_()
        if any(name.startswith(ex) for ex in exclude_layers):
            teacher_exclude_layers.add(name)
    return teacher, teacher_exclude_layers


class SpidR(nn.Module):
    """SpidR 2.0 encoder: student/teacher transformer with per-layer prediction heads and EMA codebooks."""

    def __init__(self, cfg: SpidRConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SpidRConfig()
        self.cfg: SpidRConfig = cfg

        self.feature_extractor, self.feature_projection, self.student, self.heads, self.codebooks = spidr_factory(cfg)
        self.teacher, self.teacher_exclude_layers = _init_teacher(self.student, cfg.ema_exclude_layers)
        self.ema_scheduler = partial(
            exp_ema_scheduler,
            start_decay=cfg.ema_start_decay,
            timescale=cfg.ema_timescale,
            threshold=cfg.ema_threshold,
            ema_final=cfg.ema_final,
        )
        self.reg_cfg = self.cfg.regularisation
        self.register_buffer("current_step", torch.zeros(1, dtype=torch.int64))
        self.projection_dropout = nn.Dropout(cfg.encoder_projection_dropout)
        self.enable_packing = False
        self.freeze_step = cfg.freeze_step
        self._extractor_frozen = False
        self.unfreeze_encoder_step = cfg.unfreeze_encoder_step
        self._encoder_frozen = False
        if self.unfreeze_encoder_step > 0:
            self.freeze_encoder()
        mask_embedding = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim))
        nn.init.uniform_(mask_embedding)
        self.register_parameter("mask_embedding", mask_embedding)  # type: ignore[attr-defined]

    def train(self, mode: bool = True) -> "SpidR":  # noqa: FBT001, FBT002
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    def freeze_extractor(self) -> None:
        logger.info("Freezing extractor at step %s", self.current_step.item())
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_projection.parameters():
            p.requires_grad = False
        self._extractor_frozen = True

    def freeze_encoder(self) -> None:
        logger.info("Freezing encoder at step %s", self.current_step.item())
        for p in self.student.parameters():
            p.requires_grad = False
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        logger.info("Unfreezing encoder at step %s", self.current_step.item())
        for p in self.student.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        self.train()

    @torch.no_grad()
    def _inner_ema(self, decay: torch.Tensor) -> None:
        for (ema_n, ema_p), model_p in zip(self.teacher.named_parameters(), self.student.parameters(), strict=True):
            if ema_n in self.teacher_exclude_layers:
                ema_p.copy_(model_p)
            else:
                ema_p.lerp_(model_p, 1 - decay)
        for ema_b, model_b in zip(self.teacher.buffers(), self.student.buffers(), strict=True):
            ema_b.copy_(model_b)

    def update_ema(self, step: int) -> float:
        self.current_step.fill_(step)
        decay = self.ema_scheduler(step)
        if not self._extractor_frozen and step >= self.freeze_step:
            self.freeze_extractor()
        if self._encoder_frozen and self.current_step.item() >= self.unfreeze_encoder_step:
            self.unfreeze_encoder()
        if 0.0 < decay < 1.0:
            self._inner_ema(torch.tensor(decay, device=self.current_step.device))
        return decay

    def get_intermediate_outputs(
        self,
        waveforms: Tensor,
        attention_mask: Tensor | None = None,
        *,
        layers: Iterable[int] | None = None,
    ) -> list[Tensor | None]:
        if layers is None:
            layers = list(range(1, len(self.student.layers) + 1))
        if set(layers) - set(range(1, len(self.student.layers) + 1)):
            raise ValueError(f"Layers must be in the range [1, {len(self.student.layers)}]")
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        intermediate_outputs, _ = self.student.get_intermediate_outputs(x, attention_mask, num_layers=max(layers))
        return [intermediate_outputs[i] if i + 1 in layers else None for i in range(len(self.student.layers))]

    def get_codebooks(
        self,
        waveforms: Tensor,
        *,
        attention_mask: Tensor | None = None,
        layers: Iterable[int] | None = None,
        onehot: bool = False,
    ) -> list[Tensor | None]:
        num_layers = len(self.student.layers)
        first_codebook_layer = num_layers - self.num_codebooks + 1
        if layers is None:
            layers = list(range(first_codebook_layer, num_layers + 1))
        if set(layers) - set(range(first_codebook_layer, num_layers + 1)):
            raise ValueError(f"Layers must be in the range [{first_codebook_layer}, {num_layers}]")
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        preds: list[Tensor | None] = [None] * (len(self.student.layers) - self.num_codebooks)
        intermediate_outputs, _ = self.student.get_intermediate_outputs(
            x, attention_mask=attention_mask, num_layers=max(layers)
        )
        for i in range(self.num_codebooks):
            if i + first_codebook_layer not in layers:
                preds.append(None)
                continue
            pred = self.heads[i](intermediate_outputs[i - 1 + first_codebook_layer]).float().exp().squeeze()
            if onehot:
                pred = F.one_hot(pred.argmax(dim=-1), pred.size(-1))
            preds.append(pred)
        return preds

    def forward(self, batch: SampleBatch) -> tuple[Tensor, dict[str, Tensor]]:
        feats = self.feature_extractor(batch.waveforms)
        feats = self.feature_projection(feats)
        x = feats.clone()
        x = self.projection_dropout(x)

        cu_seqlens: Tensor | None = None
        max_seqlen: int | None = None
        attention_mask = batch.attention_mask

        if self.enable_packing:
            if attention_mask is not None:
                frame_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int64)
            else:  # build_attention_mask returns None when the batch has no padding
                frame_lengths = torch.full((feats.shape[0],), feats.shape[1], dtype=torch.int64, device=feats.device)
            feats, cu_seqlens, max_seqlen, _ = pack_features(feats, frame_lengths)
            x, _, _, _ = pack_features(x, frame_lengths)
            if batch.mask is not None:
                packed_mask = pack_mask(batch.mask, frame_lengths)
                batch = replace(batch, attention_mask=None, mask=packed_mask)
            attention_mask = None

        if batch.mask is not None:
            mask = batch.mask
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        student_intermediate_outputs, student_final_pred = self.student.get_intermediate_outputs(
            x, attention_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        mask_indices = torch.nonzero(mask, as_tuple=True)
        with torch.no_grad():
            teacher_intermediate_outputs, _ = self.teacher.get_intermediate_outputs(
                feats, attention_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            targets = teacher_intermediate_outputs[-self.num_codebooks :]
            if cu_seqlens is not None:
                targets = [
                    packed_instance_norm(tl.float().transpose(1, 2), cu_seqlens).transpose(1, 2)[mask_indices]
                    for tl in targets
                ]
            else:
                targets = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)[mask_indices] for tl in targets]

        ssl_losses, loss_dict = self.get_ssl_loss(
            student_intermediate_outputs, student_final_pred, targets, mask_indices
        )

        return ssl_losses, loss_dict

    def get_ssl_loss(
        self,
        student_intermediate_outputs: list[Tensor],
        student_final_pred: Tensor,  # noqa: ARG002
        targets: list[Tensor],
        mask_indices: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute SSL loss with optional entropy regularisation.

        Args:
            student_intermediate_outputs: List of student encoder intermediate outputs
            student_final_pred: Final student prediction (unused, kept for API compatibility)
            targets: List of teacher target embeddings (normalised)
            mask_indices: Indices of masked frames

        Returns:
            (loss, loss_dict) tuple where loss_dict contains metrics for logging
        """
        device = student_intermediate_outputs[0].device
        reg_cfg = self.reg_cfg

        # Apply heads to student outputs at masked positions
        preds = [
            self.heads[i](y[mask_indices]).float()
            for i, y in enumerate(student_intermediate_outputs[-self.num_codebooks :])
        ]

        # Initialize accumulators
        num_frames = preds[0].shape[0]
        target_ppl = torch.zeros(1, device=device)
        pred_ppl = torch.zeros(1, device=device)
        ssl_loss_no_reg = torch.zeros(num_frames, device=device)  # Per-frame loss
        student_predictive_entropy = torch.zeros(1, device=device)
        student_codebook_entropy = torch.zeros(1, device=device)
        teacher_predictive_entropy = torch.zeros(1, device=device)
        teacher_codebook_entropy = torch.zeros(1, device=device)

        for i, (pred_log_probs, target) in enumerate(zip(preds, targets, strict=True)):
            pred_probs = torch.exp(pred_log_probs)

            if self.cfg.teacher_use_heads:
                # Use heads for teacher targets
                target_log_probs = self.heads[i](target).float()
                target_probs = torch.exp(target_log_probs)
                onehot_target = F.one_hot(target_log_probs.argmax(dim=-1), self.cfg.codebook_size).float()
            else:
                # Use codebook lookup
                onehot_target = self.codebooks[i](target)
                target_probs = None
                target_log_probs = None

            target_ppl += compute_ppl(onehot_target, input_onehot=True)
            pred_ppl += compute_ppl(pred_log_probs, input_onehot=False)

            # Per-frame cross-entropy loss: soft targets (teacher probs) or hard targets (one-hot)
            if self.cfg.use_soft_targets and self.cfg.teacher_use_heads:
                ssl_loss_no_reg += -torch.sum(target_probs.detach() * pred_log_probs, dim=-1)
            else:
                ssl_loss_no_reg += -torch.sum(onehot_target.detach() * pred_log_probs, dim=-1)

            s_pe, s_ce = compute_entropy(pred_probs, pred_log_probs)
            student_predictive_entropy += s_pe
            student_codebook_entropy += s_ce
            if self.cfg.teacher_use_heads:
                t_pe, t_ce = compute_entropy(target_probs, target_log_probs)
                teacher_predictive_entropy += t_pe
                teacher_codebook_entropy += t_ce

        # Get scheduled weights for entropy regularisation
        step = int(self.current_step.item())
        pred_entropy_weight = entropy_scheduler(step, reg_cfg, "predictive")
        codebook_entropy_weight = entropy_scheduler(step, reg_cfg, "codebook")

        # Apply regularisation to loss (entropy terms are scalars, broadcast to per-frame)
        ssl_loss_total = ssl_loss_no_reg.clone()

        # Entropy regularisation
        if reg_cfg.reg_student:
            ssl_loss_total += pred_entropy_weight * student_predictive_entropy
            ssl_loss_total -= codebook_entropy_weight * student_codebook_entropy
        if reg_cfg.reg_teacher and self.cfg.teacher_use_heads:
            ssl_loss_total += pred_entropy_weight * teacher_predictive_entropy
            ssl_loss_total -= codebook_entropy_weight * teacher_codebook_entropy

        # Build loss dict for logging (ssl_loss_no_reg is mean of per-frame losses)
        with torch.no_grad():
            loss_dict = {
                "target_ppl": target_ppl.detach() / self.num_codebooks,
                "pred_ppl": pred_ppl.detach() / self.num_codebooks,
                "ssl_loss_no_reg": ssl_loss_no_reg.mean().detach() / self.num_codebooks,
                "student_predictive_entropy": student_predictive_entropy.detach() / self.num_codebooks,
                "student_codebook_entropy": student_codebook_entropy.detach() / self.num_codebooks,
                "teacher_predictive_entropy": teacher_predictive_entropy.detach() / self.num_codebooks,
                "teacher_codebook_entropy": teacher_codebook_entropy.detach() / self.num_codebooks,
                "predictive_entropy_weight": torch.tensor(pred_entropy_weight, device=device),
                "codebook_entropy_weight": torch.tensor(codebook_entropy_weight, device=device),
            }

        return ssl_loss_total / self.num_codebooks, loss_dict


def spidr_factory(
    cfg: SpidRConfig,
) -> tuple[FeatureExtractor, FeatureProjection, Transformer, nn.ModuleList, nn.ModuleList]:
    feature_extractor = get_feature_extractor(
        cfg.extractor_mode,
        cfg.extractor_conv_layer_config,
        bias=cfg.extractor_conv_bias,
        channels_last=cfg.extractor_channels_last,
    )
    feature_projection, student = get_encoder(
        in_features=cfg.extractor_conv_layer_config[-1][0],
        embed_dim=cfg.encoder_embed_dim,
        pos_conv_kernel=cfg.encoder_pos_conv_kernel,
        pos_conv_groups=cfg.encoder_pos_conv_groups,
        pos_conv_depth=cfg.encoder_pos_conv_depth,
        num_layers=cfg.encoder_num_layers,
        num_heads=cfg.encoder_num_heads,
        attention_dropout=cfg.encoder_attention_dropout,
        ff_interm_features=cfg.encoder_ff_interm_features,
        ff_interm_dropout=cfg.encoder_ff_interm_dropout,
        dropout=cfg.encoder_dropout,
        layer_drop=cfg.encoder_layer_drop,
        layer_norm_first=cfg.encoder_layer_norm_first,
        qkv_bias=cfg.encoder_qkv_bias,
    )
    heads = get_heads(
        embed_dim=cfg.encoder_embed_dim,
        codebook_size=cfg.codebook_size,
        num_codebooks=cfg.num_codebooks,
        layer_norm=cfg.heads_layer_norm,
        use_weight_norm=cfg.heads_use_weight_norm,
        init=cfg.heads_init,
    )
    codebooks = nn.ModuleList(
        Codebook(cfg.encoder_embed_dim, cfg.codebook_size, cfg.codebook_decay) for _ in range(cfg.num_codebooks)
    )
    return feature_extractor, feature_projection, student, heads, codebooks
