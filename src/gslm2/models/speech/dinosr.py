"""DinoSR model definition."""

import copy
import logging
from collections.abc import Iterable
from dataclasses import replace
from functools import partial

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import functional as F

from gslm2.config import DinoSRConfig
from gslm2.data.common.packing import pack_features, pack_mask, packed_instance_norm
from gslm2.data.speech.dataset import SampleBatch
from gslm2.data.speech.tokenizer import Tokenizer
from gslm2.environment import pg_ddp
from gslm2.models.speech.components import (
    FeatureExtractor,
    FeatureProjection,
    FrameLevelCrossEntropyLoss,
    LossWeightDefinition,
    SupervisedClassifier,
    Transformer,
    get_encoder,
    get_feature_extractor,
    get_heads,
)
from gslm2.models.speech.metrics import compute_ppl

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


def ema_scheduler(step: int, start_decay: float, final_decay: float, final_step: int, freeze_step: int) -> float:
    if step < final_step:
        pct = 1 - step / final_step
        return final_decay - (final_decay - start_decay) * pct
    if step < freeze_step:
        return final_decay
    return 1


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


class DinoSR(nn.Module):
    def __init__(self, cfg: DinoSRConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DinoSRConfig()

        self.feature_extractor, self.feature_projection, self.student, self.heads, self.codebooks = dinosr_factory(cfg)
        self.teacher, self.teacher_exclude_layers = _init_teacher(self.student, cfg.ema_exclude_layers)
        self.ema_scheduler = partial(
            ema_scheduler,
            start_decay=cfg.ema_start_decay,
            final_decay=cfg.ema_final_decay,
            final_step=cfg.ema_final_step,
            freeze_step=cfg.freeze_step,
        )
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
        if cfg.supervised_loss_weight > 0.0 or cfg.supervised_every_step:
            self.supervised_loss_weight = cfg.supervised_loss_weight
            self.prepare_for_supervised_training(cfg)
        self.supervised_layer = cfg.supervised_layer

    def train(self, mode: bool = True) -> "DinoSR":  # noqa: FBT001, FBT002
        super().train(mode)
        self.teacher.eval()
        return self

    def prepare_for_supervised_training(self, cfg: DinoSRConfig) -> None:
        self.tokenizers = {}
        self.supervised_classifiers = torch.nn.ModuleDict()
        for language in cfg.supervised_languages:
            self.initialize_supervised_language_components(language, cfg.encoder_embed_dim, cfg.num_supervised_layers)
        self.supervised_criterion = FrameLevelCrossEntropyLoss()

    def initialize_supervised_language_components(
        self, language: str, encoder_embed_dim: int, num_supervised_layers: int
    ) -> None:
        logger.info("Initializing tokenizers and classifier heads for %s", language)
        self.tokenizers[language] = Tokenizer(language)
        self.supervised_classifiers[language] = SupervisedClassifier(
            encoder_embed_dim, self.tokenizers[language].vocab_size, num_supervised_layers
        )

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
        x = self.student(x, attention_mask=attention_mask)
        codebooks: list[Tensor | None] = [None] * (num_layers - self.num_codebooks)
        for i in range(self.num_codebooks):
            if i + first_codebook_layer not in layers:
                codebooks.append(None)
                continue
            codebook = self.heads[i](x).float().exp().squeeze()
            if onehot:
                codebook = F.one_hot(codebook.argmax(dim=-1), codebook.size(-1))
            codebooks.append(codebook)
        return codebooks

    def forward(
        self,
        batch: SampleBatch,
        *,
        loss_weights: LossWeightDefinition | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
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
        loss_weights = loss_weights if loss_weights else LossWeightDefinition()

        ssl_losses = torch.zeros(1, device=x.device)
        loss_dict: dict[str, Tensor] = {}

        if loss_weights.ssl > 0.0:
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
                    targets = [
                        F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)[mask_indices] for tl in targets
                    ]

            ssl_losses, loss_dict = self.get_ssl_loss(
                student_intermediate_outputs, student_final_pred, targets, mask_indices
            )

        supervision_loss = (
            self.get_supervision_loss(student_intermediate_outputs, batch.phonemes, batch.languages)
            if loss_weights.supervised > 0.0
            else torch.tensor(0.0, device=x.device)
        )

        return (
            ssl_losses,
            supervision_loss,
            loss_dict,
        )

    def get_ssl_loss(
        self,
        student_intermediate_outputs: list[Tensor],  # noqa: ARG002
        student_final_pred: Tensor,
        targets: list[Tensor],
        mask_indices: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        student_final_pred = student_final_pred[mask_indices]
        losses = torch.zeros(student_final_pred.shape[0], device=student_final_pred.device)
        target_ppl, pred_ppl = (
            torch.zeros(1, device=student_final_pred.device),
            torch.zeros(1, device=student_final_pred.device),
        )
        for i, target in enumerate(targets):
            pred = self.heads[i](student_final_pred).float()
            with torch.no_grad():
                onehot_target = self.codebooks[i](target)
                target_ppl += compute_ppl(onehot_target, input_onehot=True)
                pred_ppl += compute_ppl(pred, input_onehot=False)
            losses += torch.sum(-onehot_target * pred, dim=-1)
        loss_dict = {
            "target_ppl": (target_ppl / self.num_codebooks).detach(),
            "pred_ppl": (pred_ppl / self.num_codebooks).detach(),
        }
        return losses / self.num_codebooks, loss_dict

    def get_supervision_loss(
        self, student_intermediate_outputs: list[Tensor], phonemes: Tensor, languages: list[str]
    ) -> Tensor:
        assert phonemes is not None, "Supervised training requires phoneme alignment input."
        languages_set = set(languages)
        assert len(languages_set) == 1, (
            "Supervised training requires a single language input. Ensure you are language batching."
        )
        language = languages_set.pop()
        assert language in self.supervised_classifiers, (
            "Language %s not initialized for supervised training.",
            language,
        )
        logits = self.supervised_classifiers[language](student_intermediate_outputs[self.supervised_layer - 1])
        return self.supervised_criterion(logits, phonemes, self.tokenizers[language].ignore_id)


def dinosr_factory(
    cfg: DinoSRConfig,
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
        disable_pos_conv=cfg.encoder_disable_pos_conv,
        rope=cfg.encoder_rope,
        rope_theta=cfg.encoder_rope_theta,
        attention_bias=cfg.encoder_attention_bias,
        front_ema=cfg.encoder_front_ema,
        front_local_attn=cfg.encoder_front_local_attn,
        front_local_attn_window=cfg.encoder_front_local_attn_window,
        front_norm=cfg.encoder_front_norm,
    )
    heads = get_heads(
        embed_dim=cfg.encoder_embed_dim,
        codebook_size=cfg.codebook_size,
        num_codebooks=cfg.num_codebooks,
        layer_norm=getattr(cfg, "heads_layer_norm", "none"),
        use_weight_norm=getattr(cfg, "heads_use_weight_norm", False),
        init=getattr(cfg, "heads_init", "default"),
    )
    codebooks = nn.ModuleList(
        Codebook(cfg.encoder_embed_dim, cfg.codebook_size, cfg.codebook_decay) for _ in range(cfg.num_codebooks)
    )
    return feature_extractor, feature_projection, student, heads, codebooks
