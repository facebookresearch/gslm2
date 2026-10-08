import math
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from gslm2.config import SpidRConfig
from gslm2.models.speech.dinosr import DinoSR
from gslm2.models.speech.metrics import compute_entropy, compute_ppl
from gslm2.tools.scheduler import entropy_scheduler


def exp_ema_scheduler(
    step: int,
    start_decay: float,
    timescale: float,
    threshold: float,
    ema_final: float,
) -> float:
    decay = 1 - (1 - start_decay) * math.exp(-step / timescale)
    return decay if 1 - decay > threshold else ema_final


class SpidR(DinoSR):
    def __init__(self, cfg: SpidRConfig | None = None) -> None:
        if cfg is None:
            cfg = SpidRConfig()
        super().__init__(cfg)
        self.cfg: SpidRConfig = cfg
        self.ema_scheduler = partial(
            exp_ema_scheduler,
            start_decay=cfg.ema_start_decay,
            timescale=cfg.ema_timescale,
            threshold=cfg.ema_threshold,
            ema_final=cfg.ema_final,
        )
        self.reg_cfg = self.cfg.regularisation

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
