"""``SpeechSSLTrainer``: pure SSL speech-encoder pretraining."""

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import distributed as dist
from torch import nn

from gslm2.config import SpeechConfig
from gslm2.data.speech.dataloader import build_dataloader as build_ssl_dataloader
from gslm2.training.base import BaseTrainer
from gslm2.training.hooks import FreezeParamGroupHook, UpdateEmaHook

logger = logging.getLogger()


SSL_RECORDS_EXTRA_KEYS: list[str] = [
    "ssl_loss",
    "target_ppl",
    "pred_ppl",
    "ssl_loss_no_reg",
    "student_predictive_entropy",
    "student_codebook_entropy",
    "teacher_predictive_entropy",
    "teacher_codebook_entropy",
    "mask_fraction",
    "n_masked_frames",
]


def _ssl_compile_strategy(model: nn.Module) -> nn.Module:
    """Compile the model and its ``_inner_ema`` buffer update."""
    model.compile(dynamic=True)
    model._inner_ema = torch.compile(model._inner_ema)  # noqa: SLF001
    return model


class SpeechSSLTrainer(BaseTrainer):
    """SSL leaf trainer for speech encoder pretraining."""

    def __init__(self, cfg: SpeechConfig) -> None:
        pre_step_hooks = []
        freeze_step = getattr(cfg.model, "freeze_step", None)
        if freeze_step is not None:
            pre_step_hooks.append(FreezeParamGroupHook(freeze_step))

        post_step_hooks = [UpdateEmaHook()]
        super().__init__(
            cfg,
            post_step_hooks=post_step_hooks,
            pre_step_hooks=pre_step_hooks,
            records_extra_keys=SSL_RECORDS_EXTRA_KEYS,
            pbar_keys=["loss", "target_ppl", "ssl_loss_no_reg"],
            compile_strategy=_ssl_compile_strategy,
            find_unused_parameters=True,
        )

    def build_dataloader(self, cfg: SpeechConfig) -> Iterable:
        return build_ssl_dataloader(cfg.data, cfg.masking, cfg.model.extractor_conv_layer_config)

    def training_step(self, batch: Any, step: int) -> tuple[torch.Tensor, dict[str, Any]]:  # noqa: ANN401, ARG002
        with torch.autocast(self.device.type, self.dtype, self.cfg.optimizer.mixed_precision):
            loss, outputs = self.ddp_model(batch.to(self.device))
        # All-reduce by global frame count.
        num_frames = torch.tensor(loss.size(0), dtype=torch.long, device=self.device)
        dist.all_reduce(num_frames)
        loss = loss.sum() * self.world_size / num_frames
        loss_detached = loss.detach()
        metrics: dict[str, Any] = {
            "loss": loss_detached,
            "ssl_loss": loss_detached,
            "batch_size": batch.waveforms.size(0),
        }
        if batch.mask is not None:
            n_masked = int(batch.mask.sum().item())
            metrics["n_masked_frames"] = n_masked
            metrics["mask_fraction"] = float(batch.mask.float().mean().item())
        metrics.update(outputs)
        return loss, metrics


def train(cfg: SpeechConfig) -> None:
    SpeechSSLTrainer(cfg).fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    train(SpeechConfig.from_file(args.config))
