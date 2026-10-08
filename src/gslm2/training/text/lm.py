"""``CausalLMTrainer``: decoder-only causal LM (OPT / GPT-2) pretraining over a token manifest."""

import argparse
import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from gslm2.config import LMConfig
from gslm2.data.text.dataset import build_text_dataloader
from gslm2.training.base import BaseTrainer
from gslm2.training.text._common import prepare_text_batch

logger = logging.getLogger()


CAUSAL_LM_RECORDS_EXTRA_KEYS: list[str] = ["perplexity", "nll_loss"]


class CausalLMTrainer(BaseTrainer):
    """Causal-LM (OPT / GPT-2) trainer over tokenized text or pre-extracted units."""

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__(
            cfg,
            records_extra_keys=CAUSAL_LM_RECORDS_EXTRA_KEYS,
            pbar_keys=["loss", "perplexity"],
        )

    def build_dataloader(self, cfg: LMConfig) -> Iterable:  # type: ignore[override]
        return build_text_dataloader(cfg.data)

    def training_step(self, batch: Any, step: int) -> tuple[Tensor, dict[str, Any]]:  # noqa: ANN401, ARG002
        ids, attn, targets, doc_ids, position_ids = prepare_text_batch(batch, self.device)
        with torch.autocast(self.device.type, self.dtype, self.cfg.optimizer.mixed_precision):
            loss, log_dict = self.ddp_model(
                ids, targets=targets, attention_mask=attn, position_ids=position_ids, doc_ids=doc_ids
            )
        metrics: dict[str, Any] = {"loss": loss.detach(), "batch_size": ids.size(0)}
        metrics.update(log_dict)
        return loss, metrics

    def _extra_banner_fields(self) -> dict[str, object]:
        return {
            "Model type": self.cfg.run.model_type,
            "Batch size (per GPU)": self.cfg.data.batch_size,
            "Task": self.cfg.data.task,
        }


def train_causal_lm(cfg: LMConfig) -> None:
    CausalLMTrainer(cfg).fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    train_causal_lm(LMConfig.from_file(args.config))
