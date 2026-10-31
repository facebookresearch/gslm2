# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

import torch
from torch.utils.data import DataLoader

from gslm2.encoder.config import ABXSubset, DataConfig, ResumeConfig, SpeechConfig
from gslm2.encoder.data.dataloader import build_dataloader
from gslm2.encoder.eval.abx import ABXConfig, compute_abx
from gslm2.encoder.models import SpidR, build_model
from gslm2.shared.environment import set_seed, setup_environment, setup_pytorch
from gslm2.shared.tools import init_logger

logger = logging.getLogger()


@torch.no_grad()
def validate(model: SpidR, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_ssl_loss = torch.zeros(1, device=device)
    total_target_ppl = torch.zeros(1, device=device)
    total_pred_ppl = torch.zeros(1, device=device)
    for batch in loader:
        ssl_loss, outputs = model(batch.to(device))
        total_ssl_loss += ssl_loss.mean()
        total_target_ppl += outputs["target_ppl"]
        total_pred_ppl += outputs["pred_ppl"]
    total_ssl_loss /= len(loader)
    total_target_ppl /= len(loader)
    total_pred_ppl /= len(loader)
    return {
        "loss": total_ssl_loss.item(),
        "ssl_loss": total_ssl_loss.item(),
        "target_ppl": total_target_ppl.item(),
        "pred_ppl": total_pred_ppl.item(),
    }


def validate_existing_checkpoint(cfg: SpeechConfig, resume: ResumeConfig, data: dict, group: str) -> None:
    """Run held-out loss eval on a single dataset and append the result to JSONL."""
    init_logger()
    if resume.checkpoint is None:
        logger.error("No checkpoint found, validation failed")
        return
    logger.info("Validating %s at step %s", group, resume.step)
    set_seed(cfg.run.random_seed)
    setup_pytorch(use_deterministic=cfg.run.use_deterministic)
    setup_environment()

    device = torch.device("cuda")
    model = build_model(checkpoint=resume.checkpoint, model_type=cfg.run.model_type).eval().to(device)
    val_cfg = DataConfig(**data)
    loader = build_dataloader(val_cfg, cfg.masking, cfg.model.extractor_conv_layer_config)

    val_output = {"step": resume.step, "group": group} | validate(model, loader, device)
    logger.info("Validation on %s done: loss %s", group, val_output["loss"])
    with resume.results.open("a") as f:
        f.write(json.dumps(val_output) + "\n")


def wrapper_abx(cfg: SpeechConfig, resume: ResumeConfig, subset: ABXSubset, codebook_proba: bool) -> None:  # noqa: FBT001
    if resume.checkpoint is None:
        raise ValueError("Cannot create ABX config without a checkpoint")

    abx_cfg = ABXConfig(
        checkpoint=resume.checkpoint,
        model_type=cfg.run.model_type,
        output_scores=resume.results.parent / "abx.csv",
        manifest=subset["manifest"],
        item_file=subset["item_file"],
        speaker_mode="within",
        context_mode="within",
        layers=None,
        codebook_proba=codebook_proba,
    )
    compute_abx(abx_cfg)
