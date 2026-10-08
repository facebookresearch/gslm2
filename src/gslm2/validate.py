import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gslm2.config import ABXSubset, DataConfig, ResumeConfig, SpeechConfig
from gslm2.data.speech.dataloader import build_dataloader, speech_dataset
from gslm2.environment import set_seed, setup_environment, setup_pytorch
from gslm2.eval.encoder.abx import ABXConfig, compute_abx
from gslm2.models import DinoSR, LossWeightDefinition, SpidR, build_model
from gslm2.tools import init_logger

logger = logging.getLogger()


@torch.no_grad()
def validate(
    model: DinoSR | SpidR, loader: DataLoader, device: torch.device, loss_weights: LossWeightDefinition
) -> dict[str, float]:
    model.eval()
    total_ssl_loss = torch.zeros(1, device=device)
    total_supervised_loss = torch.zeros(1, device=device)
    total_target_ppl = torch.zeros(1, device=device)
    total_pred_ppl = torch.zeros(1, device=device)
    for batch in loader:
        ssl_loss, supervised_loss, outputs = model(batch.to(device))
        total_ssl_loss += ssl_loss.mean()
        total_supervised_loss += supervised_loss
        total_target_ppl += outputs["target_ppl"]
        total_pred_ppl += outputs["pred_ppl"]
    total_ssl_loss /= len(loader)
    total_supervised_loss /= len(loader)
    total_loss = loss_weights.supervised * total_supervised_loss + loss_weights.ssl * total_ssl_loss
    total_target_ppl /= len(loader)
    total_pred_ppl /= len(loader)
    return {
        "loss": total_loss.item(),
        "ssl_loss": total_ssl_loss.item(),
        "supervised_loss": total_supervised_loss.item(),
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

    supervised_loss_weight = cfg.model.supervised_loss_weight if val_cfg.alignments_path else 0.0
    loss_weights = LossWeightDefinition(
        ssl=1.0 - supervised_loss_weight,
        supervised=supervised_loss_weight,
    )

    val_output = {"step": resume.step, "group": group} | validate(model, loader, device, loss_weights)
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


def online_extract_representations(  # noqa: PLR0913
    model: DinoSR,
    output_dir: Path,
    manifests: list[Path],
    layers: list[int],
    *,
    codebook_proba: bool,
    onehot: bool,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device("cuda")
    # model = model.to(device).eval()
    model.eval()
    datasets = {p.stem: speech_dataset(DataConfig(manifest=p, normalize=True)) for p in set(manifests)}
    lengths = {name: len(dataset) for name, dataset in datasets.items()}

    with torch.inference_mode(), tqdm(total=sum(lengths.values())) as pbar:
        for name, dataset in datasets.items():
            pbar.set_description(name)
            for i, (waveform, _, _) in enumerate(dataset):
                outputs = (
                    model.get_codebooks(waveform.to(device).unsqueeze(0), onehot=onehot)
                    if codebook_proba
                    else model.get_intermediate_outputs(waveform.to(device).unsqueeze(0))
                )
                filename = Path(dataset.filenames[i]).with_suffix(".pt").name
                for layer in layers:
                    output_file = output_dir / f"{name}/{layer}/{filename}"
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    assert outputs[layer - 1] is not None
                    torch.save(outputs[layer - 1].squeeze().float().cpu(), output_file)
                pbar.update()
    model.train()
