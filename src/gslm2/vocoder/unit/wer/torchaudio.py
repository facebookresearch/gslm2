"""Evaluate generated speech with a torchaudio Wav2Vec2 ASR bundle."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from gslm2.vocoder.unit.evaluation import (
    EvaluationRecord,
    asr_record,
    load_generated_audio,
    read_asr_manifest,
    write_asr_results,
)
from gslm2.vocoder.unit.model import SAMPLE_RATE

if TYPE_CHECKING:
    from torchaudio import pipelines

DEFAULT_PIPELINE = "WAV2VEC2_ASR_LARGE_LV60K_960H"
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Lowercase English text and remove unsupported punctuation."""
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def greedy_decode(emission: Tensor, id_to_token: dict[int, str], blank_idx: int = 0) -> str:
    """Collapse CTC emissions into a normalized transcript."""
    hypothesis = emission.argmax(-1).unique_consecutive()
    hypothesis = hypothesis[hypothesis != blank_idx]
    text = "".join(id_to_token[int(index)].lower().replace("|", " ") for index in hypothesis)
    return " ".join(text.split())


def _get_pipeline(name: str) -> pipelines.Wav2Vec2ASRBundle:
    from torchaudio import pipelines

    try:
        return getattr(pipelines, name)
    except AttributeError as error:
        raise ValueError(f"unknown torchaudio ASR pipeline: {name}") from error


def torchaudio_word_error_rate(
    root: str | Path,
    path_manifest: str | Path,
    *,
    pipeline_name: str = DEFAULT_PIPELINE,
    device: str = "cuda",
) -> list[EvaluationRecord]:
    """Transcribe generated audio with Wav2Vec2 and return per-item errors."""
    pipeline = _get_pipeline(pipeline_name)
    model = pipeline.get_model().eval().to(device)
    id_to_token = dict(enumerate(pipeline.get_labels()))
    entries = read_asr_manifest(path_manifest)
    records: list[EvaluationRecord] = []
    for entry in tqdm(entries, desc=f"{pipeline_name} ASR"):
        waveform = load_generated_audio(root, entry.audio)
        if pipeline.sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected a 16 kHz ASR pipeline, got {pipeline.sample_rate}")
        with torch.inference_mode():
            normalized = F.layer_norm(waveform.to(device), waveform.shape)
            emission, _ = model(normalized)
        transcript = normalize_text(entry.transcript)
        hypothesis = greedy_decode(emission.squeeze(0), id_to_token)
        records.append(asr_record(entry, transcript, hypothesis, transcript_raw=entry.transcript))
    return records


def main(argv: list[str] | None = None) -> int:
    """Run torchaudio ASR evaluation and write JSONL plus aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate generated audio with torchaudio Wav2Vec2")
    parser.add_argument("root", type=Path, help="Root directory containing generated audio")
    parser.add_argument("manifest", type=Path, help="ASR JSONL with audio and transcript fields")
    parser.add_argument("output", type=Path, help="Per-item JSONL output")
    parser.add_argument("--model", default=DEFAULT_PIPELINE, help="Torchaudio pipeline constant")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    records = torchaudio_word_error_rate(
        args.root,
        args.manifest,
        pipeline_name=args.model,
        device=args.device,
    )
    wer, cer = write_asr_results(records, args.output)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("%s WER=%.2f%% CER=%.2f%%", args.model, 100 * wer, 100 * cer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
