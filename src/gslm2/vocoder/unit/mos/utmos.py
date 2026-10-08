"""Evaluate generated speech with UTMOS v1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from gslm2.vocoder.unit.evaluation import (
    EvaluationRecord,
    load_generated_audio,
    read_asr_manifest,
    write_mos_results,
)

logger = logging.getLogger(__name__)


def utmos_scores(
    root: str | Path,
    path_manifest: str | Path,
    *,
    device: str = "cuda",
) -> list[EvaluationRecord]:
    """Predict UTMOS for exactly the audio files named by an ASR manifest."""
    from utmos_pytorch import UTMOSScoreTorch

    model = UTMOSScoreTorch(device=device)
    records: list[EvaluationRecord] = []
    for entry in tqdm(read_asr_manifest(path_manifest), desc="UTMOS"):
        waveform = load_generated_audio(root, entry.audio).to(device)
        with torch.inference_mode():
            score = model.score(waveform)
        records.append({"file_path": entry.audio.as_posix(), "predicted_mos": float(score.item())})
    return records


def main(argv: list[str] | None = None) -> int:
    """Run UTMOS evaluation and write JSONL plus aggregate MOS."""
    parser = argparse.ArgumentParser(description="Evaluate generated audio with UTMOS v1")
    parser.add_argument("root", type=Path, help="Root directory containing generated audio")
    parser.add_argument("manifest", type=Path, help="ASR JSONL selecting the audio files to score")
    parser.add_argument("output", type=Path, help="Per-item JSONL output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    mean_mos = write_mos_results(utmos_scores(args.root, args.manifest, device=args.device), args.output)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("UTMOS=%.4f", mean_mos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
