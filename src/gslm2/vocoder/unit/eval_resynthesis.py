"""Resynthesis evaluation: oracle vs predicted durations (paper Tab. speech_resynthesis).

Synthesizes LJ Speech held-out utterances from ground-truth (non-deduped)
units and from duration-predictor-expanded units, then scores both with ASR
(Whisper or Qwen3-Omni WER) and UTMOS, writing one aggregate ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from gslm2.vocoder.duration.predict import (
    expand_streams,
    expand_tokens,
    load_duration_predictor,
    predict_durations,
)
from gslm2.vocoder.unit.data import STREAM_SEPARATOR, parse_units, read_manifest
from gslm2.vocoder.unit.evaluation import write_asr_results, write_mos_results
from gslm2.vocoder.unit.inference import synthesize
from gslm2.vocoder.unit.mos.utmos import utmos_scores
from gslm2.vocoder.unit.wer.whisper import whisper_word_error_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _expand_manifest(dedup_manifest: Path, duration_ckpt: Path, out_manifest: Path) -> None:
    """Predict durations for deduped units and write the expanded JSONL manifest."""
    model, log_duration = load_duration_predictor(duration_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    lines = []
    for entry in read_manifest(dedup_manifest):
        units = parse_units(entry.units)
        if units.dim() == 1:
            durations = predict_durations(model, units.tolist(), log_duration=log_duration, device=device)
            expanded = expand_tokens(units.tolist(), durations)
            units_out: object = expanded
        else:
            streams = [row.tolist() for row in units]
            durations = predict_durations(model, streams, log_duration=log_duration, device=device)
            expanded_streams = expand_streams(streams, durations)
            units_out = STREAM_SEPARATOR.join(" ".join(map(str, s)) for s in expanded_streams)
        lines.append(json.dumps({"audio": entry.audio, "units": units_out}))
    out_manifest.write_text("\n".join(lines) + "\n")
    logger.info("wrote %d expanded utterances to %s", len(lines), out_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-vs-predicted resynthesis eval (WER + MOS).")
    ap.add_argument("--oracle-manifest", type=Path, required=True, help="Non-deduped units JSONL.")
    ap.add_argument("--dedup-manifest", type=Path, required=True, help="Deduped units JSONL (same audios).")
    ap.add_argument("--asr-manifest", type=Path, required=True, help="ASR JSONL with audio + transcript refs.")
    ap.add_argument("--vocoder-checkpoint", type=Path, required=True)
    ap.add_argument("--duration-checkpoint", type=Path, required=True)
    ap.add_argument("--asr", choices=["whisper", "qwen3_omni"], default="whisper")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="ASR/MOS device (default: cuda if available else cpu).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_dir, predicted_dir = out_dir / "synth-oracle", out_dir / "synth-predicted"
    logger.info("synthesizing oracle durations")
    synthesize(oracle_dir, args.vocoder_checkpoint, args.oracle_manifest)
    logger.info("predicting durations + synthesizing")
    expanded = out_dir / "units-predicted.jsonl"
    _expand_manifest(args.dedup_manifest, args.duration_checkpoint, expanded)
    synthesize(predicted_dir, args.vocoder_checkpoint, expanded)

    if args.asr == "whisper":
        oracle_asr = whisper_word_error_rate(
            oracle_dir, args.asr_manifest, model_name=args.whisper_model, batch_size=args.batch_size, device=device
        )
        predicted_asr = whisper_word_error_rate(
            predicted_dir, args.asr_manifest, model_name=args.whisper_model, batch_size=args.batch_size, device=device
        )
    else:
        from gslm2.vocoder.unit.wer.qwen3_omni import qwen3_omni_word_error_rate

        oracle_asr = qwen3_omni_word_error_rate(oracle_dir, args.asr_manifest)
        predicted_asr = qwen3_omni_word_error_rate(predicted_dir, args.asr_manifest)
    oracle_wer, oracle_cer = write_asr_results(oracle_asr, out_dir / "asr-oracle.jsonl")
    predicted_wer, predicted_cer = write_asr_results(predicted_asr, out_dir / "asr-predicted.jsonl")

    oracle_mos = write_mos_results(
        utmos_scores(oracle_dir, args.asr_manifest, device=device), out_dir / "mos-oracle.jsonl"
    )
    predicted_mos = write_mos_results(
        utmos_scores(predicted_dir, args.asr_manifest, device=device), out_dir / "mos-predicted.jsonl"
    )

    results = {
        "oracle": {"wer": oracle_wer, "cer": oracle_cer, "mos": oracle_mos},
        "predicted": {"wer": predicted_wer, "cer": predicted_cer, "mos": predicted_mos},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(
        "oracle WER=%.2f%% MOS=%.3f | predicted WER=%.2f%% MOS=%.3f",
        100 * oracle_wer,
        oracle_mos,
        100 * predicted_wer,
        predicted_mos,
    )


if __name__ == "__main__":
    main()
