"""Shared manifest, audio, and metric utilities for unit-vocoder evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import torch
from torch import Tensor

from gslm2.vocoder.unit.data import read_manifest, relative_output_paths
from gslm2.vocoder.unit.model import SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

type EvaluationRecord = dict[str, str | int | float]


@dataclass(frozen=True)
class ASREntry:
    """One generated-audio path and its reference transcript."""

    audio: Path
    transcript: str


def read_asr_manifest(path: str | Path) -> list[ASREntry]:
    """Read and validate an ``audio``/``transcript`` JSONL reference manifest."""
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError(f"ASR manifest must be JSONL: {manifest_path}")

    entries: list[ASREntry] = []
    seen_audio: set[Path] = set()
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{manifest_path}:{line_number} must contain a JSON object")
            audio, transcript = row.get("audio"), row.get("transcript")
            if not isinstance(audio, str) or not audio:
                raise TypeError(f"{manifest_path}:{line_number} needs a non-empty audio string")
            if not isinstance(transcript, str) or not transcript.strip():
                raise TypeError(f"{manifest_path}:{line_number} needs a non-empty transcript string")
            relative_audio = Path(audio)
            if relative_audio.is_absolute() or ".." in relative_audio.parts:
                raise ValueError(f"{manifest_path}:{line_number} audio must be a safe relative path: {audio}")
            if relative_audio in seen_audio:
                raise ValueError(f"duplicate audio in {manifest_path}: {audio}")
            seen_audio.add(relative_audio)
            entries.append(ASREntry(relative_audio, transcript))

    if not entries:
        raise ValueError(f"ASR manifest is empty: {manifest_path}")
    return entries


def validate_unit_asr_alignment(unit_manifest: str | Path, asr_manifest: str | Path) -> int:
    """Require unit and ASR manifests to describe the same generated filenames."""
    unit_paths = relative_output_paths(read_manifest(unit_manifest))
    reference_paths = [entry.audio for entry in read_asr_manifest(asr_manifest)]
    if len(set(unit_paths)) != len(unit_paths):
        raise ValueError("unit manifest produces duplicate output paths")
    if set(unit_paths) != set(reference_paths):
        missing = sorted(set(reference_paths) - set(unit_paths))
        extra = sorted(set(unit_paths) - set(reference_paths))
        raise ValueError(
            "unit and ASR manifests do not align by generated path: "
            f"missing units for {missing[:5]}, extra units for {extra[:5]}"
        )
    return len(reference_paths)


def load_generated_audio(root: str | Path, relative_path: Path) -> Tensor:
    """Load one generated mono 16 kHz file as a ``(1, samples)`` float tensor."""
    audio_path = Path(root) / relative_path
    if not audio_path.is_file():
        raise FileNotFoundError(f"generated audio not found: {audio_path}")
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}: {audio_path}")
    if samples.shape[1] != 1:
        raise ValueError(f"expected mono audio, got {samples.shape[1]} channels: {audio_path}")
    if samples.shape[0] == 0:
        raise ValueError(f"generated audio is empty: {audio_path}")
    return torch.from_numpy(samples[:, 0]).unsqueeze(0)


def _edit_distance(left: Sequence[object], right: Sequence[object]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def asr_record(
    entry: ASREntry,
    transcript: str,
    hypothesis: str,
    *,
    transcript_raw: str | None = None,
    hypothesis_raw: str | None = None,
) -> EvaluationRecord:
    """Build one normalized WER/CER record with edit counts."""
    transcript_words = transcript.split()
    if not transcript_words or not transcript:
        raise ValueError(f"reference is empty after normalization: {entry.audio}")
    word_ed = _edit_distance(hypothesis.split(), transcript_words)
    char_ed = _edit_distance(hypothesis, transcript)
    record: EvaluationRecord = {
        "audio": entry.audio.as_posix(),
        "transcript": transcript,
        "hypothesis": hypothesis,
        "wer": word_ed / len(transcript_words),
        "cer": char_ed / len(transcript),
        "word_ed": word_ed,
        "word_length": len(transcript_words),
        "char_ed": char_ed,
        "char_length": len(transcript),
    }
    if transcript_raw is not None:
        record["transcript_raw"] = transcript_raw
    if hypothesis_raw is not None:
        record["hypothesis_raw"] = hypothesis_raw
    return record


def write_jsonl(records: Sequence[EvaluationRecord], output: str | Path) -> Path:
    """Write evaluation records to JSONL, requiring at least one result."""
    if not records:
        raise ValueError("evaluation produced no results")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(f"{json.dumps(record)}\n")
    return output_path


def write_asr_results(records: Sequence[EvaluationRecord], output: str | Path) -> tuple[float, float]:
    """Write per-item ASR results and a micro-averaged WER/CER summary."""
    output_path = write_jsonl(records, output)
    word_ed = sum(int(record["word_ed"]) for record in records)
    word_length = sum(int(record["word_length"]) for record in records)
    char_ed = sum(int(record["char_ed"]) for record in records)
    char_length = sum(int(record["char_length"]) for record in records)
    if word_length == 0 or char_length == 0:
        raise ValueError("ASR references have zero aggregate length")
    wer, cer = word_ed / word_length, char_ed / char_length
    output_path.with_name(f"{output_path.stem}_wer.txt").write_text(
        f"WER={wer:.2%}, CER={cer:.2%}\n",
        encoding="utf-8",
    )
    return wer, cer


def write_mos_results(records: Sequence[EvaluationRecord], output: str | Path) -> float:
    """Write per-item MOS predictions and their arithmetic mean."""
    output_path = write_jsonl(records, output)
    mean_mos = sum(float(record["predicted_mos"]) for record in records) / len(records)
    output_path.with_name(f"{output_path.stem}_mos.txt").write_text(f"MOS={mean_mos:.6f}\n", encoding="utf-8")
    return mean_mos


def main(argv: list[str] | None = None) -> int:
    """Validate that a unit manifest and ASR reference manifest are aligned."""
    parser = argparse.ArgumentParser(description="Validate unit-vocoder evaluation manifests")
    parser.add_argument("unit_manifest", type=Path)
    parser.add_argument("asr_manifest", type=Path)
    args = parser.parse_args(argv)
    count = validate_unit_asr_alignment(args.unit_manifest, args.asr_manifest)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Validated %d aligned evaluation examples", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
