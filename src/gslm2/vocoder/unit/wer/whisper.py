"""Evaluate generated speech with Whisper WER and CER."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gslm2.vocoder.unit.evaluation import (
    ASREntry,
    EvaluationRecord,
    asr_record,
    load_generated_audio,
    read_asr_manifest,
    write_asr_results,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class WhisperDataset(Dataset[tuple[Tensor, str, str]]):
    """Load generated audio and convert it to fixed-size Whisper mel features."""

    def __init__(
        self,
        root: str | Path,
        entries: list[ASREntry],
        audio_transform: Callable[[Tensor], Tensor],
    ) -> None:
        self.root = Path(root)
        self.entries = entries
        self.audio_transform = audio_transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, str, str]:
        entry = self.entries[index]
        audio = load_generated_audio(self.root, entry.audio).squeeze(0)
        return self.audio_transform(audio), entry.transcript, entry.audio.as_posix()


def whisper_word_error_rate(
    root: str | Path,
    path_manifest: str | Path,
    *,
    model_name: str = "large-v3",
    batch_size: int = 16,
    device: str = "cuda",
) -> list[EvaluationRecord]:
    """Transcribe generated audio with Whisper and return per-item errors."""
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    entries = read_asr_manifest(path_manifest)
    model = whisper.load_model(model_name, device=device).eval()

    def audio_transform(audio: Tensor) -> Tensor:
        padded = whisper.pad_or_trim(audio)
        return whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)

    loader = DataLoader(WhisperDataset(root, entries, audio_transform), batch_size=batch_size, drop_last=False)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    normalizer = EnglishTextNormalizer()
    records: list[EvaluationRecord] = []
    for mels, transcripts, audio_paths in tqdm(loader, desc="Whisper ASR"):
        decoded = model.decode(mels.to(device), options)
        if not isinstance(decoded, list):
            decoded = [decoded]
        hypotheses = [result.text for result in decoded]
        for audio, transcript_raw, hypothesis_raw in zip(audio_paths, transcripts, hypotheses, strict=True):
            entry = ASREntry(Path(audio), transcript_raw)
            records.append(
                asr_record(
                    entry,
                    normalizer(transcript_raw),
                    normalizer(hypothesis_raw),
                    transcript_raw=transcript_raw,
                    hypothesis_raw=hypothesis_raw,
                )
            )
    return records


def main(argv: list[str] | None = None) -> int:
    """Run Whisper ASR evaluation and write JSONL plus aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate generated audio with Whisper")
    parser.add_argument("root", type=Path, help="Root directory containing generated audio")
    parser.add_argument("manifest", type=Path, help="ASR JSONL with audio and transcript fields")
    parser.add_argument("output", type=Path, help="Per-item JSONL output")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    records = whisper_word_error_rate(
        args.root,
        args.manifest,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    wer, cer = write_asr_results(records, args.output)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Whisper WER=%.2f%% CER=%.2f%%", 100 * wer, 100 * cer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
