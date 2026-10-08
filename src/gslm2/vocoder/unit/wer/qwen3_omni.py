"""Evaluate generated speech with Qwen3-Omni WER and CER."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from gslm2.vocoder.unit.evaluation import (
    EvaluationRecord,
    asr_record,
    load_generated_audio,
    read_asr_manifest,
    write_asr_results,
)
from gslm2.vocoder.unit.model import SAMPLE_RATE

DEFAULT_MODEL_PATH = Path("/datasets/metaspeechssl/shared/models/llms/Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_PROMPT = "Transcribe the English audio into text. Return only the transcription."
logger = logging.getLogger(__name__)


def _conversation(audio: Path, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _generated_sequences(generated: object) -> Tensor:
    if isinstance(generated, tuple):
        generated = generated[0]
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    if not isinstance(generated, Tensor):
        raise TypeError(f"unexpected output from Qwen3-Omni generate(): {type(generated)}")
    return generated


def qwen3_omni_word_error_rate(  # noqa: PLR0913
    root: str | Path,
    path_manifest: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    prompt: str = DEFAULT_PROMPT,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
) -> list[EvaluationRecord]:
    """Transcribe generated audio with Qwen3-Omni and return per-item errors."""
    from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from whisper.normalizers import EnglishTextNormalizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    entries = read_asr_manifest(path_manifest)
    model_path = Path(model_path)
    config = Qwen3OmniMoeConfig.from_pretrained(model_path, local_files_only=True)
    config.enable_audio_output = False
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype="auto",
        device_map=device_map,
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if processor.feature_extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError(
            f"expected Qwen3-Omni to consume {SAMPLE_RATE} Hz audio, got {processor.feature_extractor.sampling_rate}"
        )

    normalizer = EnglishTextNormalizer()
    records: list[EvaluationRecord] = []
    for start in tqdm(range(0, len(entries), batch_size), desc="Qwen3-Omni ASR"):
        batch = entries[start : start + batch_size]
        audio_paths = [Path(root) / entry.audio for entry in batch]
        conversations = [_conversation(audio_path, prompt) for audio_path in audio_paths]
        texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        waveforms = [load_generated_audio(root, entry.audio).squeeze(0).numpy() for entry in batch]
        inputs = processor(text=texts, audio=waveforms, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                thinker_max_new_tokens=max_new_tokens,
            )
        sequences = _generated_sequences(generated)
        generated_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        hypotheses_raw = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for entry, hypothesis_raw in zip(batch, hypotheses_raw, strict=True):
            records.append(
                asr_record(
                    entry,
                    normalizer(entry.transcript),
                    normalizer(hypothesis_raw),
                    transcript_raw=entry.transcript,
                    hypothesis_raw=hypothesis_raw,
                )
            )
    return records


def main(argv: list[str] | None = None) -> int:
    """Run Qwen3-Omni ASR evaluation and write JSONL plus aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate generated audio with Qwen3-Omni")
    parser.add_argument("root", type=Path, help="Root directory containing generated audio")
    parser.add_argument("manifest", type=Path, help="ASR JSONL with audio and transcript fields")
    parser.add_argument("output", type=Path, help="Per-item JSONL output")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Local Qwen3-Omni checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args(argv)

    records = qwen3_omni_word_error_rate(
        args.root,
        args.manifest,
        model_path=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    wer, cer = write_asr_results(records, args.output)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Qwen3-Omni WER=%.2f%% CER=%.2f%%", 100 * wer, 100 * cer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
