"""Expand generated speech units with predicted durations and synthesize audio."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import soundfile as sf
import torch
from tqdm import tqdm

from gslm2.duration.predict import dedup_to_full, load_duration_predictor
from gslm2.shared.units import parse_units, raw_units
from gslm2.vocoder.data import read_manifest, relative_output_paths
from gslm2.vocoder.model import SAMPLE_RATE, UnitVocoder

if TYPE_CHECKING:
    from collections.abc import Sequence


@torch.inference_mode()
def synthesize(  # noqa: PLR0913
    manifest: Path,
    output_dir: Path,
    vocoder_checkpoint: Path,
    duration_checkpoint: Path,
    *,
    unit_offset: int = 0,
    special_token_ids: Sequence[int] = (),
    device: str | None = None,
) -> None:
    """Convert deduplicated JSONL units to one 16 kHz WAV per input utterance."""
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocoder = UnitVocoder.from_pretrained(vocoder_checkpoint).eval().to(resolved_device)
    duration, config = load_duration_predictor(duration_checkpoint, resolved_device)
    sizes = [vocoder.n_units] if isinstance(vocoder.n_units, int) else vocoder.n_units
    duration_vocab_size = cast("torch.nn.Embedding", duration.embeddings[0]).num_embeddings
    if duration.num_streams != len(sizes) or any(size != duration_vocab_size for size in sizes):
        raise ValueError("duration and vocoder checkpoints must use the same unit vocabularies and stream count")
    entries = read_manifest(manifest)
    paths = [path.with_suffix(".wav") for path in relative_output_paths(entries)]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest audio paths must produce unique output WAV paths")
    for entry, relative_path in tqdm(zip(entries, paths, strict=True), total=len(entries)):
        units = raw_units(
            parse_units(entry.units), sizes, unit_offset=unit_offset, special_token_ids=special_token_ids
        )
        expanded = dedup_to_full(duration, units.tolist(), log_duration=bool(config.get("log_duration", True)))
        frames = parse_units(expanded).unsqueeze(0).to(resolved_device)
        audio = vocoder.generate(frames).squeeze(0).cpu().numpy()
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, SAMPLE_RATE)


def main() -> None:
    """Synthesize generated unit sequences using duration and vocoder checkpoints."""
    parser = argparse.ArgumentParser(description="Predict speech-unit durations and synthesize 16 kHz audio")
    parser.add_argument("manifest", type=Path, help="Deduplicated single-stream or aligned multi-stream JSONL units")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--duration-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--unit-offset", type=int, default=0, help="Use 2 for speech-unit LM token IDs, 0 for raw codes"
    )
    parser.add_argument(
        "--special-token-ids", type=int, nargs="*", default=[], help="Additional LM special IDs to remove"
    )
    parser.add_argument("--device", help="Torch device; defaults to CUDA when available, otherwise CPU")
    args = parser.parse_args()
    synthesize(
        args.manifest,
        args.output_dir,
        args.vocoder_checkpoint,
        args.duration_checkpoint,
        unit_offset=args.unit_offset,
        special_token_ids=args.special_token_ids,
        device=args.device,
    )


if __name__ == "__main__":
    main()
