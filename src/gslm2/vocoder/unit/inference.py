"""Synthesize waveforms from extracted unit manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from gslm2.vocoder.unit.data import parse_units, read_manifest, relative_output_paths
from gslm2.vocoder.unit.model import SAMPLE_RATE, UnitVocoder


def synthesize(output_dir: str | Path, checkpoint: str | Path, manifest: str | Path) -> None:
    """Generate one 16 kHz waveform for every row in ``manifest``."""
    destination_root = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnitVocoder.from_pretrained(checkpoint).eval().to(device)
    entries = read_manifest(manifest)

    for entry, relative_path in tqdm(zip(entries, relative_output_paths(entries), strict=True), total=len(entries)):
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        units = parse_units(entry.units).to(device).unsqueeze(0)
        audio = model.generate(units).squeeze(0).cpu().numpy()
        sf.write(destination, audio, SAMPLE_RATE)


def main(argv: list[str] | None = None) -> int:
    """Run manifest-based unit-vocoder synthesis."""
    parser = argparse.ArgumentParser(description="Synthesize audio from discrete speech units")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    synthesize(args.output_dir, args.checkpoint, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
