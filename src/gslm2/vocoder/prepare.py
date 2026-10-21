"""Convert full-rate encoder JSONL units to vocoder manifests and duration shards."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gslm2.shared.units import parse_units, raw_units
from gslm2.vocoder.data import iter_manifest
from gslm2.vocoder.model import SAMPLE_RATE, UNITS_HOP_SIZE


def prepare_manifest(source: Path, output_dir: Path, split: str, *, num_streams: int = 1) -> None:
    """Write raw full-rate JSONL and stream-concatenated duration-training TXT."""
    if num_streams not in {1, 6}:
        raise ValueError("num_streams must be 1 or 6")
    if split not in {"train", "valid"}:
        raise ValueError("split must be train or valid")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}.jsonl"
    shard_path = shard_dir / "shard_0.txt"
    if manifest_path.exists() or shard_path.exists():
        raise FileExistsError("prepared split already exists; choose a new output directory")
    with TemporaryDirectory(prefix=".prepare-", dir=output_dir) as staging:
        staged_manifest = Path(staging) / "manifest.jsonl"
        staged_shard = Path(staging) / "shard.txt"
        with (
            staged_manifest.open("w", encoding="utf-8") as manifest,
            staged_shard.open("w", encoding="utf-8") as shard,
        ):
            for entry in iter_manifest(source, require_full_rate=True):
                units = raw_units(parse_units(entry.units), [256] * num_streams)
                record = {
                    "audio": str(Path(entry.audio).expanduser().resolve()),
                    "units": units.tolist(),
                    "num_streams": num_streams,
                    "unit_offset": 0,
                    "sample_rate": SAMPLE_RATE,
                    "units_hop_size": UNITS_HOP_SIZE,
                    "deduplicated": False,
                }
                manifest.write(json.dumps(record) + "\n")
                shard.write(" ".join(str(token) for token in units.flatten().tolist()) + "\n")
        # Publish validated artifacts without replacing an existing split.
        manifest_path.hardlink_to(staged_manifest)
        try:
            shard_path.hardlink_to(staged_shard)
        except OSError:
            manifest_path.unlink()
            raise


def main() -> None:
    """Prepare synthesis training artifacts from full-rate encoder units."""
    parser = argparse.ArgumentParser(description="Prepare full-rate vocoder and duration training data")
    parser.add_argument("manifest", type=Path, help="Unchunked, non-deduplicated encoder JSONL units")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--num-streams", type=int, choices=[1, 6], default=1)
    args = parser.parse_args()
    prepare_manifest(args.manifest, args.output_dir, args.split, num_streams=args.num_streams)


if __name__ == "__main__":
    main()
