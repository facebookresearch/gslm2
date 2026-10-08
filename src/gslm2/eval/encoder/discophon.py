import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from discophon.abx import continuous_abx, discrete_abx
from discophon.data import STEP_UNITS, read_gold_annotations
from discophon.evaluate.discovery import phoneme_discovery
from discophon.languages import get_language

from gslm2.config import ModelType
from gslm2.eval.utils import extract_codes, extract_representations, temp_dir, validate_layers
from gslm2.tools import init_logger

logger = logging.getLogger()


@dataclass
class DiscophonConfig:
    checkpoint: Path
    model_type: ModelType
    manifest: Path  # audio manifest for feature extraction
    alignments: Path  # gold phone alignment .txt file (#file onset offset #phone)
    item_file: Path  # ABX item file
    output_scores: Path  # CSV to append results to
    language: str  # ISO 639-3 code, e.g. "deu", "eng"
    output_dir: Path | None = None  # for intermediate features; uses temp dir if None
    layers: list[int] | None = None
    n_units: int = 256
    step_units: int = STEP_UNITS  # ms between consecutive units (default 20ms = 50 Hz)
    assignment: Literal["many-to-one", "one-to-one"] = "many-to-one"
    abx_kind: str = "triphone"

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint).resolve()
        self.manifest = Path(self.manifest).resolve()
        self.alignments = Path(self.alignments).resolve()
        self.item_file = Path(self.item_file).resolve()
        self.output_scores = Path(self.output_scores).resolve()
        self.output_scores.parent.mkdir(parents=True, exist_ok=True)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).resolve()

    @property
    def frequency(self) -> int:
        """Feature frequency in Hz, inverse of step_units (int for fastabx)."""
        if 1000 % self.step_units != 0:
            raise ValueError(f"step_units={self.step_units} must divide 1000")
        return 1000 // self.step_units

    @property
    def n_phonemes(self) -> int:
        return get_language(self.language).n_phonemes


def _read_codes_jsonl(jsonl_path: Path) -> dict[str, list[int]]:
    """Convert extract_codes JSONL (audio=full_path, codes=space-separated) to {fileid: list[int]}."""
    units: dict[str, list[int]] = {}
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            fileid = Path(row["audio"]).stem
            units[fileid] = list(map(int, row["codes"].split()))
    return units


def _write_discophon_units_jsonl(units: dict[str, list[int]], path: Path) -> None:
    """Write units in discophon JSONL format (fields: file, units as list of ints)."""
    with path.open("w") as f:
        for fileid, unit_list in units.items():
            f.write(json.dumps({"file": fileid, "units": unit_list}) + "\n")


def compute_discophon(cfg: DiscophonConfig) -> pd.DataFrame:
    init_logger()

    output_dir = Path(cfg.output_dir) if cfg.output_dir else temp_dir() / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    cont_dir = output_dir / "continuous"

    layers = validate_layers(cfg.checkpoint, cfg.layers, codebook_proba=True)

    logger.info("Extracting discrete codes")
    extract_codes(cfg.checkpoint, cfg.model_type, output_dir, [cfg.manifest], layers)

    logger.info("Extracting continuous representations")
    extract_representations(
        cfg.checkpoint,
        cfg.model_type,
        cont_dir,
        [cfg.manifest],
        layers,
        codebook_proba=False,
        onehot=False,
    )

    logger.info("Reading gold phone annotations from %s", cfg.alignments)
    phones = read_gold_annotations(cfg.alignments)

    results = []
    manifest_stem = cfg.manifest.stem

    for layer in layers:
        logger.info("Computing discophon metrics for layer %s", layer)

        # Load and align discrete units
        units = _read_codes_jsonl(output_dir / f"{layer}/{manifest_stem}.jsonl")
        common_ids = set(units) & set(phones)
        if not common_ids:
            logger.warning("No common file IDs between units and alignments for layer %s", layer)
            continue
        units_common = {k: units[k] for k in common_ids}
        phones_common = {k: phones[k] for k in common_ids}

        # Phoneme discovery: pnmi, per, f1, r_val
        logger.info("Running phoneme discovery for layer %s", layer)
        discovery = phoneme_discovery(
            units_common,
            phones_common,
            kind=cfg.assignment,
            n_units=cfg.n_units,
            n_phonemes=cfg.n_phonemes,
            step_units=cfg.step_units,
        )

        # Write discophon-format JSONL for discrete ABX
        discophon_jsonl = output_dir / f"{layer}/{manifest_stem}_discophon.jsonl"
        _write_discophon_units_jsonl(units_common, discophon_jsonl)

        # Discrete ABX
        logger.info("Running discrete ABX for layer %s", layer)
        disc_abx = discrete_abx(cfg.item_file, discophon_jsonl, frequency=cfg.frequency, kind=cfg.abx_kind)

        # Continuous ABX
        logger.info("Running continuous ABX for layer %s", layer)
        cont_features_dir = cont_dir / manifest_stem / str(layer)
        cont_abx = continuous_abx(cfg.item_file, cont_features_dir, frequency=cfg.frequency, kind=cfg.abx_kind)

        row = {
            "checkpoint": str(cfg.checkpoint),
            "language": cfg.language,
            "layer": layer,
            "manifest": manifest_stem,
            "kind": cfg.assignment,
            "pnmi": discovery["pnmi"],
            "per": discovery["per"],
            "f1": discovery["f1"],
            "r_val": discovery["r_val"],
            "abx_discrete_within": disc_abx["within_speaker"],
            "abx_discrete_across": disc_abx["across_speaker"],
            "abx_continuous_within": cont_abx["within_speaker"],
            "abx_continuous_across": cont_abx["across_speaker"],
        }
        logger.info(
            "Layer %s: pnmi=%.4f per=%.4f f1=%.4f r_val=%.4f "
            "abx_disc_w=%.4f abx_disc_a=%.4f abx_cont_w=%.4f abx_cont_a=%.4f",
            layer,
            row["pnmi"],
            row["per"],
            row["f1"],
            row["r_val"],
            row["abx_discrete_within"],
            row["abx_discrete_across"],
            row["abx_continuous_within"],
            row["abx_continuous_across"],
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    if cfg.output_scores.is_file():
        results_df.to_csv(cfg.output_scores, mode="a", header=False, index=False)
    else:
        results_df.to_csv(cfg.output_scores, index=False)
    logger.info("Done. Scores saved to %s", cfg.output_scores)
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Discophon evaluation: pnmi, per, f1, r_val, abx")
    parser.add_argument("checkpoint", type=Path, help="Path to the model checkpoint.")
    parser.add_argument("model_type", type=str, choices=get_args(ModelType), help="Model type.")
    parser.add_argument("manifest", type=Path, help="Path to the audio manifest CSV.")
    parser.add_argument("alignments", type=Path, help="Path to the gold phone alignment .txt file.")
    parser.add_argument("item_file", type=Path, help="Path to the ABX item file.")
    parser.add_argument("output_scores", type=Path, help="Path to the output scores CSV.")
    parser.add_argument("language", type=str, help="ISO 639-3 language code (e.g. 'deu', 'eng').")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for intermediate features.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices.")
    parser.add_argument("--n_units", type=int, default=256, help="Number of discrete units.")
    parser.add_argument("--step_units", type=int, default=STEP_UNITS, help="Step between units in ms.")
    parser.add_argument("--assignment", type=str, choices=["many-to-one", "one-to-one"], default="many-to-one")
    parser.add_argument("--abx_kind", type=str, default="triphone", help="ABX triplet kind (e.g. 'triphone').")
    args = parser.parse_args()

    cfg = DiscophonConfig(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        manifest=args.manifest,
        alignments=args.alignments,
        item_file=args.item_file,
        output_scores=args.output_scores,
        language=args.language,
        output_dir=args.output_dir,
        layers=[int(layer) for layer in args.layers.split(",")] if args.layers else None,
        n_units=args.n_units,
        step_units=args.step_units,
        assignment=args.assignment,
    )
    compute_discophon(cfg)


if __name__ == "__main__":
    main()
