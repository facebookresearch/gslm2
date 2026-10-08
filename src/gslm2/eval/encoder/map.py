import argparse
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import pandas as pd
from speech_map import build_embeddings_and_labels, mean_average_precision

from gslm2.config import ModelType
from gslm2.eval.utils import extract_representations, temp_dir, validate_layers
from gslm2.tools import init_logger

logger = logging.getLogger()


@dataclass
class MAPConfig:
    checkpoint: Path
    model_type: ModelType
    output_scores: Path
    manifest: Path
    jsonl_file: Path
    layers: list[int] | None = None
    feature_size: float = 0.02

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint).resolve()
        self.output_scores = Path(self.output_scores).resolve()
        self.manifest = Path(self.manifest).resolve()
        self.jsonl_file = Path(self.jsonl_file).resolve()


def compute_map(cfg: MAPConfig) -> pd.DataFrame:
    init_logger()
    logger.info("Extracting features")

    output_features = temp_dir() / str(uuid.uuid4())
    layers = validate_layers(cfg.checkpoint, cfg.layers, codebook_proba=False)
    extract_representations(
        cfg.checkpoint,
        cfg.model_type,
        output_features,
        [cfg.manifest],
        layers,
        codebook_proba=False,
        onehot=False,
    )

    frequency = int(1 / cfg.feature_size)
    results = []

    try:
        for layer in layers:
            logger.info("Computing MAP for layer %s, split %s", layer, cfg.manifest.stem)
            features_dir = output_features / cfg.manifest.stem / str(layer)
            embeddings, labels = build_embeddings_and_labels(
                features_dir,
                cfg.jsonl_file,
                frequency=frequency,
            )
            score = mean_average_precision(embeddings, labels)
            results.append(
                {
                    "checkpoint": cfg.checkpoint,
                    "split": cfg.manifest.stem,
                    "layer": layer,
                    "map_score": score,
                }
            )
    finally:
        logger.info("Cleaning up temporary features directory")
        shutil.rmtree(output_features, ignore_errors=True)

    logger.info("Done. Saving scores to %s", cfg.output_scores)
    results_df = pd.DataFrame(results)
    if cfg.output_scores.is_file():
        results_df.to_csv(cfg.output_scores, mode="a", header=False, index=False)
    else:
        results_df.to_csv(cfg.output_scores, index=False)
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAP@R over words with speech features")
    parser.add_argument("checkpoint", type=Path, help="Path to the model checkpoint.")
    parser.add_argument("output_scores", type=Path, help="Path to the output scores CSV.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest file.")
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL annotation file.")
    parser.add_argument("model_type", type=str, choices=get_args(ModelType), help="Model type.")
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to extract features from.",
    )
    parser.add_argument(
        "--feature_size",
        type=float,
        default=0.02,
        help="Feature frame size in seconds (default: 0.02 -> 50 Hz).",
    )
    args = parser.parse_args()

    cfg = MAPConfig(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        output_scores=args.output_scores,
        manifest=args.manifest,
        jsonl_file=args.jsonl_file,
        layers=[int(layer) for layer in args.layers.split(",")] if args.layers else None,
        feature_size=args.feature_size,
    )
    compute_map(cfg)


if __name__ == "__main__":
    main()
