# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import uuid
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from fastabx import zerospeech_abx

from gslm2.encoder.config import ModelType
from gslm2.encoder.eval.utils import extract_representations, temp_dir, validate_layers
from gslm2.shared.tools import init_logger

logger = logging.getLogger()


@dataclass
class ABXConfig:
    checkpoint: Path
    model_type: ModelType
    output_scores: Path
    manifest: Path
    item_file: Path
    speaker_mode: Literal["within", "across", "all"] = "all"
    context_mode: Literal["within", "any", "all"] = "within"
    layers: list[int] | None = None
    feature_size: float = 0.02
    codebook_proba: bool = True
    onehot: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint:
            self.checkpoint = Path(self.checkpoint).resolve()
        self.output_scores = Path(self.output_scores).resolve()
        self.manifest = Path(self.manifest).resolve()
        self.item_file = Path(self.item_file)
        if self.onehot:
            assert self.codebook_proba, "One-hot encoding requires codebook_proba"

    @property
    def distance_mode(self) -> Literal["kl_symmetric", "cosine"]:
        return "kl_symmetric" if self.codebook_proba and not self.onehot else "cosine"


def compute_abx(cfg: ABXConfig) -> pd.DataFrame:
    init_logger()
    logger.info("Extract features")

    output_features = temp_dir() / str(uuid.uuid4())
    layers = validate_layers(cfg.checkpoint, cfg.layers, codebook_proba=cfg.codebook_proba)
    extract_representations(
        cfg.checkpoint,
        cfg.model_type,
        output_features,
        [cfg.manifest],
        layers,
        codebook_proba=cfg.codebook_proba,
        onehot=cfg.onehot,
    )

    speaker_modes = ["within", "across"] if cfg.speaker_mode == "all" else [cfg.speaker_mode]
    context_modes = ["within", "any"] if cfg.context_mode == "all" else [cfg.context_mode]
    results = []
    for layer in layers:
        logger.info("Compute ABX for layer %s", layer)
        for speaker_mode, context_mode in product(speaker_modes, context_modes):
            score = zerospeech_abx(
                cfg.item_file,
                output_features / cfg.manifest.stem / str(layer),
                speaker=speaker_mode,
                context=context_mode,
                distance=cfg.distance_mode,
                frequency=int(1 / cfg.feature_size),
                max_size_group=10,
                max_x_across=5,
            )
            print(layer, speaker_mode, context_mode, score)  # noqa: T201
            results.append(
                {
                    "checkpoint": cfg.checkpoint,
                    "item": cfg.item_file.stem,
                    "layer": layer,
                    "context_mode": context_mode,
                    "speaker_mode": speaker_mode,
                    "distance_mode": cfg.distance_mode,
                    "codebook_proba": cfg.codebook_proba,
                    "onehot": cfg.onehot,
                    "score": score,
                }
            )

    logger.info("Done. Saving scores to %s", cfg.output_scores)
    results_df = pd.DataFrame(results)
    if cfg.output_scores.is_file():
        results_df.to_csv(cfg.output_scores, mode="a", header=False, index=False)
    else:
        results_df.to_csv(cfg.output_scores, index=False)
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ABX")
    parser.add_argument("checkpoint", type=Path, help="Path to the model checkpoint.")
    parser.add_argument("output_scores", type=Path, help="Path to the output scores.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest file.")
    parser.add_argument("item_file", type=Path, help="Path to the item file.")
    parser.add_argument("model_type", type=str, choices=get_args(ModelType), help="Model type.")
    parser.add_argument("--speaker_mode", type=str, choices=["within", "across", "all"], default="all")
    parser.add_argument("--context_mode", type=str, choices=["within", "any", "all"], default="within")
    parser.add_argument("--layers", type=str, help="Comma-separated list of layers to extract features from.")
    parser.add_argument("--feature_size", type=float, default=0.02, help="Feature size for ABX.")
    parser.add_argument("--codebook_proba", action="store_true", help="Use codebook probabilities.")
    parser.add_argument("--onehot", action="store_true", help="Use one-hot encoding.")
    args = parser.parse_args()

    cfg = ABXConfig(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        output_scores=args.output_scores,
        manifest=args.manifest,
        item_file=args.item_file,
        speaker_mode=args.speaker_mode,
        context_mode=args.context_mode,
        layers=[int(layer) for layer in args.layers.split(",")] if args.layers else None,
        feature_size=args.feature_size,
        codebook_proba=args.codebook_proba,
        onehot=args.onehot,
    )
    compute_abx(cfg)


if __name__ == "__main__":
    main()
