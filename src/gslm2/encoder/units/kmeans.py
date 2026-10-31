# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://raw.githubusercontent.com/facebookresearch/fairseq/refs/heads/main/examples/hubert/simple_kmeans/learn_kmeans.py

import argparse
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from gslm2.encoder.units.config import KMConfig

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("learn_kmeans")


def list_files_with_extension(folder_path: Path, extension: str) -> list[Path]:
    return sorted([file for file in folder_path.iterdir() if file.is_file() and file.suffix == extension])


def _load_single_feature(file_path: Path) -> torch.Tensor:
    return torch.load(file_path)


def load_feature(feat_dir: Path) -> np.ndarray:
    feat_files = list_files_with_extension(folder_path=feat_dir, extension=".pt")
    logger.info("Loading %d feature files", len(feat_files))
    with ThreadPoolExecutor() as executor:
        feats = list(executor.map(_load_single_feature, feat_files))
    feat = torch.cat(feats, dim=0)
    logger.info("Loaded feature with dimension %s", feat.shape)
    return feat.detach().numpy()


def learn_kmeans(
    cfg: KMConfig,
    embs_folder_path: Path,
    model_save_path: Path,
) -> None:
    feat = load_feature(embs_folder_path)
    km_model = MiniBatchKMeans(
        n_clusters=cfg.n_clusters,
        init=cfg.init,
        max_iter=cfg.max_iter,
        batch_size=cfg.batch_size,
        verbose=1,
        compute_labels=False,
        tol=cfg.tol,
        max_no_improvement=cfg.max_no_improvement,
        init_size=None,
        n_init=cfg.n_init,
        reassignment_ratio=cfg.reassignment_ratio,
        random_state=cfg.seed,
    )

    logger.info(cfg)

    logger.info("Starting Fit")
    km_model.fit(feat)
    logger.info("End Fit")

    logger.info("Dumping model to %s", model_save_path)
    joblib.dump(km_model, model_save_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total inertia: %.5f", inertia)
    logger.info("finished successfully")

    if cfg.delete_embs_folder:
        # delete embs_folder_path folder
        shutil.rmtree(embs_folder_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs_folder_path",
        type=str,
        help="Path to the embeddings folder",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="Path to save the kmeans model",
    )
    parser.add_argument("--n_clusters", type=int, default=256, help="Number of clusters.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n_init", type=int, default=20, help="Number of k-means initializations.")

    args = parser.parse_args()
    cfg = KMConfig(n_clusters=args.n_clusters, seed=args.seed, n_init=args.n_init)
    embs_folder_path = Path(args.embs_folder_path)
    model_save_path = Path(args.model_save_path)
    learn_kmeans(cfg=cfg, embs_folder_path=embs_folder_path, model_save_path=model_save_path)


if __name__ == "__main__":
    main()
