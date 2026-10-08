#!/usr/bin/env python3
"""Pairwise NMI + co-occurrence between codebooks (layers) of one SPidR run.

Reads per-layer JSONL files of extracted units and computes:
  - Pairwise NMI matrix (symmetric, L x L)
  - Co-occurrence heatmaps for each pair (Hungarian-aligned)

Usage:
    python scripts/analyze_codebook_pairwise_nmi.py \\
        --root results/pnmi/pnmi-clean__full__spidr-interm1.5__scheduled-weights__seed-1__step-final \\
        --split dev-clean \\
        --out results/codebook_similarity/spidr-interm1.5_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def load_layer_codes(jsonl_path: Path) -> dict[str, np.ndarray]:
    """Return {audio_path: codes_array} for one layer."""
    out: dict[str, np.ndarray] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[rec["audio"]] = np.fromstring(rec["codes"], sep=" ", dtype=np.int32)
    return out


def concat_aligned(layer_maps: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Return array [n_frames, n_layers] of code assignments, aligned per-utterance."""
    common = set(layer_maps[0].keys())
    for m in layer_maps[1:]:
        common &= set(m.keys())
    audio_keys = sorted(common)

    per_layer = []
    for m in layer_maps:
        pieces = []
        for k in audio_keys:
            pieces.append(m[k])
        per_layer.append(pieces)

    # trim to min length per utterance (frame counts should match, but be safe)
    aligned = []
    for i, k in enumerate(audio_keys):
        min_len = min(per_layer[l][i].shape[0] for l in range(len(layer_maps)))
        aligned.append(
            np.stack([per_layer[l][i][:min_len] for l in range(len(layer_maps))], axis=1)
        )
    return np.concatenate(aligned, axis=0)  # [T, L]


def cooc_matrix(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    idx = a.astype(np.int64) * nb + b.astype(np.int64)
    counts = np.bincount(idx, minlength=na * nb).reshape(na, nb)
    return counts


def hungarian_reorder(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder rows/cols of M to place mass on diagonal via Hungarian matching."""
    # Pad to square
    r, c = M.shape
    n = max(r, c)
    padded = np.zeros((n, n))
    padded[:r, :c] = M
    row_ind, col_ind = linear_sum_assignment(-padded)  # maximize
    return padded[np.ix_(row_ind, col_ind)], row_ind, col_ind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    layer_dirs = sorted(
        [p for p in args.root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    layers = [int(p.name) for p in layer_dirs]
    print(f"Layers: {layers}")

    layer_maps = [load_layer_codes(p / f"{args.split}.jsonl") for p in layer_dirs]
    codes = concat_aligned(layer_maps)  # [T, L]
    T, L = codes.shape
    print(f"Aligned frames: {T}, layers: {L}")

    # Determine codebook sizes per layer
    sizes = [int(codes[:, i].max()) + 1 for i in range(L)]
    print(f"Per-layer codebook sizes (max+1 observed): {sizes}")

    args.out.mkdir(parents=True, exist_ok=True)

    # --- NMI + ARI matrices ---
    nmi = np.zeros((L, L))
    ari = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            v = normalized_mutual_info_score(codes[:, i], codes[:, j])
            a = adjusted_rand_score(codes[:, i], codes[:, j])
            nmi[i, j] = nmi[j, i] = v
            ari[i, j] = ari[j, i] = a
        print(f"  layer {layers[i]} done")

    np.save(args.out / "nmi.npy", nmi)
    np.save(args.out / "ari.npy", ari)
    with (args.out / "layers.json").open("w") as f:
        json.dump({"layers": layers, "sizes": sizes, "n_frames": int(T)}, f, indent=2)

    # Plot NMI heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, name in [(axes[0], nmi, "NMI"), (axes[1], ari, "ARI")]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(layers)
        ax.set_yticklabels(layers)
        ax.set_title(f"Pairwise {name} between layer codebooks")
        for i in range(L):
            for j in range(L):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Codebook similarity — {args.root.name} ({args.split})")
    fig.tight_layout()
    fig.savefig(args.out / "nmi_ari_matrix.png", dpi=150)
    plt.close(fig)

    # --- Co-occurrence heatmaps (Hungarian-aligned) for each pair ---
    fig, axes = plt.subplots(L, L, figsize=(2.2 * L, 2.2 * L))
    for i in range(L):
        for j in range(L):
            ax = axes[i, j]
            if i == j:
                ax.axis("off")
                ax.text(0.5, 0.5, f"L{layers[i]}", ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)
                continue
            M = cooc_matrix(codes[:, i], codes[:, j], sizes[i], sizes[j])
            M_reord, _, _ = hungarian_reorder(M)
            # log for visibility
            ax.imshow(np.log1p(M_reord), cmap="magma", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"L{layers[j]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L{layers[i]}", fontsize=9)
    fig.suptitle(
        f"Hungarian-aligned co-occurrence log1p(count) — {args.root.name}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out / "cooccurrence_grid.png", dpi=120)
    plt.close(fig)

    print(f"\nWrote outputs to {args.out}")
    print("\nNMI matrix:")
    print("       " + "  ".join(f"L{l:>2}" for l in layers))
    for i, li in enumerate(layers):
        print(f"  L{li:>2} " + "  ".join(f"{nmi[i, j]:.2f}" for j in range(L)))


if __name__ == "__main__":
    main()
