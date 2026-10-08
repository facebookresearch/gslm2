#!/usr/bin/env python3
"""
Plot codebook-extracted metrics per layer for multiple models.

Reads the PNMI eval output tree produced under results/pnmi/:

    results/pnmi/<run-name>/<layer>/<split>.json

Top row:    phone purity, cluster purity, PNMI
Bottom row: codebook perplexity, bitrate, average unit duration

Runs are grouped by model variant (seed suffix stripped); with several seeds the
curve shows the mean with std error bars.

Usage:
  python plot_pnmi_layerwise.py results/pnmi/pnmi-clean__full__spidr-2.0__*__seed-*
  python plot_pnmi_layerwise.py results/pnmi/pnmi-other__* --split dev-other
  python plot_pnmi_layerwise.py results/pnmi/pnmi-clean__* --no-sil -o out.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_abx import MIN_SEEDS_FOR_STD, extract_model_name, save_figure  # noqa: E402

# (json key, axis label, subplot title). Order defines the 2x3 grid layout.
PANELS = [
    ('phone_purity', 'Phone purity', 'Phone purity'),
    ('cluster_purity', 'Cluster purity', 'Cluster purity'),
    ('pnmi', 'PNMI', 'PNMI'),
    ('codebook_perplexity', 'Perplexity', 'Codebook perplexity'),
    ('bitrate_bps', 'Bitrate (bps)', 'Bitrate'),
    ('avg_code_span_ms', 'Duration (ms)', 'Avg unit duration'),
]

# Keys that have a silence-excluded counterpart
NO_SIL_KEYS = {'phone_purity', 'cluster_purity', 'pnmi'}


def resolve_keys(no_sil: bool) -> list[tuple[str, str, str]]:
    """Swap in the _no_sil metric variants when requested."""
    if not no_sil:
        return PANELS
    return [
        (f'{key}_no_sil' if key in NO_SIL_KEYS else key, label,
         f'{title} (no sil)' if key in NO_SIL_KEYS else title)
        for key, label, title in PANELS
    ]


def load_run(run_dir: Path, split: str) -> dict[int, dict]:
    """Load {layer: metrics} for one run directory."""
    results = {}
    for layer_dir in sorted(run_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.isdigit():
            continue

        json_path = layer_dir / f'{split}.json'
        if not json_path.exists():
            continue

        with open(json_path) as f:
            results[int(layer_dir.name)] = json.load(f)

    return results


def load_models(run_dirs: list[Path], split: str) -> dict[str, dict[int, dict]]:
    """Group runs by model variant. Returns {model: {seed_key: {layer: metrics}}}.

    A model can be evaluated under both the pnmi-clean and pnmi-other run
    prefixes; those collapse to the same model name, so runs are keyed by seed
    and duplicates are reported instead of silently counted as extra seeds.
    """
    models: dict[str, dict[str, dict]] = {}

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"Warning: not a directory, skipping: {run_dir}")
            continue

        layer_results = load_run(run_dir, split)
        if not layer_results:
            print(f"Warning: no {split}.json found under {run_dir.name}")
            continue

        model_name = extract_model_name(run_dir)
        # Seed key: whatever distinguishes this run from its siblings
        seed_key = run_dir.name
        for token in run_dir.name.split('__'):
            if token.startswith('seed-'):
                seed_key = token
                break

        runs = models.setdefault(model_name, {})
        if seed_key in runs:
            print(f"Warning: duplicate {seed_key} for {model_name} "
                  f"({run_dir.name}), keeping the first")
            continue
        runs[seed_key] = layer_results

    return models


def aggregate(models: dict[str, dict[str, dict]], keys: list[str]) -> dict:
    """Compute per-model, per-layer mean/std/n for each metric key."""
    stats = {}

    for model_name, runs in models.items():
        layers = sorted({layer for lr in runs.values() for layer in lr})
        per_layer = {}

        for layer in layers:
            entry = {}
            for key in keys:
                values = [
                    float(lr[layer][key])
                    for lr in runs.values()
                    if layer in lr and key in lr[layer]
                ]
                if not values:
                    entry[key] = {'mean': np.nan, 'std': np.nan, 'n': 0}
                    continue
                entry[key] = {
                    'mean': float(np.mean(values)),
                    # sample std, matching the ABX/MAP scripts
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'n': len(values),
                }
            per_layer[layer] = entry

        stats[model_name] = per_layer

    return stats


def plot_layerwise(stats: dict, panels: list[tuple[str, str, str]],
                   output_path: str, split: str) -> None:
    """Plot the 2x3 grid of codebook metrics, mean +/- std error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    model_names = sorted(stats.keys())
    cmap1 = plt.cm.Set2(np.linspace(0, 1, 8))
    cmap2 = plt.cm.Dark2(np.linspace(0, 1, 8))
    nice_colors = np.vstack([cmap1, cmap2])[:len(model_names)]
    model_colors = dict(zip(model_names, nice_colors))

    all_layers = sorted({l for per_layer in stats.values() for l in per_layer})

    for ax, (key, ylabel, title) in zip(axes, panels):
        for model_name in model_names:
            per_layer = stats[model_name]
            layers = sorted(per_layer)
            means = np.array([per_layer[l][key]['mean'] for l in layers])
            stds = np.array([per_layer[l][key]['std'] for l in layers])
            counts = [per_layer[l][key]['n'] for l in layers]

            if np.all(np.isnan(means)):
                continue

            n_seeds = max(counts) if counts else 0
            label = model_name if n_seeds <= 1 else f'{model_name} (n={n_seeds})'
            # Error bars only once enough seeds to mean anything
            yerr = stds if (n_seeds >= MIN_SEEDS_FOR_STD) else None

            ax.errorbar(layers, means, yerr=yerr, marker='o', linewidth=2,
                       markersize=6, capsize=3, capthick=1.5, elinewidth=1.5,
                       label=label, color=model_colors[model_name])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        if all_layers:
            ax.set_xticks(all_layers)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Codebook metrics per layer ({split})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, output_path)


def print_summary(stats: dict, panels: list[tuple[str, str, str]]) -> None:
    """Print a per-layer table of mean +/- std for every model."""
    keys = [key for key, _, _ in panels]
    headers = [title for _, _, title in panels]

    print('\n' + '=' * 100)
    print('Codebook metrics per layer (mean +/- std across seeds)')
    print('=' * 100)

    for model_name in sorted(stats):
        per_layer = stats[model_name]
        n_seeds = max(
            (m['n'] for entry in per_layer.values() for m in entry.values()),
            default=0,
        )
        print(f'\n{model_name}  (n={n_seeds})')

        header = 'Layer  ' + '  '.join(f'{h:>22}' for h in headers)
        print(header)
        print('-' * len(header))

        for layer in sorted(per_layer):
            cells = []
            for key in keys:
                m = per_layer[layer][key]
                if m['n'] == 0:
                    cells.append(f'{"n/a":>22}')
                elif m['n'] >= MIN_SEEDS_FOR_STD:
                    cells.append(f'{m["mean"]:>10.3f} ± {m["std"]:<9.3f}')
                else:
                    cells.append(f'{m["mean"]:>10.3f}{"":<12}')
            print(f'{layer:>5}  ' + '  '.join(cells))

    print('=' * 100 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Plot codebook metrics per layer from results/pnmi/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_pnmi_layerwise.py results/pnmi/pnmi-clean__full__spidr-2.0__*__seed-*
  python plot_pnmi_layerwise.py results/pnmi/pnmi-other__* --split dev-other
        """
    )

    parser.add_argument('paths', nargs='*',
                       help='PNMI run directories (results/pnmi/<run-name>)')
    parser.add_argument('--split', default='dev-clean',
                       choices=['dev-clean', 'dev-other'],
                       help='Which evaluation split to read (default: dev-clean)')
    parser.add_argument('--no-sil', action='store_true',
                       help='Use the silence-excluded purity/PNMI variants')
    parser.add_argument('-o', '--output', type=str,
                       default='results/pnmi/pnmi_layerwise.png',
                       help='Output path for the plot')

    args = parser.parse_args()

    if not args.paths:
        parser.print_help()
        sys.exit(1)

    panels = resolve_keys(args.no_sil)
    run_dirs = [Path(p) for p in args.paths]

    print(f"\nProcessing {len(run_dirs)} run directory(ies), split={args.split}...")
    models = load_models(run_dirs, args.split)

    if not models:
        print("Error: no usable run directories found")
        sys.exit(1)

    stats = aggregate(models, [key for key, _, _ in panels])
    print_summary(stats, panels)
    plot_layerwise(stats, panels, args.output, args.split)


if __name__ == '__main__':
    main()
