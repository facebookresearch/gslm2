#!/usr/bin/env python3
"""
Plot MAP (mean average precision) scores from result CSV files.

Compares multiple CSV files (one per run/seed), averaging over seeds and
reporting the split-averaged best layer per model variant. MAP has only
clean/other splits (no context/speaker conditions), and HIGHER is better.

Usage:
  python plot_map.py results/map/map__*__seed-*
  python plot_map.py results/map/map__*__seed-* -o results/map/my_plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_abx import (  # noqa: E402
    MIN_SEEDS_FOR_STD,
    extract_model_name,
    model_palette,
    save_figure,
)

# Order splits clean-first regardless of how they appear in the CSVs
SPLIT_ORDER = ['dev-clean', 'dev-other']


def split_label(split: str) -> str:
    """zr2021-dev-clean-map -> dev-clean."""
    return split.replace('zr2021-', '').replace('-map', '')


def load_map_data(csv_files: list[Path]) -> pd.DataFrame | None:
    """Concatenate MAP result CSVs, dedup repeated evals, tag model/split."""
    all_data = []
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Warning: File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['filename'] = csv_file.name
        all_data.append(df)

    if not all_data:
        print("Error: No valid CSV files found")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Eval re-runs append to the same file, so a (checkpoint, split, layer) can
    # appear several times. Keep the latest row, else seed counts get inflated.
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(
        subset=['checkpoint', 'split', 'layer'], keep='last'
    )
    if len(df_all) < n_before:
        print(f"Dropped {n_before - len(df_all)} duplicate eval row(s) "
              f"(repeated checkpoint/split/layer)")

    df_all['split'] = df_all['split'].apply(split_label)
    df_all['model_name'] = df_all['filename'].apply(
        lambda x: extract_model_name(Path(x), include_step=True)
    )
    return df_all


def compute_map_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """Per (split, model, layer): mean/std/count of MAP across seeds."""
    return (
        df_all.groupby(['split', 'model_name', 'layer'])['map_score']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )


def ordered_splits(df_all: pd.DataFrame) -> list[str]:
    """Splits present in the data, clean first."""
    splits = [s for s in SPLIT_ORDER if s in set(df_all['split'])]
    return splits + sorted(set(df_all['split']) - set(SPLIT_ORDER))


def plot_map_comparison(df_all: pd.DataFrame, stats: pd.DataFrame,
                        output_path: str) -> None:
    """Plot MAP scores per layer for each model variant, averaging over seeds."""
    splits = ordered_splits(df_all)

    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5))
    axes = np.atleast_1d(axes).flatten()

    # Consistent colors across subplots
    model_names = sorted(df_all['model_name'].unique())
    model_colors = model_palette(model_names)

    for ax, split in zip(axes, splits):
        grouped = stats[stats['split'] == split]

        for model_name in model_names:
            model_data = grouped[grouped['model_name'] == model_name].sort_values('layer')
            if model_data.empty:
                continue
            color = model_colors[model_name]
            ax.plot(model_data['layer'], model_data['mean'],
                   marker='o', linewidth=2, markersize=6, label=model_name, color=color)
            if (model_data['count'] >= MIN_SEEDS_FOR_STD).all():
                ax.fill_between(model_data['layer'],
                       model_data['mean'] - model_data['std'],
                       model_data['mean'] + model_data['std'],
                       alpha=0.2, color=color)

        ax.set_title(f'{split} | MAP', fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('MAP Score (higher is better)', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Single legend at the bottom, stacked vertically
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for legend
    save_figure(fig, output_path)


def print_best_layer_summary(stats: pd.DataFrame) -> None:
    """Print, per model variant, the split-averaged best layer and its scores.

    Best layer maximizes the score averaged over all splits (MAP: higher is
    better). Scores are reported as mean +/- std across seeds when at least
    MIN_SEEDS_FOR_STD seeds are available, otherwise as a bare mean.
    """
    def fmt(mean: float, std: float, count: int) -> str:
        if count >= MIN_SEEDS_FOR_STD and not pd.isna(std):
            return f'{mean:.4f} +/- {std:.4f} (n={count})'
        return f'{mean:.4f} (n={count})'

    print('\n' + '=' * 70)
    print('Best layer per model variant (split-averaged, higher MAP is better)')
    print('=' * 70)

    for model_name, model_stats in stats.groupby('model_name'):
        # Split-averaged score per layer: only keep layers present in every split
        n_splits = model_stats['split'].nunique()
        per_layer = model_stats.groupby('layer').agg(
            avg=('mean', 'mean'), n=('split', 'nunique')
        )
        per_layer = per_layer[per_layer['n'] == n_splits]
        if per_layer.empty:
            continue

        best_layer = int(per_layer['avg'].idxmax())
        best_avg = per_layer.loc[best_layer, 'avg']

        rows = model_stats[model_stats['layer'] == best_layer]
        avg_std = rows['std'].mean()
        min_count = int(rows['count'].min())

        print(f'\n{model_name}')
        print(f'  Best layer: {best_layer}')
        print(f'  Split-averaged: {fmt(best_avg, avg_std, min_count)}')
        for _, row in rows.sort_values('split').iterrows():
            label = f"    {row['split']}:"
            print(f"{label:<52} {fmt(row['mean'], row['std'], int(row['count']))}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Plot MAP scores from result CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_map.py results/map/map__*__seed-*
  python plot_map.py results/map/map__*__seed-* -o results/map/my_plot.png
        """
    )

    parser.add_argument('paths', nargs='*', help='Paths to MAP result CSV files')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for the plot. Without it, only the '
                            'summary is printed and no figure is produced.')

    args = parser.parse_args()

    if not args.paths:
        parser.print_help()
        sys.exit(1)

    paths = [Path(p) for p in args.paths]
    print(f"\nProcessing {len(paths)} CSV file(s)...")

    df_all = load_map_data(paths)
    if df_all is None:
        sys.exit(1)

    stats = compute_map_stats(df_all)
    print_best_layer_summary(stats)

    if args.output:
        plot_map_comparison(df_all, stats, args.output)


if __name__ == '__main__':
    main()
