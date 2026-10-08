#!/usr/bin/env python3
"""
Plot ABX scores from result CSV files.

Compares multiple CSV files (one per run/seed), averaging over seeds and
reporting the task-averaged best layer per model variant.

Usage:
  python plot_abx.py results/abx/*.csv
  python plot_abx.py results/abx/*.csv -o results/abx/my_plot.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
MIN_SEEDS_FOR_STD = 3
# Leading filename token identifying the metric, stripped from model names
METRIC_PREFIXES = {'abx', 'map', 'slm', 'tde', 'pnmi'}


# =============================================================================
# Utility Functions
# =============================================================================

def extract_step_from_path(path: str) -> int | None:
    """Extract step number from path like .../step_20000.pt or step-20000."""
    match = re.search(r'step[_-](\d+)', str(path))
    return int(match.group(1)) if match else None


def extract_model_name(path: Path, include_step: bool = False) -> str:
    """Extract model name from a result filename, dropping the seed.

    Keeps every '__'-separated token except the leading 'abx*' prefix and the
    trailing 'seed-N' / 'step-X' tokens. The corpus token is kept, so runs that
    differ only by training set (full vs LL6k) do NOT collapse into one group.

    Format: abx__full__spidr-2.0__variant__seed-0__step-final  (or map__..., etc.)
            -> full__spidr-2.0__variant (or ...__step-final if include_step=True)
    """
    name = path.parent.name if path.name == 'abx.csv' else path.name
    if path.suffix == '.csv':  # don't use .stem: version tokens contain dots
        name = name[: -len('.csv')]

    parts = name.split('__')
    # Prefix may be decorated: 'abx_triphone', 'pnmi-clean', 'slm__6kh' -> 'abx'/'pnmi'/'slm'
    if parts and re.split(r'[-_]', parts[0])[0] in METRIC_PREFIXES:
        parts = parts[1:]

    step_parts = [p for p in parts if p.startswith('step-') or p.startswith('step_')]
    parts = [
        p for p in parts
        if not re.fullmatch(r'seed[-_]\d+', p) and p not in step_parts
    ]

    if include_step and step_parts:
        parts.append(step_parts[-1])

    return '__'.join(parts) if parts else name


def model_palette(model_names) -> dict:
    """Map each model to a stable color.

    Set2 + Dark2 give 16 distinct colors; beyond that we extend with tab20 and
    then cycle, so an arbitrary number of models never raises a KeyError.
    """
    names = list(model_names)
    base = np.vstack([
        plt.cm.Set2(np.linspace(0, 1, 8)),
        plt.cm.Dark2(np.linspace(0, 1, 8)),
    ])
    if len(names) > len(base):
        base = np.vstack([base, plt.cm.tab20(np.linspace(0, 1, 20))])
    return {name: base[i % len(base)] for i, name in enumerate(names)}


def save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Save figure to file, creating parent directories if needed."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")


def load_snapshot_data(csv_files: list[Path]) -> pd.DataFrame | None:
    """Concatenate ABX result CSVs, tagging each row with its source filename."""
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
        return None

    df_all = pd.concat(all_data, ignore_index=True)
    # Group scores by model variant (including step, excluding seed suffix)
    df_all['model_name'] = df_all['filename'].apply(
        lambda x: extract_model_name(Path(x), include_step=True)
    )
    df_all['setup'] = (df_all['item'] + '|ctx:' + df_all['context_mode']
                       + '|spk:' + df_all['speaker_mode'])
    return df_all


def compute_snapshot_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """Per (setup, model, layer): mean/std/count of the score across seeds."""
    return (
        df_all.groupby(['setup', 'model_name', 'layer'])['score']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )


def plot_snapshot_comparison(df_all: pd.DataFrame, stats: pd.DataFrame,
                             output_path: str) -> None:
    """Plot ABX scores comparing models, averaging over seeds."""
    setups = df_all.groupby(['item', 'context_mode', 'speaker_mode']).size().reset_index()[
        ['item', 'context_mode', 'speaker_mode']
    ]

    n_setups = len(setups)
    n_cols = 2
    n_rows = (n_setups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # One color per model, shared across every subplot
    model_colors = model_palette(sorted(df_all['model_name'].unique()))

    for idx, (_, setup) in enumerate(setups.iterrows()):
        ax = axes[idx]
        item, context_mode, speaker_mode = setup['item'], setup['context_mode'], setup['speaker_mode']
        setup_key = f'{item}|ctx:{context_mode}|spk:{speaker_mode}'

        grouped = stats[stats['setup'] == setup_key]
        df_setup = df_all[df_all['setup'] == setup_key]

        # Plot each model variant
        for model_name in grouped['model_name'].unique():
            model_data = grouped[grouped['model_name'] == model_name].sort_values('layer')
            color = model_colors[model_name]
            ax.plot(model_data['layer'], model_data['mean'],
               marker='o', linewidth=2, markersize=6, label=model_name, color=color)
            if (model_data['count'] >= MIN_SEEDS_FOR_STD).all():
                ax.fill_between(model_data['layer'],
                       model_data['mean'] - model_data['std'],
                       model_data['mean'] + model_data['std'],
                       alpha=0.2, color=color)

        layer_scores = df_setup.groupby('layer')['score'].mean().sort_index()
        ax.plot(layer_scores.index, layer_scores.values,
               marker='s', linewidth=2, linestyle='--', markersize=6)

        dataset = 'dev-clean' if 'clean' in item else 'dev-other'
        ax.set_title(f'{dataset} | context:{context_mode} | speaker:{speaker_mode}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('ABX Score (lower is better)', fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(n_setups, len(axes)):
        axes[idx].axis('off')

    # Create single legend at the bottom, stacked vertically
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for legend
    save_figure(fig, output_path)


def print_best_layer_summary(stats: pd.DataFrame) -> None:
    """Print, per model variant, the task-averaged best layer and its scores.

    Best layer is chosen by averaging the per-setup means over all setups.
    Scores are reported as mean +/- std across seeds when at least
    MIN_SEEDS_FOR_STD seeds are available, otherwise as a bare mean.
    """
    def fmt(mean: float, std: float, count: int) -> str:
        if count >= MIN_SEEDS_FOR_STD and not pd.isna(std):
            return f'{mean:.4f} +/- {std:.4f} (n={count})'
        return f'{mean:.4f} (n={count})'

    print('\n' + '=' * 70)
    print('Best layer per model variant (task-averaged)')
    print('=' * 70)

    for model_name, model_stats in stats.groupby('model_name'):
        # Task-averaged score per layer: only keep layers present in every setup
        n_setups = model_stats['setup'].nunique()
        per_layer = model_stats.groupby('layer').agg(
            avg=('mean', 'mean'), n=('setup', 'nunique')
        )
        per_layer = per_layer[per_layer['n'] == n_setups]
        if per_layer.empty:
            continue

        best_layer = int(per_layer['avg'].idxmin())
        best_avg = per_layer.loc[best_layer, 'avg']

        rows = model_stats[model_stats['layer'] == best_layer]
        avg_std = rows['std'].mean()
        min_count = int(rows['count'].min())

        print(f'\n{model_name}')
        print(f'  Best layer: {best_layer}')
        print(f'  Task-averaged: {fmt(best_avg, avg_std, min_count)}')
        for _, row in rows.sort_values('setup').iterrows():
            label = f"    {row['setup']}:"
            print(f"{label:<52} {fmt(row['mean'], row['std'], int(row['count']))}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot ABX scores from result CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_abx.py results/abx/*.csv
  python plot_abx.py results/abx/*.csv -o results/abx/my_plot.png
        """
    )

    parser.add_argument('paths', nargs='*', help='Paths to ABX result CSV files')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for the plot. Without it, only the '
                            'summary is printed and no figure is produced.')

    args = parser.parse_args()

    if not args.paths:
        parser.print_help()
        sys.exit(1)

    paths = [Path(p) for p in args.paths]
    print(f"\nProcessing {len(paths)} CSV file(s)...")

    df_all = load_snapshot_data(paths)
    if df_all is None:
        sys.exit(1)

    stats = compute_snapshot_stats(df_all)
    print_best_layer_summary(stats)

    if args.output:
        plot_snapshot_comparison(df_all, stats, args.output)


if __name__ == '__main__':
    main()
