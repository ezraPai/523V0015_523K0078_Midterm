#!/usr/bin/env python
"""Plot ablation experiment training curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot ablation comparison curves")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names matching folders under experiments/",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels in the same order as --experiments",
    )
    parser.add_argument("--title", required=True, help="Figure suptitle")
    parser.add_argument("--output", required=True, help="Output PNG path")
    return parser.parse_args()


def _load_metrics(experiment: str) -> pd.DataFrame:
    """Load metrics.csv for one experiment."""
    metrics_path = Path("experiments") / experiment / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return pd.read_csv(metrics_path)


def plot_ablation(experiments: list[str], labels: list[str], title: str, output: str | Path) -> None:
    """Plot training loss, validation loss, and validation accuracy for multiple runs."""
    if len(experiments) != len(labels):
        raise ValueError("--experiments and --labels must have the same length")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    metrics = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("val_accuracy", "Validation Accuracy", "Accuracy"),
    ]

    for exp_name, label in zip(experiments, labels):
        df = _load_metrics(exp_name)
        for ax, (column, subplot_title, y_label) in zip(axes, metrics):
            ax.plot(
                df["epoch"],
                df[column],
                marker="o",
                markersize=4,
                linewidth=2,
                label=label,
            )
            ax.set_title(subplot_title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(y_label)
            ax.legend()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    """Run the ablation plotting CLI."""
    args = _parse_args()
    plot_ablation(args.experiments, args.labels, args.title, args.output)


if __name__ == "__main__":
    main()
