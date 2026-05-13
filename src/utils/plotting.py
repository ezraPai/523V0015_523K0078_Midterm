"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def plot_training_curves(metrics_csv: str | Path, save_path: str | Path, title: str | None = None) -> None:
    """Plot training loss, validation loss, and validation accuracy for one run."""
    df = pd.read_csv(metrics_csv)
    _ensure_columns(df, ["train_loss", "val_loss", "val_accuracy"])

    epochs = df.index + 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("train_loss", "Training loss"),
        ("val_loss", "Validation loss"),
        ("val_accuracy", "Validation accuracy"),
    ]

    for axis, (column, label) in zip(axes, metrics):
        sns.lineplot(x=epochs, y=df[column], ax=axis, marker="o", markersize=5)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str | Path, title: str | None = None) -> None:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(experiment_dirs: list[Path], labels: list[str], save_path: str | Path, title: str) -> None:
    """Overlay training curves from multiple experiments."""
    if len(experiment_dirs) != len(labels):
        raise ValueError("experiment_dirs and labels must have the same length")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [
        ("train_loss", "Training loss"),
        ("val_loss", "Validation loss"),
        ("val_accuracy", "Validation accuracy"),
    ]

    for exp_dir, label in zip(experiment_dirs, labels):
        df = pd.read_csv(Path(exp_dir) / "metrics.csv")
        _ensure_columns(df, ["train_loss", "val_loss", "val_accuracy"])
        epochs = df.index + 1
        for axis, (column, metric_label) in zip(axes, metrics):
            sns.lineplot(x=epochs, y=df[column], ax=axis, marker="o", markersize=4, label=label)
            axis.set_title(metric_label)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)

    axes[0].legend(title="Run")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
