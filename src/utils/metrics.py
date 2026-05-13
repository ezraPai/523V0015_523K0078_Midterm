"""Metric utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute aggregate classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_per_class_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute precision, recall, and f1 for each class."""
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "neg_precision": float(precision[0]),
        "neg_recall": float(recall[0]),
        "neg_f1": float(f1[0]),
        "pos_precision": float(precision[1]),
        "pos_recall": float(recall[1]),
        "pos_f1": float(f1[1]),
    }


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: int = 2) -> np.ndarray:
    """Build a confusion matrix for classification outputs."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
