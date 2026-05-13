"""Main training entry point for IMDb sentiment experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from tqdm import tqdm

from src.dataset import make_loaders
from src.models import build_model
from src.utils.config import load_config, save_config, to_dict
from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    confusion_matrix,
    count_parameters,
)
from src.utils.plotting import plot_confusion_matrix, plot_training_curves
from src.utils.seed import set_seed


def _resolve_experiment_name(cli_name: str | None, cfg: Any) -> str:
    """Resolve the experiment name from CLI or config."""
    if cli_name:
        return cli_name
    run_name = getattr(cfg, "run_name", None)
    if run_name:
        return str(run_name)
    raise ValueError("An experiment name must be provided via --experiment_name or cfg.run_name")


def _get_model_output(logits_or_tuple: Any) -> tuple[Tensor, Tensor | None]:
    """Normalize model outputs to logits and optional attention weights."""
    if isinstance(logits_or_tuple, tuple):
        return logits_or_tuple[0], logits_or_tuple[1]
    return logits_or_tuple, None


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
    train: bool,
    grad_clip: float | None,
) -> dict[str, float]:
    """Run one training or validation epoch and return averaged metrics."""
    model.train(mode=train)
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        iterator = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            logits, _ = _get_model_output(model(input_ids, lengths))
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds)
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


def _load_state_if_available(model: nn.Module, optimizer: Adam | None, path: Path, device: torch.device) -> None:
    """Load a saved training state if the checkpoint exists."""
    if not path.exists():
        return
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])


def _save_checkpoint(path: Path, model: nn.Module, optimizer: Adam, epoch: int, best_val_f1: float, cfg: Any) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "config": to_dict(cfg),
        },
        path,
    )


def _evaluate_test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], list[list[int]], list[dict[str, Any]]]:
    """Evaluate a model once on the test set and collect predictions."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    rows: list[dict[str, Any]] = []

    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="test", leave=False):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]

            logits, _ = _get_model_output(model(input_ids, lengths))
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

            for i in range(labels.size(0)):
                rows.append(
                    {
                        "idx": sample_index,
                        "text": texts[i],
                        "label": int(labels[i].item()),
                        "pred": int(preds[i].item()),
                        "prob_pos": float(probs[i, 1].item()),
                    }
                )
                sample_index += 1

    test_metrics = compute_metrics(all_targets, all_preds)
    per_class = compute_per_class_metrics(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, num_classes=2).tolist()
    return test_metrics, per_class, cm, rows


def main() -> None:
    """Train a sentiment classifier, evaluate it, and save all artifacts."""
    parser = argparse.ArgumentParser(description="Train an IMDb sentiment classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--experiment_name", default=None, help="Override the experiment name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing experiment directory")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing best checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = _resolve_experiment_name(args.experiment_name, cfg)
    exp_dir = Path("experiments") / exp_name

    if exp_dir.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Experiment directory already exists: {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, exp_dir / "config.yaml")
    logger = get_logger(exp_name, exp_dir / "train.log")
    logger.info("Setup complete")

    set_seed(int(cfg.seed))
    _device_cfg = cfg.train.device
    if _device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = _device_cfg

    vocab = torch.load(Path("data/processed") / "vocab.pkl", map_location="cpu")
    train_loader, val_loader, test_loader = make_loaders(cfg)

    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    criterion = nn.CrossEntropyLoss()

    metrics_csv_path = exp_dir / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1",
            "val_loss",
            "val_accuracy",
            "val_f1",
            "learning_rate",
            "epoch_seconds",
            "is_best",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        best_val_f1 = float("-inf")
        best_epoch = 0
        best_val_acc = 0.0
        patience = 0
        start_time = time.perf_counter()

        if args.resume:
            _load_state_if_available(model, optimizer, exp_dir / "last.pt", device)

        for epoch in range(1, int(cfg.train.epochs) + 1):
            epoch_start = time.perf_counter()

            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                grad_clip=float(cfg.train.grad_clip),
            )
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
                grad_clip=None,
            )

            epoch_seconds = time.perf_counter() - epoch_start
            current_lr = float(optimizer.param_groups[0]["lr"])
            is_best = val_metrics["f1"] > best_val_f1

            if is_best:
                # Model selection is based only on validation metrics.
                best_val_f1 = val_metrics["f1"]
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                patience = 0
                _save_checkpoint(exp_dir / "best.pt", model, optimizer, epoch, best_val_f1, cfg)
            else:
                patience += 1

            if args.resume or cfg.logging.save_last_ckpt:
                _save_checkpoint(exp_dir / "last.pt", model, optimizer, epoch, best_val_f1, cfg)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "learning_rate": current_lr,
                    "epoch_seconds": epoch_seconds,
                    "is_best": int(is_best),
                }
            )
            csv_file.flush()

            logger.info(f"Epoch {epoch} completed and saved to metrics.csv")

            if patience > int(cfg.train.early_stopping_patience):
                logger.info("Early stopping triggered")
                break

    _load_state_if_available(model, None, exp_dir / "best.pt", device)

    test_metrics, per_class_metrics, cm, pred_rows = _evaluate_test(model, test_loader, device)
    num_params = count_parameters(model)
    wallclock_seconds = time.perf_counter() - start_time

    preds_df_path = exp_dir / "preds_test.csv"
    import pandas as pd

    pd.DataFrame(pred_rows).to_csv(preds_df_path, index=False)

    metrics_json = {
        "experiment_name": exp_name,
        "config_snapshot": to_dict(cfg),
        "best": {
            "epoch": int(best_epoch),
            "val_accuracy": float(best_val_acc),
            "val_f1": float(best_val_f1),
        },
        "test": {
            "accuracy": float(test_metrics["accuracy"]),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "per_class": {
                "neg": {
                    "precision": float(per_class_metrics["neg_precision"]),
                    "recall": float(per_class_metrics["neg_recall"]),
                    "f1": float(per_class_metrics["neg_f1"]),
                },
                "pos": {
                    "precision": float(per_class_metrics["pos_precision"]),
                    "recall": float(per_class_metrics["pos_recall"]),
                    "f1": float(per_class_metrics["pos_f1"]),
                },
            },
            "confusion_matrix": cm,
        },
        "wallclock_seconds": float(wallclock_seconds),
        "num_params": int(num_params),
    }

    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2)

    plot_training_curves(exp_dir / "metrics.csv", exp_dir / "training_curves.png", title=exp_name)
    plot_confusion_matrix(cm=__import__("numpy").array(cm), class_names=["neg", "pos"], save_path=exp_dir / "confusion_matrix.png", title=exp_name)

    model_type = cfg.model.type
    num_layers = getattr(cfg.model, "num_layers", 1)
    summary_md = f"""# Experiment summary — {exp_name}

**Model:** {model_type}  
**Key hyperparameters:** embed_dim={cfg.model.embed_dim}, hidden_dim={cfg.model.hidden_dim}, dropout={cfg.model.dropout}, num_layers={num_layers}  
**Parameters:** {num_params:,}  
**Wallclock:** {wallclock_seconds:.1f} s  
**Best epoch:** {best_epoch} (val_f1={best_val_f1:.4f})  

## Test metrics
| Metric | Value |
|---|---|
| Accuracy | {test_metrics['accuracy']:.4f} |
| F1 (macro) | {test_metrics['f1']:.4f} |
| Precision (macro) | {test_metrics['precision']:.4f} |
| Recall (macro) | {test_metrics['recall']:.4f} |

## Confusion matrix
See `confusion_matrix.png`.

## Training curves
See `training_curves.png`.
"""
    (exp_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    logger.info(f"Done. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
