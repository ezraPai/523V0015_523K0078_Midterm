"""Re-evaluate a saved IMDb experiment without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.dataset import IMDbDataset
from src.models import build_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, confusion_matrix


def _load_vocab(vocab_path: Path) -> Any:
    """Load the saved vocabulary object."""
    return torch.load(vocab_path, map_location="cpu")


def _get_loader(split: str) -> IMDbDataset:
    """Load the requested IMDb split dataset."""
    processed_path = Path("data/processed") / f"{split}.pt"
    return IMDbDataset(processed_path)


def _predict(model: torch.nn.Module, dataset: IMDbDataset, device: torch.device) -> tuple[dict[str, float], list[list[int]], list[dict[str, Any]]]:
    """Run inference and collect metrics, confusion matrix, and prediction rows."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "texts": [item["text"] for item in batch],
    })

    all_labels: list[int] = []
    all_preds: list[int] = []
    cm_labels: list[int] = []
    rows: list[dict[str, Any]] = []

    sample_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]

            outputs = model(input_ids, lengths)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            cm_labels.extend(labels.cpu().tolist())

            for i in range(labels.size(0)):
                rows.append(
                    {
                        "idx": sample_idx,
                        "text": texts[i],
                        "label": int(labels[i].item()),
                        "pred": int(preds[i].item()),
                        "prob_pos": float(probs[i, 1].item()),
                    }
                )
                sample_idx += 1

    metrics = compute_metrics(all_labels, all_preds)
    cm = confusion_matrix(cm_labels, all_preds, num_classes=2).tolist()
    return metrics, cm, rows


def main() -> None:
    """Recompute metrics and predictions for a saved experiment."""
    parser = argparse.ArgumentParser(description="Re-evaluate a saved IMDb experiment")
    parser.add_argument("--experiment_name", required=True, help="Experiment directory name")
    parser.add_argument("--split", default="test", choices={"test", "val"}, help="Dataset split to evaluate")
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg = load_config(exp_dir / "config.yaml")
    device = torch.device(cfg.train.device)

    vocab = _load_vocab(Path("data/processed") / "vocab.pkl")
    dataset = _get_loader(args.split)

    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics, cm, rows = _predict(model, dataset, device)

    metrics_json_path = exp_dir / "metrics.json"
    with metrics_json_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    existing[args.split] = {
        **metrics,
        "confusion_matrix": cm,
    }
    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)

    pd.DataFrame(rows, columns=["idx", "text", "label", "pred", "prob_pos"]).to_csv(
        exp_dir / f"preds_{args.split}.csv",
        index=False,
    )

    print(f"Split: {args.split}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("confusion_matrix:")
    print(cm)


if __name__ == "__main__":
    main()
