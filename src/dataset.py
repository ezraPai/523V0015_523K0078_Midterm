"""IMDb dataset and dataloader utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class IMDbDataset(Dataset):
    """Dataset backed by a processed IMDb split file."""

    def __init__(self, processed_path: str | Path) -> None:
        self.processed_path = Path(processed_path)
        self.data: dict[str, Any] = torch.load(self.processed_path, map_location="cpu")

    def __len__(self) -> int:
        return int(self.data["labels"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one processed sample."""
        return {
            "input_ids": self.data["input_ids"][idx],
            "length": int(self.data["lengths"][idx].item()),
            "label": int(self.data["labels"][idx].item()),
            "text": self.data["texts"][idx],
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of IMDb samples into tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": labels,
        "texts": texts,
    }


def make_loaders(cfg: SimpleNamespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    processed_dir = Path("data/processed")
    train_dataset = IMDbDataset(processed_dir / "train.pt")
    val_dataset = IMDbDataset(processed_dir / "val.pt")
    test_dataset = IMDbDataset(processed_dir / "test.pt")

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
