"""Mean-pooling MLP text classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanPoolMLP(nn.Module):
    """Embed tokens, masked-mean pool, and classify logits with shape [B, num_classes]."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Map input_ids [B, T] and lengths [B] to logits [B, num_classes]."""
        embeddings = self.embedding(input_ids)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).type_as(embeddings)
        summed = (embeddings * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(-1).type_as(embeddings)
        pooled = summed / denom
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
