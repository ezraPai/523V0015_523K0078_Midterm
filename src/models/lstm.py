"""LSTM-based text classifier."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    """Embed tokens, encode with LSTM, and classify logits with shape [B, num_classes]."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Map input_ids [B, T] and lengths [B] to logits [B, num_classes]."""
        embeddings = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)
