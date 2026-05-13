"""Bidirectional LSTM with additive attention classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMAttention(nn.Module):
    """Embed tokens, apply BiLSTM attention, and return logits [B, num_classes] plus attention [B, T]."""

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
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            batch_first=True,
        )
        attn_dim = hidden_dim
        self.attn_proj = nn.Linear(2 * hidden_dim, attn_dim)
        self.attn_vector = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, input_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Map input_ids [B, T] and lengths [B] to logits [B, num_classes] and attention [B, T]."""
        embeddings = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, _ = self.lstm(packed)
        padded_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        max_len = input_ids.size(1)
        if padded_outputs.size(1) < max_len:
            pad_amount = max_len - padded_outputs.size(1)
            padded_outputs = torch.nn.functional.pad(padded_outputs, (0, 0, 0, pad_amount))

        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.attn_vector(torch.tanh(self.attn_proj(padded_outputs))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), padded_outputs).squeeze(1)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights
