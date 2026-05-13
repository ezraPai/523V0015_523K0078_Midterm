"""Model factory utilities."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from .bilstm_attn import BiLSTMAttention
from .lstm import LSTMClassifier
from .mlp import MeanPoolMLP


def build_model(cfg: SimpleNamespace, vocab_size: int) -> nn.Module:
    """Build a model from cfg.model.type and return an nn.Module."""
    model_type = cfg.model.type
    pad_idx = getattr(cfg.data, "pad_idx", 0)

    if model_type == "mlp":
        return MeanPoolMLP(
            vocab_size=vocab_size,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            pad_idx=pad_idx,
        )
    if model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            pad_idx=pad_idx,
        )
    if model_type == "bilstm_attn":
        return BiLSTMAttention(
            vocab_size=vocab_size,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            pad_idx=pad_idx,
        )
    raise ValueError(f"Unknown model type: {model_type}")
