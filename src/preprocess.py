"""IMDb sentiment preprocessing pipeline."""

from __future__ import annotations

import argparse
import html
import pickle
import re
from collections import Counter
import sys
from pathlib import Path
from typing import Iterable, Iterator

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed


_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize IMDb review text."""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?;:'\-()\"]+", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split text into regex-based tokens."""
    return _TOKEN_RE.findall(text)


class Vocabulary:
    """Token-to-id vocabulary for text encoding."""

    def __init__(self, token_to_idx: dict[str, int] | None = None, special_tokens: tuple[str, str] = ("<pad>", "<unk>")) -> None:
        self.token_to_idx = token_to_idx or {special_tokens[0]: 0, special_tokens[1]: 1}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    @property
    def pad_idx(self) -> int:
        """Return the padding token index."""
        return self.token_to_idx["<pad>"]

    @property
    def unk_idx(self) -> int:
        """Return the unknown token index."""
        return self.token_to_idx["<unk>"]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @classmethod
    def build_from_iterator(
        cls,
        token_iter: Iterable[list[str]],
        max_size: int,
        min_freq: int,
        special_tokens: tuple[str, str] = ("<pad>", "<unk>"),
    ) -> "Vocabulary":
        """Build a vocabulary from token sequences."""
        counter: Counter[str] = Counter()
        for tokens in token_iter:
            counter.update(tokens)

        token_to_idx = {special_tokens[0]: 0, special_tokens[1]: 1}
        sorted_tokens = sorted(
            [token for token, freq in counter.items() if freq >= min_freq],
            key=lambda token: (-counter[token], token),
        )
        for token in sorted_tokens:
            if token in token_to_idx:
                continue
            if len(token_to_idx) >= max_size:
                break
            token_to_idx[token] = len(token_to_idx)
        return cls(token_to_idx=token_to_idx, special_tokens=special_tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to vocabulary indices."""
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Map vocabulary indices back to tokens."""
        return [self.idx_to_token.get(idx, "<unk>") for idx in ids]

    def save(self, path: str | Path) -> None:
        """Serialize the vocabulary with pickle."""
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from pickle."""
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def pad_or_truncate(ids: list[int], max_len: int, pad_idx: int) -> tuple[list[int], int]:
    """Pad or truncate encoded ids to a fixed length."""
    true_length = len(ids)
    if len(ids) >= max_len:
        return ids[:max_len], min(true_length, max_len)
    return ids + [pad_idx] * (max_len - len(ids)), true_length


def _normalize_labels(labels: list[int]) -> list[int]:
    return [int(label) for label in labels]


def _prepare_split(texts: list[str], labels: list[int]) -> tuple[list[str], list[list[str]]]:
    cleaned_texts: list[str] = []
    tokenized_texts: list[list[str]] = []
    for text in tqdm(texts, desc="Cleaning/tokenizing", leave=False):
        cleaned = clean_text(text)
        cleaned_texts.append(cleaned)
        tokenized_texts.append(tokenize(cleaned))
    return cleaned_texts, tokenized_texts


def _build_tensor_split(texts: list[str], labels: list[int], vocab: Vocabulary, max_len: int) -> dict[str, torch.Tensor | list[str]]:
    encoded_sequences: list[list[int]] = []
    lengths: list[int] = []
    raw_lengths: list[int] = []
    unk_count = 0
    total_count = 0

    for tokens in tqdm(texts, desc="Encoding/padding", leave=False):
        encoded = vocab.encode(tokens.split()) if isinstance(tokens, str) else vocab.encode(tokens)
        total_count += len(encoded)
        unk_count += sum(1 for token_id in encoded if token_id == vocab.unk_idx)
        padded, true_len = pad_or_truncate(encoded, max_len, vocab.pad_idx)
        encoded_sequences.append(padded)
        lengths.append(true_len)
        raw_lengths.append(len(encoded))

    input_ids = torch.tensor(encoded_sequences, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    raw_lengths_tensor = torch.tensor(raw_lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "lengths": lengths_tensor,
        "raw_lengths": raw_lengths_tensor,
        "labels": labels_tensor,
        "texts": texts,
        "_unk_count": torch.tensor(unk_count, dtype=torch.long),
        "_total_count": torch.tensor(total_count, dtype=torch.long),
    }


def _encode_split(tokenized_texts: list[list[str]], labels: list[int], vocab: Vocabulary, max_len: int, original_texts: list[str]) -> tuple[dict[str, torch.Tensor | list[str]], float]:
    encoded_sequences: list[list[int]] = []
    lengths: list[int] = []
    raw_lengths: list[int] = []
    unk_count = 0
    total_count = 0

    for tokens in tqdm(tokenized_texts, desc="Encoding/padding", leave=False):
        encoded = vocab.encode(tokens)
        total_count += len(encoded)
        unk_count += sum(1 for token_id in encoded if token_id == vocab.unk_idx)
        padded, true_len = pad_or_truncate(encoded, max_len, vocab.pad_idx)
        encoded_sequences.append(padded)
        lengths.append(true_len)
        raw_lengths.append(len(encoded))

    data = {
        "input_ids": torch.tensor(encoded_sequences, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "raw_lengths": torch.tensor(raw_lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": original_texts,
    }
    oov_rate = (unk_count / total_count * 100.0) if total_count else 0.0
    return data, oov_rate


def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Preprocess IMDb sentiment data")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    raw = load_dataset("imdb", cache_dir=cfg.data.cache_dir)
    train_texts = list(raw["train"]["text"])
    train_labels = _normalize_labels(list(raw["train"]["label"]))
    test_texts = list(raw["test"]["text"])
    test_labels = _normalize_labels(list(raw["test"]["label"]))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=cfg.data.val_size,
        random_state=int(cfg.seed),
        stratify=train_labels,
    )

    train_clean, train_tokens = _prepare_split(train_texts, train_labels)
    val_clean, val_tokens = _prepare_split(val_texts, val_labels)
    test_clean, test_tokens = _prepare_split(test_texts, test_labels)

    vocab = Vocabulary.build_from_iterator(
        train_tokens,
        max_size=int(cfg.data.vocab_size),
        min_freq=int(cfg.data.min_freq),
    )

    train_data, _ = _encode_split(train_tokens, train_labels, vocab, int(cfg.data.max_len), train_clean)
    val_data, val_oov = _encode_split(val_tokens, val_labels, vocab, int(cfg.data.max_len), val_clean)
    test_data, test_oov = _encode_split(test_tokens, test_labels, vocab, int(cfg.data.max_len), test_clean)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, processed_dir / "train.pt")
    torch.save(val_data, processed_dir / "val.pt")
    torch.save(test_data, processed_dir / "test.pt")
    vocab.save(processed_dir / "vocab.pkl")

    print("Preprocessing summary")
    print(f"Train size: {len(train_data['labels'])}")
    print(f"Val size: {len(val_data['labels'])}")
    print(f"Test size: {len(test_data['labels'])}")
    print(f"Val OOV rate: {val_oov:.2f}%")
    print(f"Test OOV rate: {test_oov:.2f}%")


if __name__ == "__main__":
    main()
