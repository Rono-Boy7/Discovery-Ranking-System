"""
src/retrieval/two_tower_dataset.py

Purpose
-------
Dataset utilities for the neural two-tower retrieval model.

Why this file matters
---------------------
Our current retrieval baseline is TF-IDF, which is lexical and does not learn
dense embeddings.

A two-tower model needs training data in a different form:

- user tower input
- item tower input
- binary label

This module prepares that data and provides:
- tokenization
- vocabulary building
- text encoding
- PyTorch Dataset objects
- collate function for batching

What this module does
---------------------
1. Reuse retrieval examples from src.retrieval.dataset
2. Build:
   - user_text
   - item_text
3. Build a training vocabulary from train text
4. Encode text into integer token IDs
5. Provide torch Dataset / DataLoader-ready utilities

Modeling scope
--------------
This is a lean first two-tower:
- user input = user_profile_text
- item input = category + subcategory + title + abstract
- encoder = embedding layer + masked mean pooling (in the next file)

Run local smoke test
--------------------
From the repo root:
    python3 -m src.retrieval.two_tower_dataset
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.retrieval.dataset import (
    RetrievalDatasetConfig,
    build_retrieval_examples,
)


# ------------------------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1


# ------------------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize_text(text: object) -> list[str]:
    """
    Simple tokenizer for the first two-tower model.

    Rules
    -----
    - lowercase
    - keep only alphanumeric tokens
    - return list of tokens

    Why keep it simple?
    -------------------
    We want the first neural retriever to be easy to understand and debug.
    """
    if pd.isna(text):
        return []

    text = str(text).lower().strip()
    if not text:
        return []

    return _TOKEN_PATTERN.findall(text)


# ------------------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------------------

@dataclass
class VocabularyConfig:
    """
    Configuration for building the token vocabulary.

    Attributes
    ----------
    min_freq : int
        Minimum token frequency required to keep a token.
    max_vocab_size : int | None
        Optional maximum vocabulary size including special tokens.
    """

    min_freq: int = 2
    max_vocab_size: int | None = 30000


class Vocabulary:
    """
    Lightweight token vocabulary.

    Features
    --------
    - special tokens: PAD, UNK
    - token -> id mapping
    - id -> token mapping
    - encode / decode helpers
    """

    def __init__(self) -> None:
        self.token_to_idx: dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx_to_token: list[str] = [PAD_TOKEN, UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def add_token(self, token: str) -> None:
        """
        Add a token if it does not already exist.
        """
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def lookup_token(self, token: str) -> int:
        """
        Return the token ID, falling back to UNK_IDX if missing.
        """
        return self.token_to_idx.get(token, UNK_IDX)

    def lookup_idx(self, idx: int) -> str:
        """
        Return the token string for a given ID.
        """
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
        return UNK_TOKEN

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        """
        Convert tokens into integer IDs.
        """
        return [self.lookup_token(token) for token in tokens]

    def decode_ids(self, ids: Iterable[int]) -> list[str]:
        """
        Convert integer IDs back into token strings.
        """
        return [self.lookup_idx(idx) for idx in ids]


def build_vocabulary_from_texts(
    texts: Iterable[str],
    config: VocabularyConfig | None = None,
) -> Vocabulary:
    """
    Build a vocabulary from an iterable of text strings.

    Parameters
    ----------
    texts : Iterable[str]
        Training texts used to build the vocabulary.
    config : VocabularyConfig | None
        Vocabulary settings.

    Returns
    -------
    Vocabulary
        Built vocabulary object.
    """
    config = config or VocabularyConfig()

    if config.min_freq <= 0:
        raise ValueError(f"min_freq must be positive. Got: {config.min_freq}")

    counter: Counter[str] = Counter()

    for text in texts:
        tokens = tokenize_text(text)
        counter.update(tokens)

    vocab = Vocabulary()

    # Sort by frequency descending, then token ascending for reproducibility.
    candidate_tokens = sorted(
        (
            (token, freq)
            for token, freq in counter.items()
            if freq >= config.min_freq
        ),
        key=lambda x: (-x[1], x[0]),
    )

    if config.max_vocab_size is not None:
        if config.max_vocab_size < len(vocab):
            raise ValueError(
                f"max_vocab_size ({config.max_vocab_size}) is too small to hold "
                f"special tokens."
            )

        max_non_special_tokens = config.max_vocab_size - len(vocab)
        candidate_tokens = candidate_tokens[:max_non_special_tokens]

    for token, _ in candidate_tokens:
        vocab.add_token(token)

    return vocab


# ------------------------------------------------------------------------------
# Text frame building
# ------------------------------------------------------------------------------

def _safe_text(value: object) -> str:
    """
    Convert a possibly missing value into a clean string.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_item_text(row: pd.Series) -> str:
    """
    Build the item-side text for the two-tower model.

    We include:
    - category
    - subcategory
    - title
    - abstract

    This gives the item tower richer signal than title alone.
    """
    category = _safe_text(row.get("candidate_category", ""))
    subcategory = _safe_text(row.get("candidate_subcategory", ""))
    title = _safe_text(row.get("candidate_title", ""))
    abstract = _safe_text(row.get("candidate_abstract", ""))

    text = (
        f"[CAT] {category} "
        f"[SUBCAT] {subcategory} "
        f"[TITLE] {title} "
        f"[ABSTRACT] {abstract}"
    ).strip()

    return text


def build_two_tower_text_frame(retrieval_examples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert retrieval examples into a clean two-tower text frame.

    Parameters
    ----------
    retrieval_examples_df : pd.DataFrame
        Output of src.retrieval.dataset.build_retrieval_examples().

    Returns
    -------
    pd.DataFrame
        DataFrame with the main text inputs for the two-tower model.

    Output columns
    --------------
    split
    impression_id
    user_id
    candidate_news_id
    clicked
    user_text
    item_text
    """
    required_columns = {
        "split",
        "impression_id",
        "user_id",
        "candidate_news_id",
        "clicked",
        "user_profile_text",
        "candidate_category",
        "candidate_subcategory",
        "candidate_title",
        "candidate_abstract",
    }
    missing = required_columns - set(retrieval_examples_df.columns)
    if missing:
        raise ValueError(
            "Retrieval examples are missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = retrieval_examples_df[
        [
            "split",
            "impression_id",
            "user_id",
            "candidate_news_id",
            "clicked",
            "user_profile_text",
            "candidate_category",
            "candidate_subcategory",
            "candidate_title",
            "candidate_abstract",
        ]
    ].copy()

    frame["user_text"] = frame["user_profile_text"].fillna("").astype(str).str.strip()
    frame["item_text"] = frame.apply(build_item_text, axis=1)

    output_df = frame[
        [
            "split",
            "impression_id",
            "user_id",
            "candidate_news_id",
            "clicked",
            "user_text",
            "item_text",
        ]
    ].copy()

    output_df["clicked"] = output_df["clicked"].astype("int8")
    return output_df


# ------------------------------------------------------------------------------
# Encoding helpers
# ------------------------------------------------------------------------------

@dataclass
class TextEncodingConfig:
    """
    Configuration for encoding text into token IDs.

    Attributes
    ----------
    max_user_tokens : int
        Max token count kept for the user-side text.
    max_item_tokens : int
        Max token count kept for the item-side text.
    """

    max_user_tokens: int = 64
    max_item_tokens: int = 64


def encode_text(
    text: str,
    vocab: Vocabulary,
    max_tokens: int,
) -> list[int]:
    """
    Tokenize and encode a text string into token IDs.

    Notes
    -----
    - text is truncated to max_tokens
    - if empty after tokenization, returns [UNK_IDX] so the model always sees
      at least one token
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive. Got: {max_tokens}")

    tokens = tokenize_text(text)

    if not tokens:
        return [UNK_IDX]

    token_ids = vocab.encode_tokens(tokens[:max_tokens])

    if not token_ids:
        return [UNK_IDX]

    return token_ids


def encode_two_tower_frame(
    text_df: pd.DataFrame,
    vocab: Vocabulary,
    config: TextEncodingConfig | None = None,
) -> pd.DataFrame:
    """
    Encode the user_text and item_text columns into token ID lists.

    Parameters
    ----------
    text_df : pd.DataFrame
        Output of build_two_tower_text_frame().
    vocab : Vocabulary
        Vocabulary built from training text.
    config : TextEncodingConfig | None
        Encoding length limits.

    Returns
    -------
    pd.DataFrame
        Copy of the input frame with:
        - user_input_ids
        - item_input_ids
    """
    config = config or TextEncodingConfig()

    required_columns = {"user_text", "item_text"}
    missing = required_columns - set(text_df.columns)
    if missing:
        raise ValueError(
            "Text frame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    encoded_df = text_df.copy()

    encoded_df["user_input_ids"] = encoded_df["user_text"].apply(
        lambda text: encode_text(
            text=text,
            vocab=vocab,
            max_tokens=config.max_user_tokens,
        )
    )

    encoded_df["item_input_ids"] = encoded_df["item_text"].apply(
        lambda text: encode_text(
            text=text,
            vocab=vocab,
            max_tokens=config.max_item_tokens,
        )
    )

    return encoded_df


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class TwoTowerTextDataset(Dataset):
    """
    PyTorch Dataset for two-tower retrieval training/evaluation.

    Each item contains:
    - metadata IDs
    - user token IDs
    - item token IDs
    - binary label
    """

    def __init__(self, encoded_df: pd.DataFrame) -> None:
        required_columns = {
            "impression_id",
            "user_id",
            "candidate_news_id",
            "clicked",
            "user_input_ids",
            "item_input_ids",
        }
        missing = required_columns - set(encoded_df.columns)
        if missing:
            raise ValueError(
                "Encoded DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

        self.df = encoded_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        return {
            "impression_id": str(row["impression_id"]),
            "user_id": str(row["user_id"]),
            "candidate_news_id": str(row["candidate_news_id"]),
            "clicked": float(row["clicked"]),
            "user_input_ids": list(row["user_input_ids"]),
            "item_input_ids": list(row["item_input_ids"]),
        }


# ------------------------------------------------------------------------------
# Collate / batching
# ------------------------------------------------------------------------------

def _pad_sequences(sequences: list[list[int]], pad_value: int = PAD_IDX) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length integer sequences.

    Parameters
    ----------
    sequences : list[list[int]]
        Token ID sequences.
    pad_value : int
        Padding token ID.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        padded_ids:  [batch_size, max_len]
        attention_mask: [batch_size, max_len]
            1 where token is real, 0 where token is padding
    """
    if not sequences:
        raise ValueError("Cannot pad an empty batch of sequences.")

    max_len = max(len(seq) for seq in sequences)

    padded = []
    masks = []

    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        mask = [1] * len(seq) + [0] * pad_len

        padded.append(padded_seq)
        masks.append(mask)

    padded_tensor = torch.tensor(padded, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.float32)

    return padded_tensor, mask_tensor


def collate_two_tower_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """
    Collate function for DataLoader.

    Returns
    -------
    dict
        {
            impression_id: list[str],
            user_id: list[str],
            candidate_news_id: list[str],
            labels: torch.Tensor [batch_size],
            user_input_ids: torch.Tensor [batch_size, max_user_len],
            user_attention_mask: torch.Tensor [batch_size, max_user_len],
            item_input_ids: torch.Tensor [batch_size, max_item_len],
            item_attention_mask: torch.Tensor [batch_size, max_item_len],
        }
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    user_sequences = [sample["user_input_ids"] for sample in batch]
    item_sequences = [sample["item_input_ids"] for sample in batch]

    user_input_ids, user_attention_mask = _pad_sequences(user_sequences, pad_value=PAD_IDX)
    item_input_ids, item_attention_mask = _pad_sequences(item_sequences, pad_value=PAD_IDX)

    labels = torch.tensor(
        [sample["clicked"] for sample in batch],
        dtype=torch.float32,
    )

    return {
        "impression_id": [sample["impression_id"] for sample in batch],
        "user_id": [sample["user_id"] for sample in batch],
        "candidate_news_id": [sample["candidate_news_id"] for sample in batch],
        "labels": labels,
        "user_input_ids": user_input_ids,
        "user_attention_mask": user_attention_mask,
        "item_input_ids": item_input_ids,
        "item_attention_mask": item_attention_mask,
    }


# ------------------------------------------------------------------------------
# High-level helpers
# ------------------------------------------------------------------------------

@dataclass
class TwoTowerDatasetBundle:
    """
    Container for train/dev two-tower dataset artifacts.
    """

    vocab: Vocabulary
    train_text_df: pd.DataFrame
    dev_text_df: pd.DataFrame
    train_encoded_df: pd.DataFrame
    dev_encoded_df: pd.DataFrame


def build_two_tower_dataset_bundle(
    train_config: RetrievalDatasetConfig,
    dev_config: RetrievalDatasetConfig,
    vocab_config: VocabularyConfig | None = None,
    encoding_config: TextEncodingConfig | None = None,
) -> TwoTowerDatasetBundle:
    """
    Build train/dev two-tower datasets plus a shared vocabulary.

    Workflow
    --------
    1. build retrieval examples for train/dev
    2. convert them into user_text/item_text frames
    3. build vocabulary using train text only
    4. encode train/dev using the same vocabulary
    """
    vocab_config = vocab_config or VocabularyConfig()
    encoding_config = encoding_config or TextEncodingConfig()

    train_retrieval_examples = build_retrieval_examples(train_config)
    dev_retrieval_examples = build_retrieval_examples(dev_config)

    train_text_df = build_two_tower_text_frame(train_retrieval_examples)
    dev_text_df = build_two_tower_text_frame(dev_retrieval_examples)

    training_corpus = pd.concat(
        [train_text_df["user_text"], train_text_df["item_text"]],
        ignore_index=True,
    )

    vocab = build_vocabulary_from_texts(training_corpus.tolist(), config=vocab_config)

    train_encoded_df = encode_two_tower_frame(
        train_text_df,
        vocab=vocab,
        config=encoding_config,
    )
    dev_encoded_df = encode_two_tower_frame(
        dev_text_df,
        vocab=vocab,
        config=encoding_config,
    )

    return TwoTowerDatasetBundle(
        vocab=vocab,
        train_text_df=train_text_df,
        dev_text_df=dev_text_df,
        train_encoded_df=train_encoded_df,
        dev_encoded_df=dev_encoded_df,
    )


def summarize_two_tower_frame(df: pd.DataFrame) -> dict:
    """
    Compact summary for two-tower frames.
    """
    if df.empty:
        return {
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": 0.0,
            "unique_impressions": 0,
            "unique_users": 0,
            "unique_candidates": 0,
        }

    positives = int(df["clicked"].sum())
    rows = int(len(df))
    negatives = rows - positives

    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / rows if rows else 0.0,
        "unique_impressions": int(df["impression_id"].nunique()),
        "unique_users": int(df["user_id"].nunique()),
        "unique_candidates": int(df["candidate_news_id"].nunique()),
    }


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    This builds a tiny train/dev two-tower dataset bundle and prints:
    - dataset summaries
    - vocab size
    - encoded sample rows
    - collated batch shapes
    """
    print("Running two-tower dataset smoke test...\n")

    bundle = build_two_tower_dataset_bundle(
        train_config=RetrievalDatasetConfig(
            split="train",
            max_impressions=120,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        dev_config=RetrievalDatasetConfig(
            split="dev",
            max_impressions=40,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        vocab_config=VocabularyConfig(
            min_freq=2,
            max_vocab_size=20000,
        ),
        encoding_config=TextEncodingConfig(
            max_user_tokens=48,
            max_item_tokens=48,
        ),
    )

    print("Train summary:")
    for key, value in summarize_two_tower_frame(bundle.train_text_df).items():
        print(f"  {key}: {value}")

    print("\nDev summary:")
    for key, value in summarize_two_tower_frame(bundle.dev_text_df).items():
        print(f"  {key}: {value}")

    print(f"\nVocab size: {len(bundle.vocab):,}")

    print("\nSample encoded rows:")
    preview_cols = [
        "impression_id",
        "user_id",
        "candidate_news_id",
        "clicked",
        "user_text",
        "item_text",
        "user_input_ids",
        "item_input_ids",
    ]
    print(bundle.train_encoded_df[preview_cols].head(3).to_string(index=False))

    dataset = TwoTowerTextDataset(bundle.train_encoded_df.head(4))
    batch = [dataset[i] for i in range(len(dataset))]
    collated = collate_two_tower_batch(batch)

    print("\nCollated batch tensor shapes:")
    print("  labels:", tuple(collated["labels"].shape))
    print("  user_input_ids:", tuple(collated["user_input_ids"].shape))
    print("  user_attention_mask:", tuple(collated["user_attention_mask"].shape))
    print("  item_input_ids:", tuple(collated["item_input_ids"].shape))
    print("  item_attention_mask:", tuple(collated["item_attention_mask"].shape))