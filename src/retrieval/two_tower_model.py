"""
src/retrieval/two_tower_model.py

Purpose
-------
Neural two-tower retrieval model for the discovery ranking system.

Why this file matters
---------------------
Our TF-IDF retriever is lexical and does not learn dense embeddings.

This file implements a simple but real two-tower model:
- user tower
- item tower
- embedding lookup
- masked mean pooling
- projection layer
- normalized dense embeddings
- dot-product scoring

This is a lean first two-tower baseline, designed to be:
- understandable
- trainable on a laptop
- strong enough to compare against TF-IDF offline

Training objective
------------------
We treat each row as a binary interaction:
- clicked = 1
- not clicked = 0

The model predicts a logit for each (user, item) pair.
We train with BCEWithLogitsLoss.

Architecture
------------
User tower input:
    user_input_ids, user_attention_mask

Item tower input:
    item_input_ids, item_attention_mask

Each tower:
    token embedding -> masked mean pooling -> MLP projection -> L2 normalization

Score:
    scaled dot product between user and item embeddings

Run local smoke test
--------------------
From the repo root:
    python3 -m src.retrieval.two_tower_model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_grouped_ranking
from src.retrieval.dataset import RetrievalDatasetConfig
from src.retrieval.two_tower_dataset import (
    TextEncodingConfig,
    TwoTowerTextDataset,
    Vocabulary,
    VocabularyConfig,
    build_two_tower_dataset_bundle,
    collate_two_tower_batch,
    summarize_two_tower_frame,
)


# ------------------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------------------

@dataclass
class TwoTowerModelConfig:
    """
    Configuration for the neural two-tower model.

    Attributes
    ----------
    embedding_dim : int
        Token embedding dimension.
    projection_dim : int
        Final embedding dimension produced by each tower.
    dropout : float
        Dropout applied before projection.
    logit_scale : float
        Scalar multiplier applied to dot-product similarity before BCE loss.
        Since normalized embeddings produce dot products in roughly [-1, 1],
        this helps logits have a more useful scale.
    """

    embedding_dim: int = 128
    projection_dim: int = 64
    dropout: float = 0.1
    logit_scale: float = 10.0


@dataclass
class TwoTowerTrainConfig:
    """
    Training configuration for the neural two-tower model.

    Attributes
    ----------
    batch_size : int
        Training batch size.
    learning_rate : float
        Optimizer learning rate.
    weight_decay : float
        Weight decay for AdamW.
    num_epochs : int
        Number of training epochs.
    device : str | None
        Optional device override. If None, choose automatically.
    use_class_balance : bool
        Whether to use positive-class reweighting in BCE loss.
    eval_ks : tuple[int, ...]
        Ranking cutoffs used during evaluation.
    """

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 5
    device: str | None = None
    use_class_balance: bool = True
    eval_ks: tuple[int, ...] = (5, 10, 20)


# ------------------------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------------------------

def get_default_device() -> str:
    """
    Pick the best available torch device.

    Preference:
    - CUDA
    - MPS (Apple Silicon)
    - CPU
    """
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# ------------------------------------------------------------------------------
# Tower building blocks
# ------------------------------------------------------------------------------

def masked_mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings using an attention mask.

    Parameters
    ----------
    token_embeddings : torch.Tensor
        Shape: [batch_size, seq_len, hidden_dim]
    attention_mask : torch.Tensor
        Shape: [batch_size, seq_len]
        1 for real tokens, 0 for padding

    Returns
    -------
    torch.Tensor
        Shape: [batch_size, hidden_dim]
    """
    if token_embeddings.ndim != 3:
        raise ValueError(
            f"token_embeddings must have shape [B, L, D]. Got: {tuple(token_embeddings.shape)}"
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [B, L]. Got: {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    masked_embeddings = token_embeddings * mask

    summed = masked_embeddings.sum(dim=1)  # [B, D]
    counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

    return summed / counts


class MeanPoolingTower(nn.Module):
    """
    A simple encoder tower:
    token embeddings -> masked mean pooling -> projection -> normalization
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        projection_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch into normalized dense embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape [B, L]
        attention_mask : torch.Tensor
            Shape [B, L]

        Returns
        -------
        torch.Tensor
            Shape [B, projection_dim]
        """
        token_embeddings = self.embedding(input_ids)  # [B, L, E]
        pooled = masked_mean_pool(token_embeddings, attention_mask)  # [B, E]
        projected = self.projection(pooled)  # [B, P]
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized


class TwoTowerRetrievalModel(nn.Module):
    """
    Lean neural two-tower retriever.

    User tower and item tower do not share parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        config: TwoTowerModelConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or TwoTowerModelConfig()

        self.user_tower = MeanPoolingTower(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            dropout=self.config.dropout,
            padding_idx=0,
        )

        self.item_tower = MeanPoolingTower(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            dropout=self.config.dropout,
            padding_idx=0,
        )

        self.logit_scale = self.config.logit_scale

    def encode_user(
        self,
        user_input_ids: torch.Tensor,
        user_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode user-side text into dense embeddings.
        """
        return self.user_tower(user_input_ids, user_attention_mask)

    def encode_item(
        self,
        item_input_ids: torch.Tensor,
        item_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode item-side text into dense embeddings.
        """
        return self.item_tower(item_input_ids, item_attention_mask)

    def forward(
        self,
        user_input_ids: torch.Tensor,
        user_attention_mask: torch.Tensor,
        item_input_ids: torch.Tensor,
        item_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise logits for a batch of user-item pairs.

        Returns
        -------
        torch.Tensor
            Shape [B], one logit per row.
        """
        user_emb = self.encode_user(user_input_ids, user_attention_mask)
        item_emb = self.encode_item(item_input_ids, item_attention_mask)

        # Dot product on normalized embeddings is cosine similarity.
        logits = (user_emb * item_emb).sum(dim=1) * self.logit_scale
        return logits


# ------------------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------------------

def _move_batch_to_device(batch: dict, device: str) -> dict:
    """
    Move tensor fields in a batch to the target device.
    """
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def compute_pos_weight(labels: pd.Series) -> float:
    """
    Compute positive-class weight for BCEWithLogitsLoss.

    Formula
    -------
    pos_weight = (# negatives) / (# positives)

    This helps counter class imbalance.
    """
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    if positives <= 0:
        return 1.0

    return negatives / positives


def train_one_epoch(
    model: TwoTowerRetrievalModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train the model for one epoch and return mean loss.
    """
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = model(
            user_input_ids=batch["user_input_ids"],
            user_attention_mask=batch["user_attention_mask"],
            item_input_ids=batch["item_input_ids"],
            item_attention_mask=batch["item_attention_mask"],
        )

        labels = batch["labels"]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_loss(
    model: TwoTowerRetrievalModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Evaluate mean loss without updating the model.
    """
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)

        logits = model(
            user_input_ids=batch["user_input_ids"],
            user_attention_mask=batch["user_attention_mask"],
            item_input_ids=batch["item_input_ids"],
            item_attention_mask=batch["item_attention_mask"],
        )

        labels = batch["labels"]
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def fit_two_tower_model(
    model: TwoTowerRetrievalModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    train_labels: pd.Series,
    config: TwoTowerTrainConfig | None = None,
) -> dict:
    """
    Train the two-tower model and return training history.
    """
    config = config or TwoTowerTrainConfig()
    device = config.device or get_default_device()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    pos_weight_value = compute_pos_weight(train_labels) if config.use_class_balance else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {
        "device": device,
        "pos_weight": float(pos_weight_value),
        "train_loss": [],
        "dev_loss": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        dev_loss = evaluate_loss(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(float(train_loss))
        history["dev_loss"].append(float(dev_loss))

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"dev_loss={dev_loss:.6f}"
        )

    return history


# ------------------------------------------------------------------------------
# Scoring / evaluation helpers
# ------------------------------------------------------------------------------

@torch.no_grad()
def score_two_tower_dataframe(
    model: TwoTowerRetrievalModel,
    encoded_df: pd.DataFrame,
    batch_size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Score an encoded two-tower DataFrame and return probabilities.

    Parameters
    ----------
    model : TwoTowerRetrievalModel
        Fitted model.
    encoded_df : pd.DataFrame
        Encoded frame containing token ID lists.
    batch_size : int
        Batch size for scoring.
    device : str | None
        Device override.

    Returns
    -------
    pd.DataFrame
        Copy of input df with:
        - logit
        - score   (sigmoid(logit))
    """
    device = device or get_default_device()

    dataset = TwoTowerTextDataset(encoded_df)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_two_tower_batch,
    )

    model.eval()
    model.to(device)

    logits_list: list[torch.Tensor] = []

    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)

        logits = model(
            user_input_ids=batch["user_input_ids"],
            user_attention_mask=batch["user_attention_mask"],
            item_input_ids=batch["item_input_ids"],
            item_attention_mask=batch["item_attention_mask"],
        )
        logits_list.append(logits.detach().cpu())

    all_logits = torch.cat(logits_list, dim=0)
    scores = torch.sigmoid(all_logits).numpy()

    scored_df = encoded_df.copy()
    scored_df["logit"] = all_logits.numpy()
    scored_df["score"] = scores.astype(float)

    return scored_df


def evaluate_two_tower_ranking(
    scored_df: pd.DataFrame,
    ks: Iterable[int] = (5, 10, 20),
) -> dict:
    """
    Evaluate grouped ranking metrics from a scored DataFrame.
    """
    required_columns = {"impression_id", "clicked", "score"}
    missing = required_columns - set(scored_df.columns)
    if missing:
        raise ValueError(
            "Scored DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    eval_df = scored_df[["impression_id", "clicked", "score"]].copy()
    eval_df["clicked"] = eval_df["clicked"].astype(int)

    return evaluate_grouped_ranking(eval_df, ks=ks)


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    We:
    1. build small train/dev encoded datasets
    2. create DataLoaders
    3. train a lean two-tower model
    4. score the dev set
    5. compute ranking metrics

    This checks the full neural retrieval pipeline end to end.
    """
    print("Running two-tower model smoke test...\n")

    bundle = build_two_tower_dataset_bundle(
        train_config=RetrievalDatasetConfig(
            split="train",
            max_impressions=180,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        dev_config=RetrievalDatasetConfig(
            split="dev",
            max_impressions=60,
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

    print(f"\nVocabulary size: {len(bundle.vocab):,}")

    train_dataset = TwoTowerTextDataset(bundle.train_encoded_df)
    dev_dataset = TwoTowerTextDataset(bundle.dev_encoded_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_two_tower_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_two_tower_batch,
    )

    model = TwoTowerRetrievalModel(
        vocab_size=len(bundle.vocab),
        config=TwoTowerModelConfig(
            embedding_dim=128,
            projection_dim=64,
            dropout=0.1,
            logit_scale=10.0,
        ),
    )

    print(f"Training on device: {get_default_device()}")
    history = fit_two_tower_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_labels=bundle.train_encoded_df["clicked"],
        config=TwoTowerTrainConfig(
            batch_size=128,
            learning_rate=1e-3,
            weight_decay=1e-4,
            num_epochs=5,
            device=None,
            use_class_balance=True,
            eval_ks=(5, 10, 20),
        ),
    )

    print("\nTraining history:")
    print(history)

    print("\nScoring dev set...")
    scored_dev = score_two_tower_dataframe(
        model=model,
        encoded_df=bundle.dev_encoded_df,
        batch_size=256,
        device=None,
    )

    metrics = evaluate_two_tower_ranking(
        scored_df=scored_dev,
        ks=(5, 10, 20),
    )

    print("\nTwo-tower dev ranking metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\nTop scored dev rows:")
    preview_cols = [
        "impression_id",
        "user_id",
        "candidate_news_id",
        "clicked",
        "logit",
        "score",
    ]
    print(
        scored_dev[preview_cols]
        .sort_values(["impression_id", "score"], ascending=[True, False])
        .head(12)
        .to_string(index=False)
    )