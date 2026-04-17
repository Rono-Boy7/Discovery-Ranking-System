"""
src/evaluation/metrics.py

Purpose
-------
Core ranking metrics for evaluating discovery / recommendation systems.

Why this file matters
---------------------
In a ranking system, raw accuracy is usually not the right metric.

We care more about questions like:
- Did the clicked item appear near the top?
- How many relevant items appeared in the top K?
- How good was the ranking order inside each impression?

Since MIND is organized around impressions, we evaluate predictions grouped by
impression_id.

Metrics implemented here
------------------------
1. MRR@K
   - Mean Reciprocal Rank at K
   - Rewards placing the first clicked item as high as possible

2. Recall@K
   - Of the clicked items in an impression, how many were retrieved in top K?

3. DCG@K / NDCG@K
   - Discounted Cumulative Gain / Normalized DCG
   - Rewards relevant items ranked near the top

4. Grouped evaluation helpers
   - Evaluate a full DataFrame grouped by impression_id

Expected input format
---------------------
A pandas DataFrame with at least:
- impression_id
- clicked          (ground-truth label, 0 or 1)
- score            (model score; higher means more relevant)

Run local smoke test
--------------------
From the repo root:
    python3 -m src.evaluation.metrics
"""

from __future__ import annotations

from typing import Iterable, Sequence

import math
import pandas as pd


# ------------------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------------------

def _validate_binary_labels(labels: Sequence[int]) -> None:
    """
    Ensure labels are binary: only 0 or 1.
    """
    unique_values = set(labels)
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Labels must be binary (0/1). Found values: {sorted(unique_values)}"
        )


def _top_k_pairs(labels: Sequence[int], scores: Sequence[float], k: int | None = None) -> list[tuple[int, float]]:
    """
    Sort (label, score) pairs by score descending and optionally truncate to top K.

    Parameters
    ----------
    labels : Sequence[int]
        Ground-truth binary relevance labels.
    scores : Sequence[float]
        Model scores; higher means more relevant.
    k : int | None
        Optional cutoff.

    Returns
    -------
    list[tuple[int, float]]
        Sorted (label, score) pairs in descending score order.
    """
    if len(labels) != len(scores):
        raise ValueError(
            f"Labels and scores must have the same length. "
            f"Got {len(labels)} labels and {len(scores)} scores."
        )

    _validate_binary_labels(labels)

    pairs = list(zip(labels, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    if k is not None:
        if k <= 0:
            raise ValueError(f"k must be positive when provided. Got: {k}")
        pairs = pairs[:k]

    return pairs


# ------------------------------------------------------------------------------
# Per-impression metrics
# ------------------------------------------------------------------------------

def reciprocal_rank_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """
    Compute Reciprocal Rank at K for a single impression.

    Definition
    ----------
    RR@K = 1 / rank_of_first_relevant_item, if a relevant item appears in top K
           0 otherwise

    Example
    -------
    Top-ranked labels = [0, 0, 1, 0]
    => RR = 1/3
    """
    ranked = _top_k_pairs(labels, scores, k=k)

    for rank_idx, (label, _) in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / rank_idx

    return 0.0


def recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """
    Compute Recall at K for a single impression.

    Definition
    ----------
    Recall@K = (# relevant items in top K) / (# total relevant items)

    Notes
    -----
    If an impression has no positive labels, we return 0.0.
    """
    total_relevant = sum(labels)
    if total_relevant == 0:
        return 0.0

    ranked = _top_k_pairs(labels, scores, k=k)
    retrieved_relevant = sum(label for label, _ in ranked)

    return retrieved_relevant / total_relevant


def dcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K for a single impression.

    Formula
    -------
    DCG@K = sum((2^rel_i - 1) / log2(i + 1)) for i = 1..K

    Since our labels are binary:
    - rel_i is either 0 or 1
    - (2^rel_i - 1) becomes 1 for relevant, 0 otherwise
    """
    ranked = _top_k_pairs(labels, scores, k=k)

    dcg = 0.0
    for rank_idx, (label, _) in enumerate(ranked, start=1):
        gain = (2 ** label) - 1
        discount = math.log2(rank_idx + 1)
        dcg += gain / discount

    return dcg


def idcg_at_k(labels: Sequence[int], k: int) -> float:
    """
    Compute the ideal DCG at K for a single impression.

    This is the DCG obtained when all relevant items are ranked perfectly first.
    """
    _validate_binary_labels(labels)

    ideal_labels = sorted(labels, reverse=True)
    ideal_scores = list(range(len(ideal_labels), 0, -1))  # dummy descending scores
    return dcg_at_k(ideal_labels, ideal_scores, k=k)


def ndcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """
    Compute Normalized DCG at K for a single impression.

    Definition
    ----------
    NDCG@K = DCG@K / IDCG@K

    Notes
    -----
    If there are no relevant items, return 0.0.
    """
    ideal_dcg = idcg_at_k(labels, k=k)
    if ideal_dcg == 0.0:
        return 0.0

    return dcg_at_k(labels, scores, k=k) / ideal_dcg


# ------------------------------------------------------------------------------
# DataFrame/grouped evaluation
# ------------------------------------------------------------------------------

def _validate_eval_frame(df: pd.DataFrame) -> None:
    """
    Validate that the evaluation DataFrame has the required columns.
    """
    required_columns = {"impression_id", "clicked", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Evaluation DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )


def evaluate_grouped_ranking(
    df: pd.DataFrame,
    ks: Iterable[int] = (5, 10, 20),
) -> dict:
    """
    Evaluate ranking metrics grouped by impression_id.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - impression_id
        - clicked
        - score
    ks : Iterable[int]
        Cutoff values to evaluate, e.g. (5, 10, 20)

    Returns
    -------
    dict
        Aggregated mean metrics across impressions.

    Output shape example
    --------------------
    {
      "num_impressions": 100,
      "num_rows": 2000,
      "mrr@5": 0.31,
      "recall@5": 0.28,
      "ndcg@5": 0.35,
      ...
    }
    """
    _validate_eval_frame(df)

    working_df = df.copy()
    working_df["clicked"] = working_df["clicked"].astype(int)

    grouped = working_df.groupby("impression_id", sort=False)

    results: dict[str, float | int] = {
        "num_impressions": int(grouped.ngroups),
        "num_rows": int(len(working_df)),
    }

    # Collect per-group metric values, then average them.
    for k in ks:
        rr_values: list[float] = []
        recall_values: list[float] = []
        ndcg_values: list[float] = []

        for _, group in grouped:
            labels = group["clicked"].tolist()
            scores = group["score"].tolist()

            rr_values.append(reciprocal_rank_at_k(labels, scores, k))
            recall_values.append(recall_at_k(labels, scores, k))
            ndcg_values.append(ndcg_at_k(labels, scores, k))

        results[f"mrr@{k}"] = sum(rr_values) / len(rr_values) if rr_values else 0.0
        results[f"recall@{k}"] = sum(recall_values) / len(recall_values) if recall_values else 0.0
        results[f"ndcg@{k}"] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

    return results


# ------------------------------------------------------------------------------
# Baseline scoring helpers
# ------------------------------------------------------------------------------

def add_random_scores(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Return a copy of the input DataFrame with a random baseline score column.

    Useful for smoke testing ranking pipelines before training a real model.
    """
    _validate_eval_frame_columns_for_scoring_input(df)

    scored_df = df.copy()
    scored_df["score"] = pd.Series(
        pd.array(
            pd.np.random.RandomState(seed).rand(len(scored_df)),  # type: ignore[attr-defined]
            dtype="float64",
        )
    )
    return scored_df


def add_label_prior_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a deliberately unrealistic 'cheating' score for debugging only.

    This uses the label itself as the score:
    - positives get score 1.0
    - negatives get score 0.0

    Why include this?
    -----------------
    It gives us an approximate upper-bound sanity check. If our metric functions
    are implemented correctly, this should produce very strong ranking results.

    Do NOT use this as a real model.
    """
    _validate_eval_frame_columns_for_scoring_input(df)

    scored_df = df.copy()
    scored_df["score"] = scored_df["clicked"].astype(float)
    return scored_df


def _validate_eval_frame_columns_for_scoring_input(df: pd.DataFrame) -> None:
    """
    Validate the minimal columns needed before adding a synthetic score.
    """
    required_columns = {"impression_id", "clicked"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Input DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test with tiny handcrafted examples.
    """
    print("Running ranking metric smoke tests...\n")

    # Example 1: first relevant item appears at rank 2
    labels_1 = [0, 1, 0, 0]
    scores_1 = [0.9, 0.8, 0.2, 0.1]

    print("Single-impression example")
    print("-------------------------")
    print(f"Labels: {labels_1}")
    print(f"Scores: {scores_1}")
    print(f"RR@4: {reciprocal_rank_at_k(labels_1, scores_1, 4):.4f}")
    print(f"Recall@2: {recall_at_k(labels_1, scores_1, 2):.4f}")
    print(f"NDCG@4: {ndcg_at_k(labels_1, scores_1, 4):.4f}")

    # Example 2: grouped DataFrame evaluation
    eval_df = pd.DataFrame(
        {
            "impression_id": ["A", "A", "A", "B", "B", "B"],
            "clicked":       [1,   0,   0,   0,   1,   0],
            "score":         [0.8, 0.2, 0.1, 0.7, 0.6, 0.1],
        }
    )

    print("\nGrouped-evaluation example")
    print("--------------------------")
    print(eval_df.to_string(index=False))

    metrics = evaluate_grouped_ranking(eval_df, ks=(1, 3))
    print("\nGrouped ranking metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")