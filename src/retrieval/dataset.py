"""
src/retrieval/dataset.py

Purpose
-------
Build retrieval-training examples for the discovery ranking system.

Why this file matters
---------------------
Our raw/interim MIND dataset is still organized around:
- impression-level records
- candidate items shown in each impression
- clicked vs non-clicked labels

For retrieval model training, we want a cleaner supervised training table where
each row contains:
- a user representation (built from recent reading history)
- a candidate article representation
- a binary label (clicked or not clicked)

This module creates those examples.

What this module does
---------------------
1. Loads normalized interim artifacts from src.data.loader
2. Samples impressions (optional, useful for faster experiments)
3. Keeps all positive candidates
4. Downsamples negatives per impression
5. Builds user profile text from recent history article titles
6. Attaches candidate article text and metadata

This gives us a practical first training dataset for a retrieval model.

Important note
--------------
This is still a *dataset-building* step, not the neural model yet.

Run local smoke test
--------------------
From the repo root:
    python3 -m src.retrieval.dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.loader import load_behaviors, load_candidates, load_news


# ------------------------------------------------------------------------------
# Configuration container
# ------------------------------------------------------------------------------

@dataclass
class RetrievalDatasetConfig:
    """
    Configuration for building retrieval examples.

    Attributes
    ----------
    split : str
        Dataset split to use: "train" or "dev".
    max_impressions : int | None
        If provided, randomly sample at most this many impressions.
        Useful for fast experiments and smoke tests.
    negatives_per_positive : int
        Number of negative candidates to keep per positive candidate inside each
        impression.
    max_history_items : int
        Number of most recent history articles to use when constructing the
        user profile text.
    random_state : int
        Seed for reproducible sampling.
    """

    split: str = "train"
    max_impressions: int | None = None
    negatives_per_positive: int = 4
    max_history_items: int = 10
    random_state: int = 42


# ------------------------------------------------------------------------------
# News lookup helpers
# ------------------------------------------------------------------------------

def build_news_lookup(news_df: pd.DataFrame) -> dict[str, dict]:
    """
    Build a fast lookup dictionary from news_id -> metadata dict.

    Parameters
    ----------
    news_df : pd.DataFrame
        News table loaded from the interim artifacts.

    Returns
    -------
    dict[str, dict]
        Mapping from article ID to metadata.

    Why this is useful
    ------------------
    We repeatedly need to map:
    - history article IDs -> titles
    - candidate article IDs -> title/abstract/category

    A dictionary lookup is much faster than repeated DataFrame filtering.
    """
    needed_columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "full_text",
    ]

    available_columns = [col for col in needed_columns if col in news_df.columns]
    lookup_df = news_df[available_columns].copy()

    records = lookup_df.to_dict(orient="records")
    return {record["news_id"]: record for record in records}


def _safe_text(value: object) -> str:
    """
    Convert possibly-missing values into clean strings.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_user_profile_text(
    history_ids: Iterable[str],
    news_lookup: dict[str, dict],
    max_history_items: int = 10,
) -> tuple[list[str], list[str], str]:
    """
    Build a simple text-based user profile from recent history article titles.

    Parameters
    ----------
    history_ids : Iterable[str]
        Article IDs from the user's click/read history.
    news_lookup : dict[str, dict]
        Lookup built from build_news_lookup().
    max_history_items : int
        Use only the most recent N history items.

    Returns
    -------
    tuple[list[str], list[str], str]
        (
            selected_history_ids,
            selected_history_titles,
            user_profile_text
        )

    Notes
    -----
    For our first version, we use article titles rather than full abstracts/body
    because:
    - titles are compact
    - they still carry strong signal
    - they are easier to inspect while learning

    Later we can switch to richer encoders and more sophisticated sequence models.
    """
    history_ids = list(history_ids)

    if max_history_items <= 0:
        raise ValueError(
            f"max_history_items must be positive. Got: {max_history_items}"
        )

    # Use the most recent items from the end of the history list.
    selected_ids = history_ids[-max_history_items:]

    selected_titles: list[str] = []
    for news_id in selected_ids:
        news_meta = news_lookup.get(news_id)
        if news_meta is None:
            continue

        title = _safe_text(news_meta.get("title", ""))
        if title:
            selected_titles.append(title)

    # A simple textual user profile built by joining titles in order.
    user_profile_text = " [HIST] ".join(selected_titles).strip()

    return selected_ids, selected_titles, user_profile_text


# ------------------------------------------------------------------------------
# Impression sampling helpers
# ------------------------------------------------------------------------------

def sample_behaviors(
    behaviors_df: pd.DataFrame,
    max_impressions: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Optionally sample a subset of impression rows.

    Parameters
    ----------
    behaviors_df : pd.DataFrame
        One row per impression.
    max_impressions : int | None
        If None, keep all impressions.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Possibly sampled behaviors DataFrame.
    """
    if max_impressions is None:
        return behaviors_df.copy()

    if max_impressions <= 0:
        raise ValueError(
            f"max_impressions must be positive when provided. Got: {max_impressions}"
        )

    if max_impressions >= len(behaviors_df):
        return behaviors_df.copy()

    return behaviors_df.sample(
        n=max_impressions,
        random_state=random_state,
    ).copy()


def sample_candidates_per_impression(
    candidates_df: pd.DataFrame,
    negatives_per_positive: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Keep all positive candidates and downsample negatives within each impression.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        One row per candidate item shown in an impression.
    negatives_per_positive : int
        Number of negatives to keep per positive.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A smaller training-ready candidate table.

    Example
    -------
    If an impression has:
    - 1 positive
    - 20 negatives

    and negatives_per_positive = 4,
    then we keep:
    - 1 positive
    - 4 sampled negatives
    """
    if negatives_per_positive <= 0:
        raise ValueError(
            f"negatives_per_positive must be positive. Got: {negatives_per_positive}"
        )

    rng = np.random.default_rng(random_state)
    sampled_groups: list[pd.DataFrame] = []

    grouped = candidates_df.groupby("impression_id", sort=False)

    for _, group in grouped:
        positives = group[group["clicked"] == 1]
        negatives = group[group["clicked"] == 0]

        # Keep all positives.
        positive_count = len(positives)

        # If for some reason there are no positives, we skip the impression
        # because ranking/retrieval learning signal would be weak or ambiguous.
        if positive_count == 0:
            continue

        max_negatives_to_keep = positive_count * negatives_per_positive

        if len(negatives) > max_negatives_to_keep:
            sample_seed = int(rng.integers(0, 1_000_000_000))
            negatives = negatives.sample(
                n=max_negatives_to_keep,
                random_state=sample_seed,
            )

        sampled_group = pd.concat([positives, negatives], ignore_index=True)
        sampled_groups.append(sampled_group)

    if not sampled_groups:
        return pd.DataFrame(columns=candidates_df.columns)

    sampled_df = pd.concat(sampled_groups, ignore_index=True)

    # Shuffle rows so the model doesn't see all positives first, then negatives.
    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    return sampled_df


# ------------------------------------------------------------------------------
# Main dataset builder
# ------------------------------------------------------------------------------

def build_retrieval_examples(config: RetrievalDatasetConfig) -> pd.DataFrame:
    """
    Build retrieval training examples from the interim MIND artifacts.

    Parameters
    ----------
    config : RetrievalDatasetConfig
        Configuration controlling split, sampling, and profile construction.

    Returns
    -------
    pd.DataFrame
        Training examples with:
        - user context text
        - candidate article text
        - clicked label

    Output columns
    --------------
    split
    impression_id
    user_id
    timestamp
    clicked
    history_length
    impression_size
    selected_history_ids
    selected_history_titles
    user_profile_text
    candidate_news_id
    candidate_category
    candidate_subcategory
    candidate_title
    candidate_abstract
    candidate_full_text
    """
    # --------------------------------------------------------------------------
    # Load source artifacts
    # --------------------------------------------------------------------------
    news_df = load_news(config.split)
    behaviors_df = load_behaviors(config.split)
    candidates_df = load_candidates(config.split)

    # Keep only the columns we need to control memory.
    behaviors_df = behaviors_df[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "history_list",
            "history_length",
            "impression_size",
        ]
    ].copy()

    candidates_df = candidates_df[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "candidate_news_id",
            "clicked",
        ]
    ].copy()

    # --------------------------------------------------------------------------
    # Optional impression sampling for faster experiments
    # --------------------------------------------------------------------------
    behaviors_df = sample_behaviors(
        behaviors_df=behaviors_df,
        max_impressions=config.max_impressions,
        random_state=config.random_state,
    )

    sampled_impression_ids = set(behaviors_df["impression_id"].astype(str))
    candidates_df = candidates_df[
        candidates_df["impression_id"].astype(str).isin(sampled_impression_ids)
    ].copy()

    # --------------------------------------------------------------------------
    # Downsample negatives per impression
    # --------------------------------------------------------------------------
    candidates_df = sample_candidates_per_impression(
        candidates_df=candidates_df,
        negatives_per_positive=config.negatives_per_positive,
        random_state=config.random_state,
    )

    # --------------------------------------------------------------------------
    # Build fast lookups
    # --------------------------------------------------------------------------
    news_lookup = build_news_lookup(news_df)

    # One behavior row per impression_id, so this is safe.
    behavior_lookup = {
        str(row["impression_id"]): row
        for row in behaviors_df.to_dict(orient="records")
    }

    # --------------------------------------------------------------------------
    # Build final example rows
    # --------------------------------------------------------------------------
    output_rows: list[dict] = []

    for row in candidates_df.itertuples(index=False):
        impression_id = str(row.impression_id)
        candidate_news_id = str(row.candidate_news_id)

        behavior = behavior_lookup.get(impression_id)
        if behavior is None:
            # Should not happen after filtering, but stay defensive.
            continue

        candidate_meta = news_lookup.get(candidate_news_id, {})

        selected_history_ids, selected_history_titles, user_profile_text = build_user_profile_text(
            history_ids=behavior["history_list"],
            news_lookup=news_lookup,
            max_history_items=config.max_history_items,
        )

        output_rows.append(
            {
                "split": config.split,
                "impression_id": impression_id,
                "user_id": str(row.user_id),
                "timestamp": row.timestamp,
                "clicked": int(row.clicked),
                "history_length": int(behavior["history_length"]),
                "impression_size": int(behavior["impression_size"]),
                "selected_history_ids": selected_history_ids,
                "selected_history_titles": selected_history_titles,
                "user_profile_text": user_profile_text,
                "candidate_news_id": candidate_news_id,
                "candidate_category": _safe_text(candidate_meta.get("category", "")),
                "candidate_subcategory": _safe_text(candidate_meta.get("subcategory", "")),
                "candidate_title": _safe_text(candidate_meta.get("title", "")),
                "candidate_abstract": _safe_text(candidate_meta.get("abstract", "")),
                "candidate_full_text": _safe_text(candidate_meta.get("full_text", "")),
            }
        )

    examples_df = pd.DataFrame(output_rows)

    if not examples_df.empty:
        examples_df["clicked"] = examples_df["clicked"].astype("int8")

    return examples_df


# ------------------------------------------------------------------------------
# Summary helper
# ------------------------------------------------------------------------------

def summarize_retrieval_examples(examples_df: pd.DataFrame) -> dict:
    """
    Return compact summary stats for a built retrieval dataset.
    """
    if examples_df.empty:
        return {
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": 0.0,
            "unique_impressions": 0,
            "unique_users": 0,
            "unique_candidates": 0,
        }

    positives = int(examples_df["clicked"].sum())
    rows = int(len(examples_df))
    negatives = rows - positives

    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / rows if rows else 0.0,
        "unique_impressions": int(examples_df["impression_id"].nunique()),
        "unique_users": int(examples_df["user_id"].nunique()),
        "unique_candidates": int(examples_df["candidate_news_id"].nunique()),
    }


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small smoke test.

    We intentionally use a tiny subset of the dev split so it runs quickly and
    lets us inspect the resulting training examples.
    """
    print("Building retrieval example smoke test...\n")

    config = RetrievalDatasetConfig(
        split="dev",
        max_impressions=5,
        negatives_per_positive=4,
        max_history_items=5,
        random_state=42,
    )

    examples_df = build_retrieval_examples(config)
    summary = summarize_retrieval_examples(examples_df)

    print("Retrieval dataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSample retrieval examples:")
    preview_columns = [
        "impression_id",
        "user_id",
        "clicked",
        "user_profile_text",
        "candidate_news_id",
        "candidate_title",
    ]

    available_columns = [col for col in preview_columns if col in examples_df.columns]
    print(examples_df[available_columns].head(10).to_string(index=False))