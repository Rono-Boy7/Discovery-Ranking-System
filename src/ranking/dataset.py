"""
src/ranking/dataset.py

Purpose
-------
Build second-stage ranking features for the discovery ranking system.

Why this file matters
---------------------
So far we have built:
1. raw parsing
2. interim normalized artifacts
3. retrieval examples
4. a first-stage TF-IDF retrieval baseline

That is good for candidate retrieval, but real discovery systems usually use a
second-stage reranker that looks at richer features and reorders candidates more
carefully.

This module builds that reranking dataset.

What this module does
---------------------
Given retrieval examples, it engineers ranking features such as:
- retrieval baseline score
- lexical overlap between user profile and candidate text
- title overlap
- category match features
- subcategory match features
- history length / impression size
- text length features

These features will later feed a simple reranker model such as:
- logistic regression
- gradient boosted trees
- other learning-to-rank style models

Design notes
------------
This is intentionally feature-engineering based.
That is a strong learning step because it teaches:
- what signals matter in ranking
- how two-stage systems are built
- how a reranker can improve on a retrieval baseline

Run local smoke test
--------------------
From the repo root:
    python3 -m src.ranking.dataset
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd

from src.data.loader import load_news
from src.retrieval.dataset import (
    RetrievalDatasetConfig,
    build_news_lookup,
    build_retrieval_examples,
)
from src.retrieval.model import (
    TfidfRetrievalBaseline,
    TfidfRetrievalConfig,
)


# ------------------------------------------------------------------------------
# Configuration container
# ------------------------------------------------------------------------------

@dataclass
class RankingDatasetConfig:
    """
    Configuration for building reranking features.

    Attributes
    ----------
    split : str
        Dataset split to use: "train" or "dev".
    max_impressions : int | None
        Optional max number of impressions to sample.
    negatives_per_positive : int
        Number of negatives to keep per positive.
    max_history_items : int
        Number of recent history items to use in the user profile.
    random_state : int
        Random seed for reproducibility.
    include_retrieval_score : bool
        Whether to include the retrieval model score as a ranking feature.
    """

    split: str = "train"
    max_impressions: int | None = None
    negatives_per_positive: int = 4
    max_history_items: int = 10
    random_state: int = 42
    include_retrieval_score: bool = True


# ------------------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize_text(text: object) -> list[str]:
    """
    Basic tokenizer for feature engineering.

    We keep this simple and transparent:
    - lowercase
    - keep alphanumeric tokens only

    This is enough for overlap-based baseline features.
    """
    if pd.isna(text):
        return []

    text = str(text).lower().strip()
    if not text:
        return []

    return _TOKEN_PATTERN.findall(text)


def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """
    Compute Jaccard similarity between two token sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns
    -------
    float
        Value between 0 and 1.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b
    if not union:
        return 0.0

    intersection = set_a & set_b
    return len(intersection) / len(union)


def overlap_count(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> int:
    """
    Count the number of distinct overlapping tokens between two token sets.
    """
    return len(set(tokens_a) & set(tokens_b))


def overlap_ratio(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """
    Overlap ratio defined as:
        |A ∩ B| / max(1, |A|)

    This is directional and can be useful when one side is treated as the
    "reference" set.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a:
        return 0.0

    return len(set_a & set_b) / len(set_a)


# ------------------------------------------------------------------------------
# History/category helpers
# ------------------------------------------------------------------------------

def get_history_metadata(
    selected_history_ids: list[str],
    news_lookup: dict[str, dict],
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract category, subcategory, and title history from selected history IDs.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        (
            history_categories,
            history_subcategories,
            history_titles
        )
    """
    history_categories: list[str] = []
    history_subcategories: list[str] = []
    history_titles: list[str] = []

    for news_id in selected_history_ids:
        meta = news_lookup.get(news_id)
        if meta is None:
            continue

        category = str(meta.get("category", "")).strip()
        subcategory = str(meta.get("subcategory", "")).strip()
        title = str(meta.get("title", "")).strip()

        if category:
            history_categories.append(category)
        if subcategory:
            history_subcategories.append(subcategory)
        if title:
            history_titles.append(title)

    return history_categories, history_subcategories, history_titles


def fraction_equal(values: list[str], target: str) -> float:
    """
    Fraction of values equal to the target string.
    """
    if not values:
        return 0.0
    return sum(1 for value in values if value == target) / len(values)


# ------------------------------------------------------------------------------
# Core feature engineering
# ------------------------------------------------------------------------------

def build_ranking_features_from_retrieval_examples(
    retrieval_examples_df: pd.DataFrame,
    news_lookup: dict[str, dict],
    retrieval_model: TfidfRetrievalBaseline | None = None,
    include_retrieval_score: bool = True,
) -> pd.DataFrame:
    """
    Convert retrieval examples into reranking features.

    Parameters
    ----------
    retrieval_examples_df : pd.DataFrame
        Output of build_retrieval_examples().
    news_lookup : dict[str, dict]
        Lookup from news_id -> metadata.
    retrieval_model : TfidfRetrievalBaseline | None
        Optional fitted retrieval baseline whose score can be used as a feature.
    include_retrieval_score : bool
        If True and a retrieval_model is provided, add its score as a feature.

    Returns
    -------
    pd.DataFrame
        One row per candidate with engineered ranking features.
    """
    required_columns = {
        "split",
        "impression_id",
        "user_id",
        "clicked",
        "history_length",
        "impression_size",
        "selected_history_ids",
        "selected_history_titles",
        "user_profile_text",
        "candidate_news_id",
        "candidate_category",
        "candidate_subcategory",
        "candidate_title",
        "candidate_abstract",
        "candidate_full_text",
    }
    missing = required_columns - set(retrieval_examples_df.columns)
    if missing:
        raise ValueError(
            "Retrieval examples are missing required columns: "
            + ", ".join(sorted(missing))
        )

    working_df = retrieval_examples_df.copy()

    # --------------------------------------------------------------------------
    # Optionally add retrieval baseline score
    # --------------------------------------------------------------------------
    retrieval_scores = None
    if include_retrieval_score and retrieval_model is not None:
        scored_df = retrieval_model.score_examples(working_df)
        retrieval_scores = scored_df["score"].tolist()

    # --------------------------------------------------------------------------
    # Build row-level ranking features
    # --------------------------------------------------------------------------
    output_rows: list[dict] = []

    for idx, row in enumerate(working_df.itertuples(index=False)):
        user_profile_text = str(row.user_profile_text).strip()
        candidate_title = str(row.candidate_title).strip()
        candidate_abstract = str(row.candidate_abstract).strip()
        candidate_full_text = str(row.candidate_full_text).strip()
        candidate_category = str(row.candidate_category).strip()
        candidate_subcategory = str(row.candidate_subcategory).strip()

        selected_history_ids = list(row.selected_history_ids)
        selected_history_titles = list(row.selected_history_titles)

        history_categories, history_subcategories, history_title_texts = get_history_metadata(
            selected_history_ids=selected_history_ids,
            news_lookup=news_lookup,
        )

        # ----------------------------------------------------------------------
        # Token-based features
        # ----------------------------------------------------------------------
        user_tokens = tokenize_text(user_profile_text)
        candidate_title_tokens = tokenize_text(candidate_title)
        candidate_abstract_tokens = tokenize_text(candidate_abstract)
        candidate_full_tokens = tokenize_text(candidate_full_text)

        # Compare user profile to candidate title/full text
        user_candidate_title_jaccard = jaccard_similarity(user_tokens, candidate_title_tokens)
        user_candidate_full_jaccard = jaccard_similarity(user_tokens, candidate_full_tokens)
        user_candidate_title_overlap_count = overlap_count(user_tokens, candidate_title_tokens)
        user_candidate_full_overlap_count = overlap_count(user_tokens, candidate_full_tokens)
        user_candidate_title_overlap_ratio = overlap_ratio(candidate_title_tokens, user_tokens)
        user_candidate_full_overlap_ratio = overlap_ratio(candidate_full_tokens, user_tokens)

        # Compare candidate title to each history title, then aggregate
        history_title_jaccards: list[float] = []
        history_title_overlap_counts: list[int] = []

        for history_title in history_title_texts:
            history_title_tokens = tokenize_text(history_title)
            history_title_jaccards.append(
                jaccard_similarity(history_title_tokens, candidate_title_tokens)
            )
            history_title_overlap_counts.append(
                overlap_count(history_title_tokens, candidate_title_tokens)
            )

        avg_history_title_jaccard = (
            sum(history_title_jaccards) / len(history_title_jaccards)
            if history_title_jaccards
            else 0.0
        )
        max_history_title_jaccard = max(history_title_jaccards) if history_title_jaccards else 0.0
        avg_history_title_overlap_count = (
            sum(history_title_overlap_counts) / len(history_title_overlap_counts)
            if history_title_overlap_counts
            else 0.0
        )
        max_history_title_overlap_count = (
            max(history_title_overlap_counts) if history_title_overlap_counts else 0
        )

        # ----------------------------------------------------------------------
        # Category / subcategory affinity features
        # ----------------------------------------------------------------------
        candidate_category_match_fraction = fraction_equal(
            history_categories,
            candidate_category,
        )
        candidate_subcategory_match_fraction = fraction_equal(
            history_subcategories,
            candidate_subcategory,
        )

        candidate_category_seen_in_history = 1 if candidate_category in history_categories else 0
        candidate_subcategory_seen_in_history = 1 if candidate_subcategory in history_subcategories else 0

        # ----------------------------------------------------------------------
        # Length / structure features
        # ----------------------------------------------------------------------
        user_profile_token_count = len(user_tokens)
        candidate_title_token_count = len(candidate_title_tokens)
        candidate_abstract_token_count = len(candidate_abstract_tokens)
        candidate_full_token_count = len(candidate_full_tokens)

        row_dict = {
            # IDs / labels
            "split": row.split,
            "impression_id": str(row.impression_id),
            "user_id": str(row.user_id),
            "candidate_news_id": str(row.candidate_news_id),
            "clicked": int(row.clicked),

            # Impression context features
            "history_length": int(row.history_length),
            "impression_size": int(row.impression_size),
            "selected_history_count": int(len(selected_history_ids)),

            # Text-length features
            "user_profile_token_count": user_profile_token_count,
            "candidate_title_token_count": candidate_title_token_count,
            "candidate_abstract_token_count": candidate_abstract_token_count,
            "candidate_full_token_count": candidate_full_token_count,

            # Lexical matching features
            "user_candidate_title_jaccard": user_candidate_title_jaccard,
            "user_candidate_full_jaccard": user_candidate_full_jaccard,
            "user_candidate_title_overlap_count": user_candidate_title_overlap_count,
            "user_candidate_full_overlap_count": user_candidate_full_overlap_count,
            "user_candidate_title_overlap_ratio": user_candidate_title_overlap_ratio,
            "user_candidate_full_overlap_ratio": user_candidate_full_overlap_ratio,

            # History-to-candidate title alignment features
            "avg_history_title_jaccard": avg_history_title_jaccard,
            "max_history_title_jaccard": max_history_title_jaccard,
            "avg_history_title_overlap_count": avg_history_title_overlap_count,
            "max_history_title_overlap_count": max_history_title_overlap_count,

            # Category affinity features
            "candidate_category_match_fraction": candidate_category_match_fraction,
            "candidate_subcategory_match_fraction": candidate_subcategory_match_fraction,
            "candidate_category_seen_in_history": candidate_category_seen_in_history,
            "candidate_subcategory_seen_in_history": candidate_subcategory_seen_in_history,
        }

        if retrieval_scores is not None:
            row_dict["retrieval_score"] = float(retrieval_scores[idx])

        output_rows.append(row_dict)

    features_df = pd.DataFrame(output_rows)

    if not features_df.empty:
        features_df["clicked"] = features_df["clicked"].astype("int8")

    return features_df


def build_ranking_examples(
    config: RankingDatasetConfig,
    retrieval_model: TfidfRetrievalBaseline | None = None,
) -> pd.DataFrame:
    """
    High-level convenience function to build reranking features directly from a
    dataset config.

    Parameters
    ----------
    config : RankingDatasetConfig
        Controls split, sampling, and feature options.
    retrieval_model : TfidfRetrievalBaseline | None
        Optional fitted retrieval model to add retrieval_score as a feature.

    Returns
    -------
    pd.DataFrame
        Reranking feature table.
    """
    retrieval_config = RetrievalDatasetConfig(
        split=config.split,
        max_impressions=config.max_impressions,
        negatives_per_positive=config.negatives_per_positive,
        max_history_items=config.max_history_items,
        random_state=config.random_state,
    )

    retrieval_examples_df = build_retrieval_examples(retrieval_config)
    news_df = load_news(config.split)
    news_lookup = build_news_lookup(news_df)

    return build_ranking_features_from_retrieval_examples(
        retrieval_examples_df=retrieval_examples_df,
        news_lookup=news_lookup,
        retrieval_model=retrieval_model,
        include_retrieval_score=config.include_retrieval_score,
    )


# ------------------------------------------------------------------------------
# Summary helper
# ------------------------------------------------------------------------------

def summarize_ranking_examples(features_df: pd.DataFrame) -> dict:
    """
    Summarize the reranking feature table.
    """
    if features_df.empty:
        return {
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": 0.0,
            "unique_impressions": 0,
            "unique_users": 0,
            "unique_candidates": 0,
            "num_feature_columns": 0,
        }

    positives = int(features_df["clicked"].sum())
    rows = int(len(features_df))
    negatives = rows - positives

    non_feature_cols = {
        "split",
        "impression_id",
        "user_id",
        "candidate_news_id",
        "clicked",
    }
    feature_columns = [col for col in features_df.columns if col not in non_feature_cols]

    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / rows if rows else 0.0,
        "unique_impressions": int(features_df["impression_id"].nunique()),
        "unique_users": int(features_df["user_id"].nunique()),
        "unique_candidates": int(features_df["candidate_news_id"].nunique()),
        "num_feature_columns": len(feature_columns),
    }


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    We:
    1. build a tiny sampled retrieval train set
    2. fit a tiny TF-IDF retrieval baseline
    3. build a small reranking dataset from dev
    4. inspect summary + preview

    This checks that:
    - retrieval score can be added as a feature
    - feature engineering works end to end
    """
    print("Running ranking dataset smoke test...\n")

    # --------------------------------------------------------------------------
    # Fit a small retrieval model first so we can add retrieval_score as a
    # second-stage ranking feature.
    # --------------------------------------------------------------------------
    print("Building small train retrieval examples...")
    train_retrieval_examples = build_retrieval_examples(
        RetrievalDatasetConfig(
            split="train",
            max_impressions=300,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        )
    )

    print("Fitting small TF-IDF retrieval baseline...")
    retrieval_model = TfidfRetrievalBaseline(
        config=TfidfRetrievalConfig(
            max_features=20000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            lowercase=True,
        )
    )
    retrieval_model.fit(train_retrieval_examples)

    # --------------------------------------------------------------------------
    # Build reranking features for a tiny dev sample
    # --------------------------------------------------------------------------
    print("Building reranking features...")
    ranking_df = build_ranking_examples(
        RankingDatasetConfig(
            split="dev",
            max_impressions=8,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
            include_retrieval_score=True,
        ),
        retrieval_model=retrieval_model,
    )

    summary = summarize_ranking_examples(ranking_df)

    print("\nRanking dataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSample reranking rows:")
    preview_columns = [
        "impression_id",
        "user_id",
        "candidate_news_id",
        "clicked",
        "retrieval_score",
        "history_length",
        "impression_size",
        "user_candidate_title_jaccard",
        "user_candidate_full_jaccard",
        "candidate_category_match_fraction",
        "candidate_subcategory_match_fraction",
        "max_history_title_jaccard",
    ]
    available_columns = [col for col in preview_columns if col in ranking_df.columns]
    print(ranking_df[available_columns].head(10).to_string(index=False))