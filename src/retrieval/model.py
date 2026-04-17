"""
src/retrieval/model.py

Purpose
-------
First retrieval baseline for the discovery ranking system.

What this file does
-------------------
This module implements a simple but useful retrieval model:

1. Build a text representation for the user from recent clicked history titles
2. Build a text representation for the candidate article
3. Fit a TF-IDF vectorizer
4. Score each (user, candidate) pair with cosine similarity
5. Evaluate ranking quality with MRR / Recall / NDCG

Why start with this baseline?
-----------------------------
Before jumping into neural dual encoders, we want a simple baseline that:
- is easy to understand
- is easy to debug
- gives us a reference point for future improvements

A strong ML project is not just "deep learning first".
It should show:
- baselines
- evaluation discipline
- iteration

This baseline is exactly that.

Run local smoke test
--------------------
From the repo root:
    python3 -m src.retrieval.model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation.metrics import evaluate_grouped_ranking
from src.retrieval.dataset import (
    RetrievalDatasetConfig,
    build_retrieval_examples,
    summarize_retrieval_examples,
)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class TfidfRetrievalConfig:
    """
    Configuration for the TF-IDF retrieval baseline.

    Attributes
    ----------
    max_features : int
        Maximum size of the TF-IDF vocabulary.
    ngram_min : int
        Minimum n-gram size.
    ngram_max : int
        Maximum n-gram size.
    min_df : int
        Ignore terms that appear in fewer than this many documents.
    lowercase : bool
        Whether to lowercase text.
    """

    max_features: int = 50000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    lowercase: bool = True


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class TfidfRetrievalBaseline:
    """
    A simple retrieval baseline using TF-IDF + cosine similarity.

    Idea
    ----
    We represent:
    - the user profile as a piece of text built from recent history titles
    - the candidate article as a piece of text built from title/abstract/category

    Then we score a pair by the cosine similarity between their TF-IDF vectors.

    Important
    ---------
    This is a baseline model, not the final production-grade neural retriever.
    But it is extremely useful as:
    - a sanity check
    - a benchmark
    - a debugging tool
    """

    def __init__(self, config: TfidfRetrievalConfig | None = None) -> None:
        self.config = config or TfidfRetrievalConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.is_fitted: bool = False

    # --------------------------------------------------------------------------
    # Text preparation
    # --------------------------------------------------------------------------

    @staticmethod
    def _safe_text(value: object) -> str:
        """
        Convert missing values into clean strings.
        """
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _build_candidate_text(self, examples_df: pd.DataFrame) -> pd.Series:
        """
        Build the text representation for candidate articles.

        We combine:
        - category
        - subcategory
        - title
        - abstract

        This gives the baseline a little richer signal than title alone while
        still keeping the representation interpretable.
        """
        category = examples_df["candidate_category"].fillna("").astype(str).str.strip()
        subcategory = examples_df["candidate_subcategory"].fillna("").astype(str).str.strip()
        title = examples_df["candidate_title"].fillna("").astype(str).str.strip()
        abstract = examples_df["candidate_abstract"].fillna("").astype(str).str.strip()

        candidate_text = (
            "[CAT] " + category
            + " [SUBCAT] " + subcategory
            + " [TITLE] " + title
            + " [ABSTRACT] " + abstract
        ).str.strip()

        return candidate_text

    def _build_user_text(self, examples_df: pd.DataFrame) -> pd.Series:
        """
        Build the text representation for the user.

        For now, this is simply the user_profile_text already created by the
        retrieval dataset builder from recent history titles.
        """
        return examples_df["user_profile_text"].fillna("").astype(str).str.strip()

    def _validate_examples(self, examples_df: pd.DataFrame) -> None:
        """
        Validate that the expected columns are present.
        """
        required_columns = {
            "impression_id",
            "clicked",
            "user_profile_text",
            "candidate_category",
            "candidate_subcategory",
            "candidate_title",
            "candidate_abstract",
        }
        missing = required_columns - set(examples_df.columns)
        if missing:
            raise ValueError(
                "Examples DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

    # --------------------------------------------------------------------------
    # Fitting
    # --------------------------------------------------------------------------

    def fit(self, train_examples: pd.DataFrame) -> None:
        """
        Fit the TF-IDF vectorizer on training text.

        Parameters
        ----------
        train_examples : pd.DataFrame
            Retrieval examples built from the training split.

        Notes
        -----
        We fit on both user texts and candidate texts so the vectorizer sees the
        vocabulary of both sides of the matching problem.
        """
        self._validate_examples(train_examples)

        user_text = self._build_user_text(train_examples)
        candidate_text = self._build_candidate_text(train_examples)

        training_corpus = pd.concat([user_text, candidate_text], ignore_index=True)

        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            min_df=self.config.min_df,
            lowercase=self.config.lowercase,
        )

        self.vectorizer.fit(training_corpus)
        self.is_fitted = True

    # --------------------------------------------------------------------------
    # Scoring
    # --------------------------------------------------------------------------

    def score_examples(self, examples_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score retrieval examples with cosine similarity.

        Parameters
        ----------
        examples_df : pd.DataFrame
            Retrieval examples.

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame with an added 'score' column.
        """
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        self._validate_examples(examples_df)

        scored_df = examples_df.copy()

        user_text = self._build_user_text(scored_df)
        candidate_text = self._build_candidate_text(scored_df)

        user_matrix = self.vectorizer.transform(user_text)
        candidate_matrix = self.vectorizer.transform(candidate_text)

        # Because TF-IDF vectors are L2-normalized by default, the row-wise dot
        # product gives cosine similarity for matching rows.
        similarity_scores = user_matrix.multiply(candidate_matrix).sum(axis=1).A1

        scored_df["score"] = similarity_scores.astype(float)
        return scored_df

    # --------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------

    def evaluate(
        self,
        examples_df: pd.DataFrame,
        ks: Iterable[int] = (5, 10, 20),
    ) -> dict:
        """
        Score examples and compute grouped ranking metrics.
        """
        scored_df = self.score_examples(examples_df)
        return evaluate_grouped_ranking(scored_df, ks=ks)


# ------------------------------------------------------------------------------
# Helper for readable preview
# ------------------------------------------------------------------------------

def preview_top_ranked_candidates(
    scored_df: pd.DataFrame,
    top_n_per_impression: int = 3,
    max_impressions: int = 3,
) -> pd.DataFrame:
    """
    Return a compact preview of top-ranked candidates per impression.

    This is useful for debugging whether the model is producing rankings that
    look sensible.
    """
    if "score" not in scored_df.columns:
        raise ValueError("Expected a 'score' column in scored_df.")

    working_df = scored_df.sort_values(
        by=["impression_id", "score"],
        ascending=[True, False],
    ).copy()

    preview_df = (
        working_df.groupby("impression_id", sort=False)
        .head(top_n_per_impression)
        .copy()
    )

    selected_impressions = preview_df["impression_id"].drop_duplicates().head(max_impressions)
    preview_df = preview_df[preview_df["impression_id"].isin(selected_impressions)].copy()

    preview_columns = [
        "impression_id",
        "user_id",
        "clicked",
        "score",
        "candidate_news_id",
        "candidate_title",
    ]
    available_columns = [col for col in preview_columns if col in preview_df.columns]

    return preview_df[available_columns]


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    We:
    1. Build a small sampled training set from the train split
    2. Build a small sampled evaluation set from the dev split
    3. Fit the TF-IDF baseline
    4. Evaluate ranking metrics on the dev examples

    This is just a pipeline sanity check, not a final experiment.
    """
    print("Running TF-IDF retrieval baseline smoke test...\n")

    # Small sampled training dataset
    train_dataset_config = RetrievalDatasetConfig(
        split="train",
        max_impressions=300,
        negatives_per_positive=4,
        max_history_items=5,
        random_state=42,
    )

    # Small sampled dev dataset
    dev_dataset_config = RetrievalDatasetConfig(
        split="dev",
        max_impressions=80,
        negatives_per_positive=4,
        max_history_items=5,
        random_state=42,
    )

    print("Building training examples...")
    train_examples = build_retrieval_examples(train_dataset_config)
    train_summary = summarize_retrieval_examples(train_examples)

    print("Training example summary:")
    for key, value in train_summary.items():
        print(f"{key}: {value}")

    print("\nBuilding dev examples...")
    dev_examples = build_retrieval_examples(dev_dataset_config)
    dev_summary = summarize_retrieval_examples(dev_examples)

    print("Dev example summary:")
    for key, value in dev_summary.items():
        print(f"{key}: {value}")

    print("\nFitting TF-IDF retrieval baseline...")
    model = TfidfRetrievalBaseline(
        config=TfidfRetrievalConfig(
            max_features=30000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            lowercase=True,
        )
    )
    model.fit(train_examples)

    print("Scoring dev examples...")
    scored_dev = model.score_examples(dev_examples)

    print("\nEvaluating ranking metrics...")
    metrics = model.evaluate(dev_examples, ks=(5, 10, 20))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nTop-ranked candidate preview:")
    preview_df = preview_top_ranked_candidates(
        scored_df=scored_dev,
        top_n_per_impression=3,
        max_impressions=3,
    )
    print(preview_df.to_string(index=False))