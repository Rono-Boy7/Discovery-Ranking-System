"""
src/data/loader.py

Purpose
-------
Reusable utilities for loading the normalized interim MIND-small artifacts that
were created by scripts/preprocess.py.

Why this file matters
---------------------
From this point onward, the rest of the project should avoid touching the raw
TSV files directly unless absolutely necessary.

Instead, downstream code should use this loader module to read:
- news tables
- behaviors tables
- candidate tables
- optional joined views for debugging / exploration

This keeps the project clean and gives us a single source of truth for how the
dataset is loaded.

Expected interim files
----------------------
data/interim/
├── train_news.csv.gz
├── train_behaviors.pkl
├── train_candidates.csv.gz
├── dev_news.csv.gz
├── dev_behaviors.pkl
├── dev_candidates.csv.gz
└── dataset_summary.json

Run local smoke test
--------------------
From the repo root:
    python3 -m src.data.loader
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

SplitName = Literal["train", "dev"]


# ------------------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------------------

def get_repo_root() -> Path:
    """
    Return the repository root based on this file's location.

    Current file:
        src/data/loader.py

    Repo root is therefore:
        ../../
    """
    return Path(__file__).resolve().parents[2]


def get_interim_root() -> Path:
    """
    Return the path to the interim data directory.
    """
    return get_repo_root() / "data" / "interim"


def get_summary_path() -> Path:
    """
    Return the path to dataset_summary.json.
    """
    return get_interim_root() / "dataset_summary.json"


def get_news_path(split: SplitName) -> Path:
    """
    Return the file path for the split's news table.
    """
    return get_interim_root() / f"{split}_news.csv.gz"


def get_behaviors_path(split: SplitName) -> Path:
    """
    Return the file path for the split's behaviors table.
    """
    return get_interim_root() / f"{split}_behaviors.pkl"


def get_candidates_path(split: SplitName) -> Path:
    """
    Return the file path for the split's candidates table.
    """
    return get_interim_root() / f"{split}_candidates.csv.gz"


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def _validate_split(split: str) -> SplitName:
    """
    Validate that the provided split is supported.

    Supported splits:
    - train
    - dev
    """
    if split not in {"train", "dev"}:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of: 'train', 'dev'."
        )
    return split  # type: ignore[return-value]


def validate_interim_files(split: SplitName) -> None:
    """
    Ensure the expected interim files for a split exist before trying to load.

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    required_paths = [
        get_news_path(split),
        get_behaviors_path(split),
        get_candidates_path(split),
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected interim artifact not found: {path}\n"
                "Did you run: python3 scripts/preprocess.py ?"
            )


# ------------------------------------------------------------------------------
# Summary loading
# ------------------------------------------------------------------------------

def load_dataset_summary() -> dict:
    """
    Load the dataset summary JSON created during preprocessing.

    Returns
    -------
    dict
        Parsed JSON summary.
    """
    summary_path = get_summary_path()

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Dataset summary file not found: {summary_path}\n"
            "Did you run: python3 scripts/preprocess.py ?"
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Core table loaders
# ------------------------------------------------------------------------------

def load_news(split: str) -> pd.DataFrame:
    """
    Load the news table for a given split.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    pd.DataFrame
        One row per news article.
    """
    split = _validate_split(split)
    validate_interim_files(split)

    news_df = pd.read_csv(get_news_path(split), compression="gzip")

    # Ensure key ID columns remain strings.
    if "news_id" in news_df.columns:
        news_df["news_id"] = news_df["news_id"].astype(str)

    return news_df


def load_behaviors(split: str) -> pd.DataFrame:
    """
    Load the behaviors table for a given split.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    pd.DataFrame
        One row per impression record.

    Notes
    -----
    This file is stored as a pickle because it contains Python list columns such
    as:
    - history_list
    - impression_list
    """
    split = _validate_split(split)
    validate_interim_files(split)

    behaviors_df = pd.read_pickle(get_behaviors_path(split))

    # Keep IDs as strings for safe downstream joins.
    for col in ["impression_id", "user_id"]:
        if col in behaviors_df.columns:
            behaviors_df[col] = behaviors_df[col].astype(str)

    return behaviors_df


def load_candidates(split: str) -> pd.DataFrame:
    """
    Load the candidates table for a given split.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    pd.DataFrame
        One row per candidate item shown in an impression.

    Notes
    -----
    This is the normalized version of expanded impressions and does not include
    repeated history columns on every row.
    """
    split = _validate_split(split)
    validate_interim_files(split)

    candidates_df = pd.read_csv(
        get_candidates_path(split),
        compression="gzip",
        parse_dates=["timestamp"],
    )

    # Keep IDs as strings for safe downstream joins.
    for col in ["impression_id", "user_id", "candidate_news_id"]:
        if col in candidates_df.columns:
            candidates_df[col] = candidates_df[col].astype(str)

    # Keep label type compact and explicit.
    if "clicked" in candidates_df.columns:
        candidates_df["clicked"] = candidates_df["clicked"].astype("int8")

    return candidates_df


# ------------------------------------------------------------------------------
# Convenience multi-loaders
# ------------------------------------------------------------------------------

def load_split_artifacts(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three core artifacts for a split.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (news_df, behaviors_df, candidates_df)
    """
    split = _validate_split(split)

    news_df = load_news(split)
    behaviors_df = load_behaviors(split)
    candidates_df = load_candidates(split)

    return news_df, behaviors_df, candidates_df


# ------------------------------------------------------------------------------
# Joined views for debugging / model input preparation
# ------------------------------------------------------------------------------

def build_candidate_news_frame(split: str) -> pd.DataFrame:
    """
    Join candidate rows with news metadata for the candidate article.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    pd.DataFrame
        Candidate rows enriched with the candidate article's news metadata.

    Why this is useful
    ------------------
    This is the most common join we will need for retrieval/reranking work,
    because each training row refers to a candidate_news_id and we often want
    the candidate's title, abstract, category, and full_text.
    """
    split = _validate_split(split)

    news_df = load_news(split)
    candidates_df = load_candidates(split)

    # We rename the news-side columns so it is obvious these features belong
    # to the candidate article, not the user's history articles.
    news_features = news_df.rename(
        columns={
            "news_id": "candidate_news_id",
            "category": "candidate_category",
            "subcategory": "candidate_subcategory",
            "title": "candidate_title",
            "abstract": "candidate_abstract",
            "url": "candidate_url",
            "title_entities": "candidate_title_entities",
            "abstract_entities": "candidate_abstract_entities",
            "full_text": "candidate_full_text",
        }
    )

    merged_df = candidates_df.merge(
        news_features,
        on="candidate_news_id",
        how="left",
        validate="many_to_one",
    )

    return merged_df


def build_full_debug_frame(split: str) -> pd.DataFrame:
    """
    Build a larger joined frame for debugging and exploration.

    This joins:
    - candidates
    - behaviors
    - candidate news metadata

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    pd.DataFrame
        A larger debug-friendly DataFrame.

    Warning
    -------
    This frame can get large, especially for the train split.
    Use it mainly for exploration, debugging, or sampling.
    """
    split = _validate_split(split)

    candidate_news_df = build_candidate_news_frame(split)
    behaviors_df = load_behaviors(split)

    # Keep only the columns we actually need from behaviors to avoid blowing up
    # the joined frame more than necessary.
    behavior_subset = behaviors_df[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "history",
            "history_list",
            "history_length",
            "impression_size",
        ]
    ].copy()

    merged_df = candidate_news_df.merge(
        behavior_subset,
        on=["impression_id", "user_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )

    return merged_df


# ------------------------------------------------------------------------------
# Quick dataset stats helpers
# ------------------------------------------------------------------------------

def get_label_distribution(split: str) -> dict:
    """
    Return basic label statistics for a split.

    Parameters
    ----------
    split : str
        Either 'train' or 'dev'.

    Returns
    -------
    dict
        Dictionary with total rows, positive rows, negative rows, and positive rate.
    """
    split = _validate_split(split)
    candidates_df = load_candidates(split)

    total_rows = int(len(candidates_df))
    positive_rows = int(candidates_df["clicked"].sum())
    negative_rows = total_rows - positive_rows
    positive_rate = positive_rows / total_rows if total_rows else 0.0

    return {
        "split": split,
        "total_rows": total_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_rate": positive_rate,
    }


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small local smoke test.

    Run from repo root:
        python3 -m src.data.loader
    """
    print("\nLoading dataset summary...")
    summary = load_dataset_summary()
    print(json.dumps(summary, indent=2)[:1200])  # print a trimmed preview

    for split_name in ["train", "dev"]:
        print(f"\n{'-' * 80}")
        print(f"Testing split: {split_name}")
        print(f"{'-' * 80}")

        news_df = load_news(split_name)
        behaviors_df = load_behaviors(split_name)
        candidates_df = load_candidates(split_name)

        print(f"News shape: {news_df.shape}")
        print(f"Behaviors shape: {behaviors_df.shape}")
        print(f"Candidates shape: {candidates_df.shape}")

        stats = get_label_distribution(split_name)
        print(f"Label stats: {stats}")

        candidate_news_df = build_candidate_news_frame(split_name)
        print(f"Candidate + news shape: {candidate_news_df.shape}")

        print("\nSample candidate + news rows:")
        preview_cols = [
            "impression_id",
            "user_id",
            "candidate_news_id",
            "clicked",
            "candidate_category",
            "candidate_subcategory",
            "candidate_title",
        ]
        available_cols = [col for col in preview_cols if col in candidate_news_df.columns]
        print(candidate_news_df[available_cols].head(5).to_string(index=False))