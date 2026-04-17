"""
src/data/parser.py

Purpose
-------
Utilities for parsing the raw MIND dataset files into pandas DataFrames that are
easy to work with for preprocessing, retrieval training, and reranking.

Why this file matters
---------------------
The MIND dataset ships as TSV files with some fields packed into strings:
- news.tsv contains article metadata
- behaviors.tsv contains user histories and impression strings

The impression column in behaviors.tsv looks like this:
    "N12345-1 N67890-0 N11111-0"

Each token means:
    <news_id>-<label>

Where:
- label = 1 means the user clicked that candidate
- label = 0 means the user did not click it

For ranking systems, we usually need one row per candidate impression, so this
module also provides a function to "explode" impressions into trainable rows.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ------------------------------------------------------------------------------
# MIND FILE SCHEMAS
# ------------------------------------------------------------------------------

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
]


# ------------------------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------------------------

def _normalize_text(value: object) -> str:
    """
    Convert missing/NaN values to an empty string and strip whitespace.

    We keep things simple here because the goal of this stage is just to produce
    clean, predictable strings for downstream preprocessing.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split_space_separated_ids(value: object) -> list[str]:
    """
    Convert a space-separated string of IDs into a Python list.

    Example:
        "N123 N456 N789" -> ["N123", "N456", "N789"]

    If the field is missing or empty, return an empty list.
    """
    text = _normalize_text(value)
    if not text:
        return []
    return text.split()


# ------------------------------------------------------------------------------
# NEWS PARSING
# ------------------------------------------------------------------------------

def parse_news_file(news_path: str | Path) -> pd.DataFrame:
    """
    Parse a MIND news.tsv file into a pandas DataFrame.

    Parameters
    ----------
    news_path : str | Path
        Path to a MIND news.tsv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per news article.

    Notes
    -----
    Columns in MIND news.tsv:
        1. news_id
        2. category
        3. subcategory
        4. title
        5. abstract
        6. url
        7. title_entities
        8. abstract_entities
    """
    news_path = Path(news_path)

    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}")

    df = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    # Basic cleanup so downstream code sees predictable string values.
    for col in NEWS_COLUMNS:
        df[col] = df[col].apply(_normalize_text)

    # Useful combined text field for later embedding / retrieval work.
    df["full_text"] = (
        df["title"].fillna("")
        + " "
        + df["abstract"].fillna("")
    ).str.strip()

    return df


# ------------------------------------------------------------------------------
# BEHAVIORS PARSING
# ------------------------------------------------------------------------------

def parse_behaviors_file(behaviors_path: str | Path) -> pd.DataFrame:
    """
    Parse a MIND behaviors.tsv file into a pandas DataFrame.

    Parameters
    ----------
    behaviors_path : str | Path
        Path to a MIND behaviors.tsv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per behavior record.

    Notes
    -----
    Columns in MIND behaviors.tsv:
        1. impression_id
        2. user_id
        3. timestamp
        4. history
        5. impressions

    Example row:
        1    U100    11/15/2019 9:55:12 AM    N1 N2 N3    N4-1 N5-0 N6-0

    We keep both:
    - raw string columns
    - parsed list columns for convenience
    """
    behaviors_path = Path(behaviors_path)

    if not behaviors_path.exists():
        raise FileNotFoundError(f"Behaviors file not found: {behaviors_path}")

    df = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=BEHAVIORS_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    for col in BEHAVIORS_COLUMNS:
        df[col] = df[col].apply(_normalize_text)

    # Parse timestamp. We keep invalid values as NaT rather than crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Convert user's historical clicked news IDs into a list.
    df["history_list"] = df["history"].apply(_split_space_separated_ids)

    # Keep impressions as raw strings for traceability, but also parse into a list.
    df["impression_list"] = df["impressions"].apply(_split_space_separated_ids)

    return df


# ------------------------------------------------------------------------------
# IMPRESSION EXPANSION
# ------------------------------------------------------------------------------

def _parse_impression_token(token: str) -> tuple[Optional[str], Optional[int]]:
    """
    Parse a single impression token like 'N12345-1'.

    Returns
    -------
    tuple[Optional[str], Optional[int]]
        (candidate_news_id, clicked_label)

    If parsing fails, returns (None, None).
    """
    token = _normalize_text(token)
    if not token:
        return None, None

    # MIND impression tokens are expected to look like:
    #   <news_id>-<label>
    # Example:
    #   N12345-1
    try:
        news_id, label = token.rsplit("-", 1)
        return news_id, int(label)
    except (ValueError, TypeError):
        return None, None


def expand_impressions(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the behaviors DataFrame into one row per candidate impression.

    Parameters
    ----------
    behaviors_df : pd.DataFrame
        Output of parse_behaviors_file().

    Returns
    -------
    pd.DataFrame
        One row per candidate item shown to a user in an impression set.

    Output columns
    --------------
    impression_id : str
        The original impression group ID.
    user_id : str
        User identifier.
    timestamp : datetime64
        Timestamp of the impression event.
    history : str
        Original raw history string.
    history_list : list[str]
        Parsed user history.
    candidate_news_id : str
        Candidate news article shown to the user.
    clicked : int
        1 if clicked, 0 otherwise.

    Example
    -------
    Input row:
        impressions = "N10-1 N20-0 N30-0"

    Expanded rows:
        candidate_news_id= N10, clicked=1
        candidate_news_id= N20, clicked=0
        candidate_news_id= N30, clicked=0
    """
    required_columns = {
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "history_list",
        "impression_list",
    }

    missing = required_columns - set(behaviors_df.columns)
    if missing:
        raise ValueError(
            "behaviors_df is missing required columns for expansion: "
            + ", ".join(sorted(missing))
        )

    expanded_rows: list[dict] = []

    for row in behaviors_df.itertuples(index=False):
        # row.impression_list is a list like:
        # ["N123-1", "N456-0", "N789-0"]
        for token in row.impression_list:
            candidate_news_id, clicked = _parse_impression_token(token)

            # Skip malformed tokens instead of crashing the whole pipeline.
            if candidate_news_id is None or clicked is None:
                continue

            expanded_rows.append(
                {
                    "impression_id": row.impression_id,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "history": row.history,
                    "history_list": row.history_list,
                    "candidate_news_id": candidate_news_id,
                    "clicked": clicked,
                }
            )

    expanded_df = pd.DataFrame(expanded_rows)

    # Make sure clicked is integer typed when data exists.
    if not expanded_df.empty:
        expanded_df["clicked"] = expanded_df["clicked"].astype(int)

    return expanded_df


# ------------------------------------------------------------------------------
# SPLIT-LEVEL HELPERS
# ------------------------------------------------------------------------------

def load_mind_split(split_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one MIND split directory containing:
    - news.tsv
    - behaviors.tsv

    Parameters
    ----------
    split_dir : str | Path
        Path to a split folder such as:
        data/raw/mind_small/MINDsmall_train

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (news_df, behaviors_df)
    """
    split_dir = Path(split_dir)

    news_path = split_dir / "news.tsv"
    behaviors_path = split_dir / "behaviors.tsv"

    news_df = parse_news_file(news_path)
    behaviors_df = parse_behaviors_file(behaviors_path)

    return news_df, behaviors_df


def load_and_expand_mind_split(split_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper that loads a split and also expands impressions.

    Parameters
    ----------
    split_dir : str | Path
        Path to one MIND split directory.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (news_df, behaviors_df, impressions_df)
    """
    news_df, behaviors_df = load_mind_split(split_dir)
    impressions_df = expand_impressions(behaviors_df)
    return news_df, behaviors_df, impressions_df


# ------------------------------------------------------------------------------
# OPTIONAL LOCAL SMOKE TEST
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small local smoke test.

    Run this from the repo root with:
        python -m src.data.parser

    This block assumes your current project structure matches what we created.
    It helps quickly confirm:
    - files can be found
    - parsing works
    - impression expansion works
    """
    repo_root = Path(__file__).resolve().parents[2]
    train_dir = repo_root / "data" / "raw" / "mind_small" / "MINDsmall_train"
    dev_dir = repo_root / "data" / "raw" / "mind_small" / "MINDsmall_dev"

    for split_name, split_path in [("train", train_dir), ("dev", dev_dir)]:
        print(f"\n--- Loading {split_name} split from: {split_path} ---")

        news_df, behaviors_df, impressions_df = load_and_expand_mind_split(split_path)

        print(f"News rows: {len(news_df):,}")
        print(f"Behavior rows: {len(behaviors_df):,}")
        print(f"Expanded impression rows: {len(impressions_df):,}")

        print("\nSample news rows:")
        print(news_df[["news_id", "category", "subcategory", "title"]].head(3).to_string(index=False))

        print("\nSample behavior rows:")
        print(
            behaviors_df[
                ["impression_id", "user_id", "timestamp", "history", "impressions"]
            ].head(2).to_string(index=False)
        )

        print("\nSample expanded impression rows:")
        print(impressions_df.head(5).to_string(index=False))