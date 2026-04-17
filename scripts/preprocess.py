"""
scripts/preprocess.py

Purpose
-------
Turn the raw MIND-small files into normalized interim artifacts that are faster
and cleaner to reuse in later stages of the project.

Why this matters
----------------
Right now, our parser can expand impressions into one row per candidate item.
That is useful, but it creates a *very large* table because the user's full
history gets repeated on every candidate row.

For example, if one impression has:
- one user history
- ten candidate articles

Then that same history is repeated ten times.

That is wasteful for storage and slower for future steps.

So in this preprocessing script we normalize the data into 3 reusable tables
per split:

1. news table
   - one row per article

2. behaviors table
   - one row per impression
   - includes the user's history list
   - keeps impression-level metadata

3. candidates table
   - one row per candidate item shown in an impression
   - contains the label (clicked or not clicked)
   - does NOT repeat the full history on every row

Output files
------------
These will be saved under:
    data/interim/

For each split ("train" and "dev"), we save:
- {split}_news.csv.gz
- {split}_behaviors.pkl
- {split}_candidates.csv.gz

We also save:
- dataset_summary.json

Why mixed formats?
------------------
- CSV.GZ is easy to inspect and version mentally.
- Pickle is used for the behaviors table because it contains Python list columns
  like history_list and impression_list, which do not round-trip cleanly in CSV.

Run
---
From the repo root:
    python3 scripts/preprocess.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------
# Make sure the repo root is importable
# ------------------------------------------------------------------------------
# This lets us import from src/... even when running:
#   python3 scripts/preprocess.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.parser import load_and_expand_mind_split  # noqa: E402


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
RAW_ROOT = REPO_ROOT / "data" / "raw" / "mind_small"
INTERIM_ROOT = REPO_ROOT / "data" / "interim"

TRAIN_DIR = RAW_ROOT / "MINDsmall_train"
DEV_DIR = RAW_ROOT / "MINDsmall_dev"


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Ensure the interim output directory exists.
    """
    INTERIM_ROOT.mkdir(parents=True, exist_ok=True)


def validate_required_paths() -> None:
    """
    Check that the expected raw dataset folders exist before we start.

    We fail early with a clear error if something is missing.
    """
    required_dirs = [TRAIN_DIR, DEV_DIR]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Expected dataset folder not found: {directory}")

        news_path = directory / "news.tsv"
        behaviors_path = directory / "behaviors.tsv"

        if not news_path.exists():
            raise FileNotFoundError(f"Missing news.tsv in: {directory}")

        if not behaviors_path.exists():
            raise FileNotFoundError(f"Missing behaviors.tsv in: {directory}")


def build_split_artifacts(split_name: str, split_dir: Path) -> dict:
    """
    Load one MIND split, normalize it, save reusable artifacts, and return stats.

    Parameters
    ----------
    split_name : str
        Friendly name like "train" or "dev".
    split_dir : Path
        Path to the raw split directory.

    Returns
    -------
    dict
        Summary statistics for this split.
    """
    print(f"\n{'=' * 80}")
    print(f"Building interim artifacts for split: {split_name}")
    print(f"Raw split directory: {split_dir}")
    print(f"{'=' * 80}")

    # --------------------------------------------------------------------------
    # Load raw parsed tables
    # --------------------------------------------------------------------------
    news_df, behaviors_df, expanded_df = load_and_expand_mind_split(split_dir)

    print(f"[{split_name}] Loaded news rows: {len(news_df):,}")
    print(f"[{split_name}] Loaded behavior rows: {len(behaviors_df):,}")
    print(f"[{split_name}] Loaded expanded candidate rows: {len(expanded_df):,}")

    # --------------------------------------------------------------------------
    # Clean / enrich the news table
    # --------------------------------------------------------------------------
    # There should typically be one row per news_id already, but we dedupe just
    # in case to make downstream joins safer.
    news_df = news_df.drop_duplicates(subset=["news_id"]).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # Clean / enrich the behaviors table
    # --------------------------------------------------------------------------
    # behaviors_df is the impression-level table.
    # We add a few convenience columns that will be useful later.
    behaviors_df = behaviors_df.copy()
    behaviors_df["history_length"] = behaviors_df["history_list"].apply(len)
    behaviors_df["impression_size"] = behaviors_df["impression_list"].apply(len)

    # --------------------------------------------------------------------------
    # Normalize the candidates table
    # --------------------------------------------------------------------------
    # expanded_df contains repeated history/history_list columns on every
    # candidate row. We do NOT want to store all that redundancy.
    candidates_df = expanded_df[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "candidate_news_id",
            "clicked",
        ]
    ].copy()

    # Use a smaller integer dtype for the binary label.
    candidates_df["clicked"] = candidates_df["clicked"].astype("int8")

    # --------------------------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------------------------
    expected_candidate_rows = int(behaviors_df["impression_size"].sum())
    actual_candidate_rows = len(candidates_df)

    if actual_candidate_rows != expected_candidate_rows:
        raise ValueError(
            f"[{split_name}] Candidate row mismatch. "
            f"Expected {expected_candidate_rows:,}, got {actual_candidate_rows:,}."
        )

    invalid_labels = set(candidates_df["clicked"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"[{split_name}] Found invalid clicked labels: {sorted(invalid_labels)}"
        )

    # --------------------------------------------------------------------------
    # Compute summary statistics
    # --------------------------------------------------------------------------
    positive_count = int(candidates_df["clicked"].sum())
    total_candidates = int(len(candidates_df))
    negative_count = total_candidates - positive_count
    ctr = positive_count / total_candidates if total_candidates else 0.0

    stats = {
        "split_name": split_name,
        "news_rows": int(len(news_df)),
        "behavior_rows": int(len(behaviors_df)),
        "candidate_rows": total_candidates,
        "positive_rows": positive_count,
        "negative_rows": negative_count,
        "positive_rate": round(ctr, 6),
        "avg_history_length": round(float(behaviors_df["history_length"].mean()), 4),
        "median_history_length": round(float(behaviors_df["history_length"].median()), 4),
        "avg_impression_size": round(float(behaviors_df["impression_size"].mean()), 4),
        "median_impression_size": round(float(behaviors_df["impression_size"].median()), 4),
        "unique_users": int(behaviors_df["user_id"].nunique()),
        "unique_candidate_news_ids": int(candidates_df["candidate_news_id"].nunique()),
    }

    # --------------------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------------------
    news_out = INTERIM_ROOT / f"{split_name}_news.csv.gz"
    behaviors_out = INTERIM_ROOT / f"{split_name}_behaviors.pkl"
    candidates_out = INTERIM_ROOT / f"{split_name}_candidates.csv.gz"

    print(f"[{split_name}] Saving news table -> {news_out}")
    news_df.to_csv(news_out, index=False, compression="gzip")

    print(f"[{split_name}] Saving behaviors table -> {behaviors_out}")
    behaviors_df.to_pickle(behaviors_out)

    print(f"[{split_name}] Saving candidates table -> {candidates_out}")
    candidates_df.to_csv(candidates_out, index=False, compression="gzip")

    # Free memory before moving to the next split.
    del news_df, behaviors_df, expanded_df, candidates_df
    gc.collect()

    print(f"[{split_name}] Done.")
    return stats


def main() -> None:
    """
    Main entrypoint for preprocessing the raw MIND-small dataset.
    """
    ensure_directories()
    validate_required_paths()

    print("Starting preprocessing for MIND-small...")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Raw data root: {RAW_ROOT}")
    print(f"Interim output root: {INTERIM_ROOT}")

    train_stats = build_split_artifacts("train", TRAIN_DIR)
    dev_stats = build_split_artifacts("dev", DEV_DIR)

    summary = {
        "dataset": "MIND-small",
        "raw_root": str(RAW_ROOT),
        "interim_root": str(INTERIM_ROOT),
        "splits": {
            "train": train_stats,
            "dev": dev_stats,
        },
    }

    summary_out = INTERIM_ROOT / "dataset_summary.json"
    print(f"\nSaving dataset summary -> {summary_out}")
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPreprocessing complete.")
    print("Generated files:")
    print(f"  - {INTERIM_ROOT / 'train_news.csv.gz'}")
    print(f"  - {INTERIM_ROOT / 'train_behaviors.pkl'}")
    print(f"  - {INTERIM_ROOT / 'train_candidates.csv.gz'}")
    print(f"  - {INTERIM_ROOT / 'dev_news.csv.gz'}")
    print(f"  - {INTERIM_ROOT / 'dev_behaviors.pkl'}")
    print(f"  - {INTERIM_ROOT / 'dev_candidates.csv.gz'}")
    print(f"  - {summary_out}")


if __name__ == "__main__":
    main()