"""
scripts/train_retrieval.py

Purpose
-------
Command-line entry point for training and evaluating the TF-IDF retrieval
baseline.

Why this file matters
---------------------
Right now, we can run the training pipeline with:

    python3 -m src.retrieval.trainer

That works, but a real project usually exposes a top-level script that:
- accepts experiment parameters from the command line
- makes runs easier to reproduce
- avoids editing source files for every experiment

This script wraps the trainer and lets you control:
- train/dev sample sizes
- negatives per positive
- max history size
- TF-IDF vocabulary size
- n-gram range
- whether to save the model
- run name

Example usage
-------------
Small experiment:
    python3 scripts/train_retrieval.py

Custom experiment:
    python3 scripts/train_retrieval.py \
        --run-name tfidf_exp_001 \
        --train-max-impressions 1000 \
        --dev-max-impressions 200 \
        --negatives-per-positive 4 \
        --max-history-items 8 \
        --max-features 40000 \
        --ngram-max 2

Notes
-----
This script currently trains the TF-IDF retrieval baseline only.
Later, when we add neural retrievers, this can be extended to support multiple
model types.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ------------------------------------------------------------------------------
# Make sure the repo root is importable
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.retrieval.dataset import RetrievalDatasetConfig  # noqa: E402
from src.retrieval.model import TfidfRetrievalConfig  # noqa: E402
from src.retrieval.trainer import (  # noqa: E402
    RetrievalTrainerConfig,
    train_and_evaluate_retrieval_baseline,
)


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for retrieval baseline training.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TF-IDF retrieval baseline."
    )

    # --------------------------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit run name. If omitted, a timestamp-based name is used."
        ),
    )

    # --------------------------------------------------------------------------
    # Dataset sampling / construction
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--train-max-impressions",
        type=int,
        default=500,
        help=(
            "Maximum number of train impressions to sample. "
            "Use a larger value for stronger experiments."
        ),
    )
    parser.add_argument(
        "--dev-max-impressions",
        type=int,
        default=120,
        help=(
            "Maximum number of dev impressions to sample for evaluation."
        ),
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=4,
        help=(
            "Number of negative candidates to keep per positive candidate inside "
            "each impression."
        ),
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=5,
        help=(
            "Number of most recent history articles used to build the user profile."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    # --------------------------------------------------------------------------
    # TF-IDF model parameters
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum vocabulary size for the TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms that appear in fewer than this many documents.",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing in the TF-IDF vectorizer.",
    )

    # --------------------------------------------------------------------------
    # Persistence / output options
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save the fitted model artifact.",
    )
    parser.add_argument(
        "--max-scored-dev-rows-to-save",
        type=int,
        default=500,
        help=(
            "Maximum number of scored dev rows to save for later inspection."
        ),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate CLI arguments before launching the training run.
    """
    positive_int_fields = {
        "train_max_impressions": args.train_max_impressions,
        "dev_max_impressions": args.dev_max_impressions,
        "negatives_per_positive": args.negatives_per_positive,
        "max_history_items": args.max_history_items,
        "max_features": args.max_features,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
        "max_scored_dev_rows_to_save": args.max_scored_dev_rows_to_save,
    }

    for field_name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be > 0. Got: {value}")

    if args.ngram_min > args.ngram_max:
        raise ValueError(
            f"--ngram-min ({args.ngram_min}) cannot be greater than "
            f"--ngram-max ({args.ngram_max})."
        )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, build trainer config, run the experiment, and print summary.
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    trainer_config = RetrievalTrainerConfig(
        run_name=args.run_name,
        train_dataset=RetrievalDatasetConfig(
            split="train",
            max_impressions=args.train_max_impressions,
            negatives_per_positive=args.negatives_per_positive,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
        ),
        dev_dataset=RetrievalDatasetConfig(
            split="dev",
            max_impressions=args.dev_max_impressions,
            negatives_per_positive=args.negatives_per_positive,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
        ),
        model=TfidfRetrievalConfig(
            max_features=args.max_features,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            lowercase=not args.no_lowercase,
        ),
        metrics_ks=(5, 10, 20),
        save_model=not args.no_save_model,
        max_scored_dev_rows_to_save=args.max_scored_dev_rows_to_save,
    )

    print("Launching retrieval baseline experiment with config:\n")
    print(
        json.dumps(
            {
                "run_name": trainer_config.run_name,
                "train_dataset": trainer_config.train_dataset.__dict__,
                "dev_dataset": trainer_config.dev_dataset.__dict__,
                "model": trainer_config.model.__dict__,
                "metrics_ks": trainer_config.metrics_ks,
                "save_model": trainer_config.save_model,
                "max_scored_dev_rows_to_save": trainer_config.max_scored_dev_rows_to_save,
            },
            indent=2,
        )
    )

    result = train_and_evaluate_retrieval_baseline(trainer_config)

    print("\nExperiment finished successfully.")
    print("\nFinal run summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()