"""
scripts/train_reranker.py

Purpose
-------
Command-line entry point for training and evaluating the full two-stage ranking
pipeline:

Stage 1:
- TF-IDF retrieval baseline

Stage 2:
- Logistic regression reranker

Why this file matters
---------------------
Right now, we can run the ranking trainer with:

    python3 -m src.ranking.trainer

That works, but a real ML project should also expose a top-level script that:
- accepts experiment parameters from the command line
- makes experiment runs easy to reproduce
- avoids editing source code for every experiment
- mirrors how real ML training jobs are launched

What this script controls
-------------------------
- retrieval training sample size
- ranking train/dev sample size
- negatives per positive
- max user history items
- TF-IDF vectorizer settings
- logistic reranker settings
- whether models should be saved
- run name

Example usage
-------------
Default experiment:
    python3 scripts/train_reranker.py

Larger experiment:
    python3 scripts/train_reranker.py \
        --run-name reranker_exp_001 \
        --retrieval-train-max-impressions 1000 \
        --ranking-train-max-impressions 1000 \
        --ranking-dev-max-impressions 300 \
        --negatives-per-positive 4 \
        --max-history-items 8 \
        --max-features 40000 \
        --ngram-max 2 \
        --logreg-c 0.5 \
        --logreg-max-iter 1500

Notes
-----
This script currently trains:
- TF-IDF retrieval baseline
- logistic regression reranker

Later, when we add stronger rerankers or neural retrieval models, this script can
be extended to support multiple model types.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ------------------------------------------------------------------------------
# Make repo root importable
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.ranking.dataset import RankingDatasetConfig  # noqa: E402
from src.ranking.model import LogisticRerankerConfig  # noqa: E402
from src.ranking.trainer import (  # noqa: E402
    RankingTrainerConfig,
    train_and_evaluate_reranker,
)
from src.retrieval.dataset import RetrievalDatasetConfig  # noqa: E402
from src.retrieval.model import TfidfRetrievalConfig  # noqa: E402


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the two-stage ranking experiment.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the two-stage ranking pipeline."
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
    # Retrieval training dataset
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--retrieval-train-max-impressions",
        type=int,
        default=500,
        help=(
            "Maximum number of train impressions used to fit the retrieval baseline."
        ),
    )

    # --------------------------------------------------------------------------
    # Ranking train/dev datasets
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--ranking-train-max-impressions",
        type=int,
        default=500,
        help=(
            "Maximum number of train impressions used to fit the reranker."
        ),
    )
    parser.add_argument(
        "--ranking-dev-max-impressions",
        type=int,
        default=120,
        help=(
            "Maximum number of dev impressions used to evaluate the reranker."
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
    # TF-IDF retrieval model parameters
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
        help="Minimum n-gram size for the TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for the TF-IDF vectorizer.",
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
    # Logistic reranker parameters
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help=(
            "Inverse regularization strength for logistic regression. "
            "Smaller = stronger regularization."
        ),
    )
    parser.add_argument(
        "--logreg-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression convergence.",
    )
    parser.add_argument(
        "--logreg-class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help=(
            "Class weighting strategy for logistic regression. "
            "Use 'none' to disable class weighting."
        ),
    )
    parser.add_argument(
        "--logreg-solver",
        type=str,
        default="liblinear",
        choices=["liblinear", "lbfgs"],
        help="Solver for logistic regression.",
    )

    # --------------------------------------------------------------------------
    # Output / persistence options
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save the fitted retrieval and reranker model artifacts.",
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
        "retrieval_train_max_impressions": args.retrieval_train_max_impressions,
        "ranking_train_max_impressions": args.ranking_train_max_impressions,
        "ranking_dev_max_impressions": args.ranking_dev_max_impressions,
        "negatives_per_positive": args.negatives_per_positive,
        "max_history_items": args.max_history_items,
        "random_state": args.random_state,
        "max_features": args.max_features,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
        "logreg_max_iter": args.logreg_max_iter,
        "max_scored_dev_rows_to_save": args.max_scored_dev_rows_to_save,
    }

    for field_name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(
                f"--{field_name.replace('_', '-')} must be > 0. Got: {value}"
            )

    if args.ngram_min > args.ngram_max:
        raise ValueError(
            f"--ngram-min ({args.ngram_min}) cannot be greater than "
            f"--ngram-max ({args.ngram_max})."
        )

    if args.logreg_c <= 0:
        raise ValueError(f"--logreg-c must be > 0. Got: {args.logreg_c}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, build the ranking trainer config, run the experiment, and
    print the final result summary.
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    class_weight = None if args.logreg_class_weight == "none" else args.logreg_class_weight

    trainer_config = RankingTrainerConfig(
        run_name=args.run_name,
        retrieval_train_dataset=RetrievalDatasetConfig(
            split="train",
            max_impressions=args.retrieval_train_max_impressions,
            negatives_per_positive=args.negatives_per_positive,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
        ),
        ranking_train_dataset=RankingDatasetConfig(
            split="train",
            max_impressions=args.ranking_train_max_impressions,
            negatives_per_positive=args.negatives_per_positive,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
            include_retrieval_score=True,
        ),
        ranking_dev_dataset=RankingDatasetConfig(
            split="dev",
            max_impressions=args.ranking_dev_max_impressions,
            negatives_per_positive=args.negatives_per_positive,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
            include_retrieval_score=True,
        ),
        retrieval_model=TfidfRetrievalConfig(
            max_features=args.max_features,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            lowercase=not args.no_lowercase,
        ),
        reranker_model=LogisticRerankerConfig(
            C=args.logreg_c,
            max_iter=args.logreg_max_iter,
            class_weight=class_weight,
            solver=args.logreg_solver,
            random_state=args.random_state,
        ),
        metrics_ks=(5, 10, 20),
        save_models=not args.no_save_models,
        max_scored_dev_rows_to_save=args.max_scored_dev_rows_to_save,
    )

    print("Launching two-stage ranking experiment with config:\n")
    print(
        json.dumps(
            {
                "run_name": trainer_config.run_name,
                "retrieval_train_dataset": trainer_config.retrieval_train_dataset.__dict__,
                "ranking_train_dataset": trainer_config.ranking_train_dataset.__dict__,
                "ranking_dev_dataset": trainer_config.ranking_dev_dataset.__dict__,
                "retrieval_model": trainer_config.retrieval_model.__dict__,
                "reranker_model": trainer_config.reranker_model.__dict__,
                "metrics_ks": trainer_config.metrics_ks,
                "save_models": trainer_config.save_models,
                "max_scored_dev_rows_to_save": trainer_config.max_scored_dev_rows_to_save,
            },
            indent=2,
        )
    )

    result = train_and_evaluate_reranker(trainer_config)

    print("\nExperiment finished successfully.")
    print("\nFinal run summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()