"""
scripts/compare_retrievers.py

Purpose
-------
Train and compare two retrieval approaches on the same sampled dataset:

1. TF-IDF lexical retrieval baseline
2. Neural two-tower retrieval model

Why this file matters
---------------------
Now that both retrieval pipelines exist, we want a clean and reproducible way to
compare them side by side on the same train/dev setup.

This script:
- uses the same sampled train/dev impression sets for both methods
- trains both retrievers
- evaluates both with MRR / Recall / NDCG
- prints the comparison
- saves logs and artifacts

This is the final "retrieval comparison" script that lets us close the project
with a proper baseline-vs-neural result.

Default usage
-------------
    python3 scripts/compare_retrievers.py

Optional run-name override
--------------------------
    python3 scripts/compare_retrievers.py --run-name compare_exp_001
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader


# ------------------------------------------------------------------------------
# Make repo root importable
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.retrieval.dataset import (  # noqa: E402
    RetrievalDatasetConfig,
    build_retrieval_examples,
    summarize_retrieval_examples,
)
from src.retrieval.model import (  # noqa: E402
    TfidfRetrievalBaseline,
    TfidfRetrievalConfig,
)
from src.retrieval.two_tower_dataset import (  # noqa: E402
    TextEncodingConfig,
    TwoTowerTextDataset,
    VocabularyConfig,
    build_two_tower_dataset_bundle,
    collate_two_tower_batch,
)
from src.retrieval.two_tower_model import (  # noqa: E402
    TwoTowerModelConfig,
    TwoTowerRetrievalModel,
    TwoTowerTrainConfig,
    evaluate_two_tower_ranking,
    fit_two_tower_model,
    get_default_device,
    score_two_tower_dataframe,
)


# ------------------------------------------------------------------------------
# Config containers
# ------------------------------------------------------------------------------

@dataclass
class RetrieverComparisonConfig:
    """
    Full experiment config for comparing TF-IDF vs Two-Tower retrieval.
    """

    run_name: str | None
    train_dataset: RetrievalDatasetConfig
    dev_dataset: RetrievalDatasetConfig
    tfidf_model: TfidfRetrievalConfig
    vocab: VocabularyConfig
    encoding: TextEncodingConfig
    two_tower_model: TwoTowerModelConfig
    two_tower_train: TwoTowerTrainConfig
    save_models: bool = True
    max_scored_dev_rows_to_save: int = 500


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train and compare TF-IDF vs Two-Tower retrieval models."
    )

    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--train-max-impressions", type=int, default=500)
    parser.add_argument("--dev-max-impressions", type=int, default=120)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-history-items", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--tfidf-max-features", type=int, default=30000)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-min-df", type=int, default=2)

    parser.add_argument("--vocab-min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-user-tokens", type=int, default=64)
    parser.add_argument("--max-item-tokens", type=int, default=64)

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--logit-scale", type=float, default=10.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-class-balance", action="store_true")

    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--max-scored-dev-rows-to-save", type=int, default=500)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate CLI args before launching the experiment.
    """
    positive_int_fields = {
        "train_max_impressions": args.train_max_impressions,
        "dev_max_impressions": args.dev_max_impressions,
        "negatives_per_positive": args.negatives_per_positive,
        "max_history_items": args.max_history_items,
        "random_state": args.random_state,
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_ngram_min": args.tfidf_ngram_min,
        "tfidf_ngram_max": args.tfidf_ngram_max,
        "tfidf_min_df": args.tfidf_min_df,
        "vocab_min_freq": args.vocab_min_freq,
        "max_vocab_size": args.max_vocab_size,
        "max_user_tokens": args.max_user_tokens,
        "max_item_tokens": args.max_item_tokens,
        "embedding_dim": args.embedding_dim,
        "projection_dim": args.projection_dim,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "max_scored_dev_rows_to_save": args.max_scored_dev_rows_to_save,
    }

    for field_name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(
                f"--{field_name.replace('_', '-')} must be > 0. Got: {value}"
            )

    if args.tfidf_ngram_min > args.tfidf_ngram_max:
        raise ValueError(
            f"--tfidf-ngram-min ({args.tfidf_ngram_min}) cannot be greater than "
            f"--tfidf-ngram-max ({args.tfidf_ngram_max})."
        )

    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be > 0. Got: {args.learning_rate}")

    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be >= 0. Got: {args.weight_decay}")

    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError(f"--dropout must be in [0, 1). Got: {args.dropout}")

    if args.logit_scale <= 0:
        raise ValueError(f"--logit-scale must be > 0. Got: {args.logit_scale}")


# ------------------------------------------------------------------------------
# Artifact helpers
# ------------------------------------------------------------------------------

def get_artifacts_root() -> Path:
    """
    Return the artifacts root directory.
    """
    return REPO_ROOT / "artifacts"


def get_run_name(explicit_name: str | None = None) -> str:
    """
    Generate a run name if one is not provided.
    """
    if explicit_name:
        return explicit_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"retriever_compare_{timestamp}"


def get_run_dirs(run_name: str) -> tuple[Path, Path]:
    """
    Return:
    - log directory for the run
    - model directory for the run
    """
    artifacts_root = get_artifacts_root()

    logs_dir = artifacts_root / "logs" / "retrieval_compare" / run_name
    models_dir = artifacts_root / "models" / "retrieval_compare" / run_name

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir, models_dir


def _json_default_serializer(obj):
    """
    Small serializer helper for json.dump.
    """
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as pretty JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default_serializer)


def save_pickle(obj, path: Path) -> None:
    """
    Save a Python object using pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ------------------------------------------------------------------------------
# Comparison helpers
# ------------------------------------------------------------------------------

def build_comparison(
    tfidf_metrics: dict,
    two_tower_metrics: dict,
) -> dict:
    """
    Build a comparison dictionary for TF-IDF vs Two-Tower.
    """
    improvement: dict[str, float] = {}

    for key, two_tower_value in two_tower_metrics.items():
        if key in {"num_impressions", "num_rows"}:
            continue

        tfidf_value = float(tfidf_metrics[key])
        improvement[key] = float(two_tower_value) - tfidf_value

    return {
        "tfidf": tfidf_metrics,
        "two_tower": two_tower_metrics,
        "absolute_improvement": improvement,
    }


def build_top_scored_preview(
    scored_df: pd.DataFrame,
    top_n_per_impression: int = 3,
    max_impressions: int = 10,
) -> pd.DataFrame:
    """
    Return a compact top-scored preview from a scored dev DataFrame.
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
        "candidate_news_id",
        "clicked",
        "score",
    ]
    available_columns = [col for col in preview_columns if col in preview_df.columns]
    return preview_df[available_columns]


# ------------------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------------------

def compare_retrievers(config: RetrieverComparisonConfig) -> dict:
    """
    Train TF-IDF and Two-Tower on the same sampled train/dev setup and compare them.
    """
    run_name = get_run_name(config.run_name)
    logs_dir, models_dir = get_run_dirs(run_name)

    print(f"Starting retriever comparison run: {run_name}")
    print(f"Logs directory:   {logs_dir}")
    print(f"Models directory: {models_dir}")

    # Save config snapshot
    config_payload = {
        "run_name": run_name,
        "train_dataset": asdict(config.train_dataset),
        "dev_dataset": asdict(config.dev_dataset),
        "tfidf_model": asdict(config.tfidf_model),
        "vocab": asdict(config.vocab),
        "encoding": asdict(config.encoding),
        "two_tower_model": asdict(config.two_tower_model),
        "two_tower_train": asdict(config.two_tower_train),
        "save_models": config.save_models,
        "max_scored_dev_rows_to_save": config.max_scored_dev_rows_to_save,
    }
    save_json(config_payload, logs_dir / "config.json")

    # --------------------------------------------------------------------------
    # Build shared retrieval examples for TF-IDF
    # --------------------------------------------------------------------------
    print("\nBuilding retrieval train/dev examples...")
    train_examples = build_retrieval_examples(config.train_dataset)
    dev_examples = build_retrieval_examples(config.dev_dataset)

    train_summary = summarize_retrieval_examples(train_examples)
    dev_summary = summarize_retrieval_examples(dev_examples)

    save_json(train_summary, logs_dir / "train_summary.json")
    save_json(dev_summary, logs_dir / "dev_summary.json")

    print("Train summary:")
    for key, value in train_summary.items():
        print(f"  {key}: {value}")

    print("\nDev summary:")
    for key, value in dev_summary.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # TF-IDF baseline
    # --------------------------------------------------------------------------
    print("\nFitting TF-IDF retriever...")
    tfidf_model = TfidfRetrievalBaseline(config=config.tfidf_model)
    tfidf_model.fit(train_examples)

    print("Scoring dev with TF-IDF...")
    tfidf_scored_dev = tfidf_model.score_examples(dev_examples)
    tfidf_metrics = tfidf_model.evaluate(dev_examples, ks=config.two_tower_train.eval_ks)

    print("TF-IDF metrics:")
    for key, value in tfidf_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    save_json(tfidf_metrics, logs_dir / "tfidf_metrics.json")
    tfidf_scored_dev.head(config.max_scored_dev_rows_to_save).to_csv(
        logs_dir / "tfidf_scored_dev_sample.csv",
        index=False,
    )
    build_top_scored_preview(tfidf_scored_dev).to_csv(
        logs_dir / "tfidf_preview.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Two-Tower bundle
    # --------------------------------------------------------------------------
    print("\nBuilding Two-Tower datasets...")
    bundle = build_two_tower_dataset_bundle(
        train_config=config.train_dataset,
        dev_config=config.dev_dataset,
        vocab_config=config.vocab,
        encoding_config=config.encoding,
    )

    two_tower_train_summary = summarize_retrieval_examples(bundle.train_text_df.rename(columns={"user_text": "dummy"}))
    # Keep a clean summary file for the encoded/text bundle too.
    bundle_summary = {
        "train_rows": int(len(bundle.train_encoded_df)),
        "dev_rows": int(len(bundle.dev_encoded_df)),
        "vocab_size": int(len(bundle.vocab)),
    }
    save_json(bundle_summary, logs_dir / "two_tower_bundle_summary.json")

    print(f"Two-Tower vocab size: {len(bundle.vocab):,}")

    train_dataset = TwoTowerTextDataset(bundle.train_encoded_df)
    dev_dataset = TwoTowerTextDataset(bundle.dev_encoded_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.two_tower_train.batch_size,
        shuffle=True,
        collate_fn=collate_two_tower_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.two_tower_train.batch_size,
        shuffle=False,
        collate_fn=collate_two_tower_batch,
    )

    # --------------------------------------------------------------------------
    # Two-Tower model
    # --------------------------------------------------------------------------
    print("\nFitting Two-Tower retriever...")
    two_tower_model = TwoTowerRetrievalModel(
        vocab_size=len(bundle.vocab),
        config=config.two_tower_model,
    )

    history = fit_two_tower_model(
        model=two_tower_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_labels=bundle.train_encoded_df["clicked"],
        config=config.two_tower_train,
    )
    save_json(history, logs_dir / "two_tower_history.json")

    print("\nScoring dev with Two-Tower...")
    two_tower_scored_dev = score_two_tower_dataframe(
        model=two_tower_model,
        encoded_df=bundle.dev_encoded_df,
        batch_size=256,
        device=config.two_tower_train.device,
    )
    two_tower_metrics = evaluate_two_tower_ranking(
        scored_df=two_tower_scored_dev,
        ks=config.two_tower_train.eval_ks,
    )

    print("Two-Tower metrics:")
    for key, value in two_tower_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    save_json(two_tower_metrics, logs_dir / "two_tower_metrics.json")
    two_tower_scored_dev.head(config.max_scored_dev_rows_to_save).to_csv(
        logs_dir / "two_tower_scored_dev_sample.csv",
        index=False,
    )
    build_top_scored_preview(two_tower_scored_dev).to_csv(
        logs_dir / "two_tower_preview.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Comparison
    # --------------------------------------------------------------------------
    comparison = build_comparison(
        tfidf_metrics=tfidf_metrics,
        two_tower_metrics=two_tower_metrics,
    )
    save_json(comparison, logs_dir / "comparison.json")

    print("\nAbsolute improvement (Two-Tower - TF-IDF):")
    for key, value in comparison["absolute_improvement"].items():
        print(f"  {key}: {value:+.6f}")

    # --------------------------------------------------------------------------
    # Save models
    # --------------------------------------------------------------------------
    tfidf_model_path = None
    two_tower_model_path = None

    if config.save_models:
        tfidf_model_path = models_dir / "tfidf_retriever.pkl"
        two_tower_model_path = models_dir / "two_tower_retriever.pt"

        save_pickle(tfidf_model, tfidf_model_path)
        torch.save(
            {
                "model_state_dict": two_tower_model.state_dict(),
                "model_config": asdict(config.two_tower_model),
                "train_config": asdict(config.two_tower_train),
                "vocab_config": asdict(config.vocab),
                "encoding_config": asdict(config.encoding),
                "vocab_token_to_idx": bundle.vocab.token_to_idx,
                "vocab_idx_to_token": bundle.vocab.idx_to_token,
            },
            two_tower_model_path,
        )

        print(f"\nSaved TF-IDF model to: {tfidf_model_path}")
        print(f"Saved Two-Tower model to: {two_tower_model_path}")

    result = {
        "run_name": run_name,
        "logs_dir": str(logs_dir),
        "models_dir": str(models_dir),
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "two_tower_bundle_summary": bundle_summary,
        "two_tower_history": history,
        "comparison": comparison,
        "tfidf_model_path": str(tfidf_model_path) if tfidf_model_path else None,
        "two_tower_model_path": str(two_tower_model_path) if two_tower_model_path else None,
    }

    return result


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Parse args, build config, run comparison, print final summary.
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    config = RetrieverComparisonConfig(
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
        tfidf_model=TfidfRetrievalConfig(
            max_features=args.tfidf_max_features,
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            min_df=args.tfidf_min_df,
            lowercase=True,
        ),
        vocab=VocabularyConfig(
            min_freq=args.vocab_min_freq,
            max_vocab_size=args.max_vocab_size,
        ),
        encoding=TextEncodingConfig(
            max_user_tokens=args.max_user_tokens,
            max_item_tokens=args.max_item_tokens,
        ),
        two_tower_model=TwoTowerModelConfig(
            embedding_dim=args.embedding_dim,
            projection_dim=args.projection_dim,
            dropout=args.dropout,
            logit_scale=args.logit_scale,
        ),
        two_tower_train=TwoTowerTrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            device=args.device,
            use_class_balance=not args.no_class_balance,
            eval_ks=(5, 10, 20),
        ),
        save_models=not args.no_save_models,
        max_scored_dev_rows_to_save=args.max_scored_dev_rows_to_save,
    )

    print("Launching retriever comparison with config:\n")
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "train_dataset": asdict(config.train_dataset),
                "dev_dataset": asdict(config.dev_dataset),
                "tfidf_model": asdict(config.tfidf_model),
                "vocab": asdict(config.vocab),
                "encoding": asdict(config.encoding),
                "two_tower_model": asdict(config.two_tower_model),
                "two_tower_train": asdict(config.two_tower_train),
                "save_models": config.save_models,
                "max_scored_dev_rows_to_save": config.max_scored_dev_rows_to_save,
            },
            indent=2,
        )
    )

    result = compare_retrievers(config)

    print("\nComparison finished successfully.")
    print("\nFinal summary:")
    print(json.dumps(result, indent=2, default=_json_default_serializer))


if __name__ == "__main__":
    main()