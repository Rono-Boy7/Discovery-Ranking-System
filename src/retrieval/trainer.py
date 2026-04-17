"""
src/retrieval/trainer.py

Purpose
-------
Train, evaluate, and persist the TF-IDF retrieval baseline experiment.

Why this file matters
---------------------
Up to now, we have:
- parsed raw data
- built normalized interim data
- created retrieval examples
- implemented a TF-IDF retrieval baseline
- evaluated it in a smoke test

This trainer module turns that into a real experiment pipeline that:
- builds datasets
- trains the baseline
- evaluates on dev
- saves metrics and preview outputs
- optionally saves the fitted model

This is important because strong ML repositories should not rely only on
terminal printouts. They should leave behind reproducible experiment artifacts.

What gets saved
---------------
By default, this trainer writes into:

artifacts/
├── logs/
│   └── retrieval/
│       └── <run_name>/
│           ├── config.json
│           ├── train_summary.json
│           ├── dev_summary.json
│           ├── metrics.json
│           ├── top_ranked_preview.csv
│           └── scored_dev_sample.csv
└── models/
    └── retrieval/
        └── <run_name>/
            └── tfidf_retrieval_baseline.pkl

Run local smoke test
--------------------
From the repo root:
    python3 -m src.retrieval.trainer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path

import pandas as pd

from src.retrieval.dataset import (
    RetrievalDatasetConfig,
    build_retrieval_examples,
    summarize_retrieval_examples,
)
from src.retrieval.model import (
    TfidfRetrievalBaseline,
    TfidfRetrievalConfig,
    preview_top_ranked_candidates,
)


# ------------------------------------------------------------------------------
# Config containers
# ------------------------------------------------------------------------------

@dataclass
class RetrievalTrainerConfig:
    """
    Configuration for a retrieval baseline training run.

    Attributes
    ----------
    run_name : str | None
        Optional explicit run name. If omitted, a timestamp-based name is used.
    train_dataset : RetrievalDatasetConfig
        Dataset-building config for the training split.
    dev_dataset : RetrievalDatasetConfig
        Dataset-building config for the dev split.
    model : TfidfRetrievalConfig
        TF-IDF model config.
    metrics_ks : tuple[int, ...]
        Ranking cutoffs to evaluate.
    save_model : bool
        Whether to serialize the trained model to artifacts/models/.
    max_scored_dev_rows_to_save : int
        Number of scored dev rows to save to CSV for later inspection.
    """

    run_name: str | None
    train_dataset: RetrievalDatasetConfig
    dev_dataset: RetrievalDatasetConfig
    model: TfidfRetrievalConfig
    metrics_ks: tuple[int, ...] = (5, 10, 20)
    save_model: bool = True
    max_scored_dev_rows_to_save: int = 500


# ------------------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------------------

def get_repo_root() -> Path:
    """
    Return the repository root.
    """
    return Path(__file__).resolve().parents[2]


def get_artifacts_root() -> Path:
    """
    Return the artifacts root directory.
    """
    return get_repo_root() / "artifacts"


def get_run_name(explicit_name: str | None = None) -> str:
    """
    Generate a run name if one was not provided.

    Example:
        tfidf_baseline_20260416_153022
    """
    if explicit_name:
        return explicit_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tfidf_baseline_{timestamp}"


def get_run_dirs(run_name: str) -> tuple[Path, Path]:
    """
    Return:
    - log directory for the run
    - model directory for the run
    """
    artifacts_root = get_artifacts_root()

    logs_dir = artifacts_root / "logs" / "retrieval" / run_name
    models_dir = artifacts_root / "models" / "retrieval" / run_name

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir, models_dir


# ------------------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------------------

def _json_default_serializer(obj):
    """
    JSON serializer for objects that json.dump cannot handle directly.

    We keep this small and practical:
    - Path -> str
    - tuples/lists/dataclasses are handled before this
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
    Save a Python object with pickle.

    For this project stage, pickle is sufficient for our sklearn baseline.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ------------------------------------------------------------------------------
# Core training pipeline
# ------------------------------------------------------------------------------

def train_and_evaluate_retrieval_baseline(
    config: RetrievalTrainerConfig,
) -> dict:
    """
    Run the full TF-IDF retrieval baseline experiment.

    Parameters
    ----------
    config : RetrievalTrainerConfig
        Full experiment configuration.

    Returns
    -------
    dict
        Summary of the run, including output locations and metrics.

    Steps
    -----
    1. Build train retrieval examples
    2. Build dev retrieval examples
    3. Fit the TF-IDF baseline on train
    4. Score dev examples
    5. Compute ranking metrics
    6. Save outputs and artifacts
    """
    run_name = get_run_name(config.run_name)
    logs_dir, models_dir = get_run_dirs(run_name)

    print(f"Starting retrieval baseline training run: {run_name}")
    print(f"Logs directory:   {logs_dir}")
    print(f"Models directory: {models_dir}")

    # --------------------------------------------------------------------------
    # Save run config
    # --------------------------------------------------------------------------
    config_payload = {
        "run_name": run_name,
        "train_dataset": asdict(config.train_dataset),
        "dev_dataset": asdict(config.dev_dataset),
        "model": asdict(config.model),
        "metrics_ks": list(config.metrics_ks),
        "save_model": config.save_model,
        "max_scored_dev_rows_to_save": config.max_scored_dev_rows_to_save,
    }
    save_json(config_payload, logs_dir / "config.json")

    # --------------------------------------------------------------------------
    # Build train examples
    # --------------------------------------------------------------------------
    print("\nBuilding training examples...")
    train_examples = build_retrieval_examples(config.train_dataset)
    train_summary = summarize_retrieval_examples(train_examples)
    save_json(train_summary, logs_dir / "train_summary.json")

    print("Training example summary:")
    for key, value in train_summary.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # Build dev examples
    # --------------------------------------------------------------------------
    print("\nBuilding dev examples...")
    dev_examples = build_retrieval_examples(config.dev_dataset)
    dev_summary = summarize_retrieval_examples(dev_examples)
    save_json(dev_summary, logs_dir / "dev_summary.json")

    print("Dev example summary:")
    for key, value in dev_summary.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # Fit model
    # --------------------------------------------------------------------------
    print("\nFitting TF-IDF retrieval baseline...")
    model = TfidfRetrievalBaseline(config=config.model)
    model.fit(train_examples)

    # --------------------------------------------------------------------------
    # Score and evaluate
    # --------------------------------------------------------------------------
    print("Scoring dev examples...")
    scored_dev = model.score_examples(dev_examples)

    print("Evaluating ranking metrics...")
    metrics = model.evaluate(dev_examples, ks=config.metrics_ks)
    save_json(metrics, logs_dir / "metrics.json")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # Save preview outputs
    # --------------------------------------------------------------------------
    preview_df = preview_top_ranked_candidates(
        scored_df=scored_dev,
        top_n_per_impression=3,
        max_impressions=10,
    )
    preview_path = logs_dir / "top_ranked_preview.csv"
    preview_df.to_csv(preview_path, index=False)
    print(f"\nSaved top-ranked preview to: {preview_path}")

    # Save a random-ish sample of scored dev rows for inspection.
    scored_dev_sample = scored_dev.head(config.max_scored_dev_rows_to_save).copy()
    scored_dev_sample_path = logs_dir / "scored_dev_sample.csv"
    scored_dev_sample.to_csv(scored_dev_sample_path, index=False)
    print(f"Saved scored dev sample to: {scored_dev_sample_path}")

    # --------------------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------------------
    model_path = None
    if config.save_model:
        model_path = models_dir / "tfidf_retrieval_baseline.pkl"
        save_pickle(model, model_path)
        print(f"Saved trained model to: {model_path}")

    # --------------------------------------------------------------------------
    # Return final run summary
    # --------------------------------------------------------------------------
    result = {
        "run_name": run_name,
        "logs_dir": str(logs_dir),
        "models_dir": str(models_dir),
        "metrics": metrics,
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "model_path": str(model_path) if model_path else None,
        "preview_path": str(preview_path),
        "scored_dev_sample_path": str(scored_dev_sample_path),
    }

    return result


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    This uses modest sampled datasets so it can run in a reasonable amount of
    time while still producing saved artifacts.
    """
    config = RetrievalTrainerConfig(
        run_name=None,
        train_dataset=RetrievalDatasetConfig(
            split="train",
            max_impressions=500,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        dev_dataset=RetrievalDatasetConfig(
            split="dev",
            max_impressions=120,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        model=TfidfRetrievalConfig(
            max_features=30000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            lowercase=True,
        ),
        metrics_ks=(5, 10, 20),
        save_model=True,
        max_scored_dev_rows_to_save=500,
    )

    result = train_and_evaluate_retrieval_baseline(config)

    print("\nRun complete.")
    print("Summary:")
    print(json.dumps(result, indent=2, default=_json_default_serializer))