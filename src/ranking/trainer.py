"""
src/ranking/trainer.py

Purpose
-------
Train, evaluate, and persist the second-stage ranking baseline experiment.

Why this file matters
---------------------
We now have the full pieces of a two-stage ranking system:

Stage 1:
- retrieval dataset builder
- TF-IDF retrieval baseline

Stage 2:
- ranking feature builder
- logistic regression reranker

This trainer turns those pieces into a real experiment pipeline that:
- fits the retrieval model
- builds ranking features
- fits the reranker
- compares retrieval-only vs reranked performance
- saves reproducible artifacts

Saved outputs
-------------
By default, this trainer writes into:

artifacts/
├── logs/
│   └── ranking/
│       └── <run_name>/
│           ├── config.json
│           ├── ranking_train_summary.json
│           ├── ranking_dev_summary.json
│           ├── comparison.json
│           ├── feature_importance.csv
│           ├── retrieval_scored_dev_sample.csv
│           ├── reranker_scored_dev_sample.csv
│           └── reranked_preview.csv
└── models/
    └── ranking/
        └── <run_name>/
            ├── retrieval_baseline.pkl
            └── logistic_reranker.pkl

Run local smoke test
--------------------
From the repo root:
    python3 -m src.ranking.trainer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path

import pandas as pd

from src.ranking.dataset import (
    RankingDatasetConfig,
    build_ranking_examples,
    summarize_ranking_examples,
)
from src.ranking.model import (
    LogisticReranker,
    LogisticRerankerConfig,
    compare_retrieval_vs_reranker,
)
from src.retrieval.dataset import (
    RetrievalDatasetConfig,
    build_retrieval_examples,
)
from src.retrieval.model import (
    TfidfRetrievalBaseline,
    TfidfRetrievalConfig,
)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class RankingTrainerConfig:
    """
    Configuration for a full two-stage ranking experiment.

    Attributes
    ----------
    run_name : str | None
        Optional explicit run name. If omitted, a timestamp-based name is used.
    retrieval_train_dataset : RetrievalDatasetConfig
        Dataset config for fitting the retrieval baseline.
    ranking_train_dataset : RankingDatasetConfig
        Ranking feature config for reranker training.
    ranking_dev_dataset : RankingDatasetConfig
        Ranking feature config for reranker evaluation.
    retrieval_model : TfidfRetrievalConfig
        TF-IDF retrieval model config.
    reranker_model : LogisticRerankerConfig
        Logistic reranker config.
    metrics_ks : tuple[int, ...]
        Ranking cutoffs to evaluate.
    save_models : bool
        Whether to save fitted retrieval + reranker models.
    max_scored_dev_rows_to_save : int
        How many scored dev rows to save to CSV for inspection.
    """

    run_name: str | None
    retrieval_train_dataset: RetrievalDatasetConfig
    ranking_train_dataset: RankingDatasetConfig
    ranking_dev_dataset: RankingDatasetConfig
    retrieval_model: TfidfRetrievalConfig
    reranker_model: LogisticRerankerConfig
    metrics_ks: tuple[int, ...] = (5, 10, 20)
    save_models: bool = True
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

    Example
    -------
    logistic_reranker_20260417_113015
    """
    if explicit_name:
        return explicit_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logistic_reranker_{timestamp}"


def get_run_dirs(run_name: str) -> tuple[Path, Path]:
    """
    Return:
    - logs directory for the run
    - models directory for the run
    """
    artifacts_root = get_artifacts_root()

    logs_dir = artifacts_root / "logs" / "ranking" / run_name
    models_dir = artifacts_root / "models" / "ranking" / run_name

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir, models_dir


# ------------------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------------------

def _json_default_serializer(obj):
    """
    Small serializer for json.dump.
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
# Preview helper
# ------------------------------------------------------------------------------

def build_reranked_preview(
    scored_df: pd.DataFrame,
    top_n_per_impression: int = 3,
    max_impressions: int = 10,
) -> pd.DataFrame:
    """
    Build a compact preview of reranked results.

    Notes
    -----
    The ranking feature table does not carry raw titles at this stage, so this
    preview focuses on:
    - impression_id
    - user_id
    - candidate_news_id
    - clicked
    - score
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
        "retrieval_score",
        "history_length",
        "impression_size",
    ]
    available_columns = [col for col in preview_columns if col in preview_df.columns]

    return preview_df[available_columns]


# ------------------------------------------------------------------------------
# Core experiment pipeline
# ------------------------------------------------------------------------------

def train_and_evaluate_reranker(
    config: RankingTrainerConfig,
) -> dict:
    """
    Run the full two-stage ranking experiment.

    Steps
    -----
    1. Build retrieval training examples
    2. Fit TF-IDF retrieval baseline
    3. Build ranking train/dev features using retrieval model scores
    4. Fit logistic reranker
    5. Score dev with retrieval-only and reranker
    6. Compare metrics
    7. Save artifacts and optional models
    """
    run_name = get_run_name(config.run_name)
    logs_dir, models_dir = get_run_dirs(run_name)

    print(f"Starting ranking training run: {run_name}")
    print(f"Logs directory:   {logs_dir}")
    print(f"Models directory: {models_dir}")

    # --------------------------------------------------------------------------
    # Save config
    # --------------------------------------------------------------------------
    config_payload = {
        "run_name": run_name,
        "retrieval_train_dataset": asdict(config.retrieval_train_dataset),
        "ranking_train_dataset": asdict(config.ranking_train_dataset),
        "ranking_dev_dataset": asdict(config.ranking_dev_dataset),
        "retrieval_model": asdict(config.retrieval_model),
        "reranker_model": asdict(config.reranker_model),
        "metrics_ks": list(config.metrics_ks),
        "save_models": config.save_models,
        "max_scored_dev_rows_to_save": config.max_scored_dev_rows_to_save,
    }
    save_json(config_payload, logs_dir / "config.json")

    # --------------------------------------------------------------------------
    # Step 1: fit retrieval baseline
    # --------------------------------------------------------------------------
    print("\nBuilding retrieval training examples...")
    retrieval_train_examples = build_retrieval_examples(config.retrieval_train_dataset)

    print("Fitting TF-IDF retrieval baseline...")
    retrieval_model = TfidfRetrievalBaseline(config=config.retrieval_model)
    retrieval_model.fit(retrieval_train_examples)

    # --------------------------------------------------------------------------
    # Step 2: build ranking train/dev features
    # --------------------------------------------------------------------------
    print("\nBuilding ranking train features...")
    ranking_train_df = build_ranking_examples(
        config=config.ranking_train_dataset,
        retrieval_model=retrieval_model,
    )
    ranking_train_summary = summarize_ranking_examples(ranking_train_df)
    save_json(ranking_train_summary, logs_dir / "ranking_train_summary.json")

    print("Ranking train summary:")
    for key, value in ranking_train_summary.items():
        print(f"  {key}: {value}")

    print("\nBuilding ranking dev features...")
    ranking_dev_df = build_ranking_examples(
        config=config.ranking_dev_dataset,
        retrieval_model=retrieval_model,
    )
    ranking_dev_summary = summarize_ranking_examples(ranking_dev_df)
    save_json(ranking_dev_summary, logs_dir / "ranking_dev_summary.json")

    print("Ranking dev summary:")
    for key, value in ranking_dev_summary.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # Step 3: fit reranker
    # --------------------------------------------------------------------------
    print("\nFitting logistic reranker...")
    reranker = LogisticReranker(config=config.reranker_model)
    reranker.fit(ranking_train_df)

    # --------------------------------------------------------------------------
    # Step 4: score dev
    # --------------------------------------------------------------------------
    print("Scoring dev set with reranker...")
    reranker_scored_dev = reranker.score_examples(ranking_dev_df)

    if "retrieval_score" not in ranking_dev_df.columns:
        raise ValueError(
            "Expected 'retrieval_score' in ranking dev features for comparison."
        )

    retrieval_scored_dev = ranking_dev_df.copy()
    retrieval_scored_dev["score"] = retrieval_scored_dev["retrieval_score"].astype(float)

    # --------------------------------------------------------------------------
    # Step 5: compare retrieval vs reranker
    # --------------------------------------------------------------------------
    print("Comparing retrieval vs reranker...")
    comparison = compare_retrieval_vs_reranker(
        retrieval_scored_df=retrieval_scored_dev,
        reranker_scored_df=reranker_scored_dev,
        ks=config.metrics_ks,
    )
    save_json(comparison, logs_dir / "comparison.json")

    print("\nRetrieval-only metrics:")
    for key, value in comparison["retrieval"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\nReranker metrics:")
    for key, value in comparison["reranker"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\nAbsolute improvement (reranker - retrieval):")
    for key, value in comparison["absolute_improvement"].items():
        print(f"  {key}: {value:+.6f}")

    # --------------------------------------------------------------------------
    # Step 6: save feature importance + samples
    # --------------------------------------------------------------------------
    importance_df = reranker.get_feature_importance_frame()
    importance_path = logs_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nSaved feature importance to: {importance_path}")

    retrieval_sample_path = logs_dir / "retrieval_scored_dev_sample.csv"
    retrieval_scored_dev.head(config.max_scored_dev_rows_to_save).to_csv(
        retrieval_sample_path,
        index=False,
    )
    print(f"Saved retrieval-scored dev sample to: {retrieval_sample_path}")

    reranker_sample_path = logs_dir / "reranker_scored_dev_sample.csv"
    reranker_scored_dev.head(config.max_scored_dev_rows_to_save).to_csv(
        reranker_sample_path,
        index=False,
    )
    print(f"Saved reranker-scored dev sample to: {reranker_sample_path}")

    preview_df = build_reranked_preview(
        scored_df=reranker_scored_dev,
        top_n_per_impression=3,
        max_impressions=10,
    )
    preview_path = logs_dir / "reranked_preview.csv"
    preview_df.to_csv(preview_path, index=False)
    print(f"Saved reranked preview to: {preview_path}")

    # --------------------------------------------------------------------------
    # Step 7: save models
    # --------------------------------------------------------------------------
    retrieval_model_path = None
    reranker_model_path = None

    if config.save_models:
        retrieval_model_path = models_dir / "retrieval_baseline.pkl"
        reranker_model_path = models_dir / "logistic_reranker.pkl"

        save_pickle(retrieval_model, retrieval_model_path)
        save_pickle(reranker, reranker_model_path)

        print(f"Saved retrieval model to: {retrieval_model_path}")
        print(f"Saved reranker model to: {reranker_model_path}")

    # --------------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------------
    result = {
        "run_name": run_name,
        "logs_dir": str(logs_dir),
        "models_dir": str(models_dir),
        "comparison": comparison,
        "ranking_train_summary": ranking_train_summary,
        "ranking_dev_summary": ranking_dev_summary,
        "feature_importance_path": str(importance_path),
        "retrieval_scored_dev_sample_path": str(retrieval_sample_path),
        "reranker_scored_dev_sample_path": str(reranker_sample_path),
        "preview_path": str(preview_path),
        "retrieval_model_path": str(retrieval_model_path) if retrieval_model_path else None,
        "reranker_model_path": str(reranker_model_path) if reranker_model_path else None,
    }

    return result


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    Uses modest sampled datasets so it runs in a reasonable amount of time while
    still producing saved experiment artifacts.
    """
    config = RankingTrainerConfig(
        run_name=None,
        retrieval_train_dataset=RetrievalDatasetConfig(
            split="train",
            max_impressions=500,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        ),
        ranking_train_dataset=RankingDatasetConfig(
            split="train",
            max_impressions=500,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
            include_retrieval_score=True,
        ),
        ranking_dev_dataset=RankingDatasetConfig(
            split="dev",
            max_impressions=120,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
            include_retrieval_score=True,
        ),
        retrieval_model=TfidfRetrievalConfig(
            max_features=30000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            lowercase=True,
        ),
        reranker_model=LogisticRerankerConfig(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        metrics_ks=(5, 10, 20),
        save_models=True,
        max_scored_dev_rows_to_save=500,
    )

    result = train_and_evaluate_reranker(config)

    print("\nRun complete.")
    print("Summary:")
    print(json.dumps(result, indent=2, default=_json_default_serializer))