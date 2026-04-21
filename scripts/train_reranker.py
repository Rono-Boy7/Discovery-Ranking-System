"""
scripts/train_reranker.py

Purpose
-------
Command-line entry point for training and evaluating the full two-stage ranking
pipeline using a config-driven workflow.

Stage 1:
- TF-IDF retrieval baseline

Stage 2:
- Logistic regression reranker

Why this file matters
---------------------
Previously, this script built the experiment config directly from CLI arguments.
That worked, but now we want a more realistic ML workflow where experiments are
driven by config files.

This version:
- loads a YAML or JSON config
- validates required sections
- builds the experiment objects from config
- supports a few CLI overrides for convenience

Default usage
-------------
    python3 scripts/train_reranker.py

Explicit config usage
---------------------
    python3 scripts/train_reranker.py --config configs/train.yaml

Optional run-name override
--------------------------
    python3 scripts/train_reranker.py \
        --config configs/train.yaml \
        --run-name reranker_exp_001

Notes
-----
YAML configs require PyYAML.

Install it with:
    python3 -m pip install pyyaml
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
from src.utils.config import (  # noqa: E402
    ConfigError,
    get_value,
    load_config,
    require_nested_keys,
    require_top_level_keys,
)


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the two-stage ranking experiment.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the two-stage ranking pipeline from config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help=(
            "Path to the experiment config file. "
            "Supports .yaml, .yml, and .json."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit run name override. If omitted, the trainer will "
            "use its default timestamp-based naming."
        ),
    )

    return parser


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_experiment_config(config: dict) -> None:
    """
    Validate that the config contains the required sections and keys.

    We keep validation practical and focused on the fields required to build the
    current two-stage baseline.
    """
    require_top_level_keys(config, ["experiment", "retrieval", "ranking"])

    # experiment section
    require_nested_keys(config, ["experiment", "random_state"])
    require_nested_keys(config, ["experiment", "metrics_ks"])
    require_nested_keys(config, ["experiment", "save_models"])
    require_nested_keys(config, ["experiment", "max_scored_dev_rows_to_save"])

    # retrieval section
    require_nested_keys(config, ["retrieval", "train_dataset", "split"])
    require_nested_keys(config, ["retrieval", "train_dataset", "max_impressions"])
    require_nested_keys(config, ["retrieval", "train_dataset", "negatives_per_positive"])
    require_nested_keys(config, ["retrieval", "train_dataset", "max_history_items"])
    require_nested_keys(config, ["retrieval", "train_dataset", "random_state"])

    require_nested_keys(config, ["retrieval", "model", "max_features"])
    require_nested_keys(config, ["retrieval", "model", "ngram_min"])
    require_nested_keys(config, ["retrieval", "model", "ngram_max"])
    require_nested_keys(config, ["retrieval", "model", "min_df"])
    require_nested_keys(config, ["retrieval", "model", "lowercase"])

    # ranking section
    require_nested_keys(config, ["ranking", "train_dataset", "split"])
    require_nested_keys(config, ["ranking", "train_dataset", "max_impressions"])
    require_nested_keys(config, ["ranking", "train_dataset", "negatives_per_positive"])
    require_nested_keys(config, ["ranking", "train_dataset", "max_history_items"])
    require_nested_keys(config, ["ranking", "train_dataset", "random_state"])
    require_nested_keys(config, ["ranking", "train_dataset", "include_retrieval_score"])

    require_nested_keys(config, ["ranking", "dev_dataset", "split"])
    require_nested_keys(config, ["ranking", "dev_dataset", "max_impressions"])
    require_nested_keys(config, ["ranking", "dev_dataset", "negatives_per_positive"])
    require_nested_keys(config, ["ranking", "dev_dataset", "max_history_items"])
    require_nested_keys(config, ["ranking", "dev_dataset", "random_state"])
    require_nested_keys(config, ["ranking", "dev_dataset", "include_retrieval_score"])

    require_nested_keys(config, ["ranking", "model", "C"])
    require_nested_keys(config, ["ranking", "model", "max_iter"])
    require_nested_keys(config, ["ranking", "model", "class_weight"])
    require_nested_keys(config, ["ranking", "model", "solver"])
    require_nested_keys(config, ["ranking", "model", "random_state"])

    # Practical value checks
    metrics_ks = get_value(config, ["experiment", "metrics_ks"])
    if not isinstance(metrics_ks, list) or not metrics_ks:
        raise ConfigError("experiment.metrics_ks must be a non-empty list of integers.")

    if any((not isinstance(k, int) or k <= 0) for k in metrics_ks):
        raise ConfigError("experiment.metrics_ks must contain only positive integers.")

    for key_path in [
        ["retrieval", "train_dataset", "max_impressions"],
        ["retrieval", "train_dataset", "negatives_per_positive"],
        ["retrieval", "train_dataset", "max_history_items"],
        ["retrieval", "train_dataset", "random_state"],
        ["retrieval", "model", "max_features"],
        ["retrieval", "model", "ngram_min"],
        ["retrieval", "model", "ngram_max"],
        ["retrieval", "model", "min_df"],
        ["ranking", "train_dataset", "max_impressions"],
        ["ranking", "train_dataset", "negatives_per_positive"],
        ["ranking", "train_dataset", "max_history_items"],
        ["ranking", "train_dataset", "random_state"],
        ["ranking", "dev_dataset", "max_impressions"],
        ["ranking", "dev_dataset", "negatives_per_positive"],
        ["ranking", "dev_dataset", "max_history_items"],
        ["ranking", "dev_dataset", "random_state"],
        ["ranking", "model", "max_iter"],
        ["ranking", "model", "random_state"],
        ["experiment", "max_scored_dev_rows_to_save"],
    ]:
        value = get_value(config, key_path)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"{'.'.join(key_path)} must be a positive integer. Got: {value}"
            )

    ngram_min = get_value(config, ["retrieval", "model", "ngram_min"])
    ngram_max = get_value(config, ["retrieval", "model", "ngram_max"])
    if ngram_min > ngram_max:
        raise ConfigError(
            "retrieval.model.ngram_min cannot be greater than retrieval.model.ngram_max."
        )

    logreg_c = get_value(config, ["ranking", "model", "C"])
    if not isinstance(logreg_c, (int, float)) or logreg_c <= 0:
        raise ConfigError(
            f"ranking.model.C must be > 0. Got: {logreg_c}"
        )

    save_models = get_value(config, ["experiment", "save_models"])
    if not isinstance(save_models, bool):
        raise ConfigError("experiment.save_models must be a boolean.")

    lowercase = get_value(config, ["retrieval", "model", "lowercase"])
    if not isinstance(lowercase, bool):
        raise ConfigError("retrieval.model.lowercase must be a boolean.")

    for key_path in [
        ["ranking", "train_dataset", "include_retrieval_score"],
        ["ranking", "dev_dataset", "include_retrieval_score"],
    ]:
        value = get_value(config, key_path)
        if not isinstance(value, bool):
            raise ConfigError(f"{'.'.join(key_path)} must be a boolean.")

    class_weight = get_value(config, ["ranking", "model", "class_weight"])
    if class_weight not in {"balanced", None, "none"}:
        raise ConfigError(
            "ranking.model.class_weight must be one of: 'balanced', 'none', null"
        )

    solver = get_value(config, ["ranking", "model", "solver"])
    if solver not in {"liblinear", "lbfgs"}:
        raise ConfigError(
            "ranking.model.solver must be one of: 'liblinear', 'lbfgs'"
        )


# ------------------------------------------------------------------------------
# Config -> dataclass builders
# ------------------------------------------------------------------------------

def build_retrieval_dataset_config(config: dict) -> RetrievalDatasetConfig:
    """
    Build RetrievalDatasetConfig from the retrieval.train_dataset config section.
    """
    section = config["retrieval"]["train_dataset"]

    return RetrievalDatasetConfig(
        split=section["split"],
        max_impressions=section["max_impressions"],
        negatives_per_positive=section["negatives_per_positive"],
        max_history_items=section["max_history_items"],
        random_state=section["random_state"],
    )


def build_ranking_dataset_config(section: dict) -> RankingDatasetConfig:
    """
    Build RankingDatasetConfig from a ranking dataset config section.
    """
    return RankingDatasetConfig(
        split=section["split"],
        max_impressions=section["max_impressions"],
        negatives_per_positive=section["negatives_per_positive"],
        max_history_items=section["max_history_items"],
        random_state=section["random_state"],
        include_retrieval_score=section["include_retrieval_score"],
    )


def build_tfidf_config(config: dict) -> TfidfRetrievalConfig:
    """
    Build TfidfRetrievalConfig from the retrieval.model config section.
    """
    section = config["retrieval"]["model"]

    return TfidfRetrievalConfig(
        max_features=section["max_features"],
        ngram_min=section["ngram_min"],
        ngram_max=section["ngram_max"],
        min_df=section["min_df"],
        lowercase=section["lowercase"],
    )


def build_logistic_reranker_config(config: dict) -> LogisticRerankerConfig:
    """
    Build LogisticRerankerConfig from the ranking.model config section.
    """
    section = config["ranking"]["model"]

    class_weight = section["class_weight"]
    if class_weight == "none":
        class_weight = None

    return LogisticRerankerConfig(
        C=section["C"],
        max_iter=section["max_iter"],
        class_weight=class_weight,
        solver=section["solver"],
        random_state=section["random_state"],
    )


def build_trainer_config(config: dict, run_name_override: str | None = None) -> RankingTrainerConfig:
    """
    Convert a validated config dictionary into RankingTrainerConfig.
    """
    return RankingTrainerConfig(
        run_name=run_name_override,
        retrieval_train_dataset=build_retrieval_dataset_config(config),
        ranking_train_dataset=build_ranking_dataset_config(config["ranking"]["train_dataset"]),
        ranking_dev_dataset=build_ranking_dataset_config(config["ranking"]["dev_dataset"]),
        retrieval_model=build_tfidf_config(config),
        reranker_model=build_logistic_reranker_config(config),
        metrics_ks=tuple(config["experiment"]["metrics_ks"]),
        save_models=config["experiment"]["save_models"],
        max_scored_dev_rows_to_save=config["experiment"]["max_scored_dev_rows_to_save"],
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Load config, validate it, build trainer config, run the experiment, and
    print a final summary.
    """
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config, base_dir=REPO_ROOT)
    validate_experiment_config(config)

    trainer_config = build_trainer_config(
        config=config,
        run_name_override=args.run_name,
    )

    print("Launching two-stage ranking experiment with config:\n")
    print(json.dumps(config, indent=2))

    result = train_and_evaluate_reranker(trainer_config)

    print("\nExperiment finished successfully.")
    print("\nFinal run summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()