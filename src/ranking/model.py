"""
src/ranking/model.py

Purpose
-------
Second-stage reranker baseline for the discovery ranking system.

Why this file matters
---------------------
In a two-stage ranking system, retrieval gives us a candidate set, and the
reranker uses richer features to refine the final ordering.

This module implements a simple but strong baseline reranker using:
- engineered ranking features from src.ranking.dataset
- StandardScaler
- LogisticRegression

Why logistic regression?
------------------------
It is:
- simple
- interpretable
- fast
- surprisingly strong as a baseline on engineered features

This is not the final reranker we could build, but it is exactly the kind of
baseline we want before moving to more advanced tree-based or neural rerankers.

Run local smoke test
--------------------
From the repo root:
    python3 -m src.ranking.model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_grouped_ranking
from src.ranking.dataset import (
    RankingDatasetConfig,
    build_ranking_examples,
    summarize_ranking_examples,
)
from src.retrieval.dataset import RetrievalDatasetConfig, build_retrieval_examples
from src.retrieval.model import (
    TfidfRetrievalBaseline,
    TfidfRetrievalConfig,
)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class LogisticRerankerConfig:
    """
    Configuration for the logistic regression reranker baseline.

    Attributes
    ----------
    C : float
        Inverse regularization strength for logistic regression.
        Smaller values = stronger regularization.
    max_iter : int
        Maximum number of iterations for solver convergence.
    class_weight : str | None
        Class weighting strategy. 'balanced' is useful because click labels are
        sparse and imbalanced in ranking datasets.
    solver : str
        Logistic regression solver.
    random_state : int
        Random seed for reproducibility.
    """

    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    solver: str = "liblinear"
    random_state: int = 42


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class LogisticReranker:
    """
    A simple second-stage reranker using engineered numeric features.

    Workflow
    --------
    1. Take the ranking feature table
    2. Select numeric feature columns
    3. Scale them
    4. Fit logistic regression
    5. Use predicted probability as the reranking score

    Notes
    -----
    We intentionally keep this baseline interpretable.
    That helps us:
    - understand which features matter
    - verify the ranking feature pipeline
    - establish a solid benchmark
    """

    def __init__(self, config: LogisticRerankerConfig | None = None) -> None:
        self.config = config or LogisticRerankerConfig()
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.is_fitted: bool = False

    # --------------------------------------------------------------------------
    # Validation / feature handling
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_features_df(features_df: pd.DataFrame) -> None:
        """
        Ensure the ranking feature table contains the required columns.
        """
        required_columns = {"impression_id", "clicked"}
        missing = required_columns - set(features_df.columns)
        if missing:
            raise ValueError(
                "Ranking features DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _get_non_feature_columns() -> set[str]:
        """
        Columns that are metadata / identifiers rather than model input features.
        """
        return {
            "split",
            "impression_id",
            "user_id",
            "candidate_news_id",
            "clicked",
        }

    def _select_feature_columns(self, features_df: pd.DataFrame) -> list[str]:
        """
        Select numeric columns to use as model input features.

        We intentionally keep only numeric columns because this reranker is built
        on engineered dense features, not raw text.
        """
        non_feature_columns = self._get_non_feature_columns()

        candidate_columns = [
            col
            for col in features_df.columns
            if col not in non_feature_columns
        ]

        numeric_columns = [
            col
            for col in candidate_columns
            if pd.api.types.is_numeric_dtype(features_df[col])
        ]

        if not numeric_columns:
            raise ValueError(
                "No numeric feature columns were found for reranker training."
            )

        return numeric_columns

    def _prepare_feature_matrix(
        self,
        features_df: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare the numeric feature matrix.

        Parameters
        ----------
        features_df : pd.DataFrame
            Ranking feature table.
        fit : bool
            If True, infer and store feature column order.
            If False, require that the model was already fitted and reuse the
            saved feature column order.

        Returns
        -------
        pd.DataFrame
            Numeric feature matrix with stable column order.
        """
        self._validate_features_df(features_df)

        working_df = features_df.copy()

        if fit:
            self.feature_columns = self._select_feature_columns(working_df)
        else:
            if not self.feature_columns:
                raise RuntimeError(
                    "Feature columns are not initialized. Fit the model first."
                )

        missing_columns = [col for col in self.feature_columns if col not in working_df.columns]
        if missing_columns:
            raise ValueError(
                "Input features are missing columns required by the fitted model: "
                + ", ".join(sorted(missing_columns))
            )

        X = working_df[self.feature_columns].copy()

        # Defensive fill in case any engineered feature came out missing.
        X = X.fillna(0.0)

        return X

    # --------------------------------------------------------------------------
    # Fitting
    # --------------------------------------------------------------------------

    def fit(self, train_features_df: pd.DataFrame) -> None:
        """
        Fit the reranker on the ranking feature table.
        """
        self._validate_features_df(train_features_df)

        X_train = self._prepare_feature_matrix(train_features_df, fit=True)
        y_train = train_features_df["clicked"].astype(int)

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        solver=self.config.solver,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

    # --------------------------------------------------------------------------
    # Scoring / prediction
    # --------------------------------------------------------------------------

    def score_examples(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score ranking examples and return probabilities as reranking scores.

        Returns
        -------
        pd.DataFrame
            Copy of input features_df with an added 'score' column.

        Notes
        -----
        We use predict_proba(..., class 1) because:
        - class 1 = clicked
        - higher probability should mean higher ranking relevance
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        self._validate_features_df(features_df)

        X = self._prepare_feature_matrix(features_df, fit=False)
        probabilities = self.pipeline.predict_proba(X)[:, 1]

        scored_df = features_df.copy()
        scored_df["score"] = probabilities.astype(float)

        return scored_df

    def evaluate(
        self,
        features_df: pd.DataFrame,
        ks: Iterable[int] = (5, 10, 20),
    ) -> dict:
        """
        Score ranking examples and compute grouped ranking metrics.
        """
        scored_df = self.score_examples(features_df)
        return evaluate_grouped_ranking(scored_df, ks=ks)

    # --------------------------------------------------------------------------
    # Interpretability helpers
    # --------------------------------------------------------------------------

    def get_feature_importance_frame(self) -> pd.DataFrame:
        """
        Return feature coefficients from the logistic regression reranker.

        Notes
        -----
        Positive coefficient:
            increasing the feature pushes the model more toward "clicked"

        Negative coefficient:
            increasing the feature pushes the model more away from "clicked"
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        classifier = self.pipeline.named_steps["classifier"]

        coefficients = classifier.coef_[0]
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coefficient": coefficients,
                "abs_coefficient": [abs(x) for x in coefficients],
            }
        ).sort_values("abs_coefficient", ascending=False)

        return importance_df.reset_index(drop=True)


# ------------------------------------------------------------------------------
# Comparison helpers
# ------------------------------------------------------------------------------

def compare_retrieval_vs_reranker(
    retrieval_scored_df: pd.DataFrame,
    reranker_scored_df: pd.DataFrame,
    ks: Iterable[int] = (5, 10, 20),
) -> dict:
    """
    Compare grouped ranking metrics for:
    - retrieval baseline scores
    - reranker scores

    Parameters
    ----------
    retrieval_scored_df : pd.DataFrame
        Ranking rows with retrieval score in column 'score'.
    reranker_scored_df : pd.DataFrame
        Ranking rows with reranker score in column 'score'.

    Returns
    -------
    dict
        Nested comparison dictionary.
    """
    retrieval_metrics = evaluate_grouped_ranking(retrieval_scored_df, ks=ks)
    reranker_metrics = evaluate_grouped_ranking(reranker_scored_df, ks=ks)

    improvement: dict[str, float] = {}
    for key, reranker_value in reranker_metrics.items():
        if key in {"num_impressions", "num_rows"}:
            continue

        retrieval_value = float(retrieval_metrics[key])
        reranker_value = float(reranker_value)
        improvement[key] = reranker_value - retrieval_value

    return {
        "retrieval": retrieval_metrics,
        "reranker": reranker_metrics,
        "absolute_improvement": improvement,
    }


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test.

    We:
    1. build a small retrieval train set
    2. fit the TF-IDF retrieval baseline
    3. build reranking features for train and dev
    4. fit a logistic regression reranker
    5. compare retrieval-only vs reranked metrics

    This checks the full two-stage ranking pipeline.
    """
    print("Running logistic reranker smoke test...\n")

    # --------------------------------------------------------------------------
    # Step 1: fit a small retrieval baseline
    # --------------------------------------------------------------------------
    print("Building retrieval training examples...")
    retrieval_train_examples = build_retrieval_examples(
        RetrievalDatasetConfig(
            split="train",
            max_impressions=400,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
        )
    )

    print("Fitting TF-IDF retrieval baseline...")
    retrieval_model = TfidfRetrievalBaseline(
        config=TfidfRetrievalConfig(
            max_features=25000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            lowercase=True,
        )
    )
    retrieval_model.fit(retrieval_train_examples)

    # --------------------------------------------------------------------------
    # Step 2: build reranking features
    # --------------------------------------------------------------------------
    print("Building ranking training features...")
    ranking_train_df = build_ranking_examples(
        RankingDatasetConfig(
            split="train",
            max_impressions=400,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
            include_retrieval_score=True,
        ),
        retrieval_model=retrieval_model,
    )

    train_summary = summarize_ranking_examples(ranking_train_df)
    print("Ranking train summary:")
    for key, value in train_summary.items():
        print(f"  {key}: {value}")

    print("\nBuilding ranking dev features...")
    ranking_dev_df = build_ranking_examples(
        RankingDatasetConfig(
            split="dev",
            max_impressions=100,
            negatives_per_positive=4,
            max_history_items=5,
            random_state=42,
            include_retrieval_score=True,
        ),
        retrieval_model=retrieval_model,
    )

    dev_summary = summarize_ranking_examples(ranking_dev_df)
    print("Ranking dev summary:")
    for key, value in dev_summary.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------------------------
    # Step 3: fit reranker
    # --------------------------------------------------------------------------
    print("\nFitting logistic reranker...")
    reranker = LogisticReranker(
        config=LogisticRerankerConfig(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    )
    reranker.fit(ranking_train_df)

    # --------------------------------------------------------------------------
    # Step 4: score dev with retrieval-only and reranker
    # --------------------------------------------------------------------------
    print("Scoring ranking dev features with reranker...")
    reranker_scored_dev = reranker.score_examples(ranking_dev_df)

    if "retrieval_score" not in ranking_dev_df.columns:
        raise ValueError(
            "Expected 'retrieval_score' in ranking dev features for comparison."
        )

    retrieval_scored_dev = ranking_dev_df.copy()
    retrieval_scored_dev["score"] = retrieval_scored_dev["retrieval_score"].astype(float)

    # --------------------------------------------------------------------------
    # Step 5: compare results
    # --------------------------------------------------------------------------
    print("\nComparing retrieval vs reranker...")
    comparison = compare_retrieval_vs_reranker(
        retrieval_scored_df=retrieval_scored_dev,
        reranker_scored_df=reranker_scored_dev,
        ks=(5, 10, 20),
    )

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
    # Step 6: inspect feature coefficients
    # --------------------------------------------------------------------------
    print("\nTop feature coefficients:")
    importance_df = reranker.get_feature_importance_frame()
    print(importance_df.head(12).to_string(index=False))