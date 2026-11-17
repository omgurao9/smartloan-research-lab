"""Train baseline tabular classifiers and evaluate on validation only.

The test partition is deliberately never loaded by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "loan_status"
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]
CATEGORICAL_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def evaluate_validation(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, predictions)),
        "precision_at_0_5": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc_at_0_5": float(matthews_corrcoef(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def run_benchmarks(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on training rows and evaluate only on validation rows."""
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    required_split_columns = {"original_row_index", "split", TARGET}
    if not required_split_columns.issubset(split_assignment.columns):
        raise ValueError(f"Split assignment must contain {sorted(required_split_columns)}")

    train_rows = split_assignment.loc[
        split_assignment["split"] == "train", "original_row_index"
    ].astype(int)
    validation_rows = split_assignment.loc[
        split_assignment["split"] == "validation", "original_row_index"
    ].astype(int)
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError("Training and validation partitions must both be non-empty.")
    if set(train_rows).intersection(validation_rows):
        raise ValueError("Training and validation rows overlap.")
    if (train_rows.min() < 0 or validation_rows.min() < 0
            or train_rows.max() >= len(frame) or validation_rows.max() >= len(frame)):
        raise ValueError("Split indices are outside the raw dataset bounds.")

    if not np.array_equal(
        frame.iloc[train_rows.to_numpy()][TARGET].to_numpy(),
        split_assignment.loc[split_assignment["split"] == "train", TARGET].to_numpy(),
    ):
        raise ValueError("Training target labels do not match the saved split assignment.")
    if not np.array_equal(
        frame.iloc[validation_rows.to_numpy()][TARGET].to_numpy(),
        split_assignment.loc[split_assignment["split"] == "validation", TARGET].to_numpy(),
    ):
        raise ValueError("Validation target labels do not match the saved split assignment.")

    feature_sets = {
        "full_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "without_previous_defaults": [
            feature for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
            if feature != "previous_loan_defaults_on_file"
        ],
        "without_interest_rate": [
            feature for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
            if feature != "loan_int_rate"
        ],
        "conservative_no_defaults_no_rate": [
            feature for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
            if feature not in {"previous_loan_defaults_on_file", "loan_int_rate"}
        ],
    }
    y_train = frame.iloc[train_rows.to_numpy()][TARGET].astype(int)
    y_validation = frame.iloc[validation_rows.to_numpy()][TARGET].astype(int)
    results: list[dict[str, Any]] = []
    timing: dict[str, float] = {}

    for feature_set_name, features in feature_sets.items():
        numeric = [feature for feature in NUMERIC_FEATURES if feature in features]
        categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
        x_train = frame.iloc[train_rows.to_numpy()][features]
        x_validation = frame.iloc[validation_rows.to_numpy()][features]
        for model_name, estimator in build_models(seed).items():
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(numeric, categorical)),
                    ("model", estimator),
                ]
            )
            started = time.perf_counter()
            pipeline.fit(x_train, y_train)
            probabilities = pipeline.predict_proba(x_validation)[:, 1]
            elapsed = time.perf_counter() - started
            row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "n_features": len(features),
                "fit_and_validation_seconds": round(elapsed, 3),
            }
            row.update(evaluate_validation(y_validation, probabilities))
            results.append(row)
            timing[f"{feature_set_name}/{model_name}"] = round(elapsed, 3)

    metadata = {
        "experiment_id": "B01-B02",
        "purpose": "Baseline comparison and previous-default feature ablation",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "random_state": seed,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "timings_seconds": timing,
    }
    return pd.DataFrame(results), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata = run_benchmarks(frame, splits, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    results.to_csv(args.output / "validation_metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.sort_values("average_precision", ascending=False).to_string(index=False))
    print("\nTest partition used: False")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
