"""Leakage-safe out-of-fold stacking of tabular tree models."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    evaluate_validation,
    sha256_file,
)
from research_lab.optimize import FEATURE_SETS, TARGET


def build_base_models(features: list[str], seed: int) -> list[tuple[str, Pipeline]]:
    numeric = [name for name in NUMERIC_FEATURES if name in features]
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    preprocessor = lambda: build_preprocessor(numeric, categorical)
    models: list[tuple[str, Pipeline]] = [
        ("hist_gradient_boosting", Pipeline([
            ("preprocessor", preprocessor()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed,
            )),
        ])),
        ("random_forest", Pipeline([
            ("preprocessor", preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=seed,
            )),
        ])),
    ]
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError('E01 requires LightGBM; install with .[boosting].') from exc
    models.append(("lightgbm", Pipeline([
        ("preprocessor", preprocessor()),
        ("model", LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbosity=-1,
        )),
    ])))
    return models


def run_stacking(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    required_split = {"original_row_index", "split", TARGET}
    if not required_split.issubset(split_assignment.columns):
        raise ValueError(f"Split assignment must contain {sorted(required_split)}")
    train = split_assignment.loc[split_assignment["split"] == "train"]
    validation = split_assignment.loc[split_assignment["split"] == "validation"]
    train_rows = train["original_row_index"].astype(int).to_numpy()
    validation_rows = validation["original_row_index"].astype(int).to_numpy()
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("Training and validation partitions must be non-empty.")
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError("Training and validation partitions overlap.")
    if (train_rows.min() < 0 or validation_rows.min() < 0
            or train_rows.max() >= len(frame) or validation_rows.max() >= len(frame)):
        raise ValueError("Split indices are outside the raw dataset bounds.")
    if not np.array_equal(frame.iloc[train_rows][TARGET].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Training labels do not match saved split assignment.")
    if not np.array_equal(
        frame.iloc[validation_rows][TARGET].to_numpy(), validation[TARGET].to_numpy()
    ):
        raise ValueError("Validation labels do not match saved split assignment.")

    y_train = frame.iloc[train_rows][TARGET].astype(int).to_numpy()
    y_validation = frame.iloc[validation_rows][TARGET].astype(int).to_numpy()
    if np.unique(y_train, return_counts=True)[1].min() < cv_folds:
        raise ValueError("Each training class must have at least cv_folds observations.")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    for feature_set_name, features in FEATURE_SETS.items():
        x_train = frame.iloc[train_rows][features]
        x_validation = frame.iloc[validation_rows][features]
        estimators = build_base_models(features, seed)
        started = time.perf_counter()
        # StackingClassifier generates out-of-fold base predictions for fitting
        # the meta learner, then refits each base estimator on the full train set.
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed
            ),
            stack_method="predict_proba",
            cv=cv,
            n_jobs=1,
            passthrough=False,
        )
        stack.fit(x_train, y_train)
        probabilities = stack.predict_proba(x_validation)[:, 1]
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "feature_set": feature_set_name,
            "model": "oof_stacking_hgb_rf_lightgbm",
            "n_features": len(features),
            "base_models": "hist_gradient_boosting;random_forest;lightgbm",
            "cv_folds": cv_folds,
            "fit_and_validation_seconds": round(elapsed, 3),
        }
        row.update(evaluate_validation(pd.Series(y_validation), probabilities))
        rows.append(row)
        timings[feature_set_name] = round(elapsed, 3)

    metadata = {
        "experiment_id": "E01",
        "purpose": "Leakage-safe stacking with out-of-fold training predictions",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "meta_features": "out_of_fold_base_model_probabilities",
        "cv_folds": cv_folds,
        "cv_seed": seed,
        "random_state": seed,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "timings_seconds": timings,
        "feature_sets": list(FEATURE_SETS),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/stacking"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata = run_stacking(
        frame, splits, seed=args.seed, cv_folds=args.cv_folds
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    results.to_csv(args.output / "validation_metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print("\nTest partition used: False")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
