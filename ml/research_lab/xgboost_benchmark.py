"""Benchmark XGBoost on the frozen full and conservative feature sets."""
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
from sklearn.pipeline import Pipeline

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    evaluate_validation,
    sha256_file,
)
from research_lab.optimize import FEATURE_SETS, TARGET


def build_xgboost_model(seed: int = 42) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            'G02 requires XGBoost. Install with: '
            'python -m pip install -e ".[boosting]"'
        ) from exc
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=3.5,
        random_state=seed,
        n_jobs=-1,
    )


def run_xgboost_comparison(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Pipeline]]:
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
        raise ValueError("Training and validation rows overlap.")
    if (train_rows.min() < 0 or validation_rows.min() < 0
            or train_rows.max() >= len(frame) or validation_rows.max() >= len(frame)):
        raise ValueError("Split indices are outside the raw dataset bounds.")
    if not np.array_equal(frame.iloc[train_rows][TARGET].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Training target labels do not match the saved split assignment.")
    if not np.array_equal(
        frame.iloc[validation_rows][TARGET].to_numpy(), validation[TARGET].to_numpy()
    ):
        raise ValueError("Validation target labels do not match the saved split assignment.")

    y_train = frame.iloc[train_rows][TARGET].astype(int)
    y_validation = frame.iloc[validation_rows][TARGET].astype(int)
    rows: list[dict[str, Any]] = []
    models: dict[str, Pipeline] = {}
    timings: dict[str, float] = {}
    for feature_set_name, features in FEATURE_SETS.items():
        numeric = [name for name in NUMERIC_FEATURES if name in features]
        categorical = [name for name in CATEGORICAL_FEATURES if name in features]
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", build_xgboost_model(seed)),
        ])
        x_train = frame.iloc[train_rows][features]
        x_validation = frame.iloc[validation_rows][features]
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        elapsed = time.perf_counter() - started
        result = {
            "feature_set": feature_set_name,
            "model": "xgboost",
            "n_features": len(features),
            "fit_and_validation_seconds": round(elapsed, 3),
        }
        result.update(evaluate_validation(y_validation, probabilities))
        rows.append(result)
        timings[feature_set_name] = round(elapsed, 3)
        models[feature_set_name] = pipeline

    metadata = {
        "experiment_id": "G02",
        "purpose": "Fixed-configuration XGBoost comparison",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "random_state": seed,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "timings_seconds": timings,
        "feature_sets": list(FEATURE_SETS),
    }
    return pd.DataFrame(rows), metadata, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/xgboost"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata, models = run_xgboost_comparison(frame, splits, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    results.to_csv(args.output / "validation_metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    import joblib
    for name, model in models.items():
        joblib.dump(model, args.output / f"{name}_xgboost.joblib")
    print(results.to_string(index=False))
    print("\nTest partition used: False")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
