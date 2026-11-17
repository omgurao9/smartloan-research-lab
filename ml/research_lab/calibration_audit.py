"""Training-only cross-fitted probability calibration and validation subgroup audit."""
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    sha256_file,
)
from research_lab.optimize import FEATURE_SETS, TARGET


def build_base_estimator(features: list[str], seed: int) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            'F01 requires LightGBM. Install with: '
            'python -m pip install -e ".[boosting]"'
        ) from exc
    numeric = [name for name in NUMERIC_FEATURES if name in features]
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric, categorical)),
        ("model", LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "mcc_at_0_5": float(matthews_corrcoef(y, pred)),
    }


def _subgroup_table(
    validation: pd.DataFrame,
    y: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    data = validation.reset_index(drop=True).copy()
    data["_actual"] = y
    data["_probability"] = probabilities
    data["_prediction"] = (probabilities >= 0.5).astype(int)
    data["_age_band"] = pd.cut(
        data["person_age"],
        bins=[-np.inf, 25, 35, 45, 55, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )
    # Quantile bins are derived from validation feature values only for descriptive
    # grouping; they do not affect fitting, feature selection, or predictions.
    data["_income_band"] = pd.qcut(
        data["person_income"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    group_columns = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "_age_band",
        "_income_band",
    ]
    rows: list[dict[str, Any]] = []
    for column in group_columns:
        for group_value, group in data.groupby(column, observed=True, dropna=False):
            actual = group["_actual"].to_numpy(dtype=int)
            predicted = group["_prediction"].to_numpy(dtype=int)
            rows.append({
                "grouping": column.removeprefix("_"),
                "group": str(group_value),
                "n": int(len(group)),
                "positive_count": int(actual.sum()),
                "observed_positive_rate": float(actual.mean()),
                "mean_predicted_probability": float(group["_probability"].mean()),
                "predicted_positive_rate_at_0_5": float(predicted.mean()),
                "recall_at_0_5": (
                    float(((predicted == 1) & (actual == 1)).sum() / actual.sum())
                    if actual.sum() else None
                ),
                "precision_at_0_5": (
                    float(((predicted == 1) & (actual == 1)).sum() / predicted.sum())
                    if predicted.sum() else None
                ),
            })
    return pd.DataFrame(rows)


def run_calibration_audit(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    cv_folds: int = 5,
    min_subgroup_n: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    subgroup_frames: list[pd.DataFrame] = []
    timings: dict[str, float] = {}
    for feature_set_name, features in FEATURE_SETS.items():
        x_train = frame.iloc[train_rows][features]
        x_validation = frame.iloc[validation_rows][features]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        for method in ("sigmoid", "isotonic"):
            estimator = CalibratedClassifierCV(
                estimator=build_base_estimator(features, seed),
                method=method,
                cv=cv,
                ensemble=True,
                n_jobs=1,
            )
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            probabilities = estimator.predict_proba(x_validation)[:, 1]
            elapsed = time.perf_counter() - started
            key = f"{feature_set_name}/{method}"
            timings[key] = round(elapsed, 3)
            row = {
                "feature_set": feature_set_name,
                "calibration_method": method,
                "n_features": len(features),
                "fit_and_validation_seconds": round(elapsed, 3),
            }
            row.update(_metrics(y_validation, probabilities))
            metric_rows.append(row)

            fraction_positive, mean_predicted = calibration_curve(
                y_validation, probabilities, n_bins=10, strategy="quantile"
            )
            for bin_index, (observed, predicted) in enumerate(
                zip(fraction_positive, mean_predicted, strict=True), start=1
            ):
                calibration_rows.append({
                    "feature_set": feature_set_name,
                    "calibration_method": method,
                    "bin": bin_index,
                    "observed_positive_rate": float(observed),
                    "mean_predicted_probability": float(predicted),
                })
            subgroup = _subgroup_table(
                frame.iloc[validation_rows], y_validation, probabilities
            )
            subgroup.insert(0, "calibration_method", method)
            subgroup.insert(0, "feature_set", feature_set_name)
            subgroup_frames.append(subgroup)

    subgroup_table = pd.concat(subgroup_frames, ignore_index=True)
    subgroup_table = subgroup_table.loc[subgroup_table["n"] >= min_subgroup_n].copy()
    metadata = {
        "experiment_id": "F01",
        "purpose": "Cross-fitted probability calibration and descriptive validation subgroup audit",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "calibration_training_partition": "training_only",
        "calibration_cv_folds": cv_folds,
        "calibration_cv_seed": seed,
        "calibration_methods": ["sigmoid", "isotonic"],
        "subgroup_minimum_n": min_subgroup_n,
        "random_state": seed,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "timings_seconds": timings,
        "subgroup_note": (
            "Descriptive validation-set slices only; not a legal fairness assessment "
            "or evidence of causal discrimination."
        ),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibration_rows),
        subgroup_table,
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/calibration"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-subgroup-n", type=int, default=30)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    metrics, curves, subgroups, metadata = run_calibration_audit(
        frame, splits, seed=args.seed, cv_folds=args.cv_folds,
        min_subgroup_n=args.min_subgroup_n,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    metrics.to_csv(args.output / "validation_metrics.csv", index=False)
    curves.to_csv(args.output / "calibration_curve.csv", index=False)
    subgroups.to_csv(args.output / "validation_subgroups.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"\nSubgroup rows (n >= {args.min_subgroup_n}): {len(subgroups)}")
    print("\nTest partition used: False")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
