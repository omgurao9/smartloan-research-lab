"""One-time final test evaluation after model selection is frozen.

Fits sigmoid-calibrated LightGBM on train + validation rows, then evaluates
once on the untouched test partition. Test labels are never used for fitting,
calibration, threshold selection, or model selection.
"""
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from research_lab.baseline import sha256_file
from research_lab.calibration_audit import build_base_estimator
from research_lab.optimize import FEATURE_SETS, TARGET


def _metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(y, predictions)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, predictions)),
        "precision_at_0_5": float(precision_score(y, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predictions, zero_division=0)),
        "specificity_at_0_5": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_at_0_5": float(f1_score(y, predictions, zero_division=0)),
        "mcc_at_0_5": float(matthews_corrcoef(y, predictions)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _bootstrap_intervals(
    y: np.ndarray, probabilities: np.ndarray, *, seed: int, repeats: int
) -> dict[str, dict[str, float]]:
    """Percentile intervals for test-set metric uncertainty; not model tuning."""
    rng = np.random.default_rng(seed)
    class_zero = np.flatnonzero(y == 0)
    class_one = np.flatnonzero(y == 1)
    values: dict[str, list[float]] = {
        "average_precision": [],
        "roc_auc": [],
        "brier_score": [],
    }
    for _ in range(repeats):
        sample = np.concatenate([
            rng.choice(class_zero, size=len(class_zero), replace=True),
            rng.choice(class_one, size=len(class_one), replace=True),
        ])
        rng.shuffle(sample)
        ys = y[sample]
        ps = probabilities[sample]
        values["average_precision"].append(float(average_precision_score(ys, ps)))
        values["roc_auc"].append(float(roc_auc_score(ys, ps)))
        values["brier_score"].append(float(brier_score_loss(ys, ps)))
    return {
        name: {
            "lower_95": float(np.percentile(scores, 2.5)),
            "upper_95": float(np.percentile(scores, 97.5)),
        }
        for name, scores in values.items()
    }


def run_final_test_evaluation(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    cv_folds: int = 5,
    bootstrap_repeats: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_split = {"original_row_index", "split", TARGET}
    if not required_split.issubset(split_assignment.columns):
        raise ValueError(f"Split assignment must contain {sorted(required_split)}")
    assignments = {
        name: split_assignment.loc[split_assignment["split"] == name]
        for name in ("train", "validation", "test")
    }
    indices = {
        name: part["original_row_index"].astype(int).to_numpy()
        for name, part in assignments.items()
    }
    if any(len(index) == 0 for index in indices.values()):
        raise ValueError("Train, validation, and test partitions must all be non-empty.")
    if (np.intersect1d(indices["train"], indices["validation"]).size
            or np.intersect1d(indices["train"], indices["test"]).size
            or np.intersect1d(indices["validation"], indices["test"]).size):
        raise ValueError("Saved partitions overlap.")
    if min(index.min() for index in indices.values()) < 0 or max(
        index.max() for index in indices.values()
    ) >= len(frame):
        raise ValueError("Split indices are outside the raw dataset bounds.")
    for name, index in indices.items():
        if not np.array_equal(
            frame.iloc[index][TARGET].to_numpy(),
            assignments[name][TARGET].to_numpy(),
        ):
            raise ValueError(f"{name.title()} labels do not match saved split assignment.")

    fit_indices = np.concatenate([indices["train"], indices["validation"]])
    y_fit = frame.iloc[fit_indices][TARGET].astype(int).to_numpy()
    y_test = frame.iloc[indices["test"]][TARGET].astype(int).to_numpy()
    if np.unique(y_fit, return_counts=True)[1].min() < cv_folds:
        raise ValueError("Each fit class must have at least cv_folds observations.")

    result_rows: list[dict[str, Any]] = []
    intervals: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for feature_name, features in FEATURE_SETS.items():
        print(
            f"[FINAL TEST] Fitting frozen sigmoid-LightGBM choice: {feature_name} "
            f"on {len(fit_indices):,} train+validation rows",
            flush=True,
        )
        x_fit = frame.iloc[fit_indices][features]
        x_test = frame.iloc[indices["test"]][features]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        estimator = CalibratedClassifierCV(
            estimator=build_base_estimator(features, seed),
            method="sigmoid",
            cv=cv,
            ensemble=True,
            n_jobs=1,
        )
        started = time.perf_counter()
        estimator.fit(x_fit, y_fit)
        print(
            f"[FINAL TEST] {feature_name}: fit complete; evaluating test set once "
            f"({len(indices['test']):,} rows)",
            flush=True,
        )
        probabilities = estimator.predict_proba(x_test)[:, 1]
        elapsed = time.perf_counter() - started
        metrics = _metric_values(y_test, probabilities)
        result_rows.append({
            "feature_set": feature_name,
            "model": "sigmoid_calibrated_lightgbm",
            "n_features": len(features),
            "fit_rows_train_plus_validation": int(len(fit_indices)),
            "test_rows": int(len(indices["test"])),
            "threshold": 0.5,
            **metrics,
        })
        intervals[feature_name] = _bootstrap_intervals(
            y_test, probabilities, seed=seed, repeats=bootstrap_repeats
        )
        timings[feature_name] = round(elapsed, 3)
        print(f"[FINAL TEST] {feature_name} complete.", flush=True)

    metadata = {
        "experiment_id": "FINAL_TEST_01",
        "model_selection_frozen_before_test": True,
        "primary_model": "sigmoid_calibrated_lightgbm_full_features",
        "secondary_sensitivity_model": "sigmoid_calibrated_lightgbm_conservative_features",
        "calibration_method": "sigmoid",
        "calibration_cv": cv_folds,
        "calibration_cv_fit_data": "train_plus_validation_only",
        "threshold": 0.5,
        "threshold_selected_using_test": False,
        "test_used_for_any_fitting_or_selection": False,
        "test_evaluated_once": True,
        "train_rows": int(len(indices["train"])),
        "validation_rows": int(len(indices["validation"])),
        "fit_rows_train_plus_validation": int(len(fit_indices)),
        "test_rows": int(len(indices["test"])),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_intervals": intervals,
        "random_state": seed,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "timings_seconds_including_fit_and_test_prediction": timings,
        "interpretation_limit": (
            "Single frozen dataset split; bootstrap intervals quantify test-sample "
            "uncertainty conditional on these fitted models, not training-seed or "
            "distribution-shift uncertainty. Synthetic/augmented loan_status benchmark only."
        ),
    }
    return pd.DataFrame(result_rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/final_test"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata = run_final_test_evaluation(
        frame, splits, seed=args.seed, cv_folds=args.cv_folds,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    results.to_csv(args.output / "test_metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nFINAL TEST RESULTS")
    print(results.to_string(index=False))
    print("\n95% bootstrap intervals:")
    print(json.dumps(metadata["bootstrap_intervals"], indent=2))
    print("\nTest set evaluated once; no test-driven model or threshold selection.")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
