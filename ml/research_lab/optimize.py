"""Optuna tuning of HistGradientBoosting using training-only cross-validation.

Validation is evaluated only after hyperparameter selection. The test partition
is never selected, transformed, scored, or loaded as a modeling target.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    evaluate_validation,
    sha256_file,
)

TARGET = "loan_status"
FEATURE_SETS = {
    "full_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "conservative_no_defaults_no_rate": [
        feature
        for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if feature not in {"previous_loan_defaults_on_file", "loan_int_rate"}
    ],
}


def _make_pipeline(features: list[str], params: dict[str, Any], seed: int) -> Pipeline:
    numeric = [name for name in NUMERIC_FEATURES if name in features]
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    model = HistGradientBoostingClassifier(
        **params,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", model),
        ]
    )


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63, step=4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60, step=5),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-4, 10.0, log=True
        ),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }


def optimize_feature_set(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    features: list[str],
    feature_set_name: str,
    seed: int,
    n_trials: int,
    cv_folds: int,
) -> tuple[dict[str, Any], pd.DataFrame, optuna.Study]:
    """Optimize mean CV average precision using only the training partition."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        pipeline = _make_pipeline(features, params, seed)
        scores = cross_val_score(
            pipeline,
            x_train[features],
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        trial.set_user_attr("fold_average_precision", [float(score) for score in scores])
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"O01_{feature_set_name}",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.insert(0, "feature_set", feature_set_name)
    best = {
        "feature_set": feature_set_name,
        "best_cv_average_precision": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "cv_folds": cv_folds,
        "cv_seed": seed,
    }
    return best, trials, study


def run_optimization(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    n_trials: int = 25,
    cv_folds: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Pipeline]]:
    """Tune full and conservative HGB models; evaluate each once on validation."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    required_split = {"original_row_index", "split", TARGET}
    if not required_split.issubset(split_assignment.columns):
        raise ValueError(f"Split assignment must contain {sorted(required_split)}")

    train_assignment = split_assignment.loc[split_assignment["split"] == "train"]
    validation_assignment = split_assignment.loc[
        split_assignment["split"] == "validation"
    ]
    train_rows = train_assignment["original_row_index"].astype(int).to_numpy()
    validation_rows = validation_assignment["original_row_index"].astype(int).to_numpy()
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError("Training and validation partitions must be non-empty.")
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError("Training and validation rows overlap.")
    if (
        train_rows.min() < 0
        or validation_rows.min() < 0
        or train_rows.max() >= len(frame)
        or validation_rows.max() >= len(frame)
    ):
        raise ValueError("Split indices are outside the raw dataset bounds.")
    if not np.array_equal(
        frame.iloc[train_rows][TARGET].to_numpy(),
        train_assignment[TARGET].to_numpy(),
    ):
        raise ValueError("Training target labels do not match the saved split assignment.")
    if not np.array_equal(
        frame.iloc[validation_rows][TARGET].to_numpy(),
        validation_assignment[TARGET].to_numpy(),
    ):
        raise ValueError("Validation target labels do not match the saved split assignment.")

    y_train = frame.iloc[train_rows][TARGET].astype(int)
    y_validation = frame.iloc[validation_rows][TARGET].astype(int)
    if y_train.value_counts().min() < cv_folds:
        raise ValueError("Each training class must have at least cv_folds observations.")
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    models: dict[str, Pipeline] = {}

    for feature_set_name, features in FEATURE_SETS.items():
        best, _, study = optimize_feature_set(
            frame.iloc[train_rows],
            y_train,
            features=features,
            feature_set_name=feature_set_name,
            seed=seed,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        final_model = _make_pipeline(features, best["best_params"], seed)
        final_model.fit(frame.iloc[train_rows][features], y_train)
        probabilities = final_model.predict_proba(
            frame.iloc[validation_rows][features]
        )[:, 1]
        metrics = evaluate_validation(y_validation, probabilities)
        row = {
            "feature_set": feature_set_name,
            "model": "hist_gradient_boosting_optuna",
            "n_features": len(features),
            "best_cv_average_precision": best["best_cv_average_precision"],
        }
        row.update(metrics)
        rows.append(row)
        best["validation_metrics"] = metrics
        best["best_trial_number"] = int(study.best_trial.number)
        summaries[feature_set_name] = best
        models[feature_set_name] = final_model

    metadata = {
        "experiment_id": "O01",
        "purpose": "Optuna HGB tuning with training-only stratified cross-validation",
        "optimization_metric": "average_precision",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "random_state": seed,
        "n_trials_per_feature_set": n_trials,
        "cv_folds": cv_folds,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "optuna_version": optuna.__version__,
        "feature_sets": list(FEATURE_SETS),
    }
    return pd.DataFrame(rows), summaries, metadata, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/optimization"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, summaries, metadata, models = run_optimization(
        frame,
        splits,
        seed=args.seed,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
    )

    run_id = datetime.now(timezone.utc).strftime("O01_%Y%m%dT%H%M%SZ")
    output_dir = args.output / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata["run_id"] = run_id
    metadata["dataset_sha256"] = sha256_file(args.input)
    metadata["split_sha256"] = sha256_file(args.splits)
    results.to_csv(output_dir / "validation_metrics.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}_best.joblib")
    print(results.to_string(index=False))
    print("\nTest partition used: False")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
