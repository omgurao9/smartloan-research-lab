"""TabICLv2 in-context benchmark on the frozen train/validation split."""
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

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    evaluate_validation,
    sha256_file,
)
from research_lab.optimize import FEATURE_SETS, TARGET


def run_tabicl(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    n_estimators: int = 4,
    device: str = "cpu",
    n_jobs: int = 4,
    prediction_batch_size: int = 64,
    estimator_class: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit TabICLv2 on train rows and evaluate validation rows only.

    Validation inference is chunked to bound peak attention-memory use on
    consumer hardware. The fitted training context and validation population
    are unchanged; chunking only changes how query rows are submitted.
    """
    if n_estimators < 1 or n_jobs < 1 or prediction_batch_size < 1:
        raise ValueError("n_estimators, n_jobs, and prediction_batch_size must be positive.")
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

    if estimator_class is None:
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:
            raise ImportError("N01 requires TabICL. Install with: python -m pip install tabicl") from exc
        estimator_class = TabICLClassifier

    y_train = frame.iloc[train_rows][TARGET].astype(int).to_numpy()
    y_validation = frame.iloc[validation_rows][TARGET].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    for feature_set_name, features in FEATURE_SETS.items():
        print(f"[TabICL] Feature set: {feature_set_name}; fitting on {len(train_rows):,} rows", flush=True)
        x_train = frame.iloc[train_rows][features].copy()
        x_validation = frame.iloc[validation_rows][features].copy()
        model = estimator_class(
            n_estimators=n_estimators,
            device=device,
            random_state=seed,
            n_jobs=n_jobs,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        print(
            f"[TabICL] {feature_set_name}: fit complete; predicting "
            f"{len(validation_rows):,} validation rows in batches of {prediction_batch_size}",
            flush=True,
        )
        probability_batches: list[np.ndarray] = []
        for start in range(0, len(x_validation), prediction_batch_size):
            stop = min(start + prediction_batch_size, len(x_validation))
            batch = np.asarray(model.predict_proba(x_validation.iloc[start:stop]))
            if batch.ndim != 2 or batch.shape[0] != stop - start:
                raise ValueError("TabICL returned an unexpected probability array shape.")
            probability_batches.append(batch)
            print(
                f"[TabICL] {feature_set_name}: predicted {stop:,}/{len(x_validation):,}",
                flush=True,
            )
        probabilities = np.concatenate(probability_batches, axis=0)
        elapsed = time.perf_counter() - started
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("TabICL estimator did not expose positive class 1.")
        positive_column = classes.index(1)
        positive_probabilities = probabilities[:, positive_column]
        row: dict[str, Any] = {
            "feature_set": feature_set_name,
            "model": "tabicl_v2",
            "n_features": len(features),
            "n_estimators": n_estimators,
            "device": device,
            "prediction_batch_size": prediction_batch_size,
            "fit_and_validation_seconds": round(elapsed, 3),
        }
        row.update(evaluate_validation(pd.Series(y_validation), positive_probabilities))
        rows.append(row)
        timings[feature_set_name] = round(elapsed, 3)

    metadata = {
        "experiment_id": "N01_TABICL",
        "purpose": "TabICLv2 in-context tabular classification benchmark",
        "checkpoint": "tabicl-classifier-v2-20260212.ckpt (package default)",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "random_state": seed,
        "n_estimators": n_estimators,
        "device": device,
        "n_jobs": n_jobs,
        "prediction_batch_size": prediction_batch_size,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "tabicl_version": getattr(__import__("tabicl"), "__version__", "unknown")
            if estimator_class.__module__.startswith("tabicl") else "injected-test-estimator",
        "timings_seconds": timings,
        "feature_sets": list(FEATURE_SETS),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/tabicl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata = run_tabicl(
        frame, splits, seed=args.seed, n_estimators=args.n_estimators,
        device=args.device, n_jobs=args.n_jobs,
        prediction_batch_size=args.prediction_batch_size,
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
