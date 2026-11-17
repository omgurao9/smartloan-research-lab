"""CPU-friendly TabM benchmark on the frozen train/validation split."""
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
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from research_lab.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    evaluate_validation,
    sha256_file,
)
from research_lab.optimize import FEATURE_SETS, TARGET


def _encode_features(
    train: pd.DataFrame, other: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any], np.ndarray, np.ndarray]:
    numeric = [c for c in NUMERIC_FEATURES if c in features]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    train_num = train[numeric].apply(pd.to_numeric, errors="coerce")
    other_num = other[numeric].apply(pd.to_numeric, errors="coerce")
    medians = train_num.median().fillna(0.0)
    scaler = StandardScaler()
    x_num_train = scaler.fit_transform(train_num.fillna(medians)).astype(np.float32)
    x_num_other = scaler.transform(other_num.fillna(medians)).astype(np.float32)

    train_cats: list[np.ndarray] = []
    other_cats: list[np.ndarray] = []
    cardinalities: list[int] = []
    category_maps: dict[str, dict[str, int]] = {}
    for column in categorical:
        values = train[column].fillna("__MISSING__").astype(str)
        unique = sorted(values.unique().tolist())
        mapping = {value: index for index, value in enumerate(unique)}
        # Unknown validation categories map to the first training category.
        train_cats.append(values.map(mapping).to_numpy(dtype=np.int64))
        other_values = other[column].fillna("__MISSING__").astype(str)
        other_cats.append(other_values.map(mapping).fillna(0).to_numpy(dtype=np.int64))
        cardinalities.append(max(1, len(unique)))
        category_maps[column] = mapping

    x_cat_train = (
        np.column_stack(train_cats).astype(np.int64)
        if train_cats else np.empty((len(train), 0), dtype=np.int64)
    )
    x_cat_other = (
        np.column_stack(other_cats).astype(np.int64)
        if other_cats else np.empty((len(other), 0), dtype=np.int64)
    )
    return (
        x_num_train, x_num_other, cardinalities,
        {"numeric_features": numeric, "categorical_features": categorical,
         "numeric_medians": medians.to_dict(), "category_maps": category_maps},
        x_cat_train, x_cat_other,
    )


def _fit_one(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    k: int,
    d_block: int,
    n_blocks: int,
    cat_cardinalities: list[int],
    device: str,
) -> tuple[Any, int, float]:
    from tabm import TabM
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    indices = np.arange(len(y))
    fit_idx, stop_idx = train_test_split(
        indices, test_size=0.1, random_state=seed, stratify=y
    )

    def tensors(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(x_num[idx]),
            torch.from_numpy(x_cat[idx]),
            torch.from_numpy(y[idx].astype(np.float32).reshape(-1, 1)),
        )

    fit_loader = DataLoader(tensors(fit_idx), batch_size=batch_size, shuffle=True)
    stop_x_num = torch.from_numpy(x_num[stop_idx]).to(device)
    stop_x_cat = torch.from_numpy(x_cat[stop_idx]).to(device)
    stop_y = torch.from_numpy(y[stop_idx].astype(np.float32).reshape(-1, 1)).to(device)

    model = TabM.make(
        n_num_features=x_num.shape[1],
        cat_cardinalities=cat_cardinalities or None,
        d_out=1,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_num, batch_cat, batch_y in fit_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_num, batch_cat if batch_cat.shape[1] else None)
            targets = batch_y.unsqueeze(1).expand_as(logits)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(stop_x_num, stop_x_cat if stop_x_cat.shape[1] else None)
            targets = stop_y.unsqueeze(1).expand_as(logits)
            stop_loss = float(loss_fn(logits, targets).item())
        print(f"[TabM] epoch {epoch}/{epochs}; internal stop loss={stop_loss:.6f}", flush=True)
        if stop_loss < best_loss - 1e-5:
            best_loss = stop_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, best_epoch, elapsed


def run_tabm(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 4,
    k: int = 8,
    d_block: int = 128,
    n_blocks: int = 2,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train TabM using a training-internal early-stopping split; evaluate validation once."""
    if min(epochs, batch_size, patience, k, d_block, n_blocks) < 1:
        raise ValueError("All training configuration values must be positive.")
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    train = split_assignment.loc[split_assignment["split"] == "train"]
    validation = split_assignment.loc[split_assignment["split"] == "validation"]
    train_idx = train["original_row_index"].astype(int).to_numpy()
    valid_idx = validation["original_row_index"].astype(int).to_numpy()
    if not len(train_idx) or not len(valid_idx) or np.intersect1d(train_idx, valid_idx).size:
        raise ValueError("Invalid or overlapping train/validation split.")
    if not np.array_equal(frame.iloc[train_idx][TARGET].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Training labels do not match split assignment.")
    if not np.array_equal(frame.iloc[valid_idx][TARGET].to_numpy(), validation[TARGET].to_numpy()):
        raise ValueError("Validation labels do not match split assignment.")

    y_train = frame.iloc[train_idx][TARGET].astype(int).to_numpy()
    y_valid = frame.iloc[valid_idx][TARGET].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    best_epochs: dict[str, int] = {}
    torch.set_num_threads(1)

    for feature_name, features in FEATURE_SETS.items():
        print(f"[TabM] Preparing {feature_name}", flush=True)
        train_frame = frame.iloc[train_idx][features].copy()
        valid_frame = frame.iloc[valid_idx][features].copy()
        encoded = _encode_features(train_frame, valid_frame, features)
        x_num_train, x_num_valid, cardinalities, preprocessing, x_cat_train, x_cat_valid = encoded
        model, best_epoch, elapsed = _fit_one(
            x_num_train, x_cat_train, y_train, seed=seed, epochs=epochs,
            batch_size=batch_size, patience=patience, k=k, d_block=d_block,
            n_blocks=n_blocks, cat_cardinalities=cardinalities, device=device,
        )
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(x_num_valid).to(device),
                torch.from_numpy(x_cat_valid).to(device) if x_cat_valid.shape[1] else None,
            )
            probabilities = torch.sigmoid(logits).mean(dim=1).squeeze(-1).cpu().numpy()
        row = {
            "feature_set": feature_name,
            "model": "tabm",
            "n_features": len(features),
            "best_epoch": best_epoch,
            "epochs_max": epochs,
            "k": k,
            "d_block": d_block,
            "n_blocks": n_blocks,
            "batch_size": batch_size,
            "device": device,
            "fit_seconds": round(elapsed, 3),
        }
        from research_lab.baseline import evaluate_validation
        row.update(evaluate_validation(pd.Series(y_valid), probabilities))
        rows.append(row)
        timings[feature_name] = round(elapsed, 3)
        best_epochs[feature_name] = best_epoch
        print(f"[TabM] {feature_name} complete; best epoch={best_epoch}", flush=True)

    metadata = {
        "experiment_id": "N01_TABM",
        "purpose": "TabM parameter-efficient neural tabular benchmark",
        "evaluation_partition": "validation",
        "test_partition_used": False,
        "random_state": seed,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "tabm_version": getattr(__import__("tabm"), "__version__", "unknown"),
        "device": device,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "k": k,
        "d_block": d_block,
        "n_blocks": n_blocks,
        "best_epochs": best_epochs,
        "timings_seconds": timings,
        "preprocessing": "Numeric median imputation and standardization fit on training partition; categorical integer maps fit on training partition; 10% training-internal stratified early-stopping split.",
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/split_indices.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/tabm"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--d-block", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    splits = pd.read_csv(args.splits)
    results, metadata = run_tabm(
        frame, splits, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience, k=args.k,
        d_block=args.d_block, n_blocks=args.n_blocks, device=args.device,
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
