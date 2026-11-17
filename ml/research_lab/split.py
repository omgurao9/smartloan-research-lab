"""Create one reproducible stratified train/validation/test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

from research_lab.audit import EXPECTED_COLUMNS, TARGET


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    frame: pd.DataFrame,
    *,
    max_age_exclusive: int = 84,
    min_employment_age: int = 18,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean rows using the documented rule and return a stable split assignment."""
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; resolve them before splitting.")
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be greater than zero.")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Expected a binary target with exactly two classes.")

    working = frame.copy()
    working.insert(0, "original_row_index", np.arange(len(working), dtype=int))
    age = pd.to_numeric(working["person_age"], errors="coerce")
    experience = pd.to_numeric(working["person_emp_exp"], errors="coerce")
    if age.isna().any() or experience.isna().any():
        raise ValueError("Age and employment experience must be numeric.")
    flagged = (age >= max_age_exclusive) | (
        experience > age - min_employment_age
    )
    retained = working.loc[~flagged].copy()
    if retained.empty:
        raise ValueError("Cleaning rule removed every row.")

    train_idx, holdout_idx = train_test_split(
        retained.index,
        test_size=validation_fraction + test_fraction,
        random_state=random_state,
        stratify=retained[TARGET],
    )
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    validation_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_fraction,
        random_state=random_state,
        stratify=retained.loc[holdout_idx, TARGET],
    )

    assignment = pd.concat(
        [
            pd.DataFrame({"row_index": train_idx, "split": "train"}),
            pd.DataFrame({"row_index": validation_idx, "split": "validation"}),
            pd.DataFrame({"row_index": test_idx, "split": "test"}),
        ],
        ignore_index=True,
    )
    assignment = assignment.merge(
        retained[["original_row_index", TARGET]],
        left_on="row_index",
        right_index=True,
        how="left",
        validate="one_to_one",
    ).drop(columns=["row_index"])
    assignment = assignment.rename(columns={"original_row_index": "original_row_index"})
    assignment = assignment.sort_values("original_row_index").reset_index(drop=True)

    split_counts = {}
    for name in ("train", "validation", "test"):
        subset = assignment.loc[assignment["split"] == name, TARGET]
        split_counts[name] = {
            "rows": int(len(subset)),
            "class_counts": {
                str(key): int(value)
                for key, value in subset.value_counts().sort_index().items()
            },
        }

    metadata = {
        "rows_before_cleaning": int(len(frame)),
        "rows_removed": int(flagged.sum()),
        "rows_after_cleaning": int(len(retained)),
        "cleaning_rule": (
            f"person_age >= {max_age_exclusive} OR "
            f"person_emp_exp > person_age - {min_employment_age}"
        ),
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "random_state": random_state,
        "split_counts": split_counts,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    return assignment, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    cleaning_config = config["cleaning"]
    split_config = config["split"]
    input_path = args.input or Path(dataset_config["path"])
    frame = pd.read_csv(input_path)

    assignment, metadata = prepare_split(
        frame,
        max_age_exclusive=int(cleaning_config["max_age_exclusive"]),
        min_employment_age=int(cleaning_config["min_employment_age"]),
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        random_state=int(split_config["random_state"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata["dataset_path"] = str(input_path)
    metadata["dataset_sha256"] = sha256_file(input_path)
    split_csv = assignment.to_csv(index=False)
    metadata["split_sha256"] = hashlib.sha256(split_csv.encode("utf-8")).hexdigest()
    (args.output / "split_indices.csv").write_text(split_csv, encoding="utf-8")
    (args.output / "split_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Created split assignment for {metadata['rows_after_cleaning']:,} rows.")
    for name, values in metadata["split_counts"].items():
        print(f"{name}: {values['rows']:,} rows | classes={values['class_counts']}")
    print(f"Dataset SHA-256: {metadata['dataset_sha256']}")
    print(f"Split SHA-256:   {metadata['split_sha256']}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
