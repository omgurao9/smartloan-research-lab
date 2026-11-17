"""Dataset integrity and target-relationship audit; no model fitting occurs here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "loan_status"
EXPECTED_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", TARGET,
]
NUMERIC_CANDIDATES = [
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def audit_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a serializable audit summary for a raw input dataframe."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": extra_columns,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "numeric_summary": {},
        "categorical_cardinality": {},
        "target": {},
        "feature_target_crosstabs": {},
        "cleaning_rule_impact": {},
        "warnings": [],
    }

    for column in NUMERIC_CANDIDATES:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            result["numeric_summary"][column] = _json_safe({
                "min": series.min(), "median": series.median(), "max": series.max(),
                "non_numeric_or_missing": int(series.isna().sum()),
            })

    for column in frame.select_dtypes(include=["object", "category"]).columns:
        result["categorical_cardinality"][column] = {
            "unique_values": int(frame[column].nunique(dropna=False)),
            "top_values": _json_safe(frame[column].value_counts(dropna=False).head(20).to_dict()),
        }

    if TARGET in frame.columns:
        counts = frame[TARGET].value_counts(dropna=False)
        result["target"] = {
            "class_counts": _json_safe(counts.to_dict()),
            "class_proportions": _json_safe(frame[TARGET].value_counts(normalize=True, dropna=False).to_dict()),
            "unique_values": _json_safe(frame[TARGET].drop_duplicates().tolist()),
        }
        for feature in ("previous_loan_defaults_on_file", "person_gender",
                        "person_education", "person_home_ownership", "loan_intent"):
            if feature in frame.columns:
                table = pd.crosstab(frame[feature].fillna("<MISSING>"),
                                    frame[TARGET].fillna("<MISSING>"), dropna=False)
                result["feature_target_crosstabs"][feature] = _json_safe(
                    {str(index): row.to_dict() for index, row in table.iterrows()}
                )

    if {"person_age", "person_emp_exp"}.issubset(frame.columns):
        age = pd.to_numeric(frame["person_age"], errors="coerce")
        exp = pd.to_numeric(frame["person_emp_exp"], errors="coerce")
        age_flag = age >= 84
        experience_flag = exp > age - 18
        removed = age_flag | experience_flag
        result["cleaning_rule_impact"] = {
            "rule": "person_age >= 84 OR person_emp_exp > person_age - 18",
            "age_flag_rows": int(age_flag.sum()),
            "experience_flag_rows": int(experience_flag.sum()),
            "overlap_rows": int((age_flag & experience_flag).sum()),
            "unique_rows_flagged": int(removed.sum()),
            "remaining_rows": int((~removed).sum()),
        }
        if TARGET in frame.columns:
            result["cleaning_rule_impact"]["flagged_target_counts"] = _json_safe(
                frame.loc[removed, TARGET].value_counts(dropna=False).to_dict()
            )

    if missing_columns:
        result["warnings"].append("One or more expected schema columns are missing.")
    if extra_columns:
        result["warnings"].append("Unexpected columns exist; verify whether they are identifiers or leakage risks.")
    if result["duplicate_rows"]:
        result["warnings"].append("Duplicate rows detected.")
    if TARGET not in frame.columns:
        result["warnings"].append("Target column is missing; target analyses were skipped.")
    if "previous_loan_defaults_on_file" in frame.columns and TARGET in frame.columns:
        cross = pd.crosstab(frame["previous_loan_defaults_on_file"], frame[TARGET])
        if len(cross.columns) > 1 and (cross == 0).any(axis=None):
            result["warnings"].append(
                "At least one feature category has a zero count for a target class; inspect provenance and timing."
            )
    return _json_safe(result)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        f"- Rows: **{report['rows']:,}**",
        f"- Columns: **{report['columns']}**",
        f"- Duplicate rows: **{report['duplicate_rows']:,}**",
        f"- Missing expected columns: **{len(report['missing_expected_columns'])}**",
        "",
        "## Target distribution",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label, count in report.get("target", {}).get("class_counts", {}).items():
        lines.append(f"| {label} | {count} |")
    lines += ["", "## Cleaning-rule impact", ""]
    impact = report.get("cleaning_rule_impact", {})
    if impact:
        lines += [
            f"- Rule: `{impact['rule']}`",
            f"- Rows flagged: **{impact['unique_rows_flagged']:,}**",
            f"- Rows remaining: **{impact['remaining_rows']:,}**",
            f"- Flagged target counts: `{impact.get('flagged_target_counts', {})}`",
        ]
    else:
        lines.append("Required age/experience columns were not available.")
    lines += ["", "## Warnings", ""]
    lines.extend([f"- {warning}" for warning in report.get("warnings", [])] or ["- None raised by automated checks."])
    lines += [
        "",
        "> Automated checks do not establish target meaning, feature availability at decision time,",
        "> representativeness, or suitability for real lending decisions. These require source documentation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/loan_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit"))
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}. Copy loan_data.csv from the previous project into data/raw/."
        )
    frame = pd.read_csv(args.input)
    report = audit_dataframe(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Audited {len(frame):,} rows from {args.input}")
    print(f"Report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
