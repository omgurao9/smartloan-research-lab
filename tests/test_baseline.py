import pandas as pd
import pytest

from research_lab.split import prepare_split
from research_lab.baseline import run_benchmarks


def make_frame(n=120):
    rows = []
    for index in range(n):
        rows.append({
            "person_age": 25 + (index % 10),
            "person_gender": "female" if index % 2 else "male",
            "person_education": "Bachelor" if index % 3 else "Master",
            "person_income": 50000 + index * 10,
            "person_emp_exp": 3 + (index % 5),
            "person_home_ownership": "RENT" if index % 2 else "OWN",
            "loan_amnt": 10000 + index,
            "loan_intent": "EDUCATION" if index % 2 else "MEDICAL",
            "loan_int_rate": 10.0 + (index % 4),
            "loan_percent_income": 0.2 + (index % 5) / 100,
            "cb_person_cred_hist_length": 4 + (index % 6),
            "credit_score": 600 + (index % 200),
            "previous_loan_defaults_on_file": "No" if index % 3 else "Yes",
            "loan_status": index % 2,
        })
    return pd.DataFrame(rows)


def test_baseline_runs_full_and_ablated_features_without_test_partition():
    frame = make_frame()
    splits, _ = prepare_split(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    results, metadata = run_benchmarks(frame, splits, seed=7)
    assert len(results) == 6
    assert set(results["feature_set"]) == {
        "full_features", "without_previous_defaults"
    }
    assert metadata["test_partition_used"] is False
    assert set(results["model"]) == {
        "logistic_regression", "random_forest", "hist_gradient_boosting"
    }


def test_baseline_rejects_mismatched_saved_labels():
    frame = make_frame()
    splits, _ = prepare_split(frame)
    splits.loc[splits["split"] == "train", "loan_status"] = 1
    with pytest.raises(ValueError, match="target labels do not match"):
        run_benchmarks(frame, splits)
