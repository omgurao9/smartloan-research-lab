import numpy as np
import pandas as pd
import pytest

from research_lab.optimize import run_optimization
from research_lab.split import prepare_split


def make_frame(n=180):
    rows = []
    for index in range(n):
        target = index % 2
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
            "previous_loan_defaults_on_file": "No" if target else "Yes",
            "loan_status": target,
        })
    return pd.DataFrame(rows)


def test_optimization_uses_training_cv_and_validation_only():
    frame = make_frame()
    splits, _ = prepare_split(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    results, summaries, metadata, models, trials = run_optimization(
        frame, splits, seed=7, n_trials=1, cv_folds=2
    )
    assert len(results) == 2
    assert set(results["feature_set"]) == {
        "full_features", "conservative_no_defaults_no_rate"
    }
    assert metadata["test_partition_used"] is False
    assert metadata["evaluation_partition"] == "validation"
    assert set(models) == set(summaries)
    assert len(trials) == 2
    assert all(np.isfinite(results["average_precision"]))
    assert all(summary["n_trials"] == 1 for summary in summaries.values())


def test_optimization_rejects_mismatched_validation_labels():
    frame = make_frame()
    splits, _ = prepare_split(frame)
    validation_index = splits.index[splits["split"] == "validation"][0]
    splits.loc[validation_index, "loan_status"] = 1 - splits.loc[
        validation_index, "loan_status"
    ]
    with pytest.raises(ValueError, match="Validation target labels"):
        run_optimization(frame, splits, n_trials=1, cv_folds=2)
