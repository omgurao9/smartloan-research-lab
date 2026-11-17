import numpy as np
import pandas as pd
import pytest

from research_lab.split import prepare_split
from research_lab.xgboost_benchmark import run_xgboost_comparison


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
            "loan_int_rate": 10.0 + (index % 4),
            "loan_percent_income": 0.2 + (index % 5) / 100,
            "cb_person_cred_hist_length": 4 + (index % 6),
            "credit_score": 600 + (index % 200),
            "person_emp_exp": 3 + (index % 5),
            "loan_intent": "EDUCATION" if index % 2 else "MEDICAL",
            "previous_loan_defaults_on_file": "No" if target else "Yes",
            "loan_status": target,
        })
    return pd.DataFrame(rows)


def test_xgboost_benchmark_keeps_test_partition_unopened():
    pytest.importorskip("xgboost")
    frame = make_frame()
    splits, _ = prepare_split(
        frame, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2
    )
    results, metadata, models = run_xgboost_comparison(frame, splits, seed=7)
    assert len(results) == 2
    assert set(results["model"]) == {"xgboost"}
    assert metadata["test_partition_used"] is False
    assert len(models) == 2
    assert np.isfinite(results["average_precision"]).all()
