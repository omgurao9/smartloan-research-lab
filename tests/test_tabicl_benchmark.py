import numpy as np
import pandas as pd

from research_lab.optimize import FEATURE_SETS
from research_lab.split import prepare_split
from research_lab.tabicl_benchmark import run_tabicl


def make_frame(n=240):
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
            "loan_intent": "EDUCATION" if index % 2 else "MEDICAL",
            "previous_loan_defaults_on_file": "No" if target else "Yes",
            "loan_status": target,
        })
    return pd.DataFrame(rows)


class FakeTabICL:
    def __init__(self, **kwargs):
        self.classes_ = np.array([0, 1])

    def fit(self, x, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, x):
        return np.tile([1 - self.p, self.p], (len(x), 1))


def test_tabicl_benchmark_uses_train_and_validation_only():
    frame = make_frame()
    splits, _ = prepare_split(
        frame, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2
    )
    results, metadata = run_tabicl(
        frame, splits, seed=7, n_estimators=1, estimator_class=FakeTabICL
    )
    assert len(results) == len(FEATURE_SETS) == 2
    assert set(results["model"]) == {"tabicl_v2"}
    assert metadata["test_partition_used"] is False
    assert np.isfinite(results["average_precision"]).all()
