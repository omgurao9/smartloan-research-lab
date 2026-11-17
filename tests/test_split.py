import pandas as pd
import pytest

from research_lab.split import prepare_split


def sample_frame():
    rows = []
    for index in range(40):
        rows.append({
            "person_age": 25 + (index % 10),
            "person_gender": "female" if index % 2 else "male",
            "person_education": "Bachelor",
            "person_income": 50000 + index,
            "person_emp_exp": 3,
            "person_home_ownership": "RENT",
            "loan_amnt": 10000,
            "loan_intent": "EDUCATION",
            "loan_int_rate": 10.0,
            "loan_percent_income": 0.2,
            "cb_person_cred_hist_length": 4,
            "credit_score": 650,
            "previous_loan_defaults_on_file": "No",
            "loan_status": index % 2,
        })
    return pd.DataFrame(rows)


def test_split_is_stratified_disjoint_and_reproducible():
    frame = sample_frame()
    first, metadata = prepare_split(frame)
    second, _ = prepare_split(frame)
    pd.testing.assert_frame_equal(first, second)
    assert metadata["rows_after_cleaning"] == 40
    assert first["original_row_index"].is_unique
    assert first["split"].value_counts().to_dict() == {
        "train": 28, "validation": 6, "test": 6
    }
    assert set(first["split"]) == {"train", "validation", "test"}


def test_split_applies_documented_cleaning_rule():
    frame = sample_frame()
    frame.loc[0, "person_age"] = 84
    assignment, metadata = prepare_split(frame)
    assert metadata["rows_removed"] == 1
    assert metadata["rows_after_cleaning"] == 39
    assert 0 not in assignment["original_row_index"].tolist()


def test_split_rejects_missing_values():
    frame = sample_frame()
    frame.loc[0, "person_income"] = None
    with pytest.raises(ValueError, match="missing values"):
        prepare_split(frame)
