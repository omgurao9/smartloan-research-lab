import pandas as pd

from research_lab.audit import audit_dataframe


def sample_frame():
    return pd.DataFrame({
        "person_age": [25, 30, 90],
        "person_gender": ["female", "male", "female"],
        "person_education": ["Bachelor", "Master", "Bachelor"],
        "person_income": [50000, 70000, 60000],
        "person_emp_exp": [3, 8, 70],
        "person_home_ownership": ["RENT", "OWN", "RENT"],
        "loan_amnt": [10000, 15000, 10000],
        "loan_intent": ["EDUCATION", "MEDICAL", "EDUCATION"],
        "loan_int_rate": [10.0, 12.0, 11.0],
        "loan_percent_income": [0.2, 0.21, 0.16],
        "cb_person_cred_hist_length": [4, 7, 5],
        "credit_score": [650, 700, 680],
        "previous_loan_defaults_on_file": ["No", "No", "Yes"],
        "loan_status": [1, 0, 0],
    })


def test_audit_detects_expected_schema_and_cleaning_impact():
    report = audit_dataframe(sample_frame())
    assert report["rows"] == 3
    assert report["missing_expected_columns"] == []
    assert report["cleaning_rule_impact"]["unique_rows_flagged"] == 1
    assert report["cleaning_rule_impact"]["remaining_rows"] == 2


def test_audit_flags_zero_target_cell_for_previous_default_feature():
    report = audit_dataframe(sample_frame())
    assert any("zero count" in warning for warning in report["warnings"])
    assert report["duplicate_rows"] == 0


def test_audit_reports_missing_columns_and_duplicates():
    frame = sample_frame().drop(columns=["credit_score"])
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = audit_dataframe(frame)
    assert "credit_score" in report["missing_expected_columns"]
    assert report["duplicate_rows"] == 1
