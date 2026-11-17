"""FastAPI service for the SmartLoan research-lab demo application.

Fits the preselected sigmoid-calibrated LightGBM on the frozen train+validation
rows only. The held-out test partition is never loaded for model fitting.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from research_lab.calibration_audit import build_base_estimator
from research_lab.optimize import FEATURE_SETS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/loan_data.csv"
SPLIT_PATH = ROOT / "data/processed/split_indices.csv"
FEATURES = FEATURE_SETS["full_features"]
_model = None
_model_error = None
_fit_seconds = None
_fit_rows = 0
_reference_frame = None
_class_counts = {}

class Applicant(BaseModel):
    person_age: float = Field(gt=17, lt=84)
    person_gender: str
    person_education: str
    person_income: float = Field(gt=0)
    person_emp_exp: int = Field(ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(gt=0)
    loan_intent: str
    loan_int_rate: float = Field(gt=0)
    cb_person_cred_hist_length: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=900)
    previous_loan_defaults_on_file: str

    @model_validator(mode="after")
    def validate_experience(self):
        if self.person_emp_exp > self.person_age - 18:
            raise ValueError("Employment experience is inconsistent with applicant age.")
        return self

    def features(self):
        values = self.model_dump()
        values["loan_percent_income"] = self.loan_amnt / self.person_income
        return values

class WhatIfRequest(BaseModel):
    applicant: Applicant
    person_income: float | None = Field(default=None, gt=0)
    loan_amnt: float | None = Field(default=None, gt=0)
    loan_int_rate: float | None = Field(default=None, gt=0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_error, _fit_seconds
    try:
        if not DATA_PATH.exists() or not SPLIT_PATH.exists():
            raise FileNotFoundError(
                "Missing data/raw/loan_data.csv or data/processed/split_indices.csv. "
                "Run the documented data audit and split steps first."
            )
        frame = pd.read_csv(DATA_PATH)
        split = pd.read_csv(SPLIT_PATH)
        required = set(FEATURES + [TARGET])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing required model columns: {missing}")
        if not {"original_row_index", "split", TARGET}.issubset(split.columns):
            raise ValueError("Split file is missing required index/split/target columns.")
        assignments = split[split["split"].isin(["train", "validation"])].copy()
        indices = assignments["original_row_index"].astype(int).to_numpy()
        if len(indices) == 0:
            raise ValueError("No training/validation rows in saved split.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate row indices found in the fit partition.")
        if indices.min() < 0 or indices.max() >= len(frame):
            raise ValueError("Saved split indices fall outside the dataset.")
        if not np.array_equal(frame.iloc[indices][TARGET].to_numpy(), assignments[TARGET].to_numpy()):
            raise ValueError("Saved split labels do not match the raw dataset at those indices.")
        x_fit = frame.iloc[indices][FEATURES].copy()
        y_fit = frame.iloc[indices][TARGET].astype(int).to_numpy()
        if set(np.unique(y_fit)) != {0, 1}:
            raise ValueError("The fit partition must contain both target classes.")
        global _fit_rows, _reference_frame, _class_counts
        _fit_rows = int(len(indices))
        _reference_frame = frame.iloc[indices].copy()
        _class_counts = {str(int(k)): int(v) for k, v in pd.Series(y_fit).value_counts().items()}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        estimator = CalibratedClassifierCV(
            estimator=build_base_estimator(FEATURES, seed=42),
            method="sigmoid", cv=cv, ensemble=True, n_jobs=1,
        )
        started = time.perf_counter()
        estimator.fit(x_fit, y_fit)
        _fit_seconds = round(time.perf_counter() - started, 2)
        _model = estimator
    except Exception as exc:
        _model_error = str(exc)
    yield
    _model = None

app = FastAPI(
    title="SmartLoan Research Lab API",
    version="1.0.0",
    description="Academic demo predicting the supplied loan_status dataset label.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost):517[0-9]",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_available": _model is not None,
        "model_error": _model_error,
        "fit_seconds": _fit_seconds,
        "model": "sigmoid_calibrated_lightgbm_full_features",
        "fit_partition": "train_plus_validation_only",
        "fit_rows": _fit_rows,
        "target_class_counts": _class_counts,
        "target_mapping": {"0": "rejected (working interpretation)", "1": "approved (working interpretation)"},
        "probability_note": "Model estimate of the synthetic dataset label, not a real-world approval probability.",
    }

@app.post("/api/predict")
def predict(applicant: Applicant):
    if _model is None:
        raise HTTPException(503, _model_error or "Model is still loading.")
    features = applicant.features()
    input_frame = pd.DataFrame([features], columns=FEATURES)
    probabilities = _model.predict_proba(input_frame)[0]
    classes = [int(value) for value in _model.classes_]
    probability_by_class = {str(label): float(probabilities[index]) for index, label in enumerate(classes)}
    if "1" not in probability_by_class or not np.isfinite(probabilities).all():
        raise HTTPException(500, "The model returned an invalid probability vector.")
    probability = probability_by_class["1"]
    warnings = []
    reference = _reference_frame
    for field, label in (("person_income", "Annual income"), ("loan_amnt", "Loan amount"),
                         ("credit_score", "Credit score"), ("cb_person_cred_hist_length", "Credit history")):
        series = pd.to_numeric(reference[field], errors="coerce").dropna()
        value = float(features[field])
        low, high = float(series.quantile(.01)), float(series.quantile(.99))
        if value < low or value > high:
            warnings.append(f"{label} is outside the central 98% of values in the model's reference data.")
    loan_ratio = float(features["loan_percent_income"])
    ratio_ref = pd.to_numeric(reference["loan_percent_income"], errors="coerce").dropna()
    if loan_ratio > float(ratio_ref.quantile(.99)):
        warnings.append("Loan-to-income ratio is unusually high relative to the model's reference data.")
    category_context = {}
    for field in ("person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"):
        grouped = reference.groupby(field, dropna=False)[TARGET].agg(["count", "mean"])
        key = str(features[field])
        if key in grouped.index:
            category_context[field] = {
                "value": key,
                "sample_count": int(grouped.loc[key, "count"]),
                "observed_class_1_rate": float(grouped.loc[key, "mean"]),
            }
    return {
        "class_1_probability": probability,
        "class_0_probability": probability_by_class.get("0"),
        "threshold": 0.5,
        "decision": "CLASS_1_MORE_LIKELY" if probability >= 0.5 else "CLASS_0_MORE_LIKELY",
        "target_label": "loan_status",
        "model": "sigmoid_calibrated_lightgbm_full_features",
        "input_checks": {"warnings": warnings, "loan_percent_income": loan_ratio},
        "dataset_context": category_context,
        "disclaimer": (
            "This is an estimate of the recorded label in a synthetic/augmented "
            "research dataset. It is not a real-world loan approval probability, "
            "credit decision, or prediction of repayment/default."
        ),
    }

@app.post("/api/what-if")
def what_if(request: WhatIfRequest):
    if _model is None:
        raise HTTPException(503, _model_error or "Model is still loading.")
    baseline = request.applicant.features()
    baseline_probability = float(_model.predict_proba(pd.DataFrame([baseline])[FEATURES])[0, 1])
    changes = {
        key: value for key, value in {
            "person_income": request.person_income,
            "loan_amnt": request.loan_amnt,
            "loan_int_rate": request.loan_int_rate,
        }.items() if value is not None
    }
    if not changes:
        raise HTTPException(422, "Provide at least one scenario change.")
    scenario = dict(baseline)
    scenario.update(changes)
    scenario["loan_percent_income"] = scenario["loan_amnt"] / scenario["person_income"]
    probability = float(_model.predict_proba(pd.DataFrame([scenario])[FEATURES])[0, 1])
    return {
        "baseline_probability": baseline_probability,
        "scenario_probability": probability,
        "difference": probability - baseline_probability,
        "disclaimer": "Hypothetical model sensitivity only; not a causal recommendation or lending decision.",
    }
