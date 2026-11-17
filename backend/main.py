"""FastAPI service for the SmartLoan research-lab demo application.

Fits the preselected sigmoid-calibrated LightGBM on the frozen train+validation
rows only. The held-out test partition is never loaded for model fitting.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time

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
        assignments = split[split["split"].isin(["train", "validation"])]
        indices = assignments["original_row_index"].astype(int).to_numpy()
        if len(indices) == 0:
            raise ValueError("No training/validation rows in saved split.")
        x_fit = frame.iloc[indices][FEATURES]
        y_fit = frame.iloc[indices][TARGET].astype(int).to_numpy()
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
    }

@app.post("/api/predict")
def predict(applicant: Applicant):
    if _model is None:
        raise HTTPException(503, _model_error or "Model is still loading.")
    features = applicant.features()
    probability = float(_model.predict_proba(pd.DataFrame([features])[FEATURES])[0, 1])
    threshold = 0.5
    return {
        "class_1_probability": probability,
        "threshold": threshold,
        "decision": "APPROVED" if probability >= threshold else "NOT APPROVED",
        "target_label": "loan_status",
        "disclaimer": (
            "Academic estimate of the supplied dataset label only; not a real "
            "lending decision or prediction of future repayment/default."
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
