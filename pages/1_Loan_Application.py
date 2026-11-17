from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from research_lab.calibration_audit import build_base_estimator
from research_lab.optimize import FEATURE_SETS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/loan_data.csv"
SPLIT_PATH = ROOT / "data/processed/split_indices.csv"
FEATURES = FEATURE_SETS["full_features"]
CATEGORICAL = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

st.set_page_config(page_title="Loan Assessment | SmartLoan", page_icon="🏦", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem;}
.hero {background:linear-gradient(115deg,#10243b,#17435a);padding:1.5rem 1.8rem;border-radius:16px;border:1px solid #31536c;margin-bottom:1.2rem}
.hero h1 {color:#fff;margin:0 0 .3rem 0}
.hero p {color:#d1e2ef;margin:0}
.result {padding:1.2rem;border-radius:14px;border:1px solid #31536c;background:#102b3c}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_reference_data():
    if not DATA_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Required local files are missing. Expected data/raw/loan_data.csv "
            "and data/processed/split_indices.csv. Run the documented audit and split steps first."
        )
    return pd.read_csv(DATA_PATH), pd.read_csv(SPLIT_PATH)

@st.cache_resource(show_spinner="Training the selected calibrated LightGBM model on training + validation data...")
def load_demo_model():
    frame, split = load_reference_data()
    assignments = split[split["split"].isin(["train", "validation"])]
    indices = assignments["original_row_index"].astype(int).to_numpy()
    x_fit = frame.iloc[indices][FEATURES]
    y_fit = frame.iloc[indices][TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = CalibratedClassifierCV(
        estimator=build_base_estimator(FEATURES, seed=42),
        method="sigmoid",
        cv=cv,
        ensemble=True,
        n_jobs=1,
    )
    model.fit(x_fit, y_fit)
    return model, frame

st.markdown("""
<div class="hero">
  <h1>SmartLoan · Loan Assessment</h1>
  <p>Applicant intake and ML-assisted loan-status estimate</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Academic demonstration only. This form estimates the dataset's recorded "
    "`loan_status` label. It is not an actual bank approval, credit decision, "
    "or prediction of future repayment/default."
)

try:
    with st.spinner("Loading dataset reference values..."):
        frame, _ = load_reference_data()
except Exception as exc:
    st.error(str(exc))
    st.markdown("From the repository root, first run the data audit and split commands in README.md.")
    st.stop()

def choices(column: str):
    return sorted(frame[column].dropna().astype(str).unique().tolist())

st.subheader("Applicant information")
st.caption("Enter the applicant details below. Required fields are marked by the form controls.")

with st.form("loan_application_form", clear_on_submit=False):
    st.markdown("#### Personal details")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", min_value=18, max_value=83, value=30, step=1)
    gender = c2.selectbox("Gender", choices("person_gender"))
    education = c3.selectbox("Education", choices("person_education"))

    st.markdown("#### Employment and finances")
    c1, c2, c3 = st.columns(3)
    income_min = max(1, int(frame["person_income"].min()))
    income_max = max(income_min + 1, int(frame["person_income"].max()))
    income = c1.number_input("Annual income", min_value=income_min, max_value=income_max,
                             value=int(frame["person_income"].median()), step=1000)
    exp = c2.number_input("Employment experience (years)", min_value=0,
                          max_value=max(0, int(age) - 18), value=min(5, max(0, int(age) - 18)), step=1)
    home = c3.selectbox("Home ownership", choices("person_home_ownership"))

    st.markdown("#### Loan details")
    c1, c2, c3 = st.columns(3)
    loan_min = max(1, int(frame["loan_amnt"].min()))
    loan_max = max(loan_min + 1, int(frame["loan_amnt"].max()))
    amount = c1.number_input("Requested loan amount", min_value=loan_min, max_value=loan_max,
                             value=int(frame["loan_amnt"].median()), step=500)
    intent = c2.selectbox("Loan purpose", choices("loan_intent"))
    rate = c3.number_input("Interest rate (%)", min_value=0.1, max_value=100.0,
                           value=float(frame["loan_int_rate"].median()), step=0.1)

    st.markdown("#### Credit profile")
    c1, c2, c3 = st.columns(3)
    history = c1.number_input("Credit history length (years)", min_value=0.0, max_value=80.0,
                              value=float(frame["cb_person_cred_hist_length"].median()), step=0.5)
    score = c2.number_input("Credit score", min_value=300, max_value=900,
                            value=int(frame["credit_score"].median()), step=1)
    defaults = c3.selectbox("Previous loan defaults on file",
                            choices("previous_loan_defaults_on_file"))

    st.caption("Loan-to-income ratio is calculated automatically from the requested amount and annual income.")
    submitted = st.form_submit_button("Assess application", type="primary", use_container_width=True)

if submitted:
    if exp > age - 18:
        st.error("Employment experience cannot exceed the number of years since age 18.")
        st.stop()
    if income <= 0:
        st.error("Annual income must be greater than zero.")
        st.stop()

    applicant = {
        "person_age": float(age),
        "person_gender": gender,
        "person_education": education,
        "person_income": float(income),
        "person_emp_exp": int(exp),
        "person_home_ownership": home,
        "loan_amnt": float(amount),
        "loan_intent": intent,
        "loan_int_rate": float(rate),
        "loan_percent_income": float(amount) / float(income),
        "cb_person_cred_hist_length": float(history),
        "credit_score": int(score),
        "previous_loan_defaults_on_file": defaults,
    }
    try:
        with st.spinner("Preparing the calibrated model and assessing the application..."):
            model, _ = load_demo_model()
            probability = float(model.predict_proba(pd.DataFrame([applicant])[FEATURES])[:, 1][0])
        predicted_class = int(probability >= 0.5)
        label = "Approved" if predicted_class == 1 else "Rejected"
        st.divider()
        st.subheader("Assessment result")
        col1, col2 = st.columns([1, 2])
        with col1:
            if predicted_class == 1:
                st.success(f"Predicted label: {label}")
            else:
                st.error(f"Predicted label: {label}")
            st.metric("Estimated class-1 probability", f"{probability:.1%}")
            st.caption("Fixed research threshold: 50%")
        with col2:
            st.markdown("#### Application summary")
            st.write(f"**Applicant:** {age} years · {education} · {home}")
            st.write(f"**Annual income:** {income:,.0f}")
            st.write(f"**Requested amount:** {amount:,.0f}")
            st.write(f"**Loan-to-income ratio:** {amount / income:.3f}")
            st.write(f"**Credit score:** {score}")
        st.warning(
            "This is a model classification of the benchmark dataset label, using the "
            "working interpretation that class 1 = Approved and class 0 = Rejected. "
            "It is not an actual lender decision or an offer of credit."
        )
    except Exception as exc:
        st.error(f"Could not run the model: {exc}")
        st.exception(exc)

with st.expander("About this model and its limitations"):
    st.markdown("""
- Model: sigmoid-calibrated LightGBM, using the full 13-feature research configuration.
- Training data: frozen training + validation partitions; five-fold stratified sigmoid calibration.
- Threshold: fixed at 0.50, matching the final test protocol.
- The held-out test set is not used by this application.
- Previous-default history and interest rate have unresolved feature-timing/provenance questions.
- The dataset is synthetic/augmented. The output must not be used for real lending, financial eligibility, or adverse-action decisions.
""")
