from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SmartLoan Research Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: #111827; border: 1px solid #263244; padding: 16px; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #aab7ca;}
[data-testid="stMetricValue"] {color: #f8fafc;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(115deg,#10243b,#183c54); border: 1px solid #31536c; margin-bottom: 1.3rem;}
.hero h1 {color: #f8fafc; margin: 0 0 .3rem 0; font-size: 2.1rem;}
.hero p {color: #c7d8e8; margin: 0;}
.small-note {color: #9caec1; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

FULL = {
    "Average precision (PR-AUC)": 0.939194, "ROC-AUC": 0.978665,
    "Accuracy": 0.932879, "Balanced accuracy": 0.888045,
    "Precision": 0.880727, "Recall": 0.807333, "Specificity": 0.968756,
    "F1-score": 0.842435, "MCC": 0.801116, "Log loss": 0.149439,
    "Brier score": 0.047455,
}
CONS = {
    "Average precision (PR-AUC)": 0.800031, "ROC-AUC": 0.894715,
    "Accuracy": 0.883094, "Balanced accuracy": 0.758189,
    "Precision": 0.899888, "Recall": 0.533333, "Specificity": 0.983044,
    "F1-score": 0.669736, "MCC": 0.634822, "Log loss": 0.291641,
    "Brier score": 0.088506,
}
CI = {
    "Full features": {
        "PR-AUC": (0.930651, 0.946507),
        "ROC-AUC": (0.975425, 0.981268),
        "Brier score": (0.044265, 0.051055),
    },
    "Conservative": {
        "PR-AUC": (0.784936, 0.815357),
        "ROC-AUC": (0.884775, 0.903339),
        "Brier score": (0.084663, 0.092277),
    },
}
CONFUSION = {
    "Full features": [[5085, 164], [289, 1211]],
    "Conservative": [[5160, 89], [700, 800]],
}

st.sidebar.title("SMARTLOAN")
st.sidebar.caption("RESEARCH LAB  /  MODEL EVALUATION")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model comparison", "Confusion matrices", "Calibration & uncertainty", "Research notes"],
)
st.sidebar.divider()
st.sidebar.markdown("**Evaluation status**")
st.sidebar.success("Final test completed")
st.sidebar.caption("Frozen split • 5-fold sigmoid calibration • 500 bootstrap resamples")

st.markdown("""
<div class="hero">
  <h1>SmartLoan Research Lab</h1>
  <p>Explainable model research · validation comparisons · final held-out evaluation</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic research prototype. The model estimates the dataset's recorded "
    "`loan_status` label; it is not a real lending decision system or a future-default predictor."
)

if page == "Overview":
    st.subheader("Final evaluation at a glance")
    st.caption("Held-out test set · 6,749 observations · threshold fixed at 0.50")
    a, b, c, d = st.columns(4)
    a.metric("Full model PR-AUC", "0.9392")
    b.metric("Full model ROC-AUC", "0.9787")
    c.metric("Full model Brier", "0.0475")
    d.metric("Test observations", "6,749")

    st.subheader("The two final candidates")
    left, right = st.columns(2)
    with left:
        st.markdown("### Full-feature model")
        st.metric("PR-AUC", "0.9392")
        st.caption("Sigmoid-calibrated LightGBM · 13 predictors")
        st.markdown("Best-performing benchmark in this experiment.")
    with right:
        st.markdown("### Conservative sensitivity model")
        st.metric("PR-AUC", "0.8000")
        st.caption("Sigmoid-calibrated LightGBM · 11 predictors")
        st.markdown("Excludes previous-default history and interest rate.")

    st.subheader("Discrimination comparison")
    chart = pd.DataFrame({
        "Model": ["Full features", "Conservative"],
        "PR-AUC": [FULL["Average precision (PR-AUC)"], CONS["Average precision (PR-AUC)"]],
        "ROC-AUC": [FULL["ROC-AUC"], CONS["ROC-AUC"]],
    }).melt("Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(chart, x="Metric", y="Score", color="Model", barmode="group",
                           range_y=[0, 1], text_auto=".3f",
                           color_discrete_sequence=["#2dd4bf", "#60a5fa"]),
                    use_container_width=True)

    st.info(
        "The full model leads on PR-AUC, ROC-AUC, recall, F1, MCC, log loss and Brier score. "
        "The conservative model has higher precision and specificity at the fixed 0.5 threshold."
    )

elif page == "Model comparison":
    st.subheader("Held-out test metrics")
    rows = []
    for metric in FULL:
        rows.append({"Metric": metric, "Full features": FULL[metric], "Conservative": CONS[metric]})
    df = pd.DataFrame(rows).set_index("Metric")
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    metric = st.selectbox("Visualize metric", list(FULL.keys()), index=0)
    comp = pd.DataFrame({"Model": ["Full features", "Conservative"],
                         "Score": [FULL[metric], CONS[metric]]})
    st.plotly_chart(px.bar(comp, x="Model", y="Score", color="Model", text_auto=".4f",
                           color_discrete_sequence=["#2dd4bf", "#60a5fa"]),
                    use_container_width=True)
    st.caption("Threshold-dependent metrics use the predeclared 0.50 threshold. No test-driven threshold selection.")

elif page == "Confusion matrices":
    st.subheader("Classification outcomes")
    model = st.radio("Select model", ["Full features", "Conservative"], horizontal=True)
    matrix = CONFUSION[model]
    fig = px.imshow(matrix, text_auto=True, aspect="auto",
                    x=["Predicted rejected (0)", "Predicted approved (1)"],
                    y=["Actual rejected (0)", "Actual approved (1)"],
                    color_continuous_scale="Teal",
                    labels=dict(x="Prediction", y="Actual", color="Count"))
    fig.update_layout(title=f"{model} · confusion matrix", height=430)
    st.plotly_chart(fig, use_container_width=True)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    x, y = st.columns(2)
    x.metric("Correct classifications", f"{tn + tp:,}")
    y.metric("Incorrect classifications", f"{fp + fn:,}")
    st.caption("Class 0 = rejected; class 1 = approved (working interpretation of the dataset label).")

elif page == "Calibration & uncertainty":
    st.subheader("Probability quality and bootstrap intervals")
    st.markdown("Lower Brier score and log loss indicate better probabilistic accuracy.")
    x, y = st.columns(2)
    x.metric("Full Brier score", "0.0475", help="95% bootstrap CI: 0.0443–0.0511")
    y.metric("Conservative Brier score", "0.0885", help="95% bootstrap CI: 0.0847–0.0923")
    interval_rows = []
    for model, metrics in CI.items():
        for metric, (lo, hi) in metrics.items():
            interval_rows.append({"Model": model, "Metric": metric, "Lower 95%": lo, "Upper 95%": hi})
    st.dataframe(pd.DataFrame(interval_rows).style.format(
        {"Lower 95%": "{:.4f}", "Upper 95%": "{:.4f}"}), use_container_width=True)
    selected = st.selectbox("Confidence interval metric", ["PR-AUC", "ROC-AUC", "Brier score"])
    fig = go.Figure()
    for name, color in [("Full features", "#2dd4bf"), ("Conservative", "#60a5fa")]:
        lo, hi = CI[name][selected]
        mid = (lo + hi) / 2
        fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines",
                                 line=dict(width=7, color=color), showlegend=False))
        fig.add_trace(go.Scatter(x=[mid], y=[name], mode="markers",
                                 marker=dict(size=12, color=color), name=name))
    fig.update_layout(title=f"95% stratified bootstrap interval · {selected}",
                      xaxis_title=selected, height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("500 stratified bootstrap resamples. Intervals are conditional on the fitted models and fixed split; they do not capture training-seed variation or distribution shift.")

elif page == "Research notes":
    st.subheader("Experiment journey")
    milestones = [
        ("A00 / S00", "Data audit and frozen stratified 70/15/15 split", "Completed"),
        ("B01 / B02 / V01", "Baseline models and feature-validity ablations", "Completed"),
        ("O01", "Optuna tuning of HistGradientBoosting", "Completed"),
        ("G01 / G02", "LightGBM, CatBoost and XGBoost benchmarks", "Completed"),
        ("F01", "Sigmoid/isotonic calibration and descriptive subgroup audit", "Completed"),
        ("E01", "Out-of-fold stacking comparison", "Completed"),
        ("N01", "TabM completed; TabICL inference infeasible on available CPU", "Partially completed"),
        ("FINAL_TEST_01", "Frozen final test evaluation and bootstrap intervals", "Completed"),
    ]
    for code, desc, status in milestones:
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 4, 1])
            col1.markdown(f"**{code}**")
            col2.write(desc)
            col3.caption(status)

    st.subheader("Important limitations")
    st.markdown("""
- The dataset is synthetic/augmented; external validity to real lending populations is not established.
- Label provenance and feature-generation timing remain incompletely verified.
- `previous_loan_defaults_on_file` has a striking one-way relationship with the target. Whether this is leakage, a generated rule, or a dataset construction artifact remains unresolved.
- The timing of `loan_int_rate` relative to the approval decision is not established.
- These results do not establish causality, real-world fairness, repayment/default prediction, or deployment readiness.
- The test set was evaluated once. Do not use it for further model or threshold tuning.
""")
    st.markdown("**Reproducibility:** final fit used 38,242 train + validation rows; test contained 6,749 rows; 5-fold sigmoid calibration; threshold 0.50; 500 stratified bootstrap repeats.")

st.divider()
st.markdown(
    '<p class="small-note">SMARTLOAN RESEARCH LAB · Final held-out evaluation · '
    'Results documented in docs/FINAL_TEST_RESULTS.md</p>',
    unsafe_allow_html=True,
)
