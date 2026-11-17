# Research Protocol

## Objective

Compare tabular classifiers for the observed `loan_status` label while investigating data validity, feature timing, label semantics, predictive performance, calibration, subgroup behavior, computational cost, and reproducibility.

## Claims policy

Until the dataset source and label definition are independently verified, describe the task as **loan-status label classification**. Do not call the positive class an approved loan, future default, repayment, or credit-risk event unless the source documentation establishes that meaning.

## Phase gates

1. **Data audit:** schema, types, missingness, duplicates, impossible values, class balance, feature-target cross-tabs, and cleaning impact.
2. **Provenance review:** document what each feature means and when it becomes available relative to the prediction point. Flag possible target leakage; do not infer leakage from correlation alone.
3. **Split protocol:** create one stratified train/validation/test split with a fixed seed. Save row identifiers or indices so every experiment uses identical partitions. Do not use the test set for tuning, feature selection, threshold choice, or model selection.
4. **Baseline benchmark:** simple linear and tree models plus gradient-boosted trees. Fit all preprocessing within each training fold.
5. **Optimization:** tune only on training data using cross-validation; use validation for the predeclared model-selection decision.
6. **Calibration and decision threshold:** fit calibration without test labels. Choose operating threshold on validation according to an explicitly stated objective or constraints.
7. **Final evaluation:** open test data once after choices are frozen. Report confidence intervals and avoid iterative test-driven changes.
8. **Robustness:** feature ablation, subgroup analysis, seed sensitivity, calibration, and runtime/memory.
9. **Advanced models:** add CatBoost, LightGBM, TabM, TabICL, or TabPFN only after baseline and data protocol are stable. Record package versions, hardware, licenses, and feasibility.

## Metrics

Report PR-AUC (average precision), ROC-AUC, balanced accuracy, precision, recall, specificity, F1, Matthews correlation coefficient, log loss, Brier score, and confusion matrix. Include calibration curves and subgroup counts/metrics where sample sizes permit. Accuracy alone is insufficient for the observed class imbalance.

## Critical inherited data flag

The previous project audit observed that every class-1 row had `previous_loan_defaults_on_file = No`, and the Yes group contained no class-1 rows. This must be rechecked from the raw file. Determine feature provenance and timing. Run a predeclared feature-ablation experiment with and without this feature. A strong association alone does not prove leakage.

## Reproducibility

Every experiment must record a unique ID, timestamp, code revision, config, random seed, dataset fingerprint, split fingerprint, dependency versions, hardware, metrics, runtime, and artifact locations. Never overwrite a prior result; write a new experiment record.
