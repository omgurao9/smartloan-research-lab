# F01: Calibration and Descriptive Subgroup Audit Protocol

## Objective

Assess whether LightGBM probabilities benefit from post-hoc calibration and describe validation-set performance variation across available demographic and application-feature slices. This is a descriptive benchmark audit, not a legal fairness assessment.

## Models and calibration

- Use the same fixed LightGBM configuration as G01.
- Evaluate full and conservative feature sets.
- Compare sigmoid (Platt-style) and isotonic calibration.
- Fit calibration using stratified cross-validation entirely inside the training partition (five folds by default).
- Use the frozen validation partition once for reported metrics and calibration curves.
- Do not use validation labels to fit the calibrator or choose calibration method.
- The test partition remains unopened.

The calibrated estimator uses cross-validated base estimators; validation is not passed as an early-stopping or calibration set.

## Metrics and artifacts

Report average precision, ROC-AUC, Brier score, log loss, and threshold-0.5 metrics. Export quantile-bin reliability-curve points and descriptive subgroup summaries for gender, education, home ownership, loan intent, age bands, and income quartiles. Suppress subgroup rows below the configured minimum count (30 by default).

## Interpretation constraints

- Calibration may improve probability quality while changing ranking metrics; interpret both separately.
- Small subgroup counts make estimates unstable. Always report counts and positive counts.
- Observed differences are descriptive and may reflect sample composition, label construction, or feature behavior. They do not establish causation, unlawful discrimination, or fairness.
- This dataset's synthetic/augmented provenance and unresolved feature timing remain major limitations.
- Do not use subgroup results to make deployment or lending decisions.
