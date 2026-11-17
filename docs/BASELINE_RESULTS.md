# Baseline Validation Results

**Experiments:** B01 (model comparison) and B02 (previous-default feature ablation)  
**Status:** Completed  
**Evaluation partition:** Validation only  
**Validation rows:** 6,749 (1,500 positive; 5,249 negative)  
**Decision threshold for threshold-dependent metrics:** 0.50  
**Test partition used:** No

## Results

| Feature set | Model | Features | Average precision (PR-AUC) | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Log loss | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | HistGradientBoosting | 13 | 0.934302 | 0.975728 | 0.933027 | 0.880283 | 0.900612 | 0.785333 | 0.839031 | 0.799978 | 0.156618 | 0.048914 |
| Full | Random Forest | 13 | 0.922927 | 0.971407 | 0.917321 | 0.898517 | 0.785109 | 0.864667 | 0.822970 | 0.770724 | 0.191684 | 0.061151 |
| Without previous defaults | HistGradientBoosting | 12 | 0.862328 | 0.925432 | 0.908283 | 0.821522 | 0.895067 | 0.665333 | 0.763289 | 0.719910 | 0.244304 | 0.071041 |
| Without previous defaults | Random Forest | 12 | 0.848850 | 0.916883 | 0.895392 | 0.831804 | 0.792342 | 0.717333 | 0.752974 | 0.688195 | 0.289758 | 0.084367 |
| Full | Logistic Regression | 13 | 0.841335 | 0.949086 | 0.850052 | 0.866937 | 0.610708 | 0.897333 | 0.726782 | 0.650628 | 0.287204 | 0.097145 |
| Without previous defaults | Logistic Regression | 12 | 0.671846 | 0.844088 | 0.775078 | 0.769455 | 0.496080 | 0.759333 | 0.600105 | 0.472921 | 0.473209 | 0.155016 |

## Findings

1. HistGradientBoosting with all 13 predictors has the highest validation average precision (0.934302), ROC-AUC (0.975728), and F1 at threshold 0.50 (0.839031) among the six evaluated configurations.
2. Random Forest with all predictors has the highest balanced accuracy (0.898517) and recall (0.864667) at threshold 0.50.
3. Removing `previous_loan_defaults_on_file` reduces HistGradientBoosting average precision from 0.934302 to 0.862328, an absolute decrease of 0.071974. Random Forest decreases from 0.922927 to 0.848850 (0.074077); Logistic Regression decreases from 0.841335 to 0.671846 (0.169489).
4. The ablation establishes that this feature is highly influential for these models on this validation split. It does **not** establish whether the feature is legitimate, available at decision time, or a leakage artifact.

## Interpretation and limitations

- These are validation results for a synthetic/augmented loan-approval benchmark, not evidence of real-world lending or repayment-risk performance.
- The meaning and timing of `previous_loan_defaults_on_file` require further provenance investigation. Its strong association with the target could reflect dataset construction or information unavailable at the time of approval.
- `loan_int_rate` also requires timing review before it can be treated as a valid pre-decision predictor.
- No threshold tuning was performed; threshold-dependent metrics use 0.50.
- The held-out test partition has not been used. Do not use it until model and feature decisions are frozen.

## Next research question

Before hyperparameter optimization, investigate feature validity and decision-time availability. In particular, compare a conservative feature set that excludes potentially post-decision information, and document the rationale for each retained predictor. Keep the current full-feature results as the baseline reference rather than silently replacing them.
