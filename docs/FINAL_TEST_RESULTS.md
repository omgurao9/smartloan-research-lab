# Final Test Evaluation (FINAL_TEST_01)

## Status

Completed. The frozen test set was evaluated once after model selection and validation analysis were complete. No model or threshold selection was performed using test results.

## Predeclared evaluation

- Primary candidate: sigmoid-calibrated LightGBM using all 13 predictors.
- Sensitivity candidate: sigmoid-calibrated LightGBM using the conservative 11-predictor set, excluding `previous_loan_defaults_on_file` and `loan_int_rate`.
- Fit data: 38,242 rows (training + validation).
- Test data: 6,749 rows (2,249 approved / 5,249 rejected; see confusion matrices below).
- Calibration: sigmoid calibration with 5-fold stratified CV on the combined training + validation data.
- Decision threshold: fixed at 0.5; not optimized on test.
- Bootstrap: 500 stratified bootstrap resamples; percentile 95% intervals for average precision, ROC-AUC, and Brier score.

The two models were evaluated against the same frozen test partition. Bootstrap intervals are conditional on the fitted models and fixed split; they do not capture training-seed variation or distribution shift.

## Test metrics

| Metric | Full features (13) | Conservative (11) |
|---|---:|---:|
| Average precision (PR-AUC) | **0.939194** | 0.800031 |
| ROC-AUC | **0.978665** | 0.894715 |
| Accuracy at 0.5 | **0.932879** | 0.883094 |
| Balanced accuracy at 0.5 | **0.888045** | 0.758189 |
| Precision at 0.5 | 0.880727 | **0.899888** |
| Recall at 0.5 | **0.807333** | 0.533333 |
| Specificity at 0.5 | 0.968756 | **0.983044** |
| F1 at 0.5 | **0.842435** | 0.669736 |
| MCC at 0.5 | **0.801116** | 0.634822 |
| Log loss | **0.149439** | 0.291641 |
| Brier score | **0.047455** | 0.088506 |

## Confusion matrices

Rows are actual classes; columns are predicted classes. Class 0 is rejected and class 1 is approved under the working label interpretation used in this study.

### Full features

| | Predicted rejected (0) | Predicted approved (1) |
|---|---:|---:|
| Actual rejected (0) | 5,085 | 164 |
| Actual approved (1) | 289 | 1,211 |

### Conservative features

| | Predicted rejected (0) | Predicted approved (1) |
|---|---:|---:|
| Actual rejected (0) | 5,160 | 89 |
| Actual approved (1) | 700 | 800 |

## 95% stratified bootstrap intervals

| Metric | Full features | Conservative |
|---|---:|---:|
| Average precision | 0.930651–0.946507 | 0.784936–0.815357 |
| ROC-AUC | 0.975425–0.981268 | 0.884775–0.903339 |
| Brier score | 0.044265–0.051055 | 0.084663–0.092277 |

## Interpretation

1. The full-feature model is the stronger predictive benchmark on this held-out test split. It achieves PR-AUC 0.9392, ROC-AUC 0.9787, and Brier score 0.0475.
2. The conservative sensitivity model remains materially weaker (PR-AUC 0.8000). Removing the two features is associated with a 0.1392 absolute reduction in test PR-AUC and a reduction in recall from 0.8073 to 0.5333 at the fixed 0.5 threshold.
3. The conservative model has higher precision and specificity at this threshold, but misses more positive-class examples. Threshold metrics therefore reflect a different precision/recall trade-off, not an overall superiority.
4. These results quantify predictive association in this dataset. They do not establish causal effects, lending-policy validity, or performance on real applicants.

## Limitations and claims boundary

- The dataset is synthetic/augmented benchmark data; external validity to real lending populations is not established.
- The provenance and generation process for the labels and predictors remain incompletely verified.
- The feature `previous_loan_defaults_on_file` has a striking one-way relationship with the target in this dataset. Whether this reflects target leakage, a generated rule, or a genuine label-construction mechanism is unresolved.
- The timing and availability of `loan_int_rate` relative to the approval decision are not established. It may be unsuitable for a pre-decision model if it is only assigned after approval.
- Consequently, the full-feature result is the best-performing benchmark within this experiment, not a validated deployable credit-approval system.
- No real-world fairness, causal, repayment/default prediction, or deployment-readiness claims are supported by this dataset alone.
- Test results must not be used for further feature, model, or threshold tuning. Any new modeling iteration requires a new untouched evaluation dataset or a separately predeclared validation design.

## Reproducibility

Command used:

```powershell
.\.venv\Scripts\python.exe -m research_lab.final_evaluation --cv-folds 5 --bootstrap-repeats 500
```

Local outputs are saved under the gitignored directory `experiments/results/final_test/`. The run metadata records that model selection was frozen before test evaluation, the test set was not used for fitting or selection, and the test set was evaluated once.
