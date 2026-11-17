# E01: Leakage-Safe Stacking Results

## Status

Completed. The stacking estimator was trained using out-of-fold base-model probabilities generated exclusively within the training partition. The frozen validation partition was evaluated once. The test partition was not used.

## Validation results

| Feature set | Model | Fit + validation (s) | Average precision | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Log loss | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | OOF stacking (HGB + RF + LightGBM) | 22.018 | 0.934490 | 0.975803 | 0.913765 | 0.909087 | 0.757287 | 0.900667 | 0.822777 | 0.771388 | 0.210868 | 0.062805 |
| Conservative | OOF stacking (HGB + RF + LightGBM) | 21.472 | **0.791013** | **0.880591** | 0.822789 | 0.792747 | 0.579498 | 0.738667 | 0.649472 | 0.540225 | 0.399353 | 0.125614 |

Threshold metrics use 0.5. Conservative excludes previous-default history and interest rate.

## Comparison with individual-model references

| Feature set | Stacking AP | Strong individual reference | Individual AP | Difference |
|---|---:|---|---:|---:|
| Full | 0.934490 | F01 sigmoid-calibrated LightGBM | 0.935666 | -0.001176 |
| Conservative | **0.791013** | G01 LightGBM | 0.789088 | +0.001925 |

For the full feature set, stacking does not exceed the best reported individual reference. For the conservative feature set, stacking has a small average-precision advantage over the G01 LightGBM result. This is a comparison on the same frozen validation partition, but it is not a paired significance test and does not establish a reliable improvement.

The full-feature stacking Brier score (0.062805) is worse than the F01 calibrated LightGBM Brier scores (0.048525 sigmoid; 0.048624 isotonic). The conservative stacking Brier score (0.125614) is also worse than the F01 calibrated LightGBM scores (0.090927 sigmoid; 0.090770 isotonic). Stacking therefore does not replace the calibrated models as the leading probability-quality candidates.

## Interpretation

1. **Ranking:** the stack is competitive. It is slightly behind the best calibrated full-feature reference and edges the earlier uncalibrated LightGBM conservative reference by a small margin.
2. **Threshold behavior:** the stack has relatively high recall at a 0.5 threshold in both feature sets, with lower precision. A 0.5 threshold is a reporting convention, not an established operational decision threshold.
3. **Probability quality:** Brier and log-loss values do not support selecting the stack as the best-calibrated candidate. A separate, training-only calibration study would be required before making any such claim.
4. **Complexity:** each feature-set run took approximately 21–22 seconds in the user's reported environment. The stack combines three base learners and adds computational and maintenance complexity.
5. **Model selection:** retain stacking as a competitive ensemble benchmark, especially for the conservative feature set, but do not declare it a decisive winner. Keep the calibrated LightGBM benchmark as the stronger probability-quality reference.

## Reproducibility and isolation

- Base estimators: HistGradientBoosting, Random Forest, and LightGBM.
- Meta-estimator: class-weighted Logistic Regression.
- Meta-features: out-of-fold base-model probabilities only; original features are not passed through.
- Cross-validation: five-fold stratified CV, seed 42, within training only.
- Frozen train/validation split and feature sets shared with prior experiments.
- Test partition used: **False**.
- Local artifacts: `experiments/results/stacking/validation_metrics.csv` and `experiments/results/stacking/run_metadata.json`.

## Limitations

This is a single frozen validation split. No confidence intervals or statistical tests were computed. Small metric differences may reflect sampling variation. The dataset remains synthetic/augmented with unresolved feature provenance and feature-timing questions. The target is the recorded `loan_status` label, not future repayment/default. No causal, real-world fairness, or deployment claims are supported.
