# F01: Calibration and Descriptive Subgroup Audit Results

## Status

Completed. Sigmoid and isotonic calibration were fitted using stratified cross-validation inside the training partition. The frozen validation partition was used for evaluation. The test partition was not used.

## Validation results

| Feature set | Method | Fit + validation (s) | Average precision | ROC-AUC | Brier score | Log loss | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | Sigmoid | 6.950 | **0.935666** | **0.976384** | **0.048525** | 0.155805 | 0.932434 | **0.881569** | 0.893665 | **0.790000** | 0.838641 | 0.798537 |
| Full | Isotonic | 6.788 | 0.935419 | 0.976293 | 0.048624 | **0.155088** | **0.934953** | 0.879855 | **0.914130** | 0.780667 | **0.842143** | **0.805458** |
| Conservative | Sigmoid | 6.095 | **0.788079** | **0.879115** | 0.090927 | 0.304328 | 0.884131 | **0.756951** | 0.914550 | **0.528000** | **0.669484** | 0.638860 |
| Conservative | Isotonic | 6.260 | 0.787775 | 0.878999 | **0.090770** | **0.303406** | **0.884575** | 0.755570 | **0.924617** | 0.523333 | 0.668370 | **0.640832** |

Threshold metrics use 0.5. Conservative excludes previous-default history and interest rate.

## Comparison with uncalibrated LightGBM

| Feature set | Model | Average precision | ROC-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| Full | Uncalibrated LightGBM (G01) | 0.935489 | 0.976237 | 0.057922 | 0.177841 |
| Full | Sigmoid calibrated | 0.935666 | 0.976384 | 0.048525 | 0.155805 |
| Full | Isotonic calibrated | 0.935419 | 0.976293 | 0.048624 | 0.155088 |
| Conservative | Uncalibrated LightGBM (G01) | 0.789088 | 0.879595 | 0.116507 | 0.370558 |
| Conservative | Sigmoid calibrated | 0.788079 | 0.879115 | 0.090927 | 0.304328 |
| Conservative | Isotonic calibrated | 0.787775 | 0.878999 | 0.090770 | 0.303406 |

Calibration substantially reduced Brier score and log loss in this validation run, with only small changes in ranking metrics. Sigmoid is marginally stronger on average precision and ROC-AUC; isotonic is marginally better on log loss. The full-feature sigmoid model has the lowest Brier score in this comparison. These are small single-split differences, not statistically established wins.

## Interpretation

1. **Full feature set:** both calibration methods retain essentially the same ranking quality as uncalibrated LightGBM and improve the reported probability losses. Sigmoid has the best average precision, ROC-AUC, and Brier score among the three; isotonic has the lowest log loss.
2. **Conservative feature set:** both methods improve Brier score and log loss relative to uncalibrated LightGBM, while average precision and ROC-AUC decline slightly. The conservative model remains substantially weaker in ranking quality.
3. **Threshold behavior:** isotonic yields higher precision and slightly lower recall than sigmoid at threshold 0.5 in both feature sets. A threshold of 0.5 is not automatically an appropriate operational decision threshold; any threshold choice needs a predeclared objective and must not use test labels.
4. **Subgroup audit:** 96 subgroup rows met the configured minimum count of 30 across the feature-set/calibration combinations. These are descriptive slices only. Review counts and positive counts before interpreting differences; no fairness conclusion follows from the existence of this table.
5. **Model selection:** retain sigmoid as the leading calibration candidate for the full-feature benchmark based on the primary ranking metric and Brier score; retain isotonic as a close alternative with slightly lower log loss. Do not claim a universal winner or deployment readiness.

## Reproducibility and isolation

- Calibration methods: sigmoid and isotonic.
- Five-fold stratified cross-validation within the training partition.
- Seed 42.
- Validation-only metrics and reliability-curve bins.
- Subgroups: gender, education, home ownership, loan intent, age band, and income quartile; minimum n=30.
- Test partition used: **False**.
- Local artifacts: `experiments/results/calibration/validation_metrics.csv`, `calibration_curve.csv`, `validation_subgroups.csv`, and `run_metadata.json`.

## Limitations

This is a single frozen validation split. No confidence intervals or statistical tests were computed. Calibration performance can vary with sample and distribution shift. Subgroup metrics are descriptive, may be unstable, and do not constitute a legal fairness evaluation. The dataset remains synthetic/augmented with unresolved feature provenance and timing. The target is the recorded `loan_status` label, not future repayment/default. No causal, real-world fairness, or deployment claims are supported.
