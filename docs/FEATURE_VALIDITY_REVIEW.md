# Feature Validity Review: Previous Defaults and Interest Rate

**Review date:** 2026-11-17  
**Dataset:** Ta-Wei Lo, Loan Approval Classification Dataset (45,000 rows)  
**Purpose:** Document observed associations and evaluate controlled feature ablations.

## Observed evidence from the raw dataset

### Previous defaults vs. loan status

| Previous defaults | Rejected (0) | Approved (1) | Total |
|---|---:|---:|---:|
| No | 12,142 | 10,000 | 22,142 |
| Yes | 22,858 | 0 | 22,858 |

All 10,000 approved observations have `previous_loan_defaults_on_file = No`; none of the 22,858 observations marked Yes are approved. This is a deterministic one-way separation in this file. It may reflect a generated decision rule, a policy encoded in the data, or another construction artifact. The cross-tab alone does not prove leakage: historical defaults could in principle be known before an approval decision.

### Interest rate by loan status

| Loan status | Count | Mean | Median | Standard deviation |
|---|---:|---:|---:|---:|
| Rejected (0) | 35,000 | 10.478 | 10.85 | 2.731 |
| Approved (1) | 10,000 | 12.857 | 12.98 | 3.069 |

Approved rows have a higher average and median interest rate. This does not establish the direction of causality or when the rate was assigned.

An independent public project using this same Kaggle dataset states that `loan_int_rate` is set after approval and excludes it to avoid leakage. This is secondary documentation, not confirmation from the original dataset creator. Treat the field as potentially post-decision pending primary-source confirmation.

## V01: controlled feature ablation results

All experiments use the same fixed training and validation partitions, with 6,749 validation rows. Metrics below are validation average precision (PR-AUC). The test partition was not used.

| Feature set | HistGradientBoosting | Random Forest | Logistic Regression |
|---|---:|---:|---:|
| Full features (13) | **0.934302** | **0.922927** | **0.841335** |
| Without previous defaults (12) | 0.862328 | 0.848850 | 0.671846 |
| Without interest rate (12) | 0.884712 | 0.869816 | 0.788674 |
| Conservative: without both (11) | 0.785706 | 0.753707 | 0.610803 |

### Absolute PR-AUC changes against each full-feature model

| Ablation | HistGradientBoosting | Random Forest | Logistic Regression |
|---|---:|---:|---:|
| Remove previous defaults only | -0.071974 | -0.074077 | -0.169489 |
| Remove interest rate only | -0.049590 | -0.053111 | -0.052661 |
| Remove both | -0.148596 | -0.169220 | -0.230532 |

### Interpretation

1. HistGradientBoosting with all 13 predictors remains the strongest validation configuration (PR-AUC 0.934302).
2. Removing previous-default history produces a substantial decline across all three model families.
3. Removing interest rate also reduces validation PR-AUC for all three models, though the effect is smaller for Logistic Regression and similar in magnitude to the previous-default ablation for the tree models.
4. Removing both variables yields a further reduction. The combined effect is not simply the sum of the two individual drops, indicating overlapping and/or interacting predictive information in these models.
5. The conservative HistGradientBoosting model still achieves PR-AUC 0.785706. This is a useful lower-information reference, not proof that the remaining fields are all valid at decision time.

These ablations quantify predictive dependence on the fields; they do **not** prove that either field is leakage. Conversely, high predictive value does not establish that a field is appropriate. The exact feature observation time and synthetic data-generation process remain unresolved.

## Research decisions and limitations

- Preserve the full-feature results as the benchmark reference.
- Treat `previous_loan_defaults_on_file` as a high-risk shortcut feature requiring source/generation review; do not call it proven leakage.
- Exclude `loan_int_rate` from the conservative feature set pending primary-source confirmation of its timing.
- Continue to use validation only for feature/model comparisons. The held-out test partition remains unopened.
- This is a synthetic/augmented loan-approval benchmark, not evidence of real-world lending or future repayment prediction.
- Do not make deployment, causal, or real-world fairness claims from this dataset alone.

## References

- Kaggle, Ta-Wei Lo, *Loan Approval Classification Dataset*: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Independent project methodology describing exclusion of `loan_int_rate` because it is assigned after approval: https://github.com/dpidaparthi/loan-approval-classification
- 2025 article describing the dataset as synthetic/augmented and SMOTENC-expanded: https://doi.org/10.64189/ssc.25215
