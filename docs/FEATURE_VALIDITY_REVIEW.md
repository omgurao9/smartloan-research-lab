# Feature Validity Review: Previous Defaults and Interest Rate

**Review date:** 2026-11-17  
**Dataset:** Ta-Wei Lo, Loan Approval Classification Dataset (45,000 rows)  
**Purpose:** Document observed associations and define controlled feature-ablation experiments.

## Observed evidence from the raw dataset

### Previous defaults vs. loan status

| Previous defaults | Rejected (0) | Approved (1) | Total |
|---|---:|---:|---:|
| No | 12,142 | 10,000 | 22,142 |
| Yes | 22,858 | 0 | 22,858 |

All 10,000 approved observations have previous_loan_defaults_on_file = No; none of the 22,858 observations marked Yes are approved. This is a deterministic one-way separation in this file. It may reflect a generated decision rule, a policy encoded in the data, or another construction artifact. The cross-tab alone does not prove leakage: historical defaults could in principle be known before an approval decision.

### Interest rate by loan status

| Loan status | Count | Mean | Median | Standard deviation |
|---|---:|---:|---:|---:|
| Rejected (0) | 35,000 | 10.478 | 10.85 | 2.731 |
| Approved (1) | 10,000 | 12.857 | 12.98 | 3.069 |

Approved rows have a higher average and median interest rate. This is compatible with several possible data-generation or underwriting mechanisms. It does not establish the direction of causality or when the rate was assigned.

An independent public project using this same Kaggle dataset states that loan_int_rate is set after approval and excludes it to avoid leakage. This is useful corroborating evidence but remains secondary documentation, not a statement verified from the original dataset creator. Therefore, treat loan_int_rate as potentially post-decision and exclude it from the conservative feature set pending primary-source confirmation.

## Controlled ablation design

The baseline code now evaluates four feature sets using the same fixed training and validation partitions and the same three model families:

1. full_features: all 13 predictors.
2. without_previous_defaults: removes only previous_loan_defaults_on_file.
3. without_interest_rate: removes only loan_int_rate.
4. conservative_no_defaults_no_rate: removes both potentially problematic variables.

This factorial ablation separates the individual effect of each feature from their combined removal. It is a sensitivity analysis, not proof of leakage or a claim that the conservative set is definitively the correct production feature set.

## Interpretation rules

- Compare average precision as the primary ranking metric, with ROC-AUC, threshold metrics, log loss, and Brier score as supporting measures.
- All comparisons use validation data only; the held-out test partition remains untouched.
- If performance drops after removing a feature, report the drop without automatically concluding that the feature is legitimate.
- No real-world deployment or lending-policy claims can be supported by this synthetic/augmented benchmark alone.

## References

- Kaggle, Ta-Wei Lo, Loan Approval Classification Dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Independent project methodology describing exclusion of loan_int_rate because it is assigned after approval: https://github.com/dpidaparthi/loan-approval-classification
- 2025 article describing the dataset as synthetic/augmented and SMOTENC-expanded: https://doi.org/10.64189/ssc.25215
