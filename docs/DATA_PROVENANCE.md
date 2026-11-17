# Dataset Provenance and Label Semantics

**Review date:** 2026-11-17  
**Dataset:** Ta-Wei Lo, *Loan Approval Classification Dataset* (Kaggle)  
**Dataset URL:** https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

## What can be established

The Kaggle dataset page is the identified source for this 45,000-row, 14-column file. The page was reachable but did not expose readable metadata in this review, so the following details are cross-checked against secondary documentation and must be treated accordingly.

- Multiple independent projects using this exact 45,000 × 14 dataset describe `loan_status = 1` as **approved** and `loan_status = 0` as **rejected**.
- A 2025 peer-reviewed article describes the dataset as synthetic, inspired by the original Credit Risk dataset on Kaggle, enriched with variables based on Financial Risk for Loan Approval data, and expanded using SMOTENC. This is important: the file should be treated as a synthetic/augmented benchmark, not a verified sample of real bank applications.
- The feature `previous_loan_defaults_on_file` is described as an indicator of whether an applicant has previous loan defaults.

## Label interpretation

For this research, the working label mapping is:

| Value | Working interpretation |
|---:|---|
| 0 | Rejected |
| 1 | Approved |

Therefore, the task is **loan approval classification**, not prediction of future repayment or default.

This is a working interpretation supported by multiple secondary descriptions of the exact dataset. Before making a definitive source-attributed claim, retain the original Kaggle data card/data dictionary with the project materials if it becomes accessible.

## Feature timing and leakage assessment

The available public descriptions do not specify:
- when each feature is observed relative to the loan approval decision;
- whether `loan_int_rate` is an offered/assigned rate or a pre-decision applicant attribute;
- the precise generation process for `loan_status` and `previous_loan_defaults_on_file`;
- whether the deterministic relationship observed in this copy was imposed during synthetic data generation, inherited from source rules, or reflects a genuine decision policy.

The raw audit found:

| Previous defaults | Rejected (0) | Approved (1) |
|---|---:|---:|
| No | 12,142 | 10,000 |
| Yes | 22,858 | 0 |

This means the feature is perfectly separating one category from the positive class in this dataset. It is a serious shortcut-learning / validity concern, but **not sufficient evidence by itself to prove target leakage**. A historical default indicator could plausibly exist before a decision; the source does not establish its timing or generation relationship.

## Research decisions

1. Name the target `loan_status` and describe the working classes as rejected/approved.
2. Do not describe this dataset as longitudinal default prediction or claim future repayment prediction.
3. Describe it as a synthetic/augmented loan approval benchmark, with source provenance limitations.
4. Preserve `previous_loan_defaults_on_file` in the full-feature benchmark, but run a paired ablation without it. Report both results and discuss the deterministic association prominently.
5. Separately test a conservative feature set excluding potentially post-decision variables such as `loan_int_rate` unless its timing can be verified.
6. Do not make real-world deployment, lending fairness, or causal claims from this dataset alone.

## References reviewed

- Kaggle dataset landing page: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Lo, Ta-Wei. *Loan Approval Classification Dataset* (Kaggle), dataset page above.
- *Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management*, Systems 13(7), 581 (2025). Describes the Kaggle dataset as synthetic, inspired by the original Credit Risk dataset, enriched with additional variables, and expanded with SMOTENC: https://www.mdpi.com/2079-8954/13/7/581
- Independent project documentation identifying the exact dataset's target mapping as 1 = approved, 0 = rejected: https://github.com/visnudren/loan-approval-data-mining
- Independent project documentation describing the feature dictionary and target mapping: https://github.com/saniya-khan33015/Loan-Approval-Classification

Secondary project descriptions are corroboration, not substitutes for the original provider's data card.
