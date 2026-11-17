# Raw Dataset Audit (A00)

**Status:** Initial automated audit completed; semantic/provenance review remains open.  
**Input:** `data/raw/loan_data.csv`  
**Audit date:** 2026-11-17

## Structural integrity

| Check | Result |
|---|---:|
| Rows | 45,000 |
| Columns | 14 |
| Expected columns present | 14 / 14 |
| Unexpected columns | 0 |
| Duplicate rows | 0 |
| Missing values | 0 |
| Target nulls | 0 |

The columns match the research protocol's expected schema. Numeric fields were read as numeric and categorical fields as text.

## Target distribution

| `loan_status` | Count | Share |
|---:|---:|---:|
| 0 | 35,000 | 77.78% |
| 1 | 10,000 | 22.22% |

The target is imbalanced (about 3.5:1). Use stratified splitting and report both threshold-independent and threshold-dependent metrics. Do not rely on accuracy alone.

## Cleaning-rule impact

Existing proposed rule:

`person_age >= 84 OR person_emp_exp > person_age - 18`

| Measure | Rows |
|---|---:|
| Age condition | 9 |
| Experience condition | 0 |
| Overlap | 0 |
| Unique rows flagged | 9 |
| Rows remaining | 44,991 |

All 9 flagged rows have target class 0. The age maximum is 144, so the age rule is doing all the work in this dataset. The rule should be justified and sensitivity-tested rather than treated as automatically correct.

## High-priority feature/target relationship

`previous_loan_defaults_on_file` cross-tabulation:

| Previous defaults | Target 0 | Target 1 |
|---|---:|---:|
| No | 12,142 | 10,000 |
| Yes | 22,858 | 0 |

Every class-1 record has value `No`; no class-1 record has value `Yes`. This is a near-deterministic relationship and a major research risk. It is not proof of leakage by itself. Before using this feature in a predictive claim, establish:
- what `loan_status` actually means in the source dataset;
- when `previous_loan_defaults_on_file` is measured relative to the target event/decision;
- whether the target or feature was synthetically generated or derived from the other.

Ablation experiments must compare results with and without this feature. Until provenance is established, describe model outputs as predictions of the recorded dataset label, not future default, repayment, or real-world credit risk.

## Other categorical cross-tabs

No empty target class cells were observed in the following inspected features:

- Gender: female (15,674 class 0; 4,485 class 1), male (19,326; 5,515).
- Education: Associate (9,378; 2,650), Bachelor (10,381; 3,018), Doctorate (479; 142), High School (9,301; 2,671), Master (5,461; 1,519).
- Home ownership: MORTGAGE (16,345; 2,144), OTHER (78; 39), OWN (2,729; 222), RENT (15,848; 7,595).
- Loan intent: DEBTCONSOLIDATION (4,982; 2,163), EDUCATION (7,601; 1,552), HOMEIMPROVEMENT (3,525; 1,258), MEDICAL (6,170; 2,378), PERSONAL (6,031; 1,521), VENTURE (6,691; 1,128).

## Numeric range flags for review

| Feature | Minimum | Median | Maximum |
|---|---:|---:|---:|
| person_age | 20 | 26 | 144 |
| person_income | 8,000 | 67,048 | 7,200,766 |
| person_emp_exp | 0 | 4 | 125 |
| loan_amnt | 500 | 8,000 | 35,000 |
| loan_int_rate | 5.42 | 11.01 | 20 |
| loan_percent_income | 0 | 0.12 | 0.66 |
| cb_person_cred_hist_length | 2 | 4 | 30 |
| credit_score | 390 | 640 | 850 |

The age, income and employment-experience maxima deserve source-level review. Do not silently delete or winsorize records based only on extreme values.

## A00 conclusion and next actions

The file passes basic structural checks, but it is not yet cleared for modeling claims. Next:
1. Locate and document the original dataset source, label definition, and feature timing.
2. Investigate the deterministic previous-default/target relationship.
3. Freeze a documented cleaning policy and stratified split before fitting models.
4. Keep the untouched test set out of feature selection, tuning, and threshold selection.

This audit is descriptive. It does not establish causality, representativeness, target meaning, feature availability at decision time, or suitability for lending decisions.
