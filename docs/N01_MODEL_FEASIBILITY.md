# N01: Neural and In-Context Tabular Model Feasibility

## Objective

Determine which modern tabular models are suitable for a controlled comparison on the frozen loan-status benchmark. This phase is a feasibility and protocol gate; no model has been run and no validation result is claimed here.

## Dataset and hardware constraints

- Raw dataset: 45,000 rows and 13 predictors plus the target.
- Frozen training partition: 31,493 rows; validation: 6,749 rows; test: 6,749 rows.
- Feature sets: full (13 predictors) and conservative (11 predictors, excluding previous-default history and interest rate).
- User environment: Windows 11, Python 3.12, RTX 3050 6 GB, 16 GB RAM.
- Prefer a CPU-compatible, reproducible benchmark. Do not assume CUDA is available or required.
- Keep the test partition unopened.

## Candidate assessment

### TabICLv2 — primary in-context candidate

The official TabICL documentation describes a pretrained transformer for tabular classification and regression, with training-set examples supplied as context at prediction time. Its documented pretraining range is 300–60,000 training samples and 2–100 columns. The frozen training partition (31,493 rows) and the two feature sets are within those documented ranges. The documentation states that CPU inference is supported, while a GPU is recommended for larger datasets or faster prediction.

The official project describes the implementation as open source with a permissive license. Verify the exact installed package and model-checkpoint license at execution time and record the versions. Candidate: `tabicl`.

### TabM — primary train-from-scratch neural candidate

TabM is an ICLR 2025 tabular deep-learning architecture that uses parameter-efficient ensembling with shared weights. The official repository describes parallel predictions and shared parameters as central design choices. Its package declares PyTorch and numerical-embedding dependencies and uses the Apache Software License. It is a reasonable neural candidate, but CPU runtime and training stability must be measured rather than assumed.

Candidate: `tabm` with PyTorch. Use a fixed, predeclared training budget and early stopping based only on a training-internal validation fold; do not tune against the frozen validation partition.

### TabPFN — defer from this phase

Current TabPFN model-weight releases include non-commercial restrictions for newer model generations. Its model documentation also warns that performance can degrade above 50,000 data points. The training partition is below that stated point, but the licensing restriction and additional model-weight terms make it a less suitable default candidate for a project that may later be demonstrated or reused in a product context. Do not install or benchmark it in N01 unless its exact model version and license are separately reviewed.

## N01 experimental protocol

1. Freeze the current split and feature definitions. Use only the training partition to fit models.
2. Run TabICLv2 and TabM separately on full and conservative feature sets.
3. Keep all preprocessing fit on training data. Handle categorical columns using each model's documented native interface; do not one-hot encode blindly if the estimator expects categorical inputs.
4. Fix seeds and record package, checkpoint, Python, PyTorch, scikit-learn, hardware, device, elapsed fit/predict time, peak memory where measurable, and model configuration.
5. Evaluate each fitted model once on the frozen validation partition. Report the same ranking, threshold, and probability metrics used in prior phases.
6. Do not select hyperparameters, thresholds, or feature sets using the test partition.
7. Compare each neural/in-context candidate with the leading references: F01 calibrated LightGBM and G01 LightGBM; report the conservative results separately.
8. Do not declare a winner from a tiny single-split metric difference. Consider runtime, memory, reproducibility, license, and probability quality alongside average precision.
9. Keep model-weight licenses and third-party dependencies documented independently from the research-code license.

## Feasibility decision

Proceed with TabICLv2 and TabM as the two N01 candidates. Defer TabPFN pending model-specific license review. This is a research-only benchmark of the recorded `loan_status` label, not a real-world lending, repayment, or default-prediction evaluation.

## References

- TabICL official documentation: https://tabicl.readthedocs.io/en/latest/
- TabICLv2 official repository: https://github.com/soda-inria/tabicl
- TabM official repository: https://github.com/yandex-research/tabm
- TabPFN official repository and model-license notes: https://github.com/PriorLabs/TabPFN
