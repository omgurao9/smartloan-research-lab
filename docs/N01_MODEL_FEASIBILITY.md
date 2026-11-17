# N01: Neural and In-Context Tabular Model Feasibility

## Objective

Determine which modern tabular models are suitable for a controlled comparison on the frozen loan-status benchmark. This phase is a feasibility and protocol gate; results are only reported when a complete, reproducible validation run finishes.

## Dataset and hardware constraints

- Raw dataset: 45,000 rows and 13 predictors plus the target.
- Frozen training partition: 31,493 rows; validation: 6,749 rows; test: 6,749 rows.
- Feature sets: full (13 predictors) and conservative (11 predictors, excluding previous-default history and interest rate).
- User environment: Windows 11, Python 3.12, RTX 3050 6 GB, 16 GB RAM.
- Installed PyTorch build: 2.15.0+cpu; `torch.cuda.is_available()` returned `False`. The installed environment therefore cannot use the RTX 3050 through PyTorch.
- Keep the test partition unopened.

## Candidate assessment

### TabICLv2 — attempted; not computationally feasible in the current environment

The official TabICL documentation describes a pretrained transformer for tabular classification and regression, with training-set examples supplied as context at prediction time. Its documented pretraining range is 300–60,000 training samples and 2–100 columns. The frozen training partition (31,493 rows) and the two feature sets are within those documented ranges.

The package installed and the default checkpoint `tabicl-classifier-v2-20260212.ckpt` downloaded successfully. The benchmark completed model fitting on the 31,493-row training partition. However, validation inference did not complete:

- Initial run: inference on all 6,749 validation rows was interrupted while executing PyTorch scaled dot-product attention.
- Reduced run: `n_estimators=1`, `n_jobs=1` still stalled in the same attention operation.
- Chunked run: prediction batches of 32 still did not complete the first batch; the traceback again ended inside `torch.nn.functional.scaled_dot_product_attention`.
- Hardware diagnostic: PyTorch 2.15.0+cpu; CUDA unavailable. The GPU cannot be used by this installed PyTorch build.

No TabICL validation metrics were produced. This is recorded as a hardware/runtime feasibility limitation, not as a poor predictive result or a model failure. Do not report a TabICL score or compare it as a completed benchmark. A future rerun requires a compatible CUDA-enabled PyTorch environment or a sufficiently capable CPU/runtime setup, followed by a complete validation run.

### TabM — pending

TabM is an ICLR 2025 tabular deep-learning architecture that uses parameter-efficient ensembling with shared weights. The official repository describes parallel predictions and shared parameters as central design choices. Its package declares PyTorch and numerical-embedding dependencies and uses the Apache Software License. It remains a candidate, but CPU runtime and training stability must be measured rather than assumed.

Candidate: `tabm` with PyTorch. Use a fixed, predeclared training budget and early stopping based only on a training-internal validation fold; do not tune against the frozen validation partition.

### TabPFN — defer from this phase

Current TabPFN model-weight releases include non-commercial restrictions for newer model generations. Its model documentation also warns that performance can degrade above 50,000 data points. The training partition is below that stated point, but the licensing restriction and additional model-weight terms make it a less suitable default candidate for a project that may later be demonstrated or reused in a product context. Do not install or benchmark it in N01 unless its exact model version and license are separately reviewed.

## N01 experimental protocol

1. Freeze the current split and feature definitions. Use only the training partition to fit models.
2. Run TabICLv2 and TabM separately on full and conservative feature sets, if the hardware permits.
3. Keep all preprocessing fit on training data. Handle categorical columns using each model's documented native interface; do not one-hot encode blindly if the estimator expects categorical inputs.
4. Fix seeds and record package, checkpoint, Python, PyTorch, scikit-learn, hardware, device, elapsed fit/predict time, peak memory where measurable, and model configuration.
5. Evaluate each fitted model once on the frozen validation partition. Report the same ranking, threshold, and probability metrics used in prior phases.
6. Do not select hyperparameters, thresholds, or feature sets using the test partition.
7. Compare completed neural/in-context candidates with the leading references: F01 calibrated LightGBM and G01 LightGBM; report the conservative results separately.
8. Do not declare a winner from a tiny single-split metric difference. Consider runtime, memory, reproducibility, license, and probability quality alongside average precision.
9. Keep model-weight licenses and third-party dependencies documented independently from the research-code license.

## Feasibility decision

TabICLv2 was installed and its checkpoint downloaded, but validation inference was computationally infeasible in the available CPU-only PyTorch environment. No TabICL score is claimed. TabM remains pending and may be attempted only if time and compute allow. TabPFN remains deferred pending model-specific license review.

This is a research-only benchmark of the recorded `loan_status` label, not a real-world lending, repayment, or default-prediction evaluation.

## References

- TabICL official documentation: https://tabicl.readthedocs.io/en/latest/
- TabICLv2 official repository: https://github.com/soda-inria/tabicl
- TabM official repository: https://github.com/yandex-research/tabm
- TabPFN official repository and model-license notes: https://github.com/PriorLabs/TabPFN
