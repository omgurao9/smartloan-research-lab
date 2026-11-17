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

### TabM — completed initial CPU benchmark

TabM is an ICLR 2025 tabular deep-learning architecture that uses parameter-efficient ensembling with shared weights. The benchmark used the official `tabm` package with PyTorch on CPU.

Configuration:
- Seed: 42.
- Maximum epochs: 12; patience: 3.
- Ensemble size parameter: k=8.
- Architecture: d_block=128, n_blocks=2.
- Batch size: 512.
- Device: CPU; one PyTorch CPU thread.
- Early stopping: 10% stratified holdout taken only from the training partition. The frozen validation partition was not used for epoch selection.
- Preprocessing: numeric medians and standardization fit on the training partition; categorical integer maps fit on the training partition. Unseen validation categories map to the first training category.
- Both feature sets used the same frozen 31,493-row training and 6,749-row validation partitions.

Validation results:

| Feature set | PR-AUC | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full (13 features) | 0.904513 | 0.964876 | 0.915691 | 0.849616 | 0.869151 | 0.730667 | 0.793915 | 0.745831 | 0.188814 | 0.060042 |
| Conservative (11 features) | 0.713664 | 0.820647 | 0.868277 | 0.721761 | 0.900393 | 0.458000 | 0.607159 | 0.582328 | 0.356433 | 0.106824 |

Fit times were 30.142 seconds (full) and 29.330 seconds (conservative). Both runs reached the 12-epoch maximum; therefore the configured early-stopping patience did not terminate training early.

Interpretation:
- TabM provides a valid completed validation benchmark on this CPU setup.
- Full-feature PR-AUC (0.904513) is below the strongest completed boosting/calibration references: calibrated LightGBM sigmoid (0.935666) and LightGBM (0.935489).
- Conservative-feature PR-AUC (0.713664) is also below the conservative LightGBM reference (0.789088) and conservative calibrated LightGBM sigmoid (0.788079).
- At the fixed 0.5 threshold, TabM's recall is lower than the boosting references; do not interpret accuracy alone as model superiority.
- This is one seed and one limited 12-epoch training budget. It is a benchmark result, not evidence that TabM is intrinsically inferior under all tuning budgets. No repeated-seed uncertainty or statistical significance analysis was performed.
- No test-set rows were used. Results describe prediction of the dataset's recorded `loan_status` label only.


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

TabICLv2 was installed and its checkpoint downloaded, but validation inference was computationally infeasible in the available CPU-only PyTorch environment. No TabICL score is claimed. TabM completed an initial CPU benchmark on both feature sets; its validation results are below the leading LightGBM references under the tested 12-epoch configuration. TabPFN remains deferred pending model-specific license review.

This is a research-only benchmark of the recorded `loan_status` label, not a real-world lending, repayment, or default-prediction evaluation.

## References

- TabICL official documentation: https://tabicl.readthedocs.io/en/latest/
- TabICLv2 official repository: https://github.com/soda-inria/tabicl
- TabM official repository: https://github.com/yandex-research/tabm
- TabPFN official repository and model-license notes: https://github.com/PriorLabs/TabPFN
