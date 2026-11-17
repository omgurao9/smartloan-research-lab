# O01: Optuna Hyperparameter Optimization Protocol

## Objective

Tune HistGradientBoosting for two explicitly separated feature sets:

- Full benchmark: all 13 predictors.
- Conservative benchmark: excludes `previous_loan_defaults_on_file` and `loan_int_rate`.

The goal is to test whether careful hyperparameter optimization improves validation average precision over the fixed baseline, without allowing either the validation or test partition into hyperparameter search.

## Search protocol

- Search algorithm: Optuna TPE sampler with a fixed seed.
- Objective: mean stratified cross-validation average precision.
- Cross-validation: 3 folds by default, created only from the fixed training partition.
- Trials: 25 per feature set by default.
- Execution: sequential CPU execution for reproducibility and to avoid resource contention.
- Preprocessing: imputation, scaling, and one-hot encoding are fitted inside each fold through a scikit-learn pipeline.
- Early stopping: HistGradientBoosting internal early stopping is enabled within each fold's training fit.
- Validation: after selecting the best trial from training-only CV, refit on all training rows and evaluate once on the fixed validation partition.
- Test: no test rows or labels are used for tuning or evaluation.

## Search space

| Parameter | Search range |
|---|---|
| learning_rate | 0.02–0.20, log scale |
| max_iter | 100–400, step 50 |
| max_leaf_nodes | 15–63, step 4 |
| min_samples_leaf | 10–60, step 5 |
| l2_regularization | 0.0001–10, log scale |
| max_bins | 64–255 |

## Outputs

Each run writes to a new timestamped directory under `experiments/results/optimization/`:

- `trials.csv`: trial-level parameters, objective values, and states for both feature sets.
- `best_params.json`: best CV parameters and final validation metrics.
- `validation_metrics.csv`: validation metrics for the tuned full and conservative models.
- `run_metadata.json`: seeds, fold count, trial count, dataset/split hashes, software versions, and test-isolation declaration.
- `full_features_best.joblib` and `conservative_no_defaults_no_rate_best.joblib`: fitted pipelines.

Do not overwrite or compare runs with different split fingerprints as if they were paired experiments.

## Comparison and interpretation

Compare each tuned configuration against its matching HistGradientBoosting baseline from B01/V01. Report both training-CV and validation average precision. A high CV score does not replace validation performance. If tuning improves one track but not the other, preserve that distinction. Do not select a final production model or open the test partition until the feature-validity and model-selection decisions are frozen.

## Limitations

This is optimization on a synthetic/augmented loan-approval benchmark. It does not establish real-world lending performance, causal effects, or the validity/timing of individual predictors.
