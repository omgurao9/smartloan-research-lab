# O01: Optuna Hyperparameter Optimization Protocol and Results

## Objective

Tune HistGradientBoosting for two feature sets:

- Full benchmark: all 13 predictors.
- Conservative benchmark: excludes `previous_loan_defaults_on_file` and `loan_int_rate`.

The goal is to test whether optimization improves validation average precision over the fixed baseline without using validation or test data during hyperparameter search.

## Search protocol

- Search algorithm: Optuna TPE sampler with fixed seed 42.
- Objective: mean stratified cross-validation average precision.
- Cross-validation: 3 folds, created only from the fixed training partition (31,493 rows).
- Trials: 25 per feature set (50 total).
- Execution: sequential CPU execution.
- Preprocessing: imputation, scaling, and one-hot encoding fitted inside each fold through a scikit-learn pipeline.
- Early stopping: HistGradientBoosting internal early stopping enabled within each fold's training fit.
- Validation: after selecting the best trial from training-only CV, refit on all training rows and evaluate on the fixed validation partition (6,749 rows).
- Test: no test rows or labels used for tuning or evaluation.

## Search space

| Parameter | Search range |
|---|---|
| learning_rate | 0.02–0.20, log scale |
| max_iter | 100–400, step 50 |
| max_leaf_nodes | 15–63, step 4 |
| min_samples_leaf | 10–60, step 5 |
| l2_regularization | 0.0001–10, log scale |
| max_bins | 64–255 |

## Results

Run: `O01_20261117T123638Z`

### Best training-CV results

| Feature set | Best CV average precision | Best trial |
|---|---:|---:|
| Full features | **0.936659** | 18 |
| Conservative | **0.784217** | 14 |

### Validation comparison against the fixed HistGradientBoosting baseline

| Feature set | Baseline validation PR-AUC | Tuned validation PR-AUC | Absolute change |
|---|---:|---:|---:|
| Full features | 0.934302 | **0.935259** | +0.000957 |
| Conservative | 0.785706 | **0.786295** | +0.000589 |

Supporting validation metrics:

| Feature set / model | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full tuned HGB | 0.976529 | 0.931842 | 0.877855 | 0.899386 | 0.780667 | 0.835832 | 0.796260 | 0.155173 | 0.048752 |
| Conservative tuned HGB | 0.875476 | 0.882797 | 0.756094 | 0.905143 | 0.528000 | 0.666947 | 0.633931 | 0.307962 | 0.091257 |

### Best parameter configurations

Full-feature best trial 18:

```json
{
  "learning_rate": 0.05176033923396888,
  "max_iter": 300,
  "max_leaf_nodes": 39,
  "min_samples_leaf": 35,
  "l2_regularization": 0.002635164513383049,
  "max_bins": 251
}
```

Conservative best trial 14:

```json
{
  "learning_rate": 0.04974551822769682,
  "max_iter": 350,
  "max_leaf_nodes": 59,
  "min_samples_leaf": 45,
  "l2_regularization": 0.004815200554932804,
  "max_bins": 249
}
```

## Interpretation

1. Optuna found marginal improvements in validation PR-AUC for both feature tracks: +0.000957 for full features and +0.000589 for the conservative set.
2. These gains are very small. On this split and within this search space, the fixed HistGradientBoosting baseline was already close to the best configurations found.
3. The full-feature tuned model has validation ROC-AUC 0.976529 and PR-AUC 0.935259. The conservative tuned model has ROC-AUC 0.875476 and PR-AUC 0.786295.
4. The large gap between the two tracks remains. Hyperparameter tuning does not remove the predictive dependence on the two excluded variables.
5. Do not describe these tiny improvements as a meaningful or statistically established performance gain. The experiment uses one fixed split and does not quantify uncertainty across independent datasets or repeated outer splits.
6. The full model remains a benchmark result, not evidence of a deployable lending model. Feature timing and dataset-generation limitations remain unresolved.

## Reproducibility and outputs

The local run produced the following artifacts in `experiments/results/optimization/O01_20261117T123638Z/`:

- `trials.csv`: all 50 trial records.
- `best_params.json`: best CV parameters and validation metrics.
- `validation_metrics.csv`: final validation metrics for both tuned feature sets.
- `run_metadata.json`: seeds, folds, trial count, data/split hashes, software versions, and test-isolation declaration.
- `full_features_best.joblib` and `conservative_no_defaults_no_rate_best.joblib`: fitted pipelines.

Do not compare runs with different split fingerprints as if they were paired experiments. These generated local artifacts are not committed to Git by default.

## Decision

O01 is complete. Retain the tuned pipelines as candidate models, but do not claim substantial benefit from tuning. Preserve the fixed baselines as reference points. The held-out test partition remains untouched.

## Limitations

This is optimization on a synthetic/augmented loan-approval benchmark. It does not establish real-world lending performance, causal effects, or the validity/timing of individual predictors.
