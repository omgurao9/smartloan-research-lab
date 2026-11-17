# G01: CatBoost and LightGBM Comparison Protocol

## Objective

Compare CatBoost and LightGBM against the existing HistGradientBoosting reference using the same frozen training and validation partitions. Evaluate both the full 13-feature benchmark and the conservative 11-feature set that excludes previous-default history and interest rate.

## Fixed configurations

- CatBoost: 600 iterations, depth 6, learning rate 0.05, Logloss objective, balanced class weights, seed 42.
- LightGBM: 600 estimators, learning rate 0.03, 31 leaves, minimum 30 child samples, 0.8 row and column subsampling, L2 regularization 1.0, balanced class weights, seed 42.
- Both use the same fold-independent preprocessing pipeline: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- CPU execution; no GPU-specific dependency.

These are fixed reference configurations, not hyperparameter-optimized results. Do not use the validation set for early stopping or parameter search in this experiment.

## Evaluation

- Fit on the fixed training partition only.
- Evaluate once on the fixed validation partition.
- Primary metric: average precision (PR-AUC).
- Supporting metrics: ROC-AUC, threshold-0.5 classification metrics, log loss, and Brier score.
- Test partition remains untouched.

## Outputs

- `validation_metrics.csv`: four model/feature-set comparisons.
- `run_metadata.json`: split and dataset hashes, versions, seed, timing, and test-isolation declaration.
- Four serialized fitted pipelines.

## Limitations

This experiment compares fixed configurations. A result does not establish that one algorithm family is universally superior. The dataset's synthetic/augmented nature and feature-timing concerns remain.
