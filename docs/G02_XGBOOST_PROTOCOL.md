# G02: XGBoost Benchmark Protocol

## Objective

Add XGBoost as a widely used gradient-boosted tree reference alongside HistGradientBoosting, CatBoost, and LightGBM. Compare the full 13-feature and conservative 11-feature sets on the existing frozen split.

## Fixed configuration

- 600 estimators, maximum depth 5, learning rate 0.03.
- Row subsampling 0.8 and column subsampling 0.8.
- Minimum child weight 5, L2 regularization 1.0.
- Binary logistic objective, log-loss evaluation metric, histogram tree method.
- Positive-class scale weight 3.5, seed 42, CPU execution.

This is a predeclared reference configuration, not a tuned model. Do not tune against validation results.

## Evaluation and isolation

Fit on training rows only and evaluate once on validation rows. Report average precision, ROC-AUC, threshold-0.5 metrics, log loss, Brier score, and runtime. The test partition must remain unopened.

## Outputs

- `experiments/results/xgboost/validation_metrics.csv`
- `run_metadata.json`
- One serialized pipeline per feature set

## Limitations

The fixed class weight is a modeling choice, not a calibrated probability correction. Compare ranking metrics and calibration metrics separately. The benchmark label and dataset provenance caveats in the research protocol remain in force.
