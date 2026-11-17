# Experiment Registry

| ID | Status | Question | Decision rule |
|---|---|---|---|
| A00 | Completed (provenance review pending) | Is the source dataset structurally valid and what are its label/feature relationships? | Complete audit and review before modeling |
| S00 | Completed | Create fixed stratified 70/15/15 partitions with dataset and split fingerprints. | One frozen split shared by all experiments; no test-set tuning |
| B01 | Completed | Compare Logistic Regression, Random Forest, and HistGradientBoosting on the frozen split. | Validation average precision; report threshold metrics and calibration losses |
| B02 | Completed | Measure the impact of removing previous-default history. | Paired validation comparison; test set remains untouched |
| V01 | Completed | Isolate the effects of removing previous-default history and interest rate, individually and jointly. | Four paired feature sets; validation only; test set remains untouched |
| O01 | Completed | Can Optuna improve HistGradientBoosting for full and conservative feature sets? | 25 trials each; three-fold stratified CV within training only; validation evaluation after tuning |
| G01 | Completed | Do CatBoost and LightGBM improve over the HistGradientBoosting reference? | Fixed reproducible configurations; same train/validation split; validation only |
| G02 | Implemented | Does XGBoost provide a useful comparison to the tuned HGB, LightGBM, and CatBoost references? | Fixed configuration; same frozen split; validation only; test set remains untouched |
| N01 | Planned | Do modern neural/in-context tabular models add value? | Feasibility, license, runtime and validation quality |
| E01 | Planned | Does leakage-safe stacking improve generalization? | Out-of-fold training predictions only |
| F01 | Planned | Are probabilities reliable and subgroup differences material? | Calibration and descriptive subgroup audit |
