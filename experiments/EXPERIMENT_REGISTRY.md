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
| G02 | Completed | Does XGBoost provide a useful comparison to the tuned HGB, LightGBM, and CatBoost references? | Fixed configuration; same frozen split; validation only; test set remains untouched |
| F01 | Completed | Are LightGBM probabilities calibrated, and how do descriptive validation slices differ? | Sigmoid/isotonic calibration fitted by training-only CV; validation-only metrics and subgroup summaries; test set remains untouched |
| E01 | Completed | Does out-of-fold stacking of HGB, Random Forest, and LightGBM improve validation performance? | Meta-learner trained only on training-partition OOF probabilities; validation-only evaluation; test set remains untouched |
| N01 | Partially completed (TabM benchmark completed; TabICL inference infeasible on available CPU) | Do neural and in-context tabular models add value over the boosting references? | TabM validation results recorded for full and conservative features (12-epoch CPU run); below leading LightGBM references. TabICLv2 fit completed but validation inference did not finish with CPU-only PyTorch; no metrics claimed. Frozen split; test set remains untouched. |
| FINAL_TEST_01 | Pending user execution | What is the final generalization performance of the preselected sigmoid-calibrated LightGBM? | Fit calibration on train+validation only; evaluate full-feature primary and conservative sensitivity model once on untouched test; no test-driven selection; report stratified bootstrap intervals. |
