# E01: Leakage-Safe Stacking Protocol

## Objective

Test whether a logistic meta-learner combining HistGradientBoosting, Random Forest, and LightGBM probabilities improves validation performance over the individual models.

## Method

- Evaluate full and conservative feature sets using the frozen split.
- Generate meta-training features through stratified out-of-fold predictions within the training partition only.
- Fit the logistic meta-learner on those out-of-fold predictions and training labels.
- Refit base estimators on the complete training partition as part of the stacking estimator.
- Evaluate the final stack once on validation.
- Use five folds by default, seed 42.
- Do not use validation predictions to fit the meta-learner or tune component models.
- Keep test partition unopened.

The stack uses probability outputs from three base estimators and does not pass original features to the meta-learner.

## Metrics

Report the common research-lab metrics, runtime, split fingerprint, and configuration. Compare with the corresponding single-model validation results, while noting that all comparisons use the same fixed validation set.

## Limitations

A stacking improvement on one validation split is not evidence of statistical significance or external generalization. Stacking can increase computational complexity and probability instability. Dataset provenance and feature-timing limitations remain.
