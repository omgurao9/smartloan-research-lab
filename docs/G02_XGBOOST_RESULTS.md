# G02: XGBoost Validation Results

## Status

Completed. XGBoost was trained on the frozen training partition and evaluated on the validation partition for both predeclared feature sets. The test partition was not used.

## Results

| Feature set | Features | Fit + validation (s) | Average precision | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 13 | 1.216 | 0.932122 | 0.974853 | 0.904727 | 0.905419 | 0.730005 | 0.906667 | 0.808802 | 0.754111 | 0.190203 | 0.062840 |
| Conservative | 11 | 1.141 | 0.782529 | 0.873907 | 0.836568 | 0.790417 | 0.615072 | 0.707333 | 0.657984 | 0.553625 | 0.384898 | 0.121390 |

Metrics at threshold 0.5. Conservative excludes previous-default history and interest rate.

## Comparison with current references

| Feature set | Tuned HGB AP | LightGBM AP | CatBoost AP | XGBoost AP |
|---|---:|---:|---:|---:|
| Full | 0.935259 | **0.935489** | 0.931280 | 0.932122 |
| Conservative | 0.786295 | **0.789088** | 0.781258 | 0.782529 |

On this single validation split, LightGBM remains the strongest ranking model by average precision. XGBoost is below both LightGBM and tuned HistGradientBoosting for both feature sets, but the gaps are modest. No confidence intervals or significance tests have been run, so this is a descriptive ranking only.

At threshold 0.5, XGBoost has relatively high recall (0.906667 full; 0.707333 conservative) with lower precision (0.730005 full; 0.615072 conservative). Threshold metrics should not be conflated with ranking performance.

## Interpretation

1. XGBoost is a competitive reference but does not displace LightGBM or tuned HGB on validation average precision under this fixed configuration.
2. Full-feature performance is substantially stronger than conservative-feature performance, consistent with earlier ablation results. The feature provenance and timing questions remain unresolved.
3. The benchmark used one fixed configuration, not a hyperparameter search. It would be inappropriate to conclude that XGBoost as an algorithm family is inferior based on this run.
4. Runtime was approximately 1.2 seconds per feature set in this environment; timings are environment-dependent.

## Reproducibility and isolation

- Seed: 42.
- Fixed frozen train/validation split.
- XGBoost histogram tree method; CPU execution.
- Fit on training rows and evaluate on validation rows.
- Test partition used: **False**.
- Local artifacts: `experiments/results/xgboost/validation_metrics.csv`, `run_metadata.json`, and serialized pipelines.

## Limitations

The fixed class-weight setting is a modeling choice and does not guarantee calibrated probabilities. Results are from one validation split and do not establish statistical significance, external generalization, deployment readiness, or algorithm-family superiority. The dataset is a synthetic/augmented benchmark with unresolved feature provenance and feature-timing questions. Do not interpret this task as future repayment/default prediction or make causal/fairness claims from these results.
