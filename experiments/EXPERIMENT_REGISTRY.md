# Experiment Registry

| ID | Status | Question | Decision rule |
|---|---|---|---|
| A00 | In progress | Is the source dataset structurally valid and what are its label/feature relationships? | Complete audit and review before modeling |
| B01 | Planned | What are transparent baseline results on a fixed split? | Compare validation PR-AUC and calibration |
| B02 | Planned | How much does previous-default history affect results? | Paired validation ablation; no test tuning |
| O01 | Planned | Can Optuna improve the strongest baseline? | Training-only CV; validation model selection |
| G01 | Planned | Do CatBoost and LightGBM improve over baseline? | Same split and metric protocol |
| N01 | Planned | Do modern neural/in-context tabular models add value? | Feasibility, license, runtime and validation quality |
| E01 | Planned | Does leakage-safe stacking improve generalization? | Out-of-fold training predictions only |
| F01 | Planned | Are probabilities reliable and subgroup differences material? | Calibration and descriptive subgroup audit |
