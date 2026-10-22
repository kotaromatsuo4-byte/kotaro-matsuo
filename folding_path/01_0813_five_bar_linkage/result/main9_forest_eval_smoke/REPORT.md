# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | 8 | 1 | 1.000 | 1.000 | 0.856 | 0.417 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extra_trees | 24 | 1 | 1.000 | 1.000 | 0.389 | 0.132 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| knn | 8 | 1 | 1.000 | 1.000 | 0.152 | 0.043 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| knn | 24 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.846 | 1.000 | 1.000 | 3.200 | 1.000 |
| prevalence | 8 | 1 | 0.154 | 0.053 | 1.600 | 0.796 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| prevalence | 24 | 1 | 0.154 | 0.053 | 1.603 | 0.798 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| random_forest | 8 | 1 | 0.923 | 0.905 | 0.896 | 0.438 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| random_forest | 24 | 1 | 1.000 | 1.000 | 0.443 | 0.168 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Earliest useful-filter size

- `extra_trees`: not demonstrated
- `knn`: 24 independent macros
- `prevalence`: not demonstrated
- `random_forest`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids`, `sample_weights`, `start_step`, `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
