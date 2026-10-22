# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_inputs | extra_trees | 8 | 2 | 0.815 | 0.748 | 0.923 | 0.465 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| synthetic_inputs | extra_trees | 24 | 2 | 1.000 | 1.000 | 0.401 | 0.142 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| synthetic_inputs | knn | 8 | 2 | 1.000 | 1.000 | 0.172 | 0.064 | 1.000 | 0.037 | 1.000 | 1.000 | 1.026 | 0.500 |
| synthetic_inputs | knn | 24 | 2 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.688 | 1.000 | 1.000 | 2.222 | 1.000 |
| synthetic_inputs | prevalence | 8 | 2 | 0.259 | 0.086 | 1.622 | 0.843 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| synthetic_inputs | prevalence | 24 | 2 | 0.156 | 0.054 | 1.669 | 0.823 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| synthetic_inputs | random_forest | 8 | 2 | 0.963 | 0.950 | 0.812 | 0.403 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| synthetic_inputs | random_forest | 24 | 2 | 1.000 | 1.000 | 0.460 | 0.179 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Earliest useful-filter size

- `synthetic_inputs/extra_trees`: not demonstrated
- `synthetic_inputs/knn`: 24 independent macros
- `synthetic_inputs/prevalence`: not demonstrated
- `synthetic_inputs/random_forest`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
