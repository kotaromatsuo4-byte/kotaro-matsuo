# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main9_random_corpus_b20000 | knn | 16 | 3 | 0.222 | 0.158 | 5.203 | 0.898 | 1.000 | 0.012 | 0.976 | 1.000 | 1.011 | 0.000 |
| main9_random_corpus_b20000 | knn | 24 | 3 | 0.177 | 0.140 | 7.208 | 1.014 | -- | 0.000 | 0.995 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 32 | 3 | 0.224 | 0.186 | 7.058 | 0.983 | 1.000 | 0.036 | 0.984 | -- | -- | 0.333 |
| main9_random_corpus_b20000 | knn | 48 | 3 | 0.286 | 0.244 | 6.425 | 0.935 | 1.000 | 0.057 | 0.938 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 64 | 3 | 0.323 | 0.258 | 6.364 | 0.912 | 1.000 | 0.073 | 0.906 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 96 | 3 | 0.328 | 0.263 | 5.523 | 0.889 | 1.000 | 0.120 | 0.870 | -- | -- | 0.000 |

## Earliest useful-filter size

- `main9_random_corpus_b20000/knn`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
