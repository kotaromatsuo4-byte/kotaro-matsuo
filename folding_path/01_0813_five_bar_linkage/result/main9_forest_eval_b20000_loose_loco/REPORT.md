# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main9_random_corpus_b20000 | extra_trees | 16 | 3 | 0.210 | 0.180 | 1.931 | 0.854 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 32 | 3 | 0.224 | 0.210 | 2.042 | 0.865 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 64 | 3 | 0.250 | 0.245 | 1.994 | 0.850 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 128 | 3 | 0.339 | 0.303 | 1.936 | 0.830 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 256 | 3 | 0.375 | 0.339 | 1.840 | 0.798 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 320 | 3 | 0.422 | 0.389 | 1.810 | 0.786 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 16 | 3 | 0.222 | 0.158 | 5.203 | 0.898 | 1.000 | 0.012 | 0.976 | 1.000 | 1.011 | 0.000 |
| main9_random_corpus_b20000 | knn | 32 | 3 | 0.224 | 0.186 | 7.058 | 0.983 | 1.000 | 0.036 | 0.984 | -- | -- | 0.333 |
| main9_random_corpus_b20000 | knn | 64 | 3 | 0.323 | 0.258 | 6.364 | 0.912 | 1.000 | 0.073 | 0.906 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 128 | 3 | 0.365 | 0.311 | 4.839 | 0.850 | 1.000 | 0.104 | 0.875 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 256 | 3 | 0.432 | 0.386 | 3.781 | 0.740 | 1.000 | 0.198 | 0.771 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 320 | 3 | 0.464 | 0.428 | 3.921 | 0.723 | 1.000 | 0.234 | 0.771 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 16 | 3 | 0.234 | 0.179 | 1.904 | 0.845 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 32 | 3 | 0.182 | 0.168 | 2.050 | 0.868 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 64 | 3 | 0.281 | 0.251 | 1.958 | 0.838 | -- | 0.000 | 0.995 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 128 | 3 | 0.354 | 0.305 | 1.904 | 0.819 | -- | 0.000 | 1.000 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 256 | 3 | 0.370 | 0.333 | 1.833 | 0.793 | -- | 0.000 | 0.995 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 320 | 3 | 0.411 | 0.385 | 1.752 | 0.764 | 1.000 | 0.005 | 0.990 | -- | -- | 0.000 |

## Earliest useful-filter size

- `main9_random_corpus_b20000/extra_trees`: not demonstrated
- `main9_random_corpus_b20000/knn`: not demonstrated
- `main9_random_corpus_b20000/random_forest`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
