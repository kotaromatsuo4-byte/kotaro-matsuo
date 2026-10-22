# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main9_random_corpus_b20000 | extra_trees | 16 | 3 | 0.210 | 0.180 | 1.931 | 0.854 | -- | 0.000 | 0.988 | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 32 | 3 | 0.224 | 0.210 | 2.042 | 0.865 | 1.000 | 0.026 | 0.958 | -- | -- | 0.333 |
| main9_random_corpus_b20000 | extra_trees | 64 | 3 | 0.250 | 0.245 | 1.994 | 0.850 | 1.000 | 0.073 | 0.911 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 96 | 3 | 0.302 | 0.273 | 1.968 | 0.841 | 1.000 | 0.104 | 0.833 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 128 | 3 | 0.339 | 0.303 | 1.936 | 0.830 | 1.000 | 0.099 | 0.786 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 16 | 3 | 0.234 | 0.179 | 1.904 | 0.845 | -- | 0.000 | 0.970 | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 32 | 3 | 0.182 | 0.168 | 2.050 | 0.868 | 1.000 | 0.047 | 0.922 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 64 | 3 | 0.281 | 0.251 | 1.958 | 0.838 | 1.000 | 0.109 | 0.885 | -- | -- | 0.333 |
| main9_random_corpus_b20000 | random_forest | 96 | 3 | 0.276 | 0.245 | 1.960 | 0.838 | 1.000 | 0.203 | 0.849 | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 128 | 3 | 0.354 | 0.305 | 1.904 | 0.819 | 1.000 | 0.141 | 0.828 | -- | -- | 0.000 |

## Earliest useful-filter size

- `main9_random_corpus_b20000/extra_trees`: not demonstrated
- `main9_random_corpus_b20000/random_forest`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
