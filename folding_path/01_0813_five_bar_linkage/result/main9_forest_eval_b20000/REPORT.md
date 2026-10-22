# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main9_random_corpus_b20000 | extra_trees | 4 | 3 | 0.349 | 0.269 | 1.170 | 0.681 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 8 | 3 | 0.206 | 0.169 | 1.605 | 0.801 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 16 | 3 | 0.198 | 0.161 | 1.917 | 0.849 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 24 | 3 | 0.193 | 0.170 | 2.059 | 0.871 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 32 | 3 | 0.214 | 0.190 | 2.040 | 0.864 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 48 | 3 | 0.260 | 0.230 | 2.006 | 0.853 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 64 | 3 | 0.302 | 0.285 | 1.981 | 0.845 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 96 | 3 | 0.292 | 0.251 | 1.965 | 0.840 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 128 | 3 | 0.344 | 0.302 | 1.923 | 0.825 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 192 | 3 | 0.339 | 0.304 | 1.871 | 0.810 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 256 | 3 | 0.406 | 0.357 | 1.833 | 0.796 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 320 | 3 | 0.406 | 0.379 | 1.803 | 0.785 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 4 | 3 | 0.446 | 0.318 | 1.140 | 0.650 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 8 | 3 | 0.413 | 0.269 | 3.550 | 0.771 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 16 | 3 | 0.222 | 0.158 | 5.203 | 0.898 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 24 | 3 | 0.177 | 0.140 | 7.208 | 1.014 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 32 | 3 | 0.224 | 0.186 | 7.058 | 0.983 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 48 | 3 | 0.286 | 0.244 | 6.425 | 0.935 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 64 | 3 | 0.323 | 0.258 | 6.364 | 0.912 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 96 | 3 | 0.328 | 0.263 | 5.523 | 0.889 | 1.000 | 0.010 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 128 | 3 | 0.365 | 0.311 | 4.839 | 0.850 | 1.000 | 0.026 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 192 | 3 | 0.417 | 0.369 | 3.968 | 0.774 | 1.000 | 0.036 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 256 | 3 | 0.432 | 0.386 | 3.781 | 0.740 | 1.000 | 0.052 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 320 | 3 | 0.464 | 0.428 | 3.921 | 0.723 | 1.000 | 0.057 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 4 | 3 | 0.349 | 0.163 | 1.183 | 0.687 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 8 | 3 | 0.270 | 0.085 | 1.590 | 0.793 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 16 | 3 | 0.222 | 0.052 | 2.045 | 0.882 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 24 | 3 | 0.193 | 0.040 | 2.239 | 0.903 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 32 | 3 | 0.177 | 0.038 | 2.276 | 0.905 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 48 | 3 | 0.161 | 0.035 | 2.148 | 0.891 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 64 | 3 | 0.177 | 0.038 | 2.152 | 0.889 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 96 | 3 | 0.188 | 0.039 | 2.128 | 0.884 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 128 | 3 | 0.193 | 0.040 | 2.089 | 0.877 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 192 | 3 | 0.156 | 0.033 | 2.069 | 0.873 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 256 | 3 | 0.135 | 0.030 | 2.067 | 0.873 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 320 | 3 | 0.135 | 0.030 | 2.072 | 0.874 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 4 | 3 | 0.386 | 0.314 | 1.162 | 0.675 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 8 | 3 | 0.278 | 0.233 | 1.585 | 0.790 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 16 | 3 | 0.216 | 0.165 | 1.887 | 0.839 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 24 | 3 | 0.177 | 0.151 | 2.061 | 0.871 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 32 | 3 | 0.214 | 0.179 | 2.029 | 0.861 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 48 | 3 | 0.302 | 0.274 | 1.989 | 0.848 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 64 | 3 | 0.281 | 0.260 | 1.960 | 0.839 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 96 | 3 | 0.281 | 0.244 | 1.949 | 0.836 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 128 | 3 | 0.354 | 0.302 | 1.907 | 0.819 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 192 | 3 | 0.380 | 0.339 | 1.846 | 0.799 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 256 | 3 | 0.380 | 0.340 | 1.820 | 0.788 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 320 | 3 | 0.401 | 0.376 | 1.771 | 0.772 | -- | 0.000 | -- | -- | -- | 0.000 |

## Earliest useful-filter size

- `main9_random_corpus_b20000/extra_trees`: not demonstrated
- `main9_random_corpus_b20000/knn`: not demonstrated
- `main9_random_corpus_b20000/prevalence`: not demonstrated
- `main9_random_corpus_b20000/random_forest`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.
