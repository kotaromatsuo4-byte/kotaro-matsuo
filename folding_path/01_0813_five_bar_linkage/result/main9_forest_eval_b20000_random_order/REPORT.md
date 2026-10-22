# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main9_random_corpus_b20000 | extra_trees | 4 | 3 | 0.459 | 0.349 | 0.955 | 0.598 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 8 | 3 | 0.272 | 0.220 | 1.501 | 0.768 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 16 | 3 | 0.223 | 0.183 | 2.004 | 0.860 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 24 | 3 | 0.224 | 0.192 | 2.027 | 0.860 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 32 | 3 | 0.214 | 0.195 | 2.041 | 0.864 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 48 | 3 | 0.271 | 0.231 | 1.999 | 0.851 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 64 | 3 | 0.323 | 0.284 | 1.970 | 0.842 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 96 | 3 | 0.286 | 0.246 | 1.965 | 0.841 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 128 | 3 | 0.323 | 0.306 | 1.949 | 0.837 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 192 | 3 | 0.391 | 0.358 | 1.875 | 0.811 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 256 | 3 | 0.380 | 0.339 | 1.854 | 0.804 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | extra_trees | 320 | 3 | 0.370 | 0.343 | 1.823 | 0.792 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 4 | 3 | 0.446 | 0.305 | 0.975 | 0.609 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 8 | 3 | 0.360 | 0.268 | 3.492 | 0.796 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 16 | 3 | 0.196 | 0.131 | 7.321 | 0.998 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | knn | 24 | 3 | 0.198 | 0.159 | 6.733 | 0.990 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 32 | 3 | 0.245 | 0.224 | 6.930 | 0.996 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 48 | 3 | 0.281 | 0.237 | 5.829 | 0.922 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 64 | 3 | 0.307 | 0.251 | 5.426 | 0.878 | 1.000 | 0.026 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 96 | 3 | 0.328 | 0.283 | 4.940 | 0.857 | 1.000 | 0.010 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 128 | 3 | 0.354 | 0.305 | 4.799 | 0.844 | 1.000 | 0.016 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 192 | 3 | 0.406 | 0.367 | 4.217 | 0.790 | 1.000 | 0.042 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 256 | 3 | 0.438 | 0.405 | 3.964 | 0.739 | 1.000 | 0.042 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | knn | 320 | 3 | 0.495 | 0.463 | 3.691 | 0.706 | 1.000 | 0.036 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 4 | 3 | 0.405 | 0.229 | 1.036 | 0.671 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 8 | 3 | 0.263 | 0.092 | 1.624 | 0.829 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 16 | 3 | 0.147 | 0.034 | 2.224 | 0.929 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | prevalence | 24 | 3 | 0.141 | 0.030 | 2.236 | 0.912 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 32 | 3 | 0.141 | 0.030 | 2.225 | 0.908 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 48 | 3 | 0.120 | 0.027 | 2.126 | 0.885 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 64 | 3 | 0.146 | 0.032 | 2.090 | 0.878 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 96 | 3 | 0.151 | 0.033 | 2.077 | 0.874 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 128 | 3 | 0.156 | 0.033 | 2.089 | 0.876 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 192 | 3 | 0.172 | 0.037 | 2.077 | 0.874 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 256 | 3 | 0.172 | 0.037 | 2.075 | 0.874 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | prevalence | 320 | 3 | 0.135 | 0.030 | 2.068 | 0.872 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 4 | 3 | 0.392 | 0.357 | 0.966 | 0.608 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 8 | 3 | 0.263 | 0.218 | 1.530 | 0.780 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 16 | 3 | 0.168 | 0.123 | 2.007 | 0.862 | -- | 0.000 | -- | 1.000 | 1.000 | 0.000 |
| main9_random_corpus_b20000 | random_forest | 24 | 3 | 0.214 | 0.193 | 2.033 | 0.862 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 32 | 3 | 0.229 | 0.197 | 2.049 | 0.865 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 48 | 3 | 0.229 | 0.190 | 2.003 | 0.851 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 64 | 3 | 0.260 | 0.232 | 1.956 | 0.836 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 96 | 3 | 0.297 | 0.261 | 1.932 | 0.829 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 128 | 3 | 0.271 | 0.245 | 1.929 | 0.830 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 192 | 3 | 0.396 | 0.375 | 1.847 | 0.798 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 256 | 3 | 0.370 | 0.338 | 1.820 | 0.788 | -- | 0.000 | -- | -- | -- | 0.000 |
| main9_random_corpus_b20000 | random_forest | 320 | 3 | 0.385 | 0.370 | 1.773 | 0.771 | -- | 0.000 | -- | -- | -- | 0.000 |

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
