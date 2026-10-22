# main9 feature ablation: `(q,v,K)` vs `(q,context,v,K)`

The physics and saved labels are unchanged. The final 64 chronological macros per seed are the fixed holdout, and hindsight rows remain grouped by source macro.

The earlier reachability audit found canonical T6 at 11 / 14 / 17 completed macros (seeds 800--802). Consequently 12 and 16 macros are the relevant online-data regime; 32+ macros are already generally after T6.

## Closed-set classification

Values are pooled known-class accuracy. `delta` is `(q,v,K) - full`.

| model | macros | q,v,K | q,context,v,K | delta | F1 delta |
|---|---:|---:|---:|---:|---:|
| random_forest | 8 | 0.294 | 0.286 | 0.008 | 0.020 |
| random_forest | 12 | 0.219 | 0.212 | 0.007 | 0.022 |
| random_forest | 16 | 0.210 | 0.234 | -0.024 | -0.012 |
| random_forest | 32 | 0.214 | 0.182 | 0.031 | 0.033 |
| random_forest | 64 | 0.286 | 0.281 | 0.005 | 0.017 |
| random_forest | 128 | 0.307 | 0.354 | -0.047 | -0.052 |
| random_forest | 320 | 0.401 | 0.411 | -0.010 | -0.017 |
| extra_trees | 8 | 0.294 | 0.246 | 0.048 | 0.057 |
| extra_trees | 12 | 0.278 | 0.311 | -0.033 | -0.012 |
| extra_trees | 16 | 0.222 | 0.210 | 0.012 | 0.032 |
| extra_trees | 32 | 0.198 | 0.224 | -0.026 | -0.043 |
| extra_trees | 64 | 0.260 | 0.250 | 0.010 | -0.014 |
| extra_trees | 128 | 0.312 | 0.339 | -0.026 | -0.040 |
| extra_trees | 320 | 0.417 | 0.422 | -0.005 | 0.001 |
| knn | 8 | 0.397 | 0.413 | -0.016 | 0.009 |
| knn | 12 | 0.291 | 0.291 | 0.000 | 0.015 |
| knn | 16 | 0.240 | 0.222 | 0.018 | 0.008 |
| knn | 32 | 0.219 | 0.224 | -0.005 | -0.002 |
| knn | 64 | 0.323 | 0.323 | 0.000 | 0.016 |
| knn | 128 | 0.375 | 0.365 | 0.010 | 0.014 |
| knn | 320 | 0.484 | 0.464 | 0.021 | 0.026 |
| prevalence | 8 | 0.270 | 0.270 | 0.000 | 0.000 |
| prevalence | 12 | 0.199 | 0.199 | 0.000 | 0.000 |
| prevalence | 16 | 0.222 | 0.222 | 0.000 | 0.000 |
| prevalence | 32 | 0.177 | 0.177 | 0.000 | 0.000 |
| prevalence | 64 | 0.177 | 0.177 | 0.000 | 0.000 |
| prevalence | 128 | 0.193 | 0.193 | 0.000 | 0.000 |
| prevalence | 320 | 0.135 | 0.135 | 0.000 | 0.000 |

## Safe duplicate-filter gate

Gate: pooled rejection precision >= 0.90, LOCO unknown retention >= 0.95, duplicate rejection >= 0.05, and useful in >= 0.50 of seeds.

| feature | model | first demonstrated point |
|---|---|---|
| q_v_k | random_forest | not demonstrated |
| q_v_k | extra_trees | not demonstrated |
| q_v_k | knn | not demonstrated |
| q_v_k | prevalence | not demonstrated |
| q_context_v_k | random_forest | not demonstrated |
| q_context_v_k | extra_trees | not demonstrated |
| q_context_v_k | knn | not demonstrated |
| q_context_v_k | prevalence | not demonstrated |

## Strict and relaxed trade-off at 16 and 320 macros

`dup` is pooled known-duplicate rejection; `LOCO` is retained withheld-cell fraction. Precision is omitted when nothing is rejected.

| feature | model | macros | profile | accuracy | precision | dup | LOCO | useful seeds |
|---|---|---:|---|---:|---:|---:|---:|---:|
| q_context_v_k | extra_trees | 16 | loose_groups2 | 0.210 | 0.818 | 0.054 | -- | 0.000 |
| q_context_v_k | extra_trees | 320 | loose_groups2 | 0.422 | 1.000 | 0.255 | -- | 0.000 |
| q_context_v_k | knn | 16 | loose_groups2 | 0.222 | 1.000 | 0.012 | -- | 0.000 |
| q_context_v_k | knn | 320 | loose_groups2 | 0.464 | 1.000 | 0.234 | -- | 0.000 |
| q_context_v_k | random_forest | 16 | loose_groups2 | 0.234 | 0.857 | 0.036 | -- | 0.000 |
| q_context_v_k | random_forest | 320 | loose_groups2 | 0.411 | 1.000 | 0.354 | -- | 0.000 |
| q_context_v_k | extra_trees | 16 | relative | 0.210 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | extra_trees | 320 | relative | 0.422 | 1.000 | 0.198 | -- | 0.000 |
| q_context_v_k | knn | 16 | relative | 0.222 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | knn | 320 | relative | 0.464 | 1.000 | 0.057 | -- | 0.000 |
| q_context_v_k | random_forest | 16 | relative | 0.234 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | random_forest | 320 | relative | 0.411 | 1.000 | 0.229 | -- | 0.000 |
| q_context_v_k | extra_trees | 16 | strict | 0.210 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | extra_trees | 320 | strict | 0.422 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | knn | 16 | strict | 0.222 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | knn | 320 | strict | 0.464 | 1.000 | 0.057 | -- | 0.000 |
| q_context_v_k | random_forest | 16 | strict | 0.234 | -- | 0.000 | -- | 0.000 |
| q_context_v_k | random_forest | 320 | strict | 0.411 | -- | 0.000 | -- | 0.000 |
| q_v_k | extra_trees | 16 | loose_groups2 | 0.222 | 1.000 | 0.018 | -- | 0.000 |
| q_v_k | extra_trees | 320 | loose_groups2 | 0.417 | 1.000 | 0.240 | -- | 0.000 |
| q_v_k | knn | 16 | loose_groups2 | 0.240 | 0.500 | 0.006 | -- | 0.000 |
| q_v_k | knn | 320 | loose_groups2 | 0.484 | 1.000 | 0.234 | -- | 0.000 |
| q_v_k | random_forest | 16 | loose_groups2 | 0.210 | 0.864 | 0.114 | -- | 0.000 |
| q_v_k | random_forest | 320 | loose_groups2 | 0.401 | 1.000 | 0.323 | -- | 0.000 |
| q_v_k | extra_trees | 16 | relative | 0.222 | -- | 0.000 | -- | 0.000 |
| q_v_k | extra_trees | 320 | relative | 0.417 | 1.000 | 0.203 | -- | 0.000 |
| q_v_k | knn | 16 | relative | 0.240 | -- | 0.000 | -- | 0.000 |
| q_v_k | knn | 320 | relative | 0.484 | 1.000 | 0.052 | -- | 0.000 |
| q_v_k | random_forest | 16 | relative | 0.210 | 1.000 | 0.030 | -- | 0.000 |
| q_v_k | random_forest | 320 | relative | 0.401 | 1.000 | 0.229 | -- | 0.000 |
| q_v_k | extra_trees | 16 | strict | 0.222 | -- | 0.000 | -- | 0.000 |
| q_v_k | extra_trees | 320 | strict | 0.417 | -- | 0.000 | -- | 0.000 |
| q_v_k | knn | 16 | strict | 0.240 | -- | 0.000 | -- | 0.000 |
| q_v_k | knn | 320 | strict | 0.484 | 1.000 | 0.052 | -- | 0.000 |
| q_v_k | random_forest | 16 | strict | 0.210 | -- | 0.000 | -- | 0.000 |
| q_v_k | random_forest | 320 | strict | 0.401 | -- | 0.000 | -- | 0.000 |

All threshold profiles and per-seed counts are retained in `aggregate_metrics.csv` and `seed_metrics.csv`.
