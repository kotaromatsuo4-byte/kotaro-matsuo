# main9 feature ablation: `(q,v,K)` vs `(q,context,v,K)`

The physics and saved labels are unchanged. The final 64 chronological macros per seed are the fixed holdout, and hindsight rows remain grouped by source macro.

The earlier reachability audit found canonical T6 at 11 / 14 / 17 completed macros (seeds 800--802). Consequently 12 and 16 macros are the relevant online-data regime; 32+ macros are already generally after T6.

## Closed-set classification

Values are pooled known-class accuracy. `delta` is `(q,v,K) - full`.

| model | macros | q,v,K | q,context,v,K | delta | F1 delta |
|---|---:|---:|---:|---:|---:|
| random_forest | 8 | 0.230 | 0.230 | 0.000 | -0.009 |

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

All threshold profiles and per-seed counts are retained in `aggregate_metrics.csv` and `seed_metrics.csv`.
