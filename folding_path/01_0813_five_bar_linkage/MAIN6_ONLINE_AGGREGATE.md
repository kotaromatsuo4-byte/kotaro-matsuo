# main6 K=2 online aggregate

`discovery_auc` は全予算における累積 cell 数の時間平均、
`post-warmup incremental AUC` は共通 warm-up 終了時の cell 数を
差し引いた、その後の平均追加 cell 数である。いずれも最大6で
正規化した値ではない。標準偏差は seed 間の標本標準偏差。

## Aggregate

| budget | selector | seeds | final cells | canonical coverage | discovery AUC | post-warm-up incremental AUC |
|---:|---|---:|---:|---:|---:|---:|
| 3000 | independent_random | 3 | 2.667 ± 1.155 | -- | 2.0737 ± 0.2385 | 0.5354 ± 0.9273 |
| 3000 | fixed_transport | 3 | 2.333 ± 0.577 | -- | 1.9840 ± 0.0833 | 0.4302 ± 0.7452 |
| 3000 | learned_angle_gp | 3 | 2.000 ± 0.000 | -- | 1.8893 ± 0.0810 | 0.3193 ± 0.5530 |
| 3000 | learned_angle_gp_shuffled | 3 | 2.000 ± 0.000 | -- | 1.8861 ± 0.0866 | 0.3155 ± 0.5464 |
| 10000 | independent_random | 3 | 3.333 ± 2.309 | 3.333 ± 2.309 | 2.8255 ± 1.4631 | 1.2279 ± 2.1268 |
| 10000 | fixed_transport | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.2285 ± 0.4291 | 0.6034 ± 1.0451 |
| 10000 | fixed_reverse | 3 | 2.000 ± 0.000 | 2.000 ± 0.000 | 1.9657 ± 0.0262 | 0.3284 ± 0.5688 |
| 10000 | learned_angle_gp | 3 | 2.000 ± 0.000 | 2.000 ± 0.000 | 1.9668 ± 0.0243 | 0.3296 ± 0.5708 |
| 10000 | learned_angle_gp_shuffled | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.0093 ± 0.0494 | 0.3740 ± 0.6478 |
| 10000 | learned_angle_gp_cost_aware | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.2343 ± 0.4391 | 0.6094 ± 1.0556 |
| 10000 | learned_angle_gp_cost_aware_shuffled | 3 | 2.000 ± 0.000 | 2.000 ± 0.000 | 1.9655 ± 0.0265 | 0.3282 ± 0.5685 |

## Mechanics-budget throughput

| budget | selector | rollouts / 1000 steps | completed second actions / 1000 steps |
|---:|---|---:|---:|
| 3000 | independent_random | 52.33 ± 8.08 | 42.44 ± 3.40 |
| 3000 | fixed_transport | 50.44 ± 6.83 | 39.33 ± 1.76 |
| 3000 | learned_angle_gp | 48.44 ± 4.76 | 38.78 ± 0.51 |
| 3000 | learned_angle_gp_shuffled | 52.44 ± 7.34 | 41.67 ± 2.96 |
| 10000 | independent_random | 59.17 ± 3.74 | 45.00 ± 1.59 |
| 10000 | fixed_transport | 55.20 ± 3.00 | 41.63 ± 0.76 |
| 10000 | fixed_reverse | 65.53 ± 3.51 | 50.53 ± 1.99 |
| 10000 | learned_angle_gp | 56.13 ± 0.59 | 42.23 ± 0.15 |
| 10000 | learned_angle_gp_shuffled | 58.43 ± 4.45 | 43.43 ± 1.33 |
| 10000 | learned_angle_gp_cost_aware | 68.23 ± 4.41 | 50.80 ± 2.71 |
| 10000 | learned_angle_gp_cost_aware_shuffled | 57.93 ± 2.30 | 44.57 ± 1.74 |

## Per-seed checkpoints

| budget | selector | seed | final | post-warm-up AUC | C@1000 | C@3000 | C@5000 | C@10000 | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | independent_random | 600 | 4 | 1.6061 | 2 | 4 | -- | -- | 63 | 498 | 2055 | 2337 | -- | -- |
| 3000 | independent_random | 601 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 145 | -- | -- | -- | -- |
| 3000 | independent_random | 602 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 113 | -- | -- | -- | -- |
| 3000 | fixed_transport | 600 | 3 | 1.2907 | 2 | 3 | -- | -- | 63 | 494 | 2203 | -- | -- | -- |
| 3000 | fixed_transport | 601 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 145 | -- | -- | -- | -- |
| 3000 | fixed_transport | 602 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 113 | -- | -- | -- | -- |
| 3000 | learned_angle_gp | 600 | 2 | 0.9578 | 2 | 2 | -- | -- | 63 | 549 | -- | -- | -- | -- |
| 3000 | learned_angle_gp | 601 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 145 | -- | -- | -- | -- |
| 3000 | learned_angle_gp | 602 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 113 | -- | -- | -- | -- |
| 3000 | learned_angle_gp_shuffled | 600 | 2 | 0.9465 | 2 | 2 | -- | -- | 63 | 578 | -- | -- | -- | -- |
| 3000 | learned_angle_gp_shuffled | 601 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 145 | -- | -- | -- | -- |
| 3000 | learned_angle_gp_shuffled | 602 | 2 | 0.0000 | 2 | 2 | -- | -- | 63 | 113 | -- | -- | -- | -- |
| 10000 | independent_random | 600 | 6 | 3.6838 | 2 | 4 | 5 | 6 | 63 | 498 | 2055 | 2337 | 4433 | 5464 |
| 10000 | independent_random | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | independent_random | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | fixed_transport | 600 | 3 | 1.8101 | 2 | 3 | 3 | 3 | 63 | 494 | 2203 | -- | -- | -- |
| 10000 | fixed_transport | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | fixed_transport | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | fixed_reverse | 600 | 2 | 0.9852 | 2 | 2 | 2 | 2 | 63 | 582 | -- | -- | -- | -- |
| 10000 | fixed_reverse | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | fixed_reverse | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | learned_angle_gp | 600 | 2 | 0.9887 | 2 | 2 | 2 | 2 | 63 | 549 | -- | -- | -- | -- |
| 10000 | learned_angle_gp | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | learned_angle_gp | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 600 | 3 | 1.1221 | 2 | 2 | 2 | 3 | 63 | 578 | 8696 | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware | 600 | 3 | 1.8283 | 2 | 3 | 3 | 3 | 63 | 498 | 2025 | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware_shuffled | 600 | 2 | 0.9847 | 2 | 2 | 2 | 2 | 63 | 587 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware_shuffled | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 145 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_cost_aware_shuffled | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 113 | -- | -- | -- | -- |

## learned_angle_gp versus shuffled-score control

差は `learned_angle_gp - learned_angle_gp_shuffled`。

| budget | paired seeds | Δ final cells | Δ discovery AUC | Δ post-warm-up incremental AUC |
|---:|---:|---:|---:|---:|
| 3000 | 3 | 0.000 ± 0.000 | 0.0032 ± 0.0056 | 0.0038 ± 0.0065 |
| 10000 | 3 | -0.333 ± 0.577 | -0.0425 ± 0.0736 | -0.0445 ± 0.0770 |

## learned_angle_gp_cost_aware versus shuffled-score control

差は `learned_angle_gp_cost_aware - learned_angle_gp_cost_aware_shuffled`。

| budget | paired seeds | Δ final cells | Δ discovery AUC | Δ post-warm-up incremental AUC |
|---:|---:|---:|---:|---:|
| 10000 | 3 | 0.333 ± 0.577 | 0.2688 ± 0.4656 | 0.2812 ± 0.4871 |

## Warm-up identity check

- seed 600: PASS; step=[441], cells=[1], hash=['51f2a316eca9b3be690c48f3bf13fe46aa5065a43e94b7b1366375c387766c11']
- seed 601: PASS; step=[593], cells=[2], hash=['2556b78bf36a0564a538970981d9a8584a8f1d093aaee5e0ac9c426d66dabf6a']
- seed 602: PASS; step=[563], cells=[2], hash=['7dc6edae4602ea772ffdc2ee309650c69545b68e5b87e3526ce7a314c1072b2d']
