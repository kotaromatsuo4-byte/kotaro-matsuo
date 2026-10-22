# main6 paired-warmup results

同じ `q1` から `theta=0`, `theta=+/-pi/2`, `theta=pi` の3分岐を評価する
paired warmupを16組（48終端教師）与えた後、4つの選択則を10000 mechanics stepで比較した。
`discovery AUC` は全予算上の累積cell数の時間平均、`post-warmup incremental AUC` は
warmup終了時のcell数を差し引いた後半区間の時間平均であり、最大6に正規化した値ではない。

## Paired-warmup aggregate

| selector | seeds | final cells | canonical coverage | discovery AUC | post-warmup incremental AUC |
|---|---:|---:|---:|---:|---:|
| independent_random | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1664 ± 0.3556 | 0.5889 ± 1.0200 |
| fixed_transport | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1607 ± 0.3458 | 0.5826 ± 1.0090 |
| learned_angle_gp | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1331 ± 0.2980 | 0.5517 ± 0.9557 |
| learned_angle_gp_shuffled | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1304 ± 0.2933 | 0.5487 ± 0.9504 |

## Per-seed checkpoints

| selector | seed | final | canonical | C@1000 | C@3000 | C@5000 | C@10000 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_random | 600 | 3 | 3 | 1 | 3 | 3 | 3 | 2813 | -- | -- | -- |
| independent_random | 601 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| independent_random | 602 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| fixed_transport | 600 | 3 | 3 | 1 | 3 | 3 | 3 | 2996 | -- | -- | -- |
| fixed_transport | 601 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| fixed_transport | 602 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| learned_angle_gp | 600 | 3 | 3 | 1 | 3 | 3 | 3 | 2695 | -- | -- | -- |
| learned_angle_gp | 601 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| learned_angle_gp | 602 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| learned_angle_gp_shuffled | 600 | 3 | 3 | 1 | 3 | 3 | 3 | 2701 | -- | -- | -- |
| learned_angle_gp_shuffled | 601 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |
| learned_angle_gp_shuffled | 602 | 2 | 2 | 2 | 2 | 2 | 2 | -- | -- | -- | -- |

## Paired warmup versus legacy warmup

差は `paired - legacy`。両者は同じselector・seed・10000-step予算だが、
paired warmupは48分岐、legacy warmupは16個の異なるq1で1分岐ずつなので、
この表は教師データの情報量とwarmupコストを合わせた実用上の差を表す。

| selector | paired seeds | delta final cells | delta canonical | delta discovery AUC | delta post-warmup AUC | delta warmup steps |
|---|---:|---:|---:|---:|---:|---:|
| independent_random | 3 | -1.000 ± 1.732 | -1.000 ± 1.732 | -0.6592 ± 1.1079 | -0.6390 ± 1.1069 | 723.0 ± 173.3 |
| fixed_transport | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.0678 ± 0.0846 | -0.0208 ± 0.0361 | 723.0 ± 173.3 |
| learned_angle_gp | 3 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.1663 ± 0.3222 | 0.2222 ± 0.3848 | 723.0 ± 173.3 |
| learned_angle_gp_shuffled | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.1211 ± 0.2439 | 0.1747 ± 0.3026 | 723.0 ± 173.3 |

## Warmup audit

- seed 600: PASS; selectors=4, step=[1040], cells=[1], hash=['009d8b0055e7e806f47d7b2be980d0f88ad92f72fa6ece15e4682ad904c9a3c2']
- seed 601: PASS; selectors=4, step=[1242], cells=[2], hash=['e02131252e9656c558e2293cee2be3398d0ae483b6d89afacb14d798417f9268']
- seed 602: PASS; selectors=4, step=[1484], cells=[2], hash=['1795b6e0e66153e194ac1db73cb7a141b0e80be6b7649fcbb399e2ddbe852eb6']

- 16 groups x 3 branches structural audit: PASS
- 各組でbranch index `[0,1,2]`、selection code `[14,15,16]`、
  role `[transport,orthogonal,reverse]`、同一 `q1_state_id`、
  `theta=[0,+/-pi/2,pi]`、q1コストの一回だけの計上を確認した。

## Interpretation

- learned-angle GP minus shuffled controlは平均final cellで +0.000、discovery AUCで +0.0027。scoreと結果の因果的対応はこの差で判断する。
- learned-angle GP minus independent randomは平均final cellで +0.000、discovery AUCで -0.0333。
- 3 seedsだけなので標準偏差とseed別結果を併記し、汎用性や有意差は主張しない。
- paired warmupでGPが改善しない場合、単なる教師不足だけではなく、
  theta選択より前段のparent/q1選択が探索性能を支配している可能性が残る。
