# main6 paired-warmup results

同じ補正後状態 `q1` から `theta=0`, `theta=+/-pi/2`, `theta=pi` の3分岐を
評価する paired warmup を16組（48終端教師）与え、その後のK=2探索を
10000 mechanics stepで比較した。

`discovery_auc` は全予算における累積 cell 数の時間平均、
`post-warmup incremental AUC` は共通 warm-up 終了時の cell 数を
差し引いた、その後の平均追加 cell 数である。いずれも最大6で
正規化した値ではない。標準偏差は seed 間の標本標準偏差。

## Aggregate

| budget | selector | seeds | final cells | canonical coverage | discovery AUC | post-warm-up incremental AUC |
|---:|---|---:|---:|---:|---:|---:|
| 10000 | independent_random | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1664 ± 0.3556 | 0.5889 ± 1.0200 |
| 10000 | fixed_transport | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1607 ± 0.3458 | 0.5826 ± 1.0090 |
| 10000 | learned_angle_gp | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1331 ± 0.2980 | 0.5517 ± 0.9557 |
| 10000 | learned_angle_gp_shuffled | 3 | 2.333 ± 0.577 | 2.333 ± 0.577 | 2.1304 ± 0.2933 | 0.5487 ± 0.9504 |

## Mechanics-budget throughput

| budget | selector | rollouts / 1000 steps | completed second actions / 1000 steps |
|---:|---|---:|---:|
| 10000 | independent_random | 59.67 ± 6.18 | 45.47 ± 3.43 |
| 10000 | fixed_transport | 53.47 ± 3.23 | 41.33 ± 1.65 |
| 10000 | learned_angle_gp | 58.37 ± 4.82 | 43.40 ± 1.66 |
| 10000 | learned_angle_gp_shuffled | 59.73 ± 5.00 | 45.30 ± 2.91 |

## Per-seed checkpoints

| budget | selector | seed | final | post-warm-up AUC | C@1000 | C@3000 | C@5000 | C@10000 | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | independent_random | 600 | 3 | 1.7666 | 1 | 3 | 3 | 3 | 63 | 1358 | 2813 | -- | -- | -- |
| 10000 | independent_random | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 467 | -- | -- | -- | -- |
| 10000 | independent_random | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 182 | -- | -- | -- | -- |
| 10000 | fixed_transport | 600 | 3 | 1.7477 | 1 | 3 | 3 | 3 | 63 | 1345 | 2996 | -- | -- | -- |
| 10000 | fixed_transport | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 467 | -- | -- | -- | -- |
| 10000 | fixed_transport | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 182 | -- | -- | -- | -- |
| 10000 | learned_angle_gp | 600 | 3 | 1.6552 | 1 | 3 | 3 | 3 | 63 | 2474 | 2695 | -- | -- | -- |
| 10000 | learned_angle_gp | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 467 | -- | -- | -- | -- |
| 10000 | learned_angle_gp | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 182 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 600 | 3 | 1.6462 | 1 | 3 | 3 | 3 | 63 | 2549 | 2701 | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 601 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 467 | -- | -- | -- | -- |
| 10000 | learned_angle_gp_shuffled | 602 | 2 | 0.0000 | 2 | 2 | 2 | 2 | 63 | 182 | -- | -- | -- | -- |

## learned_angle_gp versus shuffled-score control

差は `learned_angle_gp - learned_angle_gp_shuffled`。

| budget | paired seeds | Δ final cells | Δ discovery AUC | Δ post-warm-up incremental AUC |
|---:|---:|---:|---:|---:|
| 10000 | 3 | 0.000 ± 0.000 | 0.0027 ± 0.0047 | 0.0030 ± 0.0052 |

## Warm-up identity check

- seed 600: PASS; step=[1040], cells=[1], hash=['009d8b0055e7e806f47d7b2be980d0f88ad92f72fa6ece15e4682ad904c9a3c2']
- seed 601: PASS; step=[1242], cells=[2], hash=['e02131252e9656c558e2293cee2be3398d0ae483b6d89afacb14d798417f9268']
- seed 602: PASS; step=[1484], cells=[2], hash=['1795b6e0e66153e194ac1db73cb7a141b0e80be6b7649fcbb399e2ddbe852eb6']

4 selectorすべてで、seedごとにwarmup終了step、終了時cell数、木の幾何・親子関係・
archiveを含むSHA-256が一致した。したがって、各seed内のselector比較は同一の
探索prefixから開始している。

## Paired-branch structural audit

全12実行について、`rollout_diagnostics.csv` と `summary.json` を照合し、次を確認した。

- 要求16組、完了16組、完了48分岐で一致する。
- 各組にbranch index `[0,1,2]`、selection code `[14,15,16]`、role
  `[transport, orthogonal, reverse]` が1回ずつ存在する。
- 3分岐の `q1_state_id` と `parent_id` は同一で、角度は厳密に
  `[0,+/-pi/2,pi]` である。
- 共通の第1摂動・corrector費用は各組の最初の分岐にだけ計上される。

構造監査は12/12実行でPASSした。詳細は
`result/main6_paired_warmup_aggregate.json` に保存した。

## Canonical-six audit

保存されたelite cellを固定済みのfive-bar canonical witnessへ再照合したところ、
全12実行で `archive cell数 = canonical coverage数`、extra cell数は0だった。
したがって、この比較の2または3 cellはdescriptorの過分割で増えたものではない。
一方、どの手法もcanonical 6を完了せず、seed 601, 602では全手法が2 cellに留まった。

## Paired warmup versus legacy warmup

差は同じselector・seedにおける `paired warmup - legacy warmup`。paired warmupは
同一q1から48分岐、legacy warmupは16個の異なるq1から1分岐ずつなので、これは
教師情報とwarmup費用を合わせた実用上の比較であり、warmupだけの因果効果ではない。

| selector | seeds | Δ final cells | Δ canonical coverage | Δ discovery AUC | Δ post-warmup incremental AUC | Δ warmup step |
|---|---:|---:|---:|---:|---:|---:|
| independent_random | 3 | -1.000 ± 1.732 | -1.000 ± 1.732 | -0.6592 ± 1.1079 | -0.6390 ± 1.1069 | +723.0 ± 173.3 |
| fixed_transport | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.0678 ± 0.0846 | -0.0208 ± 0.0361 | +723.0 ± 173.3 |
| learned_angle_gp | 3 | +0.333 ± 0.577 | +0.333 ± 0.577 | +0.1663 ± 0.3222 | +0.2222 ± 0.3848 | +723.0 ± 173.3 |
| learned_angle_gp_shuffled | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.1211 ± 0.2439 | +0.1747 ± 0.3026 | +723.0 ± 173.3 |

paired warmupはGPのlegacy runより平均値を改善したが、改善はseed 600の1 cellだけで、
shuffled controlに対する優位にはつながらなかった。また、同じ情報追加が
independent randomを悪化させており、固定予算の約723 stepを余分なwarmupに使う
機会費用と、異なるprefixによる探索木・乱数系列の変化を無視できない。

## Interpretation

learned-angle GPとscore-shuffled controlは、3 seedすべてで最終cell数とcanonical
coverageが完全に一致した。平均差はdiscovery AUCで `+0.0027 ± 0.0047`、
post-warmup incremental AUCで `+0.0030 ± 0.0052` にすぎない。差が生じたのは
seed 600でGPが第2 cellを75 step、第3 cellを6 step早く得たためであり、seed 601,
602ではwarmup後の新規cellが両者とも0だった。したがって、この実験から
「GP scoreが有効なthetaへ選択圧をかけた」とは言えない。

paired warmupによって角度の低・中・高類似度を同一q1で教師化しても改善しなかった
ことから、教師不足だけを主要因とする説明は弱くなった。five-barでは、16候補中の
theta順位よりも、その前にランダムに決めているparentと第1摂動 `v1` が、新規basinへ
到達できるかを強く支配している可能性が高い。次の改良では、thetaだけを学習する
より、`(parent,v1)` に対する「少なくとも一つの有効thetaがある確率」と、条件付きの
theta順位を分けた二段階surrogateを検証するのが筋である。

なお3 seedのみの結果なので、平均差の統計的有意性や他構造への汎用性は主張しない。
