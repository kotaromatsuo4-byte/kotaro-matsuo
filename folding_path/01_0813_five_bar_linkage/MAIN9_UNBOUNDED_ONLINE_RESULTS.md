# main9_unbounded_b3000 forest-filter comparison (B=3000)

## 結論

現設定の forest filter は選択圧を一度も発生させていない。10 seeds 合計の actual/raw rejection は learned=0/0、shuffled=0/0。

forest の学習時間を除けば3手法は同じ探索として動作した。全20組で全 macro の候補集合 hash、物理 trajectory、最終 state hash、教師データ hash が random と完全一致した。

したがって、この panel から forest filter による高速化は主張できない。差が小さいのではなく、有効な filter action がゼロだったためである。

## 集計

AUC は mechanics step 0..3000 における累積 coverage の時間平均。T6 の `restricted` は未達を3000で右打切りした `mean(min(T6, 3000))`。rejection fraction の分母は、epsilon bypass/cold randomを除き、forest gateで実際に評価した候補数。

| selector | raw final (kin/feas) | canonical final (kin/feas) | canonical AUC (kin/feas) | canonical T6 kin | canonical T6 feas | macros / teachers | rejected/evaluated (fraction); raw | fit/score/wall [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8.000 ± 0.000 / 7.100 ± 0.568 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.082 ± 0.315 / 4.501 ± 0.415 | 1130.4 (10/10); restricted 1130.4 | 1680.4 (8/10); restricted 1944.3 | 60.700 ± 3.234 / 665.600 ± 43.864 | 0/0 (NA); raw 0 | 0.000 ± 0.000 / 0.169 ± 0.058 / 86.379 ± 14.661 |
| forest_filter | 8.000 ± 0.000 / 7.100 ± 0.568 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.082 ± 0.315 / 4.501 ± 0.415 | 1130.4 (10/10); restricted 1130.4 | 1680.4 (8/10); restricted 1944.3 | 60.700 ± 3.234 / 665.600 ± 43.864 | 0/336384 (0.00%); raw 0 | 59.012 ± 5.903 / 8.907 ± 1.474 / 143.999 ± 12.811 |
| forest_filter_shuffled | 8.000 ± 0.000 / 7.100 ± 0.568 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.082 ± 0.315 / 4.501 ± 0.415 | 1130.4 (10/10); restricted 1130.4 | 1680.4 (8/10); restricted 1944.3 | 60.700 ± 3.234 / 665.600 ± 43.864 | 0/336384 (0.00%); raw 0 | 62.101 ± 4.456 / 9.336 ± 1.141 / 150.944 ± 8.122 |


raw archive は layer-order variant を含むため、canonical target数より多い値になり得る。
random の canonical cell first-seen→first-feasible lag は、各 seed 内の観測済みcell平均をさらに10 seedで平均して 242.1 step。全 seed 合計で 2 cell が右打切りだった。

## 対応差（left − right）

coverage/AUC は正が left 有利、T6 と時間は負が left 有利。95% CI は seed 対応 bootstrap、p は exact sign-flip test。

| left | right | canonical kin AUC | canonical feas AUC | kin T6 capped | feas T6 capped | wall [s] |
|---|---|---:|---:|---:|---:|---:|
| forest_filter | random | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +57.620 [+46.038, +68.056], p=0.002 |
| forest_filter | forest_filter_shuffled | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | -6.945 [-14.246, +0.385], p=0.113 |
| forest_filter_shuffled | random | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +64.565 [+54.273, +73.987], p=0.002 |

## seed別 canonical coverage

性能列は全 selector で同一なので random の値を示す。

| seed | selector | raw final kin/feas | canonical final kin/feas | canonical AUC kin/feas | T6 kin/feas | mean first-seen→feasible lag |
|---:|---|---:|---:|---:|---:|---:|
| 930 | random | 8/7 | 6/6 | 5.172/4.490 | 751/1800 | 340.8 (0 censored) |
| 931 | random | 8/7 | 6/6 | 5.124/4.484 | 1013/1488 | 320.2 (0 censored) |
| 932 | random | 8/7 | 6/6 | 5.312/4.170 | 647/2123 | 571.0 (0 censored) |
| 933 | random | 8/6 | 6/5 | 5.203/4.439 | 708/>3000 (censored) | 0.0 (1 censored) |
| 934 | random | 8/8 | 6/6 | 5.476/5.243 | 601/905 | 116.7 (0 censored) |
| 935 | random | 8/7 | 6/6 | 4.643/4.559 | 1456/1456 | 42.0 (0 censored) |
| 936 | random | 8/7 | 6/5 | 4.470/3.909 | 2208/>3000 (censored) | 178.0 (1 censored) |
| 937 | random | 8/7 | 6/6 | 4.914/4.409 | 1522/1667 | 252.8 (0 censored) |
| 938 | random | 8/7 | 6/6 | 5.302/4.165 | 996/2602 | 568.5 (0 censored) |
| 939 | random | 8/8 | 6/6 | 5.201/5.140 | 1402/1402 | 30.7 (0 censored) |

## filter activation と trajectory 監査

learned: ready平均step 242.3, filter評価平均step 257.4, first rejectionなし。shuffled: ready平均step 242.3, filter評価平均step 257.4, first rejectionなし。

`prefix exact` は first rejectionより前の物理trajectory、`pool through rejection` はfirst rejectionを行う候補集合までのhash一致。active filterでfull trajectoryが異なるのは正常。

| guided selector | audits | prefix exact | pool through rejection exact | full trajectory exact | full pool-hash exact | final state/training hash exact |
|---|---:|---:|---:|---:|---:|---:|
| forest_filter | 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| forest_filter_shuffled | 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

## 読み方と次の判断

`cell_archive.json` の raw cell を、reachability audit の prototype shape と exact layer key に照合した。shape 判定閾値は各 run の configuration にある `rms_tolerance=0.02`, `max_tolerance=0.05` を使用した。

なお kinematic で発見済みでも feasible elite がまだない cell を一律に duplicate として除外すると feasible 修復を妨げる。少なくとも feasible 未発見 canonical cell は filter 対象外にするか、発見 emitter と feasibility-repair emitter の予算を分離すべきである。

learned と shuffled の一致は、分類器の情報価値が同程度だったことを意味しない。どちらも gate を通過する duplicate がゼロで、分類結果が action に接続されなかった。ログされた learned 選択候補の最大 class probability は 0.603、mean-probability threshold は 0.900。offline calibrationでknown-duplicate rejectionとpseudo-unknown retentionを同時に制約し、非ゼロの安全な棄却率を持つ閾値を固定して再評価する必要がある。
