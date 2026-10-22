# main9_online_b3000 forest-filter comparison (B=3000)

## 結論

現設定の forest filter は選択圧を一度も発生させていない。10 seeds 合計の actual/raw rejection は learned=0/0、shuffled=0/0。

forest の学習時間を除けば3手法は同じ探索として動作した。全20組で全 macro の候補集合 hash、物理 trajectory、最終 state hash、教師データ hash が random と完全一致した。

したがって、この panel から forest filter による高速化は主張できない。差が小さいのではなく、有効な filter action がゼロだったためである。

## 集計

AUC は mechanics step 0..3000 における累積 coverage の時間平均。T6 の `restricted` は未達を3000で右打切りした `mean(min(T6, 3000))`。rejection fraction の分母は、epsilon bypass/cold randomを除き、forest gateで実際に評価した候補数。

| selector | raw final (kin/feas) | canonical final (kin/feas) | canonical AUC (kin/feas) | canonical T6 kin | canonical T6 feas | macros / teachers | rejected/evaluated (fraction); raw | fit/score/wall [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8.000 ± 0.000 / 7.600 ± 0.699 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.014 ± 0.154 / 4.209 ± 0.505 | 1384.9 (10/10); restricted 1384.9 | 2105.9 (8/10); restricted 2284.7 | 62.100 ± 3.247 / 636.500 ± 33.357 | 0/0 (NA); raw 0 | 0.000 ± 0.000 / 0.072 ± 0.003 / 39.828 ± 1.563 |
| forest_filter | 8.000 ± 0.000 / 7.600 ± 0.699 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.014 ± 0.154 / 4.209 ± 0.505 | 1384.9 (10/10); restricted 1384.9 | 2105.9 (8/10); restricted 2284.7 | 62.100 ± 3.247 / 636.500 ± 33.357 | 0/367104 (0.00%); raw 0 | 33.686 ± 3.285 / 5.043 ± 0.622 / 77.375 ± 3.510 |
| forest_filter_shuffled | 8.000 ± 0.000 / 7.600 ± 0.699 | 6.000 ± 0.000 / 5.800 ± 0.422 | 5.014 ± 0.154 / 4.209 ± 0.505 | 1384.9 (10/10); restricted 1384.9 | 2105.9 (8/10); restricted 2284.7 | 62.100 ± 3.247 / 636.500 ± 33.357 | 0/367104 (0.00%); raw 0 | 33.742 ± 3.431 / 5.037 ± 0.590 / 77.555 ± 3.735 |


raw archive は layer-order variant を含むため、canonical target数より多い値になり得る。
random の canonical cell first-seen→first-feasible lag は、各 seed 内の観測済みcell平均をさらに10 seedで平均して 317.8 step。全 seed 合計で 2 cell が右打切りだった。

## 対応差（left − right）

coverage/AUC は正が left 有利、T6 と時間は負が left 有利。95% CI は seed 対応 bootstrap、p は exact sign-flip test。

| left | right | canonical kin AUC | canonical feas AUC | kin T6 capped | feas T6 capped | wall [s] |
|---|---|---:|---:|---:|---:|---:|
| forest_filter | random | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +37.548 [+35.263, +39.637], p=0.002 |
| forest_filter | forest_filter_shuffled | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | -0.179 [-0.446, +0.089], p=0.242 |
| forest_filter_shuffled | random | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +0.000 [+0.000, +0.000], p=1.000 | +37.727 [+35.303, +39.943], p=0.002 |

## seed別 canonical coverage

性能列は全 selector で同一なので random の値を示す。

| seed | selector | raw final kin/feas | canonical final kin/feas | canonical AUC kin/feas | T6 kin/feas | mean first-seen→feasible lag |
|---:|---|---:|---:|---:|---:|---:|
| 900 | random | 8/8 | 6/6 | 4.758/4.400 | 1696/1696 | 178.8 (0 censored) |
| 901 | random | 8/8 | 6/6 | 5.094/3.711 | 1611/2360 | 691.5 (0 censored) |
| 902 | random | 8/8 | 6/6 | 4.976/4.976 | 1749/1749 | 0.0 (0 censored) |
| 903 | random | 8/7 | 6/5 | 4.869/3.717 | 1601/>3000 (censored) | 108.4 (1 censored) |
| 904 | random | 8/8 | 6/6 | 4.990/4.407 | 1273/2216 | 291.7 (0 censored) |
| 905 | random | 8/8 | 6/6 | 5.023/4.493 | 1521/2131 | 265.0 (0 censored) |
| 906 | random | 8/8 | 6/6 | 5.244/3.768 | 970/2952 | 738.2 (0 censored) |
| 907 | random | 8/7 | 6/5 | 5.179/3.527 | 1057/>3000 (censored) | 445.6 (1 censored) |
| 908 | random | 8/6 | 6/6 | 4.865/4.252 | 1438/2656 | 306.5 (0 censored) |
| 909 | random | 8/8 | 6/6 | 5.145/4.839 | 933/1087 | 152.7 (0 censored) |

## filter activation と trajectory 監査

learned: ready平均step 265.4, filter評価平均step 276.2, first rejectionなし。shuffled: ready平均step 265.4, filter評価平均step 276.2, first rejectionなし。

`prefix exact` は first rejectionより前の物理trajectory、`pool through rejection` はfirst rejectionを行う候補集合までのhash一致。active filterでfull trajectoryが異なるのは正常。

| guided selector | audits | prefix exact | pool through rejection exact | full trajectory exact | full pool-hash exact | final state/training hash exact |
|---|---:|---:|---:|---:|---:|---:|
| forest_filter | 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| forest_filter_shuffled | 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

## 読み方と次の判断

`cell_archive.json` の raw cell を、reachability audit の prototype shape と exact layer key に照合した。shape 判定閾値は各 run の configuration にある `rms_tolerance=0.02`, `max_tolerance=0.05` を使用した。

なお kinematic で発見済みでも feasible elite がまだない cell を一律に duplicate として除外すると feasible 修復を妨げる。少なくとも feasible 未発見 canonical cell は filter 対象外にするか、発見 emitter と feasibility-repair emitter の予算を分離すべきである。

learned と shuffled の一致は、分類器の情報価値が同程度だったことを意味しない。どちらも gate を通過する duplicate がゼロで、分類結果が action に接続されなかった。ログされた learned 選択候補の最大 class probability は 0.634、mean-probability threshold は 0.900。offline calibrationでknown-duplicate rejectionとpseudo-unknown retentionを同時に制約し、非ゼロの安全な棄却率を持つ閾値を固定して再評価する必要がある。
