# main9_aggressive_b3000 forest-filter comparison (B=3000)

## 結論

aggressive filter は実際に選択圧を発生させた。3 seeds 合計で learned は 10937/104448 (10.47%)、shuffled は 8203/102912 (7.97%) の filter-evaluated candidates を棄却した。

最初の棄却より前の paired trajectory と、その棄却判断に使った candidate-pool hash は 6/6 組で random と完全一致した。棄却後の trajectory 分岐は filter が作用した結果であり、監査失敗ではない。

learned − random は canonical kinematic AUC +0.076、feasible AUC -0.095、restricted T6 は kinematic -70.0 step、feasible +563.3 step。

この 3-seed pilot では learned filter は kinematic T6をわずかに早めた一方、feasible AUCを下げ、feasible T6を遅らせた。したがって random より多様な feasible 解を高速化したとは言えない。

learned − shuffled は final feasible canonical coverage +0.667、feasible AUC +0.079、restricted feasible T6 -7.0 step。完全到達は learned 3/3、shuffled 1/3 で、learned labels による信頼性向上の兆候はあるが、3-seed pilotでは確証にならない。

## 集計

AUC は mechanics step 0..3000 における累積 coverage の時間平均。T6 の `restricted` は未達を3000で右打切りした `mean(min(T6, 3000))`。rejection fraction の分母は、epsilon bypass/cold randomを除き、forest gateで実際に評価した候補数。

| selector | raw final (kin/feas) | canonical final (kin/feas) | canonical AUC (kin/feas) | canonical T6 kin | canonical T6 feas | macros / teachers | rejected/evaluated (fraction); raw | fit/score/wall [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8.000 ± 0.000 / 7.333 ± 0.577 | 6.000 ± 0.000 / 6.000 ± 0.000 | 5.114 ± 0.384 / 4.699 ± 0.494 | 1035.7 (3/3); restricted 1035.7 | 1677.3 (3/3); restricted 1677.3 | 61.000 ± 1.000 / 709.000 ± 63.380 | 0/0 (NA); raw 0 | 0.000 ± 0.000 / 0.081 ± 0.003 / 44.512 ± 0.558 |
| forest_filter | 8.000 ± 0.000 / 7.000 ± 1.000 | 6.000 ± 0.000 / 6.000 ± 0.000 | 5.190 ± 0.197 / 4.604 ± 0.350 | 965.7 (3/3); restricted 965.7 | 2240.7 (3/3); restricted 2240.7 | 62.667 ± 2.517 / 722.000 ± 31.432 | 10937/104448 (10.47%); raw 10937 | 42.238 ± 1.947 / 5.693 ± 0.661 / 91.094 ± 2.976 |
| forest_filter_shuffled | 8.000 ± 0.000 / 6.667 ± 0.577 | 6.000 ± 0.000 / 5.333 ± 0.577 | 5.199 ± 0.239 / 4.525 ± 0.619 | 823.3 (3/3); restricted 823.3 | 743.0 (1/3); restricted 2247.7 | 60.333 ± 2.517 / 654.667 ± 20.232 | 8203/102912 (7.97%); raw 8203 | 38.905 ± 2.419 / 5.287 ± 0.433 / 88.231 ± 2.322 |


raw archive は layer-order variant を含むため、canonical target数より多い値になり得る。
random の canonical cell first-seen→first-feasible lag は、各 seed 内の観測済みcell平均をさらに3 seedで平均して 207.4 step。全 seed 合計で 0 cell が右打切りだった。

## 対応差（left − right）

coverage/AUC は正が left 有利、T6 と時間は負が left 有利。95% CI は seed 対応 bootstrap、p は exact sign-flip test。

| left | right | canonical kin AUC | canonical feas AUC | kin T6 capped | feas T6 capped | wall [s] |
|---|---|---:|---:|---:|---:|---:|
| forest_filter | random | +0.076 [-0.112, +0.340], p=1.000 | -0.095 [-0.473, +0.267], p=0.750 | -70.000 [-607.000, +397.000], p=1.000 | +563.333 [-67.000, +1523.000], p=0.500 | +46.582 [+43.418, +48.351], p=0.250 |
| forest_filter | forest_filter_shuffled | -0.009 [-0.108, +0.082], p=1.000 | +0.079 [-0.469, +0.671], p=0.750 | +142.333 [+0.000, +391.000], p=0.500 | -7.000 [-1429.000, +1517.000], p=1.000 | +2.863 [+2.191, +4.207], p=0.250 |
| forest_filter_shuffled | random | +0.085 [-0.004, +0.259], p=1.000 | -0.174 [-0.404, -0.004], p=0.250 | -212.333 [-643.000, +6.000], p=1.000 | +570.333 [+6.000, +1362.000], p=0.250 | +43.719 [+41.227, +45.787], p=0.250 |

## seed別 canonical coverage

| seed | selector | raw final kin/feas | canonical final kin/feas | canonical AUC kin/feas | T6 kin/feas | mean first-seen→feasible lag |
|---:|---|---:|---:|---:|---:|---:|
| 920 | random | 8/7 | 6/6 | 5.405/4.281 | 732/2657 | 561.8 (0 censored) |
| 920 | forest_filter | 8/8 | 6/6 | 5.405/4.202 | 732/2891 | 601.7 (0 censored) |
| 920 | forest_filter_shuffled | 8/7 | 6/5 | 5.405/4.167 | 732/>3000 (censored) | 188.4 (1 censored) |
| 921 | random | 8/8 | 6/6 | 4.678/4.572 | 1638/1638 | 53.3 (0 censored) |
| 921 | forest_filter | 8/7 | 6/6 | 5.019/4.839 | 1031/1571 | 90.0 (0 censored) |
| 921 | forest_filter_shuffled | 8/6 | 6/5 | 4.937/4.168 | 995/>3000 (censored) | 29.8 (1 censored) |
| 922 | random | 8/7 | 6/6 | 5.258/5.244 | 737/737 | 7.2 (0 censored) |
| 922 | forest_filter | 8/6 | 6/6 | 5.147/4.771 | 1134/2260 | 187.7 (0 censored) |
| 922 | forest_filter_shuffled | 8/7 | 6/6 | 5.254/5.240 | 743/743 | 7.2 (0 censored) |

## filter activation と trajectory 監査

learned: ready平均step 266.0, filter評価平均step 301.0, first rejection平均step 732.0。shuffled: ready平均step 266.0, filter評価平均step 301.0, first rejection平均step 732.0。

`prefix exact` は first rejectionより前の物理trajectory、`pool through rejection` はfirst rejectionを行う候補集合までのhash一致。active filterでfull trajectoryが異なるのは正常。

| guided selector | audits | prefix exact | pool through rejection exact | full trajectory exact | full pool-hash exact | final state/training hash exact |
|---|---:|---:|---:|---:|---:|---:|
| forest_filter | 3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| forest_filter_shuffled | 3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |

## 読み方と次の判断

`cell_archive.json` の raw cell を、reachability audit の prototype shape と exact layer key に照合した。shape 判定閾値は各 run の configuration にある `rms_tolerance=0.02`, `max_tolerance=0.05` を使用した。

なお kinematic で発見済みでも feasible elite がまだない cell を一律に duplicate として除外すると feasible 修復を妨げる。少なくとも feasible 未発見 canonical cell は filter 対象外にするか、発見 emitter と feasibility-repair emitter の予算を分離すべきである。

この aggressive panel は 3 seeds の探索的評価であり、bootstrap CIとsign-flip p値は推定精度が低い。learned-vs-randomだけでなくlearned-vs-shuffledを主要比較にして、情報を持つlabelの寄与と単なる棄却効果を分離する必要がある。
