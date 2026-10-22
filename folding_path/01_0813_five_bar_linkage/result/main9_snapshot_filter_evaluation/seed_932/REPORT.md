# main9 snapshot counterfactual filter evaluation

Seeds: 1.  各行はraw kinematic cell数が初めて2/3/4になった直後にtree・archive・教師を凍結し、別RNGの候補をcounterfactual rolloutした結果です。panel rolloutは学習にもarchive更新にも使用していません。

## 終端cell分類

|cells|model|snapshot step|train macros|panel known coverage|known top-1|open-world top-1|Brier|ECE|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|2|extra_trees|90|1.0|0.323|0.220|0.071|1.561|0.780|
|2|knn|90|1.0|0.323|0.220|0.071|1.561|0.780|
|2|random_forest|90|1.0|0.323|0.220|0.071|1.561|0.780|
|3|extra_trees|171|3.0|0.461|0.508|0.234|0.637|0.102|
|3|knn|171|3.0|0.461|0.678|0.312|0.494|0.190|
|3|random_forest|171|3.0|0.461|0.593|0.273|0.578|0.143|
|4|extra_trees|215|4.0|0.695|0.539|0.375|0.710|0.223|
|4|knn|215|4.0|0.695|0.517|0.359|0.614|0.092|
|4|random_forest|215|4.0|0.695|0.539|0.375|0.686|0.197|

top-1/Brier/ECEの分母は、snapshot archiveに存在する既知cellへ実際に到達したheld-out候補です。archiveにはあるが教師に未登場のcellも誤答・確率0として数える固定label-space評価です。
ただし2/3/4 cellでは利用可能なStateNodeと候補panel自体が変わるため、この横軸は同一test setの通常のlearning curveではなく、各時点で実際に働く予測性能です。従って単調増加は仮定しません。open-world top-1はvalidな未知cell候補をすべて誤答として同じ分母へ含めています。

## 実装どおりのstrict filter

|cells|model|rejectable|reject frac|precision|duplicate rejection|unknown retention|novelty lift|valuable retention|useless precision|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|2|extra_trees|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|2|extra_trees|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|2|knn|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|2|knn|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|2|random_forest|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|2|random_forest|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|extra_trees|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|extra_trees|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|knn|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|knn|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|random_forest|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|3|random_forest|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|extra_trees|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|extra_trees|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|knn|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|knn|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|random_forest|all_known|0.000|NA|0.000|1.000|1.000|1.000|NA|
|4|random_forest|feasible|0.000|NA|0.000|1.000|1.000|1.000|NA|

`all_known`はcoverageだけを目的に全既知cellを棄却可能とした版、`feasible`は現行main9と同じくcollision-free eliteを持つcellだけを棄却可能とした版です。20% epsilon-randomは候補固有の判定ではないため、この表には混ぜていません。実運用時は表の選択圧がさらに80%へ薄まります。

`filter_selection_pressure.png`の線は安全閾値ではなく、support gateを掛けないraw class-confidenceを順位付けして上位5–50%を仮に除いた診断です。したがって線が改善してstrict記号が動かない場合は『順位情報はあるが、安全に発火できる教師量ではない』ことを意味します。

## Files

- `classification_diagnostics.png`: top-1、Brier、ECE
- `filter_selection_pressure.png`: duplicate/unknown指標に加えvaluable retentionとuseless-reject precision
- `snapshot_metrics.csv`: snapshot/model単位の分類指標
- `filter_metrics.csv`: strictおよびranked診断
- `calibration_bins.csv`: reliability-binの生データ
- `panel_predictions.csv`: 全候補の予測とcounterfactual真値（seed directory）