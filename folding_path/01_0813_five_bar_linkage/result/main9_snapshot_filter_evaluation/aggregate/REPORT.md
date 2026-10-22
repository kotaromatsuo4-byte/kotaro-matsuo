# main9 snapshot counterfactual filter evaluation

Seeds: 3.  各行はraw kinematic cell数が初めて2/3/4になった直後にtree・archive・教師を凍結し、別RNGの候補をcounterfactual rolloutした結果です。panel rolloutは学習にもarchive更新にも使用していません。

## 先に結論

独立な教師macro数（seed平均）は 2 cell=1.0, 3 cell=3.7, 4 cell=5.0 だけでした。
実装どおりの安全なstrict filterの最大棄却率は 0.000 で、この時期にはどのモデルも選択圧を発生させませんでした。
安全gateを外した順位診断では、4 cell時点の上位10%で最良だった `knn` が reject precision=0.750, duplicate rejection=0.106, unknown retention=0.929, survivor novelty lift=1.025, valuable retention=0.908 でした。弱い順位情報は見えますが、初期filterとして安全に使える強さではありません。
全1152候補中、終端cellを確定できたvalid候補は1149件でした。filter precisionはこのvalid集合に条件付けています。表・図の集約値は候補をpoolした比率ではなく、各seed指標の単純平均です。

## 終端cell分類

|cells|model|snapshot step|train macros|panel known coverage|known top-1|open-world top-1|Brier|ECE|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|2|extra_trees|111|1.0|0.410|0.463|0.201|1.073|0.537|
|2|knn|111|1.0|0.410|0.463|0.201|1.073|0.537|
|2|random_forest|111|1.0|0.410|0.463|0.201|1.073|0.537|
|3|extra_trees|230|3.7|0.463|0.463|0.217|0.729|0.151|
|3|knn|230|3.7|0.463|0.589|0.277|0.667|0.174|
|3|random_forest|230|3.7|0.463|0.491|0.230|0.723|0.244|
|4|extra_trees|297|5.0|0.667|0.374|0.245|0.787|0.195|
|4|knn|297|5.0|0.667|0.376|0.247|0.767|0.159|
|4|random_forest|297|5.0|0.667|0.343|0.227|0.780|0.179|

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
`valuable`は現行dual archiveでnew kinematic/new feasible/elite replacementになる候補です。現在のquality keyに入っていないstrainや同一cell内の微小な節点座標差まで含む、将来の価値関数一般を表す指標ではありません。

## Files

- `classification_diagnostics.png`: top-1、Brier、ECE
- `calibration_reliability.png`: 予測confidenceと実正解率のreliability曲線
- `snapshot_filter_summary.png`: 教師macro数と主要分類・filter指標の要約
- `filter_selection_pressure.png`: duplicate/unknown指標に加えvaluable retentionとuseless-reject precision
- `snapshot_metrics.csv`: snapshot/model単位の分類指標
- `filter_metrics.csv`: strictおよびranked診断
- `calibration_bins.csv`: reliability-binの生データ
- `panel_predictions.csv`: 全候補の予測とcounterfactual真値（seed directory）