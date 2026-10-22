# main9 最終結果: 衝突継続・終端 cell 分類・duplicate filter

## 結論

今回の条件では、終端 cell classifier を安全な filter として使って
完全ランダム探索より 6 cell 発見を高速化することは確認できなかった。

一方で、次の物理規則を組み合わせたrandom探索baseline自体は高い到達性能を示した。

- 衝突は rollout の終端条件にせず、edge ごとの経路品質として記録する。
- action を `(parent state, first physical displacement, K)` とし、同じ物理方向を
  各接空間へ transport しながら `K=1..24` 回持続させる。
- K 回の後は deterministic suffix を繰り返し、flat terminal を探す。

この collision-continuing variable-K random policy は、標準オンライン評価の
10/10 seed で canonical 6 cell を運動学的に発見した。classifier を使わない
この探索器が、現時点の five-bar に対する採用アルゴリズムである。
ただし、variable-Kの効果とcollision continuationの効果をこの比較だけで分離した
わけではないため、後者単独の高速化とは主張しない。

forest filter は安全閾値では一件も候補を棄却せず random と完全同一になった。
閾値を緩めて実際に 8--10% 程度を棄却すると、運動学的発見を少し早める場合は
あったが、collision-free な 6 経路の発見は平均で悪化した。したがって、
classifier は当面 shadow 評価だけに置き、十分な prequential 証拠が得られた時だけ
有効化する設計が妥当である。

## 実装

- `main9.py`: variable-K persistent macro、collision continuation、random / forest
  filter / score-shuffled filter の共通探索器。
- `collision_continue.py`: edge collision 計測、root-to-state 経路品質、kinematic / 
  feasible 二層 archive。
- `forest_cell_model.py`: NumPy-only bootstrap Random Forest / ExtraTrees、
  class-conditional support 距離、committee disagreement、保守的 duplicate mask。
- `main9_forest_evaluation.py`: macro-group 分離の固定 holdout learning curve、
  leave-one-cell-out pseudo-unknown 評価。
- `main9_online_aggregate.py`: raw / canonical、kinematic / feasible を分離した
  online 集計と paired trajectory 監査。

入力は SVD 係数ではなく、基底不変な物理量

`x = [q_free, last_nonflat_context_free, v1_free, log(1+K)/log(25)]`

である。出力は online archive で固定された terminal cell label のみ。既知 cell
内の微小 descriptor、ひずみ、衝突数、経路長は classifier に回帰させず、terminal
取得後に外部計算する。

候補生成は各decisionでcollision-freeな非終端parentを最大16個一様抽出し、各parentに
2本の等方的な物理接方向を生成し、K=1..24を直積する。`random`はこの共通bounded
pool（最大768候補）から一様選択する。全history直積を明示生成する方式ではない。

### collision と archive

collision は flat 判定を妨げない。各 terminal cell には次を別々に保持する。

- `kinematic_elite`: collision を許して到達した代表。更新は feasibility-first。
- `feasible_elite`: root から terminal まで collision edge が 0 の代表。

同一 cell の品質比較は、collision の有無、collision edge 数、正規化幾何経路長、
摂動回数、edge 数の辞書式順序で行う。collision-free elite が短い衝突経路で
置換されることはない。

実ログでもcollisionをstop reasonにしたmacroは0であり、collisionを含んだ後も
flatまで到達したmacroは20k corpusで577、評価版B3000 randomで272件あった。
従ってcontinuationは単なる設定値ではなく実際に作動している。

### 教師

一つの macro outcome から persistent chain の残存 action
`(q_j, v_{j+1}, K-j) -> same terminal cell` を hindsight 教師として保存する。
ただし全行は同じ `macro_id` を共有し、重みの合計を 1 とする。bootstrap と
train/test split も macro 単位で行うため、長い K を独立試行として水増ししない。

filter は次の条件がすべて成立する候補だけを既知 duplicate として棄却する。

- 予測 cell に feasible elite がある。
- committee の平均確率・下側確率・vote 一致率が閾値以上。
- ensemble disagreement と class-conditional support 距離が閾値以下。
- その class の独立 macro 数が最小数以上。

未知／OOD 候補は残し、20% は常に filter を bypass する。score-shuffled 対照は
棄却数を完全に保ったまま候補との対応だけを壊す。

## 大規模 random corpus と learning curve

seed 800--802 について各 20,000 mechanics step の random tree を生成した。
合計は 60,000 state transition、1,250 macro、13,364保存行（primary＋hindsight）
である。censorを除いた分類対象は1241 macroだった。
各 seed の最後の 64 macro を固定 test とし、同一 macro の行を train/test に
跨がせず、nested training size で評価した。

canonical kinematic 6 cell の T6 は 548 / 770 / 1121 step だった。その時点で
利用できる独立 macro は 11 / 14 / 17 本だけで、保存行数 89 / 159 / 225 は
強く相関した hindsight 行である。各 terminal class の独立例は多くが 1--3 本で、
最後の cell を選ぶ直前にはさらに一例少ない。

代表的な closed-set 精度は次の通り。

| training macro | Random Forest | ExtraTrees | group-kNN | prevalence |
|---:|---:|---:|---:|---:|
| 16 | 0.216 | 0.198 | 0.222 | 0.222 |
| 32 | 0.214 | 0.214 | 0.224 | 0.177 |
| 96 | 0.281 | 0.292 | 0.328 | 0.188 |
| 320 | 0.401 | 0.406 | 0.464 | 0.135 |

全学習器が prevalence を継続的に上回り始めるのは概ね 32 macro、中央値で
約 1669 mechanics step であり、すでに corpus の T6 中央値 770 を過ぎている。
厳格filterでは RF / ExtraTrees は 320 macro でも棄却 0、kNN が初めて非ゼロの
安全棄却を出すのは 96 macro、中央値約 4918 step だった。

閾値を緩めると既知 duplicate は落とせるが、leave-one-cell-out で未知 cell を
保持できなくなる。例として ExtraTrees は training 32 macro で duplicate を
2.6% 棄却し未知を95.8%保持したが、64 macroでは棄却7.3%に対して未知保持91.1%。
Random Forest は32 macroで4.7% / 92.2%、64 macroで10.9% / 88.5%だった。

その後のtargeted LOCOで、contextありgroup-kNNだけは320 macro時点にprecision 1.0、
duplicate rejection 5.7%、unknown retention 95.8%となり、3 seed中2 seedで事前gate
を満たした。従って「分類filterは永遠に学習不能」ではなく、**約320独立macro
（中央値約15,509 mechanics step）あれば安全な弱いfilterを学べる兆候がある**。
ただしT6は11--17 macroで完了するため、多様性発見の高速化には約20倍遅い。

## なぜ学習が難しいか

第一の理由は、探索が成功する方が学習より早いことである。T6 前には独立 macro
が中央値14本しかなく、7--8 class を分類するには不足している。

第二の理由は、データを増やしても局所的な basin 混在が強いことである。全1241
macroの primary action で、最近傍が同じ cell になる率は0.380、10近傍の正規化
class entropy は0.675だった。hindsight supportを使っても0.461 / 0.599にしか
改善しない。入力距離が近い20%に限定してもsame-cell率は0.548（hindsightあり
0.593）であり、滑らかな局所写像ではない。

K は重要な探索自由度だが、K 単独で cell は決まらない。全 K が全8 raw cellを
生成し、相互情報量 `I(cell;K)=0.073 nat` は class entropy の3.55%にすぎない。
有効なのは state、初期方向、K の組であり、その境界が細かく入り組んでいる。

random-order training と chronological training の精度差は RF -1.46 point、
ExtraTrees +1.89 point、kNN -0.35 point 程度だった。従って主因は単なる時間的
covariate shiftでも、強いclass imbalanceでもない。

## Online comparison

### 現行v1.1（局所timeoutなし）の安全filter、10 seed、B=3000

forest は平均242 stepで fit 可能になり、平均257 stepから候補を評価した。しかし
10 seedの336,384評価候補で棄却は0だった。random、learned、shuffled の全20組で
候補pool hash、全macro trajectory、最終state hash、training hashが完全一致した。
各selectorで598 macroがflatへ到達し、9 macroは全体予算でcensor、local timeoutは
0だった。

| 指標 | random / learned / shuffled |
|---|---:|
| canonical kinematic 6 到達 | 10 / 10 seed |
| canonical kinematic T6 平均 | 1130.4 step |
| canonical feasible 6 到達 | 8 / 10 seed |
| feasible T6（未達を3000で打切り） | 1944.3 step |
| learned の追加wall time | +57.6 s / run |

これは小さな差ではなく、選択圧そのものがゼロだった比較である。
wall timeは30 processを同時実行した観測値なので絶対値のbenchmarkではないが、
同一batch内でforest fittingが純粋な追加費用になったことは変わらない。

### 攻めたfilter、3 seed、B=3000

offlineで検討した緩い閾値を固定して再実行すると、learned は評価対象候補の
10.47%、shuffled は7.97%を棄却した。最初の棄却まではすべての対応runで候補poolと
trajectoryが一致し、分岐後だけが異なることを監査済みである。

learned - random は canonical kinematic AUC +0.076、kinematic T6 -70 stepだったが、
canonical feasible AUC -0.095、feasible T6 +563 stepと悪化した。learned - shuffled
では feasible 6 到達が3/3対1/3だったが、N=3であり、AUC差も一貫しないため有効性
の証拠とはしない。

## 採用する探索アルゴリズム

現段階では次を採用する。

1. collision-continuing variable-K persistent random を coverage emitter とする。
2. kinematic archive と feasible archive を分離する。
3. terminal classifier は shadow mode で毎回 prequential 評価する。
4. filter は pseudo-unknown retention >= 0.95、duplicate precision >= 0.90、
   duplicate rejection >= 0.05 を macro-group validation で満たした時だけ有効化する。
5. 有効化後も最大棄却50%、20%完全random、score-shuffled監査を維持する。
6. kinematic 6 発見後は、未知cell探索と feasible path修復／既知cell品質改善を
   別quotaに分ける。

320 macro以後のgroup-kNNは、最初の6 cell探索ではなく、長時間のduplicate削減や
既知cell品質改善用filterの候補である。main9のonline比較では、T6前に動かせる可能性
を優先してforestを使ったが、結果として安全閾値では発動しなかった。

現在のfive-barでは手順4を満たさないため、実際の動作はrandomへ安全に退化する。
「classifier filterでrandomより速い」とは言えないが、誤った選択圧で探索を壊さず、
学習が本当に有用になった場合だけ使えるアルゴリズムとして実装した。
現行`main9.py`の既定selectorも`random`とし、learned filterは明示指定時だけ有効にした。

## 制約と次の改善候補

- 大規模offline corpusとaggressive pilotを生成した評価版 v1 の deterministic
  suffixには100 edgeの安全上限があり、corpusと
  online randomで各5 macroがtimeoutになった。collisionで止まったものではない。
  現行 v1.1 は既定値を0（局所timeoutなし）に変更し、flat、数値失敗、または全体予算
  まで継続する。保存済み数値を再現するときだけ`FOLDING_MAIN9_MAX_SUFFIX_STEPS=100`
  を明示する。
- canonical sixとraw archive cellを分離した。raw archiveでは重なり順variantにより
  通常8 cellとなる。
- collision品質は現在「衝突を含むedge数」であり、同一edge内の衝突部材pair数や
  penetration severityではない。必要ならcollision reportをevent/pair単位へ拡張する。
- offline corpusは3 seed、online標準比較は10 seed。five-bar以外への一般化は主張しない。
- 分類を改善するなら、単に木を深くするより、同一 `(parent,v)` に複数Kを割り当てる
  paired counterfactual、five-bar対称性augmentation、cell境界に沿ったmetric learning
  が候補。ただし、その教師生成費用がT6を超えるならオンライン高速化にはならない。

## 詳細成果物

- 実装仕様: `MAIN9_FOREST_FILTER.md`
- 評価仕様: `MAIN9_FOREST_EVALUATION.md`
- 学習可能性・局所混在の分析: `MAIN9_MODEL_ANALYSIS.md`
- `(q,v,K)` / context入力アブレーション: `MAIN9_FEATURE_ABLATION.md`
- 学習曲線図: `result/main9_model_analysis/learning_curves_t6.png`
- filter trade-off図: `result/main9_model_analysis/filter_tradeoff.png`
- 標準online結果: `MAIN9_ONLINE_RESULTS.md`
- 現行unbounded-suffix online結果: `MAIN9_UNBOUNDED_ONLINE_RESULTS.md`
- aggressive filter結果: `MAIN9_AGGRESSIVE_FILTER_RESULTS.md`
- strict learning curve: `result/main9_forest_eval_b20000/REPORT.md`
- loose/LOCO tradeoff: `result/main9_forest_eval_relative_threshold_loco/REPORT.md`
- 評価版v1 online機械可読集計: `result/main9_online_b3000_aggregate.json`
- 現行v1.1 online機械可読集計: `result/main9_unbounded_b3000_aggregate.json`
- aggressive機械可読集計: `result/main9_aggressive_b3000_aggregate.json`

保存済みoffline corpusを再現するPowerShell例（評価版v1と同じsuffix cap）は次の通り。

```powershell
$env:FOLDING_MAIN9_SELECTOR='random'
$env:FOLDING_RESULT_NAME='main9_random_corpus_b20000'
$env:FOLDING_STEP_BUDGET='20000'
$env:FOLDING_MAIN9_MAX_SUFFIX_STEPS='100'
$env:FOLDING_SAVE_TREE='0'
$env:FOLDING_VERBOSE='0'
foreach ($seed in 800..802) {
  $env:FOLDING_SEED="$seed"
  python main9.py
}

python main9_forest_evaluation.py result/main9_random_corpus_b20000 `
  --training-sizes 4,8,16,24,32,48,64,96,128,192,256,320 `
  --holdout-macros 64 `
  --model-kinds random_forest extra_trees knn prevalence `
  --output-directory result/main9_forest_eval_b20000
```

現行v1.1でflatまたは全体予算まで継続する場合は
`FOLDING_MAIN9_MAX_SUFFIX_STEPS`を削除するか0にする。
