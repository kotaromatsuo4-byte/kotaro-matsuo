# Persistent-K offline paired panel

`persistent_k_panel.py`は、同一の親状態と第1摂動
\((q,v_1)\)から、異なる摂動持続回数 \(K\) を独立に分岐実行する
offline検証用コードです。online探索中にarchiveやhistoryが変化する影響を
切り離し、次を確認します。

- \(K\)だけを変えたときに、flat/collisionや終端cellが変化するか
- 未知の\((q,v_1)\)について、GPが有望な\(K\)を順位付けできるか
- GPの順位情報が、同じscore分布をランダムに対応付けた対照より有効か
- 長い\(K\)のmechanics costと途中終了回数がどの程度か

主評価は、SVD基底に依存しない物理節点変位を入力する
`basis_free_gp`です。`raw_coefficient_gp`は、同じ第1摂動を生成したSVD係数を
入力する参考対照です。cross-validationでは、同じ\((q,v_1)\)に属するすべての
\(K\)を必ず同一foldへ入れます。

本pilot用の既定値は \(K=1,\ldots,24\)、6親、各親1方向、3-foldです。

```powershell
python persistent_k_panel.py `
  --seeds 700 701 702 `
  --parents 6 `
  --v1-per-parent 1 `
  --folds 3 `
  --k-values 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 `
  --max-suffix-steps 100 `
  --result-name persistent_k_panel_seed700_702
```

コード確認用にはseed 998、3 group、\(K=1,2,3\)、suffix上限20のsmall
smokeを使用しました。本pilotは上記コマンドで実行済みです。

各seedについて以下を保存します。

- `panel_outcomes.csv`: 各\((q,v_1,K)\)の実際の終端・cell・cost
- `cv_predictions.csv`: held-out GP scoreとgroup内shuffle score
- `panel_arrays.npz`: 二種類のGP入力と終端descriptor
- `summary.json`: group単位の順位性能、cell変化率、K別統計

全seedをまとめた値は`aggregate_summary.json`へ保存します。

## 本pilot結果：seed 700--702

各seedで6個の\((q,v_1)\) groupを作り、各groupから\(K=1,\ldots,24\)の
24 branchを実行しました。合計は18 group、432 branchです。全groupで24個の
Kが揃い、同じgroupの行が複数foldへ漏れていないことを確認しました。

| seed | panel cells | mechanics edges | flat率 | collision率 | wall time [s] |
|---:|---:|---:|---:|---:|---:|
| 700 | 3 | 5,437 | 0.542 | 0.458 | 42.31 |
| 701 | 5 | 5,017 | 0.618 | 0.382 | 37.40 |
| 702 | 6 | 3,824 | 0.458 | 0.542 | 30.80 |
| 合計 | - | 14,278 | 0.539 | 0.461 | 110.51 |

全432 branchの停止理由はflat 233件、collision 199件でした。timeout、
corrector failure、transport projection failureはありませんでした。

### Kを選択する価値があるか

| group単位の事実 | 結果 |
|---|---:|
| Kによって終端cellが変化したgroup | 12/18 = 0.667 |
| Kによって停止理由またはcellが変化したgroup | 16/18 = 0.889 |
| Kによってflat/collisionが変化したgroup | 15/18 = 0.833 |
| group当たりの異なるflat cell数 | 平均1.94、最大4 |
| 一様ランダムKの期待新規cell率 | 0.338 |
| 新規cellを生むKが存在したgroup | 15/18 = 0.833 |
| oracle top-25%の平均新規cell率 | 0.787 |

したがって、five-barでは\((q,v_1)\)を固定してもKは結果を大きく変えます。
「方向持続を使うか」だけでなく、「何回持続するか」を選択変数にする根拠が
得られました。一方、3 groupにはどのKでも新規cellが存在しないため、Kの選択
だけでは救えない親状態・第1方向もあります。

### held-out groupでのGP順位性能

表のtop-1は、各held-out groupで最高scoreのKを1個選んだときの新規cell率です。
top-25%は上位6個のKに含まれる新規cellの割合です。

| 選択規則 | top-1 | top-25% | top-25%内に1個以上新規 | row AUC | top-1 cost |
|---|---:|---:|---:|---:|---:|
| 一様ランダム期待値 | 0.338 | 0.338 | - | - | 33.05 |
| basis-free GP | **0.667 (12/18)** | **0.593 (64/108)** | 0.722 | **0.604** | 40.94 |
| basis-free score shuffle | 0.389 (7/18) | 0.324 (35/108) | 0.722 | 0.476 | 34.56 |
| raw coefficient GP | 0.444 (8/18) | 0.481 (52/108) | 0.722 | 0.509 | 36.72 |
| raw score shuffle | 0.333 (6/18) | 0.315 (34/108) | 0.833 | 0.389 | 30.83 |
| oracle | 0.833 (15/18) | 0.787 | - | - | - |

主評価のbasis-free GPは、一様ランダム期待値とscore-shuffled対照の両方を
top-1およびtop-25%平均で上回りました。同じscore集合をgroup内でshuffle
すると性能が低下したため、単に候補数を25%へ減らした効果ではなく、GPの
Kへの順位対応に情報が含まれていたと解釈できます。raw coefficient GPよりも
basis-free GPが高く、物理変位表現を採用する方針とも整合します。

ただしseed別のbasis-free top-1は、seed 700で5/6、701で5/6、702で2/6です。
shuffleとの差は700と701で正、702では同率でした。3 seed・18 groupのoffline
結果なので、汎用的な優位性の証明ではありません。またbasis-free GPは長いKを
好み、top-1の平均costが一様ランダムの33.05から40.94へ増えています。固定
mechanics予算のonline探索で、この1 rollout当たりの追加costを新規cell率の向上が
上回るかを別途確認する必要があります。

### K別結果

各Kについて18 branchを平均しました。

| K | flat | collision | 新規cell | 実行K | mechanics cost |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.667 | 0.333 | 0.000 | 1.00 | 28.61 |
| 2 | 0.667 | 0.333 | 0.111 | 1.89 | 28.39 |
| 3 | 0.667 | 0.333 | 0.222 | 2.78 | 28.33 |
| 4 | 0.556 | 0.444 | 0.222 | 3.67 | 26.06 |
| 5 | 0.556 | 0.444 | 0.222 | 4.56 | 27.22 |
| 6 | 0.444 | 0.556 | 0.167 | 5.44 | 24.89 |
| 7 | 0.500 | 0.500 | 0.222 | 6.33 | 27.33 |
| 8 | 0.500 | 0.500 | 0.278 | 7.22 | 26.67 |
| 9 | 0.500 | 0.500 | 0.278 | 8.11 | 28.22 |
| 10 | 0.611 | 0.389 | 0.333 | 9.00 | 31.39 |
| 11 | 0.500 | 0.500 | 0.333 | 9.89 | 31.22 |
| 12 | 0.611 | 0.389 | 0.444 | 10.78 | 33.44 |
| 13 | 0.667 | 0.333 | **0.500** | 11.67 | 34.94 |
| 14 | 0.556 | 0.444 | 0.444 | 12.56 | 34.56 |
| 15 | 0.556 | 0.444 | 0.444 | 13.44 | 35.28 |
| 16 | 0.611 | 0.389 | **0.500** | 14.33 | 36.00 |
| 17 | 0.556 | 0.444 | 0.444 | 15.22 | 37.50 |
| 18 | 0.556 | 0.444 | **0.500** | 16.11 | 38.83 |
| 19 | 0.500 | 0.500 | 0.444 | 17.00 | 38.83 |
| 20 | 0.444 | 0.556 | 0.389 | 17.89 | 38.39 |
| 21 | 0.389 | 0.611 | 0.389 | 18.78 | 37.94 |
| 22 | 0.389 | 0.611 | 0.389 | 19.67 | 38.11 |
| 23 | 0.444 | 0.556 | 0.389 | 20.50 | 39.83 |
| 24 | 0.500 | 0.500 | 0.444 | 21.33 | 41.22 |

新規cell率はK=1--5の平均0.156に対し、K=12--18では0.468でした。今回の
group集合では中長期の方向持続が有効です。ただしK=19--24ではcollision率が
0.556へ増え、新規cell率は0.407へ低下しました。したがって「常に最大K」では
なく、状態と第1摂動に応じた有限のKを選ぶ問題になっています。

## 保存済みpanelによるcost-aware再評価

上記432 branchを再利用し、mechanicsは再実行せずにcost-aware scoreを評価
しました。`persistent_k_cost_aware.py`は保存済みfoldをそのまま使い、学習側の
complete groupだけから

\[
  \log(1+C) \sim \operatorname{GP}
\]

を学習します。held-out候補の予測costには、安価側へ楽観するLCBではなく、
log-normal posteriorの平均

\[
  \widehat C(x)
  =\exp\!\left(\mu_{\log C}(x)
      +\frac{1}{2}\sigma^2_{\log C}(x)\right)-1
\]

を用い、最小値を1 mechanics stepとしました。評価する獲得値は

\[
  a_{\mathrm{cost}}(x)
  =\frac{a_{\mathrm{basis\mbox{-}free}}(x)}{\widehat C(x)}
\]

です。分子と分母のどちらにもheld-out branchの実測結果を使っていません。
`new/step proxy`は、選択集合内の平均新規cell指示値を、同じ集合の実測平均
mechanics costで割った値です。

| 選択規則 | top-1新規率 | top-1 cost | top-1 new/step | top-25%新規率 | top-25% cost | top-25% new/step |
|---|---:|---:|---:|---:|---:|---:|
| 一様ランダム期待値 | 0.338 | 33.05 | 0.01023 | 0.338 | 33.05 | 0.01023 |
| 通常basis-free GP | **0.667** | 40.94 | 0.01628 | 0.593 | 39.38 | 0.01505 |
| basis-free GP / 予測cost | 0.611 | **37.11** | **0.01647** | **0.639** | 37.33 | **0.01711** |
| cost-aware score shuffle | 0.500 | 36.17 | 0.01382 | 0.352 | **33.06** | 0.01064 |

top-1ではcost-aware化により平均costが40.94から37.11へ9.4%下がった一方、
新規率も0.667から0.611へ下がりました。new/step proxyの改善は1.1%に留まり、
argmax置換を正当化するほど強い結果ではありません。

top-25% filterでは、新規率が0.593から0.639へ上がり、平均costも39.38から
37.33へ5.2%下がりました。その結果new/step proxyは0.01505から0.01711へ
13.7%改善し、一様ランダムの1.67倍、score-shuffled対照の1.61倍でした。
このpanelでは、cost-aware化はargmaxよりtop-quantile filterと組み合わせた方が
有望です。

ただしcost GP自体の予測は強くありません。全行のMAEは17.59 steps、RMSEは
23.94 stepsで、全行を混ぜたSpearman相関は-0.058でした。K選択に直接関係する
group内Spearman相関は、costが一定だった2 groupを除く16 groupで平均0.274、
中央値0.251と弱い正相関です。seed別にはデータが4 training group/foldしかなく、
状態間のcost水準を十分に学べていません。

したがって、この結果からの判断は次の通りです。

- cost-aware **argmaxへの全面置換はno-go**です。効率改善が1.1%しかなく、
  新規cell率を下げています。
- cost-aware **top-25% filterのonline対照実験は条件付きgo**です。実装する場合も
  通常filterを残したpaired A/Bとし、探索規則を確定したとは扱いません。
- このoffline指標はbaseline trunkに対する新規性であり、逐次変化するarchiveでの
  発見数/固定予算を直接表しません。最終判断はonline mechanics予算を揃えた比較で
  行う必要があります。

再評価の詳細は`result/persistent_k_panel_seed700_702/`内の
`cost_aware_offline_summary.json`、各seedの`cost_aware_cv_predictions.csv`、
追記済み`aggregate_summary.json`に保存しています。
