# K=2第2摂動角のoffline paired panel

## 目的

同一の第1摂動後状態 `q1` に対して、第2摂動角だけを変えたときの真の終端をすべて計算し、次の二点を分離して調べた。

1. 直前方向をそのまま維持する `theta=0` が、本当に新規終端cellを生みやすいか。
2. GPが同一`q1`内の第2摂動角に有効な選択圧を与えられるか。

解析コードは `macro2_angle_panel.py` である。既存の `main5.py` と `gp.py` はこの解析のためには変更していない。

## panelとGP入力

第1摂動 `v1` を実行してcorrector後状態 `q1` を得る。`v1`を`q1`の接空間へ射影した単位方向を `p`、第1状態でサンプルした物理参照方向を`q1`へ射影し、`p`に対してGram--Schmidtした方向を `u` とする。第2摂動は

\[
v_2(\theta)=\rho\{\cos\theta\,p+\sin\theta\,u\}
\]

である。各`q1`について16等角度

\[
[-180,-157.5,\ldots,-22.5,0,22.5,\ldots,157.5]\ \mathrm{deg}
\]

をすべて評価し、その後は決定論的rolloutを行った。

比較した入力は次の三つである。

- `raw`: `[q1, v2]`
- `angle_aware`: `[q1, p, u, v2, cos(theta), sin(theta), projection ratios, mask]`
- `feature_shuffled`: `q1`と`v2`は正しく保ち、学習側の`p, u, angle context`だけを並べ替えた負対照

kernel splitは、`raw`ではstate blockを`q1`の`n_free`次元、後二者では`[q1,p,u]`の`3*n_free`次元とした。全16角度を同じgroupとし、group全体をまとめてholdoutした。

新規cellは、初期決定論的trunkだけから作った固定archive `A1` に対して判定した。このためarchive更新を含むonline新規性ではなく、同一の基準に対するoffline新規性である。

## pilot設定

- seed: 623, 624, 625
- 各seedの独立`q1` group: 18
- 各groupの角度: 16
- 総評価macro: 864
- 摂動ノルム: 0.20
- 最大決定論的suffix: 100 step
- group CV: 3 folds

## 真の角度依存性

3 seed、54個の`q1` groupをまとめた結果は次の通りである。

| 規則 | 新規cellを生む率 |
|---|---:|
| 16角度から一様random | 0.126 |
| 固定 `theta=0` | 0.074 |
| 固定 `theta=pi` | 0.204 |
| 各groupで真値を知るoracle top-1 | 0.370 |

`cos(theta)`と真の新規cell指示値のSpearman相関は `-0.135`、flat指示値との相関は `-0.195` だった。したがって、このpanelでは直前方向との類似度が高いほど有用という関係は得られなかった。

角度別では、`theta=-180, -157.5, +157.5 deg` の新規率がいずれも `0.204` で最も高く、`theta=0` は `0.074` だった。ただし、これは異なる`q1`で同じ角度を固定した平均であり、個別状態の最適角は異なる。

## group内選択性能

onlineの第2角度選択に直接対応するgroup内指標を主要指標とする。

| モデル | top-1新規率 | top-25%内の平均新規率 | top-25%内に新規を含むgroup率 | `theta=0`平均順位 |
|---|---:|---:|---:|---:|
| raw GP | 0.130 | 0.167 | 0.259 | 8.48 / 16 |
| angle-aware GP | **0.222** | **0.204** | **0.278** | 9.63 / 16 |
| feature-shuffled | 0.148 | 0.134 | 0.185 | 7.70 / 16 |

angle-aware GPは、54 group中12 groupでtop-1候補が新規cellを生んだ。rawは7 group、shuffledは8 groupだった。

paired比較では、angle-aware対rawは「両方成功6、angle-awareだけ成功6、rawだけ成功1」、angle-aware対shuffledは「両方成功8、angle-awareだけ成功4、shuffledだけ成功0」だった。方向contextには選択上の情報がある可能性が示されたが、3 seed・54 groupだけなので確定的な統計的主張にはまだ弱い。

一方、全候補を混ぜたnew-cell AUCはraw `0.815`、angle-aware `0.754`、shuffled `0.676`だった。raw AUCが高いのは、主に`q1` group間の難易度を識別したためと考えられる。今回のonline設計は`q1`を先に固定してから角度を選ぶので、全候補AUCよりgroup内top-1/top-25%が適切な評価指標である。

## 解釈

この結果から、以前の80/20 transport heuristicの成功を

> exactな直前方向 `theta=0` が本質的に優れていた

と解釈することはできない。より妥当なのは、次の要因を分けることである。

1. 摂動直後StateNodeをもう一度選ぶこと。
2. 小さい摂動を2回連続させ、有限距離を移動すること。
3. 第2摂動の角度を状態依存に選ぶこと。
4. 異なるbranchへ探索を分散すること。

今回のpanelは、要因2があっても要因3は`theta=0`固定でない方がよい場合が多いことを示した。さらに、angle-aware GPのtop-1新規率 `0.222` はuniform random `0.126` より高く、GPを使うなら「方向維持を再発見させる」のではなく、K=2 macro内で状態依存の第2角度を選ばせる方が有望である。

推奨online比較は次である。

1. `K=2 + theta random`
2. `K=2 + theta=0 fixed`
3. `K=2 + theta=pi fixed`
4. `K=2 + raw GP`
5. `K=2 + angle-aware GP`

全条件で第1親状態と`v1`の生成規則を揃え、総micro mechanics stepを予算とする。GPはtop-1固定だけでなく、top-25%からThompson samplingまたはsoftmax選択する条件も有用である。今回、angle-awareのtop-25%平均新規率もrandomより高かったためである。

## 重大な限界

- `A1`に対する新規性を固定している。同じ新規cellを複数候補が生む場合もすべて新規として数えるため、onlineでarchiveが埋まった後の価値を過大評価する。
- 平均panel cell数は2.33であり、このpanelだけでfive-barの全cellを網羅したわけではない。
- 親は決定論的trunk上から選び、その後に第1摂動を一回与えている。既に複数摂動を含む探索木の分布とは異なる。
- 54 groupでは、angle-awareのraw/shuffledに対する改善を強い統計的保証として主張するには少ない。
- GP classifierは近似的なlogit GPRであり、確率校正を保証しない。
- 多自由度接空間では一つの物理参照方向`u`が定める2次元断面だけを探索している。全接空間方向を列挙していない。
- offlineでは16角度をすべてrolloutして教師を得ている。この教師収集費用自体は、online探索性能の比較予算には直接置き換えられない。

## 出力

- `result/macro2_angle_panel_p18_seed623_625_aggregate.json`: 修正済みkernel splitによる主要集計
- `result/macro2_angle_panel_p18_seed623_625_statedim_nf.json`: 旧kernel splitのprovenance
- 各 `result/macro2_angle_panel_p18_seedXXX/seed_XXX/`:
  - `panel.csv`
  - `panel_arrays.npz`
  - `cv_predictions.csv`
  - `summary.json`
  - `cv_predictions_statedim_nf.csv`
  - `summary_statedim_nf.json`

