# GP サロゲートの学習曲線評価

## 1. 評価目的

本評価は、GP を用いた探索そのものの発見 cell 数ではなく、`main5.py` で用いるサロゲートが、ランダム探索から得た教師データの増加に伴って何を学習できるかを調べたものである。同一の random rollout データを三つの学習方法で共有し、最終 32 rollout を固定 holdout とした。したがって、手法ごとの探索履歴の違いを含まない、offline の予測性能比較である。

評価対象は次の三つである。

- rollout が平坦化するか衝突するかの順位付けと確率予測
- その後に到達する平坦度と、flat 条件付き終端 descriptor の回帰
- 初期決定論経路の終端だけを基準としたとき、その連続 descriptor から離れた観測済み outcome を上位に順位付けできるか

最後の項目は、後述する `root novelty` という限定的な代理評価である。探索時の候補集合をすべて rollout した new-cell ranking 評価ではない。

元データと集計結果は次に保存されている。

- `result/learning_curve_random_b4000/seed_200`--`seed_204`
- `result/learning_curve_random_b4000/gp_learning_curve_v1/seed_metrics.csv`
- `result/learning_curve_random_b4000/gp_learning_curve_v1/seed_mean_metrics.csv`
- [生成済み学習曲線](result/learning_curve_random_b4000/gp_learning_curve_v1/learning_curve.png)

## 2. データ収集条件

| 項目 | 設定 |
|---|---:|
| algorithm version | `main5_gp_v1` |
| データ収集方策 | pure random selection |
| seed | 200--204（5 seed） |
| mechanics-step 予算 | 4000 / seed |
| 初期決定論 trunk | 63 step、1 flat cell |
| `target_norm_max` | 0.10 |
| 摂動ノルム | 0.20 |
| rollout 上限 | 100 step |
| 候補親上限 / 方向数 | 64 / 16 |
| collision model | 二次元平面内判定 |
| GP 入力 / 出力次元 | 15 / 6 |
| state/action length scale | 0.50 / 0.10 |
| regression/classifier noise std. | 0.08 / 0.25 |

各 seed は予算 4000 step を使い切った。収集結果は次の通りである。`rollouts` は初期 trunk と最後の予算打切り rollout も含む。

| seed | rollouts | flat | collision | 打切り | 教師 edge | elite cells | 第2 cell 発見 step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 124 | 94 | 29 | 1 | 3955 | 2 | 215 |
| 201 | 126 | 85 | 40 | 1 | 3981 | 2 | 163 |
| 202 | 139 | 80 | 58 | 1 | 3999 | 2 | 379 |
| 203 | 138 | 88 | 49 | 1 | 3970 | 2 | 356 |
| 204 | 122 | 94 | 27 | 1 | 3981 | 2 | 235 |

すべての seed で得られた archive は 2 cell のみである。この点は、終端予測と novelty 評価の解釈を大きく制限する。

## 3. 固定 rollout holdout

各 seed について、実行済み摂動 edge を一つ持つ rollout を時系列順に並べ、最後の 32 rollout をすべての学習点に共通する test set とした。残りの先行 rollout の先頭から

\[
n\in\{4,8,16,24,32,48,64,80\}
\]

本を選び、nested な training set を作った。`n` は edge 数ではなく、摂動を伴う完了 rollout 数である。

test では各 holdout rollout の摂動 edge 一つだけを評価する。同じ rollout 内の deterministic edge は、同じ flat/collision label と同じ終端 descriptor を共有するため、rollout の一部が train、残りが test に分かれると重大な漏洩が起きる。本評価では rollout ID 単位で完全に分離され、holdout rollout の edge は一つも学習に使われていない。

各 seed の test set は 32 rollout で、平均 20.4 flat、11.6 collision であった。この test set は全 `n`、全 training mode で固定されているため、曲線上の変化は主として教師データ量の効果を表す。ただし test は同一 random tree の後半であり、別構造や独立した探索木への外挿性能を評価したものではない。

## 4. 三つの training mode

### 4.1 `all_edges`

初期 trunk と選択した `n` rollout に含まれる全 transition edge を使う。現在の `main5.py` と同じ教師化である。長い rollout ほど多くの行を持つため、その終端 label と descriptor が強く反復される。

### 4.2 `perturbation_only`

各 rollout の摂動 edge 一つだけを使う。初期 trunk には摂動 edge がないため、この mode では trunk は学習行に入らない。学習入力が実際に score する候補 action に最も近く、edge 数と GP fitting cost は小さい。一方、途中状態の deterministic transition が持つ情報は捨てる。

### 4.3 `rollout_balanced`

初期 trunk と各 rollout から最大 8 edge を等間隔に残し、摂動 edge は必ず保持する。各 rollout の寄与を近づけるための subsampling であり、観測重みを持たない現在の exact GP に対する近似的な equal-rollout weighting である。

`all_edges` と `rollout_balanced` の `train_rollout_count` は trunk を含むため `n+1`、`perturbation_only` は `n` である。

## 5. 指標

- `flat AUC`：flat と collision の順位付け。0.5 がランダム順位、1 が完全順位である。
- `flat Brier`：flat 確率と二値結果の二乗誤差。小さいほどよい。ただし現在の classifier は logit を GP 回帰する近似であり、厳密に校正された Bernoulli GP ではない。
- `flatness RMSE/correlation`：摂動後 suffix の最小 potential から作った対数 flatness target の回帰誤差と相関。
- `descriptor median W-error`：true-flat test rollout に対する6次元終端 descriptor の誤差を、shape は archive RMS tolerance、layer 部分は固定 layer scale で正規化した weighted RMS の中央値。小さいほどよく、1 以下は archive 分解能に近い精度の目安である。
- `descriptor within-one rate`：上記誤差が1以下となった true-flat test の割合。
- `root novelty AUC`：既知 archive を初期 trunk の連続 descriptor 一つに固定し、そこから W-distance が1を超える flat outcome を正例とした順位 AUC。
- `root novelty P@5`：root novelty score 上位5候補の正例率。holdout の正例率は平均 0.3125 なので、ランダム順位の期待値も約 0.3125 である。

`root novelty` は連続 descriptor 距離による代理 label であり、実際の layer key を含む archive cell 判定ではない。また、holdout は探索中に同時提示された候補集合ではなく、random policy が異なる時点で選んだ action の集合である。
正例は「flat かつ root から W-distance が1を超える outcome」、負例は root 近傍の flat と collision の両方である。そのため、この AUC には終端 novelty だけでなく collision を除く能力も含まれる。

## 6. 主要学習曲線

次表の各欄は `flat AUC / descriptor median W-error / root novelty AUC` の seed 平均である。

| training rollout数 `n` | all edges | perturbation only | rollout balanced |
|---:|---:|---:|---:|
| 4 | 0.588 / 17.66 / 0.580 | 0.557 / 18.15 / 0.593 | 0.579 / 17.41 / 0.603 |
| 8 | 0.642 / 16.77 / 0.712 | 0.543 / 19.15 / 0.574 | 0.657 / 16.59 / 0.719 |
| 16 | 0.697 / 7.87 / 0.779 | 0.608 / 12.86 / 0.642 | 0.680 / 8.20 / 0.779 |
| 24 | 0.804 / 5.77 / 0.821 | 0.681 / 6.78 / 0.709 | 0.792 / 4.98 / 0.826 |
| 32 | 0.848 / 4.95 / 0.865 | 0.751 / 4.77 / 0.782 | 0.832 / 4.80 / 0.868 |
| 48 | 0.855 / 3.92 / 0.888 | 0.814 / 3.41 / 0.830 | 0.862 / 4.27 / 0.891 |
| 64 | 0.850 / 4.12 / 0.896 | 0.793 / 3.82 / 0.849 | 0.846 / 4.15 / 0.892 |
| 80 | 0.877 / 3.60 / 0.921 | 0.823 / 3.36 / 0.885 | 0.869 / 3.63 / 0.905 |

すべての mode で完全な単調性はないものの、教師 rollout の増加に伴い、flat ranking、flatness 回帰、root novelty ranking は全体として改善した。とくに `all_edges` と `rollout_balanced` は `n=24` 付近から flat AUC が約0.8、root novelty AUC が約0.82となる。

一方、descriptor median W-error は `n=80` でも3.36--3.63であり、archive 分解能の目安である1を大きく上回る。連続終端を粗く区別する情報は学習されるが、終端を cell tolerance 内で再構成できる水準には到達していない。

## 7. `n=4`, `n=32`, `n=80` の比較

値は5 seed の平均である。括弧内に必要な箇所だけ標本標準偏差を示す。

| `n` | mode | 学習 edge | flat AUC | Brier | flatness RMSE / corr. | descriptor error | within 1 | root novelty AUC | P@5 | fit [s] |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | all | 172.0 | 0.588 | 0.236 | 1.270 / 0.083 | 17.66 | 0.385 | 0.580 | 0.36 | 0.116 |
| 4 | perturb | 4.0 | 0.557 | 0.256 | 1.316 / 0.034 | 18.15 | 0.357 | 0.593 | 0.40 | 0.0004 |
| 4 | balanced | 35.8 | 0.579 | 0.234 | 1.270 / 0.090 | 17.41 | 0.394 | 0.603 | 0.40 | 0.0004 |
| 32 | all | 1096.4 | 0.848 (0.076) | 0.160 | 0.943 / 0.574 | 4.95 (2.62) | 0.229 | 0.865 (0.060) | 0.80 | 0.675 |
| 32 | perturb | 32.0 | 0.751 (0.094) | 0.194 | 1.012 / 0.465 | 4.77 (2.95) | 0.278 | 0.782 (0.054) | 0.60 | 0.0015 |
| 32 | balanced | 241.4 | 0.832 (0.065) | 0.163 | 0.954 / 0.539 | 4.80 (3.41) | 0.249 | 0.868 (0.057) | 0.80 | 0.112 |
| 80 | all | 2561.0 | 0.877 | 0.146 | 0.903 / 0.597 | 3.60 | 0.285 | 0.921 | 0.96 | 1.974 |
| 80 | perturb | 80.0 | 0.823 | 0.171 | 0.953 / 0.555 | 3.36 | 0.374 | 0.885 | 0.92 | 0.0061 |
| 80 | balanced | 596.6 | 0.869 | 0.150 | 0.906 / 0.589 | 3.63 | 0.300 | 0.905 | 0.88 | 0.184 |

`perturbation_only, n=4` では1 seedのtraining prefixにflat descriptorがなく、descriptor GPを構成できなかった。この行のdescriptor errorとwithin-one rateだけは4 seed平均であり、その他は5 seed平均である。

`n=4` では、flat AUC と root novelty AUC はおおむね0.56--0.60であり、seed 間のばらつきも大きい。現在の cold start 4 rollout の直後には、GP の選択を信頼できるだけの教師情報が十分でない。

`n=32` では、`all_edges` と `rollout_balanced` の flat AUC はそれぞれ0.848と0.832、root novelty AUC は0.865と0.868、P@5 はともに0.80まで上昇した。したがって、少なくとも今回観測された outcome の範囲では、flat に到達し、かつ root の連続 descriptor から離れた outcome を生む action を順位付けする信号が学習されている。ただし、この代理 label は実際の archive cell と一致するとは限らない。

`n=80` では順位性能はさらに改善するが、descriptor の archive-scale 誤差は依然として3以上である。高い root novelty AUC と低い descriptor 精度が両立していることから、終端座標を cell tolerance 内で正確に回帰できなくても、「root descriptor から離れた観測済み outcomeへ向かう」粗い順位付けは可能であることが分かる。

## 8. edge の反復と training mode

`n=80` で `all_edges` は平均2561 edgeを使用するのに対し、`rollout_balanced` は597 edge、`perturbation_only` は80 edgeである。`rollout_balanced` は `all_edges` の約23%の行数で、flat AUC 0.869対0.877、root novelty AUC 0.905対0.921を得た。descriptor errorも3.63対3.60でほぼ同程度である。

これは、一つの rollout に含まれる多数の deterministic edge が無価値という意味ではない。しかし、同じ終端 label と descriptor を共有するedgeをすべて同じ重みで入れる必要性は低く、現在の exact GP では rollout-balanced subsampling の方が計算効率に優れる可能性が高い。今回の実測 fitting time は `n=80` で1.974 s対0.184 sであった。ただし、この wall time は計算機負荷に依存する参考値である。

`perturbation_only` は flat rankingでは他二方式に遅れるが、`n=80` の descriptor errorは3.36、within-one rateは0.374で三方式中最良であった。候補評価時と同じ有限摂動 action に学習分布を合わせる利点が示唆される。したがって、flat/collision と flatness には balanced edge、終端 descriptor には perturbation edgeを用いるように、三つのサロゲートで教師行を分ける設計が次の候補になる。

## 9. 1200-step 探索との対応

過去の1200-step held-out 実験では、GP は trunk と4本の cold-start rollout、すなわち本評価の `n=4` に相当する段階から guided selectionを開始していた。その時点の学習 edge は seed により139--263であり、本評価の `all_edges, n=4` の平均172 edgeと整合する。

今回の曲線によれば、この段階の flat AUC は0.588、root novelty AUC は0.580であり、ほぼ弱い順位信号しかない。したがって、初期のGP選択がrandomより安定して優れなかったことは、サロゲートがまだ未学習であったことと整合する。

`all_edges, n=32` は平均1096 edgeであり、1200 mechanics-step予算の終盤に相当する。ここでは順位性能は十分改善しているが、学習が有用な領域に入る頃には残り探索予算が少ない。さらに、descriptor errorは約5であり、archive cellを高精度に予測するには不十分である。

`n=80` の `all_edges` は平均2561 edgeを必要とし、1200-step予算では到達できない。よって、今回の学習曲線は「1200 edgeあれば十分」というより、「約1000 edgeでflat/root-away代理指標の順位信号は得られるが、cell分解能の終端回帰にはまだ足りない」ことを示す。

## 10. GP が探索に有効となるための具体条件

1. **guided selection開始を固定4 rolloutにしない**：今回の範囲では `n=16`--`24` から順位性能が明確に改善する。random比率を段階的に下げる、または rollout-level validation が所定値を超えてからGPの重みを上げる方が妥当である。
2. **rollout-balanced教師を用いる**：同じ情報を持つ長いsuffixの過重複を抑え、exact GPの cubic costを低減する。これは mechanics data収集費を減らすものではないが、再学習費を大きく抑える。
3. **モデルごとに教師を分ける**：flat/collision と flatness はbalanced edge、候補actionから終端へのdescriptor回帰はperturbation edge中心とするhybridが有望である。
4. **教師archiveに複数cellを含める**：未観測の第三以降の basinは、二 cellだけの回帰データから検証できない。root、frontier、複数摂動尺度などを混ぜたrandom corpusで、教師終端のcell coverageを先に増やす必要がある。
5. **探索予算を増やすか、事前学習を利用する**：1200-step内では、モデルが改善する頃に残り予算が少ない。別seedの共通構造データで事前学習できるならcold startを短縮できるが、探索木を跨ぐ汎化は別途holdout検証が必要である。
6. **random quotaを残す**：今回の高い novelty rankingは、観測済みoutcomeをroot-away代理labelで分けた結果である。モデルが未知 basinを既知cellと誤予測した場合に永久に除外しないため、epsilon-randomまたは不確実性探索は必要である。
7. **獲得関数を候補panelで直接検証する**：flat AUCやroot novelty proxyだけでなく、同一時点の候補集合を全rolloutし、top-kの新規cell数、random lift、cost当たり発見数を測る必要がある。

## 11. 本評価から学べること

- 15次元の状態・変位入力から、flat/collision と連続flatnessには再現可能な予測信号がある。
- 教師rolloutが増えると、既知rootの連続 descriptorから離れた観測済みoutcomeへ向かう候補の順位付けも改善する。
- 全edgeを保存した約1200行を、そのまま約1200個の独立終端観測とみなしてはならない。
- rollout-balanced subsamplingは、今回の範囲で予測性能をほぼ保ちながら学習行数を大きく削減した。
- 終端座標の粗い順位情報は得られても、archive tolerance内の精密なdescriptor回帰はまだできていない。

## 12. 本評価からは学べないことと限界

1. **未知cellの発見能力**：random corpusは全seedで2 cellしか含まない。`root novelty` はrootからの連続W-distanceで作った代理評価であり、実cell labelとの一致さえ保証されない。未観測の第三以降のcellを予測できる証拠ではない。
2. **実候補集合でのnew-cell ranking**：各decision時に生成された全候補の反実仮想rolloutは保存されていない。したがって、同じ候補pool内のprecision@k、distinct new cells@k、random/oracle regretは未評価である。
3. **動的archiveへの対応**：root noveltyではarchiveを初期終端一つに固定した。探索に伴ってarchiveが変化する非定常なnew-cell labelは扱っていない。
4. **確率校正**：Brier scoreは計算したが、reliability diagram、ECE、log loss、posterior new-cell確率の校正は未評価である。
5. **構造間・探索木間の汎化**：test rolloutは同じseedのrandom tree後半である。別の折り構造、別の入力次元、独立treeへの外挿性能は分からない。
6. **統計的確証**：seed数は5であり、表は平均と標準偏差による予備評価である。小さなmode差や非単調性を一般的な優劣として解釈できない。
7. **hyperparameter最適性**：kernel length scaleとnoiseはmain5の固定値であり、rollout-group validationによる調整は行っていない。
8. **結果のversioning**：`evaluation_config.json` にはnovelty定義、コードhash、GP hyperparameterが完結して保存されていない。同一出力先への上書きを避け、今後はschema versionとcommit/hashを結果に含める必要がある。

## 13. 結論

GP はこの問題で何も学習できないわけではない。およそ24--32本の摂動rolloutがあれば、flat到達性と、既知rootの連続 descriptorから離れた観測済みoutcomeへの粗い順位付けには明確な信号が現れる。一方、現在の4-rollout cold startは短く、1200-step予算ではモデルが有用になる時期が遅い。さらに、終端descriptorは80 rolloutでもarchive分解能に達していない。

したがって、現時点で支持される方向は、`rollout-balanced学習 + 段階的なrandom-to-GP切替 + random quota` である。ただし「GPが未知の新規cell発見を効率化する」という中心的主張には、複数cellを含む教師データと、同一候補panelを全rolloutする直接的なnew-cell ranking実験がまだ必要である。
