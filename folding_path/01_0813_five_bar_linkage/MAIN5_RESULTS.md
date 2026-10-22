# main5：GP による摂動選択とランダム選択の比較

## 1. 評価目的

`main5.py` は、初期構造から一本の決定論的 rollout を生成した後、既存の非終端 StateNode と接空間内の摂動方向の組を候補とし、選択した候補から再び決定論的 rollout を行う。ここでは、同じ候補生成規則の下で次の三手法を比較した。

- `random`：候補の StateNode–摂動対を一様ランダムに選ぶ。
- `gp_ucb`：flat 到達モデル、最小平坦度モデルおよび flat 条件付き終端 descriptor GP から計算した novelty-UCB-like スコアで選ぶ。ただし未探索領域を失わないため、cold start と ε-random を残す。
- `gp_posterior`：flat 条件付き GP の事後分布から終端 descriptor をサンプルし、各サンプルを archive 判定へ通した新規 cell score で選ぶ。cold start と ε-random は `gp_ucb` と共通である。

主目的は、GP による終端予測が「摂動位置と摂動方向に選択圧をかけること」により、同じ力学計算 step 予算で多くの elite cell を、より早く発見できるかを調べることである。

## 2. 指標

累積 elite cell 数を $C(b)$、総 mechanics-step 予算を $B$ とする。通常の discovery AUC は

\[
\operatorname{AUC}
=
\frac{1}{B}\int_0^B C(b)\,\mathrm db
\]

であり、値は 0--1 の確率ではなく「予算中に保持していた平均 cell 数」である。

初期決定論経路は全手法で共通なので、その終了 step と cell 数をそれぞれ $b_0,C_0$ とし、GP 選択の寄与を直接見るため

\[
\operatorname{AUC}_{\mathrm{post}}
=
\frac{1}{B-b_0}
\int_{b_0}^{B}\{C(b)-C_0\}\,\mathrm db
\]

も用いた。`post-trunk new cells` は $C(B)-C_0$ である。

## 3. 最終評価条件

最終評価には、調整用 seed と分離した `heldout_scale020_random`、`heldout_scale020_gp_ucb` および `heldout_scale020_gp_posterior` の seed 100--104 を用いた。全 seed で初期決定論経路は 63 step で終了し、同じ 1 cell を発見した。

| 項目 | 値 |
|---|---:|
| mechanics-step 予算 $B$ | 1200 |
| `target_norm_max` | 0.10 |
| 摂動ノルム | 0.20 |
| rollout 上限 | 100 step |
| 候補親 StateNode 上限 | 64 |
| 各親の候補方向数 | 16 |
| cold-start random rollout 数 | 4 |
| ε-random | 0.20 |
| GP 再学習間隔 | 3 rollout |
| state/action kernel length scale | 0.50 / 0.10 |
| Matérn-3/2 signal variance | 1.0 |
| 回帰/classifier noise standard deviation | 0.08 / 0.25 |
| novelty κ / reach weight | 1.0 / 0.25 |
| collision model | 二次元平面内判定 |

候補親は全 history ではなく、摂動回数とポテンシャル進行度で層化抽出した最大 64 状態である。したがって、ここで検証しているのは厳密には「有界な共通候補集合内での GP 選択」と「同集合内でのランダム選択」の差である。教師学習には候補集合の上限によらず、完了 rollout に含まれる全 edge を用いた。

## 4. Held-out 結果

### 4.1 平均値

| 手法 | seed 数 | final cells | AUC | post-trunk new cells | post-trunk AUC | wall time [s] | rollout 数 | flat / collision rollout | 選択回数の内訳 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | 5 | 2.000 | 1.6340 | 1.000 | 0.7245 | 13.17 | 38.6 | 26.0 / 11.6 | random 37.6 |
| GP-UCB-like | 5 | 2.000 | 1.6568 | 1.000 | 0.7486 | 19.86 | 37.2 | 25.4 / 10.8 | cold 4.0、GP 27.2、ε-random 5.0 |
| GP-posterior | 5 | 1.800 | 1.5118 | 0.800 | 0.5956 | 101.84 | 34.0 | 28.4 / 4.6 | cold 4.0、GP 24.4、ε-random 4.6 |

GP の wall time は平均で random の約 1.51 倍であった。一方、固定 mechanics-step 予算内の completed rollout 数はわずかに少なく、GP が比較的長い rollout を選ぶ場合があったことを示す。

GP-posterior は random の約 7.73 倍、GP-UCB-like の約 5.13 倍の wall time を要した。そのうち平均約 86.1 s が候補の事後サンプリングと archive 判定を含む score 計算である。さらに seed 100 では初期 trunk の 1 cell 以外を発見できず、最終 cell 数も平均 1.8 に低下した。random に対する paired AUC 差は平均 -0.1222、5組だけの参考 95% Student-$t$ 区間は [-0.5527, 0.3083] であった。したがって、今回のサンプル数32の素朴な事後 score は、非滑らかな cell 判定を直接扱えるという利点を探索性能へ変換できていない。

### 4.2 Seed ごとの paired 比較

差はすべて `GP - random` である。

| seed | final cells (random / GP) | AUC (random / GP) | AUC 差 | 第2 cell 発見 step (random / GP) | post-trunk AUC 差 |
|---:|---:|---:|---:|---:|---:|
| 100 | 2 / 2 | 1.6808 / 1.7317 | +0.0508 | 320 / 259 | +0.0536 |
| 101 | 2 / 2 | 1.6375 / 1.5367 | -0.1008 | 372 / 493 | -0.1064 |
| 102 | 2 / 2 | 1.4133 / 1.4708 | +0.0575 | 641 / 572 | +0.0607 |
| 103 | 2 / 2 | 1.6358 / 1.7425 | +0.1067 | 374 / 246 | +0.1126 |
| 104 | 2 / 2 | 1.8025 / 1.8025 | 0.0000 | 174 / 174 | 0.0000 |

平均 paired AUC 差は +0.0228 であったが、5 組だけから計算した参考値としての 95% Student-$t$ 区間は [-0.0750, 0.1207] であり、符号を確定できない。post-trunk AUC 差も +0.0241、95% 区間 [-0.0791, 0.1273] であった。GP は 3 seed で第2 cellを早く、1 seed で遅く発見し、1 seed は同時である。

最終 cell 数の paired 差は全 5 seed で 0 であった。これは今回の標本で差を観測しなかったことを意味し、母集団で常に差がないことの保証ではない。wall time の paired 差は平均 +6.69 s、参考 95% 区間 [5.55, 7.83] s であり、現在の実装では GP の追加計算費用は明確である。

## 5. 摂動ノルム感度

摂動ノルム 0.25 および 0.30 は調整用 seed 0--2 による補助実験であり、held-out 評価には含めない。その他の主要条件は候補親上限 64、方向数 16、cold start 4、ε=0.2、ℓq=0.5、ℓa=0.1、予算 1200 step で共通である。

| 摂動ノルム | seed 数 | final cells random / GP | AUC random / GP | paired AUC 差 | post-trunk AUC random / GP | wall time random / GP [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 held-out | 5 | 2.00 / 2.00 | 1.6340 / 1.6568 | +0.0228 | 0.7245 / 0.7486 | 13.17 / 19.86 |
| 0.25 tuning | 3 | 2.00 / 2.00 | 1.7181 / 1.6928 | -0.0253 | 0.8133 / 0.7866 | 13.67 / 20.73 |
| 0.30 tuning | 3 | 2.00 / 2.00 | 1.7183 / 1.6683 | -0.0500 | 0.8135 / 0.7608 | 13.32 / 20.52 |

0.25 と 0.30 では GP の AUC がむしろ小さく、摂動ノルムを大きくしても GP 選択の優位性は現れなかった。ただし seed 数は 3 で、同じ seed を調整にも使用しているため、この表はロバスト性の予備確認に限る。

GP は 0.25 と 0.30 で collision rollout を大きく減らし、flat rollout の割合を高めた。たとえば 0.30 では random の flat/collision 平均が 26.3/20.0、GP が 27.0/8.7 である。それでも新しい elite cell は増えなかった。この結果は、flat 到達性に対する選択圧はある程度働く一方、既知の flat basin を繰り返し選び、終端 novelty に対する選択圧が十分でない可能性を示す。

## 6. Pilot 結果の位置づけ

| 実験 | 条件差 | seed 数 | AUC random / GP | final cells random / GP | wall time random / GP [s] |
|---|---|---:|---:|---:|---:|
| `pilot1200_*` | 親上限32、方向12、予算1200 | 3 | 1.8361 / 1.8361 | 2 / 2 | 12.78 / 17.51 |
| `pilot1200_all_*` | 全親、方向12、予算1200 | 1 | 1.8417 / 1.8417 | 2 / 2 | 19.36 / 64.46 |
| `pilot2000_*` | 親上限64、方向16、予算2000 | 1 | 1.8190 / 1.8230 | 2 / 2 | 22.35 / 47.56 |

`pilot1200_*` では、第2 cell がすべて共通の cold-start random 区間で発見されたため、GP を開始した後には両手法とも新規発見がない。したがって AUC の完全一致は GP と random の一般的同等性を示さない。全親を毎回評価した `pilot1200_all_*` では発見数は変わらず、GP の wall time だけが約 3.3 倍になった。`pilot2000_*` も 1 seed のみで、予算を増やしても 2 cell に留まった。

これらは候補親数、方向数、予算および seed が held-out 条件と異なるため、held-out の標本へ混ぜて平均していない。また、反復的なコード開発中に生成された `pilot_default_*` など、cold start がなく action length scale が 0.5 であった古い出力は本資料から除外した。今回掲載したデータは保存された training input がいずれも 15 次元で、cold start 4 と action length scale 0.1 を使用していることを確認したが、当時の `config.json` にコード commit/hash が保存されていない。現行コードは今後の出力へ `algorithm_version` と GP の input/output dimension を保存するよう修正した。git 管理を導入する場合は commit hash も保存し、コード版の混在を機械的に排除する必要がある。

## 7. 結論

今回の five-bar、1200-step、摂動ノルム 0.20 の held-out 5 seed では、GP-UCB-like は第2 cellを平均的にはわずかに早く発見した。しかし差の向きは seed により変わり、paired AUC 差の区間は 0 を含む。さらに、全 seed・全摂動ノルムで最終発見数は 2 cell に留まり、GP-UCB-like は random より多くの終端 cell を発見していない。GP-posterior は平均 final cells と AUC の双方で random を下回り、計算時間も大幅に増えた。

したがって現段階で支持できるのは、「終端予測を用いた選択が一部の seed で既知の flat 終端への到達を早める場合がある」という限定的な観察までである。「GP による選択圧が多様な平坦終端の発見効率を改善する」という主張を支持する証拠は得られていない。GP は wall time を増加させているため、実計算時間を予算とする場合には現状の random より不利である。

## 8. 主な失敗理由の候補

1. **教師 edge 数と終端多様性の乖離**：約1200 edgeを学習しても、観測された flat cell は実質2種類である。同一 rollout の多数の deterministic edgeが同じ終端を持つため、教師行数ほど終端情報は増えていない。
2. **flat 到達項による既知 basin の活用**：flat probability と reach probability が高い候補を優先する結果、collision は減るが既知 cell への再到達が増え、新規性へ計算予算が配分されていない可能性がある。
3. **終端写像の不連続・多峰性**：basin境界で終端 descriptor は不連続に変わる。滑らかな Matérn GP の予測平均と対角分散は、実在しない中間終端を生成し、その既存 archive からの距離を過大評価し得る。GP-posterior でも各出力座標を独立にサンプルする現実装では、部材長拘束を満たさない非物理 descriptor を生成して新規性を過大評価し得るため、この問題は自動的には解消されない。
4. **重なり順の離散性**：archive 距離には離散的な layer key が含まれるため、現在の trace 型不確実性を加えた値は厳密な UCB ではない。
5. **候補集合の制限**：最大64親・各16方向という前段の shortlist が、有望な history 状態または狭い摂動方向を候補に含めない可能性がある。
6. **rollout cost の未考慮**：GP は同じ step 予算でやや少ない rollout しか実行していない。新規cell価値だけでなく、予測 rollout step 数で割った cost-aware acquisition が必要かもしれない。
7. **archive/物理判定側の上限**：理論上想定する six terminal states が、現在の collision 判定および terminal/layer descriptorで本当に6 cellへ分類されるかを、探索手法とは独立に再確認する必要がある。

## 9. 次に行う ablation

優先順位は次の通りである。

1. **獲得関数の分解**：`random`、flat/reach のみ、novelty のみ、現在の full score を同一候補集合で比較する。flat率だけが上がる原因を切り分ける。
2. **候補スコアの事後検証**：少数の固定 candidate pool を実際に全 rollout し、獲得値と「flatか」「新規cellか」「既存cellからの実距離」の順位相関・precision@kを測る。探索結果だけでなく surrogate と獲得関数のどちらが失敗しているかを識別する。
3. **教師構成**：全edge、摂動edgeのみ、rolloutごとに等重みを与えた全edgeを比較する。評価分割はStateNode行ではなくrollout単位とする。
4. **事後 score の改善**：今回の32サンプル全候補評価は遅く、性能も改善しなかった。少数サンプルで粗選別して上位候補だけを再評価する二段階方式、PCA/構造拘束による相関出力サンプル、または terminal-cell mixture-of-experts を試す。
5. **候補集合**：親上限64/全history、方向16/32を比較する。ただし all-history GP の大きな wall-time を必ず報告する。
6. **cost-aware化**：獲得値を予測 rollout step 数で正規化し、固定 mechanics-step 予算内の試行回数低下を防ぐ。
7. **物理・archive sanity check**：既知の6終端候補を直接 archive 判定へ通し、collision、shape descriptor、layer key のどこで除外・統合されるか確認する。
8. **最終再評価**：上記で方式を固定した後、未使用 seed を少なくとも10、可能なら20用い、摂動ノルム0.20/0.25/0.30を同じ held-out seed で比較する。
