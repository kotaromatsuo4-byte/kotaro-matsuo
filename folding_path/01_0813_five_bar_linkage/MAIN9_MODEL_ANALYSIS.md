# main9 終端 cell 分類モデルのデータ量・局所連続性監査

## 結論

`main9_random_corpus_b20000` では、終端 cell 分類器が **まったく学習不能**なのではない。しかし、random collision-continuation が canonical 6 cell を見つける時点が早すぎるため、分類器を安全な duplicate filter として使えるほど独立な教師データが集まっていない。

- canonical T6 は mechanics step `548 / 770 / 1121`（中央値 `770`）で達成された。
- その時点の独立 rollout macro は `11 / 14 / 17`（中央値 `14`）しかない。
- 保存 row は `89 / 159 / 225`（中央値 `159`）あるが、大部分は同じ rollout から作った相関の強い hindsight row であり、独立教師数ではない。
- 全モデルが prevalence baseline を継続的に上回る最初の点は `32` macro、mechanics step 中央値 `1669` で、全 seed の T6 より遅い。
- 厳格な安全 filter は RF/ExtraTrees では 320 macro まで一度も発動せず、kNN でも最初の発動は 96 macro（step 中央値 `4918`）だった。targeted LOCOではcontextありkNNが320 macroで初めて事前の安全gateを満たしたが、T6には遅すぎる。

したがって、**T6 短縮に効かない第一原因は temporal sample shortage（学習可能になる前に random が解く）**である。一方、長時間側でも精度が 0.4--0.46 程度に留まる第二原因は、現在の入力表現における basin の局所混在・終端写像の不連続性である。class imbalance と時間方向の covariate shift は二次的だった。

![T6時の教師量を重ねた学習曲線](result/main9_model_analysis/learning_curves_t6.png)

![duplicate rejectionと未知cell保持のtrade-off](result/main9_model_analysis/filter_tradeoff.png)

trade-off図の星印は、後続のtargeted LOCOで得たcontextありstrict group-kNNの320 macro点である。これはrejection 5.7%とLOCO retention 95.8%を同時に満たすが、3 seed中2 seedでの成立であり、T6時の11--17 macroよりはるかに遅い。

## データの単位

| seed | canonical T6 step | T6時の独立macro | T6時の保存row | T6時raw class | 最終macro | 最終row | hindsight row | row/macro |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 548 | 11 | 89 | 7 | 411 | 4321 | 3910 | 10.60 |
| 801 | 770 | 14 | 159 | 7 | 415 | 4583 | 4168 | 11.00 |
| 802 | 1121 | 17 | 225 | 8 | 415 | 4330 | 3915 | 10.48 |

1 macro は平均約 47.8 mechanics step を消費する。したがって、budget 20000 でも独立な教師は各 seed 約 415 個である。row 数を教師数と数えると、約 10 倍の擬似反復により学習量を過大評価する。

なお、分類器は corpus に現れた raw cell ID を学習している。raw cell は 8 個あり、T6 は reachability audit で定義された canonical 6 target の到達時刻である。seed ごとの T6 時点の macro class counts は次のとおりで、少数 class は 1 例しかない。

- seed 800: `[3, 1, 2, 1, 2, 1, 1]`
- seed 801: `[1, 4, 3, 2, 2, 1, 1]`
- seed 802: `[5, 3, 2, 1, 3, 1, 1, 1]`

第6 target を生む action をまだ実行していない選択時点では、それぞれさらに 1 macro 少ない（`10 / 13 / 16`）。つまり、「第6 cell を選ぶための分類器」が利用できる教師量は表の値よりも厳しい。

## class imbalance

最終 macro における多数 class の prevalence は `0.173 / 0.166 / 0.169`、最大 class / 最小 class の個数比は `1.78 / 2.16 / 1.89` だった。最終時点の imbalance は軽度であり、長時間側の低精度の主因ではない。

ただし、T6 以前は各 class が 1--5 macro しかなく、問題は imbalance より **class starvation** である。bootstrap や class weight は同じ macro の情報を再利用するだけなので、この不足を解消しない。

## 入力空間の局所性

各 seed 内で feature を標準化し、同じ macro を除外して nearest macro を調べた。`primary` は macro の開始入力だけ、`hindsight` は各 macro 内の residual-state row のうち他 macro に最も近いものを support として用いた結果である。

| support | query macro | 最近傍距離中央値 | 最近傍が同一cell | 10近傍の正規化entropy |
|---|---:|---:|---:|---:|
| primary only | 1241 | 0.350 | 0.380 | 0.675 |
| hindsight support | 1241 | 0.256 | 0.461 | 0.599 |

hindsight により最近傍距離は 27.0% 短縮し、same-cell 率は 8.1 percentage point 上がる。したがって hindsight は有効であり、単なる重複ではなく入力空間の support を増やしている。

しかし、最も近い距離 quintile でも結果は次の程度である。

| support | closest 20% same-cell | closest 20% entropy |
|---|---:|---:|
| primary only | 0.548 | 0.669 |
| hindsight support | 0.593 | 0.623 |

最も近い候補同士でも約 41% は別 cell へ行く。これは「入力が近ければ終端 cell もほぼ同じ」という強い filter 前提が成立していないことを示す。距離 quintile 全体の値は `result/main9_model_analysis/distance_bins.csv` に保存した。

## K の情報量

K=1--24 の各値で raw 8 cell のすべてが観測された。各 K の sample 数は 37--66、cell 分布の正規化 entropy は概ね 0.885--0.989 である。

\[
I(C;K)=0.0734\ \mathrm{nat},\qquad
H(C)=2.0674\ \mathrm{nat},\qquad
I(C;K)/H(C)=3.55\%.
\]

K 単独では終端 cell をほとんど絞れない。学習対象は `q`, `v`, `K` の相互作用であり、単純な「長い K はこの cell」という規則ではない。完全な K 別分布は `result/main9_model_analysis/k_cell_distribution.csv` にある。

## offline 分類性能

固定された末尾 64 macro/seed を holdout とし、先頭から得た macro 数を増やした。320 macro 時点の pooled result は次のとおりである。

| model | accuracy | macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| Random Forest | 0.401 | 0.376 | 1.771 | 0.772 |
| ExtraTrees | 0.406 | 0.379 | 1.803 | 0.785 |
| kNN | **0.464** | **0.428** | 3.921 | **0.723** |

kNN は hard classification では最良だが、未観測 class にゼロに近い確率を割り当てるため NLL が極端に悪い。duplicate reject を確率で判断する用途では、この非校正な高信頼をそのまま使えない。forest は確率がやや穏当だが、hard accuracy も局所 cluster 性も十分ではない。

16 macro 時点では prevalence accuracy が 0.222 に対し RF 0.216、ExtraTrees 0.198、kNN 0.222 で、実質的な改善はない。3モデルがいずれも prevalence をその後継続して上回る最初の点は 32 macro だった。

| train macro | mechanics step中央値 |
|---:|---:|
| 4 | 240 |
| 8 | 458 |
| 16 | 873 |
| 24 | 1311 |
| 32 | 1669 |
| 64 | 3258 |
| 96 | 4918 |
| 192 | 9303 |
| 320 | 15509 |

T6 教師量の範囲 11--17 macro は、まだ prevalence と同程度の領域に一致する。

## safe duplicate filter

厳格 threshold（mean probability 0.9、lower bound 0.7、committee agreement 0.8、disagreement 0.08 以下、support distance 1.0 以下、class ごとに最低4 macro）では次の結果になった。

- Random Forest: 320 macro まで reject なし。
- ExtraTrees: 320 macro まで reject なし。
- kNN: 最初の reject は 96 macro。pooled rejection 1.04%、precision 1.0。320 macro でも rejection 5.73%。

緩い threshold（0.55 / 0.25 / 0.45 / 0.25 / 2.0、class ごと2 macro）では kNN の reject は増えるが、未知 class の保持を失う。

| kNN train macro | duplicate reject | LOCO unknown retention |
|---:|---:|---:|
| 16 | 0.012 | 0.976 |
| 32 | 0.036 | 0.984 |
| 64 | 0.073 | 0.906 |
| 128 | 0.104 | 0.875 |
| 320 | 0.234 | 0.771 |

広いloose-threshold sweepでは「precision >= 0.9、unknown retention >= 0.95、duplicate rejection >= 0.05」を安定して満たす点を確認できなかった。一方、後続の特徴量アブレーションでstrict group-kNNだけをtargeted LOCO評価すると、contextあり入力は320 macroでprecision 1.0、duplicate rejection 5.7%、unknown retention 95.8%となり、3 seed中2 seedでgateを満たした。これは長時間後の弱いfilterが学習可能である兆候だが、T6時の11--17 macroとは桁が違う。contextなし`(q,v,K)`は同じ320 macroでunknown retention 91.7%に下がった。詳細は`MAIN9_FEATURE_ABLATION.md`を参照すること。

## covariate shift

学習 macro の順番を時間順からランダム順に変え、同じ末尾 holdout で評価した。全 train size にわたる accuracy 差（random order minus chronological）は次のとおりだった。

| model | 平均差 | 平均絶対差 |
|---|---:|---:|
| Random Forest | -0.0146 | 0.0295 |
| ExtraTrees | +0.0189 | 0.0337 |
| kNN | -0.0035 | 0.0166 |

ランダム混合しても性能が大きく改善しないため、単純な early/late covariate shift が主因ではない。時間順 split は実運用を模した厳しい評価として妥当だが、低精度の大部分はデータ不足と局所 basin mixing で説明される。

## 解釈と次の方針

原因は時期によって分けて考えるべきである。

1. **T6まで:** 独立 macro が中央値14しかなく、各 class 1--4例程度である。これはモデル選択や校正だけでは解消できない。
2. **長時間側:** 1241 macro を pooled しても nearest same-cell は primary 0.38、hindsight 0.46、分類精度は 0.40--0.46 である。現在の `(q, context, v, K)` と距離尺度では basin 境界が局所的に強く混在している。
3. **長時間filter:** contextありgroup-kNNは320 macroで安全gateを満たす兆候があるが、最初の6 cellを速める時期には間に合わない。
4. **副次要因:** 最終 class imbalance は軽く、train-order experiment でも covariate shift は小さい。

よって、この corpus に対する結論は「forest が悪い」でも「データを増やせば直ちに安全 filter になる」でもない。**早い random T6 と遅い learnability の時間的不一致**があり、その後には **表現上の不可約に見える局所不連続性**が残る。

今後分類器を使うなら、T6 を直接短縮する主役ではなく、次の限定用途が妥当である。

- filter は hard reject ではなく、候補 batch の一部を弱く down-rank する。
- 一定割合の完全 random/OOD 候補を必ず残す。
- raw row ではなく macro-group を有効標本数・calibration split の単位にする。
- hindsight row は support 距離改善に使うが、同一 macro の複製を独立教師として数えない。
- filter を有効化する learnability gate は「macro 数」だけでなく、group-heldout calibration、class ごとの独立 macro 数、LOCO unknown retention で判定する。
- 本当に選択圧を得たいなら、終端 cell だけでなく transport 軌跡の局所 descriptor や衝突履歴を入力へ追加し、同一 cell 内の multimodality を分離できる表現を先に検証する。

再現コマンド:

```powershell
python main9_model_analysis.py
python plot_main9_model_analysis.py
```

派生表と JSON は `result/main9_model_analysis/`、ランダム順の対照評価は `result/main9_forest_eval_b20000_random_order/` に保存されている。既存の main9、collision、corpus は変更していない。
