# ランダム初期化からのD2/D3同時学習＋適応的探索

2026-09-11。学習seed 9201。事前学習なし、256混合episodeの初期検証を完了。

## 結論

今回の短期学習ではD3側への偏りは解消していない。通常NNの最終100回評価ではD2成功・near-missとも0回、D3は両方100回。D2指定でも全100回がD≈3のwrong-flatだった。

一方、探索補助を付けた最終評価ではD2成功1/100、near-miss2/100となった。D2へ完全に到達不能というわけではない。ただし、この成功は評価中に出たもので、学習データへ戻していない。学習中のD2成功・near-missはともに0/128。両目標の安定した獲得・汎化を示した結果ではない。

## 条件

- 以前のランダム初期化からの混合学習と、初期NNの全重み・行動乱数状態が一致。D2事前学習checkpointは使わない。
- D2×4＋D3×4のbatchをシャッフル。32更新、各目標128 episode。
- 各目標の直近32学習episodeの成功率pから、e=max(0,1−p/0.2)をbatch開始時に計算。履歴なしはp=0。両目標に同じルールを適用。
- モード係数と符号付きstep幅のGaussianの平均をmu/(1+e)、分散をv+eへ変更。変換後分布でサンプリングし、同じ分布のlog-probability・entropyで学習。
- D2は全期間e=1、D3は成功率の上昇に応じてe=1から0へ低下。学習履歴をNN観測に加えず、外部の探索制御として実装。
- 報酬・NN構造・LR・学習割合は前試験と同じ。spectral重み0.05、wrong-flat罰則−2とshaping取消、joint potential差、crossはpenaltyなしreject・episode続行。
- actor LR=0.0005、critic LR=0.00015、batch全体advantage centering、最大400 step、最大step幅0.1。追加critic更新なし。
- 固定評価は各目標100 rollout、seed950000–950099。同じ評価seedをcheckpoint間で使い、評価中は更新しない。
- 通常NN評価はe=0。探索補助付き評価はそのcheckpointの学習履歴から求めたeで固定し、別フォルダへ保存。

```text
python RL_mixed_goals.py --adaptive-exploration --episodes 256 --eval-every 128 --diagnostics --workers 2
python evaluate_exploration_probe.py result/experiments/mixed_goals_seed9201_20260911_122009_729405 --updates 0 32
```

## 学習中の結果

| 目標 | episode | step | strict success | near-miss到達 |
| --- | ---: | ---: | ---: | ---: |
| D2 | 128 | 38,161 | 0 | 0 |
| D3 | 128 | 19,000 | 108 | 110 |
| 合計 | 256 | 57,161 | 108 | 110 |

near-missは経路中に一度でもF≤0.1かつ|D−D*|≤0.05となったepisodeの数で、successを含む。状態数の割合ではない。

D3の初回成功は混合45 episode目、D3の21回目、NN更新5回後。初回成功までの総stepは17,766、D3のみ8,224 step。成功episodeは247 step、終端F=0.000398873、D=2.999430053。経路は`five_d3/first_strict_success`へ保存。

D2はtimeout42回、wrong-flat86回。D3はtimeout19回、wrong-flat1回、success108回。D2の経験は38,161 stepで、以前のD2単独試験の202,970 stepの約19%にとどまる。短いwrong-flat終了が増えるため、同じepisode数でも探索stepは少なくなる。

## 通常NN評価（e=0）

| 混合episode | D2成功/100 | D2 near-miss/100 | D3成功/100 | D3 near-miss/100 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 2 |
| 128 | 0 | 0 | 100 | 100 |
| 256 | 0 | 0 | 100 | 100 |

最終D2のQ*中央値は0.07091、終端D中央値2.999307でwrong-flat100回。D3はQ*中央値1、終端D中央値2.999362で成功100回。

![通常NNの最終終端形状](terminal_distribution.png)

## 探索補助付き固定評価

| checkpoint | 目標 | e | 成功/100 | near-miss/100 | Q*中央値 |
| --- | --- | ---: | ---: | ---: | ---: |
| 初期 | D2 | 1 | 0 | 0 | 0.08248 |
| 初期 | D3 | 1 | 0 | 1 | 0.03665 |
| 混合256後 | D2 | 1 | 1 | 2 | 0.08664 |
| 混合256後 | D3 | 0 | 100 | 100 | 1.00000 |

D2は学習前後でe=1、評価seedも共通。1回の成功が得られたが、発見頻度はまだ低く、1学習seed・100評価rolloutの結果を安定した改善とは扱わない。0/100も真の確率ゼロを意味しない（95% Wilson区間0–3.70%）。

評価中に成功したD2はseed950048。178 step、F=0.0000291713、D=1.999914365。固定checkpointと同じ乱数・eから再実行し、全summary値が一致したことを検証した。経路179ファイルを`five_d2/evaluation_only_success_seed950048`へ保存。これは学習中のfirst hitではなく、評価専用の成功例である。

## 解釈と次の検証

探索制御は意図したとおり、D3の探索を弱め、未成功D2の探索を強め続けた。しかし、この強さ・256 episodeでは通常NNのD3偏重を解消できなかった。単にcontrollerを付ければ同時学習が直ちに成立するという結果ではない。

一方、D2の学習stepは単独試験より大幅に少なく、補助付き評価では成功例も出ている。同時学習が不可能、または探索強化に効果がないと確定するには早い。次は同じ設定のままD2に使うstep予算を単独試験程度まで増やし、学習中にもD2成功を獲得できるかを見ることが候補。今回は256 episodeで区切り、延長や別強度の調整はしていない。

## 検証・保存物

- 事前学習なし、初期NN・Torch乱数が以前の混合学習の初期状態と一致。
- 環境設定はepisode上限と追加critic更新0の明示以外、以前の混合学習と一致。
- U=0,16の次batchを保存状態から再現し、全episode結果・次のNN・Torch乱数が完全一致。
- 全32batchが4+4、episode記録と診断記録は256件、総step57,161と一致。
- 最終actor/critic optimizerはいずれも32更新。実行ソースSHA-256はdesign.jsonと一致。
- 既存の初期通常NN・Random評価を同条件確認後に再利用。評価は学習へ戻していない。
- グラフ8枚を保存し、終端分布を目視確認。PDFは作成していない。
- この試験で学習コードやhyperparameterの追加変更は行っていない。

[学習集計](training_summary.json)、[通常NN評価集計](evaluation_summary.csv)、[実行設定](design.json)。

[Learning curve](learning_curve.png)、[D2 near-miss](five_d2_near_miss.png)、[D3 near-miss](five_d3_near_miss.png)、[D2 Q*](five_d2_quality.png)、[D3 Q*](five_d3_quality.png)。

[D2の評価成功最終dat](five_d2/evaluation_only_success_seed950048/step178.dat)、[D3の学習初回成功最終dat](five_d3/first_strict_success/step247.dat)。
