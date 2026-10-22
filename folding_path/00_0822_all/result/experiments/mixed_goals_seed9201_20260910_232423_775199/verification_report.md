# D2の探索step数をそろえた混合学習の延長

## 結論

D2累積203,433 stepまで混合学習を延長しました。単独学習の202,970 stepに対し463 step（0.228%）超過です。最終固定NNのD2評価はstrict成功0/100、near-miss到達0/100で、単独学習の11/100、72/100に達しませんでした。D3の成功は100/100です。

このseed・報酬・LR・混合方法では、D2の探索step不足だけを補っても目標の折り分けは改善しませんでした。目標間の干渉やD3への偏りを疑う根拠は強まりましたが、勾配干渉そのものを測定した実験ではなく、全seedやすべての混合学習法に一般化する結論ではありません。

## 比較

単独比較元: `single_goal_five_d2_seed9201_20260910_220547_379979`。

| 指標 | D2単独 | 延長後D2/D3混合 |
|---|---:|---:|
| D2探索step | 202,970 | 203,433 |
| D3探索step | 0 | 191,281 |
| 全探索step | 202,970 | 394,714 |
| D2 episode | 512 | 2,016 |
| 全episode | 512 | 4,032 |
| NN更新回数 | 64 | 504 |
| D2最終strict成功 | 11/100 | 0/100 |
| D2最終near-miss到達 | 72/100 | 0/100 |
| D2最終Q*中央値 | 0.45368 | 0.08270 |
| D2学習中成功数 | 19 | 1 |

単独と同じ初期NN、seed 9201、D2報酬・環境設定を使用。D*入力あり、spectral重み0.05、cross reject/penaltyなし、最大400 step、最大step幅0.1、actor LR 0.0005、critic LR 0.00015、batch 8を維持しました。混合batchはD2/D3を4回ずつ、単独はD2を8回です。

D2 step数を一致させた比較であり、更新回数・episode数・D3の追加経験まで同じではありません。混合学習の初回D2発見は95 mixed episode（D2経験48回、11更新、全体35,119 step、D2だけ17,799 step）で、今回の延長では追加成功はありませんでした。単独の初回発見は291 episode、36更新、116,214 stepです。初回発見速度とその後の方策性能は区別してください。

## 延長中の固定NN評価

各NN・各目標100 stochastic rollout。学習から独立したseed 950000–950099をcheckpoint間で共通使用し、評価でNNを更新していません。

| 全episode | NN更新 | D2 strict | D2 near-miss | D2 Q*中央値 | D3 strict |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 128 | 0/100 | 0/100 | 0.08224 | 100/100 |
| 2,048 | 256 | 0/100 | 0/100 | 0.09044 | 100/100 |
| 3,072 | 384 | 0/100 | 0/100 | 0.08794 | 100/100 |
| 3,952 | 494 | 0/100 | 0/100 | 0.08039 | 100/100 |
| 4,032 | 504 | 0/100 | 0/100 | 0.08270 | 100/100 |

Strict条件はF<=0.001かつspread誤差<=0.05。near-miss到達は経路中にF<=0.1かつspread誤差<=0.05を一度でも満たしたrolloutの割合で、成功も含みます。Q*は経路中の最大qF*qDです。

最終D2評価はwrong-flat100回、D3評価はsuccess100回。D2の平均step134.32、D3は115.92。0/100は真の成功確率ゼロの証明ではなく、Wilson 95%区間は0–3.70%。100/100の区間は96.30–100%。これは1学習seedの実験です。

## 予算による停止と再開

最初に`mixed_goals_seed9201_20260910_214912_626246`の1,024 episodeから再開。停止条件は評価成績を見ずに、D2累積step>=202,970となった最初のbatch更新後、と事前に設定しました。episode上限は10,000です。

4,024 episode終了時はD2累積202,889 stepで予算未達、次batch終了の4,032 episodeで203,433 stepとなったため停止しました。停止判定の短い実環境テストも実施済みです。

3,952 episodeのCSV保存時にWindowsのアクセス拒否が発生しました。NN・Adam・Torch RNG・目標順序RNGはupdate00494.ptに保存済みで、episode CSVも3,952まで保存されていました。完成済みupdates.csv.tmpの最終行を元にupdates.csvとcheckpoint indexを復旧し、episode CSVから中断時summaryを再構成しました。一時CSVと復旧後の全更新行が一致することを確認しています。

中断run: `mixed_goals_seed9201_20260910_225814_393082`。そこから新しい本runへ保存状態を引き継ぎました。CSV置換時の一時PermissionErrorのみ短時間再試行する処理を追加し、数値計算・報酬・更新式は変更していません。元の完了済み実験結果は変更していません。

## 検証・保存物

- 再開元checkpoint/evaluationを全ファイルSHA-256で比較し、byte単位の一致を確認。
- 前回1,024 episode、および中断まで3,952 episodeのCSV行が引継ぎ先と完全一致。
- 全batchがD2/D3=4/4であること、予算を初めて超えたbatchで停止したことを確認。
- `training_summary.json`: 実際の停止episode、step、更新、初回成功、終了理由。
- `evaluation_summary.csv`, `evaluation/*.csv`: 集計と各rollout結果。
- `learning_curve.png`: 目標別reward/terminal flatness、生値と20目標episode移動平均。
- `five_d2_*.png`, `five_d3_*.png`: 固定NN評価。黒破線はRandom。確率の帯はWilson区間、Q*の帯はrollout四分位範囲。
- `terminal_distribution.png`: 最終NNの到達F/D分布。
- checkpointとbest reward更新時・初回成功の経路datを保存。
