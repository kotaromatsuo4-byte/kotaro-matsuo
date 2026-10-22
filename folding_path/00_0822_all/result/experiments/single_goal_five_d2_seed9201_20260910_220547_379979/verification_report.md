# D*入力付きNNのD2単独学習

## 結論

512 episodeでD2成功経路を学習中に19回発見しました。最終NNの独立100 rollout評価はstrict成功11回、near-miss到達72回でした。同じ初期重み・D2設定の混合学習では最終D2評価が両者0/100だったため、このseedでは単独学習の方がD2の探索分布を改善しました。ただし成功率11%であり、安定してD2を折れるとはまだいえません。

D*入力を追加するとD2が学べなくなる、という結果ではありません。混合学習時のD3への偏り・目標間の干渉を疑う材料になりますが、1seedの比較だけで原因を確定することはできません。

## 実行条件

`python RL_mixed_goals.py --task five_d2 --episodes 512 --seed 9201 --eval-every 128 --workers 2`

- 初期化からD2のみで学習。混合学習後のNNからの追加学習ではありません。
- 同じD*入力付きactor-critic。D*=2を入力。混合学習の初期NNと全重みが完全一致することを確認。
- seed 9201、batch 8、actor LR 0.0005、critic LR 0.00015、spectral weight 0.05。成功後もLR/batchは固定。
- horizon 400、max step width 0.1、cross reject・cross penaltyなし。D2環境設定は前回とepisode数以外完全一致。
- 各checkpointでNNを固定し100 stochastic rollout。評価seed 950000–950099はcheckpoint間で共通、学習乱数と別。評価から学習へのフィードバックなし。
- 初期NN・Randomの各100 rolloutは、初期重み・環境・評価seedの一致を確認して前回結果を再利用。
- strict: F<=0.001かつ|D-2|<=0.05。near-miss: rollout中に一度でもF<=0.1かつ|D-2|<=0.05を満たす（成功を含む）。状態数の割合ではなく、条件領域へ到達したrolloutの割合。

## 初回発見と学習全体

| 指標 | 結果 |
|---|---:|
| 初回成功episode | 291 |
| 初回成功までのNN更新回数 | 36 |
| 初回成功までの総step | 116,214 |
| 初回成功episodeのstep | 391 |
| 初回成功F | 0.0000545797 |
| 初回成功D | 1.99991685 |
| 全学習episode | 512 |
| 全NN更新回数 | 64 |
| 全学習step | 202,970 |
| 全学習中の成功数 | 19/512 |
| 最後の100学習episodeの成功数 | 16/100 |

初回成功経路は`five_d2/first_strict_success/step*.dat`に392状態（初期状態を含む）保存。`training_summary.json`の`mixed_episode`は既存出力との互換フィールド名で、単独学習では通常のepisode番号です。

## 固定NN評価

| 学習episode | NN更新 | strict成功 | near-miss到達 | Q*中央値 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0/100 | 0/100 | 0.06767 |
| 128 | 16 | 0/100 | 1/100 | 0.10700 |
| 256 | 32 | 1/100 | 5/100 | 0.14220 |
| 384 | 48 | 4/100 | 42/100 | 0.30818 |
| 512 | 64 | 11/100 | 72/100 | 0.45368 |
| Random | — | 0/100 | 0/100 | 0.06191 |

最終strict率のWilson 95%区間は6.25–18.63%、near-miss率は62.51–79.86%。最終100 rolloutは成功11・timeout89・wrong-flat0。平均stepは389.37。

評価時点では改善傾向が見られますが、すべてのNN更新で単調に改善したことを意味しません。これは1つの学習seedの評価であり、複数学習seedでの再現性は未検証です。Q*の帯はrollout間の四分位範囲、成功率・near-miss率の帯はWilson信頼区間です。

## 混合学習との比較

比較元: `mixed_goals_seed9201_20260910_214912_626246`。

| 指標 | D2単独 | D2/D3混合 |
|---|---:|---:|
| 総学習episode | 512 | 1,024 |
| D2学習episode | 512 | 512 |
| NN更新 | 64 | 128 |
| D2学習step | 202,970 | 76,967 |
| 全task学習step | 202,970 | 146,770 |
| 最終D2 strict成功 | 11/100 | 0/100 |
| 最終D2 near-miss | 72/100 | 0/100 |
| 最終D2 Q*中央値 | 0.45368 | 0.08224 |

D2のepisode数を一致させた比較です。混合はbatch内D2が4回、単独は8回で、更新回数・総step・更新ごとの目標構成は異なります。単独のD2探索stepは混合より多いため、計算予算まで完全に同じ条件での優位性を証明する比較ではありません。

また初回発見自体は混合の方が早く、混合では全体95 episode（D2経験48回）、11更新、全体35,119 stepで初回成功しています。単独の利点として今回確認できたのは後半の探索分布改善であり、初回発見の高速化ではありません。

## 保存・検証

- 全512行がD2、累積step202,970、初期を含むcheckpoint65個を確認。
- 保存された実行時ソースSHA-256が現在の対象コードと一致。
- `learning_curve.png`: reward/terminal flatnessの生値と20 episode移動平均。
- `five_d2_near_miss.png`, `five_d2_quality.png`, `five_d2_strict_success.png`: 固定NN評価推移。
- `terminal_distribution.png`: 最終NNの終端F/D分布。
- `evaluation_summary.csv`, `evaluation/*.csv`: 集計と100 rolloutごとの生データ。
- 既存の混合学習結果は変更せず、新規runに保存。
