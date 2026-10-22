# D2単独学習からD2/D3混合への切替実験

2026-09-11。学習seed 9201。学習・全checkpoint評価完了。

## 結論

D2先行による両立は、この実験では得られなかった。最終NNはD2指定で91/100回成功したが、D3指定では0/100回成功、91/100回がD≈2のwrong-flatとなった。懸念されていた逆向きの偏りが観測された。

従来のランダム初期化からの混合学習はD3側に偏り、今回はD2側に偏った。この対比は、単なるD2の難しさだけでなく、学習履歴・既習経路への探索の偏り・条件別方策の学習困難が関与するという解釈を支持する。ただし、actorの表現力不足、探索不足、共有パラメータの更新干渉を個別に確定したものではない。

## 実験条件

- 出発点: `single_goal_five_d2_seed9201_20260910_220547_379979/checkpoints/update00064.pt`。D2単独512 episode、64更新、202,970 step後の最終NNを使用。最良checkpointの事後選択は行っていない。
- 切替後: D2/D3混合1,024 episode。各batchはD2×4＋D3×4をシャッフル。各目標512 episode、128回のactor/critic更新。
- NN・actor/critic optimizer・Torch行動乱数・目標順序乱数の状態を引き継ぐ。切替後のepisode/update/step番号は0から別記。
- D*入力付きの既存NN、元の報酬・環境・更新式を維持。spectral重み0.05、crossはpenaltyなしreject・episode続行、joint milestoneはpotential差。
- actor LR=0.0005、critic LR=0.00015、batch=8、horizon=400、最大step幅=0.1。LR/batchの途中変更なし。
- advantageは元のbatch全体centering。目標別centeringやcritic追加学習は使用しない。
- 切替前と混合128 episodeごとに、固定NNを各目標100 rollout評価。seed 950000–950099を各checkpoint・目標で共用し、評価結果は学習へ戻さない。
- near-miss到達率は、経路中に一度でもF≤0.1かつ|D−D*|≤0.05を満たすrolloutの割合。状態数の割合ではない。strict successはF≤0.001かつspread許容内。Q*は各rollout中のmax(qF×qD)。

再実行コマンド（作業ディレクトリ`00_0822_all`）:

```text
python RL_mixed_goals.py --warm-start result/experiments/single_goal_five_d2_seed9201_20260910_220547_379979 --episodes 1024 --eval-every 128 --diagnostics --workers 2
```

## 固定NN評価

成功・near-miss欄は100 rollout中の回数。横軸updateは切替後の更新回数で、事前学習64更新は含めない。

| 混合episode | 切替後更新 | D2成功 | D2 near-miss | D2 Q*中央値 | D3成功 | D3 near-miss | D3 Q*中央値 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 11 | 72 | 0.45368 | 0 | 0 | 0.02884 |
| 128 | 16 | 18 | 87 | 0.51946 | 0 | 0 | 0.01544 |
| 256 | 32 | 0 | 42 | 0.32124 | 0 | 0 | 0.05646 |
| 384 | 48 | 41 | 96 | 0.75415 | 0 | 0 | 0.02040 |
| 512 | 64 | 52 | 100 | 1.00000 | 0 | 0 | 0.03994 |
| 640 | 80 | 16 | 92 | 0.54342 | 0 | 1 | 0.08181 |
| 768 | 96 | 9 | 100 | 0.66851 | 0 | 0 | 0.01159 |
| 896 | 112 | 77 | 100 | 1.00000 | 0 | 0 | 0.00000 |
| 1024 | 128 | 91 | 100 | 1.00000 | 0 | 0 | 0.00629 |

D2は最終的には向上したが、成功率は途中で0%、9%まで低下している。単調な保持・改善や、その後も安定することは確認していない。D3は640 episode後の評価でnear-missが1回だけあり、「D3近傍を一度も試せない」とは断定できない。

最終D2成功率の95% Wilson区間は83.77–95.19%。D3の0/100は区間0–3.70%で、真の成功確率が厳密に0という意味ではない。同じ評価seedを使う時系列は独立した学習seedの反復ではない。

## 最終到達形状

| 指定目標 | 成功 | wrong-flat | timeout | 終端D中央値 | 平均step |
| --- | ---: | ---: | ---: | ---: | ---: |
| D2 | 91 | 0 | 9 | 1.999838 | 208.60 |
| D3 | 0 | 91 | 9 | 1.999791 | 226.29 |

D3を指定してもD≈2へ向かっている。単にD3の成功判定直前で惜しく失敗した、という結果ではない。

![最終終端形状](terminal_distribution.png)

## 学習量と学習中の到達

| 区間・目標 | episode | step | 成功episode | near-miss到達episode |
| --- | ---: | ---: | ---: | ---: |
| 切替後D2 | 512 | 178,926 | 176 | 464 |
| 切替後D3 | 512 | 185,592 | 0 | 0 |
| 切替後合計 | 1,024 | 364,518 | 176 | 464 |

事前学習を加えた総量は1,536 episode、567,488 step、actor/criticそれぞれ192更新。D2は事前学習込み1,024 episode・381,896 step、D3は512 episode・185,592 step。

切替後D2の最初の成功は混合episode 5、D2の4回目、切替後更新0回、切替後総step 1,854（D2 step 1,454）。これは既に学習したD2の再成功であり、新しい初回発見ではない。

## 従来の混合学習との対比

| 開始方法 | 混合episode | D2評価成功/near-miss | D3評価成功/near-miss |
| --- | ---: | --- | --- |
| ランダム初期化から混合 | 1,024 | 0/100・0/100 | 100/100・100/100 |
| D2単独512 episode後から混合 | 1,024 | 91/100・100/100 | 0/100・0/100 |

従来値は`mixed_goals_seed9201_20260910_232423_775199`の混合1,024 episode評価。混合episode数は同じだが、事前学習、環境step、optimizer履歴・乱数履歴は同じではない。効率の公平な比較ではなく、学習履歴を変更した診断として扱う。

## 更新の補助診断

切替後U=0,16,32の次batchを保存状態から再現し、全episode結果と次のNNが実際の保存記録と完全一致した。3batchともD2/D3のactor方策勾配cosineは負（−0.490、−0.137、−0.051）。共有パラメータの更新方向に対立成分があることと整合する。ただしentropy・clip・Adam変換前の方策勾配であり、負のcosineだけで実際の性能低下の原因を確定できない。

[再現episode診断](advantage_diagnostic/episodes.csv)、[勾配診断](advantage_diagnostic/gradients.csv)。

## 実装・検証

- `RL_mixed_goals.py`に`--warm-start`を追加。元runは変更せず、新runへ記録。通常学習の既定設定は変更していない。
- 切替初期NN、optimizer全状態、行動・目標順序乱数が出発checkpointと完全一致。
- 切替前D2評価100回の各結果が、以前のD2単独最終評価と完全一致。
- 全128batchが4+4、全episodeの更新番号・累積step・診断件数が整合。
- 最終Adam更新数はactor/criticとも192で、事前64＋切替後128に一致。
- 出発checkpointのSHA-256は変更なし。実行ソースのSHA-256もdesign.jsonと一致。
- Randomの100回評価は同条件の既存記録をコピーして使用。
- PNGを保存し、学習曲線・成功率・最終到達形状を目視確認。PDFは作成していない。

## 解釈の限界

1学習seed・混合1,024 episodeでの診断であり、D3が今後絶対に学べないことや、汎化が不可能なことは示さない。D2先行を汎化手法として採用できる結果でもない。今回変更したのはNNだけでなくcritic・optimizerを含む学習履歴なので、actorの探索分布だけが唯一の原因と断定しない。

次に必要なのは、先に学習させる目標を探すことより、D*ごとの行動の使い分けと探索を確保できるかの切り分けである。本実験から、NNの大規模化やTransformer化が必要だとまでは判断できない。追加の対策実装・追加学習は実行していない。

## 保存図・生データ

- [Learning curve](learning_curve.png)
- [D2成功率](five_d2_strict_success.png) / [D3成功率](five_d3_strict_success.png)
- [D2 near-miss](five_d2_near_miss.png) / [D3 near-miss](five_d3_near_miss.png)
- [D2 Q*](five_d2_quality.png) / [D3 Q*](five_d3_quality.png)
- [評価集計](evaluation_summary.csv) / [学習集計](training_summary.json) / [実行設定](design.json)

評価グラフは黒実線、確率の帯は95% Wilson区間、Q*の帯はrollout間の25–75%四分位。黒破線はRandom。フォントはTimes New Roman。learning curveは各目標20 episode移動平均と生値を重ねている。
