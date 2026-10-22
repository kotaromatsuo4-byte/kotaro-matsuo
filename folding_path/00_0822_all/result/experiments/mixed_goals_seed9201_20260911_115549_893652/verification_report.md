# 成功率適応型の探索強化：D2事前学習からの混合学習

2026-09-11。学習seed 9201。まず256混合episodeの初期検証を実施。

## 結果の要点

D3の初回成功を混合244 episode目（D3の123回目）に発見した。探索強化なしの先行実験では混合1,024 episode、D3の512回まで未発見だった。今回の探索強化下ではD3の成功近傍と成功経路を実際に経験できた。

ただし、最終NNの探索補助なし評価はD3成功0/100、near-miss到達3/100。適応controllerを付けた評価では成功7/100、near-miss到達12/100。初回発見にはつながったが、探索補助なしでD2/D3を両立できたとは言えない。1seedの短期試験であり、速度向上の再現性や汎化は未検証。

## 実装した共通ルール

各目標の直近32学習episodeのstrict success率pを用い、batch開始時に探索強度をe=max(0,1−p/0.2)とする。未経験目標はp=0。事前学習の履歴も使うが、評価結果は使わない。D2/D3のどちらにも同じ式を適用し、各batch中は固定する。

モード係数・符号付きstep幅の潜在Gaussianを両方とも次のように変更する。

- 平均: mu → mu/(1+e)
- 分散: v → v+e

単なる分散下限ではなく、平均を弱めて分散を増やす組み合わせ。最大e=1では平均を半分にし、分散を1加える。物理step幅は従来どおりtanh変換で最大0.1、最大400 step/episodeのまま。

サンプリング、log-probability、entropyは同じ変換後分布から計算する。元分布のlog-probabilityを誤用していない。ただし、更新対象は探索強化後の方策分布なので、補助を外したNNの評価性能が同じになる保証はない。

controllerは学習側の外部制御。NNの観測や構造は増やしていない。成功率が下がれば探索強度は再び上がる。初回成功で直ちに探索を切らない。

初期D2は過去32回の成功率15.625%からe=0.21875、未経験のD3はe=1。試験中の範囲はD2が0–0.21875、D3が0.84375–1。最終評価時の適応値はD2が0、D3が0.84375。

## 維持した条件

- 出発点: `single_goal_five_d2_seed9201_20260910_220547_379979/checkpoints/update00064.pt`。D2単独512 episode・64更新・202,970 step。
- NN、optimizer全状態、行動と目標順序の乱数状態を引き継ぐ。
- batch内D2×4＋D3×4、actor LR=0.0005、critic LR=0.00015、advantageのbatch全体centering。
- spectral重み0.05、joint potential差、wrong-flat罰則−2とshaping取消、crossはpenaltyなしreject・episode続行。
- LR、学習割合、報酬、物理step上限は変更なし。critic追加学習なし。
- 初期、混合128、256 episode後に各目標100回評価。通常NN評価はe=0。
- 探索補助付きの固定評価は別フォルダ`exploration_probe`へ保存し、通常NNのグラフに混ぜない。
- 評価seedは950000–950099。学習とは分離し、比較対象間では同じseedを使用。

コマンド（作業ディレクトリ`00_0822_all`）:

```text
python RL_mixed_goals.py --warm-start result/experiments/single_goal_five_d2_seed9201_20260910_220547_379979 --adaptive-exploration --episodes 256 --eval-every 128 --diagnostics --workers 2
python evaluate_exploration_probe.py result/experiments/mixed_goals_seed9201_20260911_115549_893652 --updates 0 32
```

## 初回成功・学習量

| D3初回成功までの指標 | 値 |
| --- | ---: |
| 切替後の混合episode | 244 |
| D3だけのepisode数 | 123 |
| 切替後NN更新回数 | 30 |
| 切替後の総step | 93,254 |
| D3だけのstep | 48,809 |
| 成功episodeのstep | 353 |
| 終端F | 0.000277105 |
| 終端D | 2.999423464 |
| 探索強度e | 1 |

事前学習込みの初回D3成功時点は756 episode、94更新、296,224 step。更新数は成功episodeを生成する前までの数。

| 切替後の学習対象 | episode | step | strict success | near-miss到達 |
| --- | ---: | ---: | ---: | ---: |
| D2 | 128 | 47,069 | 39 | 98 |
| D3 | 128 | 50,768 | 1 | 7 |
| 合計 | 256 | 97,837 | 40 | 105 |

near-missはF≤0.1かつ|D−D*|≤0.05を経路中に一度でも満たしたepisodeの数で、successを含む。状態数の割合ではない。

最終的な事前学習込み総量は768 episode、96更新、300,807 step。初回成功経路は`five_d3/first_strict_success/step0.dat`から`step353.dat`まで354ファイルを保存。

## 通常NNの固定評価（探索補助なし）

成功・near-miss欄は100回中の回数。Q*はrollout中のmax(qF×qD)の中央値。

| 混合episode | D2成功 | D2 near-miss | D2 Q* | D3成功 | D3 near-miss | D3 Q* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11 | 72 | 0.45368 | 0 | 0 | 0.02884 |
| 128 | 34 | 100 | 0.75318 | 0 | 0 | 0.01653 |
| 256 | 30 | 51 | 0.33546 | 0 | 3 | 0.07806 |

D3 near-miss 3/100の95% Wilson区間は1.03–8.45%。成功0/100の区間は0–3.70%。評価結果だけから統計的に明確な改善・成功確率ゼロを断定しない。

探索強化なしの先行run `mixed_goals_seed9201_20260911_095038_494045`の同じ256混合episode時点では、D2成功0/100・near-miss42/100、D3成功0/100・near-miss0/100・Q*中央値0.05646。今回のD3側には小さい改善の兆候があるが、別々に進む確率的学習の1seed比較なので一般化しない。episode予算は同じでも経路長が変わり、step予算は完全一致ではない。

## 探索補助付きの固定評価

| checkpoint | 目標 | e | 成功/100 | near-miss/100 | Q*中央値 |
| --- | --- | ---: | ---: | ---: | ---: |
| 学習前 | D2 | 0.21875 | 9 | 40 | 0.29229 |
| 学習前 | D3 | 1 | 0 | 0 | 0.05381 |
| 混合256後 | D2 | 0 | 30 | 51 | 0.33546 |
| 混合256後 | D3 | 0.84375 | 7 | 12 | 0.10824 |

これは固定NNと固定eでの評価であり、評価rolloutの成功によってeを変更しない。学習前後のeが異なるので、この表だけをNN学習効果の純粋な比較とはしない。

### 同じ探索強度に固定した学習前後比較

D3に対してe=1を両checkpointで固定し、同じ100評価seedを使用した。学習前の記録は上記e=1の評価を再利用した。

| D3、e=1固定 | 成功/100 | near-miss/100 | Q*中央値 |
| --- | ---: | ---: | ---: |
| 学習前 | 0 | 0 | 0.05381 |
| 混合256後 | 5 | 12 | 0.11129 |

同じ探索補助条件でも学習後のNNでD3到達が増えた。この1回の学習では、単に補助の強さを変えただけではない改善の兆候がある。ただし、探索補助なしでは依然として成功0/100であり、補助なし方策への定着や複数学習seedでの再現性は別途検証が必要。

```text
python evaluate_exploration_probe.py result/experiments/mixed_goals_seed9201_20260911_115549_893652 --updates 32 --task five_d3 --strength 1
```

[固定e=1の最終NN評価記録](exploration_probe/five_d3_update00032_strength1.csv)。通常NN評価600 rolloutと探索補助付き評価500 rollout、既存Random200 rolloutを保存（同一条件の既存評価の再利用を含む）。

## 検証とファイル

- `test_exploration.py`: controller履歴、実際のサンプリングとlog密度の厳密一致、物理step上限、有限勾配のテスト通過。
- e=0では旧runのU=0,16,32の次batchと更新後NNを完全再現。
- 探索強化版のU=0,30,31で、controller履歴から探索強度を再計算し、episode結果・次NN・Torch乱数状態を完全再現。初回成功を挟む探索強度の変化も検証。
- 初期NNと行動乱数は探索強化なしの出発点と一致。両目標の環境設定もepisode上限以外は一致。
- 全32batchが4+4。最終actor/critic optimizer更新数は事前64＋新規32＝96。
- 実行ソースのSHA-256はdesign.jsonと一致。既存の元checkpointは変更していない。
- 初期通常NN評価とRandom評価は、同条件の既存結果をコピーして使用。
- PNGのnear-miss図と終端形状を目視確認。PDFは作成していない。

主な変更は`agent.py`、`RL_mixed_goals.py`、`mixed_diagnostics.py`。通常の既定動作は探索強化なし。`--adaptive-exploration`を指定した実験だけで有効にする。

- [Learning curve](learning_curve.png)
- [D2 near-miss](five_d2_near_miss.png) / [D3 near-miss](five_d3_near_miss.png)
- [D2 Q*](five_d2_quality.png) / [D3 Q*](five_d3_quality.png)
- [D2成功率](five_d2_strict_success.png) / [D3成功率](five_d3_strict_success.png)
- [終端形状](terminal_distribution.png)
- [通常NN評価集計](evaluation_summary.csv) / [学習集計](training_summary.json)
- [D3初回成功の最終dat](five_d3/first_strict_success/step353.dat)

## 今回言えること・次の検証

探索強化によってD3近傍と成功を経験できた点は有望。ただし1回の発見は「D3指定ならD3へ畳むNNを獲得した」ことではない。通常NNではまだ成功0/100で、終端D≈2への偏りも残る。D2のnear-miss率も100%から51%へ変動している。

次は報酬やcontrollerの値を変更せず学習を延長し、成功経験を繰り返せるか、補助なしNNへ性能が移るか、D2を保持できるかを確認するのが自然。今回は256 episode試験までを完了し、延長・複数学習seed・未知目標での評価は実施していない。
