# 主比較に加えた検証

すべて2026-09-14の5時間枠内で実施。元の主比較を置き換えず、別系列に保存する。
チェックポイント選択は既知課題の情報だけで行う。未知課題の教師は作らない。

| 系列 | 変更点 | 選択・予算 |
|---|---|---|
| graph_long | 元のfive/seven教師データで学習を延長 | 20,000更新、5 source seeds |
| graph_coverage | 既知の基準形状と既知課題でNNが訪れた状態に教師ラベルを追加 | 20,000更新、5 seeds。既知課題20条件の成功数でgraph_longと比較して選択 |
| extended | 主比較のseven D4を同じ乱数・optimizer状態から延長 | 最初の3 seedのscratch/finetune、総256,000提案。期限に達したものは時間打切り |
| graph_water | 保存されたwaterbomb RL初回発見座標への教師経路を追加 | water D1と√2のみ。20,000更新、5 seeds |
| graph_seven | sevenだけの教師データ・collectorで新規学習 | 20,000更新、3 seeds。fiveデータを間接的にも利用しない |
| bar_transfer | graph_sevenをfive D2/D3へ適用 | scratch/finetune/frozen各3 seeds、6,400提案 |
| bar_zero | 同じ転移を平均行動で試す | 各3 seeds、1経路400提案。未学習平均行動も対照にする |
| legacy_bar | 従来のD入力なし固定サイズactor | scratch各3 seeds、6,400提案。更新則・環境は共通 |
| fresh_bar | graph_seven seed5601を先に固定し、新しいfive初期形状を生成 | 8初期形状、乱数59501。scratch/frozen、6,400提案。8個のNNではない |
| nine_transfer | 新しい9節点linkageを評価前に固定 | graph_seven3 seeds、scratch/finetune/frozen、6,400提案。平均行動の対照も保存 |
| aligned_* | 符号付き幅平均を教師の有向平均に整合する結合 | ゼロショット診断のみ。主比較の方策を変更しない |
| projected_water | モードごとの非線形予測を、節点移動場のモード投影に変更 | waterを含む既知教師のみ、20,000更新、3 seeds。既知検証損失で選択 |
| projected_transfer | projected_waterの未知D4/torus転移 | 同じprojected構造のscratch/finetune/frozen、各51,200提案。時間打切りを明示 |

全体の終了期限は22:05:07 JST。計算ループは2分前に停止し、結果を保存する。
各キューの設定とledgerに、未着手・失敗・完了の状況を残す。
対照条件を含む固定された短い追試と、時間内で実行できた長い比較を区別する。

## 情報の出所

- graph_sevenの追加データ生成に使ったcollectorは、sevenだけで学習したdense actor。
- expanded_sourceのcollectorはfive/sevenで学習しているため、fiveを未知とする検証には使用しない。
- waterbombの教師目標は、water D1のseed8102・episode111の初回成功座標と、water √2のseed8101・episode23の初回成功座標。
- torusの成功座標・成功経路を教師生成や事前学習に使わない。
- 新しい9節点は、学習済みNNを評価する前に初期節点・接続・固定条件を保存した。目標座標ファイルは存在しない。

## 追加表現の位置づけ

projectedではNNの節点移動場vから、モード行列Mを用いて係数a=Mvを求め、全体ノルムで有界化する。
モード基底を直交行列Rで変換すると、係数平均もRaとなり、物理平均は同じになる。
分散は各モードで共通なので物理共分散も基底に依存しない。これを数値テストで確認する。
既存のGaussian行動サンプリング、半球正規化、幅、RL更新コードは共通だが、NNの表現と平均係数の有界化は変わる。
したがって、この比較だけから改善の原因を「基底不変性のみ」とは断定しない。

## 後半の追試と割当変更

| 系列 | 検証内容 | 独立性・情報の制限 |
|---|---|---|
| projected_d4_replication | 最初のD4成功後、残る2個の固定済みNNを6,400提案まで優先評価 | 長予算試行と同じ乱数列。早期成功は出所を保存して引き継ぎ、一度だけ数える |
| crossed_d4_source* / crossed_nine_source* | 3個の固定NNと3個の探索乱数を交差 | 対応する3条件を再利用し、残り6組を追加。9個の独立NNではない |
| d4_goal_ablation_d2/d3 | 環境はD4、NN入力だけD2またはD3 | 同じ重み・探索乱数。未知Dへの追従と探索の事前知識を切り分ける |
| projected_nine | D4教師なしの混合経験NNを9節点D3/D4へ適用 | 構造・D4とも教師に含まれない。frozen/scratch各3条件、6,400提案 |
| projected_nine_zero / *_control | 平均行動のみの9節点比較 | 学習済み/未学習、D3/D4各3条件、400提案 |
| d4_graph_matched / nine_graph_matched | 同じ教師データ・20,000更新の旧graph表現も確率的に探索 | 対応する探索乱数。D4教師を新規追加したNNは使わない |
| sequential_nine | 今回RLが発見したseven D4への教師経路を作り、新規NNからnineへ転移 | 33教師経路中19成功、1,787ラベル追加。RL actor重みは不使用。元の未知D4比較への流用を禁止 |
| nearest_prior_torus | 同じ面折り次元の既知Dで、目標Dに最も近い√2をNNへ入力 | 環境の目標・報酬はD2。frozen/scratch/finetune各3条件 |
| delta_prior_torus | 必要な広がりの変化量に近い既知D1をNNへ入力 | 環境はD2、既知NNは固定。frozen/scratch各3条件 |
| projected_bars_transfer | water経験を除き、five/seven D2/D3だけでprojectedを新規学習 | 3個全てのsource学習後に未知D4・nine D4を評価。既存scratchを再利用 |
| crossed_bars_d4_source* | D2/D3-onlyモデルのD4成功後、探索乱数を交差 | 固定済み3個×3探索乱数。元の対応3条件を再利用し、残り6組を期限内に評価 |
| fresh_d4 / gradient_fresh_d4 | 同じD2/D3-only NNを固定してから新しいseven初期形状4個を生成 | 乱数69701。全seven教師状態から距離0.03以上。frozen/scratchは6,400提案、勾配w=1は400提案。初期条件生成は共通の準備費用 |
| torus_depth | 最初のprojected seedのscratch/finetuneを総256,000提案へ延長 | 主比較2枠の終了後に開始。未着手selected 10条件の枠を振り替えた。期限による打切りを保存 |
| gradient_controls / gradient_fresh_bar | flatnessとD誤差だけの局所勾配法 | NN・経験・目標座標なし。全重み設定と失敗を報告。有限差分評価数も記録 |

これらは未知課題の結果を見て追加した探索的な検証である。未知課題の教師は使用しないが、設計を未知課題から独立に固定した最終holdout試験とは区別する。
`job_inventory.csv`は保存・未着手・時間打切り・引継ぎを確認するための成果物一覧で、行数を独立試行数として扱わない。
