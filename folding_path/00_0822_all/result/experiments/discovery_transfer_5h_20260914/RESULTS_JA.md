# 既知の折り経験による、未知課題の初回発見の高速化

更新: 2026-09-14T13:03:51.545838+00:00。実験期間: 2026-09-14 17:05～22:05 JST。終了時の保存結果を集計。

## 現時点で言えること

five/sevenのD2・D3だけで教師学習したprojected_barsは、未知seven D4を3個のNNすべてで初回発見した。提案数は673・565・281、RL更新は全て0回。対応する同じprojected actorの未学習RLは3条件とも6,400提案まで未発見だった。water経験もD4教師も不要な条件で、過去の折り経験が初回発見を短縮する例を確認できた。

さらに同じ3個のNNと3個の探索乱数を交差させた9組は、全て209～1,063提案で成功した。全て重み固定。これは9個の独立したNNを学習したという意味ではなく、3個のNNに対する探索乱数の追試である。

![D2/D3のみの経験での乱数交差比較](bars_crossed_discovery.png)

ここで成功したのは重み固定の確率的探索であり、平均行動だけの1経路では3個とも未成功だった。未知D4へ確実に折れる決定論的NNができた、という結果ではない。

![D2/D3だけで学習した同じNNのD3とD4](bars_only_d4_example.png)

![未知D4の初回発見曲線](d4_first_discovery.png)

D4教師を含まないprojected_waterのNNが、未知seven D4を612提案・RL更新0回で初回発見した。別のsource seedでも1,729提案・更新0回で成功した。教師はfive/seven D2・D3と既知waterbomb D1・√2であり、D4の成功座標はこのNNの学習に使っていない。共通6,400提案までの全3 source seedsの結果を次に示す。

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| D4_common_6400 / seven_d4 | projected・finetune | 2/3 | 1 | 0 | 612, 1729 |
| D4_common_6400 / seven_d4 | projected・frozen | 2/3 | 1 | 0 | 612, 1729 |
| D4_common_6400 / seven_d4 | projected・scratch | 0/3 | 3 | 0 | — |

この表は同じ乱数条件の短い追試と長い本試行を一度だけ数える。frozenとfinetuneが更新前に同じ経路で成功した場合も、独立した二つのNNの成功ではない。

![同じNNによる既知D3と未知D4](unseen_d4_example.png)

同じD4教師なしのNNを新しい9節点linkageへ適用すると、対応する3条件のうち1条件でD4を1,583提案・RL更新0回で発見した。同じprojected actorの未学習RLは3条件とも6,400提案まで未発見だった。ただし探索乱数を交差させる追試では9組中この1組だけが成功し、転移はまだ不安定だった。この条件では構造とDの両方が教師に含まれないが、小規模linkage系の範囲であり、torusへの転移成功を意味しない。

![未知9節点D4の初回発見](unseen_nine_example.png)

探索乱数を入れ替える追試では、seven D4は3個のNN×3個の探索乱数の9組中6組で成功した。source seed5601と5603は各3/3、5602は0/3で、学習済み重みによる差が大きい。これは独立NNを9個学習した結果ではない。

![学習済み重みと探索乱数の交差比較](crossed_discovery.png)

混合経験モデルでは、NN入力だけをD3にした対照でも、実際のD4を767・196提案で発見した。D2入力では3条件とも未発見だった。したがって、現段階で支持されるのは「経験で学んだ方策が未知課題の探索を助ける」という解釈であり、未知の数値Dへの正確な追従を証明したわけではない。D2/D3のみの追試モデルには、この入力置換対照はまだ行っていない。

![目標D入力の対照](d4_goal_input.png)
seven barのD2/D3教師経路だけで学習したNNを、未学習のfive barへそのまま適用すると、D3の初回発見を高速化できた。基準初期形状では3個のNNが47、45、53提案で成功し、RL更新は全て0回だった。新しい初期形状8個でも、最初に固定した1個のNNが33～61提案で全て成功した。

これは同じlinkage系で節点数が7から5へ変わる転移の結果である。未知D4、節点数を増やした9節点、面へ折るtorusについては、以下の結果を別々に評価する。five barでの成功をそれらの成功と同一視しない。

経験ありで初回成功が記録された未知課題: nine_d4, seven_d4。経験なしの対照で初回成功が記録された課題: seven_d4。予算内で未成功の課題を到達不可能とは判断しない。

water経験を除き、five/seven D2・D3だけで新規に学習したprojected_barsの追試は、seven_d4: 成功3/3保存済み条件（計画3条件）、初回発見の提案数673・565・281；nine_d4: 成功0/3保存済み条件（計画3条件）、初回発見の提案数なし。予算は各条件6,400提案、重み固定。混合経験の結果と区別する。

事前学習が不利になった条件もあった。初期dense比較のseed5704ではscratchがseven D4を18,751提案で発見した一方、対応する教師学習済みfinetune/frozenは51,200提案まで未発見だった。経験を付ければ常に速くなるという結果ではない。

追加した単純な幾何学的勾配法も、新しいfive初期形状8個のうち7個では22～31提案で成功し、その条件ではNNより速かった。基準初期形状のD3では勾配法の4設定は失敗した。したがって、今回のNNの効果は「未学習RLとの比較での高速化」と「初期形状による成功範囲の違い」として読むべきで、全手法に対する速度優位を示したものではない。

## ユーザーの意図との対応

既知課題の成功形状への多様な教師経路から、新しいactorを教師あり学習した。未知課題では初期節点・接続・固定条件・目標Dだけを与える。未知課題の目標座標、成功経路、教師勾配は使用しない。初回発見までの旧RL重みも転移しない。

projected_bars/projected_waterが使ったsevenの全6,952教師状態について、既に平坦D4を達成している状態が混入していないかも検査し、該当は0だった。ラベルがD2/D3というだけでなく、完成D4状態をsource状態として見ていたわけでもない（source_holdout_geometry_audit.json）。

barの既知目標は既存のユーザー提供形状を利用したため、すべての既知形状がRL発見由来であるとまでは主張しない。追加のwaterbomb教師は、保存されていた既知課題のRL初回成功座標から生成した。

後半の独立したsequential系列では、今回scratch RLが初回発見したseven D4を新たな既知経験とし、その座標への教師経路からNNを作り直してnineへ移す。RLのactor重みは使わない。このNNを元のD2/D3→D4比較へ混ぜることはコードで禁止した。D4が教師に含まれることは、この系列のnine構造への転移に限って許可する。

RLは既存のenv.py・agent.pyを変更せず利用した。モード係数のGaussian、半球正規化、符号付きtanh幅、8エピソードごとのMC actor–critic更新、報酬・交差処理は共通。初回成功を検出した時点で停止する。教師あり学習済みactorだけを移し、criticとoptimizerは新規にする。

denseは従来の固定節点数NNにD入力を追加したもの。graphは節点数を変えられる共有actorで、同じgraphのランダム初期化と比較する。legacyはD入力を持たない従来の固定サイズNNを使う参照条件。乱数はモデル初期化と行動サンプリングを分離して対応条件間で揃えたため、過去ログの乱数列そのものを再現する条件ではない。

## 指標の読み方

成功は従来環境のstrict_success（flatness ratio ≤ 0.001、D誤差 ≤ 0.05）。特定の重なり順や目標座標との一致ではない。提案数には交差などで棄却された行動も含む。保存した成功経路は部材長・固定条件・交差も再検査する。

表の「終了試行」はresult.jsonが保存された試行だけを数える。未着手・進行中を成功率の分母に混ぜず、時間打切りを別表示する。失敗を除いて発見速度を平均しない。曲線は打切り前の共通観測範囲で描く。

今回の5時間は探索的な検証であり、未知課題での失敗・成功を見て表現や補足条件を追加した。個々のcheckpointは既知課題の検証損失だけで選んでいるが、設計全体を未知課題から独立に固定した最終テストではない。未学習構造・Dの教師を使わないことと、設計選択にも使わない完全な最終holdoutであることは区別する。次の確認実験では今回の設計を固定して、新たな構造で比較する必要がある。

frozenは重み固定の確率的探索、deterministicは平均行動の1経路。finetuneは教師あり初期値からのRL。最初のRL更新より前に成功したfrozenとfinetuneは同一軌跡であり、独立した成功例として二重に数えない。実時間は探索ループのみで、起動・読み込み・既知教師の生成・事前学習を含まず、並行計算の負荷の影響も受ける。

![初回発見曲線](bar_first_discovery.png)

![既知課題の学習曲線](focused_source_learning.png)

![projectedの教師学習曲線](projected_source_learning.png)

学習曲線の損失は既知課題の未使用経路に対する値であり、未知課題の初回発見速度そのものではない。教師集合が異なるパネル間の損失値を、そのまま性能の優劣として比べない。

![同じactorのsevenからfiveへの転移](bar_transfer_example.png)

![新しい初期形状8個すべての折り経路](fresh_bar_shapes.png)

![同じ400提案までの比較](fresh_bar_controls.png)

## 初期の主比較

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| main / seven_d4 | dense・finetune | 0/5 | 5 | 0 | — |
| main / seven_d4 | dense・frozen | 0/5 | 5 | 0 | — |
| main / seven_d4 | dense・scratch | 1/5 | 4 | 0 | 18751 |
| main / torus_d2 | graph・finetune | 0/5 | 5 | 0 | — |
| main / torus_d2 | graph・frozen | 0/5 | 5 | 0 | — |
| main / torus_d2 | graph・scratch | 0/5 | 5 | 0 | — |
| zero / seven_d4 | dense・deterministic | 0/5 | 0 | 0 | — |
| zero / torus_d2 | graph・deterministic | 0/5 | 0 | 0 | — |

## D4の探索延長

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| extended / seven_d4 | dense・finetune | 0/2 | 2 | 0 | — |
| extended / seven_d4 | dense・scratch | 0/3 | 2 | 1 | — |

最初の2組ではscratch/finetuneとも総256,000提案まで未発見だった。第3組は残り時間内での観測に限られる。元の51,200提案試行の続きであり、初期試行と延長を独立に数えない。

## 経験だけで未知five barを折る検証

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| bar_transfer / five_d2 | graph・finetune | 0/3 | 3 | 0 | — |
| bar_transfer / five_d2 | graph・frozen | 0/3 | 3 | 0 | — |
| bar_transfer / five_d2 | graph・scratch | 0/3 | 3 | 0 | — |
| bar_transfer / five_d3 | graph・finetune | 3/3 | 0 | 0 | 47, 45, 53 |
| bar_transfer / five_d3 | graph・frozen | 3/3 | 0 | 0 | 47, 45, 53 |
| bar_transfer / five_d3 | graph・scratch | 2/3 | 1 | 0 | 3566, 341 |
| bar_zero / five_d2 | graph・deterministic | 0/3 | 0 | 0 | — |
| bar_zero / five_d3 | graph・deterministic | 2/3 | 0 | 0 | 36, 31 |
| bar_zero_control / five_d2 | graph・untrained_deterministic | 0/3 | 0 | 0 | — |
| bar_zero_control / five_d3 | graph・untrained_deterministic | 0/3 | 0 | 0 | — |
| legacy_bar / five_d2 | legacy・scratch | 0/3 | 3 | 0 | — |
| legacy_bar / five_d3 | legacy・scratch | 2/3 | 1 | 0 | 3596, 3426 |

## 新しい初期形状8個での追試

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| fresh_bar / five_d3 | graph・frozen | 8/8 | 0 | 0 | 61, 38, 33, 49, 46, 53, 45, 47 |
| fresh_bar / five_d3 | graph・scratch | 3/8 | 5 | 0 | 208, 331, 3591 |

6,400提案で打ち切った発見コストの平均 min(T, 6400): {'scratch': 4516.25, 'frozen': 46.5}。8条件は1個の学習済みNNに対する初期形状の追試であり、8個の独立した事前学習NNではない。初期形状はNNを固定した後に乱数59501で生成した。

## 新しい9節点構造

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| nine_transfer / nine_d3 | graph・finetune | 0/3 | 3 | 0 | — |
| nine_transfer / nine_d3 | graph・frozen | 0/3 | 3 | 0 | — |
| nine_transfer / nine_d3 | graph・scratch | 0/3 | 3 | 0 | — |
| nine_zero / nine_d3 | graph・deterministic | 0/3 | 0 | 0 | — |
| nine_zero / nine_d3 | graph・untrained_deterministic | 0/3 | 0 | 0 | — |

## 既知経験の追加とtorusへの転移

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| legacy_torus / torus_d2 | legacy・scratch | 0/5 | 5 | 0 | — |
| selected_zero / seven_d4 | graph・deterministic | 0/5 | 0 | 0 | — |
| selected_zero / torus_d2 | graph・deterministic | 0/5 | 0 | 0 | — |
| water_transfer / torus_d2 | graph・finetune | 0/4 | 3 | 1 | — |
| water_transfer / torus_d2 | graph・frozen | 0/3 | 3 | 0 | — |
| water_zero / seven_d4 | graph・deterministic | 0/5 | 0 | 0 | — |
| water_zero / torus_d2 | graph・deterministic | 0/5 | 0 | 0 | — |

## モード基底に依存しない方策

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| projected_transfer / seven_d4 | projected・finetune | 2/3 | 1 | 0 | 612, 1729 |
| projected_transfer / seven_d4 | projected・frozen | 2/3 | 1 | 0 | 612, 1729 |
| projected_transfer / seven_d4 | projected・scratch | 0/2 | 2 | 0 | — |
| projected_transfer / torus_d2 | projected・finetune | 0/2 | 1 | 1 | — |
| projected_transfer / torus_d2 | projected・frozen | 0/1 | 1 | 0 | — |
| projected_transfer / torus_d2 | projected・scratch | 0/2 | 2 | 0 | — |
| projected_zero / seven_d4 | projected・deterministic | 0/3 | 0 | 0 | — |
| projected_zero / torus_d2 | projected・deterministic | 0/3 | 0 | 0 | — |

## 未知D4の残る学習乱数の優先追試

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| projected_d4_replication / seven_d4 | projected・finetune | 1/2 | 1 | 0 | 1729 |
| projected_d4_replication / seven_d4 | projected・frozen | 1/2 | 1 | 0 | 1729 |
| projected_d4_replication / seven_d4 | projected・scratch | 0/2 | 2 | 0 | — |

最初のsource seedでの成功後、既に学習を完了・固定していた残り2 source seedsを優先して6,400提案まで評価した。条件を選り分けず残り両方を含める。元の51,200提案試行と同じ乱数列の先頭部分であり、独立した追加標本ではない。早期成功は長予算側へ出所付きで引き継ぎ、同じ発見を再計算しない。

## torusの同条件ペアを深く探索

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| torus_depth / torus_d2 | projected・finetune | 0/1 | 0 | 1 | — |
| torus_depth / torus_d2 | projected・scratch | 0/1 | 0 | 1 | — |

主比較の2枠が空いた後、projectedの最初のseedのscratch/finetuneを総256,000提案まで延長する。51,200提案の続きとしてmodel・optimizer・乱数状態を復元する。これは同じ試行の延長であり、新しい独立seedではない。未着手だったgraph_coverageのtorus追加学習10条件の枠を、この深さの比較へ振り替えた。終了期限に達した場合は実際に観測した提案数を示す。

![torusの同条件延長](torus_depth_progress.png)

この図はエピソード終端の診断である。flatnessとD誤差の個別最小値は別々の状態で生じ得るため、右図では同じ状態に対するmax(flatness/0.001, D誤差/0.05)も示す。初回発見の判定自体は全提案で行っている。

## 学習済み重みと探索乱数を入れ替える

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| crossed_d4_source5601 / seven_d4 | projected・frozen | 2/2 | 0 | 0 | 273, 234 |
| crossed_d4_source5602 / seven_d4 | projected・frozen | 0/2 | 2 | 0 | — |
| crossed_d4_source5603 / seven_d4 | projected・frozen | 2/2 | 0 | 0 | 142, 2943 |

3個の学習済みNNと3個の探索乱数を交差させる。元の対応3条件は再利用し、残り6条件だけを新たに実行する。合計9組は9個の独立した学習済みNNではない。各条件6,400提案まで、重みは更新しない。

## NNに与えるDだけを変更する対照

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| d4_goal_ablation_d2 / seven_d4 | projected・frozen | 0/3 | 3 | 0 | — |
| d4_goal_ablation_d3 / seven_d4 | projected・frozen | 2/3 | 1 | 0 | 767, 196 |

環境の成功条件はD4のまま、NN入力の目標Dだけを3または2に変更する。初期形状・学習済み重み・探索乱数は正しいD4入力の条件と揃える。これはgoal入力の寄与を調べる対照であり、正しい目標を与えた実用条件とは分ける。教師は生成せず、NNの重みも更新しない。

## 同じ複数構造の経験を9節点へ適用

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| projected_nine / nine_d3 | projected・frozen | 0/3 | 3 | 0 | — |
| projected_nine / nine_d3 | projected・scratch | 0/3 | 3 | 0 | — |
| projected_nine / nine_d4 | projected・frozen | 1/3 | 2 | 0 | 1583 |
| projected_nine / nine_d4 | projected・scratch | 0/3 | 3 | 0 | — |

未知seven D4を発見したものと同じ、既に固定済みのprojected_waterモデルをnine D3/D4へ適用する。five/seven D2/D3とwaterの経験は含むが、nineの経験もD4の教師も含まない。新たにD4教師を追加したsequential_nineとは別の検証である。

## 9節点の平均行動と探索乱数の追試

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| crossed_nine_source5601 / nine_d4 | projected・frozen | 0/2 | 2 | 0 | — |
| crossed_nine_source5602 / nine_d4 | projected・frozen | 0/2 | 2 | 0 | — |
| crossed_nine_source5603 / nine_d4 | projected・frozen | 0/2 | 2 | 0 | — |
| projected_nine_zero / nine_d3 | projected・deterministic | 0/3 | 0 | 0 | — |
| projected_nine_zero / nine_d4 | projected・deterministic | 0/3 | 0 | 0 | — |
| projected_nine_zero_control / nine_d3 | projected・untrained_deterministic | 0/3 | 0 | 0 | — |
| projected_nine_zero_control / nine_d4 | projected・untrained_deterministic | 0/3 | 0 | 0 | — |

nine D3/D4の平均行動は各3個の学習済みNN・未学習NNとも未成功だった。9節点D4の乱数交差比較はprojected_nineの対応3条件を再利用し、残り6組を追加する。重みの選び直しはしない。

## 同じ教師データ・更新数でgraphと比較

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| d4_graph_matched / seven_d4 | graph・frozen | 1/3 | 2 | 0 | 5786 |
| d4_graph_matched / seven_d4 | graph・scratch | 0/3 | 3 | 0 | — |

graph_waterの最初の3個の固定済みNNを、projected_waterと同じ教師データ・20,000更新・対応する探索乱数・6,400提案で評価する。平均行動だけのgraph対照では、確率的探索の表現比較にならないため追加した。両者は表現と係数の有界化が異なるので、差を基底不変性だけに帰属させない。

## 9節点でも同じ経験のgraph方策と比較

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| nine_graph_matched / nine_d4 | graph・frozen | 0/3 | 3 | 0 | — |

graph_waterの最初の3 source seedsをnine D4へ適用する。projected_nineと教師データ・20,000更新・探索乱数を揃える。未学習graph対照は同じ条件で実行済みのsequential_nine_controlを再利用する。新しいD4教師を含むgraph_seven_d4とは別モデルである。

## D2/D3だけの経験に限定する

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| projected_bars_transfer / nine_d4 | projected・frozen | 0/3 | 3 | 0 | — |
| projected_bars_transfer / seven_d4 | projected・finetune | 3/3 | 0 | 0 | 673, 565, 281 |
| projected_bars_transfer / seven_d4 | projected・frozen | 3/3 | 0 | 0 | 673, 565, 281 |
| projected_bars_zero / nine_d4 | projected・deterministic | 0/3 | 0 | 0 | — |
| projected_bars_zero / seven_d4 | projected・deterministic | 0/3 | 0 | 0 | — |

最初のprojectedの正の結果には既知water D1/√2の経験も含まれる。そのため、同じfive/sevenの教師データだけを残し、D2/D3以外の教師を除いたprojected_barsを新規に3個、各20,000更新で学習する。全モデルの学習とsource検証損失での選択を終えてから、未知seven D4とnine D4を評価する。scratch対照は同じprojected actor・初期形状・探索乱数なので、D4_common_6400とprojected_nineの既存結果を再利用する。geometryの抽選対象が3種類から2種類へ変わるため、water追加の有無だけでなく各barデータの抽選頻度も変わる。

## D2/D3だけの経験で探索乱数を入れ替える

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| crossed_bars_d4_source5601 / seven_d4 | projected・frozen | 2/2 | 0 | 0 | 596, 620 |
| crossed_bars_d4_source5602 / seven_d4 | projected・frozen | 2/2 | 0 | 0 | 1063, 266 |
| crossed_bars_d4_source5603 / seven_d4 | projected・frozen | 2/2 | 0 | 0 | 209, 214 |

最初の2個のD2/D3-only NNで673・565提案の成功を観測した後、固定済みの全3個と3個の探索乱数を交差させる追試を追加した。元の対応3条件は再利用し、残り6組のみを実行する。教師・checkpointは変更しない。終了期限までの観測を報告し、未着手を失敗として数えない。

![D2/D3のみの経験での乱数交差比較](bars_crossed_discovery.png)

## 新しく生成したseven初期形状からD4を初回発見する

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| fresh_d4 / seven_d4 | projected・frozen | 4/4 | 0 | 0 | 204, 626, 578, 861 |
| fresh_d4 / seven_d4 | projected・scratch | 0/4 | 4 | 0 | — |
| gradient_fresh_d4 / seven_d4 | analytic・gradient_w1p0 | 2/4 | 0 | 0 | 39, 55 |

最初のsource seed5601のD2/D3-only NNを固定してから、乱数69701で4個の可動初期形状を生成した。元の初期形状と、sevenの全教師状態から正規化距離0.03以上離れたものを使用し、NNの成否で初期形状を選ばない。同じ4形状をfrozen・scratch（各6,400提案）・幾何学的勾配w=1（400提案）に与える。

これは新しい初期条件の問題に対する比較で、元の初期形状からの探索に無料の事前歩行を付けた結果ではない。初期条件の生成は両手法共通の準備費用であり、オンライン提案数には含めない。4条件は1個の固定NNに対する初期形状の追試である。

![新しいseven初期形状からD4へ](fresh_d4_shapes.png)

![新しいseven初期形状の初回発見曲線](fresh_d4_first_discovery.png)

| 新しい初期形状 | 固定NN（予算6,400） | scratch RL（予算6,400） | 勾配法（予算400） |
|---|---:|---:|---:|
| 1 | 204 | 未成功（6400提案） | 未成功（44提案） |
| 2 | 626 | 未成功（6400提案） | 39 |
| 3 | 578 | 未成功（6400提案） | 55 |
| 4 | 861 | 未成功（6400提案） | 未成功（92提案） |

共通400提案までの成功数（全4条件が保存された手法のみ）: {'frozen': 1, 'scratch': 0, 'gradient_w1p0': 2}。勾配法は成功した初期形状でNNより速い場合がある。異なる予算の成功率からNNの一律な優位を主張しない。

## 既知の面折り条件をtorus探索の初期方策にする

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| nearest_prior_torus / torus_d2 | projected・finetune | 0/3 | 3 | 0 | — |
| nearest_prior_torus / torus_d2 | projected・frozen | 0/3 | 3 | 0 | — |
| nearest_prior_torus / torus_d2 | projected・scratch | 0/3 | 3 | 0 | — |

D4の対照で既知D3の方策も有効だったため、「同じ平坦化次元で、目標Dに最も近い既知経験を方策の初期条件にする」という規則を試す。torus D2では既知waterbombのD=√2をNNに渡す。環境の目標・報酬・成功判定はD2のまま。scratchにも同じ入力条件を与える。これは経験を探索に利用する手法で、NNが未知の数値Dに直接追従するという主張とは分ける。

## 必要な広がりの変化量で既知経験を選ぶ

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| delta_prior_torus / torus_d2 | projected・frozen | 0/3 | 3 | 0 | — |
| delta_prior_torus / torus_d2 | projected・scratch | 0/3 | 3 | 0 | — |

既知waterの初期広がりは約1.347、未知torusは約2.414。torus D2は約0.414の縮小を必要とするため、既知のD1と√2のうち、初期値からの変化量が近いD1を選ぶ。NN入力はD1、環境の目標・報酬・成功判定はD2のまま。全3 source seedsと対応scratchを6,400提案で比較する探索的な補足検証であり、既知の目標座標を未知構造の教師へ変換する操作ではない。

## ランダムな初期探索を含めたtorus初回発見

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| restart_torus / torus_d2 | graph・deterministic | 0/3 | 3 | 0 | — |
| restart_torus / torus_d2 | graph・untrained_deterministic | 0/3 | 3 | 0 | — |
| restart_torus / torus_d2 | projected・deterministic | 0/3 | 3 | 0 | — |
| restart_torus / torus_d2 | projected・untrained_deterministic | 0/3 | 3 | 0 | — |

各試行を元のtorus初期形状から開始し、40提案のランダム探索後にNNの平均行動を最大200提案行う。失敗時は同じ元形状から再試行し、合計6,400提案で打ち切る。初期探索・棄却提案も全て数える。未学習NNにも同じ乱数の初期探索を与える。重みは更新しない。これは初期形状の生成コストを除外した評価ではなく、探索手順を変えた補足比較である。

## 経験を使わない幾何学的勾配法

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| gradient_controls / five_d2 | analytic・gradient_w0p0 | 1/1 | 0 | 0 | 49 |
| gradient_controls / five_d2 | analytic・gradient_w0p1 | 1/1 | 0 | 0 | 49 |
| gradient_controls / five_d2 | analytic・gradient_w10p0 | 1/1 | 0 | 0 | 49 |
| gradient_controls / five_d2 | analytic・gradient_w1p0 | 1/1 | 0 | 0 | 49 |
| gradient_controls / five_d3 | analytic・gradient_w0p0 | 0/1 | 0 | 0 | — |
| gradient_controls / five_d3 | analytic・gradient_w0p1 | 0/1 | 0 | 0 | — |
| gradient_controls / five_d3 | analytic・gradient_w10p0 | 0/1 | 0 | 0 | — |
| gradient_controls / five_d3 | analytic・gradient_w1p0 | 0/1 | 0 | 0 | — |
| gradient_controls / nine_d3 | analytic・gradient_w0p0 | 0/1 | 0 | 0 | — |
| gradient_controls / nine_d3 | analytic・gradient_w0p1 | 0/1 | 0 | 0 | — |
| gradient_controls / nine_d3 | analytic・gradient_w10p0 | 0/1 | 0 | 0 | — |
| gradient_controls / nine_d3 | analytic・gradient_w1p0 | 0/1 | 0 | 0 | — |
| gradient_controls / seven_d4 | analytic・gradient_w0p0 | 0/1 | 0 | 0 | — |
| gradient_controls / seven_d4 | analytic・gradient_w0p1 | 0/1 | 0 | 0 | — |
| gradient_controls / seven_d4 | analytic・gradient_w10p0 | 0/1 | 0 | 0 | — |
| gradient_controls / seven_d4 | analytic・gradient_w1p0 | 0/1 | 0 | 0 | — |
| gradient_controls / torus_d2 | analytic・gradient_w0p0 | 0/1 | 0 | 0 | — |
| gradient_controls / torus_d2 | analytic・gradient_w0p1 | 0/1 | 0 | 0 | — |
| gradient_controls / torus_d2 | analytic・gradient_w10p0 | 0/1 | 0 | 0 | — |
| gradient_controls / torus_d2 | analytic・gradient_w1p0 | 0/1 | 0 | 0 | — |
| gradient_fresh_bar / five_d3 | analytic・gradient_w1p0 | 7/8 | 0 | 0 | 23, 22, 31, 28, 26, 28, 28 |

目的関数は初期値で正規化したflatness＋w(D誤差)²。モード方向の中心差分から局所勾配を計算し、交差棄却時は次の幅を縮める。w=0, 0.1, 1, 10を全て報告する。各設定は決定論的な1試行で、独立乱数4試行ではない。NNも過去の折り経験も目的座標も使わない。有限差分の目的関数評価数もresult.jsonに記録した。

基準five D3では4設定とも失敗した。一方、同じ新しい初期形状8個でw=1を固定すると7/8が22～31提案で成功し、成功した条件ではNNより速かった。NNはこの8形状全てで成功した。この結果からNNが単純な勾配法に常に勝るとは言えず、初期形状と到達する折れ方の違いを評価する必要がある。

この勾配法は1経路400提案（torus200）の短い参照条件であり、最適化手法を十分に調整した比較ではない。RLの6,400提案評価と予算が異なるため、8形状の成功率を同予算で見る場合は400提案時点で比較する。その時点では経験ありNN8/8、未学習graph RL2/8、勾配法7/8だった。

## 今回のRL初回発見を、次の未知構造へ経験として移す

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| sequential_nine / nine_d3 | graph・finetune | 0/3 | 3 | 0 | — |
| sequential_nine / nine_d3 | graph・frozen | 0/3 | 3 | 0 | — |
| sequential_nine / nine_d4 | graph・finetune | 0/3 | 3 | 0 | — |
| sequential_nine / nine_d4 | graph・frozen | 0/3 | 3 | 0 | — |
| sequential_nine_control / nine_d4 | graph・scratch | 0/3 | 3 | 0 | — |
| sequential_nine_old / nine_d4 | graph・frozen | 0/3 | 3 | 0 | — |
| sequential_nine_zero / nine_d3 | graph・deterministic | 0/3 | 0 | 0 | — |
| sequential_nine_zero / nine_d4 | graph・deterministic | 0/3 | 0 | 0 | — |

seven D4のscratch seed5704が18,751提案で初回発見した座標を、新たな既知目標として使う。基準形状＋32初期形状からその座標への教師経路を作り、元のseven D2/D3データにD4教師を追加して、fresh actorを20,000更新する。最初の3 source seedsを使い、選択はsource検証損失のみ。

33教師経路のうち19経路が成功し、1,787ラベルを追加した。D2/D3と合わせて8,245ラベル。既知seven D4の平均行動は3個の新規NNのうち2個で成功したが、nine D3/D4のfrozen/finetuneは全12条件で6,400提案まで未発見だった。既知のD4経験を追加するだけで今回のgraph actorの未知構造転移が改善する、とは示せなかった。

未知課題はnine D3/D4。sequential_nineは新しいD4経験を含むNN、sequential_nine_oldは以前のD2/D3だけのNN、sequential_nine_controlは未学習RL。nine D3の旧NN・scratch対照は既存のnine_transferと同一条件なので、前掲結果を再利用する。nineの目標座標・教師は一切生成しない。

## 方向と符号付き幅の結合診断

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| aligned_dense / seven_d4 | dense・deterministic | 0/5 | 0 | 0 | — |
| aligned_graph_coverage / seven_d4 | graph・deterministic | 0/5 | 0 | 0 | — |
| aligned_graph_coverage / torus_d2 | graph・deterministic | 0/5 | 0 | 0 | — |

## 既知課題の習得状況

| 系列 / 課題 | 方策 | 成功 / 終了試行 | 予算到達失敗 | 時間打切り | 初回発見の提案数 |
|---|---|---:|---:|---:|---|
| source_graph_coverage / five_d2 | graph・deterministic | 3/5 | 0 | 0 | 45, 51, 50 |
| source_graph_coverage / five_d3 | graph・deterministic | 5/5 | 0 | 0 | 30, 30, 30, 30, 30 |
| source_graph_coverage / seven_d2 | graph・deterministic | 1/5 | 0 | 0 | 47 |
| source_graph_coverage / seven_d3 | graph・deterministic | 5/5 | 0 | 0 | 47, 47, 49, 48, 47 |
| source_graph_long / five_d2 | graph・deterministic | 2/5 | 0 | 0 | 50, 52 |
| source_graph_long / five_d3 | graph・deterministic | 5/5 | 0 | 0 | 30, 30, 30, 30, 30 |
| source_graph_long / seven_d2 | graph・deterministic | 1/5 | 0 | 0 | 52 |
| source_graph_long / seven_d3 | graph・deterministic | 0/5 | 0 | 0 | — |
| source_graph_seven / seven_d2 | graph・deterministic | 0/3 | 0 | 0 | — |
| source_graph_seven / seven_d3 | graph・deterministic | 3/3 | 0 | 0 | 48, 47, 48 |
| source_graph_water / five_d2 | graph・deterministic | 3/5 | 0 | 0 | 49, 50, 49 |
| source_graph_water / five_d3 | graph・deterministic | 5/5 | 0 | 0 | 30, 30, 30, 30, 30 |
| source_graph_water / seven_d2 | graph・deterministic | 1/5 | 0 | 0 | 49 |
| source_graph_water / seven_d3 | graph・deterministic | 4/5 | 0 | 0 | 48, 48, 50, 47 |
| source_graph_water / water_d1 | graph・deterministic | 0/5 | 0 | 0 | — |
| source_graph_water / water_sqrt2 | graph・deterministic | 5/5 | 0 | 0 | 6, 6, 10, 6, 7 |
| source_projected_bars / five_d2 | projected・deterministic | 3/3 | 0 | 0 | 50, 51, 49 |
| source_projected_bars / five_d3 | projected・deterministic | 3/3 | 0 | 0 | 30, 30, 30 |
| source_projected_bars / seven_d2 | projected・deterministic | 0/3 | 0 | 0 | — |
| source_projected_bars / seven_d3 | projected・deterministic | 3/3 | 0 | 0 | 48, 47, 48 |
| source_projected_water / five_d2 | projected・deterministic | 2/3 | 0 | 0 | 49, 51 |
| source_projected_water / five_d3 | projected・deterministic | 2/3 | 0 | 0 | 30, 30 |
| source_projected_water / seven_d2 | projected・deterministic | 0/3 | 0 | 0 | — |
| source_projected_water / seven_d3 | projected・deterministic | 3/3 | 0 | 0 | 47, 47, 48 |
| source_projected_water / water_d1 | projected・deterministic | 0/3 | 0 | 0 | — |
| source_projected_water / water_sqrt2 | projected・deterministic | 2/3 | 0 | 0 | 6, 6 |
| source_reference / five_d2 | graph・deterministic | 2/5 | 0 | 0 | 51, 40 |
| source_reference / five_d3 | graph・deterministic | 3/5 | 0 | 0 | 31, 31, 30 |
| source_reference / seven_d2 | dense・deterministic | 3/5 | 0 | 0 | 48, 49, 48 |
| source_reference / seven_d2 | graph・deterministic | 0/5 | 0 | 0 | — |
| source_reference / seven_d3 | dense・deterministic | 5/5 | 0 | 0 | 49, 49, 48, 47, 49 |
| source_reference / seven_d3 | graph・deterministic | 0/5 | 0 | 0 | — |
| source_seven_discovered_d4 / seven_d2 | graph・deterministic | 0/3 | 0 | 0 | — |
| source_seven_discovered_d4 / seven_d3 | graph・deterministic | 3/3 | 0 | 0 | 48, 48, 47 |
| source_seven_discovered_d4 / seven_d4 | graph・deterministic | 2/3 | 0 | 0 | 62, 61 |

graph_longは元データで20,000更新、graph_coverageは既知課題でNNが訪れた状態を追加して同じ20,000更新。未知課題の結果を使わず、既知基準形状20条件での成功数（8/20対14/20）からgraph_coverageを選択した。

seven→fiveの専用モデルgraph_sevenは、元のseven教師とsevenだけで学習したdense actorの既知状態を追加した。five経験を持つcollectorを使っていない。source D3は3/3で成功したがD2は0/3であり、「D2も完全に習得した」とは言えない。

waterbombは16個の可動な初期状態と基準初期状態からD1・√2への34教師経路を生成し、33経路が成功。保存RL経路の状態も一部利用し、936ラベルを追加した。これは既知課題のみの教師追加。graph_waterは基準water √2で5/5成功する一方D1は0/5だった。

## 失敗の観察と実装上の検証

rollout_diagnostics.csvは各条件の最初の1エピソードを再現して、交差棄却・制約補正失敗・最終flatness/D誤差を記録した診断である。既存試行の再生を追加の独立試行とは数えない。sevenの一部では交差棄却の繰返しとD3付近への接近、torusではD誤差が減ってもflatnessが大きい状態を観測した。これだけで原因を断定しない。

符号付き幅が教師の有向平均を反転し得るため、幅平均を有向平均に整合させる別方策をゼロショットで試したが、D4/torusの成功はなかった。主比較の方策は変更していない。projectedは節点の移動場をモードへ投影する別表現で、モード基底の直交変換に対する物理平均・共分散の不変性をテストした。

同じ初期状態でモード基底だけを100回ランダム直交変換した診断をbasis_sensitivity.csvに保存した。graph_water seed5601のseven D4では変換前後の物理平均方向の平均cos類似度が約0.53だった。projectedは数値誤差の範囲で不変だった。これは表現の性質の検査であり、未知課題の成功を直接示す指標ではない。

![同じ教師データでの表現比較](representation_learning.png)

テストは従来rollout＋更新との重み一致、教師を読まない未知課題実行、教師あり学習がcritic/分散を変更しないこと、中断再開と連続実行の一致などを確認する。詳細はtest_results.txt。成功経路とsourceハッシュ、固定actorの一致はaudit.json。

混合経験によるseven D4の612提案・nine D4の1,583提案、およびD2/D3-onlyによるseven D4の673提案の条件を別の出力先で再実行し、提案数と成功経路が一致するかも確認する（reproduction_status.json）。この再生は新しい独立試行に数えない。

## 保存物と再現

- 全試行: all_results.csv / aggregate.json
- 主比較の曲線: first_discovery.png / terminal_progress.png
- 教師学習履歴: source/*/learning.csv / source_learning.png
- 成功モデル: source/graph_seven_seed5601/best.pt（5602、5603も保存）
- 未知D4・未知9節点のモデル: source/projected_water_seed5601/best.pt、seed5602・5603も保存
- D2/D3のみのモデル: source/projected_bars_seed5601/best.pt、seed5602・5603も保存
- 成功経路: bar_transfer/*/first_hit_path.npz、fresh_bar/*/first_hit_path.npz
- D4/9節点の成功経路: projected_transfer/*/first_hit_path.npz、projected_nine/*/first_hit_path.npz
- 動画: bar_transfer_example.gif
- 未知課題の動画: unseen_d4_example.gif、unseen_nine_example.gif
- 入力・データの由来: seven_only_source/manifest.json、fresh_bar_inputs/manifest.json、water_source/manifest.json
- [再実行の手順](REPRODUCE_JA.md)。今回の期限・出力先を変更して実行する。
- [実装の説明](METHOD_JA.md)。教師・NN入力・従来RLとの接続を記載。

## 次に主張を強くするための検証

今回の正の結果は、小規模linkage間の経験転移が初回発見を短縮し得ることを支持する。次は構造を先に固定したleave-one-structure-out評価を複数構造・複数source seedで行い、経験なしと経験ありで同じ探索予算を比較する。未知Dと未知構造は別軸で報告し、未知課題の教師による適応を混ぜない。torusについては、まず既知の面折り課題を十分習得できる表現と教師経路の範囲を確認し、その後に予算を増やした初回発見比較を行う。

## 既知データからの事前学習コスト

以下は今回の各事前学習ループの平均実時間。教師経路の生成、以前のRLによる既知形状発見、起動・データ読み込みのコストは含まない。並行実行中の値なので専有CPUでの速度としては解釈しない。

| 教師モデル | 完了モデル数 | 平均更新数 | 平均学習秒 | 平均最良source検証損失 |
|---|---:|---:|---:|---:|
| dense | 5 | 5000 | 28.7 | 0.03455 |
| graph | 5 | 5000 | 55.5 | 0.10495 |
| graph_coverage | 5 | 20000 | 268.3 | 0.04154 |
| graph_long | 5 | 20000 | 260.5 | 0.03938 |
| graph_seven | 3 | 20000 | 305.7 | 0.01767 |
| graph_seven_d4 | 3 | 20000 | 334.3 | 0.02675 |
| graph_water | 5 | 20000 | 292.9 | 0.05887 |
| projected_bars | 3 | 20000 | 309.6 | 0.02131 |
| projected_water | 3 | 20000 | 276.3 | 0.03076 |
