# 小摂動による6終端cellの可到達性と多段摂動探索

## 結論

現在の平面内衝突判定とcorrector前摂動ノルム0.20の下でも，five-bar で
従来確認していた6終端cellすべてへ到達する有効経路は存在する．したがって，
探索空間やdescriptorが6 cellを排除しているわけではない．後述する同条件の
比較は，通常のランダム探索が2 cellで停滞する一因が，複数の小摂動を一貫した
方向で連鎖させる状態・作用へ計算予算を配分できないことにあると示唆する．

同じ物理的方向を更新後の接空間へ射影し直して2回連続で加える
persistent-direction emitter は，摂動ノルム0.15と0.20の両方で6 cellを
生成した．さらに，同一version・同一seed・同一候補生成規則で10000 stepを
比較すると，ランダム選択は3/3 seedとも2 cellであったのに対し，GPを用いない
transport heuristicは3/3 seedで従来の6 canonical cellを回収した．1 seedでは
同一平面形状の別重なり順も得られ，archive上は7 cellとなった．これは
「連続摂動親を選ぶこと」と「前回方向をtransportすること」を組み合わせた
複合priorの有効性を示すが，方向transport単独の因果効果ではない．
GP-UCB-like scoreへ同じ方向priorを掛けた条件は5, 6, 4 archive cellを得たが，
canonical被覆は5, 4, 4であった．したがってGP付き方策は一様randomより有望でも，
現状のfive-barではGPなしheuristicを上回らず，終端予測の追加効果は確認できない．

## 共通の物理・数値条件

| 項目 | 値 |
|---|---:|
| deterministic predictor変位の公称最大ノルム | 0.10 |
| 通常探索のcorrector前摂動ノルム | 0.20 |
| flatness tolerance | 0.05 |
| 衝突判定 | `collision2.py` の連続平面内交差 |
| rollout上限 | 100 mechanics step |
| elite RMS / max tolerance | 0.02 / 0.05 |
| main5の親候補上限 | 64 |
| 一親あたりの通常方向数 | 16 |

発見AUCは

\[
\operatorname{AUC}
=
\frac{1}{T}\int_0^T |A_t|\,dt
\]

であり，予算中の平均累積cell数を表す．

## 1. 保存経路を用いた小摂動可到達性の監査

過去の二つの探索runに保存された代表経路を，現在の終端descriptorと
`collision2.py`で再判定した．8本のwitnessは従来確認していた6 cellへ分類され，
全edgeが現在の衝突判定を通過した．

| 診断量 | 結果 |
|---|---:|
| 保存witness数 | 8 |
| 固有終端cell数 | 6 |
| 衝突edge数 | 0 |
| corrected step normの最大値 | 0.199384 |
| `.dat`丸め後の最大拘束残差 | \(1.397\times10^{-8}\) |

旧search tree metadataから復元した6 cellの初回発見経路の摂動数は

\[
[1,2,2,3,2,3]
\]

であった．ただし，この初回発見経路の座標自体は保存されていないため，現在の
collision判定で再検査した値ではない．現在の判定で実際に再検査した8本の
保存代表経路はすべて \(K\geq2\) であった．いずれも1摂動で到達できないことの
数学的証明ではない．監査結果は
`result/reachability_audit/reachability_report.md` に保存した．

## 2. 大きいroot摂動による対照

rootへノルム0.30の摂動を一度だけ与え，以後を決定論的にrolloutした．現在の
descriptorと衝突判定の下で，5/5 seedが5000 step以内に6 cellへ到達した．

| seed | 6 cell目の発見step |
|---:|---:|
| 0 | 3197 |
| 1 | 1099 |
| 2 | 1504 |
| 3 | 1314 |
| 4 | 1712 |

6 cell目の中央値は1504 stepであり，3200 stepでは5/5 seedが6 cellへ到達した．
これはdescriptor・collision処理が6 cellを排除していないことの対照である．
ただしノルム0.30は求める経路連続性の条件と異なるため，ノルム0.20の探索性能
との直接比較には用いない．

## 3. 現行ランダム探索の大規模試行

ノルム0.20，予算10000 step，seed 300--305で，現在の通常候補生成と
ランダム選択を実行した．

| seed | 最終cell数 | AUC | rollout数 | 追加cellの主な発見 |
|---:|---:|---:|---:|---|
| 300 | 2 | 1.9800 | 314 | step 137, \(K=2\) |
| 301 | 3 | 1.9976 | 299 | step 179, \(K=1\); step 9782, \(K=4\) |
| 302 | 3 | 2.3023 | 352 | step 116, \(K=1\); step 6798, \(K=5\) |
| 303 | 2 | 1.9561 | 301 | step 376, \(K=1\) |
| 304 | 2 | 1.9386 | 320 | step 551, \(K=1\) |
| 305 | 2 | 1.9369 | 310 | step 568, \(K=1\) |

平均は2.33 cellで，6 cellへ到達したseedはなかった．seed 302を30000 stepへ
延長しても3 cellのままであった．このrunは1009 rollout，flat 594，collision
414で，木の最大摂動深さは10である．したがって，単に深い枝が存在するだけでは
十分ではなく，有効な方向相関を持つ摂動系列が重要であることを示唆する．

## 4. Persistent-direction emitterによる到達性診断

rootで選んだ物理方向を，各小摂動後の状態の接空間へ再射影し，規定ノルムへ
正規化した．各micro-stepでcorrectorとcollision判定を行った後，通常の
決定論的rolloutを実行する．終端targetは候補方向の生成・順位付けには使わない．

### 72方向

| 摂動ノルム | 連続回数 \(K\) | 実行方向 | mechanics step | cell数 |
|---:|---:|---:|---:|---:|
| 0.15 | 1 | 72 | 3210 | 2 |
| 0.15 | 2 | 57 | 2508 | 6 |
| 0.20 | 1 | 72 | 2891 | 2 |
| 0.20 | 2 | 59 | 2554 | 6 |

### 現行main5と同じ方向数・等角grid規則の16方向

| 摂動ノルム | \(K\) | 実行方向 | mechanics step | cell数 |
|---:|---:|---:|---:|---:|
| 0.15 | 1 | 16 | 761 | 2 |
| 0.15 | 2 | 13 | 610 | 6 |
| 0.20 | 1 | 16 | 638 | 2 |
| 0.20 | 2 | 16 | 653 | 5 |

72方向の \(K=2\) で最初に得た6 cellは，監査で同定した従来の6 cellと
完全に一致した．corrector後の最大摂動ノルムも0.15条件で0.149546，
0.20条件で0.198977であり，上限を満たした．

方向生成と順位付けには終端descriptorを用いていないが，既知cell数6を
停止条件には用いている．したがって7 cell目以降が存在しないことを示す
実験ではない．また，独立な2回摂動との直接比較ではないため，方向相関だけの
因果効果も分離していない．この結果は，同じ物理方向を局所接空間間で運ぶ
\(K=2\) emitterが6 cell到達に十分であることを示し，方向継続の重要性を示唆する．
なお，診断scriptは固定phase，main5は反復ごとにrandom phaseを使うため，
16方向の具体的な向きまで同一ではない．

## 5. main5へ実装した汎用transport emitter

現在状態の接空間基底を \(B(q)\)，直前のcorrector前物理摂動を
\(v_{\mathrm{prev}}\) とする．前方向の射影と規格化候補を

\[
\widetilde v
=B(q)B(q)^\dagger v_{\mathrm{prev}},
\qquad
v_{\mathrm{tr}}
=\rho\frac{\widetilde v}{\|\widetilde v\|}
\]

と定義した．これは基底を \(B(q)R\) へ符号反転・直交回転しても不変である．
候補数が増えないよう，通常gridで最も近い1方向をtransport候補で置換する．
transport候補が既評価で除外された場合は，置換していた通常方向を復元する．
射影率 \(\|\widetilde v\|/\|v_{\mathrm{prev}}\|\) が閾値未満ならtransport
候補を作らない．

さらに各候補 \(v_i\) に

\[
c_i=\max\!\left(
0,
\frac{v_i^T\widetilde v}{\|v_i\|\|\widetilde v\|}
\right)
\]

を持たせ，GPのguided scoreへ

\[
\alpha_i'
=
\alpha_i(1+\gamma c_i)
\]

という方向整合の選択圧を与えられるようにした．実装では既存のdirect-parent
bonusとの積で \(\alpha_i(1+\gamma_d I_{\rm direct})(1+\gamma c_i)\) とする．

射影 \(B(q)B(q)^\dagger\) は，SVD基底の符号反転と基底内直交回転に対して
数学的に不変である．現在の `calc_mode` は直交SVD基底を返す．

## 6. 現行v3.1の3000-step比較

seed 400--402，corrector前摂動ノルム0.20で比較した．すべて
`main5_gp_v3_1_transport_restore` であり，親候補上限64，direct-parent最低16，
root保持，transport候補ありという候補生成規則を共通にした．randomと
transport heuristicは10000-step runの先頭3000 stepを同じ階段積分で評価し，
GP二条件は3000 stepで独立に終了したrunである．

| 選択規則 | cell数/seed | 平均cell | 平均AUC |
|---|---|---:|---:|
| 候補から一様random | 2, 2, 2 | 2.00 | 1.908 |
| GP-UCB-like，alignment bonus 0 | 2, 2, 2 | 2.00 | 1.788 |
| GP-UCB-like，alignment bonus 1 | 4, 2, 4 | 3.33 | 2.532 |
| transport heuristic，GPなし | 3, 6, 4 | 4.33 | 3.255 |

alignment bonus 0から1への変更により，GP runで選ばれたtransport候補は
一run平均9.33回から19.0回へ，最大連続摂動長は7.67から16.0へ増えた．
したがって，GP scoreへ方向持続priorを加えることは探索木を実際に変えている．
一方，GPを使わずtransport候補をhardに80%選ぶheuristicの方がcell数とAUCは
高い．この時点では，終端GPよりも方向持続priorが支配的である．

## 7. 同一条件での10000-step延長

同じversion，seed，候補生成規則，mechanics-step予算で，random，GPなし
transport heuristic，GP-UCB-like＋alignment bonus 1を10000 stepまで比較した．
方策が異なれば生成historyも分岐するため，実際に現れる候補そのものは途中から
同一ではないが，候補を作る規則と乱数seedは揃えている．

| 選択規則 | archive cell/seed | canonical被覆/seed | 平均AUC | canonical 6到達step | 平均wall time |
|---|---|---|---:|---|---:|
| 一様random | 2, 2, 2 | 2, 2, 2 | 1.972 | 未達, 未達, 未達 | 120.17 s |
| transport heuristic，GPなし | 6, 7, 6 | 6, 6, 6 | 4.870 | 5370, 2052, 6495 | 119.97 s |
| GP-UCB-like＋alignment bonus 1 | 5, 6, 4 | 5, 4, 4 | 3.761 | 未達, 未達, 未達 | 250.47 s |

GP付き探索はrandomより多くのarchive cellを得たが，その差にはalignment bonusも
含まれるため，終端予測だけの効果とは言えない．またseed 401の6 archive cellは
canonical 4 cellとcanonical集合外の重なり順variant 2 cellから成り，従来の
canonical 6を完遂したわけではない．対してGPなしheuristicは3/3 seedでcanonical
6を回収した．

選択されたtransport候補は一run平均でrandom 6.0回，GP 52.0回，GPなし
heuristic 273.7回であり，最大連続摂動長はそれぞれ5.33，20.33，24.67であった．
この結果は「摂動直後の親を選ぶこと」と「前回の物理方向を現在の接空間へ
transportすること」を組み合わせた複合方策が有効であることを示す．ただし
親選択と方向選択を同時に変えているため，方向transport単独の因果効果ではない．
またseed数は3であり，five-bar以外への一般優位を示す結果でもない．

### canonical集合外の追加variant

GPなしheuristicのseed 401では，canonical 6をstep 2052で揃えた後，step 7081で
canonical集合外の7番目のarchive cellを発見した．これはouter-left終端とxy座標が
ほぼ同一である一方，隣接する部材0と1の重なり順が逆であった．

| cell | layer key | 最後の非平坦状態での正規化z分離 |
|---|---|---:|
| 追加variant | `((0,1,0),)` | +0.026888 |
| canonical variant | `((0,1,1),)` | -0.027900 |

追加variantの保存代表経路は現行collision判定で衝突0，corrector後最大step norm
0.199552，保存座標の最大拘束残差 \(1.42\times10^{-8}\) であった．したがって，
単なる座標距離の過分割とは考えにくい．「理論上6」が平面形状だけの分類なら，
重なり順も区別する現在の研究目的では6を超えるcellが存在してよい．一方，
理論上6が隣接部材の上下まで含む主張なら，理論的列挙または隣接部材の
layer定義を再確認する必要がある．

## 8. 現時点での研究上の判断

現時点で支持される主張は次である．

> 有界な小摂動と現在の衝突条件の下でも6終端cellへ到達できる．候補集合から
> 一様に状態・作用を選ぶと，有効な連続摂動系列へ予算が配分されにくい．
> 摂動直後の親を保持し，直前の物理方向を新しい接空間へtransportして選ぶ
> 複合方策は，SVD基底に依存せず，five-barの全canonical 6 cellを生成できる．

まだ支持されない主張は次である．

- GPがrandomまたはtransport heuristicより一般に高効率である．
- direct-parent bonusだけで新規cell発見が増える．
- \(K=2\) が他構造でも適切である．
- 有限予算で6 cellすべてを発見する保証がある．

同一候補規則・seed・mechanics budgetの比較まで完了したが，GPなしheuristicは
hardなtransport quota，GP版はsoftな獲得関数であり，終端予測だけの因果効果は
まだ完全には分離していない．次段階では，同じ方向priorを持つGP-free scoreとの
比較，別構造での再現，固定action catalogueまたはbranch別quotaによる初期方向の
保持を検討する．

## 再現用ファイル

- `root_reachability.py`: root大摂動の対照
- `five_bar_reachability_report.py`: 保存小摂動経路の再監査
- `five_bar_persistent_emitter.py`: target descriptorを使わないpersistent方向診断
- `five_bar_coverage_audit.py`: 保存cellのcanonical被覆と追加variantの監査
- `path_diagnostics.py`: 摂動深さ・direct/transport bonus
- `main5.py`: v3.1 transport候補，GP score，transport heuristic
- `result/reachability_audit/reachability_report.md`: 6 witness監査
- `result/persistent_direction_emitter/persistent_emitter_report.md`: 72方向診断
- `result/persistent_direction_emitter_16/persistent_emitter_report.md`: 16方向診断
- `result/gp_transport_pilot_v31_comparison.csv`: v3.1の3000-step GP比較
- `result/large_search_comparison_v31.csv`: v3.1の10000-step三手法集計
- `result/canonical_coverage_v31.csv`: archive cellとcanonical被覆の対応

## Provenance上の注意

過去の `pilot_v2_A_random_default_b3000` と
`pilot_v2_B_random_multistep_b3000` は環境変数の誤指定により実際にはGPであり，
現行表には用いていない．Section 3のseed 300--305と30000-step延長はv1の履歴的
baselineである．Section 6--7の主比較はすべて
`main5_gp_v3_1_transport_restore` で再実行した．v3.1では既評価transport候補を
除外した際に，置換されていた通常grid候補を復元するため，旧v3とは厳密には
同一trajectoryにならない．
