# K=2 摂動角選択の検証

## 1. 検証した問い

小さい摂動を一回だけ与える探索ではなく，二回の摂動を連続して与える
K=2 macro-action を考える．第1摂動後の状態を \(q_1\)，第1摂動を
\(q_1\) の接空間へ射影・正規化した方向を \(p\)，それと直交する物理方向を
\(u\) とし，第2摂動を

\[
v_2(\theta)
=
\rho\{\cos\theta\,p+\sin\theta\,u\}
\]

とした．この表現は SVD 零空間基底の符号反転と直交回転に依存しない．

比較した規則は次の通りである．

1. `independent_random`: \(\theta\) を16等角度から一様選択する．
2. `fixed_transport`: 80%で \(\theta=0\)，20%で一様選択する．
3. `fixed_reverse`: 80%で \(\theta=\pi\)，20%で一様選択する．
4. `learned_angle_gp`: GPのnovelty-UCB-like score上位25%から選び，20%は一様選択する．
5. `learned_angle_gp_shuffled`: 同じGP scoreを角度間で並べ替えた負対照である．
6. `learned_angle_gp_cost_aware`: 上記scoreを予測rollout costで割る．

全手法で親StateNodeと第1摂動 \(v_1\) の生成規則，角度候補，力学step予算，
archive判定を共通にした．GPの効果はGPとscore-shuffledのpaired差で判定する．

## 2. Offline paired counterfactual

同一の \(q_1\) について16角度をすべて実rolloutした．3 seed，54個の
\(q_1\) group，合計864 macroを用い，group単位でtrain/testを分離した．
新規性は決定論trunkだけを含む固定archive \(A_1\) に対して評価した．

### 真の角度依存性

| 角度規則 | 新規cell率 |
|---|---:|
| 一様random | 0.126 |
| \(\theta=0\) | 0.074 |
| \(\theta=\pi\) | 0.204 |
| group内oracle | 0.370 |

\(\cos\theta\) と新規cell指示値のSpearman相関は \(-0.135\) だった．
したがって，このfive-barのpanelでは「前回方向との類似度が高いほどよい」
とはいえない．K=2が有効であることと，exact transportが有効であることは
分けて扱う必要がある．

### 同一group内のGP順位

| モデル | top-1新規率 | top-25%平均新規率 | top-25%に新規を含むgroup率 |
|---|---:|---:|---:|
| raw GP \([q_1,v_2]\) | 0.130 | 0.167 | 0.259 |
| angle-aware GP | 0.222 | 0.204 | 0.278 |
| angle context shuffled | 0.148 | 0.134 | 0.185 |

この固定分布上では，angle-aware GPに状態依存の角度情報を拾う小さな正の兆候が
ある．一方，固定archive，trunk由来の親，3 seedだけの結果であり，閉ループで
archiveとhistoryが変化する場合の有効性を直接示すものではない．

## 3. Online 10,000-step比較

通常の16 rollout共通warm-up，seed 600--602の予備結果を示す．AUCは
\(10{,}000^{-1}\int_0^{10{,}000}C(s)\,ds\) であり，最大6では正規化していない．

| 手法 | 最終cell数 | discovery AUC | warm-up後incremental AUC |
|---|---:|---:|---:|
| independent random | 3.333 | 2.8255 | 1.2279 |
| fixed transport | 2.333 | 2.2285 | 0.6034 |
| fixed reverse | 2.000 | 1.9657 | 0.3284 |
| learned-angle GP | 2.000 | 1.9668 | 0.3296 |
| GP score shuffled | 2.333 | 2.0093 | 0.3740 |
| cost-aware GP | 2.333 | 2.2343 | 0.6094 |
| cost-aware shuffled | 2.000 | 1.9655 | 0.3282 |

通常GPからshuffledを引いたpaired差は，最終cell数 \(-0.333\)，AUC
\(-0.0425\) だった．現行のangle GP scoreには有効なonline選択圧を確認できない．
independent randomはseed 600だけで6 cellへstep 5464で到達したが，seed 601，602
では2 cellに留まった．このため3 seed平均の差は大きなseed依存性を含む．

cost-aware GPはcompleted second actionsを42.23から50.80回/1000 stepへ増やした．
cost-aware shuffledは44.57回/1000 stepであり，GPが短いrolloutを選ぶ情報は
持っている．しかしcell改善はseed 600の1 cellだけで，randomより多様な終端を
得たとはいえない．

## 4. 同一 \(q_1\) の高・中・低類似度を用いたwarm-up

通常warm-upは角度の周辺分布を均等にするだけで，同一 \(q_1\) では1角度しか
観測しなかった．そこで各 \(q_1\) から

\[
\theta\in\{0,\ \pm\pi/2,\ \pi\}
\]

の3枝を実行するpaired warm-upを追加した．16 group，48枝の全stepを予算に含め，
各seedで全手法のhistory，archive，終了stepのhashが一致することを確認した．

| 手法 | 最終cell数 | discovery AUC | warm-up後incremental AUC |
|---|---:|---:|---:|
| independent random | 2.333 | 2.1664 | 0.5889 |
| fixed transport | 2.333 | 2.1607 | 0.5826 |
| learned-angle GP | 2.333 | 2.1331 | 0.5517 |
| GP score shuffled | 2.333 | 2.1304 | 0.5487 |

GPからshuffledを引いたpaired差は，最終cell数0，AUC \(+0.0027\) である．
同じ \(q_1\) における角度contrastを明示的に教師化してもonline性能は改善しなかった．
したがって，失敗原因を単なる初期教師不足だけに帰すことはできない．

## 5. なぜofflineの角度順位がonline探索へ移らなかったか

主な原因は次のように整理できる．

1. **上流の候補が支配的である．** main6は親StateNodeと \(v_1\) を先にrandomで
   一つ決め，GPはその後の16角度だけを選ぶ．その \((q,v_1)\) に新規cellへ通じる
   角度が存在しなければ，完全なangle oracleでも改善できない．
2. **offlineとonlineの分布が異なる．** offlineはtrunk由来 \(q_1\) と固定
   archive \(A_1\) を用いる．onlineでは多重摂動を含むhistoryとarchiveが変化する．
3. **終端写像はbasin境界で不連続である．** 滑らかなGPの予測平均と大きな分散が，
   実際には既知basinへ行く候補をfalse noveltyとして高く評価する．
4. **UCBの不確実性は新規cell確率ではない．** 高分散はkernel距離上の未観測を
   表すが，未発見の物理basinを意味しない．seed 600で実際に新規cellを生んだ
   guided候補のscoreは1.041で，非新規候補平均4.413より低かった．
5. **角度固定則も分布依存である．** offlineでは \(\theta=\pi\) が良かったが，
   onlineのfixed reverseは全seed 2 cellだった．単一のglobal角度則は頑健でない．

## 6. 推奨する次のアルゴリズム

angleだけでなく，K=2 macro全体を候補とする．

\[
x=(q,\ v_1,\ w,\ \cos\theta,\ \sin\theta),
\]

ここで \(w\) は \(v_1\) と独立な物理参照方向であり，corrector後に接空間へ
射影して \(u\) を作る．この入力から \(q_1,p,u,v_2\) と終端は決定論的に定まる．
したがって，親位置，第1摂動，第2角度を同じjoint candidate pool内でGP filterへ
渡せる．SVD係数を入力しないため基底の符号・回転にも依存しない．

実用的な選択規則は次の通りである．

1. 全体の20%はjoint candidateから完全randomに選ぶ．
2. 残りは flat到達性，archive novelty，GP不確実性からUCB-like scoreを作る．
3. global argmaxではなく上位quantileをfilterとして残し，残存候補から抽選する．
4. rollout cost GPを用いる場合は，novelty scoreを予測costで割る．
5. score-shuffled filterを必ず負対照とし，filter構造とGP順位情報を分離する．
6. paired warm-upで高・中・低類似度を同一macro開始条件から取得する．

一方，既知basin分類器を直接使う場合は，未知cellを既知classとして誤分類して
永久に除外しないよう，高信頼のknown/collisionだけをrejectする保守的filterとし，
random bypassを残すべきである．

## 7. 現時点で主張できること

- small stepを二回連続させるK=2表現には探索上の価値がある．
- five-barでexact transportが常に最適という結果ではない．
- offlineでは状態依存角度を学ぶ余地が観測された．
- 現行の第2角度だけを選ぶGPはonlineでrandomを上回らなかった．
- cost GPは短いrolloutを選んで試行回数を増やしたが，多様性改善は限定的だった．
- 次の本命は角度GPの追加調整ではなく，親・第1摂動・第2角度のjoint macro選択である．

これらはfive-barと3 seedによる予備結果であり，構造一般の優越性ではない．
異なる構造へ同じ候補表現・予算・random比率を固定して適用し，未使用seedで
検証して初めて汎用的な有効性を論じられる．

## 8. 関連ファイル

- `macro2_angle_panel.py`: offline paired counterfactual
- `MACRO2_ANGLE_PANEL_RESULTS.md`: offline詳細
- `main6.py`: K=2角度選択とpaired warm-up，cost-aware selector
- `angle_macro.py`: 基底不変な物理方向transportと角度候補
- `main6_results_aggregate.py`: 保存結果の再集計・監査
- `MAIN6_ONLINE_AGGREGATE.md`: 通常warm-upのonline詳細
- `MAIN6_PAIRED_WARMUP_RESULTS.md`: paired warm-upのonline詳細

