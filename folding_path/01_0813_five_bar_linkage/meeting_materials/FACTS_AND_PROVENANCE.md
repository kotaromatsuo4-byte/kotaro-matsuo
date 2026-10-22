# 15枚ミーティング資料：Facts & Provenance監査

この文書は、15枚構成の本文のうちスライド3--15で使う主張と数値の監査表である。
スライド本文を簡潔に保ちながら、異なるコード版・cell定義・collision規則・評価段階を
誤って同列比較しないための内部資料として使う。

## Evidence区分

- **H: held-out / matched online** — 未使用seed、共通設定、pairedまたは負対照付き。本文の主要根拠にできる。
- **C: counterfactual / grouped offline** — 同一候補を全分岐、またはmacro groupを分離したoffline評価。機構・学習可能性の根拠だが、online高速化の証拠ではない。
- **P: pilot / tuning** — 設計を見ながら用いた少数seed。仮説生成にのみ使う。
- **A: audit / sanity check** — 可到達性、hash一致、保存経路再判定、実装動作の確認。方策優越性の証拠ではない。
- **D: design statement** — 実装・定義の説明。性能主張ではない。

## 全スライド共通の用語監査

1. **raw archive cell と canonical six**
   - raw cellは連続形状clusterとlayer keyで分けた実archiveのcellであり、重なり順variantにより6を超え得る。
   - canonical coverageは保存witnessとの再照合で得た従来の6目標の被覆数である。
   - `raw cells=8`を「8個の理論目標を発見」と表現しない。

2. **kinematic と feasible**
   - main9だけがcollisionを停止条件から外し、collisionを含んでも到達した`kinematic`と、root-to-terminalでcollision edgeが0の`feasible`を分離する。
   - main5--main8は原則としてcollisionでrollout/macroを停止する。main9のT6改善をcollision continuation単独の効果として扱わない。

3. **GPの因果効果**
   - GP対randomにはfilter、候補集中、rollout costなど複数差が含まれ得る。
   - GP scoreの候補対応そのものの効果は、原則として同じscore集合・保持数を壊さず対応だけを並べ替えた`shuffled`との差で判断する。

4. **予算とAUC**
   - 一次予算はmechanics step。AUCは予算中に保持した平均累積cell数で、0--1に正規化した確率ではない。
   - 異なるbudget、warmup、collision規則、archive定義のAUCを一つの棒グラフで直接比較しない。
   - 未到達runを捨てたT6平均だけを使わず、到達率・censor込み指標・AUCを併記する。

---

## Slide 3 — 問題設定と比較規約

### 本文で許される主張

- 目的は、固定mechanics-step予算内で多様な平坦終端cellを早く発見し、各cellの代表経路を保持すること。
- 検討は、単一摂動の終端回帰から、時間的にまとまったmacro-action、終端cell分類filterへ進んだ。
- 各段階で「行動表現の効果」と「学習した順位の効果」を分離する必要がある。

### 掲載する定義

- discovery AUC: \(B^{-1}\int_0^B C(b)\,db\)。
- T6: canonical sixの全てを初めて被覆したmechanics step。
- main9のみ、`kinematic T6`と`feasible T6`を別記する。

### 出典

- `MAIN5_RESULTS.md` §2
- `MULTISTEP_REACHABILITY_RESULTS.md`「共通の物理・数値条件」
- `MAIN9_FINAL_RESULTS.md`「collision と archive」

### Evidence

- **D**

### 比較不可・表現注意

- main5--8とmain9の発見曲線を、アルゴリズム名だけを横軸にした単一ランキングにしない。
- raw/canonicalおよびkinematic/feasibleを凡例なしで混ぜない。

---

## Slide 4 — main5：単一摂動GPはrandomを上回ったか

### 主張

- 単一 `(StateNode, perturbation)` 候補を連続終端descriptor GPで順位付けしても、1200-step held-outでは多様cell数を増やさなかった。
- GP-UCB-likeには僅かな平均AUC差があったが、seed間で符号が変わり、GP-posteriorは遅くかつ悪化した。

### 主要数値

- 条件: seed 100--104、B=1200、摂動norm 0.20、planar collision、初期trunk 63 step / 1 cell。
- random: final cells 2.000、AUC 1.6340、post-trunk AUC 0.7245、wall 13.17 s。
- GP-UCB-like: 2.000、1.6568、0.7486、19.86 s。
- paired AUC差: +0.0228、参考95% t区間 [-0.0750, 0.1207]。
- GP-posterior: final 1.800、AUC 1.5118、wall 101.84 s。

### 出典

- `MAIN5_RESULTS.md` §§3--7
- `result/heldout_scale020_comparison.csv`

### Evidence

- **H**（5 held-out seed。ただし標本は小さい）

### 比較不可・provenance注意

- これは後のpersistent macroではなく、単一摂動後にdeterministic rolloutするmain5初期系。
- この世代の保存configにはコードcommit/hashがなく、文書で監査済みの出力だけを使う。
- `main1.py`はroot-only bootstrapであり、このrandom baselineではない。

---

## Slide 5 — main5 GPは「学べない」のか「間に合わない」のか

### 主張

- GPはflat到達性とrootから離れた観測済みoutcomeの粗い順位を、24--32独立rollout程度から学ぶ。
- しかしguided開始時の4 rolloutでは信号が弱く、終端descriptorは80 rolloutでもarchive分解能に達しない。

### 主要数値

- random corpus: 5 seed、4000 step/seed、最後の32 rolloutを固定holdout。
- `all_edges`, n=4: flat AUC 0.588、root-novelty proxy AUC 0.580、descriptor W-error 17.66。
- n=32: 0.848、0.865、4.95。
- n=80: 0.877、0.921、3.60。
- archive分解能の目安はW-error 1以下。
- n=80のfit timeはall-edges 1.974 s、rollout-balanced 0.184 sで、順位性能は概ね維持。

### 出典

- `GP_LEARNING_CURVE_RESULTS.md` §§2--10, 13
- `result/learning_curve_random_b4000/gp_learning_curve_v1/learning_curve.png`
- `result/learning_curve_random_b4000/gp_learning_curve_v1/seed_mean_metrics.csv`

### Evidence

- **C**（固定holdout・rollout group分離）

### 比較不可・表現注意

- `root novelty`は初期trunk descriptorからの連続距離proxyであり、実archiveの未知cell labelではない。
- 教師行数と独立rollout数を混同しない。同一suffixの多数edgeは同じ終端を共有する。
- この曲線から「GPが未知cellを予測できた」とは言わない。

---

## Slide 6 — 小摂動でも6 cellは到達可能か

### 主張

- 現在の小摂動・descriptor・planar collision条件は6 canonical cellを排除していない。
- 通常randomの停滞は可到達性の欠如ではなく、有効な複数摂動系列への予算配分不足と整合する。

### 主要数値

- 保存witness: 8経路が6固有canonical cell、collision edge 0、corrected step norm最大0.199384。
- 旧tree metadataの初回発見経路の摂動数: `[1,2,2,3,2,3]`。現在判定で再検査した保存代表は全てK>=2。
- rootへnorm 0.30の一回摂動: 5/5 seedが5000 step内に6 cell、T6中央値1504。
- norm 0.20通常random、B=10000、seed 300--305: 平均2.33 cell、T6 0/6。seed 302を30000 stepまで延長しても3 cell。
- persistent emitter 72方向: K=1はnorm 0.15/0.20とも2 cell、K=2は両方6 cell。

### 出典

- `MULTISTEP_REACHABILITY_RESULTS.md` §§1--4
- `result/reachability_audit/reachability_report.md`
- `result/persistent_direction_emitter/persistent_emitter_report.md`
- `result/persistent_direction_emitter_16/persistent_emitter_report.md`

### Evidence

- **A/C**（可到達性監査と固定panel診断）

### 比較不可・表現注意

- root norm 0.30は対照であり、求める小摂動経路との性能比較には使わない。
- emitterは既知cell数6を停止条件に使うため、「7 cell目が存在しない」証明ではない。
- K=2 emitter panelは独立2摂動とのpaired因果比較ではない。

---

## Slide 7 — main5 v3.1 persistent transport heuristic

### 主張

- 摂動直後親とtransport方向をhard quotaで選ぶGP-free heuristicは、同じv3.1候補生成規則のrandomおよびGP+alignmentより良かった。
- この段階で強かったのは終端GPより、複数小摂動を連鎖させる構造priorだった。

### 主要数値

- version: `main5_gp_v3_1_transport_restore`、seed 400--402、B=10000。
- random: raw 2,2,2、canonical 2,2,2、平均AUC 1.972、T6 0/3。
- GP-UCB-like + alignment bonus 1: raw 5,6,4、canonical 5,4,4、AUC 3.761、T6 0/3。
- GP-free 80/20 transport heuristic: raw 6,7,6、canonical 6,6,6、AUC 4.870、T6 5370/2052/6495。
- seed 401の7番目raw cellはcanonical外のlayer-order variant。canonical sixはstep 2052で完了、variantはstep 7081。

### 出典

- `MULTISTEP_REACHABILITY_RESULTS.md` §§5--8
- `result/large_search_comparison_v31.csv`
- `result/canonical_coverage_v31.csv`
- `result/large_transport_heuristic_b10000_v31/seed_401/`

### Evidence

- **P/H相当のmatched比較**（同一version・seed・budgetだが3 seed、方式設計に使用）

### v3 / v3.1厳格注意

- 最終比較表は必ず`v31`を使う。
- `result/large_transport_heuristic_b10000_v3/seed_401/`は歴史的v3結果。v3では、既評価transport候補を除外した際に置換されていた通常grid候補を復元しない問題があった。
- v3.1は通常候補を復元するため、v3とv3.1のtrajectoryは同じseedでも厳密に同一ではない。v3の経路例とv3.1の集計値を一つのpaired結果として混ぜない。
- heuristicは「direct parent選択」と「transport方向選択」を同時に変更するため、方向類似度単独の効果ではない。

---

## Slide 8 — K=2 macro + GP filterの有望なpilot

### 主張

- global argmaxではなくtop-quantile filterとcost-aware化を用いると、設計seedではK=2 macro randomより良い兆候があった。
- ただしこれは後続held-outを正当化したpilotであり、最終的なGP有効性の証拠ではない。

### 主要数値

- seed 500--502、B=10000、version `main5_gp_v4_persistent_macro`。
- K=2 macro random: raw final 5.67、canonical T6 2/3、AUC 3.880。
- GP filter: raw 6.00、canonical T6 3/3、AUC 4.251、T6平均6969。
- cost-aware GP filter: raw 6.00、canonical T6 3/3、AUC 4.684、T6平均4488。
- cost-aware shuffled: raw 5.67、canonical T6 2/3、AUC 3.931。
- GP−cost-shuffled paired AUC差平均+0.754だがn=3のt区間は0を含む。

### 出典

- `GP_SELECTION_PRESSURE_RESULTS.md` §§2--7
- `result/selection_pressure_comparison_b10000.csv`

### Evidence

- **P**

### 比較不可・provenance注意

- seed 500--502は設計を見ながら使ったためheld-outではない。
- v4.2は診断列を追加した版で、文書上は選択ロジック不変。ただし表の本体はv4の保存結果。
- raw archive平均/AUCはlayer variantを含む。canonical到達率と同義ではない。
- このpilotの成功をmain6/main7/main8の負結果より優先しない。

---

## Slide 9 — 方向類似度仮説とmain6 angle-only GP

### 主張

- K=2が有効であることと、2回目をexact transport方向にすることは別である。
- offlineでは状態依存角度を学ぶ小さな信号があったが、第2角度だけを選ぶmain6 GPはonlineでrandomを上回らなかった。

### 主要数値

- offline: seed 623--625、54 `q1` group、16角度/group、864 macro。
- 新規率: uniform 0.126、theta=0 0.074、theta=pi 0.204、oracle 0.370。
- angle-aware GP group top-1 0.222、raw GP 0.130、context-shuffled 0.148。
- online通常warmup、seed 600--602、B=10000:
  - independent random 3.333 cell / AUC 2.8255。
  - fixed transport 2.333 / 2.2285。
  - learned angle GP 2.000 / 1.9668。
  - shuffled 2.333 / 2.0093。
- paired warmup版: 全4手法final 2.333、GP−shuffled AUC差+0.0027。

### 出典

- `MACRO2_ANGLE_PANEL_RESULTS.md`
- `MAIN6_ANGLE_SELECTION_RESULTS.md` §§2--5
- `MAIN6_ONLINE_AGGREGATE.md`
- `MAIN6_PAIRED_WARMUP_RESULTS.md`

### Evidence

- offline角度: **C**
- online main6: **P / matched**（3 seed）

### 比較不可・表現注意

- offline新規性は初期trunkだけの固定archive `A1` に対する値。onlineの動的archive価値ではない。
- 通常warmupとpaired warmupは教師収集費・共通prefixが異なるため、数値をpoolしない。
- main6は `(parent,v1)` を先にrandomで一つ固定し、その後の角度だけをGP選択する。angle GPの負結果を「joint action全体は学べない」と一般化しない。

---

## Slide 10 — main7：joint K=2 macro

### 主張

- 親・第1摂動・第2角度をまとめてscoreするjoint macroはangle-onlyより有望なtrajectoryを作ることがあった。
- しかし10000-stepでもGP−shuffled差はseed依存で、全手法T6未達。学習scoreの再現可能な優位は示せなかった。

### 主要数値

- version `main7_joint_k2_macro_gp_v1`、seed 600--602、共通paired warmup hash一致。
- B=3000: random 1.667 cell/AUC 1.417、joint GP 2.333/2.090、shuffled 2.333/1.911。
- B=10000 canonical audit:
  - random 1.667 / AUC 1.592。
  - fixed transport 2.000 / 1.805。
  - joint GP 2.667 / 2.303。
  - shuffled 2.333 / 2.207。
- GP−shuffledのAUC差はseed別 `(0,-0.757,+1.045)`、平均+0.096。全runでT6未達。
- 3000-step cost-aware GPは1.667 cell、cost-aware shuffled 2.000 cell。安価なcollisionを過度に選んだ。

### 出典

- `MAIN7_JOINT_MACRO.md`
- `MAIN7_PILOT_RESULTS.md`
- `MAIN7_B10000_RESULTS.md`
- `result/main7_b10000_canonical_coverage.json`

### Evidence

- **P / matched**（3 seed、共通prefix監査あり）

### 比較不可・表現注意

- 3000と10000は同じfrozen versionだが、別budget結果として表示する。
- 10000-stepでは全archive cellが異なるcanonicalへ対応しextra cell 0。ここではraw/canonical差はないが、他実験へ一般化しない。
- cost-aware batchとcore selector batchはCPU共有条件が違い、wall time比較に使わない。
- GP平均優位はseed 602に依存し、seed 601ではshuffledが明確に良い。

---

## Slide 11 — main8 offline：Kを選択変数にする根拠

### 主張

- 同一 `(q,v1)` でもKにより終端が大きく変わるため、固定K=2ではなく可変長persistent macroを使う根拠がある。
- grouped offlineではbasis-free GPにK順位信号があったが、これはonline高速化の証拠ではない。

### 主要数値

- seed 700--702、18 group、K=1--24、432 branch。
- Kでterminal cellが変化: 12/18=0.667。
- Kでflat/collisionが変化: 15/18=0.833。
- uniform K新規率0.338、basis-free GP top-1 0.667、score-shuffled 0.389、oracle 0.833。
- K=1--5平均新規率0.156、K=12--18は0.468、K=19--24はcollision率0.556。
- 通常GP top-1 cost 40.94、cost-aware 37.11だが、新規率0.667から0.611。new/step改善は約1.1%。

### 出典

- `PERSISTENT_K_PANEL.md`
- `result/persistent_k_panel_seed700_702/aggregate_summary.json`
- `result/persistent_k_panel_seed700_702/cost_aware_offline_summary.json`

### Evidence

- **C**（groupをfold間で完全分離）

### 比較不可・表現注意

- 新規性は固定baseline archiveに対するoffline label。
- main8 panelでは各micro-stepのcollisionで停止する。main9 collision-continuing panelではない。
- basis-free GPがofflineでshuffleを上回った事実と、onlineでGPを採用できることを同一視しない。

---

## Slide 12 — main8 online：効いたのはvariable-K、GPではない

### 主張

- 最も強く再現した改善は、GPを使わず行動空間へK=1--24を導入したこと。
- global GP filterも階層型K-GPも、held-outではscore-shuffledを上回らず、階層GPはrandomより悪かった。

### 主要数値

- GP-free ablation、B=3000、seed 700--702、warmupなし:
  - K=1: raw 2,2,1、canonical T6 0/3。
  - K=1--24 random: raw 8,8,8、canonical T6 3/3、T6 1597/1517/1686。
- global held-out、seed 703--712、B=3000、version `main8_variable_k_transport_macro_gp_v2`:
  - random / GP / shuffledはいずれもcanonical 6.0、T6 10/10。
  - canonical AUC 4.408 / 4.668 / 4.727。
  - GP−random AUC 95% CI [-0.009,0.595]、かつshuffledがGPより良い。
- hierarchical K held-out、GP側version `main8_variable_k_transport_macro_gp_v3_group_k`:
  - random T6 10/10、canonical AUC 4.408。
  - GP argmax T6 6/10、canonical AUC 4.323。
  - group-shuffled T6 7/10、canonical AUC 4.577。
  - GP−shuffled raw AUC差のbootstrap 95% CI [-0.935,-0.105]、sign-flip p=0.0488。

### 出典

- `MAIN8_FINAL_RESULTS.md` §§1--5
- `result/main8_heldout_b3000_aggregate.json`
- `result/main8_group_heldout_b3000_aggregate.json`

### Evidence

- K ablation: **P/matched**（3 seed、効果大）
- global / hierarchical: **H**（10 held-out seed、shuffled対照）

### version・比較注意

- global filterはv2、階層型K selectorはv3 group-k。random baselineの保存versionはv2。文書で共通候補・予算条件を監査しているが、globalとhierarchicalの行を一つの単一version実験としてpoolしない。
- ablationはwarmupなし、held-outは共通paired warmupあり。AUC絶対値を直接比較せず、各実験内の差だけを使う。
- raw平均7.8--8.0はlayer variant込み。目標達成はcanonical 6で判定する。
- main8ではcollision stop。main9のkinematic T6と横並びにしてcollision continuation効果を主張しない。

---

## Slide 13 — main9：collision continuationと分類filterへの再定式化

### 主張

- collisionを終端にすると終端写像を切断し、collision後にflatへ至る候補を失うため、main9ではcollisionを経路品質へ移した。
- 連続descriptorを無理に回帰せず終端cellだけを分類し、同一cell内の品質はarchiveで外部計算する。
- 未知cell labelは直接予測できないため、classifierは高信頼な既知duplicateだけを棄却するfilterとして使う。

### 実装事実・主要数値

- action: `(parent, first physical displacement, K)`, K=1--24。K後はdeterministic suffixをflatまたは全体予算まで継続。
- archive: kinematic eliteとcollision-free feasible eliteの二層。
- 20k corpusでcollisionを含んだ後にflatまで到達したmacro 577件。
- v1.1 B=3000 random評価で同種macro 272件、collision stop reason 0、local suffix timeout 0。
- macro hindsight rowは同一macro IDを共有し、重み合計1。train/testとbootstrapもmacro単位。

### 出典

- `MAIN9_FOREST_FILTER.md`
- `MAIN9_FINAL_RESULTS.md`「実装」「collision と archive」「教師」
- `collision_continue.py`
- `main9.py`

### Evidence

- **D/A**

### collision・価値定義の注意

- 「collision判定を無効化した」ではなく、「collisionをrollout停止条件から外し、品質として保持した」と書く。
- feasible eliteはcollision edge 0。kinematic発見数を実行可能な無衝突解数として扱わない。
- 現行quality keyはcollision有無・collision edge数・幾何経路長・摂動数・edge数。将来のstrainや微小形状差の全てを現在の`valuable`が含むわけではない。

---

## Slide 14 — main9：randomが解く時期とclassifierが学ぶ時期

### 主張

- classifierは長時間後には弱いfilter信号を学べるが、canonical T6より桁違いに遅い。
- 初期に効かない第一原因は独立macro不足、長時間側の上限は局所basin混在である。

### 主要数値

- random corpus: seed 800--802、合計60000 transition、1250 macro、13364保存row、valid 1241 macro。
- canonical kinematic T6: 548/770/1121、中央値770。
- T6時の独立macro: 11/14/17、保存row 89/159/225。
- fixed holdout accuracy:
  - 16 macro: RF 0.216、ET 0.198、kNN 0.222、prevalence 0.222。
  - 320 macro: 0.401 / 0.406 / 0.464。
- 全モデルがprevalenceを継続的に上回り始める32 macroはstep中央値1669。
- contextありstrict group-kNNがgateを満たす兆候: 320 macro、step中央値15509、precision 1.0、duplicate rejection 0.057、LOCO unknown retention 0.958、3 seed中2 seed。
- nearest same-cell: primary 0.380、hindsight 0.461。closest 20%でも0.548 / 0.593。
- `I(cell;K)/H(cell)=3.55%`。

### online安全filter

- v1.1、10 seed、B=3000、336384評価候補でstrict rejection 0。
- random / learned / shuffledのtrajectory・hashは完全一致。
- canonical kinematic T6: 10/10、平均1130.4。
- canonical feasible 6: 8/10、未達を3000で打切ったT6平均1944.3。
- learnedの追加wall time +57.6 s/run。

### aggressive pilot

- 3 seed、learned reject 10.47%。learned−randomはkinematic AUC +0.076、T6 -70 stepだが、feasible AUC -0.095、feasible T6 +563 step。

### 出典

- `MAIN9_MODEL_ANALYSIS.md`
- `MAIN9_FEATURE_ABLATION.md`
- `MAIN9_UNBOUNDED_ONLINE_RESULTS.md`
- `MAIN9_AGGRESSIVE_FILTER_RESULTS.md`
- `MAIN9_FINAL_RESULTS.md`
- `result/main9_model_analysis/learning_curves_t6.png`
- `result/main9_model_analysis/filter_tradeoff.png`
- `result/main9_unbounded_b3000_aggregate.json`

### Evidence

- offline learning: **C**
- strict online: **H/A**（10 seedだが選択圧ゼロなので優劣比較ではなく退化監査）
- aggressive: **P**

### 比較不可・provenance注意

- 20k offline corpusとaggressive pilotは評価版v1でsuffix cap 100。現行v1.1はcap 0（局所timeoutなし）。保存数値の再現時はcap 100を明示する。
- strict onlineでlearned=randomだったことを「同等性能の学習法」と表現しない。棄却0による完全退化である。
- kinematic改善とfeasible悪化を相殺して単一の「T6改善」と書かない。
- RF/ET/kNNのaccuracyだけでfilter有効性を判断せず、precision・duplicate rejection・unknown retentionを併記する。

---

## Slide 15 — 凍結snapshot評価と最終判断

### 主張

- 実運用の2/3/4-cell時点では独立教師が1--5 macroしかなく、安全filterは一度も発火しない。
- gateを外したconfidence rankingにも、未知保持を犠牲にしてrandomを明確に上回るだけの濃縮はない。
- 現時点の採用探索器はcollision-continuing variable-K persistent random。classifierはshadow modeで、prequential gateを満たした場合だけ有効化する。

### snapshot主要数値

- 3 seed × 3 snapshot × 128候補 = 1152候補、terminal確定1149。
- 独立教師macro平均: 2 cell=1.0、3 cell=3.7、4 cell=5.0。
- known top-1最良: 0.463 / 0.589 / 0.376。
- open-world top-1最良: 0.201 / 0.277 / 0.247。
- strict filter: 全model・全snapshot・all-known/feasibleの双方でreject fraction 0。
- 4-cell kNNの上位10%診断: reject precision 0.750、duplicate rejection 0.106、unknown retention 0.929、survivor novelty lift 1.025、valuable retention 0.908。

### 最終判断に載せる数値

- main9 v1.1 random: canonical kinematic 6は10/10、T6平均1130.4。
- canonical feasible 6は8/10、censor込み平均1944.3。
- safe learned filterはrandomと同じtrajectoryだが追加model計算を要した。

### 出典

- `result/main9_snapshot_filter_evaluation/aggregate/REPORT.md`
- `result/main9_snapshot_filter_evaluation/aggregate/snapshot_filter_summary.png`
- `result/main9_snapshot_filter_evaluation/aggregate/filter_selection_pressure.png`
- `result/main9_snapshot_filter_evaluation/aggregate/calibration_reliability.png`
- `MAIN9_FINAL_RESULTS.md`「採用する探索アルゴリズム」

### Evidence

- snapshot: **C**（厳密なfrozen counterfactual panel）
- 最終random到達率: **H**

### 比較不可・表現注意

- 2/3/4-cell snapshotでは利用可能StateNodeと候補panelが変わる。known/open-world accuracyを通常のnested learning curveとして折れ線で結び、単調性を議論しない。
- panel rolloutは学習・探索archiveへ戻していない。online性能ではなく、その時点のfilter選択圧の反実仮想評価。
- `valuable`は現行dual archiveに対するnew kinematic/new feasible/elite replacement。将来想定するstrainや同一cell内微差を全て含む一般価値ではない。
- current recommendationはfive-barでの結論。他構造への一般優位を主張しない。

---

## 本文作成時の最終チェックリスト

- [ ] v3.1比較図の入力は`large_search_comparison_v31.csv`で、v3フォルダを混ぜていない。
- [ ] `7`や`8` cellには必ず`raw`または`layer variant込み`と注記した。
- [ ] canonical six到達率はcanonical audit由来である。
- [ ] main9ではkinematicとfeasibleを色・線種の双方で区別した。
- [ ] collision-stop（main5--8）とcollision-continuing（main9）を同一条件差として扱っていない。
- [ ] GP効果の結論はGP−shuffledを優先し、GP−randomだけで因果主張していない。
- [ ] pilot seedとheld-out seedを視覚的に区別した。
- [ ] T6未達を除外した平均だけを載せていない。
- [ ] mechanics-stepとwall timeを混同していない。
- [ ] main5 root-novelty、main6/8固定archive novelty、main9の実cell filter指標を同一指標名で混ぜていない。
- [ ] snapshot accuracyを単調learning curveと呼んでいない。
- [ ] 「GPは学習不能」ではなく、「早期には独立標本不足、長時間側には局所basin混在」と結論した。
