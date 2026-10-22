# GP を用いた多様な平坦折経路探索

## main9: collision-continuing terminal-cell filter

`main9.py` は collision を rollout の終端条件から外し、物理方向を接空間へ
transport する可変長 macro `(StateNode, v1, K)` を実行します。terminal cell の
分類器は未知cellを直接予測せず、高確信な既知cellへの再到達だけを棄却する
保守的filterとして使います。kinematic archiveとcollision-free feasible archive、
Random Forest / ExtraTrees分類、score-shuffled対照、macro-group hindsight教師の
仕様は `MAIN9_FOREST_FILTER.md` を参照してください。

60,000 mechanics stepのrandom corpusによる学習曲線と局所basin解析は
`MAIN9_MODEL_ANALYSIS.md`、標準10-seed比較は`MAIN9_ONLINE_RESULTS.md`、
閾値を緩めた追加比較は`MAIN9_AGGRESSIVE_FILTER_RESULTS.md`、実装・評価全体の
採否は`MAIN9_FINAL_RESULTS.md`にまとめています。局所suffix timeoutを廃止した
現行v1.1の再比較は`MAIN9_UNBOUNDED_ONLINE_RESULTS.md`です。現結果では、安全filterは
randomと同一に退化し、攻めたfilterもrandomを一貫して上回りませんでした。
文字どおりの`(q,v,K)`とlayer履歴context入り入力の比較は
`MAIN9_FEATURE_ABLATION.md`にあります。
2/3/4 cell発見直後のtreeを凍結し、未学習候補を実際に分岐rolloutして分類精度と
filterの棄却精度を測る評価は
`result/main9_snapshot_filter_evaluation/aggregate/REPORT.md`にあります。
したがってfive-barの現行採用案はcollision継続variable-K randomで、分類器は
prequential gateを満たすまでshadow modeです。

## main7: joint K=2 macro

`main7.py` scores the complete unexecuted tuple `(parent, v1, w, theta)` so
that the GP can select the parent state and both perturbations before paying a
mechanics step. It compares independent random, fixed transport, joint GP,
score-shuffled controls, and cost-aware variants under a common paired
warmup. The physical basis-invariant input, fair-comparison rules,
environment variables, limitations, outputs, and identical-prefix smoke
audit are documented in `MAIN7_JOINT_MACRO.md`.

The matched seed 600--602, 3000-step pilot and its prefix audit are reported
in `MAIN7_PILOT_RESULTS.md`.

The frozen 10000-step extension, canonical-six coverage audit, T6 censoring,
and long-horizon interpretation are reported in `MAIN7_B10000_RESULTS.md`.

## main6: K=2 angle macro

`main6.py` compares `independent_random`, `fixed_transport`,
`fixed_reverse`, `learned_angle_gp`, and the negative control
`learned_angle_gp_shuffled`.  The optional selectors
`learned_angle_gp_cost_aware` and
`learned_angle_gp_cost_aware_shuffled` divide the same novelty acquisition by
a GP prediction of remaining mechanics cost.  They learn
`log(1 + q2-and-suffix steps)` from completed K=2 rollouts; the first action
leading to `q1` is excluded because it has already been paid when theta is
selected.  `FOLDING_MAIN6_COST_DENOMINATOR=expected` (default) uses expected
cost, while `lcb` uses an optimistic cost lower confidence bound.  Additional
settings are `FOLDING_MAIN6_COST_LCB_KAPPA` (default `1.0`),
`FOLDING_MAIN6_COST_FLOOR_STEPS` (default `1.0`), and
`FOLDING_MAIN6_COST_SCALE_FLOOR` (default `0.05`).  The expected denominator
is
`exp(mu_log_cost + variance_log_cost / 2) - 1`; the LCB denominator is
`exp(mu_log_cost - kappa * sigma_log_cost) - 1`.

This is a cost-aware ranking heuristic, not a stationary GP-UCB guarantee:
the discovery value changes whenever the elite archive changes, the flat-cell
reward is sparse, and the angle GP controls only the second action after a
random `(parent, v1)` has already fixed `q1`.  The score-shuffled control is
therefore required to attribute any online improvement to candidate ranking.
All selectors first execute the same random `(parent, v1)` rule. They differ
only in the second angle selected after the corrector has produced `q1`.
The default common balanced warmup uses 16 angles. Its size can be changed,
including to 64, with `FOLDING_MAIN6_WARMUP_ROLLOUTS` and
`FOLDING_MAIN6_NUM_ANGLES`.

The GP input order is `[q1, p, u, v2, cos(theta), sin(theta), projection
ratio, reference projection ratio, u mask]`. The kernel context block is
`[q1, p, u]`; the action block is `[v2, angle and projection scalars]`.
`angle_macro.py` constructs `p` and `u` from physical nodal displacement
vectors, so the representation is invariant to SVD-basis sign and orthogonal
rotation.

For tangent dimension greater than two, the current online
`independent_random` selector samples the angle in one randomly constructed
two-dimensional slice `span(p,u)`. It is not uniform on the full
higher-dimensional tangent sphere. A paired counterfactual comparison in
which every grid angle at an identical `q1` is fully rolled out remains an
offline angle-panel experiment. By default, online `main6.py` executes one
angle per `q1`.

An optional common paired warmup can be enabled with
`FOLDING_MAIN6_PAIRED_WARMUP=1`. For each warmup `q1`, it evaluates three
branches from the same corrected state: `theta=0`, alternating
`theta=+/-pi/2`, and `theta=pi`. The number of common `q1` groups is set by
`FOLDING_MAIN6_PAIRED_WARMUP_GROUPS` (default `4`). Every `q2` step and every
deterministic suffix is charged to the mechanics-step budget, added to the
archive independently, and stored as one teacher. Thus this warmup provides
paired low/medium/high directional-similarity examples, but it is roughly
three suffix rollouts per `q1`; it does not provide free counterfactual data.
Selection codes `14`, `15`, and `16` identify the three branches. The
`warmup_prefix_sha256` remains the audit that all selectors used an identical
prefix. The option is disabled by default, preserving legacy online runs.
These three warmup angles are constructed exactly and do not depend on the
online angle-grid resolution. A one-dimensional tangent has no orthogonal
direction `u`; only in that case the middle branch falls back to the nearest
valid direction and its actual saved `theta` must be checked.

このフォルダは、目的形状を事前指定せず、有限の力学 step 予算内で多様な平坦終端 cell と各 cell の代表経路を探す実装です。`main5.py` では、ある途中状態に微小摂動を与えた後の決定論的 rollout の結果を Gaussian Process（GP）で予測し、摂動位置と摂動方向に選択圧をかけます。

現行の保存versionは `main5_gp_v3_1_transport_restore` です。直前の物理摂動方向を現在状態の接空間へ射影したtransport候補と、その候補を明示的に選ぶGP-free heuristicも含みます。

現行実装の比較対象は `main5.py` の `random` selector です。`main1.py` は過去の root-only bootstrap 用コードであり、main5 の GP と条件をそろえた random baseline ではありません。

## 必要なパッケージ

```powershell
python -m pip install numpy scipy matplotlib cmaes plotly
```

`plotly` は対話的な探索木を保存するときに使います。

## 入力モデル

`model` フォルダに次の CSV 形式ファイルを置きます。

| ファイル | 内容 |
|---|---|
| `node_initial.dat` | 初期節点座標。`N x 3`、列は x, y, z |
| `member.dat` | 二節点部材。`M x 2`、節点番号は 0 始まり |
| `fix.dat` | 固定自由度。各行は `(節点番号, 座標軸)`。軸は x=0, y=1, z=2 |

代表長さ `L` には初期状態の平均部材長を使います。基準平面は `z = reference_point` に平行な xy 平面です。

## 実行方法

PowerShell でこのフォルダへ移動して実行します。

```powershell
Set-Location "C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0813_five_bar_linkage"

$env:FOLDING_MAIN5_SELECTOR = "gp_ucb"
$env:FOLDING_SEED = "0"
$env:FOLDING_STEP_BUDGET = "1200"
python main5.py
```

同条件の random baseline は selector だけを変えます。

```powershell
$env:FOLDING_MAIN5_SELECTOR = "random"
python main5.py
```

複数 seed を比較する例です。`FOLDING_RESULT_NAME` を以前に設定した場合は、selector ごとの既定名を使えるよう削除してください。

```powershell
Remove-Item Env:FOLDING_RESULT_NAME -ErrorAction SilentlyContinue
$env:FOLDING_STEP_BUDGET = "1200"

foreach ($selector in "random", "gp_ucb", "gp_posterior") {
    $env:FOLDING_MAIN5_SELECTOR = $selector
    foreach ($seed in 0..9) {
        $env:FOLDING_SEED = "$seed"
        python main5.py
    }
}

python compare_results.py main5_random main5_gp_ucb main5_gp_posterior
```

同じ result 名・seed を再実行すると、その seed の古い出力フォルダは削除して再生成されます。

## main5 の探索フロー

1. root から重力ポテンシャル最小化による決定論的 rollout を一本生成します。複数の root bootstrap は作りません。
2. 非終端の `StateNode` から親候補を選び、各状態の Jacobian の零空間で固定ノルムの摂動候補を生成します。この段階では corrector を実行しません。任意設定では、直前の物理摂動を現在の接空間へ射影したtransport方向で最も近い通常候補を置換します。
3. `random`、GP 獲得関数、または明示的な epsilon-random により候補を一つ選びます。
4. 選んだ候補だけ corrector を実行し、その後は平坦、衝突、rollout 上限、または総 step 予算まで決定論的に進めます。
5. 平坦なら elite archive を追加または更新し、平坦または衝突まで完了した rollout の新しい edge を教師データへ追加します。
6. GP を定期的に再学習し、予算まで 2--5 を繰り返します。

総予算は、摂動 step と決定論的 step を合わせた実際の力学 step 数です。候補の SVD 計算や GP の学習・評価時間は step 予算には含まず、wall time として別に記録します。

## 教師データの契約

教師データの基本単位は状態単体ではなく、実際に生成された `parent -> child` edge です。`StateNode.incoming_displacement` には、corrector 後の差分ではなく、corrector 前に提案した接空間上の物理変位を保存します。これにより、学習時と候補評価時の入力定義を一致させています。

自由 Cartesian 成分数を `n_free`、そのうち z 成分の数を `n_free,z` とすると、現在の入力は

```text
x = [ 現在節点の自由成分 / L,
      その経路で既知の最後の非平坦状態の自由 z 成分 / L,
      predictor 物理変位の自由成分 / L ]
```

で、次元は `2 n_free + n_free,z` です。モード係数そのものは入力しないため、SVD 基底の符号反転や基底内回転が GP 入力を変える問題を避けています。一方、候補生成は低次元のメカニズム係数空間で行うので、拘束に一次適合する方向だけを提案できます。

一つの完了 rollout からは、摂動 edge と、その後に新しく生成された全 deterministic edge を教師データにします。各 edge には、その edge より下流で実際に観測された同じ終端結果を対応させます。後から別の摂動枝が生えても、既存の祖先 edge のラベルを上書きしません。

| 教師値 | 定義 |
|---|---|
| `flat_labels` | rollout が flat なら 1、collision なら 0 |
| `flatness_targets` | 各 edge 以降の最小ポテンシャル `h_min` に対する `log(1 + h_min / flatness_tolerance)` |
| `descriptor_targets` | flat のみ。終端 xy 座標と最後の非平坦状態の z 座標からなる連続 descriptor の自由成分 |

collision の descriptor は `NaN` としてマスクされ、flat 条件付き descriptor GP には混ぜません。timeout、総予算による打ち切り、corrector failure は物理的な終端教師がないため、現在は教師データに入りません。したがって「全 StateNode で学習」とは、厳密には flat または collision まで完了した rollout 中の、ラベル付け可能な全新規 edge を使うという意味です。

## GP と獲得関数

`gp.py` は状態 block と変位 block に別々の長さ尺度を持つ Matérn-3/2 kernel を使います。出力ごとに独立な回帰を行いますが、入力 kernel と Cholesky 分解は共有します。次の三つを別々に学習します。

- flat 到達確率の近似二値 GP
- flat/collision の両方で定義できる連続 flatness GP
- flat データだけを用いる終端 descriptor GP

`gp_ucb` の主な獲得関数は概略

```text
score(x) = p_flat(x) [ d_archive(mu_D(x)) + kappa r_D(x) ]
           + reach_weight p_reach(x)
```

です。`d_archive` は予測平均と既存 elite descriptor の距離、`r_D` は descriptor GP の不確実性、`p_reach` は最小ポテンシャルが flatness 閾値以下になる確率です。これは新規 cell 発見に向けた heuristic な optimistic score であり、離散的な重なり順を含む実際の archive 判定に対する厳密な UCB 保証ではありません。

`gp_posterior` は descriptor の事後サンプルを archive 判定へ通し、新規 cell になるサンプル割合を `P_new` として使う代替です。flat 教師がまだない間は、flat 到達確率、flatness、分類器の不確実性を使う cold fallback に切り替わります。

## random baseline の正確な定義

`FOLDING_MAIN5_SELECTOR=random` は、各反復で生成された候補リストから一候補を一様ランダムに選びます。GP 版と次の条件を共有します。

- 最初の決定論的 trunk
- 力学、corrector、衝突判定、archive 判定
- 摂動ノルムと一状態あたりの方向生成規則
- rollout step 上限と総力学 step 予算
- 親候補集合の作り方

既定では全 history を毎回使うのではなく、非終端 StateNode を「経路中の摂動回数」と「初期ポテンシャルからの進行度」で層化し、最大 64 状態を抽出します。その各状態に候補方向を作った後、候補単位で一様選択します。そのため、状態ごとの候補数が異なる場合は StateNode 単位の一様選択ではありません。`FOLDING_MAIN5_PARENT_LIMIT=0` とすると全非終端 history を候補親として評価できます。

GP selector でも、完了した摂動 rollout の最初の 4 回は共通の random calibration とし、その後も既定で 20% の epsilon-random を残します。timeout や corrector failure は「完了した calibration rollout」に数えないため、cold-start 区間が 4 回の試行より長くなる場合があります。

transport候補は

\[
\bm v_{\mathrm{tr}}
=
\rho
\frac{\bm B(\bm q)\bm B(\bm q)^\dagger\bm v_{\mathrm{prev}}}
{\|\bm B(\bm q)\bm B(\bm q)^\dagger\bm v_{\mathrm{prev}}\|}
\]

で生成します。物理空間の射影を使うため、SVD基底の符号反転・基底内回転に依存しません。候補を追加せず最も近い通常方向を置換するので、一親あたりの候補数も変わりません。transport候補が既評価で除外された場合は、置換していた通常方向を復元します。

`FOLDING_MAIN5_SELECTOR=transport_heuristic` はGPを使わない追加対照です。transport候補が存在すれば既定で80%の確率でその集合から一様選択し、残り20%は全候補から一様選択します。これは物理方向を継続するpriorの効果を調べるGP-freeな比較対象です。ただし、GP版はsoftなscore、heuristic版はhardなquotaなので、両者の差だけからGPの因果効果を完全には分離できません。
`FOLDING_MAIN5_INCLUDE_TRANSPORTED_DIRECTION=1` も同時に指定してください。

## 主な環境変数

| 環境変数 | 既定値 | 意味 |
|---|---:|---|
| `FOLDING_MAIN5_SELECTOR` | `gp_ucb` | `random`, `transport_heuristic`, `gp_ucb`, `gp_posterior`, `gp_filter`, `gp_filter_shuffled`, `gp_filter_cost`, `gp_filter_cost_shuffled` |
| `FOLDING_SEED` | `0` | 乱数 seed |
| `FOLDING_STEP_BUDGET` | `1200` | 総力学 step 予算 |
| `FOLDING_COLLISION_MODEL` | `planar` | `planar` または `3d` |
| `FOLDING_TARGET_NORM` | `0.10` | deterministic predictor変位の公称最大ノルム |
| `FOLDING_PERTURBATION_SCALE` | `0.20` | corrector前の摂動変位の公称固定ノルム |
| `FOLDING_MAIN5_PERTURBATION_CHAIN_LENGTH` | `1` | 一作用内で接空間へ方向を運びながら続ける小摂動数 |
| `FOLDING_MAX_ROLLOUT_STEPS` | `100` | 一 rollout の最大 step 数 |
| `FOLDING_MAIN5_PARENT_LIMIT` | `64` | 一反復で候補化する親状態数。0 は全非終端 history |
| `FOLDING_MAIN5_DIRECTIONS` | `16` | 一状態あたりの候補方向数。ただし 1 自由度では固有な ±2 方向 |
| `FOLDING_MAIN5_DIRECT_PARENT_MINIMUM` | `0` | bounded poolへ最低限残す、摂動直後状態の数 |
| `FOLDING_MAIN5_INCLUDE_ROOT_PARENT` | `0` | 1ならrootをbounded poolへ必ず残す |
| `FOLDING_MAIN5_INCLUDE_TRANSPORTED_DIRECTION` | `0` | 1なら直前物理方向を現在の接空間へtransportした候補を含める |
| `FOLDING_MAIN5_TRANSPORT_MIN_RATIO` | `0.10` | transport射影を有効とする最小ノルム比 |
| `FOLDING_MAIN5_COLD_START_ROLLOUTS` | `4` | GP 選択前の完了 random rollout 数 |
| `FOLDING_MAIN5_EPSILON` | `0.20` | GP 開始後の random 選択率 |
| `FOLDING_MAIN5_REFIT_INTERVAL` | `3` | 完了 rollout 何回ごとに GP を再学習するか |
| `FOLDING_GP_STATE_LENGTH` | `0.50` | kernel の状態 block 長さ尺度 |
| `FOLDING_GP_ACTION_LENGTH` | `0.10` | kernel の変位 block 長さ尺度 |
| `FOLDING_GP_KAPPA` | `1.0` | descriptor 不確実性の係数 |
| `FOLDING_GP_DIRECT_PARENT_BONUS` | `0.0` | 摂動直後状態に掛けるguided scoreの乗算bonus |
| `FOLDING_GP_TRANSPORT_ALIGNMENT_BONUS` | `0.0` | transport方向とのcos類似度に掛けるguided scoreの乗算bonus |
| `FOLDING_GP_TRAINING_MODE` | `all_edges` | `all_edges`, `rollout_balanced`, `perturbation_only`, `macro_first` |
| `FOLDING_GP_BALANCED_EDGES_PER_ROLLOUT` | `8` | rollout-balanced fitで一rolloutから残す最大edge数 |
| `FOLDING_GP_REACH_WEIGHT` | `0.25` | flatness 到達項の重み |
| `FOLDING_GP_FILTER_KEEP_FRACTION` | `0.50` | GP filterで残す親候補の割合 |
| `FOLDING_GP_FILTER_AUDIT_RATE` | `0.0` | filterで落とす親候補を予算内で監査実行する割合 |
| `FOLDING_GP_COST_LCB_KAPPA` | `1.0` | cost下側信頼値の不確実性係数 |
| `FOLDING_GP_COST_OFFSET` | `1.0` | cost-aware scoreの分母offset |
| `FOLDING_GP_POSTERIOR_SAMPLES` | `32` | `gp_posterior` の事後サンプル数 |
| `FOLDING_SAVE_TREE` | `1` | 1 なら対話的探索木を保存 |
| `FOLDING_RESULT_NAME` | `main5_<selector>` | result 以下の実験名 |

衝突、flatness、layer、elite 距離の閾値なども環境変数で変更できます。実行時に使われた全値は各 seed の `config.json` に保存されるので、比較時はこのファイルも確認してください。

## ファイルの責務

| ファイル | 責務 |
|---|---|
| `main5.py` | 入力、予算、探索ループ、教師 edge の対応付け、archive 更新、GP 再学習、結果保存 |
| `gp.py` | 入出力変換、Matérn GP、flat 分類近似、候補生成、UCB/事後新規確率の計算 |
| `state.py` | `StateNode` と親子関係、入力 edge の変位・種類、経路復元 |
| `mechanics.py` | 拘束 Jacobian、SVD mode、predictor/corrector、重力ポテンシャル最小化 step |
| `roll.py` | 一 step の実行、子 StateNode の追加、flat/collision/failure 判定、衝突 backend 切替 |
| `descriptor.py` | 途中状態・終端形状・最後の非平坦状態・重なり順 descriptor |
| `archive.py` | elite cell の照合、追加、同一 cell 内の短経路への更新 |
| `collision2.py` | five-bar 用 x-z 平面内の連続的な部材交差判定 |
| `collision.py` | 一般 3D 用の近似的な部材接近・交差判定 |
| `result.py` | 代表経路、発見曲線、探索木の保存と同名 seed 出力の初期化 |
| `compare_results.py` | 複数 selector・seed の既存結果を読み、seed 別値と平均を CSV 化 |
| `gp_learning_curve.py` | 共通random corpusをrollout単位で分割し、GPのデータ量に対する予測性能を評価 |
| `gp_paired_panel.py` | 共通親のtransport・角度差・逆向き候補を全評価し、raw/Context-GPの順位性能を比較 |
| `paired_panel_aggregate.py` | paired panelのseed別summaryを平均・標準偏差へ集約 |
| `path_diagnostics.py` | 摂動深さ、連続摂動長、direct/transport score bonus |
| `online_policy_metrics.py` | fixed-step prefix、time-to-cell、親選択集中、filter診断を集計 |
| `root_reachability.py` | 現行descriptor・collision下でのroot大摂動到達性対照 |
| `five_bar_reachability_report.py` | 保存小摂動経路を現行判定で再検査 |
| `five_bar_persistent_emitter.py` | 物理方向を局所接空間へ運ぶtarget-descriptor-free到達性診断。既知cell数6は停止条件に使用 |
| `five_bar_coverage_audit.py` | 保存したarchive cellを従来のfive-bar canonical witnessへ再照合 |
| `main1.py` | 旧 root-only random bootstrap。main5 の公平な random baseline ではない |

## 出力

出力先は `result/<result_name>/seed_XXX/` です。

- `summary.json`: cell 数、AUC、trunk 後の新規 cell 数、rollout 内訳、計算時間
- `config.json`: その実行で使った設定
- `discovery_curve.dat`, `discovery_curve.png`: step 数に対する累積 elite cell 数
- `rollout_diagnostics.dat`: 各 rollout の親、消費 step、停止理由、選択方法、獲得関数内訳
- `discovery_events.json`: 各cellを初めて発見した経路の摂動数とstep種別列
- `training_data.npz`: 上記の edge 入力と教師値
- `cell[i]/0.dat, 1.dat, ...`: elite cell の代表経路上の節点座標
- `search_tree.html`: five-bar の対話的探索木。対応しない構造では `search_tree_skipped.txt`

手法比較では最終 cell 数だけでなく、早期発見を評価する `discovery_auc`、共通 trunk 後だけを評価する `post_trunk_new_cells` と `post_trunk_discovery_auc`、wall time も併記してください。

今回実行した held-out 比較、摂動ノルム感度、結果の批判的な解釈は `MAIN5_RESULTS.md` にまとめています。集計値そのものは `result/heldout_scale020_comparison.csv` です。

GPの学習曲線は、例えば次のように既存のrandom探索結果から作成できます。

```powershell
python gp_learning_curve.py learning_curve_random_b4000 `
  --train-sizes 4 8 16 24 32 48 64 80 `
  --holdout-rollouts 32 `
  --modes all_edges perturbation_only rollout_balanced
```

最後の32摂動rolloutを全学習点で共通のtest集合とし、同一rolloutのedgeがtrain/testへ分割されることを禁止しています。出力はsource result内の`gp_learning_curve/`、または`--output-directory`で指定した場所に保存されます。

今回の共通random corpusによる数値結果、学習が有効になり始めるデータ量、および評価上の限界は `GP_LEARNING_CURVE_RESULTS.md` にまとめています。

小摂動での6 cell可到達性、大規模random、persistent-direction emitter、
direct/transport bonus、およびGPを使わないtransport対照の結果は
`MULTISTEP_REACHABILITY_RESULTS.md` にまとめています。

現行v3.1の10000-step比較では、同じ候補生成規則とseed 400--402に対して、
randomは2, 2, 2 cell、GP＋方向持続scoreは5, 6, 4 cell、GPなしtransport
heuristicは6, 7, 6 cellでした。従来のcanonical 6の被覆はそれぞれ
2, 2, 2、5, 4, 4、6, 6, 6です。詳細集計は
`result/large_search_comparison_v31.csv` と `result/canonical_coverage_v31.csv` にあります。

K=2 persistent macro、GP filter、score-shuffle負対照、cost-aware filterの
新しい予備比較は `GP_SELECTION_PRESSURE_RESULTS.md` にまとめています。
seed 500--502・10000 stepでは、macro randomのcanonical 6到達率が2/3、
cost-aware GP filterが3/3で、後者の平均6番目発見stepは4488でした。
集計値は `result/selection_pressure_comparison_b10000.csv` にあります。

可変長の方向持続macro `(StateNode, v1, K)` は `main8.py` に実装しています。
実装仕様は `MAIN8_VARIABLE_K.md`、同じ `(q,v1)` に全Kを与えたoffline評価は
`PERSISTENT_K_PANEL.md`、K=1 ablation、global/階層型GP、shuffled対照、
cost-awareを含む最終結論は `MAIN8_FINAL_RESULTS.md` にまとめています。

## 汎用性の範囲

節点数、部材数、自由座標数、各状態のメカニズム自由度は入力モデルから取得します。GP はモード係数ではなく固定次元の物理変位を受け取るため、探索中に自由度や SVD 基底が変わる構造にも同じコード構成を使えます。

ただし、現状が直接仮定するのは、3 次元節点座標、二節点の定長部材、xy に平行な平坦基準面、z 方向の重なり順、同一 run 内で固定された節点・部材ラベルです。異なる節点数や topology をまたいで一つの GP を共有する実装ではありません。`planar` 衝突判定と探索木表示は five-bar 向けです。他構造では `FOLDING_COLLISION_MODEL=3d` を選べますが、現在の 3D 判定も厳密な連続 panel collision ではありません。

## 既知の限界

- 終端写像は basin 境界で不連続になり得ます。滑らかな GP の予測平均は実在しない中間形状を表す場合があります。
- descriptor 出力は条件付き独立として扱い、出力間共分散を学習していません。事後サンプルも物理拘束を必ず満たすとは限りません。
- flat データが少ない間は descriptor GP を十分に学習できず、random 探索と fallback score への依存が大きくなります。
- timeout、予算打切り、corrector failure の情報を現在は学習に利用しません。
- 一 rollout の全 edge を同じ終端へ対応付けるため、長い deterministic suffix が教師数を支配し、重要な perturbation edge が相対的に軽くなる可能性があります。
- exact GP の学習費用は教師 edge 数に対して概ね三乗で増えます。大きい予算・構造では疎 GP、局所 GP、教師点削減が必要です。
- 候補親を 64 状態に制限する既定設定では、history 中の有望状態を一時的に候補から外す可能性があります。
- archive、衝突、roll、kernel の閾値・長さ尺度は構造依存です。単一 seed や単一構造だけで優劣を結論づけず、共通設定で複数 seed と感度を確認する必要があります。
- 有限予算内の探索であり、全平坦 cell の列挙保証や新規 cell 発見の理論保証はありません。
