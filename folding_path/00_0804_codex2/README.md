# five-bar linkage：非定常な発見報酬を考慮した二段階探索

このフォルダは，多自由度の折り構造に対し，目的となる平坦形状を事前指定せず，有限の数値計算予算内で

- 最終形状と重なり順で区別される多様な平坦終端cell
- 各cellへ到達する代表最短経路

を探索する five-bar linkage の実験実装です。

現在の主実験は，次の二段階比較です。

1. 摂動位置は一様ランダムに固定し，root再始動baselineと候補集合 `frontier / event / history` を比較する。
2. `frontier / event / history` の第1段階勝者を固定し，位置選択器 `random / UCB / SM-UCB` を別seedで比較する。

旧実装の `event-frontier + lifetime UCB`，`bounded archive + lifetime UCB`，摂動系列MAP-Elitesは，研究経緯を再現するためコードと出力を残しています。ただし，現在の実行入口からは呼ばれません。

## 現在の探索設計

### 共通力学

状態座標を \(q\)，拘束Jacobianのnull-space基底を \(M(q)\) とすると，ランダムemitterは

\[
\alpha \sim \operatorname{Unif}\{a:\|a\|_2\le a_{\max}\},
\qquad
q^+=\mathcal P_{\mathcal Q}(q+M(q)\alpha)
\]

により一回の摂動を作ります。その後は，重力ポテンシャル型の量を逐次最小化する決定論的rolloutを行います。平坦化または衝突を終端とし，計算予算は新しく計算したmechanics stepの総数です。保存済みprefixの再利用は予算へ数えません。

一つの経路に与えられる摂動回数の上限は，現在のstate-restart実験では用いません。有限性は経路長上限 `max_path_steps` と総step予算 `search_step_budget` で保証します。設定に残る `max_interventions` は旧系列MAP-Elitesを再現するための項目です。

### 終端archive

終端cellは

\[
c(\tau)=
(\text{flat-shape signature},\text{layer-order signature})
\]

で定義します。同じcellでは，初期状態からの論理step数が短い経路だけがeliteを置換します。衝突，非平坦打切り，数値失敗は終端archiveへ入りません。

位置選択の報酬はcoverageだけに限定し，

\[
r_t=\mathbf 1[c(\tau_t)\notin A_t]
\]

とします。elite短縮は保存・評価しますが，選択報酬へ混ぜません。

## 第1段階：候補集合の比較

共通の3本bootstrap後に，次の二つのroot controlも同じ800 search-step予算で実行します。

- `root_random`：常に未変形rootを選び，他の三条件と同じ半径0.22のrandom emitterを適用する。摂動位置だけを比較するための主baseline。
- `root_bootstrap_random`：rootにbootstrapと同じ半径0.08の不整を繰り返す。これは文字どおり「予算までbootstrapを生成する」感度解析であり，位置と摂動半径が同時に変わる。

root集合は一状態しかないため，三集合との候補数同一化は行いません。rootを複製しても選択分布は変わらないためです。公平性は，共通bootstrap，共通random stream，共通rollout，総mechanics-step予算で担保し，rollout数を別途報告します。

途中状態の三条件では，各rolloutで新しく計算されたsuffixだけを一度登録し，再利用prefixは重複登録しません。共通rootも候補から除きます。

- `history`：これまで生成した全ての有効・非平坦・再始動可能状態。
- `frontier`：各完了rolloutの新規suffix中で最も深い再始動可能状態。
- `event`：全frontierに加え，等間隔，ポテンシャル局所極値，メカニズム基底変化，descriptor曲率，衝突直前，最後の非平坦状態のいずれかに該当する状態。

したがってraw universeは

\[
F_t^\star\subseteq E_t^\star\subseteq H_t^\star
\]

です。各state IDには固定priority \(u(v)\in(0,1)\) をhashから一度だけ割り当てます。時刻 \(t\) の共通候補数を

\[
K_t=
\min\{K_{\max},|F_t^\star|,|E_t^\star|,|H_t^\star|\}
\]

とし，各raw universeからpriority最小の \(K_t\) 状態を公開します。これにより三集合は常に同数で，各公開集合は対応するraw universeの一様bottom-\(k\) reservoir sampleです。

event severity，novelty merge，ランダムevent，UCBは第1段階の候補追加・置換に使いません。候補集合は `candidate_epoch_rollouts` 回のrollout中は固定し，新しい状態は次epochから選択可能になります。

第1段階の三集合の勝者は，事前に固定した次の辞書式規則で決めます。rootは一候補なので，Stage 2でrandom/UCB/SM-UCBを比較する意味がなく，winner選定から除外します。

1. bootstrap後に増えた終端cell数のseed間中央値が最大
2. 同率なら new-cell discovery curve AUCのseed間平均が最大
3. なお同率なら新規cell数のseed間平均が最大
4. 完全同率なら `frontier, event, history` の固定順

## 第2段階：位置選択器の比較

第1段階のseedと重ならないheld-out seedを使います。

### Random

固定epoch候補から一様に一状態を選びます。

### 通常UCB

immutable state IDごとにlifetime平均を保存し，

\[
U_t(v)=\bar r_t(v)+c\sqrt{\frac{\log t}{n_t(v)}}
\]

を最大化します。未選択状態は \(+\infty\) です。新規cell報酬はarchiveが埋まるほど減るため，このlifetime平均は現在の成功確率の不偏推定ではありません。ここではMonte Carlo Elites型のbaselineとして実装しています。

### SM-UCB

現在のarchiveに対する期待限界利得

\[
g(v,A_t)
=
\mathbb E_{\alpha}
\left[
F(A_t\cup\{Z(v,\alpha)\})-F(A_t)
\right]
\]

を，状態特徴とarchive contextのkernelで推定します。観測 \((x_i,A_i,r_i)\) に対するposteriorは

\[
\mu_t(z)=k_t(z)^\mathsf T(K_t+\sigma^2I)^{-1}r,
\qquad
s_t^2(z)=k(z,z)-k_t(z)^\mathsf T(K_t+\sigma^2I)^{-1}k_t(z)
\]

で，選択scoreは

\[
U_t^{\mathrm{SM}}(v,A_t)
=\mu_t(v,A_t)+\sqrt{\beta_t}\,s_t(v,A_t)
\]

です。

実装kernelは，状態featureのRBFとarchive cell集合のcosine kernel，archive要素数のRBFの積です。状態featureには座標descriptorだけでなく，絶対depth，残りdepth，ポテンシャル，メカニズム次元，既往摂動率を含めます。Gaussian-process計算は逐次Cholesky更新を使います。

SM-UCB原論文のregret保証は，固定ground set，低RKHS norm，有界martingale noiseなどを仮定します。本実装では候補をepoch内固定して対応を近づけていますが，新StateNodeの生成，stateの再選択，確率的emitter，可変rollout costがあるため，木探索全体へ原定理をそのまま主張しません。

## 実行方法

依存関係は Python 3.11，NumPy，SciPy，Matplotlib，`cmaes` です。

```powershell
python -m pip install numpy scipy matplotlib cmaes
```

短い動作確認：

```powershell
python run_experiments.py `
  --quick `
  --stage all `
  --output output/protocol_smoke `
  --clean-output
```

既定の5+5 seed，800 mechanics-step実験：

```powershell
python run_experiments.py `
  --stage all `
  --budget 800 `
  --output output/nonstationary_protocol `
  --clean-output
```

保存済みroot位置比較だけを再現する場合：

```powershell
python run_experiments.py `
  --stage 1 `
  --output output/root_location_comparison_full
```

第1段階だけ：

```powershell
python run_experiments.py --stage 1 --output output/nonstationary_protocol
```

保存済みwinnerを使って第2段階だけ：

```powershell
python run_experiments.py --stage 2 --output output/nonstationary_protocol
```

winnerを明示する場合：

```powershell
python run_experiments.py `
  --stage 2 `
  --winner-pool frontier `
  --stage2-seeds 101 131 151 181 211
```

設定上書き：

```powershell
python run_experiments.py `
  --set candidate_pool_capacity=32 `
  --set candidate_epoch_rollouts=10 `
  --set smucb_state_lengthscale=0.3
```

## 出力

```text
output/nonstationary_protocol/
├── effective_config.json
├── protocol.json
├── stage1_candidate_sets/
│   ├── per_run_summary.csv
│   ├── aggregate_summary.csv
│   ├── winner.json
│   ├── discovery_curves.csv
│   ├── discovery_curves.png
│   └── runs/seed_NNNNNN/<root-or-pool>_random/
└── stage2_selectors/
    ├── per_run_summary.csv
    ├── aggregate_summary.csv
    ├── design.json
    ├── discovery_curves.csv
    ├── discovery_curves.png
    └── runs/seed_NNNNNN/<winner>_<selector>/
```

各condition folderには `summary.json`，`search_log.csv`，`repository_log.csv`，`outcome_archive.csv`，各elite経路の `path.npy/actions.npy/nodes.npy` が入ります。`candidate_size_violations` は同数不変条件の診断値で，正常なら0です。

## 2026-08-05の固定設定による結果

共通条件は `K_max=24`，epoch 8 rollout，予算800 mechanics-step，通常摂動上限0.22です。第1段階seedは `11,29,47,71,89`，第2段階seedは `101,131,151,181,211` で，互いに重なりません。root比較の完成出力は `output/root_location_comparison_full/` にあります。

### 第1段階

| 条件 | 摂動半径 | 新規cell数 平均±SD | 最終総cell | 最終shape数 | new-cell AUC 平均±SD | rollout数 |
|---|---:|---:|---:|---:|---:|---:|
| root | 0.22 | 2.4 ± 0.55 | 4.0 | 4.0 | 2.165 ± 0.607 | 52.2 |
| root（bootstrap半径） | 0.08 | 0.4 ± 0.55 | 2.0 | 2.0 | 0.381 ± 0.521 | 51.0 |
| frontier | 0.22 | 1.6 ± 0.55 | 3.2 | 1.6 | 1.399 ± 0.399 | 356.8 |
| event | 0.22 | 1.2 ± 0.84 | 2.8 | 2.0 | 0.698 ± 0.576 | 130.8 |
| history | 0.22 | 1.6 ± 1.14 | 3.2 | 2.0 | 0.900 ± 0.812 | 136.0 |

半径0.22のroot条件は全5 seedsで，既往のevery-step-opt結果と同じ4形状
`+++- / ++-+ / +-++ / -+++` を発見しました。4形状が揃ったsearch stepはseed順に `76, 76, 75, 237, 59` です。一方，半径0.08のliteral bootstrap反復は全seedで2形状に留まりました。したがってこの例では，途中状態かrootかだけでなく，初期摂動半径が探索結果を支配しています。

位置以外を揃えた比較ではroot 0.22が最も多くの形状とcellを得ました。これはfive-barの最終assembly modeが経路初期に決まり，深い状態への再摂動では主に重なり順だけが変わることを示唆します。ただし5 seedsの単一構造による結果であり，一般的優位の主張ではありません。またrootは約52 rollout，frontierは約357 rolloutで，等しいのは総step数であって摂動試行数ではありません。

現在の1 mechanics stepは，摂動更新または重力更新の外側の一遷移です。重力更新内部のCMA-ES候補評価数は予算に含めていません。現行5条件は同じCMA設定なので相対比較できますが，最大200 CMA世代を使う旧every-step-optの89保存stepとは直接比較できません。

三集合内では `frontier` と `history` が最終新規cell数で同率でした。固定tie-breakのAUCにより `frontier` を第2段階へ採用しました。三集合の全runで候補数不一致は0件です。

### 第2段階

| 選択器 | 新規cell数 平均±標準偏差 | 中央値 | new-cell AUC 平均 |
|---|---:|---:|---:|
| random | 1.6 ± 0.55 | 2 | 1.429 |
| UCB | 1.6 ± 0.55 | 2 | 1.511 |
| SM-UCB | 1.6 ± 0.55 | 2 | 1.540 |

最終coverageは全seed・全選択器で同じでした。SM-UCBのAUCはrandomより5/5 seed，UCBより5/5 seedで大きく，平均差はそれぞれ0.111，0.029でした。ただしseed数は5で差も小さいため，現時点では「この設定では発見が少し早かった」という予備的観察に留めます。「より多くの終端cellを発見した」とは結論しません。

## コード構成

```text
run_experiments.py
qd_folding/
├── mechanics.py             # Jacobian/SVD, predictor-corrector, rollout
├── descriptors.py           # 状態・終端descriptor
├── archives.py              # 終端elite archiveと全trajectory history
├── candidate_sets.py        # 三raw universeと同数bottom-k候補
├── selectors.py             # random, lifetime UCB, kernel SM-UCB
├── protocol_search.py       # 一conditionの共通探索loop
├── protocol_experiment.py   # 二段階・複数seed・winner選定
├── output.py                # CSV/JSON/図/elite経路出力
└── config.py
```

旧検討を再現するモジュールは `state_search.py`，`state_repositories.py`，`sequence_search.py`，`sequence_genome.py`，`experiment.py`，`tune_hyperparameters.py` です。現在の結論や主実験のAPIと混同しないでください。

## テスト

```powershell
python tests\run_tests.py
```

候補集合の包含関係と同数性，UCB/SM-UCB更新，二段階tiny run，旧力学・archive契約を含む20テストを実行します。

## LaTeX/PDF

```powershell
latexmk -norc -r latexmkrc QD_FOLDING_METHODS.tex
```

PDFは `out/QD_FOLDING_METHODS.pdf` に生成されます。最終確認版は `output/pdf/QD_FOLDING_METHODS.pdf` に複製します。
