# ori_codex6 最終 zero-reject 候補の設計整理

更新日: 2026-08-27

対象候補: `ori6_RC_spectral_subspace_zero_reject_reward`

対象実験:

- Torus: `P0X2RCS05ZR0R1`, selected seed 4203
- Water (D^*=1): `P0X2RCS05ZR0WF12`, selected seed 4301

## 0. 最重要の訂正

最終実験で使った更新法は **PPO ではなく、episode-separated Monte-Carlo Actor--Critic** である。

- 両方の `run_config.json` は `update_method="actor_critic"`。
- `ppo_epochs=4`, `ppo_clip=0.2`, `target_kl=0.02` も設定ファイルには保存されているが、今回の分岐では使われない。
- PPO の実装自体は `ori_codex3.py:644-996` に存在するが、今回の学習では選択されていない。
- sampled action とその `log_probability` を保存する理由は、PPO の likelihood ratio のためではなく、通常の score-function policy gradient
  \(-\log\pi_\theta(a_t\mid s_t)A_t\) を計算するためである。

実行時の選択は `ori_codex3.py:2003-2019`、既定値は `ori_codex3.py:2388-2399`、実際の Actor--Critic 更新は `ori_codex2.py:601-667` にある。

## 1. コードの層構造

最終候補は一つの巨大なクラスではなく、既存実装を加算的に積み重ねている。

| 層 | 主な責務 | 主な場所 |
|---|---|---|
| `ori_codex2.py` | 座標観測、機構 mode、可変次元 action、制約射影、Monte-Carlo Actor--Critic | `145-151`, `165-235`, `336-667` |
| `ori_codex3.py` | projected spread、success / wrong-flat 判定、任意の PPO 実装、uniform random agent | `42-51`, `428-630`, `644-996`, `1565-1606` |
| `ori_codex4.py` | terminal near-miss quality \(q_F,q_D,q\) | `76-135` |
| `ori_codex5.py` | flatness と target spread の endpoint-potential、fixed joint milestones | `36-64`, `270-302`, `304-429` |
| `ori_codex6.py` | continuous spread-frontier potential、終端時 rollback | `136-182`, `184-370` |
| `ori_codex6_rejected_crossing.py` | crossing candidate を安全状態へ戻して探索継続 | `228-396` |
| `ori_codex6_spectral_subspace.py` | plane swap を表す spectral-subspace potential | `38-59`, `96-176`, `278-419` |
| `ori_codex6_zero_reject.py` | rejected crossing の即時報酬を厳密に 0 に上書き | `40-59`, `103-235` |
| `ori_codex6_zero_reject_3d.py` | 上記候補を continuous 3-D collision checker に接続 | `120-165` |
| `collision_3d.py` | 線形に動く非隣接 bar の swept proper crossing 判定 | `1-19`, `413-533` |

したがって、最終的な意味論を確認するときは `ori_codex6_zero_reject_3d.py` だけでなく、この層全体を見る必要がある。

## 2. 状態、幾何量、成功条件

### 2.1 NN に渡る情報

環境の座標観測は

\[
s=\operatorname{vec}(X/L),
\]

である。\(X\) は全 node の座標、\(L\) は初期形状における最大 bar 長である (`ori_codex2.py:85-97,145-147`)。

ただし Actor と Critic は座標だけで動くわけではない。各 step で bar-length constraint Jacobian の nullspace basis

\[
M(s)=[m_1,\dots,m_m]
\]

も計算し、座標とは別の引数として NN に渡す (`ori_codex2.py:149-151,390-437`)。SVD 由来なので mode columns は orthonormal である (`ori_codex.py:26-49`)。

最終 task の入力規模は次の通り。

| task | nodes | coordinate size | initial modes |
|---|---:|---:|---:|
| Torus | 20 | 60 | 5 |
| Water | 7 | 21 | 3 |

step count、残り horizon、milestone 獲得済み flag、reject streak は NN 入力に含まれない。continuous potentials は固定 reset の座標から決まるが、有限 horizon と once-only milestone を含む環境全体は厳密には部分観測である。

### 2.2 flatness と spread

中心化座標を

\[
X_c=U\operatorname{diag}(\sigma)V^\top
\]

と SVD する。今回の 3-D task は平面化なので \(k=2\)。flatness potential と normalized flatness ratio は

\[
P(X)=\sum_{i>k}\sigma_i^2,\qquad F(X)=\frac{P(X)}{P(X_0)}.
\]

実装は `single_solution_common.py:29-36` と `ori_codex2.py:153-163`。

spread \(D\) は、best-fit plane に射影した nodes の最大 pairwise distance を \(L\) で割った値である (`ori_codex3.py:42-51`)。spread error は

\[
e_D=|D-D^*|.
\]

### 2.3 terminal 判定

受理された状態について、最終成功条件は

\[
F\le 10^{-3},\qquad e_D\le 0.05.
\]

`wrong_flat` は \(F\le10^{-3}\) だが \(e_D>0.05\) の状態であり、直ちに episode を終える。

最終的な優先順位は次の通り。

1. `correction_failed`
2. crossing candidate なら safe state へ reject。horizon なら `timeout`、そうでなければ継続
3. `success`
4. `wrong_flat`
5. `no_modes`
6. `timeout`
7. `running`

crossed candidate 自体が success や wrong-flat になることはない。

## 3. 行動空間

現在の mode 数を \(m\) とする。policy の latent action は \(m+1\) 個の実数である。

### 3.1 機構方向

各 mode に対し

\[
\tilde c_j\sim\mathcal N(\mu_j(s,m_j),\sigma_j^2(s,m_j))
\]

を独立に sample し、

\[
c=\frac{\tilde c}{\|\tilde c\|},\qquad
u=\frac{M c}{\|M c\|}
\]

として current tangent/nullspace 内の unit direction を作る。同じ直線を表す \((u,-u)\) の重複を避けるため、\(|u_i|\) 最大の成分が正になる hemisphere に canonicalize する。移動の正負は次の signed width だけに持たせる (`ori_codex2.py:45-56,514-539`)。

### 3.2 step 幅

方向 \(u\) を条件に latent width

\[
\tilde h\sim\mathcal N(\mu_h(s,u,M),\sigma_h^2(s,u,M))
\]

を sample し、環境へ渡す幅を

\[
h=h_{\max}\tanh(\tilde h)
\]

とする (`ori_codex2.py:540-551`)。

したがって実際の action は「current feasible-mode span 内の canonical unit direction \(u\)」と「bounded signed width \(h\)」の組である。mode 数は形状により変化する。

### 3.3 action log probability

学習に保存する値は latent Gaussian 上の

\[
\log\pi(a\mid s)
=\sum_{j=1}^{m}\log\mathcal N(\tilde c_j;\mu_j,\sigma_j^2)
+\log\mathcal N(\tilde h;\mu_h,\sigma_h^2)
\]

である。environment に渡す action はこれを normalize / canonicalize / tanh した決定論的変換である (`ori_codex2.py:553-565`)。

### 3.4 NN architecture

- Mode encoder: `[state, one mode] -> 96 -> tanh -> 96 -> tanh`
- Mode coefficient head: 各 mode の mean と bounded variance
- Width body: `[state, physical direction, mean-pooled mode feature] -> 96 -> tanh -> 96 -> tanh`
- Width head: mean と bounded variance
- Critic: state branch と mean-pooled mode branch を結合し `-> 96 -> tanh -> scalar V`
- Actor parameters と Critic parameters は完全に分離

variance は

\[
10^{-4}+(2-10^{-4})\operatorname{sigmoid}(\ell)
\]

に制限される。mode mean は `tanh`、width mean は \(3\tanh(\text{raw}/3)\) に制限される (`ori_codex2.py:336-452`)。

## 4. action の力学処理

1. requested width を \([-h_{\max},h_{\max}]\) に clip。
2. \(\|u\|\le10^{-12}\) または \(|h|<10^{-4}\) なら accepted width 0 の no-op。
3. それ以外は predictor \(X_{\rm pred}=X+h u\) を bar-length constraint manifold へ射影。
4. 次を全部満たせば受理する。
   - strain \(\le10^{-8}\)
   - correction norm \(\le |h|\)
   - actual displacement \(\le1.25|h|\)
5. 満たさなければ同じ方向の幅を 0.5 倍し、再び射影する。
6. \(|h|<10^{-4}\) まで失敗すれば `correction_failed`。
7. 受理された nonzero transition に対して、corrected pre/post states の間で continuous 3-D swept collision を検査。

flatness は action の受理条件ではない。射影の縮小は constraint correction のためだけである (`ori_codex2.py:165-235`)。

3-D collision は、open motion interval \(0<t<1\) で非隣接の zero-thickness bar interiors に新たに入る proper crossing のみを禁止する。adjacent contact、endpoint contact、parallel/collinear overlap、既に存在する persistent contact は crossing にしない (`collision_3d.py:1-19`)。

## 5. 報酬設計

### 5.1 accepted valid transition の endpoint potentials

\[
C(F)=\operatorname{clip}(F,0,2),\qquad E(D)=(D-D^*)^2.
\]

基本報酬は

\[
r_{\rm base,t}
=C(F_t)-C(F_{t+1})+E(D_t)-E(D_{t+1}).
\]

これは \(\gamma=1\) で telescoping する。path length や path-integrated spread cost は使っていない (`ori_codex5.py:36-64`)。

### 5.2 continuous spread-frontier potential

初期 error を \(e_0\)、tolerance を \(\varepsilon_D=0.05\) とする。\(e_0>2\varepsilon_D\) のとき

\[
z_D(s)=\operatorname{clip}\left(
\frac{e_0-e_D(s)}{e_0-\varepsilon_D},0,1
\right),\qquad
\Phi_D(s)=0.9z_D(s).
\]

その transition reward は

\[
r_{D,t}=\Phi_D(s_{t+1})-\Phi_D(s_t).
\]

最終runは `state_potential` mode なので flatness gate \(g_F\) は掛けない。過去最大値も使わず current-state potential である (`ori_codex6.py:136-165,200-212`)。

### 5.3 joint discovery milestones

\(e_D\le0.05\) を満たした同一状態で、次の flatness threshold を episode 中に初めて越えたときだけ加点する。

| flatness threshold | once-only bonus |
|---:|---:|
| \(F\le0.5\) | 0.2 |
| \(F\le0.1\) | 0.3 |
| \(F\le0.01\) | 0.5 |

旧 hard spread milestone の bonus は 0 (`ori_codex6.py:167-182`)。

### 5.4 plane-swap spectral potential

plane の入れ替わりを表す信号は node-space projector で定義する。

\[
P=U_{:,1:k}U_{:,1:k}^\top,\qquad P_0=P(X_0),\qquad k=2,
\]

\[
\delta=\frac{\|P-P_0\|_F}{\sqrt{2k}}.
\]

境界 singular gap を

\[
a=\sigma_k,\quad b=\sigma_{k+1},\quad
\rho=\frac{a-b}{(a+b)(1+10^{-12})}
\]

とする。ただし gap が relative machine threshold 内で縮退している場合は \(\rho=0\)。spectral potential と報酬は

\[
\Phi_S=\rho\delta,\qquad
r_{S,t}=0.05\{\Phi_S(s_{t+1})-\Phi_S(s_t)\}.
\]

この成分は nonzero width の accepted valid transition だけに付く。reset state では \(P=P_0\) なので \(\Phi_S=0\) (`ori_codex6_spectral_subspace.py:38-59,96-176,278-312`)。

### 5.5 near-miss terminal quality

\[
q_F=\operatorname{clip}\left(
\frac{\log(F_{\rm far}/\max(F,10^{-3}))}
{\log(F_{\rm far}/10^{-3})},0,1\right),\qquad F_{\rm far}=1,
\]

\[
q_D=\operatorname{clip}\left(
\frac{e_{\rm far}-e_D}{e_{\rm far}-0.05},0,1\right),\qquad
q=q_Fq_D.
\]

Torus は \(e_{\rm far}=1\)、Water の最終fallbackは \(e_{\rm far}=0.12\)。Water の 0.12 は \(e_D\simeq0.115\) の境界 basin に大きな terminal quality を与えないための task-specific 調整である (`ori_codex4.py:76-99`)。

### 5.6 一般形

ordinary accepted transition の報酬は

\[
r_t=r_{\rm base,t}+r_{D,t}+r_{\rm joint,t}+r_{S,t}
+r_{\rm terminal,t}+r_{\rm rollback,t}.
\]

終端成分は次の通り。

| reason | terminal addition | rollback / retain |
|---|---|---|
| `success` | \(+10+q+0.5(1-t/H)\) | 全成分 retain |
| `timeout` | \(-1+q\) | 全成分 retain |
| `wrong_flat` | \(-2+q_D\) | episode 累積 flatness、spread-frontier、joint、spectral を cancel。target endpoint progress は retain |
| `correction_failed` | \(-10\) | spread-frontier、joint、spectral を cancel。invalid candidate 自体の progress は 0 |
| `no_modes` | \(-5\) | spread-frontier、spectral を cancel。flatness、target、joint は retain |
| ordinary crossing reject | 0 | state を戻し、過去 ledger を全てそのまま retain |
| horizon 上の crossing reject | \(-1+q(s_{\rm safe})\) | timeout reward を一度だけ適用 |

rollback の実装は `ori_codex6.py:260-342` と `ori_codex6_spectral_subspace.py:317-332`。

### 5.7 crossing は zero-reward no-op

crossing candidate が見つかった場合は、node、previous potentials、mode/geometry caches を last-safe state へ厳密に戻す。一方で `step_count` は戻さない。

- `accepted_width = 0`
- `actual_displacement = 0`
- ordinary immediate reward = 0
- base / frontier / joint / spectral / quality / rollback は全て 0
- 既に得た frontier / joint / spectral ledger は変更しない
- episode は継続し、次 step で新しい action を sample
- horizon に達した場合だけ safe state に対する \(-1+q\) を一度適用

したがって「幅 0 の action として扱う」という理解でほぼよい。ただし collision を知るまでに projection と swept collision check の計算は既に行っているため、compute cost まで 0 ではない (`ori_codex6_rejected_crossing.py:228-396`, `ori_codex6_zero_reject.py:103-235`)。

元の sampled action と `log_probability` は transition に残る。即時 reward が 0 でも、Monte-Carlo return-to-go \(G_t\) には後続報酬が含まれるため、この action の policy gradient が常に 0 になるわけではない。step 消費による機会費用は horizon と success speed bonus を通じて表れる。

## 6. 学習方法と NN 更新規則

### 6.1 rollout batch

- stochastic policy で complete episode を 8 本収集
- episode の境界を混ぜず、それぞれ backward に return-to-go を計算
- 8 episode が完了したところで Actor と Critic を各 1 回だけ更新
- minibatch 分割、GAE、bootstrap、replay buffer は使わない

各 transition の return は

\[
G_t=r_t+\gamma G_{t+1},\qquad \gamma=1.
\]

Critic を baseline とした advantage は

\[
\tilde A_t=G_t-V_\psi(s_t,M_t).
\]

これを 8 episode 内の全 transitions をまとめて mean 0、standard deviation 1 に正規化する。

### 6.2 loss

\[
L_{\rm actor}=-\operatorname{mean}_t
[\log\pi_\theta(a_t\mid s_t,M_t)A_t],
\]

\[
L_{\rm critic}=\operatorname{MSE}_t[V_\psi(s_t,M_t),G_t],
\]

\[
L_H=-\operatorname{mean}_t[H(\pi_\theta)],
\]

\[
L=L_{\rm actor}+0.5L_{\rm critic}+0.001L_H.
\]

`advantage` 内の value は detach される。Actor と Critic は separate Adam optimizer を持つが、loss を一度 backward してから、Actor gradient norm を 0.5、Critic gradient norm を 1.0 に clip し、それぞれ一度 `step()` する (`ori_codex2.py:601-667`)。

### 6.3 PPO との違い

今回使っていないもの:

- old/new policy likelihood ratio
- clipped surrogate objective
- multiple PPO epochs
- target-KL early stop
- minibatch reuse

したがって「Actor--Critic か PPO か」という問いへの答えは **Actor--Critic**。PPO も Actor と Critic を持つため名称が紛らわしいが、今回の更新規則は clipped PPO ではない。

### 6.4 learning-rate schedule

Critic LR は一定。Actor LR だけ episode schedule で段階的に下げる。これは状態連動の adaptive LR ではない。

| task | Actor LR | milestones / factors | 実際に低下する最初の update | Critic LR |
|---|---:|---|---|---:|
| Torus | \(10^{-3}\) | 1528, 4000, 7000 / 0.3, 0.1, 0.03 | ep 1536, 4008, 7008 | \(3\times10^{-4}\) |
| Water | \(5\times10^{-4}\) | 2000, 4000, 7000 / 0.3, 0.1, 0.03 | ep 2008, 4008, 7008 | \(1.5\times10^{-4}\) |

schedule は batch update 直前に適用され、条件が `episode > milestone` なので上表の境界になる (`ori_codex2.py:864-873,1149-1183`)。

## 7. 最終 task-specific 設定

| parameter | Torus | Water \(D^*=1\) |
|---|---:|---:|
| target spread \(D^*\) | 2.0 | 1.0 |
| max episode steps \(H\) | 200 | 400 |
| max requested width \(h_{\max}\) | 0.2 | 0.3 |
| flatness dimension \(k\) | 2 | 2 |
| flatness tolerance | 0.001 | 0.001 |
| spread tolerance | 0.05 | 0.05 |
| near-miss \(e_{\rm far}\) | 1.0 | 0.12 |
| batch size | 8 | 8 |
| \(\gamma\) | 1 | 1 |
| hidden size | 96 | 96 |
| entropy coefficient | 0.001 | 0.001 |
| spectral weight | 0.05 | 0.05 |
| zero crossing-reject reward | yes | yes |

Torus の主要な task-specific 対策は zero-reject と早い Actor LR decay。Water は zero-reject と near-miss spread cutoff の 1.0 から 0.12 への変更である。

## 8. 公平な random baseline

### 8.1 結論

「self-crossing したら random も再サンプリングする」は正しい。ただし **同じ step の中で当たりが出るまで無料で引き直すのではない**。

正式な matched random baseline は次の contract にする。

1. current mode basis 上で random action を 1 個だけ sample。
2. 学習 policy と同じ projection と continuous collision check を行う。
3. crossing なら exact safe state へ戻し、reward 0 の no-op とする。
4. crossing proposal も action-proposal budget と horizon を 1 消費する。
5. 次の environment step で新しい random action を sample。

これは学習側と同じ `K=1 proposal / environment step` である。

同じ step 内で collision-free action が出るまで無料再抽選すると、分布は

\[
p_{\rm random}(a\mid s,\text{non-crossing})
\]

へ変わる。これは collision checker を oracle/shield として random だけに与え、学習 policy より有利にする。実施するなら `collision-filtered random oracle` という別 baseline にし、最大再抽選回数 \(K\) を事前登録し、全 raw proposals、projection、collision checks、wall time を数えるべきである。

### 8.2 random action law

既存 `UniformRandomAgent` (`ori_codex3.py:1565-1606`) は次を使う。

- modal coefficients: \(c\sim\mathcal N(0,I)\) を unit normalize。current mode sphere 上で isotropic
- learned policy と同じ canonical hemisphere
- signed requested width: \(h\sim\operatorname{Uniform}[-h_{\max},h_{\max}]\)

これは current tangent-mode direction と width の一様 baseline であり、configuration-space volume 上の一様分布ではない。

学習効果をより厳密に分離するには、これに加えて「初期化した NN policy を freeze し、update せず sample する baseline」も有用である。

### 8.3 比較予算

primary budget は episode 数より **raw action proposals** がよい。現コードの `mechanical_steps` は実質これに相当するが、名前を `action_proposals` と明示するのが望ましい。

| budget / cost | 数え方 |
|---|---|
| action proposals | `agent.get_action` 後に `env.step` した回数。crossing no-op、tiny-width no-op、correction failure も 1 |
| projector attempts | width shrink loop の各 constraint projection。1 action で複数回あり得る |
| collision checks | projected nonzero proposalに対する checker 呼出し |
| collision pair work | broad-phase candidates と narrow checks |
| accepted moves | state が実際に変化した回数 |
| episodes | reset 回数。task horizon と early terminal に依存 |
| wall time | environment-only と、NN update を含む end-to-end を分離 |

一つの run を episode、proposal、projector、wall time の全軸で同時に厳密 censor することはできない。primary action-proposal cap を事前登録し、他の軸は同じ cap までの観測値または共通 threshold で再 censor して別々に報告する。

### 8.4 現在のコード上の gap

- 最終 staged learning runs は random search を明示的に無効化しており、最終zero-reject候補に対応する random artifacts はまだない。
- `ori_codex3.py` の inline random は、zero-reject の runtime hook 内で有効化すれば crossing 後に次 step で自動再sampleできる。ただし episode-cap中心で、strict/reject/collision内部costの記録が不足している。
- `ori_codex6_spectral_random_baseline.py` は exact-budget scaffold としてはよいが、古い spectral + crossing-penalty parent に接続されているため、最終比較にはそのまま使えない。

最小の次実装は、既存 exact-budget harness を `ori_codex6_zero_reject.py` / `ori_codex6_zero_reject_3d.py` に接続し、`UniformRandomAgent` と `K=1` を維持したまま、zero-reject identity と上記 counters を追加することである。

## 9. 推奨する比較セット

少なくとも task ごとに次の 3 群を分ける。

1. **Learned stochastic policy**: 最終 Actor--Critic policy。
2. **Matched uniform random**: current mode sphere + uniform signed width、同じ zero-reject transition、`K=1`。
3. **Frozen initial policy**: 同じ NN initialization/distribution、学習更新なし、同じ zero-reject transition。

optional な第4群として collision-filtered random oracle を置く場合は、正式 matched baseline と混ぜない。

各群で同じ strict success 定義、model files、projection parameters、3-D collision backend、task horizon、primary proposal cap、seed protocol を固定する。first-hit episode だけでなく、first-hit proposal、projector attempts、collision work、censor rate、ECDF/Kaplan--Meier/RMST と confidence interval を報告する。

## 10. 要点

- 学習法は PPO ではなく 8-episode Monte-Carlo Actor--Critic。
- action は可変次元 mode mixture と bounded signed width。
- plane swap 信号は \(0.05\Delta(\rho\delta)\) の current-state spectral potential。
- crossing candidate は state を戻し、reward 0、step を 1 消費して探索継続。
- random も crossing 後に再sampleするが、無料の inner resampling ではなく次 step で行う。
- 最終zero-reject用の厳密 random harness はまだ未接続であり、比較実験前にそこだけ実装が必要。

