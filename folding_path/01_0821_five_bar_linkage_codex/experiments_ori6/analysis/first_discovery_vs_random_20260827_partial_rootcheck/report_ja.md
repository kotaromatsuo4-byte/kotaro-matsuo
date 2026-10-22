# 5 task：学習とランダム探索の初回発見比較

生成時刻（UTC）: `2026-08-27T05:40:19.110715+00:00`  
状態: **pending**

## 判定表

一次時計は累積 action proposal 数です。各 task の打切り上限は、指定した primary learning run の初回 strict success と完全に同じ値です。

| Task | learning初回 | random hit | randomが先 | random capped RMST | RMST/learning | 状態 |
|---|---:|---:|---:|---:|---:|---|
| Five-bar D=2 | 233,824 | 0/30 | 0/30 | 233,824 | 1.000 | no_random_trial_hit_before_primary_learning |
| Five-bar D=3 | 1,033 | 0/30 | 0/30 | 1,033 | 1.000 | no_random_trial_hit_before_primary_learning |
| Water D=sqrt(2) | 17,324 | 6/10 | 6/10 | 10,185.500 | 0.588 | at_least_half_random_trials_hit_before_primary_learning |
| Torus D=2 | 636,198 | 0/0 | 0/0 | — | — | pending_random |
| Water D=1 | 126,319 | 3/3 | 3/3 | 38,082.333 | 0.301 | partial_random |

`randomが先` は random trial の first hit が primary learning hit より**厳密に小さい**場合です。同値は tie として別計数しています。capped RMST は未発見trialをtask固有capに置いた観測時間の平均です。

## 解釈上の制約

- Five-bar D=2/D=3 と Water D=sqrt(2) はtask transitionと整合するhistorical uniform authorityを使います。ただしlearningはhistorical ori5の各1 runだけなので、結果は記述的比較であり、学習法一般の統計的優越を示す検定ではありません。
- Torus D=2 と Water D=1 は3 learning seedsをcohortとして併記しますが、primary run/capはscreening後に選ばれています。post-selection biasがあるので探索的結果です。
- random actionは現在mode上の等方方向とsigned widthの一様サンプルです。方策NNの初期分布そのものとの同一性は主張しません。

## crossing と報酬の不変性監査

The uniform agent samples from its fixed action law and performs no parameter update, so rewards do not influence any later random action. Paired audits established exact nodes, actions, RNG and termination for water seed6101 over 400 steps and 20 rejections; exact forced-horizon behavior for torus and water; and exact geometry plus terminal F/D for the 257-action, 9-rejection saved water first hit. The random harness restores the last safe state and consumes one action proposal before the next sample. Its legacy rejected reward is intentionally not identical to the learned zero-reward no-op, and this difference cannot affect a reward-blind uniform agent. Torus2/water1 are therefore geometry/action-proposal matched exploratory comparisons with reward identity intentionally excluded.

監査結果: `{"checked": true, "random_method_uniform_only": true, "action_laws": ["isotropic current-mode direction and uniform signed width"], "crossing_policies": ["reject_candidate_and_continue"], "candidate_never_committed_on_crossing": true, "crossing_consumes_one_step": true, "uniform_agent_reward_blind": true, "random_reward_sha256_values": ["0c148e1eaada0da0d7ac1781abd2157fc36b516a12459e194139e843623794de"], "random_transition_semantics_sha256_values": ["19200dc483be290975ebb9aa46545a48b09c3173bc96042942a2617d2ea198cc"], "paired_semantics_audit": {"water_seed6101_400_steps": "exact nodes/actions/RNG/termination across 20 rejected crossings", "forced_horizon_torus_and_water": "exact", "water_saved_first_hit_replay": "257 actions and 9 rejected crossings: exact geometry and terminal F/D; only rejected reward differs", "scope": "campaign protocol audit supplied independently of this artifact analyzer"}}`

## 完全性

- Five-bar D=2: random seeds=[3000003, 3000004, 3000005, 3000006, 3000007, 3000008, 3000009, 3000010, 3000011, 3000012, 3000013, 3000014, 3000015, 3000016, 3000017, 3000018, 3000019, 3000020, 3000021, 3000022, 3000023, 3000024, 3000025, 3000026, 3000027, 3000028, 3000029, 3000030, 3000031, 3000032]; missing=[]; primary learning integrity=True
- Five-bar D=3: random seeds=[3000003, 3000004, 3000005, 3000006, 3000007, 3000008, 3000009, 3000010, 3000011, 3000012, 3000013, 3000014, 3000015, 3000016, 3000017, 3000018, 3000019, 3000020, 3000021, 3000022, 3000023, 3000024, 3000025, 3000026, 3000027, 3000028, 3000029, 3000030, 3000031, 3000032]; missing=[]; primary learning integrity=True
- Water D=sqrt(2): random seeds=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]; missing=[]; primary learning integrity=True
- Torus D=2: random seeds=[]; missing=[6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110]; primary learning integrity=True
- Water D=1: random seeds=[6101, 6102, 6103]; missing=[6104, 6105, 6106, 6107, 6108, 6109, 6110]; primary learning integrity=True

## 成果物

- `task_summary.csv`: task単位の主要統計
- `trial_level.csv`: learning/randomのseed単位event/censor記録
- `first_discovery_survival.png`: task別の未発見率曲線
- `normalized_summary.png`: normalized RMST と random-faster率
- `input_manifest.json`: 読み込んだauthority fileのSHA-256
