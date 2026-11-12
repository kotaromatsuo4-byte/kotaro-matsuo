# Repository Overview

松尾修論一括管理ファイルです。

## Folder Structure

```text
program/
├── configuration_space/    # 1章. 提案手法と川口型との比較用
│   ├── 2D.py                ## 4-bar-linkage configuration_space
│   ├── 3D.py                ## 5-bar-linkage configuration_space
│   ├── landscape.py         ## 図表作成用
├── module/                 # Python modules
│   ├── env.py               ## RL environment
│   ├── agent.py             ## RL agent
│   ├── policy.py            ## RL policy
│   ├── collision.py         ## Collision detection (2D/3D)
│   ├── SVD.py               ## SVD (dense/sparse)
│   └── mechanics.py         ## Mechanics, predictor/corrector (2D/3D)
├── every_step_opt.py       # Every-step optimization main (Kawaguchi's method)
├── Random.py               # Random search main
├── RL.py                   # Reinforcement learning main
├── mechanism_test.py       # mechanism / soft mode calculation test
└── README.md               # Documentation
```

モデルはこのフォルダ内の`model/`から読みます。2Dも節点座標は3列(x,y,z)で、y一定のモデル・拘束条件を使用してください。

## 0.mechanism_test

mechanism / soft mode calculation test

### 0.1.詳細

`mechanism_test.py`冒頭のグローバル変数を変更してください。

### 0.2.保存物

`result/<task>/mechanism_tes/boundary_condition/`に実行ごとに保存します。

- `displace_mode_*.dat`: mechanism / soft mode

## 1.every_step_opt

川口型の逐次計量最小化。計量の最小化問題(solver : CMAES)により各ステップのモード係数を出す。

### 1.1.詳細

`every_step_opt.py`冒頭のグローバル変数を変更してください。

### 1.2.保存物

`result/<task>/every_step_opt/<日時>/`に実行ごとに保存します。

- `config.json`　: 実行設定。
- `every_step_opt_success/step*.dat`: 初回成功経路。

## 2.random

randomで各ステップのモード係数を出す。

### 2.1.詳細

`every_step_opt.py`冒頭のグローバル変数を変更してください。

### 2.2.保存物

`result/<task>/random/<日時>/`に実行ごとに保存します。

- `config.json`　: 実行設定。
- `random_success/step*.dat`: 初回成功経路。

## 3.RL

提案手法。NNの出力で各ステップのモード係数を出す。

### 3.1.詳細

`RL.py`冒頭のグローバル変数を変更してください。

- `TASK_NAME`: `five_d2`, `five_d3`, `torus_d2`, `water_d1`, `water_sqrt2`
- `SEED`: 整数。`None`ならtaskの既定seed。
- `GEOMETRY_DIMENSION`: `2`はxz平面の衝突判定と最小二乗直線、`3`は空間衝突判定と最小二乗平面。`None`ならfiveは2、他は3。
- `SPECTRAL_WEIGHT`: 既定0.05。
- `MAX_EPISODES`: 既定10,000。10,000を超える値は学習側で10,000に制限。
- `EVALUATION_ROLLOUTS`: 既定100。
- `RUN_RANDOM`: 既定False。TrueでRLと評価の後にRandomを実行。

最大step、step幅、LR、batch数などは同じファイル内の`TASKS`/`COMMON`で指定します。
`mechanics.compute_flatness_potential`が直線と平面を同じSVD式で計算し、`mechanics.has_proper_self_crossing`が2D/3Dを切り替えます。

### 3.2.保存物

`result/<task>/rl/seed<seed>_<日時>/`に実行ごとに保存します。

- `learning_curve.png`: episode rewardとterminal flatnessの生値・移動平均。
- `best_episodeXXXXX/step*.dat`: best rewardを更新した経路。
- `first_strict_success/step*.dat`: 初回成功経路。
- `config.json`, `summary.json`, `episode_metrics.csv`: 実行設定とepisode/更新/step数。
- `policy_checkpoints/updateXXXXX.pt`, `policy_checkpoint_index.csv`: 初期NN（update 0）と全更新後NN。
- `checkpoint_evaluation/updateXXXXX.csv`: 各NNの100 rollout評価。
- `checkpoint_evaluation/checkpoint_summary.csv`: 評価集計。
- `checkpoint_evaluation/near_miss_reach_probability.png`: near-miss到達率（黒線）と95% Wilson区間（薄黒帯）。
- `checkpoint_evaluation/best_quality.png`: Q*の中央値（黒線）と25–75%四分位（薄黒帯）。

near-miss到達とは、1 rollout中に一度でも `F <= 0.1` と `|D-D*| <= spread_tolerance`（既定0.05）を同時に満たすことです。
100 rolloutのうち何本が到達したかを数えます。全stepの状態数の割合ではありません。
Q*は `max_t(qF_t*qD_t)` で、qFとqDは必ず同じstepの値を使います。初期状態を除く各step後の状態を評価し、従来の評価器と合わせています。
横軸はNN更新回数、図は黒・Times New Roman・凡例なし、PDFは出力しません。
評価のseedは学習と別にし、checkpoint間では同じ100 seedを使います。評価で学習やNN更新は行いません。
