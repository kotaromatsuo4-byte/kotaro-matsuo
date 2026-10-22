# 学習済みNNを動かす

作業ディレクトリを`00_0822_all`にする。`rollout_saved_student.py`は保存重み、構造、初期節点座標、目標Dから経路を計算する。
目的形状の節点座標は入力しない。モードと拘束条件は指定した構造の力学モデルから計算する。

## 教師4経路・174ラベルで学習したモデル

nearest教師と旧データ混合を使ったモデルは、5seed平均でD2=47.5%、D3=75.0%、D4=53.125%。
以下は同じ初期形状から3目標へ折り分けた例のD4を再実行する。

```powershell
$study = "result/experiments/generalization_5h_20260914"
python rollout_saved_student.py "$study/nearest_branch_u2000_test/seed5501_replay_n04.pt" --structure seven --goal 4 --initial "$study/multi_goal_nearest_example/node_initial.dat" --output result/replay_nearest_d4
```

同じ重み・初期形状でD2は55step、D3は62step、D4は44stepで成功した。
[折り分けの図](multi_goal_nearest_example/multi_goal.png)。これは選択した成功例で、全初期形状での成功を保証しない。

## 同じNNでD2・D3・D4を折り分ける

以下は旧D2/D3データを混ぜてD4へ適応した1つのNNを使う。
3目標に成功した同じtest初期形状を、図示用に選んである。
seed5501は説明用で、test成績が最良のseedを選んだものではない。

```powershell
$study = "result/experiments/generalization_5h_20260914"
python rollout_saved_student.py "$study/modal_replay0.5_u2000_test/seed5501_pretrained_n16.pt" --structure seven --goal 2 --initial "$study/multi_goal_example/node_initial.dat" --output result/replay_example_d2
python rollout_saved_student.py "$study/modal_replay0.5_u2000_test/seed5501_pretrained_n16.pt" --structure seven --goal 3 --initial "$study/multi_goal_example/node_initial.dat" --output result/replay_example_d3
python rollout_saved_student.py "$study/modal_replay0.5_u2000_test/seed5501_pretrained_n16.pt" --structure seven --goal 4 --initial "$study/multi_goal_example/node_initial.dat" --output result/replay_example_d4
```

この初期形状ではD2は45step、D3は43step、D4は60stepで成功した。
各目標で重みを更新していない。新しい初期形状で同じ成功を保証するものではない。
[経路図](multi_goal_example/multi_goal.png) / [選択規則](multi_goal_example/selection.json)

## barだけで学習したNNをtorusへ直接適用する

```powershell
$study = "result/experiments/generalization_5h_20260914"
python rollout_saved_student.py "$study/graph_source_mixed_seed5501_u5000.pt" --structure torus --goal 2 --initial "$study/replay_checks/torus_initial.dat" --output result/replay_example_torus_zero
```

この保存重みはfive/sevenのD2/D3だけで学習したもの。この初期形状では123stepで成功を再現した。
全5seed・通常testの平均成功率は16.25%であり、上のコマンドは成功例の再現用。
[経路とアニメーション](torus/zero_shot_example/provenance.json)

## 自分の初期形状を使う

`--initial`にn節点×3座標のカンマ区切りファイルを指定する。
指定した構造の接続・固定条件・部材長を満たす座標を使う。
省略すると、その構造の`model/.../node_initial.dat`を使用する。

- `--structure five`: five_bar_linkage
- `--structure seven`: seven_bar_linkage
- `--structure torus`: 4unit_torus
- `--structure torus6`: 6unit_torus

固定節点数のModalStudentを別の節点数へそのまま適用することはできない。
GraphModalStudentは節点数を変更できるが、成功率は別途評価する必要がある。
上の実行用スクリプトは通常のmodal、categorical modal、graph modal、graph projectedに対応する。低ランクadapter用ではない。

出力先には`path.npz`、`node_final.dat`、各stepの`metrics.csv`、成否・重みハッシュ付きの`result.json`を保存する。
以前の結果を保つため、出力先が空でなければ停止する。再実行時は新しい出力先を指定する。

実験全体の平均・ばらつき・対照条件は[RESULTS_JA.md](RESULTS_JA.md)を参照。
