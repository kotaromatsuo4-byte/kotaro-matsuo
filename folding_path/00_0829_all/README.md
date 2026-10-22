# 初回発見の学習とcheckpoint評価

`python main.py` で、初回strict successまたは最大10,000 episodeまで学習します。
初回成功があれば、初期NNとそれまでの各更新後NNを100 episodeずつ独立評価し、2枚のPNGを出力します。
成功episodeを含む追加更新はしません。成功を生成したNNまでが評価対象です。

## 実験設定

`main.py`冒頭のグローバル変数を変更してください。

- `TASK_NAME`: `five_d2`, `five_d3`, `torus_d2`, `water_d1`, `water_sqrt2`
- `SEED`: 整数。`None`ならtaskの既定seed。
- `GEOMETRY_DIMENSION`: `2`はxz平面の衝突判定と最小二乗直線、`3`は空間衝突判定と最小二乗平面。`None`ならfiveは2、他は3。
- `SPECTRAL_WEIGHT`: 既定0.05。
- `MAX_EPISODES`: 既定10,000。10,000を超える値は学習側で10,000に制限。
- `EVALUATION_ROLLOUTS`: 既定100。
- `RUN_RANDOM`: 既定False。TrueでRLと評価の後にRandomを実行。

最大step、step幅、LR、batch数などは同じファイル内の`TASKS`/`COMMON`で指定します。
最終の初回発見実験に合わせ、actor損失とentropy項は「batch内episode数 × 固定最大step数」で割り、初回成功までLRは一定です。
joint報酬はpotential差、spectral報酬も従来のpotential差です。

モデルはこのフォルダ内の`model/`から読みます。2Dも節点座標は3列(x,y,z)で、y一定のモデル・拘束条件を使用してください。
`mechanics.compute_flatness_potential`が直線と平面を同じSVD式で計算し、`mechanics.has_proper_self_crossing`が2D/3Dを切り替えます。
ファイル構成は既存の7つのPythonファイルのままです。

## 衝突と終了条件

衝突した候補は採用せず、そのstepの衝突報酬は0、同じ状態から次のstepへ進みます。
rejectも試行1stepとして数えます。最大stepに達した場合は通常のtimeout終端報酬でepisodeを終了します。
wrong-flat、モード消失、数値的な修正失敗などのepisode終了条件は旧版と同じです。
学習全体を成功として止めるのはstrict successのみです。

## 保存物

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

## 実行例・中断後の評価再開

```powershell
python main.py --task five_d3 --seed 9102
python main.py --task torus_d2 --seed 8101
python main.py --evaluate-only "result/five_d3/rl/seed9102_<日時>"
```

評価はcheckpointごとに保存するため、`--evaluate-only`で完了済みcheckpointを読み込み、残りを評価できます。
学習が10,000 episodeで未発見なら評価図は生成せず、学習結果とNNを保存します。
評価には `(更新回数+1)×100` episodeが必要です。初回発見後の評価時間は学習時間より長くなる場合があります。

## 実装時の確認（2026-09-09）

five D3・seed9102を実際に学習させ、旧結果と同じ初回成功9 episode・NN更新1回・累積3,501 stepを確認しました。
初期NNと更新1回後のNNを各100 rollout評価し、図の生成と保存済み評価からの再開を確認しました。
この確認runでnear-miss到達率は0/100から22/100、Q*中央値は0.0462から0.2645でした。
2D/3Dの衝突reject（状態不変・報酬0・継続）、最大step到達時のtimeout、未発見時のepisode上限終了も確認しています。
全5taskの10,000 episode学習は実施していません。
