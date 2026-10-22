# five-bar の教師あり方向学習

`supervised_folding.py` は、現在の節点座標と目標広がりDの1成分から、
折る方向を予測する小さなMLPです。既存のRLコードと結果は変更しません。
目的座標は教師のラベル生成にだけ使用し、NNの入力とNN経路生成には使用しません。
方向の不連続性・多峰性への特別な対策はまだ入れていません。

## 実行

このディレクトリから:

```powershell
python supervised_folding.py
```

既定は学習初期形状16、評価初期形状8、最大400step、各学習20epochです。
教師データによる初期学習をround 0とし、NN経路の収集と追加学習を3回行います。
出力は `result/experiments/supervised_five_<日時>/` です。
`--output` でも指定できますが、既存ディレクトリは上書きしません。

短い動作確認:

```powershell
python supervised_folding.py --train-starts 3 --test-starts 2 --walk-min 10 --walk-max 30 --max-steps 80 --epochs 5 --rounds 1
python -m unittest test_supervised_folding -v
```

保存した最終NNを更新せず、未知のDを含めて評価:

```powershell
python supervised_folding.py --evaluate-only result/experiments/<run> --eval-goals 2 2.5 3
```

評価専用実行は、run内に保存した形状・固定条件・幅・step上限・評価初期形状を
再利用し、新しい評価ディレクトリへ保存します。未知Dの目的座標は不要です。
任意のDが実現可能であることは保証しません。再学習・学習再開CLIは未実装です。

## データ生成と学習

1. 元の形状から運動モード内のランダム歩行を行い、拘束補正と交差確認を通った
   初期形状を作ります。各歩行は元形状から独立に開始します。
2. 学習と評価は別乱数系列で生成し、初期座標の距離が基準部材長の0.03未満の
   組は除外します。評価初期形状は全roundで固定し、ラベル生成に使いません。
3. 学習初期形状からD2/D3の教師経路を生成し、途中状態を採取します。
   基準部材長の0.03以上移動した状態と終端を残し、停滞による重複を除きます。
4. 各採取状態に対してD2/D3両方の教師方向を計算します。
5. 初期学習後はNN自身で学習初期形状から経路を生成し、その訪問状態にも
   D2/D3の教師方向を付けます。教師データと過去roundのデータは保持します。

教師方向は `mechanics_origin.py` のpredictorと同じ
`d = M @ M.T @ (target-current)` を正規化したものです。
射影勾配ノルムが1e-8以下のラベルは方向学習から除き、件数を記録します。
教師経路は既存の拘束補正・交差判定を使い、補正後も目的座標との距離が減るまで
幅を縮めます。元のpredictor/correctorと経路が完全一致する実装ではありません。
教師が停止・失敗した経路も保存し、その非ゼロ方向は局所ラベルとして使用します。
非ゼロの局所勾配があることと、目的形状へ到達できることは別です。

NNは `(3N+1) -> hidden -> hidden -> 3N` のTanh MLPです。
five-barでは16入力・15出力、seven-barでは22入力・21出力です。
各run内で節点数は固定され、異なる節点数のNNへの重み転移は行いません。
出力を現在の運動モード空間へ射影・正規化して、教師の単位方向との二乗距離を
最小化します。符号は区別します。D2/D3各半数のminibatchを復元抽出します。
移動幅・critic・分布・潜在変数は学習しません。

## 経路生成と成功判定

NNは正規化座標とDだけを見ます。拘束を扱う射影部分には既知の接続・固定条件を
使います。移動幅の上限は0.1で、拘束補正が大きい場合は縮小します。
交差候補は棄却します。同じ状態から同じ方向を繰り返す決定論的NNなので、
進めなかった時点で経路を終了し、理由を保存します。
目的座標を用いた幅探索はNN経路では行いません。

平坦度は元の `node_initial.dat` の直線からの二乗距離で正規化します。
ランダマイズした各初期形状で基準を変えません。
広がり・座標は元形状の最大部材長で正規化します。
成功は `F <= 0.001`、`abs(D-D*) <= 0.05`、部材長二乗誤差和 `<= 1e-8`。
重なり順や目的座標との一致は成功条件に含めません。
ターゲットの部材長は、入力座標の丸めを考慮して部材長基準の1e-3まで検査時に許容します。

## 保存物

- `design.json`, `model/`: 設定、乱数seed、ソースハッシュ、入力モデルのコピー。
- `starts.npz`: 学習用`train`、未学習初期形状`test`。
- `dataset.npz`: `nodes`, `projectors`, `directions`（D2,D3の順）、
  `gradient_norms`, `source_round`, `rollout_id`。
- `teacher_paths/`, `student_paths_round*/`: 経路座標のNPZ。
- `collection.csv`: 学習用経路の成功・step・終了理由。`rollout_id`は行の0始まり番号。
- `training.csv`: 学習損失、更新数、累積データ数、除外ラベル数。
- `training_epochs.csv`: epochごとの平均学習損失、累積optimizer更新数。
- `learning_curve.png`, `learning_curve.pdf`: 各round完了時に自動更新する学習曲線。
- `evaluation_summary.csv`: round・目標ごとの評価成功数、中央値、四分位。
- `checkpoint_round*.pt`, `checkpoint_final.pt`: NN、optimizer、学習乱数状態。
- `eval_initial/`, `eval_round*/`, `evaluation.csv`: 固定評価初期形状での各経路と指標。

毎roundの評価値は学習・データ収集・checkpoint選択には使用しません。
人が評価値を見て設定を調整した後の最終性能報告には、別の初期形状集合を用意してください。
この実験は同じモデルの初期形状・目標Dへの汎化です。
異なる節点数・接続への構造汎化は対象外です。

## Seven-bar: D2/D3を学習しD4を評価

seven-barにも使えるようNNの座標次元をモデルから決定します。
以前のfive-bar checkpointは座標次元15として読み込み可能です。
ターゲットがなかったため、次の節点x座標を持つ直線形状を各1種類追加しました。
y,zは全節点0。両端は-1,+1固定、6本の可動部材はほぼ単位長です。

- D2: `[-1, 0, -1, 0, 1, 0, 1]`
- D3: `[-1, 0, -1, 0, 1, 2, 1]`

学習実験:

```powershell
python supervised_folding.py --model-directory model/seven_bar_linkage --train-starts 16 --test-starts 20 --max-steps 400 --epochs 30 --rounds 3 --seed 9301 --output result/experiments/supervised_seven_d23_20260914_v1
```

学習完了後、最終NNを固定し、同じ評価初期形状でD2/D3/D4を評価:

```powershell
python evaluate_seven_d4.py result/experiments/supervised_seven_d23_20260914_v1
```

`d4_transfer/`に成功率、終端分布、全D4経路の広がり推移、代表経路、CSV、
`REPORT.md`を保存します。同じ出力先の再実行は上書きを避けるためエラーになります。
D4の教師データやNN経路を学習に追加する処理はありません。
評価前後でcheckpointファイルのSHA256が不変であることも確認します。

比較用に目的座標を知る教師も別に実行します。この教師だけが
D4座標`[-1, 0, 1, 2, 3, 2, 1]`を使い、その座標は評価出力先だけに保存します。
教師との比較は到達可能性の参考であり、同じ情報を持つ方策間の公平な比較ではありません。

### 学習量を増やす比較

評価初期形状を変えずに、学習初期形状64・データ追加10回へ増やす実験:

```powershell
python supervised_folding.py --model-directory model/seven_bar_linkage --train-starts 64 --test-starts 20 --test-starts-from result/experiments/supervised_seven_d23_20260914_v1/starts.npz --max-steps 400 --epochs 30 --rounds 10 --seed 9301 --output result/experiments/supervised_seven_d23_20260914_extended_v1
python evaluate_seven_d4.py result/experiments/supervised_seven_d23_20260914_extended_v1
python compare_seven_training.py result/experiments/supervised_seven_d23_20260914_v1 result/experiments/supervised_seven_d23_20260914_extended_v1
```

`--test-starts-from`は保存済みNPZの`test`を再利用します。個数、座標形状、
部材長、固定条件を検査し、新しい学習初期形状の生成時に近接を除外します。
入力NPZのSHA256も記録します。前のrunやcheckpointは変更しません。
比較スクリプトは評価形状の完全一致を確認し、`comparison_previous.png/pdf/csv`と
形状ごとの成功・失敗の変化`paired_success_changes.csv`を保存します。

## 学習曲線

保存済みのCSVから、再学習せずに図を生成できます。

```powershell
python plot_supervised_learning.py result/experiments/<run>
```

横軸は累積optimizer更新数です。教師方向との二乗距離、未学習初期形状での
成功率、蓄積データ数、終端平坦度、終端広がり誤差、経路step数を表示します。
初期NNの評価は更新数0です。成功率には実数（3/4など）を付けています。
終端指標とstepの帯はrollout間の25–75%四分位範囲で、信頼区間ではありません。
step数には失敗経路も含むため、短くなったことだけでは改善を意味しません。

今後の学習ではepoch平均損失を保存します。既存の初回試行にはその記録がないため、
round平均損失3点を表示します。未記録の途中経過を補完していません。
roundごとにデータセットが変わるので、損失だけで汎化を判断せず、
固定した評価初期形状での成功率と終端指標を合わせて確認します。

## 初回の実行確認（2026-09-14）

`result/experiments/supervised_pilot_20260914_v1/` に保存しています。
実行条件は以下です。評価成功を使った自動選択や早期終了は行っていません。

```powershell
python supervised_folding.py --train-starts 8 --test-starts 4 --max-steps 200 --epochs 20 --rounds 2
```

| NN | 未学習初期形状でのD2成功 | 未学習初期形状でのD3成功 |
|---|---:|---:|
| ランダム初期化 | 0/4 | 0/4 |
| 教師経路で初期学習（round 0） | 0/4 | 4/4 |
| NN訪問状態を追加（round 1） | 3/4 | 4/4 |
| 再度追加（round 2） | 3/4 | 4/4 |

教師は学習初期形状8個の両目標、計16経路すべてで成功しました。
採取状態は570→1,311→1,846件、最終教師方向は3,692件です。
最終D2評価の残り1経路は交差候補の棄却で終了しました。
この比較ではデータ追加と学習更新数の両方が増えており、改善がデータ追加だけの
効果であるとは断定できません。1seed・4評価初期形状の動作試行であり、
汎化性能の確定値や未知D・未知構造の成功を示すものではありません。

`test_supervised_folding.py` の5テストで、教師の降下方向、拘束維持、
ゼロ勾配除外、初期形状生成の再現性と分離、データ保持と両目標ラベル、
方向学習の改善、目的座標なしのNN経路生成を確認しました。
