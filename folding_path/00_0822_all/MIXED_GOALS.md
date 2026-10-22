# five D2 / D3 混合学習

## 実行

```powershell
python RL_mixed_goals.py --episodes 128 --eval-every 128 --rollouts 100 --seed 9201 --workers 2
```

本学習では `--episodes 10000` を指定します。episode数はbatch数（既定8）の倍数です。
初回成功では止めず、指定episode数まで学習します。
結果は `result/experiments/mixed_goals_seed<seed>_<日時>/` に保存します。

```powershell
# 学習のみ実行して、評価は後から行う
python RL_mixed_goals.py --episodes 1024 --seed 9201 --train-only
python RL_mixed_goals.py --evaluate-only "result/experiments/<run>" --workers 2
```

評価はcheckpoint・目標ごとのCSVに保存し、再実行時は完了分を再利用します。
`--evaluate-only`では保存済みdesign.jsonの条件とNNを使用します。
`--no-goal-input`では追加入力を常に0にした比較用学習を実行できます。同じNN規模で目標入力の有無を比較するための設定です。

## 学習設計

- NNはactor/critic各1組を両目標で共有します。
- 観測は従来の正規化座標15成分に、無次元の目標広がりD*を1成分追加した16成分です。
- モードと物理変位は従来どおり15成分です。目標は変位成分に含めません。
- actorのモード重み・幅出力、criticの両方が目標入力を受け取ります。
- batch内はD2が4 episode、D3が4 episode。順番は専用乱数生成器でシャッフルします。
- 両目標でstep幅0.1、最大400 step、spectral重み0.05を使用します。
- actor LRは5e-4、critic LRは1.5e-4で一定。actor/entropy損失の分母はbatch内episode数×400です。
- 報酬・衝突reject・strict成功判定は既存envを使用します。
- 目標ごとにenvを作り、初期spread偏差などの報酬基準もそれぞれ正しく初期化します。
- 全更新後NNを保存します。評価対象は初期NN、eval-everyごとのNN、最終NNです。
- 旧タスク用NNのgoal_size既定値は0です。旧checkpointのparameter形状を維持します。

## 評価

学習完了後にNNを固定し、各checkpointについてD2指定100 rollout、D3指定100 rolloutを実行します。
評価中はNN更新を行いません。評価seedは学習seedと別で、checkpoint間・目標間では同じ100個を使います。
Randomも同じ初期形状、step幅、最大step数で各目標100 rollout評価します。
学習中の初回成功と、固定方策の評価成功率は別の情報として保存します。

目標ごとの評価指標:

- strict success率と95% Wilson区間。
- near-miss-region到達率と95% Wilson区間。経路中に一度でも F<=0.1 と |D-D*|<=0.05 を同時に満たしたrolloutの割合。
- Q*=max_t(qF_t*qD_t) の中央値・25–75%四分位。qF,qDは同一stepのものを使います。
- 終端spreadとflatnessの散布図。指定目標と異なる平坦形状への到達も確認します。

グラフの黒実線はNN、黒破線はRandomの中心値、帯はNNの区間です。
strict/near-missの帯は信頼区間、Q*の帯はrollout間の四分位範囲で、意味が異なります。
目標ごとに図を分け、学習曲線は横軸を混合学習全体のepisode番号にします。
生値に重ねる移動平均は各目標の直近20 episodeです。
評価成功率は最大400 stepの1 rollout当たりの率です。初回発見までの累積stepに関する5seed比較を代替するものではありません。

## 主な保存物

- `design.json`: 設定、目標別env設定、ソースSHA-256。
- `episode_metrics.csv`, `updates.csv`, `training_summary.json`: 目標別の経験数・step数・初回成功・更新履歴。
- `checkpoints/`, `checkpoint_index.csv`: 初期NNと全更新後NN（optimizer・Torch乱数・目標順序乱数状態も保存）。
- `<task>/best_episode*/step*.dat`, `<task>/first_strict_success/step*.dat`: best更新時と初回成功の経路。
- `evaluation/*.csv`: 各NN・目標・Randomのrollout結果。
- `evaluation_summary.csv`: 目標別評価集計。
- `learning_curve.png`: 目標別reward/terminal flatness推移。
- `five_d2_*.png`, `five_d3_*.png`: 目標別strict、near-miss、Q*。
- `terminal_distribution.png`: 最終NNの目標別到達形状の分布。

この実装は同一five barの複数目標学習です。未知構造への汎化やGNN化はまだ行っていません。
専用NNとの公平な学習予算比較、複数学習seed、目標入力なしの対照実験は次段階です。

## Advantage診断と目標別centering

`python mixed_diagnostics.py result/experiments/<baseline-run>`は保存checkpointから次の8 episodeを再現し、目標別return・critic予測・生/混合正規化/目標別centeringのadvantageを記録します。episode結果と次のNNが保存記録と完全一致することも検証します。actor勾配の目標間cosineはentropy項を除くclip前の勾配で、実際のAdam更新の角度ではありません。

`python RL_mixed_goals.py --advantage-centering goal --diagnostics --episodes 10000 --d2-step-budget 202970 --eval-every 256 --seed 9201`

初期化から混合学習し、advantageから引く平均だけを目標別にします。標準偏差は従来どおり全transition共通、actorの固定horizon正規化・critic loss・entropy・報酬・LRは変更しません。目標別標準偏差による再重み付けではありません。`--diagnostics`で更新ごとにepisode別の統計を`advantage_diagnostics.csv`へ保存します。負のadvantageがあること自体は異常ではなく、goal centeringの成功も保証しません。省略時は従来のbatch全体centeringです。

## Critic学習不足の診断・対照

`python check_critic_fit.py result/experiments/<baseline-run>`はU128の次の8 episodeから4 stepおきに状態・returnを採取し、各目標3 episodeを学習、1 episodeを検査に分け、同じcritic構造を固定データで2,000 minibatch学習します。本番checkpointは変更しません。これは局所的な回帰能力の検査で、RL成功や未知形状汎化を意味しません。

`--critic-replay-steps 50`で、通常のactor/critic更新後に現batchだけのデータを使いcriticを50回追加学習します。各回64 transitionを復元抽出し、critic LR・loss係数・clip値は維持。独立Torch Generatorを使い、行動サンプリングの乱数を消費しません。actor構造・報酬・actor更新回数は変更しません。追加学習の成果は次batchのadvantage計算から反映されます。`--advantage-centering goal`と併用して、goal centering単独版との1要因比較を行います。環境stepは同予算ですが計算量は増えます。actor更新U回に対しcritic更新は51U回です。この版の再開CLIは現時点では無効です。

## 単独学習コマンド

`python RL_mixed_goals.py --task five_d2 --episodes 512 --seed 9201 --eval-every 128`

D*入力付きの同じNNを初期化し、D2のみで学習・評価します。`--task`省略時は従来のD2/D3混合です。単独ではbatch内8 episodeすべてD2、混合では4+4です。512単独episodeは1024混合episodeとD2経験数が一致しますが、更新回数は64対128となり、総step数も一致しません。目標ごとの経験数をそろえた対照であり、すべての計算予算を一致させた比較ではありません。

## 保存状態からの延長方法

### 成功率に応じた探索強化（任意）

`--adaptive-exploration`は各目標の直近32学習episodeの成功率pを使い、batch開始時にe=max(0,1-p/0.2)を決めます。未経験目標はp=0。warm-startでは事前学習の履歴も参照し、評価結果は使いません。各目標に同じ式を適用します。

モード係数と符号付きstep幅の潜在Gaussianについて、平均muをmu/(1+e)、分散vをv+eへ変換します。NN構造・報酬・LR・batch内4+4・物理step上限は維持します。変換した分布でサンプリングし、その同じ分布のlog-probabilityとentropyを用いて更新するため、元分布のlog-probabilityを誤って使いません。単なる分散下限ではなく、平均の弱化と分散加算を組み合わせた探索制御です。

eはbatch中固定、成功率に応じて再び上昇することもあります。環境の観測に成功履歴は追加せず、学習時の外部controllerとして扱います。`episode_metrics.csv`に事前成功率と実際のeを記録します。無効時は既存の分布計算をそのまま使います。

固定checkpoint評価ではe=0とし、探索強化なしのNN自体の性能を測ります。学習中のnear-miss増加だけではNN改善とは判断しません。学習する方策分布と評価する分布は異なる点に注意してください。`test_exploration.py`で制御則・サンプリングとlog密度の一致・物理step上限・勾配を確認できます。

`python RL_mixed_goals.py --warm-start result/experiments/<single-goal-run> --task mixed --episodes 1024 --eval-every 128 --diagnostics`

単独学習runの最終NN・actor/critic optimizer・行動および目標順序の乱数状態を引き継ぎ、D2/D3混合へ切り替える診断実験です。LR・batch・goal入力・advantage centeringは元runと同じ設定を指定します。元runは変更しません。切替後episode/update/stepは0から記録し、事前学習分はdesign.jsonのpretraining_*へ別記します。update00000はランダム初期NNではなく事前学習済みNNです。既習タスクの環境設定を引き継ぎ、新規タスクは現在の共通設定を使います。未知目標への追加学習なしの汎化評価とは区別してください。critic追加更新版との併用は未対応です。

`--d2-step-budget 202970`を指定すると、再開前を含むD2累積stepが指定値以上になった最初のbatch更新後に終了します。episode上限は`--episodes`で指定します。D2/D3の4+4構成を維持するため、最大でD2の4 episode分未満の超過が生じます。`training_summary.json`に実際のepisode数、上限、終了理由を記録します。学習条件や乱数消費を変えず、終了判定のみを追加しています。

`python RL_mixed_goals.py --resume result/experiments/<run> --episodes 1024 --eval-every 256`

完了済みrunの最終更新から、合計1024 episodeへ延長します。学習seed・LR・batch・環境設定は保存設定を継承し、元runを変更せず新しいrunへ保存します。既存のcheckpointと評価結果も引き継ぎます。評価は初期・再開元の最終・指定間隔・延長後最終で行います。再開時のseed/LRなどのCLI指定は使用しません。短い実環境テスト（4 step/episode、16 episode）で、一括学習と8 episodeからの再開の重み・optimizer・乱数・全episode指標が完全一致することを確認しました。
