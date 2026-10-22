# 再実行

作業ディレクトリは00_0822_all。今回の既定出力先は上書きしない。
スクリプトの既定期限は今回の実験終了時刻なので、後日実行するときは新しい期限・出力先を設定する。

```powershell
$env:FOLDING_DISCOVERY_OUTPUT = Join-Path (Get-Location) 'result/experiments/discovery_transfer_rerun'
$env:FOLDING_DISCOVERY_DEADLINE = [DateTimeOffset]::UtcNow.AddHours(5).ToUnixTimeSeconds().ToString()
python discovery_transfer.py pretrain --architecture dense --seed 5601 --updates 5000
python discovery_transfer.py run --task seven_d4 --architecture dense --arm scratch --seed 5701 --source-seed 5601 --budget 51200
python discovery_transfer.py run --task seven_d4 --architecture dense --arm finetune --seed 5701 --source-seed 5601 --budget 51200
python discovery_transfer.py run --task seven_d4 --architecture dense --arm frozen --seed 5701 --source-seed 5601 --budget 51200
```

torusへの転移では事前学習のarchitectureをgraphにし、runのtaskをtorus_d2、architectureをgraphにする。
既知課題の目的座標から生成済みの教師データだけを事前学習に使い、runは未知課題の目的座標を読み込まない。
arm=deterministicは方策の平均行動で1経路だけ試す。arm=frozenは同じ方策から確率的に繰り返し探索する。

同じ出力ディレクトリ・条件が既に存在すると停止する。再実行は新しい出力先またはtagで行う。
現在のsource_data.npzは以前の既知課題の教師経路を再利用している。オンライン探索の時間には、その既存データ生成コストを含めていない。

## sevenの経験だけでfive D3を初回発見する例

今回保存したチェックポイントを読み、出力は新しいディレクトリに保存する。
次の2条件は同じ初期形状・行動サンプリングseedで比較できる。

```powershell
$study = Join-Path (Get-Location) 'result/experiments/discovery_transfer_5h_20260914'
$env:FOLDING_DISCOVERY_OUTPUT = Join-Path (Get-Location) 'result/experiments/seven_to_five_rerun'
$env:FOLDING_DISCOVERY_DEADLINE = [DateTimeOffset]::UtcNow.AddHours(1).ToUnixTimeSeconds().ToString()
$checkpoint = Join-Path $study 'source/graph_seven_seed5601/best.pt'
python discovery_transfer.py run --task five_d3 --architecture graph --arm frozen --seed 5701 --source-seed 5601 --budget 6400 --checkpoint $checkpoint
python discovery_transfer.py run --task five_d3 --architecture graph --arm scratch --seed 5701 --source-seed 5601 --budget 6400
```

frozenは学習せず確率的に行動する。平均行動だけなら`--arm deterministic`、従来RLで更新するなら`--arm finetune`。
保存済み初期形状を変えるには、たとえば`--initial (Join-Path $study 'fresh_bar_inputs/start00.dat')`を追加する。
その比較では両方のarmに同じinitialを与える。

事前学習自体を再実行する場合:

```powershell
python discovery_transfer.py pretrain --architecture graph --label graph_seven --seed 5601 --updates 20000 --seven-only --data-root (Join-Path $study 'seven_only_source')
```

これは保存済みのsevenだけの教師データから新しいactorを学習する。未知fiveの教師や、過去のRL actorの重みは初期値にしない。

## NNとRLの接続

- `best.pt`はactor重みと教師データの由来を持つ。critic重みを転移するファイルではない。
- graphの入力は、正規化した現在節点位置、現在のraw SVDモード、目標D、接続・固定条件の特徴、平坦化する部分空間の次元。
- 出力はモード係数の分布と、符号付きstep幅の分布。既存agent.pyが物理方向と幅へ変換する。
- 別構造では`set_geometry`で接続と固定条件を設定する。これらは未知課題の問題定義であり、目的座標ではない。
- projectedは同じ入出力インターフェースを持つ別actor。`--architecture projected`と対応するcheckpointを組み合わせる。

同じrunを延長するときは`--resume <既存runディレクトリ>`を指定できる。
成功前で、予算による停止が完了したtimeoutエピソードと8エピソード更新境界に一致する場合だけ再開できる。
環境設定・初期形状・source checkpoint・分散条件を変更した再開は拒否する。

## 未知seven D4を発見したモデル

以下のsourceにはfive/seven D2・D3とwater D1・√2だけが含まれる。D4教師は含まれない。
同じ環境・ライブラリ・乱数条件ではfrozenは612提案で初回成功した。scratchの対応条件も保存する。

```powershell
$study = Join-Path (Get-Location) 'result/experiments/discovery_transfer_5h_20260914'
$env:FOLDING_DISCOVERY_OUTPUT = Join-Path (Get-Location) 'result/experiments/unseen_d4_rerun'
$env:FOLDING_DISCOVERY_DEADLINE = [DateTimeOffset]::UtcNow.AddHours(1).ToUnixTimeSeconds().ToString()
$checkpoint = Join-Path $study 'source/projected_water_seed5601/best.pt'
python discovery_transfer.py run --task seven_d4 --architecture projected --arm frozen --seed 5701 --source-seed 5601 --budget 6400 --checkpoint $checkpoint
python discovery_transfer.py run --task seven_d4 --architecture projected --arm scratch --seed 5701 --source-seed 5601 --budget 6400
```

source seed5603 / search seed5703では1,729提案、seed5602 / 5702では6,400提案まで未成功。成功した重みだけでなく、全3個を比較する。
teacher学習も再実行する場合は、新しい出力先のまま次を実行する。

```powershell
python discovery_transfer.py pretrain --architecture projected --label projected_water --seed 5601 --updates 20000 --include-water --data-root (Join-Path $study 'water_source')
```

D2/D3のみの教師に限定する追試は、上記から`--include-water`を外し、labelを`projected_bars`にする。同じ保存済みbar教師を使う。

D2/D3だけで学習した保存モデルの初回発見は、source/search seedの対応順に673・565・281提案だった。
この条件を再実行する場合は、新しい出力先で以下を使う。

```powershell
$env:FOLDING_DISCOVERY_OUTPUT = Join-Path (Get-Location) 'result/experiments/d2_d3_only_to_d4_rerun'
$checkpoint = Join-Path $study 'source/projected_bars_seed5601/best.pt'
python discovery_transfer.py run --task seven_d4 --architecture projected --arm frozen --seed 5701 --source-seed 5601 --budget 6400 --checkpoint $checkpoint
```

期限用の環境変数も前掲のとおり新しく設定する。これは確率的探索であり、`deterministic`に切り替えると同じ成功は再現しない。

## 未知9節点D4を発見したモデル

新しい出力先に、完成形を含まない9節点の問題定義だけをコピーする。
source seed5602 / search seed5702のfrozenは1,583提案で成功した。

```powershell
$study = Join-Path (Get-Location) 'result/experiments/discovery_transfer_5h_20260914'
$env:FOLDING_DISCOVERY_OUTPUT = Join-Path (Get-Location) 'result/experiments/unseen_nine_rerun'
$env:FOLDING_DISCOVERY_DEADLINE = [DateTimeOffset]::UtcNow.AddHours(1).ToUnixTimeSeconds().ToString()
New-Item -ItemType Directory -Path $env:FOLDING_DISCOVERY_OUTPUT -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $study 'holdout_nine') -Destination $env:FOLDING_DISCOVERY_OUTPUT -Recurse
$checkpoint = Join-Path $study 'source/projected_water_seed5602/best.pt'
python discovery_transfer.py run --task nine_d4 --architecture projected --arm frozen --seed 5702 --source-seed 5602 --budget 6400 --checkpoint $checkpoint
python discovery_transfer.py run --task nine_d4 --architecture projected --arm scratch --seed 5702 --source-seed 5602 --budget 6400
```

9節点の完成座標ファイルは不要。ほかの2 source seedsでは対応する6,400提案試行は未成功だった。探索乱数を交差させた結果は`crossed_discovery.json`を参照する。

## 目標入力の対照と、RL発見から次の経験を作る系列

`--policy-goal 3`などをrunへ追加すると、NNに与えるDだけが変わる。環境の目標・報酬・成功条件は`--task`のDのまま。
通常の転移比較ではこのオプションを付けない。対照では異なるtagを使い、通常条件と結果を混ぜない。

今回scratch RLが発見したseven D4の座標は、`seven_discovered_d4_source/manifest.json`に出所を記録してある。
この座標への教師データから新しいactorを作るコマンドは次のとおり。

```powershell
python discovery_transfer.py pretrain --architecture graph --label graph_seven_d4 --seed 5601 --updates 20000 --seven-only --include-d4 --data-root (Join-Path $study 'seven_discovered_d4_source')
```

このcheckpointはsource課題の確認、または`--tag sequential_nine`でnine D3/D4へ転移する場合だけ利用できる。
元の「D2/D3経験から未知D4」の試験へ流用するとコードが拒否する。初回発見時のRL actor重みは使わない。

最終コードは`final_code_snapshot`、従来RLの未変更確認は`core_code_audit.json`、テストは`test_results.txt`、成功形状・sourceの監査は`audit.json`。
snapshotは保存用であり、そのフォルダだけを作業ディレクトリにして実行するものではない。元の`00_0822_all`のコード配置と、記録されたデータの出所を維持して再実行する。
