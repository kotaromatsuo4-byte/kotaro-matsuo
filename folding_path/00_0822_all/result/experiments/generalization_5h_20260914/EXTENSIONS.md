# 追加検証の記録

以下は最初の2,000更新比較とは分けて保存する。評価データを学習データへ戻さない。

## 2026-09-14 09:49 JST時点

- torusのscratch学習率・更新数はseed5501のvalidation8初期形状だけで比較。
  - LR0.0003 / 2,000更新、support1/4/16本: 0/8, 0/8, 0/8。
  - LR0.001 / 2,000更新: 0/8, 0/8, 2/8。
  - LR0.003 / 2,000更新: 0/8, 0/8, 3/8。
  - LR0.0003 / 8,000更新: 0/8, 1/8, 1/8。
  - 3つのsupport予算の平均成功率でLR0.003 / 2,000更新を選び、5seedのtestを追加する。
  - この選択は1つの学習seedに依存するため、十分なハイパーパラメータ最適化とは呼ばない。
- source replay: D4の64例バッチ・2,000更新を保ち、既存D2/D3データ32例の損失を重み0.5で追加。
  追加計算量がある。validationで保持と適応のトレードオフを測る。
- source DAgger: seven-bar source初期形状だけからNN経路を生成し、D2/D3のoracleを再問い合わせ。
  各round16初期形状×2目標、上限300step、3round、各2,000更新。
  source validationのD2/D3平均成功率最大のroundを選択（同率なら早いround）。
  D4/torusの状態・ラベルは一切使わない。追加sourceデータ・更新コストを別途記録する。
- D4専用低ランク追加部分: source本体を凍結し、D4入力時だけ有効なrank8の追加部分を学習。
  frozen random featuresを使う同じ構成を対照とする。通常の全体更新とも別に比較。
  D2/D3の出力保持は単体テストで確認。成功率改善は別途検証が必要。
- graph projected構成: 節点ベクトルからモード係数を計算して制約空間へ投影。
  モード基底の符号・直交回転に物理方向が不変であることを単体テスト。
  学習済みモデルの比較はmodal構成とは別フォルダに保存する。
- mixed source: five-barの3,370ラベルとseven-barの5,688ラベルを更新ごとに等確率で選ぶ。
  合計source更新数を5,000に固定。torusラベルはsourceに含めない。
- online D4: 4つのsupport初期形状から学生の経路を生成し、各round128状態で教師へ問い合わせ、250更新。
  8roundで最大1,024問い合わせ・2,000更新。教師のD4経路は初期データにも使わない。
  scratch/pretrainedで問い合わせ数は同じだが、訪問状態は各自の方策によって異なる。
  候補不足時の重複問い合わせも予算に数える。評価は初期形状分割を維持する。
- torusの追加stress test: 元形状から100–300回のランダム歩行、乱数66004で16初期形状。
  既存support/validation/testとの近接を除外し、学習に追加しない。
  固定したsupport1/16本のモデルを評価する。元の5–25歩行のtestとは混合しない。

新規構成はまずvalidationで検証する。5seedまで反復した主比較と、1seedの探索的比較を区別する。

## 2026-09-14 10:55 JST時点の追加条件

- D4のLRはvalidation12形状、seed5501、support1/4/16/32本の平均成功率で各armごとに選択。
  候補0.0003/0.001/0.003、同率なら低いLR。scratch=0.001、pretrained=0.0003を固定。
  `d4_rate_selection.json`に全候補と選択を記録。testの2/8本も同じ選択済みLRで補完する。
- source static対照は、元の5,688ラベルだけで2,000更新を3回追加。
  source DAggerと同じAdamリセット・バッチサイズ・roundごとの乱数・source validation選択を使う。
  更新回数追加とデータ追加を区別するための条件で、新しい教師データは加えない。
- 低ランクD4追加部分はLR0.0003/0.003の両方をvalidationで確認。保持できてもD4適応は限定的だった。
- 純粋なstudent訪問状態だけからのonline D4は、seed5501のvalidationではpretrainedがscratchを上回らなかった。
  追加検証用に、固定prefixの最初の教師経路1本（62ラベル）で500更新してから、
  6round×128問い合わせ・250更新を行うbootstrap条件を用意した。合計2,000更新。
  62個の初期教師ラベルと、その後の768問い合わせは別々に数える。
- five/seven混合sourceのゼロショットは、NN更新・torus教師ラベル使用が0という意味。
  構成の探索・候補の選別にはtorus validationを使用しているため、
  対象構造に一切触れず方法まで選んだ厳密な未接触評価とは区別する。
  test初期形状の軌道や教師ラベルは学習に戻さず、全5seedと広い初期形状分布の結果を報告する。

## 後半の追加検証

- support選択の頑健性: 32本のD4教師経路を連続IDの4本ずつ、重複しない8組に分ける。
  seed5501～5503、各2,000更新、学習率は既にvalidationで選んだscratch=.001/pretrained=.0003に固定。
  各組のラベル数も報告する。成功経路だけを選ばず、8組を全て集計する。
- mixed source継続: five/seven混合5,000更新の重みから5,000更新を3round追加。
  five/sevenそれぞれD2/D3の4課題のsource validation平均で選択し、同率は早いround。
  探索総更新数20,000と選択された重みの更新数を区別する。torusでの成績は選択に使わない。
- 選択済みsourceの評価では、selection.jsonが完成するまで開始しない。
  途中の最良重みで生成されたsource評価は最終選択roundと照合し、必要なら退避して再評価する。
- graph replay: DAggerのsource選択重みと、そのroundまでのsourceデータを使う。
  torus16本のbatch32と乱数・2,000更新・LR.0003を通常の転移と一致させ、
  既存seven-barデータbatch16の損失を重み.25で追加する。追加計算量がある。
  固定重み.25のseed5501 validationでtorus成功8/8、seven-bar D2=3/12,D3=4/12を確認後、
  同じ設定の5seed testを開始。通常のtorus追加学習後のseven-bar保持率も同時に評価する。
- 更新回数に対する成功曲線: 固定16本で0/100/250/500/1000/2000更新時のvalidationを評価。
  scratch・通常fine-tuning・source replayを3seed比較し、最終重みが既存2,000更新の重みと一致することを確認。
  これは事後の学習過程診断で、主実験のtest用checkpoint選択には使わない。
- online D4のbalanced条件: 同じbootstrap1本・500更新の後、6round×128問い合わせ・250更新。
  student経路上の近接状態を間隔.03で間引き、4本の開始経路へ問い合わせ予算を均等に割り当てる。
  停滞付近の似た状態に偏る問題への予備検証で、まずseed5501のvalidationだけを使う。
- 6unit_torusの追加プローブは、barだけで学習したmixed5,000更新の全5seedをそのまま使用。
  提供された初期形状1個、事前に指定したD2/D3を評価。目的の実現可能性を独立に確認できていないため、
  失敗は「この条件では折れなかった」とだけ解釈する。

## 12:10 JST以降の教師データ選択に関する追加検証

- torus教師1本の頑健性: 先頭4経路をそれぞれ単独で使い、seed5501～5503を比較。
  sourceはsevenの5,000更新、target2,000更新。scratch LR.003、pretrained LR.0003を再調整せず使用。
  経路0は既存主比較の同じ結果を再利用し、1～3を追加する。
- D4 nearest-branch教師: 実現可能な3種類の直線配置から、その時点の座標距離が最小の配置を選ぶ。
  選んだ配置への従来oracleを1回使う。複数の完了経路を事後選択する方法ではない。
  NN入力には配置ID・目的座標を追加しない。元のsupport32初期形状から新しい教師経路を生成し、別データとして保存。
  validation教師成功6/12（従来2/12）を確認した。test教師は19/32（従来21/32）で一律改善ではなかった。
  seed5501、教師4本のNN validationでscratch3/12、pretrained6/12を確認後、5seed testを開始。
  教師4/16本、2,000更新でscratch・pretrained・replayを全て評価する。既存の学習率・replay重みを変更しない。
  教師経路本数は元実験と揃えるが、ラベル数と教師方向が変わるため別条件とする。

## 12:33 JST以降の確認と追加対照

- fresh confirmation: scratch対pretrained＋source replay、教師16本・2,000更新の比較条件と全5seedの重みハッシュを固定してから、
  新たなseven32初期形状（乱数88001、20～160歩）とtorus16初期形状（88002、5～25歩）を生成。
  既存の全分割・torusの広い初期形状集合との近接を除外する。主比較とは別の評価集合として報告。
  source/supportやトポロジーは同じなので、未使用トポロジーの確認とは呼ばない。
- scratch＋source replay対照: pretrained replayと同じ旧データ、target/sourceミニバッチ・乱数、損失重み、2,000更新で、
  重みだけランダム初期化から開始。GNNでは各seedのpretrainedが使ったsource DAggerデータをそのまま共有する。
  まず以前のscratch選択LR（seven .001、torus .003）で全5seedのtestを開始。
  追加でseed5501 validation、教師16本のtarget成功率によりLR .0003/.001/.003を比較し、同率なら低いLRを選択。
  選択LRが以前の値と異なる場合は、そのLRでも5seed testを実施。default結果と選択結果を混ぜない。
  この対照はfresh confirmationを固定した後に追加したため、freshの事前固定2条件を置き換えない。
- 教師ラベルと状態カバー範囲の分離: 元のD4 support4経路の228状態を固定し、nearest教師で方向・幅を再ラベル。
  両教師で非ゼロ方向が得られる共通状態を使い、今回は228状態全てが残った。
  座標・D・モード・経路IDが完全一致することを確認。3seed、scratch/pretrained、2,000更新、既存LRを使用する。
  元ラベルの条件は既存の同じ3seed結果を再利用し、新ラベル条件だけ追加学習する。
  nearest教師で最初から生成した4経路（174ラベル）の同じ3seed結果とも比較し、状態分布変更とラベル変更を区別する。
