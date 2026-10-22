# main8: 可変長方向持続マクロ

`main8.py` は、1回の探索判断を

\[
x=(\text{parent StateNode},\;\boldsymbol v_1,\;K)
\]

とします。最初の物理変位 \(\boldsymbol v_1\) を実行した後、直前の物理変位を各修正後状態の接空間へ射影し、元と同じノルムへ正規化して、最大 \(K\) 回まで摂動を連鎖します。その後、`FOLDING_MAX_ROLLOUT_STEPS` 回を上限とする決定論的rolloutを実行します。この決定論的suffixの上限は \(K\) と独立です。

射影に失敗した場合はランダム方向へ置き換えず、`projection_failure` としてマクロを停止します。したがって、同じ状態・第1変位・Kから得られる結果は決定論的です。

## 比較するselector

- `variable_k_random`: 共通候補集合から一様ランダム選択
- `variable_k_gp_argmax`: novelty-UCBの最大候補を選択
- `variable_k_gp_argmax_shuffled`: scoreと候補の対応だけをシャッフルするargmax負対照
- `variable_k_gp_filter`: 上位割合だけを残し、その中から一様選択
- `variable_k_gp_filter_shuffled`: scoreと候補の対応だけをシャッフルするfilter負対照
- `variable_k_gp_group_argmax`: `(parent,v1)` を一様選択した後、そのgroup内でKをargmax
- `variable_k_gp_group_argmax_shuffled`: group内だけでscoreとKの対応を壊すargmax負対照
- `variable_k_gp_group_filter`: `(parent,v1)` を一様選択した後、そのgroup内の上位Kから一様選択
- `variable_k_gp_group_filter_shuffled`: group内だけでscoreとKの対応を壊すfilter負対照

GP系列にも既定で20%の完全ランダム選択を残します。`filter_shuffled` はwarm-up、候補数、filterの保持数、ランダム枠を同じにしたまま、GPの順位情報だけを壊す対照です。

`group_*` 系列では、20%のepsilon-random時は従来どおり全候補poolから一様選択します。guided時だけ、まず `(parent,v1)` groupを一様選択し、その中のKだけにGP選択圧をかけます。これにより、GPが親状態や第1方向の選択を変える効果と、継続回数Kを学習する効果を分離できます。shuffled対照もgroupごとに独立にscoreを並べ替えるため、scoreが別の親や第1方向へ移動することはありません。

## GP入力と教師

GP入力はSVD係数を使わず、

\[
[\boldsymbol q,\;\boldsymbol q_{\mathrm{last\ nonflat}},\;
 \boldsymbol v_1,\;\log(1+K)/\log(1+K_{\max})]
\]

です。カーネル距離は特徴量の再尺度化により、

\[
r^2=
\frac{\operatorname{RMS}(q-q')^2}{\ell_q^2}
+\frac{\operatorname{RMS}(v_1-v'_1)^2}{\ell_v^2}
+\frac{(k-k')^2}{\ell_K^2}
\]

と等価になります。このため、構造の自由座標数が増えてもKの差が変位block内で薄まりません。

完了した1マクロから教師を1件だけ作ります。`flat`、`collision`、`timeout`、`corrector_failure`、`projection_failure`を学習し、連続終端descriptorはflatのときだけ保存します。cost教師は

\[
\log(1+C),\qquad
C=\text{摂動edge数}+\text{決定論edge数}
\]

です。総予算によって途中で打ち切られたマクロだけは右打切りデータなので教師から除外します。

## paired warm-up

既定では同じ `(parent, v1)` に対して

```text
K = 1, 2, 4, 8, 16, 24
```

をすべて実行します。各branchで重複する力学計算も総step予算へ算入します。全selectorで次の2つが一致すれば、warm-upの木と教師が完全に一致しています。

- `summary.json / warmup_prefix_sha256`
- `summary.json / warmup_training_sha256`

## 実行例

```powershell
$env:FOLDING_MAIN8_SELECTOR = "variable_k_gp_filter"
$env:FOLDING_SEED = "401"
$env:FOLDING_STEP_BUDGET = "10000"
$env:FOLDING_RESULT_NAME = "main8_filter_b10000"
python main8.py
```

主なmain8固有環境変数は次の通りです。

| 環境変数 | 既定値 | 意味 |
|---|---:|---|
| `FOLDING_MAIN8_K_VALUES` | `1,2,...,24` | 候補とする整数Kのカンマ区切り一覧 |
| `FOLDING_MAIN8_WARMUP_K_VALUES` | `1,2,4,8,16,24` | paired warm-upのK anchor |
| `FOLDING_MAIN8_PAIRED_WARMUP_GROUPS` | `2` | paired `(parent,v1)` の数 |
| `FOLDING_MAIN8_PARENT_LIMIT` | `16` | 1候補集合の親状態数 |
| `FOLDING_MAIN8_FIRST_DIRECTIONS` | `2` | 親状態ごとの第1変位数 |
| `FOLDING_MAIN8_EPSILON` | `0.20` | GP系列の完全ランダム確率 |
| `FOLDING_MAIN8_FILTER_KEEP_FRACTION` | `0.25` | filterが残す候補割合 |
| `FOLDING_MAIN8_GP_STATE_LENGTH` | `0.50` | 状態blockの長さ尺度 |
| `FOLDING_MAIN8_GP_V_LENGTH` | `0.50` | 第1変位blockの長さ尺度 |
| `FOLDING_MAIN8_GP_K_LENGTH` | `0.35` | log-K blockの長さ尺度 |

`rollout_diagnostics.csv` にはrequested/executed K、摂動・suffix・総mechanics cost、最小transport射影率、候補数、filter保持数、score順位、獲得関数の内訳を保存します。`training_data.npz` にはmacro単位の入力、結果、K、cost教師を保存します。

## smoke確認

seed 981、短い共通設定で5 selectorすべてを実行し、compile、出力生成、GP fit/scoreを確認しました。全selectorでwarm-upの終了step、木のhash、教師hashが一致しました。これは性能比較ではなく、実装の整合性確認です。
