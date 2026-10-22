# main8 可変長方向持続マクロ：最終比較

## 結論

今回、明確に有効だったのはGPではなく、行動を

\[
x=(\text{parent StateNode},\boldsymbol v_1,K)
\]

とし、\(\boldsymbol v_1\)を各修正後状態の接空間へtransportしながら、最大
\(K\) 回まで連続して与える可変長方向持続マクロです。

- \(K=1\) 固定では、3 seedすべてでcanonical 6 cellへ未到達でした。
- \(K=1,\ldots,24\) の一様randomでは、同じ3 seedすべてでcanonical 6 cellへ到達しました。
- 同じ \((q,v_1)\) の中でKを変えるoffline実験では、GPはKの順位を学習できました。
- しかしonline held-out比較では、global GP filterも階層型K-GPもrandomを上回りませんでした。

したがって、現時点で支持できる主張は「既存basinを越える有限距離の移動を作るには、
方向を持続する複数の小摂動を一つの可変長macroとして扱うことが有効」です。
「GPによる終端予測がrandomより探索効率を高める」という主張は支持されません。

## 実装

`main8.py` は、各候補を `(parent, v1, K)` として生成します。2回目以降の変位は

\[
\boldsymbol v_j=
\rho\frac{P_{T(q_{j-1})}\boldsymbol v_{j-1}}
{\|P_{T(q_{j-1})}\boldsymbol v_{j-1}\|}
\]

で求めます。各micro-stepでcorrectorと衝突判定を行い、flat、collision、
corrector failure、transport projection failureで直ちに停止します。K回の後に実行する
決定論suffixの上限はKと独立で、すべての力学edgeを同じstep予算へ算入します。

GP入力はSVD係数ではなく、基底回転に不変な物理量

\[
[q,\ q_{\mathrm{last\ nonflat}},\ v_1,\ \log(1+K)/\log(1+K_{\max})]
\]

です。1 macroにつき1教師とし、flat/collision等の停止種別、最小flatness、flat時の
終端形状と最後の非平坦状態、macro costを保存します。

## 1. 可変Kそのもののablation

GPを使わず、paired warm-upも使わず、同じtrunk、parent規則、摂動ノルム0.20、
3000 mechanics stepで比較しました。

| K候補 | seed | raw cells | raw AUC | canonical cells | canonical T6 |
|---|---:|---:|---:|---:|---:|
| 1 | 700 | 2 | 1.866 | 2 | 未到達 |
| 1 | 701 | 2 | 1.869 | 2 | 未到達 |
| 1 | 702 | 1 | 0.979 | 1 | 未到達 |
| 1--24 | 700 | 8 | 5.916 | 6 | 1597 |
| 1--24 | 701 | 8 | 6.218 | 6 | 1517 |
| 1--24 | 702 | 8 | 5.635 | 6 | 1686 |

平均raw cell数は1.67から8.00へ増え、canonical 6到達率は0/3から3/3へ増えました。
これは、長い方向持続macroを行動空間へ入れる効果が、GPとは独立に非常に大きいことを
示します。

## 2. 同一 \((q,v_1)\) に対するoffline全Kパネル

seeds 700--702、18個の独立な \((q,v_1)\) groupに対し、K=1--24を全分岐しました。
同じgroupをtrain/testへ分割しない3-fold評価です。

| 指標 | 結果 |
|---|---:|
| Kにより終端cellが変わるgroup | 12/18 = 0.667 |
| Kによりflat/collisionが変わるgroup | 15/18 = 0.833 |
| 一様random Kの新規cell率 | 0.338 |
| oracle top-1 | 0.833 |
| basis-free GP top-1 | 0.667 |
| basis-free GP top-25% | 0.593 |
| K-score shuffled top-1 | 0.389 |
| raw SVD係数GP top-1 | 0.444 |

K=1--5の平均新規率は0.156、K=12--18では0.468でした。一方、K=19--24では
collision率が0.556へ増えました。したがって「常に最大K」ではなく、状態と第1変位に
応じた有限Kを選ぶ余地があります。また、物理変位入力はSVD係数入力より良好でした。

## 3. global GP filter：10 seed held-out

seeds 703--712、3000 stepで、同じ候補集合、同じpaired warm-up、同じrandom quotaを
用いました。score-shuffledはGP scoreの値とfilter保持数を保ったまま、候補との対応だけを
壊す負対照です。

| 手法 | raw cells | canonical cells | canonical AUC | post-warm canonical AUC | canonical T6平均 |
|---|---:|---:|---:|---:|---:|
| variable-K random | 7.8 | 6.0 | 4.408 | 2.209 | 1899 |
| global GP filter | 7.8 | 6.0 | 4.668 | 2.516 | 1575 |
| score shuffled | 7.5 | 6.0 | **4.727** | **2.588** | **1322** |

GP filterはrandomより良い傾向を示しましたが、canonical AUC差の95% CIは
\([-0.009,0.595]\)で0を跨ぎました。さらにGP filterはshuffledより悪く、GP scoreとの
対応に因果的な有効性は確認できませんでした。そのためglobal argmax/filterは不採用です。

## 4. 親と第1方向をrandomに保つ階層型K-GP

offlineでK順位だけは学習できたため、親と第1方向の多様性を失わないように、

1. `(parent, v1)` groupを一様randomに選ぶ。
2. そのgroup内のKだけをGPで順位付けする。

という階層型を実装しました。pilotではGPとshuffledの差が見えましたが、未使用の
seeds 703--712では再現しませんでした。

| 手法 | raw cells | canonical cells | canonical 6到達 | raw AUC | canonical AUC | post-warm canonical AUC |
|---|---:|---:|---:|---:|---:|---:|
| variable-K random | 7.8 | 6.0 | 10/10 | 5.510 | **4.408** | **2.209** |
| 階層型K-GP argmax | 7.1 | 5.6 | 6/10 | 5.088 | 4.323 | 2.101 |
| group内score shuffled | 7.3 | 5.7 | 7/10 | **5.643** | **4.577** | **2.404** |

階層型K-GPはrandomにraw AUCで平均-0.422、shuffledに-0.555でした。後者の
bootstrap 95% CIは\([-0.935,-0.105]\)、exact sign-flip \(p=0.0488\)で、むしろ
GP scoreが有害でした。したがってこの方式も不採用です。

到達runだけのT6平均はcensoringにより楽観的になるため、主判断にはcanonical 6到達率、
canonical AUC、未到達を3000で打ち切ったT6を使用しています。

## 5. cost-awareの採否

保存済みoffline panelだけを使い、mechanicsを再実行せず、OOF予測cost
\(\widehat C\)でscoreを割りました。

| 選択 | top-1新規率 | 平均cost | 新規/step |
|---|---:|---:|---:|
| 通常GP | 0.667 | 40.94 | 0.01628 |
| cost-aware GP | 0.611 | 37.11 | 0.01647 |

top-1の予算効率改善は約1.1%だけで、cost GPのgroup内Spearman相関も0.274でした。
online argmaxへ追加する根拠は弱いため実装しませんでした。top-25%では改善余地が残りますが、
通常のgroup-filter自体がonline held-outで有効でなかったため、現段階では将来候補に留めます。

## なぜofflineで学べてもonlineで効かなかったか

1. **教師groupが少ない**：online開始時は2個の `(q,v1)` groupから6個ずつKを試した
   12教師だけです。offline CVは、より多くの異なるgroupから学習していました。
2. **archiveが非定常**：offlineの「新規」はtrunk直後のarchiveに対するものです。
   onlineではcellが埋まるたびに価値が変わり、残りcellは急速に希少になります。
3. **終端写像が不連続**：Kが1違うだけでflat/collisionやbasinが変わるgroupが多く、
   smooth Mat\'ern GPの局所補間仮定と合わない箇所があります。
4. **学習前にrandomがほぼ解く**：variable-K randomはおよそ1500--1900 stepで
   canonical 6へ到達します。十分な多様性を持つ教師を集める前に探索が飽和します。
5. **長いKのcost**：offline GPは新規率を上げても、選択macroの平均costを約24%増やしました。
6. **global順位の親集中**：異なる親間でscoreを比較すると、同じ親の多数のKが上位を占めます。
   階層化でこれは除けましたが、教師不足と非定常性は残りました。

## 採用するアルゴリズム

five-barについては、現時点の推奨は **GPなしvariable-K random persistent macro** です。

- 親StateNodeを候補historyから一様に選ぶ。
- 接空間から第1物理変位を一様に生成する。
- Kを1--24から一様に選ぶ。
- 各stepで前変位を現在の接空間へtransportする。
- 全micro-step後に一度だけ決定論suffixを行う。

この方針はSVD基底の符号・回転に依存せず、自由度数も固定しません。ただし、実験的な
有効性を確認したのはfive-barだけです。K上限、摂動ノルム、collision、flatness、archiveの
定義は構造依存で、別構造で同じ優位性を主張するには再検証が必要です。

GPの有効性を今後検証するなら、randomがすぐ飽和しない、より多cell・高価な構造を用い、
複数runで共通利用できる事前学習データ、より多様なpaired `(q,v1,K)` 教師、basin分類型の
mixture/treed surrogateを使うのが妥当です。高不確実性を無条件に加点するより、既知basinや
collisionを高信頼で予測できる候補だけ除外する保守filterの方が安全です。

## 主なファイル

- `main8.py`: 可変K macro、random、global GP、階層型K-GP、shuffled対照
- `MAIN8_VARIABLE_K.md`: 実装仕様と環境変数
- `persistent_k_panel.py`: 同一 `(q,v1)` に全Kを与えるoffline panel
- `persistent_k_cost_aware.py`: 保存panelのcost-aware OOF評価
- `PERSISTENT_K_PANEL.md`: offline panelの詳細
- `MAIN8_PILOT_RESULTS.md`: 3-seed pilot
- `MAIN8_HELDOUT_RESULTS.md`: global filterの10-seed held-out
- `main8_group_heldout_aggregate.py`: 階層型held-outの再集計
- `result/main8_group_heldout_b3000_aggregate.json`: 階層型held-out集計値

## 再現例

```powershell
$env:FOLDING_STEP_BUDGET = "3000"
$env:FOLDING_MAIN8_PAIRED_WARMUP_GROUPS = "2"
$env:FOLDING_MAIN8_PARENT_LIMIT = "16"
$env:FOLDING_MAIN8_FIRST_DIRECTIONS = "2"
$env:FOLDING_MAIN8_EPSILON = "0.20"
$env:FOLDING_MAIN8_FILTER_KEEP_FRACTION = "0.25"
$env:FOLDING_SAVE_TREE = "0"

# 完全random triple
$env:FOLDING_MAIN8_SELECTOR = "variable_k_random"
$env:FOLDING_RESULT_NAME = "main8_random_reproduction"
$env:FOLDING_SEED = "703"
python main8.py

# 階層型K-GP（研究上の負結果も再現可能）
$env:FOLDING_MAIN8_SELECTOR = "variable_k_gp_group_argmax"
$env:FOLDING_RESULT_NAME = "main8_group_gp_reproduction"
python main8.py
```

