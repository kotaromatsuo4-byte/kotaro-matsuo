# 方向持続macroとGP選択圧の予備比較

## 1. 比較した問い

本比較では、終端cellの増加要因を次の三層に分けた。

1. **探索表現**：直前の摂動方向を現在の接空間へ運ぶことが有効か。
2. **時間的抽象化**：二つの小摂動を一つのmacro-actionとして評価することが有効か。
3. **学習した選択圧**：同じmacro候補集合の中で、GPによる順位付けがrandomより有効か。

方向持続そのものの効果をGPの効果として数えないことが重要である。

## 2. K=2 persistent macro-action

状態を \(q\)、最初の接空間変位を \(v_1\)、correctorを \(C\) とする。
まず

\[
q_1=C(q+v_1)
\]

を計算する。次に、\(q_1\) の接空間への射影を \(P_{T(q_1)}\) として、

\[
v_2
=
\rho\frac{P_{T(q_1)}v_1}{\|P_{T(q_1)}v_1\|},
\qquad
q_2=C(q_1+v_2)
\]

とする。その後だけ重力ポテンシャル最小化による決定論rolloutを行う。
各小摂動は独立したmechanics stepとして予算へ数える。したがって、大きな一step
摂動へ戻したものではない。

macro GPの教師は

\[
x=(q,v_1)\longmapsto
\{\text{stop},\ D_{\mathrm{terminal}},\ C_{\mathrm{rollout}}\}
\]

というmacro開始edge一行だけである。二つ目の摂動edgeや決定論suffixを同じ
macro GPへ重複登録しないため、`gp_training_mode=macro_first`を用いた。

## 3. GP filter

通常のglobal argmaxは、既知の平坦basinと同じ親を繰り返し選ぶ傾向があった。
そこでGPを一点最適化器ではなくfilterとして用いた。

候補 \(x=(s,a)\) の連続descriptor novelty scoreを

\[
S_t(x)
=
p_{\mathrm{flat}}(x)
\left[
d_{A_t}(\mu_D(x))+\kappa r_D(x)
\right]
\]

とする。本比較では既知flatへの過度な活用を避けるため、独立なreach加点を0とした。
各親 \(s\) について \(\max_a S_t(s,a)\) を計算し、その値が上位50%の親だけを
残す。残った親から一様に一つ選び、その親の最大score作用を実行する。全選択の
20%はGPを通さない完全random枠として残した。

### cost-aware版

rollout step数 \(C\) に対して \(\log(1+C)\) を別GPで回帰した。予測costの
楽観的な下側値を

\[
C_t^-(x)
=
\max\left\{
1,
\exp\left(\mu_C(x)-\kappa_C\sigma_C(x)\right)-1
\right\}
\]

とし、filter順位には

\[
S_t^{\mathrm{cost}}(x)
=
\frac{S_t(x)}{C_t^-(x)+c_0}
\]

を用いた。これは「平坦になりやすい候補」だけでなく、「新規性を検証するための
mechanics stepが少ない候補」を優先する。

## 4. paired offline panel

同じ摂動直後StateNodeごとに、exact transport、\(\pm15^\circ\)、
\(\pm45^\circ\)、reverse、random二方向をすべて実rolloutした。親StateNode単位で
3-foldに分け、同一親の近接候補がtrain/testへ漏れないようにした。

seed 910--912、各12親・96 rolloutのearly-one-cell archiveに対する3 seed平均は
次の通りである。

| 手法 | new-cell AUC | average precision | success@1 |
|---|---:|---:|---:|
| raw GP | 0.9091 | 0.8657 | 0.745 |
| Context-GP | 0.9058 | 0.8390 | 0.815 |
| exact transport固定 | 0.4894 | 0.4192 | 0.579 |
| uniform random期待値 | 0.5000 | -- | 0.649 |

Context-GPはsuccess@1を改善したが、AUCとaverage precisionはraw GPよりわずかに
低く、seed間で一貫しなかった。exact transport方向自体の平均順位もraw GP 4.24、
Context-GP 5.07、random期待4.5であり、GPがtransport規則そのものを自然に
再発見したとはいえない。現段階では高次元context追加を採用する根拠は弱い。
transportを作用定義へ組み込んだK=2 macroを作り、そのmacro候補をraw GPで
rank/filterする方が単純で安定している。

## 5. online予備実験

five-bar、摂動norm 0.20、planar collision、10000 mechanics step、
seed 500--502で比較した。数値は3 seed平均であり、本実験ではない。

| 手法 | 最終archive cell | canonical 6到達率 | discovery AUC | 6番目発見step | wall time [s] |
|---|---:|---:|---:|---:|---:|
| K=2 macro random | 5.67 | 2/3 | 3.880 | 7499（到達runのみ） | 137.6 |
| shuffled-score filter | 5.33 | 1/3 | 4.318 | 2506（到達runのみ） | 201.6 |
| GP filter | 6.00 | 3/3 | 4.251 | 6969 | 172.3 |
| cost-aware score shuffle | 5.67 | 2/3 | 3.931 | 8234（到達runのみ） | 197.8 |
| cost-aware GP filter | 6.00 | 3/3 | 4.684 | 4488 | 227.2 |
| 非同期80/20 transport heuristic | 6.33 | 2/3 | 4.817 | 5725（raw archive） | 134.3 |

transport heuristicのraw archiveはseedごとに7、6、6 cellだったが、既知canonical
6との照合では6、5、6であった。余分なlayer variantがraw cell数とAUCへ含まれる。
cost-aware GP filterは3 seedすべてでcanonical 6をちょうど一度ずつ覆った。

cost-aware GP filterは平均rollout長をmacro-randomの28.4から25.4 stepへ下げ、
10000 step中の評価回数を350から391へ増やした。通常GP filterはflat率を上げすぎ、
平均rollout長が34.8 stepとなった。costモデルはこの失敗を緩和した。

## 6. 現時点の解釈

1. 小摂動でも、方向を接空間間で持続させれば一run内でcanonical 6へ到達できる。
2. K=2 macro randomだけでも3000 stepで平均3.33 cellとなり、単一摂動randomより
   到達範囲が広がった。
3. GP-UCBのglobal argmaxは既知flat basinへ集中し、macro randomより悪かった。
4. GPをtop-quantile filterとして使うと集中が弱まり、最終coverageは改善した。
5. cost-aware GP filterは3 seedすべてで6 cellへ到達し、予備結果では最も早かった。
6. cost-aware scoreを候補間でshuffleした負対照は平均5.67 cell、AUC 3.931に
   留まった。実GPとのpaired AUC差はseedごとに+0.386、+1.003、+0.873であった。
7. したがって3 seed内では、filter構造だけでなくGPの候補対応にも正の効果が
   観測された。ただし設計に使ったseedなので、独立testではない。
8. exact GPのwall timeは大きく、現状の優位はmechanics-step sample efficiencyに限る。

cost-aware GPとcost-aware shuffleのpaired AUC差は平均+0.754、標本標準偏差
0.326であるが、n=3のt区間は0を含む。したがって統計的優越性ではなく、次の
held-out検証を行う根拠となる効果量として扱う。

したがって、現時点で主張できるのは「方向持続macroという構造priorの上で、GPを
cost-aware filterとして用いる設計が有望」という予備的結論である。3 seedを見ながら
設計を修正したため、この値を最終的な統計的証拠には使わない。

## 7. 次の固定比較

次は設定を固定し、未使用seedで次を比較する。

1. K=2 macro random
2. 同じ保持率のrandom-score filter
3. inverse-visitだけを用いた親分散化
4. descriptor GP filter
5. cost-aware descriptor GP filter
6. oracle filter（小さいcandidate panelだけ全rollout）

一次指標はcanonical 6への到達stepとpost-trunk discovery AUC、二次指標は
cell@3000/5000、false reject率、親選択entropy、wall timeとする。少なくとも
未使用20 seedを用い、右打切りrunを捨てずにtime-to-sixを扱う。

一般構造へ拡張する際には、exact GPをinducing-point GPまたは局所GPへ置換し、
\(K\in\{1,2,3\}\)を別macro actionとして比較する。20%の完全random bypassは、
未知basinをモデル誤差で永久に除外しないため維持する。

## 8. provenance

上表の10000-step結果は`main5_gp_v4_persistent_macro`で生成した。現在の
`main5_gp_v4_2_persistent_macro_filter_diagnostics`は選択ロジックを変えず、将来の
held-out比較用にcandidate数、filter保持数、bypass、選択順位、audit列を
`rollout_diagnostics.dat`へ追加した版である。最終比較では全手法をv4.2で再実行し、
各`config.json`の`algorithm_version`を照合する。
