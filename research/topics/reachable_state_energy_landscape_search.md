# 到達可能な未知状態と低エネルギー経路の同時探索

更新：2026-10-22。状態：**検討中**。  
この文書は、2026年9月〜10月のChatGPTとの研究相談を整理したもの。新規性を確定した記録ではなく、既往研究との比較を通じて次の研究問題を絞るための作業ノートである。

## 1. 出発点：川口の研究との競合

現在の研究では、目的節点座標を与えず、条件を満たす未知の平坦状態とそこへの折り経路を探索してきた。

ただし、川口の研究も「目的形状を一点として指定せず、ある計量を減少させるように逐次折る」という性格を持つ。計量を複数用意して多数回実行すれば、異なる経路・終状態を得ることができるため、

- 川口＝解析
- 本研究＝探索

という単純な区別は成立しにくい。

今後の比較では、**探索するか否かではなく、何を探索圧として用いるか、どのconfiguration spaceを探索するか、経路全体の何を評価するか**を区別する必要がある。

## 2. 新規性候補として検討した3方向

### 2.1 平坦状態の列挙

異なる平坦状態を多数発見する方向は以前から試したが、探索圧の設計が難しく、一様random探索を明確に上回る安定した結果を得られていない。

このため、単純に「複数解を列挙する」ことを主軸に戻すのは現状では弱い。

### 2.2 異構造への汎用化

複数構造で学習した方策を未学習構造へ適用できれば強いが、次の問題がある。

- 未学習構造への追加学習が必要なら、川口型の軽量な逐次最適化に対する優位性が弱い。
- 学習データセットをどう構築するかが難しい。
- 計量 \(r_1\) では見つからない経路でも、別の計量 \(r_2\) を設計すれば川口型で発見できる可能性がある。

したがって、汎用化を主張するなら、**追加学習なしで未知構造へ適用できる程度の強い汎化**が必要になる。

### 2.3 ひずみを許容した探索

厳密な機構運動だけではなく、一時的な弾性変形を許容して未知状態へ到達する方向を検討した。

単に

\[
g(x)=0
\]

から

\[
\|g(x)\|\le \varepsilon
\]

へ制約を緩めるだけでは「川口法のひずみ拡張」と見られ、新規性は弱い。

そこで、研究対象を

> **平坦状態へ到達するために越える必要がある最小の弾性エネルギー障壁**

へ置き直す案を検討した。

## 3. minimum-energy-barrier問題としての定式化

平坦状態集合を

\[
\mathcal G_{\mathrm{flat}}
=
\{x\mid F(x)\le\varepsilon_F\}
\]

とする。

初期状態 \(x_0\) から、どの平坦状態へ行くかも含めて経路を探索する問題を

\[
B^*
=
\min_{\substack{
x_T\in\mathcal G_{\mathrm{flat}}\\
\tau:x_0\rightarrow x_T
}}
\max_{s\in[0,1]} U(\tau(s))
\]

と考える。

出力は

\[
(x_T^*,\tau^*,B^*)
\]

であり、

- \(x_T^*\)：最も容易に到達できる未知の平坦状態
- \(\tau^*\)：そこへの経路
- \(B^*\)：経路中に越える必要がある最小エネルギー障壁

である。

この問題は「現在状態でどの計量を最も下げるか」という局所的な逐次則とは異なり、**終状態と経路の両方を変数として経路全体の最大エネルギーを評価する**。

ただし、minimum-energy pathやminimax/bottleneck path自体は既知の問題クラスなので、「minimaxを導入したこと」だけを新規性にはできない。

## 4. Zhouらの研究との比較

比較対象として、

> Zhou et al., *Low energy fold paths in multistable origami structures*, International Journal of Solids and Structures, 2023.

を検討した。

### 4.1 Zhouらの2段階構成

Zhouらの弾性origamiに対する枠組みは大きく二段階に分かれる。

#### Stable-state discovery

Potential energy \(U(x)\) のlocal minimaをstable stateとして、

1. Latin Hypercube Sampling (LHS) で多数の初期値を作る
2. 各初期値からgradient-based local optimizationを行う
3. 重複を除いてlocal minimaを抽出する

という方法を示している。

概念的には、

\[
x_{\mathrm{LHS}}^{(1)}\rightarrow A,\qquad
x_{\mathrm{LHS}}^{(2)}\rightarrow B,\qquad
x_{\mathrm{LHS}}^{(3)}\rightarrow A
\]

のように、configuration space中の各basin of attractionへ初期値を入れることを狙う。

LHSは各変数範囲を均等に使いやすいが、

- energy landscapeそのものを見ていない
- basinが小さいstateを見逃し得る
- 高次元では依然として疎になる
- そのstateが特定の初期状態から到達可能かは分からない

という問題がある。

なお、Zhouらの主要な数値例では、LHSにより大規模高次元origamiの未知stable statesを大量列挙する実証よりも、既知のstable-state pairに対するNEBの検討が中心である。

#### NEBによるpath discovery

Nudged Elastic Band (NEB) では、

\[
Y_0=Y_A,\qquad Y_K=Y_B
\]

として**始状態と終状態を既知として固定**する。

中間imagesは節点座標の線形補間などで初期化し、各imageを独立最小化するのではなく、

- 経路に垂直な方向にはphysical energyを下げる
- 経路に平行な方向には仮想springでimage間隔を保つ

ことでband全体をlow-energy valleyへ動かす。

数値例には、90° Miura-Ori、elastic Miura-Ori、8-fold waterbomb、waterbomb arrayなどがある。

### 4.2 Zhouらに対する問題意識

Zhou型では、

\[
\text{state discovery}
\rightarrow
\text{state pair selection}
\rightarrow
\text{path discovery}
\]

が分離している。

このため、LHSでstable state \(B\) を発見しても、

\[
x_0\rightarrow B
\]

が

- 自己交差なしで可能か
- 許容ひずみ・エネルギー内で可能か
- 現実的な連続経路を持つか

はstable-state discoveryの時点では分からない。

ここから、

> **存在するstateを探すのではなく、指定初期状態から実際に到達可能な有意stateを、経路と同時に探す**

という問題設定を考えた。

## 5. reachable-state discovery

指定初期状態 \(x_0\) からしか探索を開始せず、

\[
x_0\rightarrow x_1\rightarrow x_2\rightarrow\cdots
\]

とfeasibleなlocal deformationを逐次生成する。

探索途中で初めて

\[
x_k\in\mathcal G
\]

となったとき、そのstateを発見したとみなす。

\(\mathcal G\) は例えば、

- flat-state set
- stable-state set

である。

このとき出力は常に

\[
(\text{state},\text{path})
\]

のペアとなり、stateの存在だけでなくrootからのpathも同時に得られる。

到達可能集合は例えば、

\[
\mathcal R(x_0;B)
=
\left\{
x\ \middle|\
\exists\tau:x_0\rightarrow x,\ 
\max_s U(\tau(s))\le B
\right\}
\]

と定義できる。

さらに全pathについて、

- self-intersectionなし
- boundary conditionを維持
- strain limit以下

などを課す。

flat-state探索なら、

\[
\mathcal R(x_0;B)\cap\mathcal G_{\mathrm{flat}}
\]

を探す問題になる。

## 6. T-RRT：非常に重要な比較対象

この課題感に対して非常に近い既往として、

> L. Jaillet, F. J. Corcho, J.-J. Pérez, J. Cortés,  
> “Randomized Tree Construction Algorithm to Explore Energy Landscapes,”  
> *Journal of Computational Chemistry*, 32, 3464–3474, 2011.  
> DOI: 10.1002/jcc.21931.

を確認した。

### 6.1 T-RRTの目的

T-RRT (Transition-based Rapidly-exploring Random Tree) は、

- robotics由来のRRT
- statistical physics由来のMetropolis型transition test

を組み合わせ、**一つの初期状態からenergy landscapeをtree状に探索**する手法である。

target configurationはoptionalであり、targetなしでも探索可能。

したがって、

\[
q_{\mathrm{init}}
\rightarrow
\text{tree exploration}
\rightarrow
\text{unknown reachable minima}
\]

を実現しており、「到達可能な未知stateとpathを同時に探索する」だけでは新規とは言いにくい。

### 6.2 RRTによる探索圧

各iterationで、

1. configuration spaceから \(q_{\mathrm{rand}}\) をsample
2. tree中のnearest node \(q_{\mathrm{near}}\) を選ぶ
3. \(q_{\mathrm{near}}\) から \(q_{\mathrm{rand}}\) 方向へ一定stepだけ進み \(q_{\mathrm{new}}\) を作る

RRTのVoronoi biasにより、未探索領域側のnodeほど選ばれやすく、treeが自然にspace-fillingする。

### 6.3 energy transition test

新しいstateのenergy増分を

\[
\Delta E
=
E(q_{\mathrm{new}})
-
E(q_{\mathrm{near}})
\]

とする。

\[
\Delta E\le0
\]

なら必ず受理し、energyが上がる場合は

\[
p
=
\exp\left(-\frac{\Delta E}{kT}\right)
\]

のMetropolis型確率で受理する。

よって、常にenergyを下げるgreedy descentではなく、必要に応じてenergy barrierを越えられる。

### 6.4 温度の自動調整

T-RRTでは \(T\) を固定せず、探索状況によって調整する。

- rejectionが連続して探索が止まると \(T\) を上げる
- uphill moveが成功すると \(T\) を下げる

ことで、

- 普段はlow-energy regionを優先
- 行き詰まったときだけ高energy barrierを越えやすくする

というバランスを取る。

### 6.5 T-RRTmin

targetを与えず、一つの初期状態からtreeを成長させる。

tree構築後、近傍stateよりenergyの低いnodeを抽出し、reachable energy minimaを得る。

つまり、

\[
x_0
\rightarrow
\text{reachable regionをtree探索}
\rightarrow
\text{未知のminimaを抽出}
\]

であり、見つかったnodeにはrootからのpathが必ず存在する。

### 6.6 T-RRTtrans

既知のtarget stateを与える場合は、treeがtargetに到達するまで探索し、leafからrootへtreeを逆に辿ることでtransition pathを得る。

同一frameworkで、

- unknown-state exploration
- known-state transition search

の両方を扱う。

### 6.7 数値例

#### 2D academic benchmarks

Zorro potential と Alien potentialを使用。

Zorroでは2つのenergy minima間のsaddleを通るtransition pathを探索し、exhaustive grid searchと近いtransition-state energyを得た。

AlienではA–B間に2つの主経路があり、barrier heightは類似するがpath widthが異なる。T-RRTは100試行中、より広い経路を高頻度で選んだ。

この結果は、T-RRTが単純な

\[
\min_\tau\max U(\tau)
\]

だけを解いているのではなく、energyとpath widthの両方の影響を受けることを示す。

#### Alanine dipeptide

探索変数は全Cartesian coordinatesではなく、bond length・bond angleを固定した**7個のbond torsion**。

T-RRTminで8000 nodesまで探索し、6つのenergy minimaを発見した。その後local steepest-descentでminimaを精密化。

同じ6 minimaを得る比較では、simulated annealingが660 min、T-RRTが15 minと報告されている。

これは、高次元実問題では**物理的に意味のあるinternal coordinatesを探索変数にする**ことの重要性も示す。

## 7. T-RRTを踏まえた新規性候補

T-RRTの存在により、

> 「同じ初期状態からtreeを伸ばし、到達可能な未知stateとpathを同時に探す」

だけでは新規性が弱い。

今後の差別化候補は次の3点。

### 7.1 Origami-specific local expansion

T-RRTは \(q_{\mathrm{rand}}\) へ向かう一般的なconfiguration-space extensionを用いる。

本研究では、各状態で得られる

- mechanism modes
- soft elastic modes
- Hessianの低固有モード

などを用い、

\[
\Delta x
=
\sum_k a_k m_k
+
\sum_j b_j s_j
\]

のように、**その状態で力学的に意味のある局所方向のみを用いてtreeを成長させる**ことができる。

これは、高次元Cartesian spaceを直接samplingするよりもorigamiの低次元・状態依存構造を利用できる可能性がある。

### 7.2 Feasibilityを探索生成に組み込む

Origami固有の制約として、

- compatibility
- facet/bar strain
- hinge rotation
- self-contact
- boundary condition
- flatness

をedge生成時に扱う。

単に低energy stateをsampleするのではなく、**rootから連続的にfeasibleなpathだけを保持する**。

### 7.3 Minimum barrierを直接探索する

T-RRTのtransition testは局所的なenergy incrementを確率的に評価し、厳密なminimum-energy barrierを保証しない。

そこで各node \(x\) に、

\[
C(x)
=
\max_{y\in P(x)} U(y)
\]

というrootからそのnodeまでの最大energyを持たせる。

edge \(x\rightarrow x'\) の途中も評価して、

\[
C(x')
=
\max
\left(
C(x),
\max_{\xi\in e(x,x')}U(\xi)
\right)
\]

と更新する。

\(C\) の小さいbranchを優先して展開し、terminal setへ到達するまで探索すれば、

> **reachable-state discovery と minimum-barrier path search を同じtree上で扱う**

手法になる。

この方向はT-RRTの「Metropolis確率によるenergy bias」と異なり、**bottleneck costそのものを探索目的にする**点で差別化できる可能性がある。

## 8. 数値例題の方向

手法の原理を見せるため、まず2Dまたは低自由度toy energy landscapeが有効。

例えばA, B, Cという複数のterminal stateを持ち、

- A：近いが高いbarrier
- B：一度energyが上がるが低い最大barrier
- C：広いがさらに遠い

などを意図的に設計する。

比較候補：

- random exploration
- 川口型の局所metric descent / metric sweep
- T-RRT
- 提案minimum-barrier tree search

その後、

1. 小規模linkage / branch-switching mechanism
2. non-rigid origami benchmark
3. 多自由度origami

へ進む。

## 9. 現時点での研究問題

現在もっとも整理された問いは、

> **多自由度折り構造について、具体的な終状態を事前に指定せず、指定初期状態から物理的に到達可能なconfigurationのみを探索しながら、未知の平坦／安定状態とそこへ至る低エネルギー経路を同時に発見できるか。さらに、origamiのmechanism構造を利用した局所展開とminimum-barrier costを用いることで、一般的なenergy-biased tree explorationより効率的に探索できるか。**

である。

## 10. 未解決事項

1. terminal conditionをflat stateにするかstable stateにするか、または両方扱うか。
2. 弾性energyをどのモデルで定義するか。単なるconstraint residualではなく、物理的な材料・hinge/facet stiffnessが必要。
3. branch extensionにmechanism mode、soft mode、Hessian modeのどれを用いるか。
4. random sampling / T-RRT / 川口型 metric sweep と公平に比較する計算予算。
5. minimum-barrier探索の離散化に対して、どこまで最適性を主張できるか。
6. self-contactをedge全体でどの精度で検査するか。
7. 高次元化したとき、treeのnearest-neighbor探索・重複判定・branch数をどう抑えるか。
8. T-RRTやsingle-ended energy-landscape explorationの後続研究に、origamiへより直接適用できる既往がないか追加調査する。

## 11. 現時点の注意

- 川口との差については、原論文の数式と実装条件を再確認してから最終的な新規性を記述する。
- Zhouらに対して「LHSで全stable statesを列挙した」と強く言いすぎない。論文ではstable-state finding methodを示しているが、主要数値例は既知state pairへのNEBが中心。
- 「reachableを保証」と表現する場合は、nodeだけでなくedge途中のcollision・strain・energy制約を確認する必要がある。
- T-RRTは今回の課題感に非常に近い比較対象であり、今後の新規性検討から外さない。
