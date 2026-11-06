# 現在の議論の記録

更新：2026-10-26／ChatGPT。  
[研究全体の流れ](overview.md) ／ [運用ルール](../AGENTS.md)

このファイルは、10月以降の現在の研究相談を、**本人の疑問 → AIの回答・提案 → 本人の反応 → 現時点の整理**が追えるように記録する。historyは2026-09-30までで、現在の議論はまだアーカイブしない。

---

# 1. 最新の現在地

現在の主題は、**目的節点座標を与えず、初期形状から連続的に到達可能な未知の平坦状態と、その経路を有限計算予算内で効率よく発見する方法**である。

本人が現在重視している条件は次の通り。

- 初期形状 $x_0$ は与える。
- 具体的な目的節点座標 $x_T$ は与えない。
- 終状態には平坦性と、必要に応じて広がり $D^*$ などの性質を指定する。
- 初期形状から終状態までの経路は連続であること。
- 自己交差などの不適切な状態を避けること。
- 今後は弾性変形・ひずみを許容する物理モデルを扱う。
- 低strainな経路を見つけたい。
- 無限時間ならrandom探索でも発見できる可能性があるため、研究の主眼は**有限計算予算での初回発見の高速化**に置く。

「選択的に平坦状態を探す」とは、具体的な終点座標を指定することではなく、

$$
F(x_T)\le \varepsilon_F,
\qquad
|D(x_T)-D^*|\le \varepsilon_D
$$

のように、**終状態の性質だけを指定して、その条件を満たす未知の形状を探す**ことを指す。

現時点では、SACへの置き換え、minimum-energy barrier最適化、RRT採用など、特定のアルゴリズムは確定していない。

---

# 2. 今日の議論の中心：Pure RLとRRTの弱点をどう補うか

## 2.1 本人の問題意識

これまでRLで、各状態のmechanism mode配合率を行動として未知平坦状態を探索してきた。

一方、RRT／AtlasRRTを調べると、制約多様体上をtreeで探索する方法が既に存在する。AtlasRRTでは制約ヤコビアンの核空間から接空間を構成して局所的に枝を伸ばすため、

> 「T-RRTにmechanism modeを入れる」

だけでは、既往のconstrained motion planningとの差が弱い。

しかしRRTには、接空間内に無限に近い局所方向がある中で、**どの方向が将来の平坦化に有望かを経験から学ばない**という弱点があるのではないか、と考えた。

一方Pure RLには、過去試行から有望方向を学習できる反面、

> 毎episodeを初期状態からやり直し、過去に到達した有望な途中stateを明示的に保持・再利用しない

という非効率がある。

ここから、**RRTのexplicit state memoryと、RLのlearned exploration biasの中間的な方法**を検討する方向へ進んだ。

---

# 3. RLを使う意味の再整理

## 3.1 以前の説明で弱かった点

以前はRL採用理由として、

- 終点未知
- 経路長が可変
- 途中でflatnessが悪化してもよい
- 多自由度のcontinuous action

を挙げていた。

しかし、

- 終点未知 → goal setを使うRRTでも扱える
- 可変長 → RRTでも扱える
- 途中悪化 → T-RRT等でも扱える
- constraint manifold → AtlasRRT等でも扱える

ため、これだけでは「なぜRLか」の説明として弱い。

## 3.2 現在のRL採用理由

RLに期待する本質は、

$$
a\sim\pi_\theta(a\mid s)
$$

として、同じ問題内の過去rolloutから、

> **この状態では、どの局所方向が将来の成功につながりやすいか**

を学び、action distributionそのものを有望方向へ集中させること。

RRTの主な探索圧は、Voronoi bias等による

$$
\boxed{\text{未探索領域へのspace-filling}}
$$

である。

RLでは、

$$
\boxed{\text{将来成功しそうな方向へのstate-dependent bias}}
$$

を学習できる可能性がある。

特に意味があるのは、現在のflatnessをただ改善する方向ではなく、

$$
\text{一時的なflatness/strain悪化}
\rightarrow
\text{最終的な成功}
$$

という長期的関係をreturn/valueから学習する場合。

したがって、

> **終点未知だからRL**

ではなく、

> **高次元・連続な局所行動空間に対し、将来の成功に関する探索heuristicを経験から獲得するためRLを使う**

という整理が現在は妥当。

---

# 4. Pure RLの弱点：有望途中stateを再利用できない

## 4.1 本人の疑問

例えば一度、

$$
x_0\rightarrow x_1\rightarrow x_2
$$

までかなり有望に進んだ後に失敗しても、Pure RLでは次episodeにまた $x_0$ から開始する。

RRTなら $x_2$ がtreeに残るため、

$$
x_2\rightarrow A
$$

が失敗した後、

$$
x_2\rightarrow B
$$

を試せる。

この差を利用できないか。

## 4.2 現在の案

過去に実際に到達したstateをarchiveし、

$$
\mathcal A=\{c_0,c_1,\ldots,c_M\}
$$

次episodeの一部をarchive中のstateから開始する。

checkpoint $c_i$ には、単なるconfigurationだけでなく、

$$
c_i=
\left(
x_i,\,
P_i,\,
B_i,\,
F_i,\,
D_i,\ldots
\right)
$$

を保持する案がある。

- $x_i$：途中形状
- $P_i$：初期状態からそこまでのprefix path
- $B_i$：prefix中の最大strain等のpath-dependent量
- $F_i$：flatness
- $D_i$：spread

低strain pathを扱う場合、同じconfiguration $x$ に到達しても、

$$
B=0.5\%
$$

で来た場合と、

$$
B=3.0\%
$$

で来た場合は同じcheckpointとして扱えない可能性がある。

---

# 5. Archive-start RLでの学習とcredit assignment

## 5.1 本人の疑問

昔のpolicyで

$$
x_0
\xrightarrow{a_0^{old}}
x_1
\xrightarrow{a_1^{old}}
x_2
\xrightarrow{a_2^{old}}
x_3
$$

まで到達し、$x_3$ を保存したとする。

その後、現在policyで $x_3$ から再開して、

$$
x_3\rightarrow x_4\rightarrow\cdots\rightarrow x_T
$$

と成功した場合、その成功rewardを、昔のprefix $x_0\to x_3$ のpolicyへどう反映すべきか。

## 5.2 現時点の整理

checkpoint以降のsuffixは現在policyで生成するため、Actorのon-policy更新にそのまま使える。

一方、古いprefixのactionは旧policyで生成されているので、新しいsuffix rewardを

$$
\nabla_\theta\log\pi_\theta(a_t^{old}\mid s_t)G_t
$$

として現在Actorへ直接流すのは、off-policy補正なしでは不適切。

基本案は、prefixとsuffixを分離する。

1. **Archive-start episode**  
   checkpointから現在policyで新しいsuffixを探索し、Actorはsuffixだけでon-policy更新する。

2. **Critic**  
   archive-startで得た結果からcheckpointの価値 $V(s_k)$ を学習する。

3. **Root-start episode**  
   一定割合のepisodeは本来の $x_0$ から現在policyで開始する。

4. **価値伝播**  
   TD型Criticなら、

   $$
   V(s_t)\approx r_t+\gamma V(s_{t+1})
   $$

   により、archiveで得た価値をroot側へ徐々に伝播できる。

つまり、

$$
\boxed{
\text{archiveから先を探索}
\rightarrow
V(\text{checkpoint})\text{を改善}
\rightarrow
\text{rootからfresh rollout}
\rightarrow
\text{prefix policyへ価値を伝播}
}
$$

という構成。

保存prefixをActorのpolicy-gradient教師として直接再利用するのではなく、必要なら成功trajectory完成後にbehavior cloning的な補助教師として扱う案はあるが、これは別の学習則。

---

# 6. Root-startとArchive-startを混ぜる

training時のstart-state distributionを、

$$
P(\text{root start})=\alpha,
\qquad
P(\text{archive start})=1-\alpha
$$

とする案。

archive-startだけにすると、checkpoint以降のpolicyは改善しても、

$$
x_0\rightarrow x_i
$$

へ到達するpolicyが学習されない／忘れられる可能性がある。

そのためroot-startを一定割合残す必要がある。

ここで今後重要になるのは、

- $\alpha$ を固定するか適応的にするか
- archiveからどのcheckpointを選択するか

である。

---

# 7. Restart-state RL自体には既往がある

途中stateを保存してそこから再開する考え方自体には既往があるため、

> **restartを導入したことそのもの**

は新規性にはできない。

確認した主な系統：

- **Exploring Restart Distributions / Prioritizing Starting States for Reinforcement Learning**  
  過去に訪問したstateをrestart distributionとして利用。on-policy/off-policy双方を想定。

- **Go-Explore**  
  promising stateをarchiveし、そこへreturnしてから新しくexploreする。

- **LiDER**  
  past stateへ戻り、current policyでsuffixを再rolloutする。

- **Reverse Curriculum Generation**  
  start-state distributionを学習進度に応じて変える。ただしgoal既知でgoal側からstart集合を拡張する。

したがって、現在の候補手法は、

$$
\boxed{\text{restart RLをorigamiへ適用}}
$$

だけでは不十分。

---

# 8. Origami configuration-space探索側の既往

折り紙／折り構造のconfiguration-space探索には既往がある。

- Xi & Lien：discrete-domain sampling＋PRM系
- Guo・Tachi・Yu：RRTによるrigid-origami path planning
- Tachi／Li：constraint tangent space・mechanism modeに近い局所経路生成
- AtlasRRT系：一般のconstraint manifold上でtangent-space sampling／projection
- Zhouら：elastic origamiのstable-state finding＋既知状態間NEB
- T-RRT：一般energy landscapeで初期状態からreachable minimaをtree探索

一方、今回確認した主要なorigami configuration-space探索文献では、

> **初期折り状態からreachableな途中stateをexplicit archiveし、そのstateからRL policyを再始動・更新しながら、未知flat stateを探索する**

という組合せは見つかっていない。

ただし、これは現時点の調査範囲での判断であり、「存在しない」とは断言しない。

---

# 9. Zhouらとの関係

Zhouらでは、

$$
\text{LHS＋局所最適化}
$$

でstable stateを探索し、

$$
\text{NEB}
$$

で既知のstate pair間のlow-energy pathを求める。

LHSでstateが見つかっても、

$$
x_0\rightarrow x_T
$$

が初期状態から、

- 自己交差なし
- 許容strain内
- 連続的

に到達できることは、そのstate discovery自体では確認されない。

この点から、

> **存在する状態ではなく、初期状態から実際に行ける状態とpathを同時に探したい**

という現在の問題意識が生じた。

ただし、この「reachable state + pathの同時探索」だけならT-RRT等が既に存在するため、それ自体を新規性にはできない。

---

# 10. 川口法との最も重要な比較

ここが現在の新規性検討で最も中心的。

## 10.1 本人の問題意識

川口法でも、目的形状を一点指定せず、ある計量

$$
r=\mathcal R(x)
$$

を改善するように折り進められる。

さらに、

$$
\mathcal R_1,\mathcal R_2,\ldots
$$

と計量を多数変えて高速に試せば、未知平坦状態を探索できる。

したがって、

> 「初期形状から連続的に未知平坦状態を探索する」

だけでは、川口との差を説明できない。

## 10.2 現時点で最も本質的と考えている差

川口型では、

$$
\boxed{\text{探索heuristicとなる計量を人間が事前に設計する}}
$$

のに対して、現在のRL案では、

$$
\boxed{\text{終端条件だけを与え、途中状態・行動の価値を探索経験から学ぶ}}
$$

ことを目指す。

例えば川口型では、

$$
\mathcal R(x)
$$

が現在stateの「良さ」を表す。

RLでは、

$$
V(x)
$$

または

$$
Q(x,a)
$$

によって、

> 現在は平坦でなくても、その先に成功経路があるstate/action

を高く評価できる可能性がある。

つまり、

$$
\boxed{
\text{現在の幾何学的良さ}
\neq
\text{将来の到達可能性}
}
$$

を学習対象にする。

「別の計量 $r_2$ を設計すれば川口法でも見つけられる」という反論に対しては、

> **その $r_2$ を事前にどう設計するか自体が問題であり、本研究では成功・失敗した探索経験から中間state/actionの価値尺度を獲得する**

という位置付けが候補。

## 10.3 Adaptive searchとしての差

metric sweepでは、

$$
r_1\rightarrow path_1,\qquad
r_2\rightarrow path_2,\qquad
\cdots
$$

と複数試行できるが、それぞれの試行結果を使って次の探索heuristicが自動的に改善されるわけではない。

現在の候補法では、

$$
\text{trial}
\rightarrow
\text{state/archive・成功失敗情報を保存}
\rightarrow
\text{policy/value更新}
\rightarrow
\text{次の計算資源配分を変更}
$$

とする。

したがって比較仮説は、

$$
\boxed{
\text{hand-designed non-adaptive metric sweep}
\quad\text{vs}\quad
\text{outcome-adaptive learned search}
}
$$

となる。

さらにarchiveを使えば、複数metricで毎回rootから似たprefixを再計算する代わりに、一度到達したreachable prefixを再利用できる可能性がある。

ここが、今日の議論で最も新規性候補として重要になった。

---

# 11. 現在の研究問題の候補

現時点では、次のように整理するのが最も近い。

> **多自由度折り構造について、具体的な目的形状を与えず、平坦性・広がり等の終端条件だけを与える。初期形状から自己交差を避け、許容strain内で到達できる未知平坦状態と経路を有限計算予算内で発見する。探索中に得られた成功・失敗経験から将来の成功に有望な局所方向を学習すると同時に、過去に到達した有望stateとそのprefixを明示的に保持・再利用することで、Pure RL、RRT系、手設計metric sweepより初回発見を効率化できるかを検証する。**

これは現在の作業仮説であり、新規性・性能は未実証。

---

# 12. 研究目的と評価指標

## 12.1 主目的

本人の現時点の意向：

$$
\boxed{\text{低strainな成功pathを、できるだけ早く初回発見する}}
$$

無限計算時間での完全探索ではなく、有限予算内の探索効率を問題にする。

## 12.2 終端条件

候補：

$$
F(x_T)\le\varepsilon_F,
\qquad
|D(x_T)-D^*|\le\varepsilon_D
$$

具体的な目的座標は指定しない。

## 12.3 経路品質

strainをどの物理量で定義するかは未確定。

候補として、経路中最大strain

$$
B(\tau)
=
\max_t \varepsilon_{\max}(x_t)
$$

を用いる。

ただし、これを

- hard constraintにするか
- reward/scoreにするか
- 補助評価にするか

はまだ決めていない。

## 12.4 探索効率

最低限分けて評価する候補：

- 初回成功までの力学評価回数
- wall time
- 成功率
- 発見pathの最大strain
- 打切り／未到達
- seed間ばらつき

学習に使った計算も探索予算へ含める。

---

# 13. 比較対象

最低限のbaseline候補：

1. **一様random / Pure RLの既存方式**
2. **川口型 metric-driven search / metric sweep**
3. **RRT / AtlasRRT系**
4. **T-RRT**
5. **restartなしPure RL**
6. **一般restart-state selection**
7. **提案 archive + learned-policy search**

局所変形生成方法と探索方策を分離し、比較時にはできるだけ共通化する必要がある。

---

# 14. 新規性について現在言えること／言えないこと

## 言えないこと

- 目的形状を指定しないこと自体が新規。
- ひずみを許すこと自体が新規。
- mechanism modeでRRTを伸ばすこと自体が新規。
- reachable stateとpathを同時に探すこと自体が新規。
- 途中stateからRLをrestartすること自体が新規。
- RLを使うこと自体が新規。

これらにはそれぞれ近い既往がある。

## 現在の新規性候補

単独要素ではなく、次の**探索構造と問題設定の組合せ**。

1. 目的形状ではなくflatness／$D^*$等のterminal conditionのみを指定。
2. rootから実際にfeasibleに到達したstateとprefixを明示archive。
3. 過去rolloutから、将来成功に有望な局所actionをstate-dependentに学習。
4. archive stateからrestartして、同じprefix計算を繰り返さずfrontierを拡張。
5. 低strainな成功pathの初回発見を有限予算内で高速化。
6. 手設計metricを大量にsweepする探索に対し、探索結果からheuristic／計算資源配分をadaptiveに更新。

ただし「既存要素を組み合わせたこと」だけでは不十分であり、**この組合せによって何が改善するかを比較実験で示す必要がある**。

---

# 15. 次に決めること

現時点でアルゴリズム実装へ進む前に、次を詰める。

### A. Archiveの定義

- どのstateを保存するか。
- 近いstateをどう重複判定するか。
- 同じconfigurationでもprefix strainが違えば別stateとするか。
- self-contact／重なり順等の履歴情報をどう保持するか。

### B. Restart-state選択則

最も重要な未決事項。

候補情報：

- flatness progress
- $D^*$ への接近
- prefix strain
- novelty／未探索性
- visit count
- Critic value
- policy uncertainty
- 学習進捗

restart選択則自体を手設計するのか、bandit／value／別policyで学習するのかも未確定。

### C. Root-start / Archive-startの配分

$$
P(\text{root})=\alpha
$$

を固定するか、学習進捗に応じて変えるか。

### D. Critic

現在のMonte Carlo $V(s)$ のままか、TD/GAEでarchiveから得た価値をroot側へ早く伝播させるか。

SACへの全面移行は一旦保留。

### E. 物理モデル

現在コードのconstraint residualを材料strainと同一視しない。

- bar/facet strain
- hinge energy
- elastic soft mode
- self-contact

などをどうモデル化するかは別途決める。

### F. 川口との差が出る数値例

特に、

- 一つの固定metricでは局所的に誘導しにくい
- metricを多数sweepするとprefix計算を重複する
- 探索経験から「今は悪いが将来良い」stateの価値を学べる
- archive restartでそのfrontierを効率的に掘れる

という特徴が現れる低自由度例を考える。

---

# 16. 文献上の主な比較先

現在の中心的比較先：

- 川口：骨組み構造の畳み込み解析（1997）
- Zhou et al.: *Low energy fold paths in multistable origami structures* (2023)
- Xi & Lien：rigid origami discrete-domain sampling / PRM
- Guo・Tachi・Yu：rigid origami RRT
- AtlasRRT系：constraint manifold上のsampling-based planning
- Jaillet et al.: T-RRT / T-RRTmin
- Goal-set CHOMP
- SMTO
- Exploring Restart Distributions
- Go-Explore
- LiDER
- Reverse Curriculum Generation

これらの存在により、単純な構成要素ではなく、**何を事前に与え、何を探索結果から獲得し、有限予算内で何を改善するか**を研究貢献として明確化する必要がある。

---

# 17. 現時点の一文要約

> **目的形状を直接指定せず、平坦性・広がり等の終端条件だけを与えた多自由度折り構造について、初期形状から低strain・非自己交差で到達した途中stateをarchiveし、そこから現在policyで再探索する。RLによって将来成功に有望な局所方向を学習し、explicitなreachable-state memoryと組み合わせることで、手設計metric sweep、RRT系、Pure RLより有限計算予算内の初回平坦経路発見を効率化できるかを検証する。**

これは2026-10-26時点の作業仮説であり、最終研究目的・新規性の確定ではない。

---

# 18. 根拠と記録上の注意

- 今日の中心的な整理は、2026-10-26の本人とChatGPTの会話を基にした。
- 以前のZhou・LHS・NEB・T-RRT・Goal-set CHOMP・SMTO比較は、同月の会話・currentの既存記録を背景として統合した。
- 9月以前のActor–Critic、多様解探索、転移の実験詳細は [history/](history/) を参照する。
- 過去に得たfive等でのrandom超えは保存結果であり、今回再実行していない。
- restart RLやorigami configuration-space探索の網羅的な文献調査はまだ完了していない。「主要文献で未発見」と「既往が存在しない」を区別する。
- 新しい探索法の実装・比較実験はまだ行っていない。

## 2026-10-26／Codex：GitHub更新のローカル反映

本人の依頼でGitHubのmainを取得し、ChatGPT側の5コミットをローカルへfast-forwardで取り込んだ（取得時先端 `43c949e55`）。更新されたcurrentを読み、現在の作業仮説が「到達済み途中stateとprefixのarchive＋学習した局所方向による、低strainな平坦経路の初回発見効率化」へ具体化されたことを確認した。選択性は平坦性・広がり等の終端性質の指定として記録されている。研究実装・実験を開始する指示とは扱わない。

未追跡の実験出力は保持し、研究コードは変更していない。これは依頼時点の同期であり、常時監視・自動取り込みは設定していない。以後の作業開始時も最新mainを取得し、未保存変更と比較してから取り込む。

---

# 19. 2026-11-04：発表済み成果から研究モチベーションを組み直す議論

## 19.1 現在までに発表している内容

現時点で発表済みなのは、**ひずみを考慮しないPure RLによる折り経路探索で、five barとtorusを平坦化できた**ところまで。

当時の基本的な定式化は、

- state：現在形状＋mechanism mode等
- action：mechanism modeの配合率＋step幅
- terminal condition：具体的な目標節点座標ではなく、平坦性等の条件

として、各stepで

$$
\Delta x_t=\sum_k a_{t,k}m_{t,k}
$$

のように機構モードの配合を選ぶ逐次意思決定問題として扱うもの。

当初RLを選んだ主な理由は、現在のflatnessを毎step単調に改善するだけでなく、

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
$$

という将来returnを通じて、**一時的に評価値が悪化しても最終的な平坦化につながるactionを評価できる**と考えたため。

five bar／torusの結果は、

> **具体的な終状態や折り順序を事前に指定しなくても、mechanism mode上の方策を学習して、平坦化へ至る連続経路を生成できる**

というproof of conceptとして位置付けていた。

### 注意：表現上の修正

「目的形状なし」と言うと「目的がないのになぜ畳むのか」と誤解されやすい。

本当に言いたいのは、

> **具体的なtarget configurationを事前指定しない**

ことであり、平坦性、収納寸法、広がり等の**terminal performance conditionは与える**。

今後はこの表現を優先する。

---

## 19.2 研究モチベーションへの批判

発表に対して、

> 「そもそも具体的な目的形状を与えずに畳みたくなる場面はあるのか。なぜこの研究が必要なのか分からない」

という指摘を受けた。

これに対して、月面・宇宙輸送などで限られた収納空間へ構造物を格納する必要がある、という応用例を説明したが、

$$
\text{用途がある}
\;\Rightarrow\;
\text{なぜこの探索手法が必要か}
$$

まで十分につながらず、説得力が弱かった。

別の研究者から、

> **既往研究で何ができておらず、それがなぜ問題で、その問題を解くためにこの手法が必要なのか、というストーリーを作るべき**

という助言を受けた。

この観点から、川口型のmetric-driven foldingとの比較を中心に研究意義を再構成することを検討した。

---

## 19.3 川口型との対比：現在考えている本質

川口型を抽象化すると、現在configuration $x$ に対して人間が

$$
R(x)
$$

というgeometrical measurer／探索heuristicを設計し、その計量に基づいて折り方向を選択する方法とみなせる。

これに対してRLでは、成功・失敗したrolloutから

$$
V(s),\qquad \pi(a\mid s)
$$

を学習し、**どの途中state／actionが将来の成功につながるか**を経験から獲得する。

現在の有力な対比は、

$$
\boxed{\text{hand-designed guidance}}
\qquad\text{vs}\qquad
\boxed{\text{learned guidance}}
$$

である。

特に重要なのは、

$$
\boxed{
\text{現在のconfigurationとしての良さ}
\neq
\text{そこから将来成功できる価値}
}
$$

となる場合。

ただし、**川口型では非単調経路を扱えない**、あるいは**RLだけが将来を考慮できる**と主張してはいけない。

正解経路を知っていれば、その経路へ誘導する複雑な計量を川口型にも埋め込める。

したがって本当に問題にしたいのは、

> **将来成功するための中間誘導を、事前にどのような計量として設計すべきか分からない**

という点。

RLの候補的な価値は、

> **正解経路や正しい中間誘導則を先に与えず、成功・失敗経験からそれを獲得する**

ことにある。

---

## 19.4 「川口ではできない例」が見つかった場合／見つからない場合

研究の強さには段階がある。

### より強い「前への拡張」

合理的な川口型計量ではうまく折れない、または非常に探索しにくい例があり、

$$
\text{川口型}\rightarrow\text{失敗／著しく非効率},
\qquad
\text{提案法}\rightarrow\text{成功}
$$

を、その原因まで含めて示せる場合。

この場合は、既往で扱いにくかった問題クラスを扱えるようにしたという「前への拡張」と説明しやすい。

### 「横への拡張」になり得る場合

川口型でも計量を変えればほとんどの例を容易に平坦化できる場合。

この場合、

> 「川口型でも解ける問題を別の方法でも解いた」

だけでは弱い。

そのときは、

- 構造ごとに正しい計量を人が選ぶ必要があるか
- 多数のmetric sweepが必要か
- 成功・失敗経験を次試行へ利用できるか
- 有限計算予算内の初回成功までの評価回数

など、**自動性・適応性・探索効率**で比較する必要がある。

無理に「川口では絶対解けない」と主張せず、実際の結果に応じて研究貢献を位置付ける。

---

# 20. 例題を用いて研究の有効性を語る方針

今日の重要な整理は、**一つの例題ですべてを主張しようとしない**こと。

例題ごとに、

1. 何を主張するための例なのか
2. その例から何を主張してはいけないのか

を明確に分ける。

本人は例題のインパクトから主張を広げすぎやすいため、以下を特に注意事項として残す。

---

## 20.1 「一度開いてから畳む」例

### この例で主張したいこと

「一度開くこと」自体を研究価値にするのではない。

この例は、

> **多自由度折り構造では、局所的に収納性を改善するactionが、将来成功するactionとは限らない**

ことを最小構造で可視化するための例。

例えば成功経路が、

$$
F:
1.0\rightarrow1.3\rightarrow1.5
\rightarrow0.8\rightarrow0
$$

のように、一時的にflatness等を悪化させる必要があるとする。

これはより一般的な、

- folding sequence
- branch selection
- self-interference回避
- dead-end回避
- 一時的な目的指標悪化

といった問題の象徴として用いる。

したがって一般化した主張は、

> **複雑な折り順序や分岐が存在するため、将来成功するための中間誘導を単純な局所計量として事前設定することが難しい場合がある。提案法では、将来returnからそのようなactionの価値を学習する。**

というもの。

### この例で主張してはいけないこと

- **「世の中のdeployable structureは一度開いてから畳むものばかりなので、この研究が必要」ではない。**
- **「unfoldできること自体が提案法の新規性」ではない。**
- **「川口型ではunfold→refold経路を表現できない」ではない。**
- **「RLだけが将来を考慮できる」ではない。**

川口型でも、

$$
R(q_L,q_R)
=
(q_L-q_L^*(q_R))^2+wq_R^2
$$

のように、正しいsequenceをあらかじめ計量へ埋め込めばunfold→refoldを誘導できる。

ここで本当に示したいのは、

> **その正しい中間誘導を事前に知らなくても、必要ならRLがunfoldという一見逆向きのactionすら学習結果として獲得できる**

という点。

「unfoldができる」ではなく、**「将来成功を考慮したsequenceを経験から獲得できる」**が主張。

---

## 20.2 川口型との小規模比較例

### この例で主張したいこと

toy problem／低自由度例では、

- configuration space
- branch
- collision／dead end
- metricの等高線
- RL trajectory

などを可視化し、

> **なぜ事前計量による誘導が難しくなり得るか**

を説明する。

もし川口型で実際に失敗するなら、それは強い結果。

川口型でも成功する場合も、

- 計量によって成否が大きく変わる
- 適切な計量の事前選択が必要
- metric sweepに評価回数を要する
- RLは同じterminal conditionのまま試行結果から誘導を更新する

という差を定量化する。

### この例で主張してはいけないこと

- 一つの計量で川口型が失敗しただけで「川口法では解けない」と言わない。
- RLが成功しただけで「RLの方が一般に優れる」と言わない。
- 後から正解経路を見て設計した計量との比較を、事前知識なしの公平比較として扱わない。

---

## 20.3 Five bar / Torus

### この例で主張したいこと

現在までの発表結果として、

> **具体的なtarget configurationや折り順序を事前指定せず、mechanism mode上のPure RLによって平坦化経路を獲得できる**

ことを示す。

five barは基本原理、torusはより複雑な多自由度構造でも同じ枠組みが成立することを示す例として位置付ける。

### この例で主張してはいけないこと

- five bar／torusが畳めたことだけで、RLが川口型やRRTより優れているとは言わない。
- 「目的形状を与えないこと自体が新規」とは言わない。
- これらの成功だけで、実用deployable structureへの有効性が証明されたとは言わない。

---

## 20.4 Stanford Bunny等のキャッチーな複雑形状

別の研究者から、

> **キャッチーな例題を示すことで、「なぜ畳むのか」という議論から、「どうやってこれを畳んだのか」という議論へ視線を移せる**

という助言があった。

Stanford Bunnyのような複雑形状は、この役割の**hero demo／benchmark**として考える。

### この例で主張したいこと

deployable structuresでは、複雑な立体形状を限られた収納空間へ格納するため、folding path設計が必要になる。

本研究は、その経路設計を自動化するための基礎的方法を扱う。

Bunnyは、

> **複雑な形状に対しても経路探索を適用できることを、直感的・視覚的に示すbenchmark**

として用いる。

つまり、

$$
\boxed{\text{Bunny = applicability / complexity benchmark}}
$$

である。

### この例で主張してはいけないこと

- **「ウサギを畳む実用需要があるから研究する」ではない。**
- **「Bunnyを畳めたからRLが川口型より優れている」ではない。**
- **「Bunnyだけで研究の新規性を証明した」ではない。**
- 川口型でBunnyを畳めるかどうかを、このhero demoの主張に必ずしも含める必要はない。

Bunnyの役割は**研究の必要性を比較実験で証明することではなく、手法の適用可能性とインパクトを見せること**。

---

# 21. 例題の役割分担

今後の発表・論文では、例題を次のように分業させる構成が有力。

### 原理・必要性を示す例

低自由度のbranch／sequence問題。

> **なぜ単純な局所計量だけでは不十分になり得るのか、なぜ将来価値を学習する意味があるのか**

を説明する。

### 定量評価をする例

five bar、torus、または比較可能な折り構造。

> **成功率、初回発見評価回数、計算時間等を比較し、探索法としての性能を示す。**

### Hero demo

Stanford Bunny等。

> **複雑形状への適用可能性を一目で伝える。**

この三者の役割を混ぜない。

$$
\boxed{\text{toy example = Why}}
$$

$$
\boxed{\text{quantitative examples = How well}}
$$

$$
\boxed{\text{hero demo = What can it handle}}
$$

という整理。

---

# 22. 現時点の研究ストーリー候補

現在の議論を踏まえると、発表ストーリーは次の形が有力。

### Why

deployable structuresでは、複雑な立体形状を限られた収納空間へ格納するため、適切なfolding pathの設計が必要となる。

### Problem

多自由度の折り構造には多数のfolding branchやsequenceが存在し、自己干渉やdead endを避けながら成功するには、局所的に「より畳まれた状態」へ進めばよいとは限らない。

そのため、

> **成功へ導く中間的な探索計量を事前に設計することが難しい場合がある。**

### Approach

具体的なtarget configurationや折り順序を与えず、terminal performance conditionを与える。

過去の成功・失敗rolloutから、

> **どのstate／actionが将来の成功につながるか**

をRLで学習し、folding sequenceを探索する。

### Evidence

- 最小例：局所評価と将来成功が一致しないことを示す。
- five bar／torus：Pure RLで平坦化経路を獲得できることを定量的に示す。
- 複雑形状benchmark：適用可能性を視覚的に示す。

このストーリーでは、

> **「目的形状なしだからRL」ではなく、「複雑なsequenceの中で、成功へ導く中間誘導を事前設計せず経験から獲得したいからRL」**

という説明へ軸足を移す。

---

# 23. 今後の重要な検証

今日の議論から、次の二つの例題探索を並行して進める価値が高い。

1. **研究上の比較例**  
   川口型の自然な計量では誘導が難しく、RLの将来価値学習の意味が見える小規模構造。

2. **キャッチーなbenchmark**  
   手法の複雑形状への適用可能性を視覚的に示せる対象。

前者は**研究の必要性**を支え、後者は**研究の魅力・適用可能性**を支える。

この二つを同じ主張の証拠として混同しないこと。

## 2026-11-05／Codex：11月4日の議論をローカルへ反映

本人の依頼でGitHubのmainを取得し、`e85e99077`（研究モチベーションと例題の役割の記録）をfast-forwardで取り込んだ。第19〜23節を読み、発表済み成果はひずみなしPure RLによるfive bar／torusの平坦化であり、archive・ひずみ対応の提案とは分けることを確認した。研究意義は「具体的なtarget configurationを指定せず、終端性能条件を与え、成功へ導く中間誘導を経験から獲得する」という説明を中心にする。原理説明用の小規模例、定量比較例、複雑形状の適用デモの役割を分ける整理を引き継ぐ。

今回の確認は共有ノートの読解とGit同期のみ。原論文の再検証や新しい実験は行っていない。未追跡の実験出力を保持し、研究コード・historyは変更していない。例題探索・実装の実行指示とは扱わず、次の相談ではこの最新記録を前提とする。
## 2026-11-06／Codex：次に何から着手するかの相談

**本人の問い**：今後の方針として何がよく、まず何をするべきか。

**Codexの提案（本人の採用判断前）**：まず既存のひずみなしfive bar／torusを使い、同じ局所変形・衝突判定・終端条件の下で、軽量な計量誘導と既存Pure RLを比較する小規模な診断を優先する。ひずみ対応という中期方向は維持しつつ、新物理モデル・archive・大型デモを同時に導入する前に、「学習した誘導が何を改善するか」の根拠を作る。川口原法を再現していない実装は川口型baselineと呼び、原法の限界と同一視しない。

診断案：既存成功例とその近傍の初期形状・広がり条件を、成否を見て選別せず小さな固定集合にする。一様random、複数の事前設定した幾何学計量／そのsweep、既存Pure RLを、学習・失敗試行を含む総費用で比較する。計量・報酬の調整費用と独立評価を区別し、第一段階は探索的診断と明記する。成功率、打切りを含む初回発見の力学評価回数・実時間、経路の実行可能性を記録する。予算・seed数は実行時間を測ってから決める。

結果による分岐：計量法が容易に解くなら、その例で将来価値学習の必要性を主張せず別の条件・貢献を検討する。計量による成否差がありRLが改善するなら、成功経路の非単調性だけでなく、学習前後・学習なしとの比較から原因を検討する。RLの主な無駄が同じprefixの再計算ならarchiveの有効性を切り分ける次段階へ進む。全方式が失敗する場合は可解性と数値処理を先に確認する。特定手法が失敗したことは到達不能の証明ではない。

最初の成果物は「共通条件による比較表と、差または共通の失敗が現れる代表例の経路・指標推移」。Bunny等は適用デモとして後段に置く。複数の枝を持つ小規模な弾性例で、厳密機構運動では到達できない領域へ低ひずみで移れることの確認は、ひずみ対応段階の別課題とする。

今回の作業は最新ノートの確認と方針提案のみ。文献の新規性調査、コード変更、実験は行っていない。この提案を実行指示や採用済み方針とは扱わない。