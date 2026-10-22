# ori_codex6 二タスク 6144 episode 最終報告

## 結論

学習は停止済みであり、本解析から追加学習は実行していない。torus D=2 と waterbomb D=1 は、ともに厳密成功 0/6144 だった。したがって、現候補 `P0X2RCS05` をそのまま 10000 episode まで延長する根拠はない。

ただし、単に「惜しい状態が得られなかった」のではない。waterbombでは最良の初期joint stateが反復crossing-reject報酬により強い負returnへ反転した。一方、torusのconstructive episodesは現行報酬でもnonconstructive episodesより高returnであり、それでも良い領域を失った。したがって、二タスクをreject penaltyという単一原因だけで説明することはできず、torusではon-policy保持、action-level credit assignment、critic/updateの不安定性も未解決の有力候補である。

- waterbomb D=1 では、episode 5 に `F=0.00766893`, `spread error=0.00002618`, `near-miss q=0.705088` という高品質状態が既に得られていた。しかし reject 13回の合計が `-24.0496` となり、実returnは `-21.2971`、reject項を除いた反実仮想returnは `+2.75249` だった。
- torus D=2 では、episode 1025–1536 に constructive endpoint 432/512、valid joint endpoint 17/512、positive near miss 496/512 が得られた。この区間の平均return `-0.676859` は、崩壊後1537–6144の `-0.967316` より既に高かったが、それでもepisode 1537以降は広がった非平坦basinへ完全に崩壊した。
- waterbombの学習曲線は、return上昇とterminal flatness低下を明瞭に示す。一方でspread errorは約0.40へ悪化しており、正しい平面へ入れ替わった成功方向ではなく、平坦だが幅の違うbasinを学習した。
- torusの学習曲線は、1025–1536だけreturn・flatness・spreadが同時改善するが、その後は初期状態に近いbasinへ戻る。6144まで待っても再発見はなかった。

次に試すべき最小変更は、rollbackとstep消費は維持しつつ、**非終端のrejected action一回ごとの `-2+q` をまず0にし、horizonで通常のtimeout `-1+q(last safe state)` を一度だけ与える**ことである。waterbomb episode 5の明確な符号反転を除き、torusではconstructive trajectoryの優位幅をさらに広げられる。ただしtorusの現行順位は既に正方向なので、この変更だけで保持失敗まで直るとは言えない。zero-reject-penaltyは最も小さく原因を切り分けられる次ablationであり、同時にtorusのper-update advantage、critic loss、policy driftを記録して保持・credit assignment仮説も検証する必要がある。

## 学習曲線

- [torus D=2: episode軸・累積mechanics軸](learning_curve_torus2_episode_and_mechanics.png)
- [waterbomb D=1: episode軸・累積mechanics軸](learning_curve_water1_episode_and_mechanics.png)
- [二タスク比較](learning_curve_combined_episode_and_mechanics.png)

各図はepisode 6144で打ち切り、左列をepisode軸、右列を累積mechanical steps軸とした。上から順に、episodic returnの200-episode移動平均と累積平均、terminal flatnessの移動平均・移動中央値（対数軸）、spread errorの移動平均・移動中央値、厳密成功率とtermination率、reject率とescape率、spectral・near-miss・joint信号を表示している。waterbombのepisode 5、torusのconstructive区間は図中に明示した。

## 解析終点と完全性

採用した終点は、両タスクとも最後に完全保存されたatomic checkpointの **episode 6144** である。runnerのstdoutにはtorusがepisode 6700、waterbombがepisode 6200まで表示されていたが、これらはcheckpoint・全履歴・optimizer状態が一体で保存された終点ではないため、解析から除外した。

| 検査 | torus D=2 | waterbomb D=1 |
|---|---:|---:|
| checkpoint episode | 6144 | 6144 |
| training history | 6144行、1–6144連続 | 6144行、1–6144連続 |
| update history | 768行、episode 8,16,…,6144 | 768行、episode 8,16,…,6144 |
| deterministic / stochastic eval | 各13行、episode 1,500,…,6000 | 各13行、episode 1,500,…,6000 |
| termination one-hot | pass | pass |
| configとcheckpointの意味的同一性 | pass | pass |
| model・optimizer・Torch/NumPy RNG state | 全て存在 | 全て存在 |
| 報酬・RC・spectral分解 | pass、非有限値0 | pass、非有限値0 |
| matching active process | 0 | 0 |

固定manifestは [spectral_production_two_task_full_diagnostic_d2d0b956eef9.json](../../manifests/spectral_production_two_task_full_diagnostic_d2d0b956eef9.json) で、SHA-256は `f46eff732bd96687d2ee4d033de4364db1a24b3dd5ce9695d279c54e3d6b2d9f`、再計算fingerprintは `d2d0b956eef9dce6b2b5a19730624a588377f4178066d90fb4eea718af215c00` と一致した。manifest記載のsource hashとmodel hashは全て一致した。

runnerは両方ともstatus `failed`、return code `4294967295`、stderr 0 byteである。これは学習ロジックが例外終了したという証拠ではなく、10000 episode到達前に外部停止され、finalization artifactを作らなかったrunとして分類した。episode 6144のatomic checkpointと対応履歴は完全である。

最大絶対分解誤差は、torusで通常報酬 `3.12e-14`、reject報酬 `3.33e-16`、spectral報酬 `1.99e-14`、waterbombでそれぞれ `2.48e-14`, `1.04e-16`, `2.17e-14` だった。非有限成分は両タスクとも0である。詳細は [final_metrics_and_integrity.json](final_metrics_and_integrity.json) に保存した。

## 指標の定義

- `F`: terminal flatness ratio。小さいほど平坦で、strict基準は `F <= 0.001`。
- `E`: `abs(spread - target spread)`。小さいほど目標幅に近く、strict基準は `E <= 0.05`。
- strict success: collision-freeで `F <= 0.001` かつ `E <= 0.05` を満たす終了。
- constructive proxy: `F + E^2 <= 0.5`。完全成功より緩いが、平坦性と幅を同一状態で同時評価する。
- valid joint: `F <= 0.1` かつ `E <= 0.05`、またはstrict success。
- positive near miss: terminal near-miss品質 `q > 0`。

flatnessだけの最小値とspread errorだけの最小値は別episodeでも達成できるため、それらを組み合わせて「惜しい」と判断してはいけない。本報告ではconstructive proxyとvalid jointを同一endpointで再計算した。

## episode 6144時点の正確な比較

| 指標 | torus D=2 | waterbomb D=1 |
|---|---:|---:|
| episodes | 6144 | 6144 |
| mechanical steps | 1,228,800 | 2,378,429 |
| projector calls | 1,228,681 | 2,377,647 |
| policy updates | 768 | 768 |
| strict success | 0 | 0 |
| timeout | 6144 | 5680 |
| wrong flat | 0 | 461 |
| correction failed | 0 | 3 |
| constructive proxy | 477 | 2343 |
| valid joint | 17 | 1 |
| positive near miss | 924 | 5548 |
| max near-miss q | 0.432807 | 0.705088 |
| mean episodic return | -1.011792 | -0.647925 |
| total return | -6216.4487 | -3980.8540 |
| mean return without reject項（反実仮想） | -0.669916 | -0.555010 |
| rejected actions | 1070 | 288 |
| reject streaks | 557 | 228 |
| escape after reject | 515 | 227 |
| escape rate | 0.924596 | 0.995614 |

strict successは両タスクとも0である。torusの6144 episodesはすべてtimeoutだった。waterbombはtimeout 5680、wrong-flat 461、correction-failed 3で、self-crossing終了はRCによって0になった。

### torus D=2 のlearning curve

| episode | return平均 | F平均 | E平均 | constructive | valid joint | q>0 | reject |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–512 | -0.7650 | 1.1449 | 0.2878 | 0 | 0 | 1 | 0 |
| 513–1024 | -1.9938 | 0.7668 | 0.0646 | 45 | 0 | 427 | 611 |
| 1025–1536 | -0.6769 | 0.3643 | 0.0312 | 432 | 17 | 496 | 459 |
| 1537–2048 | -0.9642 | 1.0205 | 0.3967 | 0 | 0 | 0 | 0 |
| 2049–2560 | -0.9739 | 1.0131 | 0.4023 | 0 | 0 | 0 | 0 |
| 2561–3072 | -0.9689 | 1.0164 | 0.3997 | 0 | 0 | 0 | 0 |
| 3073–3584 | -0.9687 | 1.0164 | 0.3997 | 0 | 0 | 0 | 0 |
| 3585–4096 | -0.9691 | 1.0158 | 0.4000 | 0 | 0 | 0 | 0 |
| 4097–4608 | -0.9706 | 1.0150 | 0.4008 | 0 | 0 | 0 | 0 |
| 4609–5120 | -0.9648 | 1.0183 | 0.3980 | 0 | 0 | 0 | 0 |
| 5121–5632 | -0.9646 | 1.0187 | 0.3978 | 0 | 0 | 0 | 0 |
| 5633–6144 | -0.9610 | 1.0205 | 0.3961 | 0 | 0 | 0 | 0 |

最初の200 episodesと最後の200 episodesを比べると、returnは `-0.93763 -> -0.96169`、F平均は `1.03608 -> 1.02012`、E平均は `0.38385 -> 0.39643` で、6144時点に持続的改善はない。episode 513–1024の鋭いreturn低下はreject 611回と重なり、その後のconstructive windowではreturnも回復した。したがって、learning curve上の谷をそのまま幾何学的なplane-swap barrierと読むことはできず、flatness・spread・rejectを併せて見る必要がある。

episode 1025–1536だけを取り出すと、F中央値 `0.317420`、E中央値 `0.019794`、constructive 432/512、valid joint 17/512、positive q 496/512である。spectral reward平均も `0.010552`、spectral peak potential平均も `0.213286`まで上昇した。この区間の459 rejectsは243 streaksを作り、その後230回escapeしている。すなわちRCは山を越えて安全側へ進むこと自体には成功していたが、そのたびにpenaltyを累積した。

同一endpointで最良だったepisode 1411は `F=0.0503014`, `E=0.00244462`, `q=0.432807`、joint milestone 2個である。actual returnは `+0.261096`で、reject 1回の項 `-1.72090`を除けば `+1.98199`になる。rejectは良いepisodeの優位幅を大きく削っている。

しかしcohort全体では、constructive 477 episodesのactual return平均は `+0.159882`、nonconstructive 5667 episodesは `-1.110413`で、差は `+1.270295`だった。constructive window 1025–1536の平均 `-0.676859`も、崩壊後1537–6144の `-0.967316`より高い。したがってtorusについては「rejectにより良経路とwide basinのepisode-level return順位が反転した」とは言えない。

episode 1537以降はreject 0、constructive 0、valid joint 0、positive near miss 0が6144まで続いた。これは、policyが**episode-levelでは既に有利だったconstructive領域を保持できなかった**ことを示す。原因候補は、良いepisodeが短い時期に集中したon-policy分布変化、各actionへのcredit assignment、critic shock、optimizerによるpolicy driftである。reject penaltyはmarginを削る共通confoundだが、torus崩壊の単独原因とは未介入データから確定できない。

### waterbomb D=1 のlearning curve

| episode | return平均 | F平均 | E平均 | constructive | valid joint | q>0 | wrong flat | reject |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–512 | -2.3033 | 1.4224 | 0.3338 | 74 | 1 | 194 | 38 | 287 |
| 513–1024 | -0.8504 | 0.5989 | 0.3895 | 188 | 0 | 410 | 103 | 1 |
| 1025–1536 | -0.6880 | 0.5004 | 0.3962 | 168 | 0 | 464 | 73 | 0 |
| 1537–2048 | -0.5712 | 0.5472 | 0.3946 | 121 | 0 | 479 | 21 | 0 |
| 2049–2560 | -0.4946 | 0.4253 | 0.3984 | 208 | 0 | 494 | 43 | 0 |
| 2561–3072 | -0.4536 | 0.4403 | 0.3985 | 170 | 0 | 502 | 22 | 0 |
| 3073–3584 | -0.4682 | 0.3821 | 0.4003 | 243 | 0 | 497 | 45 | 0 |
| 3585–4096 | -0.4619 | 0.4380 | 0.3987 | 189 | 0 | 500 | 20 | 0 |
| 4097–4608 | -0.3765 | 0.3808 | 0.4011 | 220 | 0 | 504 | 19 | 0 |
| 4609–5120 | -0.3823 | 0.3540 | 0.4019 | 254 | 0 | 503 | 32 | 0 |
| 5121–5632 | -0.3689 | 0.3840 | 0.4007 | 227 | 0 | 500 | 16 | 0 |
| 5633–6144 | -0.3562 | 0.3422 | 0.4022 | 281 | 0 | 501 | 29 | 0 |

waterbombでは、最初の200 episodesから最後の200 episodesへ、mean returnが `-4.12728 -> -0.378318`、F平均が `1.99198 -> 0.329381`、F中央値が `2.08509 -> 0.301085` と改善した。ユーザーが期待した「累積平均returnの上昇」と「terminal flatnessの下降」は図に明瞭に現れている。

しかしE平均は `0.273405 -> 0.403467`、E中央値は `0.302122 -> 0.406667` と悪化した。最後の200 episodesはpositive near miss 196/200、constructive 117/200である一方、valid jointは0である。near-missのflatness側だけが頻繁に活性化し、正しいspreadとの積が十分にならない。したがってこれは成功経路への収束ではなく、平坦性を改善しながら目標幅を失ったwide/wrong-plane basinへの収束である。

#### episode 5 が示す決定的な符号反転

episode 5 は、現在の6144履歴で唯一のvalid joint endpointであり、最初のActor–Critic update（episode 8）より前に得られている。

| 項 | 値 |
|---|---:|
| F | 0.0076689296 |
| E | 0.0000261817 |
| near-miss q | 0.705088416 |
| joint milestones | 3 |
| flatness progress | +0.992331 |
| target progress | +0.120607 |
| spread frontier | +0.900000 |
| joint milestone reward | +1.000000 |
| terminal fixed | -27.000000 |
| terminal quality | +2.655484 |
| spectral reward | +0.034466 |
| rejects / streaks / escapes | 13 / 12 / 12 |
| rejected-crossing reward | -24.049604 |
| actual return | -21.297112 |
| reject項を除いたreturn | +2.752492 |

これはtimeoutだったものの、`F`も`E`も非常に良く、near-miss信号も強い。したがって「history[0]側ではspreadは良いがflatnessが悪く、near missが入らなかった」という仮説だけでは現在runを説明できない。少なくともepisode 5では両者が良く、signalも入っている。問題は、良いsignalが13回のreject penaltyより約700倍小さいspectral rewardと同じepisode内で相殺されたことにある。

episode 5は最初の8 episodes中でreturn最下位だった。episode 8の最初のupdateはactor loss `-0.04007`、critic loss `41.1565`、entropy `2.14488`、actor gradient norm `0.07160`、critic gradient norm `7.10453`だった。すなわち、高品質trajectoryが「強く負のreturnを持つtrajectory」として最初の更新batchへ入った。

waterbombではepisode 1025以降rejectが0になり、その後returnが滑らかに改善した。これはactorが「reject山を越えた先」を再現したのではなく、rejectを避けて平坦だがspread error約0.40のbasinへ移ったことと一致する。

## 報酬分解

| 6144 episodes累積 | torus D=2 | waterbomb D=1 |
|---|---:|---:|
| flatness progress | +291.4443 | +3055.2135 |
| target progress | +263.7314 | -217.3860 |
| spread frontier | +1219.1039 | +80.2001 |
| joint milestone | +96.3000 | +1.5000 |
| discovery total | +1315.4039 | +81.7001 |
| terminal fixed | -8284.0000 | -7208.0000 |
| terminal quality | +189.1084 | +751.4602 |
| rejected crossing | -2100.4859 | -570.8738 |
| spectral | +7.8634 | +16.9753 |

`rejected crossing`行はreject時に生じたfixed penaltyとqualityをまとめ直した診断用subtotalであり、`terminal fixed`と`terminal quality`に既に含まれる。総returnを再構成するときにもう一度加えてはいけない。これをactual returnから除いた値を、本報告ではreject項なしの反実仮想returnとしている。

RCの設計は「crossingしそうなactionをrejectしてlast safe stateへ戻す」「stepは消費する」「それまでのprogressは保持する」という遷移としては機能した。escape rateはtorus 92.46%、waterbomb 99.56%で、crossing終了は両方0である。しかし、非終端rejectのたびに `-2+q`、horizonに達したrejectでは `-3+q` を与えたため、同じplane-swap mountainを複数回試すだけで大きな負値が累積した。

このため、幾何的には「山を越える試行を許す」設計だが、報酬上は「山に触れるたびに罰する」設計になっていた。spectral項はaccepted transitionだけに与えるpotential differenceで、waterbomb episode 5の合計は `+0.0345` にすぎず、reject項 `-24.0496` を打ち消せない。waterbombのこのepisodeについては、spectral重みを先に大幅増加するより、rejectで同じpenaltyを反復しない方が小さく直接的な修正である。torusについては、この修正は優位幅を広げるが、保持失敗を直すかは実験で確かめる必要がある。

## Actor–Criticについての修正された解釈

:codex-annotation{index="1"}

「頻繁に惜しい報酬が入り、分散を下げるActor–Criticなら、その方向へ探索圧がかかるのでうまくいくのではないか」という指摘は、**惜しいtrajectoryのfull returnが代替trajectoryより高く、正のadvantageになるなら正しい**。on-policy Actor–Criticでも、反復して得られる良いsignalは十分学習できる。

waterbomb episode 5では、q=0.705でもfull returnが `-21.297`になったため、Actor–Criticがその一本を避けても報酬に忠実な挙動である。一方torusではconstructive 477 episodesの実return平均 `+0.159882`がnonconstructiveの `-1.110413`を上回っていた。ユーザーの直感どおりepisode-levelには正しい探索圧があったはずなのに、その領域は保持されなかった。

したがって、「on-policy Actor–Criticの探索・成功trajectory保持不足」だけで二タスクを説明するのも、反対にそれを二次的問題として退けるのも正確でない。waterbombでは報酬符号が先に疑われ、torusでは保持・action-level credit assignment・critic/update安定性が既に一次候補として残る。episode returnのcohort順位が正しくても、山越えを構成する個々のactionへ正しいadvantageが配られるとは限らない。zero-reject-penalty ablationと同時にper-update advantage・critic loss・policy変化を保存し、それでもconstructive領域を失うならprioritized success bufferやbehavior cloningを前倒しで検討する。

## ori5からori6までの検討を時系列で整理

### 0. ori5の5タスクbaseline

数値は [final_ori5_comparison.json](../../../experiments_ori5/analysis/final_ori5_comparison.json) に基づく。

| task | ori5 training | 初回成功 | held-out | uniform random | 解釈 |
|---|---:|---:|---:|---:|---|
| five-bar D=2 | 10000 ep | ep 585、233,824 mechanics、73 updates後 | success 0.88、crossing 0.12、mean F 0.000478、mean E 0.000186 | 6/30 success | 学習による速度・安定性改善を支持 |
| five-bar D=3 | 10000 ep | ep 3、1,033 mechanics、update前 | success 1.00 | 25/30 success | 安定だが初回hitを学習効果とは帰属不能 |
| torus D=2 | atomic ep 8192でuser stop | なし | success 0、mean F 1.01635、mean E 0.40004 | 0/10 at 102,400 mechanics | flatness near-miss dead zone |
| waterbomb D=1 | 10000 ep | なし | success 0、crossing 1.00、mean F 1.66642、mean E 0.01708 | 0/10 at 102,400 mechanics | crossing attractor |
| waterbomb D=sqrt(2) | 10000 ep | ep 47、17,324 mechanics、5 updates後 | success 0.93、mean E 0.000119 | 10/10 success | ori_codex4に対する進歩だがrandomより効率優位なし |

ori_codex4との比較で明確な機能上の追加成功はwaterbomb D=sqrt(2)だった。ただし、ori5のuniform randomも10/10成功し、mechanics RMST比はrandom/RL `1.0467`にすぎない。calibrated random wall RMSTは20.707秒、direct RL rollout+updateは110.828秒で、wall timeではRLが約5.35倍遅い。このwall比較は別に校正したrandom step時間を用いる記述的比較で、同一runの直接対戦ではない。

five-bar D=2ではrandom/RL mechanics RMST比 `15.898`、calibrated random wall RMST 8174.399秒、direct RL 1176.802秒、比 `6.946`でRL優位を支持した。five-bar D=3は初回成功が最初のoptimizer updateより前なので、見かけの大きな速度比を学習成果として扱わない。

### 1. ori5 offline診断

waterbomb D=1では、success/CMAES経路を含む既知trajectoryを再生すると、crossing直前までspreadが改善する一方、crossing trajectoryが報酬上有利になり得た。[ori5 offline rollback](../offline_rollback_counterfactual.json)では、同じori5履歴内のoriginal crossing-minus-timeout平均差が `+0.305000`で、spread/joint rollbackだけを適用した反実仮想は `-0.562032`だった。これとは別に、[P0 seed 3101の実run](../rescue_P0_reclassification.json)は差 `+0.437862`だった。`+0.438`と`-0.562`は異なるrun/cohortであり、同一trajectoryの変更前後ではない。pairedな根拠はori5内の `+0.305 -> -0.562`である。

### 2. P0: continuous spread state potential

最小変更として、hard spread milestoneを廃止し、現在状態だけのspread potential differenceへ変更した。crossing/wrong-flat時にはprogressをrollbackし、crossing penaltyは1.25とした。

- 既知pathのreturn ordering gateは3ケース中1ケースしかpassせず、waterbombのoffline seed 2本ではcrossing attractorが残った。
- torusは3 seeds x 2048 episodesでstrict success 0。held-outはseed 3101だけcrossing 0.296875、mean F 0.45539、E 0.03776まで来たが、他2 seedsはF約1.03、E約0.39だった。
- waterbomb seed 3101は2048 episodesでsuccess 0、held-out crossing 0.96875だった。

よってP0単独は棄却した。

### 3. [P0X2](../rescue_P0X2_report.json): crossing penalty 1.25から2.0

変更はcrossing penaltyを2.0へ増やしただけである。offline既知成功pathのreturn orderingは5/5 passした。

- torus seed 3103は初回成功episode 1332、263,905 mechanics、166 updates後で、2048までに11 strict successesを得た。
- しかしtorus seed 3101/3102は0 success、seed 3103を含む全seedのlast/best held-out successは0だった。
- waterbombは3 seedsすべて2048 episodesでstrict success 0。held-outはseed 3101/3102がcrossing 0.96875/0.90625、seed 3103はwide basin（F 0.3784, E 0.4018）だった。

一度のtorus発見可能性は示したが、共通候補としては棄却した。

### 4. [P0X2B12](../rescue_P0X2B12_checkpoint1024_reject_report.json): spread frontier重み 0.9から1.2

変更はfrontier maximum bonusだけである。

- torusはseed 3101で3 successes（first ep 1419）、seed 3103で2 successes（first ep 1818）、seed 3102は0。しかし全seedのlast/best held-out successは0だった。
- waterbomb 3 seeds x 1024ではstrict success 0。good-spread endpointsは前半512の270から後半512の10へ96.30%減少し、tail 256では1/768だけだった。
- waterbomb tailのwrong-flat率はaggregate 0.1641、timeout率0.8294で、frontierを強めてもwide-spread timeout/wrong-flatへ崩壊した。

報酬量不足ではなく、状態basinとreturn orderingの問題と判断し棄却した。

### 5. [P0X2E3](../entropy_diagnostic_P0X2E3_reject_report.json): entropy係数 0.001から0.003

変更はentropy係数を3倍にしただけである。最初のoptimizer update前の8 episodesは親候補と同一で、差はepisode 9から発生した。

- 1024 episodesでwaterbomb 3 seedsともstrict success 0。
- entropyは維持されたが、tail constructive seed数はP0X2の2/3からE3の1/3へ減少した。
- tailのplane-switch endpoint proxyはseed 3101の2件だけで、seed 3102/3103は0。
- seed 3102はcrossing率を0.625から0へ下げた一方、median spread errorが0.11256から0.40251へ悪化し、good-spread率が0.14844から0になった。

探索分散不足が第一原因ではないため棄却した。

### 6. [P0X2W2](../width_diagnostic_P0X2W2_reject_report.json): waterbomb max step width 0.3から0.2

既知成功pathは細分化後も到達可能であることを確認し、step幅だけを小さくした。

- 3 seeds x 1024でstrict success 0。
- tail 256のorientation-inferred plane-switch proxyは0/3 seeds、valid jointも0/3。
- seed 3102/3103のtail median spread errorは0.40953/0.41202でwide basin。seed 3101はE 0.12754まで来たがF中央値1.59590で非平坦だった。

action幅が主要bottleneckではないため棄却した。

### 7. [P0X2RC](../crossing_continuation_diagnostic_P0X2RC_reject_report.json): rejected crossing continuation

crossing候補actionをrejectし、last safe stateへ戻し、stepを消費しつつepisodeを継続するようにした。既知5成功pathはbit-exactに保持され、遷移・報酬分解testもpassした。terminal self-crossingは0になった。

- torusのformal short gateは3 seeds x 512でconstructiveが0, 0, 38、valid jointは全0、strict success 0で未達。
- waterbombは3 seeds x 1024でconstructiveが1, 0, 0、valid jointとstrict successは全0。
- reject後escape率はwaterbomb各seedで0.9886, 1.0, 0.9945と高く、遷移としては山を避けて継続できた。

ここで、各rejectが `-2+q`、horizon rejectが `-3+q` となる設計上の欠点が明確になった。同じplane-switch mountainへの複数試行を独立に罰していた。

### 8. [P0X2RCS05](../spectral_subspace_diagnostic_P0X2RCS05_formal_report.json): spectral plane-switch potential

plane入れ替えを直接促すため、centered SVDからbest-fit plane projectorを作り、singular-gap `rho` とplane距離 `delta` の積 `Phi=rho*delta` を用いた。accepted transitionだけに `0.05 * (Phi_after - Phi_before)` を与え、invalid transitionはrollbackした。spectral重み0なら親候補とbit-exactである。

offline CMAES path比較では、各タスク内のsuccess-vs-failure peak-Phi AUCは1.0だった。一方、全履歴共通のhard thresholdは存在しない。例えばtorus failure history[1]のpeakは約0.7058で、success histories[3]/[4]は約0.86だった。よってplane-switch mountainを識別する信号にはなるが、flatness・spreadとのjoint signalが必要である。

screenでは次を得た。

- torus 3 seeds x 2048: constructive 285/569/312、valid joint 22/3/5、strict successは全0。
- waterbomb 3 seeds x 1024: constructive 0/0/2、valid joint 0/0/1。seed 3203だけepisode 22でstrict successを得た。
- waterbomb episode 22は `F=0.00066095`, `E=0.00002827`, steps 222、reject 5、reject後success 1、total return `+7.33384`、spectral total `+0.03733`だった。

可能性は見えたが1/3 seedだけなのでformal screenは棄却判定だった。ユーザー判断によりfive-barの完全nonregressionより二タスク長期診断を優先し、seed 4101を開始したのが今回の6144 runである。

### 9. five-bar D=2 の部分的nonregression

[checkpoint_512.json](../../early_gates/spectral_nonregression/P0X2RCS05/five_d2/checkpoint_512.json) のみが完了している。

- candidate successesはseed 3301/3302/3303で75/2/2、first successはepisode 298/432/315。
- ori5 reference prefix successesは28/10/0、first successはepisode 334/373/565（seed 3303のreferenceは1024まで実行され、512後に初成功）。
- reward/transition/spectral integrityはpassし、episode 512時点のcatastrophic regression evidenceはない。

しかしcandidateは512、referenceは一部1024で、事前登録した2048 episode・held-out 64評価は完了していない。five-bar D=3とwaterbomb D=sqrt(2)もnonregression未実施である。したがって「five-barへ影響なし」「共通候補が非劣性を証明済み」とは言えない。正確な表現は、**five-bar D=2に早期の壊滅的回帰は見えなかったが、非劣性検証は部分的**である。

plane入れ替えを要しないtaskではspectral重みを0にする設計は合理的で、同一報酬体系のtask hyperparameterとして扱える。ただし、RCのreject処理はspectral重みとは別なので、five-barも含めて最終的な非回帰試験が必要である。

### 10. 今回のtwo-task 6144診断

今回の目的は10000 episode完走そのものではなく、学習曲線に持続的なreturn上昇とflatness低下が現れるかを見て、可能性がなければ約6000で止めることだった。

- waterbombはreturn上昇とF低下を達成したが、Eが約0.40へ悪化し、成功方向ではなかった。
- torusはepisode 1025–1536に強いconstructive signalを作ったが、1537以降6144まで完全に失った。
- 両タスクともstrict success 0で、現設計の10000延長は中止条件を満たす。

したがって、学習停止は妥当である。特にtorusはori_codex4のように「2000以降に上がり始める」型ではなく、2000より前に上がった有益信号を、その後4000 episodes以上にわたり回復できなかった。

## randomとの比較と限界

現在のP0X2RCS05はRCによりtransitionとtermination semanticsがori5から変わっている。そのため、ori5のuniform random結果を現在候補のmatched random baselineとして流用できない。candidate-specific random harnessは存在するが、今回の6144診断では実行していない。

確実に言える範囲は次の通りである。

- 歴史的ori5 randomではtorus D=2もwaterbomb D=1も0/10 successes at 102,400 mechanicsだった。
- 現在のRLも両タスク0/6144 strict successesだった。
- しかしbudget、RC遷移、episode horizon、wall calibrationが一致しないため、現在候補についてRL対randomの速度比、RMST比、wall time優位を算出してはいけない。
- five-bar D=2とwaterbomb D=sqrt(2)のwall比較は、ori5内のcalibrated・記述的比較としてのみ有効である。

次の候補を昇格させる際は、同じRC・同じspectral設定・同じmechanical budgetを使うcandidate-specific randomを実行する必要がある。ただし、まずwaterbombの符号問題を除き、torusではconstructive signalの保持とupdate安定性を同時に見る方が優先度は高い。

## 5タスク共通報酬体系としての整理

共通体系は、task固有の成功形状を直接書き込むのではなく、次の同じ構造を全タスクに使うのがよい。

1. `flatness potential difference`: Fが下がるaccepted transitionを連続的に報いる。
2. `spread potential difference`: Eが下がるaccepted transitionを連続的に報いる。running-bestではなく現在状態potentialとし、戻れば同量を返す。
3. `joint milestones / terminal quality`: FとEが同一状態で良い場合だけ大きくする。片方だけ良いwide/wrong-plane basinを抑える。
4. `spectral plane-switch potential`: plane swapを要するtaskだけ `lambda_spectral > 0`、不要なfive-barなどでは0。式と実装は共通で係数だけtask設定にする。
5. `rejected-crossing continuation`: 幾何的に危険なactionをrollbackしてepisodeを継続するが、同じ山への試行ごとに大罰を重ねない。
6. `terminal`: strict successには大きな一回のbonus、timeoutにはlast-safe-stateのnear-miss品質を含む一回のpenalty/qualityだけを与える。

この体系ならfive-barでplane-swap項を0にでき、waterbomb/torusでは同じコード経路で有効化できる。共通性は「全タスクで全重みを同値にすること」ではなく、「全タスクで同じ意味のpotentialとterminal orderingを使い、不要な物理機構の係数だけ0にすること」と定義するのが現実的である。

## 次の方針、簡単な順

### 1. 最小修正: rejectの反復penaltyをなくす

非終端rejectはreward 0とし、rollback、step消費、progress保持は現状のままにする。horizonでは通常timeoutとして `-1 + q(last safe state)` を一度だけ与える。最初から小罰を入れず、必要なら後で `-0.01` またはepisode内cap付きpenaltyを比較する。

この変更はwaterbomb episode 5を `-21.297 -> +2.752`、torus episode 1411を `+0.261 -> +1.982` にする反実仮想と直結する。waterbombには強い直接根拠があり、torusでは既に正しいcohort順位のmarginを広げる検査になる。torus成功を単独で保証する根拠ではない。

### 2. 小さな多seed gate

torusとwaterbombを各3 seeds、まず1024、条件を満たすものだけ2048まで実行する。判定はstrict successだけでなく、後半512でconstructive/valid-joint率とreturnが維持または上昇することにする。前半の一時的発見だけでは昇格させない。

### 3. torusの保持・credit assignmentを同時診断

各update batchについてconstructive episode数、return/advantage分布、critic lossとgradient、entropy、更新前後の固定seed評価を保存する。constructive cohortがactual returnで有利なのに次updateで消えるなら、報酬より保持・credit assignment側の証拠になる。これは学習則を変えない診断なので、zero-reject-penalty screenと同時に行える。

### 4. それでもplane swap不足ならspectral重みを調整

`lambda_spectral = 0.05, 0.1, 0.2` を短いpaired ablationで比較する。plane swap不要taskは0のままにする。waterbombのreject符号を直す前にspectralを強めると問題を隠すため、この順序を逆にしない。

### 5. torusがF約0.05で止まる場合だけlate flatness shaping

log/scale-aware flatness potential、または `F=0.03, 0.01, 0.003, 0.001` のlate milestonesを追加する。torus episode 1411のようにEは十分だがFが0.05付近の状態にだけ効かせる。

### 6. gap-close → plane-swap → gap-reopenのphase情報

spectral potentialだけでphaseの向きを区別できない場合、singular gapが閉じ、planeが切り替わり、再びgapが開く順序をobservation/rewardへ入れる。これは状態と実装を複雑化するため、上の小変更が失敗してから行う。

### 7. 保持失敗が再現したらtrajectory保持を導入

CMAES/既知成功pathからbehavior cloningし、prioritized/off-policy success bufferで成功・near-success trajectoryを保持し、その後on-policy fine-tuneする。実装は複雑だが、torusでは既にcohort順位が正しいのに保持できなかったため、上の診断で同じ崩壊が再現すれば優先度を上げる。

## 「もうこれをやればうまくいく段階か」への回答

保証できる段階ではない。成功率0の二タスクに対し、単一seedの反実仮想から「必ず成功する」とは結論できない。また、torusはFを0.05から0.001へさらに50倍下げる必要があり、reject修正後にもlate flatness問題が残る可能性がある。

一方、以前より次の点はかなり明確になった。

- plane入れ替えの山を検出するspectral方針は立っており、offlineでsuccess/failureを分離し、onlineでもwaterbomb strict successを一度生成した。
- RCはcrossing終了を防ぎ、reject後に92–100%程度escapeさせる。
- 良いjoint stateとnear-miss signalは実際に得られている。
- waterbombの高品質episodeには反復rejectによる明確な符号反転があり、torusには正しいcohort順位にもかかわらず保持できない別の問題がある。

したがって正確には、**「一変更での成功保証」ではないが、「最小の報酬ablationと、torusの保持・credit assignmentを分離する明確な診断基準が揃った段階」**である。

## 成果物と再現性

- [最終整合性・全精密指標JSON](final_metrics_and_integrity.json)
- [512-episode window集計CSV](checkpoint_windows_512.csv)
- [再生成可能なstandalone analyzer](finalize_two_task_6144.py)
- [torus learning curve](learning_curve_torus2_episode_and_mechanics.png)
- [waterbomb learning curve](learning_curve_water1_episode_and_mechanics.png)
- [combined learning curve](learning_curve_combined_episode_and_mechanics.png)

analyzerは学習を実行せず、固定済みartifactを読んで検証・集計・描画するだけである。JSONの`overall_integrity_pass`はtrue、report fingerprintは `417b3966710806ef214c8a67101a93dde319bf18d736e4aa056c52cc3791442f` である。PNGはPILでdecode・寸法検査し、3枚とも3378 x 4418 pixelsである。さらに目視で軸、凡例、注記、clip、waterbomb episode 5、torus constructive windowを確認した。
