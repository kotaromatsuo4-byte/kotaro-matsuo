# 多様解探索：何を制御しようとし、どこまで効いたか

対象系列：`00_0804_codex2`、`01_0804_five_bar_linkage`、`01_0813_five_bar_linkage`。概要は [研究史03](../history/03_diverse_terminal_search.md)。以下の数値は保存報告による。今回の再実験結果ではない。

## 1. 候補集合と、候補からの選び方

問題は「良いスコアを作る」だけではない。有望な状態が候補集合に入らなければ、選択器は選べない。また新規cell報酬は、archiveが埋まるにつれて変わる。このため集合と選択器を分離した。[C03]

`00_0804_codex2` の二段階比較は、同数のfrontier/event/historyをrandomで比較し、その勝者を別seedでrandom/UCB/SM-UCBにかける。共通bootstrap、総mechanics-step予算、同数候補を使う。[S04]

800 step・5 seedでは、半径0.22のrootが新規cell平均2.4、frontier/historyは1.6、eventは1.2。集合比較内のtie-breakでfrontierを選んだ。選択器比較は最終新規cell平均がいずれも1.6。SM-UCBのAUCは少し高かったが、最終発見数の優位ではない。

この結果からrootが常に最良とは言えない。摂動半径、同じstep予算でも異なるrollout数、内部CMA-ESの評価費用が解釈に関わる。後の小摂動・持続macroと直接同じ表にして順位を付けない。

## 2. 終端を予測し、現在のarchiveに対する価値を外で計算する

NN/GPでは、状態と物理変位から終端を予測し、現在のarchiveと照合して新規性を計算する。終端写像そのものと、新規性の非定常な価値を分ける意図があった。[S05–S06]

main5のheld-out 5 seed、1200 stepではrandomとGP-UCB-likeは最終2.0 cell、GP-posteriorは1.8。GP-UCB-likeのpaired AUC差は+0.0228だが参考区間は0を含む。wall timeは13.17秒に対し19.86秒、posteriorは101.84秒。平坦への到達や衝突回避の予測ができても、新しい終端の発見効率の改善には直結しなかった。[S06]

失敗原因の候補は、終端写像の不連続性、同一終端へ行くedgeの大量重複、少ない独立rollout、候補生成の制限、モデル費用など。これらすべてが因果実験で分離されたわけではない。

## 3. 予測器の前に、別の終端へ行ける行動を作る

小さい一回の摂動では、決定論rolloutが同じbasinへ戻る。大きな一回の変位の代わりに、前の物理変位を次の接空間へ投影して方向を持続する案へ進んだ。[C05–C06]

K=2のpilot、角度学習（main6）、親と二手をまとめた選択（main7）、可変K（main8）は別の比較。main7の3 seed・10,000 stepではjoint GPの平均は良くても、shuffledに対する差はseedに依存し、全6 cell到達はなかった。[S08]

main8の3 seed・3000 stepでは、K=1はcanonical 6到達0/3、K=1..24のrandomは3/3（T6=1597/1517/1686）。一方、global GP・階層K-GPのonline held-outはrandomを上回らなかった。**可変長方向持続の効果と、それを学習で選ぶ効果を分ける**のが、この段階の主要な結論。[S09]

## 4. GP回帰から、既知終端への重複を除く分類filterへ

似た入力から異なる離散終端へ行く問題を、滑らかな座標回帰で扱う限界を疑った。衝突でも積分を続け、kinematicとcollision-free feasibleを別archiveに保存。Random Forest/ExtraTrees等で既知cellへの再到達を予測した。[C07–C08, S10]

main9 v1.1（局所suffix timeoutなし）、10 seed・3000 stepでは、randomでcanonical kinematic 6到達10/10、平均T6=1130.4。feasible 6到達は8/10、未達を3000で打ち切った平均T6=1944.3。
安全filterは336,384候補を評価して棄却0。trajectoryはrandomと一致し、モデル計算だけが増えた。「学習法もrandomと同等に良い」と評価するのではなく、選択規則が作動しなかった結果である。閾値を緩めたpilotはfeasible側を悪化させた。[S10–S11]

## 保持する知見

- raw cell数はlayer variantを含む。canonical sixの到達数と混ぜない。
- collision継続による運動学的到達と、衝突のない経路の発見を分ける。
- 方向持続だけ、衝突継続だけの寄与は、複数条件を同時に変えた比較から断定しない。
- 独立教師はedge行数ではなくmacro/rollout単位で数える。
- 「offlineで学べる」と「探索中に間に合う」は異なる。
- five-barでの採用案はvariable-K random。一般構造への優位や列挙保証は未確認。

詳細な出典対応と版混在の注意は [S12](../sources.md) にもまとめられている。
