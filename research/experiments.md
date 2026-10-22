# 代表実験の索引

2026-10-22に保存報告から整理。今回実験は実行していない。手法・課題・予算の異なる行を横断して単純な順位を付けない。

| ID | 調べた問い | 条件と保存結果 | 言える範囲 | 出典 |
|---|---|---|---|---|
| E01 | 摂動の候補集合／位置選択は効くか | 800 step、各段階5 seed。frontier上でrandom/UCB/SM-UCBの新規cell平均1.6 | SM-UCBの小さなAUC改善。最終coverageの優位なし | S04 |
| E02 | GP終端予測で多様解が増えるか | held-out 5 seed、1200 step。final cell random/GP-UCB=2.0/2.0 | 同条件のrandomより多く発見していない。GP費用増 | S06 |
| E03 | 方向持続macroそのものは効くか | 3 seed、3000 step。canonical six到達K=1:0/3、K=1..24:3/3 | 行動設計の効果。学習選択の優位ではない | S09 |
| E04 | 安全な終端分類filterは効くか | main9 v1.1、10 seed、3000 step。棄却0/336384、randomと同軌跡 | kinematic six 10/10、feasible six 8/10。filterは作動せず | S10–S11 |
| E05 | 共通報酬で5条件を学べるか | ori5。five D2初回585ep/233824step/73更新、D3は更新0。torus・water D1未発見 | 初回学習効果、初期NN効果、学習後性能を分ける | S13 |
| E06 | 衝突zero-rejectで改善するか | torus selected seed last1024成功89.4%、water fallback51.2% | 4096以降は選抜seedの診断。単独変更の一般因果効果ではない | S15 |
| E07 | C10はexact uniformより初回が速いか | RL3/random10、共通cap。random/RL RMST比 D2 4.780、D3 11.600、water√2 0.519 | fiveの正例とwaterの限界。小標本・選抜あり | S18–S19 |
| E08 | torusに大きなspectral重みが必要か | α=.05/.25/.5、各5 seed、成功4/5・4/5・2/5 | .05が打切り込み平均step最小。このscreenからの選択 | S20 |
| E09 | 初回成功前に探索分布が変わるか | five D3 seed9102、初期/1更新後各100 rollout。near-miss 0→22、Q* .0462→.2645 | READMEの実装確認。全taskの一般結果ではない | S21 |
| E10 | D2/D3混合失敗はcriticだけが原因か | seed9201。centering、critic追加でもD2 strict/near-miss 0/100、D3 strict100/100 | critic精度改善だけでは解決せず。混合全般の不可能性ではない | S23 |
| E11 | 少量D4教師に事前学習は効くか | 5 seed、同32 test starts、D4教師2経路でscratch3.12%、pretrained22.50% | target教師あり。教師部分集合によって逆転もある | S25 |
| E12 | 既知経験で未知D4を初回発見できるか | projected_bars、3 NN、673/565/281提案、更新0。同構造scratch0/3@6400 | frozen確率的探索の正例。deterministicは0/3、nine D4も0/3 | S26–S27 |

数値引用の際は、[資料一覧](sources.md) の元報告と設定を確認する。raw/canonical、kinematic/feasible、proposal/wall time、selected/held-out、source seed/探索seed/評価rolloutを混ぜない。
