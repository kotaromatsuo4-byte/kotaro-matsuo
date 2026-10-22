# Codex8 spectral reward検証（10:00確定版）

## 結論

- **torus2**: episode 2048のearly evidenceで **scale** を選択し、formal 4096 confirmation gateにも到達・通過。初回strict発見のraw action proposalsは seed8101=364199, seed8102=196191, seed8103=185999。
- **water1**: episode 2048のearly evidenceで **scale** を選択し、formal 4096 confirmation gateにも到達・通過。初回strict発見のraw action proposalsは seed8101=146207, seed8102=8789, seed8103=460232。

arm選択はreturnを使わず、初回strict発見の有無→発見までのproposal数→直近1024のstrict/joint/constructive→e→F→reject率の順です。

## Seed別の結果

| task | arm | seed | atomic ep | first hit proposals | strict / joint / constructive (last1024) | median e | median F |
|---|---:|---:|---:|---:|---:|---:|---:|
| torus2 | legacy | 8101 | 2048 | censored@409600 (not_selected_by_same_seed_rule_at_2048) | 0 / 0 / 74 | 0.06891 | 0.9858 |
| torus2 | legacy | 8102 | 2048 | 229197 | 632 / 944 / 1003 | 0.0001709 | 0.0008668 |
| torus2 | legacy | 8103 | 2048 | 184390 | 271 / 900 / 1000 | 0.001608 | 0.01129 |
| torus2 | scale | 8101 | 8192 | 364199 | 850 / 994 / 994 | 0.0001008 | 0.0006725 |
| torus2 | scale | 8102 | 8192 | 196191 | 1003 / 1012 / 1014 | 7.478e-05 | 0.000629 |
| torus2 | scale | 8103 | 8192 | 185999 | 943 / 993 / 999 | 7.71e-05 | 0.0006407 |
| torus2 | phase_v2 | 8101 | 2048 | 403592 | 3 / 144 / 619 | 0.02892 | 0.3053 |
| torus2 | phase_v2 | 8102 | 2048 | censored@409600 (not_selected_by_same_seed_rule_at_2048) | 0 / 10 / 216 | 0.397 | 1.005 |
| torus2 | phase_v2 | 8103 | 2048 | 197395 | 359 / 953 / 997 | 0.0005661 | 0.005015 |
| water1 | legacy | 8101 | 2048 | censored@663073 (not_selected_by_same_seed_rule_at_2048) | 0 / 0 / 1 | 0.1158 | 0.9621 |
| water1 | legacy | 8102 | 2048 | censored@819200 (not_selected_by_same_seed_rule_at_2048) | 0 / 0 / 7 | 0.07301 | 1.237 |
| water1 | legacy | 8103 | 2048 | 104222 | 113 / 232 / 536 | 0.007177 | 0.4682 |
| water1 | scale | 8101 | 10000 | 146207 | 674 / 692 / 710 | 1.904e-05 | 0.0007395 |
| water1 | scale | 8102 | 7168 | 8789 | 0 / 0 / 86 | 0.1141 | 0.175 |
| water1 | scale | 8103 | 7168 | 460232 | 0 / 0 / 92 | 0.1131 | 0.2946 |
| water1 | phase_v2 | 8101 | 2048 | censored@818948 (not_selected_by_same_seed_rule_at_2048) | 0 / 0 / 0 | 0.08194 | 1.164 |
| water1 | phase_v2 | 8102 | 2048 | censored@819200 (not_selected_by_same_seed_rule_at_2048) | 0 / 0 / 0 | 0.06242 | 1.328 |
| water1 | phase_v2 | 8103 | 2048 | 27178 | 119 / 216 / 481 | 0.008194 | 0.5657 |

## 同一seed pairwise差（選択endpointで打切り）

| task | endpoint | candidate | opponent | W-L-T |
|---|---:|---|---|---:|
| torus2 | 2048 | legacy | phase_v2 | 2-1-0 |
| torus2 | 2048 | legacy | scale | 1-2-0 |
| torus2 | 2048 | phase_v2 | legacy | 1-2-0 |
| torus2 | 2048 | phase_v2 | scale | 0-3-0 |
| torus2 | 2048 | scale | legacy | 2-1-0 |
| torus2 | 2048 | scale | phase_v2 | 3-0-0 |
| water1 | 2048 | legacy | phase_v2 | 2-1-0 |
| water1 | 2048 | legacy | scale | 1-2-0 |
| water1 | 2048 | phase_v2 | legacy | 1-2-0 |
| water1 | 2048 | phase_v2 | scale | 1-2-0 |
| water1 | 2048 | scale | legacy | 2-1-0 |
| water1 | 2048 | scale | phase_v2 | 2-1-0 |
各seedのhit/proposal/strict/joint/constructive/e/F/reject差は `paired_seed_comparisons.csv` に記録しています。

## ランダム探索との初回発見比較

10,000 episodeを完了してstrict hitを得た昇格armがないため、事前条件に従いrandom比較は確定していません。

random baselineとCodex8は、幾何・uniform action law・max-step・自己交差候補をrejectしてepisode継続する遷移について比較可能です。報酬値のidentityは一致条件に含めず、learningとrandomのseed cohortも異なるため、比は記述的な目安です。

## 報酬・観測の注意

- legacy/scaleはCodex6 spectral potentialとrollbackをそのまま使い、重みだけが異なります。phase_v2だけがclose→turnの順序付きpotentialです。
- actor/critic入力はCodex6の節点座標（および既存のmode-conditioned action処理）のままで、rho、delta、z、時刻は追加していません。したがってphase_v2はhidden-ledgerを持つ部分観測のablationです。
- phase Psiが高いのにF/eが悪いtimeoutを別集計し、reward farmingをarm選択から除外しています。
- 3 seedsなので不確実性は大きく、2/3規則は再現性の証明ではありません。
- 旧C8SP2はanalyzer偽陽性で中止した監査来歴に限り、fresh replayの曲線・比較には1行も混ぜていません。

## Integrity

- manifest: `f6c0dcfdbee9d2b3f6f1df8a6cacead775a5a8f01196335def32bfef754894d3`
- finalizer/test SHAはソース自身に埋め込まず、外部CLI承認値と実ファイルを開始時・公開直前に照合。
- 実行中PIDなしをfinalization開始時とatomic publish直前に再確認。
- run_config、source/formula hash、episode連続性、reward decomposition、nonfinite、atomic endpointをfail-closedで検証。
- comparison endpoint以後に書かれたoverrun rowは選択判定から除外。
