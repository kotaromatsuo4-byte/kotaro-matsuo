# Codex6 非 plane-swap 課題：optimizer 調整結果

- 作成時刻（UTC）: 2026-08-28T11:29:15.495046+00:00
- manifest fingerprint: `7f1a9303048ba683d01f8d8539ed30444e8800ff5952ec498890e694410165ab`

## 結論

- `five_d2`: `B` を選択し、3 seed とも 10000 episode を完走。
- `five_d3`: `W` を選択し、3 seed とも 10000 episode を完走。
- `watersqrt2`: `B` を選択し、3 seed とも 10000 episode を完走。

arm 選択には return を使っていません。初回 strict 成功までの累積 mechanical proposals、同一 seed の比較、strict の持続性、固定乱数 stochastic 評価で判定しています。proposal clock は `training_history.steps` の累積であり、reject された crossing candidate も1 proposalとして数えます。

## 選択された arm の結果

| task | arm | seed | formal ep | first hit ep | proposals/censor | first update前/同時 | strict(last1024) | fixed-16@formal | final-64 | overrun除外 |
|---|---|---:|---:|---:|---|:---:|---:|---:|---:|---:|
| five_d3 | W | 9101 | 10000 | 185 | 73976 | no | 1 | 1 | 1 | 0 |
| five_d3 | W | 9102 | 10000 | 9 | 3501 | no | 1 | 1 | 1 | 0 |
| five_d3 | W | 9103 | 10000 | 9 | 3600 | no | 1 | 1 | 1 | 0 |
| watersqrt2 | B | 9101 | 10000 | 163 | 65194 | no | 0.82715 | 0.8125 | 0.84375 | 0 |
| watersqrt2 | B | 9102 | 10000 | 37 | 14457 | no | 0.94629 | 0.9375 | 0.95312 | 0 |
| watersqrt2 | B | 9103 | 10000 | 1 | 137 | yes | 0.93652 | 1 | 0.89062 | 0 |
| five_d2 | B | 9101 | 10000 | 278 | 110738 | no | 0.99609 | 1 | 1 | 0 |
| five_d2 | B | 9102 | 10000 | 484 | 193492 | no | 0.99512 | 1 | 0.98438 | 0 |
| five_d2 | B | 9103 | 10000 | 394 | 157182 | no | 1 | 1 | 1 | 0 |

## optimizer arm の固定条件

| task | arm | batch | actor LR | critic LR | entropy | actor milestones | factors |
|---|---|---:|---:|---:|---:|---|---|
| five_d2 | B | 8 | 0.0005 | 0.00015 | 0.001 | [2000,4000,7000] | [0.3,0.1,0.03] |
| five_d2 | W | 8 | 0.001 | 0.00015 | 0.001 | [512,2000,4000,7000] | [0.5,0.15,0.05,0.015] |
| five_d3 | B | 8 | 0.0005 | 0.00015 | 0.001 | [2000,4000,7000] | [0.3,0.1,0.03] |
| five_d3 | W | 8 | 0.001 | 0.00015 | 0.001 | [512,2000,4000,7000] | [0.5,0.15,0.05,0.015] |
| watersqrt2 | B | 8 | 0.0005 | 0.00015 | 0.001 | [2000,4000,7000] | [0.3,0.1,0.03] |
| watersqrt2 | E | 4 | 0.00025 | 7.5e-05 | 0.001 | [2000,4000,7000] | [0.3,0.1,0.03] |

`first update前/同時=yes` の初回成功は初期方策による発見を含むため、学習が初回発見を早めた証拠とは扱いません。

## B 対 candidate の screen

| task | arm | gate ep | selected | hit seed | pre/post update hit | stable seed | proposal RMST | tail strict | fixed-16 | reason |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| five_d2 | B | 1024 | yes | 3 | 0/3 | 3 | 153804 | 0.40755 | 0.79167 | control_selected_at_1024 |
| five_d2 | W | 1024 | no | 2 | 0/2 | 2 | 202756 | 0.40625 | 0.4375 | control_selected_at_1024 |
| five_d3 | B | 512 | no | 2 | 0/2 | 3 | 23250.3333 | 0.75977 | 1 | strong_candidate_at_512 |
| five_d3 | W | 512 | yes | 2 | 0/2 | 3 | 21645.3333 | 0.83594 | 1 | strong_candidate_at_512 |
| watersqrt2 | B | 512 | yes | 3 | 1/2 | 3 | 26596 | 0.22721 | 0.70833 | both_strong_control_tiebreak_at_512 |
| watersqrt2 | E | 512 | no | 3 | 1/2 | 3 | 7103.66667 | 0.20898 | 0.58333 | both_strong_control_tiebreak_at_512 |

## 同一 seed の proposal-clock 比較

| task | candidate | seed | candidate hit@cap | B hit@cap | pre-update hit candidate/B | candidate proposals | B proposals | proposal Δ | outcome | tail Δ | fixed Δ |
|---|---|---:|:---:|:---:|:---:|---:|---:|---:|---|---:|---:|
| five_d2 | W | 9101 | no | yes | no/no | 334833 | 110738 | 224095 | loss | -0.46094 | -0.9375 |
| five_d2 | W | 9102 | yes | yes | no/no | 169295 | 193492 | -24197 | win | 0.41797 | 0.0625 |
| five_d2 | W | 9103 | yes | yes | no/no | 104140 | 157182 | -53042 | win | 0.039062 | -0.1875 |
| five_d3 | W | 9101 | no | no | no/no | 57835 | 57835 | 0 | tie | 0.18945 | 0 |
| five_d3 | W | 9102 | yes | yes | no/no | 3501 | 6335 | -2834 | win | 0.021484 | 0 |
| five_d3 | W | 9103 | yes | yes | no/no | 3600 | 5581 | -1981 | win | 0.017578 | 0 |
| watersqrt2 | E | 9101 | yes | yes | no/no | 19452 | 65194 | -45742 | win | 0.068359 | 0.0625 |
| watersqrt2 | E | 9102 | yes | yes | no/no | 1722 | 14457 | -12735 | win | -0.0625 | -0.5 |
| watersqrt2 | E | 9103 | yes | yes | yes/yes | 137 | 137 | 0 | tie | -0.060547 | 0.0625 |

## Codex5 の歴史的 anchor（記述的比較のみ）

| task | seed | first hit ep | proposals | 初期方策hit | strict(last1024) | final stochastic |
|---|---:|---:|---:|:---:|---:|---:|
| five_d2 | 2101 | 585 | 233824 | no | 0.90039 | 0.88 (100 trials) |
| five_d3 | 2101 | 3 | 1033 | yes | 0.98828 | 1 (100 trials) |
| watersqrt2 | 2101 | 47 | 17324 | no | 0.95312 | 0.93 (100 trials) |

注意: Codex5 anchor は seed2101 の1本だけで、seed が一致せず、報酬も異なり、crossing 時に episode を終了します。今回の Codex6 は crossing candidate を reject して同じ episode を継続します。したがって差を optimizer の因果効果とは解釈できません。

## 曲線と指標

選択 arm について episode 軸と累積 proposal 軸の両方を描きました。strict は MA256（実線）と MA512（破線）、固定乱数16試行は白抜き丸です。F は terminal flatness、e は spread error、C は `F<=0.5 and e<=0.1`、J は `F<=0.1 and e<=0.1`（いずれも invalid episode を除外）です。菱形は最初の update 前または同時の初回hit、星形は update 後の初回hit、X は formal endpoint での右打切りです。return は診断表示のみです。

## 完全性

manifest、execution binding、runner report、全18 jobのsource/model/config/history、gate artifactを検証し、入力 187 ファイルを SHA-256 で固定しました。
各 gate の判定は bound analyzer で再計算し、runner report および gate artifact と一致することを確認しています。formal gate 後に書かれた atomic/history overrun は件数を記録し、表・曲線・選択指標から除外しています。
