# C10 学習と exact uniform random の初回発見比較

主評価は strict compact success の初回発見までに提案した raw action 数です。自己交差で reject された提案も1 proposalとして数えます。報酬値は比較に使いません。

| task | 学習arm | cap | 学習 hit/n | random hit/n | 学習 RMST | random RMST | random/学習 | 95% bootstrap CI | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| five_d2 | B | 819,200 | 3/3 | 2/10 | 153804.0 | 735140.6 | 4.780 | [3.496, 6.639] | primary available |
| five_d3 | W | 409,600 | 3/3 | 4/10 | 27025.7 | 313490.5 | 11.600 | [4.285, 106.117] | primary available |
| watersqrt2 | B | 409,600 | 3/3 | 10/10 | 26596.0 | 13814.8 | 0.519 | [0.178, 89.406] | primary available |

`random/学習 RMST > 1` は、共通 cap において学習の初回発見が速い方向です。成功例だけの中央値ではなく、未発見 seed を cap に割り当てた RMST を主指標にしています。

## 学習前後の区別

| task | 初回更新以前のhit | 更新後のhit | final selection retained |
|---|---:|---:|---|
| five_d2 | 0 | 3 | yes |
| five_d3 | 0 | 3 | yes |
| watersqrt2 | 1 | 2 | yes |

## 不確実性と解釈上の制約

- CI は seed を単位とする percentile bootstrap (20,000 resamples, 95%) です。学習 n=3、random n=10 の小標本なので、区間は離散的かつ不安定であり、強い一般化には使えません。
- 学習3 seedすべてが共通 cap まで観測可能、random 10 seedすべてが rc=0・complete、かつ選択armが最終的に保持された場合だけ primary comparison available とします。
- 初回更新以前の hit は初期化 policy の寄与を含むため、学習で形成された探索圧の証拠とは分けて表示します。
- action law は一様 random、mechanics/H/幅/strict判定/crossing transition は C10 と一致します。optimizer と報酬は random の行動を変えないため、報酬値そのものは比較対象外です。
- C10R1 は supervisor race で中止された診断 run であり、本表には一切混入していません。
