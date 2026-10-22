# 固定horizon正規化 + LR/batch適応化 実験summary

`NN更新回数`は初回strict successより前に完了したoptimizer update数、`累積step数`は成功episodeを含む初回strict successまでのmechanical step総数である。

## five D2

| Seed | 初回strict success episode | NN更新回数 | 累積step数 |
|---:|---:|---:|---:|
| 9101 | 624 | 77 | 249,158 |
| 9102 | 482 | 60 | 192,738 |
| 9103 | 315 | 39 | 125,684 |
| 9104 | 692 | 86 | 276,524 |
| 9105 | 803 | 100 | 321,134 |
| 中央値 | 624 | 77 | 249,158 |

長期学習はseed 9103で10,000 episode、総step数2,110,118、NN更新645回。最終1,000 episodeのstrict success率は98.2%で、保持を確認した。

## five D3

| Seed | 初回strict success episode | NN更新回数 | 累積step数 |
|---:|---:|---:|---:|
| 9101 | 189 | 23 | 75,579 |
| 9102 | 9 | 1 | 3,501 |
| 9103 | 9 | 1 | 3,592 |
| 9104 | 10 | 1 | 3,954 |
| 9105 | 14 | 1 | 5,539 |
| 中央値 | 10 | 1 | 3,954 |

長期学習はseed 9103で5,000 episode、総step数362,833、NN更新313回。最終1,000 episodeのstrict success率は100%で、保持を確認した。
