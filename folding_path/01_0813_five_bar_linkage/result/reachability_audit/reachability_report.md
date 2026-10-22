# Five-bar small-perturbation reachability audit

現在の `descriptor.py` と `collision2.py` を用いて、過去に保存された経路を
再分類・再検査した結果である。探索そのものは再実行していない。

## Summary

- 保存witness数: 8
- run横断の固有cell数: 6
- 全保存edgeが現collision2で有効: True
- corrector後の最大保存step norm: 0.199384
- `.dat` 丸め後の最大拘束残差: 1.397e-08

## Source runs

| source | selector | budget | perturbation scale | deterministic max norm |
|---|---|---:|---:|---:|
| `main4_warm4_hybrid/seed_001` | hybrid | 1200 | 0.2 | 0.1 |
| `nn_pool_frontier_random_v2/seed_003` | random | 2000 | 0.2 | 0.1 |

## Six target cells

| target | terminal x signature | layer key | witnesses |
|---:|---|---|---:|
| 0 | `[-1, 0, -1, 0, 1]` | `(0,1,0) (0,2,0) (1,2,1)` | 2 |
| 1 | `[-1, 0, 1, 0, 1]` | `(1,2,2) (1,3,3) (2,3,3)` | 2 |
| 2 | `[-1, 0, -1, 0, 1]` | `(0,1,1) (0,2,2) (1,2,2)` | 1 |
| 3 | `[-1, 0, 1, 2, 1]` | `(2,3,2)` | 1 |
| 4 | `[-1, 0, 1, 0, 1]` | `(1,2,1) (1,3,1) (2,3,2)` | 1 |
| 5 | `[-1, -2, -1, 0, 1]` | `(0,1,1)` | 1 |

## Witness paths

`first K` はそのrunでcellが最初に発見された終端までの摂動edge数、
`representative K` は保存された代表経路上の摂動edge数である。

| target | source | cell | first K | representative K | max step norm | collisions | max residual |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `main4_warm4_hybrid/seed_001` | `cell[0]` | 1 | 3 | 0.198679 | 0 | 1.015e-08 |
| 0 | `nn_pool_frontier_random_v2/seed_003` | `cell[0]` | 2 | 3 | 0.197925 | 0 | 1.289e-08 |
| 1 | `main4_warm4_hybrid/seed_001` | `cell[1]` | 2 | 2 | 0.198768 | 0 | 1.155e-08 |
| 1 | `nn_pool_frontier_random_v2/seed_003` | `cell[1]` | 3 | 4 | 0.198876 | 0 | 1.167e-08 |
| 2 | `main4_warm4_hybrid/seed_001` | `cell[2]` | 2 | 2 | 0.195071 | 0 | 1.116e-08 |
| 3 | `main4_warm4_hybrid/seed_001` | `cell[3]` | 3 | 4 | 0.197423 | 0 | 1.352e-08 |
| 4 | `nn_pool_frontier_random_v2/seed_003` | `cell[2]` | 2 | 3 | 0.199384 | 0 | 1.063e-08 |
| 5 | `nn_pool_frontier_random_v2/seed_003` | `cell[3]` | 3 | 3 | 0.197285 | 0 | 1.397e-08 |

## Interpretation

この監査で6 cellが得られる場合、現在の小摂動上限と平面内衝突判定の下で
各cellへ至る数値的witnessがrun横断で存在する。単一runで6 cellを回収できる
ことや、各cellに一回の摂動では到達不可能であることまでは証明しない。
初回発見の摂動回数は、複数摂動を優先する探索設計の経験的根拠として用いる。
