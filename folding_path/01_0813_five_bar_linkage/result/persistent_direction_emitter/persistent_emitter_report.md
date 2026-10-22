# Persistent-direction emitter diagnostic

rootで選んだ物理方向を各更新状態の局所接空間へ射影し直し、targetを
使わずに連続小摂動を加えた結果である。chain length 1は同じ方向gridを
用いた比較基準である。

| scale | chain length | directions | mechanics steps | cells | reached 6 | flat | collision | max actual perturbation norm |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.15 | 1 | 72 | 3210 | 2 | False | 24 | 48 | 0.149508 |
| 0.15 | 2 | 57 | 2508 | 6 | True | 30 | 27 | 0.149546 |
| 0.20 | 1 | 72 | 2891 | 2 | False | 19 | 53 | 0.198846 |
| 0.20 | 2 | 59 | 2554 | 6 | True | 33 | 26 | 0.198977 |

`mechanics steps` は摂動stepと決定論stepを同じ1 stepとして数える。
6 cell未到達は可到達性の否定ではなく、この固定方向emitterと有限角度grid
では回収できなかったことだけを意味する。
