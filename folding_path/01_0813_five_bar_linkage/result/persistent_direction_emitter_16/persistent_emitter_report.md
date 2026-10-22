# Persistent-direction emitter diagnostic

rootで選んだ物理方向を各更新状態の局所接空間へ射影し直し、targetを
使わずに連続小摂動を加えた結果である。chain length 1は同じ方向gridを
用いた比較基準である。

| scale | chain length | directions | mechanics steps | cells | reached 6 | flat | collision | max actual perturbation norm |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.15 | 1 | 16 | 761 | 2 | False | 7 | 9 | 0.149498 |
| 0.15 | 2 | 13 | 610 | 6 | True | 8 | 5 | 0.149538 |
| 0.20 | 1 | 16 | 638 | 2 | False | 4 | 12 | 0.198829 |
| 0.20 | 2 | 16 | 653 | 5 | False | 8 | 8 | 0.198977 |

`mechanics steps` は摂動stepと決定論stepを同じ1 stepとして数える。
6 cell未到達は可到達性の否定ではなく、この固定方向emitterと有限角度grid
では回収できなかったことだけを意味する。
