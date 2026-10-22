# Persistent-direction emitter diagnostic

rootで選んだ物理方向を各更新状態の局所接空間へ射影し直し、targetを
使わずに連続小摂動を加えた結果である。chain length 1は同じ方向gridを
用いた比較基準である。

| scale | chain length | directions | mechanics steps | cells | reached 6 | flat | collision | failure/timeout |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.20 | 1 | 4 | 161 | 1 | False | 1 | 3 | 0 |
| 0.20 | 2 | 4 | 188 | 3 | False | 3 | 1 | 0 |

`mechanics steps` は摂動stepと決定論stepを同じ1 stepとして数える。
6 cell未到達は可到達性の否定ではなく、この固定方向emitterと有限角度grid
では回収できなかったことだけを意味する。
