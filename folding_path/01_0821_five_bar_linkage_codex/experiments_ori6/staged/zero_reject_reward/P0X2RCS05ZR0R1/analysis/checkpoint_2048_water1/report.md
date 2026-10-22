# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T16:06:30.992543+00:00
- Atomic endpoint: 2048 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.5554 | 0.0008116 | 0.118 | 0 | 0 | 0 | 1.08/1.63/2.33 | 1.11/2.3/4.03; 45.3% | 0.0984/0.173/0.232 | 2.24 | 0.00015 | False |
| 4202 | 0.05705 | 0.1853 | 0.1157 | 1 | 0 | 0 | 1.05/1.53/1.87 | 0.878/1.76/2.35; 37.5% | 0.0906/0.154/0.173 | 2.09 | 0.00015 | False |
| 4203 | 0.383 | 0.3252 | 0.1141 | 37 | 0 | 0 | 0.493/0.937/1.14 | 0.475/1.01/1.31; 7.8% | 0.107/0.176/0.21 | 2.14 | 0.00015 | True |

Reasons: 

