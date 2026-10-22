# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T16:49:51.006161+00:00
- Atomic endpoint: 3072 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.08597 | 0.1539 | 0.1169 | 0 | 0 | 0 | 0.994/1.41/1.56 | 0.696/1.54/2.6; 18.8% | 0.0866/0.145/0.15 | 2.26 | 0.00015 | False |
| 4202 | 0.2228 | 0.1842 | 0.116 | 1 | 0 | 0 | 0.803/1.29/1.42 | 0.694/1.37/2.53; 18.8% | 0.0933/0.133/0.164 | 2.06 | 0.00015 | False |
| 4203 | 0.601 | 0.2655 | 0.1146 | 72 | 0 | 0 | 0.344/0.698/0.904 | 0.376/0.925/1.21; 1.6% | 0.0983/0.186/0.253 | 2.13 | 0.00015 | True |

Reasons: 

