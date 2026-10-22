# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T15:35:06.215378+00:00
- Atomic endpoint: 2048 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | 0.8879 | 0.4059 | 0.03303 | 639 | 18 | 0 | 0.134/0.29/0.429 | 0.347/0.919/1.05; 4.7% | 0.526/0.984/1.18 | 2.24 | 0.0003 | True |
| 4202 | -0.07052 | 1.088 | 0.05711 | 0 | 0 | 0 | 0.00669/0.0114/0.0218 | 0.085/0.19/0.238; 0.0% | 0.219/0.363/0.505 | 2.15 | 0.0003 | False |
| 4203 | 0.2142 | 0.79 | 0.07571 | 250 | 2 | 0 | 0.282/0.595/0.882 | 0.376/0.813/1.02; 1.6% | 0.417/0.809/0.998 | 2.1 | 0.0003 | True |

Reasons: 

