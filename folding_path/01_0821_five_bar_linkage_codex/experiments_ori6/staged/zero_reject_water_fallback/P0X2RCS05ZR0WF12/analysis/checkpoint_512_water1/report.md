# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T17:51:47.270189+00:00
- Atomic endpoint: 512 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | -0.4639 | 1.813 | 0.04592 | 29 | 7 | 4 | 1.7/8.17/21.3 | 0.957/2/2.96; 43.8% | 0.098/0.177/0.251 | 2.15 | 0.0005 | False |
| 4302 | -0.558 | 0.4523 | 0.1165 | 0 | 0 | 0 | 1.5/3.03/3.64 | 1.2/2.94/3.41; 43.8% | 0.098/0.167/0.195 | 2.15 | 0.0005 | False |
| 4303 | -0.4818 | 1.958 | 0.02278 | 40 | 14 | 7 | 3.08/20.5/25.1 | 1.33/3.27/5.24; 56.2% | 0.0891/0.155/0.223 | 2.14 | 0.0005 | False |

Reasons: 

