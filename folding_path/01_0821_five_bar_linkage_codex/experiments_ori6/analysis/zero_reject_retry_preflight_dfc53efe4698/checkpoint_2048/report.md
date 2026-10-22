# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T14:40:10.431661+00:00
- Atomic endpoint: 2048 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.4811 | 1.08 | 0.159 | 1 | 0 | 0 | 0.127/0.39/0.806 | 0.667/2.06/2.82; 25.0% | 0.205/0.376/0.463 | 2.17 | 0.001 | False |
| 4202 | -0.6273 | 1.122 | 0.2795 | 0 | 0 | 0 | 0.118/0.538/1.03 | 0.696/2.5/3.4; 17.2% | 0.169/0.293/0.552 | 2.1 | 0.001 | False |
| 4203 | -0.7389 | 1.105 | 0.3217 | 0 | 0 | 0 | 0.0845/0.46/0.882 | 0.541/2.21/3.03; 12.5% | 0.197/0.316/0.398 | 2.15 | 0.001 | False |

Reasons: synthetic snapshot

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.4811 | 1.08 | 0.159 | 1 | 0 | 0 | 0.127/0.39/0.806 | 0.667/2.06/2.82; 25.0% | 0.205/0.376/0.463 | 2.17 | 0.001 | False |
| 4202 | -0.6273 | 1.122 | 0.2795 | 0 | 0 | 0 | 0.118/0.538/1.03 | 0.696/2.5/3.4; 17.2% | 0.169/0.293/0.552 | 2.1 | 0.001 | False |
| 4203 | -0.7389 | 1.105 | 0.3217 | 0 | 0 | 0 | 0.0845/0.46/0.882 | 0.541/2.21/3.03; 12.5% | 0.197/0.316/0.398 | 2.15 | 0.001 | False |

Reasons: synthetic snapshot

