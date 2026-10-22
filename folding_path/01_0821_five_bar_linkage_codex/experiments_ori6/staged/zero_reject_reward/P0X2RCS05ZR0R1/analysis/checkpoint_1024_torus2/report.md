# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T15:05:32.940311+00:00
- Atomic endpoint: 1024 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `catastrophic_stop_only_no_positive_adoption`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | 0.1097 | 0.8358 | 0.07749 | 237 | 1 | 0 | 0.221/0.594/0.848 | 0.629/1.71/2.44; 17.2% | 0.474/1/1.33 | 2.14 | 0.001 | False |
| 4202 | -0.3974 | 1.124 | 0.1137 | 0 | 0 | 0 | 0.033/0.0626/0.0856 | 0.209/0.397/0.572; 0.0% | 0.265/0.442/0.568 | 2.1 | 0.001 | False |
| 4203 | -0.2402 | 1.032 | 0.1113 | 73 | 0 | 0 | 0.22/0.67/0.984 | 0.81/2.09/2.71; 37.5% | 0.293/0.582/0.623 | 2.1 | 0.001 | False |

Reasons: no catastrophic condition; continue to 4096

