# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T21:21:38.045008+00:00
- Atomic endpoint: 10000 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 12.89 | 0.0006303 | 7.783e-05 | 987 | 982 | 915 | 15.1/42.9/50.9 | 13.5/32.1/38.9; 100.0% | 0.474/0.813/1.37 | 2.12 | 3e-05 | True |

Reasons: no collapse warning

