# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T20:41:35.193069+00:00
- Atomic endpoint: 9216 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 12.75 | 0.0006385 | 8.767e-05 | 973 | 967 | 905 | 13.1/45.6/74.3 | 14.4/39.6/70.5; 100.0% | 0.501/0.821/1.1 | 2.12 | 3e-05 | True |

Reasons: no collapse warning

