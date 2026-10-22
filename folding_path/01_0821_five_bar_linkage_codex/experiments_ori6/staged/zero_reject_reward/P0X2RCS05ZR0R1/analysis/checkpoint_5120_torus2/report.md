# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T17:35:35.971351+00:00
- Atomic endpoint: 5120 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 7.362 | 0.002618 | 0.0003051 | 953 | 908 | 445 | 34.6/44.6/50.6 | 16.5/37.4/48.4; 100.0% | 0.408/0.642/0.835 | 2.09 | 0.0001 | True |

Reasons: no collapse warning

