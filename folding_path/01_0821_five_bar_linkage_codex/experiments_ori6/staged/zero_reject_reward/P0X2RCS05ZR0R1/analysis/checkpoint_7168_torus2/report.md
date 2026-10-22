# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T19:11:29.075508+00:00
- Atomic endpoint: 7168 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 11.73 | 0.0006742 | 9.291e-05 | 957 | 941 | 822 | 19.1/36.6/43.2 | 12.3/24.2/37.8; 100.0% | 0.471/0.774/1.07 | 2.11 | 3e-05 | True |

Reasons: no collapse warning

