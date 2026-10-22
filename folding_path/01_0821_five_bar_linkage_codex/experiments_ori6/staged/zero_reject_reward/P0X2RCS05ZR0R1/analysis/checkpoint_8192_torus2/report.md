# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T19:57:16.425291+00:00
- Atomic endpoint: 8192 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 12.68 | 0.0006188 | 8.298e-05 | 973 | 962 | 899 | 14.7/35/48.1 | 11.9/24.6/41.9; 100.0% | 0.495/0.827/1.07 | 2.11 | 3e-05 | True |

Reasons: no collapse warning

