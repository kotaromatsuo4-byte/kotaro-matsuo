# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T21:51:36.275640+00:00
- Atomic endpoint: 7168 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 8.086 | 0.003481 | 4.146e-05 | 953 | 683 | 505 | 35.6/52.7/59.7 | 11.7/26.3/37.2; 98.4% | 0.139/0.222/0.363 | 2.17 | 1.5e-05 | True |

Reasons: no collapse warning

