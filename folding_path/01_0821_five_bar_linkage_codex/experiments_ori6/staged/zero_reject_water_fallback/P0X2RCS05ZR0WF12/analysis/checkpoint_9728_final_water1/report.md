# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T23:25:02.555768+00:00
- Atomic endpoint: 9728 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 8.339 | 0.0009884 | 3.73e-05 | 975 | 716 | 524 | 35.4/48.4/56.5 | 11/25.3/37.9; 100.0% | 0.139/0.242/0.352 | 2.17 | 1.5e-05 | True |

Reasons: no collapse warning

