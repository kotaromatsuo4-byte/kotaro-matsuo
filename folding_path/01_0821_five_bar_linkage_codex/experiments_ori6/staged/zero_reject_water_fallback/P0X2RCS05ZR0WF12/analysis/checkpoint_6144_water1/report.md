# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T21:18:13.196119+00:00
- Atomic endpoint: 6144 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 8.047 | 0.005409 | 4.61e-05 | 952 | 691 | 501 | 33.4/50.1/59.7 | 10.7/22.6/38.4; 98.4% | 0.132/0.234/0.297 | 2.17 | 5e-05 | True |

Reasons: no collapse warning

