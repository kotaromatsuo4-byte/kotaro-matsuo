# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T18:24:20.936248+00:00
- Atomic endpoint: 6144 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4203 | 10.02 | 0.0008097 | 0.0001351 | 970 | 951 | 670 | 27.6/41.1/45.4 | 17.8/35.9/42.9; 100.0% | 0.478/0.784/1.07 | 2.09 | 0.0001 | True |

Reasons: no collapse warning

