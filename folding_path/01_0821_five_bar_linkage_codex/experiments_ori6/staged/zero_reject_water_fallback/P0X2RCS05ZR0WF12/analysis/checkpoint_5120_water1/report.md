# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T20:44:25.795980+00:00
- Atomic endpoint: 5120 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 7.351 | 0.02387 | 0.0002021 | 925 | 649 | 443 | 31.3/46.1/52 | 6.91/16.2/18.2; 100.0% | 0.142/0.252/0.284 | 2.16 | 5e-05 | True |

Reasons: no collapse warning

