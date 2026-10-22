# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T22:57:24.968527+00:00
- Atomic endpoint: 9216 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 8.175 | 0.003074 | 3.852e-05 | 963 | 710 | 510 | 33.2/46.4/53.7 | 10.6/23.5/38.6; 100.0% | 0.143/0.256/0.274 | 2.17 | 1.5e-05 | True |

Reasons: no collapse warning

