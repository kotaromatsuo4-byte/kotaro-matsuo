# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T22:24:26.813769+00:00
- Atomic endpoint: 8192 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `selected_seed_two_consecutive_checkpoint_collapse_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 7.971 | 0.007196 | 4.783e-05 | 964 | 692 | 492 | 32.1/47.5/58.3 | 9.68/24/36.7; 100.0% | 0.146/0.285/0.389 | 2.17 | 1.5e-05 | True |

Reasons: no collapse warning

