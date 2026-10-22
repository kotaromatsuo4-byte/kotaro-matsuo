# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T20:09:57.652305+00:00
- Atomic endpoint: 4096 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `formal_2_of_3_persistent_target_plane_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 6.609 | 0.06105 | 0.000688 | 894 | 584 | 387 | 29.8/50.8/70.3 | 7.38/17.5/31.8; 93.8% | 0.141/0.251/0.359 | 2.16 | 5e-05 | True |
| 4302 | -0.1104 | 0.1871 | 0.116 | 1 | 0 | 0 | 1.13/2.14/3.28 | 0.97/2.1/4.06; 40.6% | 0.0852/0.145/0.194 | 2.24 | 5e-05 | False |
| 4303 | 6.67 | 0.06358 | 0.0007295 | 918 | 571 | 391 | 29.6/58.6/63.2 | 8.04/21.4/23.7; 95.3% | 0.129/0.207/0.287 | 2.14 | 5e-05 | True |

Reasons: 2/3 persistent evidence passed; continue selected-seed diagnostic

