# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T21:21:38.964398+00:00
- Atomic endpoint: 4096 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `formal_2_of_3_persistent_target_plane_gate`; stop=True; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.01589 | 0.1852 | 0.1165 | 1 | 0 | 0 | 0.92/1.47/1.7 | 0.681/1.35/1.75; 18.8% | 0.0908/0.155/0.245 | 2.26 | 5e-05 | False |
| 4202 | 0.2359 | 0.1309 | 0.1166 | 0 | 0 | 0 | 0.899/1.33/9.66 | 0.884/2.12/3.29; 25.0% | 0.083/0.124/0.163 | 2.04 | 5e-05 | False |
| 4203 | 0.6545 | 0.2265 | 0.1137 | 90 | 0 | 0 | 0.275/0.606/1.12 | 0.435/0.984/1.77; 4.7% | 0.104/0.185/0.235 | 2.12 | 5e-05 | True |

Reasons: fewer than 2/3 seeds passed the preregistered persistence gate

