# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T18:53:12.796500+00:00
- Atomic endpoint: 2048 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 3.561 | 0.3991 | 0.004984 | 569 | 291 | 186 | 23.2/54.7/85.9 | 3.86/10.3/22.1; 90.6% | 0.107/0.182/0.189 | 2.17 | 0.00015 | True |
| 4302 | -0.2726 | 0.3442 | 0.1148 | 3 | 0 | 0 | 1.34/2.3/2.57 | 1.08/2.25/3.1; 48.4% | 0.092/0.14/0.22 | 2.19 | 0.00015 | False |
| 4303 | 2.629 | 0.5735 | 0.007774 | 464 | 215 | 132 | 14.8/40.8/80 | 2.81/6.5/15.6; 82.8% | 0.12/0.195/0.272 | 2.14 | 0.00015 | True |

Reasons: 

