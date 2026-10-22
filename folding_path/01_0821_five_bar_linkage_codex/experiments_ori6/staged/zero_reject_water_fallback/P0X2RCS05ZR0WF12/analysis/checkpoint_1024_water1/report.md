# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T18:12:13.666093+00:00
- Atomic endpoint: 1024 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `catastrophic_stop_only_no_positive_adoption`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 0.01699 | 1.81 | 0.04158 | 91 | 41 | 31 | 6.7/26.3/43.4 | 1.33/3.51/8.9; 42.2% | 0.0939/0.149/0.254 | 2.17 | 0.0005 | False |
| 4302 | -0.4412 | 0.5855 | 0.1151 | 0 | 0 | 0 | 1.03/1.63/2.97 | 0.659/1.67/2.56; 23.4% | 0.0974/0.169/0.248 | 2.17 | 0.0005 | False |
| 4303 | 0.294 | 1.838 | 0.0141 | 150 | 60 | 38 | 8.22/32.2/48.9 | 1.59/4.65/7.38; 57.8% | 0.0922/0.146/0.23 | 2.13 | 0.0005 | False |

Reasons: no catastrophic condition; continue to 4096

