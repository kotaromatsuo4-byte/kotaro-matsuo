# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T15:23:44.225010+00:00
- Atomic endpoint: 1024 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `catastrophic_stop_only_no_positive_adoption`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.4042 | 0.03776 | 0.118 | 1 | 0 | 0 | 1.26/1.91/2.27 | 1.31/2.69/3.58; 53.1% | 0.128/0.216/0.269 | 2.21 | 0.0005 | False |
| 4202 | 0.04218 | 0.2287 | 0.1162 | 2 | 0 | 0 | 1.26/1.71/11.6 | 0.769/1.93/2.43; 23.4% | 0.0918/0.148/0.2 | 2.13 | 0.0005 | False |
| 4203 | -0.4144 | 1.284 | 0.07355 | 2 | 0 | 0 | 0.283/0.658/0.999 | 0.299/0.813/0.991; 0.0% | 0.0876/0.147/0.204 | 2.17 | 0.0005 | False |

Reasons: no catastrophic condition; continue to 4096

