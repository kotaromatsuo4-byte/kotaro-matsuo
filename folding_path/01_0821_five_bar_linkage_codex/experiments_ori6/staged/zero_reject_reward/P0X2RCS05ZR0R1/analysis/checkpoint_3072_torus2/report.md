# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T16:07:04.063645+00:00
- Atomic endpoint: 3072 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | 1.121 | 0.2753 | 0.02107 | 861 | 66 | 0 | 0.152/0.353/0.561 | 0.314/0.725/1.43; 1.6% | 0.593/1.27/1.62 | 2.27 | 0.0003 | True |
| 4202 | -0.05924 | 1.089 | 0.04796 | 0 | 0 | 0 | 0.00768/0.0129/0.0897 | 0.0722/0.151/0.328; 0.0% | 0.228/0.38/0.576 | 2.14 | 0.0003 | False |
| 4203 | 0.9706 | 0.3267 | 0.02546 | 736 | 47 | 0 | 0.219/0.531/0.609 | 0.442/0.935/1.07; 4.7% | 0.358/0.707/0.873 | 2.07 | 0.0003 | True |

Reasons: 

