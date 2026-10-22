# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T19:32:27.415876+00:00
- Atomic endpoint: 3072 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4301 | 5.392 | 0.118 | 0.001606 | 850 | 488 | 291 | 28.2/51.4/68.7 | 6/14.8/26.4; 89.1% | 0.109/0.184/0.219 | 2.17 | 0.00015 | True |
| 4302 | -0.2088 | 0.1643 | 0.1165 | 2 | 0 | 0 | 1/1.83/1.96 | 0.763/1.78/1.97; 25.0% | 0.0941/0.161/0.208 | 2.21 | 0.00015 | False |
| 4303 | 5.034 | 0.1713 | 0.002265 | 763 | 417 | 276 | 28.7/58.8/73.4 | 5.74/16.8/22.2; 89.1% | 0.119/0.208/0.26 | 2.15 | 0.00015 | True |

Reasons: 

