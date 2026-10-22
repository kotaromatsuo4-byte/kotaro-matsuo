# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T15:02:19.781358+00:00
- Atomic endpoint: 512 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## water1

Decision: `snapshot_only`; stop=False; promote=False.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | -0.2644 | 0.1179 | 0.1176 | 0 | 0 | 0 | 1.47/2.33/2.54 | 1.2/2.37/3.1; 54.7% | 0.0929/0.15/0.284 | 2.16 | 0.0005 | False |
| 4202 | -0.02217 | 0.2183 | 0.1166 | 1 | 0 | 0 | 1.26/1.75/1.89 | 0.687/1.2/2.28; 17.2% | 0.0894/0.158/0.198 | 2.15 | 0.0005 | False |
| 4203 | -0.5166 | 1.341 | 0.06664 | 0 | 0 | 0 | 0.299/0.834/1.72 | 0.588/1.75/2.76; 20.3% | 0.0962/0.162/0.258 | 2.15 | 0.0005 | False |

Reasons: 

