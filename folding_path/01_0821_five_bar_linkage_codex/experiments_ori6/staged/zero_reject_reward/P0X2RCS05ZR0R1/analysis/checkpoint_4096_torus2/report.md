# Zero-reject staged experiment checkpoint report

- Generated: 2026-08-26T16:45:00.823709+00:00
- Atomic endpoint: 4096 episodes
- Integrity: PASS
- Curves use moving episodic statistics; cumulative reward sums are not used.

## torus2

Decision: `formal_2_of_3_persistent_target_plane_gate`; stop=False; promote=True.

| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4201 | 0.4644 | 0.6559 | 0.05486 | 232 | 5 | 0 | 0.131/0.287/0.477 | 0.429/0.886/1.41; 3.1% | 0.694/1.37/1.85 | 2.32 | 0.0001 | False |
| 4202 | 0.3858 | 0.7785 | 0.04686 | 179 | 0 | 0 | 0.0382/0.0752/0.092 | 0.2/0.457/0.708; 0.0% | 0.284/0.488/0.524 | 2.15 | 0.0001 | True |
| 4203 | 2.448 | 0.08127 | 0.00882 | 930 | 567 | 64 | 14.6/34.6/53.1 | 4.99/11.3/23.6; 79.7% | 0.433/0.858/1.02 | 2.09 | 0.0001 | True |

Reasons: 2/3 persistent evidence passed; continue selected-seed diagnostic

