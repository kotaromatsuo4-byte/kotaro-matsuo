# Seven-bar D2/D3 → D4 evaluation

Training seed: 9301; training starts: 16; held-out starts: 20.
Training goals: D2, D3 only. Max steps: 400; epochs per round: 30; aggregation rounds: 3.
D4 is evaluated only after all training. NN receives current coordinates and requested spread; never target coordinates.
The teacher reference receives full target coordinates and is a feasibility/reference check, not a learned baseline.

| Method | Goal | Success | Median flatness | Median spread | Median steps |
|---|---:|---:|---:|---:|---:|
| NN | 2 | 8/20 | 0.00183287 | 2 | 48.5 |
| NN | 3 | 12/20 | 0.000780081 | 2.99988 | 46 |
| NN | 4 | 0/20 | 0.000627827 | 2.99993 | 52.5 |
| teacher_reference | 2 | 15/20 | 0.000517891 | 2.00001 | 48.5 |
| teacher_reference | 3 | 13/20 | 0.000688743 | 2.99983 | 44.5 |
| teacher_reference | 4 | 4/20 | 0.0369404 | 3.83014 | 53 |

Checkpoint SHA256 (unchanged after evaluation): `66cb5ee5f2c7d084de9b8ffb8325b3fb1b91be35a34cc2524f717a3adc73e1d0`.
Success requires F <= 0.001, spread error <= 0.05 and length-error sum <= 1e-8; crossing candidates are rejected.
One training seed, same seven-bar structure and randomized starts around one reference shape. This is target extrapolation, not transfer to an unseen structure.
The D2/D3 target configurations are one chosen pair of overlapping straight shapes; results may depend on that choice.
Failure here does not prove that D4 transfer is impossible. No hyperparameter choice was made using D4 outcomes.
