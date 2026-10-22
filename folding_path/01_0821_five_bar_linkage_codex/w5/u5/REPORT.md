# ori5 uniform-random wall-time calibration

> This report contains calibrated random wall-time estimates. It does **not** contain directly measured Actor--Critic/RL wall time.

Protocol: `ori5_uniform_walltime_v1_fixed_prefix4096`  
Run fingerprint: `03f6ed6264da7ca9473f3d37362c9c418a8da0fe67f9660c5f2e7aefce3cfe31`  
Generated: `2026-08-25T16:07:30.048609+00:00`

| Task | Timing blocks | steps/s (bootstrap CI) | random mechanics RMST | calibrated random wall RMST (joint bootstrap CI) |
|---|---:|---:|---:|---:|
| five_d2 | 30 | 454.758 [421.229, 500.943] | 3717374.233 | 2.2707 h [1.9970, 2.5157] |
| five_d3 | 30 | 512.215 [456.820, 591.106] | 1631493.267 | 0.8848 h [0.6177, 1.1827] |
| torus2 | 10 | 520.022 [510.273, 528.695] | 102400.000 | 0.0547 h [0.0538, 0.0557] |
| water1 | 10 | 874.207 [843.809, 898.919] | 102400.000 | 0.0325 h [0.0316, 0.0337] |
| watersqrt2 | 10 | 875.673 [841.448, 905.191] | 18132.500 | 0.0058 h [0.0026, 0.0092] |

The joint interval resamples both fixed-seed timing blocks and the stored common-censor random first-hit trials. The conversion assumes that seconds per mechanics step observed on the fixed prefixes is representative of the longer random search. Machine-load drift and RL forward/backward/update overhead are outside this interval.

Measured rollout mechanics: 368640  
Measured rollout wall seconds: 682.503936  
Excluded warm-up mechanics: 2560
