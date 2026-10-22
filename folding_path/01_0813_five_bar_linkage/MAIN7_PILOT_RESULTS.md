# main7 matched 3000-step pilot

## Conditions

- seeds: 600, 601, 602
- mechanics-step budget: 3000
- paired warmup groups: 4 (16 evaluated angle branches)
- parent limit: 16; first directions per parent: 2; angles: 16
- `FOLDING_SAVE_TREE=0`, `FOLDING_VERBOSE=0`
- selectors: `independent_random`, `fixed_transport`, `joint_macro_gp`,
  `joint_macro_gp_shuffled`, `joint_macro_gp_cost_aware`, and
  `joint_macro_gp_cost_aware_shuffled`

The four core selectors (12 processes) ran concurrently; the two cost-aware
selectors (6 processes) then ran as a second concurrent batch. Therefore wall
time is an execution audit under shared CPU contention, not a clean serial
speed benchmark or a direct comparison between the two batches.

The post-warmup AUC is

\[
\frac{1}{B-t_w}\int_{t_w}^{B}
  \left(N(t)-N(t_w)\right)\,dt,
\]

so it counts only cells discovered after the identical paired prefix.

## Per-seed results

| seed | selector | final cells | AUC | post-warmup AUC | post-warmup cells | wall time [s] |
|---:|---|---:|---:|---:|---:|---:|
| 600 | independent random | 2 | 1.940 | 0.000 | 0 | 73.38 |
| 600 | fixed transport | 2 | 1.940 | 0.000 | 0 | 73.54 |
| 600 | joint GP | 2 | 1.940 | 0.000 | 0 | 75.81 |
| 600 | joint GP shuffled | 2 | 1.940 | 0.000 | 0 | 75.29 |
| 600 | cost-aware joint GP | 2 | 1.940 | 0.000 | 0 | 102.74 |
| 600 | cost-aware shuffled | 2 | 1.940 | 0.000 | 0 | 92.73 |
| 601 | independent random | 2 | 1.332 | 0.397 | 1 | 71.81 |
| 601 | fixed transport | 1 | 0.979 | 0.000 | 0 | 70.21 |
| 601 | joint GP | 2 | 1.802 | 0.926 | 1 | 76.10 |
| 601 | joint GP shuffled | 3 | 1.991 | 1.139 | 2 | 76.76 |
| 601 | cost-aware joint GP | 1 | 0.979 | 0.000 | 0 | 107.15 |
| 601 | cost-aware shuffled | 2 | 1.589 | 0.686 | 1 | 92.83 |
| 602 | independent random | 1 | 0.979 | 0.000 | 0 | 70.71 |
| 602 | fixed transport | 2 | 1.833 | 0.972 | 1 | 72.85 |
| 602 | joint GP | 3 | 2.529 | 1.763 | 2 | 75.68 |
| 602 | joint GP shuffled | 2 | 1.800 | 0.934 | 1 | 76.92 |
| 602 | cost-aware joint GP | 2 | 1.415 | 0.496 | 1 | 105.21 |
| 602 | cost-aware shuffled | 2 | 1.485 | 0.576 | 1 | 94.09 |

## Aggregate (mean +/- sample SD)

| selector | final cells | AUC | post-warmup AUC | post-warmup cells | wall time [s] |
|---|---:|---:|---:|---:|---:|
| independent random | 1.667 +/- 0.577 | 1.417 +/- 0.486 | 0.132 +/- 0.229 | 0.333 +/- 0.577 | 71.97 +/- 1.34 |
| fixed transport | 1.667 +/- 0.577 | 1.584 +/- 0.527 | 0.324 +/- 0.561 | 0.333 +/- 0.577 | 72.20 +/- 1.76 |
| joint GP | 2.333 +/- 0.577 | 2.090 +/- 0.386 | 0.896 +/- 0.882 | 1.000 +/- 1.000 | 75.86 +/- 0.22 |
| joint GP shuffled | 2.333 +/- 0.577 | 1.911 +/- 0.099 | 0.691 +/- 0.607 | 1.000 +/- 1.000 | 76.33 +/- 0.90 |
| cost-aware joint GP | 1.667 +/- 0.577 | 1.445 +/- 0.481 | 0.165 +/- 0.286 | 0.333 +/- 0.577 | 105.03 +/- 2.21 |
| cost-aware shuffled | 2.000 +/- 0.000 | 1.671 +/- 0.239 | 0.421 +/- 0.368 | 0.667 +/- 0.577 | 93.21 +/- 0.76 |

Joint GP minus shuffled GP had paired final-cell differences `(0,-1,+1)`
and AUC differences `(0,-0.189,+0.729)`. Thus its mean AUC was 0.180 higher,
but its mean final cell count was identical. Three seeds do not establish
that the learned score itself caused the improvement. The joint GP is more
promising than the earlier angle-only GP because it improved over random on
this pilot, but the score-shuffled control prevents a strong efficacy claim.

Cost-aware GP was worse than its shuffled control: paired final-cell
differences were `(0,-1,0)`, and paired AUC differences were
`(0,-0.610,-0.071)`. It generated many more decision rows (mean 275.7 versus
159.7) because it strongly preferred short outcomes, but those outcomes were
often collisions and did not improve archive discovery. Dividing by expected
cost therefore optimized rollout throughput rather than useful diversity in
this experiment. It also added an exact cost GP, increasing fit/score time.

## Prefix audit

For every seed, all six selectors had the same warmup step, cell count, and
tree/archive hash.

| seed | warmup step | warmup cells | SHA-256 |
|---:|---:|---:|---|
| 600 | 610 | 2 | `730200f38558754a9af64ed482430f0ece6fec277ac66bbd792fef689cf60df0` |
| 601 | 333 | 1 | `6f9b2877d9bd18c01d886e408ca42c5d83c8fa3d850c5eef62fd7bc1bde419d3` |
| 602 | 363 | 1 | `e4fd998882783b3a9ee50ed7e9301ed5e15717a73a49f0048bde7e52f0ffe283` |

## Selection counts

Each method also evaluated four warmup branches for each of the four angle
roles (16 total). Counts below are post-warmup decisions.

| seed | independent random | fixed theta0 / epsilon | joint GP top / random-cold | shuffled top / random-cold | cost-aware top / random-cold | cost shuffled top / random-cold |
|---:|---:|---:|---:|---:|---:|---:|
| 600 | 128 | 93 / 17 | 76 / 14 | 94 / 18 | 188 / 43 | 99 / 23 |
| 601 | 153 | 123 / 25 | 72 / 13 | 129 / 27 | 233 / 58 | 134 / 27 |
| 602 | 160 | 113 / 27 | 79 / 23 | 128 / 28 | 213 / 47 | 122 / 28 |

The learned GP consumed the same step budget in fewer macro decisions than
the shuffled and random policies. This means it selected longer deterministic
suffixes on average and consequently accumulated fewer teachers (105, 100,
117 versus shuffled 127, 171, 171). A cost-aware selector is therefore a
well-motivated control, but the completed test shows that the current expected
cost denominator over-rewards cheap collisions. A better cost formulation
would model cost conditional on a feasible/flat outcome, or combine cost with
an explicit feasibility floor instead of dividing an uncertain discovery
score unconditionally.
