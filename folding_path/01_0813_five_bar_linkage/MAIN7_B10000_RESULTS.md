# main7 matched 10000-step extension

## Conditions and audit

The frozen `main7_joint_k2_macro_gp_v1` implementation was run with seeds
600--602, a 10000 mechanics-step budget, four paired warmup groups, and the
same candidate-pool and GP settings as the 3000-step pilot. The four core
selectors were run concurrently within each seed; seeds were run sequentially.

After removing only `selector` and `result_name`, saved configurations were
identical across all four methods within every seed. Warmup geometry/archive
hashes, end steps, and cell counts also matched exactly:

| seed | warmup step | warmup cells | SHA-256 |
|---:|---:|---:|---|
| 600 | 610 | 2 | `730200f38558754a9af64ed482430f0ece6fec277ac66bbd792fef689cf60df0` |
| 601 | 333 | 1 | `6f9b2877d9bd18c01d886e408ca42c5d83c8fa3d850c5eef62fd7bc1bde419d3` |
| 602 | 363 | 1 | `e4fd998882783b3a9ee50ed7e9301ed5e15717a73a49f0048bde7e52f0ffe283` |

The saved archive representatives were independently reassigned to the six
five-bar canonical witnesses. Every archive cell matched a distinct canonical
cell (`extra_cell_count=0`), so archive count and canonical coverage coincide
in this experiment.

## Per-seed results

Post-warmup AUC is the normalized integral of only the additional cells after
the common warmup. `T6` is the first mechanics step at which all six canonical
cells have been found.

| seed | selector | canonical cells | AUC | post-warmup AUC | post-warmup cells | T2 | T3 | T4 | T6 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | independent random | 2 | 1.982 | 0.000 | 0 | 116 | -- | -- | >10000 |
| 600 | fixed transport | 2 | 1.982 | 0.000 | 0 | 116 | -- | -- | >10000 |
| 600 | joint GP | 2 | 1.982 | 0.000 | 0 | 116 | -- | -- | >10000 |
| 600 | joint GP shuffled | 2 | 1.982 | 0.000 | 0 | 116 | -- | -- | >10000 |
| 601 | independent random | 2 | 1.800 | 0.834 | 1 | 1940 | -- | -- | >10000 |
| 601 | fixed transport | 2 | 1.483 | 0.506 | 1 | 5107 | -- | -- | >10000 |
| 601 | joint GP | 2 | 1.941 | 0.980 | 1 | 530 | -- | -- | >10000 |
| 601 | joint GP shuffled | 3 | 2.697 | 1.762 | 2 | 1113 | 1850 | -- | >10000 |
| 602 | independent random | 1 | 0.994 | 0.000 | 0 | -- | -- | -- | >10000 |
| 602 | fixed transport | 2 | 1.950 | 0.992 | 1 | 437 | -- | -- | >10000 |
| 602 | joint GP | 4 | 2.985 | 2.066 | 3 | 511 | 840 | 8738 | >10000 |
| 602 | joint GP shuffled | 2 | 1.940 | 0.982 | 1 | 537 | -- | -- | >10000 |

Canonical IDs covered were:

- independent random: seed 600 `{0,1}`, seed 601 `{0,1}`, seed 602 `{0}`;
- fixed transport: seed 600 `{0,1}`, seed 601 `{0,1}`, seed 602 `{0,3}`;
- joint GP: seed 600 `{0,1}`, seed 601 `{0,1}`, seed 602 `{0,1,2,3}`;
- shuffled GP: seed 600 `{0,1}`, seed 601 `{0,1,3}`, seed 602 `{0,1}`.

No method reached canonical cells 4 and 5, and no run reached all six.

## Aggregate (mean +/- sample SD)

| selector | canonical cells | AUC | post-warmup AUC | post-warmup cells | wall time [s] |
|---|---:|---:|---:|---:|---:|
| independent random | 1.667 +/- 0.577 | 1.592 +/- 0.526 | 0.278 +/- 0.481 | 0.333 +/- 0.577 | 151.38 +/- 1.42 |
| fixed transport | 2.000 +/- 0.000 | 1.805 +/- 0.279 | 0.499 +/- 0.496 | 0.667 +/- 0.577 | 150.92 +/- 3.00 |
| joint GP | 2.667 +/- 1.155 | 2.303 +/- 0.591 | 1.015 +/- 1.034 | 1.333 +/- 1.528 | 181.37 +/- 2.96 |
| joint GP shuffled | 2.333 +/- 0.577 | 2.207 +/- 0.426 | 0.915 +/- 0.883 | 1.000 +/- 1.000 | 203.20 +/- 1.88 |

The paired joint-GP minus shuffled differences were:

- final canonical cells: `(0,-1,+2)`, mean `+0.333`;
- AUC: `(0,-0.757,+1.045)`, mean `+0.096`;
- post-warmup AUC: `(0,-0.783,+1.084)`, mean `+0.100`.

The mean ordering favors joint GP, and its seed-602 trajectory is the best
single run. However, the advantage comes entirely from seed 602; shuffled GP
is clearly better on seed 601, while seed 600 is tied. Therefore this
three-seed extension still does not isolate a reliable causal benefit of the
learned acquisition score. It supports the weaker claim that joint macro
selection can produce useful search pressure on some trajectories, not that
the current GP acquisition is generally superior.

## Search throughput and compute cost

Mean terminal teacher counts were 602 (random), 539 (fixed), 353 (joint GP),
and 602 (shuffled GP). The unshuffled GP again selected substantially longer
rollouts and therefore made fewer macro decisions within the same mechanics
budget. Mean wall times were measured while four jobs shared CPU, so they are
not standalone runtime benchmarks. Nevertheless, the GP fitting/scoring
overhead is real: joint GP spent about 25.2 s fitting and 18.8 s scoring per
run on average; shuffled GP, which accumulated more teachers, spent about
58.0 s fitting and 30.9 s scoring.

The corresponding machine-readable canonical audit is
`result/main7_b10000_canonical_coverage.json` (and `.csv`).

