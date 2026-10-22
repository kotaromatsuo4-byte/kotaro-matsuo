# main8 variable-K pilot (B=3000)

## Audit

| seed | config | warmup step | raw/canonical cells | prefix/training hash count |
|---:|---|---:|---|---|
| 700 | match | [337] | [2]/[1] | 1/1 |
| 701 | match | [454] | [3]/[3] | 1/1 |
| 702 | match | [75] | [1]/[1] | 1/1 |

## Aggregate

| selector | raw/canonical final | raw/canonical AUC | postwarm raw/canonical AUC | T6 | rollouts/1000 | requested/executed K | wall [s] |
|---|---|---|---|---|---|---|---|
| variable_k_random | 8.000 +/- 0.000 / 6.000 +/- 0.000 | 5.987 +/- 0.459 / 4.634 +/- 0.455 | 4.489 +/- 1.128 / 3.342 +/- 0.729 | 3/3 (1974.0) | 30.567 +/- 1.828 | 12.631 +/- 1.651 / 10.596 +/- 1.005 | 51.169 +/- 17.169 |
| variable_k_gp_argmax | 7.667 +/- 0.577 / 6.000 +/- 0.000 | 5.263 +/- 0.295 / 4.362 +/- 0.617 | 3.709 +/- 0.917 / 3.016 +/- 1.278 | 3/3 (1915.0) | 27.590 +/- 1.291 | 11.789 +/- 1.863 / 10.924 +/- 1.711 | 53.785 +/- 18.013 |
| variable_k_gp_argmax_shuffled | 7.333 +/- 0.577 / 5.667 +/- 0.577 | 5.583 +/- 0.609 / 4.397 +/- 0.646 | 4.069 +/- 0.710 / 3.106 +/- 0.341 | 2/3 (1472.5) | 30.100 +/- 2.147 | 12.890 +/- 1.192 / 11.079 +/- 1.736 | 52.323 +/- 17.965 |
| variable_k_gp_filter | 8.000 +/- 0.000 / 6.000 +/- 0.000 | 6.018 +/- 0.252 / 4.780 +/- 0.268 | 4.536 +/- 0.864 / 3.506 +/- 0.799 | 3/3 (1508.0) | 27.499 +/- 0.837 | 13.431 +/- 1.233 / 12.260 +/- 0.878 | 52.592 +/- 18.267 |
| variable_k_gp_filter_shuffled | 7.667 +/- 0.577 / 6.000 +/- 0.000 | 5.241 +/- 0.177 / 4.799 +/- 0.595 | 3.686 +/- 0.651 / 3.518 +/- 0.824 | 3/3 (1367.3333333333333) | 30.955 +/- 0.632 | 13.289 +/- 0.500 / 11.268 +/- 0.652 | 52.904 +/- 18.033 |

## Per seed

| seed | selector | raw/canonical | raw/canonical AUC | postwarm raw/canonical AUC | T6 | discovery K |
|---:|---|---|---|---|---:|---|
| 700 | variable_k_random | 8/6 | 5.730/4.113 | 4.352/3.531 | 2708 | [17, 19, 16, 21, 17, 13] |
| 700 | variable_k_gp_argmax | 8/6 | 5.599/3.848 | 4.205/3.232 | 2468 | [20, 21, 11, 13, 12, 18] |
| 700 | variable_k_gp_argmax_shuffled | 7/6 | 4.925/4.037 | 3.445/3.445 | 2216 | [12, 24, 14, 22, 21] |
| 700 | variable_k_gp_filter | 8/6 | 5.797/4.478 | 4.427/3.942 | 1637 | [24, 24, 13, 1, 20, 23] |
| 700 | variable_k_gp_filter_shuffled | 8/6 | 5.060/4.116 | 3.597/3.534 | 2176 | [23, 14, 7, 24, 21, 8] |
| 701 | variable_k_random | 8/6 | 5.714/4.951 | 3.437/2.538 | 1469 | [8, 17, 7, 22, 24] |
| 701 | variable_k_gp_argmax | 8/6 | 5.047/4.192 | 2.651/1.643 | 2448 | [20, 19, 19, 24, 6] |
| 701 | variable_k_gp_argmax_shuffled | 8/6 | 6.125/5.143 | 3.921/2.764 | 729 | [8, 21, 12, 18, 24] |
| 701 | variable_k_gp_filter | 8/6 | 5.964/4.990 | 3.732/2.584 | 1038 | [20, 17, 18, 23, 19] |
| 701 | variable_k_gp_filter_shuffled | 8/6 | 5.415/5.077 | 3.084/2.686 | 801 | [18, 16, 24, 22, 11] |
| 702 | variable_k_random | 8/6 | 6.517/4.838 | 5.680/3.958 | 1745 | [6, 9, 11, 22, 20, 19, 23] |
| 702 | variable_k_gp_argmax | 7/6 | 5.143/5.047 | 4.270/4.172 | 829 | [20, 16, 9, 9, 5, 14] |
| 702 | variable_k_gp_argmax_shuffled | 7/5 | 5.699/4.011 | 4.841/3.110 | -- | [22, 12, 17, 23, 16, 23] |
| 702 | variable_k_gp_filter | 8/6 | 6.293/4.871 | 5.450/3.992 | 1849 | [24, 8, 11, 11, 23, 16, 24] |
| 702 | variable_k_gp_filter_shuffled | 7/6 | 5.247/5.205 | 4.377/4.334 | 1125 | [24, 22, 24, 1, 24, 22] |

## Paired deltas (left - right)

| left | right | raw AUC | canonical AUC | postwarm canonical AUC | final canonical |
|---|---|---:|---:|---:|---:|
| variable_k_gp_argmax | variable_k_random | -0.724 | -0.272 | -0.326 | +0.000 |
| variable_k_gp_argmax | variable_k_gp_argmax_shuffled | -0.320 | -0.035 | -0.090 | +0.333 |
| variable_k_gp_filter | variable_k_random | +0.031 | +0.146 | +0.164 | +0.000 |
| variable_k_gp_filter | variable_k_gp_filter_shuffled | +0.777 | -0.019 | -0.012 | +0.000 |

All steps include perturbation micro-steps, deterministic suffix steps, and failed mechanics steps.  Canonical curves ignore valid extra layer variants.

## Interpretation

The dominant result is the strength of the variable-length persistent macro
itself.  Uniform random selection of `(parent, v1, K)` reached all six
canonical cells in every seed by steps 2708, 1469, and 1745.  Its mean T6 was
1974 mechanics steps, despite using perturbations of norm 0.20.  Thus a long
sequence of individually small, transported perturbations can cross the basin
boundaries which the old single-perturbation random search rarely crossed.

Long requested chains were important in the observed discoveries.  For the
random selector, 13 of 18 post-warmup raw-cell discoveries requested `K >= 13`
(mean 16.17).  For the GP filter this was 14 of 18 (mean 17.72).  This supports
including `K` in the macro action, although it does not by itself prove that
the GP learned the correct K.

Global argmax should not be retained.  Relative to uniform random it reduced
mean canonical AUC by 0.272 and post-warmup canonical AUC by 0.326.  It also
executed about 2.98 fewer post-warmup macros per 1000 mechanics steps.  The
selected candidate acquisitions did not separate discoveries: their mean
acquisition was 21.88 for new raw cells and 25.62 for non-new outcomes.

The top-25% GP filter was safer than argmax.  Against uniform random its
canonical AUC differences were +0.365, +0.039, and +0.033, and its mean T6 was
1508 rather than 1974.  However, this is not yet evidence for the learned GP
score.  Against the score-shuffled filter, its canonical AUC differences were
+0.363, -0.086, and -0.334 (mean -0.019), and post-warm-up canonical AUC
differed by only -0.012 on average.  The shuffled control also had a lower
mean T6 of 1367.  Therefore the preregistered requirement that GP outperform
both random and its score-shuffled control is not met.

Raw archive AUC tells a different but secondary story: the GP filter exceeded
its shuffled control in all three seeds by +0.737, +0.550, and +1.046.  The
difference comes mainly from valid extra layer-order variants; after mapping
to the theoretical canonical six, it disappears.  More seeds would be needed
before treating even this raw-archive result as a reproducible benefit.

The present pilot therefore supports the structural prior
`variable-K persistent transport`, but not the current GP acquisition.  A
held-out claim of GP superiority should not be started from this acquisition
unchanged.  The next defensible GP test is a conservative negative filter:
discard only candidates predicted with high confidence to collide or return
to an existing cell, retain uncertain candidates, and keep an explicit raw
random bypass.  An offline paired `(parent, v1)` panel over every K should
first verify high new-cell recall and measure the false-rejection rate.
