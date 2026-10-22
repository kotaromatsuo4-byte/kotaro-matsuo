# main8 variable-K held-out comparison (B=3000)

Frozen `main8_variable_k_transport_macro_gp_v2`, seeds 703--712.
All methods used the same paired warm-up and mechanics budget.

## Warm-up audit

All ten seeds passed normalized-configuration, prefix-hash, and training-hash equality across the three selectors.

## Aggregate

| selector | raw/canonical final | raw/canonical AUC | postwarm raw/canonical AUC | canonical T6 | rollouts/1000 | requested/executed K | wall [s] |
|---|---|---|---|---|---|---|---|
| variable_k_random | 7.800 +/- 0.422 / 6.000 +/- 0.000 | 5.510 +/- 0.528 / 4.408 +/- 0.637 | 3.300 +/- 0.839 / 2.209 +/- 0.656 | 1899.2 (10/10) | 31.413 +/- 1.273 | 12.577 +/- 0.832 / 10.916 +/- 0.787 | 29.603 +/- 1.904 |
| variable_k_gp_filter | 7.800 +/- 0.422 / 6.000 +/- 0.000 | 5.639 +/- 0.302 / 4.668 +/- 0.329 | 3.455 +/- 0.754 / 2.516 +/- 0.549 | 1575.1 (10/10) | 29.644 +/- 1.790 | 12.166 +/- 1.122 / 10.863 +/- 0.979 | 31.371 +/- 2.305 |
| variable_k_gp_filter_shuffled | 7.500 +/- 0.527 / 6.000 +/- 0.000 | 5.534 +/- 0.374 / 4.727 +/- 0.340 | 3.334 +/- 0.748 / 2.588 +/- 0.648 | 1322.2 (10/10) | 31.784 +/- 2.243 | 12.573 +/- 0.375 / 10.481 +/- 0.631 | 31.142 +/- 2.372 |

## Paired differences (left - right)

For AUC and coverage, positive favors GP. For capped T6, negative favors GP.

| left | right | raw AUC [95% CI] | canonical AUC [95% CI] | postwarm canonical AUC [95% CI] | capped T6 [95% CI] |
|---|---|---:|---:|---:|---:|
| variable_k_gp_filter | variable_k_random | +0.129 [-0.223, +0.519] | +0.260 [-0.009, +0.595] | +0.307 [-0.012, +0.701] | -324.100 [-719.400, +29.002] |
| variable_k_gp_filter | variable_k_gp_filter_shuffled | +0.105 [-0.077, +0.277] | -0.059 [-0.268, +0.136] | -0.073 [-0.318, +0.158] | +252.900 [-75.500, +619.800] |

## Per-seed AUC and T6

| seed | selector | raw/canonical | raw/canonical AUC | postwarm canonical AUC | T6 |
|---:|---|---|---|---:|---:|
| 703 | variable_k_random | 8/6 | 5.638/4.260 | 2.792 | 2046 |
| 703 | variable_k_gp_filter | 8/6 | 5.430/4.676 | 3.287 | 1420 |
| 703 | variable_k_gp_filter_shuffled | 8/6 | 5.448/4.621 | 3.221 | 1394 |
| 704 | variable_k_random | 8/6 | 5.716/4.394 | 1.986 | 2353 |
| 704 | variable_k_gp_filter | 7/6 | 5.313/4.487 | 2.100 | 2273 |
| 704 | variable_k_gp_filter_shuffled | 7/6 | 5.787/4.960 | 2.681 | 879 |
| 705 | variable_k_random | 8/6 | 5.922/4.482 | 2.007 | 1765 |
| 705 | variable_k_gp_filter | 8/6 | 5.870/4.538 | 2.072 | 1825 |
| 705 | variable_k_gp_filter_shuffled | 7/6 | 5.812/4.907 | 2.505 | 1131 |
| 706 | variable_k_random | 8/6 | 5.921/4.947 | 2.553 | 929 |
| 706 | variable_k_gp_filter | 8/6 | 5.545/4.481 | 1.996 | 1511 |
| 706 | variable_k_gp_filter_shuffled | 7/6 | 5.072/4.293 | 1.773 | 1956 |
| 707 | variable_k_random | 8/6 | 5.137/4.297 | 2.673 | 2140 |
| 707 | variable_k_gp_filter | 8/6 | 5.196/4.267 | 2.640 | 2336 |
| 707 | variable_k_gp_filter_shuffled | 7/6 | 4.694/3.980 | 2.322 | 2082 |
| 708 | variable_k_random | 7/6 | 5.176/4.574 | 1.994 | 2668 |
| 708 | variable_k_gp_filter | 7/6 | 5.858/5.158 | 2.679 | 1061 |
| 708 | variable_k_gp_filter_shuffled | 8/6 | 5.774/4.749 | 2.199 | 1337 |
| 709 | variable_k_random | 8/6 | 4.309/2.746 | 0.977 | 2980 |
| 709 | variable_k_gp_filter | 8/6 | 5.731/4.254 | 2.750 | 1957 |
| 709 | variable_k_gp_filter_shuffled | 8/6 | 5.805/4.896 | 3.506 | 1024 |
| 710 | variable_k_random | 8/6 | 6.138/4.836 | 3.318 | 1404 |
| 710 | variable_k_gp_filter | 8/6 | 5.407/4.971 | 3.471 | 919 |
| 710 | variable_k_gp_filter_shuffled | 8/6 | 5.528/4.990 | 3.492 | 1021 |
| 711 | variable_k_random | 8/6 | 5.667/4.573 | 2.133 | 1574 |
| 711 | variable_k_gp_filter | 8/6 | 6.097/4.710 | 2.295 | 1600 |
| 711 | variable_k_gp_filter_shuffled | 7/6 | 5.710/4.878 | 2.493 | 1276 |
| 712 | variable_k_random | 7/6 | 5.478/4.971 | 1.653 | 1133 |
| 712 | variable_k_gp_filter | 8/6 | 5.944/5.140 | 1.865 | 849 |
| 712 | variable_k_gp_filter_shuffled | 8/6 | 5.714/5.001 | 1.691 | 1122 |

## Decision

The held-out result confirms that variable-length persistent transport is
highly effective, but it does not establish a benefit from the current online
GP score.

- All 30 runs reached the canonical six within 3000 mechanics steps.
- GP filter minus random improved canonical AUC by 0.260, but its paired
  bootstrap interval was `[-0.009, 0.595]` and the exact sign-flip p-value was
  0.123.  Capped T6 improved by 324 steps with interval `[-719, +29]`.
- GP filter minus score-shuffled filter had canonical AUC -0.059
  `[-0.268, 0.136]` and capped T6 +253 steps `[-76, +620]`.  Thus the learned
  score did not outperform its causal negative control.
- GP filter executed 1.77 fewer post-warm-up macros per 1000 mechanics steps
  than random.  Its longer selected actions reduce the number of online
  decisions, consistent with the offline cost observation.

The positive comparison with random is therefore insufficient for the claim
that GP learning caused the improvement.  Under the preregistered rule, the
current acquisition is a no-go for a GP-superiority claim.

## Relation to the mechanistic and offline tests

The no-warm-up mechanistic ablation was decisive.  With seeds 700--702,
`K=1` obtained canonical coverages `[2,2,1]`, whereas uniform random
`K=1..24` reached all six in every run at steps `[1597,1517,1686]`.  The main
algorithmic gain is therefore the persistent variable-K action space, not GP.

The paired offline panel nevertheless showed a real learnable K signal when
`(parent,v1)` was held fixed.  K changed the terminal cell in 12/18 groups;
basis-free GP top-1 new-cell success was 12/18 versus a uniform expectation of
0.338 and shuffled-score success of 7/18.  However, its selected rollout cost
was 40.94 mechanics steps versus 33.05 for uniform selection.

Offline and online results are compatible.  Offline GP ranks K only within
one fixed `(parent,v1)` group.  The online filter compares scores globally
across different parents, first directions, and K values while the archive
changes.  Cross-group GP calibration, nonstationary novelty, and the higher
cost of favored long-K actions can erase the useful within-group K ranking.
Moreover, reaching six in every held-out run creates a strong final-coverage
ceiling.

The next clean experiment should use a hierarchical selector.  Sample the
same `(parent,v1)` groups in both methods; let the baseline choose K uniformly,
and let GP rank K only within each group.  Preserve a random bypass and score
K by predicted new-cell probability per expected mechanics cost.  This tests
exactly the conditional K-ranking ability supported by the offline panel,
without requiring unverified score calibration across unrelated states.  Only
after that succeeds should a separate model be asked to rank perturbation
positions and first directions.
