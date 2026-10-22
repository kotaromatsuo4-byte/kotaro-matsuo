# Water D=1 near-miss spread cutoff counterfactual

## Conclusion

Recommended fallback cutoff: **0.12**.

0.12 is the smallest cutoff above the 0.10 constructive gate: it preserves a graded target-plane signal for 0.05<eD<0.10, while eD=0.114--0.118 receives only qD=0.029--0.086.  Cutoff 0.15 still gives that boundary attractor qD=0.32--0.36; cutoff 0.10 removes it completely but also drives the whole 0.05--0.10 approach band linearly to zero at the gate.

This is a one-component, recorded-trajectory counterfactual. It is not a retraining result.

## Current zero-reject checkpoint 2048 (three-seed pooled)

| Cohort | cutoff | N | positive quality | mean quality | mean return | positive return rate |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.10 | 6144 | 131 | 0.000387 | -0.441036 | 0.4154 |
| all | 0.12 | 6144 | 4330 | 0.014926 | -0.426497 | 0.4155 |
| all | 0.15 | 6144 | 4330 | 0.119129 | -0.322294 | 0.4168 |
| all | 0.40 | 6144 | 4332 | 0.292801 | -0.148622 | 0.4188 |
| constructive | 0.10 | 43 | 43 | 0.035699 | 0.612492 | 1.0000 |
| constructive | 0.12 | 43 | 43 | 0.072447 | 0.649239 | 1.0000 |
| constructive | 0.15 | 43 | 43 | 0.100007 | 0.676800 | 1.0000 |
| constructive | 0.40 | 43 | 43 | 0.145942 | 0.722734 | 1.0000 |
| boundary_flat_0p10_0p15 | 0.10 | 3579 | 0 | 0.000000 | -0.460720 | 0.5619 |
| boundary_flat_0p10_0p15 | 0.12 | 3579 | 3579 | 0.023303 | -0.437417 | 0.5619 |
| boundary_flat_0p10_0p15 | 0.15 | 3579 | 3579 | 0.199607 | -0.261113 | 0.5619 |
| boundary_flat_0p10_0p15 | 0.40 | 3579 | 3579 | 0.493446 | 0.032726 | 0.5619 |
