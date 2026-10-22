# main7: joint K=2 macro GP

## Purpose

`main6.py` chooses only the second perturbation angle after the first
corrector step has already been executed. `main7.py` instead constructs and
scores complete unexecuted macro candidates

\[
  x=(s_i,\,v_1,\,w,\,\theta)
\]

before spending either perturbation step. It can therefore put selection
pressure on the parent state, the first physical perturbation, and the second
angle jointly.

## Basis-invariant input

For every macro, the GP input is

\[
  \left[
  \frac{q_i}{L},
  \frac{q_i^{\mathrm{ctx}}}{L},
  \frac{v_1}{L},
  \frac{w}{L},
  \cos\theta,\sin\theta,m_w
  \right].
\]

Here `q_ctx` is the latest ancestor above `layer_threshold`, and `m_w`
indicates whether an independent physical reference exists. Fixed degrees of
freedom are removed, and z coordinates are measured from `reference_point`.
SVD coefficients are not saved. Hence basis signs and rotations do not alter
the representation. After selection, the code deterministically computes
`q1`, transports `v1` and `w` to obtain `(p,u)`, forms

\[
  v_2=\rho(\cos\theta\,p+\sin\theta\,u),
\]

and performs the second corrector and deterministic suffix.

For tangent dimension larger than two, one random physical orthogonal `w` is
used per first direction. Thus each group searches one random 2-D slice of
the higher-dimensional tangent space. The summary reports whether this
approximation was used.

## Compared selectors

- `independent_random`: uniform over the complete common candidate pool.
- `fixed_transport`: random parent/first direction; theta=0 with probability
  0.8 and a uniform whole-pool candidate with probability 0.2.
- `joint_macro_gp`: novelty UCB, sampled from the top 25%, with 20% uniform
  exploration.
- `joint_macro_gp_shuffled`: the same GP scores permuted among candidates.
- `joint_macro_gp_cost_aware`: discovery score divided by predicted remaining
  mechanics cost.
- `joint_macro_gp_cost_aware_shuffled`: shuffled cost-aware negative control.

The endpoint surrogate has a flat/collision classifier, a continuous
flatness GP, and a flat-conditional terminal-descriptor GP. Exactly one
endpoint row is saved per terminal macro decision. This includes a decision
whose first perturbation already reaches flat/collision: that is an observable
outcome of `(parent,v1,w,theta)` and must not be hidden from the selector.
`q2_executed` is saved separately for auditing. The cost target is
`log(1 + C)`, where `C` counts q1, q2, and every deterministic suffix step.
Every mechanics step is also charged to the global step budget.

## Fair warmup

The default warmup evaluates theta=0, pi/2, pi, and -pi/2 for the same
`(parent,v1,w)` group. Repeating q1 in each counterfactual branch is charged
to the budget. With one seed, all selectors must have identical
`warmup_end_step`, `warmup_end_cells`, and `warmup_prefix_sha256`. The smoke
test at seed 980, budget 450, one paired group produced the common values

```text
warmup_end_step       = 242
warmup_end_cells      = 2
warmup_prefix_sha256  = f9a5719f08444e6f326def3c76a1676f2a80047cf75b7daa94590672e3549f7e
```

for all six selectors.

## Running

PowerShell example:

```powershell
$env:FOLDING_MAIN7_SELECTOR = "joint_macro_gp"
$env:FOLDING_SEED = "600"
$env:FOLDING_STEP_BUDGET = "10000"
python main7.py
```

Main7-specific controls are `FOLDING_MAIN7_NUM_ANGLES` (16),
`FOLDING_MAIN7_PARENT_LIMIT` (16), `FOLDING_MAIN7_FIRST_DIRECTIONS` (2),
`FOLDING_MAIN7_PAIRED_WARMUP_GROUPS` (4), `FOLDING_MAIN7_EPSILON` (0.20),
`FOLDING_MAIN7_TOP_FRACTION` (0.25), and
`FOLDING_MAIN7_REFIT_INTERVAL` (3). Exact GP fitting remains cubic in the
number of completed macros, so long experiments may eventually require a
sparse/local GP; main7 deliberately retains every completed macro in the
current research implementation.

Outputs are written to `result/<result_name>/seed_XXX/`. In addition to the
elite paths and discovery curve, `rollout_diagnostics.csv` records the chosen
parent, theta, candidate-pool size, endpoint, acquisition components, and
predicted cost. `training_data.npz` stores endpoint and cost teachers.
