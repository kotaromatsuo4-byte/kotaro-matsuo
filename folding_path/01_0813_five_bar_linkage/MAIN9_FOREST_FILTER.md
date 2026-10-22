# main9: collision-continuing forest-filter search

`main9.py` compares selection pressure while keeping the physical macro-action
grammar fixed.  A candidate is

\[
  x=(q, q_{\mathrm{context}}, v_1, K),\qquad K\in\{1,\ldots,24\}.
\]

`v1` is a physical tangent displacement.  After each correction, the previous
physical displacement is projected into the new tangent space and rescaled to
its previous norm.  This is repeated at most `K` times.  A deterministic suffix
then continues independently of `K` until flatness, numerical failure, or the
global mechanics budget.  `FOLDING_MAIN9_MAX_SUFFIX_STEPS=0` is the default and
means no local suffix timeout; a positive value enables an explicit diagnostic
cap.  The saved v1 experiments used the earlier value 100 and report their rare
timeouts as censored outcomes.

## Stop and collision contract

- `flat`: completed terminal outcome and the only outcome assigned a cell.
- `corrector_failure`: numerical failure; no following edge is taken.
- `timeout`: the independent deterministic-suffix limit was exhausted.
- `global_budget`: the experiment-wide mechanics budget censored the macro.
- A projection failure ends only the persistent chain.  The deterministic
  suffix still starts from the last corrected state; the failure is retained
  separately in CSV/NPZ diagnostics.
- Collision never terminates a rollout.  Every colliding edge is counted in an
  external root-to-state `PathQuality` record.

The dual archive has one stable kinematic cell index.  Every valid flat path
may update its `kinematic_elite`; only a collision-free root-to-terminal path
may update its `feasible_elite`.  Candidate parents are restricted to states
with collision-free ancestry.  States after a collision may therefore finish
the endpoint identification, but cannot seed a later macro.

## Selectors and fair controls

At each decision, the shared proposal first samples up to 16 collision-free,
nonterminal parents uniformly without replacement, samples two isotropic
physical tangent directions per parent, and crosses every direction with
`K=1..24` (768 candidates for a full five-bar pool). Thus `random` is uniform
over this bounded random proposal, not a materialized cross product over every
state ever stored in the tree.

- `random`: uniform over the complete `(parent,v1,K)` pool.
- `forest_filter`: reject only candidates for which the forest gives strong,
  in-support evidence of returning to a cell that already has a feasible
  elite; sample uniformly from the remainder.
- `forest_filter_shuffled`: apply a permutation to the final reject mask.  It
  has exactly the same rejection count, but destroys correspondence between a
  candidate and the learned decision.

`random` is the current default because neither the conservative nor the
aggressive learned filter showed a held-out advantage. Set
`FOLDING_MAIN9_SELECTOR` explicitly to enable an experimental filter.

The forest filter has three safety mechanisms: an epsilon-random bypass
(default 20%), no filtering with fewer than two learned classes, and a maximum
reject fraction (default 50%).  When the raw mask is larger than the cap, only
the highest-confidence duplicate decisions are kept.  If nothing remains for
any reason, selection falls back to the full pool.

The default baseline has no warmup.  For a paired audit, set
`FOLDING_MAIN9_PAIRED_WARMUP_GROUPS` above zero.  Each group evaluates the same
`(parent,v1)` at anchor values `K={1,2,4,8,16,24}`.  Both repeated branches and
the deterministic suffixes are charged to the mechanics budget.  The summary
stores hashes of the complete tree/archive prefix and training rows at the end
of this common warmup.

## Hindsight teacher contract

One completed/censored macro is one independent `macro_id`.  Its executed
persistent prefixes provide residual examples of the same endpoint.  All rows
share the source `macro_id`, and their `sample_weights` sum to one.  The forest
bootstraps groups rather than rows, preventing a long K from becoming K
independent simulations.

`requested_k` and `executed_k` describe the residual prefix action used in the
corresponding input.  `source_requested_k` and `source_executed_k` retain the
original macro values.  `cell_labels=-1` denotes an invalid, non-flat,
ambiguous, or globally censored target and is excluded from classifier fitting.
Flat collision paths retain a kinematic cell label so the model can distinguish
that basin, but that label becomes rejectable only after a feasible elite for
the same cell exists.

The last-nonflat context is intentionally retained in the input. Removing it
to obtain literal `(q,v,K)` did not improve accuracy at the 12--16 macro T6
regime and reduced group-kNN leave-one-cell-out retention from 95.8% to 91.7%
at 320 macros. See `MAIN9_FEATURE_ABLATION.md`.

## Run examples

PowerShell baseline:

```powershell
$env:FOLDING_SEED='700'
$env:FOLDING_STEP_BUDGET='3000'
$env:FOLDING_MAIN9_SELECTOR='random'
$env:FOLDING_RESULT_NAME='main9_random_b3000'
python main9.py
```

Matched learned and shuffled controls use the same settings except:

```powershell
$env:FOLDING_MAIN9_SELECTOR='forest_filter'
$env:FOLDING_RESULT_NAME='main9_forest_filter_b3000'
python main9.py

$env:FOLDING_MAIN9_SELECTOR='forest_filter_shuffled'
$env:FOLDING_RESULT_NAME='main9_forest_filter_shuffled_b3000'
python main9.py
```

Useful variables are `FOLDING_MAIN9_K_VALUES` (comma separated),
`FOLDING_MAIN9_PARENT_LIMIT`, `FOLDING_MAIN9_DIRECTIONS`,
`FOLDING_MAIN9_MAX_SUFFIX_STEPS`, `FOLDING_MAIN9_EPSILON`,
`FOLDING_MAIN9_MAX_REJECT_FRACTION`, and
`FOLDING_MAIN9_PAIRED_WARMUP_GROUPS`.

## Artifacts

Each `result/<result_name>/seed_<seed>/` contains:

- `summary.json` and `configuration.json`;
- `rollout_diagnostics.csv`, including start/end budget, requested/executed K,
  suffix length, projection/collision/path quality, selection code, pool and
  reject counts, cell index, and feasible discovery flag;
- `training_data.npz`, including chronological steps, group IDs, weights,
  stop reasons, source/residual K, path quality, and hindsight flags;
- `cell_archive.json` and `discovery_events.json`, sufficient to recover raw or
  canonical coverage and first-discovery step;
- separate `kinematic/` and `feasible/` elite paths and discovery curves.

The forest is intentionally a conservative duplicate filter, not a claim that
online terminal-cell prediction is globally accurate.  A valid comparison is
therefore learned filter versus both full random and its own count-matched
shuffled mask, reported over held-out seeds with mechanics-step budget fixed.
