# main9 terminal-cell model evaluation

`main9_forest_evaluation.py` answers two separate questions.

1. How many independent `(q,v,K)` macros are needed before the terminal-cell
   model is accurate enough to be useful?
2. Does the conservative filter reject known-cell repeats without also
   rejecting candidates that emulate an unknown cell?

The second question is more important than ordinary classification accuracy.
A classifier can have high accuracy and still be a harmful search filter.

## Corpus schema

Each `seed_*/training_data.npz` needs:

- `inputs`: `(rows, features)` float array
- `cell_labels`: one scalar terminal-cell label per row

The following fields are strongly recommended:

- `macro_ids` (or `rollout_ids`): independent expensive rollout id
- `sample_weights`: non-negative row weights
- `macro_start_step` (or legacy `start_step`): macro start step, used to establish chronology
- `is_hindsight`: true for residual/hindsight rows

All rows from a macro must have one terminal-cell label. The evaluator always
places the complete macro in either train or test, never both. If `macro_ids`
is absent, every row is conservatively treated as an independent macro; this
fallback cannot protect against hindsight-row leakage.

main9 writes `cell_labels == -1` for corrector failures, timeouts, global
budget censoring, and ambiguous terminal descriptors. The evaluator counts
and removes those complete macros before splitting; `-1` is not a cell class.

## Offline learning curve

```powershell
python main9_forest_evaluation.py result/main9_random_corpus `
  --training-sizes 16,32,64,128,256,512,1024 `
  --holdout-macros 128 `
  --model-kinds random_forest extra_trees knn prevalence `
  --output-directory result/main9_random_corpus/model_evaluation
```

The last 128 macros are the fixed test set for every training size. Training
sets are nested chronological prefixes. To emulate drawing a random training
subset from a pre-generated tree, add `--train-order random`; one seeded
permutation is reused for all prefix sizes, so the sets remain nested.

Reported metrics include known-cell accuracy, macro F1, NLL, Brier score,
known-duplicate rejection precision/rate, naturally unseen-cell retention,
leave-one-cell-out (LOCO) pseudo-unknown retention, and terminal-cell novelty
enrichment among candidates that survive the filter.

The reported “earliest useful-filter size” is the first size meeting all
configured operational gates. Defaults are at least 0.90 rejection precision,
0.95 unknown retention, 0.05 duplicate rejection, and success in at least half
the seeds. These gates are diagnostics, not statistical-significance claims.

## Online search aggregation

```powershell
python main9_forest_evaluation.py result/main9_random_corpus `
  --online-results result/main9_online_random result/main9_online_forest `
  --online-baseline-selector variable_k_random `
  --output-directory result/main9_joint_evaluation
```

The online reader intentionally does not freeze a main9 summary schema. It
collects every scalar root-level field in `seed_*/summary.json`, aggregates it
by `selector`, and computes paired per-seed differences against the requested
baseline. This produces `online_seed_metrics.csv`,
`online_aggregate_metrics.csv`, and `online_paired_metrics.csv` in addition to
the offline outputs.

## Synthetic smoke test

```powershell
python main9_forest_evaluation.py --synthetic-smoke `
  --synthetic-seeds 2 --synthetic-macros 96 `
  --training-sizes 8,16,32 `
  --holdout-macros 24 --n-estimators 8 `
  --output-directory result/main9_forest_eval_smoke
```

This creates grouped nonlinear toy corpora, including one cell that first
appears in the fixed holdout. It is only an integration test; it says nothing
about learnability of the five-bar mechanics.
