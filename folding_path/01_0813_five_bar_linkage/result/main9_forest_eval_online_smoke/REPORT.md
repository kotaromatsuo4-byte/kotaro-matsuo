# main9 terminal-cell filter evaluation

All learning-curve points use the same final chronological holdout. Rows from one macro are never divided between train and test.

## Offline learning curve

| model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| knn | 8 | 1 | 1.000 | 1.000 | 0.310 | 0.113 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| prevalence | 8 | 1 | 0.200 | 0.067 | 1.594 | 0.794 | -- | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Earliest useful-filter size

- `knn`: not demonstrated
- `prevalence`: not demonstrated

A point is called useful only when its rejection precision, unknown-cell retention, duplicate rejection rate, and useful-seed fraction pass the configured thresholds. This is an operational gate, not a claim of statistical significance.

## Expected NPZ schema

- Required: `inputs`, `cell_labels`.
- Recommended: `macro_ids`, `sample_weights`, `start_step`, `is_hindsight`.
- Every row with the same `macro_ids` value must have the same terminal-cell label.
- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.

Configured gate: reject precision >= 0.900, unknown retention >= 0.950, duplicate rejection >= 0.050, useful-seed fraction >= 0.500.

## Online result summaries

| selector | seeds | elite_cells_mean | discovery_auc_mean | used_steps_mean | wall_time_seconds_mean |
|---|---:|---:|---:|---:|---:|
| gp_ucb | 1 | 1.000 | 0.212 | 80.000 | 0.846 |
