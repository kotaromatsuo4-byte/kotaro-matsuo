# ori5 uniform-random wall-time calibration

`calibrate_ori5_uniform_walltime.py` converts the audited uniform-random
first-discovery RMST in mechanics steps to a current-machine wall-time
estimate. It does not train a network and does not write into any training or
first-discovery result directory.

## Fixed protocol

The stored protocol `ori5_uniform_walltime_protocol.json` fixes:

- the original random-baseline seeds (30 seeds for each five-bar target and
  10 seeds for each 3-D target);
- one 4,096-mechanics-step measured prefix per seed;
- a separate 512-step warm-up per task, excluded from timing;
- one foreground process and one numerical-library thread;
- current ori5 reward/mechanics and the current planar or continuous 3-D
  collision backend;
- 20,000 seed-cluster bootstrap replicates.

Strict success does not stop calibration: a new random episode starts and the
stored prefix continues until its exact mechanics budget is consumed. This
keeps every timing block the same size and avoids timing-dependent censoring.

## Commands

Validate the complete protocol, hashes, models, and baseline compatibility
without running any rollout:

```powershell
python experiments_ori5/analysis/tools/calibrate_ori5_uniform_walltime.py --dry-run
```

Run the complete calibration in a single foreground process:

```powershell
python experiments_ori5/analysis/tools/calibrate_ori5_uniform_walltime.py `
  --output-dir experiments_ori5/analysis/walltime_calibration/final_single_process
```

Run only stored tasks (seeds and prefix lengths cannot be overridden):

```powershell
python experiments_ori5/analysis/tools/calibrate_ori5_uniform_walltime.py `
  --tasks torus2 water1 watersqrt2 `
  --output-dir experiments_ori5/analysis/walltime_calibration/spatial_only
```

Use a new, empty output directory for every run. For a defensible comparison,
run this after parallel training jobs have stopped and avoid other CPU-heavy
work during calibration.

## Expected runtime and outputs

The complete protocol measures 368,640 rollout mechanics steps in 90 timing
blocks, plus 2,560 excluded warm-up steps. Historical uniform throughput on
this machine suggests roughly 10--15 minutes when otherwise idle; heavy CPU or
OneDrive activity can make it materially longer. This is a planning estimate,
not a calibration result.

Each completed run writes:

- `calibration_results.json`: complete results, bootstrap intervals,
  provenance, limitations, source/model hashes, and trace hashes;
- `timing_blocks.csv`: one row per fixed seed prefix;
- `task_summary.csv`: compact throughput and converted-RMST table;
- `REPORT.md`: human-readable result table;
- `run_config.json`: immutable protocol snapshot and run fingerprint.

The main interval jointly resamples timing blocks and the stored
common-censor random first-hit trials. The reported wall RMST is explicitly a
`calibrated_uniform_random_wall_equivalent`. It is not directly measured RL
wall time: policy-network inference, backpropagation, optimizer updates,
evaluation, and checkpoint overhead are excluded.

## Tiny verification

The test suite runs only nine warm-up/measured mechanics steps across the
planar, torus, and waterbomb adapters:

```powershell
python -m unittest -v test_ori_codex5_walltime_calibration.py
```
