# Torus D2 spectral-alpha first-success screen

## Conditions

- Spectral alpha: 0.05, 0.25, 0.5
- Paired seeds: 8101--8105
- Stop: first strict success; otherwise right-censor at 10,000 episodes
- Maximum 200 steps per episode; maximum step width 0.2
- Batch size 8; actor learning rate 0.001; critic learning rate 0.0003
- Actor/entropy reduction: sum divided by batch episodes times reference horizon 200
- Cross: reject as a zero-reward no-op and continue the episode
- The discovery-phase learning rate is fixed. Post-success adaptation is not reached because the run stops at first success.

## Seed-level results

`--` means no strict success by 10,000 episodes. For a censored run, the values in parentheses are effort at the cap.

| alpha | seed | first-success episode | NN updates before success | cumulative steps |
|---:|---:|---:|---:|---:|
| 0.05 | 8101 | 2,632 | 328 | 526,397 |
| 0.05 | 8102 | 1,569 | 196 | 313,798 |
| 0.05 | 8103 | 1,446 | 180 | 289,199 |
| 0.05 | 8104 | -- (10,000) | -- (1,250) | -- (2,000,000) |
| 0.05 | 8105 | 1,259 | 157 | 251,788 |
| 0.25 | 8101 | 2,619 | 327 | 523,796 |
| 0.25 | 8102 | 3,105 | 388 | 620,999 |
| 0.25 | 8103 | 853 | 106 | 170,599 |
| 0.25 | 8104 | 1,322 | 165 | 264,398 |
| 0.25 | 8105 | -- (10,000) | -- (1,250) | -- (2,000,000) |
| 0.5 | 8101 | 1,659 | 207 | 331,785 |
| 0.5 | 8102 | 1,126 | 140 | 225,185 |
| 0.5 | 8103 | -- (10,000) | -- (1,250) | -- (2,000,000) |
| 0.5 | 8104 | -- (10,000) | -- (1,250) | -- (2,000,000) |
| 0.5 | 8105 | -- (10,000) | -- (1,250) | -- (2,000,000) |

## Alpha-level comparison

The conditional medians use successful seeds only and are selection-biased when success rates differ. The restricted means use the actual stopping effort, treating a no-hit run at its 10,000-episode cap.

| alpha | success | conditional median episode | conditional median updates | conditional median steps | restricted mean episode | restricted mean updates | restricted mean steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 4/5 | 1,507.5 | 188 | 301,498.5 | 3,381.2 | 422.2 | 676,236.4 |
| 0.25 | 4/5 | 1,970.5 | 246 | 394,097 | 3,579.8 | 447.2 | 715,958.4 |
| 0.5 | 2/5 | 1,392.5 | 173.5 | 278,485 | 6,557.0 | 819.4 | 1,311,394.0 |

## Interpretation

Alpha 0.5 is not necessary for first discovery: both 0.05 and 0.25 succeeded in four of five paired seeds. Alpha 0.5 produced two fast hits but three long censored runs, so its successful-run median alone is misleading. At the common 10,000-episode cap, alpha 0.05 has the smallest restricted mean effort and uses about half the steps of alpha 0.5. For a task-independent common setting, alpha 0.05 is the most defensible candidate from this screen. This screen is hyperparameter selection evidence; final performance should be evaluated on different seeds.
