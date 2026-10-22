# Five D2/D3 mixed learning: extension to 1,024 episodes

## Conclusion

Same-seed extension did not recover goal-dependent folding. The final frozen policy succeeded in 0/100 D2 trials and 100/100 D3 trials. All 100 D2 trials terminated as wrong-flat at D approximately 3. This is a failure of the current seed/settings to learn both goals, not evidence that goal-conditioned learning or generalization is impossible.

## Conditions

- Training seed: 9201. Continued from episode 128 of `mixed_goals_seed9201_20260910_213428_669282` without resetting model, Adam optimizer states, Torch action RNG, or goal-schedule RNG.
- Total training: 1,024 episodes, 512 per goal; 128 optimizer updates; 146,770 physical steps.
- Each batch contains four D2 and four D3 episodes, shuffled. Actor LR 0.0005, critic LR 0.00015; fixed throughout this run. Spectral weight 0.05, cross reject without penalty, horizon 400, max width 0.1. All numerical settings inherited from the pilot.
- Evaluation: frozen stochastic policy, 100 rollouts per goal per checkpoint, evaluation seeds 950000–950099 reused across checkpoints/goals. Evaluation never changes training state. Initial-policy and random baselines are reused from the pilot.
- Strict success: F <= 0.001 and spread error <= 0.05. Near-miss reach: at least one visited post-step state with F <= 0.1 and spread error <= 0.05, including success.

## Frozen-policy evaluation

| Mixed episodes | NN updates | D2 strict | D3 strict | D2 near-miss | D3 near-miss | D2 median Q* | D3 median Q* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0/100 | 0/100 | 0/100 | 2/100 | 0.06767 | 0.02811 |
| 128 | 16 | 0/100 | 97/100 | 0/100 | 100/100 | 0.07825 | 1.00000 |
| 256 | 32 | 0/100 | 100/100 | 0/100 | 100/100 | 0.07331 | 1.00000 |
| 512 | 64 | 0/100 | 100/100 | 0/100 | 100/100 | 0.06831 | 1.00000 |
| 768 | 96 | 0/100 | 100/100 | 0/100 | 100/100 | 0.07015 | 1.00000 |
| 1024 | 128 | 0/100 | 100/100 | 0/100 | 100/100 | 0.08224 | 1.00000 |

Final median terminal D: D2 input 2.999434; D3 input 2.999321. Final mean episode steps: D2 84.07 (wrong-flat), D3 81.89 (success). Shorter D2 rollouts are not an improvement: they reach the wrong shape faster.

For 0/100 and 100/100 observations, Wilson 95% intervals are [0, 0.03699] and [0.96301, 1]. The observed rates do not establish exact underlying probabilities of zero or one. This is one training seed; checkpoint evaluations are paired by evaluation seed and are not independent training replications.

## Training discovery versus retained performance

| Goal | First mixed episode | Goal-specific episode count at discovery | Updates before discovery | All-task steps at discovery | Goal-specific steps at discovery | Total training successes |
|---|---:|---:|---:|---:|---:|---:|
| D2 | 95 | 48 | 11 | 35,119 | 17,799 | 1/512 |
| D3 | 28 | 15 | 3 | 11,026 | 5,830 | 478/512 |

The extension produced no additional D2 success. The final 100 training episodes per goal contained zero D2 successes and 100 D3 successes. Both goals become flat, but only D3 satisfies the requested spread. Thus flatness alone conceals the failed conditioning.

## Verification and artifacts

- A short actual-environment split/resume test (16 episodes, horizon 4) exactly matched uninterrupted training in model weights, both optimizers, RNG states, counters, and every episode metric.
- All original numerical-source hashes (main/env/mechanics/policy/agent/RL_main) match the pilot. Only the mixed-run orchestration and documentation were extended.
- The first 128 episode records are identical to the pilot; every batch has a 4:4 goal balance. Copied pilot checkpoints and raw evaluations are byte-identical. The original run was preserved.
- `learning_curve.png`: per-goal reward and terminal-flatness curves, raw values and 20-goal-episode moving averages.
- `five_d2_*.png`, `five_d3_*.png`: checkpoint strict/near-miss probabilities and Q* distributions. Probability bands are Wilson intervals; Q* bands are rollout quartiles. Random reference is dashed.
- `terminal_distribution.png`: final goal-specific terminal F/D scatter, common axes.
- `evaluation_summary.csv`, `evaluation/*.csv`: full numerical results.

Next diagnostic, not performed here: compare D2-only goal-conditioned training and several mixed-training seeds at matched goal-specific experience budgets. This would help separate insufficient D2 learnability under the new input from interference between goals. Architectural or optimization changes require a separate experiment; this extension alone does not identify the causal mechanism.
