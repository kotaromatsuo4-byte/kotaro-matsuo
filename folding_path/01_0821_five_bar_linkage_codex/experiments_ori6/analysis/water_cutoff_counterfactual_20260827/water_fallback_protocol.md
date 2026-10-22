# Water D=1 cutoff-0.12 fallback protocol

Status: **drafted, tested, not authorized, not launched**.

The fallback is conditional on a normal, integrity-passing formal rejection of
water1 at episode 4096 in the active `P0X2RCS05ZR0R1` experiment. It refuses to
authorize for a missing gate, a passing gate, an integrity failure, an incomplete
three-seed cohort, or any change to the bound R1 manifest/runner/analyzer/core.

## Configuration

- Candidate: `ori6_RC_spectral_subspace_zero_reject_reward`
- Only causal change from the active water configuration:
  `near_miss_spread_error_far = 0.12` instead of `0.40`
- Fresh seeds: 4301, 4302, 4303
- Fresh root:
  `experiments_ori6/staged/zero_reject_water_fallback/P0X2RCS05ZR0WF12`
- Screen checkpoints: 512, 1024, 2048, 3072, 4096
- Formal promotion: at least two of three seeds pass the unchanged water
  target-plane persistence gate at 4096
- Post-4096: only the best promising seed continues as a production diagnostic
  through 10000; two consecutive collapse warnings stop it
- Safety stop: 2026-08-27 08:25 JST at the last atomic checkpoint

## Commands

Before the formal R1 rejection, the only permitted command is read-only draft:

```powershell
python run_ori_codex6_zero_reject_water_fallback.py --draft
```

After and only after the formal rejection, create the immutable authorization
manifest (still no training):

```powershell
python run_ori_codex6_zero_reject_water_fallback.py --authorize-manifest
```

The authorization output prints the exact separate `--execute-manifest` command.
Launching remains a distinct operator decision. Never reuse or delete a previous
fallback root.

## Evidence for cutoff 0.12

For the current common episode-2048 prefix, the observed boundary band
`eD in [0.114, 0.118]` receives mean `qD` about 0.05 at cutoff 0.12, about 0.335
at cutoff 0.15, and about 0.810 at cutoff 0.40. Cutoff 0.10 removes the boundary
credit entirely but also drives the useful `0.05 < eD < 0.10` approach signal to
zero at the constructive gate. The saved strict-success endpoint receives
`qD = 1` under every candidate cutoff.

These values are recorded-trajectory terminal-reward counterfactuals, not a
claim about a retrained policy.
