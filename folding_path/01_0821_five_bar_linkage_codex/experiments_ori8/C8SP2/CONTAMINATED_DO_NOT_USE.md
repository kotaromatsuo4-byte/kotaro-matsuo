# C8SP2 is not a scientific comparison authority

This campaign was stopped at 2026-08-28 01:42:17 JST.

At the 512-episode torus checkpoint, the frozen analyzer incorrectly required
`ori6_zero_reject_total_reward` (ordinary rejected proposals only) to equal
`ori6_zero_reject_timeout_reward` (the separate horizon-boundary timeout
reward). Valid horizon-crossing episodes therefore produced false integrity
failures. The runner stopped all nine torus jobs at 512 episodes and launched
the nine water jobs. The supervisor and water jobs were then terminated before
their first atomic checkpoint.

The underlying torus histories have valid reward decompositions, but the
campaign's stopping and task-transition decisions are invalid. Preserve this
directory only as forensic evidence. Do not use it for arm selection,
learning-curve comparison, or random-baseline comparison.

Authorized manifest fingerprint:
`56a0da899663735a1be5768c56c4e6a8f0138389492992a51d71516909fd3620`.
