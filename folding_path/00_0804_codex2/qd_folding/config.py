"""Configuration shared by the three search methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    """Numerical, search, and output parameters.

    The bootstrap budget is reported separately.  ``search_step_budget`` counts
    only newly simulated path transitions after the common bootstrap has been
    generated.
    """

    seed: int = 17
    bootstrap_count: int = 3
    initial_imperfection: float = 0.08

    max_path_steps: int = 40
    search_step_budget: int = 800
    flatness_tolerance: float = 0.05
    layer_threshold: float = 0.10
    layer_separation_tolerance: float = 1.0e-5
    overlap_tolerance: float = 0.06
    collision_tolerance: float = 1.0e-8
    corrector_iterations: int = 20
    corrector_tolerance: float = 1.0e-7
    residual_tolerance: float = 1.0e-5
    no_progress_tolerance: float = 1.0e-5
    no_progress_patience: int = 4

    gravity_step_norm: float = 0.35
    local_cma_generations: int = 8
    local_cma_population: int = 6
    local_cma_sigma: float = 0.30

    perturb_max_norm: float = 0.22
    # ``max_interventions`` is kept for the legacy intervention-sequence
    # experiment.  The current state-restart protocol intentionally does not
    # use an intervention-count cut-off; ``max_path_steps`` and the common
    # mechanics-step budget are its only path/experiment limits.
    max_interventions: int = 3

    # Current two-stage protocol.  Candidate pools are refreshed only after
    # this many completed rollouts, and remain fixed inside each epoch.
    candidate_pool_capacity: int = 24
    candidate_epoch_rollouts: int = 8

    ucb_exploration: float = 1.0
    smucb_state_lengthscale: float = 0.25
    smucb_archive_lengthscale: float = 4.0
    smucb_noise_variance: float = 0.10
    smucb_delta: float = 0.10
    smucb_beta_scale: float = 1.0

    # The fields below are retained only so that the pre-nonstationarity
    # experiments can still be reproduced and inspected.
    ucb_reward_step_scale: float = 20.0
    reward_new_cell: float = 1.0
    reward_elite_improvement: float = 0.35
    reward_exploration_gain: float = 0.10

    event_frontier_capacity: int = 160
    event_max_add_per_rollout: int = 8
    event_stride: int = 4
    event_potential_prominence: float = 2.0e-3
    event_basis_change_threshold: float = 0.08
    event_curvature_threshold: float = 0.40
    event_novelty_tolerance: float = 0.030
    event_merge_tolerance: float = 0.012
    event_random_points: int = 2
    event_peak_window: int = 1
    event_novelty_as_trigger: bool = False

    state_bin_width: float = 0.15
    state_archive_max_cells: int = 256
    state_representatives_per_cell: int = 2

    sequence_max_wait: int = 4
    sequence_initial_random: int = 8
    sequence_parent_probability: float = 0.85
    sequence_coefficient_sigma: float = 0.08
    sequence_wait_sigma: float = 2.0
    sequence_insert_probability: float = 0.35
    sequence_delete_probability: float = 0.20
    sequence_fresh_random_probability: float = 0.15

    save_all_histories: bool = False
    output_root: str = "output/nonstationary_protocol"

    def validate(self) -> None:
        if self.bootstrap_count <= 0:
            raise ValueError("bootstrap_count must be positive")
        if self.max_path_steps <= 0 or self.search_step_budget <= 0:
            raise ValueError("step limits must be positive")
        if self.local_cma_generations <= 0 or self.local_cma_population <= 1:
            raise ValueError("local CMA settings must be positive")
        if not 0.0 < self.flatness_tolerance < self.layer_threshold:
            raise ValueError(
                "Require 0 < flatness_tolerance < layer_threshold"
            )
        if self.perturb_max_norm <= 0.0:
            raise ValueError("perturb_max_norm must be positive")
        if self.max_interventions < 1:
            raise ValueError("max_interventions must be at least one")
        if self.candidate_pool_capacity <= 0:
            raise ValueError("candidate_pool_capacity must be positive")
        if self.candidate_epoch_rollouts <= 0:
            raise ValueError("candidate_epoch_rollouts must be positive")
        if self.ucb_exploration < 0.0:
            raise ValueError("ucb_exploration must be non-negative")
        if self.smucb_state_lengthscale <= 0.0:
            raise ValueError("smucb_state_lengthscale must be positive")
        if self.smucb_archive_lengthscale <= 0.0:
            raise ValueError("smucb_archive_lengthscale must be positive")
        if self.smucb_noise_variance <= 0.0:
            raise ValueError("smucb_noise_variance must be positive")
        if not 0.0 < self.smucb_delta < 1.0:
            raise ValueError("smucb_delta must lie strictly between zero and one")
        if self.smucb_beta_scale <= 0.0:
            raise ValueError("smucb_beta_scale must be positive")
        if self.event_frontier_capacity <= 0:
            raise ValueError("event_frontier_capacity must be positive")
        if self.event_merge_tolerance > self.event_novelty_tolerance:
            raise ValueError(
                "event_merge_tolerance must not exceed novelty tolerance"
            )
        if self.state_bin_width <= 0.0:
            raise ValueError("state_bin_width must be positive")
        if self.state_archive_max_cells <= 0:
            raise ValueError("state_archive_max_cells must be positive")
        if self.state_representatives_per_cell <= 0:
            raise ValueError(
                "state_representatives_per_cell must be positive"
            )
        probabilities = {
            "sequence_parent_probability": self.sequence_parent_probability,
            "sequence_insert_probability": self.sequence_insert_probability,
            "sequence_delete_probability": self.sequence_delete_probability,
            "sequence_fresh_random_probability": (
                self.sequence_fresh_random_probability
            ),
        }
        invalid = [
            name
            for name, value in probabilities.items()
            if not 0.0 <= value <= 1.0
        ]
        if invalid:
            raise ValueError(
                "Sequence probabilities must lie in [0, 1]: "
                + ", ".join(invalid)
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise KeyError(f"Unknown configuration keys: {unknown}")
        config = cls(**values)
        config.validate()
        return config

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)
