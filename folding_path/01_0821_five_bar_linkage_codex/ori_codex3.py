"""Compact-target flat-folding with a variable-mode Actor--Critic.

This experiment reuses the mechanics and variable-mode policy in
``ori_codex2`` but adds a compact-flat terminal goal.  The default dense reward
combines bounded flatness improvement, target-spread potential improvement,
and squared target-spread deviation integrated along the actual corrected
nodal path.  Temporary motion away from the target is allowed, but the
distance travelled away from it accumulates a cost.  The default update is the
episode-separated Monte-Carlo Actor--Critic from ``ori_codex2``; clipped PPO
remains available as a fallback experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

import ori_codex2 as base


_BASE_RUN_EPISODE = base.run_episode
_BASE_AGGREGATE_EVALUATIONS = base.aggregate_stochastic_evaluations
_BASE_MAKE_ENVIRONMENT = base.make_environment
_BASE_SAVE_TRAINING_PLOTS = base.save_training_plots
_BASE_POLICY_SELECTION_SCORE = base.policy_selection_score
_BASE_AGENT_CLASS = base.ActorCriticAgent


def projected_spread(node, subspace_dimension, coordinate_scale=1.0):
    """Diameter after projection into the best-fit affine subspace."""
    coordinates = np.asarray(node, dtype=float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    dimension = min(int(subspace_dimension), right_vectors.shape[0])
    projected = centered @ right_vectors[:dimension].T
    differences = projected[:, np.newaxis, :] - projected[np.newaxis, :, :]
    diameter = np.max(np.linalg.norm(differences, axis=2))
    return float(diameter / max(float(coordinate_scale), 1e-12))


def target_deviation_energy(spread, target_spread):
    """Squared target error E=((D-D*)/1)^2 with a finite-value guard."""
    normalised_deviation = float(spread) - float(target_spread)
    if not np.isfinite(normalised_deviation):
        return 1.0e12
    normalised_deviation = float(np.clip(
        normalised_deviation, -1.0e6, 1.0e6
    ))
    return float(normalised_deviation * normalised_deviation)


def integrated_target_deviation_cost(
    spread_before,
    spread_after,
    normalised_path_increment,
    target_spread,
):
    """Trapezoidal line integral of the squared target-spread deviation.

    The path increment is the Frobenius norm of the *corrected* nodal
    displacement divided by the fixed coordinate scale.  Using a line
    integral makes the cost approximately invariant to splitting one physical
    move into several smaller solver steps.
    """
    path_increment = max(float(normalised_path_increment), 0.0)
    before = target_deviation_energy(spread_before, target_spread)
    after = target_deviation_energy(spread_after, target_spread)
    return float(0.5 * path_increment * (before + after))


def target_deviation_reward_components(
    flatness_ratio_before,
    flatness_ratio_after,
    spread_before,
    spread_after,
    normalised_path_increment,
    target_spread,
    flatness_clip,
    flatness_progress_weight,
    target_progress_weight,
    target_deviation_cost_weight,
):
    """Return the exact dense-reward decomposition used by the environment."""
    bounded_before = float(np.clip(
        flatness_ratio_before, 0.0, flatness_clip
    ))
    bounded_after = float(np.clip(
        flatness_ratio_after, 0.0, flatness_clip
    ))
    flatness_reward = float(
        flatness_progress_weight * (bounded_before - bounded_after)
    )
    energy_before = target_deviation_energy(spread_before, target_spread)
    energy_after = target_deviation_energy(spread_after, target_spread)
    target_progress_reward = float(
        target_progress_weight * (energy_before - energy_after)
    )
    path_cost = integrated_target_deviation_cost(
        spread_before,
        spread_after,
        normalised_path_increment,
        target_spread,
    )
    path_cost_reward = float(-target_deviation_cost_weight * path_cost)
    return {
        "bounded_flatness_before": bounded_before,
        "bounded_flatness_after": bounded_after,
        "flatness_progress_reward": flatness_reward,
        "target_deviation_energy_before": energy_before,
        "target_deviation_energy_after": energy_after,
        "target_progress_reward": target_progress_reward,
        "integrated_target_deviation_cost": path_cost,
        "target_deviation_path_reward": path_cost_reward,
        "dense_reward": float(
            flatness_reward + target_progress_reward + path_cost_reward
        ),
    }


class CompactTargetFoldingEnv(base.FixedStartFoldingEnv):
    """Flat-folding environment with compactness-controlled terminal quality."""

    def __init__(
        self,
        *args,
        target_spread=2.0,
        spread_sigma=0.35,
        spread_tolerance=0.05,
        flat_quality_clip=1.0,
        quality_combination="path_integrated_target_deviation",
        flat_quality_weight=0.3,
        flat_signal_floor=0.1,
        compact_potential_weight=1.0,
        flatness_progress_weight=1.0,
        target_progress_weight=1.0,
        target_deviation_cost_weight=1.0,
        shaping_discount=1.0,
        flat_terminal_bonus=0.0,
        joint_success_bonus=10.0,
        wrong_flat_penalty=0.0,
        terminate_wrong_flat=True,
        training_flatness_tolerances=(1e-3,),
        training_flatness_milestones=(),
        curriculum_waypoint_bonus=0.0,
        curriculum_waypoint_start_episode=0,
        gated_compact_cost_weight=1.2,
        compact_gate_scale=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluation_flatness_tolerance = float(self.flatness_tolerance)
        self.target_spread = float(target_spread)
        self.spread_sigma = float(spread_sigma)
        self.spread_tolerance = float(spread_tolerance)
        self.flat_quality_clip = float(flat_quality_clip)
        self.quality_combination = str(quality_combination)
        self.flat_quality_weight = float(flat_quality_weight)
        self.flat_signal_floor = float(flat_signal_floor)
        self.compact_potential_weight = float(compact_potential_weight)
        self.flatness_progress_weight = float(flatness_progress_weight)
        self.target_progress_weight = float(target_progress_weight)
        self.target_deviation_cost_weight = float(
            target_deviation_cost_weight
        )
        self.shaping_discount = float(shaping_discount)
        self.flat_terminal_bonus = float(flat_terminal_bonus)
        self.joint_success_bonus = float(joint_success_bonus)
        self.wrong_flat_penalty = abs(float(wrong_flat_penalty))
        self.terminate_wrong_flat = bool(
            terminate_wrong_flat
            or self.quality_combination
            == "path_integrated_target_deviation"
        )
        self.training_flatness_tolerances = tuple(
            float(value) for value in training_flatness_tolerances
        )
        self.training_flatness_milestones = tuple(
            int(value) for value in training_flatness_milestones
        )
        self.training_episode_count = 0
        self.episode_is_training = False
        self.active_flatness_tolerance = self.evaluation_flatness_tolerance
        self.curriculum_waypoint_bonus = float(curriculum_waypoint_bonus)
        self.curriculum_waypoint_start_episode = int(
            curriculum_waypoint_start_episode
        )
        self.curriculum_waypoint_mode_active = False
        self.curriculum_waypoint_awarded = False
        self.curriculum_waypoint_bonus_count = 0
        self.gated_compact_cost_weight = float(gated_compact_cost_weight)
        self.compact_gate_scale = float(compact_gate_scale)
        if not np.isfinite(self.target_spread) or self.target_spread < 0.0:
            raise ValueError("target_spread must be finite and non-negative.")
        if not np.isfinite(self.spread_sigma) or self.spread_sigma <= 0.0:
            raise ValueError("spread_sigma must be finite and positive.")
        if (
            not np.isfinite(self.spread_tolerance)
            or self.spread_tolerance < 0.0
        ):
            raise ValueError(
                "spread_tolerance must be finite and non-negative."
            )
        if self.flat_quality_clip <= 0.0:
            raise ValueError("flat_quality_clip must be positive.")
        if self.quality_combination not in (
            "weighted_sum", "product", "minimum", "flatness_only",
            "gated_cost", "path_integrated_target_deviation",
        ):
            raise ValueError(
                "quality_combination must be 'weighted_sum', 'product', "
                "'minimum', 'flatness_only', 'gated_cost', or "
                "'path_integrated_target_deviation'."
            )
        if not 0.0 <= self.flat_quality_weight <= 1.0:
            raise ValueError("flat_quality_weight must lie in [0, 1].")
        if not -1.0 <= self.flat_signal_floor <= 1.0:
            raise ValueError("flat_signal_floor must lie in [-1, 1].")
        if not 0.0 <= self.shaping_discount <= 1.0:
            raise ValueError("shaping_discount must lie in [0, 1].")
        if self.compact_potential_weight < 0.0:
            raise ValueError("compact_potential_weight cannot be negative.")
        if self.flatness_progress_weight < 0.0:
            raise ValueError("flatness_progress_weight cannot be negative.")
        if self.target_progress_weight < 0.0:
            raise ValueError("target_progress_weight cannot be negative.")
        if self.target_deviation_cost_weight < 0.0:
            raise ValueError(
                "target_deviation_cost_weight cannot be negative."
            )
        if (
            self.quality_combination == "path_integrated_target_deviation"
            and abs(self.shaping_discount - 1.0) > 1e-12
        ):
            raise ValueError(
                "path_integrated_target_deviation requires gamma=1 so "
                "potential improvements and geometric path cost retain "
                "their intended undiscounted meaning."
            )
        if self.flat_terminal_bonus < 0.0:
            raise ValueError("flat_terminal_bonus cannot be negative.")
        if self.joint_success_bonus < 0.0:
            raise ValueError("joint_success_bonus cannot be negative.")
        if self.curriculum_waypoint_bonus < 0.0:
            raise ValueError("curriculum_waypoint_bonus cannot be negative.")
        if self.curriculum_waypoint_start_episode < 0:
            raise ValueError(
                "curriculum_waypoint_start_episode cannot be negative."
            )
        if self.gated_compact_cost_weight < 0.0:
            raise ValueError("gated_compact_cost_weight cannot be negative.")
        if self.compact_gate_scale <= 0.0:
            raise ValueError("compact_gate_scale must be positive.")
        if len(self.training_flatness_tolerances) != (
            len(self.training_flatness_milestones) + 1
        ):
            raise ValueError(
                "training_flatness_tolerances must contain exactly one more "
                "entry than training_flatness_milestones."
            )
        if any(value <= 0.0 for value in self.training_flatness_tolerances):
            raise ValueError("All training flatness tolerances must be positive.")
        if any(
            value < self.evaluation_flatness_tolerance
            for value in self.training_flatness_tolerances
        ):
            raise ValueError(
                "Training flatness tolerances cannot be stricter than the "
                "evaluation tolerance."
            )
        if any(
            later > earlier
            for earlier, later in zip(
                self.training_flatness_tolerances,
                self.training_flatness_tolerances[1:],
            )
        ):
            raise ValueError(
                "Training flatness tolerances must be non-increasing."
            )
        if any(value < 1 for value in self.training_flatness_milestones):
            raise ValueError("Training flatness milestones must be positive.")
        if tuple(sorted(self.training_flatness_milestones)) != (
            self.training_flatness_milestones
        ):
            raise ValueError("Training flatness milestones must be sorted.")
        self.previous_compact_potential = self.compactness_metrics()[
            "combined_quality"
        ]

    def configure_episode_tolerance(self, training):
        """Select a curriculum tolerance without relaxing evaluation."""
        self.episode_is_training = bool(training)
        if training:
            self.training_episode_count += 1
            phase = sum(
                self.training_episode_count > milestone
                for milestone in self.training_flatness_milestones
            )
            tolerance = self.training_flatness_tolerances[phase]
        else:
            tolerance = self.evaluation_flatness_tolerance
        self.active_flatness_tolerance = float(tolerance)
        self.curriculum_waypoint_mode_active = bool(
            training
            and self.curriculum_waypoint_bonus > 0.0
            and self.training_episode_count
            > self.curriculum_waypoint_start_episode
        )
        self.flatness_tolerance = (
            self.evaluation_flatness_tolerance
            if (not training or self.curriculum_waypoint_mode_active)
            else self.active_flatness_tolerance
        )

    def reset(self):
        state = super().reset()
        self.curriculum_waypoint_awarded = False
        self.curriculum_waypoint_bonus_count = 0
        self.previous_compact_potential = self.compactness_metrics()[
            "combined_quality"
        ]
        return state

    def compactness_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        flatness_potential, flatness_ratio, flatness_rms = (
            self.flatness_metrics(coordinates)
        )
        spread = projected_spread(
            coordinates,
            self.flatness_subspace_dimension,
            self.coordinate_scale,
        )
        spread_deviation = spread - self.target_spread
        spread_error = abs(spread_deviation)
        spread_quality = np.exp(
            -0.5 * (spread_deviation / self.spread_sigma) ** 2
        )
        spread_cost = 1.0 - spread_quality
        compact_gate = np.exp(
            -max(float(flatness_ratio), 0.0) / self.compact_gate_scale
        )
        clipped_flatness = np.clip(
            flatness_ratio, 0.0, self.flat_quality_clip
        )
        flat_quality = 1.0 - clipped_flatness / self.flat_quality_clip
        if self.quality_combination == "weighted_sum":
            combined_quality = (
                self.flat_quality_weight * flat_quality
                + (1.0 - self.flat_quality_weight) * spread_quality
            )
        elif self.quality_combination == "product":
            # A negative floor makes a flat but clearly over-wide state worse
            # than the initial state, while retaining a full flatness signal
            # whenever the projected spread remains close to its target.
            combined_quality = flat_quality * (
                self.flat_signal_floor
                + (1.0 - self.flat_signal_floor) * spread_quality
            )
        elif self.quality_combination == "minimum":
            # Bottleneck quality: improving an already-good component cannot
            # compensate for the other component being poor.
            combined_quality = min(flat_quality, spread_quality)
        elif self.quality_combination == "flatness_only":
            # This is exactly the bounded flatness-progress potential used by
            # ori_codex2, expressed as Phi so that reward = Phi(s') - Phi(s).
            # Compactness then affects only the joint terminal goal.
            combined_quality = -float(np.clip(
                flatness_ratio, 0.0, self.flatness_reward_clip
            ))
        elif self.quality_combination == "path_integrated_target_deviation":
            # This is only an instantaneous diagnostic.  The actual reward
            # below uses flatness and target-error *improvements* plus the
            # non-conservative line integral of target error along the path.
            flatness_cost = float(np.clip(
                flatness_ratio, 0.0, self.flatness_reward_clip
            ))
            combined_quality = -(
                self.flatness_progress_weight * flatness_cost
                + self.target_progress_weight
                * target_deviation_energy(spread, self.target_spread)
            )
        else:
            # Compactness is irrelevant far from flatness, where the initial
            # linkage already happens to have the target projected diameter.
            # It is activated smoothly near the flat manifold.  Expressing the
            # entire objective as one Markov state potential preserves the
            # telescoping, non-monotonic-path property of the original reward.
            flatness_cost = float(np.clip(
                flatness_ratio, 0.0, self.flatness_reward_clip
            ))
            combined_quality = -(
                flatness_cost
                + self.gated_compact_cost_weight
                * compact_gate
                * spread_cost
            )
        return {
            "flatness_potential": float(flatness_potential),
            "flatness_ratio": float(flatness_ratio),
            "flatness_rms": float(flatness_rms),
            "spread": float(spread),
            "spread_deviation": float(spread_deviation),
            "spread_error": float(spread_error),
            "target_deviation_energy": target_deviation_energy(
                spread, self.target_spread
            ),
            "spread_quality": float(spread_quality),
            "spread_cost": float(spread_cost),
            "compact_gate": float(compact_gate),
            "flat_quality": float(flat_quality),
            "combined_quality": float(combined_quality),
        }

    def step(self, physical_direction, requested_width, mode_coefficients):
        metrics_before = self.compactness_metrics()
        potential_before = metrics_before["combined_quality"]
        next_state, _, _, info = super().step(
            physical_direction, requested_width, mode_coefficients
        )
        compactness = self.compactness_metrics()
        potential_after = compactness["combined_quality"]

        bounded_flatness_before = float(np.clip(
            metrics_before["flatness_ratio"],
            0.0,
            self.flatness_reward_clip,
        ))
        bounded_flatness_after = float(np.clip(
            compactness["flatness_ratio"],
            0.0,
            self.flatness_reward_clip,
        ))
        normalised_path_increment = float(
            info["actual_displacement"]
            / max(self.coordinate_scale, 1e-12)
        )
        if not np.isfinite(normalised_path_increment):
            normalised_path_increment = 0.0
        transition_is_valid = bool(
            not info["self_crossing"] and not info["correction_failed"]
        )

        if (
            self.quality_combination == "path_integrated_target_deviation"
            and transition_is_valid
        ):
            reward_components = target_deviation_reward_components(
                metrics_before["flatness_ratio"],
                compactness["flatness_ratio"],
                metrics_before["spread"],
                compactness["spread"],
                normalised_path_increment,
                self.target_spread,
                self.flatness_reward_clip,
                self.flatness_progress_weight,
                self.target_progress_weight,
                self.target_deviation_cost_weight,
            )
            flatness_progress_reward = reward_components[
                "flatness_progress_reward"
            ]
            target_progress_reward = reward_components[
                "target_progress_reward"
            ]
            target_deviation_path_cost = reward_components[
                "integrated_target_deviation_cost"
            ]
            target_deviation_path_reward = reward_components[
                "target_deviation_path_reward"
            ]
            dense_reward = reward_components["dense_reward"]
        elif self.quality_combination == "path_integrated_target_deviation":
            # A crossed or numerically invalid candidate is not a state on an
            # admissible folding path.  Do not extract either progress or path
            # cost from it; termination supplies the learning signal.
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            dense_reward = 0.0
        else:
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            dense_reward = self.compact_potential_weight * (
                self.shaping_discount * potential_after - potential_before
            )

        flat_reached = bool(
            info["flatness_ratio"] <= self.flatness_tolerance
            and not info["self_crossing"]
            and not info["correction_failed"]
        )
        compact_success = bool(
            flat_reached
            and compactness["spread_error"] <= self.spread_tolerance
        )
        wrong_flat = bool(flat_reached and not compact_success)
        curriculum_waypoint_candidate = bool(
            self.episode_is_training
            and self.curriculum_waypoint_mode_active
            and not self.curriculum_waypoint_awarded
            and info["flatness_ratio"] <= self.active_flatness_tolerance
            and compactness["spread_error"] <= self.spread_tolerance
            and not info["self_crossing"]
            and not info["correction_failed"]
        )

        reward = float(dense_reward)
        terminal_reward_component = 0.0
        if curriculum_waypoint_candidate:
            reward += self.curriculum_waypoint_bonus
            terminal_reward_component += self.curriculum_waypoint_bonus
            self.curriculum_waypoint_awarded = True
            self.curriculum_waypoint_bonus_count += 1
            if self.curriculum_waypoint_bonus_count > 1:
                raise RuntimeError("Curriculum waypoint bonus was awarded twice.")
        if info["correction_failed"]:
            # Invalid candidates receive no dense reward.  Numerical failure
            # retains its explicit diagnostic penalty.
            dense_reward = 0.0
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            terminal_reward_component = -self.numerical_failure_penalty
            reward = terminal_reward_component
            termination_reason = "correction_failed"
        elif info["self_crossing"]:
            # Deliberate design choice: crossing only terminates the episode.
            # It has neither a terminal penalty nor reward from the invalid
            # transition itself; in particular, it cannot claim flatness
            # progress from a crossed candidate.
            dense_reward = 0.0
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            terminal_reward_component = 0.0
            reward = 0.0
            termination_reason = "self_crossing"
        elif compact_success:
            success_component = (
                self.flat_terminal_bonus * compactness["spread_quality"]
                + self.joint_success_bonus
            )
            reward += success_component
            terminal_reward_component += success_component
            termination_reason = "success"
        elif wrong_flat and (
            self.quality_combination == "path_integrated_target_deviation"
            or self.terminate_wrong_flat
        ):
            if (
                self.quality_combination
                != "path_integrated_target_deviation"
            ):
                wrong_flat_component = (
                    self.flat_terminal_bonus * compactness["spread_quality"]
                    - self.wrong_flat_penalty
                )
                reward += wrong_flat_component
                terminal_reward_component += wrong_flat_component
            # In the target-deviation path-integral design, wrong-flat is a
            # first-hit terminal with no extra bonus or penalty.
            termination_reason = "wrong_flat"
        elif info["no_modes"]:
            reward -= self.no_mode_penalty
            terminal_reward_component -= self.no_mode_penalty
            termination_reason = "no_modes"
        elif self.step_count >= self.max_steps:
            reward -= self.timeout_penalty
            terminal_reward_component -= self.timeout_penalty
            termination_reason = "timeout"
        else:
            termination_reason = "running"

        self.previous_compact_potential = potential_after
        info.update(compactness)
        info.update({
            "bounded_flatness_before": bounded_flatness_before,
            "bounded_flatness_after": bounded_flatness_after,
            "flatness_progress_reward": float(flatness_progress_reward),
            "spread_before": float(metrics_before["spread"]),
            "target_deviation_energy_before": float(
                metrics_before["target_deviation_energy"]
            ),
            "target_deviation_energy_after": float(
                compactness["target_deviation_energy"]
            ),
            "target_progress_reward": float(target_progress_reward),
            "normalised_path_increment": normalised_path_increment,
            "integrated_target_deviation_cost": float(
                target_deviation_path_cost
            ),
            "target_deviation_path_reward": float(
                target_deviation_path_reward
            ),
            "dense_reward": float(dense_reward),
            "terminal_reward_component": float(terminal_reward_component),
            "after_state_valid": int(transition_is_valid),
            "flat_reached": flat_reached,
            "wrong_flat": wrong_flat,
            "success": compact_success,
            "compact_success": compact_success,
            "wrong_flat_current": wrong_flat,
            "curriculum_waypoint_reached": curriculum_waypoint_candidate,
            "curriculum_waypoint_awarded": self.curriculum_waypoint_awarded,
            "curriculum_waypoint_bonus_count": (
                self.curriculum_waypoint_bonus_count
            ),
            "termination_reason": termination_reason,
            "timeout": termination_reason == "timeout",
        })
        return next_state, float(reward), termination_reason != "running", info


@dataclass
class PPORolloutTransition:
    reward: float
    state: torch.Tensor
    modes: torch.Tensor
    raw_mixture: torch.Tensor
    raw_width: torch.Tensor
    old_log_probability: torch.Tensor
    old_value: torch.Tensor


class PPOActorCriticAgent(_BASE_AGENT_CLASS):
    """Variable-mode Actor--Critic with clipped on-policy PPO updates."""

    def __init__(
        self,
        *args,
        ppo_epochs=4,
        ppo_clip=0.2,
        target_kl=0.02,
        reset_critic_at_waypoint_start=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_clip = float(ppo_clip)
        self.target_kl = float(target_kl)
        self.reset_critic_at_waypoint_start = bool(
            reset_critic_at_waypoint_start
        )
        self.critic_reset_count = 0
        self.finished_episode_outcomes = []
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be at least one.")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("ppo_clip must lie in (0, 1).")
        if self.target_kl < 0.0:
            raise ValueError("target_kl cannot be negative.")
        self._pending_latent_action = None

    def record_finished_episode_outcome(self, success, self_crossing):
        """Attach the true terminal labels to the latest PPO episode."""
        if len(self.finished_episodes) == len(self.finished_episode_outcomes):
            # An episode with no sampled action (for example, no initial mode)
            # is not placed in the PPO rollout buffer.
            return
        if len(self.finished_episodes) != len(self.finished_episode_outcomes) + 1:
            raise RuntimeError("PPO episode outcomes are out of sync.")
        self.finished_episode_outcomes.append({
            "success": bool(success),
            "self_crossing": bool(self_crossing),
        })

    def reset_critic_for_curriculum(self):
        """Discard a value function trained under the relaxed terminal task."""
        if (
            self.current_episode
            or self.finished_episodes
            or self.finished_episode_outcomes
        ):
            raise RuntimeError(
                "Critic reset must occur at an empty PPO rollout boundary."
            )
        learning_rate = self.critic_learning_rate
        for module in (
            self.model.critic_state,
            self.model.critic_mode,
            self.model.critic,
        ):
            module.apply(
                lambda child: child.reset_parameters()
                if hasattr(child, "reset_parameters")
                else None
            )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameter_list, lr=learning_rate
        )
        self.critic_reset_count += 1

    def get_action(self, state, modes, deterministic=False):
        state_tensor = torch.as_tensor(
            state[np.newaxis, :], dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        mean, variance, modal_features = (
            self.model.modal_distribution_parameters(state_tensor, mode_tensor)
        )
        if mean.numel() == 0:
            raise ValueError("Cannot sample an action without mechanism modes.")

        mixture_distribution = Normal(mean, torch.sqrt(variance))
        raw_mixture = mean if deterministic else mixture_distribution.sample()
        mixture_norm = torch.linalg.vector_norm(raw_mixture)
        if float(mixture_norm.detach()) <= 1e-12:
            normalised_mixture = torch.zeros_like(raw_mixture)
            physical_direction = state_tensor.new_zeros(state_tensor.shape[1])
        else:
            normalised_mixture = raw_mixture / mixture_norm
            normalised_mixture, physical_direction = (
                base.canonical_physical_direction(
                    normalised_mixture, mode_tensor
                )
            )

        width_mean, width_variance = (
            self.model.width_distribution_parameters(
                state_tensor, physical_direction, modal_features
            )
        )
        width_distribution = Normal(width_mean, torch.sqrt(width_variance))
        raw_width = (
            width_mean if deterministic else width_distribution.sample()
        )
        signed_width = self.max_step_width * torch.tanh(raw_width)
        log_probability = (
            mixture_distribution.log_prob(raw_mixture).sum()
            + width_distribution.log_prob(raw_width).sum()
        )
        entropy = (
            mixture_distribution.entropy().mean()
            + width_distribution.entropy().sum()
        )
        value = self.model.value(state_tensor, mode_tensor)

        self._pending_latent_action = {
            "state": state_tensor.detach().clone(),
            "modes": mode_tensor.detach().clone(),
            "raw_mixture": raw_mixture.detach().clone(),
            "raw_width": raw_width.detach().clone(),
            "old_log_probability": log_probability.detach().clone(),
            "old_value": value.detach().clone(),
        }
        return base.PolicyAction(
            physical_direction=physical_direction.detach().cpu().numpy(),
            mode_coefficients=normalised_mixture.detach().cpu().numpy(),
            requested_width=float(signed_width.detach()),
            log_probability=log_probability,
            value=value,
            entropy=entropy,
            mean_mixture_variance=float(variance.mean().detach()),
            width_mean=float(width_mean.detach()),
            width_variance=float(width_variance.detach()),
        )

    def add_transition(self, transition):
        if self._pending_latent_action is None:
            raise RuntimeError("A PPO transition has no pending latent action.")
        latent = self._pending_latent_action
        self.current_episode.append(PPORolloutTransition(
            reward=float(transition.reward),
            state=latent["state"],
            modes=latent["modes"],
            raw_mixture=latent["raw_mixture"],
            raw_width=latent["raw_width"],
            old_log_probability=latent["old_log_probability"],
            old_value=latent["old_value"],
        ))
        self._pending_latent_action = None

    def evaluate_latent_action(self, transition, include_value=True):
        mean, variance, modal_features = (
            self.model.modal_distribution_parameters(
                transition.state, transition.modes
            )
        )
        mixture_distribution = Normal(mean, torch.sqrt(variance))
        mixture_norm = torch.linalg.vector_norm(transition.raw_mixture)
        if float(mixture_norm.detach()) <= 1e-12:
            normalised_mixture = torch.zeros_like(transition.raw_mixture)
            physical_direction = transition.state.new_zeros(
                transition.state.shape[1]
            )
        else:
            normalised_mixture = transition.raw_mixture / mixture_norm
            _, physical_direction = base.canonical_physical_direction(
                normalised_mixture, transition.modes
            )
        width_mean, width_variance = (
            self.model.width_distribution_parameters(
                transition.state, physical_direction, modal_features
            )
        )
        width_distribution = Normal(width_mean, torch.sqrt(width_variance))
        log_probability = (
            mixture_distribution.log_prob(transition.raw_mixture).sum()
            + width_distribution.log_prob(transition.raw_width).sum()
        )
        entropy = (
            mixture_distribution.entropy().mean()
            + width_distribution.entropy().sum()
        )
        value = (
            self.model.value(transition.state, transition.modes)
            if include_value
            else None
        )
        return log_probability, entropy, value

    def _evaluate_batch(self, transitions, include_values=False):
        log_probabilities = []
        entropies = []
        values = []
        for transition in transitions:
            log_probability, entropy, value = self.evaluate_latent_action(
                transition, include_value=include_values
            )
            log_probabilities.append(log_probability)
            entropies.append(entropy)
            if include_values:
                values.append(value)
        result = (
            torch.stack(log_probabilities),
            torch.stack(entropies),
        )
        if include_values:
            return (*result, torch.stack(values))
        return result

    def update(self):
        if not self.finished_episodes:
            return None

        transitions = []
        returns = []
        batch_has_success = False
        batch_has_crossing = False
        for episode_index, episode in enumerate(self.finished_episodes):
            discounted_return = 0.0
            episode_returns = []
            for transition in reversed(episode):
                discounted_return = (
                    transition.reward + self.gamma * discounted_return
                )
                episode_returns.append(discounted_return)
            episode_returns.reverse()
            transitions.extend(episode)
            returns.extend(episode_returns)
            if episode_returns:
                if episode_index < len(self.finished_episode_outcomes):
                    outcome = self.finished_episode_outcomes[episode_index]
                    batch_has_success = (
                        batch_has_success or outcome["success"]
                    )
                    batch_has_crossing = (
                        batch_has_crossing or outcome["self_crossing"]
                    )
                else:
                    # Backward-compatible fallback for direct synthetic tests
                    # that populate the PPO buffer without an environment.
                    terminal_return = episode_returns[0]
                    batch_has_success = (
                        batch_has_success or terminal_return > 5.0
                    )

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        old_log_probabilities = torch.stack([
            transition.old_log_probability for transition in transitions
        ])
        old_values = torch.stack([
            transition.old_value for transition in transitions
        ])
        advantages = returns_tensor - old_values
        if advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)
        advantages = advantages.detach()

        # The critic gets one Monte-Carlo regression step per on-policy batch.
        self.critic_optimizer.zero_grad()
        _, _, current_values = self._evaluate_batch(
            transitions, include_values=True
        )
        critic_loss = nn.functional.mse_loss(
            current_values, returns_tensor
        )
        (self.value_coefficient * critic_loss).backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            self.critic_parameter_list, self.critic_gradient_clip
        )
        self.critic_optimizer.step()

        actor_loss = torch.zeros(())
        entropy_mean = torch.zeros(())
        actor_gradient_norm = torch.zeros(())
        approx_kl = 0.0
        clip_fraction = 0.0
        importance_ratio_mean = 1.0
        importance_ratio_max = 1.0
        epochs_completed = 0
        for _ in range(self.ppo_epochs):
            self.actor_optimizer.zero_grad()
            new_log_probabilities, entropies = self._evaluate_batch(
                transitions
            )
            log_ratio = new_log_probabilities - old_log_probabilities
            importance_ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
            unclipped = importance_ratio * advantages
            clipped = torch.clamp(
                importance_ratio,
                1.0 - self.ppo_clip,
                1.0 + self.ppo_clip,
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            entropy_mean = entropies.mean()
            actor_loss = (
                policy_loss - self.entropy_coefficient * entropy_mean
            )
            actor_loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                self.actor_parameter_list, self.actor_gradient_clip
            )
            self.actor_optimizer.step()
            epochs_completed += 1

            with torch.no_grad():
                updated_log_probabilities, _ = self._evaluate_batch(
                    transitions
                )
                updated_log_ratio = (
                    updated_log_probabilities - old_log_probabilities
                )
                updated_ratio = torch.exp(
                    torch.clamp(updated_log_ratio, -20.0, 20.0)
                )
                # Schulman's non-negative k3 estimator is much less likely than
                # mean(old_logp - new_logp) to miss an oversized PPO update on
                # a small rollout batch merely because of sampling noise.
                approx_kl = float(
                    ((updated_ratio - 1.0) - updated_log_ratio).mean()
                )
                clip_fraction = float(
                    (torch.abs(updated_ratio - 1.0) > self.ppo_clip)
                    .float()
                    .mean()
                )
                importance_ratio_mean = float(updated_ratio.mean())
                importance_ratio_max = float(updated_ratio.max())
            if self.target_kl > 0.0 and approx_kl > self.target_kl:
                break

        total_loss = (
            actor_loss.detach()
            + self.value_coefficient * critic_loss.detach()
        )
        self.finished_episodes = []
        self.finished_episode_outcomes = []
        return {
            "loss": float(total_loss),
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "actor_gradient_norm": float(actor_gradient_norm),
            "critic_gradient_norm": float(critic_gradient_norm),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "ppo_epochs_completed": int(epochs_completed),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "importance_ratio_mean": float(importance_ratio_mean),
            "importance_ratio_max": float(importance_ratio_max),
            "batch_has_success": int(batch_has_success),
            "batch_has_crossing": int(batch_has_crossing),
        }


def make_environment(args, node, member, fix):
    return CompactTargetFoldingEnv(
        node=node,
        member=member,
        fix=fix,
        max_steps=args.max_steps,
        max_step_width=args.max_step_width,
        min_step_width=args.min_step_width,
        projection_shrink=args.projection_shrink,
        max_correction_ratio=args.max_correction_ratio,
        max_actual_step_ratio=args.max_actual_step_ratio,
        flatness_subspace_dimension=args.flatness_subspace_dimension,
        flatness_tolerance=args.flatness_tolerance,
        strain_tolerance=args.strain_tolerance,
        crossing_tolerance=args.crossing_tolerance,
        flatness_reward_weight=0.0,
        flatness_reward_clip=args.flatness_reward_clip,
        success_bonus=0.0,
        crossing_penalty=args.crossing_penalty,
        numerical_failure_penalty=args.numerical_failure_penalty,
        no_mode_penalty=args.no_mode_penalty,
        timeout_penalty=args.timeout_penalty,
        target_spread=args.target_spread,
        spread_sigma=args.spread_sigma,
        spread_tolerance=args.spread_tolerance,
        flat_quality_clip=args.flat_quality_clip,
        quality_combination=args.quality_combination,
        flat_quality_weight=args.flat_quality_weight,
        flat_signal_floor=args.flat_signal_floor,
        compact_potential_weight=args.compact_potential_weight,
        flatness_progress_weight=getattr(
            args, "flatness_progress_weight", 1.0
        ),
        target_progress_weight=getattr(
            args, "target_progress_weight", 1.0
        ),
        target_deviation_cost_weight=getattr(
            args, "target_deviation_cost_weight", 1.0
        ),
        shaping_discount=args.gamma,
        flat_terminal_bonus=args.flat_terminal_bonus,
        joint_success_bonus=args.joint_success_bonus,
        wrong_flat_penalty=args.wrong_flat_penalty,
        terminate_wrong_flat=args.terminate_wrong_flat,
        training_flatness_tolerances=args.training_flatness_tolerances,
        training_flatness_milestones=args.training_flatness_milestones,
        curriculum_waypoint_bonus=args.curriculum_waypoint_bonus,
        curriculum_waypoint_start_episode=(
            args.curriculum_waypoint_start_episode
        ),
        gated_compact_cost_weight=args.gated_compact_cost_weight,
        compact_gate_scale=args.compact_gate_scale,
    )


def run_episode(env, agent, deterministic=False, training=False):
    reset_critic_now = bool(
        training
        and env.curriculum_waypoint_bonus > 0.0
        and env.curriculum_waypoint_start_episode > 0
        and env.training_episode_count == env.curriculum_waypoint_start_episode
        and getattr(agent, "reset_critic_at_waypoint_start", False)
    )
    if reset_critic_now:
        agent.reset_critic_for_curriculum()
    env.configure_episode_tolerance(training=training)
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    if training and hasattr(agent, "record_finished_episode_outcome"):
        agent.record_finished_episode_outcome(
            summary["success"], summary["self_crossing"]
        )
    terminal_metrics = env.compactness_metrics(nodes[-1])
    terminal_is_valid = not (
        summary["self_crossing"]
        or summary["correction_failed"]
        or summary["no_modes"]
    )
    terminal_flat_reached = bool(
        terminal_is_valid
        and summary["flatness_ratio"] <= env.active_flatness_tolerance
    )
    terminal_wrong_flat = bool(
        terminal_flat_reached
        and terminal_metrics["spread_error"] > env.spread_tolerance
    )
    strict_flat_reached = bool(
        terminal_is_valid
        and summary["flatness_ratio"] <= env.evaluation_flatness_tolerance
    )
    strict_compact_success = bool(
        strict_flat_reached
        and terminal_metrics["spread_error"] <= env.spread_tolerance
    )
    trajectory_metrics = [env.compactness_metrics(node) for node in nodes]
    trajectory_ratios = [env.flatness_metrics(node)[1] for node in nodes]
    valid_trajectory_length = (
        len(nodes) - 1
        if (summary["self_crossing"] or summary["correction_failed"])
        else len(nodes)
    )
    valid_trajectory_metrics = trajectory_metrics[:valid_trajectory_length]
    valid_trajectory_ratios = trajectory_ratios[:valid_trajectory_length]
    ever_flat_reached = any(
        ratio <= env.active_flatness_tolerance
        for ratio in valid_trajectory_ratios
    )
    ever_wrong_flat = any(
        ratio <= env.active_flatness_tolerance
        and metrics["spread_error"] > env.spread_tolerance
        for ratio, metrics in zip(
            valid_trajectory_ratios, valid_trajectory_metrics
        )
    )
    summary.update(terminal_metrics)
    summary.update({
        "target_spread": float(env.target_spread),
        "spread_tolerance": float(env.spread_tolerance),
        "quality_combination": env.quality_combination,
        "flat_quality_weight": float(env.flat_quality_weight),
        "active_flatness_tolerance": float(env.active_flatness_tolerance),
        "evaluation_flatness_tolerance": float(
            env.evaluation_flatness_tolerance
        ),
        "flat_reached": int(terminal_flat_reached),
        "wrong_flat": int(terminal_wrong_flat),
        "strict_flat_reached": int(strict_flat_reached),
        "strict_compact_success": int(strict_compact_success),
        "relaxed_only_success": int(
            summary["success"] and not strict_compact_success
        ),
        "ever_flat_reached": int(ever_flat_reached),
        "ever_wrong_flat": int(ever_wrong_flat),
        "compact_success": int(summary["success"]),
        "curriculum_waypoint_reached": int(
            env.curriculum_waypoint_awarded
        ),
        "curriculum_waypoint_bonus_count": int(
            env.curriculum_waypoint_bonus_count
        ),
        "critic_reset_at_curriculum_start": int(reset_critic_now),
        "maximum_combined_quality": float(max(
            metrics["combined_quality"]
            for metrics in valid_trajectory_metrics
        )),
    })
    previous_metrics = env.compactness_metrics(nodes[0])
    previous_quality = previous_metrics["combined_quality"]
    waypoint_seen_in_saved_path = False
    total_flatness_progress_reward = 0.0
    total_target_progress_reward = 0.0
    total_integrated_target_deviation_cost = 0.0
    total_target_deviation_path_reward = 0.0
    total_normalised_path_length = 0.0
    for action_index, (action, node) in enumerate(zip(actions, nodes[1:])):
        metrics = env.compactness_metrics(node)
        terminal_failure_at_step = bool(
            action_index == len(actions) - 1
            and (summary["self_crossing"] or summary["correction_failed"])
        )
        normalised_path_increment = float(
            action["actual_displacement"]
            / max(env.coordinate_scale, 1e-12)
        )
        if not np.isfinite(normalised_path_increment):
            normalised_path_increment = 0.0
        if terminal_failure_at_step:
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            shaping_reward = 0.0
        elif (
            env.quality_combination
            == "path_integrated_target_deviation"
        ):
            reward_components = target_deviation_reward_components(
                previous_metrics["flatness_ratio"],
                metrics["flatness_ratio"],
                previous_metrics["spread"],
                metrics["spread"],
                normalised_path_increment,
                env.target_spread,
                env.flatness_reward_clip,
                env.flatness_progress_weight,
                env.target_progress_weight,
                env.target_deviation_cost_weight,
            )
            flatness_progress_reward = reward_components[
                "flatness_progress_reward"
            ]
            target_progress_reward = reward_components[
                "target_progress_reward"
            ]
            target_deviation_path_cost = reward_components[
                "integrated_target_deviation_cost"
            ]
            target_deviation_path_reward = reward_components[
                "target_deviation_path_reward"
            ]
            shaping_reward = reward_components["dense_reward"]
        else:
            flatness_progress_reward = 0.0
            target_progress_reward = 0.0
            target_deviation_path_cost = 0.0
            target_deviation_path_reward = 0.0
            shaping_reward = env.compact_potential_weight * (
                env.shaping_discount * metrics["combined_quality"]
                - previous_quality
            )
        action.update(metrics)
        waypoint_at_step = bool(
            training
            and env.curriculum_waypoint_mode_active
            and not waypoint_seen_in_saved_path
            and not terminal_failure_at_step
            and action["flatness_ratio"] <= env.active_flatness_tolerance
            and metrics["spread_error"] <= env.spread_tolerance
        )
        waypoint_seen_in_saved_path |= waypoint_at_step
        action.update({
            "combined_quality_before": float(previous_quality),
            "compact_shaping_reward": float(shaping_reward),
            "flatness_progress_reward": float(
                flatness_progress_reward
            ),
            "spread_before": float(previous_metrics["spread"]),
            "target_deviation_energy_before": float(
                previous_metrics["target_deviation_energy"]
            ),
            "target_deviation_energy_after": float(
                metrics["target_deviation_energy"]
            ),
            "target_progress_reward": float(target_progress_reward),
            "normalised_path_increment": normalised_path_increment,
            "integrated_target_deviation_cost": float(
                target_deviation_path_cost
            ),
            "target_deviation_path_reward": float(
                target_deviation_path_reward
            ),
            "dense_reward": float(shaping_reward),
            "terminal_reward_component": float(action["reward"] - shaping_reward),
            "after_state_valid": int(not terminal_failure_at_step),
            "termination_reason": (
                summary["termination_reason"]
                if action_index == len(actions) - 1
                else "running"
            ),
            "self_crossing_at_step": int(
                action_index == len(actions) - 1
                and summary["self_crossing"]
            ),
            "correction_failed_at_step": int(
                action_index == len(actions) - 1
                and summary["correction_failed"]
            ),
            "success_at_step": int(
                action_index == len(actions) - 1
                and summary["success"]
            ),
            "timeout_at_step": int(
                action_index == len(actions) - 1
                and summary["timeout"]
            ),
            "curriculum_waypoint_reached_at_step": int(waypoint_at_step),
            "flat_reached_at_step": int(
                not terminal_failure_at_step
                and action["flatness_ratio"] <= env.active_flatness_tolerance
            ),
            "wrong_flat_at_step": int(
                not terminal_failure_at_step
                and action["flatness_ratio"] <= env.active_flatness_tolerance
                and metrics["spread_error"] > env.spread_tolerance
            ),
        })
        total_flatness_progress_reward += flatness_progress_reward
        total_target_progress_reward += target_progress_reward
        total_integrated_target_deviation_cost += (
            target_deviation_path_cost
        )
        total_target_deviation_path_reward += (
            target_deviation_path_reward
        )
        if not terminal_failure_at_step:
            total_normalised_path_length += normalised_path_increment
        previous_metrics = metrics
        previous_quality = metrics["combined_quality"]
    summary.update({
        "total_flatness_progress_reward": float(
            total_flatness_progress_reward
        ),
        "total_target_progress_reward": float(
            total_target_progress_reward
        ),
        "total_integrated_target_deviation_cost": float(
            total_integrated_target_deviation_cost
        ),
        "total_target_deviation_path_reward": float(
            total_target_deviation_path_reward
        ),
        "total_terminal_reward_component": float(
            summary["total_reward"]
            - total_flatness_progress_reward
            - total_target_progress_reward
            - total_target_deviation_path_reward
        ),
        "total_normalised_path_length": float(
            total_normalised_path_length
        ),
        "maximum_spread": float(max(
            metrics["spread"] for metrics in valid_trajectory_metrics
        )),
        "maximum_target_deviation_energy": float(max(
            metrics["target_deviation_energy"]
            for metrics in valid_trajectory_metrics
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE_EVALUATIONS(evaluations, episode=episode)
    if result is None:
        return None
    result.update({
        "mean_return": float(np.mean([
            row["total_reward"] for row in evaluations
        ])),
        "flat_reached_rate": float(np.mean([
            row["flat_reached"] for row in evaluations
        ])),
        "wrong_flat_rate": float(np.mean([
            row["wrong_flat"] for row in evaluations
        ])),
        "strict_compact_success_rate": float(np.mean([
            row["strict_compact_success"] for row in evaluations
        ])),
        "relaxed_only_success_rate": float(np.mean([
            row["relaxed_only_success"] for row in evaluations
        ])),
        "curriculum_waypoint_rate": float(np.mean([
            row["curriculum_waypoint_reached"] for row in evaluations
        ])),
        "maximum_curriculum_waypoint_bonus_count": int(max(
            row["curriculum_waypoint_bonus_count"] for row in evaluations
        )),
        "ever_flat_reached_rate": float(np.mean([
            row["ever_flat_reached"] for row in evaluations
        ])),
        "ever_wrong_flat_rate": float(np.mean([
            row["ever_wrong_flat"] for row in evaluations
        ])),
        "mean_terminal_spread": float(np.mean([
            row["spread"] for row in evaluations
        ])),
        "mean_spread_error": float(np.mean([
            row["spread_error"] for row in evaluations
        ])),
        "median_spread_error": float(np.median([
            row["spread_error"] for row in evaluations
        ])),
        "mean_combined_quality": float(np.mean([
            row["combined_quality"] for row in evaluations
        ])),
        "mean_target_progress_reward": float(np.mean([
            row["total_target_progress_reward"] for row in evaluations
        ])),
        "mean_integrated_target_deviation_cost": float(np.mean([
            row["total_integrated_target_deviation_cost"]
            for row in evaluations
        ])),
        "mean_flatness_progress_reward": float(np.mean([
            row["total_flatness_progress_reward"] for row in evaluations
        ])),
        "mean_terminal_reward_component": float(np.mean([
            row["total_terminal_reward_component"] for row in evaluations
        ])),
        "mean_normalised_path_length": float(np.mean([
            row["total_normalised_path_length"] for row in evaluations
        ])),
        "mean_maximum_target_deviation_energy": float(np.mean([
            row["maximum_target_deviation_energy"]
            for row in evaluations
        ])),
    })
    return result


def policy_selection_score(evaluation):
    # Success is the task objective.  Crossing is a failed path even though the
    # requested reward design gives its terminal transition no explicit
    # penalty, so checkpoint selection uses crossing rate before return.
    return (
        float(evaluation["success_rate"]),
        -float(evaluation["self_crossing_rate"]),
        float(evaluation["mean_return"]),
        float(evaluation["mean_combined_quality"]),
        -float(evaluation["mean_spread_error"]),
        -float(evaluation["mean_terminal_ratio"]),
        -float(evaluation["wrong_flat_rate"]),
        -float(evaluation["mean_steps"]),
    )


def _plot_raw_and_average(axis, episodes, values, window, log=False):
    values = np.asarray(values, dtype=float)
    axis.plot(episodes, np.maximum(values, 1e-12) if log else values,
              color="0.78", linewidth=0.45)
    x, averaged = base.moving_average(values, window)
    if len(x):
        axis.plot(x, np.maximum(averaged, 1e-12) if log else averaged,
                  color="black", linewidth=1.8)
    if log:
        axis.set_yscale("log")


def save_training_plots(
    output_directory,
    history,
    evaluation_history,
    stochastic_evaluation_history,
    window,
):
    episodes = np.asarray([row["episode"] for row in history])
    rewards = np.asarray([row["total_reward"] for row in history])
    ratios = np.asarray([row["flatness_ratio"] for row in history])
    spreads = np.asarray([row["spread"] for row in history])
    qualities = np.asarray([row["combined_quality"] for row in history])
    integrated_target_deviation_costs = np.asarray([
        row.get("total_integrated_target_deviation_cost", 0.0)
        for row in history
    ])
    successes = np.asarray([row["success"] for row in history])
    strict_successes = np.asarray([
        row.get("strict_compact_success", row["success"]) for row in history
    ])
    flat_reached = np.asarray([row["flat_reached"] for row in history])
    ever_flat_reached = np.asarray([
        row.get("ever_flat_reached", row["flat_reached"]) for row in history
    ])
    wrong_flat = np.asarray([row["wrong_flat"] for row in history])
    ever_wrong_flat = np.asarray([
        row.get("ever_wrong_flat", row["wrong_flat"]) for row in history
    ])
    curriculum_waypoints = np.asarray([
        row.get("curriculum_waypoint_reached", 0) for row in history
    ])
    crossings = np.asarray([row["self_crossing"] for row in history])
    steps = np.asarray([row["steps"] for row in history])

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    _plot_raw_and_average(axes[0, 0], episodes, rewards, window)
    axes[0, 0].set_title("Episode return")

    _plot_raw_and_average(axes[0, 1], episodes, ratios, window, log=True)
    active_tolerances = np.asarray([
        row.get("active_flatness_tolerance", row.get("flatness_tolerance", 1e-3))
        for row in history
    ], dtype=float)
    axes[0, 1].plot(
        episodes,
        active_tolerances,
        color="tab:purple",
        linestyle="--",
        linewidth=1.2,
        label="training tolerance",
    )
    strict_tolerance = history[0].get(
        "evaluation_flatness_tolerance", 1e-3
    )
    axes[0, 1].axhline(
        strict_tolerance,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        label="evaluation tolerance",
    )
    if evaluation_history:
        axes[0, 1].plot(
            [row["episode"] for row in evaluation_history],
            [max(row["flatness_ratio"], 1e-12) for row in evaluation_history],
            color="tab:red", marker="o", markersize=2.5, linewidth=1,
            label="deterministic",
        )
    axes[0, 1].legend()
    axes[0, 1].set_title("Terminal flatness ratio")

    _plot_raw_and_average(axes[0, 2], episodes, spreads, window)
    target = history[0].get("target_spread", None)
    if target is None:
        target = 2.0
    axes[0, 2].axhline(target, color="tab:green", linestyle="--",
                       linewidth=1.2, label="target spread")
    tolerance = history[0].get("spread_tolerance", 0.0)
    axes[0, 2].axhspan(
        target - tolerance,
        target + tolerance,
        color="tab:green",
        alpha=0.10,
        label="success tolerance",
    )
    if evaluation_history:
        axes[0, 2].plot(
            [row["episode"] for row in evaluation_history],
            [row["spread"] for row in evaluation_history],
            color="tab:red", marker="o", markersize=2.5, linewidth=1,
            label="deterministic",
        )
    axes[0, 2].legend()
    axes[0, 2].set_title("Terminal projected spread")

    for values, label, color in (
        (successes, "target-shape success", "tab:green"),
        (strict_successes, "strict target-shape success", "black"),
        (flat_reached, "terminal flat", "tab:blue"),
        (ever_flat_reached, "ever reached flat", "tab:cyan"),
        (wrong_flat, "terminal wrong flat", "tab:orange"),
        (ever_wrong_flat, "ever reached wrong flat", "tab:brown"),
        (curriculum_waypoints, "curriculum waypoint", "tab:purple"),
        (crossings, "crossing", "tab:red"),
    ):
        x, averaged = base.moving_average(values, window)
        if len(x):
            axes[1, 0].plot(x, averaged, color=color, label=label)
    if stochastic_evaluation_history:
        axes[1, 0].plot(
            [row["episode"] for row in stochastic_evaluation_history],
            [row["success_rate"] for row in stochastic_evaluation_history],
            color="black", marker="o", markersize=3, linewidth=1,
            label="frozen stochastic success",
        )
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Moving terminal rates")
    axes[1, 0].legend()

    if (
        history[0].get("quality_combination")
        == "path_integrated_target_deviation"
    ):
        _plot_raw_and_average(
            axes[1, 1], episodes, integrated_target_deviation_costs, window
        )
        axes[1, 1].set_title("Path-integrated target deviation")
    else:
        _plot_raw_and_average(axes[1, 1], episodes, qualities, window)
        finite_qualities = np.asarray(qualities, dtype=float)
        finite_qualities = finite_qualities[np.isfinite(finite_qualities)]
        quality_min = min(-0.02, float(finite_qualities.min()) - 0.02)
        quality_max = max(1.02, float(finite_qualities.max()) + 0.02)
        axes[1, 1].set_ylim(quality_min, quality_max)
        axes[1, 1].set_title("Combined flat-and-compact quality")

    _plot_raw_and_average(axes[1, 2], episodes, steps, window)
    axes[1, 2].set_title("Episode length")

    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "learning_curves.png", dpi=180)
    figure.savefig(output_directory / "training_curves.png", dpi=180)
    plt.close(figure)


class UniformRandomAgent:
    """Uniform random physical action with the policy's sign convention."""

    def __init__(self, max_step_width, rng):
        self.max_step_width = float(max_step_width)
        self.rng = rng

    def get_action(self, state, modes, deterministic=False):
        del state, deterministic
        mode_count = int(modes.shape[1])
        if mode_count == 0:
            raise ValueError("Cannot sample an action without mechanism modes.")
        coefficients = self.rng.normal(size=mode_count)
        coefficient_norm = float(np.linalg.norm(coefficients))
        while coefficient_norm <= 1e-12:
            coefficients = self.rng.normal(size=mode_count)
            coefficient_norm = float(np.linalg.norm(coefficients))
        coefficients /= coefficient_norm
        coefficient_tensor = torch.as_tensor(
            coefficients, dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        coefficient_tensor, physical_direction = (
            base.canonical_physical_direction(
                coefficient_tensor, mode_tensor
            )
        )
        requested_width = float(self.rng.uniform(
            -self.max_step_width, self.max_step_width
        ))
        zero = torch.zeros(())
        return base.PolicyAction(
            physical_direction=physical_direction.cpu().numpy(),
            mode_coefficients=coefficient_tensor.cpu().numpy(),
            requested_width=requested_width,
            log_probability=zero,
            value=zero,
            entropy=zero,
            mean_mixture_variance=1.0,
            width_mean=0.0,
            width_variance=(self.max_step_width ** 2) / 3.0,
        )


def _load_problem_data(args):
    script_directory = Path(__file__).resolve().parent
    data_directory = (
        Path(args.data_dir).resolve()
        if args.data_dir is not None
        else script_directory
    )
    node = np.atleast_2d(np.loadtxt(
        data_directory / "node_initial.dat", delimiter=","
    ))
    member = np.atleast_2d(np.loadtxt(
        data_directory / "member.dat", delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(
        data_directory / "fix.dat", delimiter=",", dtype=int
    ))
    return node, member, fix


def run_uniform_random_comparison(args, output_directory):
    """Measure first-hit cost for independent uniform-random searches."""
    enabled = bool(args.random_search and args.random_search_count > 0)
    result = {
        "enabled": enabled,
        "search_count": int(args.random_search_count if enabled else 0),
        "max_episodes_per_search": int(args.random_search_max_episodes),
        "base_seed": int(args.random_search_seed),
        "random_success_rate": 0.0,
        "random_censored_count": 0,
        "restricted_mean_first_success_episode": None,
        "restricted_mean_first_success_mechanical_steps": None,
        "restricted_mean_first_success_projector_calls": None,
    }
    if not enabled:
        return result

    comparison_directory = (
        Path(output_directory) / "uniform_random_search"
    )
    comparison_directory.mkdir(parents=True, exist_ok=True)
    node, member, fix = _load_problem_data(args)
    search_rows = []
    all_start = time.perf_counter()
    episode_fieldnames = (
        "search_index", "search_seed", "episode", "success",
        "termination_reason", "mechanical_steps", "projector_calls",
        "cumulative_mechanical_steps", "cumulative_projector_calls",
        "elapsed_seconds", "total_reward", "flatness_ratio", "spread",
        "spread_error", "self_crossing", "correction_failed", "timeout",
    )

    for search_index in range(1, args.random_search_count + 1):
        search_seed = int(args.random_search_seed + search_index - 1)
        rng = np.random.default_rng(search_seed)
        agent = UniformRandomAgent(args.max_step_width, rng)
        environment = make_environment(args, node, member, fix)
        search_directory = (
            comparison_directory / f"search_{search_index:04d}"
        )
        search_directory.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        cumulative_steps = 0
        cumulative_projector_calls = 0
        termination_counts = {}
        success_summary = None
        success_nodes = None
        success_actions = None

        with open(
            search_directory / "episode_history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as history_file:
            writer = csv.DictWriter(
                history_file, fieldnames=episode_fieldnames
            )
            writer.writeheader()
            for episode in range(1, args.random_search_max_episodes + 1):
                summary, nodes, actions = run_episode(
                    environment,
                    agent,
                    deterministic=False,
                    training=False,
                )
                cumulative_steps += int(summary["steps"])
                cumulative_projector_calls += int(
                    summary["projection_attempts"]
                )
                reason = summary["termination_reason"]
                termination_counts[reason] = (
                    termination_counts.get(reason, 0) + 1
                )
                writer.writerow({
                    "search_index": search_index,
                    "search_seed": search_seed,
                    "episode": episode,
                    "success": int(summary["success"]),
                    "termination_reason": reason,
                    "mechanical_steps": int(summary["steps"]),
                    "projector_calls": int(summary["projection_attempts"]),
                    "cumulative_mechanical_steps": cumulative_steps,
                    "cumulative_projector_calls": (
                        cumulative_projector_calls
                    ),
                    "elapsed_seconds": time.perf_counter() - start,
                    "total_reward": float(summary["total_reward"]),
                    "flatness_ratio": float(summary["flatness_ratio"]),
                    "spread": float(summary["spread"]),
                    "spread_error": float(summary["spread_error"]),
                    "self_crossing": int(summary["self_crossing"]),
                    "correction_failed": int(summary["correction_failed"]),
                    "timeout": int(summary["timeout"]),
                })
                if summary["success"]:
                    success_summary = summary
                    success_nodes = nodes
                    success_actions = actions
                    break

        elapsed = time.perf_counter() - start
        episodes_used = (
            episode if args.random_search_max_episodes > 0 else 0
        )
        success = success_summary is not None
        first_success_path = ""
        if success:
            path_directory = search_directory / "hit_path"
            base.save_rollout(
                path_directory, success_nodes, success_actions
            )
            first_success_path = str(path_directory)
        search_row = {
            "search_index": search_index,
            "search_seed": search_seed,
            "success": int(success),
            "episodes": int(episodes_used),
            "mechanical_steps": int(cumulative_steps),
            "projector_calls": int(cumulative_projector_calls),
            "elapsed_seconds": float(elapsed),
            "first_success_episode": int(episodes_used) if success else "",
            "first_success_mechanical_steps": (
                int(cumulative_steps) if success else ""
            ),
            "first_success_projector_calls": (
                int(cumulative_projector_calls) if success else ""
            ),
            "first_success_path": first_success_path,
            "success_terminal_spread": (
                float(success_summary["spread"]) if success else ""
            ),
            "success_terminal_flatness_ratio": (
                float(success_summary["flatness_ratio"])
                if success else ""
            ),
            "wrong_flat_episodes": int(
                termination_counts.get("wrong_flat", 0)
            ),
            "crossing_episodes": int(
                termination_counts.get("self_crossing", 0)
            ),
            "timeout_episodes": int(
                termination_counts.get("timeout", 0)
            ),
            "correction_failed_episodes": int(
                termination_counts.get("correction_failed", 0)
            ),
        }
        search_rows.append(search_row)
        base.save_csv(comparison_directory / "summary.csv", search_rows)
        print(
            f"Random search {search_index:3d}/{args.random_search_count}: "
            f"success={success} episodes={episodes_used} "
            f"steps={cumulative_steps} calls={cumulative_projector_calls} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    successful_rows = [row for row in search_rows if row["success"]]
    successful_first_hit_episodes = [
        int(row["first_success_episode"]) for row in successful_rows
    ]
    successful_first_hit_steps = [
        int(row["first_success_mechanical_steps"])
        for row in successful_rows
    ]
    successful_first_hit_calls = [
        int(row["first_success_projector_calls"])
        for row in successful_rows
    ]
    observed_episodes = [int(row["episodes"]) for row in search_rows]
    observed_steps = [int(row["mechanical_steps"]) for row in search_rows]
    observed_calls = [int(row["projector_calls"]) for row in search_rows]
    random_success_rate = float(len(successful_rows) / len(search_rows))
    result.update({
        "successes": len(successful_rows),
        "success_rate": random_success_rate,
        "random_success_rate": random_success_rate,
        "random_censored_count": len(search_rows) - len(successful_rows),
        "total_episodes": int(sum(row["episodes"] for row in search_rows)),
        "total_mechanical_steps": int(sum(
            row["mechanical_steps"] for row in search_rows
        )),
        "total_projector_calls": int(sum(
            row["projector_calls"] for row in search_rows
        )),
        "total_elapsed_seconds": float(time.perf_counter() - all_start),
        "first_success_episodes": successful_first_hit_episodes,
        "first_success_mechanical_steps": successful_first_hit_steps,
        "first_success_projector_calls": successful_first_hit_calls,
        # Restricted means include every trial.  Successful trials contribute
        # their observed first-hit cost, while unsuccessful trials contribute
        # their censoring cost at the configured search limit.
        "restricted_mean_first_success_episode": float(np.mean(
            observed_episodes
        )),
        "restricted_mean_first_success_mechanical_steps": float(np.mean(
            observed_steps
        )),
        "restricted_mean_first_success_projector_calls": float(np.mean(
            observed_calls
        )),
        "mean_first_success_episode": (
            float(np.mean([
                row["first_success_episode"] for row in successful_rows
            ])) if successful_rows else None
        ),
        "median_first_success_episode": (
            float(np.median([
                row["first_success_episode"] for row in successful_rows
            ])) if successful_rows else None
        ),
        "mean_first_success_mechanical_steps": (
            float(np.mean([
                row["first_success_mechanical_steps"]
                for row in successful_rows
            ])) if successful_rows else None
        ),
        "median_first_success_mechanical_steps": (
            float(np.median([
                row["first_success_mechanical_steps"]
                for row in successful_rows
            ])) if successful_rows else None
        ),
        "mean_first_success_projector_calls": (
            float(np.mean([
                row["first_success_projector_calls"]
                for row in successful_rows
            ])) if successful_rows else None
        ),
        "median_first_success_projector_calls": (
            float(np.median([
                row["first_success_projector_calls"]
                for row in successful_rows
            ])) if successful_rows else None
        ),
    })
    with open(
        comparison_directory / "aggregate.json", "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
    return result


def save_first_hit_comparison_plot(
    output_directory,
    random_result,
    training_first_success_episode,
    training_first_success_mechanical_steps,
):
    """Plot uniform-random first-hit ECDFs and the learning-run first hit.

    The ECDF denominator is every independent random search, including a
    search that exhausted its episode budget without succeeding.  The curve
    therefore remains below one when there are censored random trials.
    """
    if not random_result.get("enabled"):
        return
    search_count = int(random_result.get("search_count", 0))
    if search_count <= 0:
        return
    random_episodes = np.sort(np.asarray(
        random_result.get("first_success_episodes", []), dtype=float
    ))
    random_steps = np.sort(np.asarray(
        random_result.get("first_success_mechanical_steps", []), dtype=float
    ))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (
        (
            axes[0], random_episodes, training_first_success_episode,
            random_result.get("restricted_mean_first_success_episode"),
            "Episode", "First-hit episode",
        ),
        (
            axes[1], random_steps, training_first_success_mechanical_steps,
            random_result.get(
                "restricted_mean_first_success_mechanical_steps"
            ),
            "Cumulative mechanics steps", "First-hit mechanics cost",
        ),
    )
    for axis, values, learning_hit, restricted_mean, xlabel, title in panels:
        if len(values):
            probabilities = np.arange(1, len(values) + 1) / search_count
            axis.step(
                values, probabilities, where="post", color="tab:blue",
                linewidth=2, label="uniform random",
            )
        if restricted_mean is not None:
            axis.axvline(
                float(restricted_mean), color="tab:blue", linestyle=":",
                linewidth=1.8, label="uniform random restricted mean",
            )
        if learning_hit is not None:
            axis.axvline(
                float(learning_hit), color="tab:orange", linestyle="--",
                linewidth=2, label="learning run first hit",
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Fraction with a success")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    successes = int(random_result.get("successes", 0))
    censored = int(random_result.get(
        "random_censored_count", search_count - successes
    ))
    success_rate = float(random_result.get(
        "random_success_rate", successes / search_count
    ))
    figure.suptitle(
        f"Uniform random: success {successes}/{search_count} "
        f"({success_rate:.1%}), censored {censored}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        Path(output_directory) / "first_hit_random_comparison.png", dpi=180
    )
    plt.close(figure)


def train(args):
    # Keep the recorded configuration identical to the effective default-task
    # semantics requested for ori_codex3.
    args.crossing_penalty = 0.0
    if args.random_search_count < 0:
        raise ValueError("random_search_count cannot be negative.")
    if args.random_search_max_episodes < 1:
        raise ValueError("random_search_max_episodes must be positive.")
    if args.quality_combination == "path_integrated_target_deviation":
        args.terminate_wrong_flat = True
        args.wrong_flat_penalty = 0.0
    if (
        args.update_method == "actor_critic"
        and args.reset_critic_at_waypoint_start
    ):
        raise ValueError(
            "reset_critic_at_waypoint_start is available only with PPO."
        )
    if any(
        milestone % args.batch_size != 0
        for milestone in args.training_flatness_milestones
    ):
        raise ValueError(
            "Each training flatness milestone must be divisible by batch_size "
            "so one on-policy batch update never mixes two curriculum phases."
        )
    if (
        args.curriculum_waypoint_bonus > 0.0
        and args.curriculum_waypoint_start_episode > 0
        and args.curriculum_waypoint_start_episode % args.batch_size != 0
    ):
        raise ValueError(
            "curriculum_waypoint_start_episode must be divisible by batch_size "
            "so one on-policy batch update never mixes terminal and waypoint "
            "regimes."
        )
    objective_name = (
        "path_integrated_target_deviation"
        if args.quality_combination == "path_integrated_target_deviation"
        else "compact_target_combined_potential"
    )
    update_name = (
        "actor_critic"
        if args.update_method == "actor_critic"
        else "ppo_actor_critic"
    )
    if not getattr(args, "architecture_name", None):
        args.architecture_name = f"variable_mode_{objective_name}_{update_name}"

    def make_selected_agent(*agent_args, **agent_kwargs):
        if args.update_method == "actor_critic":
            # This is exactly the episode-separated Monte-Carlo Actor--Critic
            # used by ori_codex2: one update per rollout batch, a learned value
            # baseline, advantage normalisation, entropy regularisation, and
            # separate clipped Actor/Critic optimisers.
            return _BASE_AGENT_CLASS(*agent_args, **agent_kwargs)
        return PPOActorCriticAgent(
            *agent_args,
            ppo_epochs=args.ppo_epochs,
            ppo_clip=args.ppo_clip,
            target_kl=args.target_kl,
            reset_critic_at_waypoint_start=(
                args.reset_critic_at_waypoint_start
            ),
            **agent_kwargs,
        )

    original_components = (
        base.make_environment,
        base.run_episode,
        base.aggregate_stochastic_evaluations,
        base.policy_selection_score,
        base.save_training_plots,
        base.ActorCriticAgent,
    )
    base.make_environment = make_environment
    base.run_episode = run_episode
    base.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    base.policy_selection_score = policy_selection_score
    base.save_training_plots = save_training_plots
    base.ActorCriticAgent = make_selected_agent
    try:
        output_directory, summary = base.train(args)
    finally:
        (
            base.make_environment,
            base.run_episode,
            base.aggregate_stochastic_evaluations,
            base.policy_selection_score,
            base.save_training_plots,
            base.ActorCriticAgent,
        ) = original_components

    with open(
        Path(output_directory) / "training_history.csv",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    recent = rows[-min(1000, len(rows)):]
    cumulative_training_steps = 0
    cumulative_training_projector_calls = 0
    first_success_episode = None
    first_success_steps = None
    first_success_projector_calls = None
    for row in rows:
        cumulative_training_steps += int(float(row["steps"]))
        cumulative_training_projector_calls += int(float(
            row["projection_attempts"]
        ))
        if first_success_episode is None and int(float(row["success"])):
            first_success_episode = int(float(row["episode"]))
            first_success_steps = cumulative_training_steps
            first_success_projector_calls = (
                cumulative_training_projector_calls
            )
    summary.update({
        "target_spread": args.target_spread,
        "spread_tolerance": args.spread_tolerance,
        "update_method": args.update_method,
        "quality_combination": args.quality_combination,
        "flat_quality_weight": args.flat_quality_weight,
        "flatness_progress_weight": args.flatness_progress_weight,
        "target_progress_weight": args.target_progress_weight,
        "target_deviation_cost_weight": (
            args.target_deviation_cost_weight
        ),
        "training_total_mechanical_steps": int(
            cumulative_training_steps
        ),
        "training_total_projector_calls": int(
            cumulative_training_projector_calls
        ),
        "training_first_success_episode": first_success_episode,
        "training_first_success_mechanical_steps": first_success_steps,
        "training_first_success_projector_calls": (
            first_success_projector_calls
        ),
        "training_policy_updates_before_first_success": (
            int((first_success_episode - 1) // args.batch_size)
            if first_success_episode is not None else None
        ),
        "last_1000_compact_success_rate": float(np.mean([
            float(row["compact_success"]) for row in recent
        ])),
        "last_1000_strict_compact_success_rate": float(np.mean([
            float(row["strict_compact_success"]) for row in recent
        ])),
        "last_1000_terminal_flat_rate": float(np.mean([
            float(row["flat_reached"]) for row in recent
        ])),
        "last_1000_any_flat_rate": float(np.mean([
            float(row["ever_flat_reached"]) for row in recent
        ])),
        "last_1000_wrong_flat_rate": float(np.mean([
            float(row["wrong_flat"]) for row in recent
        ])),
        "last_1000_ever_wrong_flat_rate": float(np.mean([
            float(row["ever_wrong_flat"]) for row in recent
        ])),
        "last_1000_curriculum_waypoint_rate": float(np.mean([
            float(row["curriculum_waypoint_reached"]) for row in recent
        ])),
        "last_1000_mean_terminal_spread": float(np.mean([
            float(row["spread"]) for row in recent
        ])),
        "last_1000_mean_spread_error": float(np.mean([
            float(row["spread_error"]) for row in recent
        ])),
        "last_1000_mean_target_progress_reward": float(np.mean([
            float(row["total_target_progress_reward"])
            for row in recent
        ])),
        "last_1000_mean_integrated_target_deviation_cost": float(np.mean([
            float(row["total_integrated_target_deviation_cost"])
            for row in recent
        ])),
        "last_1000_mean_flatness_progress_reward": float(np.mean([
            float(row["total_flatness_progress_reward"])
            for row in recent
        ])),
        "last_1000_mean_normalised_path_length": float(np.mean([
            float(row["total_normalised_path_length"])
            for row in recent
        ])),
    })
    random_result = run_uniform_random_comparison(args, output_directory)
    summary["uniform_random_search"] = random_result
    random_episodes = random_result.get("first_success_episodes", [])
    random_steps = random_result.get(
        "first_success_mechanical_steps", []
    )
    random_search_count = int(random_result.get("search_count", 0))
    summary["first_hit_comparison"] = {
        "training_first_success_episode": first_success_episode,
        "training_first_success_mechanical_steps": first_success_steps,
        "random_search_count": random_search_count,
        "random_successes": int(random_result.get("successes", 0)),
        "random_success_rate": random_result.get("random_success_rate"),
        "random_censored_count": random_result.get(
            "random_censored_count"
        ),
        "restricted_mean_first_success_episode": random_result.get(
            "restricted_mean_first_success_episode"
        ),
        "restricted_mean_first_success_mechanical_steps": random_result.get(
            "restricted_mean_first_success_mechanical_steps"
        ),
        "restricted_mean_first_success_projector_calls": random_result.get(
            "restricted_mean_first_success_projector_calls"
        ),
        "random_median_first_success_episode": random_result.get(
            "median_first_success_episode"
        ),
        "random_median_first_success_mechanical_steps": random_result.get(
            "median_first_success_mechanical_steps"
        ),
        "fraction_random_hit_by_training_first_success_episode": (
            float(sum(
                value <= first_success_episode for value in random_episodes
            ) / random_search_count)
            if random_search_count > 0 and first_success_episode is not None
            else None
        ),
        "fraction_random_hit_by_training_first_success_mechanical_steps": (
            float(sum(
                value <= first_success_steps for value in random_steps
            ) / random_search_count)
            if random_search_count > 0 and first_success_steps is not None
            else None
        ),
    }
    save_first_hit_comparison_plot(
        output_directory,
        random_result,
        first_success_episode,
        first_success_steps,
    )
    with open(
        Path(output_directory) / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    print("Compact-target metrics")
    print(json.dumps({
        key: summary[key]
        for key in (
            "last_1000_compact_success_rate",
            "last_1000_strict_compact_success_rate",
            "last_1000_any_flat_rate",
            "last_1000_wrong_flat_rate",
            "last_1000_mean_terminal_spread",
            "last_1000_mean_target_progress_reward",
            "last_1000_mean_integrated_target_deviation_cost",
        )
    }, indent=2))
    return output_directory, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Variable-mode Actor--Critic for compact flat folding."
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument(
        "--periodic-stochastic-eval-episodes", type=int, default=50
    )
    # With the physically smaller predictor width requested for this
    # experiment, a directed folding path needs more solver steps.  Four
    # hundred steps leaves comfortable room for the known 40--130 step paths
    # at h=0.1.  Batch eight keeps the worst-case transition count per update
    # close to the former 32 x 80 configuration.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-step-width", type=float, default=0.1)
    parser.add_argument("--min-step-width", type=float, default=1e-4)
    parser.add_argument("--projection-shrink", type=float, default=0.5)
    parser.add_argument("--max-correction-ratio", type=float, default=1.0)
    parser.add_argument("--max-actual-step-ratio", type=float, default=1.25)
    parser.add_argument(
        "--flatness-subspace-dimension", type=int, choices=(1, 2), default=1
    )
    parser.add_argument("--flatness-tolerance", type=float, default=1e-3)
    parser.add_argument("--strain-tolerance", type=float, default=1e-8)
    parser.add_argument("--crossing-tolerance", type=float, default=1e-10)
    parser.add_argument("--flatness-reward-clip", type=float, default=2.0)
    parser.add_argument(
        "--crossing-penalty",
        type=float,
        default=0.0,
        help=(
            "Legacy option retained in run configs.  ori_codex3 terminates "
            "a self-crossing transition with reward 0 and applies no "
            "crossing penalty."
        ),
    )
    parser.add_argument(
        "--numerical-failure-penalty", type=float, default=10.0
    )
    parser.add_argument("--no-mode-penalty", type=float, default=5.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)
    parser.add_argument("--target-spread", type=float, default=2.0)
    parser.add_argument("--spread-sigma", type=float, default=0.35)
    parser.add_argument("--spread-tolerance", type=float, default=0.05)
    parser.add_argument("--flat-quality-clip", type=float, default=1.0)
    parser.add_argument(
        "--quality-combination",
        choices=(
            "weighted_sum", "product", "minimum", "flatness_only",
            "gated_cost", "path_integrated_target_deviation",
        ),
        default="path_integrated_target_deviation",
    )
    parser.add_argument("--flat-quality-weight", type=float, default=0.3)
    parser.add_argument("--flat-signal-floor", type=float, default=0.1)
    parser.add_argument("--compact-potential-weight", type=float, default=1.0)
    parser.add_argument(
        "--flatness-progress-weight",
        type=float,
        default=1.0,
        help=(
            "Weight alpha on the bounded flatness-ratio improvement in the "
            "path-integrated reward."
        ),
    )
    parser.add_argument(
        "--target-progress-weight",
        type=float,
        default=1.0,
        help=(
            "Weight beta on improvement of squared target-spread error."
        ),
    )
    parser.add_argument(
        "--target-deviation-cost-weight",
        type=float,
        default=1.0,
        help=(
            "Weight lambda on the corrected-path integral of "
            "(spread-target)^2."
        ),
    )
    parser.add_argument("--flat-terminal-bonus", type=float, default=0.0)
    parser.add_argument("--joint-success-bonus", type=float, default=10.0)
    parser.add_argument("--wrong-flat-penalty", type=float, default=0.0)
    parser.add_argument(
        "--near-miss-bonus-weight",
        type=float,
        default=0.0,
        help=(
            "Reserved for ori_codex4's valid-terminal near-miss reward. "
            "It is parsed but intentionally unused by ori_codex3."
        ),
    )
    parser.add_argument(
        "--near-miss-flatness-far",
        type=float,
        default=1.0,
        help="Flatness ratio mapped to zero near-miss quality in ori_codex4.",
    )
    parser.add_argument(
        "--near-miss-spread-error-far",
        type=float,
        default=1.0,
        help="Spread error mapped to zero near-miss quality in ori_codex4.",
    )
    parser.add_argument(
        "--terminate-wrong-flat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Terminate when flatness is reached at the wrong spread.  Disable "
            "to allow unfolding and refolding toward the joint compact goal "
            "in legacy reward modes.  The path-integrated mode always "
            "terminates at the first wrong-flat state."
        ),
    )
    parser.add_argument(
        "--training-flatness-tolerances",
        type=float,
        nargs="+",
        default=[1e-3],
        help="One tolerance per curriculum phase; evaluation stays strict.",
    )
    parser.add_argument(
        "--training-flatness-milestones",
        type=int,
        nargs="*",
        default=[],
        help="Last episode of each curriculum phase except the final phase.",
    )
    parser.add_argument(
        "--curriculum-waypoint-bonus",
        type=float,
        default=0.0,
        help=(
            "If positive, relaxed curriculum thresholds award this bonus once "
            "without ending the episode; strict evaluation tolerance remains "
            "the only compact-success termination threshold."
        ),
    )
    parser.add_argument(
        "--curriculum-waypoint-start-episode",
        type=int,
        default=0,
        help=(
            "First training episode after which relaxed thresholds become "
            "one-time waypoints and strict flatness becomes the only terminal "
            "success condition.  Before this episode, the active curriculum "
            "tolerance remains a terminal success condition."
        ),
    )
    parser.add_argument(
        "--gated-compact-cost-weight", type=float, default=1.2
    )
    parser.add_argument("--compact-gate-scale", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--actor-learning-rate", "--learning-rate",
        dest="actor_learning_rate", type=float, default=5e-4,
    )
    parser.add_argument("--critic-learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--actor-lr-milestones", type=int, nargs="*",
        default=[2000, 4000, 7000],
    )
    parser.add_argument(
        "--actor-lr-factors", type=float, nargs="*",
        default=[0.3, 0.1, 0.03],
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument(
        "--update-method",
        choices=("actor_critic", "ppo"),
        default="actor_critic",
        help=(
            "Policy update rule. actor_critic reproduces ori_codex2's "
            "one-pass episode-separated Monte-Carlo Actor--Critic; ppo "
            "reuses each on-policy batch with clipping and a target-KL guard."
        ),
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--reset-critic-at-waypoint-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reinitialize the critic and its optimizer when relaxed terminal "
            "success changes into a strict terminal plus waypoint task."
        ),
    )
    parser.add_argument("--actor-gradient-clip", type=float, default=0.5)
    parser.add_argument("--critic-gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument(
        "--variance-bias", type=float, default=-1.0986122886681098
    )
    parser.add_argument("--width-mean-limit", type=float, default=3.0)
    parser.add_argument("--moving-average-window", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--selection-evaluation-seed", type=int, default=1000003
    )
    parser.add_argument(
        "--final-evaluation-seed", type=int, default=2000003
    )
    parser.add_argument(
        "--random-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run independent uniform-random searches after policy training; "
            "use --no-random-search to disable the comparison."
        ),
    )
    parser.add_argument(
        "--random-search-count",
        type=int,
        default=30,
        help="Recommended number of independent random searches.",
    )
    parser.add_argument(
        "--random-search-seed", type=int, default=3000003
    )
    parser.add_argument(
        "--random-search-max-episodes", type=int, default=10000
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str,
        default=(
            "ori_codex3_h010_batch8_target_spread_2_"
            "actor_critic_10000_results"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
