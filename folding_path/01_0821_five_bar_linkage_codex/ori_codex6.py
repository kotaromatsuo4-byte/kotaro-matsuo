"""Minimal common-reward revision built on :mod:`ori_codex5`.

The mechanics, observation, action distribution, Actor--Critic optimiser,
endpoint potentials, joint milestones, terminal constants, and gamma=1
semantics are unchanged.  The discontinuous 0.9 spread milestone is replaced
by a bounded continuous state potential.  Credit from that potential and from
joint milestones is explicitly rolled back on self crossing, so a reliably
compact-but-crossed path cannot become an attractive terminal mode.

The default ``state_potential`` frontier is Markov in the existing coordinate
observation: ``Phi_D = B_D z_D``.  A flatness-gated state-potential ablation is
available, but no running-best/history-dependent frontier is implemented in
this minimal revision and the Actor/Critic input is therefore unchanged.

This module is additive.  It never mutates the source or saved-run semantics
of :mod:`ori_codex5`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex5 as ori5


REWARD_DESIGN_NAME = "ori6_continuous_spread_state_potential_rollback"
REWARD_SCHEMA_VERSION = "1.0.0"

SPREAD_FRONTIER_MODES = (
    "state_potential",
    "flatness_gated_state_potential",
)

endpoint_reward_components = ori5.endpoint_reward_components

_ORIGINAL_ORI5_RUN_EPISODE = ori5.run_episode
_ORIGINAL_ORI5_AGGREGATE_EVALUATIONS = (
    ori5.aggregate_stochastic_evaluations
)


class ContinuousSpreadPotentialTargetFoldingEnv(
    ori5.CommonRewardTargetFoldingEnv
):
    """ori5 mechanics with a continuous spread potential and rollback ledger."""

    def __init__(
        self,
        *args,
        ori6_spread_frontier_bonus=0.9,
        ori6_spread_frontier_mode="state_potential",
        ori6_spread_frontier_flatness_cap=2.0,
        **kwargs,
    ):
        self.ori6_spread_frontier_bonus = float(
            ori6_spread_frontier_bonus
        )
        self.ori6_spread_frontier_mode = str(
            ori6_spread_frontier_mode
        )
        self.ori6_spread_frontier_flatness_cap = float(
            ori6_spread_frontier_flatness_cap
        )

        # The hard ori5 spread milestone is disabled unconditionally.  Its
        # eligibility factor and all joint milestone settings remain useful
        # and retain their ori5 meanings.
        kwargs["ori5_spread_milestone_bonus"] = 0.0
        super().__init__(*args, **kwargs)
        self._validate_ori6_parameters()
        self.quality_combination = REWARD_DESIGN_NAME
        self._reset_ori6_episode_state()

    def _validate_ori6_parameters(self):
        if (
            not np.isfinite(self.ori6_spread_frontier_bonus)
            or self.ori6_spread_frontier_bonus < 0.0
        ):
            raise ValueError(
                "ori6_spread_frontier_bonus must be finite and non-negative."
            )
        if self.ori6_spread_frontier_mode not in SPREAD_FRONTIER_MODES:
            raise ValueError(
                "ori6_spread_frontier_mode must be one of "
                f"{SPREAD_FRONTIER_MODES}."
            )
        if (
            not np.isfinite(self.ori6_spread_frontier_flatness_cap)
            or self.ori6_spread_frontier_flatness_cap
            <= self.evaluation_flatness_tolerance
        ):
            raise ValueError(
                "ori6_spread_frontier_flatness_cap must be finite and "
                "greater than the evaluation flatness tolerance."
            )

    def _reset_ori6_episode_state(self):
        initial = self.compactness_metrics()
        self.ori6_initial_spread_error = float(initial["spread_error"])
        self.ori6_spread_frontier_eligible = bool(
            self.ori6_initial_spread_error
            > self.ori5_spread_milestone_initial_factor
            * self.spread_tolerance
        )
        initial_components = self.spread_frontier_components(initial)
        self.ori6_current_spread_progress = float(initial_components["z_D"])
        self.ori6_current_flatness_gate = float(initial_components["g_F"])
        self.ori6_current_spread_potential = float(
            initial_components["phi_D"]
        )
        self.ori6_maximum_spread_progress = float(
            self.ori6_current_spread_progress
        )
        self.ori6_maximum_spread_potential = float(
            self.ori6_current_spread_potential
        )
        self.ori6_cumulative_spread_frontier_reward = 0.0
        self.ori6_cumulative_joint_milestone_reward = 0.0

    def reset(self):
        state = super().reset()
        # Guard parent-construction dispatch before ori6 parameters exist.
        if hasattr(self, "ori6_spread_frontier_bonus"):
            self._reset_ori6_episode_state()
        return state

    def spread_frontier_components(self, metrics):
        """Return ``z_D``, optional ``g_F``, and bounded ``Phi_D``."""

        if not self.ori6_spread_frontier_eligible:
            z_D = 0.0
        else:
            denominator = (
                self.ori6_initial_spread_error - self.spread_tolerance
            )
            z_D = (
                self.ori6_initial_spread_error
                - float(metrics["spread_error"])
            ) / denominator
            z_D = float(np.clip(z_D, 0.0, 1.0))

        if self.ori6_spread_frontier_mode == "state_potential":
            g_F = 1.0
        else:
            denominator = (
                self.ori6_spread_frontier_flatness_cap
                - self.evaluation_flatness_tolerance
            )
            g_F = (
                self.ori6_spread_frontier_flatness_cap
                - float(metrics["flatness_ratio"])
            ) / denominator
            g_F = float(np.clip(g_F, 0.0, 1.0))

        phi_D = float(self.ori6_spread_frontier_bonus * z_D * g_F)
        return {"z_D": float(z_D), "g_F": float(g_F), "phi_D": phi_D}

    def _joint_milestone_components(self, metrics_after):
        """Keep ori5's fixed joint milestones, without its spread milestone."""

        bonuses = [0.0 for _ in self.ori5_joint_flatness_thresholds]
        if metrics_after["spread_error"] <= self.spread_tolerance:
            for index, (threshold, bonus) in enumerate(zip(
                self.ori5_joint_flatness_thresholds,
                self.ori5_joint_milestone_bonuses,
            )):
                if (
                    not self.ori5_joint_milestones_awarded[index]
                    and metrics_after["flatness_ratio"] <= threshold
                ):
                    bonuses[index] = float(bonus)
                    self.ori5_joint_milestones_awarded[index] = True
        return bonuses

    def step(self, physical_direction, requested_width, mode_coefficients):
        metrics_before = self.compactness_metrics()
        qF_before, qD_before, q_before = self.near_miss_quality(
            metrics_before["flatness_ratio"], metrics_before["spread_error"]
        )
        frontier_before = self.spread_frontier_components(metrics_before)

        # Retain ori3 mechanics and terminal-priority classification while
        # replacing every reward component explicitly below.
        next_state, _, done, info = ori3.CompactTargetFoldingEnv.step(
            self, physical_direction, requested_width, mode_coefficients
        )
        metrics_after = self.compactness_metrics()
        transition_is_valid = bool(info.get("after_state_valid", 0))
        termination_reason = str(info["termination_reason"])

        if transition_is_valid:
            base = endpoint_reward_components(
                metrics_before["flatness_ratio"],
                metrics_after["flatness_ratio"],
                metrics_before["spread"],
                metrics_after["spread"],
                self.target_spread,
                self.ori5_flatness_clip,
            )
            frontier_after = self.spread_frontier_components(metrics_after)
            spread_frontier_reward = float(
                frontier_after["phi_D"] - frontier_before["phi_D"]
            )
            joint_milestones = self._joint_milestone_components(metrics_after)
        else:
            base = endpoint_reward_components(
                metrics_before["flatness_ratio"],
                metrics_before["flatness_ratio"],
                metrics_before["spread"],
                metrics_before["spread"],
                self.target_spread,
                self.ori5_flatness_clip,
            )
            frontier_after = dict(frontier_before)
            spread_frontier_reward = 0.0
            joint_milestones = [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ]

        joint_milestone_reward = float(sum(joint_milestones))
        discovery_reward = float(
            spread_frontier_reward + joint_milestone_reward
        )
        flatness_reward = float(base["flatness_progress_reward"])
        target_reward = float(base["target_progress_reward"])
        base_reward = float(base["base_reward"])

        self.ori5_cumulative_flatness_reward += flatness_reward
        self.ori5_cumulative_target_reward += target_reward
        # Compatibility: ori5's milestone ledger now means the sum of the
        # continuous spread frontier and unchanged joint milestones.
        self.ori5_cumulative_milestone_reward += discovery_reward
        self.ori6_cumulative_spread_frontier_reward += (
            spread_frontier_reward
        )
        self.ori6_cumulative_joint_milestone_reward += (
            joint_milestone_reward
        )
        self.ori6_current_spread_progress = float(frontier_after["z_D"])
        self.ori6_current_flatness_gate = float(frontier_after["g_F"])
        self.ori6_current_spread_potential = float(frontier_after["phi_D"])
        self.ori6_maximum_spread_progress = max(
            self.ori6_maximum_spread_progress,
            self.ori6_current_spread_progress,
        )
        self.ori6_maximum_spread_potential = max(
            self.ori6_maximum_spread_potential,
            self.ori6_current_spread_potential,
        )

        qF_after, qD_after, q_after = self.near_miss_quality(
            metrics_after["flatness_ratio"], metrics_after["spread_error"]
        )
        qF_used, qD_used, q_used = qF_after, qD_after, q_after
        quality_source = "after_valid"
        terminal_fixed_reward = 0.0
        terminal_quality_reward = 0.0
        terminal_speed_reward = 0.0
        flatness_rollback_reward = 0.0
        frontier_rollback_reward = 0.0
        joint_milestone_rollback_reward = 0.0

        if termination_reason == "correction_failed":
            terminal_fixed_reward = -self.numerical_failure_penalty
            frontier_rollback_reward = -(
                self.ori6_cumulative_spread_frontier_reward
            )
            joint_milestone_rollback_reward = -(
                self.ori6_cumulative_joint_milestone_reward
            )
            quality_source = "none"
            qF_used = qD_used = q_used = 0.0
        elif termination_reason == "self_crossing":
            terminal_fixed_reward = -self.ori5_crossing_penalty
            terminal_quality_reward = self.ori5_quality_weight * q_before
            frontier_rollback_reward = -(
                self.ori6_cumulative_spread_frontier_reward
            )
            joint_milestone_rollback_reward = -(
                self.ori6_cumulative_joint_milestone_reward
            )
            quality_source = "last_valid_before_crossing"
            qF_used, qD_used, q_used = qF_before, qD_before, q_before
        elif termination_reason == "success":
            terminal_fixed_reward = self.ori5_success_bonus
            terminal_quality_reward = self.ori5_quality_weight * q_after
            terminal_speed_reward = (
                self.ori5_success_speed_bonus_weight
                * (1.0 - self.step_count / self.max_steps)
            )
        elif termination_reason == "wrong_flat":
            terminal_fixed_reward = -self.ori5_wrong_flat_penalty
            terminal_quality_reward = self.ori5_quality_weight * qD_after
            flatness_rollback_reward = -(
                self.ori5_cumulative_flatness_reward
            )
            frontier_rollback_reward = -(
                self.ori6_cumulative_spread_frontier_reward
            )
            joint_milestone_rollback_reward = -(
                self.ori6_cumulative_joint_milestone_reward
            )
            quality_source = "after_valid_spread_only"
        elif termination_reason == "no_modes":
            terminal_fixed_reward = -self.no_mode_penalty
            # Preserve ori5's joint-milestone handling, but invalid numerical
            # completion cannot keep ori6 spread-frontier credit.
            frontier_rollback_reward = -(
                self.ori6_cumulative_spread_frontier_reward
            )
            quality_source = "none"
            qF_used = qD_used = q_used = 0.0
        elif termination_reason == "timeout":
            terminal_fixed_reward = -self.ori5_timeout_penalty
            terminal_quality_reward = self.ori5_quality_weight * q_after

        terminal_correction_reward = float(
            flatness_rollback_reward
            + frontier_rollback_reward
            + joint_milestone_rollback_reward
        )

        if termination_reason in ("correction_failed", "self_crossing"):
            reward = (
                terminal_fixed_reward + terminal_quality_reward
                + terminal_correction_reward
            )
        else:
            reward = (
                base_reward + discovery_reward + terminal_fixed_reward
                + terminal_quality_reward + terminal_speed_reward
                + terminal_correction_reward
            )

        if transition_is_valid:
            self.ori5_last_valid_metrics = dict(metrics_after)

        terminal_reward_component = float(
            terminal_fixed_reward + terminal_quality_reward
            + terminal_speed_reward + terminal_correction_reward
        )
        decomposition_error = float(
            reward - base_reward - discovery_reward
            - terminal_reward_component
        )
        audited_components = (
            flatness_reward,
            target_reward,
            spread_frontier_reward,
            joint_milestone_reward,
            terminal_fixed_reward,
            terminal_quality_reward,
            terminal_speed_reward,
            flatness_rollback_reward,
            frontier_rollback_reward,
            joint_milestone_rollback_reward,
            reward,
            decomposition_error,
        )
        nonfinite_component_count = int(sum(
            not np.isfinite(value) for value in audited_components
        ))

        trace = {
            "ori6_reward_design": REWARD_DESIGN_NAME,
            "ori6_reward_schema_version": REWARD_SCHEMA_VERSION,
            "ori6_flatness_progress_reward": flatness_reward,
            "ori6_target_progress_reward": target_reward,
            "ori6_base_reward": base_reward,
            "ori6_spread_progress_before": float(frontier_before["z_D"]),
            "ori6_spread_progress_after": float(frontier_after["z_D"]),
            "ori6_flatness_gate_before": float(frontier_before["g_F"]),
            "ori6_flatness_gate_after": float(frontier_after["g_F"]),
            "ori6_spread_potential_before": float(frontier_before["phi_D"]),
            "ori6_spread_potential_after": float(frontier_after["phi_D"]),
            "ori6_spread_frontier_reward": spread_frontier_reward,
            "ori6_hard_spread_milestone_reward": 0.0,
            "ori6_joint_milestone_rewards": joint_milestones,
            "ori6_joint_milestone_reward": joint_milestone_reward,
            "ori6_discovery_reward": discovery_reward,
            "ori6_terminal_fixed_reward": float(terminal_fixed_reward),
            "ori6_terminal_quality_reward": float(terminal_quality_reward),
            "ori6_terminal_speed_reward": float(terminal_speed_reward),
            "ori6_flatness_rollback_reward": float(
                flatness_rollback_reward
            ),
            "ori6_frontier_rollback_reward": float(
                frontier_rollback_reward
            ),
            "ori6_joint_milestone_rollback_reward": float(
                joint_milestone_rollback_reward
            ),
            "ori6_terminal_correction_reward": terminal_correction_reward,
            "ori6_terminal_reward": terminal_reward_component,
            "ori6_qF": float(qF_used),
            "ori6_qD": float(qD_used),
            "ori6_q": float(q_used),
            "ori6_quality_source": quality_source,
            "ori6_spread_frontier_eligible": int(
                self.ori6_spread_frontier_eligible
            ),
            "ori6_spread_frontier_mode": self.ori6_spread_frontier_mode,
            "ori6_joint_milestones_awarded": [
                int(value) for value in self.ori5_joint_milestones_awarded
            ],
            "ori6_cumulative_flatness_reward": float(
                self.ori5_cumulative_flatness_reward
            ),
            "ori6_cumulative_target_reward": float(
                self.ori5_cumulative_target_reward
            ),
            "ori6_cumulative_spread_frontier_reward": float(
                self.ori6_cumulative_spread_frontier_reward
            ),
            "ori6_cumulative_joint_milestone_reward": float(
                self.ori6_cumulative_joint_milestone_reward
            ),
            "ori6_reward_decomposition_error": decomposition_error,
            "ori6_nonfinite_reward_component_count": (
                nonfinite_component_count
            ),
            # ori5-named compatibility aliases allow the unchanged rollout,
            # CSV, optimiser, and evaluator plumbing to consume ori6 exactly.
            "ori5_reward_design": REWARD_DESIGN_NAME,
            "ori5_reward_schema_version": REWARD_SCHEMA_VERSION,
            "ori5_flatness_progress_reward": flatness_reward,
            "ori5_target_progress_reward": target_reward,
            "ori5_base_reward": base_reward,
            "ori5_spread_milestone_reward": spread_frontier_reward,
            "ori5_joint_milestone_rewards": joint_milestones,
            "ori5_joint_milestone_reward": joint_milestone_reward,
            "ori5_milestone_reward": discovery_reward,
            "ori5_terminal_fixed_reward": float(terminal_fixed_reward),
            "ori5_terminal_quality_reward": float(terminal_quality_reward),
            "ori5_terminal_speed_reward": float(terminal_speed_reward),
            "ori5_terminal_correction_reward": terminal_correction_reward,
            "ori5_terminal_reward": terminal_reward_component,
            "ori5_qF": float(qF_used),
            "ori5_qD": float(qD_used),
            "ori5_q": float(q_used),
            "ori5_quality_source": quality_source,
            "ori5_spread_milestone_eligible": int(
                self.ori6_spread_frontier_eligible
            ),
            "ori5_spread_milestone_awarded": 0,
            "ori5_joint_milestones_awarded": [
                int(value) for value in self.ori5_joint_milestones_awarded
            ],
            "ori5_cumulative_flatness_reward": float(
                self.ori5_cumulative_flatness_reward
            ),
            "ori5_cumulative_target_reward": float(
                self.ori5_cumulative_target_reward
            ),
            "ori5_cumulative_milestone_reward": float(
                self.ori5_cumulative_milestone_reward
            ),
            "ori5_reward_decomposition_error": decomposition_error,
            "ori5_nonfinite_reward_component_count": (
                nonfinite_component_count
            ),
            "bounded_flatness_before": base["bounded_flatness_before"],
            "bounded_flatness_after": base["bounded_flatness_after"],
            "flatness_progress_reward": flatness_reward,
            "target_deviation_energy_before": base[
                "target_deviation_energy_before"
            ],
            "target_deviation_energy_after": base[
                "target_deviation_energy_after"
            ],
            "target_progress_reward": target_reward,
            "integrated_target_deviation_cost": 0.0,
            "target_deviation_path_reward": 0.0,
            "dense_reward": float(base_reward + discovery_reward),
            "terminal_reward_component": terminal_reward_component,
            "near_miss_qF": float(qF_used),
            "near_miss_qD": float(qD_used),
            "near_miss_q": float(q_used),
            "near_miss_bonus": float(terminal_quality_reward),
            "near_miss_terminal_eligible": int(
                termination_reason
                in ("success", "wrong_flat", "timeout", "self_crossing")
            ),
        }
        info.update(trace)
        self.ori5_reward_trace.append(dict(trace, **{
            "termination_reason": termination_reason,
            "after_state_valid": int(transition_is_valid),
        }))
        return next_state, float(reward), done, info


# Public name parallel to ori5 while retaining a descriptive implementation
# class name for metadata and debugging.
CommonRewardTargetFoldingEnv = ContinuousSpreadPotentialTargetFoldingEnv


def make_environment(args, node, member, fix):
    return ContinuousSpreadPotentialTargetFoldingEnv(
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
        flatness_reward_clip=args.ori5_flatness_clip,
        success_bonus=0.0,
        crossing_penalty=0.0,
        numerical_failure_penalty=args.numerical_failure_penalty,
        no_mode_penalty=args.no_mode_penalty,
        timeout_penalty=args.ori5_timeout_penalty,
        target_spread=args.target_spread,
        spread_sigma=args.spread_sigma,
        spread_tolerance=args.spread_tolerance,
        flat_quality_clip=args.flat_quality_clip,
        flat_quality_weight=args.flat_quality_weight,
        flat_signal_floor=args.flat_signal_floor,
        compact_potential_weight=1.0,
        flatness_progress_weight=1.0,
        target_progress_weight=1.0,
        target_deviation_cost_weight=0.0,
        shaping_discount=1.0,
        flat_terminal_bonus=0.0,
        joint_success_bonus=0.0,
        wrong_flat_penalty=0.0,
        terminate_wrong_flat=True,
        training_flatness_tolerances=args.training_flatness_tolerances,
        training_flatness_milestones=args.training_flatness_milestones,
        curriculum_waypoint_bonus=args.curriculum_waypoint_bonus,
        curriculum_waypoint_start_episode=(
            args.curriculum_waypoint_start_episode
        ),
        gated_compact_cost_weight=args.gated_compact_cost_weight,
        compact_gate_scale=args.compact_gate_scale,
        near_miss_bonus_weight=args.ori5_quality_weight,
        near_miss_flatness_far=args.near_miss_flatness_far,
        near_miss_spread_error_far=args.near_miss_spread_error_far,
        ori5_flatness_clip=args.ori5_flatness_clip,
        ori5_spread_milestone_initial_factor=(
            args.ori5_spread_milestone_initial_factor
        ),
        ori5_joint_flatness_thresholds=(
            args.ori5_joint_flatness_thresholds
        ),
        ori5_joint_milestone_bonuses=args.ori5_joint_milestone_bonuses,
        ori5_crossing_penalty=args.ori5_crossing_penalty,
        ori5_wrong_flat_penalty=args.ori5_wrong_flat_penalty,
        ori5_timeout_penalty=args.ori5_timeout_penalty,
        ori5_success_bonus=args.ori5_success_bonus,
        ori5_quality_weight=args.ori5_quality_weight,
        ori5_success_speed_bonus_weight=(
            args.ori5_success_speed_bonus_weight
        ),
        ori6_spread_frontier_bonus=args.ori6_spread_frontier_bonus,
        ori6_spread_frontier_mode=args.ori6_spread_frontier_mode,
        ori6_spread_frontier_flatness_cap=(
            args.ori6_spread_frontier_flatness_cap
        ),
    )


def run_episode(env, agent, deterministic=False, training=False):
    """Use ori5 rollout plumbing and append the complete ori6 ledger."""

    summary, nodes, actions = _ORIGINAL_ORI5_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in actions))

    terminal = actions[-1] if actions else {}
    # As in ori5, modes can vanish before an action is issued.  That penalty
    # is then applied by the rollout shell rather than env.step, so it has no
    # action-row trace.  Include it in ori6's terminal-fixed ledger explicitly
    # to preserve the episode reward decomposition for this edge case.
    untraced_no_mode_reward = 0.0
    if summary["termination_reason"] == "no_modes" and (
        not actions or terminal.get("termination_reason") != "no_modes"
    ):
        untraced_no_mode_reward = -float(env.no_mode_penalty)
    total_terminal_fixed = (
        total("ori6_terminal_fixed_reward") + untraced_no_mode_reward
    )
    summary.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination": REWARD_DESIGN_NAME,
        "ori6_initial_spread_error": float(env.ori6_initial_spread_error),
        "ori6_spread_frontier_eligible": int(
            env.ori6_spread_frontier_eligible
        ),
        "ori6_spread_frontier_mode": env.ori6_spread_frontier_mode,
        "ori6_maximum_spread_progress": float(
            env.ori6_maximum_spread_progress
        ),
        "ori6_maximum_spread_potential": float(
            env.ori6_maximum_spread_potential
        ),
        "ori6_joint_milestones_awarded": int(sum(
            env.ori5_joint_milestones_awarded
        )),
        "ori6_total_flatness_progress_reward": total(
            "ori6_flatness_progress_reward"
        ),
        "ori6_total_target_progress_reward": total(
            "ori6_target_progress_reward"
        ),
        "ori6_total_spread_frontier_reward": total(
            "ori6_spread_frontier_reward"
        ),
        "ori6_total_hard_spread_milestone_reward": total(
            "ori6_hard_spread_milestone_reward"
        ),
        "ori6_total_joint_milestone_reward": total(
            "ori6_joint_milestone_reward"
        ),
        "ori6_total_discovery_reward": total("ori6_discovery_reward"),
        "ori6_total_terminal_fixed_reward": total_terminal_fixed,
        "ori6_total_terminal_quality_reward": total(
            "ori6_terminal_quality_reward"
        ),
        "ori6_total_terminal_speed_reward": total(
            "ori6_terminal_speed_reward"
        ),
        "ori6_total_flatness_rollback_reward": total(
            "ori6_flatness_rollback_reward"
        ),
        "ori6_total_frontier_rollback_reward": total(
            "ori6_frontier_rollback_reward"
        ),
        "ori6_total_joint_milestone_rollback_reward": total(
            "ori6_joint_milestone_rollback_reward"
        ),
        "ori6_total_terminal_correction_reward": total(
            "ori6_terminal_correction_reward"
        ),
        "ori6_reward_decomposition_error": float(
            summary["total_reward"] - total_terminal_fixed
            - sum(total(name) for name in (
                "ori6_flatness_progress_reward",
                "ori6_target_progress_reward",
                "ori6_spread_frontier_reward",
                "ori6_joint_milestone_reward",
                "ori6_terminal_quality_reward",
                "ori6_terminal_speed_reward",
                "ori6_terminal_correction_reward",
            ))
        ),
        "ori6_maximum_abs_step_reward_decomposition_error": float(max(
            [abs(float(row["ori6_reward_decomposition_error"]))
             for row in actions]
            or [0.0]
        )),
        "ori6_nonfinite_reward_component_count": int(sum(
            int(row["ori6_nonfinite_reward_component_count"])
            for row in actions
        )),
        "ori6_terminal_quality_source": terminal.get(
            "ori6_quality_source", "none"
        ),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _ORIGINAL_ORI5_AGGREGATE_EVALUATIONS(
        evaluations, episode=episode
    )
    if result is None:
        return None
    mean_fields = {
        "ori6_total_spread_frontier_reward": (
            "mean_ori6_spread_frontier_reward"
        ),
        "ori6_total_joint_milestone_reward": (
            "mean_ori6_joint_milestone_reward"
        ),
        "ori6_total_frontier_rollback_reward": (
            "mean_ori6_frontier_rollback_reward"
        ),
        "ori6_total_joint_milestone_rollback_reward": (
            "mean_ori6_joint_milestone_rollback_reward"
        ),
        "ori6_maximum_spread_progress": "mean_ori6_maximum_spread_progress",
    }
    for source, destination in mean_fields.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result["maximum_abs_ori6_reward_decomposition_error"] = float(max(
        abs(float(row["ori6_reward_decomposition_error"]))
        for row in evaluations
    ))
    return result


def reward_parameters(args):
    """Return the complete, JSON-serialisable ori6 reward specification."""

    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "gamma": 1.0,
        "endpoint_potential": {
            "flatness_clip": float(args.ori5_flatness_clip),
            "flatness_weight": 1.0,
            "squared_spread_error_weight": 1.0,
            "path_integrated_cost_weight": 0.0,
        },
        "spread_frontier": {
            "mode": str(args.ori6_spread_frontier_mode),
            "maximum_bonus": float(args.ori6_spread_frontier_bonus),
            "spread_error_definition": "e_D = abs(D - D_target)",
            "progress_formula": (
                "z_D = clip((e_D_initial - e_D) / "
                "(e_D_initial - spread_tolerance), 0, 1)"
            ),
            "initial_eligibility_factor": float(
                args.ori5_spread_milestone_initial_factor
            ),
            "eligibility_formula": (
                "e_D_initial > initial_eligibility_factor * "
                "spread_tolerance"
            ),
            "flatness_gate_cap": float(
                args.ori6_spread_frontier_flatness_cap
            ),
            "flatness_gate_formula": (
                "g_F = clip((flatness_gate_cap - F) / "
                "(flatness_gate_cap - evaluation_flatness_tolerance), "
                "0, 1)"
            ),
            "state_potential_formula": (
                "Phi_D = maximum_bonus * z_D for state_potential; "
                "Phi_D = maximum_bonus * z_D * g_F for "
                "flatness_gated_state_potential"
            ),
            "transition_reward_formula": (
                "r_D,t = Phi_D(s_{t+1}) - Phi_D(s_t)"
            ),
            "hard_spread_milestone_bonus_effective": 0.0,
            "is_running_best": False,
            "is_markov_in_existing_coordinate_observation": True,
            "markov_assumption": (
                "the reset geometry, and therefore e_D_initial, is fixed "
                "within each trained environment"
            ),
            "randomised_reset_caveat": (
                "if reset geometry is randomised, include e_D_initial in "
                "the observation or replace it by a fixed normaliser"
            ),
            "actor_critic_frontier_scalar_input_added": False,
        },
        "joint_flatness_thresholds": list(
            args.ori5_joint_flatness_thresholds
        ),
        "joint_milestone_bonuses": list(
            args.ori5_joint_milestone_bonuses
        ),
        "crossing_penalty": float(args.ori5_crossing_penalty),
        "wrong_flat_penalty": float(args.ori5_wrong_flat_penalty),
        "timeout_penalty": float(args.ori5_timeout_penalty),
        "success_bonus": float(args.ori5_success_bonus),
        "quality_weight": float(args.ori5_quality_weight),
        "success_speed_bonus_weight": float(
            args.ori5_success_speed_bonus_weight
        ),
        "near_miss_flatness_far": float(args.near_miss_flatness_far),
        "near_miss_spread_error_far": float(
            args.near_miss_spread_error_far
        ),
        "rollback": {
            "self_crossing": ["spread_frontier", "joint_milestones"],
            "wrong_flat": [
                "flatness_progress", "spread_frontier", "joint_milestones"
            ],
            "correction_failed": [
                "spread_frontier", "joint_milestones"
            ],
            "no_modes": ["spread_frontier"],
            "timeout": [],
            "success": [],
        },
        "collision_quality_reference": "last_valid_before_crossing",
    }


def _moving_average(episodes, values, window):
    indices, averaged = mechanics.moving_average(values, window)
    if not len(indices):
        return np.asarray([]), np.asarray([])
    return episodes[np.asarray(indices, dtype=int) - 1], averaged


def save_ori6_learning_curves(output_directory, rows, window):
    """Save the new frontier ledger and unchanged joint/terminal signals."""

    if not rows:
        return
    episodes = np.asarray([float(row["episode"]) for row in rows])
    figure, axes = ori3.plt.subplots(1, 3, figsize=(15.0, 4.1))
    signals = (
        (
            "ori6_total_spread_frontier_reward",
            "Continuous spread-potential reward",
            "tab:green",
        ),
        (
            "ori6_total_joint_milestone_reward",
            "Joint milestone reward",
            "tab:purple",
        ),
    )
    for axis, (field, title, color) in zip(axes[:2], signals):
        values = np.asarray([float(row[field]) for row in rows])
        axis.plot(episodes, values, color="0.82", linewidth=0.5)
        x, average = _moving_average(episodes, values, window)
        if len(x):
            axis.plot(x, average, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    for field, label, color in (
        ("success", "success", "tab:blue"),
        ("self_crossing", "crossing", "tab:red"),
        ("wrong_flat", "wrong flat", "tab:orange"),
        ("timeout", "timeout", "tab:gray"),
    ):
        values = np.asarray([float(row[field]) for row in rows])
        x, average = _moving_average(episodes, values, window)
        if len(x):
            axes[2].plot(x, average, color=color, label=label)
    axes[2].set_title("Terminal-reason moving rates")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "ori6_frontier_learning_curve.png", dpi=180
    )
    ori3.plt.close(figure)

    # Retain the familiar near-miss diagnostic with an ori6 filename.
    quality = np.asarray([float(row["near_miss_q"]) for row in rows])
    bonus = np.asarray([float(row["near_miss_bonus"]) for row in rows])
    figure, axes = ori3.plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, values, title in (
        (axes[0], quality, "Terminal/last-valid near-miss q"),
        (axes[1], bonus, "Terminal graded-quality reward"),
    ):
        axis.plot(episodes, values, color="0.78", linewidth=0.5)
        x, average = _moving_average(episodes, values, window)
        if len(x):
            axis.plot(x, average, color="tab:blue", linewidth=1.8)
        axis.set_xlabel("Episode")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "ori6_near_miss_learning_curve.png", dpi=180
    )
    ori3.plt.close(figure)


def _validate_args(args):
    ori5._validate_args(args)
    if args.ori6_spread_frontier_mode not in SPREAD_FRONTIER_MODES:
        raise ValueError(
            f"--ori6-spread-frontier-mode must be one of {SPREAD_FRONTIER_MODES}."
        )
    if (
        not np.isfinite(args.ori6_spread_frontier_bonus)
        or args.ori6_spread_frontier_bonus < 0.0
    ):
        raise ValueError(
            "--ori6-spread-frontier-bonus must be finite and non-negative."
        )
    if (
        not np.isfinite(args.ori6_spread_frontier_flatness_cap)
        or args.ori6_spread_frontier_flatness_cap
        <= args.flatness_tolerance
    ):
        raise ValueError(
            "--ori6-spread-frontier-flatness-cap must exceed flatness tolerance."
        )
    if float(args.ori5_spread_milestone_bonus) != 0.0:
        raise ValueError("ori6 requires the hard ori5 spread milestone to be 0.")


@contextmanager
def _patched_ori5_training_runtime():
    """Reuse ori5's tested training shell without persistent monkey patches."""

    original = (
        ori5.REWARD_DESIGN_NAME,
        ori5.REWARD_SCHEMA_VERSION,
        ori5.make_environment,
        ori5.run_episode,
        ori5.aggregate_stochastic_evaluations,
        ori5.reward_parameters,
        ori5.save_ori5_learning_curves,
    )
    ori5.REWARD_DESIGN_NAME = REWARD_DESIGN_NAME
    ori5.REWARD_SCHEMA_VERSION = REWARD_SCHEMA_VERSION
    ori5.make_environment = make_environment
    ori5.run_episode = run_episode
    ori5.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    ori5.reward_parameters = reward_parameters
    ori5.save_ori5_learning_curves = save_ori6_learning_curves
    try:
        yield
    finally:
        (
            ori5.REWARD_DESIGN_NAME,
            ori5.REWARD_SCHEMA_VERSION,
            ori5.make_environment,
            ori5.run_episode,
            ori5.aggregate_stochastic_evaluations,
            ori5.reward_parameters,
            ori5.save_ori5_learning_curves,
        ) = original


def train(args):
    """Train with ori5's optimiser shell and ori6's isolated reward hooks."""

    _validate_args(args)
    parameters = reward_parameters(args)
    with _patched_ori5_training_runtime():
        output_directory, summary = ori5.train(args)

    output_directory = Path(output_directory)
    history_path = output_directory / "training_history.csv"
    with open(history_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    recent = rows[-min(1000, len(rows)):]
    summary.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination": REWARD_DESIGN_NAME,
        "ori6_reward_parameters": parameters,
        "last_1000_mean_ori6_spread_frontier_reward": float(np.mean([
            float(row["ori6_total_spread_frontier_reward"])
            for row in recent
        ])),
        "last_1000_mean_ori6_frontier_rollback_reward": float(np.mean([
            float(row["ori6_total_frontier_rollback_reward"])
            for row in recent
        ])),
        "maximum_abs_ori6_reward_decomposition_error": float(max(
            abs(float(row["ori6_reward_decomposition_error"]))
            for row in rows
        )),
        "maximum_abs_ori6_step_reward_decomposition_error": float(max(
            abs(float(row[
                "ori6_maximum_abs_step_reward_decomposition_error"
            ]))
            for row in rows
        )),
        "ori6_nonfinite_reward_component_count": int(sum(
            int(float(row["ori6_nonfinite_reward_component_count"]))
            for row in rows
        )),
    })

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination_effective": REWARD_DESIGN_NAME,
        "ori6_reward_parameters": parameters,
        "reward_component_identity": (
            "total_reward = flatness_endpoint + target_endpoint "
            "+ continuous_spread_state_potential + joint_milestones "
            "+ terminal_fixed + terminal_quality + terminal_speed "
            "+ flatness/frontier/joint_rollbacks"
        ),
        "observation_semantics": {
            "actor_critic_input_changed_from_ori5": False,
            "default_frontier_markov_in_existing_coordinates": True,
            "running_best_history_frontier_present": False,
            "non_markov_caveat": (
                "Markov under the current deterministic reset geometry. "
                "A randomised reset would require e_D_initial in the "
                "observation or a fixed normaliser."
            ),
        },
        "maximum_abs_ori6_reward_decomposition_error": summary[
            "maximum_abs_ori6_reward_decomposition_error"
        ],
        "maximum_abs_ori6_step_reward_decomposition_error": summary[
            "maximum_abs_ori6_step_reward_decomposition_error"
        ],
        "ori6_nonfinite_reward_component_count": summary[
            "ori6_nonfinite_reward_component_count"
        ],
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _ori6_argument_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--ori6-spread-frontier-bonus", type=float, default=0.9
    )
    parser.add_argument(
        "--ori6-spread-frontier-mode",
        choices=SPREAD_FRONTIER_MODES,
        default="state_potential",
    )
    parser.add_argument(
        "--ori6-spread-frontier-flatness-cap", type=float, default=2.0
    )
    return parser


def parse_args(argv=None):
    user_arguments = sys.argv[1:] if argv is None else list(argv)
    ori6_options, remaining = _ori6_argument_parser().parse_known_args(
        user_arguments
    )
    args = ori5.parse_args([
        "--output-dir", "ori_codex6_common_reward_results",
        *remaining,
    ])
    for name, value in vars(ori6_options).items():
        setattr(args, name, value)
    # Disable the inherited hard milestone even if an old ori5 option was
    # accidentally supplied.  The effective value is recorded in metadata.
    args.ori5_spread_milestone_bonus = 0.0
    args.reward_design = REWARD_DESIGN_NAME
    args.reward_schema_version = REWARD_SCHEMA_VERSION
    return args


if __name__ == "__main__":
    train(parse_args())
