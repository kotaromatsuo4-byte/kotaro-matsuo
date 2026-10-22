"""Common endpoint-potential reward for planar and spatial flat folding.

``ori_codex5`` keeps the mechanics, observation, action, and Actor--Critic
implementation used by :mod:`ori_codex4`, but replaces its reward.  The new
reward has no path-length or path-integrated spread cost.  Valid transitions
receive endpoint-potential differences, a small set of once-only milestones
supplies discovery credit, and every terminal reason has an explicit and
auditable decomposition.

The implementation deliberately lives in a new module.  Neither
``ori_codex4.py`` nor its saved-run semantics are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex4 as ori4


REWARD_DESIGN_NAME = "ori5_endpoint_potential_fixed_milestones"
REWARD_SCHEMA_VERSION = "1.0.0"

_ORI3_RUN_EPISODE = ori3.run_episode
_ORI3_AGGREGATE_EVALUATIONS = ori3.aggregate_stochastic_evaluations


def endpoint_reward_components(
    flatness_before,
    flatness_after,
    spread_before,
    spread_after,
    target_spread,
    flatness_clip=2.0,
):
    """Return the two conservative endpoint-potential increments.

    With ``gamma=1`` their sums telescope to the difference between the
    initial and final endpoint costs, irrespective of detours.
    """

    bounded_before = float(np.clip(flatness_before, 0.0, flatness_clip))
    bounded_after = float(np.clip(flatness_after, 0.0, flatness_clip))
    error_before = float(spread_before) - float(target_spread)
    error_after = float(spread_after) - float(target_spread)
    flatness_reward = bounded_before - bounded_after
    target_reward = error_before * error_before - error_after * error_after
    return {
        "bounded_flatness_before": bounded_before,
        "bounded_flatness_after": bounded_after,
        "flatness_progress_reward": float(flatness_reward),
        "target_deviation_energy_before": float(error_before * error_before),
        "target_deviation_energy_after": float(error_after * error_after),
        "target_progress_reward": float(target_reward),
        "base_reward": float(flatness_reward + target_reward),
    }


class CommonRewardTargetFoldingEnv(ori4.NearMissTargetFoldingEnv):
    """ori4 mechanics with the ori5 common reward and episode memory."""

    def __init__(
        self,
        *args,
        ori5_flatness_clip=2.0,
        ori5_spread_milestone_initial_factor=2.0,
        ori5_spread_milestone_bonus=0.9,
        ori5_joint_flatness_thresholds=(0.5, 0.1, 0.01),
        ori5_joint_milestone_bonuses=(0.2, 0.3, 0.5),
        ori5_crossing_penalty=1.25,
        ori5_wrong_flat_penalty=2.0,
        ori5_timeout_penalty=1.0,
        ori5_success_bonus=10.0,
        ori5_quality_weight=1.0,
        ori5_success_speed_bonus_weight=0.5,
        **kwargs,
    ):
        # ori3's compatibility logger asks for the same geometric quantities
        # several times per saved state, while the mechanics asks for the
        # just-computed mode basis again at the start of the next step.  Keep
        # these pure, episode-local calculations cached.  Keys contain the
        # complete coordinate bytes, so an accepted, crossed, or corrected
        # candidate can never alias a different configuration.
        self._clear_ori5_geometry_caches()
        # A supported legacy mode is used during parent construction.  The
        # dense reward returned by the parent is replaced below, and the
        # public name is changed after validation has completed.
        kwargs["quality_combination"] = "flatness_only"
        kwargs["terminate_wrong_flat"] = True
        super().__init__(*args, **kwargs)

        self.ori5_flatness_clip = float(ori5_flatness_clip)
        self.ori5_spread_milestone_initial_factor = float(
            ori5_spread_milestone_initial_factor
        )
        self.ori5_spread_milestone_bonus = float(
            ori5_spread_milestone_bonus
        )
        self.ori5_joint_flatness_thresholds = tuple(
            float(value) for value in ori5_joint_flatness_thresholds
        )
        self.ori5_joint_milestone_bonuses = tuple(
            float(value) for value in ori5_joint_milestone_bonuses
        )
        self.ori5_crossing_penalty = abs(float(ori5_crossing_penalty))
        self.ori5_wrong_flat_penalty = abs(float(ori5_wrong_flat_penalty))
        self.ori5_timeout_penalty = abs(float(ori5_timeout_penalty))
        self.ori5_success_bonus = float(ori5_success_bonus)
        self.ori5_quality_weight = float(ori5_quality_weight)
        self.ori5_success_speed_bonus_weight = float(
            ori5_success_speed_bonus_weight
        )
        self._validate_ori5_parameters()
        self.quality_combination = REWARD_DESIGN_NAME
        # Parent construction evaluates the initial state before all ori5
        # parameters above exist.  Discard those construction-time entries.
        self._clear_ori5_geometry_caches()
        self._reset_ori5_episode_state()

    def _validate_ori5_parameters(self):
        positive = {
            "ori5_flatness_clip": self.ori5_flatness_clip,
            "ori5_spread_milestone_initial_factor": (
                self.ori5_spread_milestone_initial_factor
            ),
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "ori5_spread_milestone_bonus": (
                self.ori5_spread_milestone_bonus
            ),
            "ori5_crossing_penalty": self.ori5_crossing_penalty,
            "ori5_wrong_flat_penalty": self.ori5_wrong_flat_penalty,
            "ori5_timeout_penalty": self.ori5_timeout_penalty,
            "ori5_success_bonus": self.ori5_success_bonus,
            "ori5_quality_weight": self.ori5_quality_weight,
            "ori5_success_speed_bonus_weight": (
                self.ori5_success_speed_bonus_weight
            ),
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if len(self.ori5_joint_flatness_thresholds) != len(
            self.ori5_joint_milestone_bonuses
        ):
            raise ValueError(
                "ori5 joint thresholds and bonuses must have equal length."
            )
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in self.ori5_joint_flatness_thresholds
        ):
            raise ValueError(
                "ori5 joint flatness thresholds must be finite and positive."
            )
        if any(
            later >= earlier
            for earlier, later in zip(
                self.ori5_joint_flatness_thresholds,
                self.ori5_joint_flatness_thresholds[1:],
            )
        ):
            raise ValueError(
                "ori5 joint flatness thresholds must be strictly decreasing."
            )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self.ori5_joint_milestone_bonuses
        ):
            raise ValueError(
                "ori5 joint milestone bonuses must be finite and non-negative."
            )

    @staticmethod
    def _ori5_geometry_cache_key(node):
        coordinates = np.asarray(node, dtype=float)
        return coordinates.shape, coordinates.tobytes(order="C")

    def _clear_ori5_geometry_caches(self):
        """Start a fresh cache generation at construction/reset boundaries."""

        self._ori5_flatness_metrics_cache = {}
        self._ori5_compactness_metrics_cache = {}
        self._ori5_mode_cache_key = None
        self._ori5_mode_cache = None

    def flatness_metrics(self, node=None):
        """Return the unchanged flatness metric, memoised by coordinates."""

        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        cache = getattr(self, "_ori5_flatness_metrics_cache", None)
        if cache is None:
            return super().flatness_metrics(coordinates)
        key = self._ori5_geometry_cache_key(coordinates)
        if key not in cache:
            cache[key] = tuple(super().flatness_metrics(coordinates))
        # Preserve the original value semantics rather than exposing cache
        # storage to a caller that might mutate a returned container.
        return tuple(cache[key])

    def get_modes(self):
        """Reuse the current state's SVD basis without exposing cache storage."""

        if not hasattr(self, "_ori5_mode_cache_key"):
            return super().get_modes()
        key = self._ori5_geometry_cache_key(self.node)
        if self._ori5_mode_cache_key != key:
            self._ori5_mode_cache = np.asarray(
                super().get_modes(), dtype=float
            ).copy()
            self._ori5_mode_cache_key = key
        return self._ori5_mode_cache.copy()

    def compactness_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        cache = getattr(self, "_ori5_compactness_metrics_cache", None)
        key = self._ori5_geometry_cache_key(coordinates)
        if cache is not None and key in cache:
            return dict(cache[key])

        metrics = super().compactness_metrics(coordinates)
        flatness_clip = float(getattr(
            self, "ori5_flatness_clip", self.flatness_reward_clip
        ))
        metrics["combined_quality"] = float(-(
            np.clip(metrics["flatness_ratio"], 0.0, flatness_clip)
            + metrics["target_deviation_energy"]
        ))
        if cache is not None:
            cache[key] = dict(metrics)
        return dict(metrics)

    def _reset_ori5_episode_state(self):
        initial = self.compactness_metrics()
        self.ori5_initial_spread_error = float(initial["spread_error"])
        self.ori5_spread_milestone_eligible = bool(
            self.ori5_initial_spread_error
            > self.ori5_spread_milestone_initial_factor
            * self.spread_tolerance
        )
        self.ori5_spread_milestone_awarded = False
        self.ori5_joint_milestones_awarded = [
            False for _ in self.ori5_joint_flatness_thresholds
        ]
        self.ori5_cumulative_flatness_reward = 0.0
        self.ori5_cumulative_target_reward = 0.0
        self.ori5_cumulative_milestone_reward = 0.0
        self.ori5_reward_trace = []
        self.ori5_last_valid_metrics = dict(initial)

    def reset(self):
        # In particular, remove crossed candidates and modes from the prior
        # episode before the parent evaluates the fixed initial state.
        self._clear_ori5_geometry_caches()
        state = super().reset()
        self._reset_ori5_episode_state()
        return state

    def _milestone_components(self, metrics_after, termination_reason):
        # Every valid after-state is eligible.  This matters when one move
        # jumps several nested thresholds and lands directly in success, or
        # when a timeout/no-mode state is the best near miss.  Invalid crossed
        # and correction-failed candidates never call this method.  A
        # wrong-flat state may provisionally claim milestones here, but its
        # terminal ledger cancels every milestone earned in the episode.
        spread_bonus = 0.0
        joint_bonuses = [0.0 for _ in self.ori5_joint_flatness_thresholds]

        spread_is_good = bool(
            metrics_after["spread_error"] <= self.spread_tolerance
        )
        if (
            self.ori5_spread_milestone_eligible
            and not self.ori5_spread_milestone_awarded
            and spread_is_good
        ):
            spread_bonus = self.ori5_spread_milestone_bonus
            self.ori5_spread_milestone_awarded = True

        if spread_is_good:
            for index, (threshold, bonus) in enumerate(zip(
                self.ori5_joint_flatness_thresholds,
                self.ori5_joint_milestone_bonuses,
            )):
                if (
                    not self.ori5_joint_milestones_awarded[index]
                    and metrics_after["flatness_ratio"] <= threshold
                ):
                    joint_bonuses[index] = bonus
                    self.ori5_joint_milestones_awarded[index] = True
        return float(spread_bonus), [float(value) for value in joint_bonuses]

    def step(self, physical_direction, requested_width, mode_coefficients):
        metrics_before = self.compactness_metrics()
        qF_before, qD_before, q_before = self.near_miss_quality(
            metrics_before["flatness_ratio"], metrics_before["spread_error"]
        )

        # Skip ori4's old terminal-only bonus but retain the exact ori3
        # mechanics and terminal-priority classification.
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
            spread_milestone, joint_milestones = self._milestone_components(
                metrics_after, termination_reason
            )
        else:
            base = endpoint_reward_components(
                metrics_before["flatness_ratio"],
                metrics_before["flatness_ratio"],
                metrics_before["spread"],
                metrics_before["spread"],
                self.target_spread,
                self.ori5_flatness_clip,
            )
            spread_milestone = 0.0
            joint_milestones = [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ]

        joint_milestone_reward = float(sum(joint_milestones))
        milestone_reward = float(
            spread_milestone + joint_milestone_reward
        )
        flatness_reward = float(base["flatness_progress_reward"])
        target_reward = float(base["target_progress_reward"])
        base_reward = float(base["base_reward"])

        self.ori5_cumulative_flatness_reward += flatness_reward
        self.ori5_cumulative_target_reward += target_reward
        self.ori5_cumulative_milestone_reward += milestone_reward

        qF_after, qD_after, q_after = self.near_miss_quality(
            metrics_after["flatness_ratio"], metrics_after["spread_error"]
        )
        qF_used = qF_after
        qD_used = qD_after
        q_used = q_after
        quality_source = "after_valid"
        terminal_fixed_reward = 0.0
        terminal_quality_reward = 0.0
        terminal_speed_reward = 0.0
        terminal_correction_reward = 0.0

        if termination_reason == "correction_failed":
            terminal_fixed_reward = -self.numerical_failure_penalty
            reward = terminal_fixed_reward
            quality_source = "none"
            qF_used = qD_used = q_used = 0.0
        elif termination_reason == "self_crossing":
            terminal_fixed_reward = -self.ori5_crossing_penalty
            terminal_quality_reward = self.ori5_quality_weight * q_before
            reward = terminal_fixed_reward + terminal_quality_reward
            quality_source = "last_valid_before_crossing"
            qF_used, qD_used, q_used = qF_before, qD_before, q_before
        elif termination_reason == "success":
            terminal_fixed_reward = self.ori5_success_bonus
            terminal_quality_reward = self.ori5_quality_weight * q_after
            terminal_speed_reward = (
                self.ori5_success_speed_bonus_weight
                * (1.0 - self.step_count / self.max_steps)
            )
            reward = (
                base_reward + milestone_reward + terminal_fixed_reward
                + terminal_quality_reward + terminal_speed_reward
            )
        elif termination_reason == "wrong_flat":
            terminal_fixed_reward = -self.ori5_wrong_flat_penalty
            terminal_quality_reward = self.ori5_quality_weight * qD_after
            # Remove all flatness and milestone credit accumulated in the
            # episode.  Squared target-error progress is intentionally kept.
            terminal_correction_reward = -(
                self.ori5_cumulative_flatness_reward
                + self.ori5_cumulative_milestone_reward
            )
            reward = (
                base_reward + milestone_reward + terminal_fixed_reward
                + terminal_quality_reward + terminal_correction_reward
            )
            quality_source = "after_valid_spread_only"
        elif termination_reason == "no_modes":
            terminal_fixed_reward = -self.no_mode_penalty
            reward = base_reward + milestone_reward + terminal_fixed_reward
            quality_source = "none"
        elif termination_reason == "timeout":
            terminal_fixed_reward = -self.ori5_timeout_penalty
            terminal_quality_reward = self.ori5_quality_weight * q_after
            reward = (
                base_reward + milestone_reward + terminal_fixed_reward
                + terminal_quality_reward
            )
        else:
            reward = base_reward + milestone_reward

        if transition_is_valid:
            self.ori5_last_valid_metrics = dict(metrics_after)

        terminal_reward_component = float(
            terminal_fixed_reward + terminal_quality_reward
            + terminal_speed_reward + terminal_correction_reward
        )
        decomposition_error = float(
            reward - base_reward - milestone_reward
            - terminal_reward_component
        )
        audited_components = (
            flatness_reward,
            target_reward,
            milestone_reward,
            terminal_fixed_reward,
            terminal_quality_reward,
            terminal_speed_reward,
            terminal_correction_reward,
            reward,
            decomposition_error,
        )
        nonfinite_component_count = int(sum(
            not np.isfinite(value) for value in audited_components
        ))
        trace = {
            "ori5_reward_design": REWARD_DESIGN_NAME,
            "ori5_reward_schema_version": REWARD_SCHEMA_VERSION,
            "ori5_flatness_progress_reward": flatness_reward,
            "ori5_target_progress_reward": target_reward,
            "ori5_base_reward": base_reward,
            "ori5_spread_milestone_reward": float(spread_milestone),
            "ori5_joint_milestone_rewards": joint_milestones,
            "ori5_joint_milestone_reward": joint_milestone_reward,
            "ori5_milestone_reward": milestone_reward,
            "ori5_terminal_fixed_reward": float(terminal_fixed_reward),
            "ori5_terminal_quality_reward": float(terminal_quality_reward),
            "ori5_terminal_speed_reward": float(terminal_speed_reward),
            "ori5_terminal_correction_reward": float(
                terminal_correction_reward
            ),
            "ori5_terminal_reward": terminal_reward_component,
            "ori5_qF": float(qF_used),
            "ori5_qD": float(qD_used),
            "ori5_q": float(q_used),
            "ori5_quality_source": quality_source,
            "ori5_spread_milestone_eligible": int(
                self.ori5_spread_milestone_eligible
            ),
            "ori5_spread_milestone_awarded": int(
                self.ori5_spread_milestone_awarded
            ),
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
            "dense_reward": float(base_reward + milestone_reward),
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


def make_environment(args, node, member, fix):
    """Construct the common-reward environment on ori3/ori4 mechanics."""

    return CommonRewardTargetFoldingEnv(
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
        ori5_spread_milestone_bonus=args.ori5_spread_milestone_bonus,
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
    )


def run_episode(env, agent, deterministic=False, training=False):
    """Run one rollout and replace legacy reward logs with exact ori5 logs."""

    summary, nodes, actions = _ORI3_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    trace = list(env.ori5_reward_trace)
    if len(trace) != len(actions):
        raise RuntimeError(
            "ori5 reward trace length does not match the saved action count."
        )
    for action, reward_fields in zip(actions, trace):
        action.update(reward_fields)
        action["compact_shaping_reward"] = float(
            reward_fields["dense_reward"]
        )

    # If modes vanish before an action is issued, ori_codex2 applies the
    # no-mode penalty outside env.step.  Account for that summary-only reward.
    untraced_no_mode_reward = 0.0
    if summary["termination_reason"] == "no_modes" and (
        not trace or trace[-1]["termination_reason"] != "no_modes"
    ):
        untraced_no_mode_reward = -float(env.no_mode_penalty)

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in trace))

    total_flatness = total("ori5_flatness_progress_reward")
    total_target = total("ori5_target_progress_reward")
    total_milestone = total("ori5_milestone_reward")
    total_terminal_fixed = (
        total("ori5_terminal_fixed_reward") + untraced_no_mode_reward
    )
    total_terminal_quality = total("ori5_terminal_quality_reward")
    total_terminal_speed = total("ori5_terminal_speed_reward")
    total_terminal_correction = total("ori5_terminal_correction_reward")
    decomposed_total = (
        total_flatness + total_target + total_milestone
        + total_terminal_fixed + total_terminal_quality
        + total_terminal_speed + total_terminal_correction
    )
    terminal = trace[-1] if trace else {}
    last_valid = dict(env.ori5_last_valid_metrics)
    summary.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination": REWARD_DESIGN_NAME,
        "ori5_initial_spread_error": float(env.ori5_initial_spread_error),
        "ori5_spread_milestone_eligible": int(
            env.ori5_spread_milestone_eligible
        ),
        "ori5_spread_milestone_awarded": int(
            env.ori5_spread_milestone_awarded
        ),
        "ori5_joint_milestones_awarded": int(sum(
            env.ori5_joint_milestones_awarded
        )),
        "ori5_total_flatness_progress_reward": total_flatness,
        "ori5_total_target_progress_reward": total_target,
        "ori5_total_spread_milestone_reward": total(
            "ori5_spread_milestone_reward"
        ),
        "ori5_total_joint_milestone_reward": total(
            "ori5_joint_milestone_reward"
        ),
        "ori5_total_milestone_reward": total_milestone,
        "ori5_total_terminal_fixed_reward": total_terminal_fixed,
        "ori5_total_terminal_quality_reward": total_terminal_quality,
        "ori5_total_terminal_speed_reward": total_terminal_speed,
        "ori5_total_terminal_correction_reward": total_terminal_correction,
        "ori5_reward_decomposition_error": float(
            summary["total_reward"] - decomposed_total
        ),
        "ori5_maximum_abs_step_reward_decomposition_error": float(max(
            [abs(float(row["ori5_reward_decomposition_error"]))
             for row in trace]
            or [0.0]
        )),
        "ori5_nonfinite_reward_component_count": int(sum(
            int(row["ori5_nonfinite_reward_component_count"])
            for row in trace
        )),
        "ori5_last_valid_flatness_ratio": float(
            last_valid["flatness_ratio"]
        ),
        "ori5_last_valid_spread": float(last_valid["spread"]),
        "ori5_last_valid_spread_error": float(last_valid["spread_error"]),
        "ori5_terminal_quality_source": terminal.get(
            "ori5_quality_source", "none"
        ),
        "near_miss_qF": float(terminal.get("ori5_qF", 0.0)),
        "near_miss_qD": float(terminal.get("ori5_qD", 0.0)),
        "near_miss_q": float(terminal.get("ori5_q", 0.0)),
        "near_miss_bonus": float(terminal.get(
            "ori5_terminal_quality_reward", 0.0
        )),
        "near_miss_terminal_eligible": int(terminal.get(
            "near_miss_terminal_eligible", 0
        )),
        "near_miss_bonus_awarded": int(
            abs(float(terminal.get(
                "ori5_terminal_quality_reward", 0.0
            ))) > 0.0
        ),
        # Compatibility fields consumed by ori3 plots and summaries.
        "total_flatness_progress_reward": total_flatness,
        "total_target_progress_reward": total_target,
        "total_integrated_target_deviation_cost": 0.0,
        "total_target_deviation_path_reward": 0.0,
        "total_terminal_reward_component": float(
            total_terminal_fixed + total_terminal_quality
            + total_terminal_speed + total_terminal_correction
        ),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _ORI3_AGGREGATE_EVALUATIONS(evaluations, episode=episode)
    if result is None:
        return None
    mean_fields = {
        "near_miss_qF": "mean_near_miss_qF",
        "near_miss_qD": "mean_near_miss_qD",
        "near_miss_q": "mean_near_miss_q",
        "near_miss_bonus": "mean_near_miss_bonus",
        "near_miss_terminal_eligible": (
            "near_miss_terminal_eligible_rate"
        ),
        "ori5_total_flatness_progress_reward": (
            "mean_ori5_flatness_progress_reward"
        ),
        "ori5_total_target_progress_reward": (
            "mean_ori5_target_progress_reward"
        ),
        "ori5_total_milestone_reward": "mean_ori5_milestone_reward",
        "ori5_total_terminal_correction_reward": (
            "mean_ori5_terminal_correction_reward"
        ),
    }
    for source, destination in mean_fields.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result["maximum_abs_ori5_reward_decomposition_error"] = float(max(
        abs(float(row["ori5_reward_decomposition_error"]))
        for row in evaluations
    ))
    return result


def reward_parameters(args):
    """Return the JSON-serialisable effective ori5 reward specification."""

    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "gamma": 1.0,
        "flatness_clip": float(args.ori5_flatness_clip),
        "spread_milestone_initial_factor": float(
            args.ori5_spread_milestone_initial_factor
        ),
        "spread_milestone_bonus": float(args.ori5_spread_milestone_bonus),
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
        "path_integrated_cost_weight": 0.0,
        "milestone_after_state_semantics": (
            "all valid running/success/timeout/no_modes/wrong_flat states; "
            "wrong_flat cancels the full episode milestone ledger"
        ),
    }


def _moving_average_on_episode_axis(episodes, values, window):
    indices, averaged = mechanics.moving_average(values, window)
    if not len(indices):
        return np.asarray([]), np.asarray([])
    integer_indices = np.asarray(indices, dtype=int) - 1
    return episodes[integer_indices], averaged


def save_ori5_learning_curves(output_directory, rows, window):
    """Save ori5 milestone, terminal-reason, and graded-quality curves."""

    if not rows:
        return
    episodes = np.asarray([float(row["episode"]) for row in rows])
    milestone_fields = (
        (
            "ori5_total_spread_milestone_reward",
            "Spread milestone reward",
            "tab:green",
        ),
        (
            "ori5_total_joint_milestone_reward",
            "Joint milestones reward",
            "tab:purple",
        ),
    )
    figure, axes = ori3.plt.subplots(1, 3, figsize=(15.0, 4.1))
    for axis, (field, title, color) in zip(axes[:2], milestone_fields):
        values = np.asarray([float(row[field]) for row in rows])
        axis.plot(episodes, values, color="0.82", linewidth=0.5)
        x_average, averaged = _moving_average_on_episode_axis(
            episodes, values, window
        )
        if len(x_average):
            axis.plot(x_average, averaged, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)

    reason_fields = (
        ("success", "success", "tab:blue"),
        ("self_crossing", "crossing", "tab:red"),
        ("wrong_flat", "wrong flat", "tab:orange"),
        ("timeout", "timeout", "tab:gray"),
    )
    for field, label, color in reason_fields:
        values = np.asarray([float(row[field]) for row in rows])
        x_average, averaged = _moving_average_on_episode_axis(
            episodes, values, window
        )
        if len(x_average):
            axes[2].plot(
                x_average, averaged, color=color, linewidth=1.6, label=label
            )
    axes[2].set_title("Terminal-reason moving rates")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "ori5_milestone_learning_curve.png", dpi=180
    )
    ori3.plt.close(figure)

    quality = np.asarray([float(row["near_miss_q"]) for row in rows])
    quality_reward = np.asarray([
        float(row["near_miss_bonus"]) for row in rows
    ])
    figure, axes = ori3.plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, values, title in (
        (
            axes[0], quality,
            "Reason-graded q (includes last-valid crossing q)",
        ),
        (
            axes[1], quality_reward,
            "Terminal graded-quality reward",
        ),
    ):
        axis.plot(episodes, values, color="0.78", linewidth=0.5)
        x_average, averaged = _moving_average_on_episode_axis(
            episodes, values, window
        )
        if len(x_average):
            axis.plot(
                x_average, averaged, color="tab:blue", linewidth=1.8,
                label=f"{window}-episode average",
            )
            axis.legend()
        axis.set_xlabel("Episode")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "near_miss_learning_curve.png", dpi=180
    )
    ori3.plt.close(figure)


def _validate_args(args):
    ori4._validate_args(args)
    if abs(float(args.gamma) - 1.0) > 1.0e-12:
        raise ValueError("ori_codex5 requires --gamma 1.")
    if float(args.curriculum_waypoint_bonus) != 0.0:
        raise ValueError(
            "ori_codex5 fixed milestones replace curriculum waypoint reward; "
            "set --curriculum-waypoint-bonus 0."
        )
    if len(args.ori5_joint_flatness_thresholds) != len(
        args.ori5_joint_milestone_bonuses
    ):
        raise ValueError(
            "ori5 joint thresholds and bonuses must have equal length."
        )


def train(args):
    """Train using ori3's unchanged optimiser with exception-safe hooks."""

    _validate_args(args)
    args.quality_combination = "flatness_only"
    args.target_deviation_cost_weight = 0.0
    args.flatness_progress_weight = 1.0
    args.target_progress_weight = 1.0
    args.terminate_wrong_flat = True
    args.reward_design = REWARD_DESIGN_NAME
    args.reward_schema_version = REWARD_SCHEMA_VERSION
    args.ori5_reward_parameters = reward_parameters(args)
    original_ori3_components = (
        ori3.make_environment,
        ori3.run_episode,
        ori3.aggregate_stochastic_evaluations,
    )
    original_mechanics_train = mechanics.train

    def train_with_ori5_architecture(selected_args):
        update = (
            "actor_critic"
            if selected_args.update_method == "actor_critic"
            else "ppo_actor_critic"
        )
        selected_args.architecture_name = (
            f"variable_mode_{REWARD_DESIGN_NAME}_{update}"
        )
        return original_mechanics_train(selected_args)

    ori3.make_environment = make_environment
    ori3.run_episode = run_episode
    ori3.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    mechanics.train = train_with_ori5_architecture
    try:
        output_directory, summary = ori3.train(args)
    finally:
        (
            ori3.make_environment,
            ori3.run_episode,
            ori3.aggregate_stochastic_evaluations,
        ) = original_ori3_components
        mechanics.train = original_mechanics_train

    history_path = Path(output_directory) / "training_history.csv"
    with open(history_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    recent = rows[-min(1000, len(rows)):]
    summary.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination": REWARD_DESIGN_NAME,
        "ori5_reward_parameters": reward_parameters(args),
        "last_1000_mean_near_miss_q": float(np.mean([
            float(row["near_miss_q"]) for row in recent
        ])),
        "last_1000_mean_near_miss_bonus": float(np.mean([
            float(row["near_miss_bonus"]) for row in recent
        ])),
        "last_1000_mean_ori5_milestone_reward": float(np.mean([
            float(row["ori5_total_milestone_reward"]) for row in recent
        ])),
        "last_1000_mean_ori5_terminal_correction_reward": float(np.mean([
            float(row["ori5_total_terminal_correction_reward"])
            for row in recent
        ])),
        "maximum_abs_ori5_reward_decomposition_error": float(max(
            abs(float(row["ori5_reward_decomposition_error"]))
            for row in rows
        )),
        "maximum_abs_ori5_step_reward_decomposition_error": float(max(
            abs(float(row[
                "ori5_maximum_abs_step_reward_decomposition_error"
            ]))
            for row in rows
        )),
        "ori5_nonfinite_reward_component_count": int(sum(
            int(float(row["ori5_nonfinite_reward_component_count"]))
            for row in rows
        )),
    })
    save_ori5_learning_curves(
        output_directory, rows, args.moving_average_window
    )

    config_path = Path(output_directory) / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "reward_design": REWARD_DESIGN_NAME,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "quality_combination_effective": REWARD_DESIGN_NAME,
        "ori5_reward_parameters": reward_parameters(args),
        "reward_component_identity": (
            "total_reward = flatness_endpoint + target_endpoint + milestones "
            "+ terminal_fixed + terminal_quality + terminal_speed "
            "+ terminal_correction"
        ),
        "maximum_abs_ori5_reward_decomposition_error": summary[
            "maximum_abs_ori5_reward_decomposition_error"
        ],
        "maximum_abs_ori5_step_reward_decomposition_error": summary[
            "maximum_abs_ori5_step_reward_decomposition_error"
        ],
        "ori5_nonfinite_reward_component_count": summary[
            "ori5_nonfinite_reward_component_count"
        ],
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(
        Path(output_directory) / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _ori5_argument_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--ori5-flatness-clip", type=float, default=2.0)
    parser.add_argument(
        "--ori5-spread-milestone-initial-factor", type=float, default=2.0
    )
    parser.add_argument(
        "--ori5-spread-milestone-bonus", type=float, default=0.9
    )
    parser.add_argument(
        "--ori5-joint-flatness-thresholds", type=float, nargs="+",
        default=[0.5, 0.1, 0.01],
    )
    parser.add_argument(
        "--ori5-joint-milestone-bonuses", type=float, nargs="+",
        default=[0.2, 0.3, 0.5],
    )
    parser.add_argument("--ori5-crossing-penalty", type=float, default=1.25)
    parser.add_argument("--ori5-wrong-flat-penalty", type=float, default=2.0)
    parser.add_argument("--ori5-timeout-penalty", type=float, default=1.0)
    parser.add_argument("--ori5-success-bonus", type=float, default=10.0)
    parser.add_argument("--ori5-quality-weight", type=float, default=1.0)
    parser.add_argument(
        "--ori5-success-speed-bonus-weight", type=float, default=0.5
    )
    return parser


def parse_args(argv=None):
    user_arguments = sys.argv[1:] if argv is None else list(argv)
    ori5_options, remaining = _ori5_argument_parser().parse_known_args(
        user_arguments
    )
    args = ori4.parse_args([
        "--quality-combination", "flatness_only",
        "--target-deviation-cost-weight", "0",
        "--gamma", "1",
        "--output-dir", "ori_codex5_common_reward_results",
        *remaining,
    ])
    for name, value in vars(ori5_options).items():
        setattr(args, name, value)
    args.flatness_reward_clip = float(args.ori5_flatness_clip)
    args.near_miss_bonus_weight = float(args.ori5_quality_weight)
    args.joint_success_bonus = 0.0
    args.wrong_flat_penalty = float(args.ori5_wrong_flat_penalty)
    args.timeout_penalty = float(args.ori5_timeout_penalty)
    args.terminate_wrong_flat = True
    return args


if __name__ == "__main__":
    train(parse_args())
