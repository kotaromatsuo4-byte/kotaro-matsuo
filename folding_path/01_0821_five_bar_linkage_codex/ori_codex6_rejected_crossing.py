"""Additive ori6 ablation that rejects self-crossing candidates.

The P0X2 reward and all non-crossing mechanics remain in :mod:`ori_codex6`.
This separate entry point preserves the byte identity and immutable evidence
hashes of ori6 while making the changed transition semantics explicit.

A rejected crossing consumes one step, restores the complete last-safe
mechanical state, retains past spread-frontier and joint-milestone ledgers,
and rewards the attempt with ``-crossing_penalty + q(last_safe)`` (including
the configured quality weight).  If it exhausts the horizon, timeout is the
terminal classification and its fixed penalty is added once.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex6 as ori6


CANDIDATE_NAME = "ori6_P0X2_rejected_crossing_continuation"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CROSSING_POLICY = "reject_candidate_and_continue"
CLI_FLAG = "--ori6-rejected-crossing-continuation"

_BASE_RUN_EPISODE = ori6.run_episode
_BASE_AGGREGATE_EVALUATIONS = ori6.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = ori6.reward_parameters
_BASE_SAVE_LEARNING_CURVES = ori6.save_ori6_learning_curves
_BASE_MAKE_ENVIRONMENT = ori6.make_environment


class RejectedCrossingContinuationTargetFoldingEnv(
    ori6.ContinuousSpreadPotentialTargetFoldingEnv
):
    """Ori6 environment with non-committing rejected crossings."""

    _SAFE_STATE_ATTRIBUTES = (
        "node",
        "previous_potential",
        "previous_ratio",
        "previous_compact_potential",
        "_ori5_flatness_metrics_cache",
        "_ori5_compactness_metrics_cache",
        "_ori5_mode_cache_key",
        "_ori5_mode_cache",
    )

    def _enable_rejected_crossing_continuation(self):
        self.ori6_crossing_policy = CROSSING_POLICY
        self._reset_rejected_crossing_episode_state()

    def reset(self):
        state = super().reset()
        if hasattr(self, "ori6_crossing_policy"):
            self._reset_rejected_crossing_episode_state()
        return state

    def _reset_rejected_crossing_episode_state(self):
        self.ori6_rejected_crossing_count = 0
        self.ori6_rejected_crossing_streak_count = 0
        self.ori6_consecutive_rejected_crossings = 0
        self.ori6_max_consecutive_rejected_crossings = 0
        self.ori6_escape_after_rejected_crossing_count = 0
        self.ori6_success_after_rejected_crossing = False

    @staticmethod
    def _copy_safe_value(value):
        return copy.deepcopy(value)

    def _safe_state_snapshot(self):
        return {
            name: self._copy_safe_value(getattr(self, name))
            for name in self._SAFE_STATE_ATTRIBUTES
            if hasattr(self, name)
        }

    def _restore_safe_state(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, self._copy_safe_value(value))

    @staticmethod
    def _candidate_metric_fields(info):
        return {
            f"ori6_rejected_candidate_{name}": info.get(name)
            for name in (
                "flatness_potential",
                "flatness_ratio",
                "flatness_rms",
                "spread",
                "spread_error",
                "target_deviation_energy",
                "requested_width",
                "accepted_width",
                "strain",
                "correction_norm",
                "actual_displacement",
                "correction_to_width_ratio",
                "actual_to_width_ratio",
            )
        }

    def _crossing_log_fields(
        self,
        *,
        rejected,
        streak_started=0,
        escape=0,
        candidate_fields=None,
        timeout_boundary=0,
        event_fixed_reward=0.0,
        event_quality_reward=0.0,
        event_timeout_reward=0.0,
    ):
        candidate_defaults = {
            f"ori6_rejected_candidate_{name}": None
            for name in (
                "flatness_potential",
                "flatness_ratio",
                "flatness_rms",
                "spread",
                "spread_error",
                "target_deviation_energy",
                "requested_width",
                "accepted_width",
                "strain",
                "correction_norm",
                "actual_displacement",
                "correction_to_width_ratio",
                "actual_to_width_ratio",
            )
        }
        if candidate_fields:
            candidate_defaults.update(candidate_fields)
        event_reward = float(
            event_fixed_reward
            + event_quality_reward
            + event_timeout_reward
        )
        return {
            "ori6_candidate": CANDIDATE_NAME,
            "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori6_crossing_policy": CROSSING_POLICY,
            "ori6_rejected_crossing": int(rejected),
            "ori6_rejected_crossing_streak_started": int(streak_started),
            "ori6_consecutive_rejected_crossings": int(
                self.ori6_consecutive_rejected_crossings
            ),
            "ori6_rejected_crossing_count_so_far": int(
                self.ori6_rejected_crossing_count
            ),
            "ori6_rejected_crossing_streak_count_so_far": int(
                self.ori6_rejected_crossing_streak_count
            ),
            "ori6_escape_after_rejected_crossing": int(escape),
            "ori6_escape_after_rejected_crossing_count_so_far": int(
                self.ori6_escape_after_rejected_crossing_count
            ),
            "ori6_rejected_crossing_timeout_boundary": int(
                timeout_boundary
            ),
            "ori6_rejected_crossing_fixed_reward": float(
                event_fixed_reward
            ),
            "ori6_rejected_crossing_quality_reward": float(
                event_quality_reward
            ),
            "ori6_rejected_crossing_timeout_reward": float(
                event_timeout_reward
            ),
            "ori6_rejected_crossing_reward": event_reward,
            "ori6_rejected_crossing_reward_decomposition_error": float(
                event_reward
                - event_fixed_reward
                - event_quality_reward
                - event_timeout_reward
            ),
            "ori6_rejected_candidate_base_reward": 0.0,
            "ori6_rejected_candidate_spread_frontier_reward": 0.0,
            "ori6_rejected_candidate_joint_milestone_reward": 0.0,
            **candidate_defaults,
        }

    def _replace_latest_reward_trace(self, info, extra_fields):
        if not self.ori5_reward_trace:
            raise RuntimeError(
                "Rejected-crossing step did not create an ori6 reward trace."
            )
        row = self.ori5_reward_trace[-1]
        for name in tuple(row):
            if name in info:
                row[name] = info[name]
        row.update(extra_fields)
        row["termination_reason"] = info["termination_reason"]
        row["after_state_valid"] = int(info["after_state_valid"])

    def _annotate_non_rejected_step(self, info):
        had_open_streak = self.ori6_consecutive_rejected_crossings > 0
        valid_accepted = bool(
            info.get("after_state_valid", 0)
            and abs(float(info.get("accepted_width", 0.0))) > 0.0
        )
        escape = int(had_open_streak and valid_accepted)
        if escape:
            self.ori6_escape_after_rejected_crossing_count += 1
        # A streak is a maximal consecutive run.  Its first non-rejected
        # successor either escapes immediately or leaves it unescaped.
        self.ori6_consecutive_rejected_crossings = 0
        if (
            str(info.get("termination_reason")) == "success"
            and self.ori6_rejected_crossing_count > 0
        ):
            self.ori6_success_after_rejected_crossing = True
        extra = self._crossing_log_fields(rejected=0, escape=escape)
        info.update(extra)
        self._replace_latest_reward_trace(info, extra)

    def _convert_crossing_to_rejection(
        self,
        info,
        safe_snapshot,
        safe_metrics,
        qF_before,
        qD_before,
        q_before,
    ):
        candidate_fields = self._candidate_metric_fields(info)
        self._restore_safe_state(safe_snapshot)

        streak_started = int(
            self.ori6_consecutive_rejected_crossings == 0
        )
        if streak_started:
            self.ori6_rejected_crossing_streak_count += 1
        self.ori6_rejected_crossing_count += 1
        self.ori6_consecutive_rejected_crossings += 1
        self.ori6_max_consecutive_rejected_crossings = max(
            self.ori6_max_consecutive_rejected_crossings,
            self.ori6_consecutive_rejected_crossings,
        )

        timeout_boundary = bool(self.step_count >= self.max_steps)
        crossing_fixed = -float(self.ori5_crossing_penalty)
        crossing_quality = float(self.ori5_quality_weight * q_before)
        timeout_fixed = (
            -float(self.ori5_timeout_penalty) if timeout_boundary else 0.0
        )
        terminal_fixed = float(crossing_fixed + timeout_fixed)
        reward = float(terminal_fixed + crossing_quality)
        termination_reason = "timeout" if timeout_boundary else "running"

        frontier = self.spread_frontier_components(safe_metrics)
        bounded_flatness = float(np.clip(
            safe_metrics["flatness_ratio"], 0.0, self.ori5_flatness_clip
        ))
        target_energy = float(
            (safe_metrics["spread"] - self.target_spread) ** 2
        )

        # Ordinary fields describe the effective safe state.  Candidate
        # geometry is preserved only under ori6_rejected_candidate_*.
        info.update(safe_metrics)
        info.update({
            "self_crossing": False,
            "self_crossing_candidate": True,
            "correction_failed": False,
            "no_modes": False,
            "timeout": timeout_boundary,
            "termination_reason": termination_reason,
            "after_state_valid": 0,
            "success": False,
            "compact_success": False,
            "flat_reached": False,
            "wrong_flat": False,
            "wrong_flat_current": False,
            "accepted_width": 0.0,
            "correction_norm": 0.0,
            "actual_displacement": 0.0,
            "correction_to_width_ratio": 0.0,
            "actual_to_width_ratio": 0.0,
            "bounded_flatness_before": bounded_flatness,
            "bounded_flatness_after": bounded_flatness,
            "flatness_progress_reward": 0.0,
            "spread_before": float(safe_metrics["spread"]),
            "target_deviation_energy_before": target_energy,
            "target_deviation_energy_after": target_energy,
            "target_progress_reward": 0.0,
            "normalised_path_increment": 0.0,
            "integrated_target_deviation_cost": 0.0,
            "target_deviation_path_reward": 0.0,
            "dense_reward": 0.0,
            "terminal_reward_component": reward,
            "near_miss_qF": float(qF_before),
            "near_miss_qD": float(qD_before),
            "near_miss_q": float(q_before),
            "near_miss_bonus": crossing_quality,
            "near_miss_terminal_eligible": 1,
            "ori6_flatness_progress_reward": 0.0,
            "ori6_target_progress_reward": 0.0,
            "ori6_base_reward": 0.0,
            "ori6_spread_progress_before": float(frontier["z_D"]),
            "ori6_spread_progress_after": float(frontier["z_D"]),
            "ori6_flatness_gate_before": float(frontier["g_F"]),
            "ori6_flatness_gate_after": float(frontier["g_F"]),
            "ori6_spread_potential_before": float(frontier["phi_D"]),
            "ori6_spread_potential_after": float(frontier["phi_D"]),
            "ori6_spread_frontier_reward": 0.0,
            "ori6_hard_spread_milestone_reward": 0.0,
            "ori6_joint_milestone_rewards": [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ],
            "ori6_joint_milestone_reward": 0.0,
            "ori6_discovery_reward": 0.0,
            "ori6_terminal_fixed_reward": terminal_fixed,
            "ori6_terminal_quality_reward": crossing_quality,
            "ori6_terminal_speed_reward": 0.0,
            "ori6_flatness_rollback_reward": 0.0,
            "ori6_frontier_rollback_reward": 0.0,
            "ori6_joint_milestone_rollback_reward": 0.0,
            "ori6_terminal_correction_reward": 0.0,
            "ori6_terminal_reward": reward,
            "ori6_qF": float(qF_before),
            "ori6_qD": float(qD_before),
            "ori6_q": float(q_before),
            "ori6_quality_source": "last_safe_before_rejected_crossing",
            "ori6_reward_decomposition_error": 0.0,
            "ori6_nonfinite_reward_component_count": int(
                not np.isfinite(reward)
            ),
            # Aliases consumed by ori6's unchanged rollout shell.
            "ori5_flatness_progress_reward": 0.0,
            "ori5_target_progress_reward": 0.0,
            "ori5_base_reward": 0.0,
            "ori5_spread_milestone_reward": 0.0,
            "ori5_joint_milestone_rewards": [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ],
            "ori5_joint_milestone_reward": 0.0,
            "ori5_milestone_reward": 0.0,
            "ori5_terminal_fixed_reward": terminal_fixed,
            "ori5_terminal_quality_reward": crossing_quality,
            "ori5_terminal_speed_reward": 0.0,
            "ori5_terminal_correction_reward": 0.0,
            "ori5_terminal_reward": reward,
            "ori5_qF": float(qF_before),
            "ori5_qD": float(qD_before),
            "ori5_q": float(q_before),
            "ori5_quality_source": "last_safe_before_rejected_crossing",
            "ori5_reward_decomposition_error": 0.0,
            "ori5_nonfinite_reward_component_count": int(
                not np.isfinite(reward)
            ),
        })
        extra = self._crossing_log_fields(
            rejected=1,
            streak_started=streak_started,
            candidate_fields=candidate_fields,
            timeout_boundary=int(timeout_boundary),
            event_fixed_reward=crossing_fixed,
            event_quality_reward=crossing_quality,
            event_timeout_reward=timeout_fixed,
        )
        info.update(extra)
        self._replace_latest_reward_trace(info, extra)
        return self.get_state(), reward, timeout_boundary, info

    def step(self, physical_direction, requested_width, mode_coefficients):
        safe_snapshot = self._safe_state_snapshot()
        safe_metrics = self.compactness_metrics()
        qF_before, qD_before, q_before = self.near_miss_quality(
            safe_metrics["flatness_ratio"], safe_metrics["spread_error"]
        )
        next_state, reward, done, info = super().step(
            physical_direction, requested_width, mode_coefficients
        )
        if bool(info.get("self_crossing", False)):
            return self._convert_crossing_to_rejection(
                info,
                safe_snapshot,
                safe_metrics,
                qF_before,
                qD_before,
                q_before,
            )
        self._annotate_non_rejected_step(info)
        return next_state, reward, done, info


CommonRewardTargetFoldingEnv = RejectedCrossingContinuationTargetFoldingEnv


def make_environment(args, node, member, fix):
    """Construct base ori6 exactly, then add the state-free policy subclass."""

    environment = _BASE_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.__class__ = RejectedCrossingContinuationTargetFoldingEnv
    environment._enable_rejected_crossing_continuation()
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in actions))

    rejected_count = int(env.ori6_rejected_crossing_count)
    streak_count = int(env.ori6_rejected_crossing_streak_count)
    escape_count = int(env.ori6_escape_after_rejected_crossing_count)
    summary.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_rejected_crossing_count": rejected_count,
        "ori6_rejected_crossing_rate": float(
            rejected_count / max(int(summary["steps"]), 1)
        ),
        "ori6_rejected_crossing_streak_count": streak_count,
        "ori6_max_consecutive_rejected_crossings": int(
            env.ori6_max_consecutive_rejected_crossings
        ),
        "ori6_escape_after_rejected_crossing_count": escape_count,
        "ori6_rejected_crossing_escape_rate": float(
            escape_count / max(streak_count, 1)
        ),
        "ori6_unescaped_rejected_crossing_streak_count": int(
            streak_count - escape_count
        ),
        "ori6_success_after_rejected_crossing": int(
            env.ori6_success_after_rejected_crossing
        ),
        "ori6_total_rejected_crossing_reward": total(
            "ori6_rejected_crossing_reward"
        ),
        "ori6_maximum_abs_rejected_crossing_reward_"
        "decomposition_error": float(max(
            [abs(float(row.get(
                "ori6_rejected_crossing_reward_decomposition_error", 0.0
            ))) for row in actions]
            or [0.0]
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE_EVALUATIONS(
        evaluations, episode=episode
    )
    if result is None:
        return None
    for source, destination in {
        "ori6_rejected_crossing_count": (
            "mean_ori6_rejected_crossing_count"
        ),
        "ori6_rejected_crossing_rate": (
            "mean_ori6_rejected_crossing_rate"
        ),
        "ori6_rejected_crossing_streak_count": (
            "mean_ori6_rejected_crossing_streak_count"
        ),
        "ori6_rejected_crossing_escape_rate": (
            "mean_ori6_rejected_crossing_escape_rate"
        ),
        "ori6_success_after_rejected_crossing": (
            "success_after_rejected_crossing_rate"
        ),
    }.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(args)
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "crossing_transition": {
            "policy": CROSSING_POLICY,
            "candidate_committed": False,
            "step_count_increments": True,
            "ordinary_rejected_reward_formula": (
                "-crossing_penalty + quality_weight * q(last_safe)"
            ),
            "horizon_rejected_reward_formula": (
                "-crossing_penalty + quality_weight * q(last_safe) "
                "- timeout_penalty"
            ),
            "horizon_priority": (
                "correction_failed; else reject crossing candidate; if "
                "step_count >= max_steps classify terminal as timeout; "
                "crossed candidate cannot be success or wrong_flat"
            ),
            "rejected_candidate_endpoint_reward": 0.0,
            "rejected_candidate_spread_frontier_reward": 0.0,
            "rejected_candidate_new_milestone_reward": 0.0,
            "past_frontier_and_joint_ledgers_retained": True,
            "escape_definition": (
                "one escape per maximal consecutive rejected-crossing "
                "streak when its immediately following non-rejected action "
                "is a valid accepted transition; unresolved terminal counts "
                "as zero"
            ),
        },
        "collision_quality_reference": (
            "last_safe_before_rejected_crossing"
        ),
    })
    parameters["rollback"] = dict(parameters["rollback"])
    parameters["rollback"]["self_crossing"] = []
    return parameters


def save_learning_curves(output_directory, rows, window):
    _BASE_SAVE_LEARNING_CURVES(output_directory, rows, window)
    if not rows:
        return
    episodes = np.asarray([float(row["episode"]) for row in rows])
    figure, axes = ori3.plt.subplots(1, 3, figsize=(14.5, 4.0))
    fields = (
        (
            "ori6_rejected_crossing_rate",
            "Rejected attempts / episode steps",
            "tab:red",
        ),
        (
            "ori6_rejected_crossing_escape_rate",
            "Escape rate per reject streak",
            "tab:green",
        ),
        (
            "ori6_rejected_crossing_count",
            "Rejected attempts per episode",
            "tab:purple",
        ),
    )
    for axis, (field, title, color) in zip(axes, fields):
        values = np.asarray([float(row[field]) for row in rows])
        axis.plot(episodes, values, color="0.82", linewidth=0.5)
        x, average = ori6._moving_average(episodes, values, window)
        if len(x):
            axis.plot(x, average, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "ori6_rejected_crossing_learning_curve.png",
        dpi=180,
    )
    ori3.plt.close(figure)


@contextmanager
def _patched_ori6_training_runtime():
    original = (
        ori6.make_environment,
        ori6.run_episode,
        ori6.aggregate_stochastic_evaluations,
        ori6.reward_parameters,
        ori6.save_ori6_learning_curves,
    )
    ori6.make_environment = make_environment
    ori6.run_episode = run_episode
    ori6.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    ori6.reward_parameters = reward_parameters
    ori6.save_ori6_learning_curves = save_learning_curves
    try:
        yield
    finally:
        (
            ori6.make_environment,
            ori6.run_episode,
            ori6.aggregate_stochastic_evaluations,
            ori6.reward_parameters,
            ori6.save_ori6_learning_curves,
        ) = original


def train(args):
    ori6._validate_args(args)
    if not bool(args.ori6_rejected_crossing_continuation):
        raise ValueError(f"Candidate requires explicit {CLI_FLAG} opt-in.")
    parameters = reward_parameters(args)
    with _patched_ori6_training_runtime():
        output_directory, summary = ori6.train(args)

    output_directory = Path(output_directory)
    with open(
        output_directory / "training_history.csv",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    recent = rows[-min(1000, len(rows)):]
    summary.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_reward_parameters": parameters,
        "last_1000_mean_ori6_rejected_crossing_count": float(np.mean([
            float(row["ori6_rejected_crossing_count"]) for row in recent
        ])),
        "last_1000_mean_ori6_rejected_crossing_rate": float(np.mean([
            float(row["ori6_rejected_crossing_rate"]) for row in recent
        ])),
        "last_1000_mean_ori6_rejected_crossing_escape_rate": float(np.mean([
            float(row["ori6_rejected_crossing_escape_rate"])
            for row in recent
        ])),
        "last_1000_ori6_success_after_rejected_crossing_rate": float(
            np.mean([
                float(row["ori6_success_after_rejected_crossing"])
                for row in recent
            ])
        ),
    })
    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_reward_parameters": parameters,
        "crossing_transition_policy": parameters["crossing_transition"],
        "candidate_entrypoint": Path(__file__).name,
        "base_ori6_source_is_unmodified": True,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(
        output_directory / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _option_was_supplied(arguments, option):
    return any(
        str(argument) == option
        or str(argument).startswith(f"{option}=")
        for argument in arguments
    )


def _candidate_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, action="store_true", required=True)
    return parser


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    candidate, remaining = _candidate_parser().parse_known_args(arguments)
    args = ori6.parse_args(remaining)
    args.ori6_rejected_crossing_continuation = bool(
        candidate.ori6_rejected_crossing_continuation
    )
    args.ori6_crossing_policy = CROSSING_POLICY
    if not _option_was_supplied(arguments, "--ori5-crossing-penalty"):
        args.ori5_crossing_penalty = 2.0
    if not _option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex6_rejected_crossing_results"
    return args


if __name__ == "__main__":
    train(parse_args())
