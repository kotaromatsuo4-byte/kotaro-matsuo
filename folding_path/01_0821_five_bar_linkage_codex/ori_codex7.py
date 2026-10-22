"""Codex7 fixed-penalty terminal-crossing ablation.

This additive entry point freezes the successful ori6 spectral reward on every
accepted transition and changes only the reward assigned when an action
proposal would self-intersect.  The rejected candidate is never committed;
the last-safe state is returned, one proposal is consumed, and the episode
terminates.  The two pre-registered arms isolate the fixed term exactly::

    no_fixed:        r_cross = q_safe
    historical_1p25: r_cross = -1.25 + q_safe

``q_safe`` is the existing last-safe near-miss quality, without another
quality-weight multiplication.  Candidate geometry earns no flatness, spread,
frontier, milestone, or spectral credit, while every ledger accumulated by the
valid prefix is retained.  A crossing on the horizon is still a crossing and
does not receive a timeout penalty.

Training remains the repository's episode-separated Monte-Carlo Actor-Critic.
The originally sampled action and log probability stay in the trace because
the score-function actor update must be evaluated on the action that actually
produced the terminal crossing event.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex6 as ori6
import ori_codex6_rejected_crossing as rejected
import ori_codex6_spectral_subspace as spectral


CANDIDATE_NAME = "ori7_spectral_terminal_crossing_fixed_penalty_ablation"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CROSSING_POLICY = "reject_candidate_restore_last_safe_and_terminate"
CLI_FLAG = "--ori7-crossing-terminal-mode"
MODE_NO_FIXED = "no_fixed"
MODE_HISTORICAL_1P25 = "historical_1p25"
CROSSING_MODES = (MODE_NO_FIXED, MODE_HISTORICAL_1P25)
HISTORICAL_FIXED_PENALTY = 1.25
FROZEN_SPECTRAL_WEIGHT = 0.05

REWARD_FORMULA = {
    "accepted_transition": {
        "flatness": "existing ori6 flatness progress potential",
        "spread": "existing target progress plus continuous spread frontier",
        "joint_milestones": "existing ori6 joint milestones",
        "spectral": "0.05 * existing reset-referenced spectral potential difference",
        "terminal": "existing ori6 terminal components",
    },
    "crossing_terminal": {
        MODE_NO_FIXED: "q_safe",
        MODE_HISTORICAL_1P25: "-1.25 + q_safe",
        "q_safe": "existing last-safe near-miss q without weight reapplication",
        "candidate_components": {
            "flatness": 0.0,
            "target_progress": 0.0,
            "spread_frontier": 0.0,
            "joint_milestones": 0.0,
            "spectral": 0.0,
        },
        "timeout_component_even_at_horizon": 0.0,
    },
    "learning_rule": (
        "episode-separated Monte-Carlo Actor-Critic; retain sampled action "
        "and log_probability for the score-function update"
    ),
}

TRANSITION_FORMULA = {
    "crossing_candidate": "invalid and never committed",
    "returned_state": "exact last-safe state and caches",
    "proposal_cost": "one environment step",
    "terminal": True,
    "termination_reason": "self_crossing",
    "terminal_priority": (
        "self_crossing > correction_failed > success > wrong_flat > "
        "no_modes > timeout"
    ),
    "horizon_priority": "crossing; no timeout reward is added",
    "sampled_action": "retained unchanged in rollout trace",
    "valid_prefix_ledgers": "retained unchanged, including spectral cumulative reward",
    "post_terminal_step": "forbidden until reset",
}


def _formula_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


REWARD_FORMULA_SHA256 = _formula_hash(REWARD_FORMULA)
TRANSITION_FORMULA_SHA256 = _formula_hash(TRANSITION_FORMULA)

_BASE_MAKE_ENVIRONMENT = spectral.make_environment
_BASE_RUN_EPISODE = spectral.run_episode
_BASE_AGGREGATE = spectral.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = spectral.reward_parameters


def fixed_component(mode):
    if mode == MODE_NO_FIXED:
        return 0.0
    if mode == MODE_HISTORICAL_1P25:
        return -HISTORICAL_FIXED_PENALTY
    raise ValueError(f"unknown Codex7 crossing mode: {mode!r}")


def crossing_terminal_selected(info):
    """Crossing outranks every competing terminal classification."""
    return bool(info.get("self_crossing", False))


class Codex7TerminalCrossingEnv(spectral.SpectralSubspaceTargetFoldingEnv):
    """Frozen spectral parent with terminal, non-committing crossings."""

    def _enable_ori7(self, mode):
        if mode not in CROSSING_MODES:
            raise ValueError(f"unknown Codex7 crossing mode: {mode!r}")
        self.ori7_crossing_terminal_mode = mode
        self.ori6_crossing_policy = CROSSING_POLICY
        self._ori7_terminal_crossing_emitted = False

    def reset(self):
        state = super().reset()
        if hasattr(self, "ori7_crossing_terminal_mode"):
            self._ori7_terminal_crossing_emitted = False
        return state

    def _ori7_common_fields(self, terminal):
        return {
            "ori7_candidate": CANDIDATE_NAME,
            "ori7_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori7_crossing_policy": CROSSING_POLICY,
            "ori7_crossing_terminal_mode": self.ori7_crossing_terminal_mode,
            "ori7_reward_formula_sha256": REWARD_FORMULA_SHA256,
            "ori7_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
            "ori7_frozen_spectral_weight": FROZEN_SPECTRAL_WEIGHT,
            "ori7_terminal_crossing": int(terminal),
        }

    def _annotate_non_rejected_step(self, info):
        super()._annotate_non_rejected_step(info)
        fields = self._ori7_common_fields(False)
        fields.update({
            "ori7_crossing_fixed_reward": 0.0,
            "ori7_crossing_quality_reward": 0.0,
            "ori7_crossing_reward": 0.0,
            "ori7_crossing_reward_decomposition_error": 0.0,
            "ori7_crossing_nonfinite_component_count": 0,
            "ori7_sampled_action_retained": 0,
            "ori7_valid_prefix_ledgers_retained": 0,
        })
        info.update(fields)
        self._replace_latest_reward_trace(info, fields)

    def _convert_crossing_to_rejection(
        self,
        info,
        safe_snapshot,
        safe_metrics,
        qF_before,
        qD_before,
        q_before,
    ):
        if self._ori7_terminal_crossing_emitted:
            raise RuntimeError("Codex7 environment stepped after terminal crossing")

        # Delegate state restoration, action-proposal accounting, candidate
        # diagnostics, and rejection counters to the immutable parent.  Its
        # continuation/horizon reward is overwritten completely below.
        state, _, _, info = rejected.RejectedCrossingContinuationTargetFoldingEnv._convert_crossing_to_rejection(
            self,
            info,
            safe_snapshot,
            safe_metrics,
            qF_before,
            qD_before,
            q_before,
        )
        self._ori7_terminal_crossing_emitted = True

        fixed_reward = float(fixed_component(self.ori7_crossing_terminal_mode))
        quality_reward = float(q_before)
        reward = float(fixed_reward + quality_reward)
        nonfinite_count = int(sum(not np.isfinite(value) for value in (
            fixed_reward, quality_reward, reward
        )))
        decomposition_error = float(reward - fixed_reward - quality_reward)
        frontier = self.spread_frontier_components(safe_metrics)
        bounded_flatness = float(np.clip(
            safe_metrics["flatness_ratio"], 0.0, self.ori5_flatness_clip
        ))
        target_energy = float(
            (safe_metrics["spread"] - self.target_spread) ** 2
        )

        # Ordinary geometry describes the restored last-safe state.  Crossed
        # candidate diagnostics remain under ori6_rejected_candidate_* only.
        info.update(safe_metrics)
        info.update({
            "self_crossing": True,
            "self_crossing_candidate": True,
            "correction_failed": False,
            "no_modes": False,
            "timeout": False,
            "termination_reason": "self_crossing",
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
            "near_miss_q": quality_reward,
            "near_miss_bonus": quality_reward,
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
            "ori6_terminal_fixed_reward": fixed_reward,
            "ori6_terminal_quality_reward": quality_reward,
            "ori6_terminal_speed_reward": 0.0,
            "ori6_flatness_rollback_reward": 0.0,
            "ori6_frontier_rollback_reward": 0.0,
            "ori6_joint_milestone_rollback_reward": 0.0,
            "ori6_terminal_correction_reward": 0.0,
            "ori6_terminal_reward": reward,
            "ori6_qF": float(qF_before),
            "ori6_qD": float(qD_before),
            "ori6_q": quality_reward,
            "ori6_quality_source": "last_safe_before_terminal_crossing",
            "ori6_reward_decomposition_error": decomposition_error,
            "ori6_nonfinite_reward_component_count": nonfinite_count,
            "ori5_flatness_progress_reward": 0.0,
            "ori5_target_progress_reward": 0.0,
            "ori5_base_reward": 0.0,
            "ori5_spread_milestone_reward": 0.0,
            "ori5_joint_milestone_rewards": [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ],
            "ori5_joint_milestone_reward": 0.0,
            "ori5_milestone_reward": 0.0,
            "ori5_terminal_fixed_reward": fixed_reward,
            "ori5_terminal_quality_reward": quality_reward,
            "ori5_terminal_speed_reward": 0.0,
            "ori5_terminal_correction_reward": 0.0,
            "ori5_terminal_reward": reward,
            "ori5_qF": float(qF_before),
            "ori5_qD": float(qD_before),
            "ori5_q": quality_reward,
            "ori5_quality_source": "last_safe_before_terminal_crossing",
            "ori5_reward_decomposition_error": decomposition_error,
            "ori5_nonfinite_reward_component_count": nonfinite_count,
            # Parent rejection aliases are overwritten so neither its fixed
            # penalty nor its horizon timeout can leak into learning/logs.
            "ori6_rejected_crossing_timeout_boundary": 0,
            "ori6_rejected_crossing_fixed_reward": fixed_reward,
            "ori6_rejected_crossing_quality_reward": quality_reward,
            "ori6_rejected_crossing_timeout_reward": 0.0,
            "ori6_rejected_crossing_reward": reward,
            "ori6_rejected_crossing_reward_decomposition_error": decomposition_error,
        })
        fields = self._ori7_common_fields(True)
        fields.update({
            "ori7_crossing_last_safe_qF": float(qF_before),
            "ori7_crossing_last_safe_qD": float(qD_before),
            "ori7_crossing_last_safe_q": quality_reward,
            "ori7_crossing_fixed_reward": fixed_reward,
            "ori7_crossing_quality_reward": quality_reward,
            "ori7_crossing_timeout_reward": 0.0,
            "ori7_crossing_reward": reward,
            "ori7_crossing_reward_decomposition_error": decomposition_error,
            "ori7_crossing_nonfinite_component_count": nonfinite_count,
            "ori7_sampled_action_retained": 1,
            "ori7_valid_prefix_ledgers_retained": 1,
            "ori7_candidate_flatness_reward": 0.0,
            "ori7_candidate_target_progress_reward": 0.0,
            "ori7_candidate_spread_frontier_reward": 0.0,
            "ori7_candidate_joint_milestone_reward": 0.0,
            "ori7_candidate_spectral_reward": 0.0,
        })
        info.update(fields)
        self._replace_latest_reward_trace(info, fields)
        return state, reward, True, info

    def step(self, physical_direction, requested_width, mode_coefficients):
        if self._ori7_terminal_crossing_emitted:
            raise RuntimeError("Codex7 environment stepped after terminal crossing")

        metrics_before = self.spectral_subspace_metrics()
        cumulative_before = float(self.ori6_spectral_cumulative_reward)
        safe_snapshot = self._safe_state_snapshot()
        safe_metrics = self.compactness_metrics()
        qF_before, qD_before, q_before = self.near_miss_quality(
            safe_metrics["flatness_ratio"], safe_metrics["spread_error"]
        )

        core_state, core_reward, done, info = ori6.ContinuousSpreadPotentialTargetFoldingEnv.step(
            self, physical_direction, requested_width, mode_coefficients
        )
        if crossing_terminal_selected(info):
            core_state, core_reward, done, info = self._convert_crossing_to_rejection(
                info, safe_snapshot, safe_metrics, qF_before, qD_before, q_before
            )
        else:
            self._annotate_non_rejected_step(info)

        terminal_crossing = bool(info.get("ori7_terminal_crossing", 0))
        rejected_crossing = bool(info.get("ori6_rejected_crossing", 0))
        metrics_after = (
            copy.deepcopy(metrics_before)
            if rejected_crossing else self.spectral_subspace_metrics()
        )
        accepted_valid = bool(
            not rejected_crossing
            and info.get("after_state_valid", 0)
            and abs(float(info.get("accepted_width", 0.0))) > 0.0
        )
        if accepted_valid:
            potential_difference = float(
                metrics_after["potential"] - metrics_before["potential"]
            )
            spectral_progress_reward = float(
                self.ori6_spectral_weight * potential_difference
            )
        else:
            potential_difference = 0.0
            spectral_progress_reward = 0.0

        provisional_cumulative = float(cumulative_before + spectral_progress_reward)
        termination_reason = str(info.get("termination_reason", "running"))
        rollback_reason = (
            "none" if terminal_crossing else
            termination_reason if termination_reason in spectral.SPECTRAL_ROLLBACK_REASONS
            else "none"
        )
        if rollback_reason != "none":
            spectral_rollback_reward = -provisional_cumulative
            cumulative_after = 0.0
            self.ori6_spectral_rollback_count += 1
        else:
            spectral_rollback_reward = 0.0
            cumulative_after = provisional_cumulative
        spectral_reward = float(spectral_progress_reward + spectral_rollback_reward)
        self.ori6_spectral_cumulative_reward = float(cumulative_after)
        self.ori6_spectral_current_potential = float(metrics_after["potential"])
        if accepted_valid:
            self.ori6_spectral_maximum_potential = max(
                self.ori6_spectral_maximum_potential, float(metrics_after["potential"])
            )
            self.ori6_spectral_maximum_delta = max(
                self.ori6_spectral_maximum_delta, float(metrics_after["delta"])
            )
            self.ori6_spectral_maximum_rho = max(
                self.ori6_spectral_maximum_rho, float(metrics_after["rho"])
            )

        # A terminal crossing already contains its complete frozen ablation
        # reward.  Its spectral transition and rollback components are both
        # exactly zero; accepted paths preserve the parent's float ordering.
        reward = (
            core_reward if terminal_crossing
            else core_reward if self.ori6_spectral_weight == 0.0
            else float(core_reward + spectral_reward)
        )
        audited = (
            metrics_before["rho"], metrics_before["delta"], metrics_before["potential"],
            metrics_after["rho"], metrics_after["delta"], metrics_after["potential"],
            potential_difference, spectral_progress_reward, spectral_rollback_reward,
            spectral_reward, cumulative_before, cumulative_after, core_reward, reward,
        )
        nonfinite_count = int(sum(not np.isfinite(value) for value in audited))
        decomposition_error = float(
            reward - core_reward - spectral_progress_reward - spectral_rollback_reward
        )
        spectral_fields = {
            "ori6_spectral_candidate": spectral.CANDIDATE_NAME,
            "ori6_spectral_candidate_schema_version": spectral.CANDIDATE_SCHEMA_VERSION,
            "ori6_spectral_formula_sha256": spectral.SPECTRAL_FORMULA_SHA256,
            "ori6_spectral_weight": float(self.ori6_spectral_weight),
            "ori6_spectral_relative_epsilon": float(self.ori6_spectral_relative_epsilon),
            "ori6_spectral_subspace_dimension": int(self.flatness_subspace_dimension),
            "ori6_spectral_accepted_valid_transition": int(accepted_valid),
            "ori6_spectral_rejected_crossing_zeroed": int(rejected_crossing),
            "ori6_spectral_potential_difference": potential_difference,
            "ori6_spectral_progress_reward": spectral_progress_reward,
            "ori6_spectral_rollback_reward": float(spectral_rollback_reward),
            "ori6_spectral_reward": spectral_reward,
            "ori6_spectral_core_reward": float(core_reward),
            "ori6_spectral_cumulative_reward_before": cumulative_before,
            "ori6_spectral_cumulative_reward_after": float(cumulative_after),
            "ori6_spectral_rollback_reason": rollback_reason,
            "ori6_spectral_reward_decomposition_error": decomposition_error,
            "ori6_spectral_nonfinite_component_count": nonfinite_count,
            **self._metric_log_fields("before", metrics_before),
            **self._metric_log_fields("after", metrics_after),
        }
        info.update(spectral_fields)
        if not self.ori5_reward_trace:
            raise RuntimeError("Codex7 step did not create a parent reward trace")
        self.ori5_reward_trace[-1].update(spectral_fields)
        return core_state, reward, bool(done), info


CommonRewardTargetFoldingEnv = Codex7TerminalCrossingEnv


def make_environment(args, node, member, fix):
    environment = _BASE_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.__class__ = Codex7TerminalCrossingEnv
    environment._enable_ori7(args.ori7_crossing_terminal_mode)
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    terminal = [row for row in actions if int(row.get("ori7_terminal_crossing", 0))]
    summary.update({
        "ori7_candidate": CANDIDATE_NAME,
        "ori7_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori7_crossing_policy": CROSSING_POLICY,
        "ori7_crossing_terminal_mode": env.ori7_crossing_terminal_mode,
        "ori7_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori7_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "ori7_terminal_crossing_count": len(terminal),
        "ori7_terminal_crossing_reward": float(sum(
            float(row.get("ori7_crossing_reward", 0.0)) for row in terminal
        )),
        "ori7_maximum_abs_crossing_reward_decomposition_error": float(max(
            [abs(float(row.get("ori7_crossing_reward_decomposition_error", 0.0)))
             for row in actions] or [0.0]
        )),
        "ori7_crossing_nonfinite_component_count": int(sum(
            int(row.get("ori7_crossing_nonfinite_component_count", 0))
            for row in actions
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE(evaluations, episode=episode)
    if result is None:
        return None
    for source, destination in {
        "ori7_terminal_crossing_count": "mean_ori7_terminal_crossing_count",
        "ori7_terminal_crossing_reward": "mean_ori7_terminal_crossing_reward",
    }.items():
        result[destination] = float(np.mean([float(row[source]) for row in evaluations]))
    result["maximum_abs_ori7_crossing_reward_decomposition_error"] = float(max(
        abs(float(row["ori7_maximum_abs_crossing_reward_decomposition_error"]))
        for row in evaluations
    ))
    result["total_ori7_crossing_nonfinite_component_count"] = int(sum(
        int(row["ori7_crossing_nonfinite_component_count"]) for row in evaluations
    ))
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(args)
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": spectral.CANDIDATE_NAME,
        "crossing_policy": CROSSING_POLICY,
        "crossing_terminal_mode": args.ori7_crossing_terminal_mode,
        "crossing_fixed_reward": fixed_component(args.ori7_crossing_terminal_mode),
        "crossing_quality_reward": "q(last_safe), without quality-weight reapplication",
        "reward_formula": copy.deepcopy(REWARD_FORMULA),
        "reward_formula_sha256": REWARD_FORMULA_SHA256,
        "transition_formula": copy.deepcopy(TRANSITION_FORMULA),
        "transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "spectral_weight_frozen": FROZEN_SPECTRAL_WEIGHT,
        "update_method": "episode_separated_monte_carlo_actor_critic",
        "score_function_action_contract": "retain sampled action and log_probability",
    })
    return parameters


@contextmanager
def _patched_spectral_training_runtime():
    original = (
        spectral.make_environment,
        spectral.run_episode,
        spectral.aggregate_stochastic_evaluations,
        spectral.reward_parameters,
    )
    spectral.make_environment = make_environment
    spectral.run_episode = run_episode
    spectral.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    spectral.reward_parameters = reward_parameters
    try:
        yield
    finally:
        (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
            spectral.reward_parameters,
        ) = original


def train(args):
    _validate_args(args)
    _bind_checkpoint_metadata(args)
    parameters = reward_parameters(args)
    with _patched_spectral_training_runtime():
        output_directory, summary = spectral.train(args)
    output_directory = Path(output_directory)
    with (output_directory / "training_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary.update({
        "ori7_candidate": CANDIDATE_NAME,
        "ori7_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori7_crossing_terminal_mode": args.ori7_crossing_terminal_mode,
        "ori7_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori7_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "ori7_reward_parameters": parameters,
        "maximum_abs_ori7_crossing_reward_decomposition_error": float(max(
            [abs(float(row.get(
                "ori7_maximum_abs_crossing_reward_decomposition_error", 0.0
            ))) for row in rows] or [0.0]
        )),
    })
    config_path = output_directory / "run_config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config.update({
        "ori7_candidate": CANDIDATE_NAME,
        "ori7_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori7_crossing_terminal_mode": args.ori7_crossing_terminal_mode,
        "ori7_reward_parameters": parameters,
        "reward_formula": copy.deepcopy(REWARD_FORMULA),
        "reward_formula_sha256": REWARD_FORMULA_SHA256,
        "transition_formula": copy.deepcopy(TRANSITION_FORMULA),
        "transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "candidate_entrypoint": Path(__file__).name,
        "base_ori6_authorities_unmodified": True,
    })
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_directory, summary


def _validate_args(args):
    if args.ori7_crossing_terminal_mode not in CROSSING_MODES:
        raise ValueError(f"{CLI_FLAG} must be one of {CROSSING_MODES}")
    spectral._validate_spectral_args(args)
    if float(args.ori6_spectral_weight) != FROZEN_SPECTRAL_WEIGHT:
        raise ValueError("Codex7 freezes --ori6-spectral-weight at exactly 0.05")
    if str(args.update_method) != "actor_critic":
        raise ValueError("Codex7 requires episode-separated Monte-Carlo Actor-Critic")
    if float(args.ori5_quality_weight) != 1.0:
        raise ValueError("Codex7 freezes --ori5-quality-weight at exactly 1")
    if float(args.ori5_crossing_penalty) != HISTORICAL_FIXED_PENALTY:
        raise ValueError("Codex7 freezes --ori5-crossing-penalty metadata at exactly 1.25")


def _bind_checkpoint_metadata(args):
    """Bind authority fields before the base trainer writes run_config.json."""
    args.ori7_candidate = CANDIDATE_NAME
    args.ori7_candidate_schema_version = CANDIDATE_SCHEMA_VERSION
    args.ori7_crossing_policy = CROSSING_POLICY
    args.ori7_reward_formula_sha256 = REWARD_FORMULA_SHA256
    args.ori7_transition_formula_sha256 = TRANSITION_FORMULA_SHA256
    args.ori7_update_method = "episode_separated_monte_carlo_actor_critic"
    args.ori7_reward_parameters = reward_parameters(args)


def _ori7_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, choices=CROSSING_MODES, required=True)
    return parser


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = _ori7_parser().parse_known_args(arguments)
    if not rejected._option_was_supplied(remaining, spectral.CLI_FLAG):
        remaining = [spectral.CLI_FLAG, *remaining]
    if not rejected._option_was_supplied(remaining, "--ori6-spectral-weight"):
        remaining = [
            "--ori6-spectral-weight", str(FROZEN_SPECTRAL_WEIGHT), *remaining
        ]
    if not rejected._option_was_supplied(remaining, "--ori5-crossing-penalty"):
        remaining = [
            "--ori5-crossing-penalty", str(HISTORICAL_FIXED_PENALTY), *remaining
        ]
    args = spectral.parse_args(remaining)
    args.ori7_crossing_terminal_mode = selected.ori7_crossing_terminal_mode
    args.ori6_crossing_policy = CROSSING_POLICY
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex7_results"
    _validate_args(args)
    _bind_checkpoint_metadata(args)
    return args


if __name__ == "__main__":
    train(parse_args())
