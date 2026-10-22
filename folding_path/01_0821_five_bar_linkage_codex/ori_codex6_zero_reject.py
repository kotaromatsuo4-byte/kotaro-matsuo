"""Additive zero-reward rejected-crossing candidate for ori6.

This entry point leaves the ori6, rejected-crossing, and spectral candidates
byte-identical.  It changes only the reward assigned to a rejected crossing:

* an ordinary rejected proposal is a zero-reward no-op;
* the originally sampled policy action remains in the rollout trace;
* the mechanical state and caches are restored by the parent candidate;
* the step is consumed and all existing discovery ledgers are retained; and
* a rejected proposal at the horizon receives the ordinary timeout reward
  ``-timeout_penalty + quality_weight * q(last_safe)`` exactly once.

The spectral potential remains optional and unchanged.  In particular, its
transition reward is zero on a rejected proposal because the effective state
does not change.
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

import ori_codex6_rejected_crossing as rejected
import ori_codex6_spectral_subspace as spectral


CANDIDATE_NAME = "ori6_RC_spectral_subspace_zero_reject_reward"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CROSSING_POLICY = "reject_candidate_as_zero_reward_noop_and_continue"
CLI_FLAG = "--ori6-zero-reject-reward"

ZERO_REJECT_FORMULA = {
    "ordinary_rejected_reward": "0",
    "horizon_rejected_reward": (
        "-timeout_penalty + quality_weight * q(last_safe)"
    ),
    "effective_transition": (
        "restore exact last-safe state and caches; accepted_width=0; "
        "actual_displacement=0; consume one step"
    ),
    "policy_trace": "retain originally sampled action and log_probability",
    "ordinary_components": {
        "base": 0.0,
        "frontier": 0.0,
        "joint": 0.0,
        "spectral": 0.0,
        "quality": 0.0,
        "rollback": 0.0,
    },
    "past_ledgers": "retained unchanged",
}
ZERO_REJECT_FORMULA_SHA256 = hashlib.sha256(
    json.dumps(
        ZERO_REJECT_FORMULA,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_BASE_MAKE_ENVIRONMENT = spectral.make_environment
_BASE_RUN_EPISODE = spectral.run_episode
_BASE_AGGREGATE_EVALUATIONS = spectral.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = spectral.reward_parameters


class ZeroRejectRewardSpectralTargetFoldingEnv(
    spectral.SpectralSubspaceTargetFoldingEnv
):
    """Spectral rejected-crossing environment with zero-cost no-ops."""

    def _enable_zero_reject_reward(self):
        self.ori6_crossing_policy = CROSSING_POLICY
        self.ori6_zero_reject_reward_enabled = True

    def _annotate_non_rejected_step(self, info):
        super()._annotate_non_rejected_step(info)
        fields = {
            "ori6_candidate": CANDIDATE_NAME,
            "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori6_crossing_policy": CROSSING_POLICY,
            "ori6_zero_reject_formula_sha256": (
                ZERO_REJECT_FORMULA_SHA256
            ),
            "ori6_zero_reject_reward_enabled": 1,
            "ori6_zero_reject_nonterminal": 0,
            "ori6_zero_reject_timeout_fixed_reward": 0.0,
            "ori6_zero_reject_timeout_quality_reward": 0.0,
            "ori6_zero_reject_timeout_reward": 0.0,
            "ori6_zero_reject_reward": 0.0,
            "ori6_zero_reject_reward_decomposition_error": 0.0,
        }
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
        # Delegate every mechanical operation and counter update to the
        # immutable rejected-crossing parent, then replace only its reward
        # account.  The spectral layer has not run yet at this point.
        state, _, done, info = super()._convert_crossing_to_rejection(
            info,
            safe_snapshot,
            safe_metrics,
            qF_before,
            qD_before,
            q_before,
        )

        timeout_boundary = bool(
            info.get("ori6_rejected_crossing_timeout_boundary", 0)
        )
        timeout_fixed = (
            -float(self.ori5_timeout_penalty) if timeout_boundary else 0.0
        )
        timeout_quality = (
            float(self.ori5_quality_weight * q_before)
            if timeout_boundary
            else 0.0
        )
        timeout_reward = float(timeout_fixed + timeout_quality)
        reward = timeout_reward if timeout_boundary else 0.0
        quality_source = (
            "last_safe_timeout_after_rejected_crossing"
            if timeout_boundary
            else "none"
        )
        qF_used = float(qF_before) if timeout_boundary else 0.0
        qD_used = float(qD_before) if timeout_boundary else 0.0
        q_used = float(q_before) if timeout_boundary else 0.0

        # Every legacy and ori6 reward component is explicit.  This prevents
        # the parent's crossing penalty or q bonus from leaking into PPO,
        # summaries, CSV output, or a future analyser.
        fields = {
            "ori6_candidate": CANDIDATE_NAME,
            "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori6_crossing_policy": CROSSING_POLICY,
            "ori6_zero_reject_formula_sha256": (
                ZERO_REJECT_FORMULA_SHA256
            ),
            "ori6_zero_reject_reward_enabled": 1,
            "ori6_zero_reject_nonterminal": int(not timeout_boundary),
            "ori6_zero_reject_last_safe_qF": float(qF_before),
            "ori6_zero_reject_last_safe_qD": float(qD_before),
            "ori6_zero_reject_last_safe_q": float(q_before),
            "ori6_zero_reject_timeout_fixed_reward": timeout_fixed,
            "ori6_zero_reject_timeout_quality_reward": timeout_quality,
            "ori6_zero_reject_timeout_reward": timeout_reward,
            "ori6_zero_reject_reward": reward,
            "ori6_zero_reject_reward_decomposition_error": 0.0,
            # The rejected event itself has no fixed or quality component.
            # At the horizon its separate timeout component contains the
            # complete ordinary timeout reward, including q exactly once.
            "ori6_rejected_crossing_fixed_reward": 0.0,
            "ori6_rejected_crossing_quality_reward": 0.0,
            "ori6_rejected_crossing_timeout_reward": timeout_reward,
            "ori6_rejected_crossing_reward": reward,
            "ori6_rejected_crossing_reward_decomposition_error": 0.0,
            "ori6_flatness_progress_reward": 0.0,
            "ori6_target_progress_reward": 0.0,
            "ori6_base_reward": 0.0,
            "ori6_spread_frontier_reward": 0.0,
            "ori6_hard_spread_milestone_reward": 0.0,
            "ori6_joint_milestone_rewards": [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ],
            "ori6_joint_milestone_reward": 0.0,
            "ori6_discovery_reward": 0.0,
            "ori6_terminal_fixed_reward": timeout_fixed,
            "ori6_terminal_quality_reward": timeout_quality,
            "ori6_terminal_speed_reward": 0.0,
            "ori6_flatness_rollback_reward": 0.0,
            "ori6_frontier_rollback_reward": 0.0,
            "ori6_joint_milestone_rollback_reward": 0.0,
            "ori6_terminal_correction_reward": 0.0,
            "ori6_terminal_reward": reward,
            "ori6_qF": qF_used,
            "ori6_qD": qD_used,
            "ori6_q": q_used,
            "ori6_quality_source": quality_source,
            "ori6_reward_decomposition_error": 0.0,
            "ori6_nonfinite_reward_component_count": int(
                not np.isfinite(reward)
            ),
            "ori5_flatness_progress_reward": 0.0,
            "ori5_target_progress_reward": 0.0,
            "ori5_base_reward": 0.0,
            "ori5_spread_milestone_reward": 0.0,
            "ori5_joint_milestone_rewards": [
                0.0 for _ in self.ori5_joint_flatness_thresholds
            ],
            "ori5_joint_milestone_reward": 0.0,
            "ori5_milestone_reward": 0.0,
            "ori5_terminal_fixed_reward": timeout_fixed,
            "ori5_terminal_quality_reward": timeout_quality,
            "ori5_terminal_speed_reward": 0.0,
            "ori5_terminal_correction_reward": 0.0,
            "ori5_terminal_reward": reward,
            "ori5_qF": qF_used,
            "ori5_qD": qD_used,
            "ori5_q": q_used,
            "ori5_quality_source": quality_source,
            "ori5_reward_decomposition_error": 0.0,
            "ori5_nonfinite_reward_component_count": int(
                not np.isfinite(reward)
            ),
            "flatness_progress_reward": 0.0,
            "target_progress_reward": 0.0,
            "target_deviation_path_reward": 0.0,
            "dense_reward": 0.0,
            "terminal_reward_component": reward,
            "near_miss_qF": qF_used,
            "near_miss_qD": qD_used,
            "near_miss_q": q_used,
            "near_miss_bonus": timeout_quality,
            "near_miss_terminal_eligible": int(timeout_boundary),
        }
        info.update(fields)
        self._replace_latest_reward_trace(info, fields)
        return state, reward, bool(done), info


CommonRewardTargetFoldingEnv = ZeroRejectRewardSpectralTargetFoldingEnv


def make_environment(args, node, member, fix):
    """Construct the immutable spectral parent, then add zero-reject reward."""

    environment = _BASE_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.__class__ = ZeroRejectRewardSpectralTargetFoldingEnv
    environment._enable_zero_reject_reward()
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env,
        agent,
        deterministic=deterministic,
        training=training,
    )

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in actions))

    ordinary = [
        row for row in actions
        if int(row.get("ori6_rejected_crossing", 0))
        and not int(row.get("ori6_rejected_crossing_timeout_boundary", 0))
    ]
    summary.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_parent_candidate": spectral.CANDIDATE_NAME,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_zero_reject_formula_sha256": ZERO_REJECT_FORMULA_SHA256,
        "ori6_zero_reject_count": len(ordinary),
        "ori6_zero_reject_total_reward": float(sum(
            float(row.get("ori6_zero_reject_reward", 0.0))
            for row in ordinary
        )),
        "ori6_zero_reject_timeout_reward": total(
            "ori6_zero_reject_timeout_reward"
        ),
        "ori6_maximum_abs_zero_reject_reward_"
        "decomposition_error": float(max(
            [abs(float(row.get(
                "ori6_zero_reject_reward_decomposition_error", 0.0
            ))) for row in actions]
            or [0.0]
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE_EVALUATIONS(
        evaluations,
        episode=episode,
    )
    if result is None:
        return None
    for source, destination in {
        "ori6_zero_reject_count": "mean_ori6_zero_reject_count",
        "ori6_zero_reject_total_reward": (
            "mean_ori6_zero_reject_total_reward"
        ),
        "ori6_zero_reject_timeout_reward": (
            "mean_ori6_zero_reject_timeout_reward"
        ),
    }.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result[
        "maximum_abs_ori6_zero_reject_reward_decomposition_error"
    ] = float(max(
        abs(float(row[
            "ori6_maximum_abs_zero_reject_reward_decomposition_error"
        ]))
        for row in evaluations
    ))
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(args)
    transition = copy.deepcopy(parameters["crossing_transition"])
    transition.update({
        "policy": CROSSING_POLICY,
        "ordinary_rejected_reward_formula": "0",
        "horizon_rejected_reward_formula": (
            "-timeout_penalty + quality_weight * q(last_safe)"
        ),
        "rejected_crossing_penalty_applied": False,
        "rejected_crossing_quality_applied": False,
        "ordinary_rejected_total_reward_exactly_zero": True,
        "original_policy_action_and_log_probability_retained": True,
        "accepted_width": 0.0,
        "actual_displacement": 0.0,
        "timeout_reward_applied_exactly_once": True,
        "past_frontier_and_joint_ledgers_retained": True,
        "spectral_transition_reward_on_reject": 0.0,
    })
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": spectral.CANDIDATE_NAME,
        "crossing_transition": transition,
        "zero_reject_formula": copy.deepcopy(ZERO_REJECT_FORMULA),
        "zero_reject_formula_sha256": ZERO_REJECT_FORMULA_SHA256,
        "collision_quality_reference": "last_safe_timeout_only",
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
    spectral.aggregate_stochastic_evaluations = (
        aggregate_stochastic_evaluations
    )
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
    _validate_zero_reject_args(args)
    parameters = reward_parameters(args)
    with _patched_spectral_training_runtime():
        output_directory, summary = spectral.train(args)

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
        "ori6_parent_candidate": spectral.CANDIDATE_NAME,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_zero_reject_formula_sha256": ZERO_REJECT_FORMULA_SHA256,
        "ori6_reward_parameters": parameters,
        "last_1000_mean_ori6_zero_reject_count": float(np.mean([
            float(row["ori6_zero_reject_count"]) for row in recent
        ])),
        "last_1000_mean_ori6_zero_reject_total_reward": float(np.mean([
            float(row["ori6_zero_reject_total_reward"])
            for row in recent
        ])),
        "maximum_abs_ori6_zero_reject_reward_decomposition_error": (
            float(max(
                [abs(float(row[
                    "ori6_maximum_abs_zero_reject_reward_"
                    "decomposition_error"
                ])) for row in rows]
                or [0.0]
            ))
        ),
    })

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_parent_candidate": spectral.CANDIDATE_NAME,
        "ori6_crossing_policy": CROSSING_POLICY,
        "ori6_reward_parameters": parameters,
        "zero_reject_formula": copy.deepcopy(ZERO_REJECT_FORMULA),
        "zero_reject_formula_sha256": ZERO_REJECT_FORMULA_SHA256,
        "candidate_entrypoint": Path(__file__).name,
        "base_rejected_and_spectral_authorities_unmodified": True,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(
        output_directory / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _validate_zero_reject_args(args):
    if not bool(args.ori6_zero_reject_reward):
        raise ValueError(f"Candidate requires explicit {CLI_FLAG} opt-in.")
    spectral._validate_spectral_args(args)


def _zero_reject_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, action="store_true", required=True)
    return parser


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = _zero_reject_parser().parse_known_args(arguments)
    if not rejected._option_was_supplied(remaining, spectral.CLI_FLAG):
        remaining = [spectral.CLI_FLAG, *remaining]
    args = spectral.parse_args(remaining)
    args.ori6_zero_reject_reward = bool(selected.ori6_zero_reject_reward)
    args.ori6_crossing_policy = CROSSING_POLICY
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex6_zero_reject_results"
    _validate_zero_reject_args(args)
    return args


if __name__ == "__main__":
    train(parse_args())
