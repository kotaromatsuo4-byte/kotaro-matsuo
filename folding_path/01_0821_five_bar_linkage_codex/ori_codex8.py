"""Additive Codex8 spectral reward ablation on final Codex6 zero-reject.

The mechanical transition, observation, action distribution, network and
episode-separated Monte-Carlo Actor--Critic update are inherited unchanged.
The legacy and scaled arms delegate literally to the Codex6 spectral parent;
only phase_v2 disables that parent term and installs a replacement wrapper.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex6_rejected_crossing as rejected
import ori_codex6_spectral_subspace as spectral
import ori_codex6_zero_reject as zero_reject


CANDIDATE_NAME = "ori8_zero_reject_spectral_reward_ablation"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CLI_FLAG = "--ori8-spectral-ablation"
ARM_FLAG = "--ori8-spectral-arm"
LAMBDA_FLAG = "--ori8-lambda"
ARMS = ("legacy", "scale", "phase_v2")
DEFAULT_SCALE_MULTIPLIER = 10.0
DEFAULT_CLOSE_RHOBAR_THRESHOLD = 0.1
DEFAULT_TURN_DELTA_THRESHOLD = 0.7
UPDATE_METHOD = "episode_separated_monte_carlo_actor_critic"
OBSERVATION_CONTRACT = (
    "Codex6 coordinates-only observation unchanged; rho, delta, rho_bar, "
    "close, z, step and horizon are not observed"
)
POMDP_CAVEAT = (
    "phase_v2 close/z are path-dependent environment ledgers omitted from "
    "the unchanged actor and critic observation; the reward-only phase arm "
    "is therefore partially observable"
)

REWARD_FORMULA = {
    "parent": zero_reject.CANDIDATE_NAME,
    "effective_parent_weight": {
        "legacy": "lambda",
        "scale": "lambda*scale_multiplier (scale_multiplier frozen at 10)",
        "phase_v2": 0.0,
    },
    "geometry": "reuse parent spectral rho and delta; rho0=rho(reset)",
    "rho_bar": "clip(rho/rho0, 0, 1); rho0 must be finite and positive",
    "arms": {
        "legacy": "literal Codex6 spectral parent at weight lambda",
        "scale": (
            "literal Codex6 spectral parent at weight "
            "lambda*scale_multiplier"
        ),
        "phase_v2_pre": "Psi = 0.1*(1-rho_bar) while z=0",
        "phase_v2_post": "Psi = 0.1+0.4*rho_bar*delta while z=1",
    },
    "phase_events": {
        "close": "accepted_valid and rho_bar_after <= 0.1",
        "turn": (
            "accepted_valid and close was already set before this accepted "
            "step and delta_after >= 0.7"
        ),
        "same_step_close_turn_cascade": False,
        "z_latched": True,
    },
    "phase_v2_transition_reward": (
        "accepted_valid * lambda * (Psi_after-Psi_before)"
    ),
    "phase_v2_rollback": {
        "reasons": sorted(spectral.SPECTRAL_ROLLBACK_REASONS),
        "reward": "-provisional cumulative phase reward",
        "retain": ["success", "timeout", "running"],
    },
    "noncommit": (
        "rejected crossing or invalid/unaccepted candidate leaves close, z, "
        "Psi and cumulative phase reward unchanged and gives phase reward 0"
    ),
    "lambda_zero": (
        "exact Codex6 zero-reject parent with effective spectral weight 0; no "
        "Codex8 environment or history fields are installed"
    ),
}
TRANSITION_FORMULA = {
    "parent_crossing_policy": zero_reject.CROSSING_POLICY,
    "crossing_candidate_committed": False,
    "crossing_step_consumed": True,
    "ordinary_crossing_reward": 0.0,
    "horizon_crossing_reward": (
        "-timeout_penalty + quality_weight*q(last_safe) exactly once"
    ),
    "sampled_action_and_log_probability_retained": True,
    "phase_candidate_commit": "accepted_valid only, after parent step",
    "phase_invalid_noncommit": True,
    "network_observation_action_update_changed": False,
}


def _formula_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


REWARD_FORMULA_SHA256 = _formula_sha(REWARD_FORMULA)
TRANSITION_FORMULA_SHA256 = _formula_sha(TRANSITION_FORMULA)

_BASE_MAKE_ENVIRONMENT = zero_reject.make_environment
_BASE_RUN_EPISODE = zero_reject.run_episode
_BASE_AGGREGATE = zero_reject.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = zero_reject.reward_parameters


def _normalised_rho(rho, rho0):
    rho = float(rho)
    rho0 = float(rho0)
    if not np.isfinite(rho) or not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("Codex8 requires finite rho and finite positive rho0")
    return float(np.clip(rho / rho0, 0.0, 1.0))


def phase_v2_potential(rho_bar, delta, z):
    rho_bar = float(rho_bar)
    delta = float(delta)
    if not np.isfinite(rho_bar) or not 0.0 <= rho_bar <= 1.0:
        raise ValueError("rho_bar must be finite and in [0, 1]")
    if not np.isfinite(delta) or not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be finite and in [0, 1]")
    return float(
        0.1 + 0.4 * rho_bar * delta
        if bool(z) else 0.1 * (1.0 - rho_bar)
    )


def arm_potential(arm, rho, rho_bar, delta, z, scale_multiplier):
    if arm == "legacy":
        return float(rho * delta)
    if arm == "scale":
        return float(scale_multiplier * rho * delta)
    if arm == "phase_v2":
        return phase_v2_potential(rho_bar, delta, z)
    raise ValueError(f"Unknown Codex8 spectral arm: {arm!r}")


def effective_parent_spectral_weight(args):
    """Return the literal Codex6 spectral weight for the selected arm."""

    arm = str(args.ori8_spectral_arm)
    coefficient = float(args.ori8_lambda)
    if arm == "legacy":
        return coefficient
    if arm == "scale":
        return float(coefficient * float(args.ori8_scale_multiplier))
    if arm == "phase_v2":
        return 0.0
    raise ValueError(f"Unknown Codex8 spectral arm: {arm!r}")


class Codex8SpectralTargetFoldingEnv(
    zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv
):
    """Final zero-reject environment with one replacement spectral reward."""

    def _enable_ori8(self, args):
        if str(args.ori8_spectral_arm) != "phase_v2":
            raise RuntimeError("Codex8 wrapper is exclusive to phase_v2")
        if float(self.ori6_spectral_weight) != 0.0:
            raise RuntimeError("Codex8 parent spectral reward must be disabled")
        self.ori8_spectral_arm = str(args.ori8_spectral_arm)
        self.ori8_lambda = float(args.ori8_lambda)
        self.ori8_scale_multiplier = float(args.ori8_scale_multiplier)
        self.ori8_close_rhobar_threshold = float(
            args.ori8_close_rhobar_threshold
        )
        self.ori8_turn_delta_threshold = float(args.ori8_turn_delta_threshold)
        self._reset_ori8_episode_state()

    def reset(self):
        state = super().reset()
        if hasattr(self, "ori8_spectral_arm"):
            self._reset_ori8_episode_state()
        return state

    def _reset_ori8_episode_state(self):
        initial = self.spectral_subspace_metrics()
        self.ori8_rho0 = float(initial["rho"])
        if not np.isfinite(self.ori8_rho0) or self.ori8_rho0 <= 0.0:
            raise ValueError("Codex8 reset spectral rho0 must be positive")
        self.ori8_close_latched = False
        self.ori8_z_latched = False
        rho_bar = _normalised_rho(initial["rho"], self.ori8_rho0)
        self.ori8_current_psi = arm_potential(
            self.ori8_spectral_arm,
            float(initial["rho"]),
            rho_bar,
            float(initial["delta"]),
            False,
            self.ori8_scale_multiplier,
        )
        self.ori8_initial_psi = float(self.ori8_current_psi)
        self.ori8_cumulative_reward = 0.0
        self.ori8_accepted_transition_count = 0
        self.ori8_noncommit_count = 0
        self.ori8_close_event_count = 0
        self.ori8_turn_event_count = 0
        self.ori8_rollback_count = 0

    def step(self, physical_direction, requested_width, mode_coefficients):
        close_before = bool(self.ori8_close_latched)
        z_before = bool(self.ori8_z_latched)
        psi_before = float(self.ori8_current_psi)
        cumulative_before = float(self.ori8_cumulative_reward)

        state, core_reward, done, info = super().step(
            physical_direction, requested_width, mode_coefficients
        )
        if float(info.get("ori6_spectral_weight", np.nan)) != 0.0:
            raise RuntimeError("Codex8 detected a stacked parent spectral reward")
        if float(info.get("ori6_spectral_reward", np.nan)) != 0.0:
            raise RuntimeError("Codex8 parent spectral reward was not exactly zero")

        rejected_crossing = bool(info.get("ori6_rejected_crossing", 0))
        accepted_valid = bool(
            not rejected_crossing
            and info.get("after_state_valid", 0)
            and abs(float(info.get("accepted_width", 0.0))) > 0.0
        )

        rho_before = float(info["ori6_spectral_rho_before"])
        delta_before = float(info["ori6_spectral_delta_before"])
        rho_bar_before = _normalised_rho(rho_before, self.ori8_rho0)
        observed_rho_after = float(info["ori6_spectral_rho_after"])
        observed_delta_after = float(info["ori6_spectral_delta_after"])

        if accepted_valid:
            rho_after = observed_rho_after
            delta_after = observed_delta_after
            rho_bar_after = _normalised_rho(rho_after, self.ori8_rho0)
            close_event = bool(
                self.ori8_spectral_arm == "phase_v2"
                and not close_before
                and rho_bar_after <= self.ori8_close_rhobar_threshold
            )
            # Deliberately use close_before: a first close cannot turn in the
            # same accepted transition even when delta already exceeds 0.7.
            turn_event = bool(
                self.ori8_spectral_arm == "phase_v2"
                and close_before
                and not z_before
                and delta_after >= self.ori8_turn_delta_threshold
            )
            close_after = bool(close_before or close_event)
            z_after = bool(z_before or turn_event)
            evaluated_before = arm_potential(
                self.ori8_spectral_arm,
                rho_before,
                rho_bar_before,
                delta_before,
                z_before,
                self.ori8_scale_multiplier,
            )
            if not np.isclose(evaluated_before, psi_before, rtol=0.0, atol=1e-14):
                raise RuntimeError("Codex8 phase ledger disagrees with state geometry")
            psi_after = arm_potential(
                self.ori8_spectral_arm,
                rho_after,
                rho_bar_after,
                delta_after,
                z_after,
                self.ori8_scale_multiplier,
            )
            psi_difference = float(psi_after - psi_before)
            progress_reward = float(self.ori8_lambda * psi_difference)
            self.ori8_close_latched = close_after
            self.ori8_z_latched = z_after
            self.ori8_current_psi = float(psi_after)
            self.ori8_accepted_transition_count += 1
            self.ori8_close_event_count += int(close_event)
            self.ori8_turn_event_count += int(turn_event)
        else:
            # Candidate geometry is diagnostic only.  The complete Codex8
            # state and reward ledger remains the exact pre-proposal ledger.
            rho_after = rho_before
            delta_after = delta_before
            rho_bar_after = rho_bar_before
            close_event = False
            turn_event = False
            close_after = close_before
            z_after = z_before
            psi_after = psi_before
            psi_difference = 0.0
            progress_reward = 0.0
            self.ori8_noncommit_count += 1

        provisional_cumulative = float(cumulative_before + progress_reward)
        termination_reason = str(info.get("termination_reason", "running"))
        rollback_reason = (
            termination_reason
            if termination_reason in spectral.SPECTRAL_ROLLBACK_REASONS
            else "none"
        )
        if rollback_reason != "none":
            rollback_reward = float(-provisional_cumulative)
            cumulative_after = 0.0
            self.ori8_rollback_count += 1
        else:
            rollback_reward = 0.0
            cumulative_after = provisional_cumulative
        phase_reward = float(progress_reward + rollback_reward)
        self.ori8_cumulative_reward = float(cumulative_after)

        # Preserve the parent's exact float (including a possible signed
        # zero) whenever the replacement term is exactly zero.  In
        # particular, ordinary and horizon rejected crossings retain the
        # last-safe phase prefix and receive no phase contribution.
        reward = (
            core_reward
            if phase_reward == 0.0
            else float(core_reward + phase_reward)
        )
        audited = (
            rho_before, delta_before, rho_bar_before, psi_before,
            rho_after, delta_after, rho_bar_after, psi_after,
            psi_difference, progress_reward, rollback_reward, phase_reward,
            cumulative_before,
            cumulative_after, core_reward, reward,
        )
        nonfinite_count = int(sum(not np.isfinite(v) for v in audited))
        decomposition_error = float(
            reward - core_reward - progress_reward - rollback_reward
        )
        fields = {
            "ori8_candidate": CANDIDATE_NAME,
            "ori8_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori8_reward_formula_sha256": REWARD_FORMULA_SHA256,
            "ori8_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
            "ori8_spectral_arm": self.ori8_spectral_arm,
            "ori8_lambda": self.ori8_lambda,
            "ori8_scale_multiplier": self.ori8_scale_multiplier,
            "ori8_close_rhobar_threshold": self.ori8_close_rhobar_threshold,
            "ori8_turn_delta_threshold": self.ori8_turn_delta_threshold,
            "ori8_rho0": self.ori8_rho0,
            "ori8_accepted_valid_transition": int(accepted_valid),
            "ori8_candidate_noncommit": int(not accepted_valid),
            "ori8_rejected_crossing_noncommit": int(rejected_crossing),
            "ori8_rho_before": rho_before,
            "ori8_delta_before": delta_before,
            "ori8_rhobar_before": rho_bar_before,
            "ori8_psi_before": psi_before,
            "ori8_observed_candidate_rho_after": observed_rho_after,
            "ori8_observed_candidate_delta_after": observed_delta_after,
            "ori8_rho_after": rho_after,
            "ori8_delta_after": delta_after,
            "ori8_rhobar_after": rho_bar_after,
            "ori8_psi_after": psi_after,
            "ori8_close_before": int(close_before),
            "ori8_close_event": int(close_event),
            "ori8_close_after": int(close_after),
            "ori8_z_before": int(z_before),
            "ori8_turn_event": int(turn_event),
            "ori8_z_after": int(z_after),
            "ori8_psi_difference": psi_difference,
            "ori8_progress_reward": progress_reward,
            "ori8_rollback_reward": rollback_reward,
            "ori8_rollback_reason": rollback_reason,
            "ori8_reward": phase_reward,
            "ori8_core_reward": float(core_reward),
            "ori8_cumulative_reward_before": cumulative_before,
            "ori8_cumulative_reward_after": cumulative_after,
            "ori8_reward_decomposition_error": decomposition_error,
            "ori8_nonfinite_component_count": nonfinite_count,
        }
        info.update(fields)
        if not self.ori5_reward_trace:
            raise RuntimeError("Codex8 step did not create a parent reward trace")
        self.ori5_reward_trace[-1].update(fields)
        return state, reward, done, info


CommonRewardTargetFoldingEnv = Codex8SpectralTargetFoldingEnv


def _parent_args(args):
    parent = copy.copy(args)
    parent.ori6_spectral_weight = effective_parent_spectral_weight(args)
    return parent


def make_environment(args, node, member, fix):
    environment = _BASE_MAKE_ENVIRONMENT(
        _parent_args(args), node, member, fix
    )
    if (
        float(args.ori8_lambda) == 0.0
        or str(args.ori8_spectral_arm) in {"legacy", "scale"}
    ):
        # Literal Codex6 path: legacy and scale differ only by the parent
        # weight, and lambda=0 does not install Codex8 fields at all.
        return environment
    environment.__class__ = Codex8SpectralTargetFoldingEnv
    environment._enable_ori8(args)
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    if not hasattr(env, "ori8_spectral_arm"):
        return summary, nodes, actions

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in actions))

    total_phase = total("ori8_reward")
    total_progress = total("ori8_progress_reward")
    total_rollback = total("ori8_rollback_reward")
    total_core = total("ori8_core_reward")
    action_total = float(sum(float(row["reward"]) for row in actions))
    untraced = float(summary["total_reward"] - action_total)
    own_error = float(
        summary["total_reward"] - untraced - total_core - total_phase
    )
    # Parent reward shells see Codex8 only through total_reward.  Remove this
    # separately audited additive term from each inherited residual once.
    for name in (
        "ori6_reward_decomposition_error",
        "ori5_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
    ):
        if name in summary:
            summary[name] = float(summary[name] - total_phase)
    summary.update({
        "ori8_candidate": CANDIDATE_NAME,
        "ori8_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori8_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori8_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "ori8_spectral_arm": env.ori8_spectral_arm,
        "ori8_lambda": env.ori8_lambda,
        "ori8_scale_multiplier": env.ori8_scale_multiplier,
        "ori8_effective_parent_spectral_weight": 0.0,
        "ori8_rho0": env.ori8_rho0,
        "ori8_initial_psi": env.ori8_initial_psi,
        "ori8_total_reward": total_phase,
        "ori8_total_progress_reward": total_progress,
        "ori8_total_rollback_reward": total_rollback,
        "ori8_total_core_reward": total_core,
        "ori8_untraced_core_reward": untraced,
        "ori8_cumulative_reward": env.ori8_cumulative_reward,
        "ori8_terminal_psi": env.ori8_current_psi,
        "ori8_close_latched": int(env.ori8_close_latched),
        "ori8_z_latched": int(env.ori8_z_latched),
        "ori8_close_event_count": int(env.ori8_close_event_count),
        "ori8_turn_event_count": int(env.ori8_turn_event_count),
        "ori8_rollback_count": int(env.ori8_rollback_count),
        "ori8_accepted_transition_count": int(
            env.ori8_accepted_transition_count
        ),
        "ori8_noncommit_count": int(env.ori8_noncommit_count),
        "ori8_reward_decomposition_error": own_error,
        "ori8_maximum_abs_step_reward_decomposition_error": float(max(
            [abs(float(row.get(
                "ori8_reward_decomposition_error", 0.0
            ))) for row in actions] or [0.0]
        )),
        "ori8_nonfinite_component_count": int(sum(
            int(row.get("ori8_nonfinite_component_count", 0))
            for row in actions
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE(evaluations, episode=episode)
    if result is None or not evaluations or "ori8_total_reward" not in evaluations[0]:
        return result
    for source, destination in {
        "ori8_total_reward": "mean_ori8_total_reward",
        "ori8_total_progress_reward": "mean_ori8_total_progress_reward",
        "ori8_total_rollback_reward": "mean_ori8_total_rollback_reward",
        "ori8_terminal_psi": "mean_ori8_terminal_psi",
        "ori8_close_event_count": "mean_ori8_close_event_count",
        "ori8_turn_event_count": "mean_ori8_turn_event_count",
        "ori8_noncommit_count": "mean_ori8_noncommit_count",
    }.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result["maximum_abs_ori8_reward_decomposition_error"] = float(max(
        abs(float(row["ori8_reward_decomposition_error"]))
        for row in evaluations
    ))
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(_parent_args(args))
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": zero_reject.CANDIDATE_NAME,
        "parent_spectral_weight_effective": float(
            effective_parent_spectral_weight(args)
        ),
        "ori8_spectral_arm": str(args.ori8_spectral_arm),
        "ori8_lambda": float(args.ori8_lambda),
        "ori8_scale_multiplier": float(args.ori8_scale_multiplier),
        "ori8_close_rhobar_threshold": float(
            args.ori8_close_rhobar_threshold
        ),
        "ori8_turn_delta_threshold": float(args.ori8_turn_delta_threshold),
        "reward_formula": copy.deepcopy(REWARD_FORMULA),
        "reward_formula_sha256": REWARD_FORMULA_SHA256,
        "transition_formula": copy.deepcopy(TRANSITION_FORMULA),
        "transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "update_method": UPDATE_METHOD,
        "observation_contract": OBSERVATION_CONTRACT,
        "pomdp_caveat": POMDP_CAVEAT,
    })
    return parameters


@contextmanager
def _patched_zero_reject_training_runtime():
    original = (
        zero_reject.make_environment,
        zero_reject.run_episode,
        zero_reject.aggregate_stochastic_evaluations,
        zero_reject.reward_parameters,
    )
    zero_reject.make_environment = make_environment
    zero_reject.run_episode = run_episode
    zero_reject.aggregate_stochastic_evaluations = (
        aggregate_stochastic_evaluations
    )
    zero_reject.reward_parameters = reward_parameters
    try:
        yield
    finally:
        (
            zero_reject.make_environment,
            zero_reject.run_episode,
            zero_reject.aggregate_stochastic_evaluations,
            zero_reject.reward_parameters,
        ) = original


def train(args):
    _validate_ori8_args(args)
    parameters = reward_parameters(args)
    with _patched_zero_reject_training_runtime():
        output_directory, summary = zero_reject.train(args)

    output_directory = Path(output_directory)
    summary.update({
        "ori8_candidate": CANDIDATE_NAME,
        "ori8_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori8_parent_candidate": zero_reject.CANDIDATE_NAME,
        "ori8_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori8_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "ori8_spectral_arm": str(args.ori8_spectral_arm),
        "ori8_lambda": float(args.ori8_lambda),
        "ori8_scale_multiplier": float(args.ori8_scale_multiplier),
        "ori8_effective_parent_spectral_weight": float(
            effective_parent_spectral_weight(args)
        ),
        "ori8_reward_parameters": parameters,
        "ori8_lambda_zero_parent_exact": bool(float(args.ori8_lambda) == 0.0),
    })
    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "ori8_candidate": CANDIDATE_NAME,
        "ori8_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori8_parent_candidate": zero_reject.CANDIDATE_NAME,
        "ori8_reward_formula": copy.deepcopy(REWARD_FORMULA),
        "ori8_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori8_transition_formula": copy.deepcopy(TRANSITION_FORMULA),
        "ori8_transition_formula_sha256": TRANSITION_FORMULA_SHA256,
        "ori8_spectral_arm": str(args.ori8_spectral_arm),
        "ori8_lambda": float(args.ori8_lambda),
        "ori8_scale_multiplier": float(args.ori8_scale_multiplier),
        "ori8_effective_parent_spectral_weight": float(
            effective_parent_spectral_weight(args)
        ),
        "ori8_reward_parameters": parameters,
        "candidate_entrypoint": Path(__file__).name,
        "parent_sources_unmodified": True,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _ori8_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, action="store_true", required=True)
    parser.add_argument(ARM_FLAG, choices=ARMS, required=True)
    parser.add_argument(LAMBDA_FLAG, type=float, required=True)
    parser.add_argument(
        "--ori8-scale-multiplier",
        type=float,
        default=DEFAULT_SCALE_MULTIPLIER,
    )
    parser.add_argument(
        "--ori8-close-rhobar-threshold",
        type=float,
        default=DEFAULT_CLOSE_RHOBAR_THRESHOLD,
    )
    parser.add_argument(
        "--ori8-turn-delta-threshold",
        type=float,
        default=DEFAULT_TURN_DELTA_THRESHOLD,
    )
    return parser


def _bind_ori8_args(args, selected, original_arguments):
    if rejected._option_was_supplied(
        original_arguments, "--ori6-spectral-weight"
    ) and float(args.ori6_spectral_weight) != 0.0:
        raise ValueError("Codex8 forbids a nonzero parent ori6 spectral weight")
    args.ori8_spectral_ablation = bool(selected.ori8_spectral_ablation)
    args.ori8_spectral_arm = str(selected.ori8_spectral_arm)
    args.ori8_lambda = float(selected.ori8_lambda)
    args.ori8_scale_multiplier = float(selected.ori8_scale_multiplier)
    args.ori8_close_rhobar_threshold = float(
        selected.ori8_close_rhobar_threshold
    )
    args.ori8_turn_delta_threshold = float(
        selected.ori8_turn_delta_threshold
    )
    args.ori6_spectral_weight = effective_parent_spectral_weight(args)
    args.ori8_effective_parent_spectral_weight = float(
        args.ori6_spectral_weight
    )
    # These are deliberately set before training so atomic checkpoints and
    # the initial run_config bind the candidate without final-only updates.
    args.ori8_candidate = CANDIDATE_NAME
    args.ori8_candidate_schema_version = CANDIDATE_SCHEMA_VERSION
    args.ori8_reward_formula_sha256 = REWARD_FORMULA_SHA256
    args.ori8_transition_formula_sha256 = TRANSITION_FORMULA_SHA256
    args.ori8_update_method = UPDATE_METHOD
    args.ori8_observation_contract = OBSERVATION_CONTRACT
    args.ori8_pomdp_caveat = POMDP_CAVEAT
    _validate_ori8_args(args)
    return args


def _validate_ori8_args(args):
    if not bool(args.ori8_spectral_ablation):
        raise ValueError(f"Codex8 requires explicit {CLI_FLAG} opt-in")
    zero_reject._validate_zero_reject_args(args)
    if str(args.ori8_spectral_arm) not in ARMS:
        raise ValueError(f"{ARM_FLAG} must be one of {ARMS}")
    if not np.isfinite(float(args.ori8_lambda)) or float(args.ori8_lambda) < 0.0:
        raise ValueError(f"{LAMBDA_FLAG} must be finite and non-negative")
    exact = (
        ("ori8_scale_multiplier", DEFAULT_SCALE_MULTIPLIER),
        ("ori8_close_rhobar_threshold", DEFAULT_CLOSE_RHOBAR_THRESHOLD),
        ("ori8_turn_delta_threshold", DEFAULT_TURN_DELTA_THRESHOLD),
    )
    for name, expected in exact:
        if not np.isfinite(float(getattr(args, name))):
            raise ValueError(f"{name} must be finite")
        if float(getattr(args, name)) != expected:
            raise ValueError(f"{name} is frozen at {expected}")
    expected_parent_weight = effective_parent_spectral_weight(args)
    if float(args.ori6_spectral_weight) != expected_parent_weight:
        raise ValueError(
            "ori6_spectral_weight disagrees with the selected Codex8 arm"
        )
    if (
        float(args.ori8_effective_parent_spectral_weight)
        != expected_parent_weight
    ):
        raise ValueError("Codex8 effective parent weight metadata mismatch")
    if str(args.update_method) != "actor_critic":
        raise ValueError("Codex8 requires the unchanged actor_critic update")


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = _ori8_parser().parse_known_args(arguments)
    if not rejected._option_was_supplied(
        remaining, zero_reject.CLI_FLAG
    ):
        remaining = [zero_reject.CLI_FLAG, *remaining]
    if not rejected._option_was_supplied(
        remaining, "--ori6-spectral-weight"
    ):
        remaining = ["--ori6-spectral-weight", "0", *remaining]
    args = zero_reject.parse_args(remaining)
    _bind_ori8_args(args, selected, arguments)
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex8_results"
    return args


if __name__ == "__main__":
    train(parse_args())
