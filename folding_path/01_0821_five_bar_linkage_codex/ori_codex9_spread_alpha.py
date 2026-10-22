"""Water-D=1 spread-potential scaling on the frozen Codex8 scale arm.

This candidate leaves every Codex8 source byte-identical.  It adds only the
extra reward needed to replace the two Codex6 spread terms

    target_progress + spread_frontier

by ``alpha`` times their parent values, with ``alpha`` frozen to either 2 or
3.  The frontier rollback is scaled by the same factor, so wrong-flat and
invalid terminal outcomes retain the parent cancellation semantics.  The
zero-reward rejected-crossing policy already reports all three affected
components as zero; rejected proposals therefore remain exact no-ops.
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
import ori_codex6_zero_reject as zero_reject
import ori_codex8 as ori8


CANDIDATE_NAME = "ori9_water1_spread_alpha"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CLI_FLAG = "--ori9-spread-alpha-ablation"
ALPHA_FLAG = "--ori9-spread-alpha"
ALPHAS = (2.0, 3.0)
PARENT_ARM = "scale"
PARENT_LAMBDA = 0.025
PARENT_EFFECTIVE_SPECTRAL_WEIGHT = 0.25
TARGET_SPREAD = 1.0
PARENT_SPREAD_FRONTIER_BONUS = 0.9
PARENT_SPREAD_FRONTIER_MODE = "state_potential"
UPDATE_METHOD = "episode_separated_monte_carlo_actor_critic"
OBSERVATION_CONTRACT = ori8.OBSERVATION_CONTRACT

REWARD_FORMULA = {
    "parent": ori8.CANDIDATE_NAME,
    "scope": {
        "model": "waterbomb",
        "target_spread": TARGET_SPREAD,
        "parent_spectral_arm": PARENT_ARM,
        "parent_spectral_weight": PARENT_EFFECTIVE_SPECTRAL_WEIGHT,
        "alpha_values": list(ALPHAS),
    },
    "parent_spread_reward": (
        "r_target + r_frontier + r_frontier_rollback"
    ),
    "additive_reward": (
        "(alpha-1) * (r_target + r_frontier + r_frontier_rollback)"
    ),
    "effective_reward": (
        "r_parent + (alpha-1) * "
        "(r_target + r_frontier + r_frontier_rollback)"
    ),
    "target_progress": (
        "r_target = e_D(before)^2 - e_D(after)^2 on accepted valid steps"
    ),
    "frontier_progress": (
        "r_frontier = Phi_D(after)-Phi_D(before), Phi_D=0.9*z_D"
    ),
    "frontier_rollback": (
        "scale the parent's frontier rollback by alpha; this cancels the "
        "complete scaled frontier ledger for the same terminal reasons"
    ),
    "unchanged": [
        "flatness reward",
        "joint milestones",
        "terminal fixed, quality, and speed rewards",
        "spectral reward",
        "mechanics",
        "observation",
        "action distribution",
        "network",
        "optimizer and update rule",
    ],
    "rejected_crossing": (
        "parent target, frontier, and frontier rollback fields are all zero; "
        "ordinary and horizon rejected proposals get no alpha contribution"
    ),
}


def _formula_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


REWARD_FORMULA_SHA256 = _formula_sha(REWARD_FORMULA)

_BASE_MAKE_ENVIRONMENT = ori8.make_environment
_BASE_RUN_EPISODE = ori8.run_episode
_BASE_AGGREGATE = ori8.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = ori8.reward_parameters


def alpha_increment_components(alpha, target, frontier, frontier_rollback):
    """Return the three additive components and their exact sum."""

    alpha = float(alpha)
    values = tuple(float(value) for value in (
        target, frontier, frontier_rollback
    ))
    if not np.isfinite(alpha) or alpha < 1.0:
        raise ValueError("alpha must be finite and at least one")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("parent spread reward components must be finite")
    multiplier = float(alpha - 1.0)
    target_increment = float(multiplier * values[0])
    frontier_increment = float(multiplier * values[1])
    frontier_rollback_increment = float(multiplier * values[2])
    total = float(
        target_increment
        + frontier_increment
        + frontier_rollback_increment
    )
    return {
        "target": target_increment,
        "frontier": frontier_increment,
        "frontier_rollback": frontier_rollback_increment,
        "total": total,
    }


class SpreadAlphaTargetFoldingEnv(
    zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv
):
    """State-free additive reward wrapper around the Codex8 scale parent."""

    def _enable_ori9_spread_alpha(self, args):
        self.ori9_spread_alpha = float(args.ori9_spread_alpha)
        self._reset_ori9_episode_state()

    def _reset_ori9_episode_state(self):
        self.ori9_cumulative_reward = 0.0
        self.ori9_cumulative_target_increment_reward = 0.0
        self.ori9_cumulative_frontier_increment_reward = 0.0
        self.ori9_cumulative_frontier_rollback_increment_reward = 0.0
        self.ori9_step_count = 0

    def reset(self):
        state = super().reset()
        if hasattr(self, "ori9_spread_alpha"):
            self._reset_ori9_episode_state()
        return state

    def step(self, physical_direction, requested_width, mode_coefficients):
        state, core_reward, done, info = super().step(
            physical_direction, requested_width, mode_coefficients
        )
        parent_target = float(info["ori6_target_progress_reward"])
        parent_frontier = float(info["ori6_spread_frontier_reward"])
        parent_frontier_rollback = float(
            info["ori6_frontier_rollback_reward"]
        )
        increments = alpha_increment_components(
            self.ori9_spread_alpha,
            parent_target,
            parent_frontier,
            parent_frontier_rollback,
        )
        alpha_reward = float(increments["total"])

        # Preserve the exact parent float, including signed zero, whenever
        # this step has no alpha contribution.  This is essential for both
        # ordinary and horizon zero-reject crossing transitions.
        reward = (
            core_reward
            if alpha_reward == 0.0
            else float(core_reward + alpha_reward)
        )
        self.ori9_cumulative_target_increment_reward += increments["target"]
        self.ori9_cumulative_frontier_increment_reward += increments[
            "frontier"
        ]
        self.ori9_cumulative_frontier_rollback_increment_reward += increments[
            "frontier_rollback"
        ]
        self.ori9_cumulative_reward += alpha_reward
        self.ori9_step_count += 1

        decomposition_error = float(
            reward
            - core_reward
            - increments["target"]
            - increments["frontier"]
            - increments["frontier_rollback"]
        )
        audited = (
            core_reward,
            parent_target,
            parent_frontier,
            parent_frontier_rollback,
            increments["target"],
            increments["frontier"],
            increments["frontier_rollback"],
            alpha_reward,
            reward,
            decomposition_error,
        )
        fields = {
            "ori9_candidate": CANDIDATE_NAME,
            "ori9_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "ori9_reward_formula_sha256": REWARD_FORMULA_SHA256,
            "ori9_spread_alpha": self.ori9_spread_alpha,
            "ori9_core_reward": float(core_reward),
            "ori9_parent_target_progress_reward": parent_target,
            "ori9_parent_spread_frontier_reward": parent_frontier,
            "ori9_parent_frontier_rollback_reward": (
                parent_frontier_rollback
            ),
            "ori9_target_increment_reward": float(increments["target"]),
            "ori9_frontier_increment_reward": float(increments["frontier"]),
            "ori9_frontier_rollback_increment_reward": float(
                increments["frontier_rollback"]
            ),
            "ori9_reward": alpha_reward,
            "ori9_reward_decomposition_error": decomposition_error,
            "ori9_nonfinite_component_count": int(sum(
                not np.isfinite(value) for value in audited
            )),
        }
        info.update(fields)
        self._replace_latest_reward_trace(info, fields)
        return state, float(reward), done, info


CommonRewardTargetFoldingEnv = SpreadAlphaTargetFoldingEnv


def make_environment(args, node, member, fix):
    """Construct the exact Codex8 scale parent, then install the wrapper."""

    _validate_ori9_args(args)
    environment = _BASE_MAKE_ENVIRONMENT(args, node, member, fix)
    if type(environment) is not zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv:
        raise RuntimeError(
            "Codex9 spread alpha requires the literal Codex8 scale parent"
        )
    environment.__class__ = SpreadAlphaTargetFoldingEnv
    environment._enable_ori9_spread_alpha(args)
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    if not hasattr(env, "ori9_spread_alpha"):
        return summary, nodes, actions

    def total(name):
        return float(sum(float(row.get(name, 0.0)) for row in actions))

    total_alpha = total("ori9_reward")
    total_target = total("ori9_target_increment_reward")
    total_frontier = total("ori9_frontier_increment_reward")
    total_frontier_rollback = total(
        "ori9_frontier_rollback_increment_reward"
    )
    total_core = total("ori9_core_reward")
    action_total = float(sum(float(row["reward"]) for row in actions))
    untraced = float(summary["total_reward"] - action_total)
    own_error = float(
        summary["total_reward"] - untraced - total_core - total_alpha
    )

    # Parent episode shells see this candidate only through total_reward.
    # Remove the separately audited additive term from their residuals once.
    for name in (
        "ori6_reward_decomposition_error",
        "ori5_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori8_reward_decomposition_error",
    ):
        if name in summary:
            summary[name] = float(summary[name] - total_alpha)

    summary.update({
        "ori9_candidate": CANDIDATE_NAME,
        "ori9_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori9_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori9_spread_alpha": env.ori9_spread_alpha,
        "ori9_total_reward": total_alpha,
        "ori9_total_target_increment_reward": total_target,
        "ori9_total_frontier_increment_reward": total_frontier,
        "ori9_total_frontier_rollback_increment_reward": (
            total_frontier_rollback
        ),
        "ori9_total_core_reward": total_core,
        "ori9_untraced_core_reward": untraced,
        "ori9_cumulative_reward": env.ori9_cumulative_reward,
        "ori9_reward_decomposition_error": own_error,
        "ori9_maximum_abs_step_reward_decomposition_error": float(max(
            [abs(float(row.get(
                "ori9_reward_decomposition_error", 0.0
            ))) for row in actions] or [0.0]
        )),
        "ori9_nonfinite_component_count": int(sum(
            int(row.get("ori9_nonfinite_component_count", 0))
            for row in actions
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE(evaluations, episode=episode)
    if result is None:
        return None
    for source, destination in {
        "ori9_total_reward": "mean_ori9_reward",
        "ori9_total_target_increment_reward": (
            "mean_ori9_target_increment_reward"
        ),
        "ori9_total_frontier_increment_reward": (
            "mean_ori9_frontier_increment_reward"
        ),
        "ori9_total_frontier_rollback_increment_reward": (
            "mean_ori9_frontier_rollback_increment_reward"
        ),
    }.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result["maximum_abs_ori9_reward_decomposition_error"] = float(max(
        abs(float(row["ori9_reward_decomposition_error"]))
        for row in evaluations
    ))
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(args)
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": ori8.CANDIDATE_NAME,
        "spread_alpha": float(args.ori9_spread_alpha),
        "reward_formula": copy.deepcopy(REWARD_FORMULA),
        "reward_formula_sha256": REWARD_FORMULA_SHA256,
        "observation_contract": OBSERVATION_CONTRACT,
        "update_method": UPDATE_METHOD,
    })
    return parameters


@contextmanager
def patched_ori8_training_runtime():
    """Temporarily route Codex8 training through this additive candidate."""

    original = (
        ori8.make_environment,
        ori8.run_episode,
        ori8.aggregate_stochastic_evaluations,
        ori8.reward_parameters,
    )
    ori8.make_environment = make_environment
    ori8.run_episode = run_episode
    ori8.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    ori8.reward_parameters = reward_parameters
    try:
        yield
    finally:
        (
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
            ori8.reward_parameters,
        ) = original


def train(args):
    _validate_ori9_args(args)
    parameters = reward_parameters(args)
    with patched_ori8_training_runtime():
        output_directory, summary = ori8.train(args)

    output_directory = Path(output_directory)
    summary.update({
        "ori9_candidate": CANDIDATE_NAME,
        "ori9_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori9_parent_candidate": ori8.CANDIDATE_NAME,
        "ori9_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori9_spread_alpha": float(args.ori9_spread_alpha),
        "ori9_reward_parameters": parameters,
    })
    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "ori9_candidate": CANDIDATE_NAME,
        "ori9_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori9_parent_candidate": ori8.CANDIDATE_NAME,
        "ori9_reward_formula": copy.deepcopy(REWARD_FORMULA),
        "ori9_reward_formula_sha256": REWARD_FORMULA_SHA256,
        "ori9_spread_alpha": float(args.ori9_spread_alpha),
        "ori9_reward_parameters": parameters,
        "candidate_entrypoint": Path(__file__).name,
        "parent_sources_unmodified": True,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def _ori9_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, action="store_true", required=True)
    parser.add_argument(ALPHA_FLAG, type=float, choices=ALPHAS, required=True)
    return parser


def _bind_ori9_args(args, selected):
    args.ori9_spread_alpha_ablation = bool(
        selected.ori9_spread_alpha_ablation
    )
    args.ori9_spread_alpha = float(selected.ori9_spread_alpha)
    args.ori9_candidate = CANDIDATE_NAME
    args.ori9_candidate_schema_version = CANDIDATE_SCHEMA_VERSION
    args.ori9_reward_formula_sha256 = REWARD_FORMULA_SHA256
    args.ori9_update_method = UPDATE_METHOD
    args.ori9_observation_contract = OBSERVATION_CONTRACT
    _validate_ori9_args(args)
    return args


def _validate_ori9_args(args):
    if not bool(getattr(args, "ori9_spread_alpha_ablation", False)):
        raise ValueError(f"Codex9 requires explicit {CLI_FLAG} opt-in")
    alpha = float(args.ori9_spread_alpha)
    if not np.isfinite(alpha) or alpha not in ALPHAS:
        raise ValueError(f"{ALPHA_FLAG} must be exactly one of {ALPHAS}")
    ori8._validate_ori8_args(args)
    if str(args.ori8_spectral_arm) != PARENT_ARM:
        raise ValueError("Codex9 spread alpha requires the Codex8 scale arm")
    if float(args.ori8_lambda) != PARENT_LAMBDA:
        raise ValueError(
            f"Codex9 spread alpha freezes ori8 lambda at {PARENT_LAMBDA}"
        )
    if (
        float(args.ori8_effective_parent_spectral_weight)
        != PARENT_EFFECTIVE_SPECTRAL_WEIGHT
        or float(args.ori6_spectral_weight)
        != PARENT_EFFECTIVE_SPECTRAL_WEIGHT
    ):
        raise ValueError(
            "Codex9 spread alpha requires effective spectral weight 0.25"
        )
    if float(args.target_spread) != TARGET_SPREAD:
        raise ValueError("Codex9 spread alpha is restricted to target D=1")
    if Path(args.data_dir).name.lower() != "waterbomb":
        raise ValueError("Codex9 spread alpha is restricted to waterbomb")
    if float(args.ori6_spread_frontier_bonus) != PARENT_SPREAD_FRONTIER_BONUS:
        raise ValueError("Codex9 freezes the parent spread frontier bonus at 0.9")
    if str(args.ori6_spread_frontier_mode) != PARENT_SPREAD_FRONTIER_MODE:
        raise ValueError(
            "Codex9 requires the parent state-potential spread frontier"
        )
    if str(args.update_method) != "actor_critic":
        raise ValueError("Codex9 requires the unchanged actor_critic update")


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = _ori9_parser().parse_known_args(arguments)
    args = ori8.parse_args(remaining)
    _bind_ori9_args(args, selected)
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex9_spread_alpha_results"
    return args


if __name__ == "__main__":
    train(parse_args())
