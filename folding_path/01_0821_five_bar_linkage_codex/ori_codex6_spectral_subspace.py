"""Additive spectral-subspace potential on rejected-crossing ori6.

This module deliberately leaves ori6 and the rejected-crossing candidate
byte-identical.  It adds a current-state, reset-referenced Markov potential
whose reward telescopes on accepted valid transitions.  Invalid terminal
outcomes roll back only this additive spectral ledger.
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

import ori_codex3 as ori3
import ori_codex6_rejected_crossing as rejected


CANDIDATE_NAME = "ori6_RC_spectral_subspace_potential"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
CLI_FLAG = "--ori6-spectral-subspace"
DEFAULT_SPECTRAL_WEIGHT = 0.05
DEFAULT_RELATIVE_EPSILON = 1e-12
SPECTRAL_ROLLBACK_REASONS = frozenset({
    "wrong_flat",
    "correction_failed",
    "no_modes",
    "self_crossing",
})

SPECTRAL_FORMULA = {
    "centered_coordinates": "Xc = X - mean_rows(X)",
    "svd": "Xc = U diag(sigma) V^T",
    "subspace_dimension": "k = flatness_subspace_dimension",
    "projector": "P = U[:, :k] U[:, :k]^T",
    "reference": "P0 is recomputed exactly from the reset state",
    "gap_symbols": "a=sigma[k-1], b=sigma[k], s=a+b, gap=a-b",
    "relative_epsilon_rule": (
        "if s==0 or gap<=eps_rel*max(s,sigma[0]) then rho=0; "
        "else rho=gap/(s*(1+eps_rel))"
    ),
    "distance": "delta = ||P-P0||_F/sqrt(2*k)",
    "potential": "Phi = rho*delta",
    "transition_reward": (
        "accepted_valid * lambda * (Phi_after-Phi_before)"
    ),
    "retain": ["success", "timeout"],
    "rollback": sorted(SPECTRAL_ROLLBACK_REASONS),
    "rejected_crossing": (
        "last-safe state is restored; spectral transition reward is zero"
    ),
}
SPECTRAL_FORMULA_SHA256 = hashlib.sha256(
    json.dumps(
        SPECTRAL_FORMULA,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_BASE_MAKE_ENVIRONMENT = rejected.make_environment
_BASE_RUN_EPISODE = rejected.run_episode
_BASE_AGGREGATE_EVALUATIONS = rejected.aggregate_stochastic_evaluations
_BASE_REWARD_PARAMETERS = rejected.reward_parameters
_BASE_SAVE_LEARNING_CURVES = rejected.save_learning_curves


def _canonical_array_sha256(array):
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def subspace_projector(basis, k):
    """Return the node-space projector of the first ``k`` basis columns."""

    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("basis must be a two-dimensional array")
    k = int(k)
    if k < 1 or k > basis.shape[1]:
        raise ValueError("k must select at least one available basis column")
    selected = basis[:, :k]
    return selected @ selected.T


def spectral_subspace_geometry(
    node,
    k,
    reference_projector=None,
    relative_epsilon=DEFAULT_RELATIVE_EPSILON,
):
    """Compute the invariant geometry used by the spectral potential.

    ``relative_epsilon`` is dimensionless.  Consequently both the gap-zero
    decision and nonzero ``rho`` are invariant under any nonzero uniform
    coordinate scale, apart from unavoidable floating-point roundoff.
    """

    coordinates = np.asarray(node, dtype=float)
    if coordinates.ndim != 2:
        raise ValueError("node coordinates must be a two-dimensional array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node coordinates must be finite")
    k = int(k)
    relative_epsilon = float(relative_epsilon)
    if not np.isfinite(relative_epsilon) or relative_epsilon < 0.0:
        raise ValueError("relative_epsilon must be finite and non-negative")

    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if k < 1 or k >= len(singular_values):
        raise ValueError(
            "flatness_subspace_dimension must leave sigma[k] available"
        )

    projector = subspace_projector(left, k)
    a = float(singular_values[k - 1])
    b = float(singular_values[k])
    singular_sum = float(a + b)
    gap = float(max(a - b, 0.0))
    sigma0 = float(singular_values[0])
    degeneracy_threshold = float(
        relative_epsilon * max(singular_sum, sigma0)
    )
    gap_degenerate = bool(
        singular_sum == 0.0 or gap <= degeneracy_threshold
    )
    if gap_degenerate:
        rho = 0.0
    else:
        rho = float(
            gap / (singular_sum * (1.0 + relative_epsilon))
        )

    if reference_projector is None:
        delta = 0.0
    else:
        reference = np.asarray(reference_projector, dtype=float)
        if reference.shape != projector.shape:
            raise ValueError(
                "reference projector shape does not match current projector"
            )
        delta = float(
            np.linalg.norm(projector - reference, ord="fro")
            / np.sqrt(2.0 * k)
        )
        # Rank-k orthogonal projectors give delta in [0, 1].  Clamp only
        # roundoff beyond the mathematical range.
        delta = float(np.clip(delta, 0.0, 1.0))
    potential = float(rho * delta)
    return {
        "projector": np.array(projector, copy=True),
        "singular_values": np.array(singular_values, copy=True),
        "sigma_k": a,
        "sigma_k_plus_1": b,
        "singular_sum": singular_sum,
        "spectral_gap": gap,
        "degeneracy_threshold": degeneracy_threshold,
        "gap_degenerate": int(gap_degenerate),
        "rho": rho,
        "delta": delta,
        "potential": potential,
    }


class SpectralSubspaceTargetFoldingEnv(
    rejected.RejectedCrossingContinuationTargetFoldingEnv
):
    """Rejected-crossing environment with one additive Markov potential."""

    def _enable_spectral_subspace(self, weight, relative_epsilon):
        self.ori6_spectral_weight = float(weight)
        self.ori6_spectral_relative_epsilon = float(relative_epsilon)
        if (
            not np.isfinite(self.ori6_spectral_weight)
            or self.ori6_spectral_weight < 0.0
        ):
            raise ValueError("ori6 spectral weight must be finite and non-negative")
        if (
            not np.isfinite(self.ori6_spectral_relative_epsilon)
            or self.ori6_spectral_relative_epsilon < 0.0
        ):
            raise ValueError(
                "ori6 spectral relative epsilon must be finite and non-negative"
            )
        self._reset_spectral_subspace_episode_state()

    def reset(self):
        state = super().reset()
        if hasattr(self, "ori6_spectral_weight"):
            self._reset_spectral_subspace_episode_state()
        return state

    def _reset_spectral_subspace_episode_state(self):
        self._ori6_spectral_metrics_cache = {}
        reference = spectral_subspace_geometry(
            self.node,
            self.flatness_subspace_dimension,
            reference_projector=None,
            relative_epsilon=self.ori6_spectral_relative_epsilon,
        )
        self.ori6_spectral_reference_projector = np.array(
            reference["projector"],
            copy=True,
        )
        self.ori6_spectral_reference_projector_sha256 = (
            _canonical_array_sha256(
                self.ori6_spectral_reference_projector
            )
        )
        self.ori6_spectral_reference_node_sha256 = _canonical_array_sha256(
            self.node
        )
        self.ori6_spectral_current_potential = 0.0
        self.ori6_spectral_cumulative_reward = 0.0
        self.ori6_spectral_maximum_potential = 0.0
        self.ori6_spectral_maximum_delta = 0.0
        self.ori6_spectral_maximum_rho = 0.0
        self.ori6_spectral_rollback_count = 0
        initial = dict(reference)
        initial.update({"delta": 0.0, "potential": 0.0})
        self._ori6_spectral_metrics_cache[
            self._spectral_cache_key(self.node)
        ] = copy.deepcopy(initial)

    @staticmethod
    def _spectral_cache_key(node):
        value = np.ascontiguousarray(np.asarray(node, dtype="<f8"))
        return (tuple(value.shape), value.tobytes(order="C"))

    def spectral_subspace_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        key = self._spectral_cache_key(coordinates)
        cached = self._ori6_spectral_metrics_cache.get(key)
        if cached is None:
            cached = spectral_subspace_geometry(
                coordinates,
                self.flatness_subspace_dimension,
                reference_projector=(
                    self.ori6_spectral_reference_projector
                ),
                relative_epsilon=self.ori6_spectral_relative_epsilon,
            )
            self._ori6_spectral_metrics_cache[key] = copy.deepcopy(cached)
        return copy.deepcopy(cached)

    @staticmethod
    def _metric_log_fields(prefix, metrics):
        return {
            f"ori6_spectral_sigma_k_{prefix}": float(metrics["sigma_k"]),
            f"ori6_spectral_sigma_k_plus_1_{prefix}": float(
                metrics["sigma_k_plus_1"]
            ),
            f"ori6_spectral_gap_{prefix}": float(metrics["spectral_gap"]),
            f"ori6_spectral_gap_degenerate_{prefix}": int(
                metrics["gap_degenerate"]
            ),
            f"ori6_spectral_rho_{prefix}": float(metrics["rho"]),
            f"ori6_spectral_delta_{prefix}": float(metrics["delta"]),
            f"ori6_spectral_potential_{prefix}": float(
                metrics["potential"]
            ),
        }

    def step(self, physical_direction, requested_width, mode_coefficients):
        metrics_before = self.spectral_subspace_metrics()
        cumulative_before = float(self.ori6_spectral_cumulative_reward)
        core_state, core_reward, done, info = super().step(
            physical_direction,
            requested_width,
            mode_coefficients,
        )

        rejected_crossing = bool(info.get("ori6_rejected_crossing", 0))
        if rejected_crossing:
            # The parent restored the complete last-safe mechanical state.
            # The parent never accesses this candidate's pure geometry cache,
            # and after-state spectral metrics are intentionally deferred
            # until this branch.  Its keys, values, and object identity are
            # therefore already unchanged without an O(steps^2) deep copy.
            metrics_after = copy.deepcopy(metrics_before)
        else:
            metrics_after = self.spectral_subspace_metrics()

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

        provisional_cumulative = float(
            cumulative_before + spectral_progress_reward
        )
        termination_reason = str(info.get("termination_reason", "running"))
        rollback_reason = (
            termination_reason
            if termination_reason in SPECTRAL_ROLLBACK_REASONS
            else "none"
        )
        if rollback_reason != "none":
            spectral_rollback_reward = -provisional_cumulative
            cumulative_after = 0.0
            self.ori6_spectral_rollback_count += 1
        else:
            spectral_rollback_reward = 0.0
            cumulative_after = provisional_cumulative
        spectral_reward = float(
            spectral_progress_reward + spectral_rollback_reward
        )
        self.ori6_spectral_cumulative_reward = float(cumulative_after)
        self.ori6_spectral_current_potential = float(
            metrics_after["potential"]
        )
        if accepted_valid:
            self.ori6_spectral_maximum_potential = max(
                self.ori6_spectral_maximum_potential,
                float(metrics_after["potential"]),
            )
            self.ori6_spectral_maximum_delta = max(
                self.ori6_spectral_maximum_delta,
                float(metrics_after["delta"]),
            )
            self.ori6_spectral_maximum_rho = max(
                self.ori6_spectral_maximum_rho,
                float(metrics_after["rho"]),
            )

        # Preserve exact parent floats when the ablation weight is zero.
        reward = (
            core_reward
            if self.ori6_spectral_weight == 0.0
            else float(core_reward + spectral_reward)
        )
        audited = (
            metrics_before["rho"],
            metrics_before["delta"],
            metrics_before["potential"],
            metrics_after["rho"],
            metrics_after["delta"],
            metrics_after["potential"],
            potential_difference,
            spectral_progress_reward,
            spectral_rollback_reward,
            spectral_reward,
            cumulative_before,
            cumulative_after,
            core_reward,
            reward,
        )
        nonfinite_count = int(sum(not np.isfinite(value) for value in audited))
        decomposition_error = float(
            reward - core_reward
            - spectral_progress_reward
            - spectral_rollback_reward
        )
        fields = {
            "ori6_spectral_candidate": CANDIDATE_NAME,
            "ori6_spectral_candidate_schema_version": (
                CANDIDATE_SCHEMA_VERSION
            ),
            "ori6_spectral_formula_sha256": SPECTRAL_FORMULA_SHA256,
            "ori6_spectral_weight": float(self.ori6_spectral_weight),
            "ori6_spectral_relative_epsilon": float(
                self.ori6_spectral_relative_epsilon
            ),
            "ori6_spectral_subspace_dimension": int(
                self.flatness_subspace_dimension
            ),
            "ori6_spectral_accepted_valid_transition": int(accepted_valid),
            "ori6_spectral_rejected_crossing_zeroed": int(
                rejected_crossing
            ),
            "ori6_spectral_potential_difference": potential_difference,
            "ori6_spectral_progress_reward": spectral_progress_reward,
            "ori6_spectral_rollback_reward": float(
                spectral_rollback_reward
            ),
            "ori6_spectral_reward": spectral_reward,
            "ori6_spectral_core_reward": float(core_reward),
            "ori6_spectral_cumulative_reward_before": cumulative_before,
            "ori6_spectral_cumulative_reward_after": float(
                cumulative_after
            ),
            "ori6_spectral_rollback_reason": rollback_reason,
            "ori6_spectral_reward_decomposition_error": (
                decomposition_error
            ),
            "ori6_spectral_nonfinite_component_count": nonfinite_count,
            **self._metric_log_fields("before", metrics_before),
            **self._metric_log_fields("after", metrics_after),
        }
        info.update(fields)
        if not self.ori5_reward_trace:
            raise RuntimeError("spectral step did not create a parent reward trace")
        self.ori5_reward_trace[-1].update(fields)
        return core_state, reward, done, info


CommonRewardTargetFoldingEnv = SpectralSubspaceTargetFoldingEnv


def make_environment(args, node, member, fix):
    environment = _BASE_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.__class__ = SpectralSubspaceTargetFoldingEnv
    environment._enable_spectral_subspace(
        args.ori6_spectral_weight,
        args.ori6_spectral_relative_epsilon,
    )
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

    total_progress = total("ori6_spectral_progress_reward")
    total_rollback = total("ori6_spectral_rollback_reward")
    total_spectral = total("ori6_spectral_reward")
    total_core = total("ori6_spectral_core_reward")
    action_reward_total = float(sum(float(row["reward"]) for row in actions))
    untraced_core = float(summary["total_reward"] - action_reward_total)
    decomposition_error = float(
        summary["total_reward"]
        - untraced_core
        - total_core
        - total_progress
        - total_rollback
    )

    # Parent ledgers intentionally remain semantically pure.  Correct their
    # episode-level residuals by the separately audited additive component.
    summary["ori6_reward_decomposition_error"] = float(
        summary["ori6_reward_decomposition_error"] - total_spectral
    )
    summary["ori5_reward_decomposition_error"] = float(
        summary["ori5_reward_decomposition_error"] - total_spectral
    )
    terminal = env.spectral_subspace_metrics()
    summary.update({
        "ori6_spectral_candidate": CANDIDATE_NAME,
        "ori6_spectral_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_spectral_formula_sha256": SPECTRAL_FORMULA_SHA256,
        "ori6_spectral_parent_candidate": rejected.CANDIDATE_NAME,
        "ori6_spectral_weight": float(env.ori6_spectral_weight),
        "ori6_spectral_relative_epsilon": float(
            env.ori6_spectral_relative_epsilon
        ),
        "ori6_spectral_reference_projector_sha256": (
            env.ori6_spectral_reference_projector_sha256
        ),
        "ori6_spectral_reference_node_sha256": (
            env.ori6_spectral_reference_node_sha256
        ),
        "ori6_spectral_total_progress_reward": total_progress,
        "ori6_spectral_total_rollback_reward": total_rollback,
        "ori6_spectral_total_reward": total_spectral,
        "ori6_spectral_total_core_reward": total_core,
        "ori6_spectral_untraced_core_reward": untraced_core,
        "ori6_spectral_cumulative_reward": float(
            env.ori6_spectral_cumulative_reward
        ),
        "ori6_spectral_terminal_potential": float(terminal["potential"]),
        "ori6_spectral_terminal_delta": float(terminal["delta"]),
        "ori6_spectral_terminal_rho": float(terminal["rho"]),
        "ori6_spectral_maximum_potential": float(
            env.ori6_spectral_maximum_potential
        ),
        "ori6_spectral_maximum_delta": float(
            env.ori6_spectral_maximum_delta
        ),
        "ori6_spectral_maximum_rho": float(
            env.ori6_spectral_maximum_rho
        ),
        "ori6_spectral_rollback_count": int(
            env.ori6_spectral_rollback_count
        ),
        "ori6_spectral_reward_decomposition_error": decomposition_error,
        "ori6_spectral_maximum_abs_step_reward_decomposition_error": float(
            max([
                abs(float(row.get(
                    "ori6_spectral_reward_decomposition_error", 0.0
                )))
                for row in actions
            ] or [0.0])
        ),
        "ori6_spectral_nonfinite_component_count": int(sum(
            int(row.get("ori6_spectral_nonfinite_component_count", 0))
            for row in actions
        )),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_AGGREGATE_EVALUATIONS(evaluations, episode=episode)
    if result is None:
        return None
    for source, destination in {
        "ori6_spectral_total_reward": "mean_ori6_spectral_total_reward",
        "ori6_spectral_total_progress_reward": (
            "mean_ori6_spectral_total_progress_reward"
        ),
        "ori6_spectral_total_rollback_reward": (
            "mean_ori6_spectral_total_rollback_reward"
        ),
        "ori6_spectral_terminal_potential": (
            "mean_ori6_spectral_terminal_potential"
        ),
        "ori6_spectral_maximum_potential": (
            "mean_ori6_spectral_maximum_potential"
        ),
        "ori6_spectral_rollback_count": (
            "mean_ori6_spectral_rollback_count"
        ),
    }.items():
        result[destination] = float(np.mean([
            float(row[source]) for row in evaluations
        ]))
    result["maximum_abs_ori6_spectral_reward_decomposition_error"] = float(
        max(abs(float(row["ori6_spectral_reward_decomposition_error"]))
            for row in evaluations)
    )
    return result


def reward_parameters(args):
    parameters = _BASE_REWARD_PARAMETERS(args)
    parent_candidate = parameters.get("candidate", rejected.CANDIDATE_NAME)
    parameters.update({
        "candidate": CANDIDATE_NAME,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": parent_candidate,
        "spectral_subspace": {
            "weight": float(args.ori6_spectral_weight),
            "relative_epsilon": float(
                args.ori6_spectral_relative_epsilon
            ),
            "formula": copy.deepcopy(SPECTRAL_FORMULA),
            "formula_sha256": SPECTRAL_FORMULA_SHA256,
            "current_state_markov": True,
            "reference_reset_exact": True,
            "accepted_valid_transitions_only": True,
            "weight_zero_parent_bit_exact": True,
        },
    })
    return parameters


def save_learning_curves(output_directory, rows, window):
    _BASE_SAVE_LEARNING_CURVES(output_directory, rows, window)
    if not rows:
        return
    episodes = np.asarray([float(row["episode"]) for row in rows])
    figure, axes = ori3.plt.subplots(1, 3, figsize=(14.5, 4.0))
    fields = (
        (
            "ori6_spectral_maximum_potential",
            "Maximum spectral potential",
            "tab:blue",
        ),
        (
            "ori6_spectral_total_reward",
            "Retained spectral reward",
            "tab:green",
        ),
        (
            "ori6_spectral_total_rollback_reward",
            "Spectral rollback reward",
            "tab:red",
        ),
    )
    for axis, (field, title, color) in zip(axes, fields):
        values = np.asarray([float(row[field]) for row in rows])
        axis.plot(episodes, values, color="0.82", linewidth=0.5)
        x, average = rejected.ori6._moving_average(
            episodes,
            values,
            window,
        )
        if len(x):
            axis.plot(x, average, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "ori6_spectral_subspace_learning_curve.png",
        dpi=180,
    )
    ori3.plt.close(figure)


@contextmanager
def _patched_rejected_training_runtime():
    original = (
        rejected.make_environment,
        rejected.run_episode,
        rejected.aggregate_stochastic_evaluations,
        rejected.reward_parameters,
        rejected.save_learning_curves,
    )
    rejected.make_environment = make_environment
    rejected.run_episode = run_episode
    rejected.aggregate_stochastic_evaluations = (
        aggregate_stochastic_evaluations
    )
    rejected.reward_parameters = reward_parameters
    rejected.save_learning_curves = save_learning_curves
    try:
        yield
    finally:
        (
            rejected.make_environment,
            rejected.run_episode,
            rejected.aggregate_stochastic_evaluations,
            rejected.reward_parameters,
            rejected.save_learning_curves,
        ) = original


def train(args):
    _validate_spectral_args(args)
    parameters = reward_parameters(args)
    with _patched_rejected_training_runtime():
        output_directory, summary = rejected.train(args)

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
        "ori6_parent_candidate": rejected.CANDIDATE_NAME,
        "ori6_spectral_formula_sha256": SPECTRAL_FORMULA_SHA256,
        "ori6_reward_parameters": parameters,
        "last_1000_mean_ori6_spectral_total_reward": float(np.mean([
            float(row["ori6_spectral_total_reward"]) for row in recent
        ])),
        "last_1000_mean_ori6_spectral_maximum_potential": float(np.mean([
            float(row["ori6_spectral_maximum_potential"])
            for row in recent
        ])),
        "maximum_abs_ori6_spectral_reward_decomposition_error": float(max(
            [abs(float(row[
                "ori6_spectral_reward_decomposition_error"
            ])) for row in rows]
            or [0.0]
        )),
        "ori6_spectral_nonfinite_component_count": int(sum(
            int(float(row["ori6_spectral_nonfinite_component_count"]))
            for row in rows
        )),
    })

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "ori6_candidate": CANDIDATE_NAME,
        "ori6_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "ori6_parent_candidate": rejected.CANDIDATE_NAME,
        "ori6_reward_parameters": parameters,
        "spectral_subspace_formula": copy.deepcopy(SPECTRAL_FORMULA),
        "spectral_subspace_formula_sha256": SPECTRAL_FORMULA_SHA256,
        "candidate_entrypoint": Path(__file__).name,
        "base_and_rejected_authorities_unmodified": True,
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


def _validate_spectral_args(args):
    if not bool(args.ori6_spectral_subspace):
        raise ValueError(f"Candidate requires explicit {CLI_FLAG} opt-in.")
    if not bool(args.ori6_rejected_crossing_continuation):
        raise ValueError("Spectral candidate requires rejected crossing parent.")
    weight = float(args.ori6_spectral_weight)
    epsilon = float(args.ori6_spectral_relative_epsilon)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("--ori6-spectral-weight must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(
            "--ori6-spectral-relative-epsilon must be finite and non-negative"
        )


def _spectral_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(CLI_FLAG, action="store_true", required=True)
    parser.add_argument(
        "--ori6-spectral-weight",
        type=float,
        default=DEFAULT_SPECTRAL_WEIGHT,
    )
    parser.add_argument(
        "--ori6-spectral-relative-epsilon",
        type=float,
        default=DEFAULT_RELATIVE_EPSILON,
    )
    return parser


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    spectral, remaining = _spectral_parser().parse_known_args(arguments)
    if not rejected._option_was_supplied(remaining, rejected.CLI_FLAG):
        remaining = [rejected.CLI_FLAG, *remaining]
    args = rejected.parse_args(remaining)
    args.ori6_spectral_subspace = bool(spectral.ori6_spectral_subspace)
    args.ori6_spectral_weight = float(spectral.ori6_spectral_weight)
    args.ori6_spectral_relative_epsilon = float(
        spectral.ori6_spectral_relative_epsilon
    )
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        args.output_dir = "ori_codex6_spectral_subspace_results"
    _validate_spectral_args(args)
    return args


if __name__ == "__main__":
    train(parse_args())
