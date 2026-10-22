"""Offline mechanics/reward authority for the water1 ``h=0.2`` screen.

The archived constructive waterbomb-D=1 success used predictor widths up to
0.3.  This audit subdivides every archived interval, re-projects each
sub-target secant into the *current* mechanism nullspace, canonicalises the
result exactly as a policy action, and applies the unchanged repository
predictor/corrector with the continuous 3-D collision checker at ``h=0.2``.

It is a reachability and return-ordering witness only.  It makes no claim
about the probability that a learned or uniformly random policy finds it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

import analyze_ori_codex6_cross2_counterfactual as counterfactual
import ori_codex6_3d as adapter
import run_ori_codex6_experiments as runner


ROOT = Path(__file__).resolve().parent
ARCHIVE = (
    ROOT / "experiments_3d" / "feasibility"
    / "waterbomb_spread1_collision_free_path"
)
ARCHIVED_ACTIONS = ARCHIVE / "actions.json"
ARCHIVED_SUMMARY = ARCHIVE / "summary.json"
PARENT_ORDERING = runner.P0_CROSS2_OFFLINE_AUDIT
OUTPUT = (
    runner.EXPERIMENT_ROOT / "analysis"
    / "offline_P0X2W2_water1_width_feasibility.json"
)
TARGET_SPREAD = 1.0
PARENT_MAX_STEP_WIDTH = 0.3
CANDIDATE_MAX_STEP_WIDTH = 0.2
MAX_STEPS = 400
COMPONENTS = counterfactual.COMPONENTS


def _load_node(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, dtype=float))


def _P0X2_arguments() -> list[str]:
    common = list(runner.COMMON_ARGUMENTS)
    penalty_index = common.index("--ori5-crossing-penalty")
    common[penalty_index + 1] = format(
        runner.candidate_crossing_penalty(runner.P0_CROSS2_MODE), ".17g"
    )
    return [
        "--model", "waterbomb",
        "--max-steps", str(MAX_STEPS),
        "--max-step-width", format(CANDIDATE_MAX_STEP_WIDTH, ".17g"),
        "--target-spread", format(TARGET_SPREAD, ".17g"),
        "--output-dir", str(runner.EXPERIMENT_ROOT / "_offline_width2_unused"),
        *common,
        "--ori5-spread-milestone-bonus", "0",
        "--ori6-spread-frontier-mode", runner.P0_MODE,
        "--ori6-spread-frontier-bonus", format(
            runner.candidate_frontier_bonus(runner.P0_CROSS2_MODE), ".17g"
        ),
        "--ori6-spread-frontier-flatness-cap", "2",
        "--data-dir", str((ROOT / "model" / "waterbomb").resolve()),
        "--no-rigid-diagnostics",
    ]


def _canonical_tangent_action(
    modes: np.ndarray, secant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return a policy-representable canonical tangent direction and sign."""

    secant = np.asarray(secant, dtype=float).reshape(-1)
    projection = modes @ (modes.T @ secant)
    norm = float(np.linalg.norm(projection))
    if norm <= 1.0e-12:
        raise RuntimeError("A subdivided reference secant has zero tangent projection")
    direction = projection / norm
    coefficients = modes.T @ direction
    sign = 1.0
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
        coefficients = -coefficients
        sign = -1.0
    reconstruction = modes @ coefficients
    error = float(np.linalg.norm(reconstruction - direction))
    return direction, coefficients, sign, error


def replay_subdivided_success() -> dict:
    with open(ARCHIVED_ACTIONS, encoding="utf-8") as handle:
        archived_actions = json.load(handle)
    archived_nodes = [
        _load_node(ARCHIVE / f"step{index}.dat")
        for index in range(len(archived_actions) + 1)
    ]

    args = adapter.parse_args(_P0X2_arguments())
    node, member, fix, model_files = adapter.load_problem_data(args.data_dir)
    lengths = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    fixed_indices = 3 * fix[:, 0] + fix[:, 1]
    records: list[dict] = []
    rewards: list[float] = []
    done = False

    with adapter.patched_three_dimensional_runtime(args, member):
        env = adapter.make_environment(args, node, member, fix)
        env.configure_episode_tolerance(training=False)
        env.reset()
        for archived_index, archived_action in enumerate(archived_actions):
            width = abs(float(archived_action["requested_width"]))
            subdivision_count = max(
                1, int(math.ceil(width / CANDIDATE_MAX_STEP_WIDTH))
            )
            reference_before = archived_nodes[archived_index]
            reference_after = archived_nodes[archived_index + 1]
            for subdivision_index in range(1, subdivision_count + 1):
                target = reference_before + (
                    subdivision_index / subdivision_count
                ) * (reference_after - reference_before)
                modes = env.get_modes()
                direction, coefficients, sign, reconstruction_error = (
                    _canonical_tangent_action(modes, target - env.node)
                )
                target_distance = float(np.linalg.norm(target - env.node))
                requested_width = sign * target_distance
                if abs(requested_width) > CANDIDATE_MAX_STEP_WIDTH + 1.0e-12:
                    raise RuntimeError(
                        "Reference subdivision exceeds the candidate width: "
                        f"{requested_width}"
                    )
                jacobian = adapter.mechanics.dedx(
                    env.node, member, env.free
                )
                tangency_residual = float(np.linalg.norm(
                    jacobian @ direction[env.free]
                ))
                _, reward, done, info = env.step(
                    direction, requested_width, coefficients
                )
                metrics = env.compactness_metrics()
                rewards.append(float(reward))
                records.append({
                    "step": int(env.step_count),
                    "archived_interval": int(archived_index + 1),
                    "subdivision_index": int(subdivision_index),
                    "subdivision_count": int(subdivision_count),
                    "mode_count": int(modes.shape[1]),
                    "requested_width": float(info["requested_width"]),
                    "accepted_width": float(info["accepted_width"]),
                    "projection_attempts": int(info["projection_attempts"]),
                    "correction_norm": float(info["correction_norm"]),
                    "actual_displacement": float(info["actual_displacement"]),
                    "correction_to_width_ratio": float(
                        info["correction_to_width_ratio"]
                    ),
                    "actual_to_width_ratio": float(
                        info["actual_to_width_ratio"]
                    ),
                    "mechanism_tangency_residual": tangency_residual,
                    "mode_reconstruction_error": reconstruction_error,
                    "reference_target_coordinate_max_abs_error": float(
                        np.max(np.abs(env.node - target))
                    ),
                    "strain": float(info["strain"]),
                    "flatness_ratio": float(info["flatness_ratio"]),
                    "spread": float(metrics["spread"]),
                    "spread_error": float(metrics["spread_error"]),
                    "termination_reason": str(info["termination_reason"]),
                    "after_state_valid": int(info["after_state_valid"]),
                    "self_crossing": int(info["self_crossing"]),
                    "correction_failed": int(info["correction_failed"]),
                    "reward": float(reward),
                    "reward_decomposition_error": float(
                        info["ori6_reward_decomposition_error"]
                    ),
                })
                if done:
                    break
            if done:
                break

        component_sums = {
            name: float(sum(float(row.get(name, 0.0)) for row in env.ori5_reward_trace))
            for name in COMPONENTS
        }
        final_metrics = env.compactness_metrics()
        member_error = np.abs(
            np.linalg.norm(
                env.node[member[:, 0]] - env.node[member[:, 1]], axis=1
            ) - lengths
        )
        fixed_drift = np.abs(
            env.node.reshape(-1)[fixed_indices] - node.reshape(-1)[fixed_indices]
        )
        result = {
            "success": bool(records and records[-1]["termination_reason"] == "success"),
            "termination_reason": records[-1]["termination_reason"] if records else None,
            "transition_count": len(records),
            "horizon_limit": MAX_STEPS,
            "maximum_absolute_requested_width": float(max(
                (abs(row["requested_width"]) for row in records), default=0.0
            )),
            "maximum_absolute_accepted_width": float(max(
                (abs(row["accepted_width"]) for row in records), default=0.0
            )),
            "self_crossing_count": int(sum(
                row["self_crossing"] for row in records
            )),
            "correction_failure_count": int(sum(
                row["correction_failed"] for row in records
            )),
            "all_after_states_valid": bool(all(
                row["after_state_valid"] for row in records
            )),
            "maximum_mechanism_tangency_residual": float(max(
                (row["mechanism_tangency_residual"] for row in records),
                default=0.0,
            )),
            "maximum_mode_reconstruction_error": float(max(
                (row["mode_reconstruction_error"] for row in records),
                default=0.0,
            )),
            "maximum_reference_target_coordinate_max_abs_error": float(max(
                (row["reference_target_coordinate_max_abs_error"] for row in records),
                default=0.0,
            )),
            "final_archived_coordinate_max_abs_error": float(np.max(np.abs(
                env.node - archived_nodes[-1]
            ))),
            "maximum_member_length_error": float(max(member_error, default=0.0)),
            "maximum_fixed_coordinate_drift": float(max(fixed_drift, default=0.0)),
            "final_flatness_ratio": float(final_metrics["flatness_ratio"]),
            "final_spread": float(final_metrics["spread"]),
            "final_spread_error": float(final_metrics["spread_error"]),
            "total_return": float(sum(rewards)),
            "component_sums": component_sums,
            "step_reward_identity_max_abs_error": float(max(
                (abs(row["reward_decomposition_error"]) for row in records),
                default=0.0,
            )),
            "environment": {
                "max_steps": int(args.max_steps),
                "max_step_width": float(args.max_step_width),
                "min_step_width": float(args.min_step_width),
                "projection_shrink": float(args.projection_shrink),
                "max_correction_ratio": float(args.max_correction_ratio),
                "max_actual_step_ratio": float(args.max_actual_step_ratio),
                "flatness_tolerance": float(args.flatness_tolerance),
                "spread_tolerance": float(args.spread_tolerance),
                "strain_tolerance": float(args.strain_tolerance),
                "crossing_tolerance": float(args.crossing_tolerance),
                "target_spread": float(args.target_spread),
            },
            "model_file_sha256": {
                name: runner.file_sha256(path)
                for name, path in model_files.items()
            },
            "transitions": records,
        }
    return result


def build_report() -> dict:
    replay = replay_subdivided_success()
    with open(PARENT_ORDERING, encoding="utf-8") as handle:
        parent = json.load(handle)
    water_parent = parent["orderings"]["water1"]
    near_cross = counterfactual.replay_action_path(
        counterfactual.WATER_PATHS["near_cross"],
        target=TARGET_SPREAD,
        initial_spread=counterfactual.INITIAL_WATER_SPREAD,
        max_steps=MAX_STEPS,
        penalty=runner.candidate_crossing_penalty(runner.P0_CROSS2_MODE),
        frontier_bonus=runner.candidate_frontier_bonus(runner.P0_CROSS2_MODE),
    )
    wide_wrong = counterfactual.replay_action_path(
        counterfactual.WATER_PATHS["wide_wrong"],
        target=TARGET_SPREAD,
        initial_spread=counterfactual.INITIAL_WATER_SPREAD,
        max_steps=MAX_STEPS,
        penalty=runner.candidate_crossing_penalty(runner.P0_CROSS2_MODE),
        frontier_bonus=runner.candidate_frontier_bonus(runner.P0_CROSS2_MODE),
    )
    returns = {
        "subdivided_ideal_success": replay["total_return"],
        "near_cross": near_cross["total_return"],
        "wide_wrong": wide_wrong["total_return"],
    }
    ordering_pass = bool(
        returns["subdivided_ideal_success"] > returns["near_cross"]
        > returns["wide_wrong"]
    )
    feasibility_pass = bool(
        replay["success"]
        and replay["transition_count"] <= MAX_STEPS
        and replay["maximum_absolute_requested_width"]
        <= CANDIDATE_MAX_STEP_WIDTH + 1.0e-12
        and replay["self_crossing_count"] == 0
        and replay["correction_failure_count"] == 0
        and replay["all_after_states_valid"]
        and replay["step_reward_identity_max_abs_error"] <= 1.0e-10
    )
    report = {
        "schema_version": 1,
        "authority_type": "ori6_water1_subdivided_mechanics_feasibility",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_mode_slug": "P0X2W2",
        "parent_candidate_mode": runner.P0_CROSS2_MODE,
        "task": "water1",
        "single_changed_mechanics_parameter": {
            "name": "max_step_width",
            "P0X2_value": PARENT_MAX_STEP_WIDTH,
            "candidate_value": CANDIDATE_MAX_STEP_WIDTH,
        },
        "reward_identity_reused_from_parent": True,
        "candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ),
        "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ),
        "optimizer_identity_reused_from_parent": True,
        "candidate_optimizer_sha256": runner.candidate_optimizer_sha256(
            runner.P0_CROSS2_MODE,
            runner.PROFILES[runner.TASKS["water1"].profile],
        ),
        "parent_candidate_optimizer_sha256": runner.candidate_optimizer_sha256(
            runner.P0_CROSS2_MODE,
            runner.PROFILES[runner.TASKS["water1"].profile],
        ),
        "source_saved_success": {
            "actions_path": str(ARCHIVED_ACTIONS.resolve()),
            "actions_sha256": runner.file_sha256(ARCHIVED_ACTIONS),
            "summary_path": str(ARCHIVED_SUMMARY.resolve()),
            "summary_sha256": runner.file_sha256(ARCHIVED_SUMMARY),
            "step_file_sha256": {
                f"step{index}.dat": runner.file_sha256(
                    ARCHIVE / f"step{index}.dat"
                )
                for index in range(21)
            },
        },
        "replay_method": {
            "description": (
                "Subdivide each archived interval by ceil(abs(archived "
                "requested_width)/0.2); at every current corrected state, "
                "project the next linearly interpolated reference sub-target "
                "secant into the instantaneous mechanism nullspace, "
                "canonicalise the direction/sign, then call the unchanged "
                "predictor/corrector and continuous 3-D collision checker."
            ),
            "policy_action_representability_checked": True,
            "continuous_collision_checked_on_every_corrected_transition": True,
            "metric_interpolation_used_for_mechanics": False,
        },
        "subdivided_replay": replay,
        "return_ordering": {
            "required_order": [
                "subdivided_ideal_success", "near_cross", "wide_wrong"
            ],
            "P0X2_returns": returns,
            "adjacent_margins": [
                returns["subdivided_ideal_success"] - returns["near_cross"],
                returns["near_cross"] - returns["wide_wrong"],
            ],
            "pass": ordering_pass,
            "parent_water1_required_order": water_parent["required_order"],
            "parent_water1_pass": water_parent["pass"],
            "unchanged_reward_all_five_task_ordering_authority": str(
                PARENT_ORDERING.resolve()
            ),
            "unchanged_reward_all_five_task_ordering_authority_sha256": (
                runner.file_sha256(PARENT_ORDERING)
            ),
            "other_four_tasks_mechanics_unchanged_by_water1_only_screen": True,
        },
        "feasibility_gate": {
            "pass": feasibility_pass,
            "strict_success": replay["success"],
            "collision_free": replay["self_crossing_count"] == 0,
            "correction_failure_free": replay["correction_failure_count"] == 0,
            "transition_count_le_400": replay["transition_count"] <= MAX_STEPS,
            "all_requested_widths_le_0_2": (
                replay["maximum_absolute_requested_width"]
                <= CANDIDATE_MAX_STEP_WIDTH + 1.0e-12
            ),
            "reward_ordering_unchanged": ordering_pass,
            "reward_integrity": (
                replay["step_reward_identity_max_abs_error"] <= 1.0e-10
            ),
        },
        "all_required_checks_pass": bool(feasibility_pass and ordering_pass),
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task)
            for name, task in runner.TASKS.items()
        },
        "scope_limitations": [
            "This is a constructive reachability witness, not a discovery-probability estimate.",
            "No new random baseline is claimed; one is required only if P0X2W2 is adopted.",
            "Only water1 mechanics change in the bounded diagnostic.",
        ],
    }
    if not report["all_required_checks_pass"]:
        raise RuntimeError("The h=0.2 saved-path feasibility gate did not pass")
    return report


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
