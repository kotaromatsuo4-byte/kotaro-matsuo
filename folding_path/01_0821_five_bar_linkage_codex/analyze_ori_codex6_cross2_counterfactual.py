"""Offline saved-path audit for ori6 P0 with crossing penalty 2.0.

Only the existing ``ori5_crossing_penalty`` is changed.  The ori6 core mode is
still ``state_potential``; no trainer or mechanics source is modified.  This
script replays archived metric sequences through the actual ori6 reward ledger
and writes a derived JSON authority below ``experiments_ori6/analysis``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

import collision_3d
import ori_codex3 as ori3
import ori_codex6 as ori6
import run_ori_codex6_experiments as runner


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    runner.EXPERIMENT_ROOT / "analysis" / "offline_P0X2_saved_path_ordering.json"
)
INITIAL_WATER_SPREAD = 1.34728491587901
INITIAL_TORUS_SPREAD = 2.414213685053052
BASE_PENALTY = 1.25
CANDIDATE_PENALTY = 2.0

WATER_PATHS = {
    "ideal_success": ROOT / "experiments_3d" / "feasibility"
    / "waterbomb_spread1_collision_free_path" / "actions.json",
    "near_cross": ROOT / "experiments_3d" / "pilot"
    / "water1_B_b8_lr5e4_s107" / "final_training_path" / "actions.json",
    "wide_wrong": ROOT / "experiments_3d" / "pilot"
    / "water1_B_b8_lr5e4_s107" / "best_flatness_path" / "actions.json",
}
WATERSQRT2_PATHS = {
    "ideal_success": ROOT / "experiments_ori5" / "full" / "watersqrt2"
    / "B" / "seed2101" / "first_success_path" / "actions.json",
    "pilot_timeout": ROOT / "experiments_3d" / "pilot"
    / "watersqrt2_B_b8_lr5e4_s107" / "final_training_path"
    / "actions.json",
}
FIVE_PATHS = {
    "ideal_success": ROOT / "experiments_ori5" / "full" / "five_d2"
    / "B" / "seed2101" / "first_success_path" / "actions.json",
    "wrong_flat": ROOT / "experiments_ori5" / "tuning" / "r1"
    / "five_d2" / "B" / "seed1101" / "best_flatness_path"
    / "actions.json",
}
FIVE_D3_PATHS = {
    "ideal_success": ROOT / "experiments_ori5" / "full" / "five_d3"
    / "B" / "seed2101" / "first_success_path" / "actions.json",
    "wrong_d2": FIVE_PATHS["ideal_success"],
}
TORUS_SUCCESS = (
    ROOT / "experiments_3d" / "feasibility"
    / "torus_spread2_collision_free_path" / "actions.json"
)
TORUS_CMAES_ROOT = (
    ROOT.parent / "02_4unit_torus_CMAES"
    / "randomize+gravity_potential_history2"
)

COMPONENTS = (
    "ori6_flatness_progress_reward",
    "ori6_target_progress_reward",
    "ori6_spread_frontier_reward",
    "ori6_joint_milestone_reward",
    "ori6_terminal_fixed_reward",
    "ori6_terminal_quality_reward",
    "ori6_terminal_speed_reward",
    "ori6_terminal_correction_reward",
)


def metric(flatness: float, spread: float, target: float) -> dict:
    error = abs(float(spread) - float(target))
    return {
        "flatness_ratio": float(flatness),
        "spread": float(spread),
        "spread_error": error,
        "target_deviation_energy": error * error,
    }


class ScriptedRewardEnvironment(ori6.CommonRewardTargetFoldingEnv):
    def compactness_metrics(self, node=None):
        del node
        return dict(self.scripted_metrics[self.scripted_index])


def scripted_environment(
    states: list[dict], target: float, max_steps: int, penalty: float,
    frontier_bonus: float = 0.9,
) -> ScriptedRewardEnvironment:
    env = object.__new__(ScriptedRewardEnvironment)
    env.scripted_metrics = list(states)
    env.scripted_index = 0
    env.target_spread = float(target)
    env.spread_tolerance = 0.05
    env.evaluation_flatness_tolerance = 0.001
    env.near_miss_flatness_far = 1.0
    env.near_miss_spread_error_far = 1.0
    env.ori5_flatness_clip = 2.0
    env.ori5_spread_milestone_initial_factor = 2.0
    env.ori5_spread_milestone_bonus = 0.0
    env.ori5_joint_flatness_thresholds = (0.5, 0.1, 0.01)
    env.ori5_joint_milestone_bonuses = (0.2, 0.3, 0.5)
    env.ori5_crossing_penalty = float(penalty)
    env.ori5_wrong_flat_penalty = 2.0
    env.ori5_timeout_penalty = 1.0
    env.ori5_success_bonus = 10.0
    env.ori5_quality_weight = 1.0
    env.ori5_success_speed_bonus_weight = 0.5
    env.numerical_failure_penalty = 10.0
    env.no_mode_penalty = 5.0
    env.max_steps = int(max_steps)
    env.step_count = 0
    env.ori6_spread_frontier_bonus = float(frontier_bonus)
    env.ori6_spread_frontier_mode = runner.P0_MODE
    env.ori6_spread_frontier_flatness_cap = 2.0
    env._reset_ori5_episode_state()
    env._validate_ori6_parameters()
    env._reset_ori6_episode_state()
    return env


def scripted_parent_step(env, physical_direction, requested_width, coefficients):
    del physical_direction, requested_width, coefficients
    index = env.scripted_index
    env.scripted_index += 1
    env.step_count += 1
    reason = env.scripted_reasons[index]
    valid = env.scripted_validity[index]
    return None, 0.0, reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(valid),
    }


def replay(
    states: list[dict], reasons: list[str], validity: list[bool],
    *, target: float, max_steps: int, penalty: float,
    frontier_bonus: float = 0.9,
) -> dict:
    if len(states) != len(reasons) + 1 or len(reasons) != len(validity):
        raise ValueError("Scripted state/transition lengths are inconsistent")
    env = scripted_environment(
        states, target, max_steps, penalty, frontier_bonus
    )
    env.scripted_reasons = list(reasons)
    env.scripted_validity = list(validity)
    rewards = []
    infos = []
    with mock.patch.object(
        ori3.CompactTargetFoldingEnv, "step", scripted_parent_step,
    ):
        for _ in reasons:
            _, reward, _, info = ori6.CommonRewardTargetFoldingEnv.step(
                env, None, 0.0, []
            )
            rewards.append(float(reward))
            infos.append(info)
    component_sums = {
        name: float(sum(float(info[name]) for info in infos))
        for name in COMPONENTS
    }
    step_identity_error = max(
        [
            abs(float(reward) - sum(float(info[name]) for name in COMPONENTS))
            for reward, info in zip(rewards, infos)
        ] or [0.0]
    )
    result = {
        "total_return": float(sum(rewards)),
        "transition_count": len(reasons),
        "terminal_reason": reasons[-1] if reasons else None,
        "initial_metrics": states[0],
        "last_valid_metrics": (
            states[-2] if reasons and not validity[-1] else states[-1]
        ),
        "candidate_terminal_metrics": states[-1],
        "component_sums": component_sums,
        "step_reward_identity_max_abs_error": float(step_identity_error),
        "reported_decomposition_max_abs_error": float(max(
            [abs(float(info["ori6_reward_decomposition_error"])) for info in infos]
            or [0.0]
        )),
        "nonfinite_reward_component_count": int(sum(
            int(info["ori6_nonfinite_reward_component_count"])
            for info in infos
        )),
        "terminal_ledger": {
            name: infos[-1].get(name) for name in (
                "ori6_terminal_fixed_reward",
                "ori6_terminal_quality_reward",
                "ori6_terminal_speed_reward",
                "ori6_terminal_correction_reward",
                "ori6_frontier_rollback_reward",
                "ori6_joint_milestone_rollback_reward",
                "ori6_quality_source",
            )
        } if infos else {},
    }
    if step_identity_error > 1.0e-10:
        raise RuntimeError(f"Step ledger identity failed: {step_identity_error}")
    if result["reported_decomposition_max_abs_error"] > 1.0e-10:
        raise RuntimeError("Reported reward decomposition failed")
    if result["nonfinite_reward_component_count"]:
        raise RuntimeError("Non-finite reward component found")
    return result


def action_states(
    path: Path, target: float, initial_spread: float,
    *, force_final_reason: str | None = None,
) -> tuple[list[dict], list[str], list[bool], list[dict]]:
    with open(path, encoding="utf-8") as handle:
        actions = json.load(handle)
    states = [metric(1.0, initial_spread, target)]
    states.extend(
        metric(action["flatness_ratio"], action["spread"], target)
        for action in actions
    )
    reasons = [str(action["termination_reason"]) for action in actions]
    validity = [bool(action["after_state_valid"]) for action in actions]
    if force_final_reason is not None:
        reasons[-1] = force_final_reason
        validity[-1] = False
    return states, reasons, validity, actions


def replay_action_path(
    path: Path, *, target: float, initial_spread: float, max_steps: int,
    penalty: float, force_final_reason: str | None = None,
    frontier_bonus: float = 0.9,
) -> dict:
    states, reasons, validity, actions = action_states(
        path, target, initial_spread,
        force_final_reason=force_final_reason,
    )
    result = replay(
        states, reasons, validity, target=target, max_steps=max_steps,
        penalty=penalty, frontier_bonus=frontier_bonus,
    )
    result["source"] = {
        "path": str(path.resolve()),
        "sha256": runner.file_sha256(path),
        "archived_terminal_reason": actions[-1]["termination_reason"],
        "forced_terminal_reason": force_final_reason,
    }
    return result


def torus_metric_functions():
    model_dir = ROOT / "model" / "4unit_torus"
    node = np.atleast_2d(np.loadtxt(model_dir / "node_initial.dat", delimiter=","))
    member = np.atleast_2d(np.loadtxt(
        model_dir / "member.dat", delimiter=",", dtype=int,
    ))
    lengths = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    scale = float(np.max(lengths))

    def potential(coordinates):
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        return float(np.sum(singular[2:] ** 2))

    initial_potential = potential(node)

    def metrics(coordinates):
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ right[:2].T
        spread = float(np.max(np.linalg.norm(
            projected[:, None, :] - projected[None, :, :], axis=2,
        )) / scale)
        return metric(potential(coordinates) / initial_potential, spread, 2.0)

    checker = collision_3d.ContinuousBarCollisionChecker(
        member,
        pair_indices=collision_3d.nonadjacent_member_pairs(member),
        distance_tolerance=1.0e-10,
    )
    return metrics, checker


def replay_torus_cmaes(
    index: int, penalty: float, frontier_bonus: float = 0.9,
) -> dict:
    directory = TORUS_CMAES_ROOT / f"randomize+gravity_potential_history[{index}]"
    paths = sorted(directory.glob("*.dat"), key=lambda path: int(path.stem))
    if len(paths) < 2:
        raise RuntimeError(f"Missing CMA-ES state sequence: {directory}")
    coordinates = [np.atleast_2d(np.loadtxt(path)) for path in paths]
    metric_function, checker = torus_metric_functions()
    states = [metric_function(node) for node in coordinates]
    collision_indices = [
        transition
        for transition, (before, after) in enumerate(
            zip(coordinates[:-1], coordinates[1:]), start=1,
        )
        if checker.check(before, after).collision
    ]
    if collision_indices != [len(paths) - 1]:
        raise RuntimeError(
            f"history[{index}] current collision transitions {collision_indices}"
        )
    reasons = ["running"] * (len(states) - 1)
    reasons[-1] = "self_crossing"
    validity = [True] * (len(states) - 1)
    validity[-1] = False
    result = replay(
        states, reasons, validity, target=2.0, max_steps=200,
        penalty=penalty, frontier_bonus=frontier_bonus,
    )
    hashes = {path.name: runner.file_sha256(path) for path in paths}
    result["source"] = {
        "path": str(directory.resolve()),
        "state_file_count": len(paths),
        "state_file_sha256": hashes,
        "state_manifest_sha256": runner.canonical_digest(hashes),
        "current_collision_transition_indices": collision_indices,
        "start_state_scope": (
            "CMA-ES randomized feasible state0; this is a reward-ledger "
            "ordering audit, not a canonical-reset reachability proof"
        ),
    }
    return result


def pair(base: dict, candidate: dict) -> dict:
    delta = candidate["total_return"] - base["total_return"]
    expected = (
        -(CANDIDATE_PENALTY - BASE_PENALTY)
        if candidate["terminal_reason"] == "self_crossing" else 0.0
    )
    if abs(delta - expected) > 1.0e-10:
        raise RuntimeError(f"Unexpected crossing-penalty delta: {delta} != {expected}")
    return {
        "P0_crossing_penalty_1_25": base,
        "P0X2_crossing_penalty_2_0": candidate,
        "return_delta_P0X2_minus_P0": float(delta),
        "expected_delta": expected,
        "single_parameter_delta_exact": True,
    }


def ordering(name: str, cases: dict, names: list[str]) -> dict:
    totals = {
        case: cases[case]["P0X2_crossing_penalty_2_0"]["total_return"]
        for case in names
    }
    margins = [
        {
            "higher": left,
            "lower": right,
            "return_margin": float(totals[left] - totals[right]),
            "pass": bool(totals[left] > totals[right]),
        }
        for left, right in zip(names[:-1], names[1:])
    ]
    return {
        "structure": name,
        "required_order": names,
        "P0X2_total_returns": totals,
        "adjacent_margins": margins,
        "pass": all(row["pass"] for row in margins),
        "cases": cases,
    }


def main() -> int:
    if runner.candidate_crossing_penalty(runner.P0_CROSS2_MODE) != CANDIDATE_PENALTY:
        raise RuntimeError("Runner P0X2 penalty no longer equals this audit")

    water_cases = {}
    for name, path in WATER_PATHS.items():
        water_cases[name] = pair(*[
            replay_action_path(
                path, target=1.0, initial_spread=INITIAL_WATER_SPREAD,
                max_steps=400, penalty=penalty,
            )
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ])

    with open(FIVE_PATHS["ideal_success"], encoding="utf-8") as handle:
        five_actions = json.load(handle)
    five_initial_spread = float(five_actions[0]["spread_before"])
    five_specs = {
        "ideal_success": (FIVE_PATHS["ideal_success"], None),
        # Same archived feasible prefix, with only its final proposed state
        # declared invalid.  The penultimate state is therefore the audited
        # last-valid near-cross geometry.
        "near_cross": (FIVE_PATHS["ideal_success"], "self_crossing"),
        "wrong_flat": (FIVE_PATHS["wrong_flat"], None),
    }
    five_cases = {}
    for name, (path, forced) in five_specs.items():
        five_cases[name] = pair(*[
            replay_action_path(
                path, target=2.0, initial_spread=five_initial_spread,
                max_steps=400, penalty=penalty,
                force_final_reason=forced,
            )
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ])

    with open(FIVE_D3_PATHS["ideal_success"], encoding="utf-8") as handle:
        five_d3_actions = json.load(handle)
    five_d3_initial_spread = float(five_d3_actions[0]["spread_before"])
    five_d3_specs = {
        "ideal_success": (FIVE_D3_PATHS["ideal_success"], None),
        "near_cross": (FIVE_D3_PATHS["ideal_success"], "self_crossing"),
        # The archived D=2 success is a flat but wrong-spread branch when the
        # target is D=3, so only its terminal label is changed to wrong-flat.
        "wrong_d2": (FIVE_D3_PATHS["wrong_d2"], "wrong_flat"),
    }
    five_d3_cases = {}
    for name, (path, forced) in five_d3_specs.items():
        five_d3_cases[name] = pair(*[
            replay_action_path(
                path, target=3.0, initial_spread=five_d3_initial_spread,
                max_steps=400, penalty=penalty,
                force_final_reason=forced,
            )
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ])

    watersqrt2_specs = {
        "ideal_success": (WATERSQRT2_PATHS["ideal_success"], None),
        "near_cross": (
            WATERSQRT2_PATHS["ideal_success"], "self_crossing",
        ),
        "pilot_timeout": (WATERSQRT2_PATHS["pilot_timeout"], None),
    }
    watersqrt2_cases = {}
    for name, (path, forced) in watersqrt2_specs.items():
        watersqrt2_cases[name] = pair(*[
            replay_action_path(
                path, target=float(np.sqrt(2.0)),
                initial_spread=INITIAL_WATER_SPREAD, max_steps=400,
                penalty=penalty, force_final_reason=forced,
            )
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ])

    torus_cases = {
        "success": pair(*[
            replay_action_path(
                TORUS_SUCCESS, target=2.0,
                initial_spread=INITIAL_TORUS_SPREAD, max_steps=200,
                penalty=penalty,
            )
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ]),
        "h0_near_cross": pair(*[
            replay_torus_cmaes(0, penalty)
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ]),
        "h1_bad_cross": pair(*[
            replay_torus_cmaes(1, penalty)
            for penalty in (BASE_PENALTY, CANDIDATE_PENALTY)
        ]),
    }

    audits = {
        "water1": ordering(
            "waterbomb D=1", water_cases,
            ["ideal_success", "near_cross", "wide_wrong"],
        ),
        "torus2": ordering(
            "4-unit torus D=2", torus_cases,
            ["success", "h0_near_cross", "h1_bad_cross"],
        ),
        "five_d2": ordering(
            "five-bar D=2", five_cases,
            ["ideal_success", "near_cross", "wrong_flat"],
        ),
        "five_d3": ordering(
            "five-bar D=3", five_d3_cases,
            ["ideal_success", "near_cross", "wrong_d2"],
        ),
        "watersqrt2": ordering(
            "waterbomb D=sqrt(2)", watersqrt2_cases,
            ["ideal_success", "near_cross", "pilot_timeout"],
        ),
    }
    all_pass = all(value["pass"] for value in audits.values())
    if not all_pass:
        raise RuntimeError("P0X2 violates a required saved-path ordering")
    result = {
        "schema_version": 1,
        "analysis_type": "offline_saved_metric_path_reward_replay",
        "candidate_mode": runner.P0_CROSS2_MODE,
        "candidate_mode_slug": runner.MODE_SLUGS[runner.P0_CROSS2_MODE],
        "core_frontier_mode": runner.core_frontier_mode(
            runner.P0_CROSS2_MODE
        ),
        "single_changed_reward_parameter": {
            "name": "ori5_crossing_penalty",
            "P0_value": BASE_PENALTY,
            "candidate_value": CANDIDATE_PENALTY,
        },
        "candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ),
        "all_required_orderings_pass": all_pass,
        "orderings": audits,
        "interpretation_limits": [
            "This verifies reward ordering on archived metric sequences; it does not estimate policy discovery probability.",
            "The five-bar near-cross case is a counterfactual invalid final proposal on an archived successful prefix.",
            "The five-bar D=3 wrong-D=2 case re-labels an archived D=2 success as wrong-flat under target D=3.",
            "The waterbomb sqrt(2) near-cross case is a counterfactual invalid final proposal on an archived successful prefix.",
            "The two CMA-ES torus cases begin at their archived randomized feasible state0, not at the canonical RL reset.",
        ],
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task)
            for name, task in runner.TASKS.items()
        },
    }
    runner.atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
