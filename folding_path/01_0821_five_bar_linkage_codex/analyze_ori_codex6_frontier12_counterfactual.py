"""Saved-path audit for P0X2 with frontier bonus 0.9 -> 1.2.

The core reward implementation and mechanics are unchanged.  Both sides use
crossing penalty 2.0 and the core ``state_potential`` mode; only the existing
``ori6_spread_frontier_bonus`` CLI parameter differs.  Archived metric paths
for all five target tasks are replayed through the actual ori6 reward ledger.
"""

from __future__ import annotations

import json

import numpy as np

import analyze_ori_codex6_cross2_counterfactual as base
import run_ori_codex6_experiments as runner


OUTPUT = (
    runner.EXPERIMENT_ROOT / "analysis"
    / "offline_P0X2B12_saved_path_ordering.json"
)
PARENT_MODE = runner.P0_CROSS2_MODE
CANDIDATE_MODE = runner.P0_CROSS2_FRONTIER12_MODE
CROSSING_PENALTY = 2.0
BASE_BONUS = 0.9
CANDIDATE_BONUS = 1.2
ALLOWED_COMPONENT_DELTAS = {
    "ori6_spread_frontier_reward",
    # Crossing/correction rollback is recorded in this terminal component.
    "ori6_terminal_correction_reward",
}


def pair(parent: dict, candidate: dict) -> dict:
    component_deltas = {
        name: float(
            candidate["component_sums"][name]
            - parent["component_sums"][name]
        )
        for name in base.COMPONENTS
    }
    unexpected = {
        name: value
        for name, value in component_deltas.items()
        if name not in ALLOWED_COMPONENT_DELTAS and abs(value) > 1.0e-10
    }
    return_delta = float(candidate["total_return"] - parent["total_return"])
    component_delta = float(sum(component_deltas.values()))
    exact = not unexpected and abs(return_delta - component_delta) <= 1.0e-10
    if not exact:
        raise RuntimeError(
            "Frontier-only replay delta failed: "
            f"return={return_delta}, components={component_delta}, "
            f"unexpected={unexpected}"
        )
    return {
        "P0X2_frontier_bonus_0_9": parent,
        "P0X2B12_frontier_bonus_1_2": candidate,
        "return_delta_P0X2B12_minus_P0X2": return_delta,
        "component_deltas": component_deltas,
        "allowed_component_delta_names": sorted(ALLOWED_COMPONENT_DELTAS),
        "single_parameter_delta_exact": True,
    }


def paired_action_path(
    path, *, target: float, initial_spread: float, max_steps: int,
    force_final_reason: str | None = None,
) -> dict:
    return pair(*[
        base.replay_action_path(
            path, target=target, initial_spread=initial_spread,
            max_steps=max_steps, penalty=CROSSING_PENALTY,
            force_final_reason=force_final_reason,
            frontier_bonus=bonus,
        )
        for bonus in (BASE_BONUS, CANDIDATE_BONUS)
    ])


def paired_torus_cmaes(index: int) -> dict:
    return pair(*[
        base.replay_torus_cmaes(
            index, CROSSING_PENALTY, frontier_bonus=bonus,
        )
        for bonus in (BASE_BONUS, CANDIDATE_BONUS)
    ])


def ordering(name: str, cases: dict, names: list[str]) -> dict:
    totals = {
        case: cases[case]["P0X2B12_frontier_bonus_1_2"]["total_return"]
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
        "P0X2B12_total_returns": totals,
        "adjacent_margins": margins,
        "pass": all(row["pass"] for row in margins),
        "cases": cases,
    }


def build_orderings() -> dict:
    water_cases = {
        name: paired_action_path(
            path, target=1.0, initial_spread=base.INITIAL_WATER_SPREAD,
            max_steps=400,
        )
        for name, path in base.WATER_PATHS.items()
    }

    with open(base.FIVE_PATHS["ideal_success"], encoding="utf-8") as handle:
        five_initial_spread = float(json.load(handle)[0]["spread_before"])
    five_specs = {
        "ideal_success": (base.FIVE_PATHS["ideal_success"], None),
        "near_cross": (base.FIVE_PATHS["ideal_success"], "self_crossing"),
        "wrong_flat": (base.FIVE_PATHS["wrong_flat"], None),
    }
    five_cases = {
        name: paired_action_path(
            path, target=2.0, initial_spread=five_initial_spread,
            max_steps=400, force_final_reason=forced,
        )
        for name, (path, forced) in five_specs.items()
    }

    with open(
        base.FIVE_D3_PATHS["ideal_success"], encoding="utf-8"
    ) as handle:
        five_d3_initial_spread = float(json.load(handle)[0]["spread_before"])
    five_d3_specs = {
        "ideal_success": (base.FIVE_D3_PATHS["ideal_success"], None),
        "near_cross": (
            base.FIVE_D3_PATHS["ideal_success"], "self_crossing",
        ),
        "wrong_d2": (base.FIVE_D3_PATHS["wrong_d2"], "wrong_flat"),
    }
    five_d3_cases = {
        name: paired_action_path(
            path, target=3.0, initial_spread=five_d3_initial_spread,
            max_steps=400, force_final_reason=forced,
        )
        for name, (path, forced) in five_d3_specs.items()
    }

    watersqrt2_specs = {
        "ideal_success": (base.WATERSQRT2_PATHS["ideal_success"], None),
        "near_cross": (
            base.WATERSQRT2_PATHS["ideal_success"], "self_crossing",
        ),
        "pilot_timeout": (base.WATERSQRT2_PATHS["pilot_timeout"], None),
    }
    watersqrt2_cases = {
        name: paired_action_path(
            path, target=float(np.sqrt(2.0)),
            initial_spread=base.INITIAL_WATER_SPREAD, max_steps=400,
            force_final_reason=forced,
        )
        for name, (path, forced) in watersqrt2_specs.items()
    }

    torus_cases = {
        "success": paired_action_path(
            base.TORUS_SUCCESS, target=2.0,
            initial_spread=base.INITIAL_TORUS_SPREAD, max_steps=200,
        ),
        "h0_near_cross": paired_torus_cmaes(0),
        "h1_bad_cross": paired_torus_cmaes(1),
    }

    return {
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


def main() -> int:
    if runner.candidate_crossing_penalty(CANDIDATE_MODE) != CROSSING_PENALTY:
        raise RuntimeError("Runner P0X2B12 crossing penalty is not 2.0")
    if runner.candidate_frontier_bonus(PARENT_MODE) != BASE_BONUS:
        raise RuntimeError("Runner P0X2 frontier bonus is not 0.9")
    if runner.candidate_frontier_bonus(CANDIDATE_MODE) != CANDIDATE_BONUS:
        raise RuntimeError("Runner P0X2B12 frontier bonus is not 1.2")

    with open(runner.P0_CROSS2_REJECT_REPORT, encoding="utf-8") as handle:
        parent_reject = json.load(handle)
    if not (
        parent_reject.get("candidate_mode") == PARENT_MODE
        and parent_reject.get("decision") == "reject_candidate"
        and parent_reject.get("rescue_gate_pass") is False
    ):
        raise RuntimeError("P0X2 formal rescue rejection is not established")

    audits = build_orderings()
    all_pass = all(value["pass"] for value in audits.values())
    if not all_pass:
        raise RuntimeError("P0X2B12 violates a required saved-path ordering")
    result = {
        "schema_version": 1,
        "analysis_type": "offline_saved_metric_path_reward_replay",
        "candidate_mode": CANDIDATE_MODE,
        "candidate_mode_slug": runner.MODE_SLUGS[CANDIDATE_MODE],
        "parent_candidate_mode": PARENT_MODE,
        "core_frontier_mode": runner.core_frontier_mode(CANDIDATE_MODE),
        "single_changed_reward_parameter": {
            "name": "ori6_spread_frontier_bonus",
            "P0X2_value": BASE_BONUS,
            "candidate_value": CANDIDATE_BONUS,
        },
        "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
            PARENT_MODE
        ),
        "candidate_reward_sha256": runner.candidate_reward_sha256(
            CANDIDATE_MODE
        ),
        "parent_candidate_reject_authority": str(
            runner.P0_CROSS2_REJECT_REPORT.resolve()
        ),
        "parent_candidate_reject_authority_sha256": runner.file_sha256(
            runner.P0_CROSS2_REJECT_REPORT
        ),
        "all_required_orderings_pass": all_pass,
        "orderings": audits,
        "interpretation_limits": [
            "This verifies reward ordering on archived metric sequences; it does not estimate policy discovery probability.",
            "Only the existing frontier bonus CLI value changes; the ori6 core reward implementation is untouched.",
            "Crossing and correction paths include exact rollback of frontier credit.",
            "Counterfactual terminal labels and CMA-ES start-state limits are the same as in the P0X2 audit.",
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
