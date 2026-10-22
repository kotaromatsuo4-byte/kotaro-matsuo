"""C10-compatible checkpoint analysis for the zero-reject non-swap screen.

The optimizer selection, proposal-censored discovery statistics, and staged
stopping gates are the frozen C10 implementation.  This additive adapter
changes only the authority checks needed by ``ori_codex6_zero_reject``:
ordinary rejected crossings must contribute exactly zero reward, while a
rejection on the horizon keeps the ordinary timeout/last-safe-quality term.

The C10 analyzer is loaded under a private module name so importing this file
cannot mutate the historical C10 analysis module in a shared test process.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import ori_codex6_zero_reject as zero_reject


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BASE_ANALYZER = SCRIPT_DIRECTORY / "analyze_ori_codex10_nonswap_staged.py"
_BASE_MODULE_NAME = "_ori11_c10_nonswap_analysis_base"


def _load_frozen_base():
    existing = sys.modules.get(_BASE_MODULE_NAME)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        _BASE_MODULE_NAME, BASE_ANALYZER
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen analyzer: {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[_BASE_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


_base = _load_frozen_base()
_ORIGINAL_INTEGRITY_ERRORS = _base._integrity_errors
_ORIGINAL_ANALYSE_TASK = _base.analyse_task
_ORIGINAL_ANALYSE_SELECTED = _base.analyse_selected

SEEDS = _base.SEEDS
TASK_ORDER = _base.TASK_ORDER
SCREEN_CHECKPOINTS = _base.SCREEN_CHECKPOINTS
PRODUCTION_CHECKPOINTS = _base.PRODUCTION_CHECKPOINTS
PRODUCTION_READY_SEED_TAIL512 = _base.PRODUCTION_READY_SEED_TAIL512
PRODUCTION_READY_SEED_FIXED = _base.PRODUCTION_READY_SEED_FIXED
PRODUCTION_READY_MEDIAN_TAIL512 = _base.PRODUCTION_READY_MEDIAN_TAIL512
PRODUCTION_MEDIAN_TAIL1024 = _base.PRODUCTION_MEDIAN_TAIL1024
PRODUCTION_MIN_SEED_TAIL1024 = _base.PRODUCTION_MIN_SEED_TAIL1024
PRODUCTION_MEDIAN_FIXED = _base.PRODUCTION_MEDIAN_FIXED

CANDIDATE = zero_reject.CANDIDATE_NAME
PARENT_CANDIDATE = "ori6_RC_spectral_subspace_potential"
CROSSING_POLICY = zero_reject.CROSSING_POLICY
FORMULA_SHA256 = zero_reject.ZERO_REJECT_FORMULA_SHA256


def validate_config(path: Path, expected: dict) -> list[str]:
    """Validate parser binding plus the effective zero-reject reward design."""
    if not path.is_file():
        return ["missing run_config.json"]
    try:
        with open(path, encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [f"config read failed: {error}"]

    errors: list[str] = []
    for key, value in expected.items():
        if not _base._same(value, observed.get(key)):
            errors.append(f"config {key}: {observed.get(key)!r} != {value!r}")
    if "output_dir" in expected and not _base._same(
        expected["output_dir"], observed.get("output_directory")
    ):
        errors.append(
            "config output_directory: "
            f"{observed.get('output_directory')!r} != {expected['output_dir']!r}"
        )

    architecture = (
        "variable_mode_ori6_continuous_spread_state_potential_rollback_"
        "actor_critic"
    )
    for key in ("architecture_name", "architecture"):
        if observed.get(key) != architecture:
            errors.append(
                f"config {key}: {observed.get(key)!r} != {architecture!r}"
            )

    parameters = observed.get("ori6_reward_parameters")
    if not isinstance(parameters, dict):
        parameters = observed.get("ori5_reward_parameters")
    if not isinstance(parameters, dict):
        errors.append("missing nested effective reward parameters")
        parameters = {}
    transition = parameters.get("crossing_transition", {})
    spectral = parameters.get("spectral_subspace", {})
    frontier = parameters.get("spread_frontier", {})
    nested_expected = {
        "reward_design": (
            observed.get("reward_design"),
            "ori6_continuous_spread_state_potential_rollback",
        ),
        "candidate": (parameters.get("candidate"), CANDIDATE),
        "parent_candidate": (
            parameters.get("parent_candidate"), PARENT_CANDIDATE
        ),
        "candidate_schema_version": (
            parameters.get("candidate_schema_version"),
            zero_reject.CANDIDATE_SCHEMA_VERSION,
        ),
        "crossing_policy": (transition.get("policy"), CROSSING_POLICY),
        "candidate_committed": (transition.get("candidate_committed"), False),
        "ordinary_reject_formula": (
            transition.get("ordinary_rejected_reward_formula"), "0"
        ),
        "horizon_reject_formula": (
            transition.get("horizon_rejected_reward_formula"),
            "-timeout_penalty + quality_weight * q(last_safe)",
        ),
        "crossing_penalty_applied": (
            transition.get("rejected_crossing_penalty_applied"), False
        ),
        "crossing_quality_applied": (
            transition.get("rejected_crossing_quality_applied"), False
        ),
        "ordinary_total_exactly_zero": (
            transition.get("ordinary_rejected_total_reward_exactly_zero"), True
        ),
        "policy_trace_retained": (
            transition.get("original_policy_action_and_log_probability_retained"),
            True,
        ),
        "accepted_width": (transition.get("accepted_width"), 0.0),
        "actual_displacement": (transition.get("actual_displacement"), 0.0),
        "timeout_exactly_once": (
            transition.get("timeout_reward_applied_exactly_once"), True
        ),
        "past_ledgers_retained": (
            transition.get("past_frontier_and_joint_ledgers_retained"), True
        ),
        "spectral_reject_reward": (
            transition.get("spectral_transition_reward_on_reject"), 0.0
        ),
        "reject_step_count_increments": (
            transition.get("step_count_increments"), True
        ),
        "rejected_endpoint_reward": (
            transition.get("rejected_candidate_endpoint_reward"), 0.0
        ),
        "rejected_frontier_reward": (
            transition.get("rejected_candidate_spread_frontier_reward"), 0.0
        ),
        "rejected_milestone_reward": (
            transition.get("rejected_candidate_new_milestone_reward"), 0.0
        ),
        # The configured penalty is intentionally retained as a frozen
        # counterfactual parameter; the zero adapter proves it is not applied.
        "configured_crossing_penalty": (
            parameters.get("crossing_penalty"), 2.0
        ),
        "quality_weight": (parameters.get("quality_weight"), 1.0),
        "timeout_penalty": (parameters.get("timeout_penalty"), 1.0),
        "zero_formula_sha256": (
            parameters.get("zero_reject_formula_sha256"), FORMULA_SHA256
        ),
        "collision_quality_reference": (
            parameters.get("collision_quality_reference"),
            "last_safe_timeout_only",
        ),
        "spectral_weight": (spectral.get("weight"), 0.05),
        "spectral_relative_epsilon": (
            spectral.get("relative_epsilon"), 1e-12
        ),
        "spectral_accepted_valid_only": (
            spectral.get("accepted_valid_transitions_only"), True
        ),
        "spectral_formula_sha256": (
            spectral.get("formula_sha256"),
            "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6",
        ),
        "frontier_mode": (frontier.get("mode"), "state_potential"),
        "frontier_weight": (frontier.get("maximum_bonus"), 0.9),
        "frontier_flatness_gate_cap": (
            frontier.get("flatness_gate_cap"), 2.0
        ),
    }
    for label, (actual, required) in nested_expected.items():
        if not _base._same(required, actual):
            errors.append(f"nested {label}: {actual!r} != {required!r}")

    optional_top_level = {
        "ori6_candidate": CANDIDATE,
        "ori6_parent_candidate": PARENT_CANDIDATE,
        "ori6_crossing_policy": CROSSING_POLICY,
        "zero_reject_formula_sha256": FORMULA_SHA256,
        "base_rejected_and_spectral_authorities_unmodified": True,
    }
    for key, required in optional_top_level.items():
        if key in observed and not _base._same(required, observed[key]):
            errors.append(f"top-level {key}: {observed[key]!r} != {required!r}")
    return errors


def _integrity_errors(rows: Sequence[dict], max_steps: int) -> list[str]:
    """Retain every C10 check and add exact zero-reject row checks."""
    translated = [copy.copy(row) for row in rows]
    for row in translated:
        # Translate only identity labels so the frozen C10 numerical and
        # decomposition checks can be reused byte-for-byte.
        row["ori6_candidate"] = "ori6_P0X2_rejected_crossing_continuation"
        row["ori6_crossing_policy"] = "reject_candidate_and_continue"
    errors = list(_ORIGINAL_INTEGRITY_ERRORS(translated, max_steps))

    required = (
        "ori6_parent_candidate",
        "ori6_zero_reject_formula_sha256",
        "ori6_zero_reject_count",
        "ori6_zero_reject_total_reward",
        "ori6_zero_reject_timeout_reward",
        "ori6_maximum_abs_zero_reject_reward_decomposition_error",
    )
    if rows:
        for name in required:
            if name not in rows[0]:
                errors.append(f"missing zero-reject integrity field {name}")

    for row_index, row in enumerate(rows, start=1):
        episode = _base._int(row, ("episode",), row_index)
        identities = {
            "ori6_candidate": CANDIDATE,
            "ori6_candidate_schema_version": zero_reject.CANDIDATE_SCHEMA_VERSION,
            "ori6_parent_candidate": PARENT_CANDIDATE,
            "ori6_crossing_policy": CROSSING_POLICY,
            "ori6_zero_reject_formula_sha256": FORMULA_SHA256,
        }
        for name, required_value in identities.items():
            if row.get(name) != required_value and len(errors) < 50:
                errors.append(
                    f"episode {episode}: {name}={row.get(name)!r} "
                    f"!= {required_value!r}"
                )
        try:
            zero_count = _base._required_int(row, "ori6_zero_reject_count")
            reject_count = _base._required_int(
                row, "ori6_rejected_crossing_count"
            )
        except ValueError as error:
            if len(errors) < 50:
                errors.append(f"episode {episode}: {error}")
        else:
            if (
                zero_count < 0
                or zero_count > reject_count
                or reject_count - zero_count not in (0, 1)
            ) and len(errors) < 50:
                errors.append(
                    f"episode {episode}: ordinary/total reject counts "
                    f"{zero_count}/{reject_count} are inconsistent"
                )
        ordinary_reward = _base._float(
            row, ("ori6_zero_reject_total_reward",)
        )
        if (
            not math.isfinite(ordinary_reward)
            or ordinary_reward != 0.0
        ) and len(errors) < 50:
            errors.append(
                f"episode {episode}: ordinary zero-reject reward "
                f"{ordinary_reward!r} is not exactly zero"
            )
        for name in (
            "ori6_zero_reject_timeout_reward",
            "ori6_maximum_abs_zero_reject_reward_decomposition_error",
        ):
            value = _base._float(row, (name,))
            if not math.isfinite(value) and len(errors) < 50:
                errors.append(f"episode {episode}: nonfinite or missing {name}")
            if (
                name.endswith("decomposition_error")
                and math.isfinite(value)
                and abs(value) > 1e-8
                and len(errors) < 50
            ):
                errors.append(
                    f"episode {episode}: zero-reject decomposition "
                    f"{value:.3g} > 1e-8"
                )
    return errors


def _decorate(report: dict) -> dict:
    report["candidate"] = CANDIDATE
    report["reward_ablation"] = {
        "crossing_policy": CROSSING_POLICY,
        "ordinary_rejected_reward": "0",
        "horizon_rejected_reward": (
            "-timeout_penalty + quality_weight * q(last_safe)"
        ),
        "formula_sha256": FORMULA_SHA256,
    }
    return report


def analyse_task(jobs: Sequence[dict], task: str, endpoint: int) -> dict:
    return _decorate(_ORIGINAL_ANALYSE_TASK(jobs, task, endpoint))


def analyse_selected(
    jobs: Sequence[dict], task: str, arm: str, endpoint: int,
    prior_production_warning_streak: int = 0,
) -> dict:
    return _decorate(_ORIGINAL_ANALYSE_SELECTED(
        jobs, task, arm, endpoint,
        prior_production_warning_streak=prior_production_warning_streak,
    ))


# Install adaptations only into the privately loaded C10 module.  Its main()
# can therefore be used directly without contaminating the historical import.
_base.validate_config = validate_config
_base._integrity_errors = _integrity_errors
_base.analyse_task = analyse_task
_base.analyse_selected = analyse_selected

read_prefix = _base.read_prefix
read_updates = _base.read_updates
read_prefix_sparse = _base.read_prefix_sparse
seed_report = _base.seed_report
write_json = _base.write_json
main = _base.main


if __name__ == "__main__":
    main()
