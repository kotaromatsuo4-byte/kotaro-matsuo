"""Fail-closed staged analysis for the Codex8 spectral-reward experiment.

The screen is deliberately based on first strict discovery and geometric
quality, never on return.  Rows beyond ``endpoint`` are ignored so a runner
can safely analyse an atomic checkpoint after a process has written ahead.
The legacy and scale controls use the native Codex6 spectral implementation;
only ``phase_v2`` has the Codex8 phase-ledger columns.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Sequence

import ori_codex6_spectral_subspace as ori6_spectral
import ori_codex6_zero_reject as zero_reject
import ori_codex8 as core


TASKS = ("torus2", "water1")
ARMS = ("legacy", "scale", "phase_v2")
NEW_ARMS = ("scale", "phase_v2")
SEEDS = (8101, 8102, 8103)
SCREEN_CHECKPOINTS = (512, 1024, 2048, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
CHECKPOINTS = SCREEN_CHECKPOINTS + PRODUCTION_CHECKPOINTS
FORMAL_ENDPOINT = 4096
SELECTION_RULE_VERSION = "codex8_same_seed_lexicographic_v1"

# Controls retain the exact Codex6 potential, including its native rollback.
# phase_v2 disables the parent term and uses the replacement phase potential.
PARENT_SPECTRAL_WEIGHT = {
    "torus2": {"legacy": 0.05, "scale": 0.50, "phase_v2": 0.0},
    "water1": {"legacy": 0.05, "scale": 0.25, "phase_v2": 0.0},
}
PHASE_LAMBDA = {"torus2": 1.0, "water1": 0.5}
ARM_LAMBDA = {
    "torus2": {"legacy": 0.05, "scale": 0.05, "phase_v2": 1.0},
    "water1": {"legacy": 0.05, "scale": 0.025, "phase_v2": 0.5},
}
TASK_RUNTIME = {
    "torus2": {
        "data_dir": str((Path(__file__).resolve().parent / "model/4unit_torus").resolve()),
        "selected_model": "4unit_torus",
        "target_spread": 2.0, "max_steps": 200, "max_step_width": 0.2,
        "actor_learning_rate": 1.0e-3, "critic_learning_rate": 3.0e-4,
        "actor_lr_milestones": [1528, 4000, 7000],
        "near_miss_spread_error_far": 1.0,
    },
    "water1": {
        "data_dir": str((Path(__file__).resolve().parent / "model/waterbomb").resolve()),
        "selected_model": "waterbomb",
        "target_spread": 1.0, "max_steps": 400, "max_step_width": 0.3,
        "actor_learning_rate": 5.0e-4, "critic_learning_rate": 1.5e-4,
        "actor_lr_milestones": [2000, 4000, 7000],
        "near_miss_spread_error_far": 0.12,
    },
}


@dataclass(frozen=True)
class QualityGate:
    constructive_flatness: float = 0.5
    joint_flatness: float = 0.1
    spread_error: float = 0.1
    high_phase_psi: float = 0.35
    farming_fraction: float = 0.20


GATE = QualityGate()


def run_directory(root: Path, task: str, arm: str, seed: int) -> Path:
    """The runner and analyser share this exact, profile-free layout."""

    return Path(root) / task / arm / f"seed{int(seed)}"


def _float(row: dict, *names: str, default: float = math.nan) -> float:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return default


def _int(row: dict, *names: str, default: int = 0) -> int:
    value = _float(row, *names, default=float(default))
    return int(value) if math.isfinite(value) else default


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else math.nan


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else math.nan


def _rate(count: int | float, total: int | float) -> float:
    return float(count / total) if total else 0.0


def _finite_or(value, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def read_history(path: Path, endpoint: int) -> tuple[list[dict], dict, list[str]]:
    """Read exactly 1..endpoint and audit, but do not consume overrun rows."""

    audit = {
        "path": str(path), "requested_endpoint": int(endpoint),
        "file_row_count": 0, "file_max_episode": 0, "overrun_row_count": 0,
        "overrun_capped": False,
    }
    if not path.is_file():
        return [], audit, [f"missing history: {path}"]
    rows: dict[int, dict] = {}
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        try:
            source = list(csv.DictReader(handle))
        except csv.Error as error:
            return [], audit, [f"history CSV error: {error}"]
    audit["file_row_count"] = len(source)
    for row_number, row in enumerate(source, 2):
        episode = _int(row, "episode", default=-1)
        if episode <= 0:
            errors.append(f"invalid episode at CSV row {row_number}")
            continue
        audit["file_max_episode"] = max(audit["file_max_episode"], episode)
        if episode > endpoint:
            audit["overrun_row_count"] += 1
            continue
        if episode in rows:
            errors.append(f"duplicate episode {episode}")
        rows[episode] = row
    audit["overrun_capped"] = bool(audit["overrun_row_count"])
    expected = list(range(1, endpoint + 1))
    observed = sorted(rows)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        errors.append(
            f"history is not contiguous 1..{endpoint}; missing={missing[:8]}"
        )
    return [rows[key] for key in observed], audit, errors


def read_updates(path: Path, endpoint: int, batch_size: int = 8) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], [f"missing update history: {path}"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen: dict[int, dict] = {}
    errors: list[str] = []
    for row in rows:
        episode = _int(row, "episode", default=-1)
        if 0 < episode <= endpoint:
            if episode in chosen:
                errors.append(f"duplicate optimizer update episode {episode}")
            chosen[episode] = row
    expected = set(range(batch_size, endpoint + 1, batch_size))
    missing = sorted(expected - set(chosen))
    if missing:
        errors.append(f"missing optimizer updates {missing[:8]}")
    for episode, row in chosen.items():
        for field in (
            "critic_loss", "critic_gradient_norm", "actor_gradient_norm",
            "actor_learning_rate", "critic_learning_rate",
        ):
            if not math.isfinite(_float(row, field)):
                errors.append(f"nonfinite optimizer {field} at episode {episode}")
    return [chosen[key] for key in sorted(chosen)], errors


def expected_parent_weight(task: str, arm: str) -> float:
    return float(PARENT_SPECTRAL_WEIGHT[task][arm])


def _nested_values(payload, key: str) -> list:
    values: list = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key:
                values.append(value)
            values.extend(_nested_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_nested_values(value, key))
    return values


def expected_runtime_config(task: str, arm: str, seed: int) -> dict:
    expected = {
        "seed": seed, "episodes": 10000, "batch_size": 8,
        "checkpoint_every": 512, "eval_interval": 512,
        "periodic_stochastic_eval_episodes": 16, "eval_episodes": 64,
        "print_every": 128, "selection_evaluation_seed": 9320000,
        "final_evaluation_seed": 9420000,
        "flatness_subspace_dimension": 2, "entropy_coefficient": 0.001,
        "actor_lr_factors": [0.3, 0.1, 0.03], "update_method": "actor_critic",
        "random_search": False, "spread_tolerance": 0.05,
        "flatness_tolerance": 0.001, "strain_tolerance": 1.0e-8,
        "crossing_tolerance": 1.0e-10, "min_step_width": 1.0e-4,
        "projection_shrink": 0.5, "max_correction_ratio": 1.0,
        "max_actual_step_ratio": 1.25, "terminate_wrong_flat": True,
        "numerical_failure_penalty": 10.0, "no_mode_penalty": 5.0,
        "gamma": 1.0, "near_miss_flatness_far": 1.0,
        "ori5_flatness_clip": 2.0,
        "ori5_spread_milestone_initial_factor": 2.0,
        "ori5_spread_milestone_bonus": 0.0,
        "ori5_joint_flatness_thresholds": [0.5, 0.1, 0.01],
        "ori5_joint_milestone_bonuses": [0.2, 0.3, 0.5],
        "ori5_crossing_penalty": 2.0, "ori5_wrong_flat_penalty": 2.0,
        "ori5_timeout_penalty": 1.0, "ori5_success_bonus": 10.0,
        "ori5_quality_weight": 1.0, "ori5_success_speed_bonus_weight": 0.5,
        "ori6_spread_frontier_mode": "state_potential",
        "ori6_spread_frontier_bonus": 0.9,
        "ori6_spread_frontier_flatness_cap": 2.0,
        "ori6_spectral_subspace": True, "ori6_spectral_relative_epsilon": 1.0e-12,
        "ori6_zero_reject_reward": True,
        "rigid_motion_diagnostics": False,
        "ori8_spectral_ablation": True, "ori8_spectral_arm": arm,
        "ori8_lambda": ARM_LAMBDA[task][arm], "ori8_scale_multiplier": 10.0,
        "ori8_close_rhobar_threshold": 0.1, "ori8_turn_delta_threshold": 0.7,
        "ori8_effective_parent_spectral_weight": expected_parent_weight(task, arm),
        "ori8_observation_contract": core.OBSERVATION_CONTRACT,
        "ori8_pomdp_caveat": core.POMDP_CAVEAT,
    }
    expected.update(TASK_RUNTIME[task])
    return expected


def _same_config_value(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_config_value(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, str) and ("data_dir" in expected or "\\" in expected or "/" in expected):
        try:
            return Path(str(actual)).resolve() == Path(expected).resolve()
        except (TypeError, OSError):
            return False
    return actual == expected


def validate_config(path: Path, task: str, arm: str, seed: int) -> dict:
    if not path.is_file():
        return {"pass": False, "errors": [f"missing config: {path}"], "checks": {}}
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        return {"pass": False, "errors": [f"config read failed: {error}"], "checks": {}}

    def same_number(actual, expected, tolerance=1e-12):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False

    checks = {
        "seed": _int(config, "seed", default=-1) == seed,
        "episodes_cap": _int(config, "episodes", default=-1) == 10000,
        "checkpoint_every": _int(config, "checkpoint_every", default=-1) == 512,
        "batch_size": _int(config, "batch_size", default=-1) == 8,
        "actor_critic": config.get("update_method") == "actor_critic",
        "candidate": config.get("ori8_candidate") == core.CANDIDATE_NAME,
        "schema": config.get("ori8_candidate_schema_version") == core.CANDIDATE_SCHEMA_VERSION,
        "reward_formula": config.get("ori8_reward_formula_sha256") == core.REWARD_FORMULA_SHA256,
        "transition_formula": config.get("ori8_transition_formula_sha256") == core.TRANSITION_FORMULA_SHA256,
        "arm": config.get("ori8_spectral_arm", config.get("ori8_reward_parameters", {}).get("ori8_spectral_arm")) == arm,
        "parent_weight": same_number(
            config.get(
                "ori8_effective_parent_spectral_weight",
                config.get("ori6_spectral_weight", config.get("ori8_parent_spectral_weight")),
            ),
            expected_parent_weight(task, arm),
        ),
    }
    for field, expected in expected_runtime_config(task, arm, seed).items():
        checks[f"runtime_{field}"] = _same_config_value(config.get(field), expected)
    observation_values = (
        _nested_values(config, "observation_contract")
        + _nested_values(config, "ori8_observation_contract")
    )
    checks["observation_unchanged"] = any(
        "unchanged" in str(value).lower() or "coordinates-only" in str(value).lower()
        for value in observation_values
    )
    if arm == "phase_v2":
        checks["pomdp_caveat"] = (
            config.get("ori8_pomdp_caveat") == core.POMDP_CAVEAT
            or core.POMDP_CAVEAT in _nested_values(config, "pomdp_caveat")
        )
    errors = [name for name, passed in checks.items() if not passed]
    return {"pass": not errors, "errors": errors, "checks": checks, "config": config}


def _require_number(row: dict, field: str, errors: list[str]) -> float:
    if row.get(field) in (None, ""):
        errors.append(f"missing_{field}")
        return math.nan
    value = _float(row, field)
    if not math.isfinite(value):
        errors.append(f"nonfinite_{field}")
    return value


def validate_history_row(row: dict, task: str, arm: str) -> list[str]:
    errors: list[str] = []
    # Every field used by the selection key is explicit and bounded.  Silent
    # defaults here could otherwise create a false same-seed win.
    steps_value = _require_number(row, "steps", errors)
    max_steps = int(TASK_RUNTIME[task]["max_steps"])
    if math.isfinite(steps_value) and (
        steps_value != int(steps_value) or not 1 <= int(steps_value) <= max_steps
    ):
        errors.append("selection_steps_bounds")
    for field in (
        "strict_compact_success", "self_crossing", "correction_failed",
        "no_modes", "timeout",
    ):
        value = _require_number(row, field, errors)
        if math.isfinite(value) and value not in (0.0, 1.0):
            errors.append(f"selection_binary_{field}")
    if _int(row, "timeout") and math.isfinite(steps_value) and steps_value != max_steps:
        errors.append("selection_timeout_must_reach_horizon")
    flatness = _require_number(row, "flatness_ratio", errors)
    spread_error = _require_number(row, "spread_error", errors)
    if math.isfinite(flatness) and flatness < 0.0:
        errors.append("selection_negative_flatness")
    if math.isfinite(spread_error) and spread_error < 0.0:
        errors.append("selection_negative_spread_error")
    rejects = _require_number(row, "ori6_rejected_crossing_count", errors)
    reject_rate = _require_number(row, "ori6_rejected_crossing_rate", errors)
    if all(math.isfinite(value) for value in (rejects, steps_value)) and (
        rejects != int(rejects) or rejects < 0 or rejects > steps_value
    ):
        errors.append("selection_reject_count_bounds")
    if math.isfinite(reject_rate) and not 0.0 <= reject_rate <= 1.0:
        errors.append("selection_reject_rate_bounds")
    if all(math.isfinite(value) for value in (rejects, steps_value, reject_rate)) and steps_value > 0:
        if not math.isclose(reject_rate, rejects / steps_value, rel_tol=0.0, abs_tol=1e-12):
            errors.append("selection_reject_rate_consistency")
    invalid = any(_int(row, field) for field in (
        "self_crossing", "correction_failed", "no_modes"
    ))
    if _int(row, "strict_compact_success") and invalid:
        errors.append("selection_strict_invalid_inconsistent")
    expected_weight = expected_parent_weight(task, arm)
    weight = _require_number(row, "ori6_spectral_weight", errors)
    if math.isfinite(weight) and not math.isclose(weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
        errors.append("parent_spectral_weight")
    if row.get("ori6_spectral_formula_sha256") != ori6_spectral.SPECTRAL_FORMULA_SHA256:
        errors.append("ori6_spectral_formula_sha256")

    for field in (
        "ori5_reward_decomposition_error",
        "ori5_maximum_abs_step_reward_decomposition_error",
        "ori6_reward_decomposition_error",
        "ori6_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
        "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
        "ori6_maximum_abs_zero_reject_reward_decomposition_error",
    ):
        value = _require_number(row, field, errors)
        if math.isfinite(value) and abs(value) > 1e-8:
            errors.append(f"decomposition_{field}")
    for field in (
        "ori5_nonfinite_reward_component_count",
        "ori6_nonfinite_reward_component_count",
        "ori6_spectral_nonfinite_component_count",
    ):
        value = _require_number(row, field, errors)
        if math.isfinite(value) and value != 0.0:
            errors.append(f"nonzero_{field}")
    if row.get("ori6_zero_reject_formula_sha256") != zero_reject.ZERO_REJECT_FORMULA_SHA256:
        errors.append("ori6_zero_reject_formula_sha256")
    zero_reject_count = _require_number(row, "ori6_zero_reject_count", errors)
    zero_reject_reward = _require_number(row, "ori6_zero_reject_total_reward", errors)
    zero_reject_timeout = _require_number(row, "ori6_zero_reject_timeout_reward", errors)
    rejected_total = _require_number(row, "ori6_total_rejected_crossing_reward", errors)
    # zero_reject_total_reward deliberately sums only ordinary, non-horizon
    # rejected proposals (zero each).  A rejected proposal at H is instead the
    # one ordinary timeout reward and is reported separately in timeout_reward.
    if math.isfinite(zero_reject_reward) and abs(zero_reject_reward) > 1e-12:
        errors.append("zero_reject_ordinary_total_not_zero")
    if all(math.isfinite(value) for value in (zero_reject_count, rejects)):
        if (
            zero_reject_count != int(zero_reject_count)
            or not 0 <= zero_reject_count <= rejects
            or rejects - zero_reject_count not in (0, 1)
        ):
            errors.append("zero_reject_count_vs_rejected_count")
        elif rejects - zero_reject_count == 1:
            if not _int(row, "timeout") or str(row.get("termination_reason")) != "timeout":
                errors.append("zero_reject_horizon_termination")
            if math.isfinite(steps_value) and steps_value != max_steps:
                errors.append("zero_reject_horizon_steps")
            if row.get("ori6_terminal_quality_source") != "last_safe_timeout_after_rejected_crossing":
                errors.append("zero_reject_horizon_quality_source")
            if all(math.isfinite(value) for value in (rejected_total, zero_reject_timeout)) and not math.isclose(
                rejected_total, zero_reject_timeout, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append("zero_reject_horizon_timeout_decomposition")
        else:
            if row.get("ori6_terminal_quality_source") == "last_safe_timeout_after_rejected_crossing":
                errors.append("zero_reject_horizon_quality_source_without_horizon_reject")
            if math.isfinite(zero_reject_timeout) and abs(zero_reject_timeout) > 1e-12:
                errors.append("zero_reject_timeout_without_horizon_reject")
            if math.isfinite(rejected_total) and abs(rejected_total) > 1e-12:
                errors.append("zero_reject_rejected_total_without_horizon_reject")

    if arm == "phase_v2":
        exact = {
            "ori8_candidate": core.CANDIDATE_NAME,
            "ori8_candidate_schema_version": core.CANDIDATE_SCHEMA_VERSION,
            "ori8_reward_formula_sha256": core.REWARD_FORMULA_SHA256,
            "ori8_transition_formula_sha256": core.TRANSITION_FORMULA_SHA256,
            "ori8_spectral_arm": "phase_v2",
        }
        for field, expected in exact.items():
            if row.get(field) != expected:
                errors.append(field)
        phase_lambda = _require_number(row, "ori8_lambda", errors)
        if math.isfinite(phase_lambda) and not math.isclose(
            phase_lambda, PHASE_LAMBDA[task], rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("ori8_lambda")
        for field in (
            "ori8_reward_decomposition_error",
            "ori8_maximum_abs_step_reward_decomposition_error",
        ):
            value = _require_number(row, field, errors)
            if math.isfinite(value) and abs(value) > 1e-8:
                errors.append(f"decomposition_{field}")
        nonfinite = _require_number(row, "ori8_nonfinite_component_count", errors)
        if math.isfinite(nonfinite) and nonfinite != 0.0:
            errors.append("nonzero_ori8_nonfinite_component_count")
        total = _require_number(row, "ori8_total_reward", errors)
        total_progress = _require_number(row, "ori8_total_progress_reward", errors)
        total_rollback = _require_number(row, "ori8_total_rollback_reward", errors)
        cumulative = _require_number(row, "ori8_cumulative_reward", errors)
        initial_psi = _require_number(row, "ori8_initial_psi", errors)
        terminal_psi = _require_number(row, "ori8_terminal_psi", errors)
        rollback_count = _require_number(row, "ori8_rollback_count", errors)
        if math.isfinite(total) and math.isfinite(cumulative) and not math.isclose(
            total, cumulative, rel_tol=0.0, abs_tol=1e-8
        ):
            errors.append("phase_telescoping_total_vs_cumulative")
        if all(math.isfinite(value) for value in (total, total_progress, total_rollback)) and not math.isclose(
            total, total_progress + total_rollback, rel_tol=0.0, abs_tol=1e-8
        ):
            errors.append("phase_progress_rollback_decomposition")
        if math.isfinite(rollback_count) and rollback_count == 0:
            if math.isfinite(total_rollback) and abs(total_rollback) > 1e-8:
                errors.append("phase_nonrollback_has_rollback_reward")
            if all(math.isfinite(value) for value in (
                total, terminal_psi, initial_psi, phase_lambda
            )) and not math.isclose(
                total, phase_lambda * (terminal_psi - initial_psi),
                rel_tol=0.0, abs_tol=1e-8,
            ):
                errors.append("phase_telescoping_terminal_potential")
        elif math.isfinite(rollback_count) and rollback_count > 0:
            if str(row.get("termination_reason", "")) not in ori6_spectral.SPECTRAL_ROLLBACK_REASONS:
                errors.append("phase_rollback_reason")
            if math.isfinite(total) and abs(total) > 1e-8:
                errors.append("phase_rollback_total_not_zero")
            if math.isfinite(cumulative) and abs(cumulative) > 1e-8:
                errors.append("phase_rollback_cumulative_not_zero")
            if all(math.isfinite(value) for value in (total_progress, total_rollback)) and not math.isclose(
                total_rollback, -total_progress, rel_tol=0.0, abs_tol=1e-8
            ):
                errors.append("phase_rollback_not_negative_prefix")
    else:
        # Controls must not accidentally stack the replacement phase reward.
        if row.get("ori8_total_reward") not in (None, "", "0", "0.0"):
            if abs(_float(row, "ori8_total_reward")) > 1e-12:
                errors.append("stacked_ori8_control_reward")
    return errors


def classify(row: dict) -> dict:
    flatness = _float(row, "flatness_ratio", "ori5_last_valid_flatness_ratio")
    spread_error = _float(row, "spread_error", "ori5_last_valid_spread_error")
    invalid = any(_int(row, name) != 0 for name in (
        "self_crossing", "correction_failed", "no_modes"
    ))
    strict = bool(_int(row, "strict_compact_success") and not invalid)
    constructive = bool(
        not invalid and math.isfinite(flatness) and math.isfinite(spread_error)
        and flatness <= GATE.constructive_flatness
        and spread_error <= GATE.spread_error
    )
    joint = bool(
        not invalid and math.isfinite(flatness) and math.isfinite(spread_error)
        and flatness <= GATE.joint_flatness
        and spread_error <= GATE.spread_error
    )
    psi = _float(row, "ori8_terminal_psi", default=0.0)
    phase_reward = _float(row, "ori8_total_reward", default=0.0)
    timeout = bool(_int(row, "timeout"))
    z = bool(_int(row, "ori8_z_latched"))
    high_psi_bad = bool(psi >= GATE.high_phase_psi and not constructive)
    return {
        "episode": _int(row, "episode"),
        "steps": _int(row, "steps"),
        "reward": _float(row, "total_reward"),
        "flatness": flatness,
        "spread_error": spread_error,
        "strict": strict,
        "joint": joint,
        "constructive": constructive,
        "invalid": invalid,
        "timeout": timeout,
        "reject_count": _int(row, "ori6_rejected_crossing_count"),
        "reject_rate": _float(row, "ori6_rejected_crossing_rate", default=0.0),
        "phase_reward": phase_reward,
        "phase_psi": psi,
        "phase_close": bool(_int(row, "ori8_close_latched")),
        "phase_z": z,
        "phase_close_events": _int(row, "ori8_close_event_count"),
        "phase_turn_events": _int(row, "ori8_turn_event_count"),
        "phase_noncommits": _int(row, "ori8_noncommit_count"),
        "high_psi_bad": high_psi_bad,
        "high_psi_bad_timeout": bool(high_psi_bad and timeout),
        "z_bad_timeout": bool(z and timeout and not constructive),
        "positive_phase_bad": bool(phase_reward > 0.0 and not constructive),
    }


def first_hit(records: Sequence[dict]) -> dict:
    proposals = 0
    for record in records:
        proposals += int(record["steps"])
        if record["strict"]:
            return {
                "hit": True, "episode": record["episode"],
                "action_proposals": proposals,
                "policy_updates_before_hit": (record["episode"] - 1) // 8,
            }
    return {
        "hit": False, "episode": None, "action_proposals": None,
        "policy_updates_before_hit": None,
        "censor_action_proposals": proposals,
    }


def window(records: Sequence[dict], start: int, end: int) -> dict:
    chosen = [row for row in records if start <= row["episode"] <= end]
    n = len(chosen)
    timeouts = sum(row["timeout"] for row in chosen)
    high_bad_timeouts = sum(row["high_psi_bad_timeout"] for row in chosen)
    return {
        "start_episode": start, "end_episode": end, "episode_count": n,
        "strict_count": sum(row["strict"] for row in chosen),
        "strict_rate": _rate(sum(row["strict"] for row in chosen), n),
        "joint_count": sum(row["joint"] for row in chosen),
        "joint_rate": _rate(sum(row["joint"] for row in chosen), n),
        "constructive_count": sum(row["constructive"] for row in chosen),
        "constructive_rate": _rate(sum(row["constructive"] for row in chosen), n),
        "invalid_count": sum(row["invalid"] for row in chosen),
        "timeout_count": timeouts,
        "flatness_median": _median(row["flatness"] for row in chosen),
        "spread_error_median": _median(row["spread_error"] for row in chosen),
        "return_mean_diagnostic_only": _mean(row["reward"] for row in chosen),
        "action_proposals": sum(row["steps"] for row in chosen),
        "reject_count": sum(row["reject_count"] for row in chosen),
        "reject_rate_mean": _mean(row["reject_rate"] for row in chosen),
        "phase_reward_mean": _mean(row["phase_reward"] for row in chosen),
        "phase_psi_mean": _mean(row["phase_psi"] for row in chosen),
        "phase_close_rate": _rate(sum(row["phase_close"] for row in chosen), n),
        "phase_z_rate": _rate(sum(row["phase_z"] for row in chosen), n),
        "phase_close_event_count": sum(row["phase_close_events"] for row in chosen),
        "phase_turn_event_count": sum(row["phase_turn_events"] for row in chosen),
        "phase_noncommit_count": sum(row["phase_noncommits"] for row in chosen),
        "phase_high_psi_bad_count": sum(row["high_psi_bad"] for row in chosen),
        "phase_high_psi_bad_timeout_count": high_bad_timeouts,
        "phase_high_psi_bad_timeout_rate": _rate(high_bad_timeouts, n),
        "phase_high_psi_bad_fraction_of_timeouts": _rate(high_bad_timeouts, timeouts),
        "phase_z_bad_timeout_count": sum(row["z_bad_timeout"] for row in chosen),
        "phase_positive_reward_bad_count": sum(row["positive_phase_bad"] for row in chosen),
    }


def optimizer_summary(rows: Sequence[dict], start: int, end: int) -> dict:
    chosen = [row for row in rows if start <= _int(row, "episode") <= end]
    return {
        "update_count": len(chosen),
        "critic_loss_mean": _mean(_float(row, "critic_loss") for row in chosen),
        "critic_loss_max": max([_float(row, "critic_loss") for row in chosen] or [math.nan]),
        "critic_gradient_norm_mean": _mean(_float(row, "critic_gradient_norm") for row in chosen),
        "actor_gradient_norm_mean": _mean(_float(row, "actor_gradient_norm") for row in chosen),
        "entropy_mean": _mean(_float(row, "entropy") for row in chosen),
        "actor_learning_rate_first": _float(chosen[0], "actor_learning_rate") if chosen else math.nan,
        "actor_learning_rate_last": _float(chosen[-1], "actor_learning_rate") if chosen else math.nan,
    }


def seed_report(root: Path, task: str, arm: str, seed: int, endpoint: int) -> dict:
    directory = run_directory(root, task, arm, seed)
    rows, history_audit, errors = read_history(directory / "training_history.csv", endpoint)
    config = validate_config(directory / "run_config.json", task, arm, seed)
    if not config["pass"]:
        errors.extend(f"config:{error}" for error in config["errors"])
    batch_size = _int(config.get("config", {}), "batch_size", default=8)
    updates, update_errors = read_updates(directory / "update_history.csv", endpoint, batch_size)
    errors.extend(update_errors)
    for row in rows:
        episode = _int(row, "episode", default=-1)
        errors.extend(
            f"episode_{episode}:{error}"
            for error in validate_history_row(row, task, arm)
        )
    records = [classify(row) for row in rows]
    recent = window(records, max(1, endpoint - 1023), endpoint)
    late = window(records, max(1, endpoint - 511), endpoint)
    previous = window(records, max(1, endpoint - 1023), max(1, endpoint - 512))
    farming = bool(
        arm == "phase_v2"
        and recent["phase_high_psi_bad_timeout_rate"] >= GATE.farming_fraction
        and recent["constructive_rate"] < 0.01
    )
    return {
        "task": task, "arm": arm, "seed": seed, "endpoint": endpoint,
        "directory": str(directory.resolve()),
        "integrity_pass": not errors, "integrity_errors": errors,
        "history_audit": history_audit, "config_validation": config,
        "first_hit": first_hit(records),
        "full": window(records, 1, endpoint), "last1024": recent,
        "last512": late, "previous512": previous,
        "optimizer_last1024": optimizer_summary(updates, max(1, endpoint - 1023), endpoint),
        "phase_reward_hacking": {
            "farming_detected": farming,
            "rule": "high-Psi poor-quality timeout rate >=20% and constructive rate <1%",
            "observation_caveat": (
                "phase ledger z is not in the unchanged Codex6 observation; "
                "phase_v2 is intentionally a reward-only partially observable arm"
            ) if arm == "phase_v2" else None,
        },
        "records": records,
        "optimizer_records": updates,
    }


def selection_key(report: dict) -> tuple:
    """Pre-registered lexicographic key.  Return is intentionally absent."""

    hit = report["first_hit"]
    recent = report["last1024"]
    return (
        int(bool(hit["hit"])),
        -int(hit["action_proposals"]) if hit["hit"] else 0,
        int(recent["strict_count"]),
        int(recent["joint_count"]),
        int(recent["constructive_count"]),
        -_finite_or(recent["spread_error_median"], math.inf),
        -_finite_or(recent["flatness_median"], math.inf),
        -_finite_or(recent["reject_rate_mean"], math.inf),
    )


def compare_reports(candidate: dict, opponent: dict) -> dict:
    if candidate["seed"] != opponent["seed"]:
        raise ValueError("same-seed comparison required")
    left, right = selection_key(candidate), selection_key(opponent)
    winner = candidate["arm"] if left > right else opponent["arm"] if right > left else "tie"
    return {
        "seed": candidate["seed"], "winner": winner,
        "candidate": candidate["arm"], "opponent": opponent["arm"],
        "candidate_key": list(left), "opponent_key": list(right),
        "candidate_first_hit": candidate["first_hit"],
        "opponent_first_hit": opponent["first_hit"],
    }


def paired_comparison(candidate: Sequence[dict], opponent: Sequence[dict]) -> dict:
    candidate_by_seed = {row["seed"]: row for row in candidate}
    opponent_by_seed = {row["seed"]: row for row in opponent}
    per_seed = [compare_reports(candidate_by_seed[seed], opponent_by_seed[seed]) for seed in SEEDS]
    candidate_arm = candidate[0]["arm"]
    opponent_arm = opponent[0]["arm"]
    return {
        "wins": sum(row["winner"] == candidate_arm for row in per_seed),
        "losses": sum(row["winner"] == opponent_arm for row in per_seed),
        "ties": sum(row["winner"] == "tie" for row in per_seed),
        "per_seed": per_seed,
        "rule": SELECTION_RULE_VERSION,
    }


def catastrophic(reports: Sequence[dict], endpoint: int) -> tuple[bool, str | None, list[str]]:
    if any(not report["integrity_pass"] for report in reports):
        return True, "integrity", ["one or more seed histories/configs failed authority checks"]
    if endpoint < 1024:
        return False, None, []
    no_signal = all(
        report["last512"]["strict_count"] == 0
        and report["last512"]["constructive_count"] == 0
        for report in reports
    )
    invalid_lock = all(
        report["last512"]["invalid_count"] >= 0.95 * report["last512"]["episode_count"]
        for report in reports
    )
    reject_lock = all(
        report["last512"]["reject_rate_mean"] >= 0.80 for report in reports
    )
    farming = all(report["phase_reward_hacking"]["farming_detected"] for report in reports)
    reasons = []
    if no_signal and invalid_lock:
        reasons.append("all-seed invalid-transition lock with no geometric signal")
    if no_signal and reject_lock:
        reasons.append("all-seed reject lock with no geometric signal")
    if farming:
        reasons.append("all-seed phase-reward farming")
    return bool(reasons), "catastrophic" if reasons else None, reasons


def analyze(root: Path, endpoint: int, tasks=TASKS, arms=ARMS) -> dict:
    if endpoint not in CHECKPOINTS:
        raise ValueError(f"endpoint must be one of {CHECKPOINTS}")
    root = Path(root)
    reports: dict = {}
    decisions: dict = {}
    task_decisions: dict = {}
    for task in tasks:
        reports[task] = {}
        decisions[task] = {}
        for arm in arms:
            cohort = [seed_report(root, task, arm, seed, endpoint) for seed in SEEDS]
            reports[task][arm] = cohort
            stop, stop_kind, reasons = catastrophic(cohort, endpoint)
            decisions[task][arm] = {
                "task": task, "arm": arm, "endpoint": endpoint,
                "stop": stop, "stop_kind": stop_kind, "reasons": reasons,
            }

        pairwise = {arm: {} for arm in arms}
        for candidate in arms:
            for opponent in arms:
                if candidate == opponent:
                    continue
                pairwise[candidate][opponent] = paired_comparison(
                    reports[task][candidate], reports[task][opponent]
                )
        selected = None
        eligible = []
        strong_early = False
        early_evidence_seed_count = 0
        all_task_integrity = all(
            row["integrity_pass"]
            for arm in arms for row in reports[task][arm]
        )
        if endpoint >= 2048:
            for candidate in NEW_ARMS:
                if candidate not in reports[task] or set(arms) != set(ARMS):
                    continue
                integrity = all_task_integrity and not decisions[task][candidate]["stop"]
                no_farming = not any(
                    row["phase_reward_hacking"]["farming_detected"]
                    for row in reports[task][candidate]
                )
                beats_every_arm = all(
                    pairwise[candidate][opponent]["wins"] >= 2
                    for opponent in arms if opponent != candidate
                )
                strict_evidence = sum(
                    row["first_hit"]["hit"] for row in reports[task][candidate]
                )
                persistent_evidence = sum(
                    row["last512"]["joint_count"] >= 2
                    and row["last512"]["constructive_count"] >= 6
                    and row["previous512"]["joint_count"] >= 1
                    and row["previous512"]["constructive_count"] >= 3
                    for row in reports[task][candidate]
                )
                evidence_count = max(strict_evidence, persistent_evidence)
                early_evidence_ok = endpoint >= FORMAL_ENDPOINT or evidence_count >= 2
                if integrity and no_farming and beats_every_arm and early_evidence_ok:
                    eligible.append(candidate)
            if len(eligible) == 1:
                selected = eligible[0]
                if endpoint < FORMAL_ENDPOINT:
                    strong_early = True
                    strict_evidence = sum(
                        row["first_hit"]["hit"] for row in reports[task][selected]
                    )
                    persistent_evidence = sum(
                        row["last512"]["joint_count"] >= 2
                        and row["last512"]["constructive_count"] >= 6
                        and row["previous512"]["joint_count"] >= 1
                        and row["previous512"]["constructive_count"] >= 3
                        for row in reports[task][selected]
                    )
                    early_evidence_seed_count = max(strict_evidence, persistent_evidence)
        min_wins = (
            min(pairwise[selected][opponent]["wins"] for opponent in arms if opponent != selected)
            if selected else 0
        )
        selected_integrity = bool(
            selected and all(row["integrity_pass"] for row in reports[task][selected])
        )
        selected_farming_gate = bool(
            selected and not any(
                row["phase_reward_hacking"]["farming_detected"]
                for row in reports[task][selected]
            )
        )
        descriptive_winners = [
            candidate for candidate in arms
            if len(arms) == len(ARMS) and all(
                pairwise[candidate][opponent]["wins"] >= 2
                for opponent in arms if opponent != candidate
            )
        ]
        descriptive_winner = descriptive_winners[0] if len(descriptive_winners) == 1 else None
        task_decisions[task] = {
            "task": task, "endpoint": endpoint,
            "selection_rule_version": SELECTION_RULE_VERSION,
            "formal_decision_available": endpoint >= FORMAL_ENDPOINT,
            "strong_early_promotion": strong_early,
            "early_evidence_seed_count": early_evidence_seed_count,
            "strong_early_rule": (
                ">=2/3 paired wins versus both arms, all-task integrity, no farming, "
                "and >=2/3 strict hits OR >=2/3 persistent seeds where last512 "
                "joint>=2/constructive>=6 and previous512 joint>=1/constructive>=3"
            ),
            "paired_wins": pairwise,
            "eligible_new_arms": eligible,
            "selected_arm": selected,
            "selected_seed": None,
            "selected_arm_min_pairwise_wins": min_wins,
            "all_candidate_seeds_integrity_pass": selected_integrity,
            "candidate_farming_gate_pass": selected_farming_gate,
            "all_task_integrity": bool(selected and all_task_integrity),
            "descriptive_winner_arm": descriptive_winner,
            "control_non_improvement": bool(
                endpoint >= FORMAL_ENDPOINT and descriptive_winner == "legacy"
            ),
            "promote_selected_to_10000": selected is not None,
            "continuation_scope": "all three seeds of selected new arm",
            "return_used_for_selection": False,
        }
    return {
        "schema_version": 1,
        "analysis": "codex8_spectral_reward_three_arm_staged",
        "root": str(root.resolve()), "endpoint": endpoint,
        "tasks": list(tasks), "arms": list(arms), "seeds": list(SEEDS),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "overrun_policy": "rows with episode > endpoint are ignored",
        "formal_endpoint": FORMAL_ENDPOINT,
        "reports": reports, "arm_decisions": decisions,
        "task_decisions": task_decisions,
    }


def _public_seed_row(report: dict) -> dict:
    hit, recent = report["first_hit"], report["last1024"]
    return {
        "task": report["task"], "arm": report["arm"], "seed": report["seed"],
        "endpoint": report["endpoint"], "integrity_pass": report["integrity_pass"],
        "hit": int(hit["hit"]), "first_hit_episode": hit["episode"],
        "first_hit_action_proposals": hit["action_proposals"],
        "strict_last1024": recent["strict_count"], "joint_last1024": recent["joint_count"],
        "constructive_last1024": recent["constructive_count"],
        "spread_error_median_last1024": recent["spread_error_median"],
        "flatness_median_last1024": recent["flatness_median"],
        "reject_rate_last1024": recent["reject_rate_mean"],
        "phase_close_rate_last1024": recent.get("phase_close_rate", 0.0),
        "phase_z_rate_last1024": recent.get("phase_z_rate", 0.0),
        "phase_high_psi_bad_timeout_rate": recent.get("phase_high_psi_bad_timeout_rate", 0.0),
        "phase_farming_detected": report["phase_reward_hacking"]["farming_detected"],
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_artifacts(payload: dict, output: Path) -> dict:
    """Write a clearly preliminary checkpoint bundle; this is not final publication."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compact = json.loads(json.dumps(payload))
    for task in compact["tasks"]:
        for arm in compact["arms"]:
            for report in compact["reports"][task][arm]:
                report.pop("records", None)
                report.pop("optimizer_records", None)
                report.get("config_validation", {}).pop("config", None)
    json_path = output / "checkpoint_analysis.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    rows = [
        _public_seed_row(report)
        for task in payload["tasks"] for arm in payload["arms"]
        for report in payload["reports"][task][arm]
    ]
    csv_path = output / "seed_metrics.csv"
    _write_csv(csv_path, rows)
    md_path = output / "checkpoint_report_ja.md"
    lines = [
        "# Codex8 spectral staged checkpoint（暫定）", "",
        f"atomic endpoint: {payload['endpoint']}", "",
        "この資料は実行中checkpointの判定用であり、最終結果の公開物ではありません。returnはarm選択に使用していません。", "",
    ]
    for task in payload["tasks"]:
        decision = payload["task_decisions"][task]
        lines.append(
            f"- {task}: selected={decision['selected_arm']}, "
            f"promote={decision['promote_selected_to_10000']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def production_decision(
    reports: Sequence[dict] | dict, baseline_reports: Sequence[dict] | dict,
    prior_warning_count: int
) -> dict:
    """Stop immediately on integrity; require two consecutive collapse warnings."""

    reports = [reports] if isinstance(reports, dict) else list(reports)
    baseline_reports = (
        [baseline_reports] if isinstance(baseline_reports, dict)
        else list(baseline_reports)
    )
    if any(not row["integrity_pass"] for row in reports):
        return {
            "stop": True, "stop_kind": "integrity", "warning": False,
            "consecutive_warning_count": 0, "reasons": ["production integrity failure"],
        }
    current_signal = sum(
        4 * row["last512"]["strict_count"]
        + 2 * row["last512"]["joint_count"]
        + row["last512"]["constructive_count"] for row in reports
    )
    baseline_signal = sum(
        4 * row["last512"]["strict_count"]
        + 2 * row["last512"]["joint_count"]
        + row["last512"]["constructive_count"] for row in baseline_reports
    )
    farming = any(row["phase_reward_hacking"]["farming_detected"] for row in reports)
    warning = bool(current_signal < max(1, 0.2 * baseline_signal) or farming)
    warning_count = prior_warning_count + 1 if warning else 0
    return {
        "stop": warning_count >= 2,
        "stop_kind": "production_collapse" if warning_count >= 2 else None,
        "warning": warning, "consecutive_warning_count": warning_count,
        "reasons": ["two consecutive geometric-signal collapse warnings"] if warning_count >= 2 else [],
        "current_signal": current_signal, "baseline_signal": baseline_signal,
        "phase_reward_farming": farming,
    }


def write_production_artifacts(report: dict, decision: dict, output: Path) -> dict:
    """Persist a per-seed production checkpoint used by the staged runner."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compact = json.loads(json.dumps(report))
    compact.pop("records", None)
    compact.pop("optimizer_records", None)
    compact.get("config_validation", {}).pop("config", None)
    json_path = output / "production_checkpoint.json"
    json_path.write_text(json.dumps({
        "report": compact, "production_decision": decision,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output / "selected_seed_window.csv"
    _write_csv(csv_path, [_public_seed_row(report)])
    md_path = output / "checkpoint_report_ja.md"
    md_path.write_text(
        "# Codex8 production checkpoint（暫定）\n\n"
        f"- {report['task']} / {report['arm']} / seed{report['seed']}\n"
        f"- endpoint={report['endpoint']}, warning={decision['warning']}, "
        f"stop={decision['stop']}\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", type=int, choices=CHECKPOINTS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = analyze(args.root, args.endpoint)
    result = write_artifacts(payload, args.output) if args.output else None
    print(json.dumps({"task_decisions": payload["task_decisions"], "artifacts": result}, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
