"""Checkpoint-bounded analysis for the two-arm Codex7 staged experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Sequence

import ori_codex7 as core


TASKS = ("torus2", "water1", "five_d2", "five_d3", "watersqrt2")
ARMS = ("no_fixed", "historical_1p25")
SEEDS = (7101, 7102, 7103)
PROFILES = {
    "five_d2": "B", "five_d3": "B", "torus2": "A",
    "water1": "B", "watersqrt2": "B",
}
TASK_CONFIG = {
    "torus2": (2.0, 200, .2, 2, 1e-3, 3e-4, [1528, 4000, 7000], 1.0, "model/4unit_torus"),
    "water1": (1.0, 400, .3, 2, 5e-4, 1.5e-4, [2000, 4000, 7000], .12, "model/waterbomb"),
    "five_d2": (2.0, 400, .1, 1, 5e-4, 1.5e-4, [2000, 4000, 7000], 1.0, "model/five_bar_linkage"),
    "five_d3": (3.0, 400, .1, 1, 5e-4, 1.5e-4, [2000, 4000, 7000], 1.0, "model/five_bar_linkage"),
    "watersqrt2": (math.sqrt(2), 400, .3, 2, 5e-4, 1.5e-4, [2000, 4000, 7000], 1.0, "model/waterbomb"),
}
CHECKPOINTS = (512, 1024, 2048, 4096, 5120, 6144, 7168, 8192, 9216, 10000)


@dataclass(frozen=True)
class Gate:
    constructive_last1024: int
    joint_last1024: int
    median_flatness_cap: float
    median_spread_error_cap: float
    crossing_rate_cap: float


GATES = {
    "five_d2": Gate(6, 2, 0.70, 0.22, 0.25),
    "five_d3": Gate(6, 2, 0.70, 0.22, 0.10),
    "torus2": Gate(12, 2, 0.80, 0.25, 0.25),
    "water1": Gate(6, 2, 0.70, 0.22, 0.25),
    "watersqrt2": Gate(6, 2, 0.70, 0.22, 0.10),
}


def _float(row: dict, *names: str, default=math.nan) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return default


def _int(row: dict, *names: str, default=0) -> int:
    value = _float(row, *names, default=float(default))
    return int(value) if math.isfinite(value) else default


def _median(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else math.nan


def _mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(values)) if values else math.nan


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def read_history(path: Path, endpoint: int) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], [f"missing:{path}"]
    by_episode = {}
    errors = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = _int(row, "episode", default=-1)
            if 1 <= episode <= endpoint:
                if episode in by_episode:
                    errors.append(f"duplicate_episode:{episode}")
                by_episode[episode] = row
    missing = sorted(set(range(1, endpoint + 1)) - set(by_episode))
    if missing:
        errors.append(f"missing_episodes:{missing[:8]}:count={len(missing)}")
    return [by_episode[index] for index in sorted(by_episode)], errors


def classify(row: dict) -> dict:
    flatness = _float(row, "flatness_ratio", "flatness_potential")
    spread_error = _float(row, "spread_error", "spread_deviation")
    termination_reason = str(row.get("termination_reason", "")).lower()
    crossing = bool(
        _int(
            row, "ori7_terminal_crossing_count", "ori7_terminal_crossing",
            "ori7_crossing_terminal",
            "ori7_crossing_candidate_terminal", "self_crossing",
        )
        or termination_reason in {
            "self_crossing", "terminal_crossing", "crossing_terminal",
            "ori7_terminal_crossing",
        }
    )
    mode = str(row.get("ori7_crossing_terminal_mode", ""))
    crossing_fixed = (-1.25 if mode == "historical_1p25" else 0.0) if crossing else 0.0
    crossing_reward = _float(
        row, "ori7_terminal_crossing_reward", "ori7_crossing_terminal_reward",
        default=_float(row, "total_reward") if crossing else 0.0,
    )
    crossing_q_safe = _float(
        row, "ori7_crossing_terminal_q_safe", "ori7_terminal_crossing_q_safe",
        default=(crossing_reward - crossing_fixed) if crossing else 0.0,
    )
    strict = bool(
        not crossing and _int(row, "strict_compact_success", "success")
    )
    invalid = bool(
        crossing or _int(row, "correction_failed", "no_modes")
        or not math.isfinite(flatness)
        or not math.isfinite(_float(row, "total_reward"))
    )
    constructive = bool(
        not crossing and not invalid
        and (
            strict
            or (math.isfinite(spread_error) and spread_error <= 0.10)
            or _float(row, "ori6_maximum_spread_progress", default=0.0) >= 0.8
        )
    )
    joint = bool(
        not crossing and not invalid
        and math.isfinite(flatness) and flatness <= 0.10
        and math.isfinite(spread_error) and spread_error <= 0.10
    )
    decomposition = max(abs(_float(row, name, default=0.0)) for name in (
        "ori7_maximum_abs_crossing_reward_decomposition_error",
        "ori7_reward_decomposition_error",
        "ori7_maximum_abs_step_reward_decomposition_error",
        "ori6_reward_decomposition_error",
        "ori6_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
    ))
    nonfinite = sum(_int(row, name) for name in (
        "ori7_crossing_nonfinite_component_count",
        "ori7_nonfinite_reward_component_count",
        "ori6_nonfinite_reward_component_count",
        "ori6_spectral_nonfinite_component_count",
    ))
    return {
        "episode": _int(row, "episode"),
        "reward": _float(row, "total_reward"),
        "flatness": flatness,
        "spread_error": spread_error,
        "strict": strict,
        "constructive": constructive,
        "joint": joint,
        "crossing": crossing,
        "crossing_fixed_reward": crossing_fixed,
        "crossing_reward": crossing_reward,
        "crossing_q_safe": crossing_q_safe,
        "noncrossing_reward": _float(row, "total_reward") if not crossing else math.nan,
        "invalid": invalid,
        "decomposition": decomposition,
        "nonfinite": nonfinite,
        "steps": _int(row, "steps"),
        "projectors": _int(row, "projection_attempts"),
    }


def window(records: Sequence[dict], start: int, end: int) -> dict:
    rows = [row for row in records if start <= row["episode"] <= end]
    count = len(rows)
    return {
        "start": start, "end": end, "count": count,
        "return_mean": _mean(row["reward"] for row in rows),
        "flatness_median": _median(row["flatness"] for row in rows),
        "spread_error_median": _median(row["spread_error"] for row in rows),
        "strict_count": sum(row["strict"] for row in rows),
        "strict_rate": _rate(sum(row["strict"] for row in rows), count),
        "constructive_count": sum(row["constructive"] for row in rows),
        "joint_count": sum(row["joint"] for row in rows),
        "crossing_count": sum(row["crossing"] for row in rows),
        "crossing_rate": _rate(sum(row["crossing"] for row in rows), count),
        "crossing_fixed_reward_total": sum(
            row["crossing_fixed_reward"] for row in rows if row["crossing"]
        ),
        "crossing_reward_mean": _mean(
            row["crossing_reward"] for row in rows if row["crossing"]
        ),
        "crossing_q_safe_mean": _mean(
            row["crossing_q_safe"] for row in rows if row["crossing"]
        ),
        "noncrossing_return_mean": _mean(row["noncrossing_reward"] for row in rows),
        "invalid_count": sum(row["invalid"] for row in rows),
        "nonfinite_count": sum(row["nonfinite"] for row in rows),
        "maximum_decomposition_error": max(
            [row["decomposition"] for row in rows] or [0.0]
        ),
        "mechanical_steps": sum(row["steps"] for row in rows),
        "projector_calls": sum(row["projectors"] for row in rows),
    }


def run_directory(root: Path, arm: str, task: str, seed: int) -> Path:
    return root / arm / task / PROFILES[task] / f"seed{seed}"


def _same(actual, expected) -> bool:
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, float):
        try: return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError): return False
    return actual == expected


def expected_config(task: str, arm: str, seed: int) -> dict:
    target, steps, width, dimension, actor_lr, critic_lr, milestones, far, model = TASK_CONFIG[task]
    return {
        "seed": seed, "target_spread": target, "max_steps": steps,
        "max_step_width": width, "flatness_subspace_dimension": dimension,
        "actor_learning_rate": actor_lr, "critic_learning_rate": critic_lr,
        "actor_lr_milestones": milestones, "actor_lr_factors": [.3, .1, .03],
        "batch_size": 8, "entropy_coefficient": .001, "gamma": 1.0,
        "episodes": 10000, "random_search": False,
        "update_method": "actor_critic",
        "ori6_spectral_subspace": True,
        "ori6_spectral_relative_epsilon": 1e-12,
        "ori6_spectral_weight": .05, "ori5_crossing_penalty": 1.25,
        "ori5_quality_weight": 1.0, "near_miss_spread_error_far": far,
        "ori5_flatness_clip": 2.0,
        "ori5_spread_milestone_initial_factor": 2.0,
        "ori5_spread_milestone_bonus": 0.0,
        "ori5_joint_flatness_thresholds": [.5, .1, .01],
        "ori5_joint_milestone_bonuses": [.2, .3, .5],
        "ori6_spread_frontier_mode": "state_potential",
        "ori6_spread_frontier_bonus": .9,
        "ori6_spread_frontier_flatness_cap": 2.0,
        "ori5_timeout_penalty": 1.0, "ori5_wrong_flat_penalty": 2.0,
        "ori5_success_bonus": 10.0, "ori5_success_speed_bonus_weight": .5,
        "spread_tolerance": .05, "flatness_tolerance": .001,
        "numerical_failure_penalty": 10.0, "no_mode_penalty": 5.0,
        "strain_tolerance": 1e-8, "crossing_tolerance": 1e-10,
        "min_step_width": 1e-4, "projection_shrink": .5,
        "max_correction_ratio": 1.0, "max_actual_step_ratio": 1.25,
        "terminate_wrong_flat": True,
        "data_dir": str((Path(core.__file__).resolve().parent / model).resolve()),
        "ori7_crossing_terminal_mode": arm,
    }


def validate_history_row(row: dict, arm: str) -> list[str]:
    errors = []
    required_exact = {
        "ori7_candidate": core.CANDIDATE_NAME,
        "ori7_candidate_schema_version": core.CANDIDATE_SCHEMA_VERSION,
        "ori7_crossing_policy": core.CROSSING_POLICY,
        "ori7_crossing_terminal_mode": arm,
        "ori7_reward_formula_sha256": core.REWARD_FORMULA_SHA256,
        "ori7_transition_formula_sha256": core.TRANSITION_FORMULA_SHA256,
    }
    for field, expected in required_exact.items():
        if row.get(field) != expected: errors.append(f"history_{field}")
    numeric_required = (
        "ori7_terminal_crossing_count", "ori7_terminal_crossing_reward",
        "ori7_maximum_abs_crossing_reward_decomposition_error",
        "ori7_crossing_nonfinite_component_count", "self_crossing",
    )
    parsed = {}
    for field in numeric_required:
        if row.get(field) in (None, ""):
            errors.append(f"history_missing_{field}"); continue
        try: parsed[field] = float(row[field])
        except (TypeError, ValueError): errors.append(f"history_nonfinite_{field}"); continue
        if not math.isfinite(parsed[field]): errors.append(f"history_nonfinite_{field}")
    count = parsed.get("ori7_terminal_crossing_count")
    self_crossing = parsed.get("self_crossing")
    reason = str(row.get("termination_reason", "")).lower()
    if not reason: errors.append("history_missing_termination_reason")
    if count not in (0.0, 1.0): errors.append("history_crossing_count_not_binary")
    if count == 1.0:
        if self_crossing != 1.0 or reason != "self_crossing":
            errors.append("history_crossing_terminal_inconsistent")
        quality = _float(row, "ori5_total_terminal_quality_reward")
        reward = parsed.get("ori7_terminal_crossing_reward", math.nan)
        expected = (0.0 if arm == "no_fixed" else -1.25) + quality
        if not math.isfinite(quality) or not math.isfinite(reward) or not math.isclose(reward, expected, abs_tol=1e-8):
            errors.append("history_crossing_reward_contract")
    elif count == 0.0:
        if self_crossing != 0.0 or reason == "self_crossing":
            errors.append("history_noncrossing_terminal_inconsistent")
        for field in ("ori7_terminal_crossing_reward",
                      "ori7_maximum_abs_crossing_reward_decomposition_error",
                      "ori7_crossing_nonfinite_component_count"):
            if parsed.get(field) != 0.0: errors.append(f"history_noncrossing_{field}_nonzero")
    if parsed.get("ori7_crossing_nonfinite_component_count", 1.0) != 0.0:
        errors.append("history_crossing_nonfinite")
    if abs(parsed.get("ori7_maximum_abs_crossing_reward_decomposition_error", math.inf)) > 1e-8:
        errors.append("history_crossing_decomposition")
    global_decomposition = (
        "ori6_reward_decomposition_error",
        "ori6_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
    )
    for field in global_decomposition:
        if row.get(field) in (None, ""):
            errors.append(f"history_missing_{field}"); continue
        try: value = float(row[field])
        except (TypeError, ValueError):
            errors.append(f"history_nonfinite_{field}"); continue
        if not math.isfinite(value): errors.append(f"history_nonfinite_{field}")
        elif abs(value) > 1e-8: errors.append(f"history_decomposition_{field}")
    for field in ("ori6_nonfinite_reward_component_count",
                  "ori6_spectral_nonfinite_component_count"):
        if row.get(field) in (None, ""):
            errors.append(f"history_missing_{field}"); continue
        try: value = float(row[field])
        except (TypeError, ValueError):
            errors.append(f"history_nonfinite_{field}"); continue
        if not math.isfinite(value) or value != 0.0:
            errors.append(f"history_nonzero_{field}")
    return errors


def validate_config(path: Path, arm: str, task: str, seed: int) -> list[str]:
    if not path.is_file():
        return [f"missing:{path}"]
    config = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    checks = expected_config(task, arm, seed)
    for field, expected in checks.items():
        actual = config.get(field)
        valid = _same(actual, expected)
        if not valid:
            errors.append(f"config_{field}")
    actual_arm = config.get("ori7_crossing_terminal_mode")
    if actual_arm != arm:
        errors.append("config_arm")
    for field, expected in {
        "ori7_candidate": core.CANDIDATE_NAME,
        "ori7_candidate_schema_version": core.CANDIDATE_SCHEMA_VERSION,
        "ori7_crossing_policy": core.CROSSING_POLICY,
        "ori7_update_method": "episode_separated_monte_carlo_actor_critic",
    }.items():
        if config.get(field) != expected:
            errors.append(f"config_{field}")
    for primary, fallback, expected in (
        ("ori7_reward_formula_sha256", "reward_formula_sha256", core.REWARD_FORMULA_SHA256),
        ("ori7_transition_formula_sha256", "transition_formula_sha256", core.TRANSITION_FORMULA_SHA256),
    ):
        if config.get(primary, config.get(fallback)) != expected:
            errors.append(f"config_{primary}")
    fixed = config.get("ori7_reward_parameters", {}).get("crossing_fixed_reward")
    try:
        expected_fixed_reward = 0.0 if arm == "no_fixed" else -1.25
        if not math.isclose(float(fixed), expected_fixed_reward, abs_tol=1e-12):
            errors.append("config_crossing_fixed_penalty")
    except (TypeError, ValueError):
        errors.append("config_crossing_fixed_penalty")
    return errors


def seed_report(root: Path, arm: str, task: str, seed: int, endpoint: int) -> dict:
    directory = run_directory(root, arm, task, seed)
    rows, errors = read_history(directory / "training_history.csv", endpoint)
    for row in rows:
        errors.extend(
            f"episode_{_int(row, 'episode')}:{error}"
            for error in validate_history_row(row, arm)
        )
    errors += validate_config(directory / "run_config.json", arm, task, seed)
    records = [classify(row) for row in rows]
    full = window(records, 1, endpoint)
    recent = window(records, max(1, endpoint - 1023), endpoint)
    late = window(records, max(1, endpoint - 511), endpoint)
    integrity = bool(
        not errors
        and full["nonfinite_count"] == 0
        and full["maximum_decomposition_error"] <= 1e-8
    )
    gate = GATES[task]
    evidence = bool(
        full["strict_count"] > 0
        or (
            recent["constructive_count"] >= gate.constructive_last1024
            and recent["joint_count"] >= gate.joint_last1024
        )
    )
    persistence = bool(
        late["strict_count"] > 0
        or (
            late["constructive_count"] >= max(1, gate.constructive_last1024 // 2)
            and late["joint_count"] >= 1
        )
    )
    quality = bool(
        math.isfinite(recent["flatness_median"])
        and recent["flatness_median"] <= gate.median_flatness_cap
        and math.isfinite(recent["spread_error_median"])
        and recent["spread_error_median"] <= gate.median_spread_error_cap
        and recent["crossing_rate"] <= gate.crossing_rate_cap
    )
    promising = bool(endpoint >= 4096 and integrity and evidence and persistence and quality)
    return {
        "arm": arm, "task": task, "seed": seed, "endpoint": endpoint,
        "directory": str(directory), "integrity_pass": integrity,
        "integrity_errors": errors, "full": full, "last1024": recent,
        "last512": late, "evidence": evidence, "persistence": persistence,
        "quality": quality, "promising_at_4096": promising,
    }


def catastrophic(reports: Sequence[dict], endpoint: int) -> tuple[bool, list[str]]:
    if any(not report["integrity_pass"] for report in reports):
        return True, ["integrity failure"]
    if endpoint < 1024:
        return False, []
    no_signal = all(
        report["full"]["strict_count"] == 0
        and report["last512"]["constructive_count"] == 0
        for report in reports
    )
    invalid_lock = all(
        report["last512"]["invalid_count"] >= 0.95 * report["last512"]["count"]
        for report in reports
    )
    if no_signal and invalid_lock:
        return True, ["all-seed no-signal invalid lock"]
    return False, []


def seed_rank(report: dict):
    def finite(value, fallback):
        return float(value) if math.isfinite(float(value)) else fallback
    return (
        report["last1024"]["strict_count"],
        report["last512"]["strict_count"],
        report["last1024"]["joint_count"],
        report["last1024"]["constructive_count"],
        -finite(report["last1024"]["spread_error_median"], math.inf),
        -finite(report["last1024"]["flatness_median"], math.inf),
        finite(report["last1024"]["noncrossing_return_mean"], -math.inf),
        -report["seed"],
    )


def arm_decision(reports: Sequence[dict], endpoint: int) -> dict:
    arm, task = reports[0]["arm"], reports[0]["task"]
    stop, reasons = catastrophic(reports, endpoint)
    result = {
        "arm": arm, "task": task, "endpoint": endpoint,
        "rule": "integrity/catastrophic only before 4096; 2-of-3 at 4096",
        "stop": stop, "reasons": reasons,
        "promote": False, "promising_seed_count": 0,
        "promising_seeds": [], "selected_seed": None,
    }
    if endpoint >= 4096 and not stop:
        promising = [report for report in reports if report["promising_at_4096"]]
        result.update({
            "promising_seed_count": len(promising),
            "promising_seeds": [row["seed"] for row in promising],
            "promote": len(promising) >= 2,
            "stop": len(promising) < 2,
            "reasons": [
                "2/3 persistent evidence" if len(promising) >= 2
                else "fewer than 2/3 promising seeds"
            ],
            "selected_seed": (
                max(promising, key=seed_rank)["seed"] if len(promising) >= 2
                else None
            ),
        })
    return result


def _finite(value, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def _paired_metric(report: dict) -> tuple:
    recent = report["last1024"]
    return (
        int(recent["strict_count"]), int(recent["joint_count"]),
        int(recent["constructive_count"]),
        _finite(recent["noncrossing_return_mean"], -math.inf),
        -_finite(recent["crossing_rate"], math.inf),
        -_finite(recent["spread_error_median"], math.inf),
        -_finite(recent["flatness_median"], math.inf),
    )


def paired_arm_comparison(no_fixed: Sequence[dict], historical: Sequence[dict]) -> dict:
    """Pre-registered same-seed comparison; best-seed cross-arm selection is forbidden."""

    no_by_seed = {row["seed"]: row for row in no_fixed}
    hi_by_seed = {row["seed"]: row for row in historical}
    pairs = []
    wins = {"no_fixed": 0, "historical_1p25": 0, "ties": 0}
    delta_fields = (
        "strict_count", "joint_count", "constructive_count",
        "noncrossing_return_mean", "crossing_rate",
        "spread_error_median", "flatness_median",
    )
    for seed in SEEDS:
        left, right = no_by_seed[seed], hi_by_seed[seed]
        left_metric, right_metric = _paired_metric(left), _paired_metric(right)
        winner = "no_fixed" if left_metric > right_metric else (
            "historical_1p25" if right_metric > left_metric else "ties"
        )
        wins[winner] += 1
        deltas = {}
        for field in delta_fields:
            a = _finite(left["last1024"].get(field), math.nan)
            b = _finite(right["last1024"].get(field), math.nan)
            deltas[field] = a - b if math.isfinite(a) and math.isfinite(b) else None
        pairs.append({"seed": seed, "winner": winner, "no_fixed_minus_historical": deltas})
    selected = next((arm for arm in ARMS if wins[arm] >= 2), None)
    status = "selected_by_at_least_2_of_3_paired_wins" if selected else "inconclusive"
    return {"wins": wins, "pairs": pairs, "selected_arm": selected,
            "status": status, "rule": "requires >=2/3 same-seed wins; ties never auto-promote"}


def production_decision(report: dict, baseline_4096: dict,
                        prior_warning_count: int) -> dict:
    """Integrity stops immediately; collapse requires two consecutive warnings."""

    gate = GATES[report["task"]]
    recent = report["last512"]
    def signal(window_report):
        return (4 * window_report["strict_count"] + 2 * window_report["joint_count"]
                + window_report["constructive_count"])
    current_signal = signal(recent)
    baseline_signal = signal(baseline_4096["last512"])
    signal_floor = max(1, gate.constructive_last1024 // 4)
    warning_threshold = max(signal_floor, 0.2 * baseline_signal)
    no_signal = current_signal < warning_threshold
    degraded = (
        _finite(recent["flatness_median"], math.inf) > gate.median_flatness_cap
        or _finite(recent["spread_error_median"], math.inf) > gate.median_spread_error_cap
        or _finite(recent["crossing_rate"], math.inf) > gate.crossing_rate_cap
    )
    warning = bool(report["integrity_pass"] and no_signal and degraded)
    count = prior_warning_count + 1 if warning else 0
    stop = not report["integrity_pass"] or count >= 2
    reasons = (["integrity failure"] if not report["integrity_pass"] else
               (["two consecutive production collapse warnings"] if count >= 2 else
                (["production collapse warning 1/2"] if warning else [])))
    return {"warning": warning, "consecutive_warning_count": count,
            "stop": stop, "reasons": reasons, "no_signal": no_signal,
            "degraded_quality_or_crossing": degraded,
            "current_primary_signal": current_signal,
            "baseline_4096_primary_signal": baseline_signal,
            "warning_signal_threshold": warning_threshold}


def analyze(root: Path, endpoint: int, tasks=TASKS, arms=ARMS) -> dict:
    reports = {}
    decisions = {}
    task_decisions = {}
    for task in tasks:
        reports[task] = {}
        decisions[task] = {}
        for arm in arms:
            cohort = [seed_report(root, arm, task, seed, endpoint) for seed in SEEDS]
            reports[task][arm] = cohort
            decisions[task][arm] = arm_decision(cohort, endpoint)
        passing = [
            arm for arm in arms if decisions[task][arm]["promote"]
        ]
        selected_arm = None
        selected_seed = None
        paired = None
        if endpoint >= 4096 and passing:
            if len(passing) == 2:
                paired = paired_arm_comparison(
                    reports[task]["no_fixed"], reports[task]["historical_1p25"]
                )
                selected_arm = paired["selected_arm"]
            else:
                selected_arm = passing[0]
            if selected_arm is not None:
                selected_seed = decisions[task][selected_arm]["selected_seed"]
        task_decisions[task] = {
            "task": task, "passing_arms": passing,
            "selected_arm": selected_arm, "selected_seed": selected_seed,
            "promote_selected_to_10000": selected_arm is not None,
            "paired_arm_comparison": paired,
            "comparative_conclusion": bool(
                selected_arm is not None and (len(passing) == 1 or paired is not None)
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "codex7_terminal_crossing_two_arm_staged",
        "root": str(root.resolve()), "endpoint": endpoint,
        "tasks": list(tasks), "arms": list(arms), "seeds": list(SEEDS),
        "reports": reports, "arm_decisions": decisions,
        "task_decisions": task_decisions,
    }


def write_artifacts(payload: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "checkpoint_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for task, arms in payload["reports"].items():
        for arm, reports in arms.items():
            for report in reports:
                rows.append({
                    "task": task, "arm": arm, "seed": report["seed"],
                    "endpoint": report["endpoint"],
                    "integrity_pass": report["integrity_pass"],
                    "strict_last1024": report["last1024"]["strict_count"],
                    "constructive_last1024": report["last1024"]["constructive_count"],
                    "joint_last1024": report["last1024"]["joint_count"],
                    "crossing_rate_last1024": report["last1024"]["crossing_rate"],
                    "crossing_fixed_reward_total_last1024": report["last1024"]["crossing_fixed_reward_total"],
                    "crossing_reward_mean_last1024": report["last1024"]["crossing_reward_mean"],
                    "crossing_q_safe_mean_last1024": report["last1024"]["crossing_q_safe_mean"],
                    "noncrossing_return_mean_last1024": report["last1024"]["noncrossing_return_mean"],
                    "flatness_median_last1024": report["last1024"]["flatness_median"],
                    "spread_error_median_last1024": report["last1024"]["spread_error_median"],
                    "promising": report["promising_at_4096"],
                })
    csv_path = output / "seed_windows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    plots = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for task in payload["tasks"]:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for arm in payload["arms"]:
                for report in payload["reports"][task][arm]:
                    history, _ = read_history(
                        Path(report["directory"]) / "training_history.csv",
                        payload["endpoint"],
                    )
                    x = [_int(row, "episode") for row in history]
                    axes[0].plot(x, [_float(row, "total_reward") for row in history],
                                 alpha=.35, linewidth=.5, label=f"{arm}-{report['seed']}")
                    axes[1].plot(x, [_float(row, "flatness_ratio", "flatness_potential") for row in history],
                                 alpha=.35, linewidth=.5)
            axes[0].set_title(f"{task}: return"); axes[1].set_title(f"{task}: terminal flatness")
            axes[0].set_xlabel("episode"); axes[1].set_xlabel("episode")
            axes[0].legend(fontsize=6, ncol=2)
            path = output / f"learning_curve_{task}.png"
            figure.savefig(path, dpi=150); plt.close(figure); plots.append(str(path))
    except Exception as error:
        payload["plot_error"] = repr(error)
    report_path = output / "report.md"
    lines = ["# Codex7 terminal-crossing staged checkpoint", ""]
    for task, decision in payload["task_decisions"].items():
        lines.append(
            f"- {task}: passing={decision['passing_arms']}, "
            f"selected={decision['selected_arm']}/{decision['selected_seed']}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path),
            "plots": plots, "report": str(report_path)}


def write_production_artifacts(report: dict, decision: dict, output: Path) -> dict:
    """Persist every selected-seed production checkpoint, including warnings."""

    output.mkdir(parents=True, exist_ok=True)
    payload = {"report": report, "production_decision": decision}
    json_path = output / "production_checkpoint.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output / "selected_seed_window.csv"
    row = {
        "task": report["task"], "arm": report["arm"], "seed": report["seed"],
        "endpoint": report["endpoint"], "integrity_pass": report["integrity_pass"],
        "strict_last512": report["last512"]["strict_count"],
        "constructive_last512": report["last512"]["constructive_count"],
        "joint_last512": report["last512"]["joint_count"],
        "crossing_rate_last512": report["last512"]["crossing_rate"],
        "noncrossing_return_last512": report["last512"]["noncrossing_return_mean"],
        "flatness_median_last512": report["last512"]["flatness_median"],
        "spread_error_median_last512": report["last512"]["spread_error_median"],
        "warning": decision["warning"],
        "consecutive_warning_count": decision["consecutive_warning_count"],
        "stop": decision["stop"],
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    plot_path = output / "learning_curve_selected_seed.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history, _ = read_history(Path(report["directory"]) / "training_history.csv", report["endpoint"])
        x = [_int(row, "episode") for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(x, [_float(row, "total_reward") for row in history], linewidth=.6)
        axes[1].plot(x, [_float(row, "flatness_ratio", "flatness_potential") for row in history], linewidth=.6)
        axes[0].set_title("return"); axes[1].set_title("terminal flatness")
        for axis in axes: axis.set_xlabel("episode")
        figure.savefig(plot_path, dpi=150); plt.close(figure)
    except Exception:
        plot_path = None
    report_path = output / "report.md"
    report_path.write_text(
        f"# Production checkpoint {report['endpoint']}\n\n"
        f"- {report['task']} {report['arm']} seed{report['seed']}\n"
        f"- warning={decision['warning']}, consecutive={decision['consecutive_warning_count']}, stop={decision['stop']}\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path),
            "plot": str(plot_path) if plot_path else None, "report": str(report_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", type=int, choices=CHECKPOINTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = analyze(args.root, args.endpoint)
    artifacts = write_artifacts(payload, args.output)
    print(json.dumps({"payload": payload, "artifacts": artifacts}, indent=2))
    return payload


if __name__ == "__main__":
    main()
