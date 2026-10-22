"""Checkpoint-safe analysis for the ori6 zero-reject staged experiment.

Only rows at or before an atomic model checkpoint are authoritative.  The
module deliberately has no dependency on the older authority-bound analyzers.
It is imported by ``run_ori_codex6_zero_reject_staged.py`` and can also be run
directly to rebuild JSON/CSV/PNG/Markdown artifacts from an experiment root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


CANDIDATE = "ori6_RC_spectral_subspace_zero_reject_reward"
SEEDS = (4201, 4202, 4203)
PROFILES = {"torus2": "A", "water1": "B"}
TASK_ORDER = ("torus2", "water1")


@dataclass(frozen=True)
class TaskGate:
    constructive_count_last_1024: int
    joint_count_last_1024: int
    median_flatness_cap: float
    median_spread_error_cap: float
    minimum_late_to_peak_ratio: float


TASK_GATES = {
    "torus2": TaskGate(12, 2, 0.80, 0.25, 0.20),
    "water1": TaskGate(6, 2, 0.70, 0.22, 0.20),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _float(row: dict, names: Sequence[str], default: float = math.nan) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return parsed
    return default


def _int(row: dict, names: Sequence[str], default: int = 0) -> int:
    value = _float(row, names, float(default))
    return int(value) if math.isfinite(value) else default


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else math.nan


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else math.nan


def _percentile(values: Iterable[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def read_history(path: Path, endpoint: int) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [f"missing history: {path}"]
    rows_by_episode: dict[int, dict] = {}
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    episode = int(float(row.get("episode", "")))
                except (TypeError, ValueError):
                    errors.append("unparseable episode row")
                    continue
                if episode <= endpoint:
                    if episode in rows_by_episode:
                        errors.append(f"duplicate episode {episode}")
                    rows_by_episode[episode] = row
    except (OSError, csv.Error) as error:
        return [], [f"history read failed: {error}"]
    expected = set(range(1, endpoint + 1))
    observed = set(rows_by_episode)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(
            f"missing episodes: {missing[:5]}" +
            (f" ... ({len(missing)} total)" if len(missing) > 5 else "")
        )
    if extra:
        errors.append(f"unexpected episodes: {extra[:5]}")
    return [rows_by_episode[key] for key in sorted(rows_by_episode)], errors


def read_updates(path: Path, endpoint: int) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [f"missing update history: {path}"]
    rows_by_episode: dict[int, dict] = {}
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                episode = _int(row, ("episode",), -1)
                if 0 < episode <= endpoint:
                    if episode in rows_by_episode:
                        errors.append(f"duplicate update episode {episode}")
                    rows_by_episode[episode] = row
    except (OSError, csv.Error) as error:
        return [], [f"update history read failed: {error}"]
    expected = set(range(8, endpoint + 1, 8))
    missing = sorted(expected - set(rows_by_episode))
    if missing:
        errors.append(
            f"missing update episodes: {missing[:5]}" +
            (f" ... ({len(missing)} total)" if len(missing) > 5 else "")
        )
    return [rows_by_episode[key] for key in sorted(rows_by_episode)], errors


def update_window_summary(rows: Sequence[dict], start: int, end: int) -> dict:
    chosen = [row for row in rows if start <= _int(row, ("episode",)) <= end]
    critic_loss = [_float(row, ("critic_loss",)) for row in chosen]
    critic_grad = [_float(row, ("critic_gradient_norm",)) for row in chosen]
    actor_grad = [_float(row, ("actor_gradient_norm",)) for row in chosen]
    entropy = [_float(row, ("entropy",)) for row in chosen]
    actor_lr = [_float(row, ("actor_learning_rate",)) for row in chosen]
    return {
        "update_count": len(chosen),
        "critic_loss_mean": _mean(critic_loss),
        "critic_loss_p95": _percentile(critic_loss, 0.95),
        "critic_loss_max": max([value for value in critic_loss if math.isfinite(value)] or [math.nan]),
        "critic_gradient_norm_mean": _mean(critic_grad),
        "critic_gradient_norm_p95": _percentile(critic_grad, 0.95),
        "critic_gradient_norm_max": max([value for value in critic_grad if math.isfinite(value)] or [math.nan]),
        "critic_gradient_over_clip_fraction": _safe_ratio(
            sum(math.isfinite(value) and value > 1.0 for value in critic_grad), len(chosen)
        ),
        "actor_gradient_norm_mean": _mean(actor_grad),
        "actor_gradient_norm_p95": _percentile(actor_grad, 0.95),
        "actor_gradient_norm_max": max([value for value in actor_grad if math.isfinite(value)] or [math.nan]),
        "actor_gradient_over_clip_fraction": _safe_ratio(
            sum(math.isfinite(value) and value > 0.5 for value in actor_grad), len(chosen)
        ),
        "entropy_mean": _mean(entropy),
        "actor_learning_rate_first": actor_lr[0] if actor_lr else math.nan,
        "actor_learning_rate_last": actor_lr[-1] if actor_lr else math.nan,
        "actor_learning_rate_min": min([value for value in actor_lr if math.isfinite(value)] or [math.nan]),
        "actor_learning_rate_max": max([value for value in actor_lr if math.isfinite(value)] or [math.nan]),
    }


def normalise(row: dict) -> dict:
    flatness = _float(
        row, ("ori5_last_valid_flatness_ratio", "flatness_ratio")
    )
    spread_error = _float(
        row, ("ori5_last_valid_spread_error", "spread_error")
    )
    invalid = any(
        _int(row, (field,)) != 0
        for field in ("self_crossing", "correction_failed", "no_modes")
    )
    strict = _int(row, ("success", "strict_compact_success")) != 0
    path_joint_discovery = (
        not invalid
        and _int(
            row,
            ("ori6_joint_milestones_awarded", "ori5_joint_milestones_awarded"),
        ) > 0
    )
    constructive = (
        not invalid
        and math.isfinite(flatness)
        and math.isfinite(spread_error)
        and flatness <= 0.5
        and spread_error <= 0.1
    )
    terminal_joint = constructive and flatness <= 0.1
    decomposition_fields = tuple(
        name for name in row
        if name.endswith("reward_decomposition_error")
        or name.endswith("abs_step_reward_decomposition_error")
    )
    nonfinite_fields = tuple(
        name for name in row if "nonfinite" in name and "count" in name
    )
    return {
        "episode": _int(row, ("episode",)),
        "reward": _float(row, ("total_reward",)),
        "flatness": flatness,
        "spread_error": spread_error,
        "spread": _float(row, ("ori5_last_valid_spread", "spread")),
        "steps": _int(row, ("steps",)),
        "strict": strict,
        "joint": terminal_joint,
        "path_joint_discovery": path_joint_discovery,
        "constructive": constructive,
        "near_miss": _int(row, ("near_miss_bonus_awarded",)) != 0,
        "wrong_flat": _int(row, ("wrong_flat",)) != 0,
        "invalid": invalid,
        "rejects": _int(row, ("ori6_rejected_crossing_count",)),
        "reject_rate": _float(row, ("ori6_rejected_crossing_rate",), 0.0),
        "decomposition_max": max(
            [abs(_float(row, (name,), 0.0)) for name in decomposition_fields]
            or [0.0]
        ),
        "nonfinite_count": sum(
            _int(row, (name,)) for name in nonfinite_fields
        ),
        "zero_reject_ordinary_reward": _float(
            row, ("ori6_zero_reject_total_reward",), 0.0
        ),
    }


def window_summary(records: Sequence[dict], start: int, end: int) -> dict:
    chosen = [row for row in records if start <= row["episode"] <= end]
    count = len(chosen)
    return {
        "start_episode": start,
        "end_episode": end,
        "episode_count": count,
        "mean_return": _mean(row["reward"] for row in chosen),
        "median_return": _median(row["reward"] for row in chosen),
        "mean_flatness": _mean(row["flatness"] for row in chosen),
        "median_flatness": _median(row["flatness"] for row in chosen),
        "mean_spread_error": _mean(row["spread_error"] for row in chosen),
        "median_spread_error": _median(row["spread_error"] for row in chosen),
        "strict_count": sum(row["strict"] for row in chosen),
        "strict_rate": _safe_ratio(sum(row["strict"] for row in chosen), count),
        "valid_joint_count": sum(row["joint"] for row in chosen),
        "valid_joint_rate": _safe_ratio(sum(row["joint"] for row in chosen), count),
        "constructive_count": sum(row["constructive"] for row in chosen),
        "constructive_rate": _safe_ratio(
            sum(row["constructive"] for row in chosen), count
        ),
        "path_joint_discovery_count": sum(
            row["path_joint_discovery"] for row in chosen
        ),
        "near_miss_count": sum(row["near_miss"] for row in chosen),
        "wrong_flat_count": sum(row["wrong_flat"] for row in chosen),
        "invalid_count": sum(row["invalid"] for row in chosen),
        "mean_reject_count": _mean(row["rejects"] for row in chosen),
        "mean_reject_rate": _mean(row["reject_rate"] for row in chosen),
        "mechanical_steps": sum(row["steps"] for row in chosen),
        "max_abs_reward_decomposition_error": max(
            [row["decomposition_max"] for row in chosen] or [math.inf]
        ),
        "nonfinite_component_count": sum(
            row["nonfinite_count"] for row in chosen
        ),
        "max_abs_zero_reject_ordinary_reward": max(
            [abs(row["zero_reject_ordinary_reward"]) for row in chosen] or [math.inf]
        ),
    }


def _recursive_values(payload, key: str) -> list:
    values = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key:
                values.append(value)
            values.extend(_recursive_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_recursive_values(value, key))
    return values


def validate_config(path: Path, task: str, seed: int) -> dict:
    errors: list[str] = []
    if not path.is_file():
        return {"pass": False, "errors": [f"missing config: {path}"]}
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {"pass": False, "errors": [f"config read failed: {error}"]}
    expected_cutoff = 1.0 if task == "torus2" else 0.40
    expected_milestones = [1528, 4000, 7000] if task == "torus2" else [2000, 4000, 7000]
    checks = {
        "episodes": config.get("episodes") == 10000,
        "seed": config.get("seed") == seed,
        "checkpoint_every": config.get("checkpoint_every") == 512,
        "near_miss_spread_error_far": math.isclose(
            float(config.get("near_miss_spread_error_far", math.nan)),
            expected_cutoff,
            abs_tol=1e-12,
        ),
        "actor_lr_milestones": config.get("actor_lr_milestones") == expected_milestones,
        "actor_lr_factors": config.get("actor_lr_factors") == [0.3, 0.1, 0.03],
        "spectral_weight": math.isclose(
            float(config.get("ori6_spectral_weight", math.nan)), 0.05,
            abs_tol=1e-15,
        ),
        "zero_reject_enabled": config.get("ori6_zero_reject_reward") is True,
    }
    candidate_values = _recursive_values(config, "candidate") + [
        config.get("ori6_candidate"), config.get("ori6_zero_reject_candidate")
    ]
    checks["candidate"] = CANDIDATE in candidate_values
    ordinary_formulas = _recursive_values(config, "ordinary_rejected_reward_formula")
    checks["ordinary_reject_formula"] = any(
        str(value).strip().lower() in {"0", "0.0", "zero"}
        for value in ordinary_formulas
    )
    horizon_formulas = _recursive_values(config, "horizon_rejected_reward_formula")
    checks["horizon_timeout_formula"] = any(
        "timeout" in str(value).lower() for value in horizon_formulas
    )
    errors.extend(name for name, passed in checks.items() if not passed)
    return {"pass": not errors, "checks": checks, "errors": errors}


def seed_report(root: Path, task: str, seed: int, endpoint: int) -> dict:
    profile = PROFILES[task]
    run_dir = root / task / profile / f"seed{seed}"
    rows, history_errors = read_history(run_dir / "training_history.csv", endpoint)
    update_rows, update_errors = read_updates(run_dir / "update_history.csv", endpoint)
    records = [normalise(row) for row in rows]
    config = validate_config(run_dir / "run_config.json", task, seed)
    first_end = min(1024, endpoint)
    first = window_summary(records, 1, first_end)
    recent = window_summary(records, max(1, endpoint - 1023), endpoint)
    late = window_summary(records, max(1, endpoint - 511), endpoint)
    windows = []
    for start in range(1, endpoint + 1, 512):
        end = min(start + 511, endpoint)
        windows.append({
            **window_summary(records, start, end),
            **update_window_summary(update_rows, start, end),
        })
    integrity_errors = [*history_errors, *update_errors]
    if recent["nonfinite_component_count"]:
        integrity_errors.append("nonfinite reward component")
    if recent["max_abs_reward_decomposition_error"] > 1.0e-8:
        integrity_errors.append("reward decomposition error exceeds 1e-8")
    if recent["max_abs_zero_reject_ordinary_reward"] > 1.0e-12:
        integrity_errors.append("ordinary zero-reject reward is not exactly zero")
    if not config["pass"]:
        integrity_errors.extend(f"config:{item}" for item in config["errors"])
    threshold = TASK_GATES[task]
    peak_constructive = max(
        (window["constructive_rate"] for window in windows), default=0.0
    )
    persistence_floor = min(
        1.0,
        max(1.0 / max(1, late["episode_count"]),
            threshold.minimum_late_to_peak_ratio * peak_constructive),
    )
    evidence = (
        recent["strict_count"] > 0
        or recent["valid_joint_count"] >= threshold.joint_count_last_1024
        or recent["constructive_count"] >= threshold.constructive_count_last_1024
    )
    persistent = (
        late["strict_count"] > 0
        or late["valid_joint_count"] > 0
        or late["constructive_rate"] >= persistence_floor
    )
    quality = (
        recent["median_flatness"] <= threshold.median_flatness_cap
        and recent["median_spread_error"] <= threshold.median_spread_error_cap
    )
    trend = (
        recent["mean_return"] >= first["mean_return"] - 0.15
        and recent["median_flatness"] <= first["median_flatness"] + 0.10
        and recent["median_spread_error"] <= first["median_spread_error"] + 0.05
    )
    promising = not integrity_errors and evidence and persistent and quality and trend
    joint_cost = (
        recent["median_flatness"] / 0.5
        + recent["median_spread_error"] / 0.1
    )
    return {
        "task": task,
        "profile": profile,
        "seed": seed,
        "endpoint": endpoint,
        "run_directory": str(run_dir.resolve()),
        "integrity_pass": not integrity_errors,
        "integrity_errors": integrity_errors,
        "config_validation": config,
        "first_1024": first,
        "last_1024": recent,
        "last_512": late,
        "windows_512": windows,
        "update_diagnostics_last_512": update_window_summary(
            update_rows, max(1, endpoint - 511), endpoint
        ),
        "update_diagnostics_last_1024": update_window_summary(
            update_rows, max(1, endpoint - 1023), endpoint
        ),
        "peak_constructive_rate_512": peak_constructive,
        "persistence_floor": persistence_floor,
        "promising_components": {
            "evidence": evidence,
            "persistence": persistent,
            "quality": quality,
            "trend": trend,
        },
        "promising_at_4096": promising,
        "selection_key": {
            "strict_hits": sum(row["strict"] for row in records),
            "last_1024_valid_joint_rate": recent["valid_joint_rate"],
            "last_1024_constructive_rate": recent["constructive_rate"],
            "last_1024_terminal_joint_cost": joint_cost,
            "last_1024_mean_return": recent["mean_return"],
        },
        "records": records,
    }


def catastrophic_1024(reports: Sequence[dict]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if any(not report["integrity_pass"] for report in reports):
        reasons.append("integrity failure")
    no_signal = all(
        report["last_512"]["strict_count"] == 0
        and report["last_512"]["valid_joint_count"] == 0
        and report["last_512"]["constructive_count"] == 0
        for report in reports
    )
    pooled_rejects = sum(
        report["last_512"]["mean_reject_count"]
        * report["last_512"]["episode_count"]
        for report in reports
    )
    pooled_steps = sum(
        report["last_512"]["mechanical_steps"] for report in reports
    )
    reject_lock = pooled_steps > 0 and pooled_rejects / pooled_steps >= 0.80
    invalid_lock = all(
        report["last_512"]["invalid_count"]
        >= 0.95 * report["last_512"]["episode_count"]
        for report in reports
    )
    extreme_return = all(
        report["last_512"]["mean_return"] <= -20.0 for report in reports
    )
    if no_signal and reject_lock:
        reasons.append("all-seed reject lock with no target-plane signal")
    if no_signal and invalid_lock:
        reasons.append("all-seed invalid-transition lock")
    if no_signal and extreme_return:
        reasons.append("all-seed extreme negative return with no signal")
    return bool(reasons), reasons


def _selection_tuple(report: dict) -> tuple:
    key = report["selection_key"]
    return (
        report["last_512"]["valid_joint_rate"],
        report["last_512"]["constructive_rate"],
        key["last_1024_valid_joint_rate"],
        key["last_1024_constructive_rate"],
        key["strict_hits"],
        -key["last_1024_terminal_joint_cost"],
        key["last_1024_mean_return"],
        -report["seed"],
    )


def task_decision(task: str, reports: Sequence[dict], endpoint: int) -> dict:
    decision = {
        "task": task,
        "checkpoint_episode": endpoint,
        "rule": "snapshot_only",
        "stop": False,
        "promote": False,
        "selected_seed": None,
        "reasons": [],
    }
    if endpoint == 1024:
        stop, reasons = catastrophic_1024(reports)
        decision.update({
            "rule": "catastrophic_stop_only_no_positive_adoption",
            "stop": stop,
            "reasons": reasons or ["no catastrophic condition; continue to 4096"],
        })
    elif endpoint == 4096:
        promising = [report for report in reports if report["promising_at_4096"]]
        promote = len(promising) >= 2
        selected = max(promising, key=_selection_tuple) if promote else None
        decision.update({
            "rule": "formal_2_of_3_persistent_target_plane_gate",
            "stop": not promote,
            "promote": promote,
            "selected_seed": selected["seed"] if selected else None,
            "promising_seed_count": len(promising),
            "promising_seeds": [report["seed"] for report in promising],
            "selection_order": [
                report["seed"]
                for report in sorted(promising, key=_selection_tuple, reverse=True)
            ],
            "reasons": (
                ["2/3 persistent evidence passed; continue selected-seed diagnostic"]
                if promote else
                ["fewer than 2/3 seeds passed the preregistered persistence gate"]
            ),
        })
    return decision


def production_warning(task: str, report: dict, baseline_4096: dict) -> dict:
    recent = report["last_1024"]
    baseline = baseline_4096["last_1024"]
    constructive_floor = max(
        2, int(math.ceil(0.20 * baseline["constructive_count"]))
    )
    no_signal = (
        recent["strict_count"] == 0
        and recent["valid_joint_count"] == 0
        and recent["constructive_count"] < constructive_floor
    )
    threshold = TASK_GATES[task]
    geometry_loss = (
        recent["median_flatness"] > max(
            threshold.median_flatness_cap,
            baseline["median_flatness"] + 0.20,
        )
        or recent["median_spread_error"] > max(
            threshold.median_spread_error_cap,
            baseline["median_spread_error"] + 0.08,
        )
    )
    return_loss = recent["mean_return"] < baseline["mean_return"] - 0.20
    if task == "water1":
        structural_wrong_basin = (
            recent["median_flatness"] <= 0.5
            and recent["median_spread_error"] >= 0.30
        )
    else:
        structural_wrong_basin = (
            recent["median_flatness"] >= 0.9
            and recent["median_spread_error"] >= 0.30
        )
    reject_spam = no_signal and recent["mean_reject_rate"] >= 0.75
    warning = (
        not report["integrity_pass"]
        or (no_signal and (
            geometry_loss or return_loss or structural_wrong_basin or reject_spam
        ))
    )
    return {
        "warning": warning,
        "no_signal": no_signal,
        "constructive_count_floor_from_4096": constructive_floor,
        "geometry_loss_from_4096": geometry_loss,
        "return_loss_from_4096": return_loss,
        "structural_wrong_basin": structural_wrong_basin,
        "reject_spam": reject_spam,
        "integrity_pass": report["integrity_pass"],
    }


def analyse_checkpoint(
    root: Path,
    endpoint: int,
    tasks: Sequence[str] = TASK_ORDER,
    seeds: Sequence[int] = SEEDS,
    selected_seeds: dict[str, int] | None = None,
    baseline_4096: dict[str, dict] | None = None,
    previous_warnings: dict[str, bool] | None = None,
) -> dict:
    selected_seeds = selected_seeds or {}
    previous_warnings = previous_warnings or {}
    payload = {
        "schema_version": 1,
        "analysis_type": "ori6_zero_reject_staged_checkpoint",
        "generated_utc": utc_now(),
        "experiment_root": str(root.resolve()),
        "checkpoint_episode": endpoint,
        "candidate": CANDIDATE,
        "tasks": {},
    }
    for task in tasks:
        task_seeds = (selected_seeds[task],) if task in selected_seeds else tuple(seeds)
        reports = [seed_report(root, task, seed, endpoint) for seed in task_seeds]
        decision = task_decision(task, reports, endpoint)
        if endpoint > 4096 and task in selected_seeds:
            if baseline_4096 is None or task not in baseline_4096:
                raise ValueError("production analysis requires the 4096 baseline")
            warning = production_warning(task, reports[0], baseline_4096[task])
            sustained = warning["warning"] and previous_warnings.get(task, False)
            decision = {
                "task": task,
                "checkpoint_episode": endpoint,
                "rule": "selected_seed_two_consecutive_checkpoint_collapse_gate",
                "stop": sustained or not reports[0]["integrity_pass"],
                "promote": True,
                "selected_seed": selected_seeds[task],
                "collapse_warning": warning,
                "previous_checkpoint_warning": previous_warnings.get(task, False),
                "sustained_collapse": sustained,
                "reasons": (
                    ["two consecutive collapse warnings"] if sustained else
                    ["single warning; require confirmation"] if warning["warning"] else
                    ["no collapse warning"]
                ),
            }
        payload["tasks"][task] = {"seeds": reports, "decision": decision}
    payload["overall_integrity_pass"] = all(
        report["integrity_pass"]
        for item in payload["tasks"].values() for report in item["seeds"]
    )
    return payload


def _serialisable(payload):
    if isinstance(payload, dict):
        return {key: _serialisable(value) for key, value in payload.items() if key != "records"}
    if isinstance(payload, list):
        return [_serialisable(value) for value in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def write_csv(payload: dict, path: Path) -> None:
    preferred_fields = [
        "task", "seed", "start_episode", "end_episode", "episode_count",
        "mean_return", "median_flatness", "median_spread_error",
        "constructive_count", "constructive_rate", "valid_joint_count",
        "path_joint_discovery_count", "strict_count", "near_miss_count", "wrong_flat_count",
        "mean_reject_count", "mean_reject_rate", "mechanical_steps",
        "max_abs_reward_decomposition_error", "nonfinite_component_count",
        "max_abs_zero_reject_ordinary_reward", "critic_loss_mean",
        "critic_loss_p95", "critic_loss_max", "critic_gradient_norm_mean",
        "critic_gradient_norm_p95", "critic_gradient_norm_max",
        "critic_gradient_over_clip_fraction", "actor_gradient_norm_mean",
        "actor_gradient_norm_p95", "actor_gradient_norm_max",
        "actor_gradient_over_clip_fraction", "entropy_mean",
        "actor_learning_rate_first", "actor_learning_rate_last",
        "actor_learning_rate_min", "actor_learning_rate_max",
    ]
    rows = []
    for task, item in payload["tasks"].items():
        for report in item["seeds"]:
            for window in report["windows_512"]:
                rows.append({"task": task, "seed": report["seed"], **window})
    # Window schemas grow as diagnostics are added.  Derive the complete union
    # instead of maintaining a second brittle hand-written schema.  Preferred
    # fields keep the stable human-facing order; new fields follow sorted.
    all_fields = {key for row in rows for key in row}
    fields = [field for field in preferred_fields if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    if not rows or not fields:
        raise RuntimeError("Checkpoint CSV cannot be written without window rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_plots(payload: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for task, item in payload["tasks"].items():
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for report in item["seeds"]:
            records = report["records"]
            episode = np.array([row["episode"] for row in records], dtype=float)
            label = f"seed {report['seed']}"
            series = {
                "reward": np.array([row["reward"] for row in records]),
                "flatness": np.array([row["flatness"] for row in records]),
                "spread_error": np.array([row["spread_error"] for row in records]),
                "constructive": np.array([row["constructive"] for row in records], dtype=float),
            }
            window = min(200, max(1, len(records)))
            kernel = np.ones(window) / window
            x = episode[window - 1:]
            for axis, name in zip(axes.flat, ("reward", "flatness", "spread_error", "constructive")):
                axis.plot(x, np.convolve(series[name], kernel, mode="valid"), label=label, linewidth=1.2)
        axes[0, 0].set_ylabel("episodic return (MA200)")
        axes[0, 1].set_ylabel("terminal flatness F (MA200)")
        axes[1, 0].set_ylabel("terminal spread error (MA200)")
        axes[1, 1].set_ylabel("constructive rate (MA200)")
        axes[0, 1].set_yscale("log")
        axes[1, 0].axhline(0.1, color="black", linewidth=0.8, linestyle="--")
        for axis in axes.flat:
            axis.set_xlabel("episode")
            axis.grid(alpha=0.22)
        axes[0, 0].legend(loc="best", fontsize=8)
        figure.suptitle(f"Zero-reject staged learning curve: {task}")
        path = output / f"learning_curve_{task}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(str(path.resolve()))
    return written


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Zero-reject staged experiment checkpoint report",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Atomic endpoint: {payload['checkpoint_episode']} episodes",
        f"- Integrity: {'PASS' if payload['overall_integrity_pass'] else 'FAIL'}",
        "- Curves use moving episodic statistics; cumulative reward sums are not used.",
        "",
    ]
    for task, item in payload["tasks"].items():
        decision = item["decision"]
        lines.extend([
            f"## {task}", "",
            f"Decision: `{decision['rule']}`; stop={decision['stop']}; promote={decision['promote']}.",
            "",
            "| seed | return (last1024) | F median | spread error median | constructive | terminal joint | strict | critic loss mean/p95/max (last512) | critic grad mean/p95/max; >1 | actor grad mean/p95/max | entropy | actor LR | promising |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for report in item["seeds"]:
            recent = report["last_1024"]
            update = report["update_diagnostics_last_512"]
            lines.append(
                f"| {report['seed']} | {recent['mean_return']:.4g} | "
                f"{recent['median_flatness']:.4g} | {recent['median_spread_error']:.4g} | "
                f"{recent['constructive_count']} | {recent['valid_joint_count']} | "
                f"{recent['strict_count']} | "
                f"{update['critic_loss_mean']:.3g}/{update['critic_loss_p95']:.3g}/{update['critic_loss_max']:.3g} | "
                f"{update['critic_gradient_norm_mean']:.3g}/{update['critic_gradient_norm_p95']:.3g}/{update['critic_gradient_norm_max']:.3g}; {update['critic_gradient_over_clip_fraction']:.1%} | "
                f"{update['actor_gradient_norm_mean']:.3g}/{update['actor_gradient_norm_p95']:.3g}/{update['actor_gradient_norm_max']:.3g} | "
                f"{update['entropy_mean']:.3g} | {update['actor_learning_rate_last']:.3g} | "
                f"{report['promising_at_4096']} |"
            )
        lines.extend(["", "Reasons: " + "; ".join(decision["reasons"]), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_artifacts(payload: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "checkpoint_analysis.json"
    csv_path = output / "windows_512.csv"
    report_path = output / "report.md"
    atomic_json(json_path, _serialisable(payload))
    write_csv(payload, csv_path)
    plots = write_plots(payload, output)
    write_report(payload, report_path)
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "report": str(report_path.resolve()),
        "plots": plots,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--endpoint", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=list(TASK_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    payload = analyse_checkpoint(
        options.experiment_root.resolve(), options.endpoint,
        options.tasks, options.seeds,
    )
    output = options.output or (
        options.experiment_root / "analysis" / f"checkpoint_{options.endpoint}"
    )
    artifacts = write_artifacts(payload, output.resolve())
    print(json.dumps({"analysis": _serialisable(payload), "artifacts": artifacts}, ensure_ascii=False, indent=2))
    return 0 if payload["overall_integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
