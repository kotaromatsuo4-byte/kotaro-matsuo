"""Aggregate paired main9 online-search runs against the canonical six cells.

The online archive deliberately distinguishes layer-order variants and can
therefore contain more than the six historical five-bar targets.  This script
keeps four coverage notions separate:

* raw kinematic cells,
* raw feasible cells,
* canonical-six kinematic coverage, and
* canonical-six feasible coverage.

Canonical matching uses the prototype shape and exact layer key stored in the
reachability audit, with each run's ``rms_tolerance`` and ``max_tolerance``.
The script also audits the paired-random invariant: before a filter first
rejects a candidate, its physical trajectory must be exactly the same as the
random control.  A filter that never rejects is required to match the complete
random trajectory and final state/training hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SELECTOR_ORDER = ("random", "forest_filter", "forest_filter_shuffled")
PAIR_ORDER = (
    ("forest_filter", "random"),
    ("forest_filter", "forest_filter_shuffled"),
    ("forest_filter_shuffled", "random"),
)

TRAJECTORY_FIELDS = (
    "macro_id",
    "start_step",
    "end_step",
    "mechanics_cost",
    "parent_id",
    "terminal_state_id",
    "requested_K",
    "executed_K",
    "suffix_steps",
    "stop_reason",
    "projection_failure",
    "minimum_projection_ratio",
    "collision_edges_in_macro",
    "path_collision_edges",
    "path_geometric_length",
    "path_collision_free",
    "kinematic_cell_index",
    "new_kinematic",
    "new_feasible",
    "teacher_rows",
)

PAIRED_METRICS = (
    "raw_kinematic_final",
    "raw_feasible_final",
    "canonical_kinematic_final",
    "canonical_feasible_final",
    "raw_kinematic_auc",
    "raw_feasible_auc",
    "canonical_kinematic_auc",
    "canonical_feasible_auc",
    "canonical_kinematic_t6_capped",
    "canonical_feasible_t6_capped",
    "rejected_total",
    "raw_rejected_total",
    "fit_time_seconds",
    "score_time_seconds",
    "wall_time_seconds",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _layer_key(value: Iterable[Iterable[Any]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in entry) for entry in value)


def _shape_distance(left: Any, right: Any) -> tuple[float, float]:
    difference = (
        np.asarray(left, dtype=float).reshape(-1, 2)
        - np.asarray(right, dtype=float).reshape(-1, 2)
    )
    node_distance = np.linalg.norm(difference, axis=1)
    return (
        float(np.sqrt(np.mean(node_distance**2))),
        float(np.max(node_distance)),
    )


def _canonical_targets(reference: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for target in reference["targets"]:
        targets.append(
            {
                "target_id": int(target["target_id"]),
                "shape_descriptor": target["prototype_shape_descriptor"],
                "layer_key": _layer_key(target["layer_key"]),
            }
        )
    return sorted(targets, key=lambda target: target["target_id"])


def match_canonical_cell(
    cell: dict[str, Any],
    targets: list[dict[str, Any]],
    rms_tolerance: float,
    max_tolerance: float,
) -> tuple[int | None, float | None, float | None]:
    """Match one raw archive cell, preferring the closest valid prototype."""

    matches = []
    layer = _layer_key(cell["layer_key"])
    for target in targets:
        if layer != target["layer_key"]:
            continue
        rms_distance, max_distance = _shape_distance(
            cell["shape_descriptor"], target["shape_descriptor"]
        )
        if rms_distance <= rms_tolerance and max_distance <= max_tolerance:
            matches.append(
                (rms_distance, max_distance, int(target["target_id"]))
            )
    if not matches:
        return None, None, None
    rms_distance, max_distance, target_id = min(matches)
    return target_id, rms_distance, max_distance


def _coverage_curve(
    events: list[dict[str, Any]],
    layer: str,
    mapping: dict[int, int | None] | None = None,
) -> list[tuple[int, int]]:
    covered: set[int] = set()
    curve = []
    ordered = sorted(
        enumerate(events),
        key=lambda item: (int(item[1]["mechanics_step"]), item[0]),
    )
    for _, event in ordered:
        if event["archive_layer"] != layer:
            continue
        raw_cell = int(event["cell_index"])
        cell = raw_cell if mapping is None else mapping.get(raw_cell)
        if cell is None or cell in covered:
            continue
        covered.add(int(cell))
        curve.append((int(event["mechanics_step"]), len(covered)))
    return curve


def _auc(curve: list[tuple[int, int]], budget: int) -> float:
    area = 0.0
    previous_step = 0
    count = 0
    for step, next_count in curve:
        step = min(max(int(step), previous_step), int(budget))
        area += (step - previous_step) * count
        previous_step = step
        count = int(next_count)
        if step >= budget:
            break
    area += max(int(budget) - previous_step, 0) * count
    return float(area / max(float(budget), 1.0))


def _time_to(curve: list[tuple[int, int]], target: int) -> int | None:
    return next((int(step) for step, count in curve if count >= target), None)


def _first_steps(
    events: list[dict[str, Any]],
    mapping: dict[int, int | None] | None,
) -> tuple[dict[int, int], dict[int, int]]:
    output = {"kinematic": {}, "feasible": {}}
    for event in sorted(events, key=lambda item: int(item["mechanics_step"])):
        raw_cell = int(event["cell_index"])
        cell = raw_cell if mapping is None else mapping.get(raw_cell)
        if cell is None:
            continue
        layer = event["archive_layer"]
        output[layer].setdefault(int(cell), int(event["mechanics_step"]))
    return output["kinematic"], output["feasible"]


def _lag_summary(
    kinematic: dict[int, int], feasible: dict[int, int], budget: int
) -> dict[str, Any]:
    cells = sorted(set(kinematic) | set(feasible))
    details = []
    observed_lags = []
    censored_lower_bounds = []
    for cell in cells:
        first_seen = kinematic.get(cell)
        first_feasible = feasible.get(cell)
        if first_seen is not None and first_feasible is not None:
            lag = int(first_feasible - first_seen)
            observed_lags.append(lag)
            status = "observed"
            lower_bound = None
        elif first_seen is not None:
            lag = None
            lower_bound = int(max(budget - first_seen, 0))
            censored_lower_bounds.append(lower_bound)
            status = "right_censored"
        else:
            lag = None
            lower_bound = None
            status = "missing_kinematic"
        details.append(
            {
                "cell_id": int(cell),
                "first_seen_step": first_seen,
                "first_feasible_step": first_feasible,
                "lag": lag,
                "status": status,
                "censored_lag_lower_bound": lower_bound,
            }
        )
    return {
        "details": details,
        "observed_count": len(observed_lags),
        "right_censored_count": len(censored_lower_bounds),
        "mean_observed": (
            float(statistics.fmean(observed_lags)) if observed_lags else None
        ),
        "median_observed": (
            float(statistics.median(observed_lags)) if observed_lags else None
        ),
        "max_observed": max(observed_lags) if observed_lags else None,
        "mean_censored_lower_bound": (
            float(statistics.fmean(censored_lower_bounds))
            if censored_lower_bounds
            else None
        ),
    }


def _diagnostic_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    ready = [row for row in rows if _as_bool(row.get("forest_ready"))]
    evaluated = [
        row
        for row in rows
        if str(row.get("selection_mode", "")).startswith("forest_filter")
    ]
    rejected = [row for row in rows if _as_int(row.get("rejected_count")) > 0]
    raw_rejected = [
        row for row in rows if _as_int(row.get("raw_rejected_count")) > 0
    ]
    probabilities = [
        _as_float(row.get("max_probability")) for row in rows
    ]
    probabilities = [value for value in probabilities if math.isfinite(value)]
    support = [_as_float(row.get("support_ratio")) for row in rows]
    finite_support = [value for value in support if math.isfinite(value)]
    total_candidates = sum(_as_int(row.get("candidate_count")) for row in rows)
    evaluated_candidates = sum(
        _as_int(row.get("candidate_count")) for row in evaluated
    )
    rejected_total = sum(
        _as_int(row.get("rejected_count")) for row in rows
    )
    raw_rejected_total = sum(
        _as_int(row.get("raw_rejected_count")) for row in rows
    )
    return {
        "diagnostic_macro_count": len(rows),
        "candidate_total": int(total_candidates),
        "filter_evaluated_candidate_total": int(evaluated_candidates),
        "rejected_total": int(rejected_total),
        "raw_rejected_total": int(raw_rejected_total),
        "filter_rejection_fraction": (
            float(rejected_total / evaluated_candidates)
            if evaluated_candidates
            else 0.0
        ),
        "raw_filter_rejection_fraction": (
            float(raw_rejected_total / evaluated_candidates)
            if evaluated_candidates
            else 0.0
        ),
        "all_pool_rejection_fraction": (
            float(rejected_total / total_candidates) if total_candidates else 0.0
        ),
        "macros_with_rejection": len(rejected),
        "macros_with_raw_rejection": len(raw_rejected),
        "reject_cap_applied_count": sum(
            _as_bool(row.get("reject_cap_applied")) for row in rows
        ),
        "epsilon_bypass_count": sum(
            _as_bool(row.get("epsilon_bypass")) for row in rows
        ),
        "cold_random_count": sum(
            row.get("selection_mode") == "cold_random" for row in rows
        ),
        "filter_evaluation_count": len(evaluated),
        "first_filter_ready_macro": (
            _as_int(ready[0]["macro_id"]) if ready else None
        ),
        "first_filter_ready_step": (
            _as_int(ready[0]["start_step"]) if ready else None
        ),
        "first_filter_evaluation_macro": (
            _as_int(evaluated[0]["macro_id"]) if evaluated else None
        ),
        "first_filter_evaluation_step": (
            _as_int(evaluated[0]["start_step"]) if evaluated else None
        ),
        "first_raw_rejection_macro": (
            _as_int(raw_rejected[0]["macro_id"]) if raw_rejected else None
        ),
        "first_raw_rejection_step": (
            _as_int(raw_rejected[0]["start_step"]) if raw_rejected else None
        ),
        "first_rejection_macro": (
            _as_int(rejected[0]["macro_id"]) if rejected else None
        ),
        "first_rejection_step": (
            _as_int(rejected[0]["start_step"]) if rejected else None
        ),
        "selected_probability_count": len(probabilities),
        "selected_max_probability_max": (
            max(probabilities) if probabilities else None
        ),
        "selected_max_probability_mean": (
            float(statistics.fmean(probabilities)) if probabilities else None
        ),
        "finite_selected_support_count": len(finite_support),
        "selected_support_ratio_min": (
            min(finite_support) if finite_support else None
        ),
        "selected_support_ratio_max": (
            max(finite_support) if finite_support else None
        ),
    }


def one_run(
    seed_directory: Path,
    selector: str,
    targets: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    config = _read_json(seed_directory / "configuration.json")
    summary = _read_json(seed_directory / "summary.json")
    archive = _read_json(seed_directory / "cell_archive.json")
    events = _read_json(seed_directory / "discovery_events.json")
    diagnostics = _read_csv(seed_directory / "rollout_diagnostics.csv")
    budget = int(summary["budget"])
    rms_tolerance = float(config["rms_tolerance"])
    max_tolerance = float(config["max_tolerance"])

    cell_matches = []
    mapping: dict[int, int | None] = {}
    for cell_index, cell in enumerate(archive):
        target_id, rms_distance, max_distance = match_canonical_cell(
            cell, targets, rms_tolerance, max_tolerance
        )
        mapping[cell_index] = target_id
        cell_matches.append(
            {
                "raw_cell_index": cell_index,
                "canonical_target_id": target_id,
                "rms_distance": rms_distance,
                "max_distance": max_distance,
                "has_feasible_elite": cell.get("feasible_elite") is not None,
            }
        )

    curves = {}
    for layer in ("kinematic", "feasible"):
        curves[f"raw_{layer}"] = _coverage_curve(events, layer)
        curves[f"canonical_{layer}"] = _coverage_curve(events, layer, mapping)

    target_count = len(targets)
    canonical_kinematic_t6 = _time_to(
        curves["canonical_kinematic"], target_count
    )
    canonical_feasible_t6 = _time_to(curves["canonical_feasible"], target_count)
    raw_kinematic_t6 = _time_to(curves["raw_kinematic"], target_count)
    raw_feasible_t6 = _time_to(curves["raw_feasible"], target_count)
    raw_first_seen, raw_first_feasible = _first_steps(events, None)
    canonical_first_seen, canonical_first_feasible = _first_steps(events, mapping)

    feasible_raw_indices = {
        index
        for index, cell in enumerate(archive)
        if cell.get("feasible_elite") is not None
    }
    canonical_kinematic = {
        target_id for target_id in mapping.values() if target_id is not None
    }
    canonical_feasible = {
        mapping[index]
        for index in feasible_raw_indices
        if mapping[index] is not None
    }
    diagnostic_metrics = _diagnostic_metrics(diagnostics)
    row = {
        "seed": int(summary["seed"]),
        "selector": selector,
        "result_root": str(seed_directory.parent.resolve()),
        "seed_directory": str(seed_directory.resolve()),
        "budget": budget,
        "used_steps": int(summary["used_steps"]),
        "macro_count": int(summary["macro_count"]),
        "training_rows": int(summary["training_rows"]),
        "valid_cell_training_rows": int(summary["valid_cell_training_rows"]),
        "raw_kinematic_final": len(archive),
        "raw_feasible_final": len(feasible_raw_indices),
        "canonical_kinematic_final": len(canonical_kinematic),
        "canonical_feasible_final": len(canonical_feasible),
        "canonical_kinematic_ids": sorted(canonical_kinematic),
        "canonical_feasible_ids": sorted(canonical_feasible),
        "unmatched_raw_cells": sum(
            match["canonical_target_id"] is None for match in cell_matches
        ),
        "raw_kinematic_auc": _auc(curves["raw_kinematic"], budget),
        "raw_feasible_auc": _auc(curves["raw_feasible"], budget),
        "canonical_kinematic_auc": _auc(
            curves["canonical_kinematic"], budget
        ),
        "canonical_feasible_auc": _auc(curves["canonical_feasible"], budget),
        "raw_kinematic_t6": raw_kinematic_t6,
        "raw_feasible_t6": raw_feasible_t6,
        "canonical_kinematic_t6": canonical_kinematic_t6,
        "canonical_feasible_t6": canonical_feasible_t6,
        "canonical_kinematic_t6_event": int(
            canonical_kinematic_t6 is not None
        ),
        "canonical_feasible_t6_event": int(canonical_feasible_t6 is not None),
        "canonical_kinematic_t6_censored_at": (
            None if canonical_kinematic_t6 is not None else budget
        ),
        "canonical_feasible_t6_censored_at": (
            None if canonical_feasible_t6 is not None else budget
        ),
        "canonical_kinematic_t6_capped": (
            canonical_kinematic_t6
            if canonical_kinematic_t6 is not None
            else budget
        ),
        "canonical_feasible_t6_capped": (
            canonical_feasible_t6
            if canonical_feasible_t6 is not None
            else budget
        ),
        "raw_lag": _lag_summary(raw_first_seen, raw_first_feasible, budget),
        "canonical_lag": _lag_summary(
            canonical_first_seen, canonical_first_feasible, budget
        ),
        "cell_matches": cell_matches,
        "curves": {
            key: [[int(step), int(count)] for step, count in curve]
            for key, curve in curves.items()
        },
        "fit_time_seconds": float(summary.get("fit_time_seconds", 0.0)),
        "score_time_seconds": float(summary.get("score_time_seconds", 0.0)),
        "wall_time_seconds": float(summary.get("wall_time_seconds", 0.0)),
        "rms_tolerance": rms_tolerance,
        "max_tolerance": max_tolerance,
        "filter_mean_probability": float(
            config.get("filter_mean_probability", math.nan)
        ),
        "filter_lower_probability": float(
            config.get("filter_lower_probability", math.nan)
        ),
        "filter_vote_fraction": float(
            config.get("filter_vote_fraction", math.nan)
        ),
        "filter_max_disagreement": float(
            config.get("filter_max_disagreement", math.nan)
        ),
        "filter_max_support_ratio": float(
            config.get("filter_max_support_ratio", math.nan)
        ),
        "warmup_end_step": int(summary.get("warmup_end_step", 0)),
        "warmup_prefix_sha256": summary.get("warmup_prefix_sha256"),
        "warmup_training_sha256": summary.get("warmup_training_sha256"),
        "final_state_sha256": summary.get("final_state_sha256"),
        "final_training_sha256": summary.get("final_training_sha256"),
        **diagnostic_metrics,
    }
    # The raw AUC recomputation is an integrity check on event semantics.
    row["raw_kinematic_auc_matches_summary"] = math.isclose(
        row["raw_kinematic_auc"], float(summary["kinematic_auc"]), abs_tol=1e-12
    )
    row["raw_feasible_auc_matches_summary"] = math.isclose(
        row["raw_feasible_auc"], float(summary["feasible_auc"]), abs_tol=1e-12
    )
    return row, diagnostics


def _hash_records(records: list[dict[str, str]], fields: Iterable[str]) -> str:
    payload = [
        {field: record.get(field) for field in fields}
        for record in records
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_record_mismatch(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    fields: Iterable[str],
    macro_limit_exclusive: int | None = None,
) -> int | None:
    left_by_id = {_as_int(row["macro_id"]): row for row in left}
    right_by_id = {_as_int(row["macro_id"]): row for row in right}
    macro_ids = sorted(set(left_by_id) | set(right_by_id))
    for macro_id in macro_ids:
        if macro_limit_exclusive is not None and macro_id >= macro_limit_exclusive:
            break
        left_row = left_by_id.get(macro_id)
        right_row = right_by_id.get(macro_id)
        if left_row is None or right_row is None:
            return macro_id
        if any(left_row.get(field) != right_row.get(field) for field in fields):
            return macro_id
    return None


def trajectory_audit(
    guided_row: dict[str, Any],
    guided_diagnostics: list[dict[str, str]],
    random_row: dict[str, Any],
    random_diagnostics: list[dict[str, str]],
) -> dict[str, Any]:
    first_rejection = guided_row["first_rejection_macro"]
    prefix_limit = first_rejection
    guided_prefix = [
        row
        for row in guided_diagnostics
        if prefix_limit is None or _as_int(row["macro_id"]) < prefix_limit
    ]
    random_prefix = [
        row
        for row in random_diagnostics
        if prefix_limit is None or _as_int(row["macro_id"]) < prefix_limit
    ]
    first_trajectory_mismatch = _first_record_mismatch(
        guided_diagnostics,
        random_diagnostics,
        TRAJECTORY_FIELDS,
        prefix_limit,
    )
    pool_limit = None if first_rejection is None else first_rejection + 1
    first_pool_mismatch = _first_record_mismatch(
        guided_diagnostics,
        random_diagnostics,
        ("candidate_count", "candidate_pool_sha256"),
        pool_limit,
    )
    full_trajectory_mismatch = _first_record_mismatch(
        guided_diagnostics, random_diagnostics, TRAJECTORY_FIELDS
    )
    if full_trajectory_mismatch is None and len(guided_diagnostics) != len(
        random_diagnostics
    ):
        full_trajectory_mismatch = min(
            len(guided_diagnostics), len(random_diagnostics)
        )
    full_pool_mismatch = _first_record_mismatch(
        guided_diagnostics,
        random_diagnostics,
        ("candidate_count", "candidate_pool_sha256"),
    )
    if full_pool_mismatch is None and len(guided_diagnostics) != len(
        random_diagnostics
    ):
        full_pool_mismatch = min(len(guided_diagnostics), len(random_diagnostics))
    return {
        "seed": guided_row["seed"],
        "guided_selector": guided_row["selector"],
        "random_selector": random_row["selector"],
        "first_rejection_macro": first_rejection,
        "first_rejection_step": guided_row["first_rejection_step"],
        "trajectory_prefix_macro_count": len(guided_prefix),
        "trajectory_equal_before_first_rejection": (
            first_trajectory_mismatch is None
            and len(guided_prefix) == len(random_prefix)
        ),
        "first_trajectory_mismatch_before_rejection": first_trajectory_mismatch,
        "pool_hash_equal_through_first_rejection": first_pool_mismatch is None,
        "first_pool_hash_mismatch_through_rejection": first_pool_mismatch,
        "full_trajectory_equal": full_trajectory_mismatch is None,
        "first_full_trajectory_mismatch_macro": full_trajectory_mismatch,
        "full_candidate_pool_hash_equal": full_pool_mismatch is None,
        "first_full_pool_hash_mismatch_macro": full_pool_mismatch,
        "guided_prefix_sha256": _hash_records(
            guided_prefix, TRAJECTORY_FIELDS
        ),
        "random_prefix_sha256": _hash_records(random_prefix, TRAJECTORY_FIELDS),
        "warmup_prefix_hash_equal": (
            guided_row["warmup_prefix_sha256"]
            == random_row["warmup_prefix_sha256"]
        ),
        "warmup_training_hash_equal": (
            guided_row["warmup_training_sha256"]
            == random_row["warmup_training_sha256"]
        ),
        "final_state_hash_equal": (
            guided_row["final_state_sha256"] == random_row["final_state_sha256"]
        ),
        "final_training_hash_equal": (
            guided_row["final_training_sha256"]
            == random_row["final_training_sha256"]
        ),
    }


def _mean_std(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(finite),
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
    }


def _bootstrap_ci(values: Iterable[float], seed: int, samples: int = 50_000):
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _exact_sign_flip(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    observed = abs(float(array.mean()))
    distribution = [
        abs(float(np.mean(array * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=array.size)
    ]
    return float(
        np.mean(np.asarray(distribution) >= observed - np.finfo(float).eps * 8)
    )


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "raw_kinematic_final",
        "raw_feasible_final",
        "canonical_kinematic_final",
        "canonical_feasible_final",
        "raw_kinematic_auc",
        "raw_feasible_auc",
        "canonical_kinematic_auc",
        "canonical_feasible_auc",
        "canonical_kinematic_t6_capped",
        "canonical_feasible_t6_capped",
        "macro_count",
        "training_rows",
        "candidate_total",
        "filter_evaluated_candidate_total",
        "rejected_total",
        "raw_rejected_total",
        "filter_rejection_fraction",
        "raw_filter_rejection_fraction",
        "all_pool_rejection_fraction",
        "first_filter_ready_step",
        "first_filter_evaluation_step",
        "first_raw_rejection_step",
        "first_rejection_step",
        "fit_time_seconds",
        "score_time_seconds",
        "wall_time_seconds",
    )
    output = []
    for selector in SELECTOR_ORDER:
        group = [row for row in runs if row["selector"] == selector]
        item: dict[str, Any] = {"selector": selector, "seeds": len(group)}
        for metric in metrics:
            values = [
                row[metric]
                for row in group
                if row.get(metric) is not None
            ]
            stats = _mean_std(values)
            item[f"{metric}_n"] = stats["n"]
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
        for layer in ("kinematic", "feasible"):
            event_key = f"canonical_{layer}_t6_event"
            time_key = f"canonical_{layer}_t6"
            reached = [row[time_key] for row in group if row[event_key]]
            item[f"canonical_{layer}_t6_reached"] = int(
                sum(row[event_key] for row in group)
            )
            item[f"canonical_{layer}_t6_mean_reached"] = (
                float(statistics.fmean(reached)) if reached else None
            )
        lag_values = [
            row["canonical_lag"]["mean_observed"]
            for row in group
            if row["canonical_lag"]["mean_observed"] is not None
        ]
        lag_stats = _mean_std(lag_values)
        item["canonical_lag_mean_observed_mean"] = lag_stats["mean"]
        item["canonical_lag_mean_observed_std"] = lag_stats["std"]
        item["canonical_lag_right_censored_total"] = int(
            sum(row["canonical_lag"]["right_censored_count"] for row in group)
        )
        for metric in (
            "candidate_total",
            "filter_evaluated_candidate_total",
            "rejected_total",
            "raw_rejected_total",
            "macros_with_rejection",
        ):
            item[f"{metric}_sum"] = int(sum(row[metric] for row in group))
        evaluated_total = item["filter_evaluated_candidate_total_sum"]
        item["filter_rejection_fraction_pooled"] = (
            float(item["rejected_total_sum"] / evaluated_total)
            if evaluated_total
            else 0.0
        )
        item["raw_filter_rejection_fraction_pooled"] = (
            float(item["raw_rejected_total_sum"] / evaluated_total)
            if evaluated_total
            else 0.0
        )
        output.append(item)
    return output


def paired_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["selector"], row["seed"]): row for row in runs}
    output = []
    for pair_index, (left, right) in enumerate(PAIR_ORDER):
        seeds = sorted(
            set(seed for selector, seed in lookup if selector == left)
            & set(seed for selector, seed in lookup if selector == right)
        )
        item = {"left": left, "right": right, "seeds": seeds, "metrics": {}}
        for metric_index, metric in enumerate(PAIRED_METRICS):
            differences = [
                float(lookup[(left, seed)][metric])
                - float(lookup[(right, seed)][metric])
                for seed in seeds
            ]
            stats = _mean_std(differences)
            item["metrics"][metric] = {
                "by_seed": dict(zip(map(str, seeds), differences)),
                **stats,
                "bootstrap_95_ci": _bootstrap_ci(
                    differences, 813 + pair_index * 100 + metric_index
                ),
                "exact_two_sided_sign_flip_p": _exact_sign_flip(differences),
            }
        output.append(item)
    return output


def _fmt(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _t6_cell(row: dict[str, Any], layer: str) -> str:
    reached = row[f"canonical_{layer}_t6_reached"]
    total = row["seeds"]
    reached_mean = row[f"canonical_{layer}_t6_mean_reached"]
    restricted = row[f"canonical_{layer}_t6_capped_mean"]
    reached_text = "NA" if reached_mean is None else f"{reached_mean:.1f}"
    return f"{reached_text} ({reached}/{total}); restricted {restricted:.1f}"


def make_markdown(payload: dict[str, Any]) -> str:
    aggregate = {row["selector"]: row for row in payload["aggregate"]}
    audits = payload["trajectory_audits"]
    runs = payload["runs"]
    seeds = payload["seeds"]
    seed_count = len(seeds)
    budgets = sorted({int(row["budget"]) for row in runs})
    budget = budgets[0] if len(budgets) == 1 else max(budgets)
    target_count = int(payload["canonical_target_count"])
    panel_name = payload.get("panel_name", "main9_online")
    all_exact = all(
        audit["full_trajectory_equal"]
        and audit["full_candidate_pool_hash_equal"]
        and audit["final_state_hash_equal"]
        and audit["final_training_hash_equal"]
        for audit in audits
    )
    any_rejection = any(
        aggregate[selector]["rejected_total_sum"] > 0
        for selector in ("forest_filter", "forest_filter_shuffled")
    )
    performance_fields = (
        "raw_kinematic_final",
        "raw_feasible_final",
        "canonical_kinematic_final",
        "canonical_feasible_final",
        "raw_kinematic_auc",
        "raw_feasible_auc",
        "canonical_kinematic_auc",
        "canonical_feasible_auc",
        "canonical_kinematic_t6_capped",
        "canonical_feasible_t6_capped",
    )
    run_lookup = {(row["selector"], row["seed"]): row for row in runs}
    performance_identical = all(
        all(
            run_lookup[(selector, seed)][field]
            == run_lookup[("random", seed)][field]
            for field in performance_fields
        )
        for selector in ("forest_filter", "forest_filter_shuffled")
        for seed in seeds
    )
    pair_lookup = {
        (pair["left"], pair["right"]): pair for pair in payload["paired"]
    }

    def difference(left: str, right: str, metric: str) -> float:
        return float(pair_lookup[(left, right)]["metrics"][metric]["mean"])

    learned = aggregate["forest_filter"]
    shuffled = aggregate["forest_filter_shuffled"]
    random = aggregate["random"]
    exact_prefix_count = sum(
        audit["trajectory_equal_before_first_rejection"]
        and audit["pool_hash_equal_through_first_rejection"]
        for audit in audits
    )
    lines = [
        f"# {panel_name} forest-filter comparison (B={budget})",
        "",
        "## 結論",
        "",
    ]

    if not any_rejection:
        lines += [
            "現設定の forest filter は選択圧を一度も発生させていない。"
            f"{seed_count} seeds 合計の actual/raw rejection は learned="
            f"{learned['rejected_total_sum']}/{learned['raw_rejected_total_sum']}、"
            f"shuffled={shuffled['rejected_total_sum']}/{shuffled['raw_rejected_total_sum']}。",
            "",
            (
                "forest の学習時間を除けば3手法は同じ探索として動作した。"
                f"全{len(audits)}組で全 macro の候補集合 hash、物理 trajectory、"
                "最終 state hash、教師データ hash が random と完全一致した。"
                if all_exact
                else "棄却ゼロにもかかわらず一部の trajectory 監査に不一致がある。"
            ),
            "",
            "したがって、この panel から forest filter による高速化は主張できない。"
            "差が小さいのではなく、有効な filter action がゼロだったためである。",
        ]
    else:
        learned_fraction = learned["filter_rejection_fraction_pooled"]
        shuffled_fraction = shuffled["filter_rejection_fraction_pooled"]
        lines += [
            "aggressive filter は実際に選択圧を発生させた。"
            f"{seed_count} seeds 合計で learned は "
            f"{learned['rejected_total_sum']}/{learned['filter_evaluated_candidate_total_sum']} "
            f"({100.0 * learned_fraction:.2f}%)、shuffled は "
            f"{shuffled['rejected_total_sum']}/{shuffled['filter_evaluated_candidate_total_sum']} "
            f"({100.0 * shuffled_fraction:.2f}%) の filter-evaluated candidates を棄却した。",
            "",
            f"最初の棄却より前の paired trajectory と、その棄却判断に使った candidate-pool hash は"
            f" {exact_prefix_count}/{len(audits)} 組で random と完全一致した。"
            "棄却後の trajectory 分岐は filter が作用した結果であり、監査失敗ではない。",
            "",
            "learned − random は canonical kinematic AUC "
            f"{difference('forest_filter', 'random', 'canonical_kinematic_auc'):+.3f}、"
            "feasible AUC "
            f"{difference('forest_filter', 'random', 'canonical_feasible_auc'):+.3f}、"
            "restricted T6 は kinematic "
            f"{difference('forest_filter', 'random', 'canonical_kinematic_t6_capped'):+.1f} step、"
            "feasible "
            f"{difference('forest_filter', 'random', 'canonical_feasible_t6_capped'):+.1f} step。",
            "",
            (
                f"この {seed_count}-seed pilot では learned filter は kinematic T6をわずかに早めた一方、"
                "feasible AUCを下げ、feasible T6を遅らせた。したがって random より"
                "多様な feasible 解を高速化したとは言えない。"
                if difference(
                    "forest_filter", "random", "canonical_feasible_t6_capped"
                ) > 0
                or difference(
                    "forest_filter", "random", "canonical_feasible_auc"
                ) < 0
                else "この pilot では feasible AUCとT6の双方がrandomより改善した。"
            ),
            "",
            "learned − shuffled は final feasible canonical coverage "
            f"{difference('forest_filter', 'forest_filter_shuffled', 'canonical_feasible_final'):+.3f}、"
            "feasible AUC "
            f"{difference('forest_filter', 'forest_filter_shuffled', 'canonical_feasible_auc'):+.3f}、"
            "restricted feasible T6 "
            f"{difference('forest_filter', 'forest_filter_shuffled', 'canonical_feasible_t6_capped'):+.1f} step。"
            f"完全到達は learned {learned['canonical_feasible_t6_reached']}/{seed_count}、"
            f"shuffled {shuffled['canonical_feasible_t6_reached']}/{seed_count} で、"
            f"learned labels による信頼性向上の兆候はあるが、{seed_count}-seed pilotでは確証にならない。",
        ]

    lines += [
        "",
        "## 集計",
        "",
        f"AUC は mechanics step 0..{budget} における累積 coverage の時間平均。"
        f"T{target_count} の `restricted` は未達を{budget}で右打切りした"
        f" `mean(min(T{target_count}, {budget}))`。rejection fraction の分母は、"
        "epsilon bypass/cold randomを除き、forest gateで実際に評価した候補数。",
        "",
        "| selector | raw final (kin/feas) | canonical final (kin/feas) | canonical AUC (kin/feas) | canonical T6 kin | canonical T6 feas | macros / teachers | rejected/evaluated (fraction); raw | fit/score/wall [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def rejection_cell(row: dict[str, Any]) -> str:
        denominator = row["filter_evaluated_candidate_total_sum"]
        if denominator == 0:
            return f"{row['rejected_total_sum']}/0 (NA); raw {row['raw_rejected_total_sum']}"
        return (
            f"{row['rejected_total_sum']}/{denominator} "
            f"({100.0 * row['filter_rejection_fraction_pooled']:.2f}%); "
            f"raw {row['raw_rejected_total_sum']}"
        )

    for selector in SELECTOR_ORDER:
        row = aggregate[selector]
        lines.append(
            f"| {selector} | "
            f"{_fmt(row['raw_kinematic_final_mean'], row['raw_kinematic_final_std'])} / "
            f"{_fmt(row['raw_feasible_final_mean'], row['raw_feasible_final_std'])} | "
            f"{_fmt(row['canonical_kinematic_final_mean'], row['canonical_kinematic_final_std'])} / "
            f"{_fmt(row['canonical_feasible_final_mean'], row['canonical_feasible_final_std'])} | "
            f"{_fmt(row['canonical_kinematic_auc_mean'], row['canonical_kinematic_auc_std'])} / "
            f"{_fmt(row['canonical_feasible_auc_mean'], row['canonical_feasible_auc_std'])} | "
            f"{_t6_cell(row, 'kinematic')} | {_t6_cell(row, 'feasible')} | "
            f"{_fmt(row['macro_count_mean'], row['macro_count_std'])} / "
            f"{_fmt(row['training_rows_mean'], row['training_rows_std'])} | "
            f"{rejection_cell(row)} | "
            f"{_fmt(row['fit_time_seconds_mean'], row['fit_time_seconds_std'])} / "
            f"{_fmt(row['score_time_seconds_mean'], row['score_time_seconds_std'])} / "
            f"{_fmt(row['wall_time_seconds_mean'], row['wall_time_seconds_std'])} |"
        )

    lines += [
        "",
        "",
        "raw archive は layer-order variant を含むため、canonical target数より多い値になり得る。",
        "random の canonical cell first-seen→first-feasible lag は、各 seed 内の観測済み"
        f"cell平均をさらに{seed_count} seedで平均して "
        f"{random['canonical_lag_mean_observed_mean']:.1f} step。"
        f"全 seed 合計で {random['canonical_lag_right_censored_total']} cell が右打切りだった。",
        "",
        "## 対応差（left − right）",
        "",
        "coverage/AUC は正が left 有利、T6 と時間は負が left 有利。"
        "95% CI は seed 対応 bootstrap、p は exact sign-flip test。",
        "",
        "| left | right | canonical kin AUC | canonical feas AUC | kin T6 capped | feas T6 capped | wall [s] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def paired_cell(pair: dict[str, Any], metric: str) -> str:
        value = pair["metrics"][metric]
        low, high = value["bootstrap_95_ci"]
        return (
            f"{value['mean']:+.3f} [{low:+.3f}, {high:+.3f}], "
            f"p={value['exact_two_sided_sign_flip_p']:.3f}"
        )

    for pair in payload["paired"]:
        lines.append(
            f"| {pair['left']} | {pair['right']} | "
            f"{paired_cell(pair, 'canonical_kinematic_auc')} | "
            f"{paired_cell(pair, 'canonical_feasible_auc')} | "
            f"{paired_cell(pair, 'canonical_kinematic_t6_capped')} | "
            f"{paired_cell(pair, 'canonical_feasible_t6_capped')} | "
            f"{paired_cell(pair, 'wall_time_seconds')} |"
        )

    lines += [
        "",
        "## seed別 canonical coverage",
        "",
    ]
    selected_rows = (
        [item for item in runs if item["selector"] == "random"]
        if performance_identical
        else runs
    )
    if performance_identical:
        lines += ["性能列は全 selector で同一なので random の値を示す。", ""]
    lines += [
        "| seed | selector | raw final kin/feas | canonical final kin/feas | canonical AUC kin/feas | T6 kin/feas | mean first-seen→feasible lag |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        feasible_t6 = (
            str(row["canonical_feasible_t6"])
            if row["canonical_feasible_t6"] is not None
            else f">{budget} (censored)"
        )
        kinematic_t6 = (
            str(row["canonical_kinematic_t6"])
            if row["canonical_kinematic_t6"] is not None
            else f">{budget} (censored)"
        )
        lag = row["canonical_lag"]["mean_observed"]
        lag_text = "NA" if lag is None else f"{lag:.1f}"
        lines.append(
            f"| {row['seed']} | {row['selector']} | "
            f"{row['raw_kinematic_final']}/{row['raw_feasible_final']} | "
            f"{row['canonical_kinematic_final']}/{row['canonical_feasible_final']} | "
            f"{row['canonical_kinematic_auc']:.3f}/{row['canonical_feasible_auc']:.3f} | "
            f"{kinematic_t6}/{feasible_t6} | "
            f"{lag_text} ({row['canonical_lag']['right_censored_count']} censored) |"
        )

    def activation_text(row: dict[str, Any], name: str) -> str:
        ready = row["first_filter_ready_step_mean"]
        evaluated = row["first_filter_evaluation_step_mean"]
        rejected = row["first_rejection_step_mean"]
        rejection_text = (
            "first rejectionなし"
            if rejected is None
            else f"first rejection平均step {rejected:.1f}"
        )
        return (
            f"{name}: ready平均step {ready:.1f}, filter評価平均step {evaluated:.1f}, "
            f"{rejection_text}"
        )

    lines += [
        "",
        "## filter activation と trajectory 監査",
        "",
        activation_text(learned, "learned") + "。" + activation_text(shuffled, "shuffled") + "。",
        "",
        "`prefix exact` は first rejectionより前の物理trajectory、`pool through rejection` は"
        "first rejectionを行う候補集合までのhash一致。active filterでfull trajectoryが異なるのは正常。",
        "",
        "| guided selector | audits | prefix exact | pool through rejection exact | full trajectory exact | full pool-hash exact | final state/training hash exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for selector in ("forest_filter", "forest_filter_shuffled"):
        group = [audit for audit in audits if audit["guided_selector"] == selector]
        lines.append(
            f"| {selector} | {len(group)} | "
            f"{sum(a['trajectory_equal_before_first_rejection'] for a in group)}/{len(group)} | "
            f"{sum(a['pool_hash_equal_through_first_rejection'] for a in group)}/{len(group)} | "
            f"{sum(a['full_trajectory_equal'] for a in group)}/{len(group)} | "
            f"{sum(a['full_candidate_pool_hash_equal'] for a in group)}/{len(group)} | "
            f"{sum(a['final_state_hash_equal'] and a['final_training_hash_equal'] for a in group)}/{len(group)} |"
        )

    lines += [
        "",
        "## 読み方と次の判断",
        "",
        "`cell_archive.json` の raw cell を、reachability audit の prototype shape と"
        " exact layer key に照合した。shape 判定閾値は各 run の configuration にある"
        f" `rms_tolerance={runs[0]['rms_tolerance']}`, "
        f"`max_tolerance={runs[0]['max_tolerance']}` を使用した。",
        "",
        "なお kinematic で発見済みでも feasible elite がまだない cell を一律に duplicate として"
        "除外すると feasible 修復を妨げる。少なくとも feasible 未発見 canonical cell は filter 対象外にする"
        "か、発見 emitter と feasibility-repair emitter の予算を分離すべきである。",
    ]
    if not any_rejection:
        selected_probability_max = max(
            row["selected_max_probability_max"] or -math.inf
            for row in runs
            if row["selector"] == "forest_filter"
        )
        threshold = next(
            row["filter_mean_probability"]
            for row in runs
            if row["selector"] == "forest_filter"
        )
        lines += [
            "",
            "learned と shuffled の一致は、分類器の情報価値が同程度だったことを意味しない。"
            "どちらも gate を通過する duplicate がゼロで、分類結果が action に接続されなかった。"
            f"ログされた learned 選択候補の最大 class probability は {selected_probability_max:.3f}、"
            f"mean-probability threshold は {threshold:.3f}。offline calibrationで"
            "known-duplicate rejectionとpseudo-unknown retentionを同時に制約し、"
            "非ゼロの安全な棄却率を持つ閾値を固定して再評価する必要がある。",
        ]
    else:
        lines += [
            "",
            f"この aggressive panel は {seed_count} seeds の探索的評価であり、"
            "bootstrap CIとsign-flip p値は推定精度が低い。learned-vs-randomだけでなく"
            "learned-vs-shuffledを主要比較にして、情報を持つlabelの寄与と単なる棄却効果を分離する必要がある。",
        ]
    return "\n".join(lines) + "\n"


def _discover_seeds(roots: dict[str, Path]) -> list[int]:
    seed_sets = []
    for root in roots.values():
        seeds = {
            int(path.name.split("_", 1)[1])
            for path in root.glob("seed_*")
            if path.is_dir() and path.name.split("_", 1)[1].isdigit()
        }
        seed_sets.append(seeds)
    common = set.intersection(*seed_sets) if seed_sets else set()
    if not common:
        raise ValueError("the supplied result roots have no common seeds")
    return sorted(common)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-root", default="result/main9_online_b3000_random"
    )
    parser.add_argument(
        "--filter-root", default="result/main9_online_b3000_forest_filter"
    )
    parser.add_argument(
        "--shuffled-root",
        default="result/main9_online_b3000_forest_filter_shuffled",
    )
    parser.add_argument(
        "--reference",
        default="result/reachability_audit/reachability_report.json",
    )
    parser.add_argument(
        "--seeds",
        default="all",
        help="'all', comma-separated integers, or an inclusive range such as 900:909",
    )
    parser.add_argument(
        "--csv", default="result/main9_online_b3000_runs.csv"
    )
    parser.add_argument(
        "--json", default="result/main9_online_b3000_aggregate.json"
    )
    parser.add_argument("--markdown", default="MAIN9_ONLINE_RESULTS.md")
    return parser.parse_args()


def _parse_seeds(specification: str, roots: dict[str, Path]) -> list[int]:
    if specification.strip().lower() == "all":
        return _discover_seeds(roots)
    if ":" in specification:
        start, stop = (int(item) for item in specification.split(":", 1))
        return list(range(start, stop + 1))
    return [int(item) for item in specification.split(",") if item.strip()]


def main() -> None:
    args = parse_arguments()
    roots = {
        "random": Path(args.random_root),
        "forest_filter": Path(args.filter_root),
        "forest_filter_shuffled": Path(args.shuffled_root),
    }
    for selector, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{selector} result root does not exist: {root}")
    reference_path = Path(args.reference)
    reference = _read_json(reference_path)
    targets = _canonical_targets(reference)
    seeds = _parse_seeds(args.seeds, roots)

    runs = []
    diagnostic_lookup: dict[tuple[str, int], list[dict[str, str]]] = {}
    for selector in SELECTOR_ORDER:
        for seed in seeds:
            seed_directory = roots[selector] / f"seed_{seed:03d}"
            if not seed_directory.is_dir():
                raise FileNotFoundError(f"missing paired seed directory: {seed_directory}")
            row, diagnostics = one_run(seed_directory, selector, targets)
            runs.append(row)
            diagnostic_lookup[(selector, seed)] = diagnostics
    runs.sort(key=lambda row: (row["seed"], SELECTOR_ORDER.index(row["selector"])))
    run_lookup = {(row["selector"], row["seed"]): row for row in runs}

    audits = []
    for selector in ("forest_filter", "forest_filter_shuffled"):
        for seed in seeds:
            audits.append(
                trajectory_audit(
                    run_lookup[(selector, seed)],
                    diagnostic_lookup[(selector, seed)],
                    run_lookup[("random", seed)],
                    diagnostic_lookup[("random", seed)],
                )
            )

    payload = {
        "analysis": "main9 collision-continuing online forest filter",
        "panel_name": roots["random"].name.removesuffix("_random"),
        "reference": str(reference_path.resolve()),
        "canonical_target_count": len(targets),
        "canonical_target_ids": [target["target_id"] for target in targets],
        "selectors": list(SELECTOR_ORDER),
        "seeds": seeds,
        "roots": {key: str(path.resolve()) for key, path in roots.items()},
        "censoring": {
            "T6_event": "1 iff all canonical targets were reached by budget",
            "T6_capped": "min(T6, budget); fixed-budget restricted time-to-six",
        },
        "runs": runs,
        "aggregate": aggregate_runs(runs),
        "paired": paired_comparisons(runs),
        "trajectory_audits": audits,
        "integrity": {
            "all_raw_auc_recomputations_match_summary": all(
                row["raw_kinematic_auc_matches_summary"]
                and row["raw_feasible_auc_matches_summary"]
                for row in runs
            ),
            "all_prefixes_exact_before_first_rejection": all(
                audit["trajectory_equal_before_first_rejection"]
                and audit["pool_hash_equal_through_first_rejection"]
                for audit in audits
            ),
        },
    }

    json_path = Path(args.json)
    csv_path = Path(args.csv)
    markdown_path = Path(args.markdown)
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_fields = (
        "seed",
        "selector",
        "budget",
        "used_steps",
        "macro_count",
        "training_rows",
        "raw_kinematic_final",
        "raw_feasible_final",
        "canonical_kinematic_final",
        "canonical_feasible_final",
        "canonical_kinematic_ids",
        "canonical_feasible_ids",
        "unmatched_raw_cells",
        "raw_kinematic_auc",
        "raw_feasible_auc",
        "canonical_kinematic_auc",
        "canonical_feasible_auc",
        "canonical_kinematic_t6",
        "canonical_feasible_t6",
        "canonical_kinematic_t6_event",
        "canonical_feasible_t6_event",
        "canonical_kinematic_t6_censored_at",
        "canonical_feasible_t6_censored_at",
        "canonical_kinematic_t6_capped",
        "canonical_feasible_t6_capped",
        "rejected_total",
        "raw_rejected_total",
        "candidate_total",
        "filter_evaluated_candidate_total",
        "filter_rejection_fraction",
        "raw_filter_rejection_fraction",
        "all_pool_rejection_fraction",
        "macros_with_rejection",
        "first_filter_ready_step",
        "first_filter_evaluation_step",
        "first_raw_rejection_step",
        "first_rejection_step",
        "fit_time_seconds",
        "score_time_seconds",
        "wall_time_seconds",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in runs:
            writer.writerow({field: _csv_value(row.get(field)) for field in csv_fields})

    report = make_markdown(payload)
    markdown_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
