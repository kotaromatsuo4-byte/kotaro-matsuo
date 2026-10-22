"""Counterfactual snapshot evaluation of the main9 terminal-cell filter.

This script answers a narrower question than an ordinary learning curve:

* freeze the random main9 search immediately after raw kinematic cells 2, 3,
  and 4 are first discovered;
* train only on macros that existed at that instant;
* generate a fresh candidate panel from that exact search tree;
* execute every panel member on a transactionally rolled-back copy of the
  mechanics state, without updating the search archive; and
* compare predicted known-cell probabilities and reject decisions with the
  counterfactual terminal cell obtained by the actual rollout.

The expensive counterfactual rollout is evaluation-only.  It never supplies a
teacher to the snapshot model and never changes the replay trajectory.  This
lets us measure both closed-set classification and the selection pressure that
the online filter would have applied at the same historical state.

Examples
--------
Run one seed (``--panel-size 0`` evaluates one complete candidate pool)::

    python main9_snapshot_filter_evaluation.py run --seed 800 --panel-size 96

Aggregate already completed seed directories and redraw the figures::

    python main9_snapshot_filter_evaluation.py aggregate \
        result/main9_snapshot_filter_evaluation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from collision_continue import (
    dual_archive_counts,
    initialize_path_metrics,
    new_dual_archive,
)
from forest_cell_model import (
    conservative_duplicate_mask,
    fit_cell_filter,
    predict_cell_filter,
)
from main9 import (
    _append_teachers,
    _archive_update,
    _candidate_pool,
    _configuration9,
    _execute_macro,
    _initial_trunk,
    _make_model,
    _new_training_data,
    _rejectable_classes,
    _select,
    _thresholds,
)
from main9_forest_evaluation import _GroupedKNN
from mechanics import compute_gravity_potential
from state import create_history


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_KINDS = ("random_forest", "extra_trees", "knn")
DEFAULT_TARGETS = (2, 3, 4)
DEFAULT_RANK_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    def coerce(text: str) -> Any:
        if text == "":
            return ""
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            value = float(text)
        except ValueError:
            return text
        if math.isfinite(value) and value.is_integer() and not any(
            marker in lowered for marker in (".", "e")
        ):
            return int(value)
        return value

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _snapshot_training_arrays(
    training: Mapping[str, Sequence[Any]], feature_dimension: int
) -> dict[str, np.ndarray]:
    labels = np.asarray(training["cell_labels"], dtype=int)
    valid = labels >= 0
    inputs = np.asarray(training["inputs"], dtype=float).reshape(
        -1, feature_dimension
    )
    return {
        "inputs": inputs[valid],
        "labels": labels[valid],
        "groups": np.asarray(training["macro_ids"], dtype=int)[valid],
        "weights": np.asarray(training["sample_weights"], dtype=float)[valid],
    }


def _fit_and_predict(
    model_kind: str,
    training: Mapping[str, np.ndarray],
    query: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    n_estimators: int,
    knn_neighbors: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    if len(training["labels"]) == 0:
        raise ValueError("snapshot contains no completed macro teacher")
    if model_kind in ("random_forest", "extra_trees"):
        fitted = fit_cell_filter(
            training["inputs"],
            training["labels"],
            training["groups"],
            training["weights"],
            model_kind=model_kind,
            rng_seed=int(seed),
            n_estimators=int(n_estimators),
            max_depth=int(config["forest_max_depth"]),
            min_samples_leaf=int(config["forest_min_leaf"]),
        )
        prediction = predict_cell_filter(fitted, query)
        return np.asarray(fitted.classes_, dtype=object), prediction, bool(
            fitted.filter_ready
        )
    if model_kind == "knn":
        fitted = _GroupedKNN(neighbors=int(knn_neighbors)).fit(
            training["inputs"], training["labels"], training["groups"]
        )
        prediction = fitted.predict(query)
        return np.asarray(fitted.classes_, dtype=object), prediction, bool(
            fitted.filter_ready
        )
    raise ValueError(f"unsupported model kind: {model_kind}")


def _cap_mask(
    mask: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    max_fraction: float,
) -> np.ndarray:
    """Apply main9's strongest-confidence cap to one frozen panel."""
    result = np.asarray(mask, dtype=bool).copy()
    cap = int(np.floor(float(max_fraction) * len(result)))
    rejected = np.flatnonzero(result)
    if len(rejected) <= cap:
        return result
    confidence = np.asarray(prediction["max_probability"], dtype=float)
    strongest = rejected[
        np.argsort(confidence[rejected], kind="stable")[::-1][:cap]
    ]
    result[:] = False
    result[strongest] = True
    return result


def _duplicate_score(prediction: Mapping[str, np.ndarray]) -> np.ndarray:
    """Continuous confidence used only for ranked diagnostic ablations.

    It is deliberately not a new production acquisition function.  The score
    uses raw class-confidence only, without the strict support/group gates.
    Ranking it asks whether *any* useful duplicate ordering is already present
    before the safe threshold fires.  In particular, an infinite support ratio
    in a one-class cold start must not turn the whole diagnostic score to zero.
    """
    lower = np.clip(
        np.asarray(prediction["lower_probability"], dtype=float), 0.0, 1.0
    )
    agreement = np.clip(
        np.asarray(prediction["committee_agreement"], dtype=float), 0.0, 1.0
    )
    epistemic = np.maximum(
        np.asarray(prediction["epistemic"], dtype=float), 0.0
    )
    return lower * agreement * np.exp(-4.0 * epistemic)


def _rank_mask(
    score: np.ndarray,
    predicted_labels: np.ndarray,
    rejectable_classes: Iterable[int],
    fraction: float,
    tie_breaker: np.ndarray,
) -> np.ndarray:
    rejectable = {int(value) for value in rejectable_classes}
    eligible = np.asarray(
        [int(label) in rejectable for label in predicted_labels], dtype=bool
    )
    desired = min(
        int(np.floor(float(fraction) * len(score))), int(np.count_nonzero(eligible))
    )
    result = np.zeros(len(score), dtype=bool)
    if desired <= 0:
        return result
    indices = np.flatnonzero(eligible)
    # A one-class model can assign exactly equal confidence to every query.
    # Seeded random tie-breaking makes that honest random-baseline behavior,
    # rather than accidentally preferring the candidate-pool/K order.
    ordering = np.lexsort(
        (np.asarray(tie_breaker, dtype=float)[indices], np.asarray(score)[indices])
    )[::-1]
    result[indices[ordering[:desired]]] = True
    return result


def _filter_metrics(
    truth_known: np.ndarray,
    truth_valid: np.ndarray,
    truth_valuable: np.ndarray,
    rejected: np.ndarray,
) -> dict[str, float | int]:
    valid = np.asarray(truth_valid, dtype=bool)
    duplicate = valid & np.asarray(truth_known, dtype=bool)
    unknown = valid & ~np.asarray(truth_known, dtype=bool)
    valuable = valid & np.asarray(truth_valuable, dtype=bool)
    useless = valid & ~np.asarray(truth_valuable, dtype=bool)
    reject = valid & np.asarray(rejected, dtype=bool)
    survive = valid & ~np.asarray(rejected, dtype=bool)
    valid_count = int(np.count_nonzero(valid))
    duplicate_count = int(np.count_nonzero(duplicate))
    unknown_count = int(np.count_nonzero(unknown))
    rejected_count = int(np.count_nonzero(reject))
    baseline_novelty = _safe_ratio(unknown_count, valid_count)
    survivor_count = int(np.count_nonzero(survive))
    survivor_novelty = _safe_ratio(
        np.count_nonzero(survive & unknown), survivor_count
    )
    baseline_valuable = _safe_ratio(np.count_nonzero(valuable), valid_count)
    survivor_valuable = _safe_ratio(
        np.count_nonzero(survive & valuable), survivor_count
    )
    return {
        "valid_panel_count": valid_count,
        "duplicate_count": duplicate_count,
        "unknown_count": unknown_count,
        "valuable_count": int(np.count_nonzero(valuable)),
        "useless_count": int(np.count_nonzero(useless)),
        "rejected_count": rejected_count,
        "actual_reject_fraction": _safe_ratio(rejected_count, valid_count),
        "reject_precision": _safe_ratio(
            np.count_nonzero(reject & duplicate), rejected_count
        ),
        "duplicate_rejection_rate": _safe_ratio(
            np.count_nonzero(reject & duplicate), duplicate_count
        ),
        "unknown_retention": _safe_ratio(
            np.count_nonzero(survive & unknown), unknown_count
        ),
        "valuable_retention": _safe_ratio(
            np.count_nonzero(survive & valuable), np.count_nonzero(valuable)
        ),
        "useless_reject_precision": _safe_ratio(
            np.count_nonzero(reject & useless), rejected_count
        ),
        "useless_rejection_rate": _safe_ratio(
            np.count_nonzero(reject & useless), np.count_nonzero(useless)
        ),
        "baseline_novelty_rate": baseline_novelty,
        "survivor_novelty_rate": survivor_novelty,
        "survivor_novelty_lift": _safe_ratio(
            survivor_novelty, baseline_novelty
        ),
        "baseline_valuable_rate": baseline_valuable,
        "survivor_valuable_rate": survivor_valuable,
        "survivor_valuable_lift": _safe_ratio(
            survivor_valuable, baseline_valuable
        ),
    }


def _classification_metrics(
    truth_cell: np.ndarray,
    truth_known: np.ndarray,
    truth_valid: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    known_cell_count: int,
    calibration_bins: int = 10,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Score against every snapshot-known class, including unseen classes.

    If a snapshot archive contains a cell with no completed macro teacher, the
    model assigns it probability zero and an occurrence of that cell is an
    ordinary error.  This fixed label space avoids the changing-denominator
    artifact of the older conditional learning curve.
    """
    expanded = np.zeros((len(probabilities), int(known_cell_count)), dtype=float)
    for source, raw_label in enumerate(classes):
        label = int(raw_label)
        if 0 <= label < known_cell_count:
            expanded[:, label] = probabilities[:, source]
    valid_rows = np.flatnonzero(np.asarray(truth_valid, dtype=bool))
    rows = np.flatnonzero(
        np.asarray(truth_valid, dtype=bool) & np.asarray(truth_known, dtype=bool)
    )
    calibration: list[dict[str, float | int]] = []
    if len(rows) == 0:
        for bin_index in range(calibration_bins):
            calibration.append(
                {
                    "bin": bin_index,
                    "lower": bin_index / calibration_bins,
                    "upper": (bin_index + 1) / calibration_bins,
                    "count": 0,
                    "mean_confidence": float("nan"),
                    "empirical_accuracy": float("nan"),
                }
            )
        return {
            "valid_test_count": int(len(valid_rows)),
            "known_test_count": 0,
            "panel_known_target_coverage": 0.0,
            "known_top1_accuracy": float("nan"),
            "open_world_top1_accuracy": 0.0 if len(valid_rows) else float("nan"),
            "known_brier": float("nan"),
            "known_nll": float("nan"),
            "known_ece": float("nan"),
            "mean_confidence": float("nan"),
        }, calibration

    target = np.asarray(truth_cell[rows], dtype=int)
    prediction = np.argmax(expanded[rows], axis=1)
    correct = prediction == target
    confidence = np.max(expanded[rows], axis=1)
    selected = np.maximum(expanded[rows, target], 1.0e-15)
    one_hot = np.zeros_like(expanded[rows])
    one_hot[np.arange(len(rows)), target] = 1.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for bin_index in range(calibration_bins):
        if bin_index + 1 == calibration_bins:
            mask = (confidence >= edges[bin_index]) & (
                confidence <= edges[bin_index + 1]
            )
        else:
            mask = (confidence >= edges[bin_index]) & (
                confidence < edges[bin_index + 1]
            )
        count = int(np.count_nonzero(mask))
        mean_confidence = float(np.mean(confidence[mask])) if count else float("nan")
        empirical = float(np.mean(correct[mask])) if count else float("nan")
        if count:
            ece += count / len(rows) * abs(empirical - mean_confidence)
        calibration.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical,
            }
        )
    known_correct = int(np.count_nonzero(correct))
    return {
        "valid_test_count": int(len(valid_rows)),
        "known_test_count": int(len(rows)),
        "panel_known_target_coverage": _safe_ratio(len(rows), len(valid_rows)),
        "known_top1_accuracy": float(np.mean(correct)),
        # The classifier has no unknown output class.  Every actually novel
        # valid endpoint is therefore an open-world top-1 error by definition.
        "open_world_top1_accuracy": _safe_ratio(known_correct, len(valid_rows)),
        "known_brier": float(np.mean(np.sum((expanded[rows] - one_hot) ** 2, axis=1))),
        "known_nll": float(-np.mean(np.log(selected))),
        "known_ece": float(ece),
        "mean_confidence": float(np.mean(confidence)),
    }, calibration


def _make_panel(
    history,
    path_metrics,
    model,
    config,
    rng: np.random.Generator,
    panel_size: int,
) -> list[dict[str, Any]]:
    """Generate independent pools until the requested panel is full.

    ``panel_size == 0`` means one complete bounded pool.  For a capped panel,
    candidates are uniformly subsampled within every fresh pool so K/order in
    the pool implementation cannot bias a prefix.
    """
    first = _candidate_pool(history, path_metrics, model, config, rng)
    if not first:
        return []
    if panel_size == 0:
        result = first
    else:
        result: list[dict[str, Any]] = []
        pool = first
        pool_index = 0
        while len(result) < panel_size:
            remaining = panel_size - len(result)
            take = min(remaining, len(pool))
            chosen = rng.choice(len(pool), size=take, replace=False)
            for index in chosen:
                item = dict(pool[int(index)])
                item["panel_pool_id"] = int(pool_index)
                result.append(item)
            pool_index += 1
            if len(result) >= panel_size:
                break
            pool = _candidate_pool(history, path_metrics, model, config, rng)
            if not pool:
                break
    for candidate_index, item in enumerate(result):
        item.setdefault("panel_pool_id", 0)
        item["panel_candidate_id"] = int(candidate_index)
    return result


def _evaluate_counterfactual_panel(
    history,
    path_metrics,
    snapshot_archive,
    panel: Sequence[Mapping[str, Any]],
    model,
    config,
    rollout_budget: int,
) -> list[dict[str, Any]]:
    """Execute each candidate, then restore history and path metadata exactly."""
    known_count = len(snapshot_archive)
    oracle_archive = copy.deepcopy(snapshot_archive)
    rows: list[dict[str, Any]] = []
    base_length = len(history)
    random_state = np.random.get_state()
    try:
        for candidate_index, candidate in enumerate(panel):
            if len(history) != base_length:
                raise RuntimeError("counterfactual rollback lost history alignment")
            parent_id = int(candidate["parent_id"])
            child_count = len(history[parent_id].children_ids)
            state_before = np.random.get_state()
            try:
                result = _execute_macro(
                    history,
                    path_metrics,
                    candidate,
                    int(rollout_budget),
                    model,
                    config,
                )
                # Each value decision must see exactly the frozen archive.
                # An accumulating oracle archive is used only to assign stable
                # ids to distinct novel panel basins.
                trial_archive = copy.deepcopy(snapshot_archive)
                trial_update = _archive_update(
                    trial_archive,
                    history,
                    path_metrics,
                    result["terminal"].state_id,
                    model,
                    config,
                )
                oracle_update = _archive_update(
                    oracle_archive,
                    history,
                    path_metrics,
                    result["terminal"].state_id,
                    model,
                    config,
                )
                valid = bool(
                    trial_update["accepted"] and result["stop_reason"] == "flat"
                )
                cell = int(oracle_update["cell_index"]) if valid else -1
                snapshot_cell = (
                    int(trial_update["cell_index"]) if valid else -1
                )
                valuable = bool(
                    valid
                    and (
                        trial_update["new_kinematic"]
                        or trial_update["new_feasible"]
                        or trial_update["kinematic_replaced"]
                        or trial_update["feasible_replaced"]
                    )
                )
                quality = path_metrics[result["terminal"].state_id]
                rows.append(
                    {
                        "candidate_id": int(candidate_index),
                        "panel_pool_id": int(candidate.get("panel_pool_id", 0)),
                        "parent_id": parent_id,
                        "K": int(candidate["k"]),
                        "truth_valid": int(valid),
                        "truth_cell": cell,
                        "snapshot_cell": snapshot_cell,
                        "truth_known": int(valid and snapshot_cell < known_count),
                        "truth_unknown": int(valid and snapshot_cell >= known_count),
                        "truth_valuable": int(valuable),
                        "truth_useless": int(valid and not valuable),
                        "new_kinematic": int(valid and trial_update["new_kinematic"]),
                        "new_feasible": int(valid and trial_update["new_feasible"]),
                        "kinematic_replaced": int(
                            valid and trial_update["kinematic_replaced"]
                        ),
                        "feasible_replaced": int(
                            valid and trial_update["feasible_replaced"]
                        ),
                        "truth_feasible": int(valid and quality.collision_free),
                        "stop_reason": str(result["stop_reason"]),
                        "mechanics_cost": int(result["used_steps"]),
                        "executed_K": int(result["executed_k"]),
                        "suffix_steps": int(result["suffix_steps"]),
                        "projection_failure": int(result["projection_failure"]),
                        "path_collision_edges": int(quality.collision_edge_count),
                    }
                )
            finally:
                for state_id in range(base_length, len(history)):
                    path_metrics.pop(state_id, None)
                del history[base_length:]
                del history[parent_id].children_ids[child_count:]
                np.random.set_state(state_before)
                if len(history) != base_length or len(path_metrics) != base_length:
                    raise RuntimeError(
                        "counterfactual transaction did not restore tree metadata"
                    )
    finally:
        np.random.set_state(random_state)
    return rows


def _evaluate_snapshot(
    *,
    target_cells: int,
    used_steps: int,
    macro_count: int,
    history,
    path_metrics,
    archive,
    training,
    model,
    config,
    panel_size: int,
    panel_rollout_budget: int,
    panel_seed: int,
    n_estimators: int,
    knn_neighbors: int,
    rank_fractions: Sequence[float],
) -> dict[str, list[dict[str, Any]]]:
    panel_rng = np.random.default_rng(int(panel_seed))
    panel = _make_panel(
        history, path_metrics, model, config, panel_rng, int(panel_size)
    )
    truth_rows = _evaluate_counterfactual_panel(
        history,
        path_metrics,
        archive,
        panel,
        model,
        config,
        panel_rollout_budget,
    )
    if not panel:
        raise RuntimeError(f"candidate panel is empty at {target_cells} cells")

    query = np.asarray([item["input"] for item in panel], dtype=float)
    truth_valid = np.asarray([row["truth_valid"] for row in truth_rows], dtype=bool)
    truth_known = np.asarray([row["truth_known"] for row in truth_rows], dtype=bool)
    truth_valuable = np.asarray(
        [row["truth_valuable"] for row in truth_rows], dtype=bool
    )
    truth_cell = np.asarray([row["truth_cell"] for row in truth_rows], dtype=int)
    arrays = _snapshot_training_arrays(training, model["input_dimension"])
    feasible_classes = _rejectable_classes(archive)
    rank_tie_breaker = np.random.default_rng(int(panel_seed) + 700001).random(
        len(panel)
    )
    snapshot_common = {
        "seed": int(config["seed"]),
        "target_cells": int(target_cells),
        "snapshot_step": int(used_steps),
        "snapshot_macro_count": int(macro_count),
        "archive_cells": int(len(archive)),
        "archive_feasible_cells": int(dual_archive_counts(archive)[1]),
        "train_rows": int(len(arrays["labels"])),
        "train_macros": int(len(np.unique(arrays["groups"]))) if len(arrays["groups"]) else 0,
        "train_classes": int(len(np.unique(arrays["labels"]))) if len(arrays["labels"]) else 0,
        "panel_count": int(len(panel)),
        "valid_panel_count": int(np.count_nonzero(truth_valid)),
        "known_panel_count": int(np.count_nonzero(truth_valid & truth_known)),
        "unknown_panel_count": int(np.count_nonzero(truth_valid & ~truth_known)),
        "censored_panel_count": int(np.count_nonzero(~truth_valid)),
        "valuable_panel_count": int(np.count_nonzero(truth_valid & truth_valuable)),
        "useless_panel_count": int(np.count_nonzero(truth_valid & ~truth_valuable)),
    }
    model_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_index, model_kind in enumerate(MODEL_KINDS):
        classes, prediction, filter_ready = _fit_and_predict(
            model_kind,
            arrays,
            query,
            config,
            seed=int(panel_seed + 1009 * (model_index + 1)),
            n_estimators=n_estimators,
            knn_neighbors=knn_neighbors,
        )
        classification, calibration = _classification_metrics(
            truth_cell,
            truth_known,
            truth_valid,
            np.asarray(prediction["probabilities"], dtype=float),
            classes,
            len(archive),
        )
        model_row = {
            **snapshot_common,
            "model_kind": model_kind,
            "filter_ready": int(filter_ready),
            "model_classes": int(len(classes)),
            "archive_class_coverage": _safe_ratio(len(classes), len(archive)),
            **classification,
        }
        model_rows.append(model_row)
        for calibration_row in calibration:
            calibration_rows.append(
                {
                    "seed": int(config["seed"]),
                    "target_cells": int(target_cells),
                    "model_kind": model_kind,
                    **calibration_row,
                }
            )

        strict_definitions = (
            ("strict_all_known", list(range(len(archive)))),
            ("strict_feasible", feasible_classes),
        )
        strict_masks: dict[str, np.ndarray] = {}
        for policy, rejectable in strict_definitions:
            reject = conservative_duplicate_mask(
                prediction, rejectable, thresholds=_thresholds(config)
            )
            reject = _cap_mask(
                reject, prediction, float(config["max_reject_fraction"])
            )
            strict_masks[policy] = reject.copy()
            filter_rows.append(
                {
                    **snapshot_common,
                    "model_kind": model_kind,
                    "filter_ready": int(filter_ready),
                    "policy": policy,
                    "requested_reject_fraction": float("nan"),
                    **_filter_metrics(
                        truth_known, truth_valid, truth_valuable, reject
                    ),
                }
            )

        score = _duplicate_score(prediction)
        labels = np.asarray(prediction["predicted_labels"], dtype=int)
        for fraction in rank_fractions:
            reject = _rank_mask(
                score,
                labels,
                range(len(archive)),
                float(fraction),
                rank_tie_breaker,
            )
            filter_rows.append(
                {
                    **snapshot_common,
                    "model_kind": model_kind,
                    "filter_ready": int(filter_ready),
                    "policy": "ranked_confidence_diagnostic",
                    "requested_reject_fraction": float(fraction),
                    **_filter_metrics(
                        truth_known, truth_valid, truth_valuable, reject
                    ),
                }
            )

        probabilities = np.asarray(prediction["probabilities"], dtype=float)
        for index, truth in enumerate(truth_rows):
            probability_map = {
                str(int(label)): float(probabilities[index, source])
                for source, label in enumerate(classes)
            }
            prediction_rows.append(
                {
                    "seed": int(config["seed"]),
                    "target_cells": int(target_cells),
                    "model_kind": model_kind,
                    **truth,
                    "predicted_cell": int(prediction["predicted_labels"][index]),
                    "max_probability": float(prediction["max_probability"][index]),
                    "lower_probability": float(prediction["lower_probability"][index]),
                    "committee_agreement": float(
                        prediction["committee_agreement"][index]
                    ),
                    "epistemic": float(prediction["epistemic"][index]),
                    "support_ratio": float(prediction["support_ratio"][index]),
                    "class_group_count": int(
                        prediction["class_group_count"][index]
                    ),
                    "duplicate_score": float(score[index]),
                    "strict_all_known_reject": int(
                        strict_masks["strict_all_known"][index]
                    ),
                    "strict_feasible_reject": int(
                        strict_masks["strict_feasible"][index]
                    ),
                    "probabilities_json": json.dumps(probability_map, sort_keys=True),
                }
            )

    return {
        "model_rows": model_rows,
        "filter_rows": filter_rows,
        "calibration_rows": calibration_rows,
        "prediction_rows": prediction_rows,
    }


def run_replay(args: argparse.Namespace) -> Path:
    config = _configuration9()
    config.update(
        {
            "seed": int(args.seed),
            "total_step_budget": int(args.replay_budget),
            "selector": "random",
            # The experiment under test always completes the deterministic
            # suffix.  Only the explicit replay/panel mechanics budgets may
            # censor it; an inherited environment timeout must not do so.
            "max_suffix_steps": 0,
            "verbose": False,
            "save_tree": False,
        }
    )
    if args.panel_size < 0 or args.panel_rollout_budget <= 0:
        raise ValueError("panel-size must be nonnegative and panel budget positive")
    targets = sorted(set(int(value) for value in args.targets))
    if not targets or min(targets) <= 0:
        raise ValueError("targets must contain positive cell counts")
    if any(not 0.0 < value < 1.0 for value in args.rank_fractions):
        raise ValueError("rank fractions must be in (0, 1)")

    base_directory = SCRIPT_DIRECTORY
    initial_node, model = _make_model(base_directory, config)
    history = create_history(
        initial_node,
        compute_gravity_potential(initial_node, config["reference_point"]),
    )
    path_metrics = initialize_path_metrics(0)
    archive = new_dual_archive()
    training = _new_training_data()
    candidate_rng = np.random.default_rng(int(args.seed) + 9101)
    selection_rng = np.random.default_rng(int(args.seed) + 9202)
    shuffle_rng = np.random.default_rng(int(args.seed) + 9404)
    np.random.seed(int(args.seed))
    used_steps = 0
    macro_id = 0
    outputs = {
        "model_rows": [],
        "filter_rows": [],
        "calibration_rows": [],
        "prediction_rows": [],
    }
    completed_targets: set[int] = set()

    trunk, spent, trunk_reason = _initial_trunk(
        history, path_metrics, config["total_step_budget"], model, config
    )
    used_steps += int(spent)
    if trunk_reason == "flat":
        _archive_update(
            archive, history, path_metrics, trunk.state_id, model, config
        )

    def evaluate_new_targets() -> None:
        for target in targets:
            if target in completed_targets or len(archive) < target:
                continue
            evaluated = _evaluate_snapshot(
                target_cells=target,
                used_steps=used_steps,
                macro_count=macro_id,
                history=history,
                path_metrics=path_metrics,
                archive=copy.deepcopy(archive),
                training=copy.deepcopy(training),
                model=model,
                config=config,
                panel_size=int(args.panel_size),
                panel_rollout_budget=int(args.panel_rollout_budget),
                panel_seed=int(args.panel_seed_base + args.seed * 100 + target),
                n_estimators=int(args.n_estimators),
                knn_neighbors=int(args.knn_neighbors),
                rank_fractions=args.rank_fractions,
            )
            for key in outputs:
                outputs[key].extend(evaluated[key])
            completed_targets.add(target)
            print(
                f"seed={args.seed} snapshot={target} cells step={used_steps} "
                f"macros={macro_id} panel={args.panel_size or 'full'}",
                flush=True,
            )

    evaluate_new_targets()
    while used_steps < config["total_step_budget"] and len(completed_targets) < len(targets):
        pool = _candidate_pool(history, path_metrics, model, config, candidate_rng)
        if not pool:
            break
        selected_index, _ = _select(
            pool,
            "random",
            None,
            archive,
            selection_rng,
            shuffle_rng,
            config,
        )
        selected = pool[selected_index]
        macro_start = int(used_steps)
        result = _execute_macro(
            history,
            path_metrics,
            selected,
            config["total_step_budget"] - used_steps,
            model,
            config,
        )
        used_steps += int(result["used_steps"])
        update = _archive_update(
            archive,
            history,
            path_metrics,
            result["terminal"].state_id,
            model,
            config,
        )
        label = update["cell_index"] if update["accepted"] else -1
        _append_teachers(
            training,
            history,
            path_metrics,
            result,
            macro_id,
            macro_start,
            used_steps,
            label,
            model,
            config,
        )
        macro_id += 1
        evaluate_new_targets()

    output_directory = Path(args.output_root).resolve() / f"seed_{args.seed:03d}"
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "snapshot_metrics.csv", outputs["model_rows"])
    _write_csv(output_directory / "filter_metrics.csv", outputs["filter_rows"])
    _write_csv(output_directory / "calibration_bins.csv", outputs["calibration_rows"])
    _write_csv(output_directory / "panel_predictions.csv", outputs["prediction_rows"])
    summary = {
        "algorithm": "main9_snapshot_counterfactual_filter_evaluation_v1",
        "seed": int(args.seed),
        "replay_budget": int(args.replay_budget),
        "used_steps": int(used_steps),
        "macro_count": int(macro_id),
        "final_raw_kinematic_cells": int(len(archive)),
        "final_feasible_cells": int(dual_archive_counts(archive)[1]),
        "requested_targets": targets,
        "completed_targets": sorted(completed_targets),
        "missing_targets": sorted(set(targets) - completed_targets),
        "panel_size": int(args.panel_size),
        "panel_rollout_budget": int(args.panel_rollout_budget),
        "n_estimators": int(args.n_estimators),
        "knn_neighbors": int(args.knn_neighbors),
        "rank_fractions": [float(value) for value in args.rank_fractions],
        "configuration": _jsonable(config),
    }
    (output_directory / "run_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregate_results([output_directory], output_directory)
    print(json.dumps(_jsonable(summary), ensure_ascii=False))
    return output_directory


def _discover_seed_directories(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if (path / "snapshot_metrics.csv").exists():
            result.append(path)
        elif path.is_dir():
            result.extend(
                candidate.parent
                for candidate in sorted(path.rglob("snapshot_metrics.csv"))
            )
        else:
            raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError("no snapshot_metrics.csv files were found")
    return unique


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]
) -> list[dict[str, Any]]:
    def group_value(value: Any) -> Any:
        if isinstance(value, (int, float, np.number)) and not _finite(value):
            # NaN is not equal to itself and would split one strict policy into
            # one group per seed.  None is the intended "not applicable" key.
            return None
        return value

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(group_value(row.get(key)) for key in group_keys)].append(row)
    result: list[dict[str, Any]] = []
    excluded = set(group_keys) | {"seed"}
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(group_keys, key))
        record["seed_count"] = len({row.get("seed") for row in selected})
        numeric_keys = sorted(
            {
                field
                for row in selected
                for field, value in row.items()
                if field not in excluded and _finite(value)
            }
        )
        for field in numeric_keys:
            values = np.asarray(
                [float(row[field]) for row in selected if _finite(row.get(field))],
                dtype=float,
            )
            if not len(values):
                continue
            record[f"{field}_mean"] = float(np.mean(values))
            record[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            record[f"{field}_n"] = int(len(values))
        result.append(record)
    return result


def _plot_results(
    model_rows: Sequence[Mapping[str, Any]],
    filter_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "random_forest": "#2474B5",
        "extra_trees": "#E07A1F",
        "knn": "#2B9B57",
    }
    markers = {2: "o", 3: "s", 4: "^"}
    model_labels = {
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "knn": "group-kNN",
    }

    aggregate_model = _aggregate_rows(
        model_rows, ("target_cells", "model_kind")
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    specifications = (
        ("known_top1_accuracy", "Known-cell top-1 accuracy", (0.0, 1.02)),
        (
            "open_world_top1_accuracy",
            "Open-world top-1 (unknown = error)",
            (0.0, 1.02),
        ),
        ("known_brier", "Brier score (lower is better)", None),
        ("known_ece", "Calibration error, ECE (lower is better)", None),
    )
    for model_kind in MODEL_KINDS:
        selected = sorted(
            [row for row in aggregate_model if row["model_kind"] == model_kind],
            key=lambda row: int(row["target_cells"]),
        )
        if not selected:
            continue
        x = np.asarray([int(row["target_cells"]) for row in selected])
        for axis, (metric, title, ylim) in zip(axes, specifications):
            mean_key = f"{metric}_mean"
            valid = [row for row in selected if _finite(row.get(mean_key))]
            if not valid:
                continue
            vx = np.asarray([int(row["target_cells"]) for row in valid])
            y = np.asarray([float(row[mean_key]) for row in valid])
            error = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in valid])
            axis.errorbar(
                vx,
                y,
                yerr=error,
                color=colors[model_kind],
                marker="o",
                linewidth=2,
                capsize=3,
                label=model_labels[model_kind],
            )
            axis.set_title(title)
            axis.set_xlabel("Known cells at frozen snapshot")
            axis.set_xticks(sorted(set(int(value) for value in x)))
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("Panel metric")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False)
    fig.suptitle(
        "Can the model predict held-out counterfactual rollouts at the time?",
        fontsize=13,
    )
    fig.savefig(output_directory / "classification_diagnostics.png", dpi=180)
    plt.close(fig)

    # A compact headline figure keeps the main causal story legible without
    # requiring the reader to decode every threshold in the detailed plot.
    aggregate_filter = _aggregate_rows(
        filter_rows,
        ("target_cells", "model_kind", "policy", "requested_reject_fraction"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    flat_axes = axes.reshape(-1)

    # The number of independent macros, not the many correlated hindsight
    # rows, is the relevant amount of online supervision.
    support_rows = [row for row in model_rows if row["model_kind"] == MODEL_KINDS[0]]
    support_targets = sorted({int(row["target_cells"]) for row in support_rows})
    for target in support_targets:
        values = [
            float(row["train_macros"])
            for row in support_rows
            if int(row["target_cells"]) == target
        ]
        offsets = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else [0.0]
        flat_axes[0].scatter(
            np.asarray(offsets) + target,
            values,
            color="#777777",
            s=38,
            alpha=0.8,
            zorder=2,
        )
        flat_axes[0].scatter(
            [target],
            [float(np.mean(values))],
            color="black",
            marker="D",
            s=58,
            zorder=3,
        )
    flat_axes[0].set_title("Independent training macros")
    flat_axes[0].set_ylabel("Macro groups available online")

    headline_specs = (
        (1, "model", "known_top1_accuracy", "Known-cell top-1", (0.0, 1.02)),
        (2, "model", "open_world_top1_accuracy", "Open-world top-1", (0.0, 1.02)),
        (3, "strict", "actual_reject_fraction", "Production strict rejection", (-0.005, 0.105)),
        (4, "ranked", "reject_precision", "Diagnostic top-10% reject precision", (0.0, 1.02)),
        (5, "ranked", "duplicate_rejection_rate", "Diagnostic duplicate rejection", (0.0, 0.35)),
        (6, "ranked", "unknown_retention", "Diagnostic unknown retention", (0.0, 1.02)),
        (7, "ranked", "survivor_novelty_lift", "Diagnostic survivor novelty lift", None),
    )
    for axis_index, source, metric, title, ylim in headline_specs:
        axis = flat_axes[axis_index]
        for model_kind in MODEL_KINDS:
            if source == "model":
                selected = [
                    row for row in aggregate_model if row["model_kind"] == model_kind
                ]
            elif source == "strict":
                selected = [
                    row
                    for row in aggregate_filter
                    if row["model_kind"] == model_kind
                    and row["policy"] == "strict_all_known"
                ]
            else:
                selected = [
                    row
                    for row in aggregate_filter
                    if row["model_kind"] == model_kind
                    and row["policy"] == "ranked_confidence_diagnostic"
                    and abs(float(row["requested_reject_fraction"]) - 0.10) < 1.0e-12
                ]
            selected = sorted(selected, key=lambda row: int(row["target_cells"]))
            valid = [row for row in selected if _finite(row.get(f"{metric}_mean"))]
            if not valid:
                continue
            axis.plot(
                [int(row["target_cells"]) for row in valid],
                [float(row[f"{metric}_mean"]) for row in valid],
                color=colors[model_kind],
                marker="o",
                linewidth=2,
                label=model_labels[model_kind],
            )
        axis.set_title(title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        if metric == "survivor_novelty_lift":
            axis.axhline(1.0, color="black", linewidth=1, alpha=0.45)
        axis.grid(alpha=0.25)
    for axis in flat_axes:
        axis.set_xlabel("Known cells at frozen snapshot")
        axis.set_xticks(support_targets)
    handles, labels = flat_axes[1].get_legend_handles_labels()
    if handles:
        flat_axes[1].legend(frameon=False, fontsize=8)
    flat_axes[3].text(
        0.5,
        0.70,
        "all models: 0 rejection",
        transform=flat_axes[3].transAxes,
        ha="center",
        fontsize=9,
    )
    fig.suptitle(
        "Terminal-cell model and duplicate-filter readiness at each discovery snapshot\n"
        "gray dots/black diamonds: seed support/mean; top-10% ranking is diagnostic, not the deployed gate",
        fontsize=13,
    )
    fig.savefig(output_directory / "snapshot_filter_summary.png", dpi=180)
    plt.close(fig)

    # Reliability is easier to judge from the full curve than from ECE alone.
    # Pool bins by their actual candidate count rather than averaging the
    # per-seed bin means, which would over-weight sparse bins.
    targets = sorted({int(row["target_cells"]) for row in calibration_rows})
    if targets:
        fig, axes = plt.subplots(
            len(MODEL_KINDS),
            len(targets),
            figsize=(4.4 * len(targets), 4.0 * len(MODEL_KINDS)),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        for model_index, model_kind in enumerate(MODEL_KINDS):
            for target_index, target in enumerate(targets):
                axis = axes[model_index, target_index]
                selected = [
                    row
                    for row in calibration_rows
                    if row["model_kind"] == model_kind
                    and int(row["target_cells"]) == target
                ]
                grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
                for row in selected:
                    grouped[int(row["bin"])].append(row)
                confidence: list[float] = []
                accuracy: list[float] = []
                counts: list[int] = []
                for bin_index in sorted(grouped):
                    rows = grouped[bin_index]
                    count = int(
                        sum(int(row["count"]) for row in rows if _finite(row.get("count")))
                    )
                    if count <= 0:
                        continue
                    confidence.append(
                        sum(
                            int(row["count"]) * float(row["mean_confidence"])
                            for row in rows
                            if int(row["count"]) > 0
                            and _finite(row.get("mean_confidence"))
                        )
                        / count
                    )
                    accuracy.append(
                        sum(
                            int(row["count"]) * float(row["empirical_accuracy"])
                            for row in rows
                            if int(row["count"]) > 0
                            and _finite(row.get("empirical_accuracy"))
                        )
                        / count
                    )
                    counts.append(count)
                axis.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.45)
                if counts:
                    size = 32.0 + 90.0 * np.sqrt(np.asarray(counts) / max(counts))
                    axis.plot(
                        confidence,
                        accuracy,
                        color=colors[model_kind],
                        linewidth=1.8,
                        alpha=0.8,
                    )
                    axis.scatter(
                        confidence,
                        accuracy,
                        s=size,
                        color=colors[model_kind],
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                    axis.text(
                        0.03,
                        0.94,
                        f"known endpoints n={sum(counts)}",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=8,
                    )
                axis.set_xlim(0.0, 1.03)
                axis.set_ylim(0.0, 1.03)
                axis.grid(alpha=0.2)
                if model_index == 0:
                    axis.set_title(f"{target} known cells")
                if target_index == 0:
                    axis.set_ylabel(f"{model_labels[model_kind]}\nEmpirical accuracy")
                if model_index + 1 == len(MODEL_KINDS):
                    axis.set_xlabel("Predicted confidence")
        fig.suptitle(
            "Reliability of the predicted terminal-cell confidence\n"
            "marker area reflects the number of counterfactual endpoints",
            fontsize=13,
        )
        fig.savefig(output_directory / "calibration_reliability.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    metric_specs = (
        ("reject_precision", "Reject precision: P(known | rejected)", (0.0, 1.02)),
        ("duplicate_rejection_rate", "Known-duplicate rejection rate", (0.0, 1.02)),
        ("unknown_retention", "Unknown-cell retention", (0.0, 1.02)),
        ("survivor_novelty_lift", "Novelty enrichment among survivors", None),
        ("valuable_retention", "Valuable-candidate retention", (0.0, 1.02)),
        (
            "useless_reject_precision",
            "Reject precision: P(useless | rejected)",
            (0.0, 1.02),
        ),
    )
    ranked = [
        row
        for row in aggregate_filter
        if row["policy"] == "ranked_confidence_diagnostic"
    ]
    for model_kind in MODEL_KINDS:
        for target in sorted({int(row["target_cells"]) for row in ranked}):
            selected = sorted(
                [
                    row
                    for row in ranked
                    if row["model_kind"] == model_kind
                    and int(row["target_cells"]) == target
                ],
                key=lambda row: float(row["requested_reject_fraction"]),
            )
            if not selected:
                continue
            x = np.asarray(
                [float(row["actual_reject_fraction_mean"]) for row in selected]
            )
            for axis, (metric, title, ylim) in zip(axes.reshape(-1), metric_specs):
                valid = [row for row in selected if _finite(row.get(f"{metric}_mean"))]
                if not valid:
                    continue
                vx = np.asarray(
                    [float(row["actual_reject_fraction_mean"]) for row in valid]
                )
                vy = np.asarray([float(row[f"{metric}_mean"]) for row in valid])
                axis.plot(
                    vx,
                    vy,
                    color=colors[model_kind],
                    marker=markers.get(target, "o"),
                    linestyle={2: "-", 3: "--", 4: ":"}.get(target, "-"),
                    linewidth=1.7,
                    label=f"{model_labels[model_kind]}, {target} cells",
                )
                axis.set_title(title)
                axis.set_xlabel("Actual rejected fraction of valid panel")
                if ylim is not None:
                    axis.set_ylim(*ylim)
                axis.set_xlim(left=0.0)
                axis.grid(alpha=0.25)

    strict_markers = {"strict_all_known": "*", "strict_feasible": "X"}
    for policy, marker in strict_markers.items():
        for row in aggregate_filter:
            if row["policy"] != policy:
                continue
            model_kind = str(row["model_kind"])
            target = int(row["target_cells"])
            for axis, (metric, _, _) in zip(axes.reshape(-1), metric_specs):
                x_key = "actual_reject_fraction_mean"
                y_key = f"{metric}_mean"
                if _finite(row.get(x_key)) and _finite(row.get(y_key)):
                    axis.scatter(
                        [float(row[x_key])],
                        [float(row[y_key])],
                        color=colors.get(model_kind, "black"),
                        marker=marker,
                        s=100,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=5,
                    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].axhline(1.0, color="black", linewidth=1, alpha=0.35)
    fig.suptitle(
        "Does model confidence create useful duplicate-filter selection pressure?\n"
        "lines: ranked diagnostic; stars: strict all-known; X: strict feasible-only",
        fontsize=13,
    )
    fig.savefig(output_directory / "filter_selection_pressure.png", dpi=180)
    plt.close(fig)


def _format(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if _finite(value) else "NA"


def _write_report(
    seed_directories: Sequence[Path],
    model_rows: Sequence[Mapping[str, Any]],
    filter_rows: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> None:
    aggregate_model = _aggregate_rows(
        model_rows, ("target_cells", "model_kind")
    )
    aggregate_filter = _aggregate_rows(
        filter_rows,
        ("target_cells", "model_kind", "policy", "requested_reject_fraction"),
    )
    last_target = max(int(row["target_cells"]) for row in aggregate_model)
    last_ranked = [
        row
        for row in aggregate_filter
        if int(row["target_cells"]) == last_target
        and row["policy"] == "ranked_confidence_diagnostic"
        and abs(float(row["requested_reject_fraction"]) - 0.10) < 1.0e-12
    ]
    best_ranked = max(
        last_ranked,
        key=lambda row: float(row.get("survivor_novelty_lift_mean", -np.inf)),
    )
    strict_max = max(
        float(row.get("actual_reject_fraction_mean", 0.0))
        for row in aggregate_filter
        if str(row["policy"]).startswith("strict_")
    )
    macro_support = []
    for target in sorted({int(row["target_cells"]) for row in aggregate_model}):
        representative = next(
            row
            for row in aggregate_model
            if int(row["target_cells"]) == target
        )
        macro_support.append(
            f"{target} cell={_format(representative.get('train_macros_mean'), 1)}"
        )
    representative_rows = [
        row for row in model_rows if row["model_kind"] == MODEL_KINDS[0]
    ]
    total_panel = sum(int(row["panel_count"]) for row in representative_rows)
    total_valid = sum(int(row["valid_panel_count"]) for row in representative_rows)
    lines = [
        "# main9 snapshot counterfactual filter evaluation",
        "",
        f"Seeds: {len(seed_directories)}.  各行はraw kinematic cell数が初めて"
        "2/3/4になった直後にtree・archive・教師を凍結し、別RNGの候補を"
        "counterfactual rolloutした結果です。panel rolloutは学習にもarchive"
        "更新にも使用していません。",
        "",
        "## 先に結論",
        "",
        f"独立な教師macro数（seed平均）は {', '.join(macro_support)} だけでした。",
        f"実装どおりの安全なstrict filterの最大棄却率は {strict_max:.3f} で、"
        "この時期にはどのモデルも選択圧を発生させませんでした。",
        f"安全gateを外した順位診断では、{last_target} cell時点の上位10%で最良だった"
        f" `{best_ranked['model_kind']}` が reject precision="
        f"{_format(best_ranked.get('reject_precision_mean'))}, duplicate rejection="
        f"{_format(best_ranked.get('duplicate_rejection_rate_mean'))}, unknown retention="
        f"{_format(best_ranked.get('unknown_retention_mean'))}, survivor novelty lift="
        f"{_format(best_ranked.get('survivor_novelty_lift_mean'))}, valuable retention="
        f"{_format(best_ranked.get('valuable_retention_mean'))} でした。弱い順位情報は"
        "見えますが、初期filterとして安全に使える強さではありません。",
        f"全{total_panel}候補中、終端cellを確定できたvalid候補は{total_valid}件でした。"
        "filter precisionはこのvalid集合に条件付けています。表・図の集約値は候補を"
        "poolした比率ではなく、各seed指標の単純平均です。",
        "",
        "## 終端cell分類",
        "",
        "|cells|model|snapshot step|train macros|panel known coverage|known top-1|open-world top-1|Brier|ECE|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        aggregate_model,
        key=lambda item: (int(item["target_cells"]), str(item["model_kind"])),
    ):
        lines.append(
            "|{target}|{model}|{step}|{macros}|{coverage}|{accuracy}|{open_world}|{brier}|{ece}|".format(
                target=int(row["target_cells"]),
                model=row["model_kind"],
                step=_format(row.get("snapshot_step_mean"), 0),
                macros=_format(row.get("train_macros_mean"), 1),
                coverage=_format(row.get("panel_known_target_coverage_mean")),
                accuracy=_format(row.get("known_top1_accuracy_mean")),
                open_world=_format(row.get("open_world_top1_accuracy_mean")),
                brier=_format(row.get("known_brier_mean")),
                ece=_format(row.get("known_ece_mean")),
            )
        )
    lines.extend(
        [
            "",
            "top-1/Brier/ECEの分母は、snapshot archiveに存在する既知cellへ"
            "実際に到達したheld-out候補です。archiveにはあるが教師に未登場の"
            "cellも誤答・確率0として数える固定label-space評価です。",
            "ただし2/3/4 cellでは利用可能なStateNodeと候補panel自体が変わるため、"
            "この横軸は同一test setの通常のlearning curveではなく、各時点で実際に"
            "働く予測性能です。従って単調増加は仮定しません。open-world top-1は"
            "validな未知cell候補をすべて誤答として同じ分母へ含めています。",
            "",
            "## 実装どおりのstrict filter",
            "",
            "|cells|model|rejectable|reject frac|precision|duplicate rejection|unknown retention|novelty lift|valuable retention|useless precision|",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    strict = [row for row in aggregate_filter if str(row["policy"]).startswith("strict_")]
    for row in sorted(
        strict,
        key=lambda item: (
            int(item["target_cells"]),
            str(item["model_kind"]),
            str(item["policy"]),
        ),
    ):
        lines.append(
            "|{target}|{model}|{policy}|{fraction}|{precision}|{duplicate}|{retention}|{lift}|{valuable}|{useless}|".format(
                target=int(row["target_cells"]),
                model=row["model_kind"],
                policy=str(row["policy"]).replace("strict_", ""),
                fraction=_format(row.get("actual_reject_fraction_mean")),
                precision=_format(row.get("reject_precision_mean")),
                duplicate=_format(row.get("duplicate_rejection_rate_mean")),
                retention=_format(row.get("unknown_retention_mean")),
                lift=_format(row.get("survivor_novelty_lift_mean")),
                valuable=_format(row.get("valuable_retention_mean")),
                useless=_format(row.get("useless_reject_precision_mean")),
            )
        )
    lines.extend(
        [
            "",
            "`all_known`はcoverageだけを目的に全既知cellを棄却可能とした版、"
            "`feasible`は現行main9と同じくcollision-free eliteを持つcellだけを"
            "棄却可能とした版です。20% epsilon-randomは候補固有の判定ではない"
            "ため、この表には混ぜていません。実運用時は表の選択圧がさらに80%"
            "へ薄まります。",
            "",
            "`filter_selection_pressure.png`の線は安全閾値ではなく、support gate"
            "を掛けないraw class-confidenceを順位付けして上位5–50%を仮に除いた"
            "診断です。したがって"
            "線が改善してstrict記号が動かない場合は『順位情報はあるが、安全に"
            "発火できる教師量ではない』ことを意味します。",
            "`valuable`は現行dual archiveでnew kinematic/new feasible/elite replacement"
            "になる候補です。現在のquality keyに入っていないstrainや同一cell内の微小な"
            "節点座標差まで含む、将来の価値関数一般を表す指標ではありません。",
            "",
            "## Files",
            "",
            "- `classification_diagnostics.png`: top-1、Brier、ECE",
            "- `calibration_reliability.png`: 予測confidenceと実正解率のreliability曲線",
            "- `snapshot_filter_summary.png`: 教師macro数と主要分類・filter指標の要約",
            "- `filter_selection_pressure.png`: duplicate/unknown指標に加えvaluable retentionとuseless-reject precision",
            "- `snapshot_metrics.csv`: snapshot/model単位の分類指標",
            "- `filter_metrics.csv`: strictおよびranked診断",
            "- `calibration_bins.csv`: reliability-binの生データ",
            "- `panel_predictions.csv`: 全候補の予測とcounterfactual真値（seed directory）",
        ]
    )
    (output_directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_results(seed_directories: Sequence[Path], output_directory: Path) -> Path:
    directories = _discover_seed_directories(list(seed_directories))
    model_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for directory in directories:
        model_rows.extend(_read_csv(directory / "snapshot_metrics.csv"))
        filter_rows.extend(_read_csv(directory / "filter_metrics.csv"))
        calibration_path = directory / "calibration_bins.csv"
        if calibration_path.exists():
            calibration_rows.extend(_read_csv(calibration_path))
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "snapshot_metrics_all_seeds.csv", model_rows)
    _write_csv(output_directory / "filter_metrics_all_seeds.csv", filter_rows)
    _write_csv(output_directory / "calibration_bins_all_seeds.csv", calibration_rows)
    aggregate_model = _aggregate_rows(model_rows, ("target_cells", "model_kind"))
    aggregate_filter = _aggregate_rows(
        filter_rows,
        ("target_cells", "model_kind", "policy", "requested_reject_fraction"),
    )
    _write_csv(output_directory / "snapshot_metrics_aggregate.csv", aggregate_model)
    _write_csv(output_directory / "filter_metrics_aggregate.csv", aggregate_filter)
    _plot_results(model_rows, filter_rows, calibration_rows, output_directory)
    _write_report(directories, model_rows, filter_rows, output_directory)
    report = {
        "seed_directories": [str(path) for path in directories],
        "seed_count": len(directories),
        "snapshot_rows": len(model_rows),
        "filter_rows": len(filter_rows),
        "snapshot_aggregate": aggregate_model,
        "filter_aggregate": aggregate_filter,
    }
    (output_directory / "report.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="replay one random seed and evaluate snapshots")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--replay-budget", type=int, default=5000)
    run.add_argument("--targets", type=int, nargs="+", default=list(DEFAULT_TARGETS))
    run.add_argument(
        "--panel-size",
        type=int,
        default=128,
        help="candidates per snapshot; zero evaluates one complete pool",
    )
    run.add_argument("--panel-rollout-budget", type=int, default=500)
    run.add_argument("--panel-seed-base", type=int, default=2026081700)
    run.add_argument("--n-estimators", type=int, default=64)
    run.add_argument("--knn-neighbors", type=int, default=5)
    run.add_argument(
        "--rank-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_RANK_FRACTIONS),
    )
    run.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "main9_snapshot_filter_evaluation",
    )
    aggregate = subparsers.add_parser("aggregate", help="aggregate completed seed directories")
    aggregate.add_argument("inputs", type=Path, nargs="+")
    aggregate.add_argument(
        "--output-directory",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "main9_snapshot_filter_evaluation" / "aggregate",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        if args.replay_budget <= 0 or args.n_estimators <= 0 or args.knn_neighbors <= 0:
            raise ValueError("budgets, estimators, and kNN neighbors must be positive")
        run_replay(args)
    else:
        output = aggregate_results(args.inputs, args.output_directory.resolve())
        print(f"wrote aggregate snapshot evaluation to {output}")


if __name__ == "__main__":
    main()
