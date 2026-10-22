"""Freeze the offline spectral-subspace design evidence.

This standalone finalizer intentionally does not import the live learner,
runner, analyzer, or spectral candidate.  It recomputes the coordinate-only
design statistics directly from immutable saved path files.  The resulting
authority is descriptive offline evidence; it is not a training/adoption
authority and it does not claim that a spectral event is a strict success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUT = (
    ROOT
    / "experiments_ori6/analysis/"
    "spectral_subspace_offline_design_authority.json"
)

SCHEMA_VERSION = 1
AUTHORITY_TYPE = "ori6_spectral_subspace_offline_design"
RELATIVE_EPSILON = 1.0e-12
SCREEN_WEIGHT = 0.05

TASK_DIMENSION = {
    "five_d2": 1,
    "five_d3": 1,
    "torus2": 2,
    "water1": 2,
    "watersqrt2": 2,
}

DATASET_ROOTS = (
    ROOT / "experiments_ori5/tuning/r1",
    ROOT / "experiments_ori5/tuning/r2",
    ROOT / "experiments_ori5/full",
    ROOT / "experiments_ori6/screen/rescue/P0X2",
    ROOT / "experiments_ori6/screen/rescue/P0X2B12",
)

EXPECTED_RAW_COUNTS = {
    "five_d2": {"success": 12, "failure": 22},
    "five_d3": {"success": 19, "failure": 16},
    "torus2": {"success": 6, "failure": 45},
    "water1": {"success": 3, "failure": 43},
    "watersqrt2": {"success": 20, "failure": 20},
}
EXPECTED_UNIQUE_COUNTS = {
    "five_d2": {"success": 12, "failure": 18},
    "five_d3": {"success": 17, "failure": 14},
    "torus2": {"success": 5, "failure": 39},
    "water1": {"success": 2, "failure": 37},
    "watersqrt2": {"success": 19, "failure": 15},
}

FORMULA = {
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
    "rollback": [
        "correction_failed",
        "no_modes",
        "self_crossing",
        "wrong_flat",
    ],
    "rejected_crossing": (
        "last-safe state is restored; spectral transition reward is zero"
    ),
}
FORMULA_SHA256 = hashlib.sha256(
    json.dumps(
        FORMULA,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

EXPECTED_SOURCE_SHA256 = {
    "ori_codex6.py": (
        "38a1f6ac1466088b3e208fe9000b32f52cecb4c2537d0da591458bafea851c63"
    ),
    "ori_codex6_3d.py": (
        "2f065459de5022248ab1bb7fada04cf317bb875151bb5128d88208152ab8e46d"
    ),
    "ori_codex6_rejected_crossing.py": (
        "0040a552693135ae7d48655865f2535d5fa5f10321a4b235ab006cd34f6f5a94"
    ),
    "ori_codex6_rejected_crossing_3d.py": (
        "7e9d78bb0cf73b123b57f458dd29b1c7a94baca1414a2451fdedfbec24942ea2"
    ),
    "ori_codex6_spectral_subspace.py": (
        "4e309f92a412742b6889d44708409be21dd46afbf1066338d56e231e186bd998"
    ),
    "ori_codex6_spectral_subspace_3d.py": (
        "83a3d1cf69e1e0f4efbd0b3177e37d8d3e690b2713e906da01d756e3ec1aa953"
    ),
    "test_ori_codex6_spectral_subspace.py": (
        "c1b0d5d91ccb8c4a156514585bd4ef77ee2170072168d38aed471ee6a09ae98c"
    ),
}

EXPECTED_FORMULA_SHA256 = (
    "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6"
)

RC_AUTHORITY = (
    ROOT
    / "experiments_ori6/analysis/"
    "crossing_continuation_diagnostic_P0X2RC_reject_report.json"
)
EXPECTED_RC_AUTHORITY_SHA256 = (
    "f520049575f0eb54b582107688bd9cada51e38f57aa64a9a28b8a1a7127d0489"
)
EXPECTED_RC_AUTHORITY_FINGERPRINT = (
    "09e6c608d4fee4ffada426531976550050c900d21c985e8adad54b4b0c2afba9"
)

HARD_EVENT_SPECIFICATION = {
    "purpose": "diagnostic_only_not_reward_not_success_gate",
    "closure": "some strictly earlier rho/rho0 < 0.1",
    "reopen": "current absolute rho > 0.3",
    "turn": "current delta > 0.6",
    "closure_ratio_threshold": 0.1,
    "absolute_reopen_threshold": 0.3,
    "projector_distance_threshold": 0.6,
}

CMAES_SUCCESS_PATHS = {
    "history3",
    "history4",
    "randomize+gravity_potential_history[2]",
    "randomize+gravity_potential_history[4]",
    "randomize+gravity_potential_history[8]",
    "randomize+gravity_potential_history[9]",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        require(
            resolved.is_relative_to(WORKSPACE),
            f"evidence path escapes workspace: {resolved}",
        )
        return "../" + resolved.relative_to(WORKSPACE).as_posix()


def numeric_files(directory: Path, actor_path: bool) -> list[Path]:
    pattern = re.compile(r"step(\d+)\.dat" if actor_path else r"(\d+)\.dat")
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob("*.dat"):
        match = pattern.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    require(indexed, f"no numeric states in {relative(directory)}")
    require(
        [index for index, _ in indexed] == list(range(len(indexed))),
        f"non-contiguous state numbering in {relative(directory)}",
    )
    return [path for _, path in indexed]


def file_inventory_digest(paths: list[Path]) -> tuple[str, int, int]:
    inventory = [
        {
            "name": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return (
        canonical_digest(inventory),
        len(inventory),
        sum(row["size_bytes"] for row in inventory),
    )


def task_for_path(path: Path) -> str:
    matches = [part for part in path.parts if part in TASK_DIMENSION]
    require(len(matches) == 1, f"cannot determine task for {relative(path)}")
    return matches[0]


def geometry(
    node: np.ndarray,
    k: int,
    reference_projector: np.ndarray | None = None,
) -> dict[str, Any]:
    coordinates = np.asarray(node, dtype=float)
    require(coordinates.ndim == 2, "saved coordinates are not a matrix")
    require(np.all(np.isfinite(coordinates)), "saved coordinates are nonfinite")
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    require(1 <= k < len(singular_values), "invalid saved-path subspace dimension")
    selected = left[:, :k]
    projector = selected @ selected.T
    a = float(singular_values[k - 1])
    b = float(singular_values[k])
    singular_sum = float(a + b)
    gap = float(max(a - b, 0.0))
    sigma0 = float(singular_values[0])
    threshold = float(RELATIVE_EPSILON * max(singular_sum, sigma0))
    degenerate = bool(singular_sum == 0.0 or gap <= threshold)
    rho = (
        0.0
        if degenerate
        else float(gap / (singular_sum * (1.0 + RELATIVE_EPSILON)))
    )
    if reference_projector is None:
        delta = 0.0
    else:
        require(
            reference_projector.shape == projector.shape,
            "reference projector shape mismatch",
        )
        delta = float(
            np.linalg.norm(projector - reference_projector, ord="fro")
            / math.sqrt(2.0 * k)
        )
        delta = float(np.clip(delta, 0.0, 1.0))
    potential = float(rho * delta)
    require(
        all(math.isfinite(value) for value in (rho, delta, potential)),
        "nonfinite spectral metric",
    )
    return {
        "projector": projector,
        "rho": rho,
        "delta": delta,
        "potential": potential,
        "gap_degenerate": int(degenerate),
    }


def first_prefix_fraction(values: list[float], fraction: float) -> float | None:
    peak = max(values)
    if peak <= 0.0:
        return None
    threshold = float(fraction * peak)
    index = next(index for index, value in enumerate(values) if value >= threshold)
    return float(index / max(len(values) - 1, 1))


def path_metrics(states: list[np.ndarray], k: int) -> dict[str, Any]:
    shape = tuple(np.asarray(states[0]).shape)
    require(all(tuple(np.asarray(state).shape) == shape for state in states), "state shape changed within path")
    reference = geometry(states[0], k)["projector"]
    rows = [geometry(state, k, reference) for state in states]
    rho = [float(row["rho"]) for row in rows]
    delta = [float(row["delta"]) for row in rows]
    potential = [float(row["potential"]) for row in rows]
    require(potential[0] == 0.0, "reset-referenced potential is not exactly zero")

    initial_rho = rho[0]
    require(initial_rho > 0.0, "initial rho is zero; diagnostic ratio undefined")
    closure_seen = False
    event_index: int | None = None
    for index in range(len(rows)):
        if (
            closure_seen
            and rho[index] > HARD_EVENT_SPECIFICATION["absolute_reopen_threshold"]
            and delta[index] > HARD_EVENT_SPECIFICATION["projector_distance_threshold"]
        ):
            event_index = index
            break
        if (
            rho[index] / initial_rho
            < HARD_EVENT_SPECIFICATION["closure_ratio_threshold"]
        ):
            closure_seen = True

    increments = np.diff(np.asarray(potential, dtype=float))
    positive_count = int(np.count_nonzero(increments > 1.0e-6))
    frontier = -math.inf
    frontier_updates = 0
    for value in potential:
        if value > frontier + 1.0e-12:
            frontier = value
            frontier_updates += 1
    return {
        "state_shape": list(shape),
        "state_count": len(states),
        "transition_count": len(states) - 1,
        "initial_rho": initial_rho,
        "minimum_rho": min(rho),
        "maximum_rho": max(rho),
        "maximum_delta": max(delta),
        "terminal_rho": rho[-1],
        "terminal_delta": delta[-1],
        "peak_potential": max(potential),
        "terminal_potential": potential[-1],
        "positive_increment_count_gt_1e_6": positive_count,
        "positive_increment_fraction": float(
            positive_count / max(len(increments), 1)
        ),
        "running_frontier_update_count_gt_1e_12": frontier_updates,
        "peak_prefix_fraction": {
            str(percent): first_prefix_fraction(potential, percent / 100.0)
            for percent in (25, 50, 75, 90)
        },
        "hard_event": event_index is not None,
        "hard_event_first_state_index": event_index,
        "gap_degenerate_state_count": sum(
            int(row["gap_degenerate"]) for row in rows
        ),
        "nonfinite_metric_count": 0,
    }


def actor_path_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action_paths = sorted(
        actions
        for dataset_root in DATASET_ROOTS
        for actions in dataset_root.rglob("actions.json")
        if any(actions.parent.glob("step*.dat"))
    )
    require(len(action_paths) == 206, f"expected 206 actor paths, found {len(action_paths)}")

    raw_records: list[dict[str, Any]] = []
    unique_groups: dict[str, list[dict[str, Any]]] = {}
    for actions_path in action_paths:
        directory = actions_path.parent
        actions = load_json(actions_path)
        require(isinstance(actions, list) and actions, f"invalid actions payload: {relative(actions_path)}")
        require(all(isinstance(row, dict) for row in actions), f"invalid action row: {relative(actions_path)}")
        state_files = numeric_files(directory, actor_path=True)
        require(
            len(state_files) == len(actions) + 1,
            f"actions/state count mismatch: {relative(directory)}",
        )
        task = task_for_path(directory)
        terminal_reason = str(actions[-1].get("termination_reason"))
        label = "success" if terminal_reason == "success" else "failure"
        action_hash = file_sha256(actions_path)
        state_digest, state_count, state_bytes = file_inventory_digest(state_files)
        states = [np.loadtxt(path) for path in state_files]
        metrics = path_metrics(states, TASK_DIMENSION[task])
        archived_return = float(sum(float(row["reward"]) for row in actions))
        require(math.isfinite(archived_return), "nonfinite archived return")
        retained = terminal_reason in {"success", "timeout"}
        candidate_bonus = float(
            SCREEN_WEIGHT * metrics["terminal_potential"] if retained else 0.0
        )
        candidate_return = float(archived_return + candidate_bonus)
        record = {
            "path": relative(directory),
            "task": task,
            "label": label,
            "terminal_reason": terminal_reason,
            "actions_sha256": action_hash,
            "actions_size_bytes": actions_path.stat().st_size,
            "action_count": len(actions),
            "state_count": state_count,
            "state_files_total_bytes": state_bytes,
            "state_file_inventory_sha256": state_digest,
            "path_content_fingerprint": canonical_digest({
                "actions_sha256": action_hash,
                "state_file_inventory_sha256": state_digest,
            }),
            "archived_return": archived_return,
            "lambda_0p05_terminal_retained": retained,
            "lambda_0p05_spectral_bonus": candidate_bonus,
            "lambda_0p05_counterfactual_return": candidate_return,
            "metrics": metrics,
        }
        raw_records.append(record)
        unique_groups.setdefault(action_hash, []).append(record)

    raw_counts = {
        task: {
            label: sum(
                row["task"] == task and row["label"] == label
                for row in raw_records
            )
            for label in ("success", "failure")
        }
        for task in TASK_DIMENSION
    }
    require(raw_counts == EXPECTED_RAW_COUNTS, f"raw label counts changed: {raw_counts}")
    require(len(unique_groups) == 178, f"expected 178 unique actions, found {len(unique_groups)}")

    unique_records: list[dict[str, Any]] = []
    for action_hash, group in sorted(unique_groups.items()):
        identities = {(row["task"], row["label"], row["terminal_reason"]) for row in group}
        require(len(identities) == 1, f"duplicate action hash has conflicting identity: {action_hash}")
        content_fingerprints = sorted({row["path_content_fingerprint"] for row in group})
        representatives = sorted(group, key=lambda row: row["path"])
        representative = representatives[0]
        # Coordinate files are hashed separately.  If textual formatting ever
        # differs under an identical action trace, preserve every variant in
        # the authority instead of silently treating them as byte-identical.
        metric_fingerprints = sorted({
            canonical_digest(row["metrics"]) for row in group
        })
        require(
            len(metric_fingerprints) == 1,
            f"duplicate action trace has different spectral metrics: {action_hash}",
        )
        unique_records.append({
            "actions_sha256": action_hash,
            "task": representative["task"],
            "label": representative["label"],
            "terminal_reason": representative["terminal_reason"],
            "duplicate_count": len(group),
            "raw_paths": [row["path"] for row in representatives],
            "path_content_fingerprints": content_fingerprints,
            "representative_path": representative["path"],
            "representative_state_file_inventory_sha256": (
                representative["state_file_inventory_sha256"]
            ),
            "archived_return": representative["archived_return"],
            "lambda_0p05_terminal_retained": (
                representative["lambda_0p05_terminal_retained"]
            ),
            "lambda_0p05_spectral_bonus": (
                representative["lambda_0p05_spectral_bonus"]
            ),
            "lambda_0p05_counterfactual_return": (
                representative["lambda_0p05_counterfactual_return"]
            ),
            "metrics": representative["metrics"],
        })

    unique_counts = {
        task: {
            label: sum(
                row["task"] == task and row["label"] == label
                for row in unique_records
            )
            for label in ("success", "failure")
        }
        for task in TASK_DIMENSION
    }
    require(
        unique_counts == EXPECTED_UNIQUE_COUNTS,
        f"unique label counts changed: {unique_counts}",
    )
    return raw_records, unique_records


def auc(success: list[float], failure: list[float]) -> float:
    require(success and failure, "AUC requires both classes")
    wins = 0.0
    for positive in success:
        for negative in failure:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return float(wins / (len(success) * len(failure)))


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    require(array.size > 0 and np.all(np.isfinite(array)), "invalid quantile input")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def threshold_classification(
    records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    success = [row for row in records if row["label"] == "success"]
    failure = [row for row in records if row["label"] == "failure"]
    true_positive = sum(row["metrics"]["peak_potential"] >= threshold for row in success)
    false_positive = sum(row["metrics"]["peak_potential"] >= threshold for row in failure)
    return {
        "threshold": threshold,
        "true_positive_count": true_positive,
        "success_count": len(success),
        "true_positive_rate": float(true_positive / len(success)),
        "false_positive_count": false_positive,
        "failure_count": len(failure),
        "false_positive_rate": float(false_positive / len(failure)),
    }


def aggregate_actor_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task in TASK_DIMENSION:
        task_rows = [row for row in records if row["task"] == task]
        success = [row for row in task_rows if row["label"] == "success"]
        failure = [row for row in task_rows if row["label"] == "failure"]
        success_peak = [row["metrics"]["peak_potential"] for row in success]
        failure_peak = [row["metrics"]["peak_potential"] for row in failure]
        success_old_return = [row["archived_return"] for row in success]
        failure_old_return = [row["archived_return"] for row in failure]
        success_new_return = [row["lambda_0p05_counterfactual_return"] for row in success]
        failure_new_return = [row["lambda_0p05_counterfactual_return"] for row in failure]
        success_event = sum(row["metrics"]["hard_event"] for row in success)
        failure_event = sum(row["metrics"]["hard_event"] for row in failure)
        per_task[task] = {
            "success_count": len(success),
            "failure_count": len(failure),
            "success_peak_potential": quantiles(success_peak),
            "failure_peak_potential": quantiles(failure_peak),
            "peak_potential_auc": auc(success_peak, failure_peak),
            "archived_return_auc": auc(success_old_return, failure_old_return),
            "lambda_0p05_counterfactual_return_auc": auc(
                success_new_return,
                failure_new_return,
            ),
            "lambda_0p05_success_return": quantiles(success_new_return),
            "lambda_0p05_failure_return": quantiles(failure_new_return),
            "hard_event": {
                "success_count": success_event,
                "failure_count": failure_event,
                "success_rate": float(success_event / len(success)),
                "false_positive_rate": float(failure_event / len(failure)),
            },
            "success_positive_increment_fraction": quantiles([
                row["metrics"]["positive_increment_fraction"]
                for row in success
            ]),
            "success_running_frontier_update_count": quantiles([
                float(row["metrics"]["running_frontier_update_count_gt_1e_12"])
                for row in success
            ]),
            "success_peak_prefix_fraction_median": {
                percent: float(statistics.median([
                    row["metrics"]["peak_prefix_fraction"][percent]
                    for row in success
                    if row["metrics"]["peak_prefix_fraction"][percent] is not None
                ]))
                for percent in ("25", "50", "75", "90")
            },
        }
        require(per_task[task]["peak_potential_auc"] == 1.0, f"{task}: peak AUC changed")
        require(per_task[task]["archived_return_auc"] == 1.0, f"{task}: archived return ordering changed")
        require(per_task[task]["lambda_0p05_counterfactual_return_auc"] == 1.0, f"{task}: lambda ordering changed")

    all_success = [row["metrics"]["peak_potential"] for row in records if row["label"] == "success"]
    all_failure = [row["metrics"]["peak_potential"] for row in records if row["label"] == "failure"]
    return {
        "per_task": per_task,
        "overall_peak_potential_auc": auc(all_success, all_failure),
        "common_threshold_diagnostics": {
            "peak_ge_0p6": threshold_classification(records, 0.6),
            "peak_ge_0p1": threshold_classification(records, 0.1),
        },
        "interpretation": (
            "Per-task ordering is perfect in this selected archive, but the "
            "overall AUC is lower because legitimate success signal scales "
            "differ by task. No common hard threshold is adopted."
        ),
    }


def cmaes_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = WORKSPACE / "02_4unit_torus_CMAES"
    directories = [root / f"history{index}" for index in range(1, 5)]
    randomized_root = root / "randomize+gravity_potential_history2"
    directories.extend(
        randomized_root / f"randomize+gravity_potential_history[{index}]"
        for index in range(10)
    )
    require(all(path.is_dir() for path in directories), "CMAES path is absent")

    records: list[dict[str, Any]] = []
    for directory in directories:
        state_files = numeric_files(directory, actor_path=False)
        state_digest, state_count, state_bytes = file_inventory_digest(state_files)
        states = [np.loadtxt(path) for path in state_files]
        metrics = path_metrics(states, 2)
        identity = directory.name
        label = "success" if identity in CMAES_SUCCESS_PATHS else "failure"
        records.append({
            "path": relative(directory),
            "identity": identity,
            "label": label,
            "classification_provenance": (
                "manual_original_CMAES_objective_classification"
            ),
            "state_count": state_count,
            "state_files_total_bytes": state_bytes,
            "state_file_inventory_sha256": state_digest,
            "metrics": metrics,
        })
    records.sort(key=lambda row: row["path"])
    success = [row for row in records if row["label"] == "success"]
    failure = [row for row in records if row["label"] == "failure"]
    require(len(success) == 6 and len(failure) == 8, "CMAES labels changed")
    event_success = sum(row["metrics"]["hard_event"] for row in success)
    event_failure = sum(row["metrics"]["hard_event"] for row in failure)
    peak_0p8_success = sum(
        row["metrics"]["peak_potential"] >= 0.8 for row in success
    )
    peak_0p8_failure = sum(
        row["metrics"]["peak_potential"] >= 0.8 for row in failure
    )
    require((event_success, event_failure) == (6, 2), "CMAES event counts changed")
    require(
        (peak_0p8_success, peak_0p8_failure) == (6, 0),
        "CMAES peak>=0.8 counts changed",
    )
    aggregate = {
        "path_count": len(records),
        "success_count": len(success),
        "failure_count": len(failure),
        "peak_potential_auc": auc(
            [row["metrics"]["peak_potential"] for row in success],
            [row["metrics"]["peak_potential"] for row in failure],
        ),
        "hard_event_success_count": event_success,
        "hard_event_failure_count": event_failure,
        "peak_ge_0p8_success_count": peak_0p8_success,
        "peak_ge_0p8_failure_count": peak_0p8_failure,
        "interpretation": (
            "This is an external optimizer-selected diagnostic with manual "
            "original-objective labels and heterogeneous randomized state0; "
            "it is not training evidence or a success gate."
        ),
    }
    return records, aggregate


def source_binding() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = ROOT / name
        require(path.is_file(), f"bound source is absent: {name}")
        actual = file_sha256(path)
        require(actual == expected, f"bound source hash changed: {name}")
        files[name] = {
            "path": name,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    require(
        FORMULA_SHA256 == EXPECTED_FORMULA_SHA256,
        "offline formula no longer matches the stable implementation formula",
    )
    return {
        "files": files,
        "generator": {
            "path": relative(Path(__file__)),
            "sha256": file_sha256(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
        },
        "implementation_formula_sha256": EXPECTED_FORMULA_SHA256,
        "existing_files_are_read_only_authorities": True,
    }


def rc_authority_binding() -> dict[str, Any]:
    require(RC_AUTHORITY.is_file(), "RC reject authority is absent")
    actual_sha = file_sha256(RC_AUTHORITY)
    require(
        actual_sha == EXPECTED_RC_AUTHORITY_SHA256,
        "RC reject authority file hash changed",
    )
    payload = load_json(RC_AUTHORITY)
    stored = payload.get("authority_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("authority_fingerprint", None)
    recomputed = canonical_digest(unsigned)
    require(stored == recomputed, "RC authority self digest is invalid")
    require(
        stored == EXPECTED_RC_AUTHORITY_FINGERPRINT,
        "RC authority fingerprint changed",
    )
    require(payload.get("decision") == "reject_candidate", "RC decision changed")
    return {
        "path": relative(RC_AUTHORITY),
        "sha256": actual_sha,
        "size_bytes": RC_AUTHORITY.stat().st_size,
        "authority_type": payload.get("authority_type"),
        "authority_fingerprint": stored,
        "decision": payload.get("decision"),
        "role": (
            "parent rejected-crossing screen; this offline design authority "
            "does not overturn its rejection"
        ),
    }


DERIVED_ROLLOUT_EVIDENCE = (
    {
        "task": "torus2",
        "candidate": "P0X2",
        "seed": 3101,
        "directory": "experiments_ori6/screen/rescue/P0X2/torus2/A/seed3101",
        "checkpoint_sha256": "11fe546bb727ae5e55eb1b2c30aad225ef5487f04ab920d851cca84dfee632db",
        "run_config_sha256": "dcf6fb999c131a925a88328832313bc85bac7fde6aab644b5e0853a56abade14",
        "rollout_seed_first": 7410000,
        "rollout_seed_last": 7410015,
        "rollout_count": 16,
        "closure_enter_count": 16,
        "event_complete_count": 10,
        "strict_success_count": 0,
        "termination_reason_counts": {"timeout": 16},
    },
    {
        "task": "torus2",
        "candidate": "P0X2",
        "seed": 3102,
        "directory": "experiments_ori6/screen/rescue/P0X2/torus2/A/seed3102",
        "checkpoint_sha256": "e138a6f7e1c7ffdb16e5fa02c0e4128ff909e87780dbbdc399a2df3bd712529a",
        "run_config_sha256": "f4a7be240958e3f56dc9b8446e7cb70524c846a599b2e7c24b32d4e74c2c0a71",
        "rollout_seed_first": 7410000,
        "rollout_seed_last": 7410015,
        "rollout_count": 16,
        "closure_enter_count": 0,
        "event_complete_count": 0,
        "strict_success_count": 0,
        "termination_reason_counts": {"timeout": 16},
    },
    {
        "task": "torus2",
        "candidate": "P0X2",
        "seed": 3103,
        "directory": "experiments_ori6/screen/rescue/P0X2/torus2/A/seed3103",
        "checkpoint_sha256": "0f3dc66197cf8bb37c226f3c3ca1106ff38fc1a42ff5970fd6d51785c599a4ec",
        "run_config_sha256": "c304ad7355c1d968501df445781c5f77aa52c03c80d7149e8e6edb48585e41ee",
        "rollout_seed_first": 7410000,
        "rollout_seed_last": 7410015,
        "rollout_count": 16,
        "closure_enter_count": 16,
        "event_complete_count": 16,
        "strict_success_count": 0,
        "termination_reason_counts": {"self_crossing": 5, "timeout": 11},
    },
    {
        "task": "water1",
        "candidate": "P0X2RC",
        "seed": 3201,
        "directory": "experiments_ori6/screen/crossing_continuation_diagnostic/P0X2RC/water1/B/seed3201",
        "checkpoint_sha256": "c0fb203ba599ca32fe232b4dd2e34ecb4cc0236615bbd3d03ca10f253d0437a2",
        "run_config_sha256": "61aa5eddca931ff49f1b5ae4a6cf6b8653f18f9ad3d76777221c917ec4be7d62",
        "rollout_seed_first": 7420200,
        "rollout_seed_last": 7420207,
        "rollout_count": 8,
        "closure_enter_count": 3,
        "event_complete_count": 0,
        "strict_success_count": 0,
        "termination_reason_counts": {"correction_failed": 1, "timeout": 7},
    },
    {
        "task": "water1",
        "candidate": "P0X2RC",
        "seed": 3202,
        "directory": "experiments_ori6/screen/crossing_continuation_diagnostic/P0X2RC/water1/B/seed3202",
        "checkpoint_sha256": "13189b197ff4aee35db0cd5c3792e704420b29f1a9075521bc9c498b0fbaf250",
        "run_config_sha256": "84b9fff583adf26540f1259db11a5968879e26fbdfce555efead82f76846ce0a",
        "rollout_seed_first": 7420200,
        "rollout_seed_last": 7420207,
        "rollout_count": 8,
        "closure_enter_count": 0,
        "event_complete_count": 0,
        "strict_success_count": 0,
        "termination_reason_counts": {"timeout": 4, "wrong_flat": 4},
    },
    {
        "task": "water1",
        "candidate": "P0X2RC",
        "seed": 3203,
        "directory": "experiments_ori6/screen/crossing_continuation_diagnostic/P0X2RC/water1/B/seed3203",
        "checkpoint_sha256": "1c361fc9da7eeb1933a2269b91e9c060c533b9d059413d2e3912d539a9bf038e",
        "run_config_sha256": "42c1b69143b954216db775003e80b30ccf476d77f4178361d79c7239b20fbe17",
        "rollout_seed_first": 7420200,
        "rollout_seed_last": 7420207,
        "rollout_count": 8,
        "closure_enter_count": 0,
        "event_complete_count": 0,
        "strict_success_count": 0,
        "termination_reason_counts": {"timeout": 7, "wrong_flat": 1},
    },
)


def derived_fixed_rollout_provenance() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for source in DERIVED_ROLLOUT_EVIDENCE:
        row = dict(source)
        directory = ROOT / row.pop("directory")
        checkpoint = directory / "checkpoint.pt"
        config_path = directory / "run_config.json"
        require(checkpoint.is_file() and config_path.is_file(), "derived rollout evidence absent")
        require(
            file_sha256(checkpoint) == row["checkpoint_sha256"],
            f"derived checkpoint hash changed: {relative(checkpoint)}",
        )
        require(
            file_sha256(config_path) == row["run_config_sha256"],
            f"derived run config hash changed: {relative(config_path)}",
        )
        config = load_json(config_path)
        require(int(config["seed"]) == row["seed"], "derived rollout seed binding changed")
        expected_final_seed = 7410000 if row["candidate"] == "P0X2" else 7420000
        require(
            int(config["final_evaluation_seed"]) == expected_final_seed,
            "derived rollout evaluation seed binding changed",
        )
        require(
            sum(row["termination_reason_counts"].values()) == row["rollout_count"],
            "derived rollout termination counts are inconsistent",
        )
        require(
            0 <= row["event_complete_count"] <= row["closure_enter_count"] <= row["rollout_count"],
            "derived rollout event counts are inconsistent",
        )
        row.update({
            "directory": relative(directory),
            "checkpoint_path": relative(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "run_config_path": relative(config_path),
            "run_config_size_bytes": config_path.stat().st_size,
            "configured_final_evaluation_seed": int(config["final_evaluation_seed"]),
        })
        records.append(row)
    return {
        "provenance_class": (
            "derived_from_ephemeral_read_only_rollouts_not_raw_authority"
        ),
        "raw_coordinate_sequences_saved": False,
        "recomputed_on_verify": False,
        "eligible_as_training_history_or_adoption_gate": False,
        "legacy_exploratory_event_definition": {
            "rho_cov": "(sigma2^2-sigma3^2)/sum_i sigma_i^2",
            "normalization": "rho_cov/rho_cov_at_state0",
            "closure": "some strictly earlier normalized rho_cov < 0.1",
            "reopen": "current normalized rho_cov > 0.3",
            "turn": "current node-space projector delta > 0.6",
            "warning": (
                "This differs from the adopted scale-free singular-gap rho "
                "and is retained only as explicitly derived provenance."
            ),
        },
        "records": records,
        "aggregate": {
            "torus2": {
                "rollout_count": 48,
                "closure_enter_count": 32,
                "event_complete_count": 26,
                "strict_success_count": 0,
            },
            "water1": {
                "rollout_count": 24,
                "closure_enter_count": 3,
                "event_complete_count": 0,
                "strict_success_count": 0,
            },
        },
    }


def weight_zero_contract() -> dict[str, Any]:
    fixture = ROOT / (
        "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
        "final_training_path/actions.json"
    )
    require(fixture.is_file(), "weight-zero replay fixture is absent")
    fixture_hash = file_sha256(fixture)
    require(
        fixture_hash == "0ae2882a30ed4f1afd35bf4debabfb49accc210148d352ec1bea867f4fa63375",
        "weight-zero replay fixture changed",
    )
    return {
        "weight": 0.0,
        "semantic_expectation": (
            "All parent mechanics, accepted/restored states, actions, rewards, "
            "termination, parent summary/evaluation fields, model parameters, "
            "actor and critic optimizer states, and Torch/NumPy RNG states are "
            "bit-exact to rejected-crossing ori6; additive spectral diagnostic "
            "columns may be present but contribute exactly zero reward."
        ),
        "actual_regression_test": {
            "path": "test_ori_codex6_spectral_subspace.py",
            "sha256": EXPECTED_SOURCE_SHA256["test_ori_codex6_spectral_subspace.py"],
            "coverage": [
                "saved 3D near-crossing action replay state/action/reward identity",
                "repeated rejected crossing and timeout boundary zero spectral delta",
                "one-episode planar Actor-Critic/PPO training parent CSV columns",
                "parent summary core and evaluation field equality",
                "policy model_state_dict equality",
                "actor_optimizer_state_dict equality",
                "critic_optimizer_state_dict equality",
                "Torch RNG state equality",
                "NumPy RNG state equality",
            ],
        },
        "saved_replay_fixture": {
            "path": relative(fixture),
            "sha256": fixture_hash,
            "size_bytes": fixture.stat().st_size,
        },
        "scope_warning": (
            "This is an executable regression contract, not a claim that a "
            "nonzero spectral weight is parent-bit-exact."
        ),
    }


def build_report(generated_utc: str) -> dict[str, Any]:
    raw_records, unique_records = actor_path_records()
    actor_aggregate = aggregate_actor_metrics(unique_records)
    cmaes, cmaes_aggregate = cmaes_records()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority_type": AUTHORITY_TYPE,
        "generated_utc": generated_utc,
        "decision": "pilot_worthy_not_adopted",
        "candidate": {
            "name": "ori6_RC_spectral_subspace_potential",
            "schema_version": "1.0.0",
            "parent": "ori6_rejected_crossing_continuation",
            "default_weight": SCREEN_WEIGHT,
            "relative_epsilon": RELATIVE_EPSILON,
        },
        "scope": {
            "evidence_type": "offline_saved_coordinate_path_design_screen",
            "tasks": list(TASK_DIMENSION),
            "actor_raw_path_count": len(raw_records),
            "actor_unique_actions_sha256_count": len(unique_records),
            "cmaes_external_path_count": len(cmaes),
            "training_executed_by_finalizer": False,
            "rollouts_executed_by_finalizer": False,
            "strict_success_claim_from_spectral_signal": False,
            "adoption_or_nonregression_claim": False,
        },
        "historical_binding": {
            "source": source_binding(),
            "parent_rc_reject_authority": rc_authority_binding(),
        },
        "spectral_formulation": {
            "formula": FORMULA,
            "formula_sha256": FORMULA_SHA256,
            "offline_screen_parameters": {
                "relative_epsilon": RELATIVE_EPSILON,
                "lambda": SCREEN_WEIGHT,
                "state_increment_threshold": 1.0e-6,
                "frontier_increment_threshold": 1.0e-12,
            },
            "normalization_and_invariance": {
                "centering_removes_translation": True,
                "rho_is_uniform_scale_invariant": True,
                "node_space_projector_is_invariant_to_global_orthogonal_ambient_transforms": True,
                "delta_range": [0.0, 1.0],
                "delta_normalizer": "sqrt(2*k)",
                "degenerate_gap_behavior": "rho=0 suppresses ambiguous projector changes",
                "current_state_markov": (
                    "Phi depends only on current coordinates and the fixed reset-state P0"
                ),
                "telescoping": (
                    "sum lambda*(Phi_after-Phi_before) = lambda*Phi_terminal "
                    "on accepted valid transitions from reset Phi=0"
                ),
            },
        },
        "hard_event_diagnostic": HARD_EVENT_SPECIFICATION,
        "actor_archive": {
            "dataset_roots": [relative(path) for path in DATASET_ROOTS],
            "inventory": {
                "raw_path_count": len(raw_records),
                "unique_actions_sha256_count": len(unique_records),
                "expected_raw_counts": EXPECTED_RAW_COUNTS,
                "expected_unique_counts": EXPECTED_UNIQUE_COUNTS,
                "raw_records_sha256": canonical_digest(raw_records),
                "unique_records_sha256": canonical_digest(unique_records),
            },
            "raw_path_records": raw_records,
            "actions_sha256_unique_records": unique_records,
            "aggregate": actor_aggregate,
        },
        "external_cmaes_diagnostic": {
            "records_sha256": canonical_digest(cmaes),
            "records": cmaes,
            "aggregate": cmaes_aggregate,
        },
        "derived_fixed_rollout_provenance": derived_fixed_rollout_provenance(),
        "weight_zero_contract": weight_zero_contract(),
        "conclusion": {
            "minimal_safe_candidate": (
                "Add only a small bounded accepted-valid current-state "
                "difference reward lambda*(Phi_after-Phi_before); retain all "
                "terminal flatness/spread rewards and rejected-crossing semantics."
            ),
            "why_not_hard_event": (
                "The hard event has false positives and needs trajectory memory; "
                "success signal scale also differs substantially across tasks."
            ),
            "evidence_summary": (
                "Peak Phi and lambda=0.05 counterfactual return preserve perfect "
                "within-task archived success/failure ordering across all five "
                "tasks, but selected-path bias, small hard-task positive samples, "
                "and late signal prevent an adoption claim."
            ),
        },
        "limitations": [
            "Saved paths are policy- and optimizer-selected, not an IID population.",
            "Unique hard-task positives are only torus2 n=5 and water1 n=2.",
            "Deduplication is by exact actions.json SHA-256; coordinate inventories remain separately bound.",
            "The spectral signal is often late and is not an early exploration oracle.",
            "Spectral switch evidence cannot replace flatness, spread, validity, or terminal success checks.",
            "RC endpoint CSVs do not contain per-step coordinates, so spectral events are not inferred from them.",
            "The fixed-rollout diagnostic raw coordinate sequences were not saved and are explicitly non-authoritative.",
        ],
        "self_digest_scheme": {
            "field": "authority_fingerprint",
            "algorithm": "sha256",
            "payload": "canonical JSON with authority_fingerprint omitted",
            "canonical_json": (
                "UTF-8, sorted keys, separators comma/colon, "
                "ensure_ascii=false, allow_nan=false"
            ),
        },
    }
    report["authority_fingerprint"] = canonical_digest(report)
    return report


def verify_self_digest(report: dict[str, Any]) -> None:
    stored = report.get("authority_fingerprint")
    require(isinstance(stored, str) and len(stored) == 64, "authority fingerprint absent")
    unsigned = dict(report)
    unsigned.pop("authority_fingerprint", None)
    require(canonical_digest(unsigned) == stored, "authority self digest mismatch")


def atomic_write_new(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite existing authority: {relative(path)}")
    temporary: Path | None = None
    try:
        for _ in range(100):
            candidate = path.parent / f".tmp-{secrets.token_hex(8)}"
            try:
                with candidate.open("x", encoding="utf-8", newline="\n") as handle:
                    json.dump(
                        report,
                        handle,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary = candidate
                break
            except FileExistsError:
                continue
        require(temporary is not None, "could not reserve a short authority temp file")
        require(not path.exists(), f"authority appeared during finalization: {relative(path)}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute from raw evidence and verify the existing authority",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        require(OUTPUT.is_file(), f"authority is absent: {relative(OUTPUT)}")
        stored = load_json(OUTPUT)
        require(isinstance(stored, dict), "stored authority is not an object")
        verify_self_digest(stored)
        generated_utc = stored.get("generated_utc")
        require(isinstance(generated_utc, str), "stored generated_utc is absent")
        rebuilt = build_report(generated_utc)
        require(
            canonical_bytes(rebuilt) == canonical_bytes(stored),
            "stored authority differs from independently rebuilt raw evidence",
        )
        print(
            json.dumps({
                "status": "verified",
                "path": relative(OUTPUT),
                "sha256": file_sha256(OUTPUT),
                "authority_fingerprint": stored["authority_fingerprint"],
            }, sort_keys=True)
        )
        return 0

    require(not OUTPUT.exists(), f"refusing to overwrite existing authority: {relative(OUTPUT)}")
    report = build_report(datetime.now(timezone.utc).isoformat())
    verify_self_digest(report)
    atomic_write_new(OUTPUT, report)
    print(
        json.dumps({
            "status": "created",
            "path": relative(OUTPUT),
            "sha256": file_sha256(OUTPUT),
            "authority_fingerprint": report["authority_fingerprint"],
        }, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
