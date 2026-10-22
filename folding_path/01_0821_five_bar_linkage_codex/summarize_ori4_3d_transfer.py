"""Build the compact, reproducible ori_codex4 three-dimensional report.

The report intentionally combines three different kinds of evidence without
silently treating them as interchangeable:

* selected training runs and their independent held-out rollouts;
* no-evaluation, common-budget first-discovery trials; and
* an optional periodic-selection replay used to compare the last and selected
  checkpoints on independent held-out rollouts.

No partial first-discovery report is written.  All three benchmark aggregates
must exist and declare ``complete: true`` before this program creates the
analysis directory.  Use ``--wait-seconds`` to poll for a bounded amount of
time, or leave it at zero to fail immediately and list the missing work.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from collision_3d import (
    COLLISION_BACKEND_NAME,
    COLLISION_BACKEND_VERSION,
    COLLISION_SEMANTICS,
    ContinuousBarCollisionChecker,
)


SCHEMA_VERSION = 1
ANALYZER_VERSION = "1.2.0"
EXPECTED_PILOT_RUN_COUNT = 24
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DISCOVERY_ROOT = (
    SCRIPT_DIRECTORY
    / "experiments_3d"
    / "first_discovery"
    / "screen_n10_b102400"
)
DEFAULT_PILOT_ROOT = SCRIPT_DIRECTORY / "experiments_3d" / "pilot"
DEFAULT_VALIDATION_ROOT = SCRIPT_DIRECTORY / "experiments_3d" / "validation"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIRECTORY / "experiments_3d" / "analysis"
METHODS = ("learning", "frozen", "uniform")


# These are analysis inputs, not outcomes selected by this script.  The raw
# requested diameters are explicitly marked as manual in results.json.  Pilot,
# discovery, validation, and feasibility quantities are read from artifacts.
TASK_SPECS = (
    {
        "task_id": "torus2",
        "display_name": "4-unit torus: diameter 4",
        "model_name": "4unit_torus",
        "selected_pilot": "torus2_C_b32_lr1e3_s107",
        "manual_requested_raw_diameter": 4.0,
        "feasibility_artifact": (
            SCRIPT_DIRECTORY
            / "experiments_3d"
            / "feasibility"
            / "torus_spread2_collision_free_path"
        ),
        "manual_feasibility_prior": (
            "The constructive torus spread-2 witness is independently "
            "rechecked from its persisted coordinates when available."
        ),
    },
    {
        "task_id": "water1",
        "display_name": "waterbomb: diameter 3",
        "model_name": "waterbomb",
        "selected_pilot": "water1_B_b8_lr5e4_s107",
        "manual_requested_raw_diameter": 3.0,
        "feasibility_artifact": (
            SCRIPT_DIRECTORY
            / "experiments_3d"
            / "feasibility"
            / "waterbomb_spread1_collision_free_path"
        ),
        "manual_feasibility_prior": (
            "The constructive waterbomb spread-1 witness is independently "
            "rechecked from its persisted coordinates when available."
        ),
    },
    {
        "task_id": "watersqrt2",
        "display_name": "waterbomb: diameter 3 sqrt(2)",
        "model_name": "waterbomb",
        "selected_pilot": "watersqrt2_B_b8_lr5e4_s107",
        "manual_requested_raw_diameter": 3.0 * math.sqrt(2.0),
        "manual_feasibility_prior": (
            "A flat branch near normalized spread sqrt(2) was reported by "
            "the pre-analysis geometry audit; strict successes in the current "
            "artifacts independently demonstrate reachability."
        ),
    },
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(directory):
    """Return a deterministic recursive file inventory and manifest hash."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        return {
            "status": "missing",
            "directory": str(directory),
            "file_count": 0,
            "manifest_sha256": None,
            "files": [],
        }
    files = []
    manifest = hashlib.sha256()
    for path in sorted(
        (candidate for candidate in directory.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(directory).as_posix(),
    ):
        relative = path.relative_to(directory).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        files.append({
            "relative_path": relative,
            "size_bytes": size,
            "sha256": digest,
        })
        manifest.update(relative.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")
    return {
        "status": "inventoried",
        "directory": str(directory),
        "file_count": len(files),
        "manifest_sha256": manifest.hexdigest(),
        "files": files,
    }


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(value, default=None):
    if value in (None, ""):
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric artifact value: {value!r}")
    return result


def as_int(value, default=None):
    number = as_float(value, default=None)
    return default if number is None else int(number)


def flatness_potential(node, subspace_dimension=2):
    coordinates = np.asarray(node, dtype=float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return float(np.sum(singular_values[int(subspace_dimension):] ** 2))


def projected_spread(node, coordinate_scale, subspace_dimension=2):
    coordinates = np.asarray(node, dtype=float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    dimension = min(int(subspace_dimension), right_vectors.shape[0])
    projected = centered @ right_vectors[:dimension].T
    differences = projected[:, np.newaxis, :] - projected[np.newaxis, :, :]
    diameter = float(np.max(np.linalg.norm(differences, axis=2)))
    return diameter / float(coordinate_scale)


def verify_constructive_path_artifact(spec, config):
    """Independently audit a persisted ``stepN.dat`` feasibility witness."""

    directory = Path(spec["feasibility_artifact"]).resolve()
    inventory = directory_manifest(directory)
    result = {
        "source_type": "constructive_path_artifact",
        "status": "missing" if not directory.is_dir() else "verification_failed",
        "directory": str(directory),
        "manifest_sha256": inventory["manifest_sha256"],
        "file_count": inventory["file_count"],
        "summary_sha256": None,
        "checks": {},
        "failed_checks": [],
    }
    if not directory.is_dir():
        return result

    try:
        summary_path = directory / "summary.json"
        actions_path = directory / "actions.json"
        summary = load_json(summary_path)
        actions = load_json(actions_path)
        outcome = summary["outcome"]
        environment = summary["environment"]
        collision_declaration = summary["collision"]
        saved_count = int(outcome["saved_node_count"])
        transition_count = int(outcome["transition_count"])
        expected_names = [f"step{index}.dat" for index in range(saved_count)]
        actual_step_paths = sorted(
            directory.glob("step*.dat"),
            key=lambda path: int(path.stem[4:]),
        )
        actual_names = [path.name for path in actual_step_paths]
        declared_step_hashes = summary["files"]["step_file_sha256"]
        nodes = [np.atleast_2d(np.loadtxt(path)) for path in actual_step_paths]

        model_directory = SCRIPT_DIRECTORY / "model" / spec["model_name"]
        model_paths = {
            "node_initial.dat": model_directory / "node_initial.dat",
            "member.dat": model_directory / "member.dat",
            "fix.dat": model_directory / "fix.dat",
        }
        initial_model = np.atleast_2d(np.loadtxt(
            model_paths["node_initial.dat"], delimiter=","
        ))
        member = np.atleast_2d(np.loadtxt(
            model_paths["member.dat"], delimiter=",", dtype=int
        ))
        fix = np.atleast_2d(np.loadtxt(
            model_paths["fix.dat"], delimiter=",", dtype=int
        ))
        initial_lengths = np.linalg.norm(
            initial_model[member[:, 0]] - initial_model[member[:, 1]], axis=1
        )
        coordinate_scale = float(np.max(initial_lengths))
        initial_potential = flatness_potential(initial_model)
        final_potential = flatness_potential(nodes[-1])
        final_F = final_potential / initial_potential
        final_D = projected_spread(nodes[-1], coordinate_scale)
        maximum_member_error = max(
            float(np.max(np.abs(
                np.linalg.norm(
                    node[member[:, 0]] - node[member[:, 1]], axis=1
                ) - initial_lengths
            )))
            for node in nodes
        )
        maximum_fixed_drift = max(
            abs(float(node[int(node_index), int(coordinate)])
                - float(initial_model[int(node_index), int(coordinate)]))
            for node in nodes
            for node_index, coordinate in fix
        )

        settings = collision_declaration["settings"]
        checker = ContinuousBarCollisionChecker(member, **settings)
        collision_results = [
            checker.check(before, after)
            for before, after in zip(nodes[:-1], nodes[1:])
        ]
        collision_count = sum(item.collision for item in collision_results)
        maximum_requested_width = max(
            (abs(float(action["requested_width"])) for action in actions),
            default=0.0,
        )
        maximum_accepted_width = max(
            (abs(float(action["accepted_width"])) for action in actions),
            default=0.0,
        )

        local_hashes = {
            name: sha256(path) for name, path in model_paths.items()
        }
        declared_model_hashes = summary["model"]["model_file_sha256"]
        adapter_hashes = config["three_dimensional_adapter"][
            "model_file_sha256"
        ]
        adapter_hashes_by_filename = {
            "node_initial.dat": adapter_hashes["node"],
            "member.dat": adapter_hashes["member"],
            "fix.dat": adapter_hashes["fix"],
        }
        persisted = outcome["persisted_step_file_round_trip"]
        checks = {
            "summary_and_actions_present": (
                summary_path.is_file() and actions_path.is_file()
            ),
            "contiguous_step_filenames": actual_names == expected_names,
            "saved_count_equals_transitions_plus_one": (
                saved_count == transition_count + 1
            ),
            "actions_equal_transition_count": len(actions) == transition_count,
            "step_hash_keyset_exact": (
                list(declared_step_hashes) == expected_names
            ),
            "step_hashes_exact": all(
                declared_step_hashes.get(path.name) == sha256(path)
                for path in actual_step_paths
            ),
            "actions_hash_exact": (
                summary["files"]["actions_sha256"] == sha256(actions_path)
            ),
            "model_hashes_match_local_files": declared_model_hashes == local_hashes,
            "model_hashes_match_selected_pilot": (
                adapter_hashes_by_filename == local_hashes
            ),
            "step0_matches_model_with_coordinate_rounding": (
                nodes[0].shape == initial_model.shape
                and np.allclose(nodes[0], initial_model, rtol=0.0, atol=5.0e-11)
            ),
            "all_coordinates_finite": all(
                np.all(np.isfinite(node)) for node in nodes
            ),
            "all_actions_mark_state_valid": all(
                int(action.get("after_state_valid", 0)) == 1
                for action in actions
            ),
            "requested_width_within_limit": (
                maximum_requested_width
                <= float(environment["max_step_width"]) + 1.0e-12
            ),
            "accepted_width_within_limit": (
                maximum_accepted_width
                <= float(environment["max_step_width"]) + 1.0e-12
            ),
            "member_lengths_within_strain_tolerance": (
                maximum_member_error <= float(environment["strain_tolerance"])
            ),
            "fixed_coordinates_within_strain_tolerance": (
                maximum_fixed_drift <= float(environment["strain_tolerance"])
            ),
            "recomputed_flatness_meets_threshold": (
                final_F <= float(environment["flatness_tolerance"])
            ),
            "recomputed_spread_meets_threshold": (
                abs(final_D - float(environment["target_spread_normalised"]))
                <= float(environment["spread_tolerance"])
            ),
            "recomputed_collision_count_zero": collision_count == 0,
            "declared_collision_count_zero": (
                int(collision_declaration["collision_count"]) == 0
                and int(collision_declaration[
                    "persisted_step_file_collision_count"
                ]) == 0
            ),
            "collision_backend_matches_current_source": (
                collision_declaration["backend"] == COLLISION_BACKEND_NAME
                and collision_declaration["backend_version"]
                == COLLISION_BACKEND_VERSION
                and collision_declaration["semantics"] == COLLISION_SEMANTICS
            ),
            "collision_settings_match_selected_pilot": (
                settings == config["three_dimensional_adapter"][
                    "collision_parameters"
                ]
            ),
            "collision_source_hash_matches_current_source": (
                summary["source_sha256"]["collision_3d.py"]
                == sha256(SCRIPT_DIRECTORY / "collision_3d.py")
            ),
            "recomputed_F_matches_round_trip_declaration": math.isclose(
                final_F,
                float(persisted["final_flatness_ratio_F"]),
                rel_tol=1.0e-8,
                abs_tol=1.0e-15,
            ),
            "recomputed_D_matches_round_trip_declaration": math.isclose(
                final_D,
                float(persisted["final_spread_normalised_D"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            "recomputed_member_error_matches_round_trip_declaration": math.isclose(
                maximum_member_error,
                float(persisted["maximum_member_length_error_over_path"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-14,
            ),
            "declared_outcome_is_success": (
                bool(outcome["success"])
                and outcome["termination_reason"] == "success"
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        result.update({
            "status": "verified" if not failed else "verification_failed",
            "summary_sha256": sha256(summary_path),
            "checks": checks,
            "failed_checks": failed,
            "declared": {
                "transition_count": transition_count,
                "final_flatness_ratio_F": outcome["final_flatness_ratio_F"],
                "final_spread_normalised_D": outcome[
                    "final_spread_normalised_D"
                ],
                "collision_count": collision_declaration["collision_count"],
                "target_spread_normalised": environment[
                    "target_spread_normalised"
                ],
                "target_planar_diameter_raw": environment[
                    "target_planar_diameter_raw"
                ],
            },
            "recomputed_from_step_files": {
                "initial_flatness_potential": initial_potential,
                "final_flatness_potential": final_potential,
                "final_flatness_ratio_F": final_F,
                "final_spread_normalised_D": final_D,
                "final_planar_diameter_raw": final_D * coordinate_scale,
                "maximum_member_length_error_over_path": maximum_member_error,
                "maximum_fixed_coordinate_drift": maximum_fixed_drift,
                "collision_count": collision_count,
                "maximum_requested_width": maximum_requested_width,
                "maximum_accepted_width": maximum_accepted_width,
            },
            "collision_replay": {
                "backend": COLLISION_BACKEND_NAME,
                "backend_version": COLLISION_BACKEND_VERSION,
                "semantics": COLLISION_SEMANTICS,
                "settings": settings,
                "checked_transitions": len(collision_results),
                "nonadjacent_member_pair_count": int(
                    len(checker.pair_indices)
                ),
            },
            "rigid_motion_artifact": summary.get("rigid_motion"),
            "rigid_motion_artifact_interpretation": (
                "Logging-only diagnostics for this constructive witness; "
                "not a policy held-out statistic and not used in success."
                if summary.get("rigid_motion") is not None else None
            ),
            "generation_method_artifact": summary.get("generation_method"),
            "upstream_source_artifact": summary.get("source_path"),
            "model_file_sha256": local_hashes,
        })
    except Exception as error:  # Preserve the report, but never claim verified.
        result.update({
            "status": "verification_error",
            "error_type": type(error).__name__,
            "error": str(error),
        })
    return result


def task_feasibility_evidence(spec, config, artifact_success):
    if "feasibility_artifact" in spec:
        return verify_constructive_path_artifact(spec, config)
    if "external_feasibility_source" in spec:
        inventory = directory_manifest(spec["external_feasibility_source"])
        return {
            "source_type": "external_CMAES_path_inventory_not_replayed",
            "status": (
                "inventoried_not_replayed"
                if inventory["status"] == "inventoried" else "missing"
            ),
            "directory": inventory["directory"],
            "file_count": inventory["file_count"],
            "manifest_sha256": inventory["manifest_sha256"],
            "files": inventory["files"],
            "manual_prior_note": spec["manual_feasibility_prior"],
        }
    return {
        "source_type": (
            "strict_success_records_in_current_artifacts"
            if artifact_success else "manual_pre_analysis_input"
        ),
        "status": (
            "constructively_demonstrated_by_strict_success_records"
            if artifact_success else "manual_prior_not_recomputed"
        ),
        "manual_prior_note": spec["manual_feasibility_prior"],
    }


def mean(rows, field):
    values = [as_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def count_reason(rows, reason):
    return sum(row.get("termination_reason") == reason for row in rows)


def wilson_interval(successes, trials, z=1.959963984540054):
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    if low < 1.0e-15:
        low = 0.0
    if 1.0 - high < 1.0e-15:
        high = 1.0
    return float(low), float(high)


def bootstrap_mean_interval(values, rng, replicates):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"point": None, "ci95_low": None, "ci95_high": None}
    if not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap input contains a non-finite value.")
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "point": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_figure(path, figure):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def visible_completed_trials(task_root):
    return sum(
        1
        for path in (Path(task_root) / "trials").glob("*/seed*/summary.json")
        if load_json(path).get("complete")
    )


def preflight(discovery_root, pilot_root):
    errors = []
    for spec in TASK_SPECS:
        task_root = Path(discovery_root) / spec["task_id"]
        aggregate_path = task_root / "aggregate.json"
        trials_path = task_root / "trials.csv"
        config_path = task_root / "run_config.json"
        if not aggregate_path.is_file():
            visible = visible_completed_trials(task_root) if task_root.exists() else 0
            errors.append(
                f"{spec['task_id']}: aggregate.json is missing "
                f"({visible} completed trial summaries currently visible)"
            )
            continue
        aggregate = load_json(aggregate_path)
        if not aggregate.get("complete"):
            errors.append(
                f"{spec['task_id']}: aggregate is incomplete "
                f"({aggregate.get('completed_trials')}/"
                f"{aggregate.get('expected_trials')}); missing="
                f"{aggregate.get('missing_method_seed_pairs')}"
            )
        if not trials_path.is_file():
            errors.append(f"{spec['task_id']}: trials.csv is missing")
        if not config_path.is_file():
            errors.append(f"{spec['task_id']}: run_config.json is missing")

        selected = Path(pilot_root) / spec["selected_pilot"]
        required = (
            "run_config.json",
            "summary.json",
            "training_history.csv",
            "evaluation_summary.csv",
            "best_policy_evaluation_summary.csv",
        )
        missing = [name for name in required if not (selected / name).is_file()]
        if missing:
            errors.append(
                f"{spec['task_id']}: selected pilot {selected.name} misses {missing}"
            )
    return errors


def wait_for_preflight(discovery_root, pilot_root, wait_seconds):
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        errors = preflight(discovery_root, pilot_root)
        if not errors:
            return
        if time.monotonic() >= deadline:
            message = [
                "REFUSING TO WRITE ANALYSIS: all three first-discovery "
                "aggregates must be complete."
            ]
            message.extend(f"  - {error}" for error in errors)
            raise RuntimeError("\n".join(message))
        time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))


def discovery_trial_records(spec, task_root, root_config, aggregate):
    """Return normalized per-trial records and validate the harness output."""

    raw_rows = load_csv(Path(task_root) / "trials.csv")
    expected = int(aggregate["expected_trials"])
    if len(raw_rows) != expected:
        raise ValueError(
            f"{spec['task_id']}: trials.csv has {len(raw_rows)} rows; "
            f"aggregate expects {expected}."
        )
    effective = root_config["effective_ori4_args"]
    target = float(effective["target_spread"])
    adapter = root_config.get("three_dimensional_adapter", {})
    # The first-discovery config stores adapter metadata at the top level in
    # current artifacts, while ordinary training stores it below one key.
    if not adapter:
        adapter = {
            "model_name": effective.get("selected_model"),
            "initial_maximum_member_length": None,
        }
    budget = int(aggregate["mechanics_budget"])
    episode_horizon = int(aggregate["common_episode_horizon"])
    normalized = []
    seen = set()
    for row in raw_rows:
        method = row["method"]
        seed = as_int(row["seed"])
        pair = (method, seed)
        if method not in METHODS or pair in seen:
            raise ValueError(f"Invalid or duplicate discovery trial {pair}.")
        seen.add(pair)
        success = as_int(row["success"], 0)
        first_episode = as_int(row.get("first_success_episode"))
        first_steps = as_int(row.get("first_success_mechanical_steps"))
        episode_event = int(
            bool(success)
            and first_episode is not None
            and first_episode <= episode_horizon
        )
        normalized.append({
            "task_id": spec["task_id"],
            "model_name": spec["model_name"],
            "target_spread": target,
            "method": method,
            "seed": seed,
            "success": success,
            "censored": as_int(row.get("censored"), 1 - success),
            "censor_reason": row.get("censor_reason") or "",
            "mechanics_budget": budget,
            "common_episode_horizon": episode_horizon,
            "episodes": as_int(row["episodes"]),
            "mechanical_steps": as_int(row["mechanical_steps"]),
            "projector_calls": as_int(row["projector_calls"]),
            "first_success_episode": first_episode,
            "first_success_mechanical_steps": first_steps,
            "first_success_projector_calls": as_int(
                row.get("first_success_projector_calls")
            ),
            "episode_event_at_common_horizon": episode_event,
            "assigned_mechanical_cost": first_steps if success else budget,
            "assigned_episode_cost": (
                min(first_episode, episode_horizon)
                if success and first_episode is not None
                else episode_horizon
            ),
            "policy_updates": as_int(row.get("policy_updates"), 0),
            "terminal_flatness_ratio": as_float(
                row.get("terminal_flatness_ratio")
            ),
            "terminal_spread": as_float(row.get("terminal_spread")),
            "initial_model_sha256": row.get("initial_model_sha256") or "",
            "first_batch_action_trace_sha256": (
                row.get("first_batch_action_trace_sha256") or ""
            ),
            "summary_path": row.get("summary_path") or "",
        })
    expected_pairs = {
        (method, int(seed))
        for method in root_config["methods"]
        for seed in root_config["seeds"]
    }
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        extra = sorted(seen - expected_pairs)
        raise ValueError(
            f"{spec['task_id']}: discovery pair mismatch; "
            f"missing={missing}, extra={extra}."
        )
    return normalized


def discovery_statistics(records, aggregate, rng, replicates):
    by_method = {}
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        mechanics = [row["assigned_mechanical_cost"] for row in rows]
        episodes = [row["assigned_episode_cost"] for row in rows]
        successes = sum(row["success"] for row in rows)
        episode_successes = sum(
            row["episode_event_at_common_horizon"] for row in rows
        )
        success_ci = wilson_interval(successes, len(rows))
        episode_ci = wilson_interval(episode_successes, len(rows))
        by_method[method] = {
            "trials": len(rows),
            "successes": successes,
            "success_at_budget": successes / len(rows),
            "success_at_budget_wilson95": {
                "low": success_ci[0], "high": success_ci[1]
            },
            "mechanical_rmst": bootstrap_mean_interval(
                mechanics, rng, replicates
            ),
            "episode_successes_at_common_horizon": episode_successes,
            "success_at_common_episode_horizon": episode_successes / len(rows),
            "episode_success_wilson95": {
                "low": episode_ci[0], "high": episode_ci[1]
            },
            "episode_rmst": bootstrap_mean_interval(
                episodes, rng, replicates
            ),
            "mechanics_budget": int(aggregate["mechanics_budget"]),
            "common_episode_horizon": int(
                aggregate["common_episode_horizon"]
            ),
        }

    indexed = {
        (row["method"], int(row["seed"])): row for row in records
    }
    paired = {}
    for baseline in ("frozen", "uniform"):
        seeds = sorted(
            seed for method, seed in indexed
            if method == "learning" and (baseline, seed) in indexed
        )
        mechanical_differences = [
            indexed[("learning", seed)]["assigned_mechanical_cost"]
            - indexed[(baseline, seed)]["assigned_mechanical_cost"]
            for seed in seeds
        ]
        episode_differences = [
            indexed[("learning", seed)]["assigned_episode_cost"]
            - indexed[(baseline, seed)]["assigned_episode_cost"]
            for seed in seeds
        ]
        initial_matches = []
        trace_matches = []
        if baseline == "frozen":
            for seed in seeds:
                learning = indexed[("learning", seed)]
                other = indexed[(baseline, seed)]
                initial_matches.append(
                    bool(learning["initial_model_sha256"])
                    and bool(other["initial_model_sha256"])
                    and learning["initial_model_sha256"]
                    == other["initial_model_sha256"]
                )
                trace_matches.append(
                    bool(learning["first_batch_action_trace_sha256"])
                    and bool(other["first_batch_action_trace_sha256"])
                    and learning["first_batch_action_trace_sha256"]
                    == other["first_batch_action_trace_sha256"]
                )
        mechanics_ci = bootstrap_mean_interval(
            mechanical_differences, rng, replicates
        )
        episodes_ci = bootstrap_mean_interval(
            episode_differences, rng, replicates
        )
        paired[baseline] = {
            "paired_seeds": seeds,
            "mechanical_learning_minus_baseline": mechanics_ci,
            "episode_learning_minus_baseline": episodes_ci,
            "learning_earlier_mechanical_fraction": float(np.mean([
                difference < 0 for difference in mechanical_differences
            ])),
            "mechanical_tie_fraction": float(np.mean([
                difference == 0 for difference in mechanical_differences
            ])),
            "identical_initial_actor_all_pairs": (
                bool(all(initial_matches)) if initial_matches else None
            ),
            "identical_preupdate_action_trace_all_pairs": (
                bool(all(trace_matches)) if trace_matches else None
            ),
        }

    # Check the point estimates against the harness aggregate.  Bootstrap
    # intervals are new, but the unbootstrapped means must match exactly up to
    # floating-point serialization.
    consistency = []
    for method, stats in by_method.items():
        artifact = aggregate["methods"][method]
        checks = {
            "success_at_budget": stats["success_at_budget"],
            "restricted_mean_first_hit_mechanical_steps": stats[
                "mechanical_rmst"
            ]["point"],
            "restricted_mean_first_hit_episode": stats["episode_rmst"][
                "point"
            ],
        }
        for key, calculated in checks.items():
            recorded = float(artifact[key])
            consistency.append({
                "field": f"{method}.{key}",
                "recorded": recorded,
                "calculated": calculated,
                "absolute_error": abs(recorded - calculated),
            })
    maximum_error = max(row["absolute_error"] for row in consistency)
    if maximum_error > 1e-9:
        raise ValueError(
            f"Discovery aggregate consistency error: {maximum_error:.3g}"
        )
    return {
        "methods": by_method,
        "paired_learning_comparisons": paired,
        "artifact_consistency_max_abs": maximum_error,
    }


def summarize_episode_rows(rows):
    count = len(rows)
    successes = count_reason(rows, "success")
    crossings = count_reason(rows, "self_crossing")
    wrong_flat = count_reason(rows, "wrong_flat")
    timeouts = count_reason(rows, "timeout")
    correction_failed = count_reason(rows, "correction_failed")
    no_modes = count_reason(rows, "no_modes")
    success_ci = wilson_interval(successes, count)
    crossing_ci = wilson_interval(crossings, count)
    success_rows = [row for row in rows if row.get("termination_reason") == "success"]
    return {
        "episodes": count,
        "successes": successes,
        "self_crossings": crossings,
        "wrong_flat": wrong_flat,
        "timeouts": timeouts,
        "correction_failed": correction_failed,
        "no_modes": no_modes,
        "success_rate": successes / count if count else None,
        "success_rate_wilson95": {
            "low": success_ci[0], "high": success_ci[1]
        },
        "self_crossing_rate": crossings / count if count else None,
        "self_crossing_rate_wilson95": {
            "low": crossing_ci[0], "high": crossing_ci[1]
        },
        "mean_flatness_ratio": mean(rows, "flatness_ratio"),
        "mean_spread": mean(rows, "spread"),
        "mean_spread_error": mean(rows, "spread_error"),
        "mean_near_miss_q": mean(rows, "near_miss_q"),
        "mean_return": mean(rows, "total_reward"),
        "success_mean_flatness_ratio": mean(success_rows, "flatness_ratio"),
        "success_mean_spread": mean(success_rows, "spread"),
        "success_mean_spread_error": mean(success_rows, "spread_error"),
        "success_max_spread_error": (
            max(as_float(row["spread_error"]) for row in success_rows)
            if success_rows else None
        ),
    }


def reward_invariant_audit(rows, config, tolerance=1.0e-10):
    """Check serialized ori4 return identities without re-running mechanics."""

    component_fields = (
        "total_reward",
        "total_flatness_progress_reward",
        "total_target_progress_reward",
        "total_integrated_target_deviation_cost",
        "total_target_deviation_path_reward",
        "total_terminal_reward_component",
        "near_miss_q",
        "near_miss_bonus",
    )
    missing_fields = sorted({
        field for field in component_fields
        if any(field not in row or row[field] == "" for row in rows)
    })
    total_errors = []
    path_reward_errors = []
    near_miss_errors = []
    finite = True
    all_numeric_finite = True
    numeric_cells_checked = 0
    terminal_flag_mismatches = 0
    reason_to_flag = {
        "success": "success",
        "self_crossing": "self_crossing",
        "wrong_flat": "wrong_flat",
        "timeout": "timeout",
        "correction_failed": "correction_failed",
        "no_modes": "no_modes",
    }
    near_miss_weight = float(config["near_miss_bonus_weight"])
    for row in rows:
        for serialized in row.values():
            if serialized in (None, ""):
                continue
            try:
                numeric = float(serialized)
            except (TypeError, ValueError):
                continue
            numeric_cells_checked += 1
            all_numeric_finite = all_numeric_finite and math.isfinite(numeric)
        values = {}
        for field in component_fields:
            if field in row and row[field] != "":
                value = float(row[field])
                finite = finite and math.isfinite(value)
                values[field] = value
        if not missing_fields:
            expected_total = (
                values["total_flatness_progress_reward"]
                + values["total_target_progress_reward"]
                - values["total_integrated_target_deviation_cost"]
                + values["total_terminal_reward_component"]
            )
            total_errors.append(abs(values["total_reward"] - expected_total))
            path_reward_errors.append(abs(
                values["total_target_deviation_path_reward"]
                + values["total_integrated_target_deviation_cost"]
            ))
            near_miss_errors.append(abs(
                values["near_miss_bonus"]
                - near_miss_weight * values["near_miss_q"]
            ))
        reason = row.get("termination_reason")
        expected_flag = reason_to_flag.get(reason)
        if expected_flag is None:
            terminal_flag_mismatches += 1
            continue
        for flag in reason_to_flag.values():
            observed = as_int(row.get(flag), 0)
            expected = int(flag == expected_flag)
            terminal_flag_mismatches += int(observed != expected)

    maximum_total_error = max(total_errors, default=None)
    maximum_path_error = max(path_reward_errors, default=None)
    maximum_near_miss_error = max(near_miss_errors, default=None)
    passed = bool(
        rows
        and not missing_fields
        and finite
        and all_numeric_finite
        and terminal_flag_mismatches == 0
        and maximum_total_error <= tolerance
        and maximum_path_error <= tolerance
        and maximum_near_miss_error <= tolerance
    )
    return {
        "rows": len(rows),
        "tolerance": tolerance,
        "all_required_values_finite": bool(finite),
        "all_serialized_numeric_values_finite": bool(all_numeric_finite),
        "serialized_numeric_cells_checked": numeric_cells_checked,
        "missing_fields": missing_fields,
        "terminal_flag_mismatches": terminal_flag_mismatches,
        "maximum_total_reward_component_error": maximum_total_error,
        "maximum_path_reward_sign_error": maximum_path_error,
        "maximum_near_miss_bonus_error": maximum_near_miss_error,
        "passed": passed,
    }


def first_training_hit(training_rows):
    cumulative = 0
    for index, row in enumerate(training_rows, start=1):
        cumulative += as_int(row["steps"], 0)
        if row.get("termination_reason") == "success":
            return {"episode": index, "mechanical_steps": cumulative}
    return {"episode": None, "mechanical_steps": None}


def serialized_numeric_finite_audit(rows):
    """Audit every CSV cell that can be parsed as a number."""

    checked = 0
    finite = True
    for row in rows:
        for value in row.values():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            checked += 1
            finite = finite and math.isfinite(number)
    return {"numeric_cells_checked": checked, "all_finite": bool(finite)}


def structured_numeric_finite_audit(value):
    """Audit finite JSON-native numeric values recursively."""

    checked = 0
    finite = True

    def visit(item):
        nonlocal checked, finite
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            checked += 1
            finite = finite and math.isfinite(float(item))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return {"numeric_values_checked": checked, "all_finite": bool(finite)}


def portable_path(path):
    """Prefer a stable repository-relative POSIX path in tabular output."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(SCRIPT_DIRECTORY).as_posix()
    except ValueError:
        return str(resolved)


def task_spec_for_training_config(config, pilot_root):
    """Map a pilot to exactly one task using artifact model and target."""

    matches = []
    for spec in TASK_SPECS:
        reference = load_json(
            Path(pilot_root) / spec["selected_pilot"] / "run_config.json"
        )
        if (
            spec["model_name"] == config.get("selected_model")
            and math.isclose(
                float(reference["target_spread"]),
                float(config["target_spread"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            matches.append(spec)
    if len(matches) != 1:
        raise ValueError(
            "Pilot task classification is not unique for model="
            f"{config.get('selected_model')!r}, target="
            f"{config.get('target_spread')!r}: "
            f"{[spec['task_id'] for spec in matches]}"
        )
    return matches[0]


def optional_file_hash(directory, name):
    path = Path(directory) / name
    return sha256(path) if path.is_file() else ""


def compact_hash_set(hash_set):
    """Hash a labeled file-hash mapping without depending on JSON spacing."""

    digest = hashlib.sha256()
    for name, value in sorted(hash_set.items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collect_tuning_runs(pilot_root, preservation_names):
    """Create one compact, independently auditable record per pilot run."""

    pilot_root = Path(pilot_root)
    directories = sorted(
        directory for directory in pilot_root.iterdir()
        if directory.is_dir() and (directory / "run_config.json").is_file()
    )
    if len(directories) != EXPECTED_PILOT_RUN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PILOT_RUN_COUNT} pilot directories before "
            f"cleanup, found {len(directories)} below {pilot_root}."
        )

    required_files = (
        "run_config.json",
        "summary.json",
        "training_history.csv",
        "update_history.csv",
        "evaluation_summary.csv",
        "best_policy_evaluation_summary.csv",
        "policy.pt",
        "best_policy.pt",
    )
    rows = []
    for directory in directories:
        companion_logs = {
            "stdout_log": pilot_root / f"{directory.name}.stdout.log",
            "stderr_log": pilot_root / f"{directory.name}.stderr.log",
        }
        missing = [
            name for name in required_files
            if not (directory / name).is_file()
        ]
        missing.extend(
            path.name for path in companion_logs.values()
            if not path.is_file()
        )
        if missing:
            raise ValueError(
                f"Pilot {directory.name} is incomplete; missing {missing}."
            )

        config = load_json(directory / "run_config.json")
        summary = load_json(directory / "summary.json")
        training_rows = load_csv(directory / "training_history.csv")
        update_rows = load_csv(directory / "update_history.csv")
        heldout_rows = load_csv(directory / "evaluation_summary.csv")
        heldout_best_rows = load_csv(
            directory / "best_policy_evaluation_summary.csv"
        )
        spec = task_spec_for_training_config(config, pilot_root)
        training = summarize_episode_rows(training_rows)
        last64 = summarize_episode_rows(training_rows[-64:])
        heldout = summarize_episode_rows(heldout_rows)
        first_hit = first_training_hit(training_rows)

        reward_audits = {
            "training": reward_invariant_audit(training_rows, config),
            "heldout_last": reward_invariant_audit(heldout_rows, config),
            "heldout_best": reward_invariant_audit(
                heldout_best_rows, config
            ),
        }
        numeric_audits = {
            "run_config": structured_numeric_finite_audit(config),
            "summary": structured_numeric_finite_audit(summary),
            "training": serialized_numeric_finite_audit(training_rows),
            "updates": serialized_numeric_finite_audit(update_rows),
            "heldout_last": serialized_numeric_finite_audit(heldout_rows),
            "heldout_best": serialized_numeric_finite_audit(
                heldout_best_rows
            ),
        }
        all_numeric_finite = all(
            audit["all_finite"] for audit in numeric_audits.values()
        )
        numeric_values_checked = sum(
            audit.get("numeric_cells_checked", 0)
            + audit.get("numeric_values_checked", 0)
            for audit in numeric_audits.values()
        )

        mechanics = sum(as_int(row["steps"], 0) for row in training_rows)
        recorded_hit_episode = as_int(
            summary.get("training_first_success_episode")
        )
        recorded_hit_mechanics = as_int(
            summary.get("training_first_success_mechanical_steps")
        )
        computed_updates_before_hit = (
            (int(first_hit["episode"]) - 1) // int(config["batch_size"])
            if first_hit["episode"] is not None else None
        )
        recorded_updates_before_hit = as_int(
            summary.get("training_policy_updates_before_first_success")
        )
        reason_total = sum(
            training[key] for key in (
                "successes", "self_crossings", "wrong_flat", "timeouts",
                "correction_failed", "no_modes",
            )
        )
        heldout_reason_total = sum(
            heldout[key] for key in (
                "successes", "self_crossings", "wrong_flat", "timeouts",
                "correction_failed", "no_modes",
            )
        )
        maximum_total_error = max(
            (
                audit["maximum_total_reward_component_error"]
                for audit in reward_audits.values()
                if audit["maximum_total_reward_component_error"] is not None
            ),
            default=None,
        )
        maximum_path_error = max(
            (
                audit["maximum_path_reward_sign_error"]
                for audit in reward_audits.values()
                if audit["maximum_path_reward_sign_error"] is not None
            ),
            default=None,
        )
        maximum_near_miss_error = max(
            (
                audit["maximum_near_miss_bonus_error"]
                for audit in reward_audits.values()
                if audit["maximum_near_miss_bonus_error"] is not None
            ),
            default=None,
        )

        file_hashes = {
            name: optional_file_hash(directory, name)
            for name in required_files
        }
        manifest = directory_manifest(directory)
        companion_log_hashes = {
            name: sha256(path) for name, path in companion_logs.items()
        }
        complete_run_fingerprint = compact_hash_set({
            "directory_manifest_sha256": manifest["manifest_sha256"],
            **{
                f"{name}_sha256": digest
                for name, digest in companion_log_hashes.items()
            },
        })
        adapter = config["three_dimensional_adapter"]
        model_hashes = adapter.get("model_file_sha256", {})
        row = {
            "directory": portable_path(directory),
            "run_name": directory.name,
            "retention_class": (
                "preserve_required"
                if directory.name in preservation_names
                else "delete_candidate_after_compact_archive"
            ),
            "task": spec["task_id"],
            "model": spec["model_name"],
            "target": float(config["target_spread"]),
            "seed": int(config["seed"]),
            "episodes": len(training_rows),
            "episodes_configured": int(config["episodes"]),
            "episodes_match_config": int(
                len(training_rows) == int(config["episodes"])
                == as_int(summary.get("training_episodes"))
            ),
            "batch": int(config["batch_size"]),
            "actor_LR": float(config["actor_learning_rate"]),
            "critic_LR": float(config["critic_learning_rate"]),
            "H": int(config["max_steps"]),
            "h": float(config["max_step_width"]),
            "hidden_size": int(config["hidden_size"]),
            "near_miss_bonus_weight": float(
                config["near_miss_bonus_weight"]
            ),
            "near_miss_flatness_far": float(
                config["near_miss_flatness_far"]
            ),
            "near_miss_spread_error_far": float(
                config["near_miss_spread_error_far"]
            ),
            "S": training["successes"],
            "X": training["self_crossings"],
            "W": training["wrong_flat"],
            "C": training["correction_failed"],
            "T": training["timeouts"],
            "N": training["no_modes"],
            "mechanical_steps": mechanics,
            "mechanics_match_summary": int(
                mechanics
                == as_int(summary.get("training_total_mechanical_steps"))
            ),
            "first_hit_episode": first_hit["episode"],
            "first_hit_mechanical_steps": first_hit["mechanical_steps"],
            "updates_before_first_hit": recorded_updates_before_hit,
            "first_hit_matches_summary": int(
                first_hit["episode"] == recorded_hit_episode
                and first_hit["mechanical_steps"] == recorded_hit_mechanics
                and computed_updates_before_hit == recorded_updates_before_hit
            ),
            "last64_n": last64["episodes"],
            "last64_S": last64["successes"],
            "last64_X": last64["self_crossings"],
            "last64_W": last64["wrong_flat"],
            "last64_C": last64["correction_failed"],
            "last64_T": last64["timeouts"],
            "last64_N": last64["no_modes"],
            "last64_q": last64["mean_near_miss_q"],
            "last64_F": last64["mean_flatness_ratio"],
            "last64_E": last64["mean_spread_error"],
            "last64_return": last64["mean_return"],
            "heldout_n": heldout["episodes"],
            "heldout_S": heldout["successes"],
            "heldout_X": heldout["self_crossings"],
            "heldout_W": heldout["wrong_flat"],
            "heldout_C": heldout["correction_failed"],
            "heldout_T": heldout["timeouts"],
            "heldout_N": heldout["no_modes"],
            "heldout_F": heldout["mean_flatness_ratio"],
            "heldout_E": heldout["mean_spread_error"],
            "all_numeric_finite": int(all_numeric_finite),
            "numeric_values_checked": numeric_values_checked,
            "reward_invariant_pass": int(all(
                audit["passed"] for audit in reward_audits.values()
            )),
            "reward_max_total_error": maximum_total_error,
            "reward_max_path_sign_error": maximum_path_error,
            "reward_max_near_miss_error": maximum_near_miss_error,
            "terminal_count_matches_episodes": int(
                reason_total == len(training_rows)
                and heldout_reason_total == len(heldout_rows)
            ),
            "model_node_sha256": model_hashes.get("node", ""),
            "model_member_sha256": model_hashes.get("member", ""),
            "model_fix_sha256": model_hashes.get("fix", ""),
            "run_config_sha256": file_hashes["run_config.json"],
            "summary_sha256": file_hashes["summary.json"],
            "training_history_sha256": file_hashes[
                "training_history.csv"
            ],
            "update_history_sha256": file_hashes["update_history.csv"],
            "evaluation_summary_sha256": file_hashes[
                "evaluation_summary.csv"
            ],
            "best_policy_evaluation_summary_sha256": file_hashes[
                "best_policy_evaluation_summary.csv"
            ],
            "policy_sha256": file_hashes["policy.pt"],
            "best_policy_sha256": file_hashes["best_policy.pt"],
            "stdout_log_sha256": companion_log_hashes["stdout_log"],
            "stderr_log_sha256": companion_log_hashes["stderr_log"],
            "run_fingerprint_sha256": complete_run_fingerprint,
            "directory_manifest_sha256": manifest["manifest_sha256"],
            "directory_file_count": manifest["file_count"],
        }
        rows.append(row)

    if len({row["directory"] for row in rows}) != EXPECTED_PILOT_RUN_COUNT:
        raise ValueError("Pilot inventory contains duplicate directory rows.")
    return rows


def evaluation_rollout_records(
    spec, run_directory, run_kind, policy_label, csv_name, config,
    duplicate_of_policy="",
):
    path = Path(run_directory) / csv_name
    if not path.is_file():
        return []
    rows = load_csv(path)
    evaluation_seed = config.get("final_evaluation_seed")
    records = []
    for index, row in enumerate(rows):
        records.append({
            "task_id": spec["task_id"],
            "model_name": spec["model_name"],
            "target_spread": float(config["target_spread"]),
            "run_kind": run_kind,
            "run_directory": Path(run_directory).name,
            "training_seed": int(config["seed"]),
            "policy_label": policy_label,
            "duplicate_of_policy": duplicate_of_policy,
            "evaluation_seed": evaluation_seed,
            "rollout": as_int(
                row.get("evaluation_episode"), default=index
            ),
            "termination_reason": row.get("termination_reason", ""),
            "steps": as_int(row.get("steps")),
            "success": as_int(row.get("success"), 0),
            "self_crossing": as_int(row.get("self_crossing"), 0),
            "wrong_flat": as_int(row.get("wrong_flat"), 0),
            "timeout": as_int(row.get("timeout"), 0),
            "correction_failed": as_int(row.get("correction_failed"), 0),
            "no_modes": as_int(row.get("no_modes"), 0),
            "flatness_ratio": as_float(row.get("flatness_ratio")),
            "spread": as_float(row.get("spread")),
            "spread_error": as_float(row.get("spread_error")),
            "near_miss_q": as_float(row.get("near_miss_q")),
            "total_reward": as_float(row.get("total_reward")),
        })
    return records


def find_matching_pilot(validation_config, pilot_root):
    compared_fields = (
        "selected_model", "target_spread", "max_step_width", "max_steps",
        "episodes", "batch_size", "actor_learning_rate",
        "critic_learning_rate", "near_miss_bonus_weight",
        "near_miss_flatness_far", "near_miss_spread_error_far", "seed",
    )
    matches = []
    for directory in sorted(Path(pilot_root).iterdir()):
        config_path = directory / "run_config.json"
        history_path = directory / "training_history.csv"
        if not directory.is_dir() or not config_path.is_file() or not history_path.is_file():
            continue
        config = load_json(config_path)
        if all(config.get(field) == validation_config.get(field) for field in compared_fields):
            matches.append(directory)
    return matches


def discover_validation_runs(validation_root, pilot_root, task_specs):
    """Read complete optional validation replays without requiring one."""

    output = {spec["task_id"]: [] for spec in task_specs}
    validation_root = Path(validation_root)
    if not validation_root.is_dir():
        return output
    for directory in sorted(validation_root.iterdir()):
        if not directory.is_dir():
            continue
        config_path = directory / "run_config.json"
        summary_path = directory / "summary.json"
        if not config_path.is_file():
            continue
        config = load_json(config_path)
        matching_specs = [
            spec for spec in task_specs
            if spec["model_name"] == config.get("selected_model")
            and math.isclose(
                float(config.get("target_spread")),
                float(load_json(
                    Path(pilot_root)
                    / spec["selected_pilot"]
                    / "run_config.json"
                )["target_spread"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(matching_specs) != 1:
            continue
        spec = matching_specs[0]
        record = {
            "directory": str(directory),
            "status": "incomplete",
            "run_config": config,
        }
        if summary_path.is_file():
            summary = load_json(summary_path)
            expected_episodes = int(config["episodes"])
            history_path = directory / "training_history.csv"
            history_rows = load_csv(history_path) if history_path.is_file() else []
            complete = (
                int(summary.get("training_episodes", -1)) == expected_episodes
                and len(history_rows) == expected_episodes
                and (directory / "evaluation_summary.csv").is_file()
                and (directory / "best_policy_evaluation_summary.csv").is_file()
            )
            record.update({
                "status": "complete" if complete else "incomplete",
                "summary": summary,
                "training_rows": history_rows,
            })
            matches = find_matching_pilot(config, pilot_root)
            record["matching_pilot_directories"] = [
                str(path) for path in matches
            ]
            if complete and len(matches) == 1:
                source_history = matches[0] / "training_history.csv"
                record["training_trace_audit"] = {
                    "source": str(source_history),
                    "source_sha256": sha256(source_history),
                    "replay_sha256": sha256(history_path),
                    "byte_exact": sha256(source_history) == sha256(history_path),
                    "row_count_source": len(load_csv(source_history)),
                    "row_count_replay": len(history_rows),
                }
                source_updates = matches[0] / "update_history.csv"
                replay_updates = directory / "update_history.csv"
                if source_updates.is_file() and replay_updates.is_file():
                    record["update_trace_audit"] = {
                        "source_sha256": sha256(source_updates),
                        "replay_sha256": sha256(replay_updates),
                        "byte_exact": sha256(source_updates) == sha256(replay_updates),
                    }
        output[spec["task_id"]].append(record)
    return output


def collect_rigid_posthoc(selected_runs, validation_root, task_details):
    persisted = []
    validation_root = Path(validation_root)
    if validation_root.is_dir():
        for path in sorted(validation_root.rglob("*rigid*.csv")):
            persisted.append({
                "path": str(path),
                "format": "csv",
                "rows": load_csv(path),
                "sha256": sha256(path),
            })
        for path in sorted(validation_root.rglob("*rigid*.json")):
            persisted.append({
                "path": str(path),
                "format": "json",
                "value": load_json(path),
                "sha256": sha256(path),
            })
    logged = []
    for task_id, run_directory in selected_runs.items():
        for csv_name, scope in (
            ("training_history.csv", "training"),
            ("evaluation_summary.csv", "heldout_last"),
        ):
            rows = load_csv(Path(run_directory) / csv_name)
            values = [
                as_float(row.get("mean_actual_rigid_energy_fraction"))
                for row in rows
            ]
            values = [value for value in values if value is not None]
            if values:
                logged.append({
                    "task_id": task_id,
                    "scope": scope,
                    "episodes": len(values),
                    "mean_actual_rigid_energy_fraction": float(np.mean(values)),
                })
    constructive_witnesses = []
    for task_id, detail in task_details.items():
        evidence = detail["geometry"]["feasibility_evidence"]
        rigid = evidence.get("rigid_motion_artifact")
        if rigid is not None:
            constructive_witnesses.append({
                "task_id": task_id,
                "scope": "constructive_feasibility_witness",
                "source_directory": evidence.get("directory"),
                "source_type": (
                    "artifact_declared_logging_only_not_policy_heldout"
                ),
                **rigid,
            })
    if persisted or logged or constructive_witnesses:
        return {
            "status": "available",
            "persisted_posthoc_artifacts": persisted,
            "logged_diagnostics": logged,
            "constructive_witness_diagnostics": constructive_witnesses,
            "interpretation": (
                "Constructive-witness rigid diagnostics describe only that "
                "path. Selected pilots and validation replays used "
                "--no-rigid-diagnostics, so policy-level rigid claims remain "
                "unavailable."
            ),
        }
    return {
        "status": "unavailable",
        "reason": (
            "No persisted rigid posthoc artifact was found, and selected "
            "runs used --no-rigid-diagnostics so their rigid columns are null."
        ),
        "suggested_columns": [
            "run", "path", "transitions",
            "mean_commanded_rigid_energy_fraction",
            "mean_actual_rigid_energy_fraction",
            "path_actual_rigid_energy_fraction",
            "total_rigid_path_length_normalised",
            "total_internal_path_length_normalised",
            "total_actual_path_length_normalised",
            "first25_actual_fraction", "last25_actual_fraction",
        ],
    }


def classify_discovery(discovery_stats):
    if all(
        discovery_stats["methods"][method]["successes"] == 0
        for method in METHODS
    ):
        return "all_methods_no_hits_at_budget_comparison_unresolved"
    comparison = discovery_stats["paired_learning_comparisons"]["uniform"]
    interval = comparison["mechanical_learning_minus_baseline"]
    if interval["ci95_high"] < 0.0:
        return "learning_faster_than_uniform_bootstrap95"
    if interval["ci95_low"] > 0.0:
        return "learning_slower_than_uniform_bootstrap95"
    return "no_clear_learning_uniform_difference"


def classify_stability(training_stats, heldout_stats):
    if heldout_stats["episodes"] and heldout_stats["successes"] == heldout_stats["episodes"]:
        return "all_heldout_rollouts_succeeded"
    if heldout_stats["successes"] > 0:
        return "partial_heldout_success"
    if training_stats["successes"] > 0:
        return "training_hits_without_heldout_success"
    return "no_strict_success_observed"


def classify_replicated_stability(selected_heldout, validation_summaries):
    if not validation_summaries:
        return "no_independent_validation_replay_available"
    validation_last_successes = sum(
        item["last_heldout"]["successes"] for item in validation_summaries
    )
    validation_best_successes = sum(
        item["best_heldout"]["successes"] for item in validation_summaries
    )
    if (
        selected_heldout["successes"] > 0
        and validation_last_successes == 0
        and validation_best_successes == 0
    ):
        return "selected_seed_success_not_replicated_by_validation_seed"
    if validation_last_successes > 0 or validation_best_successes > 0:
        return "strict_success_observed_in_validation_replay"
    return "no_strict_success_in_validation_replay"


def classify_spread(training_stats, heldout_stats, tolerance):
    if heldout_stats["successes"] > 0:
        maximum = heldout_stats["success_max_spread_error"]
        return (
            "heldout_successes_within_spread_tolerance"
            if maximum is not None and maximum <= tolerance
            else "heldout_success_spread_violation"
        )
    if training_stats["successes"] > 0:
        return "training_only_target_spread_attainment"
    return "no_strict_target_spread_attainment"


def selected_run_analysis(
    spec, pilot_root, discovery_stats, validation_runs,
):
    directory = Path(pilot_root) / spec["selected_pilot"]
    config = load_json(directory / "run_config.json")
    summary = load_json(directory / "summary.json")
    training_rows = load_csv(directory / "training_history.csv")
    heldout_last_rows = load_csv(directory / "evaluation_summary.csv")
    heldout_best_rows = load_csv(
        directory / "best_policy_evaluation_summary.csv"
    )
    training = summarize_episode_rows(training_rows)
    heldout_last = summarize_episode_rows(heldout_last_rows)
    heldout_best = summarize_episode_rows(heldout_best_rows)
    first64 = summarize_episode_rows(training_rows[:64])
    last64 = summarize_episode_rows(training_rows[-64:])
    reward_audits = {
        "training": reward_invariant_audit(training_rows, config),
        "heldout_last": reward_invariant_audit(heldout_last_rows, config),
        "heldout_best": reward_invariant_audit(heldout_best_rows, config),
    }
    first_hit = first_training_hit(training_rows)
    adapter = config["three_dimensional_adapter"]
    maximum_member_length = float(adapter["initial_maximum_member_length"])
    target_spread = float(config["target_spread"])
    deterministic = summary["last_policy_deterministic_evaluation"]
    best_equals_last = (
        sha256(directory / "evaluation_summary.csv")
        == sha256(directory / "best_policy_evaluation_summary.csv")
    )

    complete_validations = [
        record for record in validation_runs if record["status"] == "complete"
    ]
    validation_summary = []
    for record in complete_validations:
        validation_directory = Path(record["directory"])
        last_rows = load_csv(validation_directory / "evaluation_summary.csv")
        best_rows = load_csv(
            validation_directory / "best_policy_evaluation_summary.csv"
        )
        replay_summary = record["summary"]
        validation_summary.append({
            "directory": str(validation_directory),
            "training_trace_audit": record.get("training_trace_audit"),
            "update_trace_audit": record.get("update_trace_audit"),
            "last_heldout": summarize_episode_rows(last_rows),
            "best_heldout": summarize_episode_rows(best_rows),
            "reward_invariants": {
                "last_heldout": reward_invariant_audit(
                    last_rows, record["run_config"]
                ),
                "best_heldout": reward_invariant_audit(
                    best_rows, record["run_config"]
                ),
            },
            "best_policy_selection_evaluation": replay_summary.get(
                "best_policy_selection_evaluation"
            ),
            "last_policy_deterministic_evaluation": replay_summary.get(
                "last_policy_deterministic_evaluation"
            ),
            "best_policy_deterministic_evaluation": replay_summary.get(
                "best_policy_deterministic_evaluation"
            ),
        })

    artifact_success = bool(
        training["successes"]
        or heldout_last["successes"]
        or discovery_stats["methods"]["learning"]["successes"]
        or discovery_stats["methods"]["frozen"]["successes"]
        or discovery_stats["methods"]["uniform"]["successes"]
        or any(
            item["last_heldout"]["successes"]
            or item["best_heldout"]["successes"]
            for item in validation_summary
        )
    )
    feasibility = task_feasibility_evidence(
        spec, config, artifact_success
    )
    stability_status = classify_stability(training, heldout_last)
    replicated_stability_status = classify_replicated_stability(
        heldout_last, validation_summary
    )
    spread_status = classify_spread(
        training, heldout_last, float(config["spread_tolerance"])
    )
    discovery_status = classify_discovery(discovery_stats)

    detail = {
        "task_id": spec["task_id"],
        "display_name": spec["display_name"],
        "model_name": spec["model_name"],
        "selected_run": str(directory),
        "selection_source": (
            "manual analysis input: finalist from the preceding tuning "
            "workflow; not re-selected by this summarizer"
        ),
        "configuration": config,
        "adapter": adapter,
        "training": training,
        "training_first_success": first_hit,
        "training_first_success_updates_reported": summary.get(
            "training_policy_updates_before_first_success"
        ),
        "training_first64": first64,
        "training_last64": last64,
        "reward_invariants": reward_audits,
        "heldout_last": heldout_last,
        "heldout_best": heldout_best,
        "best_heldout_is_byte_identical_to_last": best_equals_last,
        "deterministic_last": deterministic,
        "first_discovery": discovery_stats,
        "validation_replays": validation_summary,
        "geometry": {
            "initial_maximum_member_length_artifact": maximum_member_length,
            "target_spread_artifact": target_spread,
            "target_raw_diameter_from_artifact": (
                maximum_member_length * target_spread
            ),
            "manual_requested_raw_diameter": spec[
                "manual_requested_raw_diameter"
            ],
            "manual_feasibility_prior": {
                "source_type": "manual_context_note",
                "note": spec["manual_feasibility_prior"],
            },
            "feasibility_evidence": feasibility,
            "strict_success_observed_in_current_artifacts": artifact_success,
            "interpretation": (
                "A strict success is constructive feasibility evidence; "
                "absence of a success is not evidence of infeasibility."
            ),
        },
        "evidence_status": {
            "first_discovery": discovery_status,
            "stability": stability_status,
            "replicated_stability": replicated_stability_status,
            "spread": spread_status,
        },
    }

    learning = discovery_stats["methods"]["learning"]
    frozen = discovery_stats["methods"]["frozen"]
    uniform = discovery_stats["methods"]["uniform"]
    pair_uniform = discovery_stats["paired_learning_comparisons"]["uniform"]
    pair_frozen = discovery_stats["paired_learning_comparisons"]["frozen"]
    row = {
        "task_id": spec["task_id"],
        "model_name": spec["model_name"],
        "target_spread": target_spread,
        "target_raw_diameter_from_artifact": maximum_member_length * target_spread,
        "manual_requested_raw_diameter": spec["manual_requested_raw_diameter"],
        "initial_maximum_member_length": maximum_member_length,
        "step_width": config["max_step_width"],
        "max_steps": config["max_steps"],
        "selected_run": directory.name,
        "training_seed": config["seed"],
        "training_episodes": len(training_rows),
        "batch_size": config["batch_size"],
        "actor_learning_rate": config["actor_learning_rate"],
        "critic_learning_rate": config["critic_learning_rate"],
        "train_mechanical_steps": sum(as_int(row_["steps"], 0) for row_ in training_rows),
        "train_successes": training["successes"],
        "train_self_crossings": training["self_crossings"],
        "train_wrong_flat": training["wrong_flat"],
        "train_timeouts": training["timeouts"],
        "first_success_episode": first_hit["episode"],
        "first_success_mechanical_steps": first_hit["mechanical_steps"],
        "updates_before_first_success": summary.get(
            "training_policy_updates_before_first_success"
        ),
        "first64_success_rate": first64["success_rate"],
        "first64_self_crossing_rate": first64["self_crossing_rate"],
        "first64_wrong_flat_rate": first64["wrong_flat"] / first64["episodes"],
        "first64_mean_q": first64["mean_near_miss_q"],
        "first64_mean_F": first64["mean_flatness_ratio"],
        "first64_mean_E": first64["mean_spread_error"],
        "first64_mean_return": first64["mean_return"],
        "last64_success_rate": last64["success_rate"],
        "last64_self_crossing_rate": last64["self_crossing_rate"],
        "last64_wrong_flat_rate": last64["wrong_flat"] / last64["episodes"],
        "last64_mean_q": last64["mean_near_miss_q"],
        "last64_mean_F": last64["mean_flatness_ratio"],
        "last64_mean_E": last64["mean_spread_error"],
        "last64_mean_return": last64["mean_return"],
        "heldout_last_n": heldout_last["episodes"],
        "heldout_last_successes": heldout_last["successes"],
        "heldout_last_success_rate": heldout_last["success_rate"],
        "heldout_last_success_ci95_low": heldout_last[
            "success_rate_wilson95"
        ]["low"],
        "heldout_last_success_ci95_high": heldout_last[
            "success_rate_wilson95"
        ]["high"],
        "heldout_last_self_crossings": heldout_last["self_crossings"],
        "heldout_last_self_crossing_rate": heldout_last[
            "self_crossing_rate"
        ],
        "heldout_last_self_crossing_ci95_low": heldout_last[
            "self_crossing_rate_wilson95"
        ]["low"],
        "heldout_last_self_crossing_ci95_high": heldout_last[
            "self_crossing_rate_wilson95"
        ]["high"],
        "heldout_last_mean_F": heldout_last["mean_flatness_ratio"],
        "heldout_last_mean_spread": heldout_last["mean_spread"],
        "heldout_last_mean_E": heldout_last["mean_spread_error"],
        "heldout_last_mean_return": heldout_last["mean_return"],
        "heldout_best_success_rate": heldout_best["success_rate"],
        "heldout_best_success_ci95_low": heldout_best[
            "success_rate_wilson95"
        ]["low"],
        "heldout_best_success_ci95_high": heldout_best[
            "success_rate_wilson95"
        ]["high"],
        "heldout_best_self_crossing_rate": heldout_best[
            "self_crossing_rate"
        ],
        "heldout_best_self_crossing_ci95_low": heldout_best[
            "self_crossing_rate_wilson95"
        ]["low"],
        "heldout_best_self_crossing_ci95_high": heldout_best[
            "self_crossing_rate_wilson95"
        ]["high"],
        "heldout_best_equals_last": int(best_equals_last),
        "deterministic_last_success": as_int(deterministic.get("success"), 0),
        "deterministic_last_reason": deterministic.get("termination_reason"),
        "discovery_learning_hits": learning["successes"],
        "discovery_learning_n": learning["trials"],
        "discovery_learning_rmst_steps": learning["mechanical_rmst"]["point"],
        "discovery_learning_rmst_steps_ci95_low": learning[
            "mechanical_rmst"
        ]["ci95_low"],
        "discovery_learning_rmst_steps_ci95_high": learning[
            "mechanical_rmst"
        ]["ci95_high"],
        "discovery_learning_rmst_episodes": learning["episode_rmst"]["point"],
        "discovery_frozen_hits": frozen["successes"],
        "discovery_frozen_n": frozen["trials"],
        "discovery_frozen_rmst_steps": frozen["mechanical_rmst"]["point"],
        "discovery_uniform_hits": uniform["successes"],
        "discovery_uniform_n": uniform["trials"],
        "discovery_uniform_rmst_steps": uniform["mechanical_rmst"]["point"],
        "discovery_uniform_rmst_episodes": uniform["episode_rmst"]["point"],
        "paired_learning_minus_uniform_steps": pair_uniform[
            "mechanical_learning_minus_baseline"
        ]["point"],
        "paired_learning_minus_uniform_steps_ci95_low": pair_uniform[
            "mechanical_learning_minus_baseline"
        ]["ci95_low"],
        "paired_learning_minus_uniform_steps_ci95_high": pair_uniform[
            "mechanical_learning_minus_baseline"
        ]["ci95_high"],
        "paired_learning_earlier_uniform_fraction": pair_uniform[
            "learning_earlier_mechanical_fraction"
        ],
        "paired_learning_minus_frozen_steps": pair_frozen[
            "mechanical_learning_minus_baseline"
        ]["point"],
        "paired_learning_earlier_frozen_fraction": pair_frozen[
            "learning_earlier_mechanical_fraction"
        ],
        "artifact_strict_success_observed": int(artifact_success),
        "reward_invariants_pass": int(all(
            audit["passed"] for audit in reward_audits.values()
        )),
        "reward_invariants_max_total_error": max(
            (audit["maximum_total_reward_component_error"]
             for audit in reward_audits.values()
             if audit["maximum_total_reward_component_error"] is not None),
            default=None,
        ),
        "reward_invariants_max_path_sign_error": max(
            (audit["maximum_path_reward_sign_error"]
             for audit in reward_audits.values()
             if audit["maximum_path_reward_sign_error"] is not None),
            default=None,
        ),
        "reward_invariants_max_near_miss_error": max(
            (audit["maximum_near_miss_bonus_error"]
             for audit in reward_audits.values()
             if audit["maximum_near_miss_bonus_error"] is not None),
            default=None,
        ),
        "feasibility_status": feasibility["status"],
        "feasibility_source_type": feasibility["source_type"],
        "feasibility_manifest_sha256": feasibility.get("manifest_sha256"),
        "feasibility_transition_count": feasibility.get(
            "declared", {}
        ).get("transition_count"),
        "feasibility_recomputed_F": feasibility.get(
            "recomputed_from_step_files", {}
        ).get("final_flatness_ratio_F"),
        "feasibility_recomputed_spread": feasibility.get(
            "recomputed_from_step_files", {}
        ).get("final_spread_normalised_D"),
        "feasibility_recomputed_collision_count": feasibility.get(
            "recomputed_from_step_files", {}
        ).get("collision_count"),
        "discovery_status": discovery_status,
        "stability_status": stability_status,
        "replicated_stability_status": replicated_stability_status,
        "spread_status": spread_status,
        "validation_complete_runs": len(complete_validations),
    }
    if validation_summary:
        latest = validation_summary[-1]
        row.update({
            "validation_training_trace_exact": int(bool(
                latest.get("training_trace_audit", {}).get("byte_exact")
            )),
            "validation_last_n": latest["last_heldout"]["episodes"],
            "validation_last_success_rate": latest["last_heldout"][
                "success_rate"
            ],
            "validation_last_crossing_rate": latest["last_heldout"][
                "self_crossing_rate"
            ],
            "validation_last_success_ci95_low": latest["last_heldout"][
                "success_rate_wilson95"
            ]["low"],
            "validation_last_success_ci95_high": latest["last_heldout"][
                "success_rate_wilson95"
            ]["high"],
            "validation_last_crossing_ci95_low": latest["last_heldout"][
                "self_crossing_rate_wilson95"
            ]["low"],
            "validation_last_crossing_ci95_high": latest["last_heldout"][
                "self_crossing_rate_wilson95"
            ]["high"],
            "validation_best_n": latest["best_heldout"]["episodes"],
            "validation_best_success_rate": latest["best_heldout"][
                "success_rate"
            ],
            "validation_best_crossing_rate": latest["best_heldout"][
                "self_crossing_rate"
            ],
            "validation_best_success_ci95_low": latest["best_heldout"][
                "success_rate_wilson95"
            ]["low"],
            "validation_best_success_ci95_high": latest["best_heldout"][
                "success_rate_wilson95"
            ]["high"],
            "validation_best_crossing_ci95_low": latest["best_heldout"][
                "self_crossing_rate_wilson95"
            ]["low"],
            "validation_best_crossing_ci95_high": latest["best_heldout"][
                "self_crossing_rate_wilson95"
            ]["high"],
        })
    return detail, row, training_rows, heldout_last_rows, heldout_best_rows


def step_ecdf(axis, times, events, limit, label, color):
    event_times = sorted(time for time, event in zip(times, events) if event)
    x_values = [0.0]
    y_values = [0.0]
    for index, event_time in enumerate(event_times, start=1):
        x_values.extend([event_time, event_time])
        y_values.extend([(index - 1) / len(times), index / len(times)])
    x_values.append(limit)
    y_values.append(len(event_times) / len(times))
    axis.step(x_values, y_values, where="post", label=label, color=color, lw=2)
    censored = len(times) - len(event_times)
    if censored:
        axis.scatter(
            [limit] * censored,
            [len(event_times) / len(times)] * censored,
            marker="|", color=color, s=65, linewidths=1.2,
        )


def plot_first_discovery(path, task_records, task_details):
    colors = {"learning": "#1f77b4", "frozen": "#ff7f0e", "uniform": "#2ca02c"}
    figure, axes = plt.subplots(
        len(TASK_SPECS), 2, figsize=(11.5, 10.0), squeeze=False
    )
    for row_index, spec in enumerate(TASK_SPECS):
        records = task_records[spec["task_id"]]
        detail = task_details[spec["task_id"]]["first_discovery"]
        mechanics_limit = max(row["mechanics_budget"] for row in records)
        episode_limit = max(row["common_episode_horizon"] for row in records)
        for method in METHODS:
            rows = [row for row in records if row["method"] == method]
            method_detail = detail["methods"][method]
            label = (
                f"{method}: {method_detail['successes']}/"
                f"{method_detail['trials']}"
            )
            step_ecdf(
                axes[row_index, 0],
                [row["assigned_mechanical_cost"] for row in rows],
                [bool(row["success"]) for row in rows],
                mechanics_limit, label, colors[method],
            )
            step_ecdf(
                axes[row_index, 1],
                [row["assigned_episode_cost"] for row in rows],
                [bool(row["episode_event_at_common_horizon"]) for row in rows],
                episode_limit, label, colors[method],
            )
        axes[row_index, 0].set_title(f"{spec['display_name']} — mechanics")
        axes[row_index, 1].set_title(f"{spec['display_name']} — episodes")
        for column in range(2):
            axis = axes[row_index, column]
            axis.set_ylim(-0.02, 1.04)
            axis.grid(alpha=0.25)
            axis.set_ylabel("Cumulative strict-success probability")
            axis.legend(loc="lower right", fontsize=8)
        axes[row_index, 0].set_xlabel("Cumulative mechanical steps")
        axes[row_index, 1].set_xlabel("Episodes (common censored horizon)")
    figure.suptitle(
        "First-discovery empirical event curves (vertical ticks are censoring)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    atomic_figure(path, figure)


def plot_flatness_spread(path, heldout_records, task_details):
    reason_colors = {
        "success": "#2ca02c",
        "self_crossing": "#d62728",
        "wrong_flat": "#9467bd",
        "timeout": "#7f7f7f",
        "correction_failed": "#8c564b",
        "no_modes": "#e377c2",
    }
    policy_markers = {
        "selected_last": "o",
        "selected_best": "s",
        "validation_last": "^",
        "validation_best": "D",
    }
    figure, axes = plt.subplots(1, len(TASK_SPECS), figsize=(15.2, 4.8), squeeze=False)
    for index, spec in enumerate(TASK_SPECS):
        axis = axes[0, index]
        detail = task_details[spec["task_id"]]
        config = detail["configuration"]
        target = float(config["target_spread"])
        spread_tolerance = float(config["spread_tolerance"])
        flat_tolerance = float(config["flatness_tolerance"])
        rows = [row for row in heldout_records if row["task_id"] == spec["task_id"]]
        y_values = [
            max(float(row["flatness_ratio"]), flat_tolerance / 20.0)
            for row in rows if row["flatness_ratio"] is not None
        ]
        y_max = max(y_values + [flat_tolerance * 10.0]) * 1.5
        axis.add_patch(Rectangle(
            (target - spread_tolerance, flat_tolerance / 20.0),
            2.0 * spread_tolerance,
            flat_tolerance - flat_tolerance / 20.0,
            facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.12,
            label="strict success box",
        ))
        for policy, marker in policy_markers.items():
            policy_rows = [row for row in rows if row["policy_label"] == policy]
            for reason, color in reason_colors.items():
                subset = [
                    row for row in policy_rows
                    if row["termination_reason"] == reason
                    and row["spread"] is not None
                    and row["flatness_ratio"] is not None
                ]
                if subset:
                    axis.scatter(
                        [row["spread"] for row in subset],
                        [max(row["flatness_ratio"], flat_tolerance / 20.0) for row in subset],
                        c=color, marker=marker, s=34, alpha=0.68,
                        edgecolors="none",
                    )
        axis.axvline(target, color="black", lw=1, ls="--")
        axis.axhline(flat_tolerance, color="black", lw=1, ls=":")
        axis.set_yscale("log")
        axis.set_ylim(flat_tolerance / 20.0, y_max)
        axis.grid(alpha=0.22, which="both")
        axis.set_title(spec["display_name"])
        axis.set_xlabel("Normalized projected spread")
        axis.set_ylabel("Terminal flatness ratio F")
    reason_legend = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=reason)
        for reason, color in reason_colors.items()
    ]
    policy_legend = [
        Line2D([0], [0], marker=marker, linestyle="", color="black", label=policy)
        for policy, marker in policy_markers.items()
        if any(row["policy_label"] == policy for row in heldout_records)
    ]
    figure.legend(
        handles=reason_legend + policy_legend,
        loc="lower center", ncol=5, fontsize=8,
    )
    figure.suptitle("Held-out terminal geometry and strict-success regions", fontsize=13)
    figure.tight_layout(rect=(0, 0.14, 1, 0.95))
    atomic_figure(path, figure)


def rate_with_interval(rows, reason):
    count = sum(row.get("termination_reason") == reason for row in rows)
    low, high = wilson_interval(count, len(rows))
    rate = count / len(rows) if rows else 0.0
    return rate, low, high


def plot_stability(path, stability_groups):
    figure, axes = plt.subplots(1, len(TASK_SPECS), figsize=(15.2, 5.2), squeeze=False)
    for index, spec in enumerate(TASK_SPECS):
        axis = axes[0, index]
        groups = stability_groups[spec["task_id"]]
        labels = [group["label"] for group in groups]
        x = np.arange(len(groups), dtype=float)
        width = 0.35
        success_rates = []
        success_low = []
        success_high = []
        crossing_rates = []
        crossing_low = []
        crossing_high = []
        for group in groups:
            sr, sl, sh = rate_with_interval(group["rows"], "success")
            cr, cl, ch = rate_with_interval(group["rows"], "self_crossing")
            success_rates.append(sr)
            success_low.append(max(0.0, sr - sl))
            success_high.append(max(0.0, sh - sr))
            crossing_rates.append(cr)
            crossing_low.append(max(0.0, cr - cl))
            crossing_high.append(max(0.0, ch - cr))
        axis.bar(
            x - width / 2, success_rates, width, color="#2ca02c",
            label="success", yerr=np.asarray([success_low, success_high]),
            capsize=3,
        )
        axis.bar(
            x + width / 2, crossing_rates, width, color="#d62728",
            label="self-crossing", yerr=np.asarray([crossing_low, crossing_high]),
            capsize=3,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylim(0.0, 1.18)
        axis.set_ylabel("Rate with Wilson 95% interval")
        axis.set_title(spec["display_name"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Training-tail versus independent held-out stability", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    atomic_figure(path, figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-discovery-root", type=Path, default=DEFAULT_DISCOVERY_ROOT
    )
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument(
        "--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument(
        "--wait-seconds", type=float, default=0.0,
        help="bounded polling time before refusing an incomplete benchmark",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1000:
        parser.error("--bootstrap-replicates must be at least 1000")
    if args.wait_seconds < 0:
        parser.error("--wait-seconds cannot be negative")
    return args


def ordered_union_fields(rows):
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def main(argv=None):
    args = parse_args(argv)
    try:
        wait_for_preflight(
            args.first_discovery_root, args.pilot_root, args.wait_seconds
        )
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    # The output directory is created only after the all-or-nothing discovery
    # preflight above succeeds.
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    discovery_records_by_task = {}
    discovery_details = {}
    discovery_configs = {}
    discovery_aggregates = {}
    all_discovery_records = []
    for spec in TASK_SPECS:
        task_root = args.first_discovery_root / spec["task_id"]
        root_config = load_json(task_root / "run_config.json")
        aggregate = load_json(task_root / "aggregate.json")
        records = discovery_trial_records(
            spec, task_root, root_config, aggregate
        )
        stats = discovery_statistics(
            records, aggregate, rng, args.bootstrap_replicates
        )
        discovery_records_by_task[spec["task_id"]] = records
        discovery_details[spec["task_id"]] = stats
        discovery_configs[spec["task_id"]] = root_config
        discovery_aggregates[spec["task_id"]] = aggregate
        all_discovery_records.extend(records)

    validations = discover_validation_runs(
        args.validation_root, args.pilot_root, TASK_SPECS
    )
    task_details = {}
    task_summary_rows = []
    heldout_records = []
    stability_groups = {}
    selected_runs = {}
    for spec in TASK_SPECS:
        task_id = spec["task_id"]
        detail, summary_row, training_rows, last_rows, best_rows = (
            selected_run_analysis(
                spec, args.pilot_root, discovery_details[task_id],
                validations[task_id],
            )
        )
        task_details[task_id] = detail
        task_summary_rows.append(summary_row)
        selected_directory = args.pilot_root / spec["selected_pilot"]
        selected_runs[task_id] = selected_directory
        config = detail["configuration"]
        duplicate = (
            "selected_last"
            if detail["best_heldout_is_byte_identical_to_last"] else ""
        )
        heldout_records.extend(evaluation_rollout_records(
            spec, selected_directory, "selected_pilot", "selected_last",
            "evaluation_summary.csv", config,
        ))
        heldout_records.extend(evaluation_rollout_records(
            spec, selected_directory, "selected_pilot", "selected_best",
            "best_policy_evaluation_summary.csv", config,
            duplicate_of_policy=duplicate,
        ))
        groups = [
            {"label": "train last64", "rows": training_rows[-64:]},
            {"label": "heldout last", "rows": last_rows},
            {"label": "heldout best", "rows": best_rows},
        ]
        for validation_index, validation in enumerate(validations[task_id]):
            if validation["status"] != "complete":
                continue
            directory = Path(validation["directory"])
            validation_config = validation["run_config"]
            validation_last = load_csv(directory / "evaluation_summary.csv")
            validation_best = load_csv(
                directory / "best_policy_evaluation_summary.csv"
            )
            suffix = "" if validation_index == 0 else str(validation_index + 1)
            heldout_records.extend(evaluation_rollout_records(
                spec, directory, "validation_replay",
                f"validation_last{suffix}", "evaluation_summary.csv",
                validation_config,
            ))
            heldout_records.extend(evaluation_rollout_records(
                spec, directory, "validation_replay",
                f"validation_best{suffix}",
                "best_policy_evaluation_summary.csv", validation_config,
            ))
            groups.extend([
                {"label": f"replay last{suffix}", "rows": validation_last},
                {"label": f"replay best{suffix}", "rows": validation_best},
            ])
        stability_groups[task_id] = groups

    preservation_names = {
        spec["selected_pilot"] for spec in TASK_SPECS
    }
    for records in validations.values():
        for record in records:
            if record["status"] != "complete":
                continue
            preservation_names.update(
                Path(directory).name
                for directory in record.get(
                    "matching_pilot_directories", []
                )
            )
    tuning_rows = collect_tuning_runs(
        args.pilot_root, preservation_names
    )
    tuning_inventory_qa = {
        "expected_rows": EXPECTED_PILOT_RUN_COUNT,
        "observed_rows": len(tuning_rows),
        "unique_directories": len({
            row["directory"] for row in tuning_rows
        }),
        "preserve_required_count": sum(
            row["retention_class"] == "preserve_required"
            for row in tuning_rows
        ),
        "delete_candidate_count": sum(
            row["retention_class"]
            == "delete_candidate_after_compact_archive"
            for row in tuning_rows
        ),
        "all_numeric_finite": all(
            bool(row["all_numeric_finite"]) for row in tuning_rows
        ),
        "all_reward_invariants_pass": all(
            bool(row["reward_invariant_pass"]) for row in tuning_rows
        ),
        "all_episode_counts_match": all(
            bool(row["episodes_match_config"]) for row in tuning_rows
        ),
        "all_mechanics_match": all(
            bool(row["mechanics_match_summary"]) for row in tuning_rows
        ),
        "all_first_hits_match": all(
            bool(row["first_hit_matches_summary"]) for row in tuning_rows
        ),
        "all_terminal_counts_match": all(
            bool(row["terminal_count_matches_episodes"])
            for row in tuning_rows
        ),
        "maximum_reward_total_error": max(
            row["reward_max_total_error"] for row in tuning_rows
        ),
        "maximum_reward_path_sign_error": max(
            row["reward_max_path_sign_error"] for row in tuning_rows
        ),
        "maximum_reward_near_miss_error": max(
            row["reward_max_near_miss_error"] for row in tuning_rows
        ),
    }
    required_inventory_checks = (
        tuning_inventory_qa["observed_rows"]
        == tuning_inventory_qa["expected_rows"]
        == tuning_inventory_qa["unique_directories"]
        and tuning_inventory_qa["all_numeric_finite"]
        and tuning_inventory_qa["all_reward_invariants_pass"]
        and tuning_inventory_qa["all_episode_counts_match"]
        and tuning_inventory_qa["all_mechanics_match"]
        and tuning_inventory_qa["all_first_hits_match"]
        and tuning_inventory_qa["all_terminal_counts_match"]
    )
    if not required_inventory_checks:
        raise ValueError(
            "Pilot tuning inventory QA failed: "
            f"{tuning_inventory_qa}"
        )

    discovery_fields = (
        "task_id", "model_name", "target_spread", "method", "seed",
        "success", "censored", "censor_reason", "mechanics_budget",
        "common_episode_horizon", "episodes", "mechanical_steps",
        "projector_calls", "first_success_episode",
        "first_success_mechanical_steps", "first_success_projector_calls",
        "episode_event_at_common_horizon", "assigned_mechanical_cost",
        "assigned_episode_cost", "policy_updates",
        "terminal_flatness_ratio", "terminal_spread",
        "initial_model_sha256", "first_batch_action_trace_sha256",
        "summary_path",
    )
    heldout_fields = (
        "task_id", "model_name", "target_spread", "run_kind",
        "run_directory", "training_seed", "policy_label",
        "duplicate_of_policy", "evaluation_seed", "rollout",
        "termination_reason", "steps", "success", "self_crossing",
        "wrong_flat", "timeout", "correction_failed", "no_modes",
        "flatness_ratio", "spread",
        "spread_error", "near_miss_q", "total_reward",
    )
    task_summary_path = args.output_root / "task_summary.csv"
    discovery_trials_path = args.output_root / "first_discovery_trials.csv"
    heldout_path = args.output_root / "heldout_rollouts.csv"
    tuning_path = args.output_root / "tuning_runs.csv"
    atomic_csv(
        task_summary_path, task_summary_rows,
        ordered_union_fields(task_summary_rows),
    )
    atomic_csv(discovery_trials_path, all_discovery_records, discovery_fields)
    atomic_csv(heldout_path, heldout_records, heldout_fields)
    atomic_csv(
        tuning_path, tuning_rows, ordered_union_fields(tuning_rows)
    )

    first_plot = args.output_root / "first_discovery_km_ecdf.png"
    geometry_plot = args.output_root / "flatness_spread_scatter.png"
    stability_plot = args.output_root / "stability_crossing_comparison.png"
    plot_first_discovery(
        first_plot, discovery_records_by_task, task_details
    )
    plot_flatness_spread(geometry_plot, heldout_records, task_details)
    plot_stability(stability_plot, stability_groups)

    source_hash_sets = {
        task_id: {
            "root_fingerprint": config.get("root_fingerprint"),
            "source_sha256": config.get("source_sha256", {}),
            "model_file_sha256": config.get("model_file_sha256", {}),
        }
        for task_id, config in discovery_configs.items()
    }
    collision = {
        task_id: {
            "backend": detail["adapter"].get("collision_backend"),
            "backend_version": detail["adapter"].get(
                "collision_backend_version"
            ),
            "semantics": detail["adapter"].get("collision_semantics"),
            "parameters": detail["adapter"].get("collision_parameters"),
        }
        for task_id, detail in task_details.items()
    }
    all_targets_have_heldout_success = all(
        detail["heldout_last"]["successes"] > 0
        for detail in task_details.values()
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "generated_utc": utc_now(),
        "analysis_inputs": {
            "first_discovery_root": str(args.first_discovery_root),
            "pilot_root": str(args.pilot_root),
            "validation_root": str(args.validation_root),
            "selected_pilot_directories": {
                spec["task_id"]: spec["selected_pilot"]
                for spec in TASK_SPECS
            },
            "selected_run_policy": (
                "Manual finalist inputs from the preceding tuning workflow; "
                "the analyzer does not optimize over pilot directories."
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "manual_geometry_inputs": {
                spec["task_id"]: {
                    "requested_raw_diameter": spec[
                        "manual_requested_raw_diameter"
                    ],
                    "source_type": "manual_user_requirement",
                    "context_note": spec["manual_feasibility_prior"],
                    "constructive_artifact_path": (
                        str(spec["feasibility_artifact"])
                        if "feasibility_artifact" in spec else None
                    ),
                    "external_source_path": (
                        str(spec["external_feasibility_source"])
                        if "external_feasibility_source" in spec else None
                    ),
                }
                for spec in TASK_SPECS
            },
        },
        "definitions": {
            "strict_success": (
                "terminal flatness_ratio <= flatness_tolerance, "
                "spread_error <= spread_tolerance, and an admissible path"
            ),
            "F": "terminal flatness_ratio",
            "E": "absolute normalized spread error",
            "spread": (
                "diameter after projection onto the best-fit plane divided "
                "by the maximum initial member length"
            ),
            "mechanical_rmst": (
                "mean assigned first-hit cost with every administratively "
                "censored trial assigned the common mechanics budget"
            ),
            "episode_rmst": (
                "mean min(first-hit episode, floor(budget/max_steps)); hits "
                "after that common horizon are episode-censored"
            ),
            "bootstrap_ci": (
                "fixed-seed nonparametric percentile 95% interval over "
                "trials, or over paired seed differences"
            ),
            "rate_ci": "two-sided Wilson score 95% interval",
            "first_discovery_curve": (
                "Empirical cumulative incidence of strict first hits. Since "
                "all non-hits share one administrative censoring horizon, "
                "this equals one minus the Kaplan-Meier survival estimate "
                "through that horizon."
            ),
            "best_vs_last": (
                "selected pilot best/last files are reported separately and "
                "explicitly marked when byte-identical"
            ),
            "tuning_runs_H": "configured maximum steps per episode",
            "tuning_runs_h": "configured maximum action step width",
            "tuning_runs_terminal_codes": {
                "S": "success",
                "X": "self_crossing",
                "W": "wrong_flat",
                "C": "correction_failed",
                "T": "timeout",
                "N": "no_modes",
            },
            "tuning_runs_heldout": (
                "independent final evaluation_summary.csv for the last policy"
            ),
            "tuning_runs_retention_class": (
                "preserve_required includes each manually selected finalist "
                "and any pilot that is the byte-trace source of a complete "
                "validation replay; all other rows are cleanup candidates "
                "only after this compact archive is preserved"
            ),
            "tuning_runs_run_fingerprint": (
                "SHA-256 over the complete pilot-directory manifest hash "
                "and the sibling stdout/stderr log hashes"
            ),
            "tuning_runs_directory_manifest": (
                "SHA-256 over every relative file path and file SHA-256 "
                "inside the pilot directory"
            ),
        },
        "inference_rules": {
            "discovery": (
                "The bootstrap CI for paired learning-minus-uniform assigned "
                "mechanical cost must lie wholly below zero to label learning "
                "faster, wholly above zero to label it slower."
            ),
            "stability": (
                "Training hits are not stability evidence. Stability labels "
                "are determined from independent held-out rollouts. The "
                "replicated-stability label separately reports whether a "
                "matching validation replay at another training seed "
                "reproduces strict held-out success."
            ),
            "spread_control": (
                "Per-task spread attainment is conditional on strict held-out "
                "success. Cross-target control requires held-out success for "
                "every requested target."
            ),
        },
        "provenance": {
            "analyzer_sha256": sha256(__file__),
            "first_discovery_artifacts": source_hash_sets,
            "collision": collision,
            "feasibility_evidence": {
                task_id: {
                    "source_type": detail["geometry"][
                        "feasibility_evidence"
                    ].get("source_type"),
                    "status": detail["geometry"][
                        "feasibility_evidence"
                    ].get("status"),
                    "directory": detail["geometry"][
                        "feasibility_evidence"
                    ].get("directory"),
                    "manifest_sha256": detail["geometry"][
                        "feasibility_evidence"
                    ].get("manifest_sha256"),
                    "summary_sha256": detail["geometry"][
                        "feasibility_evidence"
                    ].get("summary_sha256"),
                }
                for task_id, detail in task_details.items()
            },
            "output_sha256": {
                task_summary_path.name: sha256(task_summary_path),
                discovery_trials_path.name: sha256(discovery_trials_path),
                heldout_path.name: sha256(heldout_path),
                tuning_path.name: sha256(tuning_path),
                first_plot.name: sha256(first_plot),
                geometry_plot.name: sha256(geometry_plot),
                stability_plot.name: sha256(stability_plot),
            },
        },
        "tuning_run_inventory": {
            "qa": tuning_inventory_qa,
            "preserve_required": sorted(
                row["run_name"] for row in tuning_rows
                if row["retention_class"] == "preserve_required"
            ),
            "delete_candidates_after_compact_archive": sorted(
                row["run_name"] for row in tuning_rows
                if row["retention_class"]
                == "delete_candidate_after_compact_archive"
            ),
            "source_scope": (
                "all pilot directories visible before cleanup; one row per "
                "directory, with hashes calculated before any deletion"
            ),
        },
        "first_discovery_aggregates": discovery_aggregates,
        "tasks": task_details,
        "validation_discovery": {
            task_id: [
                {
                    key: value for key, value in record.items()
                    if key not in ("training_rows", "run_config", "summary")
                }
                for record in records
            ]
            for task_id, records in validations.items()
        },
        "rigid_posthoc": collect_rigid_posthoc(
            selected_runs, args.validation_root, task_details
        ),
        "cross_target_spread_control": {
            "all_requested_targets_have_selected_last_heldout_success": (
                all_targets_have_heldout_success
            ),
            "status": (
                "observed_for_all_targets"
                if all_targets_have_heldout_success
                else "not_demonstrated_for_all_targets"
            ),
        },
    }
    results_path = args.output_root / "results.json"
    atomic_json(results_path, results)
    print(f"Wrote reproducible analysis to {args.output_root}")
    for path in (
        task_summary_path, discovery_trials_path, heldout_path, tuning_path,
        results_path, first_plot, geometry_plot, stability_plot,
    ):
        print(f"  {path.name}: {sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
