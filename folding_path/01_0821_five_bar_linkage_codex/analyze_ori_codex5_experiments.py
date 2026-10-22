"""Validate, rank, aggregate, and plot ori_codex5 experiments.

The analyzer has two independent responsibilities:

* audit whether the saved uniform-random artifacts are mechanically reusable;
* summarize a completed training stage and select one common optimizer profile
  for the five-bar pair, one for the waterbomb pair, and one for the torus.

Reward parameters are checked before any cross-task ranking.  A mixed-reward
collection is reported as invalid and is never used to emit a selection file.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

import run_ori_codex5_experiments as runner


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_ROOT = runner.EXPERIMENT_ROOT
DEFAULT_ANALYSIS_DIRECTORY = EXPERIMENT_ROOT / "analysis"


BASELINES = {
    "five_d2": {
        "kind": "ori3_uniform_search",
        "root": (
            "ori_codex4_h010_valid_terminal_near_miss_"
            "actor_critic_10000_results"
        ),
    },
    "five_d3": {
        "kind": "ori3_uniform_search",
        "root": (
            "ori_codex4_2_target_spread_3_h010_valid_terminal_"
            "near_miss_actor_critic_10000_results"
        ),
    },
    "torus2": {
        "kind": "three_dimensional_first_discovery",
        "root": "experiments_3d/first_discovery/screen_n10_b102400/torus2",
    },
    "water1": {
        "kind": "three_dimensional_first_discovery",
        "root": "experiments_3d/first_discovery/screen_n10_b102400/water1",
    },
    "watersqrt2": {
        "kind": "three_dimensional_first_discovery",
        "root": (
            "experiments_3d/first_discovery/screen_n10_b102400/watersqrt2"
        ),
    },
}

EXPECTED_TWO_D_COMMON_MECHANICS_CAPS = {
    "five_d2": 3_999_760,
    "five_d3": 3_999_980,
}


MECHANICS_FIELDS = (
    "max_steps",
    "max_step_width",
    "min_step_width",
    "projection_shrink",
    "max_correction_ratio",
    "max_actual_step_ratio",
    "flatness_subspace_dimension",
    "flatness_tolerance",
    "strain_tolerance",
    "crossing_tolerance",
    "target_spread",
    "spread_tolerance",
    "terminate_wrong_flat",
)


GROUPS = {
    "five": ("five_d2", "five_d3"),
    "torus": ("torus2",),
    "waterbomb": ("water1", "watersqrt2"),
}

STAGE_GROUP_PROFILE_CANDIDATES = {
    "stage1": {
        "five": runner.STAGE1_PROFILE_NAMES,
        "torus": runner.STAGE1_PROFILE_NAMES,
        "waterbomb": runner.STAGE1_PROFILE_NAMES,
    },
    "stage2": {
        "five": ("A", "B"),
        "torus": ("A", "C"),
        "waterbomb": ("B", "E", "F"),
    },
}


# The user explicitly stopped the only full torus run after its last atomic
# checkpoint had been committed.  This exception is deliberately pinned to
# exact file hashes and row counts so an arbitrary incomplete output can never
# be silently promoted to an accepted endpoint.
USER_STOPPED_FULL_ENDPOINTS = {
    "torus2": {
        "profile": "A",
        "checkpoint_episode": 8192,
        "update_rows": 1024,
        "last_periodic_evaluation_episode": 8000,
        "periodic_evaluation_rows": 17,
        "checkpoint_sha256": (
            "63bf9f5e3e3908f7016ad0cab7bcc50d25077980bf5970108ae035b073c04037"
        ),
        "best_policy_episode": 1000,
        "best_policy_sha256": (
            "c2f05db4bef97460a2d4d5a6c2fc947de6a4238e667fcaa4d0d1d16a267bb990"
        ),
        "exploratory_stdout_last_episode": 8800,
        "status": "user_stopped_at_checkpoint",
    },
}

# The derived evaluator writes outside the canonical training directory.  The
# exact schema and policy hashes are checked before its metrics are consumed.
USER_STOPPED_DERIVED_EVALUATION = (
    EXPERIMENT_ROOT / "analysis" / "derived_torus_user_stopped_ep8192"
    / "evaluation_summary.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float_value(row: dict, name: str, default=0.0) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def mean_field(rows: list[dict], name: str) -> float:
    return float(np.mean([float_value(row, name) for row in rows])) if rows else 0.0


def evaluation_residual_termination_rate(evaluation: dict) -> float:
    """Return timeout/no-mode/correction residual absent from saved aggregates.

    The upstream evaluation aggregate records success, self-crossing and
    wrong-flat rates but not timeout separately.  Calling the residual a
    timeout rate would silently misclassify rare no-mode/correction failures,
    so the summary table labels it explicitly as a residual.
    """

    return max(0.0, 1.0 - sum(float(evaluation.get(key, 0.0)) for key in (
        "success_rate", "self_crossing_rate", "wrong_flat_rate",
    )))


def audit_png_artifacts(directory: Path) -> dict:
    from PIL import Image

    names = (
        "learning_curves.png", "ori5_milestone_learning_curve.png",
        "near_miss_learning_curve.png",
    )
    rows = {}
    for name in names:
        path = directory / name
        row = {"bytes": path.stat().st_size if path.is_file() else 0}
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                row.update({
                    "width": int(image.width), "height": int(image.height),
                    "format": image.format,
                })
            row["valid"] = bool(
                row["bytes"] >= 1024
                and row["width"] >= 64 and row["height"] >= 64
                and row["format"] == "PNG"
            )
        except (OSError, ValueError) as error:
            row.update({"valid": False, "error": str(error)})
        rows[name] = row
    return {
        "valid": all(row["valid"] for row in rows.values()),
        "minimum_bytes": min(row["bytes"] for row in rows.values()),
        "files": rows,
    }


def mechanics_expected(task: runner.Task) -> dict:
    return {
        "max_steps": task.max_steps,
        "max_step_width": task.max_step_width,
        "min_step_width": 1.0e-4,
        "projection_shrink": 0.5,
        "max_correction_ratio": 1.0,
        "max_actual_step_ratio": 1.25,
        "flatness_subspace_dimension": task.flatness_subspace_dimension,
        "flatness_tolerance": 1.0e-3,
        "strain_tolerance": 1.0e-8,
        "crossing_tolerance": 1.0e-10,
        "target_spread": task.target_spread,
        "spread_tolerance": 0.05,
        "terminate_wrong_flat": True,
    }


def model_hashes(model_directory: Path) -> dict:
    return {
        "node": sha256(model_directory / "node_initial.dat"),
        "member": sha256(model_directory / "member.dat"),
        "fix": sha256(model_directory / "fix.dat"),
    }


def baseline_paths(task_name: str) -> tuple[Path, Path, Path]:
    entry = BASELINES[task_name]
    root = SCRIPT_DIRECTORY / entry["root"]
    if entry["kind"] == "ori3_uniform_search":
        return (
            root / "run_config.json",
            root / "uniform_random_search" / "aggregate.json",
            root,
        )
    return root / "run_config.json", root / "aggregate.json", root


def restricted_mean(events: list[float], censored: int, cap: float) -> float:
    if any(float(value) > float(cap) for value in events):
        raise ValueError("Event lies beyond the common administrative cap.")
    count = len(events) + int(censored)
    return (
        float(sum(float(value) for value in events) + int(censored) * cap)
        / count if count else float("nan")
    )


def two_dimensional_common_censor_data(
    task_name: str, root: Path, aggregate: dict,
) -> dict:
    """Recensor unequal mechanics horizons at their common minimum cap."""

    censored_horizons = []
    for history_path in sorted(root.glob("uniform_random_search/search_*/episode_history.csv")):
        rows = load_csv(history_path)
        if rows and float_value(rows[-1], "success") < 0.5:
            censored_horizons.append(int(float_value(
                rows[-1], "cumulative_mechanical_steps"
            )))
    if not censored_horizons:
        raise ValueError(f"No censored 2-D searches available for {task_name}.")
    common_mechanics_cap = min(censored_horizons)
    expected_cap = EXPECTED_TWO_D_COMMON_MECHANICS_CAPS[task_name]
    if common_mechanics_cap != expected_cap:
        raise ValueError(
            f"{task_name} common cap {common_mechanics_cap} != {expected_cap}"
        )
    episode_events = [
        int(value) for value in aggregate.get("first_success_episodes", [])
    ]
    mechanics_events = [
        int(value)
        for value in aggregate.get("first_success_mechanical_steps", [])
    ]
    censored = int(aggregate.get("random_censored_count", 0))
    episode_cap = int(aggregate["max_episodes_per_search"])
    return {
        "first_success_episodes": episode_events,
        "first_success_mechanical_steps": mechanics_events,
        "censored_count": censored,
        "common_episode_censor_cap": episode_cap,
        "common_mechanical_censor_cap": common_mechanics_cap,
        "restricted_mean_first_success_episode": restricted_mean(
            episode_events, censored, episode_cap
        ),
        "restricted_mean_first_success_mechanical_steps": restricted_mean(
            mechanics_events, censored, common_mechanics_cap
        ),
        "mechanics_censor_rule": (
            "all censored trials recensored at the minimum observed censored "
            "mechanics horizon"
        ),
        "rmst_is_exact_common_administrative_censor": True,
    }


def three_dimensional_event_data(root: Path, uniform: dict) -> dict:
    summaries = [
        load_json(path) for path in sorted(
            (root / "trials" / "uniform").glob("seed*/summary.json")
        )
    ]
    successful = [row for row in summaries if int(row.get("success", 0))]
    return {
        "first_success_episodes": [
            int(row["first_success_episode"]) for row in successful
        ],
        "first_success_mechanical_steps": [
            int(row["first_success_mechanical_steps"]) for row in successful
        ],
        "censored_count": sum(int(row.get("censored", 0)) for row in summaries),
        "common_episode_censor_cap": int(uniform["common_episode_horizon"]),
        "common_mechanical_censor_cap": int(uniform["mechanics_budget"]),
        "mechanics_censor_rule": uniform.get("rmst_censor_assignment"),
        "rmst_is_exact_common_administrative_censor": uniform.get(
            "rmst_is_exact_common_administrative_censor"
        ),
    }


def audit_random_baseline_static(task_name: str) -> dict:
    task = runner.TASKS[task_name]
    entry = BASELINES[task_name]
    config_path, aggregate_path, root = baseline_paths(task_name)
    errors = []
    for path in (config_path, aggregate_path):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        return {
            "task": task_name,
            "compatible": False,
            "errors": errors,
            "baseline_root": str(root),
        }
    config = load_json(config_path)
    aggregate = load_json(aggregate_path)
    effective = (
        config if entry["kind"] == "ori3_uniform_search"
        else config.get("effective_ori4_args", {})
    )
    expected = mechanics_expected(task)
    for field in MECHANICS_FIELDS:
        if field not in effective:
            errors.append(f"baseline config missing {field}")
        elif not runner.values_equal(effective[field], expected[field]):
            errors.append(
                f"{field}: {effective[field]!r} != {expected[field]!r}"
            )
    provenance = {}
    if entry["kind"] == "three_dimensional_first_discovery":
        if config.get("collision_backend") != "continuous_proper_crossing_3d":
            errors.append("unexpected collision backend")
        expected_model_hashes = config.get("model_file_sha256", {})
        current_model_hashes = model_hashes(
            SCRIPT_DIRECTORY / str(task.model_directory)
        )
        if current_model_hashes != expected_model_hashes:
            errors.append("3-D model-file hash mismatch")
        current_source_hashes = {}
        for name, recorded_hash in config.get("source_sha256", {}).items():
            path = SCRIPT_DIRECTORY / name
            if not path.is_file():
                errors.append(f"missing recorded source file {name}")
                continue
            current_source_hashes[name] = sha256(path)
            if current_source_hashes[name] != recorded_hash:
                errors.append(f"source hash changed: {name}")
        provenance = {
            "recorded_model_sha256": expected_model_hashes,
            "current_model_sha256": current_model_hashes,
            "recorded_source_sha256": config.get("source_sha256", {}),
            "current_source_sha256": current_source_hashes,
            "collision_backend": config.get("collision_backend"),
            "aggregate_complete": aggregate.get("complete"),
        }
        if aggregate.get("complete") is not True:
            errors.append("3-D random aggregate is incomplete")
        uniform = aggregate.get("methods", {}).get("uniform", {})
        event_data = three_dimensional_event_data(root, uniform)
        random_summary = {
            "search_count": uniform.get("trials"),
            "successes": uniform.get("successes"),
            "success_rate": uniform.get("success_at_budget"),
            "mechanics_budget": uniform.get("mechanics_budget"),
            "restricted_mean_first_success_episode": uniform.get(
                "restricted_mean_first_hit_episode"
            ),
            "restricted_mean_first_success_mechanical_steps": uniform.get(
                "restricted_mean_first_hit_mechanical_steps"
            ),
            "median_first_success_mechanical_steps": uniform.get(
                "median_first_hit_mechanical_steps_among_successes"
            ),
            **event_data,
            "wall_time_comparison_status": (
                "historical_reference_only; not a fresh same-load ori5 timing"
            ),
        }
    else:
        common_censor = two_dimensional_common_censor_data(
            task_name, root, aggregate
        )
        random_summary = {
            "search_count": aggregate.get("search_count"),
            "successes": aggregate.get("successes"),
            "success_rate": aggregate.get("random_success_rate"),
            "max_episodes_per_search": aggregate.get("max_episodes_per_search"),
            "median_first_success_mechanical_steps": aggregate.get(
                "median_first_success_mechanical_steps"
            ),
            **common_censor,
            "saved_unequal_censor_mechanics_rmst_not_used": aggregate.get(
                "restricted_mean_first_success_mechanical_steps"
            ),
            "wall_time_comparison_status": (
                "historical_reference_only; not a fresh same-load ori5 timing"
            ),
        }
        if not aggregate.get("enabled"):
            errors.append("saved 2-D random aggregate is not enabled")
    return {
        "task": task_name,
        "compatible": not errors,
        "errors": errors,
        "baseline_kind": entry["kind"],
        "baseline_root": str(root.resolve()),
        "mechanics_signature_expected": expected,
        "random_summary": random_summary,
        "provenance": provenance,
        "reuse_argument": (
            "Uniform actions and mechanics/termination are reward-independent; "
            "only reward values differ between ori4 and ori5."
        ),
        "two_dimensional_provenance_limit": (
            entry["kind"] == "ori3_uniform_search"
        ),
    }


def action_digest(actions: list[dict]) -> str:
    compact = [{
        "physical_direction": row["physical_direction"],
        "mode_coefficients": row["mode_coefficients"],
        "requested_width": row["requested_width"],
    } for row in actions]
    return runner.canonical_digest(compact)


def compare_replay_row(actual: dict, recorded: dict, *, action_hash=None) -> list[str]:
    errors = []
    exact_fields = {
        "termination_reason": str(actual["termination_reason"]),
        "mechanical_steps": int(actual["steps"]),
        "projector_calls": int(actual["projection_attempts"]),
        "self_crossing": int(actual.get("self_crossing", 0)),
        "correction_failed": int(actual.get("correction_failed", 0)),
        "timeout": int(actual.get("timeout", 0)),
    }
    for field, value in exact_fields.items():
        recorded_value = recorded.get(field)
        if field != "termination_reason":
            recorded_value = int(float(recorded_value))
        if recorded_value != value:
            errors.append(f"{field}: replay {value!r} != saved {recorded_value!r}")
    for field in ("flatness_ratio", "spread", "spread_error"):
        if not math.isclose(
            float(actual[field]), float(recorded[field]),
            rel_tol=0.0, abs_tol=2.0e-10,
        ):
            errors.append(
                f"{field}: replay {actual[field]} != saved {recorded[field]}"
            )
    if action_hash is not None and recorded.get("action_trace_sha256") != action_hash:
        errors.append("action_trace_sha256 mismatch")
    return errors


def replay_random_prefix(task_name: str, episodes: int) -> dict:
    """Replay a tiny saved random prefix as a functional mechanics fingerprint."""

    if episodes <= 0:
        return {"attempted": False, "episodes": 0, "compatible": None}
    task = runner.TASKS[task_name]
    entry = BASELINES[task_name]
    errors = []
    if entry["kind"] == "ori3_uniform_search":
        import ori_codex3 as ori3
        import ori_codex5 as ori5

        _, _, root = baseline_paths(task_name)
        rows = load_csv(
            root / "uniform_random_search" / "search_0001"
            / "episode_history.csv"
        )[:episodes]
        seed = int(float(rows[0]["search_seed"]))
        arguments = [
            "--max-steps", str(task.max_steps),
            "--max-step-width", str(task.max_step_width),
            "--flatness-subspace-dimension", str(task.flatness_subspace_dimension),
            "--target-spread", format(task.target_spread, ".17g"),
            "--no-random-search",
        ]
        args = ori5.parse_args(arguments)
        node, member, fix = ori3._load_problem_data(args)
        environment = ori5.make_environment(args, node, member, fix)
        agent = ori3.UniformRandomAgent(
            args.max_step_width, np.random.default_rng(seed)
        )
        for index, recorded in enumerate(rows, 1):
            summary, _, _ = ori5.run_episode(
                environment, agent, deterministic=False, training=False
            )
            for error in compare_replay_row(summary, recorded):
                errors.append(f"episode {index}: {error}")
    else:
        import torch
        import ori_codex3 as ori3
        import ori_codex5_3d as adapter

        _, _, root = baseline_paths(task_name)
        seed_directories = sorted((root / "trials" / "uniform").glob("seed*"))
        if not seed_directories:
            return {
                "attempted": True, "episodes": 0, "compatible": False,
                "errors": ["no saved 3-D uniform trial"],
            }
        rows = load_csv(seed_directories[0] / "episode_history.csv")[:episodes]
        seed = int(float(rows[0]["seed"]))
        arguments = [
            "--max-steps", str(task.max_steps),
            "--max-step-width", str(task.max_step_width),
            "--target-spread", format(task.target_spread, ".17g"),
            "--data-dir", str((SCRIPT_DIRECTORY / str(task.model_directory)).resolve()),
            "--no-random-search", "--no-rigid-diagnostics",
        ]
        args = adapter.parse_args(arguments)
        node, member, fix, _ = adapter.load_problem_data(args.data_dir)
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment = adapter.make_environment(args, node, member, fix)
        agent = ori3.UniformRandomAgent(
            args.max_step_width, np.random.default_rng(seed)
        )
        with adapter.patched_three_dimensional_runtime(args, member):
            for index, recorded in enumerate(rows, 1):
                summary, _, actions = adapter.run_episode(
                    environment, agent, deterministic=False, training=True
                )
                for error in compare_replay_row(
                    summary, recorded, action_hash=action_digest(actions)
                ):
                    errors.append(f"episode {index}: {error}")
    return {
        "attempted": True,
        "episodes": min(episodes, len(rows)),
        "compatible": not errors,
        "errors": errors,
    }


def audit_random_baselines(
    replay_episodes: int, replay_tasks: tuple[str, ...] | list[str] | None = None,
) -> dict:
    replay_set = set(runner.TASKS if replay_tasks is None else replay_tasks)
    rows = []
    for task_name in runner.TASKS:
        row = audit_random_baseline_static(task_name)
        row["functional_replay"] = replay_random_prefix(
            task_name, replay_episodes if task_name in replay_set else 0
        )
        if row["functional_replay"]["compatible"] is False:
            row["compatible"] = False
            row["errors"].extend(row["functional_replay"]["errors"])
        rows.append(row)
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "all_compatible": all(row["compatible"] for row in rows),
        "functional_replay_episodes": replay_episodes,
        "functional_replay_tasks": sorted(replay_set),
        "tasks": {row["task"]: row for row in rows},
    }


def preserve_stronger_functional_replays(new_audit: dict, path: Path) -> dict:
    """Do not erase an earlier functional replay during a static status pass."""

    if not path.is_file():
        return new_audit
    try:
        old_audit = load_json(path)
    except (OSError, json.JSONDecodeError):
        return new_audit
    for task_name, new_row in new_audit["tasks"].items():
        old_replay = (
            old_audit.get("tasks", {}).get(task_name, {})
            .get("functional_replay", {})
        )
        new_replay = new_row.get("functional_replay", {})
        if old_replay.get("attempted") and not new_replay.get("attempted"):
            new_row["functional_replay"] = old_replay
            if old_replay.get("compatible") is False:
                new_row["compatible"] = False
                new_row["errors"].extend(old_replay.get("errors", []))
    new_audit["all_compatible"] = all(
        row["compatible"] for row in new_audit["tasks"].values()
    )
    new_audit["functional_replay_tasks"] = sorted([
        task_name for task_name, row in new_audit["tasks"].items()
        if row.get("functional_replay", {}).get("attempted")
    ])
    return new_audit


def find_valid_artifact(
    task: runner.Task, profile: runner.Profile, stage: runner.Stage,
) -> tuple[Path | None, dict | None]:
    return runner.find_valid_stage_artifact(task, profile, stage)


def _episode_numbers(rows: list[dict]) -> list[int]:
    return [int(float_value(row, "episode", -1)) for row in rows]


def _derived_policy_evaluation(
    payload: dict, key: str, canonical_filename: str,
    expected_policy_sha256: str, expected_episode: int,
) -> tuple[dict, list[str]]:
    errors = []
    entry = payload.get(key)
    if not isinstance(entry, dict):
        return {}, [f"derived evaluation missing {key}"]
    canonical_record = (
        payload.get("provenance", {}).get("canonical_files", {})
        .get(canonical_filename, {})
    )
    if str(canonical_record.get("sha256", "")).lower() != expected_policy_sha256:
        errors.append(f"{key} canonical policy hash mismatch")
    if int(entry.get("episode", -1)) != expected_episode:
        errors.append(f"{key} policy episode mismatch")
    protocol = payload.get("evaluation_protocol", {})
    if int(protocol.get("first_seed", -1)) != 6_410_000:
        errors.append(f"{key} first evaluation seed mismatch")
    if int(protocol.get("last_seed", -1)) != 6_410_099:
        errors.append(f"{key} last evaluation seed mismatch")
    if int(protocol.get("evaluation_episodes", -1)) != 100:
        errors.append(f"{key} evaluation episode count mismatch")
    metrics = entry
    if isinstance(metrics, dict):
        required_metrics = (
            "strict_compact_success_rate", "self_crossing_rate",
            "wrong_flat_rate", "mean_terminal_ratio", "mean_spread_error",
            "mean_return", "mean_steps", "mean_near_miss_q",
            "mean_ori5_milestone_reward",
            "maximum_abs_ori5_reward_decomposition_error",
        )
        for metric in required_metrics:
            if metric not in metrics:
                errors.append(f"{key} metrics missing {metric}")
        try:
            if abs(float(metrics[
                "maximum_abs_ori5_reward_decomposition_error"
            ])) > 1.0e-8:
                errors.append(f"{key} reward decomposition exceeds 1e-8")
        except (KeyError, TypeError, ValueError):
            pass
    else:
        errors.append(f"{key} metrics missing")
        metrics = {}
    return metrics, errors


def validate_user_stopped_full_artifact(
    path: Path, task: runner.Task, profile: runner.Profile,
    stage: runner.Stage, specification: dict,
) -> tuple[dict | None, dict]:
    """Validate the one explicitly accepted atomic full-stage checkpoint."""

    errors = []
    required = (
        "run_config.json", "training_history.csv", "update_history.csv",
        "deterministic_evaluation_history.csv",
        "stochastic_evaluation_history.csv", "checkpoint.pt",
        "best_policy.pt", "best_policy_metrics.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        errors.append("missing files: " + ", ".join(missing))
        return None, {"valid": False, "errors": errors, "path": str(path)}

    matches, config_errors = runner.config_matches_job(
        path, task, profile, stage
    )
    if not matches:
        errors.extend(config_errors)
    try:
        config = load_json(path / "run_config.json")
        best_metrics = load_json(path / "best_policy_metrics.json")
        history = load_csv(path / "training_history.csv")
        updates = load_csv(path / "update_history.csv")
        deterministic = load_csv(path / "deterministic_evaluation_history.csv")
        stochastic = load_csv(path / "stochastic_evaluation_history.csv")
    except (OSError, csv.Error, json.JSONDecodeError) as error:
        errors.append(f"cannot read checkpoint artifact: {error}")
        return None, {"valid": False, "errors": errors, "path": str(path)}

    endpoint = int(specification["checkpoint_episode"])
    if _episode_numbers(history) != list(range(1, endpoint + 1)):
        errors.append("training history is not the exact episode 1..8192 prefix")
    expected_updates = list(range(profile.batch_size, endpoint + 1,
                                  profile.batch_size))
    if _episode_numbers(updates) != expected_updates:
        errors.append("update history is not the exact batch-boundary prefix")
    expected_evaluations = [1, *range(
        stage.eval_interval,
        int(specification["last_periodic_evaluation_episode"]) + 1,
        stage.eval_interval,
    )]
    for name, rows in (
        ("deterministic", deterministic), ("stochastic", stochastic)
    ):
        if _episode_numbers(rows) != expected_evaluations:
            errors.append(f"{name} evaluation history episode grid mismatch")

    if len(updates) != int(specification["update_rows"]):
        errors.append("update history row count mismatch")
    if len(deterministic) != int(specification["periodic_evaluation_rows"]):
        errors.append("deterministic evaluation row count mismatch")
    if len(stochastic) != int(specification["periodic_evaluation_rows"]):
        errors.append("stochastic evaluation row count mismatch")
    if sha256(path / "checkpoint.pt") != specification["checkpoint_sha256"]:
        errors.append("checkpoint SHA-256 mismatch")
    if sha256(path / "best_policy.pt") != specification["best_policy_sha256"]:
        errors.append("best-policy SHA-256 mismatch")
    if int(best_metrics.get("episode", -1)) != int(
        specification["best_policy_episode"]
    ):
        errors.append("best-policy metrics episode mismatch")

    try:
        import torch
        checkpoint = torch.load(
            path / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        best_policy = torch.load(
            path / "best_policy.pt", map_location="cpu", weights_only=False
        )
        if int(checkpoint.get("episode", -1)) != endpoint:
            errors.append("checkpoint payload episode mismatch")
        if int(best_policy.get("episode", -1)) != int(
            specification["best_policy_episode"]
        ):
            errors.append("best-policy payload episode mismatch")
        if checkpoint.get("config") != best_policy.get("config"):
            errors.append("checkpoint and best-policy effective configs differ")
    except Exception as error:  # torch deserialisation is itself an audit gate.
        errors.append(f"cannot audit saved policies: {error}")

    nonfinite = int(sum(
        int(float_value(row, "ori5_nonfinite_reward_component_count"))
        for row in history
    ))
    decomposition = max([
        abs(float_value(row, "ori5_reward_decomposition_error"))
        for row in history
    ] or [0.0])
    step_decomposition = max([
        abs(float_value(
            row, "ori5_maximum_abs_step_reward_decomposition_error"
        )) for row in history
    ] or [0.0])
    if nonfinite != 0:
        errors.append("non-finite reward component detected")
    if decomposition > 1.0e-8:
        errors.append("reward decomposition error exceeds 1e-8")
    if step_decomposition > 1.0e-8:
        errors.append("step reward decomposition error exceeds 1e-8")

    derived_path = USER_STOPPED_DERIVED_EVALUATION
    if not derived_path.is_file():
        errors.append(f"missing derived policy evaluation: {derived_path}")
        derived = {}
    else:
        try:
            derived = load_json(derived_path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read derived policy evaluation: {error}")
            derived = {}
    if derived:
        if derived.get("status") != specification["status"]:
            errors.append("derived evaluation endpoint status mismatch")
        if int(derived.get("checkpoint_episode", -1)) != endpoint:
            errors.append("derived evaluation checkpoint episode mismatch")
        if int(derived.get("best_policy_episode", -1)) != int(
            specification["best_policy_episode"]
        ):
            errors.append("derived best-policy episode mismatch")
        provenance = derived.get("provenance", {})
        if provenance.get("canonical_files_unchanged_during_evaluation") is not True:
            errors.append("derived evaluator did not certify canonical immutability")
        selection_replay = derived.get("best_selection_replay", {})
        if selection_replay.get("discrete_terminal_outcomes_matched") is not True:
            errors.append("best-policy selection replay outcomes did not match")
        if int(selection_replay.get("evaluation_episodes", -1)) != 50:
            errors.append("best-policy selection replay count mismatch")
        if int(selection_replay.get("evaluation_seed", -1)) != 6_310_000:
            errors.append("best-policy selection replay seed mismatch")
        try:
            replay_drift = float(selection_replay[
                "maximum_absolute_continuous_aggregate_delta"
            ])
            if not math.isfinite(replay_drift) or replay_drift > 2.0e-3:
                errors.append("best-policy selection replay drift exceeds 2e-3")
        except (KeyError, TypeError, ValueError):
            errors.append("best-policy selection replay drift is missing")
        partial_audit = derived.get("partial_artifact_validation", {})
        if int(partial_audit.get("checkpoint_episode", -1)) != endpoint:
            errors.append("derived partial-artifact checkpoint mismatch")
        if int(partial_audit.get("training_history_rows", -1)) != endpoint:
            errors.append("derived partial-artifact history count mismatch")
        if int(partial_audit.get("update_history_rows", -1)) != int(
            specification["update_rows"]
        ):
            errors.append("derived partial-artifact update count mismatch")
        if int(partial_audit.get("nonfinite_reward_component_count", -1)) != 0:
            errors.append("derived partial-artifact non-finite reward count")
        checkpoint_evaluation, checkpoint_errors = _derived_policy_evaluation(
            derived, "last_checkpoint_held_out_evaluation", "checkpoint.pt",
            specification["checkpoint_sha256"], endpoint,
        )
        best_evaluation, best_errors = _derived_policy_evaluation(
            derived, "best_policy_held_out_evaluation", "best_policy.pt",
            specification["best_policy_sha256"],
            int(specification["best_policy_episode"]),
        )
        errors.extend(checkpoint_errors)
        errors.extend(best_errors)
    else:
        checkpoint_evaluation = {}
        best_evaluation = {}

    validation = {
        "valid": not errors,
        "errors": errors,
        "path": str(path),
        "status": specification["status"],
        "checkpoint_episode": endpoint,
        "training_rows": len(history),
        "update_rows": len(updates),
        "periodic_evaluation_rows": len(stochastic),
        "last_periodic_evaluation_episode": (
            _episode_numbers(stochastic)[-1] if stochastic else None
        ),
        "checkpoint_sha256": sha256(path / "checkpoint.pt"),
        "best_policy_sha256": sha256(path / "best_policy.pt"),
        "maximum_reward_decomposition_error": decomposition,
        "maximum_step_reward_decomposition_error": step_decomposition,
        "nonfinite_reward_components": nonfinite,
        "derived_evaluation_path": str(derived_path),
    }
    if errors:
        return None, validation

    success_rows = [
        row for row in history
        if float_value(row, "strict_compact_success") > 0.5
    ]
    first_success = success_rows[0] if success_rows else None
    first_episode = (
        int(float_value(first_success, "episode")) if first_success else None
    )
    first_steps = None
    if first_episode is not None:
        first_steps = int(sum(
            float_value(row, "steps") for row in history[:first_episode]
        ))
    checkpoint_evaluation = dict(checkpoint_evaluation)
    checkpoint_evaluation.setdefault("episode", endpoint)
    best_evaluation = dict(best_evaluation)
    best_evaluation.setdefault("episode", specification["best_policy_episode"])
    synthetic_summary = {
        "training_first_success_episode": first_episode,
        "training_first_success_mechanical_steps": first_steps,
        "training_policy_updates_before_first_success": (
            (first_episode - 1) // profile.batch_size
            if first_episode is not None else None
        ),
        "training_total_mechanical_steps": int(sum(
            float_value(row, "steps") for row in history
        )),
        "last_policy_held_out_evaluation": checkpoint_evaluation,
        "best_policy_held_out_evaluation": best_evaluation,
        "best_training_flatness_ratio": min([
            float_value(row, "flatness_ratio") for row in history
        ] or [None]),
        "ori5_reward_parameters": config.get("ori5_reward_parameters"),
        "maximum_abs_ori5_reward_decomposition_error": decomposition,
        "maximum_abs_ori5_step_reward_decomposition_error": step_decomposition,
        "ori5_nonfinite_reward_component_count": nonfinite,
    }
    return synthetic_summary, validation


def summarize_artifact(
    path: Path, task: runner.Task, profile: runner.Profile,
    stage: runner.Stage, *, summary_override: dict | None = None,
    artifact_status: str = "complete",
) -> dict:
    summary = (
        load_json(path / "summary.json")
        if summary_override is None else summary_override
    )
    history = load_csv(path / "training_history.csv")
    tail_count = min(1000, max(1, len(history) // 4))
    tail = history[-tail_count:]
    first_success_episode = summary.get("training_first_success_episode")
    first_success_steps = summary.get("training_first_success_mechanical_steps")
    total_steps = int(summary.get("training_total_mechanical_steps", 0))
    heldout = summary.get("last_policy_held_out_evaluation") or {}
    best_heldout = summary.get("best_policy_held_out_evaluation") or {}
    png_audit = (
        audit_png_artifacts(path)
        if artifact_status == "complete"
        else {"valid": True, "minimum_bytes": None}
    )
    success_rows = [row for row in history if float_value(row, "strict_compact_success") > 0.5]
    tail_endpoint_costs = [
        float(np.clip(
            float_value(row, "ori5_last_valid_flatness_ratio"), 0.0, 2.0
        )) + float_value(row, "ori5_last_valid_spread_error") ** 2
        for row in tail
    ]
    all_endpoint_costs = [
        float(np.clip(
            float_value(row, "ori5_last_valid_flatness_ratio"), 0.0, 2.0
        )) + float_value(row, "ori5_last_valid_spread_error") ** 2
        for row in history
    ]
    first_fraction = (
        float(first_success_steps) / max(1, total_steps)
        if first_success_steps is not None else 1.0
    )
    record = {
        "stage": stage.name,
        "task": task.name,
        "profile": profile.name,
        "batch_size": profile.batch_size,
        "actor_learning_rate": profile.actor_learning_rate,
        "critic_learning_rate": profile.critic_learning_rate,
        "entropy_coefficient": profile.entropy_coefficient,
        "seed": stage.seed,
        "output_directory": str(path.resolve()),
        "artifact_status": artifact_status,
        "requested_training_episodes": stage.episodes,
        "training_endpoint_censored": artifact_status != "complete",
        "training_censor_reason": (
            artifact_status if artifact_status != "complete" else None
        ),
        "training_censor_episode": (
            len(history) if artifact_status != "complete" else None
        ),
        "episodes": len(history),
        "training_successes": len(success_rows),
        "training_success_rate": len(success_rows) / max(1, len(history)),
        "training_crossing_rate": mean_field(history, "self_crossing"),
        "training_wrong_flat_rate": mean_field(history, "wrong_flat"),
        "training_timeout_rate": mean_field(history, "timeout"),
        "training_mean_last_valid_flatness_ratio": mean_field(
            history, "ori5_last_valid_flatness_ratio"
        ),
        "training_mean_last_valid_spread_error": mean_field(
            history, "ori5_last_valid_spread_error"
        ),
        "training_spread_milestone_rate": mean_field(
            history, "ori5_spread_milestone_awarded"
        ),
        "training_joint_milestone_rate": float(np.mean([
            float_value(row, "ori5_joint_milestones_awarded") > 0.0
            for row in history
        ])),
        "first_success_episode": first_success_episode,
        "first_success_mechanical_steps": first_success_steps,
        "policy_updates_before_first_success": summary.get(
            "training_policy_updates_before_first_success"
        ),
        "first_success_fraction_of_training_mechanics": first_fraction,
        "training_total_mechanical_steps": total_steps,
        "tail_episode_count": tail_count,
        "tail_success_rate": mean_field(tail, "strict_compact_success"),
        "tail_crossing_rate": mean_field(tail, "self_crossing"),
        "tail_wrong_flat_rate": mean_field(tail, "wrong_flat"),
        "tail_timeout_rate": mean_field(tail, "timeout"),
        "tail_mean_near_miss_q": mean_field(tail, "near_miss_q"),
        "tail_mean_total_reward": mean_field(tail, "total_reward"),
        "tail_mean_flatness_progress_reward": mean_field(
            tail, "ori5_total_flatness_progress_reward"
        ),
        "tail_mean_target_progress_reward": mean_field(
            tail, "ori5_total_target_progress_reward"
        ),
        "tail_mean_milestone_reward": mean_field(
            tail, "ori5_total_milestone_reward"
        ),
        "tail_mean_terminal_fixed_reward": mean_field(
            tail, "ori5_total_terminal_fixed_reward"
        ),
        "tail_mean_terminal_quality_reward": mean_field(
            tail, "ori5_total_terminal_quality_reward"
        ),
        "tail_mean_flatness_ratio": mean_field(tail, "flatness_ratio"),
        "tail_mean_spread_error": mean_field(tail, "spread_error"),
        "tail_mean_last_valid_flatness_ratio": mean_field(
            tail, "ori5_last_valid_flatness_ratio"
        ),
        "tail_mean_last_valid_spread_error": mean_field(
            tail, "ori5_last_valid_spread_error"
        ),
        "tail_spread_milestone_rate": mean_field(
            tail, "ori5_spread_milestone_awarded"
        ),
        "tail_mean_joint_milestone_count": mean_field(
            tail, "ori5_joint_milestones_awarded"
        ),
        "tail_mean_endpoint_cost": float(np.mean(tail_endpoint_costs)),
        "minimum_terminal_endpoint_cost": float(min(all_endpoint_costs)),
        "tail_joint_milestone_rate": float(np.mean([
            float_value(row, "ori5_joint_milestones_awarded") > 0.0
            for row in tail
        ])),
        "maximum_near_miss_q": max(
            [float_value(row, "near_miss_q") for row in history], default=0.0
        ),
        "maximum_joint_milestones": int(max(
            [float_value(row, "ori5_joint_milestones_awarded") for row in history],
            default=0.0,
        )),
        "heldout_strict_success_rate": float(
            heldout.get("strict_compact_success_rate", 0.0)
        ),
        "heldout_crossing_rate": float(heldout.get("self_crossing_rate", 0.0)),
        "heldout_wrong_flat_rate": float(heldout.get("wrong_flat_rate", 0.0)),
        "heldout_timeout_or_other_residual_rate": (
            evaluation_residual_termination_rate(heldout)
        ),
        "heldout_evaluation_episode": heldout.get("episode"),
        "heldout_mean_flatness_ratio": float(
            heldout.get("mean_terminal_ratio", 0.0)
        ),
        "heldout_mean_spread_error": float(heldout.get("mean_spread_error", 0.0)),
        "heldout_mean_near_miss_q": float(heldout.get("mean_near_miss_q", 0.0)),
        "heldout_mean_milestone_reward": float(
            heldout.get("mean_ori5_milestone_reward", 0.0)
        ),
        "best_heldout_strict_success_rate": float(
            best_heldout.get("strict_compact_success_rate", 0.0)
        ),
        "best_heldout_crossing_rate": float(
            best_heldout.get("self_crossing_rate", 0.0)
        ),
        "best_heldout_wrong_flat_rate": float(
            best_heldout.get("wrong_flat_rate", 0.0)
        ),
        "best_heldout_timeout_or_other_residual_rate": (
            evaluation_residual_termination_rate(best_heldout)
        ),
        "best_heldout_evaluation_episode": best_heldout.get("episode"),
        "best_heldout_mean_flatness_ratio": float(
            best_heldout.get("mean_terminal_ratio", 0.0)
        ),
        "best_heldout_mean_spread_error": float(
            best_heldout.get("mean_spread_error", 0.0)
        ),
        "best_heldout_mean_near_miss_q": float(
            best_heldout.get("mean_near_miss_q", 0.0)
        ),
        "best_heldout_mean_milestone_reward": float(
            best_heldout.get("mean_ori5_milestone_reward", 0.0)
        ),
        "best_training_flatness_ratio": summary.get(
            "best_training_flatness_ratio"
        ),
        "reward_valid": runner.values_equal(
            summary.get("ori5_reward_parameters"),
            runner.COMMON_REWARD_PARAMETERS,
        ),
        "maximum_reward_decomposition_error": summary.get(
            "maximum_abs_ori5_reward_decomposition_error"
        ),
        "maximum_step_reward_decomposition_error": summary.get(
            "maximum_abs_ori5_step_reward_decomposition_error"
        ),
        "nonfinite_reward_components": summary.get(
            "ori5_nonfinite_reward_component_count"
        ),
        "png_artifacts_valid": png_audit["valid"],
        "minimum_png_artifact_bytes": png_audit["minimum_bytes"],
    }
    return record


def collect_stage(stage_name: str) -> tuple[list[dict], list[dict]]:
    stage = runner.STAGES[stage_name]
    records = []
    statuses = []
    processes = runner.active_python_processes()
    for task in runner.TASKS.values():
        task_stage = runner.stage_for_task(stage, task)
        profile_names = runner.profile_names_for_stage_task(
            stage_name, task.name
        )
        if stage_name == "full":
            # Full artifacts are identified by their effective config.  Scan
            # all profiles because the selected mapping is an analyzer output.
            profile_names = tuple(runner.PROFILES)
        for profile_name in profile_names:
            profile = runner.PROFILES[profile_name]
            partial_specification = (
                USER_STOPPED_FULL_ENDPOINTS.get(task.name)
                if stage_name == "full" else None
            )
            if (
                partial_specification is not None
                and profile_name == partial_specification["profile"]
            ):
                partial_path = runner.output_directory(
                    task_stage, task, profile
                )
                synthetic, validation = validate_user_stopped_full_artifact(
                    partial_path, task, profile, task_stage,
                    partial_specification,
                )
                if synthetic is not None:
                    records.append(summarize_artifact(
                        partial_path, task, profile, task_stage,
                        summary_override=synthetic,
                        artifact_status=partial_specification["status"],
                    ))
                    status = partial_specification["status"]
                else:
                    status = "invalid_user_stopped_checkpoint"
                statuses.append({
                    "stage": stage_name,
                    "task": task.name,
                    "profile": profile_name,
                    "status": status,
                    "path": str(partial_path),
                    "validation_errors": " | ".join(
                        validation.get("errors", [])
                    ),
                    "checkpoint_episode": partial_specification[
                        "checkpoint_episode"
                    ],
                    "exploratory_stdout_last_episode_ignored": (
                        partial_specification[
                            "exploratory_stdout_last_episode"
                        ]
                    ),
                    "derived_evaluation_path": str(
                        USER_STOPPED_DERIVED_EVALUATION
                    ),
                })
                continue
            path, validation = find_valid_artifact(task, profile, task_stage)
            if path is not None:
                records.append(summarize_artifact(
                    path, task, profile, task_stage
                ))
                status = "complete"
            else:
                base = runner.output_directory(task_stage, task, profile)
                external = runner.external_process_for_destination(base, processes)
                status = "active" if external else (
                    "invalid" if validation is not None else "missing"
                )
            statuses.append({
                "stage": stage_name,
                "task": task.name,
                "profile": profile_name,
                "status": status,
                "path": str(path or runner.output_directory(
                    task_stage, task, profile
                )),
                "validation_errors": (
                    " | ".join(validation.get("errors", []))
                    if validation is not None else ""
                ),
            })
    return records, statuses


def group_profile_rows(
    records: list[dict], group_name: str, profile_names: tuple[str, ...],
) -> list[dict]:
    task_names = GROUPS[group_name]
    rows = []
    for profile_name in profile_names:
        selected = [
            row for row in records
            if row["task"] in task_names and row["profile"] == profile_name
        ]
        if len(selected) != len(task_names):
            rows.append({
                "group": group_name,
                "profile": profile_name,
                "complete_tasks": len(selected),
                "expected_tasks": len(task_names),
                "eligible": 0,
            })
            continue
        heldout = [row["heldout_strict_success_rate"] for row in selected]
        tail_success = [row["tail_success_rate"] for row in selected]
        tail_q = [row["tail_mean_near_miss_q"] for row in selected]
        endpoint_cost = [row["tail_mean_endpoint_cost"] for row in selected]
        assigned_fraction = [
            row["first_success_fraction_of_training_mechanics"] for row in selected
        ]
        group_row = {
            "group": group_name,
            "profile": profile_name,
            "complete_tasks": len(selected),
            "expected_tasks": len(task_names),
            "eligible": int(all(row["reward_valid"] for row in selected)),
            "tasks_with_training_hit": sum(
                row["first_success_episode"] is not None for row in selected
            ),
            "minimum_heldout_success_rate": min(heldout),
            "mean_heldout_success_rate": float(np.mean(heldout)),
            "minimum_tail_success_rate": min(tail_success),
            "mean_tail_success_rate": float(np.mean(tail_success)),
            "worst_assigned_first_hit_fraction": max(assigned_fraction),
            "mean_assigned_first_hit_fraction": float(np.mean(assigned_fraction)),
            "minimum_tail_near_miss_q": min(tail_q),
            "mean_tail_near_miss_q": float(np.mean(tail_q)),
            "worst_tail_endpoint_cost": max(endpoint_cost),
            "mean_tail_endpoint_cost": float(np.mean(endpoint_cost)),
            "mean_tail_crossing_rate": float(np.mean([
                row["tail_crossing_rate"] for row in selected
            ])),
            "mean_tail_wrong_flat_rate": float(np.mean([
                row["tail_wrong_flat_rate"] for row in selected
            ])),
        }
        rows.append(group_row)
    eligible = [row for row in rows if row.get("eligible")]
    eligible.sort(key=lambda row: (
        -row["tasks_with_training_hit"],
        -row["minimum_heldout_success_rate"],
        -row["mean_heldout_success_rate"],
        -row["minimum_tail_success_rate"],
        -row["mean_tail_success_rate"],
        row["worst_assigned_first_hit_fraction"],
        row["mean_assigned_first_hit_fraction"],
        -row["minimum_tail_near_miss_q"],
        -row["mean_tail_near_miss_q"],
        row["worst_tail_endpoint_cost"],
        row["mean_tail_endpoint_cost"],
        row["mean_tail_crossing_rate"],
        row["mean_tail_wrong_flat_rate"],
        row["profile"],
    ))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    return rows


def select_group_profiles(
    records: list[dict], candidates: dict[str, tuple[str, ...]],
) -> tuple[dict, list[dict]]:
    rankings = []
    winners = {}
    for group_name in GROUPS:
        rows = group_profile_rows(
            records, group_name, candidates[group_name]
        )
        rankings.extend(rows)
        winner = next((row for row in rows if row.get("rank") == 1), None)
        if winner is not None:
            winners[group_name] = winner["profile"]
    selected = {}
    if len(winners) == len(GROUPS):
        for group_name, tasks in GROUPS.items():
            for task_name in tasks:
                selected[task_name] = winners[group_name]
    return selected, rankings


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )
    return result


def plot_stage_learning_curves(records: list[dict], analysis_directory: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task_name in runner.TASKS:
        task_records = [row for row in records if row["task"] == task_name]
        if not task_records:
            continue
        figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
        ordered_records = sorted(task_records, key=lambda row: row["profile"])
        for record_index, record in enumerate(ordered_records):
            colour = f"C{record_index % 10}"
            history = load_csv(
                Path(record["output_directory"]) / "training_history.csv"
            )
            episode = np.asarray([float_value(row, "episode") for row in history])
            mechanics = np.cumsum(np.asarray([
                float_value(row, "steps") for row in history
            ]))
            window = min(64, max(8, len(history) // 8))
            series = (
                ("strict_compact_success", axes[0, 0]),
                ("near_miss_q", axes[0, 1]),
                ("ori5_total_milestone_reward", axes[0, 2]),
            )
            for field, axis in series:
                values = np.asarray([float_value(row, field) for row in history])
                axis.plot(
                    episode, moving_average(values, window), linewidth=1.6,
                    color=colour, label=record["profile"],
                )

            flatness = np.asarray([
                float_value(row, "ori5_last_valid_flatness_ratio")
                for row in history
            ])
            spread_error = np.asarray([
                float_value(row, "ori5_last_valid_spread_error")
                for row in history
            ])
            axes[1, 0].plot(
                episode, moving_average(flatness, window), color=colour,
                linewidth=1.45, label=f"{record['profile']} F",
            )
            axes[1, 0].plot(
                episode, moving_average(spread_error, window), color=colour,
                linewidth=1.25, linestyle="--", label=f"{record['profile']} e_D",
            )

            evaluation_history = load_csv(
                Path(record["output_directory"])
                / "stochastic_evaluation_history.csv"
            )
            evaluation_episode = np.asarray([
                float_value(row, "episode") for row in evaluation_history
            ])
            evaluation_success = np.asarray([
                float_value(row, "strict_compact_success_rate")
                for row in evaluation_history
            ])
            evaluation_crossing = np.asarray([
                float_value(row, "self_crossing_rate")
                for row in evaluation_history
            ])
            evaluation_wrong_flat = np.asarray([
                float_value(row, "wrong_flat_rate")
                for row in evaluation_history
            ])
            axes[1, 1].plot(
                evaluation_episode, evaluation_success, color=colour,
                linewidth=1.6, marker="o", markersize=2.8,
                label=f"{record['profile']} success",
            )
            axes[1, 1].plot(
                evaluation_episode, evaluation_crossing, color=colour,
                linewidth=1.25, linestyle="--", marker="x", markersize=3.0,
                label=f"{record['profile']} crossing",
            )
            axes[1, 1].plot(
                evaluation_episode, evaluation_wrong_flat, color=colour,
                linewidth=1.15, linestyle=":", marker="^", markersize=2.8,
                label=f"{record['profile']} wrong flat",
            )

            success = np.asarray([
                float_value(row, "strict_compact_success") for row in history
            ])
            axes[1, 2].plot(
                mechanics, moving_average(success, window), color=colour,
                linewidth=1.6, label=record["profile"],
            )
            if record.get("artifact_status") == "user_stopped_at_checkpoint":
                endpoint = float(record["episodes"])
                for axis in (
                    axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1],
                ):
                    axis.axvline(
                        endpoint, color="black", linewidth=1.05,
                        linestyle=":", alpha=0.75,
                    )
                axes[1, 2].axvline(
                    float(record["training_total_mechanical_steps"]),
                    color="black", linewidth=1.05, linestyle=":", alpha=0.75,
                )
                axes[0, 0].axvline(
                    endpoint, color="black", linewidth=1.05,
                    linestyle=":", alpha=0.75,
                    label=f"user stop @ {int(endpoint)}",
                )
                axes[1, 1].plot(
                    [endpoint], [record["heldout_strict_success_rate"]],
                    marker="*", markersize=9, color=colour,
                    linestyle="none", label="checkpoint final-seed eval",
                )

        for axis, title in zip(axes[0], (
            "Strict-success moving rate",
            "Terminal q moving mean",
            "Milestone-reward moving mean",
        )):
            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylim(-0.02, 1.02)
        axes[0, 1].set_ylim(-0.02, 1.02)
        axes[0, 0].legend(title="Profile")

        axes[1, 0].set_title("Terminal geometry (moving mean)")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("F (solid), spread error (dashed)")
        axes[1, 0].set_ylim(bottom=-0.02)
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend(fontsize=7, ncol=2)

        axes[1, 1].set_title("Fixed-seed stochastic evaluation")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Rate")
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(fontsize=7, ncol=2)

        axes[1, 2].set_title("Strict-success rate by mechanics cost")
        axes[1, 2].set_xlabel("Cumulative mechanical steps (log scale)")
        axes[1, 2].set_ylabel("Moving success rate")
        axes[1, 2].set_xscale("log")
        axes[1, 2].set_ylim(-0.02, 1.02)
        axes[1, 2].grid(alpha=0.25)
        axes[1, 2].legend(title="Profile")
        stopped = next((
            row for row in ordered_records
            if row.get("artifact_status") == "user_stopped_at_checkpoint"
        ), None)
        title = task_name
        if stopped is not None:
            title += (
                f" — user-stopped atomic checkpoint at episode "
                f"{stopped['episodes']} (censored)"
            )
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(
            analysis_directory / f"{record['stage']}_{task_name}_profiles.png",
            dpi=180,
        )
        plt.close(figure)


def plot_random_first_hit_comparison(
    records: list[dict], audit: dict, analysis_directory: Path,
) -> None:
    """Plot learned first-hit cost against the compatible random RMST."""

    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda row: (
        list(runner.TASKS).index(row["task"]), row["profile"]
    ))
    labels = [f"{row['task']}\n{row['profile']}" for row in ordered]
    learned = []
    censored = []
    random_values = []
    for row in ordered:
        hit = row["first_success_mechanical_steps"]
        learned.append(float(
            hit if hit is not None else row["training_total_mechanical_steps"]
        ))
        censored.append(hit is None)
        random_values.append(float(
            audit["tasks"][row["task"]]["random_summary"][
                "restricted_mean_first_success_mechanical_steps"
            ]
        ))
    x = np.arange(len(ordered), dtype=float)
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(10.0, 0.72 * len(ordered)), 5.0))
    bars = axis.bar(
        x - width / 2, learned, width, color="tab:blue", label="ori5 first hit"
    )
    for bar, is_censored in zip(bars, censored):
        if is_censored:
            bar.set_hatch("///")
            bar.set_alpha(0.55)
    axis.bar(
        x + width / 2, random_values, width, color="0.55",
        label="uniform-random restricted mean",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Mechanical steps (log scale)")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title(
        "First strict-flat discovery: ori5 versus saved uniform random\n"
        "hatched ori5 bars are censored at the observed training cost"
    )
    figure.tight_layout()
    figure.savefig(
        analysis_directory / f"{ordered[0]['stage']}_random_first_hit_comparison.png",
        dpi=180,
    )
    plt.close(figure)


def plot_random_discovery_survival(
    records: list[dict], audit: dict, analysis_directory: Path,
) -> None:
    """Plot random discovery ECDF/survival with administrative censor ticks."""

    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_name = records[0]["stage"]
    for task_name in runner.TASKS:
        task_records = sorted(
            (row for row in records if row["task"] == task_name),
            key=lambda row: row["profile"],
        )
        if not task_records:
            continue
        baseline = audit["tasks"][task_name]["random_summary"]
        total = int(baseline["search_count"])
        panels = (
            (
                "first_success_episodes", "common_episode_censor_cap",
                "first_success_episode", "Episodes",
            ),
            (
                "first_success_mechanical_steps",
                "common_mechanical_censor_cap",
                "first_success_mechanical_steps", "Mechanical steps",
            ),
        )
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
        for axis, (events_key, cap_key, learned_key, xlabel) in zip(axes, panels):
            events = sorted(float(value) for value in baseline[events_key])
            cap = float(baseline[cap_key])
            start = 1.0
            x = np.asarray([start, *events, cap], dtype=float)
            cdf = np.asarray(
                [0.0, *[(index + 1) / total for index in range(len(events))],
                 len(events) / total],
                dtype=float,
            )
            axis.step(
                x, cdf, where="post", color="tab:blue", linewidth=2.0,
                label="Uniform-random discovery ECDF",
            )
            axis.step(
                x, 1.0 - cdf, where="post", color="tab:gray",
                linewidth=1.7, linestyle="--", label="Uniform-random survival",
            )
            censored = int(baseline["censored_count"])
            if censored:
                axis.plot(
                    [cap], [censored / total], marker="|", markersize=16,
                    markeredgewidth=2.2, color="black", linestyle="none",
                    label=f"Administrative censor (n={censored})",
                )
            for index, row in enumerate(task_records):
                hit = row[learned_key]
                if hit is not None:
                    axis.axvline(
                        float(hit), color=f"C{index + 2}", linewidth=1.25,
                        alpha=0.9, label=f"ori5 {row['profile']} first hit",
                    )
                else:
                    observed = (
                        float(row["episodes"]) if learned_key.endswith("episode")
                        else float(row["training_total_mechanical_steps"])
                    )
                    axis.plot(
                        [observed], [0.03 + 0.025 * index], marker="|",
                        markersize=12, markeredgewidth=1.8, color=f"C{index + 2}",
                        linestyle="none", label=f"ori5 {row['profile']} censored",
                    )
            axis.set_xscale("log")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel(f"First strict-flat discovery: {xlabel} (log scale)")
            axis.set_ylabel("Empirical fraction")
            axis.grid(alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[1].legend(
            by_label.values(), by_label.keys(), fontsize=7.5,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
        )
        figure.suptitle(
            f"{task_name}: first-discovery reference\n"
            "Episode/mechanics only; calibrated wall comparison is shown separately."
        )
        figure.tight_layout()
        figure.savefig(
            analysis_directory
            / f"{stage_name}_{task_name}_random_ecdf_survival.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(figure)


def add_random_comparison(records: list[dict], audit: dict) -> list[dict]:
    result = []
    for row in records:
        baseline = audit["tasks"][row["task"]]["random_summary"]
        first_episode = row["first_success_episode"]
        first_steps = row["first_success_mechanical_steps"]
        random_episode_rmst = baseline.get(
            "restricted_mean_first_success_episode"
        )
        random_rmst = baseline.get("restricted_mean_first_success_mechanical_steps")
        comparison = dict(row)
        comparison.update({
            "random_baseline_compatible": audit["tasks"][row["task"]][
                "compatible"
            ],
            "random_search_count": baseline.get("search_count"),
            "random_success_rate": baseline.get("success_rate"),
            "random_common_episode_censor_cap": baseline.get(
                "common_episode_censor_cap"
            ),
            "random_common_mechanical_censor_cap": baseline.get(
                "common_mechanical_censor_cap"
            ),
            "random_restricted_mean_first_success_episode": (
                random_episode_rmst
            ),
            "random_restricted_mean_first_success_mechanical_steps": random_rmst,
            "rl_first_hit_beats_random_episode_rmst": (
                first_episode is not None and random_episode_rmst is not None
                and float(first_episode) < float(random_episode_rmst)
            ),
            "rl_first_hit_episode_rmst_speedup": (
                float(random_episode_rmst) / float(first_episode)
                if first_episode not in (None, 0)
                and random_episode_rmst is not None else None
            ),
            "rl_first_hit_beats_random_rmst": (
                first_steps is not None and random_rmst is not None
                and float(first_steps) < float(random_rmst)
            ),
            "rl_first_hit_mechanical_rmst_speedup": (
                float(random_rmst) / float(first_steps)
                if first_steps not in (None, 0) and random_rmst is not None
                else None
            ),
            "random_wall_time_comparison_status": baseline.get(
                "wall_time_comparison_status"
            ),
        })
        result.append(comparison)
    return result


def quantile_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean": None, "q00": None, "q25": None, "q50": None,
            "q75": None, "q100": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "q00": float(np.quantile(array, 0.00)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q100": float(np.quantile(array, 1.00)),
    }


def episode_reward_decomposition(row: dict) -> dict:
    flatness = float_value(row, "ori5_total_flatness_progress_reward")
    target = float_value(row, "ori5_total_target_progress_reward")
    fixed = float_value(row, "ori5_total_terminal_fixed_reward")
    quality = float_value(row, "ori5_total_terminal_quality_reward")
    speed = float_value(row, "ori5_total_terminal_speed_reward")
    correction = float_value(row, "ori5_total_terminal_correction_reward")
    return {
        "episode": int(float_value(row, "episode")),
        "termination_reason": row.get("termination_reason", "unknown"),
        "total_reward": float_value(row, "total_reward"),
        "base_reward": flatness + target,
        "flatness_progress_reward": flatness,
        "target_progress_reward": target,
        "milestone_reward": float_value(row, "ori5_total_milestone_reward"),
        "spread_milestone_reward": float_value(
            row, "ori5_total_spread_milestone_reward"
        ),
        "joint_milestone_reward": float_value(
            row, "ori5_total_joint_milestone_reward"
        ),
        "terminal_reward": fixed + quality + speed + correction,
        "terminal_fixed_reward": fixed,
        "terminal_quality_reward": quality,
        "terminal_speed_reward": speed,
        "terminal_correction_reward": correction,
        "q": float_value(row, "near_miss_q"),
        "last_valid_flatness_ratio": float_value(
            row, "ori5_last_valid_flatness_ratio"
        ),
        "last_valid_spread_error": float_value(
            row, "ori5_last_valid_spread_error"
        ),
        "terminal_flatness_ratio": float_value(row, "flatness_ratio"),
        "terminal_spread_error": float_value(row, "spread_error"),
        "spread_milestone_awarded": int(float_value(
            row, "ori5_spread_milestone_awarded"
        )),
        "joint_milestones_awarded": int(float_value(
            row, "ori5_joint_milestones_awarded"
        )),
    }


def water1_termination_diagnostics(records: list[dict]) -> dict:
    """Quantify the milestone/crossing attractor for every completed profile."""

    profile_diagnostics = {}
    flat_rows = []
    b_first_success = None
    metric_names = (
        "total_reward", "base_reward", "milestone_reward", "q",
        "terminal_reward", "flatness_progress_reward",
        "target_progress_reward", "last_valid_flatness_ratio",
        "last_valid_spread_error",
    )
    for record in sorted(
        (row for row in records if row["task"] == "water1"),
        key=lambda row: row["profile"],
    ):
        profile = record["profile"]
        history = load_csv(
            Path(record["output_directory"]) / "training_history.csv"
        )
        decomposed = [episode_reward_decomposition(row) for row in history]
        reasons = sorted({row["termination_reason"] for row in decomposed})
        reason_rows = {}
        for reason in reasons:
            selected = [
                row for row in decomposed if row["termination_reason"] == reason
            ]
            summaries = {
                name: quantile_summary([row[name] for row in selected])
                for name in metric_names
            }
            reason_record = {
                "profile": profile,
                "termination_reason": reason,
                "count": len(selected),
                "rate": len(selected) / len(decomposed),
                "metrics": summaries,
            }
            if reason == "self_crossing":
                spread = [row["spread_milestone_awarded"] > 0 for row in selected]
                joint = [row["joint_milestones_awarded"] > 0 for row in selected]
                reason_record["crossing_milestones"] = {
                    "any_milestone_rate": float(np.mean([
                        row["milestone_reward"] > 0.0 for row in selected
                    ])),
                    "spread_milestone_rate": float(np.mean(spread)),
                    "joint_milestone_rate": float(np.mean(joint)),
                    "spread_only_rate": float(np.mean([
                        has_spread and not has_joint
                        for has_spread, has_joint in zip(spread, joint)
                    ])),
                    "no_milestone_rate": float(np.mean([
                        not has_spread and not has_joint
                        for has_spread, has_joint in zip(spread, joint)
                    ])),
                    "joint_milestone_count_distribution": {
                        str(level): sum(
                            row["joint_milestones_awarded"] == level
                            for row in selected
                        )
                        for level in range(4)
                    },
                }
            reason_rows[reason] = reason_record
            flattened = {
                "profile": profile,
                "termination_reason": reason,
                "count": len(selected),
                "rate": len(selected) / len(decomposed),
            }
            for metric, summary in summaries.items():
                for statistic, value in summary.items():
                    flattened[f"{metric}_{statistic}"] = value
            if reason == "self_crossing":
                flattened.update(reason_record["crossing_milestones"])
                flattened["joint_milestone_count_distribution"] = json.dumps(
                    flattened["joint_milestone_count_distribution"],
                    sort_keys=True,
                )
            flat_rows.append(flattened)
        successes = [
            row for row in decomposed if row["termination_reason"] == "success"
        ]
        first_success = min(successes, key=lambda row: row["episode"]) if successes else None
        crossings = [
            row for row in decomposed
            if row["termination_reason"] == "self_crossing"
        ]
        profile_diagnostics[profile] = {
            "episodes": len(decomposed),
            "first_success": first_success,
            "termination_reasons": reason_rows,
            "crossing_count": len(crossings),
        }
        if profile == "B" and first_success is not None:
            b_first_success = first_success
            if int(first_success["episode"]) != 340:
                b_first_success = dict(first_success)
                b_first_success["note"] = (
                    "Observed first success episode differs from the expected 340."
                )
    comparison = None
    b_cross = profile_diagnostics.get("B", {}).get(
        "termination_reasons", {}
    ).get("self_crossing")
    if b_first_success is not None and b_cross is not None:
        comparison = {
            "profile": "B",
            "success_episode": b_first_success,
            "crossing_distribution": b_cross,
            "success_minus_crossing_mean": {
                metric: (
                    b_first_success[metric]
                    - b_cross["metrics"][metric]["mean"]
                )
                for metric in metric_names
            },
        }
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "completed_profiles": sorted(profile_diagnostics),
        "profiles": profile_diagnostics,
        "profile_B_first_success_vs_crossing": comparison,
        "flat_rows": flat_rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(runner.STAGES), default="stage1")
    parser.add_argument(
        "--analysis-dir", default=str(DEFAULT_ANALYSIS_DIRECTORY)
    )
    parser.add_argument(
        "--replay-random-episodes", type=int, default=0,
        help="Replay this many saved uniform episodes per task (0 = static audit).",
    )
    parser.add_argument(
        "--replay-random-tasks", nargs="+", choices=tuple(runner.TASKS),
        default=None,
        help="Restrict functional replay to these tasks; static-audit all others.",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    analysis_directory = Path(args.analysis_dir).resolve()
    runner.ensure_within_experiment_root(analysis_directory)
    analysis_directory.mkdir(parents=True, exist_ok=True)

    audit = audit_random_baselines(
        args.replay_random_episodes, args.replay_random_tasks
    )
    audit_path = analysis_directory / "random_baseline_compatibility.json"
    audit = preserve_stronger_functional_replays(audit, audit_path)
    runner.atomic_json(audit_path, audit)
    records, statuses = collect_stage(args.stage)
    write_csv(analysis_directory / f"{args.stage}_status.csv", statuses)
    comparison_records = add_random_comparison(records, audit)
    write_csv(
        analysis_directory / f"{args.stage}_run_summary.csv",
        comparison_records,
    )
    if args.stage in ("stage1", "stage2") and any(
        row["task"] == "water1" for row in records
    ):
        water_diagnostics = water1_termination_diagnostics(records)
        runner.atomic_json(
            analysis_directory
            / f"{args.stage}_water1_termination_diagnostics.json",
            {
                key: value for key, value in water_diagnostics.items()
                if key != "flat_rows"
            },
        )
        write_csv(
            analysis_directory
            / f"{args.stage}_water1_termination_diagnostics.csv",
            water_diagnostics["flat_rows"],
        )
    reward_invariant = all(row["reward_valid"] for row in records)
    png_artifacts_valid = bool(records) and all(
        row["png_artifacts_valid"] for row in records
    )
    expected_count = 5 if args.stage == "full" else sum(
        len(runner.profile_names_for_stage_task(args.stage, task_name))
        for task_name in runner.TASKS
    )
    strict_complete_count = sum(
        row["artifact_status"] == "complete" for row in records
    )
    user_stopped_count = sum(
        row["artifact_status"] == "user_stopped_at_checkpoint"
        for row in records
    )
    if args.stage == "full":
        accepted_endpoint_collection = (
            len(records) == expected_count
            and strict_complete_count == expected_count - len(
                USER_STOPPED_FULL_ENDPOINTS
            )
            and user_stopped_count == len(USER_STOPPED_FULL_ENDPOINTS)
            and png_artifacts_valid
        )
    else:
        accepted_endpoint_collection = (
            len(records) == expected_count and png_artifacts_valid
        )
    # For --require-complete, "complete" means that every task has the exact
    # protocol-approved endpoint.  The report separately states that one
    # requested 10k training run was user-censored, avoiding any claim that all
    # trainers naturally reached their requested episode count.
    complete = accepted_endpoint_collection
    selection, rankings = ({}, [])
    if args.stage in STAGE_GROUP_PROFILE_CANDIDATES:
        selection, rankings = select_group_profiles(
            records, STAGE_GROUP_PROFILE_CANDIDATES[args.stage]
        )
        write_csv(
            analysis_directory / f"{args.stage}_group_ranking.csv", rankings
        )
        if complete and reward_invariant and len(selection) == len(runner.TASKS):
            selection_payload = {
                "schema_version": 1,
                "generated_utc": utc_now(),
                "source_stage": args.stage,
                "selection_rule": (
                    "lexicographic group-common ranking: discoveries, "
                    "worst/mean held-out and tail stability, assigned "
                    "first-hit fraction, near-miss q, endpoint-cost "
                    "fallback, crossing/wrong-flat"
                ),
                "groups": GROUPS,
                "group_profile_candidates": (
                    STAGE_GROUP_PROFILE_CANDIDATES[args.stage]
                ),
                "selected_profiles": selection,
            }
            if args.stage == "stage2":
                stage2_manifest_path, stage2_manifest = (
                    runner.current_stage_manifest("stage2")
                )
                selection_payload["stage2_manifest"] = {
                    "path": str(stage2_manifest_path),
                    "manifest_fingerprint": stage2_manifest[
                        "manifest_fingerprint"
                    ],
                    "source_sha256": stage2_manifest["source_sha256"],
                }
                selection_payload["model_file_sha256_by_task"] = {
                    task_name: runner.task_model_file_hashes(task)
                    for task_name, task in runner.TASKS.items()
                }
            runner.atomic_json(
                analysis_directory / f"{args.stage}_selected_profiles.json",
                selection_payload,
            )
            # Full-stage runner consumes this stable alias; the source_stage
            # field prevents ambiguity when Stage 2 supersedes Stage 1.
            stable_selection_path = analysis_directory / "selected_profiles.json"
            if args.stage == "stage2" or not stable_selection_path.is_file():
                runner.atomic_json(stable_selection_path, selection_payload)
    if not args.no_plots and records:
        plot_stage_learning_curves(records, analysis_directory)
        plot_random_first_hit_comparison(records, audit, analysis_directory)
        plot_random_discovery_survival(records, audit, analysis_directory)
    report = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "stage": args.stage,
        "expected_artifacts": expected_count,
        "valid_analysis_endpoints": len(records),
        "valid_complete_artifacts": strict_complete_count,
        "complete": complete,
        "accepted_endpoint_collection": accepted_endpoint_collection,
        "strict_natural_completions": strict_complete_count,
        "accepted_user_stopped_checkpoints": user_stopped_count,
        "all_requested_training_runs_reached_episode_target": (
            user_stopped_count == 0 and strict_complete_count == expected_count
        ),
        "completion_semantics": (
            "Every task has a validated analysis endpoint; full/torus2 is "
            "explicitly censored at the user-stopped atomic episode-8192 "
            "checkpoint, while the other four full runs satisfy the strict "
            "10,000-episode artifact schema."
            if args.stage == "full" else
            "Every expected artifact satisfies the strict stage schema."
        ),
        "user_stopped_full_endpoints": (
            USER_STOPPED_FULL_ENDPOINTS if args.stage == "full" else {}
        ),
        "reward_invariant": reward_invariant,
        "png_artifacts_valid": png_artifacts_valid,
        "random_baselines_compatible": audit["all_compatible"],
        "selected_profiles": selection,
        "status_counts": {
            state: sum(row["status"] == state for row in statuses)
            for state in sorted({row["status"] for row in statuses})
        },
    }
    runner.atomic_json(
        analysis_directory / f"{args.stage}_analysis_report.json", report
    )
    print(json.dumps(report, indent=2))
    if args.require_complete and not complete:
        return 2
    if not reward_invariant or not audit["all_compatible"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
