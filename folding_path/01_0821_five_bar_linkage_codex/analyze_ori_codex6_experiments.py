"""Analyze ori6 staged experiments without mutating training artifacts.

The analyzer has three roles: (1) audit ori5 rollback-only counterfactuals,
(2) apply explicit rescue/regression/full promotion gates, and (3) produce
episode/mechanics learning curves and random first-discovery comparisons.
Plane diagnostics are derived from fixed-seed checkpoint rollouts and written
below ``experiments_ori6/analysis``; canonical training directories are hashed
before/after each derived evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import run_ori_codex6_experiments as runner


ANALYSIS_DIRECTORY = runner.EXPERIMENT_ROOT / "analysis"
ORI5_FULL = runner.SCRIPT_DIRECTORY / "experiments_ori5" / "full"
ROLLING_WINDOW = 200


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def f(row: dict, name: str, default=0.0) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def mean(values) -> float | None:
    data = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(data)) if data else None


def quantiles(values) -> dict:
    data = np.asarray([float(value) for value in values], dtype=float)
    if not len(data):
        return {"count": 0, "mean": None, "q05": None, "median": None, "q95": None}
    return {
        "count": int(len(data)), "mean": float(np.mean(data)),
        "q05": float(np.quantile(data, 0.05)),
        "median": float(np.median(data)),
        "q95": float(np.quantile(data, 0.95)),
    }


def bootstrap_mean_difference(left, right, seed=6202601, samples=20000) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return {"estimable": False, "difference": None, "ci95": [None, None]}
    generator = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for index in range(samples):
        differences[index] = (
            np.mean(generator.choice(left, size=len(left), replace=True))
            - np.mean(generator.choice(right, size=len(right), replace=True))
        )
    return {
        "estimable": True,
        "difference": float(np.mean(left) - np.mean(right)),
        "ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "bootstrap_samples": samples,
    }


def canonical_files_digest(directory: Path, names: tuple[str, ...]) -> dict:
    return {
        name: runner.file_sha256(directory / name)
        for name in names if (directory / name).is_file()
    }


def resolve_job_artifact(job: dict, allow_partial=True) -> tuple[Path, dict, str]:
    destination = Path(job["output_directory"])
    status_payload = runner.load_status(job, destination)
    if status_payload is None:
        raise RuntimeError(f"Missing runner provenance for {job['job_id']}")
    if status_payload.get("job_fingerprint") != job["fingerprint"]:
        raise RuntimeError(f"Stale runner provenance for {job['job_id']}")
    effective, validation = runner.discover_output(job, destination)
    if effective is not None and validation and validation["valid"]:
        return effective, validation, "complete"
    state, path, detail = runner.classify(job, False, [])
    if allow_partial and state == "early_stopped" and detail and detail["valid"]:
        return path, detail, state
    raise RuntimeError(
        f"Artifact is not analyzable for {job['job_id']}: state={state}, detail={detail}"
    )


def manifest_uses_current_candidate_authorities(payload: dict, mode: str) -> bool:
    """Return whether an immutable manifest binds the current evidence files.

    Earlier P0X2 manifests can have the same source and job fingerprints while
    referring to missing or superseded offline evidence.  Select by the
    evidence bindings themselves; the mutable stale-manifest index remains
    diagnostic metadata rather than an authority.
    """

    if mode not in (
        runner.P0_CROSS2_MODE, runner.P0_CROSS2_FRONTIER12_MODE,
        runner.P0_CROSS2_ENTROPY3_MODE, runner.P0_CROSS2_WIDTH2_MODE,
        runner.P0_CROSS2_REJECTED_CONTINUATION_MODE,
        runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        return True
    if (
        payload.get("core_frontier_mode") != runner.core_frontier_mode(mode)
        or payload.get("candidate_reward_sha256")
        != runner.candidate_reward_sha256(mode)
    ):
        return False
    if mode == runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
        protocol_name = payload.get("protocol", {}).get("name")
        if protocol_name not in (
            "spectral_subspace_diagnostic", "spectral_nonregression",
        ):
            return False
        protocol = runner.PROTOCOLS[protocol_name]
        selected_tasks = list(protocol.tasks)
        parent_mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        profiles = {
            runner.TASKS[name].profile:
            runner.PROFILES[runner.TASKS[name].profile]
            for name in selected_tasks
        }
        expected_optimizer = {
            name: runner.candidate_optimizer_sha256(mode, profile)
            for name, profile in profiles.items()
        }
        parent_optimizer = {
            name: runner.candidate_optimizer_sha256(parent_mode, profile)
            for name, profile in profiles.items()
        }
        expected_mechanics = {
            name: runner.candidate_mechanics_sha256(
                mode, runner.TASKS[name]
            ) for name in selected_tasks
        }
        parent_mechanics = {
            name: runner.candidate_mechanics_sha256(
                parent_mode, runner.TASKS[name]
            ) for name in selected_tasks
        }
        if protocol_name == "spectral_subspace_diagnostic":
            expected_scope = {
                "stage": "spectral_subspace_diagnostic",
                "tasks": selected_tasks,
                "seeds": list(protocol.seeds),
                "episodes": protocol.episodes,
                "checkpoint_every": protocol.checkpoint_every,
                "early_gate_checkpoints": [512, 1024, 2048],
                "task_cohorts_are_independent_and_fixed": True,
                "paired_parent_seeds": [3201, 3202, 3203],
                "parent_torus_endpoint": 512,
                "parent_water1_endpoint": 1024,
            }
            expected_gate = runner.spectral_subspace_gate_specification()
        else:
            expected_scope = {
                "stage": "spectral_nonregression",
                "tasks": selected_tasks,
                "seeds": list(protocol.seeds),
                "episodes": protocol.episodes,
                "checkpoint_every": protocol.checkpoint_every,
                "live_early_gate_checkpoints": [512],
                "implementations": list(protocol.implementations),
                "jobs_per_task": 6,
                "total_jobs": 18,
                "task_cohorts_are_independent_and_fixed": True,
            }
            expected_gate = runner.spectral_nonregression_gate_specification()
        expected_authorities = [
            (
                "parent_RC_reject_authority",
                "parent_RC_reject_authority_sha256",
                runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT,
            ),
            (
                "spectral_offline_design_authority",
                "spectral_offline_design_authority_sha256",
                runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY,
            ),
        ]
        if (
            payload.get("parent_candidate_mode") != parent_mode
            or payload.get("ori6_candidate")
            != runner.SPECTRAL_SUBSPACE_CANDIDATE_NAME
            or payload.get("ori6_candidate_schema_version")
            != runner.SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
            or payload.get("parent_ori6_candidate")
            != runner.REJECTED_CROSSING_CANDIDATE_NAME
            or payload.get("candidate_additive_source_sha256") != {
                **runner.rejected_crossing_additive_source_hashes(),
                **runner.spectral_subspace_additive_source_hashes(),
            }
            or payload.get("spectral_additive_source_sha256")
            != runner.spectral_subspace_additive_source_hashes()
            or payload.get("candidate_orchestration_source_sha256")
            != runner.rejected_crossing_orchestration_source_hashes()
            or payload.get("parent_candidate_reward_sha256")
            != runner.candidate_reward_sha256(parent_mode)
            or payload.get("single_changed_reward_parameter") != {
                "name": "ori6_spectral_subspace.weight",
                "parent_value": 0.0,
                "candidate_value": runner.SPECTRAL_SUBSPACE_WEIGHT,
                "parent_bit_exact_at_zero": True,
            }
            or payload.get("spectral_formula_sha256")
            != runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
            or payload.get("spectral_formula")
            != runner.spectral_subspace_runtime_identity()["formula"]
            or payload.get("candidate_optimizer_sha256_by_profile")
            != expected_optimizer
            or payload.get("parent_candidate_optimizer_sha256_by_profile")
            != parent_optimizer
            or payload.get("optimizer_identity_reused_from_parent") is not True
            or expected_optimizer != parent_optimizer
            or payload.get("candidate_mechanics_sha256_by_task")
            != expected_mechanics
            or payload.get("parent_candidate_mechanics_sha256_by_task")
            != parent_mechanics
            or payload.get("physical_mechanics_identity_reused_from_parent")
            is not True
            or expected_mechanics != parent_mechanics
            or payload.get("candidate_transition_semantics_sha256")
            != runner.candidate_transition_semantics_sha256(mode)
            or payload.get("parent_transition_semantics_sha256")
            != runner.candidate_transition_semantics_sha256(parent_mode)
            or payload.get("crossing_transition_semantics_reused_from_parent")
            is not True
            or payload.get("screen_scope") != expected_scope
            or payload.get("early_gate_specification") != expected_gate
            or payload.get("random_baseline_authority")
            != runner.rejected_crossing_random_baseline_disposition()
        ):
            return False
        if protocol_name == "spectral_subspace_diagnostic":
            if (
                payload.get("parent_comparison_scope") != {
                    "same_seeds": [3201, 3202, 3203],
                    "common_prefix_only": True,
                    "performance_comparison_is_diagnostic_not_a_gate": True,
                    "missing_parent_1024_torus_does_not_invalidate": True,
                    "parent_artifacts_are_immutable_reject_evidence": True,
                }
                or payload.get("adoption_requirements") != {
                    "fresh_confirmation_seeds": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "fresh_successful_task_regression": True,
                    "screen_pass_does_not_authorize_full_training": True,
                }
            ):
                return False
        else:
            expected_reference = {
                "implementation": "ori5_reference",
                "reward_sha256": runner.canonical_digest({
                    "reward_design": runner.ORI5_REWARD_DESIGN,
                    "reward_schema_version": runner.ORI5_REWARD_SCHEMA_VERSION,
                    "effective_parameters": runner.expected_ori5_reward(),
                }),
                "mechanics_and_optimizer_profiles_are_exactly_paired": True,
            }
            if (
                payload.get("paired_reference") != expected_reference
                or payload.get("promotion_requirements") != {
                    "formal_spectral_report_explicitly_permits_"
                    "preregistered_regression": True,
                    "formal_spectral_report_does_not_authorize_"
                    "adoption_or_full": True,
                    "all_three_successful_task_nonregression_gates_must_pass": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "regression_pass_is_necessary_but_not_sufficient_for_full": True,
                }
            ):
                return False
            try:
                binding = runner.spectral_screen_report_binding()
            except (OSError, RuntimeError, ValueError):
                return False
            if not (
                payload.get("spectral_screen_report") == binding["path"]
                and payload.get("spectral_screen_report_sha256")
                == binding["sha256"]
                and payload.get("spectral_screen_report_status")
                == binding["status"]
            ):
                return False
            if binding["status"] == "bound_regression_permission_authority":
                expected_authorities.append((
                    "spectral_screen_report",
                    "spectral_screen_report_sha256",
                    runner.SPECTRAL_SUBSPACE_SCREEN_REPORT,
                ))
        for path_field, digest_field, expected_path in expected_authorities:
            if not expected_path.is_file() or not payload.get(path_field):
                return False
            try:
                if Path(payload[path_field]).resolve() != expected_path.resolve():
                    return False
            except (OSError, TypeError, ValueError):
                return False
            if payload.get(digest_field) != runner.file_sha256(expected_path):
                return False
        return True
    if mode == runner.P0_CROSS2_MODE:
        if payload.get("single_changed_reward_parameter") != {
            "name": "ori5_crossing_penalty",
            "P0_value": runner.candidate_crossing_penalty(runner.P0_MODE),
            "candidate_value": runner.candidate_crossing_penalty(mode),
        }:
            return False
        authority_fields = (
            (
                "offline_saved_path_ordering_authority",
                "offline_saved_path_ordering_authority_sha256",
                runner.P0_CROSS2_OFFLINE_AUDIT,
            ),
            (
                "P0_crossing_attractor_authority",
                "P0_crossing_attractor_authority_sha256",
                runner.P0_RECLASSIFICATION_REPORT,
            ),
        )
    elif mode == runner.P0_CROSS2_FRONTIER12_MODE:
        if (
            payload.get("parent_candidate_mode") != runner.P0_CROSS2_MODE
            or payload.get("parent_candidate_reward_sha256")
            != runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
            or payload.get("single_changed_reward_parameter") != {
                "name": "ori6_spread_frontier_bonus",
                "P0X2_value": runner.candidate_frontier_bonus(
                    runner.P0_CROSS2_MODE
                ),
                "candidate_value": runner.candidate_frontier_bonus(mode),
            }
        ):
            return False
        authority_fields = (
            (
                "offline_saved_path_ordering_authority",
                "offline_saved_path_ordering_authority_sha256",
                runner.P0_CROSS2_FRONTIER12_OFFLINE_AUDIT,
            ),
            (
                "parent_candidate_reject_authority",
                "parent_candidate_reject_authority_sha256",
                runner.P0_CROSS2_REJECT_REPORT,
            ),
        )
    elif mode == runner.P0_CROSS2_ENTROPY3_MODE:
        profile = runner.PROFILES[runner.TASKS["water1"].profile]
        if (
            payload.get("parent_candidate_mode") != runner.P0_CROSS2_MODE
            or payload.get("parent_candidate_reward_sha256")
            != runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
            or payload.get("reward_identity_reused_from_parent") is not True
            or payload.get("candidate_optimizer_sha256")
            != runner.candidate_optimizer_sha256(mode, profile)
            or payload.get("parent_candidate_optimizer_sha256")
            != runner.candidate_optimizer_sha256(
                runner.P0_CROSS2_MODE, profile
            )
            or payload.get("single_changed_training_parameter") != {
                "name": "entropy_coefficient",
                "P0X2_value": runner.candidate_entropy_coefficient(
                    runner.P0_CROSS2_MODE, profile
                ),
                "candidate_value": runner.candidate_entropy_coefficient(
                    mode, profile
                ),
            }
            or payload.get("screen_scope") != {
                "stage": "entropy_diagnostic",
                "tasks": ["water1"],
                "seeds": list(runner.PROTOCOLS["entropy_diagnostic"].seeds),
                "episodes": runner.PROTOCOLS["entropy_diagnostic"].episodes,
                "checkpoint_every": runner.PROTOCOLS["entropy_diagnostic"].checkpoint_every,
                "early_gate_checkpoints": [512, 1024],
            }
        ):
            return False
        authority_fields = (
            (
                "offline_saved_path_ordering_authority",
                "offline_saved_path_ordering_authority_sha256",
                runner.P0_CROSS2_OFFLINE_AUDIT,
            ),
            (
                "preceding_candidate_reject_authority",
                "preceding_candidate_reject_authority_sha256",
                runner.P0_CROSS2_FRONTIER12_REJECT_REPORT,
            ),
            (
                "entropy_preflight_authority",
                "entropy_preflight_authority_sha256",
                runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT,
            ),
        )
    elif mode == runner.P0_CROSS2_WIDTH2_MODE:
        task = runner.TASKS["water1"]
        profile = runner.PROFILES[task.profile]
        scope = payload.get("screen_scope", {})
        if (
            payload.get("parent_candidate_mode") != runner.P0_CROSS2_MODE
            or payload.get("parent_candidate_reward_sha256")
            != runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
            or payload.get("reward_identity_reused_from_parent") is not True
            or payload.get("candidate_optimizer_sha256")
            != runner.candidate_optimizer_sha256(mode, profile)
            or payload.get("parent_candidate_optimizer_sha256")
            != runner.candidate_optimizer_sha256(
                runner.P0_CROSS2_MODE, profile
            )
            or payload.get("optimizer_identity_reused_from_parent") is not True
            or payload.get("candidate_mechanics_sha256")
            != runner.candidate_mechanics_sha256(mode, task)
            or payload.get("parent_candidate_mechanics_sha256")
            != runner.candidate_mechanics_sha256(
                runner.P0_CROSS2_MODE, task
            )
            or payload.get("single_changed_mechanics_parameter") != {
                "name": "max_step_width",
                "P0X2_value": runner.candidate_max_step_width(
                    runner.P0_CROSS2_MODE, task
                ),
                "candidate_value": runner.candidate_max_step_width(mode, task),
            }
            or scope.get("stage") != "width_diagnostic"
            or scope.get("tasks") != ["water1"]
            or scope.get("seeds")
            != list(runner.PROTOCOLS["width_diagnostic"].seeds)
            or scope.get("episodes")
            != runner.PROTOCOLS["width_diagnostic"].episodes
            or scope.get("checkpoint_every")
            != runner.PROTOCOLS["width_diagnostic"].checkpoint_every
            or scope.get("early_gate_checkpoints") != [512, 1024, 2048]
            or payload.get("random_baseline_authority") != {
                "status": "not_used_for_bounded_mechanics_diagnostic",
                "reason": "max_step_width differs from the existing random baseline",
                "new_random_baseline_required_only_if_adopted": True,
            }
        ):
            return False
        authority_fields = (
            (
                "offline_saved_path_ordering_authority",
                "offline_saved_path_ordering_authority_sha256",
                runner.P0_CROSS2_OFFLINE_AUDIT,
            ),
            (
                "preceding_candidate_reject_authority",
                "preceding_candidate_reject_authority_sha256",
                runner.P0_CROSS2_ENTROPY3_REJECT_REPORT,
            ),
            (
                "width_feasibility_authority",
                "width_feasibility_authority_sha256",
                runner.P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT,
            ),
        )
    else:
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        selected_tasks = list(protocol.tasks)
        profiles = {
            runner.TASKS[name].profile:
            runner.PROFILES[runner.TASKS[name].profile]
            for name in selected_tasks
        }
        expected_optimizer = {
            name: runner.candidate_optimizer_sha256(mode, profile)
            for name, profile in profiles.items()
        }
        parent_optimizer = {
            name: runner.candidate_optimizer_sha256(
                runner.P0_CROSS2_MODE, profile
            )
            for name, profile in profiles.items()
        }
        expected_mechanics = {
            name: runner.candidate_mechanics_sha256(
                mode, runner.TASKS[name]
            )
            for name in selected_tasks
        }
        parent_mechanics = {
            name: runner.candidate_mechanics_sha256(
                runner.P0_CROSS2_MODE, runner.TASKS[name]
            )
            for name in selected_tasks
        }
        expected_scope = {
            "stage": "crossing_continuation_diagnostic",
            "tasks": selected_tasks,
            "seeds": list(protocol.seeds),
            "episodes": protocol.episodes,
            "checkpoint_every": protocol.checkpoint_every,
            "early_gate_checkpoints": [512, 1024, 2048],
            "task_cohorts_are_independent_and_fixed": True,
        }
        expected_change = {
            "name": "self_crossing_transition_policy",
            "P0X2_value": "terminate_on_self_crossing",
            "candidate_value": runner.REJECTED_CROSSING_POLICY,
            "P0X2_crossed_candidate_committed": True,
            "candidate_crossed_candidate_committed": False,
            "rejected_attempt_consumes_step": True,
            "last_safe_state_restored": True,
            "past_frontier_and_joint_ledgers_retained": True,
        }
        if (
            payload.get("parent_candidate_mode")
            != runner.P0_CROSS2_MODE
            or payload.get("ori6_candidate")
            != runner.REJECTED_CROSSING_CANDIDATE_NAME
            or payload.get("ori6_candidate_schema_version")
            != runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            or payload.get("parent_candidate_reward_sha256")
            != runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
            or payload.get("numeric_reward_hyperparameters_reused_from_parent")
            is not True
            or payload.get("candidate_additive_source_sha256")
            != runner.rejected_crossing_additive_source_hashes()
            or payload.get("candidate_orchestration_source_sha256")
            != runner.rejected_crossing_orchestration_source_hashes()
            or payload.get("candidate_optimizer_sha256_by_profile")
            != expected_optimizer
            or payload.get("parent_candidate_optimizer_sha256_by_profile")
            != parent_optimizer
            or payload.get("optimizer_identity_reused_from_parent") is not True
            or expected_optimizer != parent_optimizer
            or payload.get("candidate_mechanics_sha256_by_task")
            != expected_mechanics
            or payload.get("parent_candidate_mechanics_sha256_by_task")
            != parent_mechanics
            or payload.get("physical_mechanics_identity_reused_from_parent")
            is not True
            or expected_mechanics != parent_mechanics
            or payload.get("candidate_transition_semantics_sha256")
            != runner.candidate_transition_semantics_sha256(mode)
            or payload.get("parent_transition_semantics_sha256")
            != runner.candidate_transition_semantics_sha256(
                runner.P0_CROSS2_MODE
            )
            or payload.get("single_changed_environment_semantics")
            != expected_change
            or payload.get("screen_scope") != expected_scope
            or payload.get("early_gate_specification")
            != runner.crossing_continuation_gate_specification()
            or payload.get("parent_saved_path_scope") != (
                "numeric P0X2 reward and collision-free path evidence only; "
                "terminal self-crossing path returns are not reused"
            )
            or payload.get("random_baseline_authority")
            != runner.rejected_crossing_random_baseline_disposition()
            or payload.get("adoption_requirements") != {
                "fresh_random_baseline_all_five_tasks": True,
                "fresh_successful_task_regression": True,
                "screen_pass_does_not_authorize_full_training": True,
            }
        ):
            return False
        authority_fields = (
            (
                "parent_saved_path_ordering_authority",
                "parent_saved_path_ordering_authority_sha256",
                runner.P0_CROSS2_OFFLINE_AUDIT,
            ),
            (
                "preceding_candidate_reject_authority",
                "preceding_candidate_reject_authority_sha256",
                runner.P0_CROSS2_WIDTH2_REJECT_REPORT,
            ),
        )
    for path_field, digest_field, expected_path in authority_fields:
        if not expected_path.is_file():
            return False
        recorded_path = payload.get(path_field)
        if not recorded_path:
            return False
        try:
            if Path(recorded_path).resolve() != expected_path.resolve():
                return False
        except (OSError, TypeError, ValueError):
            return False
        if payload.get(digest_field) != runner.file_sha256(expected_path):
            return False
    return True


def current_manifest_authority(protocol_name: str, mode: str, jobs: list[dict]) -> dict:
    expected_fingerprints = sorted(job["fingerprint"] for job in jobs)
    pattern = f"{protocol_name}_{runner.MODE_SLUGS[mode]}_*.json"
    matches = []
    for path in (runner.EXPERIMENT_ROOT / "manifests").glob(pattern):
        if "stale_manifest_exclusions" in path.name:
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        content = dict(payload)
        recorded = content.pop("manifest_fingerprint", None)
        content.pop("created_utc", None)
        if recorded != runner.canonical_digest(content):
            continue
        if (
            payload.get("protocol", {}).get("name") == protocol_name
            and payload.get("candidate_mode") == mode
            and payload.get("source_sha256") == runner.source_hashes()
            and sorted(row["fingerprint"] for row in payload.get("jobs", []))
            == expected_fingerprints
            and manifest_uses_current_candidate_authorities(payload, mode)
        ):
            matches.append((path.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one current immutable manifest for {protocol_name}/{mode}; "
            f"found {len(matches)}"
        )
    path, payload = matches[0]
    return {
        "path": str(path),
        "sha256": runner.file_sha256(path),
        "manifest_fingerprint": payload["manifest_fingerprint"],
    }


def first_hit(history: list[dict], batch_size: int) -> dict:
    mechanics = 0
    projector_calls = 0
    for row in history:
        mechanics += int(f(row, "steps"))
        projector_calls += int(f(row, "projection_attempts"))
        if int(f(row, "success")):
            episode = int(f(row, "episode"))
            return {
                "episode": episode,
                "mechanical_steps": mechanics,
                "projector_calls": projector_calls,
                "policy_updates_before_hit": (episode - 1) // batch_size,
                "censored": False,
            }
    return {
        "episode": None,
        "mechanical_steps": None,
        "projector_calls": None,
        "policy_updates_before_hit": None,
        "censored": True,
        "censor_episode": len(history),
        "censor_mechanical_steps": mechanics,
        "censor_projector_calls": projector_calls,
    }


def rate(rows: list[dict], field: str) -> float:
    return float(np.mean([f(row, field) for row in rows])) if rows else 0.0


def rejected_crossing_constructive_proxy(row: dict, task_name: str) -> bool:
    if not (
        f(row, "self_crossing") == 0.0
        and f(row, "correction_failed") == 0.0
        and f(row, "no_modes") == 0.0
    ):
        return False
    flatness = f(row, "flatness_ratio", math.inf)
    spread_error = f(row, "spread_error", math.inf)
    if task_name == "torus2":
        return flatness + spread_error ** 2 <= 0.5
    if task_name == "water1":
        return flatness <= 0.5 and spread_error <= 0.1
    if task_name in ("five_d2", "five_d3", "watersqrt2"):
        return flatness <= 0.5 and spread_error <= 0.1
    raise ValueError(f"No rejected-crossing proxy for {task_name}")


def rejected_crossing_summary(rows: list[dict], task_name: str) -> dict:
    rejected = int(sum(f(row, "ori6_rejected_crossing_count") for row in rows))
    streaks = int(sum(
        f(row, "ori6_rejected_crossing_streak_count") for row in rows
    ))
    escapes = int(sum(
        f(row, "ori6_escape_after_rejected_crossing_count") for row in rows
    ))
    steps = int(sum(f(row, "steps") for row in rows))
    required = {
        "ori6_candidate", "ori6_candidate_schema_version",
        "ori6_crossing_policy", "ori6_rejected_crossing_count",
        "ori6_rejected_crossing_streak_count",
        "ori6_escape_after_rejected_crossing_count",
        "ori6_rejected_crossing_rate",
        "ori6_rejected_crossing_escape_rate",
        "ori6_unescaped_rejected_crossing_streak_count",
        "ori6_max_consecutive_rejected_crossings",
        "ori6_success_after_rejected_crossing",
        "ori6_total_rejected_crossing_reward",
        "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
    }

    def row_integrity(row: dict) -> bool:
        if not required.issubset(row):
            return False
        count = f(row, "ori6_rejected_crossing_count", math.inf)
        streak_count = f(
            row, "ori6_rejected_crossing_streak_count", math.inf
        )
        escape_count = f(
            row, "ori6_escape_after_rejected_crossing_count", math.inf
        )
        unescaped = f(
            row,
            "ori6_unescaped_rejected_crossing_streak_count",
            math.inf,
        )
        maximum_consecutive = f(
            row, "ori6_max_consecutive_rejected_crossings", math.inf
        )
        success_after = f(
            row, "ori6_success_after_rejected_crossing", math.inf
        )
        steps = f(row, "steps", -1.0)
        integer_values = (
            count, streak_count, escape_count, unescaped,
            maximum_consecutive, success_after,
        )
        return bool(
            row.get("ori6_candidate")
            == runner.REJECTED_CROSSING_CANDIDATE_NAME
            and row.get("ori6_candidate_schema_version")
            == runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            and row.get("ori6_crossing_policy")
            == runner.REJECTED_CROSSING_POLICY
            and all(np.isfinite(value) and value.is_integer()
                    for value in integer_values)
            and 0.0 <= escape_count <= streak_count <= count <= steps
            and 0.0 <= maximum_consecutive <= count
            and success_after in (0.0, 1.0)
            and math.isclose(
                f(row, "ori6_rejected_crossing_rate", math.inf),
                count / max(steps, 1.0), rel_tol=0.0, abs_tol=1.0e-12,
            )
            and math.isclose(
                f(row, "ori6_rejected_crossing_escape_rate", math.inf),
                escape_count / max(streak_count, 1.0),
                rel_tol=0.0, abs_tol=1.0e-12,
            )
            and math.isclose(
                unescaped, streak_count - escape_count,
                rel_tol=0.0, abs_tol=1.0e-12,
            )
            and f(row, "self_crossing") == 0.0
            and abs(f(
                row,
                "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
                math.inf,
            )) <= 1.0e-10
            and math.isfinite(f(
                row, "ori6_total_rejected_crossing_reward", math.inf
            ))
        )

    integrity = bool(rows) and all(row_integrity(row) for row in rows)
    return {
        "episodes": len(rows),
        "rejected_action_count": rejected,
        "episode_step_count": steps,
        "pooled_rejected_action_rate": float(rejected / max(steps, 1)),
        "rejected_streak_count": streaks,
        "escape_after_rejected_streak_count": escapes,
        "pooled_escape_rate": (
            float(escapes / streaks) if streaks else None
        ),
        "constructive_proxy_episodes": int(sum(
            rejected_crossing_constructive_proxy(row, task_name)
            for row in rows
        )),
        "constructive_proxy_rate": float(np.mean([
            rejected_crossing_constructive_proxy(row, task_name)
            for row in rows
        ])) if rows else 0.0,
        "valid_joint_or_strict_episodes": int(sum(
            (
                f(row, "flatness_ratio", math.inf) <= 0.1
                and f(row, "spread_error", math.inf) <= 0.05
                and f(row, "self_crossing") == 0.0
                and f(row, "correction_failed") == 0.0
                and f(row, "no_modes") == 0.0
            )
            or f(row, "success") > 0.5
            for row in rows
        )),
        "strict_success_episodes": int(sum(
            f(row, "success") > 0.5 for row in rows
        )),
        "transition_integrity_pass": integrity,
    }


def spectral_subspace_summary(rows: list[dict]) -> dict:
    required = {
        "ori6_spectral_candidate",
        "ori6_spectral_candidate_schema_version",
        "ori6_spectral_formula_sha256",
        "ori6_spectral_parent_candidate",
        "ori6_spectral_weight",
        "ori6_spectral_relative_epsilon",
        "ori6_spectral_total_progress_reward",
        "ori6_spectral_total_rollback_reward",
        "ori6_spectral_total_reward",
        "ori6_spectral_cumulative_reward",
        "ori6_spectral_terminal_potential",
        "ori6_spectral_terminal_delta",
        "ori6_spectral_terminal_rho",
        "ori6_spectral_maximum_potential",
        "ori6_spectral_maximum_delta",
        "ori6_spectral_maximum_rho",
        "ori6_spectral_reward_decomposition_error",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_nonfinite_component_count",
    }

    def row_integrity(row: dict) -> bool:
        if not required.issubset(row):
            return False
        finite_fields = (
            "ori6_spectral_total_progress_reward",
            "ori6_spectral_total_rollback_reward",
            "ori6_spectral_total_reward",
            "ori6_spectral_cumulative_reward",
            "ori6_spectral_terminal_potential",
            "ori6_spectral_terminal_delta",
            "ori6_spectral_terminal_rho",
            "ori6_spectral_maximum_potential",
            "ori6_spectral_maximum_delta",
            "ori6_spectral_maximum_rho",
        )
        total = f(row, "ori6_spectral_total_reward", math.inf)
        cumulative = f(row, "ori6_spectral_cumulative_reward", math.inf)
        rollback_terminal = row.get("termination_reason") in {
            "wrong_flat", "correction_failed", "no_modes", "self_crossing",
        }
        return bool(
            row["ori6_spectral_candidate"]
            == runner.SPECTRAL_SUBSPACE_CANDIDATE_NAME
            and row["ori6_spectral_candidate_schema_version"]
            == runner.SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
            and row["ori6_spectral_formula_sha256"]
            == runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
            and row["ori6_spectral_parent_candidate"]
            == runner.REJECTED_CROSSING_CANDIDATE_NAME
            and runner.values_equal(
                f(row, "ori6_spectral_weight"),
                runner.SPECTRAL_SUBSPACE_WEIGHT,
            )
            and runner.values_equal(
                f(row, "ori6_spectral_relative_epsilon"),
                runner.SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
            )
            and all(math.isfinite(f(row, field, math.inf)) for field in finite_fields)
            and all(
                -1.0e-12 <= f(row, field) <= 1.0 + 1.0e-12
                for field in (
                    "ori6_spectral_terminal_potential",
                    "ori6_spectral_terminal_delta",
                    "ori6_spectral_terminal_rho",
                    "ori6_spectral_maximum_potential",
                    "ori6_spectral_maximum_delta",
                    "ori6_spectral_maximum_rho",
                )
            )
            and math.isclose(
                total,
                f(row, "ori6_spectral_total_progress_reward")
                + f(row, "ori6_spectral_total_rollback_reward"),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            and math.isclose(total, cumulative, rel_tol=0.0, abs_tol=1.0e-10)
            and (
                not rollback_terminal
                or abs(cumulative) <= 1.0e-10
            )
            and abs(f(
                row, "ori6_spectral_reward_decomposition_error", math.inf
            )) <= 1.0e-10
            and abs(f(
                row,
                "ori6_spectral_maximum_abs_step_reward_decomposition_error",
                math.inf,
            )) <= 1.0e-10
            and f(
                row, "ori6_spectral_nonfinite_component_count", math.inf
            ) == 0.0
        )

    integrity = bool(rows) and all(row_integrity(row) for row in rows)
    return {
        "episodes": len(rows),
        "integrity_pass": integrity,
        "maximum_potential": max([
            f(row, "ori6_spectral_maximum_potential") for row in rows
        ] or [0.0]),
        "maximum_delta": max([
            f(row, "ori6_spectral_maximum_delta") for row in rows
        ] or [0.0]),
        "positive_peak_episodes": int(sum(
            f(row, "ori6_spectral_maximum_potential") > 1.0e-12
            for row in rows
        )),
        "delta_gt_1e_6_episodes": int(sum(
            f(row, "ori6_spectral_maximum_delta") > 1.0e-6
            for row in rows
        )),
        "mean_total_reward": mean(
            f(row, "ori6_spectral_total_reward") for row in rows
        ),
        "maximum_abs_reward_decomposition_error": max([
            abs(f(row, "ori6_spectral_reward_decomposition_error"))
            for row in rows
        ] or [0.0]),
        "nonfinite_component_count": int(sum(
            f(row, "ori6_spectral_nonfinite_component_count")
            for row in rows
        )),
    }


def heldout_summary(
    path: Path, name="evaluation_summary.csv", task_name: str | None = None,
) -> dict | None:
    file = path / name
    if not file.is_file():
        return None
    rows = load_csv(file)
    successes = [row for row in rows if f(row, "success") > 0.5]
    result = {
        "episodes": len(rows),
        "success_rate": rate(rows, "success"),
        "crossing_rate": rate(rows, "self_crossing"),
        "wrong_flat_rate": rate(rows, "wrong_flat"),
        "timeout_rate": rate(rows, "timeout"),
        "mean_flatness_ratio": mean(f(row, "flatness_ratio") for row in rows),
        "mean_spread_error": mean(f(row, "spread_error") for row in rows),
        "successful_mean_spread_error": mean(f(row, "spread_error") for row in successes),
        "mean_return": mean(f(row, "total_reward") for row in rows),
        "mean_steps": mean(f(row, "steps") for row in rows),
        "mean_maximum_spread_progress": mean(
            f(row, "ori6_maximum_spread_progress") for row in rows
        ),
        "same_state_joint_cost_le_0_5_rate": float(np.mean([
            (
                f(row, "flatness_ratio", math.inf)
                + f(row, "spread_error", math.inf) ** 2
            ) <= 0.5
            for row in rows
        ])) if rows else 0.0,
    }
    if rows and "ori6_rejected_crossing_count" in rows[0]:
        result["rejected_crossing"] = rejected_crossing_summary(
            rows, task_name
        )
    if rows and "ori6_spectral_candidate" in rows[0]:
        result["spectral_subspace"] = spectral_subspace_summary(rows)
    return result


def return_by_termination(history: list[dict]) -> dict:
    reasons = ("success", "self_crossing", "wrong_flat", "timeout", "correction_failed", "no_modes")
    result = {}
    for reason in reasons:
        rows = [row for row in history if row.get("termination_reason") == reason]
        result[reason] = {
            "total_reward": quantiles(f(row, "total_reward") for row in rows),
            "flatness_ratio": quantiles(f(row, "flatness_ratio") for row in rows),
            "spread_error": quantiles(f(row, "spread_error") for row in rows),
            "near_miss_q": quantiles(f(row, "near_miss_q") for row in rows),
            "spread_frontier": quantiles(
                f(row, "ori6_total_spread_frontier_reward") for row in rows
            ),
            "frontier_rollback": quantiles(
                f(row, "ori6_total_frontier_rollback_reward") for row in rows
            ),
            "joint_milestone": quantiles(
                f(row, "ori6_total_joint_milestone_reward") for row in rows
            ),
            "joint_rollback": quantiles(
                f(row, "ori6_total_joint_milestone_rollback_reward") for row in rows
            ),
        }
    crossing = [row for row in history if row.get("termination_reason") == "self_crossing"]
    result["crossing_rollback_audit"] = {
        "count": len(crossing),
        "maximum_abs_frontier_plus_rollback": max([
            abs(
                f(row, "ori6_total_spread_frontier_reward")
                + f(row, "ori6_total_frontier_rollback_reward")
            ) for row in crossing
        ] or [0.0]),
        "maximum_abs_joint_plus_rollback": max([
            abs(
                f(row, "ori6_total_joint_milestone_reward")
                + f(row, "ori6_total_joint_milestone_rollback_reward")
            ) for row in crossing
        ] or [0.0]),
    }
    return result


def summarize_job(job: dict, path: Path, validation: dict, status: str) -> dict:
    history = load_csv(path / "training_history.csv")
    tail = history[-min(512, len(history)):]
    config = load_json(path / "run_config.json")
    summary = {
        "job_id": job["job_id"],
        "task": job["task"]["name"],
        "profile": job["profile"]["name"],
        "seed": int(job["seed"]),
        "implementation": job["implementation"],
        "candidate_mode": job["candidate_mode"],
        "status": status,
        "artifact_path": str(path.resolve()),
        "endpoint_episode": len(history),
        "artifact_validation": validation,
        "first_hit": first_hit(history, int(job["profile"]["batch_size"])),
        "training": {
            "success_rate": rate(history, "success"),
            "crossing_rate": rate(history, "self_crossing"),
            "wrong_flat_rate": rate(history, "wrong_flat"),
            "timeout_rate": rate(history, "timeout"),
            "minimum_terminal_flatness_ratio": min(
                [f(row, "flatness_ratio", math.inf) for row in history] or [math.inf]
            ),
            "minimum_same_state_joint_cost": min([
                f(row, "flatness_ratio", math.inf) + f(row, "spread_error", math.inf) ** 2
                for row in history
            ] or [math.inf]),
            "maximum_spread_progress": max([
                f(row, "ori6_maximum_spread_progress") for row in history
            ] or [0.0]),
            "positive_frontier_episodes": sum(
                f(row, "ori6_total_spread_frontier_reward") > 1.0e-12 for row in history
            ),
            "positive_joint_milestone_episodes": sum(
                f(row, "ori6_total_joint_milestone_reward") > 1.0e-12 for row in history
            ),
            "maximum_near_miss_q": max([f(row, "near_miss_q") for row in history] or [0.0]),
            "total_mechanical_steps": int(sum(f(row, "steps") for row in history)),
            "maximum_abs_decomposition_error": max([
                abs(f(row, "ori6_reward_decomposition_error")) for row in history
            ] or [0.0]),
            "nonfinite_reward_components": int(sum(
                f(row, "ori6_nonfinite_reward_component_count") for row in history
            )),
        },
        "tail512": {
            "success_rate": rate(tail, "success"),
            "crossing_rate": rate(tail, "self_crossing"),
            "wrong_flat_rate": rate(tail, "wrong_flat"),
            "timeout_rate": rate(tail, "timeout"),
            "mean_flatness_ratio": mean(f(row, "flatness_ratio") for row in tail),
            "median_flatness_ratio": float(np.median([f(row, "flatness_ratio") for row in tail])) if tail else None,
            "mean_spread_error": mean(f(row, "spread_error") for row in tail),
            "median_spread_error": float(np.median([f(row, "spread_error") for row in tail])) if tail else None,
            "mean_return": mean(f(row, "total_reward") for row in tail),
        },
        "last_policy_heldout": heldout_summary(
            path, task_name=job["task"]["name"]
        ),
        "best_policy_heldout": heldout_summary(
            path, "best_policy_evaluation_summary.csv",
            task_name=job["task"]["name"],
        ),
        "return_by_termination": return_by_termination(history),
        "run_config_sha256": runner.file_sha256(path / "run_config.json"),
        "training_history_sha256": runner.file_sha256(path / "training_history.csv"),
        "model_sha256": job["model_sha256"],
        "source_sha256": job["source_sha256"],
        "reward_design": config.get("reward_design"),
        "candidate_identity": {
            "ori6_candidate": config.get("ori6_candidate"),
            "ori6_candidate_schema_version": config.get(
                "ori6_candidate_schema_version"
            ),
            "ori6_crossing_policy": config.get("ori6_crossing_policy"),
            "candidate_entrypoint": config.get("candidate_entrypoint"),
        },
    }
    if job["candidate_mode"] in (
        runner.P0_CROSS2_REJECTED_CONTINUATION_MODE,
        runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        summary["training"]["rejected_crossing"] = (
            rejected_crossing_summary(history, job["task"]["name"])
        )
        summary["tail512"]["rejected_crossing"] = (
            rejected_crossing_summary(tail, job["task"]["name"])
        )
    if job["candidate_mode"] == (
        runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        summary["training"]["spectral_subspace"] = (
            spectral_subspace_summary(history)
        )
        summary["tail512"]["spectral_subspace"] = (
            spectral_subspace_summary(tail)
        )
        summary["candidate_identity"].update({
            "ori6_parent_candidate": config.get("ori6_parent_candidate"),
            "spectral_formula_sha256": config.get(
                "spectral_subspace_formula_sha256"
            ),
            "spectral_weight": config.get("ori6_spectral_weight"),
        })
    return summary


def collect(protocol_name: str, mode: str, tasks: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    protocol = runner.PROTOCOLS[protocol_name]
    selected = list(protocol.tasks) if tasks is None else tasks
    jobs = runner.build_jobs(protocol, mode, selected)
    current_manifest_authority(protocol_name, mode, jobs)
    records = []
    errors = []
    for job in jobs:
        try:
            path, validation, status = resolve_job_artifact(job)
            records.append(summarize_job(job, path, validation, status))
        except Exception as error:
            errors.append({"job_id": job["job_id"], "error": repr(error)})
    return records, errors


def offline_rollback_counterfactual() -> dict:
    reference = {
        "five_d2": ORI5_FULL / "five_d2" / "B" / "seed2101",
        "five_d3": ORI5_FULL / "five_d3" / "B" / "seed2101",
        "torus2": ORI5_FULL / "torus2" / "A" / "seed2101",
        "water1": ORI5_FULL / "water1" / "B" / "seed2101",
        "watersqrt2": ORI5_FULL / "watersqrt2" / "B" / "seed2101",
    }
    result = {
        "schema_version": 1,
        "status": "offline_counterfactual_only_no_training",
        "formula": (
            "on ori5 self_crossing rows: counterfactual_return = total_reward "
            "- ori5_total_spread_milestone_reward - ori5_total_joint_milestone_reward"
        ),
        "tasks": {},
    }
    for task, path in reference.items():
        history_path = path / "training_history.csv"
        rows = load_csv(history_path)
        crossing = [row for row in rows if row.get("termination_reason") == "self_crossing"]
        timeout = [row for row in rows if row.get("termination_reason") == "timeout"]
        old = [f(row, "total_reward") for row in crossing]
        rolled = [
            f(row, "total_reward")
            - f(row, "ori5_total_spread_milestone_reward")
            - f(row, "ori5_total_joint_milestone_reward")
            for row in crossing
        ]
        result["tasks"][task] = {
            "training_history": str(history_path.resolve()),
            "training_history_sha256": runner.file_sha256(history_path),
            "crossing_count": len(crossing),
            "timeout_count": len(timeout),
            "crossing_original_return": quantiles(old),
            "crossing_rollback_only_counterfactual_return": quantiles(rolled),
            "timeout_original_return": quantiles(f(row, "total_reward") for row in timeout),
            "counterfactual_cross_minus_timeout_mean": (
                float(np.mean(rolled) - np.mean([f(row, "total_reward") for row in timeout]))
                if rolled and timeout else None
            ),
        }
    result["source_sha256"] = runner.source_hashes()
    result["random_baseline_authority_sha256"] = runner.random_baseline_authority()["sha256"]
    path = ANALYSIS_DIRECTORY / "offline_rollback_counterfactual.json"
    runner.atomic_json(path, result)
    return result


def best_fit_plane_projector(nodes: np.ndarray) -> tuple[np.ndarray, float]:
    centered = np.asarray(nodes, dtype=float) - np.mean(nodes, axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    projector = np.eye(3) - np.outer(normal, normal)
    trace = max(float(np.sum(eigenvalues)), np.finfo(float).eps)
    gap = float(max(eigenvalues[1] - eigenvalues[0], 0.0) / trace)
    return projector, gap


def kabsch_align_to_reference(nodes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    moving = np.asarray(nodes, dtype=float) - np.mean(nodes, axis=0, keepdims=True)
    target = np.asarray(reference, dtype=float) - np.mean(reference, axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(moving.T @ target)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return moving @ rotation


def plane_sequence_metrics(nodes: list, actions: list[dict], gap_threshold=1.0e-4) -> dict:
    if len(nodes) < 2:
        return {
            "states": len(nodes), "transitions": 0,
            "low_gap_state_fraction": None, "positive_frontier_reward": 0.0,
            "low_gap_positive_frontier_credit_fraction": None,
        }
    reference = np.asarray(nodes[0], dtype=float)
    raw_projectors = []
    aligned_projectors = []
    gaps = []
    for state in nodes:
        raw, gap = best_fit_plane_projector(np.asarray(state, dtype=float))
        aligned, _ = best_fit_plane_projector(
            kabsch_align_to_reference(np.asarray(state, dtype=float), reference)
        )
        raw_projectors.append(raw)
        aligned_projectors.append(aligned)
        gaps.append(gap)
    raw_delta = []
    aligned_delta = []
    well_defined = []
    for index in range(1, len(nodes)):
        raw_delta.append(float(
            np.linalg.norm(raw_projectors[index] - raw_projectors[index - 1], ord="fro")
            / math.sqrt(2.0)
        ))
        aligned_delta.append(float(
            np.linalg.norm(
                aligned_projectors[index] - aligned_projectors[index - 1], ord="fro"
            ) / math.sqrt(2.0)
        ))
        well_defined.append(
            gaps[index] >= gap_threshold and gaps[index - 1] >= gap_threshold
        )
    positive_credit = 0.0
    low_gap_positive_credit = 0.0
    for index, action in enumerate(actions[:len(nodes) - 1]):
        credit = max(f(action, "ori6_spread_frontier_reward"), 0.0)
        positive_credit += credit
        if gaps[index + 1] < gap_threshold:
            low_gap_positive_credit += credit
    selected_aligned = [
        value for value, valid in zip(aligned_delta, well_defined) if valid
    ]
    return {
        "states": len(nodes),
        "transitions": len(nodes) - 1,
        "plane_gap_definition": "(lambda_2-lambda_3)/trace for descending PCA eigenvalues",
        "gap_threshold": gap_threshold,
        "minimum_plane_spectral_gap": float(min(gaps)),
        "median_plane_spectral_gap": float(np.median(gaps)),
        "low_gap_state_fraction": float(np.mean(np.asarray(gaps) < gap_threshold)),
        "raw_projector_delta_mean": float(np.mean(raw_delta)),
        "raw_projector_delta_max": float(max(raw_delta)),
        "kabsch_aligned_projector_delta_mean_well_defined": (
            float(np.mean(selected_aligned)) if selected_aligned else None
        ),
        "kabsch_aligned_projector_delta_max_well_defined": (
            float(max(selected_aligned)) if selected_aligned else None
        ),
        "kabsch_aligned_switch_count_delta_ge_0_25": int(sum(
            value >= 0.25 for value in selected_aligned
        )),
        "positive_frontier_reward": float(positive_credit),
        "low_gap_positive_frontier_reward": float(low_gap_positive_credit),
        "low_gap_positive_frontier_credit_fraction": (
            float(low_gap_positive_credit / positive_credit)
            if positive_credit > 1.0e-15 else None
        ),
        "interpretation_limit": (
            "This is a diagnostic proxy, not a terminal objective. Raw plane "
            "motion is confounded by rigid rotation; the Kabsch-aligned value "
            "reduces but cannot eliminate deformation/alignment ambiguity."
        ),
    }


def build_checkpoint_agent(args: SimpleNamespace, checkpoint: dict):
    import ori_codex2 as mechanics

    model = mechanics.VariableModeActorCritic(
        coordinate_size=int(checkpoint["config"]["coordinate_size"]),
        hidden_size=int(args.hidden_size),
        variance_bias=float(args.variance_bias),
        width_mean_limit=float(args.width_mean_limit),
    )
    agent = mechanics.ActorCriticAgent(
        model=model,
        max_step_width=float(args.max_step_width),
        gamma=float(args.gamma),
        actor_learning_rate=float(args.actor_learning_rate),
        critic_learning_rate=float(args.critic_learning_rate),
        entropy_coefficient=float(args.entropy_coefficient),
        value_coefficient=float(args.value_coefficient),
        actor_gradient_clip=float(args.actor_gradient_clip),
        critic_gradient_clip=float(args.critic_gradient_clip),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return agent


def derive_plane_proxy(record: dict, evaluation_episodes=16) -> dict:
    if record["task"] != "torus2" or record["implementation"] != "ori6":
        raise ValueError("Plane proxy is defined here only for the ori6 torus task")
    import torch
    import ori_codex6_3d as adapter

    run_directory = Path(record["artifact_path"])
    canonical_names = (
        "run_config.json", "training_history.csv", "update_history.csv",
        "deterministic_evaluation_history.csv", "stochastic_evaluation_history.csv",
        "checkpoint.pt", "best_policy.pt", "best_policy_metrics.json",
    )
    before = canonical_files_digest(run_directory, canonical_names)
    config = load_json(run_directory / "run_config.json")
    checkpoint = torch.load(
        run_directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    args = SimpleNamespace(**config)
    node, member, fix, _ = adapter.load_problem_data(args.data_dir)
    agent = build_checkpoint_agent(args, checkpoint)
    seed = int(config["final_evaluation_seed"])
    rows = []
    torch.set_num_threads(1)
    with adapter.patched_three_dimensional_runtime(args, member):
        environment = adapter.make_environment(args, node, member, fix)
        for index in range(evaluation_episodes):
            case_seed = seed + index
            torch.manual_seed(case_seed)
            np.random.seed(case_seed)
            with torch.no_grad():
                summary, nodes, actions = adapter.run_episode(
                    environment, agent, deterministic=False, training=False
                )
            rows.append({
                "evaluation_index": index,
                "evaluation_seed": case_seed,
                "success": int(summary["success"]),
                "self_crossing": int(summary["self_crossing"]),
                "termination_reason": summary["termination_reason"],
                "flatness_ratio": float(summary["flatness_ratio"]),
                "spread_error": float(summary["spread_error"]),
                "maximum_spread_progress": float(summary["ori6_maximum_spread_progress"]),
                **plane_sequence_metrics(nodes, actions),
            })
    after = canonical_files_digest(run_directory, canonical_names)
    if after != before:
        raise RuntimeError("Canonical training artifact changed during plane evaluation")
    credit_rows = [
        row for row in rows
        if row["low_gap_positive_frontier_credit_fraction"] is not None
    ]
    total_positive = sum(row["positive_frontier_reward"] for row in rows)
    total_low_gap = sum(row["low_gap_positive_frontier_reward"] for row in rows)
    result = {
        "schema_version": 1,
        "task": record["task"], "seed": record["seed"],
        "candidate_mode": record["candidate_mode"],
        "checkpoint_episode": int(checkpoint["episode"]),
        "evaluation_protocol": {
            "episodes": evaluation_episodes,
            "seed_start": seed,
            "stochastic_policy": True,
            "collision_adapter": "ori_codex6_3d",
            "canonical_artifact_mutated": False,
        },
        "rollouts": rows,
        "aggregate": {
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "crossing_rate": float(np.mean([row["self_crossing"] for row in rows])),
            "maximum_spread_progress": float(max(row["maximum_spread_progress"] for row in rows)),
            "positive_frontier_reward": float(total_positive),
            "low_gap_positive_frontier_reward": float(total_low_gap),
            "low_gap_positive_frontier_credit_fraction": (
                float(total_low_gap / total_positive) if total_positive > 1.0e-15 else None
            ),
            "evaluable_frontier_rollouts": len(credit_rows),
            "maximum_kabsch_aligned_projector_delta_well_defined": max([
                row["kabsch_aligned_projector_delta_max_well_defined"]
                for row in rows
                if row["kabsch_aligned_projector_delta_max_well_defined"] is not None
            ] or [0.0]),
        },
        "canonical_sha256_before_after": before,
        "source_sha256": runner.source_hashes(),
    }
    output = (
        ANALYSIS_DIRECTORY / "plane_proxy" / runner.MODE_SLUGS[record["candidate_mode"]]
        / f"torus2_seed{record['seed']}_ep{record['endpoint_episode']}.json"
    )
    runner.atomic_json(output, result)
    result["output_path"] = str(output.resolve())
    return result


def _history_for_record(record: dict) -> list[dict]:
    return load_csv(Path(record["artifact_path"]) / "training_history.csv")


def _torus_frontier_before_flat(record: dict) -> dict:
    rows = _history_for_record(record)
    first_below = next(
        (index for index, row in enumerate(rows) if f(row, "flatness_ratio") < 1.0),
        None,
    )
    endpoint = len(rows) if first_below is None else first_below + 1
    prefix = rows[:endpoint]
    return {
        "first_terminal_F_below_1_episode": (
            int(f(rows[first_below], "episode")) if first_below is not None else None
        ),
        "positive_frontier_before_or_at_first_F_below_1": any(
            f(row, "ori6_total_spread_frontier_reward") > 1.0e-12
            for row in prefix
        ),
        "maximum_spread_progress_before_or_at_first_F_below_1": max([
            f(row, "ori6_maximum_spread_progress") for row in prefix
        ] or [0.0]),
    }


def _water_seed_return_order(record: dict) -> dict:
    rows = _history_for_record(record)
    crossing = [f(row, "total_reward") for row in rows if row.get("termination_reason") == "self_crossing"]
    timeout = [f(row, "total_reward") for row in rows if row.get("termination_reason") == "timeout"]
    result = bootstrap_mean_difference(crossing, timeout, seed=6202601 + record["seed"])
    result.update({"crossing_count": len(crossing), "timeout_count": len(timeout)})
    return result


def rescue_report(mode: str, derive_planes: bool) -> dict:
    records, errors = collect("rescue", mode)
    if errors or len(records) != 6:
        raise RuntimeError(f"Rescue analysis requires six valid endpoints: {errors}")
    torus = sorted(
        [row for row in records if row["task"] == "torus2"], key=lambda row: row["seed"]
    )
    water = sorted(
        [row for row in records if row["task"] == "water1"], key=lambda row: row["seed"]
    )
    plane = []
    if derive_planes:
        plane = [derive_plane_proxy(row) for row in torus]
    plane_by_seed = {row["seed"]: row for row in plane}
    torus_seed_rows = []
    for record in torus:
        frontier_order = _torus_frontier_before_flat(record)
        proxy = plane_by_seed.get(record["seed"])
        torus_seed_rows.append({
            "seed": record["seed"],
            "endpoint_episode": record["endpoint_episode"],
            "complete_2048": record["status"] == "complete" and record["endpoint_episode"] == 2048,
            "maximum_spread_progress": record["training"]["maximum_spread_progress"],
            "positive_frontier_episodes": record["training"]["positive_frontier_episodes"],
            "minimum_terminal_flatness_ratio": record["training"]["minimum_terminal_flatness_ratio"],
            "minimum_same_state_joint_cost": record["training"]["minimum_same_state_joint_cost"],
            "strict_hit": not record["first_hit"]["censored"],
            "tail512": record["tail512"],
            "joint_milestone_episodes": record["training"]["positive_joint_milestone_episodes"],
            **frontier_order,
            "plane_proxy": proxy["aggregate"] if proxy else None,
            "last_policy_heldout": record["last_policy_heldout"],
        })
    frontier_each_seed = all(
        row["maximum_spread_progress"] > 0.0 and row["positive_frontier_episodes"] > 0
        for row in torus_seed_rows
    )
    substantial_frontier_seeds = sum(
        row["maximum_spread_progress"] >= 0.25 for row in torus_seed_rows
    )
    joint_progress_seeds = sum(
        row["minimum_same_state_joint_cost"] <= 0.5 for row in torus_seed_rows
    )
    any_strict_hit = any(row["strict_hit"] for row in torus_seed_rows)
    frontier_precedes_flat = all(
        row["positive_frontier_before_or_at_first_F_below_1"]
        for row in torus_seed_rows
        if row["first_terminal_F_below_1_episode"] is not None
    ) and any(
        row["first_terminal_F_below_1_episode"] is not None for row in torus_seed_rows
    )
    evaluable_plane = [
        row["plane_proxy"] for row in torus_seed_rows
        if row["plane_proxy"] is not None
        and row["plane_proxy"]["low_gap_positive_frontier_credit_fraction"] is not None
    ]
    plane_proxy_pass = (
        len(evaluable_plane) >= 2
        and all(
            row["low_gap_positive_frontier_credit_fraction"] <= 0.25
            for row in evaluable_plane
        )
    ) if derive_planes else None
    torus_safe_timeout_seeds = sum(
        row["tail512"]["timeout_rate"] >= 0.8
        and row["tail512"]["median_flatness_ratio"] is not None
        and row["tail512"]["median_flatness_ratio"] >= 1.0
        and row["tail512"]["median_spread_error"] is not None
        and row["tail512"]["median_spread_error"] <= 0.05
        and row["joint_milestone_episodes"] == 0
        for row in torus_seed_rows
    )
    torus_safe_timeout_attractor = torus_safe_timeout_seeds >= 2
    heldout_joint_progress_seeds = sum(
        row["last_policy_heldout"] is not None
        and (
            row["last_policy_heldout"]["same_state_joint_cost_le_0_5_rate"] >= 0.10
            or row["last_policy_heldout"]["success_rate"] > 0.0
        )
        for row in torus_seed_rows
    )
    torus_gate_pass = all(row["complete_2048"] for row in torus_seed_rows) and (
        frontier_each_seed
        and substantial_frontier_seeds >= 2
        and ((joint_progress_seeds >= 2 and frontier_precedes_flat) or any_strict_hit)
        and plane_proxy_pass is True
    )

    water_seed_rows = []
    pooled_crossing = []
    pooled_timeout = []
    for record in water:
        order = _water_seed_return_order(record)
        history = _history_for_record(record)
        pooled_crossing.extend(
            f(row, "total_reward") for row in history
            if row.get("termination_reason") == "self_crossing"
        )
        pooled_timeout.extend(
            f(row, "total_reward") for row in history
            if row.get("termination_reason") == "timeout"
        )
        rollback = record["return_by_termination"]["crossing_rollback_audit"]
        tail = record["tail512"]
        water_seed_rows.append({
            "seed": record["seed"],
            "endpoint_episode": record["endpoint_episode"],
            "complete_2048": record["status"] == "complete" and record["endpoint_episode"] == 2048,
            "crossing_vs_timeout_return": order,
            "crossing_rollback_audit": rollback,
            "tail512": tail,
            "last_policy_heldout": record["last_policy_heldout"],
            "strict_hit": not record["first_hit"]["censored"],
            "joint_milestone_episodes": record["training"]["positive_joint_milestone_episodes"],
        })
    pooled_order = bootstrap_mean_difference(
        pooled_crossing, pooled_timeout, seed=6202699
    )
    pooled_order.update({
        "crossing_count": len(pooled_crossing), "timeout_count": len(pooled_timeout)
    })
    rollback_exact = all(
        row["crossing_rollback_audit"]["maximum_abs_frontier_plus_rollback"] <= 1.0e-10
        and row["crossing_rollback_audit"]["maximum_abs_joint_plus_rollback"] <= 1.0e-10
        for row in water_seed_rows
    )
    seed_order_passes = sum(
        row["crossing_vs_timeout_return"]["estimable"]
        and row["crossing_vs_timeout_return"]["difference"] < 0.0
        for row in water_seed_rows
    )
    low_cross_or_success = sum(
        row["tail512"]["crossing_rate"] <= 0.75
        or row["tail512"]["success_rate"] >= 0.25
        for row in water_seed_rows
    )
    heldout_cross_passes = sum(
        row["last_policy_heldout"] is not None
        and row["last_policy_heldout"]["crossing_rate"] < 0.8
        for row in water_seed_rows
    )
    safe_timeout_seeds = sum(
        row["tail512"]["timeout_rate"] >= 0.8
        and row["tail512"]["median_flatness_ratio"] is not None
        and row["tail512"]["median_flatness_ratio"] >= 1.0
        and row["tail512"]["median_spread_error"] is not None
        and row["tail512"]["median_spread_error"] <= 0.05
        and row["joint_milestone_episodes"] == 0
        for row in water_seed_rows
    )
    safe_timeout_attractor = safe_timeout_seeds >= 2
    if not pooled_crossing:
        return_order_pass = True
    else:
        return_order_pass = (
            pooled_order["estimable"]
            and pooled_order["difference"] < 0.0
            and seed_order_passes >= 2
        )
    water_gate_pass = all(row["complete_2048"] for row in water_seed_rows) and (
        rollback_exact and return_order_pass and low_cross_or_success >= 2
        and heldout_cross_passes >= 2 and not safe_timeout_attractor
    )
    rescue_gate_pass = bool(torus_gate_pass and water_gate_pass)
    if rescue_gate_pass:
        decision = "proceed_to_successful_task_regression"
    elif (
        mode == runner.P0_MODE
        and (safe_timeout_attractor or torus_safe_timeout_attractor)
        and substantial_frontier_seeds >= 2
    ):
        decision = "trigger_P1"
    else:
        decision = "reject_candidate"
    result = {
        "schema_version": 1,
        "stage": "rescue",
        "candidate_mode": mode,
        "candidate_mode_slug": runner.MODE_SLUGS[mode],
        "decision": decision,
        "rescue_gate_pass": rescue_gate_pass,
        "torus_gate": {
            "pass": bool(torus_gate_pass),
            "frontier_each_seed": frontier_each_seed,
            "substantial_frontier_seed_count": substantial_frontier_seeds,
            "joint_progress_seed_count": joint_progress_seeds,
            "heldout_joint_progress_seed_count": heldout_joint_progress_seeds,
            "any_strict_hit": any_strict_hit,
            "frontier_precedes_first_F_below_1": frontier_precedes_flat,
            "plane_proxy_requested": derive_planes,
            "plane_proxy_evaluable_seed_count": len(evaluable_plane),
            "plane_proxy_pass": plane_proxy_pass,
            "safe_timeout_seed_count": torus_safe_timeout_seeds,
            "safe_timeout_attractor": torus_safe_timeout_attractor,
            "per_seed": torus_seed_rows,
        },
        "water1_gate": {
            "pass": bool(water_gate_pass),
            "rollback_exact": rollback_exact,
            "pooled_crossing_vs_timeout_return": pooled_order,
            "seed_return_order_pass_count": seed_order_passes,
            "low_cross_or_success_seed_count": low_cross_or_success,
            "heldout_cross_below_0_8_seed_count": heldout_cross_passes,
            "safe_timeout_seed_count": safe_timeout_seeds,
            "safe_timeout_attractor": safe_timeout_attractor,
            "per_seed": water_seed_rows,
        },
        "records": records,
        "manifest_authority": current_manifest_authority(
            "rescue", mode,
            runner.build_jobs(runner.PROTOCOLS["rescue"], mode, list(runner.PROTOCOLS["rescue"].tasks)),
        ),
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task) for name, task in runner.TASKS.items()
        },
        "random_baseline_authority_sha256": runner.random_baseline_authority()["sha256"],
    }
    output = ANALYSIS_DIRECTORY / f"rescue_{runner.MODE_SLUGS[mode]}_report.json"
    runner.atomic_json(output, result)
    write_csv(
        ANALYSIS_DIRECTORY / f"rescue_{runner.MODE_SLUGS[mode]}_summary.csv",
        [flatten_record(record) for record in records],
    )
    return result


def paired_cluster_bootstrap(differences: list[float], seed: int) -> dict:
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return {"difference": None, "ci95": [None, None], "seed_count": 0}
    generator = np.random.default_rng(seed)
    samples = np.asarray([
        np.mean(generator.choice(values, size=len(values), replace=True))
        for _ in range(20000)
    ])
    return {
        "difference": float(np.mean(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "one_sided_95_lower": float(np.quantile(samples, 0.05)),
        "one_sided_95_upper": float(np.quantile(samples, 0.95)),
        "seed_count": len(values), "bootstrap_samples": len(samples),
    }


def crossing_continuation_gate_authorities(
    task_name: str, records: list[dict],
) -> dict:
    rows_by_seed = {
        int(record["seed"]): _history_for_record(record)
        for record in records
    }
    rows = []
    for checkpoint in (512, 1024, 2048):
        path = (
            runner.EXPERIMENT_ROOT / "early_gates"
            / "crossing_continuation_diagnostic" / "P0X2RC"
            / task_name / f"checkpoint_{checkpoint}.json"
        ).resolve()
        row = {
            "checkpoint_episode": checkpoint,
            "path": str(path),
            "present": path.is_file(),
            "sha256": runner.file_sha256(path) if path.is_file() else None,
            "valid_and_passed": False,
        }
        if path.is_file():
            try:
                payload = load_json(path)
                recomputed = runner.early_gate_decision(
                    "crossing_continuation_diagnostic", task_name,
                    rows_by_seed, checkpoint,
                    runner.P0_CROSS2_REJECTED_CONTINUATION_MODE,
                )
                recorded_semantics = dict(payload)
                recomputed_semantics = dict(recomputed)
                recorded_semantics.pop("evaluated_utc", None)
                recomputed_semantics.pop("evaluated_utc", None)
                row["semantic_recomputation_match"] = (
                    recorded_semantics == recomputed_semantics
                )
                formal = payload.get(
                    "formal_crossing_continuation_gate", {}
                )
                row["valid_and_passed"] = bool(
                    payload.get("protocol")
                    == "crossing_continuation_diagnostic"
                    and payload.get("task") == task_name
                    and payload.get("checkpoint_episode") == checkpoint
                    and payload.get("candidate_mode")
                    == runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
                    and payload.get("cohort_shape_valid") is True
                    and payload.get("preregistered_gate_specification")
                    == runner.crossing_continuation_gate_specification()
                    and set(payload.get("per_seed", {}))
                    == {"3201", "3202", "3203"}
                    and formal.get("pass") is True
                    and payload.get("stop") is False
                    and row["semantic_recomputation_match"] is True
                )
                row["reason"] = payload.get("reason")
                row["formal_gate"] = formal
            except (OSError, ValueError, json.JSONDecodeError) as error:
                row["error"] = repr(error)
        rows.append(row)
    return {
        "all_three_present_and_passed": all(
            row["valid_and_passed"] for row in rows
        ),
        "checkpoints": rows,
    }


def crossing_continuation_report(mode: str) -> dict:
    if mode != runner.P0_CROSS2_REJECTED_CONTINUATION_MODE:
        raise ValueError(
            "crossing_continuation_diagnostic is reserved for P0X2RC"
        )
    protocol_name = "crossing_continuation_diagnostic"
    records, errors = collect(protocol_name, mode)
    if errors or len(records) != 6:
        raise RuntimeError(
            "Crossing-continuation analysis requires six valid endpoints: "
            f"{errors}"
        )
    task_results = {}
    for task_name in runner.PROTOCOLS[protocol_name].tasks:
        selected = sorted(
            [row for row in records if row["task"] == task_name],
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in selected] != [3201, 3202, 3203]:
            raise RuntimeError(f"Unexpected P0X2RC seed cohort for {task_name}")
        gate_authorities = crossing_continuation_gate_authorities(
            task_name, selected
        )
        per_seed = []
        for record in selected:
            candidate_curve = (
                Path(record["artifact_path"])
                / "ori6_rejected_crossing_learning_curve.png"
            )
            training_rc = record["training"].get("rejected_crossing", {})
            heldout = record.get("last_policy_heldout") or {}
            heldout_rc = heldout.get("rejected_crossing", {})
            complete = bool(
                record["status"] == "complete"
                and record["endpoint_episode"] == 2048
                and heldout_rc
            )
            training_strict = not record["first_hit"]["censored"]
            training_constructive = bool(
                training_rc.get("constructive_proxy_episodes", 0) >= 1
            )
            reward_transition_integrity = bool(
                record["artifact_validation"].get("valid") is True
                and record["training"]["maximum_abs_decomposition_error"]
                <= 1.0e-10
                and record["training"]["nonfinite_reward_components"] == 0
                and training_rc.get("transition_integrity_pass") is True
                and (
                    not complete
                    or heldout_rc.get("transition_integrity_pass") is True
                )
            )
            heldout_constructive_or_strict = bool(
                complete
                and (
                    heldout_rc.get("constructive_proxy_episodes", 0) >= 1
                    or heldout.get("success_rate", 0.0) > 0.0
                )
            )
            per_seed.append({
                "seed": record["seed"],
                "status": record["status"],
                "endpoint_episode": record["endpoint_episode"],
                "complete_2048_with_heldout": complete,
                "training_strict_hit": training_strict,
                "training_constructive_proxy": training_constructive,
                "reward_and_transition_integrity_pass": (
                    reward_transition_integrity
                ),
                "training_rejected_crossing": training_rc,
                "tail512_rejected_crossing": record["tail512"].get(
                    "rejected_crossing"
                ),
                "heldout_success_rate": heldout.get("success_rate"),
                "heldout_constructive_or_strict_support": (
                    heldout_constructive_or_strict
                ),
                "heldout_rejected_crossing": heldout_rc or None,
                "artifact_path": record["artifact_path"],
                "candidate_learning_curve": (
                    str(candidate_curve.resolve())
                    if candidate_curve.is_file() else None
                ),
                "candidate_learning_curve_sha256": (
                    runner.file_sha256(candidate_curve)
                    if candidate_curve.is_file() else None
                ),
            })
        complete_count = sum(
            row["complete_2048_with_heldout"] for row in per_seed
        )
        strict_count = sum(row["training_strict_hit"] for row in per_seed)
        constructive_count = sum(
            row["training_constructive_proxy"] for row in per_seed
        )
        heldout_support_count = sum(
            row["heldout_constructive_or_strict_support"] for row in per_seed
        )
        integrity = all(
            row["reward_and_transition_integrity_pass"] for row in per_seed
        )
        gate_pass = bool(
            complete_count == 3
            and strict_count >= 1
            and constructive_count >= 2
            and heldout_support_count >= 2
            and integrity
            and gate_authorities["all_three_present_and_passed"]
        )
        task_results[task_name] = {
            "pass": gate_pass,
            "complete_2048_with_heldout_seed_count": complete_count,
            "complete_2048_with_heldout_required_seed_count": 3,
            "training_strict_hit_seed_count": strict_count,
            "training_strict_hit_required_seed_count": 1,
            "training_constructive_proxy_seed_count": constructive_count,
            "training_constructive_proxy_required_seed_count": 2,
            "heldout_constructive_or_strict_seed_count": (
                heldout_support_count
            ),
            "heldout_constructive_or_strict_required_seed_count": 2,
            "reward_and_transition_integrity_pass": integrity,
            "checkpoint_gate_authorities": gate_authorities,
            "constructive_proxy_definition": (
                runner.crossing_continuation_gate_specification()[
                    "constructive_proxy_by_task"
                ][task_name]
            ),
            "per_seed": per_seed,
        }
    screen_pass = all(row["pass"] for row in task_results.values())
    decision = (
        "bounded_screen_pass_requires_fresh_random_and_regression"
        if screen_pass else "reject_candidate"
    )
    jobs = runner.build_jobs(
        runner.PROTOCOLS[protocol_name], mode,
        list(runner.PROTOCOLS[protocol_name].tasks),
    )
    result = {
        "schema_version": 1,
        "stage": protocol_name,
        "candidate_mode": mode,
        "candidate_mode_slug": runner.MODE_SLUGS[mode],
        "decision": decision,
        "crossing_continuation_screen_pass": screen_pass,
        "screen_pass_authorizes_full_training": False,
        "task_gates": task_results,
        "records": records,
        "manifest_authority": current_manifest_authority(
            protocol_name, mode, jobs
        ),
        "source_sha256": runner.source_hashes(),
        "candidate_additive_source_sha256": (
            runner.rejected_crossing_additive_source_hashes()
        ),
        "candidate_orchestration_source_sha256": (
            runner.rejected_crossing_orchestration_source_hashes()
        ),
        "model_sha256_by_task": {
            name: runner.model_hashes(runner.TASKS[name])
            for name in runner.PROTOCOLS[protocol_name].tasks
        },
        "candidate_reward_sha256": runner.candidate_reward_sha256(mode),
        "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ),
        "candidate_transition_semantics_sha256": (
            runner.candidate_transition_semantics_sha256(mode)
        ),
        "parent_transition_semantics_sha256": (
            runner.candidate_transition_semantics_sha256(
                runner.P0_CROSS2_MODE
            )
        ),
        "preceding_W2_reject_authority": str(
            runner.P0_CROSS2_WIDTH2_REJECT_REPORT.resolve()
        ),
        "preceding_W2_reject_authority_sha256": runner.file_sha256(
            runner.P0_CROSS2_WIDTH2_REJECT_REPORT
        ),
        "existing_random_baseline": (
            runner.rejected_crossing_random_baseline_disposition()
        ),
        "adoption_blockers_even_if_screen_passes": [
            "fresh_random_baseline_all_five_tasks",
            "fresh_successful_task_regression",
        ],
        "learning_curves": plot_learning_curves(
            records, f"{protocol_name}_{runner.MODE_SLUGS[mode]}"
        ),
    }
    output = (
        ANALYSIS_DIRECTORY
        / "crossing_continuation_diagnostic_P0X2RC_report.json"
    )
    runner.atomic_json(output, result)
    write_csv(
        ANALYSIS_DIRECTORY
        / "crossing_continuation_diagnostic_P0X2RC_summary.csv",
        [flatten_record(record) for record in records],
    )
    return result


def spectral_subspace_gate_authorities(
    task_name: str, records: list[dict],
) -> dict:
    rows_by_seed = {
        int(record["seed"]): _history_for_record(record)
        for record in records
    }
    rows = []
    for checkpoint in (512, 1024, 2048):
        path = (
            runner.EXPERIMENT_ROOT / "early_gates"
            / "spectral_subspace_diagnostic" / "P0X2RCS05"
            / task_name / f"checkpoint_{checkpoint}.json"
        ).resolve()
        row = {
            "checkpoint_episode": checkpoint,
            "path": str(path),
            "present": path.is_file(),
            "sha256": runner.file_sha256(path) if path.is_file() else None,
            "valid_and_passed": False,
        }
        if path.is_file():
            try:
                payload = load_json(path)
                recomputed = runner.early_gate_decision(
                    "spectral_subspace_diagnostic", task_name,
                    rows_by_seed, checkpoint,
                    runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
                )
                recorded_semantics = dict(payload)
                recomputed_semantics = dict(recomputed)
                recorded_semantics.pop("evaluated_utc", None)
                recomputed_semantics.pop("evaluated_utc", None)
                row["semantic_recomputation_match"] = (
                    recorded_semantics == recomputed_semantics
                )
                formal = payload.get("formal_spectral_subspace_gate", {})
                row["valid_and_passed"] = bool(
                    payload.get("protocol")
                    == "spectral_subspace_diagnostic"
                    and payload.get("task") == task_name
                    and payload.get("checkpoint_episode") == checkpoint
                    and payload.get("candidate_mode")
                    == runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                    and payload.get("cohort_shape_valid") is True
                    and payload.get("preregistered_gate_specification")
                    == runner.spectral_subspace_gate_specification()
                    and set(payload.get("per_seed", {}))
                    == {"3201", "3202", "3203"}
                    and formal.get("pass") is True
                    and payload.get("stop") is False
                    and row["semantic_recomputation_match"] is True
                )
                row["reason"] = payload.get("reason")
                row["formal_gate"] = formal
            except (OSError, ValueError, json.JSONDecodeError) as error:
                row["error"] = repr(error)
        rows.append(row)
    return {
        "all_three_present_and_passed": all(
            row["valid_and_passed"] for row in rows
        ),
        "checkpoints": rows,
    }


def spectral_nonregression_gate_authority(
    task_name: str, candidate: list[dict], reference: list[dict],
) -> dict:
    candidate_rows = {
        int(record["seed"]): _history_for_record(record)
        for record in candidate
    }
    reference_rows = {
        int(record["seed"]): _history_for_record(record)
        for record in reference
    }
    path = (
        runner.EXPERIMENT_ROOT / "early_gates"
        / "spectral_nonregression" / "P0X2RCS05"
        / task_name / "checkpoint_512.json"
    ).resolve()
    result = {
        "checkpoint_episode": 512,
        "path": str(path),
        "present": path.is_file(),
        "sha256": runner.file_sha256(path) if path.is_file() else None,
        "valid_and_passed": False,
    }
    if not path.is_file():
        return result
    try:
        payload = load_json(path)
        recomputed = runner.early_gate_decision(
            "spectral_nonregression", task_name, candidate_rows, 512,
            runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
            reference_rows_by_seed=reference_rows,
        )
        recorded_semantics = dict(payload)
        recomputed_semantics = dict(recomputed)
        recorded_semantics.pop("evaluated_utc", None)
        recomputed_semantics.pop("evaluated_utc", None)
        semantic_match = recorded_semantics == recomputed_semantics
        formal = payload.get("formal_spectral_nonregression_gate", {})
        result.update({
            "semantic_recomputation_match": semantic_match,
            "reason": payload.get("reason"),
            "formal_gate": formal,
            "valid_and_passed": bool(
                payload.get("protocol") == "spectral_nonregression"
                and payload.get("task") == task_name
                and payload.get("checkpoint_episode") == 512
                and payload.get("candidate_mode")
                == runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                and payload.get("preregistered_gate_specification")
                == runner.spectral_nonregression_gate_specification()
                and set(payload.get("per_seed", {}))
                == {"3301", "3302", "3303"}
                and set(payload.get("reference_per_seed", {}))
                == {"3301", "3302", "3303"}
                and formal.get("pass") is True
                and payload.get("stop") is False
                and semantic_match
            ),
        })
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        result["error"] = repr(error)
    return result


def spectral_subspace_report(mode: str) -> dict:
    if mode != runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
        raise ValueError(
            "spectral_subspace_diagnostic is reserved for P0X2RCS05"
        )
    protocol_name = "spectral_subspace_diagnostic"
    records, errors = collect(protocol_name, mode)
    if errors or len(records) != 6:
        raise RuntimeError(
            "Spectral-subspace analysis requires six valid endpoints: "
            f"{errors}"
        )
    task_results = {}
    parent_comparisons = {}
    for task_name in runner.PROTOCOLS[protocol_name].tasks:
        selected = sorted(
            [row for row in records if row["task"] == task_name],
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in selected] != [3201, 3202, 3203]:
            raise RuntimeError(
                f"Unexpected P0X2RCS05 seed cohort for {task_name}"
            )
        gate_authorities = spectral_subspace_gate_authorities(
            task_name, selected
        )
        histories = {
            int(record["seed"]): _history_for_record(record)
            for record in selected
        }
        parent_comparisons[task_name] = (
            runner.spectral_parent_common_prefix_comparison(
                histories, task_name,
                min(record["endpoint_episode"] for record in selected),
            )
        )
        per_seed = []
        for record in selected:
            artifact = Path(record["artifact_path"])
            rejected_curve = (
                artifact / "ori6_rejected_crossing_learning_curve.png"
            )
            spectral_curve = (
                artifact / "ori6_spectral_subspace_learning_curve.png"
            )
            training_rc = record["training"].get("rejected_crossing", {})
            training_spectral = record["training"].get(
                "spectral_subspace", {}
            )
            heldout = record.get("last_policy_heldout") or {}
            heldout_rc = heldout.get("rejected_crossing", {})
            heldout_spectral = heldout.get("spectral_subspace", {})
            complete = bool(
                record["status"] == "complete"
                and record["endpoint_episode"] == 2048
                and heldout_rc and heldout_spectral
            )
            training_strict = not record["first_hit"]["censored"]
            training_constructive = bool(
                training_rc.get("constructive_proxy_episodes", 0) >= 1
            )
            integrity = bool(
                record["artifact_validation"].get("valid") is True
                and record["training"]["maximum_abs_decomposition_error"]
                <= 1.0e-10
                and record["training"]["nonfinite_reward_components"] == 0
                and training_rc.get("transition_integrity_pass") is True
                and training_spectral.get("integrity_pass") is True
                and (
                    not complete
                    or (
                        heldout_rc.get("transition_integrity_pass") is True
                        and heldout_spectral.get("integrity_pass") is True
                    )
                )
            )
            heldout_support = bool(
                complete and (
                    heldout_rc.get("constructive_proxy_episodes", 0) >= 1
                    or heldout.get("success_rate", 0.0) > 0.0
                )
            )
            per_seed.append({
                "seed": record["seed"],
                "status": record["status"],
                "endpoint_episode": record["endpoint_episode"],
                "complete_2048_with_heldout": complete,
                "training_strict_hit": training_strict,
                "training_constructive_proxy": training_constructive,
                "reward_spectral_and_transition_integrity_pass": integrity,
                "training_rejected_crossing": training_rc,
                "training_spectral_subspace": training_spectral,
                "heldout_success_rate": heldout.get("success_rate"),
                "heldout_constructive_or_strict_support": heldout_support,
                "heldout_rejected_crossing": heldout_rc or None,
                "heldout_spectral_subspace": heldout_spectral or None,
                "artifact_path": record["artifact_path"],
                "rejected_crossing_learning_curve": (
                    str(rejected_curve.resolve())
                    if rejected_curve.is_file() else None
                ),
                "rejected_crossing_learning_curve_sha256": (
                    runner.file_sha256(rejected_curve)
                    if rejected_curve.is_file() else None
                ),
                "spectral_learning_curve": (
                    str(spectral_curve.resolve())
                    if spectral_curve.is_file() else None
                ),
                "spectral_learning_curve_sha256": (
                    runner.file_sha256(spectral_curve)
                    if spectral_curve.is_file() else None
                ),
            })
        complete_count = sum(
            row["complete_2048_with_heldout"] for row in per_seed
        )
        strict_count = sum(row["training_strict_hit"] for row in per_seed)
        constructive_count = sum(
            row["training_constructive_proxy"] for row in per_seed
        )
        heldout_support_count = sum(
            row["heldout_constructive_or_strict_support"] for row in per_seed
        )
        integrity = all(
            row["reward_spectral_and_transition_integrity_pass"]
            for row in per_seed
        )
        gate_pass = bool(
            complete_count == 3
            and strict_count >= 1
            and constructive_count >= 2
            and heldout_support_count >= 2
            and integrity
            and gate_authorities["all_three_present_and_passed"]
        )
        task_results[task_name] = {
            "pass": gate_pass,
            "complete_2048_with_heldout_seed_count": complete_count,
            "complete_2048_with_heldout_required_seed_count": 3,
            "training_strict_hit_seed_count": strict_count,
            "training_strict_hit_required_seed_count": 1,
            "training_constructive_proxy_seed_count": constructive_count,
            "training_constructive_proxy_required_seed_count": 2,
            "heldout_constructive_or_strict_seed_count": heldout_support_count,
            "heldout_constructive_or_strict_required_seed_count": 2,
            "reward_spectral_and_transition_integrity_pass": integrity,
            "checkpoint_gate_authorities": gate_authorities,
            "constructive_proxy_definition": (
                runner.spectral_subspace_gate_specification()[
                    "constructive_proxy_by_task"
                ][task_name]
            ),
            "per_seed": per_seed,
        }
    screen_pass = all(row["pass"] for row in task_results.values())
    decision = (
        "bounded_screen_pass_requires_fresh_confirmation_random_and_regression"
        if screen_pass else "reject_candidate"
    )
    jobs = runner.build_jobs(
        runner.PROTOCOLS[protocol_name], mode,
        list(runner.PROTOCOLS[protocol_name].tasks),
    )
    result = {
        "schema_version": 1,
        "stage": protocol_name,
        "candidate_mode": mode,
        "candidate_mode_slug": runner.MODE_SLUGS[mode],
        "decision": decision,
        "spectral_subspace_screen_pass": screen_pass,
        "screen_pass_authorizes_full_training": False,
        "task_gates": task_results,
        "parent_RC_common_prefix_comparison": parent_comparisons,
        "parent_comparison_is_diagnostic_not_a_gate": True,
        "records": records,
        "manifest_authority": current_manifest_authority(
            protocol_name, mode, jobs
        ),
        "source_sha256": runner.source_hashes(),
        "parent_additive_source_sha256": (
            runner.rejected_crossing_additive_source_hashes()
        ),
        "spectral_additive_source_sha256": (
            runner.spectral_subspace_additive_source_hashes()
        ),
        "candidate_orchestration_source_sha256": (
            runner.rejected_crossing_orchestration_source_hashes()
        ),
        "model_sha256_by_task": {
            name: runner.model_hashes(runner.TASKS[name])
            for name in runner.PROTOCOLS[protocol_name].tasks
        },
        "candidate_reward_sha256": runner.candidate_reward_sha256(mode),
        "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        ),
        "spectral_formula_sha256": (
            runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
        ),
        "parent_RC_reject_authority": str(
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT.resolve()
        ),
        "parent_RC_reject_authority_sha256": runner.file_sha256(
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
        ),
        "spectral_offline_design_authority": str(
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY.resolve()
        ),
        "spectral_offline_design_authority_sha256": runner.file_sha256(
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
        ),
        "existing_random_baseline": (
            runner.rejected_crossing_random_baseline_disposition()
        ),
        "adoption_blockers_even_if_screen_passes": [
            "fresh_confirmation_seeds",
            "fresh_random_baseline_all_five_tasks",
            "fresh_successful_task_regression",
        ],
        "learning_curves": plot_learning_curves(
            records, f"{protocol_name}_{runner.MODE_SLUGS[mode]}"
        ),
    }
    output = (
        ANALYSIS_DIRECTORY
        / "spectral_subspace_diagnostic_P0X2RCS05_report.json"
    )
    runner.atomic_json(output, result)
    write_csv(
        ANALYSIS_DIRECTORY
        / "spectral_subspace_diagnostic_P0X2RCS05_summary.csv",
        [flatten_record(record) for record in records],
    )
    return result


def common_cap_rmst(candidate: list[dict], reference: list[dict]) -> dict:
    all_records = [*candidate, *reference]
    cap = min(
        int(row["first_hit"].get("censor_mechanical_steps") or row["training"]["total_mechanical_steps"])
        for row in all_records
    )
    def estimate(records):
        values = []
        events = 0
        for row in records:
            hit = row["first_hit"]
            if not hit["censored"] and hit["mechanical_steps"] <= cap:
                values.append(hit["mechanical_steps"])
                events += 1
            else:
                values.append(cap)
        return {"rmst_mechanical_steps": float(np.mean(values)), "events": events, "n": len(values)}
    left = estimate(candidate)
    right = estimate(reference)
    return {
        "common_mechanics_cap": cap,
        "candidate": left,
        "ori5_reference": right,
        "candidate_over_reference_ratio": (
            left["rmst_mechanical_steps"] / right["rmst_mechanical_steps"]
            if right["rmst_mechanical_steps"] else None
        ),
    }


def spectral_nonregression_report(mode: str) -> dict:
    if mode != runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
        raise ValueError("spectral_nonregression is reserved for P0X2RCS05")
    runner.validate_spectral_screen_report_authority(
        runner.SPECTRAL_SUBSPACE_SCREEN_REPORT
    )
    protocol_name = "spectral_nonregression"
    records, errors = collect(protocol_name, mode)
    if errors or len(records) != 18:
        raise RuntimeError(
            "Spectral nonregression requires 18 complete endpoints: "
            f"{errors}"
        )

    def base_reward_integrity(record: dict, prefix: str) -> bool:
        artifact = Path(record["artifact_path"])
        rows = load_csv(artifact / "evaluation_summary.csv")
        implementation = record["implementation"]
        stem = "ori6" if implementation == "ori6" else "ori5"
        required = (
            f"{stem}_reward_decomposition_error",
            f"{stem}_maximum_abs_step_reward_decomposition_error",
            f"{stem}_nonfinite_reward_component_count",
        )
        return bool(
            len(rows) == 64
            and all(
                all(field in row for field in required)
                and abs(f(row, required[0], math.inf)) <= 1.0e-10
                and abs(f(row, required[1], math.inf)) <= 1.0e-10
                and f(row, required[2], math.inf) == 0.0
                for row in rows
            )
        )

    task_results = {}
    for task_index, task_name in enumerate(
        runner.PROTOCOLS[protocol_name].tasks
    ):
        candidate = sorted([
            row for row in records
            if row["task"] == task_name and row["implementation"] == "ori6"
        ], key=lambda row: row["seed"])
        reference = sorted([
            row for row in records
            if row["task"] == task_name
            and row["implementation"] == "ori5_reference"
        ], key=lambda row: row["seed"])
        expected_seeds = [3301, 3302, 3303]
        if (
            [row["seed"] for row in candidate] != expected_seeds
            or [row["seed"] for row in reference] != expected_seeds
        ):
            raise RuntimeError(f"Unpaired spectral regression seeds for {task_name}")
        checkpoint_authority = spectral_nonregression_gate_authority(
            task_name, candidate, reference
        )
        success_differences = [
            left["last_policy_heldout"]["success_rate"]
            - right["last_policy_heldout"]["success_rate"]
            for left, right in zip(candidate, reference)
        ]
        crossing_differences = [
            left["last_policy_heldout"]["crossing_rate"]
            - right["last_policy_heldout"]["crossing_rate"]
            for left, right in zip(candidate, reference)
        ]
        success_comparison = paired_cluster_bootstrap(
            success_differences, 6330101 + task_index * 10
        )
        crossing_comparison = paired_cluster_bootstrap(
            crossing_differences, 6330102 + task_index * 10
        )
        success_noninferiority_pass = bool(
            success_comparison["one_sided_95_lower"] > -0.10
        )
        crossing_noninferiority_pass = bool(
            crossing_comparison["one_sided_95_upper"] < 0.10
        )
        candidate_spread = mean(
            row["last_policy_heldout"]["successful_mean_spread_error"]
            for row in candidate
        )
        reference_spread = mean(
            row["last_policy_heldout"]["successful_mean_spread_error"]
            for row in reference
        )
        spread_pass = bool(
            candidate_spread is not None
            and candidate_spread <= 0.05
            and (
                reference_spread is None
                or candidate_spread <= reference_spread + 0.01
            )
        )
        candidate_tail = mean(
            row["tail512"]["success_rate"] for row in candidate
        )
        reference_tail = mean(
            row["tail512"]["success_rate"] for row in reference
        )
        tail_pass = bool(
            candidate_tail is not None
            and reference_tail is not None
            and candidate_tail >= reference_tail - 0.15
        )
        mechanics = common_cap_rmst(candidate, reference)
        mechanics_evaluated = task_name in ("five_d2", "watersqrt2")
        mechanics_pass = bool(
            not mechanics_evaluated
            or mechanics["candidate_over_reference_ratio"] <= 1.5
        )
        zero_update_hits = [
            {
                "implementation": row["implementation"],
                "seed": row["seed"],
                "updates_before_hit": row["first_hit"].get(
                    "policy_updates_before_hit"
                ),
            }
            for row in [*candidate, *reference]
            if not row["first_hit"]["censored"]
            and row["first_hit"].get("policy_updates_before_hit") == 0
        ]
        candidate_integrity = all(
            row["artifact_validation"].get("valid") is True
            and row["endpoint_episode"] == 2048
            and row["training"]["maximum_abs_decomposition_error"]
            <= 1.0e-10
            and row["training"]["nonfinite_reward_components"] == 0
            and row["training"].get("rejected_crossing", {}).get(
                "transition_integrity_pass"
            ) is True
            and row["training"].get("spectral_subspace", {}).get(
                "integrity_pass"
            ) is True
            and (row.get("last_policy_heldout") or {}).get(
                "episodes"
            ) == 64
            and (row.get("last_policy_heldout") or {}).get(
                "rejected_crossing", {}
            ).get("transition_integrity_pass") is True
            and (row.get("last_policy_heldout") or {}).get(
                "spectral_subspace", {}
            ).get("integrity_pass") is True
            and base_reward_integrity(row, "ori6")
            for row in candidate
        )
        reference_integrity = all(
            row["artifact_validation"].get("valid") is True
            and row["endpoint_episode"] == 2048
            and (row.get("last_policy_heldout") or {}).get("episodes") == 64
            and base_reward_integrity(row, "ori5")
            for row in reference
        )
        integrity_pass = bool(candidate_integrity and reference_integrity)
        task_pass = bool(
            checkpoint_authority["valid_and_passed"]
            and success_noninferiority_pass
            and crossing_noninferiority_pass
            and spread_pass
            and tail_pass
            and mechanics_pass
            and integrity_pass
        )
        task_results[task_name] = {
            "pass": task_pass,
            "checkpoint_512_authority": checkpoint_authority,
            "paired_seed_success_rate_difference_candidate_minus_reference": (
                success_comparison
            ),
            "success_one_sided_lower_bound_strictly_gt_minus_0_10": (
                success_noninferiority_pass
            ),
            "paired_seed_crossing_rate_difference_candidate_minus_reference": (
                crossing_comparison
            ),
            "crossing_one_sided_upper_bound_strictly_lt_plus_0_10": (
                crossing_noninferiority_pass
            ),
            "candidate_successful_mean_spread_error": candidate_spread,
            "ori5_successful_mean_spread_error": reference_spread,
            "spread_precision_pass": spread_pass,
            "candidate_tail512_success_rate": candidate_tail,
            "ori5_tail512_success_rate": reference_tail,
            "tail_collapse_gate_pass": tail_pass,
            "screen_mechanics_rmst": mechanics,
            "mechanics_gate_evaluated": mechanics_evaluated,
            "mechanics_gate_pass": mechanics_pass,
            "learning_speed_claim_exclusion": (
                "five_d3 has a zero-update discovery mode; its first-hit "
                "mechanics are reported but excluded from learning-speed claims"
                if task_name == "five_d3" else None
            ),
            "zero_update_hits": zero_update_hits,
            "candidate_reward_transition_spectral_integrity_pass": (
                candidate_integrity
            ),
            "reference_reward_integrity_pass": reference_integrity,
            "candidate_records": candidate,
            "ori5_reference_records": reference,
        }
    overall_pass = all(row["pass"] for row in task_results.values())
    jobs = runner.build_jobs(
        runner.PROTOCOLS[protocol_name], mode,
        list(runner.PROTOCOLS[protocol_name].tasks),
    )
    result = {
        "schema_version": 1,
        "stage": protocol_name,
        "candidate_mode": mode,
        "candidate_mode_slug": runner.MODE_SLUGS[mode],
        "status": "pass" if overall_pass else "reject_or_noninferiority_not_established",
        "successful_task_regression_gate_pass": overall_pass,
        "regression_pass_is_required_before_full_manifest": True,
        "full_manifest_authorized": False,
        "full_manifest_remaining_blockers": [
            "fresh_random_baseline_all_five_tasks_under_rejected_crossing_semantics"
        ],
        "gate_specification": runner.spectral_nonregression_gate_specification(),
        "task_results": task_results,
        "records": records,
        "manifest_authority": current_manifest_authority(
            protocol_name, mode, jobs
        ),
        "source_sha256": runner.source_hashes(),
        "parent_additive_source_sha256": (
            runner.rejected_crossing_additive_source_hashes()
        ),
        "spectral_additive_source_sha256": (
            runner.spectral_subspace_additive_source_hashes()
        ),
        "candidate_orchestration_source_sha256": (
            runner.rejected_crossing_orchestration_source_hashes()
        ),
        "model_sha256_by_task": {
            name: runner.model_hashes(runner.TASKS[name])
            for name in runner.PROTOCOLS[protocol_name].tasks
        },
        "candidate_reward_sha256": runner.candidate_reward_sha256(mode),
        "reference_reward_sha256": runner.canonical_digest({
            "reward_design": runner.ORI5_REWARD_DESIGN,
            "reward_schema_version": runner.ORI5_REWARD_SCHEMA_VERSION,
            "effective_parameters": runner.expected_ori5_reward(),
        }),
        "spectral_screen_report": str(
            runner.SPECTRAL_SUBSPACE_SCREEN_REPORT.resolve()
        ),
        "spectral_screen_report_sha256": runner.file_sha256(
            runner.SPECTRAL_SUBSPACE_SCREEN_REPORT
        ),
        "parent_RC_reject_authority": str(
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT.resolve()
        ),
        "parent_RC_reject_authority_sha256": runner.file_sha256(
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
        ),
        "spectral_offline_design_authority": str(
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY.resolve()
        ),
        "spectral_offline_design_authority_sha256": runner.file_sha256(
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
        ),
        "existing_random_baseline": (
            runner.rejected_crossing_random_baseline_disposition()
        ),
        "learning_curves": plot_learning_curves(
            records, f"{protocol_name}_{runner.MODE_SLUGS[mode]}"
        ),
    }
    output = (
        ANALYSIS_DIRECTORY
        / "spectral_nonregression_P0X2RCS05_report.json"
    )
    runner.atomic_json(output, result)
    write_csv(
        ANALYSIS_DIRECTORY
        / "spectral_nonregression_P0X2RCS05_summary.csv",
        [flatten_record(record) for record in records],
    )
    return result


def regression_report(mode: str) -> dict:
    records, errors = collect("regression", mode)
    if errors or len(records) != 18:
        raise RuntimeError(f"Regression analysis requires 18 complete endpoints: {errors}")
    task_results = {}
    any_fail = False
    any_inconclusive = False
    for task_name in runner.PROTOCOLS["regression"].tasks:
        candidate = sorted([
            row for row in records
            if row["task"] == task_name and row["implementation"] == "ori6"
        ], key=lambda row: row["seed"])
        reference = sorted([
            row for row in records
            if row["task"] == task_name and row["implementation"] == "ori5_reference"
        ], key=lambda row: row["seed"])
        if [row["seed"] for row in candidate] != [row["seed"] for row in reference]:
            raise RuntimeError(f"Unpaired regression seeds for {task_name}")
        success_differences = [
            left["last_policy_heldout"]["success_rate"]
            - right["last_policy_heldout"]["success_rate"]
            for left, right in zip(candidate, reference)
        ]
        crossing_differences = [
            left["last_policy_heldout"]["crossing_rate"]
            - right["last_policy_heldout"]["crossing_rate"]
            for left, right in zip(candidate, reference)
        ]
        success_comparison = paired_cluster_bootstrap(success_differences, 6202701)
        crossing_comparison = paired_cluster_bootstrap(crossing_differences, 6202702)
        candidate_spread = mean(
            row["last_policy_heldout"]["successful_mean_spread_error"] for row in candidate
        )
        reference_spread = mean(
            row["last_policy_heldout"]["successful_mean_spread_error"] for row in reference
        )
        spread_pass = (
            candidate_spread is not None and candidate_spread <= 0.05
            and (reference_spread is None or candidate_spread <= reference_spread + 0.01)
        )
        candidate_tail = mean(row["tail512"]["success_rate"] for row in candidate)
        reference_tail = mean(row["tail512"]["success_rate"] for row in reference)
        tail_pass = candidate_tail is not None and reference_tail is not None and (
            candidate_tail >= reference_tail - 0.15
        )
        mechanics = common_cap_rmst(candidate, reference)
        mechanics_pass = (
            True if task_name == "five_d3"
            else mechanics["candidate_over_reference_ratio"] <= 1.5
        )
        definite_fail = (
            success_comparison["difference"] < -0.10
            or crossing_comparison["difference"] > 0.10
            or not spread_pass or not tail_pass or not mechanics_pass
        )
        statistical_pass = (
            success_comparison["ci95"][0] >= -0.10
            and crossing_comparison["ci95"][1] <= 0.10
        )
        if definite_fail:
            status = "fail"
            any_fail = True
        elif not statistical_pass:
            status = "inconclusive_add_seeds_3104_3105"
            any_inconclusive = True
        else:
            status = "pass"
        task_results[task_name] = {
            "status": status,
            "paired_seed_success_rate_difference_ori6_minus_ori5": success_comparison,
            "paired_seed_crossing_rate_difference_ori6_minus_ori5": crossing_comparison,
            "candidate_successful_mean_spread_error": candidate_spread,
            "ori5_successful_mean_spread_error": reference_spread,
            "spread_precision_pass": spread_pass,
            "candidate_tail512_success_rate": candidate_tail,
            "ori5_tail512_success_rate": reference_tail,
            "tail_collapse_gate_pass": tail_pass,
            "screen_mechanics_rmst": mechanics,
            "mechanics_gate_pass": mechanics_pass,
            "learning_claim_exclusion": (
                "five_d3 can succeed before the first policy update; first-hit speed is not "
                "attributed to learning" if task_name == "five_d3" else None
            ),
            "candidate_records": candidate,
            "ori5_reference_records": reference,
        }
    overall_status = (
        "reject_candidate" if any_fail else
        "add_two_seeds_before_selection" if any_inconclusive else
        "pass"
    )
    report = {
        "schema_version": 1,
        "stage": "regression",
        "candidate_mode": mode,
        "status": overall_status,
        "successful_task_regression_gate_pass": overall_status == "pass",
        "task_results": task_results,
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task) for name, task in runner.TASKS.items()
        },
        "random_baseline_authority_sha256": runner.random_baseline_authority()["sha256"],
        "records": records,
        "manifest_authority": current_manifest_authority(
            "regression", mode,
            runner.build_jobs(
                runner.PROTOCOLS["regression"], mode,
                list(runner.PROTOCOLS["regression"].tasks),
            ),
        ),
    }
    output = ANALYSIS_DIRECTORY / f"regression_{runner.MODE_SLUGS[mode]}_report.json"
    runner.atomic_json(output, report)
    write_csv(
        ANALYSIS_DIRECTORY / f"regression_{runner.MODE_SLUGS[mode]}_summary.csv",
        [flatten_record(record) for record in records],
    )
    if overall_status == "pass":
        rescue = load_json(
            ANALYSIS_DIRECTORY / f"rescue_{runner.MODE_SLUGS[mode]}_report.json"
        )
        selection = {
            "schema_version": 1,
            "selected_candidate_mode": mode,
            "rescue_gate_pass": rescue.get("rescue_gate_pass") is True,
            "successful_task_regression_gate_pass": True,
            "primary_full_gate_pass": False,
            "source_sha256": runner.source_hashes(),
            "model_sha256_by_task": report["model_sha256_by_task"],
            "random_baseline_authority_sha256": report["random_baseline_authority_sha256"],
            "rescue_report": str((ANALYSIS_DIRECTORY / f"rescue_{runner.MODE_SLUGS[mode]}_report.json").resolve()),
            "rescue_report_sha256": runner.file_sha256(
                ANALYSIS_DIRECTORY / f"rescue_{runner.MODE_SLUGS[mode]}_report.json"
            ),
            "regression_report": str(output.resolve()),
            "regression_report_sha256": runner.file_sha256(output),
        }
        runner.atomic_json(ANALYSIS_DIRECTORY / "selected_candidate.json", selection)
    return report


def flatten_record(record: dict) -> dict:
    heldout = record.get("last_policy_heldout") or {}
    hit = record["first_hit"]
    identity = record.get("candidate_identity") or {}
    training_rc = record.get("training", {}).get("rejected_crossing") or {}
    heldout_rc = heldout.get("rejected_crossing") or {}
    training_spectral = record.get("training", {}).get(
        "spectral_subspace"
    ) or {}
    heldout_spectral = heldout.get("spectral_subspace") or {}
    return {
        "task": record["task"],
        "implementation": record["implementation"],
        "candidate_mode": record["candidate_mode"],
        "ori6_candidate": identity.get("ori6_candidate"),
        "ori6_candidate_schema_version": identity.get(
            "ori6_candidate_schema_version"
        ),
        "ori6_crossing_policy": identity.get("ori6_crossing_policy"),
        "candidate_entrypoint": identity.get("candidate_entrypoint"),
        "ori6_parent_candidate": identity.get("ori6_parent_candidate"),
        "spectral_formula_sha256": identity.get("spectral_formula_sha256"),
        "spectral_weight": identity.get("spectral_weight"),
        "profile": record["profile"],
        "seed": record["seed"],
        "status": record["status"],
        "endpoint_episode": record["endpoint_episode"],
        "first_hit_episode": hit.get("episode"),
        "first_hit_mechanical_steps": hit.get("mechanical_steps"),
        "updates_before_first_hit": hit.get("policy_updates_before_hit"),
        "training_success_rate": record["training"]["success_rate"],
        "tail512_success_rate": record["tail512"]["success_rate"],
        "tail512_crossing_rate": record["tail512"]["crossing_rate"],
        "tail512_wrong_flat_rate": record["tail512"]["wrong_flat_rate"],
        "tail512_timeout_rate": record["tail512"]["timeout_rate"],
        "heldout_success_rate": heldout.get("success_rate"),
        "heldout_crossing_rate": heldout.get("crossing_rate"),
        "heldout_successful_mean_spread_error": heldout.get(
            "successful_mean_spread_error"
        ),
        "minimum_terminal_flatness_ratio": record["training"][
            "minimum_terminal_flatness_ratio"
        ],
        "minimum_same_state_joint_cost": record["training"][
            "minimum_same_state_joint_cost"
        ],
        "maximum_spread_progress": record["training"]["maximum_spread_progress"],
        "positive_frontier_episodes": record["training"]["positive_frontier_episodes"],
        "maximum_abs_decomposition_error": record["training"][
            "maximum_abs_decomposition_error"
        ],
        "nonfinite_reward_components": record["training"][
            "nonfinite_reward_components"
        ],
        "training_rejected_action_count": training_rc.get(
            "rejected_action_count"
        ),
        "training_pooled_rejected_action_rate": training_rc.get(
            "pooled_rejected_action_rate"
        ),
        "training_rejected_streak_count": training_rc.get(
            "rejected_streak_count"
        ),
        "training_escape_after_rejected_streak_count": training_rc.get(
            "escape_after_rejected_streak_count"
        ),
        "training_pooled_escape_rate": training_rc.get(
            "pooled_escape_rate"
        ),
        "training_constructive_proxy_episodes": training_rc.get(
            "constructive_proxy_episodes"
        ),
        "heldout_pooled_rejected_action_rate": heldout_rc.get(
            "pooled_rejected_action_rate"
        ),
        "heldout_pooled_escape_rate": heldout_rc.get("pooled_escape_rate"),
        "heldout_constructive_proxy_rate": heldout_rc.get(
            "constructive_proxy_rate"
        ),
        "training_spectral_integrity_pass": training_spectral.get(
            "integrity_pass"
        ),
        "training_maximum_spectral_potential": training_spectral.get(
            "maximum_potential"
        ),
        "heldout_spectral_integrity_pass": heldout_spectral.get(
            "integrity_pass"
        ),
        "artifact_path": record["artifact_path"],
    }


def moving_average(values, window=ROLLING_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1:] = (
            cumulative[window:] - cumulative[:-window]
        ) / window
    return result


def plot_learning_curves(records: list[dict], label: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_paths = []
    for task_name in sorted({record["task"] for record in records}):
        selected = [record for record in records if record["task"] == task_name]
        figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=False)
        colors = {"ori6": "tab:blue", "ori5_reference": "tab:orange"}
        for record in selected:
            history = _history_for_record(record)
            episodes = np.asarray([int(f(row, "episode")) for row in history])
            mechanics = np.cumsum([f(row, "steps") for row in history])
            implementation = record["implementation"]
            color = colors.get(implementation, "tab:gray")
            seed_label = f"{implementation} seed{record['seed']}"
            success = moving_average([f(row, "success") for row in history])
            crossing = moving_average([f(row, "self_crossing") for row in history])
            axes[0, 0].plot(episodes, success, color=color, alpha=0.55, linewidth=1.0, label=seed_label)
            axes[0, 0].plot(episodes, crossing, color=color, alpha=0.28, linewidth=0.8, linestyle="--")
            axes[0, 1].plot(mechanics, success, color=color, alpha=0.55, linewidth=1.0, label=seed_label)
            axes[1, 0].plot(
                episodes,
                moving_average([f(row, "flatness_ratio") for row in history]),
                color=color, alpha=0.55, linewidth=1.0,
            )
            axes[1, 0].plot(
                episodes,
                moving_average([f(row, "spread_error") for row in history]),
                color=color, alpha=0.3, linewidth=0.9, linestyle=":",
            )
            if implementation == "ori6":
                axes[1, 1].plot(
                    episodes,
                    moving_average([
                        f(row, "ori6_total_spread_frontier_reward") for row in history
                    ]),
                    color=color, alpha=0.65, linewidth=1.0,
                )
                axes[1, 1].plot(
                    episodes,
                    moving_average([
                        f(row, "ori6_total_frontier_rollback_reward") for row in history
                    ]),
                    color="tab:red", alpha=0.35, linewidth=0.9,
                )
            stochastic_path = Path(record["artifact_path"]) / "stochastic_evaluation_history.csv"
            if stochastic_path.is_file():
                evaluations = load_csv(stochastic_path)
                axes[0, 0].scatter(
                    [f(row, "episode") for row in evaluations],
                    [f(row, "success_rate") for row in evaluations],
                    color=color, s=15, marker="o", edgecolors="none", alpha=0.9,
                )
        axes[0, 0].set_title("200-episode success (solid), crossing (dashed); fixed eval dots")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Rate")
        axes[0, 0].set_ylim(-0.03, 1.03)
        axes[0, 1].set_title("Success against exact cumulative mechanics")
        axes[0, 1].set_xlabel("Cumulative mechanical steps")
        axes[0, 1].set_ylabel("200-episode success rate")
        axes[0, 1].set_ylim(-0.03, 1.03)
        axes[1, 0].set_title("Terminal F (solid) and spread error (dotted), rolling mean")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_yscale("symlog", linthresh=1.0e-3)
        axes[1, 1].set_title("ori6 spread frontier (blue) / rollback (red), rolling mean")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].axhline(0.0, color="0.3", linewidth=0.7)
        for axis in axes.flat:
            axis.grid(alpha=0.22)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=8, loc="best")
        figure.suptitle(f"ori6 {label}: {task_name}")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        output = ANALYSIS_DIRECTORY / "curves" / f"{label}_{task_name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        output_paths.append(str(output.resolve()))
    return output_paths


def random_comparison(record: dict) -> dict:
    random_summary = runner.random_baseline_authority()["task_random_summaries"][record["task"]]
    hit = record["first_hit"]
    rmst_mechanics = float(random_summary["restricted_mean_first_success_mechanical_steps"])
    rmst_episode = float(random_summary["restricted_mean_first_success_episode"])
    cap_mechanics = float(random_summary["common_mechanical_censor_cap"])
    cap_episode = float(random_summary["common_episode_censor_cap"])
    hit_within_random_cap = (
        not hit["censored"]
        and hit["mechanical_steps"] <= cap_mechanics
        and hit["episode"] <= cap_episode
    )
    return {
        "task": record["task"],
        "seed": record["seed"],
        "rl_first_hit_episode": hit.get("episode"),
        "rl_first_hit_mechanical_steps": hit.get("mechanical_steps"),
        "rl_updates_before_first_hit": hit.get("policy_updates_before_hit"),
        "rl_censored": hit["censored"],
        "random_search_count": int(random_summary["search_count"]),
        "random_successes": int(random_summary["successes"]),
        "random_common_episode_cap": cap_episode,
        "random_common_mechanics_cap": cap_mechanics,
        "random_episode_rmst": rmst_episode,
        "random_mechanics_rmst": rmst_mechanics,
        "rl_hit_within_random_common_cap": hit_within_random_cap,
        "random_rmst_over_rl_hit_mechanics": (
            rmst_mechanics / hit["mechanical_steps"]
            if not hit["censored"] and hit["mechanical_steps"] else None
        ),
        "comparison_status": (
            "comparable_within_common_random_cap" if hit_within_random_cap
            else "random_horizon_too_short_or_rl_censored"
        ),
        "learning_claim_caveat": (
            "0 policy updates before hit; do not attribute first discovery to learning"
            if hit.get("policy_updates_before_hit") == 0 else None
        ),
    }


def plot_random_comparisons(records: list[dict], label: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    authority = runner.random_baseline_authority()["task_random_summaries"]
    for record in records:
        task = record["task"]
        random_summary = authority[task]
        values = np.sort(np.asarray(
            random_summary.get("first_success_mechanical_steps", []), dtype=float
        ))
        n = int(random_summary["search_count"])
        cap = float(random_summary["common_mechanical_censor_cap"])
        figure, axis = plt.subplots(figsize=(7.4, 4.5))
        if len(values):
            axis.step(values, np.arange(1, len(values) + 1) / n, where="post", label="uniform random ECDF")
        censored = n - len(values)
        if censored:
            axis.scatter([cap], [len(values) / n], marker="|", s=180, color="tab:blue", label=f"random censor x{censored}")
        axis.axvline(
            float(random_summary["restricted_mean_first_success_mechanical_steps"]),
            color="tab:blue", linestyle=":", linewidth=1.8, label="random mechanics RMST",
        )
        hit = record["first_hit"]
        if not hit["censored"]:
            axis.axvline(
                hit["mechanical_steps"], color="tab:orange", linestyle="--",
                linewidth=2.0, label=f"RL seed{record['seed']} first hit",
            )
        else:
            axis.axvline(
                hit["censor_mechanical_steps"], color="tab:orange", linestyle="-.",
                linewidth=1.8, label=f"RL seed{record['seed']} censored",
            )
        axis.set_xlabel("Exact cumulative mechanical steps")
        axis.set_ylabel("Fraction discovered")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_title(f"{task}: uniform random vs ori6 first discovery")
        figure.tight_layout()
        output = (
            ANALYSIS_DIRECTORY / "random_comparison"
            / f"{label}_{task}_seed{record['seed']}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(str(output.resolve()))
    return outputs


def primary_task_gate(record: dict) -> dict:
    thresholds = {
        "five_d2": (0.75, 0.25, 0.70),
        "five_d3": (0.90, 0.10, 0.90),
        "watersqrt2": (0.75, 0.10, 0.70),
        "torus2": (0.50, 0.40, 0.50),
        "water1": (0.50, 0.40, 0.50),
    }
    heldout_success, heldout_cross, tail_success = thresholds[record["task"]]
    heldout = record["last_policy_heldout"] or {}
    hit_required = record["task"] in ("torus2", "water1")
    hit_pass = not record["first_hit"]["censored"] if hit_required else True
    spread_error = heldout.get("successful_mean_spread_error")
    checks = {
        "first_strict_hit": hit_pass,
        "heldout_success_rate": heldout.get("success_rate", 0.0) >= heldout_success,
        "heldout_crossing_rate": heldout.get("crossing_rate", 1.0) <= heldout_cross,
        "tail512_success_rate": record["tail512"]["success_rate"] >= tail_success,
        "successful_spread_error": spread_error is not None and spread_error <= 0.05,
        "numeric_audit": (
            record["training"]["maximum_abs_decomposition_error"] <= 1.0e-10
            and record["training"]["nonfinite_reward_components"] == 0
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "heldout_success_rate_min": heldout_success,
            "heldout_crossing_rate_max": heldout_cross,
            "tail512_success_rate_min": tail_success,
            "successful_spread_error_max": 0.05,
        },
    }


def full_report(protocol_name: str, mode: str) -> dict:
    records, errors = collect(protocol_name, mode)
    expected = 5 if protocol_name == "full_primary" else 10
    if errors or len(records) != expected:
        raise RuntimeError(f"{protocol_name} requires {expected} complete artifacts: {errors}")
    task_gates = {}
    comparisons = []
    for record in records:
        key = f"{record['task']}:seed{record['seed']}"
        task_gates[key] = primary_task_gate(record)
        comparisons.append(random_comparison(record))
    overall = all(row["pass"] for row in task_gates.values())
    label = f"{protocol_name}_{runner.MODE_SLUGS[mode]}"
    curves = plot_learning_curves(records, label)
    random_plots = plot_random_comparisons(records, label)
    report = {
        "schema_version": 1,
        "stage": protocol_name,
        "candidate_mode": mode,
        "primary_full_gate_pass": overall if protocol_name == "full_primary" else None,
        "replication_gate_pass": overall if protocol_name == "full_replication" else None,
        "task_seed_gates": task_gates,
        "random_comparisons": comparisons,
        "records": records,
        "manifest_authority": current_manifest_authority(
            protocol_name, mode,
            runner.build_jobs(
                runner.PROTOCOLS[protocol_name], mode,
                list(runner.PROTOCOLS[protocol_name].tasks),
            ),
        ),
        "learning_curve_pngs": curves,
        "random_comparison_pngs": random_plots,
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task) for name, task in runner.TASKS.items()
        },
        "random_baseline_authority_sha256": runner.random_baseline_authority()["sha256"],
        "wall_time_claim": None,
        "primary_cost_axis": "exact cumulative mechanical steps",
    }
    output = ANALYSIS_DIRECTORY / f"{protocol_name}_{runner.MODE_SLUGS[mode]}_report.json"
    runner.atomic_json(output, report)
    write_csv(
        ANALYSIS_DIRECTORY / f"{protocol_name}_{runner.MODE_SLUGS[mode]}_summary.csv",
        [flatten_record(record) for record in records],
    )
    write_csv(
        ANALYSIS_DIRECTORY / f"{protocol_name}_{runner.MODE_SLUGS[mode]}_random_comparison.csv",
        comparisons,
    )
    if protocol_name == "full_primary":
        selection_path = ANALYSIS_DIRECTORY / "selected_candidate.json"
        selection = load_json(selection_path)
        if selection.get("source_sha256") != runner.source_hashes():
            raise RuntimeError("Selection source hash changed during full analysis")
        selection.update({
            "primary_full_gate_pass": overall,
            "primary_full_report": str(output.resolve()),
            "primary_full_report_sha256": runner.file_sha256(output),
        })
        runner.atomic_json(selection_path, selection)
    return report


def write_protocol() -> dict:
    protocol = {
        "schema_version": 1,
        "candidate_order": [
            "offline rollback-only counterfactual",
            "P0 state_potential rescue on torus2/water1",
            "P1 gated state_potential only after a P0 safe-timeout attractor",
            (
                "bounded P0X2 rejected-crossing continuation diagnostic; "
                "screen pass alone cannot authorize adoption"
            ),
            (
                "bounded RC+spectral-subspace diagnostic; 512 is "
                "integrity-only and a pass requires later fresh confirmation"
            ),
            (
                "paired P0X2RCS05 versus ori5_reference nonregression on "
                "five_d2/five_d3/watersqrt2"
            ),
            "paired successful-task regression",
            "10k primary seed4101",
            "10k replication seeds4102/4103 only after primary passes",
        ],
        "protocols": {name: asdict(value) for name, value in runner.PROTOCOLS.items()},
        "early_checkpoints": [512, 1024],
        "early_checkpoints_by_protocol": {
            "legacy_rescue_regression": [512, 1024],
            "width_diagnostic": [512, 1024, 2048],
            "crossing_continuation_diagnostic": [512, 1024, 2048],
            "spectral_subspace_diagnostic": [512, 1024, 2048],
            "spectral_nonregression": [512],
        },
        "crossing_continuation_gate_specification": (
            runner.crossing_continuation_gate_specification()
        ),
        "spectral_subspace_gate_specification": (
            runner.spectral_subspace_gate_specification()
        ),
        "spectral_nonregression_gate_specification": (
            runner.spectral_nonregression_gate_specification()
        ),
        "success_regression_margins": {
            "heldout_success_rate_noninferiority": -0.10,
            "crossing_rate_increase_max": 0.10,
            "screen_mechanics_rmst_ratio_max": 1.5,
            "spread_error_increase_max": 0.01,
        },
        "random_reuse": runner.random_baseline_authority(),
        "crossing_continuation_random_baseline_disposition": (
            runner.rejected_crossing_random_baseline_disposition()
        ),
        "random_extension_rule": (
            "If an ori6 3-D first hit exceeds 102400 mechanics, the existing "
            "random artifact cannot establish speed. Extend uniform trials to "
            "a preregistered common cap before making a speed claim."
        ),
        "curve_specification": (
            "thin seed traces plus task-level comparison, 200-episode rolling "
            "training rates, fixed-seed evaluation points, episode and exact "
            "cumulative-mechanics axes, F/spread/frontier/rollback diagnostics"
        ),
        "source_sha256": runner.source_hashes(),
    }
    runner.atomic_json(ANALYSIS_DIRECTORY / "ori6_staged_protocol.json", protocol)
    return protocol


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "protocol", "offline", "rescue",
            "crossing_continuation_diagnostic", "regression",
            "spectral_subspace_diagnostic",
            "spectral_nonregression",
            "full_primary", "full_replication",
        ),
        default="protocol",
    )
    parser.add_argument("--candidate-mode", choices=runner.CANDIDATE_MODES, default=runner.P0_MODE)
    parser.add_argument("--derive-plane-proxy", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    ANALYSIS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_protocol()
    if options.stage == "protocol":
        print(ANALYSIS_DIRECTORY / "ori6_staged_protocol.json")
        return 0
    if options.stage == "offline":
        result = offline_rollback_counterfactual()
    elif options.stage == "rescue":
        result = rescue_report(options.candidate_mode, options.derive_plane_proxy)
    elif options.stage == "crossing_continuation_diagnostic":
        result = crossing_continuation_report(options.candidate_mode)
    elif options.stage == "spectral_subspace_diagnostic":
        result = spectral_subspace_report(options.candidate_mode)
    elif options.stage == "spectral_nonregression":
        result = spectral_nonregression_report(options.candidate_mode)
    elif options.stage == "regression":
        result = regression_report(options.candidate_mode)
    else:
        result = full_report(options.stage, options.candidate_mode)
    if options.require_complete:
        gate = (
            result.get("rescue_gate_pass")
            if options.stage == "rescue" else
            result.get("crossing_continuation_screen_pass")
            if options.stage == "crossing_continuation_diagnostic" else
            result.get("spectral_subspace_screen_pass")
            if options.stage == "spectral_subspace_diagnostic" else
            result.get("successful_task_regression_gate_pass")
            if options.stage == "spectral_nonregression" else
            result.get("successful_task_regression_gate_pass")
            if options.stage == "regression" else
            result.get("primary_full_gate_pass")
            if options.stage == "full_primary" else
            result.get("replication_gate_pass")
            if options.stage == "full_replication" else True
        )
        if gate is not True:
            raise RuntimeError(f"{options.stage} promotion gate did not pass")
    print(json.dumps({
        "stage": options.stage,
        "candidate_mode": options.candidate_mode,
        "status": result.get("status", result.get("decision", "complete")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
