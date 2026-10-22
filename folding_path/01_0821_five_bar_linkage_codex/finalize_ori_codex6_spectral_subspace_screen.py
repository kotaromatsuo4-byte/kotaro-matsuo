"""Freeze the completed P0X2RCS05 spectral-subspace screen evidence.

This finalizer is intentionally standalone.  It does not import the live ori6
learner, runner, or analyzer: the authority must remain reproducible from the
immutable manifest, gate JSONs, endpoint artifacts, and runner status files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
MANIFEST = (
    ROOT
    / "experiments_ori6/manifests/"
    "spectral_subspace_diagnostic_P0X2RCS05_9ae939802f62.json"
)
OUTPUT = (
    ROOT
    / "experiments_ori6/analysis/"
    "spectral_subspace_diagnostic_P0X2RCS05_formal_report.json"
)
PARENT_RC_AUTHORITY = (
    ROOT
    / "experiments_ori6/analysis/"
    "crossing_continuation_diagnostic_P0X2RC_reject_report.json"
)
OFFLINE_AUTHORITY = (
    ROOT
    / "experiments_ori6/analysis/"
    "spectral_subspace_offline_design_authority.json"
)
RUNNER_STDOUT = (
    ROOT
    / "experiments_ori6/_runner_logs/"
    "manual_spectral_subspace_launch_20260826T1200/runner.stdout.log"
)
RUNNER_STDERR = RUNNER_STDOUT.with_name("runner.stderr.log")

EXPECTED_MANIFEST_SHA256 = (
    "09f5116596adb382efacfc1d0c6994c7cc51f9156ec2a043a1034efdfb1b6880"
)
EXPECTED_MANIFEST_FINGERPRINT = (
    "9ae939802f624681c04e1267140adc843b7fca0b0a04a18e3f4cce4812817e7f"
)
EXPECTED_PARENT_RC_SHA256 = (
    "f520049575f0eb54b582107688bd9cada51e38f57aa64a9a28b8a1a7127d0489"
)
EXPECTED_OFFLINE_SHA256 = (
    "52b1388038b58e42f32fd1036dcf08a3db4833a45a6338e9d78e5c642f5a0856"
)
EXPECTED_MODE = (
    "state_potential_crossing_penalty_2_rejected_crossing_continuation_"
    "spectral_subspace_0p05"
)
EXPECTED_SLUG = "P0X2RCS05"
EXPECTED_SEEDS = (3201, 3202, 3203)
EXPECTED_FORMULA_SHA256 = (
    "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6"
)
EXPECTED_SPECTRAL_WEIGHT = 0.05
EXPECTED_RELATIVE_EPSILON = 1.0e-12

EXPECTED_GATES = {
    ("torus2", 512): {
        "sha256": "d387fd4530284c80cda5c55fced9a64868be5edc09bca2c5bea06b12bdf064ff",
        "stop": False,
        "reason": "spectral_subspace_integrity_continue_to_1024",
    },
    ("torus2", 1024): {
        "sha256": "52558e32a2f818633b63f416540b45127ed9f9e95e63aaa5a32bcf07139fb095",
        "stop": False,
        "reason": "spectral_subspace_continue_to_2048",
    },
    ("torus2", 2048): {
        "sha256": "3f1dd938f0a3393e8d108b29a163637f9002661a82ecbd5a0ec9bb2eeec5f8c1",
        "stop": True,
        "reason": "spectral_subspace_2048_gate_reject",
    },
    ("water1", 512): {
        "sha256": "b305e7c6833730d4f8796db5fc436d5ef5c767ac1197afefb907c8bd90182eb8",
        "stop": False,
        "reason": "spectral_subspace_integrity_continue_to_1024",
    },
    ("water1", 1024): {
        "sha256": "60d0c73f067d576a9bf4b8f23a9e718f64583165da94606af7d643beed69d458",
        "stop": True,
        "reason": "spectral_subspace_1024_gate_not_met",
    },
}

TASK_PROFILE = {"torus2": "A", "water1": "B"}
TASK_ENDPOINT = {"torus2": 2048, "water1": 1024}
PARENT_ENDPOINT = {"torus2": 512, "water1": 1024}

REQUIRED_ENDPOINT_ARTIFACTS = (
    "checkpoint.pt",
    "training_history.csv",
    "update_history.csv",
    "run_config.json",
    "best_policy.pt",
    "best_policy_metrics.json",
    "deterministic_evaluation_history.csv",
    "stochastic_evaluation_history.csv",
)

REWARD_COMPONENT_FIELDS = (
    "ori6_total_flatness_progress_reward",
    "ori6_total_target_progress_reward",
    "ori6_total_spread_frontier_reward",
    "ori6_total_hard_spread_milestone_reward",
    "ori6_total_joint_milestone_reward",
    "ori6_total_terminal_fixed_reward",
    "ori6_total_terminal_quality_reward",
    "ori6_total_terminal_speed_reward",
    "ori6_total_terminal_correction_reward",
    "ori6_spectral_total_reward",
)

RC_FIELDS = (
    "ori6_candidate",
    "ori6_candidate_schema_version",
    "ori6_crossing_policy",
    "ori6_rejected_crossing_count",
    "ori6_rejected_crossing_rate",
    "ori6_rejected_crossing_streak_count",
    "ori6_escape_after_rejected_crossing_count",
    "ori6_rejected_crossing_escape_rate",
    "ori6_unescaped_rejected_crossing_streak_count",
    "ori6_max_consecutive_rejected_crossings",
    "ori6_success_after_rejected_crossing",
    "ori6_total_rejected_crossing_reward",
    "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
)

SPECTRAL_FIELDS = (
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
    "ori6_spectral_maximum_potential",
    "ori6_spectral_maximum_delta",
    "ori6_spectral_reward_decomposition_error",
    "ori6_spectral_maximum_abs_step_reward_decomposition_error",
    "ori6_spectral_nonfinite_component_count",
)

ROLLBACK_REASONS = {
    "correction_failed",
    "no_modes",
    "self_crossing",
    "wrong_flat",
}

CHECKPOINT_CONFIG_RUNTIME_ONLY_FIELDS = {
    "base_and_rejected_authorities_unmodified",
    "base_ori6_source_is_unmodified",
    "candidate_entrypoint",
    "crossing_transition_policy",
    "ori6_candidate",
    "ori6_candidate_schema_version",
    "ori6_parent_candidate",
    "ori6_reward_parameters",
    "spectral_subspace_formula",
    "spectral_subspace_formula_sha256",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_under_root(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escapes authority root: {path}") from error
    return resolved


def relative(path: Path | str) -> str:
    return resolve_under_root(path).relative_to(ROOT).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"missing or invalid finite field {field}") from error
    require(math.isfinite(value), f"nonfinite field {field}")
    return value


def expected_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and expected_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                expected_subset(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    return canonical_bytes(actual) == canonical_bytes(expected)


def verify_self_digest(payload: dict[str, Any], label: str) -> str:
    stored = payload.get("authority_fingerprint")
    require(
        isinstance(stored, str) and len(stored) == 64,
        f"{label}: authority fingerprint absent",
    )
    unsigned = dict(payload)
    unsigned.pop("authority_fingerprint", None)
    require(
        canonical_digest(unsigned) == stored,
        f"{label}: authority self digest mismatch",
    )
    return stored


def file_binding(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing evidence file: {relative(path)}")
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def recursive_inventory(directory: Path) -> dict[str, Any]:
    require(directory.is_dir(), f"missing artifact directory: {relative(directory)}")
    entries: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    ):
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(entries),
        "total_size_bytes": sum(row["size_bytes"] for row in entries),
        "canonical_entries_sha256": canonical_digest(entries),
    }


def direct_file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file():
            inventory[path.name] = file_binding(path)
    for name in REQUIRED_ENDPOINT_ARTIFACTS:
        require(name in inventory, f"missing required endpoint artifact {name}")
    return inventory


def artifact_directory(task: str, seed: int) -> Path:
    return (
        ROOT
        / "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05"
        / task
        / TASK_PROFILE[task]
        / f"seed{seed}"
    )


def gate_path(task: str, episode: int) -> Path:
    return (
        ROOT
        / "experiments_ori6/early_gates/spectral_subspace_diagnostic/P0X2RCS05"
        / task
        / f"checkpoint_{episode}.json"
    )


def runner_status_path(task: str, seed: int) -> Path:
    root = (
        ROOT
        / "experiments_ori6/_runner_logs/spectral_subspace_diagnostic/P0X2RCS05"
        / task
        / "ori6"
        / f"seed{seed}"
    )
    matches = sorted(root.glob("*/status.json"))
    require(
        len(matches) == 1,
        f"{task} seed{seed}: expected one status, found {len(matches)}",
    )
    return matches[0]


def valid_endpoint(row: dict[str, str]) -> bool:
    return (
        finite_float(row, "self_crossing") == 0.0
        and finite_float(row, "correction_failed") == 0.0
        and finite_float(row, "no_modes") == 0.0
    )


def constructive_endpoint(row: dict[str, str], task: str) -> bool:
    if not valid_endpoint(row):
        return False
    flatness = finite_float(row, "flatness_ratio")
    spread_error = finite_float(row, "spread_error")
    if task == "torus2":
        return flatness + spread_error**2 <= 0.5
    return flatness <= 0.5 and spread_error <= 0.1


def joint_or_strict_endpoint(row: dict[str, str]) -> bool:
    return (
        valid_endpoint(row)
        and finite_float(row, "flatness_ratio") <= 0.1
        and finite_float(row, "spread_error") <= 0.05
    ) or finite_float(row, "success") > 0.0


def prefix_metrics(rows: list[dict[str, str]], task: str) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "constructive_proxy_episodes": sum(
            constructive_endpoint(row, task) for row in rows
        ),
        "valid_joint_or_strict_episodes": sum(
            joint_or_strict_endpoint(row) for row in rows
        ),
        "strict_success_episodes": sum(
            finite_float(row, "success") > 0.0 for row in rows
        ),
        "maximum_spread_progress": max(
            [finite_float(row, "ori6_maximum_spread_progress") for row in rows]
            or [0.0]
        ),
    }


def rejection_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    rejected = int(
        sum(finite_float(row, "ori6_rejected_crossing_count") for row in rows)
    )
    streaks = int(
        sum(
            finite_float(row, "ori6_rejected_crossing_streak_count")
            for row in rows
        )
    )
    escapes = int(
        sum(
            finite_float(row, "ori6_escape_after_rejected_crossing_count")
            for row in rows
        )
    )
    steps = int(sum(finite_float(row, "steps") for row in rows))
    return {
        "rejected_count": rejected,
        "streak_count": streaks,
        "escape_count": escapes,
        "step_count": steps,
        "rejected_action_rate": rejected / max(steps, 1),
        "escape_rate": escapes / streaks if streaks else None,
    }


def audit_training_rows(
    rows: list[dict[str, str]], task: str, seed: int
) -> dict[str, Any]:
    expected_episode = TASK_ENDPOINT[task]
    require(len(rows) == expected_episode, f"{task} seed{seed}: history length")
    require(
        [int(finite_float(row, "episode")) for row in rows]
        == list(range(1, expected_episode + 1)),
        f"{task} seed{seed}: noncontiguous episode history",
    )

    max_saved_decomposition = 0.0
    max_recomputed_decomposition = 0.0
    max_step_decomposition = 0.0
    nonfinite_count = 0
    max_rc_decomposition = 0.0
    max_spectral_decomposition = 0.0
    max_spectral_step_decomposition = 0.0
    max_total_progress_rollback_error = 0.0
    max_total_cumulative_error = 0.0
    max_retained_telescope_error = 0.0
    max_failure_cumulative = 0.0
    termination_counts: dict[str, int] = {}

    for row in rows:
        for field in REWARD_COMPONENT_FIELDS + (
            "total_reward",
            "ori6_reward_decomposition_error",
            "ori6_maximum_abs_step_reward_decomposition_error",
        ):
            finite_float(row, field)
        recomputed = sum(
            finite_float(row, field) for field in REWARD_COMPONENT_FIELDS
        )
        total = finite_float(row, "total_reward")
        max_recomputed_decomposition = max(
            max_recomputed_decomposition, abs(total - recomputed)
        )
        max_saved_decomposition = max(
            max_saved_decomposition,
            abs(finite_float(row, "ori6_reward_decomposition_error")),
        )
        max_step_decomposition = max(
            max_step_decomposition,
            abs(
                finite_float(
                    row, "ori6_maximum_abs_step_reward_decomposition_error"
                )
            ),
        )
        nonfinite_count += int(
            finite_float(row, "ori6_nonfinite_reward_component_count")
        )

        require(
            all(field in row for field in RC_FIELDS),
            f"{task} seed{seed}: missing RC field",
        )
        require(
            row["ori6_candidate"]
            == "ori6_P0X2_rejected_crossing_continuation",
            f"{task} seed{seed}: RC candidate mismatch",
        )
        require(
            row["ori6_candidate_schema_version"] == "1.0.0"
            and row["ori6_crossing_policy"] == "reject_candidate_and_continue",
            f"{task} seed{seed}: RC schema/policy mismatch",
        )
        rejected = finite_float(row, "ori6_rejected_crossing_count")
        streaks = finite_float(row, "ori6_rejected_crossing_streak_count")
        escapes = finite_float(
            row, "ori6_escape_after_rejected_crossing_count"
        )
        unescaped = finite_float(
            row, "ori6_unescaped_rejected_crossing_streak_count"
        )
        maximum_consecutive = finite_float(
            row, "ori6_max_consecutive_rejected_crossings"
        )
        success_after = finite_float(
            row, "ori6_success_after_rejected_crossing"
        )
        steps = finite_float(row, "steps")
        require(
            all(
                value.is_integer()
                for value in (
                    rejected,
                    streaks,
                    escapes,
                    unescaped,
                    maximum_consecutive,
                    success_after,
                )
            ),
            f"{task} seed{seed}: noninteger RC count",
        )
        require(
            0.0 <= escapes <= streaks <= rejected <= steps,
            f"{task} seed{seed}: invalid RC count ordering",
        )
        require(
            math.isclose(
                finite_float(row, "ori6_rejected_crossing_rate"),
                rejected / max(steps, 1.0),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"{task} seed{seed}: rejected rate mismatch",
        )
        require(
            math.isclose(
                finite_float(row, "ori6_rejected_crossing_escape_rate"),
                escapes / max(streaks, 1.0),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"{task} seed{seed}: escape rate mismatch",
        )
        require(
            math.isclose(
                unescaped, streaks - escapes, rel_tol=0.0, abs_tol=1.0e-12
            )
            and 0.0 <= maximum_consecutive <= rejected
            and success_after in (0.0, 1.0),
            f"{task} seed{seed}: RC ledger mismatch",
        )
        max_rc_decomposition = max(
            max_rc_decomposition,
            abs(
                finite_float(
                    row,
                    "ori6_maximum_abs_rejected_crossing_reward_"
                    "decomposition_error",
                )
            ),
        )
        finite_float(row, "ori6_total_rejected_crossing_reward")
        require(
            finite_float(row, "self_crossing") == 0.0,
            f"{task} seed{seed}: committed crossing in RC history",
        )

        require(
            all(field in row for field in SPECTRAL_FIELDS),
            f"{task} seed{seed}: missing spectral field",
        )
        require(
            row["ori6_spectral_candidate"]
            == "ori6_RC_spectral_subspace_potential"
            and row["ori6_spectral_candidate_schema_version"] == "1.0.0"
            and row["ori6_spectral_formula_sha256"]
            == EXPECTED_FORMULA_SHA256
            and row["ori6_spectral_parent_candidate"]
            == "ori6_P0X2_rejected_crossing_continuation",
            f"{task} seed{seed}: spectral identity mismatch",
        )
        require(
            finite_float(row, "ori6_spectral_weight")
            == EXPECTED_SPECTRAL_WEIGHT
            and finite_float(row, "ori6_spectral_relative_epsilon")
            == EXPECTED_RELATIVE_EPSILON,
            f"{task} seed{seed}: spectral numeric identity mismatch",
        )
        progress = finite_float(row, "ori6_spectral_total_progress_reward")
        rollback = finite_float(row, "ori6_spectral_total_rollback_reward")
        spectral_total = finite_float(row, "ori6_spectral_total_reward")
        cumulative = finite_float(row, "ori6_spectral_cumulative_reward")
        terminal_potential = finite_float(
            row, "ori6_spectral_terminal_potential"
        )
        terminal_delta = finite_float(row, "ori6_spectral_terminal_delta")
        maximum_potential = finite_float(
            row, "ori6_spectral_maximum_potential"
        )
        maximum_delta = finite_float(row, "ori6_spectral_maximum_delta")
        require(
            0.0 <= terminal_potential <= 1.0 + 1.0e-12
            and 0.0 <= terminal_delta <= 1.0 + 1.0e-12
            and 0.0 <= maximum_potential <= 1.0 + 1.0e-12
            and 0.0 <= maximum_delta <= 1.0 + 1.0e-12,
            f"{task} seed{seed}: spectral potential/delta out of range",
        )
        max_spectral_decomposition = max(
            max_spectral_decomposition,
            abs(
                finite_float(row, "ori6_spectral_reward_decomposition_error")
            ),
        )
        max_spectral_step_decomposition = max(
            max_spectral_step_decomposition,
            abs(
                finite_float(
                    row,
                    "ori6_spectral_maximum_abs_step_reward_"
                    "decomposition_error",
                )
            ),
        )
        require(
            finite_float(row, "ori6_spectral_nonfinite_component_count")
            == 0.0,
            f"{task} seed{seed}: spectral nonfinite component",
        )
        max_total_progress_rollback_error = max(
            max_total_progress_rollback_error,
            abs(spectral_total - progress - rollback),
        )
        max_total_cumulative_error = max(
            max_total_cumulative_error, abs(spectral_total - cumulative)
        )
        reason = row["termination_reason"]
        if reason in ROLLBACK_REASONS:
            max_failure_cumulative = max(
                max_failure_cumulative, abs(cumulative)
            )
        else:
            max_retained_telescope_error = max(
                max_retained_telescope_error,
                abs(
                    cumulative
                    - EXPECTED_SPECTRAL_WEIGHT * terminal_potential
                ),
            )
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    audit = {
        "episode_count": len(rows),
        "termination_counts": dict(sorted(termination_counts.items())),
        "maximum_abs_saved_episode_reward_decomposition_error": (
            max_saved_decomposition
        ),
        "maximum_abs_recomputed_episode_reward_decomposition_error": (
            max_recomputed_decomposition
        ),
        "maximum_abs_step_reward_decomposition_error": max_step_decomposition,
        "nonfinite_reward_component_count": nonfinite_count,
        "maximum_abs_rejected_crossing_reward_decomposition_error": (
            max_rc_decomposition
        ),
        "maximum_abs_spectral_reward_decomposition_error": (
            max_spectral_decomposition
        ),
        "maximum_abs_spectral_step_reward_decomposition_error": (
            max_spectral_step_decomposition
        ),
        "maximum_abs_spectral_total_minus_progress_minus_rollback": (
            max_total_progress_rollback_error
        ),
        "maximum_abs_spectral_total_minus_cumulative": (
            max_total_cumulative_error
        ),
        "maximum_abs_retained_cumulative_minus_weight_times_phi": (
            max_retained_telescope_error
        ),
        "maximum_abs_failure_cumulative_after_rollback": (
            max_failure_cumulative
        ),
    }
    audit["pass"] = bool(
        max_saved_decomposition <= 1.0e-10
        and max_recomputed_decomposition <= 1.0e-10
        and max_step_decomposition <= 1.0e-10
        and nonfinite_count == 0
        and max_rc_decomposition <= 1.0e-10
        and max_spectral_decomposition <= 1.0e-10
        and max_spectral_step_decomposition <= 1.0e-10
        and max_total_progress_rollback_error <= 1.0e-10
        and max_total_cumulative_error <= 1.0e-10
        and max_retained_telescope_error <= 1.0e-10
        and max_failure_cumulative <= 1.0e-10
    )
    require(audit["pass"], f"{task} seed{seed}: reward/integrity audit failed")
    return audit


def critic_diagnostics(
    rows: list[dict[str, str]], run_config: dict[str, Any], endpoint: int
) -> dict[str, Any]:
    threshold = float(run_config["critic_gradient_clip"])
    expected_episodes = list(range(8, endpoint + 1, 8))
    episodes = [int(finite_float(row, "episode")) for row in rows]
    require(episodes == expected_episodes, "update history schedule mismatch")
    norms = [finite_float(row, "critic_gradient_norm") for row in rows]

    phases: list[dict[str, Any]] = []
    phase_start = 1
    for phase_end in range(512, endpoint + 1, 512):
        selected = [
            finite_float(row, "critic_gradient_norm")
            for row in rows
            if phase_start <= int(finite_float(row, "episode")) <= phase_end
        ]
        clipped = sum(value > threshold for value in selected)
        phases.append(
            {
                "episode_start": phase_start,
                "episode_end": phase_end,
                "update_count": len(selected),
                "preclip_norm_gt_threshold_count": clipped,
                "preclip_norm_gt_threshold_rate": clipped / len(selected),
                "mean_preclip_norm": sum(selected) / len(selected),
                "maximum_preclip_norm": max(selected),
            }
        )
        phase_start = phase_end + 1

    clipped = sum(value > threshold for value in norms)
    return {
        "logged_norm_semantics": "pre-clipping gradient norm",
        "critic_gradient_clip_threshold": threshold,
        "update_count": len(norms),
        "preclip_norm_gt_threshold_count": clipped,
        "preclip_norm_gt_threshold_rate": clipped / len(norms),
        "mean_preclip_norm": sum(norms) / len(norms),
        "maximum_preclip_norm": max(norms),
        "phases": phases,
    }


def endpoint_signal_metrics(
    rows: list[dict[str, str]], task: str
) -> dict[str, Any]:
    metrics = prefix_metrics(rows, task)
    metrics.update(
        {
            "maximum_spectral_potential": max(
                finite_float(row, "ori6_spectral_maximum_potential")
                for row in rows
            ),
            "maximum_spectral_delta": max(
                finite_float(row, "ori6_spectral_maximum_delta")
                for row in rows
            ),
            "positive_spectral_peak_episodes": sum(
                finite_float(row, "ori6_spectral_maximum_potential")
                > 1.0e-12
                for row in rows
            ),
            "spectral_delta_gt_1e_6_episodes": sum(
                finite_float(row, "ori6_spectral_maximum_delta") > 1.0e-6
                for row in rows
            ),
            "spectral_progress_reward_sum": sum(
                finite_float(row, "ori6_spectral_total_progress_reward")
                for row in rows
            ),
            "spectral_rollback_reward_sum": sum(
                finite_float(row, "ori6_spectral_total_rollback_reward")
                for row in rows
            ),
            "spectral_total_reward_sum": sum(
                finite_float(row, "ori6_spectral_total_reward")
                for row in rows
            ),
            "rejected_crossing": rejection_counts(rows),
        }
    )
    return metrics


def recompute_formal_gate(
    task: str,
    episode: int,
    rows_by_seed: dict[int, list[dict[str, str]]],
    audits_by_seed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    prefixes = {seed: rows[:episode] for seed, rows in rows_by_seed.items()}
    shape = bool(
        set(prefixes) == set(EXPECTED_SEEDS)
        and all(len(rows) >= episode for rows in prefixes.values())
        and all(
            [int(finite_float(row, "episode")) for row in rows]
            == list(range(1, episode + 1))
            for rows in prefixes.values()
        )
    )
    per_seed = {
        seed: endpoint_signal_metrics(rows, task)
        for seed, rows in prefixes.items()
    }
    integrity = bool(
        shape
        and all(audits_by_seed[seed]["pass"] for seed in EXPECTED_SEEDS)
    )
    constructive_count = sum(
        row["constructive_proxy_episodes"] >= 1
        for row in per_seed.values()
    )
    joint_count = sum(
        row["valid_joint_or_strict_episodes"] >= 1
        for row in per_seed.values()
    )
    strict_count = sum(
        row["strict_success_episodes"] >= 1 for row in per_seed.values()
    )
    positive_peak_count = sum(
        row["maximum_spectral_potential"] > 1.0e-12
        for row in per_seed.values()
    )
    delta_count = sum(
        row["maximum_spectral_delta"] > 1.0e-6
        for row in per_seed.values()
    )

    if episode == 512:
        passed = integrity
        phase = "pathology_and_integrity_only"
        reason = (
            "spectral_subspace_integrity_continue_to_1024"
            if passed
            else "spectral_subspace_512_integrity_failure"
        )
    elif episode == 1024:
        passed = bool(
            integrity
            and constructive_count >= 2
            and joint_count >= 1
            and positive_peak_count >= 2
            and delta_count >= 2
        )
        phase = "constructive_spectral_screen"
        reason = (
            "spectral_subspace_continue_to_2048"
            if passed
            else "spectral_subspace_1024_gate_not_met"
        )
    else:
        passed = bool(
            integrity and strict_count >= 1 and constructive_count >= 2
        )
        phase = "finalize_for_heldout_analysis"
        reason = (
            "spectral_subspace_finalize_heldout"
            if passed
            else "spectral_subspace_2048_gate_reject"
        )

    return {
        "cohort_shape_valid": shape,
        "reward_spectral_and_transition_integrity_pass": integrity,
        "constructive_prefix_seed_count": constructive_count,
        "valid_F_le_0_1_e_le_0_05_or_strict_seed_count": joint_count,
        "strict_hit_seed_count": strict_count,
        "positive_spectral_peak_seed_count": positive_peak_count,
        "spectral_delta_gt_1e_6_seed_count": delta_count,
        "phase": phase,
        "pass": passed,
        "stop": not passed,
        "reason": reason,
        "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
    }


def compare_gate_to_recomputation(
    payload: dict[str, Any], recomputed: dict[str, Any], task: str, episode: int
) -> None:
    formal = payload.get("formal_spectral_subspace_gate", {})
    for key in (
        "cohort_shape_valid",
        "reward_spectral_and_transition_integrity_pass",
        "constructive_prefix_seed_count",
        "valid_F_le_0_1_e_le_0_05_or_strict_seed_count",
        "strict_hit_seed_count",
        "positive_spectral_peak_seed_count",
        "spectral_delta_gt_1e_6_seed_count",
        "phase",
        "pass",
    ):
        require(
            formal.get(key) == recomputed[key],
            f"{task} checkpoint{episode}: formal field mismatch {key}",
        )
    require(
        payload.get("stop") == recomputed["stop"]
        and payload.get("reason") == recomputed["reason"],
        f"{task} checkpoint{episode}: stop/reason mismatch",
    )
    require(
        payload.get("cohort_shape_valid") is recomputed["cohort_shape_valid"],
        f"{task} checkpoint{episode}: cohort shape mismatch",
    )
    recorded_per_seed = payload.get("per_seed", {})
    require(
        set(recorded_per_seed) == {str(seed) for seed in EXPECTED_SEEDS},
        f"{task} checkpoint{episode}: gate seed set mismatch",
    )
    for seed in EXPECTED_SEEDS:
        recorded = recorded_per_seed[str(seed)]
        fresh = recomputed["per_seed"][str(seed)]
        exact_fields = {
            "prefix_constructive_proxy_episodes": fresh[
                "constructive_proxy_episodes"
            ],
            "prefix_valid_joint_target_episodes": fresh[
                "valid_joint_or_strict_episodes"
            ],
            "success_episodes": fresh["strict_success_episodes"],
            "maximum_spectral_potential": fresh[
                "maximum_spectral_potential"
            ],
            "maximum_spectral_delta": fresh["maximum_spectral_delta"],
            "positive_spectral_peak_episodes": fresh[
                "positive_spectral_peak_episodes"
            ],
            "spectral_delta_gt_1e_6_episodes": fresh[
                "spectral_delta_gt_1e_6_episodes"
            ],
        }
        for key, value in exact_fields.items():
            require(
                recorded.get(key) == value,
                f"{task} checkpoint{episode} seed{seed}: {key} mismatch",
            )
        for key in (
            "reward_integrity",
            "rejected_crossing_integrity",
            "spectral_integrity",
        ):
            require(
                recorded.get(key) is True,
                f"{task} checkpoint{episode} seed{seed}: {key} failed",
            )
        require(
            recorded.get("prefix_rejected_crossing")
            == fresh["rejected_crossing"],
            f"{task} checkpoint{episode} seed{seed}: RC aggregate mismatch",
        )


def parent_comparison(
    task: str,
    candidate_rows: dict[int, list[dict[str, str]]],
    parent_authority: dict[str, Any],
    requested_endpoint: int | None = None,
) -> dict[str, Any]:
    endpoint = TASK_ENDPOINT[task] if requested_endpoint is None else requested_endpoint
    parent_records = {
        (row["task"], int(row["seed"])): row
        for row in parent_authority["atomic_checkpoint_integrity"]["per_seed"]
    }
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        record = parent_records[(task, seed)]
        parent_directory = resolve_under_root(record["artifact_path"])
        parent_history = parent_directory / "training_history.csv"
        bound = record["artifact_inventory"]["training_history.csv"]
        require(
            file_sha256(parent_history) == bound["sha256"]
            and parent_history.stat().st_size == bound["size_bytes"],
            f"{task} seed{seed}: parent history authority mismatch",
        )
        rows = read_csv(parent_history)
        common = min(
            endpoint, int(record["checkpoint_episode"]), len(rows)
        )
        candidate = prefix_metrics(candidate_rows[seed][:common], task)
        parent = prefix_metrics(rows[:common], task)
        per_seed[str(seed)] = {
            "available": common > 0,
            "parent_atomic_endpoint": int(record["checkpoint_episode"]),
            "common_prefix_episode": common,
            "candidate": candidate,
            "parent_RC": parent,
            "candidate_minus_parent_RC": {
                key: candidate[key] - parent[key]
                for key in (
                    "constructive_proxy_episodes",
                    "valid_joint_or_strict_episodes",
                    "strict_success_episodes",
                    "maximum_spread_progress",
                )
            },
        }
    return {
        "gating": False,
        "requested_candidate_endpoint": endpoint,
        "parent_requested_endpoint_available": all(
            row["parent_atomic_endpoint"] >= endpoint
            for row in per_seed.values()
        ),
        "missing_parent_requested_endpoint_invalidates": False,
        "per_seed": per_seed,
    }


def water_strict_success_evidence(
    histories: dict[tuple[str, int], list[dict[str, str]]]
) -> dict[str, Any]:
    successes: list[tuple[str, int, dict[str, str]]] = []
    for (task, seed), rows in histories.items():
        for row in rows:
            if finite_float(row, "success") > 0.0:
                successes.append((task, seed, row))
    require(len(successes) == 1, "expected exactly one strict screen success")
    task, seed, row = successes[0]
    require(
        task == "water1" and seed == 3203 and int(row["episode"]) == 22,
        "unexpected strict screen success identity",
    )
    require(
        valid_endpoint(row)
        and finite_float(row, "flatness_ratio") <= 0.001
        and finite_float(row, "spread_error") <= 0.05
        and row["termination_reason"] == "success",
        "water episode22 strict criteria failed",
    )

    path = artifact_directory(task, seed) / "first_success_path"
    actions_path = path / "actions.json"
    actions = load_json(actions_path)
    require(isinstance(actions, list) and len(actions) == 222, "success actions")
    state_paths = sorted(
        path.glob("step*.dat"),
        key=lambda item: int(item.stem.removeprefix("step")),
    )
    require(len(state_paths) == 223, "success state count")
    require(
        [int(item.stem.removeprefix("step")) for item in state_paths]
        == list(range(223)),
        "success state sequence",
    )
    last_action = actions[-1]
    require(
        last_action.get("step") == 222
        and last_action.get("termination_reason") == "success"
        and last_action.get("success_at_step") == 1,
        "success final action semantics",
    )
    for action_field, history_field in (
        ("flatness_ratio", "flatness_ratio"),
        ("spread", "spread"),
        ("spread_error", "spread_error"),
        ("ori6_spectral_potential_after", "ori6_spectral_terminal_potential"),
        ("ori6_spectral_cumulative_reward_after", "ori6_spectral_total_reward"),
    ):
        require(
            math.isclose(
                float(last_action[action_field]),
                finite_float(row, history_field),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            f"success action/history mismatch {action_field}",
        )

    return {
        "task": task,
        "seed": seed,
        "episode": 22,
        "steps": int(finite_float(row, "steps")),
        "termination_reason": row["termination_reason"],
        "flatness_ratio": finite_float(row, "flatness_ratio"),
        "spread": finite_float(row, "spread"),
        "spread_error": finite_float(row, "spread_error"),
        "total_reward": finite_float(row, "total_reward"),
        "spectral_terminal_potential": finite_float(
            row, "ori6_spectral_terminal_potential"
        ),
        "spectral_maximum_potential": finite_float(
            row, "ori6_spectral_maximum_potential"
        ),
        "spectral_total_reward": finite_float(
            row, "ori6_spectral_total_reward"
        ),
        "rejected_crossing_count": int(
            finite_float(row, "ori6_rejected_crossing_count")
        ),
        "rejected_crossing_streak_count": int(
            finite_float(row, "ori6_rejected_crossing_streak_count")
        ),
        "escape_after_rejected_crossing_count": int(
            finite_float(row, "ori6_escape_after_rejected_crossing_count")
        ),
        "success_after_rejected_crossing": int(
            finite_float(row, "ori6_success_after_rejected_crossing")
        ),
        "strict_criteria_recomputed": True,
        "first_success_path": {
            "path": relative(path),
            "actions": file_binding(actions_path),
            "action_count": len(actions),
            "state_file_count": len(state_paths),
            "recursive_inventory": recursive_inventory(path),
            "last_action_matches_history": True,
        },
    }


def verify_manifest() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    require(file_sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256, "manifest SHA")
    manifest = load_json(MANIFEST)
    require(manifest["manifest_fingerprint"] == EXPECTED_MANIFEST_FINGERPRINT, "manifest fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    unsigned.pop("created_utc", None)
    require(canonical_digest(unsigned) == EXPECTED_MANIFEST_FINGERPRINT, "manifest self digest")
    require(manifest["candidate_mode"] == EXPECTED_MODE, "manifest mode")
    require(manifest["candidate_mode_slug"] == EXPECTED_SLUG, "manifest slug")
    require(manifest["spectral_formula_sha256"] == EXPECTED_FORMULA_SHA256, "formula SHA")
    require(manifest["spectral_relative_epsilon"] == EXPECTED_RELATIVE_EPSILON, "relative epsilon")
    require(manifest["single_changed_reward_parameter"]["candidate_value"] == EXPECTED_SPECTRAL_WEIGHT, "spectral weight")
    jobs = {(job["task"]["name"], int(job["seed"])): job for job in manifest["jobs"]}
    require(
        set(jobs)
        == {(task, seed) for task in TASK_PROFILE for seed in EXPECTED_SEEDS},
        "manifest job cohort",
    )
    # These are historical launch identities.  Later candidates are allowed to
    # move the live sources on, so (as in the W2 historical finalizer) bind the
    # immutable manifest/job/status fingerprints rather than requiring the
    # current workspace bytes to retain an old experiment's hashes.
    return manifest, jobs


def build_report(generated_utc: str) -> dict[str, Any]:
    manifest, jobs = verify_manifest()
    require(file_sha256(PARENT_RC_AUTHORITY) == EXPECTED_PARENT_RC_SHA256, "parent RC authority SHA")
    parent_authority = load_json(PARENT_RC_AUTHORITY)
    parent_fingerprint = verify_self_digest(parent_authority, "parent RC")
    require(file_sha256(OFFLINE_AUTHORITY) == EXPECTED_OFFLINE_SHA256, "offline authority SHA")
    offline_authority = load_json(OFFLINE_AUTHORITY)
    offline_fingerprint = verify_self_digest(offline_authority, "offline spectral")

    histories: dict[tuple[str, int], list[dict[str, str]]] = {}
    audits: dict[tuple[str, int], dict[str, Any]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    endpoint_metrics: dict[str, dict[str, Any]] = {task: {} for task in TASK_PROFILE}

    for task in TASK_PROFILE:
        endpoint = TASK_ENDPOINT[task]
        for seed in EXPECTED_SEEDS:
            job = jobs[(task, seed)]
            directory = artifact_directory(task, seed)
            require(directory.resolve() == Path(job["output_directory"]).resolve(), f"{task} seed{seed}: output path")
            direct_inventory = direct_file_inventory(directory)
            tree_inventory = recursive_inventory(directory)
            history_path = directory / "training_history.csv"
            update_path = directory / "update_history.csv"
            config_path = directory / "run_config.json"
            rows = read_csv(history_path)
            updates = read_csv(update_path)
            run_config = load_json(config_path)
            histories[(task, seed)] = rows
            audit = audit_training_rows(rows, task, seed)
            audits[(task, seed)] = audit
            metrics = endpoint_signal_metrics(rows, task)
            endpoint_metrics[task][str(seed)] = metrics
            critic = critic_diagnostics(updates, run_config, endpoint)
            critic_rows.append({"task": task, "seed": seed, **critic})

            missing_run_config_fields = set(job["expected_config"]) - set(run_config)
            require(
                missing_run_config_fields
                in (set(), CHECKPOINT_CONFIG_RUNTIME_ONLY_FIELDS),
                f"{task} seed{seed}: unexpected run_config field set",
            )
            run_config_expected = {
                key: value
                for key, value in job["expected_config"].items()
                if key in run_config
            }
            require(
                expected_subset(run_config, run_config_expected),
                f"{task} seed{seed}: run_config training subset",
            )
            checkpoint = torch.load(
                directory / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            require(int(checkpoint["episode"]) == endpoint, f"{task} seed{seed}: checkpoint episode")
            checkpoint_config = checkpoint["config"]
            missing_checkpoint_fields = set(job["expected_config"]) - set(
                checkpoint_config
            )
            require(
                missing_checkpoint_fields == CHECKPOINT_CONFIG_RUNTIME_ONLY_FIELDS,
                f"{task} seed{seed}: unexpected checkpoint config field set",
            )
            checkpoint_expected = {
                key: value
                for key, value in job["expected_config"].items()
                if key in checkpoint_config
            }
            require(
                expected_subset(checkpoint_config, checkpoint_expected),
                f"{task} seed{seed}: checkpoint config training subset",
            )
            require(len(updates) == endpoint // 8, f"{task} seed{seed}: update rows")

            status_path = runner_status_path(task, seed)
            status = load_json(status_path)
            stderr_path = status_path.with_name("stderr.log")
            stdout_path = status_path.with_name("stdout.log")
            require(status["job_id"] == job["job_id"], f"{task} seed{seed}: status job id")
            require(status["job_fingerprint"] == job["fingerprint"], f"{task} seed{seed}: status fingerprint")
            require(status["command"] == job["command"], f"{task} seed{seed}: status command")
            require(Path(status["requested_output_directory"]).resolve() == directory.resolve(), f"{task} seed{seed}: requested output")
            require(Path(status["effective_output_directory"]).resolve() == directory.resolve(), f"{task} seed{seed}: effective output")
            require(stderr_path.stat().st_size == 0, f"{task} seed{seed}: stderr nonempty")

            if task == "torus2" and seed in (3201, 3202):
                require(status["status"] == "complete" and status["return_code"] == 0, f"{task} seed{seed}: complete status")
                validation = status["artifact_validation"]
                require(validation["valid"] is True and validation["complete"] is True, f"{task} seed{seed}: final validation")
                require("summary.json" in direct_inventory, f"{task} seed{seed}: summary absent")
            else:
                require(status["status"] == "early_stopped_at_checkpoint" and status["return_code"] == 1, f"{task} seed{seed}: early-stop status")
                require(int(status["checkpoint_episode"]) == endpoint and int(status["cohort_decision_checkpoint_episode"]) == endpoint, f"{task} seed{seed}: stop endpoint")
                validation = status["partial_artifact_validation"]
                require(validation["valid"] is True and validation["complete"] is False, f"{task} seed{seed}: partial validation")
                require("summary.json" not in direct_inventory, f"{task} seed{seed}: unexpected summary")
            require(
                int(validation["endpoint_episode"]) == endpoint
                and int(validation["training_rows"]) == endpoint
                and int(validation["update_rows"]) == endpoint // 8,
                f"{task} seed{seed}: status validation counts",
            )

            endpoint_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "job_id": job["job_id"],
                    "job_fingerprint": job["fingerprint"],
                    "artifact_path": relative(directory),
                    "checkpoint_episode": endpoint,
                    "training_rows": len(rows),
                    "update_rows": len(updates),
                    "run_config_training_subset_matches_manifest_job": True,
                    "run_config_runtime_only_fields_missing": sorted(
                        missing_run_config_fields
                    ),
                    "checkpoint_config_training_subset_matches_manifest_job": True,
                    "checkpoint_config_runtime_only_fields_recorded_in_run_config": sorted(
                        CHECKPOINT_CONFIG_RUNTIME_ONLY_FIELDS
                    ),
                    "checkpoint_episode_matches_endpoint": True,
                    "direct_file_inventory": direct_inventory,
                    "recursive_artifact_inventory": tree_inventory,
                    "runner_status": file_binding(status_path),
                    "runner_status_value": status["status"],
                    "runner_return_code": int(status["return_code"]),
                    "runner_stdout": file_binding(stdout_path),
                    "runner_stderr": {
                        **file_binding(stderr_path),
                        "empty": True,
                    },
                    "artifact_validation": validation,
                    "reward_and_transition_integrity": audit,
                }
            )

    gate_rows: list[dict[str, Any]] = []
    gate_recomputations: dict[tuple[str, int], dict[str, Any]] = {}
    for (task, episode), expected in EXPECTED_GATES.items():
        path = gate_path(task, episode)
        require(file_sha256(path) == expected["sha256"], f"{task} checkpoint{episode}: gate SHA")
        payload = load_json(path)
        rows_by_seed = {
            seed: histories[(task, seed)] for seed in EXPECTED_SEEDS
        }
        task_audits = {seed: audits[(task, seed)] for seed in EXPECTED_SEEDS}
        recomputed = recompute_formal_gate(
            task, episode, rows_by_seed, task_audits
        )
        compare_gate_to_recomputation(payload, recomputed, task, episode)
        require(payload["stop"] == expected["stop"] and payload["reason"] == expected["reason"], f"{task} checkpoint{episode}: expected disposition")
        gate_recomputations[(task, episode)] = recomputed
        gate_rows.append(
            {
                **file_binding(path),
                "task": task,
                "checkpoint_episode": episode,
                "payload": payload,
                "critical_semantic_recomputation": recomputed,
                "critical_semantic_recomputation_match": True,
            }
        )

    water_2048 = gate_path("water1", 2048)
    require(not water_2048.exists(), "water checkpoint2048 unexpectedly exists")

    parent_comparisons = {
        task: parent_comparison(
            task,
            {seed: histories[(task, seed)] for seed in EXPECTED_SEEDS},
            parent_authority,
        )
        for task in TASK_PROFILE
    }
    require(
        parent_comparisons["torus2"]["parent_requested_endpoint_available"]
        is False
        and parent_comparisons["water1"][
            "parent_requested_endpoint_available"
        ]
        is True,
        "parent common-prefix availability mismatch",
    )
    gate_parent_comparisons: dict[str, Any] = {}
    for task in ("torus2", "water1"):
        gate_parent = next(
            row["payload"]["formal_spectral_subspace_gate"].get(
                "parent_common_prefix_comparison"
            )
            for row in gate_rows
            if row["task"] == task and row["checkpoint_episode"] == 1024
        )
        gate_parent_comparisons[task] = parent_comparison(
            task,
            {seed: histories[(task, seed)] for seed in EXPECTED_SEEDS},
            parent_authority,
            requested_endpoint=1024,
        )
        require(
            gate_parent == gate_parent_comparisons[task],
            f"{task}: parent common-prefix recomputation mismatch",
        )

    strict_evidence = water_strict_success_evidence(histories)
    torus_final = gate_recomputations[("torus2", 2048)]
    water_final = gate_recomputations[("water1", 1024)]
    require(
        torus_final["constructive_prefix_seed_count"] == 3
        and torus_final[
            "valid_F_le_0_1_e_le_0_05_or_strict_seed_count"
        ]
        == 3
        and torus_final["strict_hit_seed_count"] == 0,
        "torus final evidence mismatch",
    )
    require(
        water_final["constructive_prefix_seed_count"] == 1
        and water_final["strict_hit_seed_count"] == 1,
        "water final evidence mismatch",
    )

    torus_status_rows = [row for row in endpoint_rows if row["task"] == "torus2"]
    water_status_rows = [row for row in endpoint_rows if row["task"] == "water1"]
    require(
        [row["runner_status_value"] for row in torus_status_rows]
        == ["complete", "complete", "early_stopped_at_checkpoint"],
        "torus status cohort",
    )
    require(
        all(
            row["runner_status_value"] == "early_stopped_at_checkpoint"
            for row in water_status_rows
        ),
        "water status cohort",
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "authority_type": "ori6_spectral_subspace_atomic_screen_formal_report",
        "generated_utc": generated_utc,
        "stage": "spectral_subspace_diagnostic",
        "candidate_mode": EXPECTED_MODE,
        "candidate_mode_slug": EXPECTED_SLUG,
        "decision": "formal_gate_reject_but_promising_signal_for_regression",
        "spectral_subspace_screen_pass": False,
        "candidate_adopted": False,
        "full_10000_episode_training_authorized_by_this_report": False,
        "fresh_confirmation_required": True,
        "decision_reason": {
            "formal_reject_predicates": [
                {
                    "task": "torus2",
                    "checkpoint_episode": 2048,
                    "predicate": "strict_hit_seed_count >= 1",
                    "observed": 0,
                    "required": 1,
                    "exact_reason": "spectral_subspace_2048_gate_reject",
                },
                {
                    "task": "water1",
                    "checkpoint_episode": 1024,
                    "predicate": "constructive_prefix_seed_count >= 2",
                    "observed": 1,
                    "required": 2,
                    "exact_reason": "spectral_subspace_1024_gate_not_met",
                },
            ],
            "promising_signal": [
                "water1 seed3203 produced the only strict screen success at episode 22; its paired RC parent prefix had zero strict successes",
                "torus2 reached constructive and valid-joint evidence in all three seeds by episode 2048 despite zero strict hits",
                "all spectral reward, rollback, transition, and nonfinite checks passed",
            ],
        },
        "user_defined_possibility_rationale": {
            "classification": "possibility_visible_but_not_confirmed",
            "why_promising": (
                "The minimal lambda=0.05 change produced a new strict water1 path "
                "and broad torus constructive/joint evidence, so the signal is not "
                "classified as hopeless."
            ),
            "why_not_adoption": (
                "The preregistered task gates rejected: torus had no strict hit, "
                "water support remained one seed, and water did not reach episode "
                "2048.  One early water success is possibility evidence, not "
                "multi-seed learning confirmation."
            ),
            "permitted_next_claim": (
                "A separately preregistered fresh-seed regression/confirmation "
                "may be considered; this report does not itself authorize adoption "
                "or a 10000-episode production run."
            ),
        },
        "scope": {
            "protocol": "spectral_subspace_diagnostic",
            "seeds": list(EXPECTED_SEEDS),
            "tasks": ["torus2", "water1"],
            "checkpoint_every": 512,
            "preregistered_episode_cap": 2048,
            "task_atomic_endpoints": TASK_ENDPOINT,
            "fixed_independent_task_cohorts": True,
            "water_episode_2048_executed_or_claimed": False,
            "complete_three_seed_heldout_cohort_claimed": False,
            "heldout_metrics_used_for_decision": False,
            "fresh_random_baseline_executed_or_claimed": False,
            "raw_artifacts_modified_by_finalizer": False,
            "runner_or_core_modified_by_finalizer": False,
        },
        "historical_binding": {
            "manifest": {
                **file_binding(MANIFEST),
                "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
            },
            "parent_RC_reject_authority": {
                **file_binding(PARENT_RC_AUTHORITY),
                "authority_fingerprint": parent_fingerprint,
            },
            "spectral_offline_design_authority": {
                **file_binding(OFFLINE_AUTHORITY),
                "authority_fingerprint": offline_fingerprint,
            },
            "source_sha256": manifest["source_sha256"],
            "parent_additive_source_sha256": manifest[
                "parent_additive_source_sha256"
            ],
            "spectral_additive_source_sha256": manifest[
                "spectral_additive_source_sha256"
            ],
            "candidate_orchestration_source_sha256": manifest[
                "candidate_orchestration_source_sha256"
            ],
            "historical_source_binding_semantics": {
                "identity_source": "immutable launch manifest plus matching job/status fingerprints",
                "live_workspace_source_rehash_required": False,
                "reason": (
                    "Later candidates may legitimately update live source files; "
                    "that does not rewrite the completed P0X2RCS05 launch identity."
                ),
            },
            "model_sha256_by_task": {
                task: manifest["model_sha256_by_task"][task]
                for task in TASK_PROFILE
            },
            "candidate_reward_sha256": manifest["candidate_reward_sha256"],
            "parent_candidate_reward_sha256": manifest[
                "parent_candidate_reward_sha256"
            ],
            "candidate_optimizer_sha256_by_profile": manifest[
                "candidate_optimizer_sha256_by_profile"
            ],
            "parent_optimizer_sha256_by_profile": manifest[
                "parent_candidate_optimizer_sha256_by_profile"
            ],
            "candidate_mechanics_sha256_by_task": manifest[
                "candidate_mechanics_sha256_by_task"
            ],
            "parent_mechanics_sha256_by_task": manifest[
                "parent_candidate_mechanics_sha256_by_task"
            ],
            "candidate_transition_semantics_sha256": manifest[
                "candidate_transition_semantics_sha256"
            ],
            "parent_transition_semantics_sha256": manifest[
                "parent_transition_semantics_sha256"
            ],
            "spectral_formula": manifest["spectral_formula"],
            "spectral_formula_sha256": EXPECTED_FORMULA_SHA256,
            "spectral_weight": EXPECTED_SPECTRAL_WEIGHT,
            "spectral_relative_epsilon": EXPECTED_RELATIVE_EPSILON,
            "finalizer": file_binding(Path(__file__).resolve()),
        },
        "preregistered_gates": gate_rows,
        "atomic_endpoint_integrity": {
            "pass": all(
                row["reward_and_transition_integrity"]["pass"]
                for row in endpoint_rows
            ),
            "per_seed": endpoint_rows,
        },
        "reward_transition_and_spectral_integrity": {
            "all_six_pass": all(
                row["reward_and_transition_integrity"]["pass"]
                for row in endpoint_rows
            ),
            "spectral_identity": (
                "total=progress+rollback=cumulative; retained cumulative="
                "lambda*terminal_Phi; configured failures roll back cumulative to zero"
            ),
            "per_seed": [
                {
                    "task": row["task"],
                    "seed": row["seed"],
                    **row["reward_and_transition_integrity"],
                }
                for row in endpoint_rows
            ],
        },
        "endpoint_signal_metrics": endpoint_metrics,
        "parent_RC_common_prefix_comparison": parent_comparisons,
        "checkpoint_1024_parent_RC_common_prefix_recomputation": gate_parent_comparisons,
        "parent_comparison_is_diagnostic_not_a_gate": True,
        "water_episode22_strict_success": strict_evidence,
        "critic_clipping_diagnostics": {
            "interpretation": (
                "Rates count logged pre-clipping critic gradient norms greater "
                "than the configured threshold; they diagnose critic clipping, "
                "not actor entropy collapse."
            ),
            "per_seed": critic_rows,
        },
        "execution_disposition": {
            "all_runner_and_child_statuses_terminal": True,
            "all_six_stderr_empty": True,
            "runner_stdout": file_binding(RUNNER_STDOUT),
            "runner_stderr": {**file_binding(RUNNER_STDERR), "empty": True},
            "torus_statuses_by_seed": {
                "3201": "complete",
                "3202": "complete",
                "3203": "early_stopped_at_checkpoint",
            },
            "water_statuses_by_seed": {
                str(seed): "early_stopped_at_checkpoint"
                for seed in EXPECTED_SEEDS
            },
            "completed_seed_retention_exercised": True,
            "completed_seed_retention_evidence": (
                "torus seeds3201 and 3202 finalized before seed3203; the atomic "
                "2048 gate still contained all three preregistered seeds"
            ),
        },
        "negative_scope_evidence": {
            "water_checkpoint_2048_path": relative(water_2048),
            "water_checkpoint_2048_absent": True,
            "water_endpoint_episode": 1024,
            "heldout": {
                "complete_three_seed_heldout_cohort": False,
                "heldout_metrics_inspected_or_used_by_this_report": False,
                "note": (
                    "Fast torus seeds3201/3202 finalized and therefore have "
                    "finalization artifacts, but seed3203 did not; no complete "
                    "fixed-cohort heldout inference is made."
                ),
            },
            "fresh_random_baseline": {
                "executed_by_candidate_jobs": False,
                "claimed": False,
                "all_manifest_jobs_random_search_false": all(
                    job["expected_config"]["random_search"] is False
                    for job in jobs.values()
                ),
            },
        },
        "fresh_confirmation_requirements": [
            "fresh preregistered confirmation seeds",
            "fresh random baseline on all five tasks",
            "fresh successful-task regression on five_d2, five_d3, and watersqrt2",
            "a separate authorization decision before any 10000-episode training",
        ],
        "conclusion": {
            "formal_screen": "rejected",
            "signal_assessment": "promising_for_regression_not_adoption",
            "water": (
                "A strict episode-22 path is new versus the paired RC parent, "
                "but only one seed supplied constructive evidence by episode 1024."
            ),
            "torus": (
                "All three seeds supplied constructive and joint evidence by "
                "episode 2048, but none achieved strict success."
            ),
        },
        "limitations": [
            "The preregistered screen rejected; promising-signal wording is not a gate override.",
            "The water strict hit occurred once at episode 22 and did not establish multi-seed learning reliability.",
            "The rejected RC torus parent ends at episode 512, so later torus comparison has no paired parent endpoint.",
            "No water episode-2048 result exists.",
            "No complete three-seed heldout cohort or fresh random-baseline result is claimed.",
            "Critic clipping diagnostics are descriptive and do not by themselves identify a causal failure mechanism.",
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


def atomic_write_new(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(
        not path.exists(),
        f"refusing to overwrite existing authority: {relative(path)}",
    )
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
        require(temporary is not None, "could not reserve authority temp file")
        require(
            not path.exists(),
            f"authority appeared during finalization: {relative(path)}",
        )
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
        help="recompute raw evidence and verify the existing authority",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        require(OUTPUT.is_file(), f"authority is absent: {relative(OUTPUT)}")
        stored = load_json(OUTPUT)
        require(isinstance(stored, dict), "stored authority is not an object")
        verify_self_digest(stored, "spectral screen")
        generated_utc = stored.get("generated_utc")
        require(isinstance(generated_utc, str), "stored generated_utc absent")
        rebuilt = build_report(generated_utc)
        require(
            canonical_bytes(rebuilt) == canonical_bytes(stored),
            "stored authority differs from rebuilt raw evidence",
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "path": relative(OUTPUT),
                    "sha256": file_sha256(OUTPUT),
                    "authority_fingerprint": stored["authority_fingerprint"],
                },
                sort_keys=True,
            )
        )
        return 0

    require(
        not OUTPUT.exists(),
        f"refusing to overwrite existing authority: {relative(OUTPUT)}",
    )
    report = build_report(datetime.now(timezone.utc).isoformat())
    verify_self_digest(report, "spectral screen")
    atomic_write_new(OUTPUT, report)
    print(
        json.dumps(
            {
                "status": "created",
                "path": relative(OUTPUT),
                "sha256": file_sha256(OUTPUT),
                "authority_fingerprint": report["authority_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
