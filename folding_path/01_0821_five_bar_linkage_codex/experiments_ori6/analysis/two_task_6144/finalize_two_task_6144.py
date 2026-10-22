"""Audit and plot the externally stopped ori6 two-task run at episode 6144.

This is a read-only analyzer with respect to training artifacts.  It writes
only beside itself, under experiments_ori6/analysis/two_task_6144.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
ENDPOINT = 6144
ROLLING_WINDOW = 200
MANIFEST = (
    REPO_ROOT
    / "experiments_ori6/manifests/"
    "spectral_production_two_task_full_diagnostic_d2d0b956eef9.json"
)
MANIFEST_SHA256 = (
    "f46eff732bd96687d2ee4d033de4364db1a24b3dd5ce9695d279c54e3d6b2d9f"
)
MANIFEST_FINGERPRINT = (
    "d2d0b956eef9dce6b2b5a19730624a588377f4178066d90fb4eea718af215c00"
)
FORMULA_SHA256 = (
    "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6"
)
CANDIDATE = (
    "state_potential_crossing_penalty_2_rejected_crossing_continuation_"
    "spectral_subspace_0p05"
)

TASKS = {
    "torus2": {
        "label": "Torus D=2",
        "profile": "A",
        "artifact": REPO_ROOT
        / "experiments_ori6/full/two_task_full_diagnostic/P0X2RCS05/"
        "torus2/A/seed4101",
        "model": REPO_ROOT / "model/4unit_torus",
        "status": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/torus2/ori6/seed4101/5f78bd1ba0e5/status.json",
        "stderr": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/torus2/ori6/seed4101/5f78bd1ba0e5/stderr.log",
        "stdout": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/torus2/ori6/seed4101/5f78bd1ba0e5/stdout.log",
        "color": "#0072B2",
    },
    "water1": {
        "label": "Waterbomb D=1",
        "profile": "B",
        "artifact": REPO_ROOT
        / "experiments_ori6/full/two_task_full_diagnostic/P0X2RCS05/"
        "water1/B/seed4101",
        "model": REPO_ROOT / "model/waterbomb",
        "status": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/water1/ori6/seed4101/36063a7486c1/status.json",
        "stderr": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/water1/ori6/seed4101/36063a7486c1/stderr.log",
        "stdout": REPO_ROOT
        / "experiments_ori6/_runner_logs/two_task_full_diagnostic/"
        "P0X2RCS05/water1/ori6/seed4101/36063a7486c1/stdout.log",
        "color": "#D55E00",
    },
}

REQUIRED_COLUMNS = {
    "episode",
    "total_reward",
    "flatness_ratio",
    "spread_error",
    "strict_compact_success",
    "termination_reason",
    "steps",
    "projection_attempts",
    "success",
    "self_crossing",
    "correction_failed",
    "no_modes",
    "wrong_flat",
    "timeout",
    "near_miss_qF",
    "near_miss_qD",
    "near_miss_q",
    "near_miss_bonus",
    "near_miss_bonus_awarded",
    "ori6_joint_milestones_awarded",
    "ori6_total_discovery_reward",
    "ori6_total_rejected_crossing_reward",
    "ori6_rejected_crossing_count",
    "ori6_rejected_crossing_streak_count",
    "ori6_escape_after_rejected_crossing_count",
    "ori6_spectral_total_progress_reward",
    "ori6_spectral_total_rollback_reward",
    "ori6_spectral_total_reward",
    "ori6_spectral_maximum_potential",
    "ori6_reward_decomposition_error",
    "ori6_nonfinite_reward_component_count",
    "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
    "ori6_spectral_reward_decomposition_error",
    "ori6_spectral_nonfinite_component_count",
    "ori6_spectral_formula_sha256",
    "ori6_spectral_weight",
    "ori6_candidate",
    "ori6_crossing_policy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-14))
    return left == right


def expected_subset_equal(expected: Any, observed: Any) -> bool:
    """Allow run_config to carry descriptive keys added around manifest data."""

    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(
            key in observed and expected_subset_equal(value, observed[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(observed, list) and len(expected) == len(observed) and all(
            expected_subset_equal(left, right)
            for left, right in zip(expected, observed)
        )
    return values_equal(expected, observed)


def direct_config_check(expected: dict[str, Any], observed: dict[str, Any]) -> dict:
    """Compare every manifest field that is directly represented in run_config."""

    compared: dict[str, bool] = {}
    absent_metadata: list[str] = []
    for key, value in expected.items():
        if key in observed:
            compared[key] = expected_subset_equal(value, observed[key])
        else:
            absent_metadata.append(key)
    mismatches = [key for key, okay in compared.items() if not okay]
    return {
        "pass": not mismatches,
        "direct_field_count": len(compared),
        "direct_fields_all_match": not mismatches,
        "mismatches": mismatches,
        "manifest_only_metadata_fields": absent_metadata,
    }


def source_and_model_audit(manifest: dict[str, Any]) -> dict:
    source_checks: dict[str, dict[str, Any]] = {}
    for section in (
        "base_source_sha256",
        "candidate_additive_source_sha256",
        "production_source_sha256",
    ):
        source_checks[section] = {}
        for name, expected in manifest[section].items():
            path = REPO_ROOT / name
            observed = sha256(path) if path.is_file() else None
            source_checks[section][name] = {
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "pass": observed == expected,
            }

    model_checks: dict[str, Any] = {}
    names = {"node": "node_initial.dat", "member": "member.dat", "fix": "fix.dat"}
    for task, spec in TASKS.items():
        model_checks[task] = {}
        for logical, filename in names.items():
            path = spec["model"] / filename
            expected = manifest["model_sha256_by_task"][task][logical]
            observed = sha256(path) if path.is_file() else None
            model_checks[task][logical] = {
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "pass": observed == expected,
            }

    all_source = all(
        row["pass"]
        for section in source_checks.values()
        for row in section.values()
    )
    all_model = all(
        row["pass"]
        for task in model_checks.values()
        for row in task.values()
    )
    return {
        "all_source_hashes_match": all_source,
        "all_model_hashes_match": all_model,
        "source": source_checks,
        "model": model_checks,
    }


def active_training_processes() -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    error = None
    try:
        import psutil

        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if (
                "ori_codex6_spectral_subspace_3d.py" in command
                and "two_task_full_diagnostic" in command
                and "seed4101" in command
            ):
                matches.append({"pid": process.info["pid"], "command": command})
    except Exception as exception:  # pragma: no cover - environment diagnostic
        error = repr(exception)
        # psutil is not a project dependency.  On Windows, use a read-only WMI
        # query restricted to Python executables, so the PowerShell query does
        # not match its own command line.
        try:
            command = (
                "$rows = Get-CimInstance Win32_Process | "
                "Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq "
                "'pythonw.exe') -and $_.CommandLine -match "
                "'ori_codex6_spectral_subspace_3d.py' -and "
                "$_.CommandLine -match 'two_task_full_diagnostic' -and "
                "$_.CommandLine -match 'seed4101' }; "
                "$rows | Select-Object ProcessId,CommandLine | "
                "ConvertTo-Json -Compress"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            payload = completed.stdout.strip()
            decoded = json.loads(payload) if payload else []
            if isinstance(decoded, dict):
                decoded = [decoded]
            matches = [
                {
                    "pid": int(row["ProcessId"]),
                    "command": row.get("CommandLine", ""),
                }
                for row in decoded
            ]
            error = None
        except Exception as fallback_exception:
            error = f"{error}; fallback={fallback_exception!r}"
    return {
        "checked": error is None,
        "matching_process_count": len(matches),
        "matching_processes": matches,
        "error": error,
        "pass": error is None and not matches,
    }


def rolling(series: pd.Series, operation: str = "mean") -> pd.Series:
    window = series.rolling(ROLLING_WINDOW, min_periods=20)
    return getattr(window, operation)()


def rolling_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = numerator.rolling(ROLLING_WINDOW, min_periods=20).sum()
    bottom = denominator.rolling(ROLLING_WINDOW, min_periods=20).sum()
    return top.div(bottom.where(bottom > 0))


def enrich(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["mechanics"] = frame["steps"].cumsum()
    frame["cumulative_mean_return"] = frame["total_reward"].expanding().mean()
    frame["constructive_proxy"] = (
        frame["flatness_ratio"] + frame["spread_error"] ** 2 <= 0.5
    ).astype(float)
    frame["valid_joint"] = (
        (
            (frame["flatness_ratio"] <= 0.1)
            & (frame["spread_error"] <= 0.05)
        )
        | (frame["strict_compact_success"] == 1)
    ).astype(float)
    frame["non_timeout_termination"] = (frame["termination_reason"] != "timeout").astype(float)
    frame["rejected_action_rate_roll"] = rolling_ratio(
        frame["ori6_rejected_crossing_count"], frame["steps"]
    )
    frame["escape_rate_roll"] = rolling_ratio(
        frame["ori6_escape_after_rejected_crossing_count"],
        frame["ori6_rejected_crossing_streak_count"],
    )
    return frame


def window_metrics(frame: pd.DataFrame, start: int, end: int) -> dict[str, Any]:
    part = frame[(frame["episode"] >= start) & (frame["episode"] <= end)]
    streaks = int(part["ori6_rejected_crossing_streak_count"].sum())
    return {
        "episode_start": start,
        "episode_end": end,
        "episodes": int(len(part)),
        "mean_return": float(part["total_reward"].mean()),
        "mean_terminal_flatness_ratio": float(part["flatness_ratio"].mean()),
        "median_terminal_flatness_ratio": float(part["flatness_ratio"].median()),
        "mean_spread_error": float(part["spread_error"].mean()),
        "median_spread_error": float(part["spread_error"].median()),
        "strict_successes": int(part["strict_compact_success"].sum()),
        "timeouts": int(part["timeout"].sum()),
        "wrong_flat": int(part["wrong_flat"].sum()),
        "correction_failed": int(part["correction_failed"].sum()),
        "constructive_proxy_episodes": int(part["constructive_proxy"].sum()),
        "valid_joint_episodes": int(part["valid_joint"].sum()),
        "positive_near_miss_episodes": int((part["near_miss_q"] > 0).sum()),
        "rejected_actions": int(part["ori6_rejected_crossing_count"].sum()),
        "rejected_streaks": streaks,
        "escapes_after_rejected_crossing": int(
            part["ori6_escape_after_rejected_crossing_count"].sum()
        ),
        "escape_rate": (
            float(part["ori6_escape_after_rejected_crossing_count"].sum() / streaks)
            if streaks
            else None
        ),
        "mean_spectral_reward": float(part["ori6_spectral_total_reward"].mean()),
        "mean_spectral_peak_potential": float(
            part["ori6_spectral_maximum_potential"].mean()
        ),
        "mechanical_steps": int(part["steps"].sum()),
    }


def row_subset(row: pd.Series) -> dict[str, Any]:
    names = [
        "episode",
        "total_reward",
        "flatness_ratio",
        "spread_error",
        "termination_reason",
        "steps",
        "near_miss_qF",
        "near_miss_qD",
        "near_miss_q",
        "near_miss_bonus",
        "ori6_joint_milestones_awarded",
        "ori6_total_flatness_progress_reward",
        "ori6_total_target_progress_reward",
        "ori6_total_spread_frontier_reward",
        "ori6_total_joint_milestone_reward",
        "ori6_total_terminal_fixed_reward",
        "ori6_total_terminal_quality_reward",
        "ori6_rejected_crossing_count",
        "ori6_rejected_crossing_streak_count",
        "ori6_escape_after_rejected_crossing_count",
        "ori6_total_rejected_crossing_reward",
        "ori6_spectral_total_reward",
        "ori6_spectral_terminal_potential",
        "ori6_spectral_maximum_potential",
    ]
    payload: dict[str, Any] = {}
    for name in names:
        value = row[name]
        if isinstance(value, (np.integer, int)):
            payload[name] = int(value)
        elif isinstance(value, (np.floating, float)):
            payload[name] = float(value)
        else:
            payload[name] = str(value)
    payload["counterfactual_return_without_rejected_crossing_reward"] = float(
        row["total_reward"] - row["ori6_total_rejected_crossing_reward"]
    )
    return payload


def audit_task(task: str, spec: dict[str, Any], job: dict[str, Any]) -> tuple[dict, pd.DataFrame]:
    artifact = spec["artifact"]
    paths = {
        name: artifact / name
        for name in (
            "run_config.json",
            "training_history.csv",
            "update_history.csv",
            "deterministic_evaluation_history.csv",
            "stochastic_evaluation_history.csv",
            "checkpoint.pt",
            "best_policy.pt",
            "best_policy_metrics.json",
        )
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{task}: missing atomic artifact files: {missing}")

    run_config = load_json(paths["run_config.json"])
    history_raw = pd.read_csv(paths["training_history.csv"])
    updates = pd.read_csv(paths["update_history.csv"])
    deterministic = pd.read_csv(paths["deterministic_evaluation_history.csv"])
    stochastic = pd.read_csv(paths["stochastic_evaluation_history.csv"])
    history = enrich(history_raw)
    checkpoint = torch.load(paths["checkpoint.pt"], map_location="cpu", weights_only=False)
    status = load_json(spec["status"])
    stderr_bytes = spec["stderr"].stat().st_size
    stdout_text = spec["stdout"].read_text(encoding="utf-8", errors="replace")
    printed = [int(value) for value in re.findall(r"Episode\s+(\d+)\s+\|", stdout_text)]

    errors: list[str] = []
    if len(history) != ENDPOINT:
        errors.append("training_history_row_count")
    if history["episode"].tolist() != list(range(1, ENDPOINT + 1)):
        errors.append("training_episode_sequence")
    if set(history.columns).isdisjoint(REQUIRED_COLUMNS) or not REQUIRED_COLUMNS.issubset(
        history.columns
    ):
        errors.append("required_training_columns")
    if len(updates) != ENDPOINT // int(run_config["batch_size"]):
        errors.append("update_history_row_count")
    expected_update_episodes = list(
        range(int(run_config["batch_size"]), ENDPOINT + 1, int(run_config["batch_size"]))
    )
    if updates["episode"].tolist() != expected_update_episodes:
        errors.append("update_episode_sequence")
    expected_eval_episodes = [1] + list(range(500, 6001, 500))
    if deterministic["episode"].tolist() != expected_eval_episodes:
        errors.append("deterministic_evaluation_sequence")
    if stochastic["episode"].tolist() != expected_eval_episodes:
        errors.append("stochastic_evaluation_sequence")
    if checkpoint.get("episode") != ENDPOINT:
        errors.append("checkpoint_episode")
    if checkpoint.get("config") != run_config:
        errors.append("checkpoint_config_identity")

    config_audit = direct_config_check(job["expected_config"], run_config)
    if not config_audit["pass"]:
        errors.append("manifest_run_config_direct_fields")
    critical_identity = {
        "seed": run_config.get("seed") == 4101,
        "candidate_mode": job.get("candidate_mode") == CANDIDATE,
        "spectral_enabled": run_config.get("ori6_spectral_subspace") is True,
        "spectral_weight": values_equal(run_config.get("ori6_spectral_weight"), 0.05),
        "spectral_formula": bool(
            (history["ori6_spectral_formula_sha256"] == FORMULA_SHA256).all()
        ),
        "rejected_continuation": run_config.get("ori6_rejected_crossing_continuation")
        is True,
        "crossing_policy": run_config.get("ori6_crossing_policy")
        == "reject_candidate_and_continue",
        "history_candidate": bool(
            (history["ori6_candidate"] == "ori6_P0X2_rejected_crossing_continuation").all()
        ),
    }
    if not all(critical_identity.values()):
        errors.append("candidate_identity")

    numeric_columns = [
        "total_reward",
        "flatness_ratio",
        "spread_error",
        "near_miss_q",
        "ori6_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
    ]
    if not np.isfinite(history[numeric_columns].to_numpy(dtype=float)).all():
        errors.append("nonfinite_core_history")
    reward_integrity = {
        "maximum_abs_ori6_reward_decomposition_error": float(
            history["ori6_reward_decomposition_error"].abs().max()
        ),
        "maximum_abs_rejected_reward_decomposition_error": float(
            history[
                "ori6_maximum_abs_rejected_crossing_reward_decomposition_error"
            ].abs().max()
        ),
        "maximum_abs_spectral_reward_decomposition_error": float(
            history["ori6_spectral_reward_decomposition_error"].abs().max()
        ),
        "ori6_nonfinite_reward_component_count": int(
            history["ori6_nonfinite_reward_component_count"].sum()
        ),
        "spectral_nonfinite_component_count": int(
            history["ori6_spectral_nonfinite_component_count"].sum()
        ),
        "maximum_abs_spectral_cumulative_identity_error": float(
            (
                history["ori6_spectral_total_reward"]
                - history["ori6_spectral_total_progress_reward"]
                - history["ori6_spectral_total_rollback_reward"]
            ).abs().max()
        ),
    }
    reward_integrity["pass"] = bool(
        reward_integrity["maximum_abs_ori6_reward_decomposition_error"] <= 1e-10
        and reward_integrity["maximum_abs_rejected_reward_decomposition_error"] <= 1e-10
        and reward_integrity["maximum_abs_spectral_reward_decomposition_error"] <= 1e-10
        and reward_integrity["maximum_abs_spectral_cumulative_identity_error"] <= 1e-10
        and reward_integrity["ori6_nonfinite_reward_component_count"] == 0
        and reward_integrity["spectral_nonfinite_component_count"] == 0
    )
    if not reward_integrity["pass"]:
        errors.append("reward_rc_spectral_integrity")

    flag_sum = history[
        ["success", "correction_failed", "no_modes", "wrong_flat", "timeout"]
    ].sum(axis=1)
    if not bool((flag_sum == 1).all()):
        errors.append("termination_one_hot")
    reason_to_flag = {
        "success": "success",
        "correction_failed": "correction_failed",
        "no_modes": "no_modes",
        "wrong_flat": "wrong_flat",
        "timeout": "timeout",
    }
    for reason, flag in reason_to_flag.items():
        selected = history["termination_reason"] == reason
        if selected.any() and not bool((history.loc[selected, flag] == 1).all()):
            errors.append(f"termination_reason_{reason}")
    if bool((history["self_crossing"] != 0).any()):
        errors.append("self_crossing_should_be_rejected_not_terminal")

    external_stop = {
        "runner_status": status.get("status"),
        "return_code": status.get("return_code"),
        "stderr_bytes": stderr_bytes,
        "stderr_empty": stderr_bytes == 0,
        "stdout_max_printed_episode": max(printed) if printed else None,
        "requested_episodes": run_config.get("episodes"),
        "atomic_checkpoint_episode": checkpoint.get("episode"),
        "missing_finalization_artifacts_recorded": status.get("artifact_validation", {}).get(
            "errors", []
        ),
        "classification": "externally_stopped_incomplete_run_with_valid_atomic_checkpoint",
    }
    external_stop["pass"] = bool(
        status.get("status") == "failed"
        and status.get("return_code") == 4294967295
        and stderr_bytes == 0
        and checkpoint.get("episode") == ENDPOINT
        and (max(printed) if printed else 0) >= ENDPOINT
        and not any("Traceback" in line for line in stdout_text.splitlines())
    )
    if not external_stop["pass"]:
        errors.append("external_stop_evidence")

    term_counts = {
        str(key): int(value)
        for key, value in history["termination_reason"].value_counts().items()
    }
    total_streaks = int(history["ori6_rejected_crossing_streak_count"].sum())
    best_cost_index = (
        history["flatness_ratio"] + history["spread_error"] ** 2
    ).idxmin()
    best_joint = row_subset(history.loc[best_cost_index])
    best_joint["joint_cost_F_plus_E_squared"] = float(
        history.loc[best_cost_index, "flatness_ratio"]
        + history.loc[best_cost_index, "spread_error"] ** 2
    )
    constructive_mask = history["constructive_proxy"].astype(bool)
    counterfactual_return = (
        history["total_reward"]
        - history["ori6_total_rejected_crossing_reward"]
    )

    def return_cohort(mask: pd.Series) -> dict[str, Any]:
        cohort = history.loc[mask]
        return {
            "episodes": int(mask.sum()),
            "mean_actual_return": (
                float(cohort["total_reward"].mean()) if len(cohort) else None
            ),
            "mean_counterfactual_return_without_rejected_crossing_reward": (
                float(counterfactual_return.loc[mask].mean()) if len(cohort) else None
            ),
        }

    windows = []
    for start in range(1, ENDPOINT + 1, 512):
        windows.append(window_metrics(history, start, min(start + 511, ENDPOINT)))

    metrics: dict[str, Any] = {
        "episodes": len(history),
        "mechanical_steps": int(history["steps"].sum()),
        "projector_calls": int(history["projection_attempts"].sum()),
        "policy_updates": len(updates),
        "termination_counts": term_counts,
        "strict_successes": int(history["strict_compact_success"].sum()),
        "constructive_proxy_definition": "flatness_ratio + spread_error^2 <= 0.5",
        "constructive_proxy_episodes": int(history["constructive_proxy"].sum()),
        "valid_joint_definition": "F <= 0.1 and spread_error <= 0.05, or strict success",
        "valid_joint_episodes": int(history["valid_joint"].sum()),
        "positive_near_miss_episodes": int((history["near_miss_q"] > 0).sum()),
        "maximum_near_miss_q": float(history["near_miss_q"].max()),
        "minimum_terminal_flatness_ratio": float(history["flatness_ratio"].min()),
        "minimum_spread_error": float(history["spread_error"].min()),
        "mean_return": float(history["total_reward"].mean()),
        "total_return": float(history["total_reward"].sum()),
        "counterfactual_mean_return_without_rejected_crossing_reward": float(
            (
                history["total_reward"]
                - history["ori6_total_rejected_crossing_reward"]
            ).mean()
        ),
        "constructive_vs_nonconstructive_return_ordering": {
            "constructive": return_cohort(constructive_mask),
            "nonconstructive": return_cohort(~constructive_mask),
            "actual_mean_difference_constructive_minus_nonconstructive": float(
                history.loc[constructive_mask, "total_reward"].mean()
                - history.loc[~constructive_mask, "total_reward"].mean()
            ),
            "interpretation_caveat": (
                "Episode-level cohort ordering is descriptive and does not prove "
                "per-action advantage or causal policy-update behavior."
            ),
        },
        "reward_decomposition_totals": {
            "flatness_progress": float(
                history["ori6_total_flatness_progress_reward"].sum()
            ),
            "target_progress": float(
                history["ori6_total_target_progress_reward"].sum()
            ),
            "spread_frontier": float(
                history["ori6_total_spread_frontier_reward"].sum()
            ),
            "joint_milestone": float(
                history["ori6_total_joint_milestone_reward"].sum()
            ),
            "discovery": float(history["ori6_total_discovery_reward"].sum()),
            "terminal_fixed": float(
                history["ori6_total_terminal_fixed_reward"].sum()
            ),
            "terminal_quality": float(
                history["ori6_total_terminal_quality_reward"].sum()
            ),
            "rejected_crossing": float(
                history["ori6_total_rejected_crossing_reward"].sum()
            ),
            "spectral": float(history["ori6_spectral_total_reward"].sum()),
        },
        "rejected_crossing": {
            "rejected_actions": int(history["ori6_rejected_crossing_count"].sum()),
            "rejected_streaks": total_streaks,
            "escapes": int(
                history["ori6_escape_after_rejected_crossing_count"].sum()
            ),
            "escape_rate": (
                float(
                    history["ori6_escape_after_rejected_crossing_count"].sum()
                    / total_streaks
                )
                if total_streaks
                else None
            ),
            "rejected_action_rate": float(
                history["ori6_rejected_crossing_count"].sum()
                / history["steps"].sum()
            ),
        },
        "first_200": window_metrics(history, 1, 200),
        "last_200": window_metrics(history, ENDPOINT - 199, ENDPOINT),
        "checkpoint_windows_512": windows,
        "best_joint_endpoint": best_joint,
    }

    if task == "water1":
        episode5 = row_subset(history.loc[history["episode"] == 5].iloc[0])
        first8 = history[history["episode"] <= 8]
        episode5["return_rank_ascending_within_first_8"] = int(
            first8["total_reward"].rank(method="min", ascending=True).loc[
                first8["episode"] == 5
            ].iloc[0]
        )
        first_update = updates.iloc[0]
        metrics["episode5_high_quality_near_miss"] = episode5
        metrics["first_actor_critic_update_episode8"] = {
            name: float(first_update[name])
            for name in (
                "loss",
                "actor_loss",
                "critic_loss",
                "entropy",
                "actor_gradient_norm",
                "critic_gradient_norm",
                "actor_learning_rate",
                "critic_learning_rate",
            )
        }
        metrics["rejected_actions_after_episode1024"] = int(
            history.loc[history["episode"] > 1024, "ori6_rejected_crossing_count"].sum()
        )
    else:
        metrics["constructive_window_1025_1536"] = window_metrics(
            history, 1025, 1536
        )
        metrics["post_constructive_window_1537_6144"] = window_metrics(
            history, 1537, 6144
        )
        metrics["rejected_actions_after_episode1536"] = int(
            history.loc[history["episode"] > 1536, "ori6_rejected_crossing_count"].sum()
        )

    file_hashes = {
        name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    audit = {
        "pass": not errors,
        "errors": errors,
        "task": task,
        "label": spec["label"],
        "artifact_directory": str(artifact),
        "file_hashes": file_hashes,
        "config": config_audit,
        "critical_identity": critical_identity,
        "checkpoint": {
            "episode": checkpoint.get("episode"),
            "config_byte_semantic_identity": checkpoint.get("config") == run_config,
            "actor_learning_rate": checkpoint.get("actor_learning_rate"),
            "critic_learning_rate": checkpoint.get("critic_learning_rate"),
            "model_state_present": "model_state_dict" in checkpoint,
            "actor_optimizer_state_present": "actor_optimizer_state_dict" in checkpoint,
            "critic_optimizer_state_present": "critic_optimizer_state_dict" in checkpoint,
            "torch_rng_state_present": "torch_rng_state" in checkpoint,
            "numpy_rng_state_present": "numpy_rng_state" in checkpoint,
        },
        "history": {
            "training_rows": len(history),
            "update_rows": len(updates),
            "deterministic_evaluation_rows": len(deterministic),
            "stochastic_evaluation_rows": len(stochastic),
            "episode_sequence_contiguous_1_to_6144": history["episode"].tolist()
            == list(range(1, ENDPOINT + 1)),
            "update_sequence_exact": updates["episode"].tolist()
            == expected_update_episodes,
            "evaluation_sequence_exact": deterministic["episode"].tolist()
            == expected_eval_episodes
            and stochastic["episode"].tolist() == expected_eval_episodes,
            "termination_one_hot": bool((flag_sum == 1).all()),
        },
        "reward_rc_spectral_integrity": reward_integrity,
        "external_stop": external_stop,
        "metrics": metrics,
    }
    return audit, history


def apply_common_axis_style(axis: plt.Axes, xlabel: str, x_max: float) -> None:
    axis.set_xlim(1 if xlabel == "Episode" else 0, x_max)
    axis.set_xlabel(xlabel)
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def shade_context(axis: plt.Axes, frame: pd.DataFrame, x_name: str, task: str) -> None:
    if task == "torus2":
        if x_name == "episode":
            left, right = 1025, 1536
        else:
            left = float(frame.loc[frame["episode"] == 1025, "mechanics"].iloc[0])
            right = float(frame.loc[frame["episode"] == 1536, "mechanics"].iloc[0])
        axis.axvspan(left, right, color="#009E73", alpha=0.09, zorder=0)


def plot_task(task: str, frame: pd.DataFrame) -> Path:
    label = TASKS[task]["label"]
    figure, axes = plt.subplots(6, 2, figsize=(17, 22), sharex="col")
    figure.suptitle(
        f"ori6 P0X2RCS05 — {label} — seed 4101 — exact atomic endpoint 6144",
        fontsize=17,
        y=0.997,
    )
    axis_sets = [("episode", "Episode"), ("mechanics", "Cumulative mechanical steps")]
    palette = {
        "main": "#0072B2",
        "secondary": "#D55E00",
        "green": "#009E73",
        "purple": "#CC79A7",
        "gold": "#E69F00",
        "gray": "#666666",
    }
    for column, (x_name, xlabel) in enumerate(axis_sets):
        x = frame[x_name]
        x_max = ENDPOINT if x_name == "episode" else float(x.iloc[-1])
        for axis in axes[:, column]:
            shade_context(axis, frame, x_name, task)
            apply_common_axis_style(axis, xlabel, x_max)

        axis = axes[0, column]
        axis.plot(x, rolling(frame["total_reward"]), color=palette["main"], lw=1.6, label="return: rolling mean (200 ep)")
        axis.plot(x, frame["cumulative_mean_return"], color=palette["secondary"], lw=1.6, ls="--", label="return: cumulative mean")
        axis.axhline(0, color="black", lw=0.7, alpha=0.45)
        axis.set_ylabel("Episodic return")
        axis.set_title("Reward learning curve")
        axis.legend(fontsize=8, loc="best")

        axis = axes[1, column]
        axis.plot(x, rolling(frame["flatness_ratio"], "mean").clip(lower=1e-8), color=palette["main"], lw=1.6, label="rolling mean")
        axis.plot(x, rolling(frame["flatness_ratio"], "median").clip(lower=1e-8), color=palette["secondary"], lw=1.4, ls="--", label="rolling median")
        axis.axhline(0.001, color="black", lw=0.8, ls=":", label="strict F tolerance")
        axis.set_yscale("log")
        axis.set_ylabel("Terminal flatness F")
        axis.set_title("Terminal flatness (log scale)")
        axis.legend(fontsize=8, loc="best")

        axis = axes[2, column]
        axis.plot(x, rolling(frame["spread_error"], "mean").clip(lower=1e-8), color=palette["main"], lw=1.6, label="rolling mean")
        axis.plot(x, rolling(frame["spread_error"], "median").clip(lower=1e-8), color=palette["secondary"], lw=1.4, ls="--", label="rolling median")
        axis.axhline(0.05, color="black", lw=0.8, ls=":", label="spread tolerance")
        axis.set_yscale("log")
        axis.set_ylabel("Terminal spread error")
        axis.set_title("Terminal spread error (log scale)")
        axis.legend(fontsize=8, loc="best")

        axis = axes[3, column]
        for name, color, style, shown in (
            ("strict_compact_success", palette["green"], "-", "strict success"),
            ("timeout", palette["main"], "-", "timeout"),
            ("wrong_flat", palette["secondary"], "--", "wrong-flat"),
            ("correction_failed", palette["purple"], ":", "correction failure"),
        ):
            axis.plot(x, rolling(frame[name].astype(float)), color=color, ls=style, lw=1.35, label=shown)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Rolling rate")
        axis.set_title("Strict success and termination rates (200 ep)")
        axis.legend(fontsize=8, ncol=2, loc="best")

        axis = axes[4, column]
        axis.plot(x, frame["rejected_action_rate_roll"], color=palette["secondary"], lw=1.5, label="rejected actions / steps")
        axis.plot(x, frame["escape_rate_roll"], color=palette["green"], lw=1.5, ls="--", label="escaped / rejected streaks")
        axis.set_ylim(bottom=-0.03)
        axis.set_ylabel("Rolling rate")
        axis.set_title("Rejected crossing and escape (200 ep)")
        axis.legend(fontsize=8, loc="best")

        axis = axes[5, column]
        axis.plot(x, rolling(frame["ori6_spectral_maximum_potential"]), color=palette["purple"], lw=1.4, label="spectral peak potential")
        axis.plot(x, rolling(frame["near_miss_q"]), color=palette["gold"], lw=1.5, label="near-miss q")
        axis.plot(x, rolling(frame["valid_joint"]), color=palette["green"], lw=1.5, ls="--", label="valid joint rate")
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Rolling mean / rate")
        axis.set_title("Spectral, near-miss, and joint signals (200 ep)")
        axis.legend(fontsize=8, loc="best")

        if task == "water1":
            episode5 = frame.loc[frame["episode"] == 5].iloc[0]
            marker_x = float(episode5[x_name])
            axes[1, column].scatter([marker_x], [episode5["flatness_ratio"]], color="#000000", s=28, zorder=5)
            axes[2, column].scatter([marker_x], [episode5["spread_error"]], color="#000000", s=28, zorder=5)

    context_note = (
        "Green shading marks the torus constructive window (episodes 1025–1536)."
        if task == "torus2"
        else "Black markers on the log panels mark the episode-5 high-quality near miss."
    )
    figure.text(
        0.5,
        0.003,
        "Solid reward summary is a rolling mean; the dashed cumulative curve is a cumulative mean, not a cumulative sum. "
        + context_note,
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.015, 1, 0.985))
    path = OUTPUT_DIR / f"learning_curve_{task}_episode_and_mechanics.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_combined(frames: dict[str, pd.DataFrame]) -> Path:
    figure, axes = plt.subplots(6, 2, figsize=(17, 22), sharex="col")
    figure.suptitle(
        "ori6 P0X2RCS05 — two-task learning summary — exact atomic endpoint 6144",
        fontsize=17,
        y=0.997,
    )
    axis_sets = [("episode", "Episode"), ("mechanics", "Cumulative mechanical steps")]
    titles = [
        "Reward learning curve",
        "Terminal flatness (log scale)",
        "Terminal spread error (log scale)",
        "Strict success / non-timeout termination",
        "Rejected crossing / escape",
        "Spectral / near-miss / valid joint",
    ]
    ylabels = [
        "Episodic return",
        "Terminal flatness F",
        "Terminal spread error",
        "Rolling rate",
        "Rolling rate",
        "Rolling mean / rate",
    ]
    for column, (x_name, xlabel) in enumerate(axis_sets):
        x_max = ENDPOINT if x_name == "episode" else max(float(frame[x_name].iloc[-1]) for frame in frames.values())
        for row, axis in enumerate(axes[:, column]):
            apply_common_axis_style(axis, xlabel, x_max)
            axis.set_title(titles[row])
            axis.set_ylabel(ylabels[row])
        # Torus constructive window is defined on the torus mechanics ledger.
        shade_context(axes[0, column], frames["torus2"], x_name, "torus2")
        for row in range(1, 6):
            shade_context(axes[row, column], frames["torus2"], x_name, "torus2")

        for task, frame in frames.items():
            x = frame[x_name]
            color = TASKS[task]["color"]
            label = TASKS[task]["label"]
            axes[0, column].plot(x, rolling(frame["total_reward"]), color=color, lw=1.5, label=f"{label}: rolling")
            axes[0, column].plot(x, frame["cumulative_mean_return"], color=color, lw=1.25, ls="--", label=f"{label}: cumulative mean")
            axes[1, column].plot(x, rolling(frame["flatness_ratio"], "mean").clip(lower=1e-8), color=color, lw=1.5, label=f"{label}: mean")
            axes[1, column].plot(x, rolling(frame["flatness_ratio"], "median").clip(lower=1e-8), color=color, lw=1.15, ls="--", label=f"{label}: median")
            axes[2, column].plot(x, rolling(frame["spread_error"], "mean").clip(lower=1e-8), color=color, lw=1.5, label=f"{label}: mean")
            axes[2, column].plot(x, rolling(frame["spread_error"], "median").clip(lower=1e-8), color=color, lw=1.15, ls="--", label=f"{label}: median")
            axes[3, column].plot(x, rolling(frame["strict_compact_success"].astype(float)), color=color, lw=1.5, label=f"{label}: strict")
            axes[3, column].plot(x, rolling(frame["non_timeout_termination"]), color=color, lw=1.2, ls="--", label=f"{label}: non-timeout")
            axes[4, column].plot(x, frame["rejected_action_rate_roll"], color=color, lw=1.5, label=f"{label}: reject/step")
            axes[4, column].plot(x, frame["escape_rate_roll"], color=color, lw=1.2, ls="--", label=f"{label}: escape/streak")
            axes[5, column].plot(x, rolling(frame["ori6_spectral_maximum_potential"]), color=color, lw=1.25, ls=":", label=f"{label}: spectral")
            axes[5, column].plot(x, rolling(frame["near_miss_q"]), color=color, lw=1.5, label=f"{label}: near-miss")
            axes[5, column].plot(x, rolling(frame["valid_joint"]), color=color, lw=1.15, ls="--", label=f"{label}: joint")

        axes[0, column].axhline(0, color="black", lw=0.7, alpha=0.45)
        axes[1, column].axhline(0.001, color="black", lw=0.8, ls=":")
        axes[2, column].axhline(0.05, color="black", lw=0.8, ls=":")
        axes[1, column].set_yscale("log")
        axes[2, column].set_yscale("log")
        axes[3, column].set_ylim(-0.03, 1.03)
        axes[4, column].set_ylim(bottom=-0.03)
        axes[5, column].set_ylim(-0.03, 1.03)
        for axis in axes[:, column]:
            axis.legend(fontsize=7, ncol=2, loc="best")

    figure.text(
        0.5,
        0.003,
        "All smooth curves use a trailing 200-episode window. Green shading: torus constructive window, episodes 1025–1536. "
        "Cumulative reward line means cumulative mean episodic return, not cumulative sum.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.015, 1, 0.985))
    path = OUTPUT_DIR / "learning_curve_combined_episode_and_mechanics.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def png_audit(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "mode": mode,
                "readable": width >= 2000 and height >= 2000,
            }
        )
    return records


def write_window_csv(audits: dict[str, dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / "checkpoint_windows_512.csv"
    rows = []
    for task, audit in audits.items():
        for row in audit["metrics"]["checkpoint_windows_512"]:
            rows.append({"task": task, **row})
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_json(MANIFEST)
    manifest_content = dict(manifest)
    recorded_fingerprint = manifest_content.pop("manifest_fingerprint", None)
    manifest_content.pop("created_utc", None)
    manifest_audit = {
        "path": str(MANIFEST),
        "expected_sha256": MANIFEST_SHA256,
        "observed_sha256": sha256(MANIFEST),
        "sha256_match": sha256(MANIFEST) == MANIFEST_SHA256,
        "expected_fingerprint": MANIFEST_FINGERPRINT,
        "recorded_fingerprint": recorded_fingerprint,
        "recomputed_fingerprint": canonical_digest(manifest_content),
        "fingerprint_match": recorded_fingerprint
        == MANIFEST_FINGERPRINT
        == canonical_digest(manifest_content),
        "stage": manifest.get("stage"),
        "candidate_mode": manifest.get("candidate_mode"),
        "job_count": len(manifest.get("jobs", [])),
    }
    manifest_audit["pass"] = bool(
        manifest_audit["sha256_match"]
        and manifest_audit["fingerprint_match"]
        and manifest.get("stage") == "two_task_full_diagnostic"
        and manifest.get("candidate_mode") == CANDIDATE
        and len(manifest.get("jobs", [])) == 2
    )

    source_models = source_and_model_audit(manifest)
    jobs = {job["task"]["name"]: job for job in manifest["jobs"]}
    if set(jobs) != set(TASKS):
        raise RuntimeError(f"Unexpected manifest cohort: {set(jobs)}")

    audits: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for task, spec in TASKS.items():
        audit, frame = audit_task(task, spec, jobs[task])
        audits[task] = audit
        frames[task] = frame

    process_audit = active_training_processes()
    png_paths = [plot_task(task, frames[task]) for task in TASKS]
    png_paths.append(plot_combined(frames))
    png_records = png_audit(png_paths)
    csv_path = write_window_csv(audits)

    report = {
        "schema_version": 1,
        "report_type": "ori6_two_task_external_stop_atomic_episode_6144",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "tasks": list(TASKS),
            "seed": 4101,
            "candidate_mode": CANDIDATE,
            "candidate_slug": "P0X2RCS05",
            "spectral_weight": 0.05,
            "atomic_endpoint_episode": ENDPOINT,
            "rolling_window_episodes": ROLLING_WINDOW,
            "training_was_run_by_this_analyzer": False,
        },
        "manifest_integrity": manifest_audit,
        "source_and_model_integrity": source_models,
        "active_process_audit": process_audit,
        "task_audits": audits,
        "plots": png_records,
        "window_csv": {
            "path": str(csv_path),
            "sha256": sha256(csv_path),
            "bytes": csv_path.stat().st_size,
        },
        "overall_integrity_pass": bool(
            manifest_audit["pass"]
            and source_models["all_source_hashes_match"]
            and source_models["all_model_hashes_match"]
            and process_audit["pass"]
            and all(audit["pass"] for audit in audits.values())
            and all(row["readable"] for row in png_records)
        ),
        "interpretation": {
            "strict_successes": {task: audit["metrics"]["strict_successes"] for task, audit in audits.items()},
            "torus": "A constructive/joint phase existed at episodes 1025–1536, then collapsed to the original wide/unflat timeout basin.",
            "water1": "Return rose and terminal F fell, but terminal spread error stayed near 0.40; this is not target-shape learning.",
            "primary_failure_mechanism": "Repeated rejected-crossing penalties reverse the return ordering of rare high-quality paths; water episode 5 is the decisive example.",
            "next_minimal_ablation": "Give nonterminal rejected actions zero reward while retaining rollback and step consumption; apply the ordinary timeout reward once at horizon.",
            "success_guaranteed_by_next_ablation": False,
        },
    }
    report_path = OUTPUT_DIR / "final_metrics_and_integrity.json"
    report["report_fingerprint"] = canonical_digest(report)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "overall_integrity_pass": report["overall_integrity_pass"],
                "plots": [str(path) for path in png_paths],
                "window_csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
