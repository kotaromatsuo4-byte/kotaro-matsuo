"""Fail-closed 21:00 finalizer for the immutable Codex7 staged campaign.

This module is strictly additive.  It never launches or terminates training and
refuses to read a campaign while a Codex7 trainer/runner PID is alive.  The
report is built in a temporary sibling directory and published atomically only
after all integrity and visual checks pass.
"""

from __future__ import annotations

import argparse
import bisect
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Iterable, Sequence


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY / "experiments_ori7" / "C7TC2A"
DEFAULT_OUTPUT = SCRIPT_DIRECTORY / "experiments_ori7" / "analysis" / "C7TC2A_2100"
MANIFEST_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori7" / "manifests"
EXPECTED_MANIFEST_FINGERPRINT = \
    "14af9b73e5c1c8726bab6f39217a1ac2a430d26053299c4ff552df1035b79f47"
EXPECTED_PROTOCOL = "codex7_terminal_crossing_two_arm_staged_to_10000"
FORMAL_ENDPOINT = 4096
ROLLING_WINDOW = 256
ARMS = ("no_fixed", "historical_1p25")
TASKS = ("torus2", "water1", "five_d2", "five_d3", "watersqrt2")
SOURCE_NAMES = (
    "ori_codex7.py", "ori_codex7_3d.py",
    "ori_codex6_spectral_subspace.py", "ori_codex6_spectral_subspace_3d.py",
    "ori_codex6_rejected_crossing.py", "ori_codex6_rejected_crossing_3d.py",
    "ori_codex6.py", "ori_codex6_3d.py", "ori_codex5.py", "ori_codex5_3d.py",
    "ori_codex4.py", "ori_codex4_3d.py", "ori_codex3.py", "ori_codex2.py",
    "ori_codex.py", "collision_3d.py", "single_solution_common.py",
    "analyze_ori_codex7_terminal_crossing_staged.py",
    "run_ori_codex7_terminal_crossing_staged.py",
)
JOB_INPUT_NAMES = (
    "run_config.json", "training_history.csv", "update_history.csv",
    "checkpoint.pt", "summary.json",
)
FINALIZER_NAMES = ("finalize_ori_codex7_2100.py", "test_finalize_ori_codex7_2100.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def require_below(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Path escaped authority root: {path}")
    return path


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being read: {path}")
    return data


def load_json(path: Path) -> dict:
    return json.loads(stable_bytes(path).decode("utf-8-sig"))


def load_csv(path: Path) -> list[dict]:
    data = stable_bytes(path).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(data)))


def active_codex7_pids() -> list[dict]:
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and ((($_.CommandLine -like '*ori_codex7.py*') "
        "-or ($_.CommandLine -like '*ori_codex7_3d.py*')) -or "
        "($_.CommandLine -like '*run_ori_codex7_terminal_crossing_staged.py*'))}; "
        "$p | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    return [{"pid": int(row["ProcessId"]), "command": row.get("CommandLine", "")}
            for row in payload]


def _float(row: dict, name: str, default=math.nan) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _int(row: dict, name: str, default=0) -> int:
    value = _float(row, name, float(default))
    return int(value) if math.isfinite(value) else default


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(values)) if values else math.nan


def _median(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else math.nan


def rolling(values: Sequence[float], window: int, kind: str = "mean") -> list[float]:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result = []
    queue = deque()
    if kind == "median":
        ordered = []
        for raw in values:
            value = float(raw)
            finite = math.isfinite(value)
            queue.append(value if finite else None)
            if finite:
                bisect.insort(ordered, value)
            if len(queue) > window:
                expired = queue.popleft()
                if expired is not None:
                    ordered.pop(bisect.bisect_left(ordered, expired))
            count = len(ordered)
            if not count:
                result.append(math.nan)
            elif count % 2:
                result.append(float(ordered[count // 2]))
            else:
                result.append(float((ordered[count // 2 - 1] + ordered[count // 2]) / 2))
        return result
    total = 0.0
    count = 0
    for raw in values:
        value = float(raw)
        finite = math.isfinite(value)
        queue.append(value if finite else None)
        if finite:
            total += value; count += 1
        if len(queue) > window:
            expired = queue.popleft()
            if expired is not None:
                total -= expired; count -= 1
        result.append(float(total / count) if count else math.nan)
    return result


def contiguous_history(rows: Sequence[dict]) -> int:
    episodes = [_int(row, "episode", -1) for row in rows]
    if episodes != list(range(1, len(rows) + 1)):
        raise RuntimeError("training_history.csv is not unique and contiguous from episode 1")
    return len(rows)


def checkpoint_episode(path: Path) -> int:
    if not path.is_file():
        return 0
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episode = int(payload.get("episode", 0))
    if episode < 0 or episode > 10000:
        raise RuntimeError(f"Invalid checkpoint episode {episode}: {path}")
    return episode


def atomic_endpoint(directory: Path, rows: Sequence[dict]) -> tuple[int, dict]:
    history_endpoint = contiguous_history(rows)
    checkpoint = checkpoint_episode(directory / "checkpoint.pt")
    summary_endpoint = 0
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary_endpoint = int(load_json(summary_path).get("training_episodes", 0))
    if summary_endpoint == 10000:
        if history_endpoint < 10000:
            raise RuntimeError(f"Final summary without 10000 contiguous history rows: {directory}")
        endpoint = 10000
        authority = "summary_and_history"
    else:
        if checkpoint <= 0:
            raise RuntimeError(f"Launched run has no atomic checkpoint: {directory}")
        endpoint = min(history_endpoint, checkpoint)
        authority = "checkpoint_and_history"
    if endpoint <= 0:
        raise RuntimeError(f"No positive atomic endpoint: {directory}")
    return endpoint, {
        "history_endpoint": history_endpoint,
        "checkpoint_endpoint": checkpoint,
        "summary_endpoint": summary_endpoint,
        "authority": authority,
    }


def validate_history_authority(rows: Sequence[dict], arm: str, endpoint: int,
                               source: dict) -> None:
    exact = {
        "ori7_candidate": "ori7_spectral_terminal_crossing_fixed_penalty_ablation",
        "ori7_candidate_schema_version": "1.0.0",
        "ori7_crossing_policy": "reject_candidate_restore_last_safe_and_terminate",
        "ori7_crossing_terminal_mode": arm,
        "ori7_reward_formula_sha256":
            "508f2f185f41bbd31b7edbd0d0248163bec39554bcd00a1351deff7ae158cb3d",
        "ori7_transition_formula_sha256":
            "061539186dc820c791fa4e2c57032d66e05e5232dae881bbf3d9bbb4984ece32",
    }
    numeric_zero = (
        "ori6_nonfinite_reward_component_count",
        "ori6_spectral_nonfinite_component_count",
        "ori7_crossing_nonfinite_component_count",
    )
    decomposition = (
        "ori6_reward_decomposition_error",
        "ori6_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
        "ori7_maximum_abs_crossing_reward_decomposition_error",
    )
    for index, row in enumerate(rows[:endpoint], 1):
        for field, expected in exact.items():
            if row.get(field) != expected:
                raise RuntimeError(f"History authority mismatch episode {index}: {field}")
        for field in numeric_zero:
            if row.get(field) in (None, "") or _float(row, field) != 0.0:
                raise RuntimeError(f"History nonfinite contract episode {index}: {field}")
        for field in decomposition:
            value = _float(row, field)
            if not math.isfinite(value) or abs(value) > 1e-8:
                raise RuntimeError(f"History decomposition contract episode {index}: {field}")
        crossing = _int(row, "ori7_terminal_crossing_count")
        if crossing not in (0, 1):
            raise RuntimeError(f"Non-binary crossing count episode {index}")
        if crossing:
            fixed = 0.0 if arm == "no_fixed" else -1.25
            quality = _float(row, "ori5_total_terminal_quality_reward")
            reward = _float(row, "ori7_terminal_crossing_reward")
            if (_int(row, "self_crossing") != 1
                    or str(row.get("termination_reason", "")).lower() != "self_crossing"
                    or not math.isclose(reward, fixed + quality, abs_tol=1e-8)):
                raise RuntimeError(f"Crossing reward/termination contract episode {index}")
        elif (_int(row, "self_crossing") != 0
              or _float(row, "ori7_terminal_crossing_reward", 0.0) != 0.0):
            raise RuntimeError(f"Noncrossing contract episode {index}")
    if source.get("ori_codex7.py") != \
            "20fda5bf54a1136713cebf89ab4ebf65166f9929b648316cb965cc771d565887":
        raise RuntimeError("Unexpected bound Codex7 planar core")
    if source.get("ori_codex7_3d.py") != \
            "d6a2aea943bc2efb1a269e7f640e96e70ee27a2de04c409ec59dd5b9643e208e":
        raise RuntimeError("Unexpected bound Codex7 spatial core")


def episode_curves(rows: Sequence[dict], endpoint: int, identity: dict) -> list[dict]:
    base = []
    for row in rows[:endpoint]:
        crossing = bool(_int(row, "ori7_terminal_crossing_count"))
        flatness = _float(row, "flatness_ratio")
        spread_error = _float(row, "spread_error")
        strict = int(not crossing and bool(_int(row, "strict_compact_success")))
        joint = int(not crossing and math.isfinite(flatness) and flatness <= .1
                    and math.isfinite(spread_error) and spread_error <= .1)
        base.append({
            **identity, "episode": _int(row, "episode"),
            "reward": _float(row, "total_reward"),
            "terminal_flatness": flatness, "spread_error": spread_error,
            "strict": strict, "joint": joint, "crossing": int(crossing),
        })
    series = {
        "reward_rolling": rolling([row["reward"] for row in base], ROLLING_WINDOW),
        "terminal_flatness_rolling_median": rolling(
            [row["terminal_flatness"] for row in base], ROLLING_WINDOW, "median"),
        "spread_error_rolling_median": rolling(
            [row["spread_error"] for row in base], ROLLING_WINDOW, "median"),
        "strict_rolling_rate": rolling([row["strict"] for row in base], ROLLING_WINDOW),
        "joint_rolling_rate": rolling([row["joint"] for row in base], ROLLING_WINDOW),
        "crossing_rolling_rate": rolling([row["crossing"] for row in base], ROLLING_WINDOW),
    }
    for field, values in series.items():
        for row, value in zip(base, values):
            row[field] = value
    return base


def optimizer_curves(rows: Sequence[dict], endpoint: int, identity: dict) -> list[dict]:
    result = []
    required = (
        "critic_loss", "critic_gradient_norm", "actor_learning_rate",
        "critic_learning_rate",
    )
    last_episode = 0
    for row in rows:
        episode = _int(row, "episode", -1)
        if episode <= 0 or episode > endpoint:
            continue
        if episode < last_episode:
            raise RuntimeError("update_history.csv episode order regressed")
        last_episode = episode
        parsed = {field: _float(row, field) for field in required}
        if any(not math.isfinite(value) for value in parsed.values()):
            raise RuntimeError(f"Nonfinite optimizer metric at episode {episode}")
        result.append({**identity, "episode": episode, **parsed})
    if not result:
        raise RuntimeError("Atomic run has no optimizer updates")
    return result


def summarize(curves: Sequence[dict], endpoint: int) -> dict:
    sample = list(curves[max(0, endpoint - 1024):endpoint])
    late = list(curves[max(0, endpoint - 512):endpoint])
    return {
        "reward_mean_last1024": _mean(row["reward"] for row in sample),
        "terminal_flatness_median_last1024": _median(
            row["terminal_flatness"] for row in sample),
        "spread_error_median_last1024": _median(row["spread_error"] for row in sample),
        "strict_count_last1024": sum(row["strict"] for row in sample),
        "strict_rate_last1024": _mean(row["strict"] for row in sample),
        "joint_count_last1024": sum(row["joint"] for row in sample),
        "joint_rate_last1024": _mean(row["joint"] for row in sample),
        "crossing_rate_last1024": _mean(row["crossing"] for row in sample),
        "reward_mean_last512": _mean(row["reward"] for row in late),
        "strict_count_total": sum(row["strict"] for row in curves[:endpoint]),
        "joint_count_total": sum(row["joint"] for row in curves[:endpoint]),
        "crossing_count_total": sum(row["crossing"] for row in curves[:endpoint]),
    }


def find_manifest(fingerprint: str, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        candidates = sorted(MANIFEST_DIRECTORY.glob(
            f"codex7_terminal_crossing_staged_{fingerprint[:12]}*.json"
        ))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one bound manifest, found {len(candidates)}")
    return candidates[0]


def validate_exact_campaign_contract(manifest: dict, binding: dict,
                                     report: dict, root: Path) -> None:
    fingerprint = binding.get("manifest_fingerprint")
    if fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise RuntimeError("This finalizer is bound only to the authorized C7TC2A campaign")
    if manifest.get("manifest_fingerprint") != fingerprint \
            or report.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError("Control-plane manifest fingerprint mismatch")
    if Path(manifest["experiment_root"]).resolve() != root.resolve():
        raise RuntimeError("Manifest root mismatch")
    if root.resolve() != DEFAULT_ROOT.resolve():
        raise RuntimeError("This finalizer rejects arbitrary campaign roots")
    expected_matrix = {
        (task, arm, seed) for task in TASKS for arm in ARMS for seed in (7101, 7102, 7103)
    }
    actual_matrix = {
        (job.get("task", {}).get("name"), job.get("arm"), int(job.get("seed", -1)))
        for job in manifest.get("jobs", [])
    }
    protocol_contract = {
        "schema_version": 1, "protocol": EXPECTED_PROTOCOL,
        "tasks": list(TASKS), "arms": list(ARMS), "seeds": [7101, 7102, 7103],
        "episodes_cap": 10000,
        "screen_checkpoints": [512, 1024, 2048, 4096],
        "production_checkpoints": [5120, 6144, 7168, 8192, 9216, 10000],
        "job_order": list(TASKS), "max_parallel": 12,
        "training_stop_local": "20:45", "report_deadline_local": "21:00",
    }
    changed = [field for field, expected in protocol_contract.items()
               if manifest.get(field) != expected]
    if changed or actual_matrix != expected_matrix or len(manifest.get("jobs", [])) != 30:
        raise RuntimeError(f"Exact campaign matrix/protocol mismatch: {changed}")
    binding_contract = {
        "job_order": list(TASKS), "max_parallel": 12,
        "training_stop_local": "20:45", "report_deadline_local": "21:00",
    }
    binding_changed = [field for field, expected in binding_contract.items()
                       if binding.get(field) != expected]
    if binding_changed:
        raise RuntimeError(f"Execution binding protocol mismatch: {binding_changed}")
    if (report.get("schema_version") != 1
            or report.get("protocol") != EXPECTED_PROTOCOL
            or report.get("training_deadline_local") != "20:45"
            or report.get("report_deadline_local") != "21:00"
            or len(report.get("endpoints", [])) != 30):
        raise RuntimeError("Runner report schema/protocol mismatch")


def verify_control_plane(root: Path, manifest_path: Path | None) -> tuple[dict, dict, dict, dict]:
    binding_path = root / "execution_binding.json"
    report_path = root / "runner_report.json"
    if not report_path.is_file():
        raise RuntimeError("runner_report.json is absent; runner has not finished")
    binding = load_json(binding_path)
    manifest_file = find_manifest(binding["manifest_fingerprint"], manifest_path)
    manifest = load_json(manifest_file)
    report = load_json(report_path)
    content = dict(manifest)
    recorded = content.pop("manifest_fingerprint", None)
    if recorded != canonical_digest(content):
        raise RuntimeError("Manifest self-fingerprint mismatch")
    validate_exact_campaign_contract(manifest, binding, report, root)
    if binding.get("source_sha256") != manifest.get("source_sha256"):
        raise RuntimeError("Execution/source binding mismatch")
    sources = manifest["source_sha256"]
    if set(sources) != set(SOURCE_NAMES):
        raise RuntimeError("Unexpected source authority keyset")
    for name, expected in sources.items():
        if sha256_file(SCRIPT_DIRECTORY / name) != expected:
            raise RuntimeError(f"Bound source changed: {name}")
    inputs = {
        str(manifest_file.resolve()): sha256_file(manifest_file),
        str(binding_path.resolve()): sha256_file(binding_path),
        str(report_path.resolve()): sha256_file(report_path),
    }
    return manifest, binding, report, inputs


def decision_evidence(report: dict, root: Path, inputs: dict) -> tuple[dict, dict]:
    """Load bound runner decisions used to derive causal censor labels."""
    arm_stops, production_stops = {}, {}
    for raw in artifact_paths(report.get("artifacts", [])):
        path = raw if raw.is_absolute() else (SCRIPT_DIRECTORY / raw)
        path = require_below(path, root)
        if not path.is_file():
            raise RuntimeError(f"Runner artifact missing: {path}")
        inputs[str(path.resolve())] = sha256_file(path)
        if path.name == "checkpoint_analysis.json":
            payload = load_json(path)
            endpoint = int(payload.get("endpoint", 0))
            for task, arms in payload.get("arm_decisions", {}).items():
                for arm, decision in arms.items():
                    if decision.get("stop"):
                        arm_stops[(task, arm)] = {
                            "endpoint": endpoint, "reasons": list(decision.get("reasons", []))
                        }
        elif path.name == "production_checkpoint.json":
            payload = load_json(path)
            report_row = payload.get("report", {})
            decision = payload.get("production_decision", {})
            if decision.get("stop"):
                production_stops[(
                    report_row.get("task"), report_row.get("arm"),
                    int(report_row.get("seed", -1)), int(report_row.get("endpoint", 0)),
                )] = list(decision.get("reasons", []))
    return arm_stops, production_stops


def causal_censor_reason(entry: dict, report: dict, arm_stops: dict,
                         production_stops: dict) -> str:
    """Return a causal censor label; runner's selected/screened status is never causal."""
    if not entry["launched"]:
        return ("queued_at_training_deadline"
                if entry["runner_status"] == "queued_censored_at_training_deadline"
                else "not_launched_by_staged_task_order")
    endpoint = int(entry.get("atomic_endpoint", 0))
    if endpoint == 10000:
        return "natural_completion_10000"
    task, arm, seed = entry["task"], entry["arm"], entry["seed"]
    matching_production = [
        reasons for (t, a, s, stop_endpoint), reasons in production_stops.items()
        if (t, a, s) == (task, arm, seed) and stop_endpoint <= endpoint
    ]
    if matching_production:
        reasons = " ".join(matching_production[-1]).lower()
        return ("production_integrity_stop" if "integrity" in reasons
                else "production_collapse_stop")
    stopped = arm_stops.get((task, arm))
    if stopped and int(stopped["endpoint"]) < FORMAL_ENDPOINT:
        reasons = " ".join(stopped["reasons"]).lower()
        return ("integrity_failure_before_4096" if "integrity" in reasons
                else "catastrophic_stop_before_4096")
    if endpoint < FORMAL_ENDPOINT:
        if not report.get("deadline_stop"):
            raise RuntimeError(
                f"Sub-formal censor without deadline or registered stop: {task}/{arm}/seed{seed}"
            )
        return ("deadline_before_first_atomic_checkpoint" if endpoint == 0
                else "deadline_right_censored_before_4096")
    selection = report.get("task_selections", {}).get(task, {})
    passing = selection.get("passing_arms", [])
    selected_arm, selected_seed = selection.get("selected_arm"), selection.get("selected_seed")
    if arm not in passing:
        return "rejected_at_4096_arm_gate"
    if selected_arm is None:
        return "paired_comparison_inconclusive_at_4096"
    if arm != selected_arm:
        return "not_selected_by_paired_arm_comparison"
    if seed != selected_seed:
        return "not_selected_seed_after_4096"
    if report.get("deadline_stop"):
        return "deadline_right_censored_selected_run"
    raise RuntimeError(f"Unable to derive causal censor reason: {task}/{arm}/seed{seed}")


def artifact_paths(value) -> Iterable[Path]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from artifact_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from artifact_paths(nested)
    elif isinstance(value, str):
        path = Path(value)
        if path.suffix.lower() in {".json", ".csv", ".md", ".png"}:
            yield path


def analyze_campaign(root: Path, manifest_path: Path | None = None,
                     active_check=True) -> dict:
    if active_check:
        active = active_codex7_pids()
        if active:
            raise RuntimeError(f"Active Codex7 PID(s); refusing finalization: {active}")
    root = root.resolve()
    manifest, binding, report, inputs = verify_control_plane(root, manifest_path)
    arm_stops, production_stops = decision_evidence(report, root, inputs)
    runner_by_identity = {
        (row["task"], row["arm"], int(row["seed"])): row
        for row in report.get("endpoints", [])
    }
    if len(runner_by_identity) != len(manifest["jobs"]):
        raise RuntimeError("Runner endpoint table is incomplete or duplicated")
    jobs, all_episode_curves, all_optimizer_curves = [], [], []
    for job in manifest["jobs"]:
        task = job["task"]["name"]
        arm, seed = job["arm"], int(job["seed"])
        identity = {"task": task, "arm": arm, "seed": seed}
        runner_row = runner_by_identity[(task, arm, seed)]
        launched = bool(runner_row.get("launched"))
        directory = require_below(Path(job["output_directory"]), root)
        model_directory = require_below(
            SCRIPT_DIRECTORY / job["task"]["model_directory"], SCRIPT_DIRECTORY
        )
        for name, expected_hash in job.get("model_sha256", {}).items():
            model_path = model_directory / name
            if not model_path.is_file() or sha256_file(model_path) != expected_hash:
                raise RuntimeError(f"Bound model changed: {model_path}")
            inputs[str(model_path.resolve())] = expected_hash
        entry = {
            **identity, "launched": launched,
            "runner_status": runner_row.get("status"),
            "selected": bool(runner_row.get("selected")),
            "runner_endpoint": int(runner_row.get("endpoint", 0)),
            "atomic_endpoint": 0,
            "evidence_phase": "unlaunched_censored",
            "censor_reason": runner_row.get("status") if not launched else "",
            "integrity_status": "not_applicable_unlaunched" if not launched else "pending",
        }
        if not launched:
            if directory.exists():
                raise RuntimeError(f"Unlaunched job unexpectedly has output: {directory}")
            jobs.append(entry)
            continue
        log_directory = root / "_runner_logs" / arm / task / f"seed{seed}"
        for name in ("stdout.log", "stderr.log"):
            log_path = log_directory / name
            if not log_path.is_file():
                raise RuntimeError(f"Launched job log missing: {log_path}")
            inputs[str(log_path.resolve())] = sha256_file(log_path)
        if not directory.is_dir():
            if int(runner_row.get("endpoint", 0)) != 0:
                raise RuntimeError(f"Positive runner endpoint without output directory: {directory}")
            entry.update({"evidence_phase": "launched_no_atomic_endpoint",
                          "censor_reason": "launched_but_no_atomic_output",
                          "integrity_status": "censored_no_atomic_endpoint"})
            jobs.append(entry)
            continue
        for name in JOB_INPUT_NAMES:
            path = directory / name
            if path.is_file():
                inputs[str(path.resolve())] = sha256_file(path)
        history_path = directory / "training_history.csv"
        checkpoint_path = directory / "checkpoint.pt"
        summary_path = directory / "summary.json"
        if not history_path.is_file() or (
            not checkpoint_path.is_file() and not summary_path.is_file()
        ):
            if int(runner_row.get("endpoint", 0)) != 0:
                raise RuntimeError(f"Positive runner endpoint without atomic files: {directory}")
            entry.update({"evidence_phase": "launched_no_atomic_endpoint",
                          "censor_reason": "launched_but_no_atomic_checkpoint_history",
                          "integrity_status": "censored_no_atomic_endpoint"})
            jobs.append(entry)
            continue
        config = load_json(directory / "run_config.json")
        expected_config = {
            "seed": seed,
            "target_spread": job["task"]["target_spread"],
            "max_steps": job["task"]["max_steps"],
            "max_step_width": job["task"]["max_step_width"],
            "flatness_subspace_dimension": job["task"]["flatness_dimension"],
            "actor_learning_rate": job["task"]["actor_learning_rate"],
            "critic_learning_rate": job["task"]["critic_learning_rate"],
            "actor_lr_milestones": job["task"]["actor_lr_milestones"],
            "near_miss_spread_error_far": job["task"]["near_miss_spread_error_far"],
            "data_dir": str(model_directory.resolve()),
            "episodes": 10000, "random_search": False,
            "update_method": "actor_critic",
            "ori5_quality_weight": 1.0, "ori5_crossing_penalty": 1.25,
            "ori6_spectral_subspace": True, "ori6_spectral_weight": .05,
            "ori7_candidate": "ori7_spectral_terminal_crossing_fixed_penalty_ablation",
            "ori7_crossing_policy": "reject_candidate_restore_last_safe_and_terminate",
            "ori7_crossing_terminal_mode": arm,
            "ori7_reward_formula_sha256":
                "508f2f185f41bbd31b7edbd0d0248163bec39554bcd00a1351deff7ae158cb3d",
            "ori7_transition_formula_sha256":
                "061539186dc820c791fa4e2c57032d66e05e5232dae881bbf3d9bbb4984ece32",
        }
        mismatched = []
        for field, expected in expected_config.items():
            actual = config.get(field)
            if isinstance(expected, float):
                try:
                    same = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                except (TypeError, ValueError):
                    same = False
            else:
                same = actual == expected
            if not same:
                mismatched.append(field)
        fixed = config.get("ori7_reward_parameters", {}).get("crossing_fixed_reward")
        expected_fixed = 0.0 if arm == "no_fixed" else -1.25
        try:
            fixed_matches = math.isclose(float(fixed), expected_fixed, abs_tol=1e-12)
        except (TypeError, ValueError):
            fixed_matches = False
        if not fixed_matches:
            mismatched.append("ori7_reward_parameters.crossing_fixed_reward")
        if mismatched:
            raise RuntimeError(f"Run config contract mismatch {mismatched}: {directory}")
        rows = load_csv(directory / "training_history.csv")
        endpoint, authority = atomic_endpoint(directory, rows)
        if int(runner_row.get("endpoint", 0)) != endpoint:
            raise RuntimeError(
                f"Runner/atomic endpoint mismatch {task}/{arm}/seed{seed}: "
                f"{runner_row.get('endpoint')} != {endpoint}"
            )
        validate_history_authority(rows, arm, endpoint, manifest["source_sha256"])
        updates = load_csv(directory / "update_history.csv")
        curves = episode_curves(rows, endpoint, identity)
        for curve in curves:
            curve["analysis_segment"] = (
                "formal_screen" if curve["episode"] <= FORMAL_ENDPOINT
                else ("selected_production_longitudinal" if entry["selected"]
                      else "post_gate_overrun_descriptive")
            )
        optimizer = optimizer_curves(updates, endpoint, identity)
        atomic_metrics = summarize(curves, endpoint)
        decision_endpoint = min(endpoint, FORMAL_ENDPOINT)
        decision_metrics = summarize(curves, decision_endpoint)
        entry.update(authority)
        entry.update({f"atomic_{field}": value for field, value in atomic_metrics.items()})
        entry.update({f"decision_{field}": value for field, value in decision_metrics.items()})
        entry.update({
            "atomic_endpoint": endpoint,
            "decision_evidence_endpoint": decision_endpoint,
            "post_gate_overrun_episodes": max(0, endpoint - FORMAL_ENDPOINT),
            "primary_metric_scope": (
                "formal_exact_4096" if endpoint >= FORMAL_ENDPOINT
                else "early_evidence_at_atomic_endpoint"
            ),
            "evidence_phase": "formal_exact_4096_available" if endpoint >= FORMAL_ENDPOINT
                              else "early_evidence_only",
            "censor_reason": "" if endpoint >= 10000 else (
                runner_row.get("status") or "right_censored_before_10000"
            ),
            "integrity_status": "verified",
        })
        jobs.append(entry)
        all_episode_curves.extend(curves)
        all_optimizer_curves.extend(optimizer)
    for entry in jobs:
        entry["censor_reason"] = causal_censor_reason(
            entry, report, arm_stops, production_stops
        )
    paired = paired_differences(jobs, all_episode_curves)
    finalizer_sources = {}
    for name in FINALIZER_NAMES:
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise RuntimeError(f"Finalizer authority missing: {path}")
        finalizer_sources[name] = sha256_file(path)
    return {
        "schema_version": 1,
        "analysis": "ori_codex7_2100_atomic_finalizer",
        "created_utc": utc_now(), "root": str(root),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "formal_endpoint": FORMAL_ENDPOINT,
        "formal_caveat": (
            "atomic_endpoint < 4096 is early evidence only and must not be interpreted "
            "as a formal arm comparison"
        ),
        "source_sha256": manifest["source_sha256"],
        "finalizer_source_sha256": finalizer_sources,
        "input_sha256": dict(sorted(inputs.items())),
        "runner_report": report,
        "jobs": jobs, "paired_seed_differences": paired,
        "episode_curves": all_episode_curves,
        "optimizer_curves": all_optimizer_curves,
    }


def paired_differences(jobs: Sequence[dict], curves: Sequence[dict]) -> list[dict]:
    job_map = {(row["task"], row["arm"], row["seed"]): row for row in jobs}
    curve_map = {}
    for row in curves:
        curve_map.setdefault((row["task"], row["arm"], row["seed"]), []).append(row)
    result = []
    for task in TASKS:
        for seed in sorted({row["seed"] for row in jobs}):
            no = job_map.get((task, "no_fixed", seed), {})
            hi = job_map.get((task, "historical_1p25", seed), {})
            observed_common = min(
                int(no.get("atomic_endpoint", 0)), int(hi.get("atomic_endpoint", 0))
            )
            common = min(observed_common, FORMAL_ENDPOINT)
            row = {
                "task": task, "seed": seed,
                "decision_common_endpoint": common,
                "observed_common_atomic_endpoint": observed_common,
                "no_fixed_post_gate_overrun_episodes": max(
                    0, int(no.get("atomic_endpoint", 0)) - FORMAL_ENDPOINT),
                "historical_post_gate_overrun_episodes": max(
                    0, int(hi.get("atomic_endpoint", 0)) - FORMAL_ENDPOINT),
                "comparison_phase": "formal_exact_4096" if common >= FORMAL_ENDPOINT
                                    else ("early_evidence_only" if common > 0
                                          else "unavailable_censored"),
            }
            if common <= 0:
                result.append(row)
                continue
            metrics = {}
            for arm in ARMS:
                sample = curve_map[(task, arm, seed)][:common]
                metrics[arm] = summarize(sample, common)
            for field in (
                "reward_mean_last1024", "terminal_flatness_median_last1024",
                "spread_error_median_last1024", "strict_rate_last1024",
                "joint_rate_last1024", "crossing_rate_last1024",
            ):
                row[f"no_fixed_minus_historical_{field}"] = (
                    metrics["no_fixed"][field] - metrics["historical_1p25"][field]
                )
            result.append(row)
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _plot_values(rows: Sequence[dict], field: str, max_points=1200):
    step = max(1, len(rows) // max_points)
    picked = list(rows[::step])
    if rows and picked[-1] is not rows[-1]:
        picked.append(rows[-1])
    return [row["episode"] for row in picked], [row[field] for row in picked]


def render_plots(payload: dict, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode_map = {}
    for row in payload["episode_curves"]:
        episode_map.setdefault((row["task"], row["arm"], row["seed"]), []).append(row)
    update_map = {}
    for row in payload["optimizer_curves"]:
        update_map.setdefault((row["task"], row["arm"], row["seed"]), []).append(row)
    colors = {"no_fixed": "#1f77b4", "historical_1p25": "#d62728"}
    lines = {7101: "-", 7102: "--", 7103: ":"}
    made = []
    learning_fields = (
        ("reward_rolling", "Return (rolling mean)"),
        ("terminal_flatness_rolling_median", "Terminal F (rolling median)"),
        ("spread_error_rolling_median", "Spread error (rolling median)"),
        ("strict_rolling_rate", "Strict rate (rolling)"),
        ("joint_rolling_rate", "Joint rate (rolling)"),
        ("crossing_rolling_rate", "Crossing rate (rolling)"),
    )
    optimizer_fields = (
        ("critic_loss", "Critic loss"),
        ("critic_gradient_norm", "Critic gradient norm"),
        ("actor_learning_rate", "Actor learning rate"),
        ("critic_learning_rate", "Critic learning rate"),
    )
    for task in TASKS:
        figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
        for axis, (field, label) in zip(axes.flat, learning_fields):
            for (row_task, arm, seed), rows in episode_map.items():
                if row_task != task:
                    continue
                x, y = _plot_values(rows, field)
                axis.plot(x, y, color=colors[arm], linestyle=lines.get(seed, "-"),
                          linewidth=.9, alpha=.82, label=f"{arm} s{seed}")
            axis.set_xlabel("Episode"); axis.set_ylabel(label); axis.grid(alpha=.2)
            if axis.get_xlim()[1] >= FORMAL_ENDPOINT:
                axis.axvline(FORMAL_ENDPOINT, color="0.35", linewidth=.7, alpha=.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=7, ncol=2)
        figure.suptitle(f"{task}: learning curves (window={ROLLING_WINDOW})")
        path = output / f"{task}_learning_curves.png"
        figure.savefig(path, dpi=150); plt.close(figure); made.append(path)

        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, (field, label) in zip(axes.flat, optimizer_fields):
            for (row_task, arm, seed), rows in update_map.items():
                if row_task != task:
                    continue
                x, y = _plot_values(rows, field)
                axis.plot(x, y, color=colors[arm], linestyle=lines.get(seed, "-"),
                          linewidth=.85, alpha=.8, label=f"{arm} s{seed}")
            axis.set_xlabel("Episode"); axis.set_ylabel(label); axis.grid(alpha=.2)
            if axis.get_xlim()[1] >= FORMAL_ENDPOINT:
                axis.axvline(FORMAL_ENDPOINT, color="0.35", linewidth=.7, alpha=.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=7, ncol=2)
        figure.suptitle(f"{task}: optimizer curves")
        path = output / f"{task}_optimizer_curves.png"
        figure.savefig(path, dpi=150); plt.close(figure); made.append(path)

    paired = [row for row in payload["paired_seed_differences"]
              if row["decision_common_endpoint"] > 0]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    paired_fields = (
        ("no_fixed_minus_historical_reward_mean_last1024", "Delta return"),
        ("no_fixed_minus_historical_terminal_flatness_median_last1024", "Delta terminal F"),
        ("no_fixed_minus_historical_spread_error_median_last1024", "Delta spread error"),
        ("no_fixed_minus_historical_strict_rate_last1024", "Delta strict rate"),
        ("no_fixed_minus_historical_joint_rate_last1024", "Delta joint rate"),
        ("no_fixed_minus_historical_crossing_rate_last1024", "Delta crossing rate"),
    )
    labels = [f"{row['task']}\ns{row['seed']}" for row in paired]
    for axis, (field, label) in zip(axes.flat, paired_fields):
        values = [row.get(field, math.nan) for row in paired]
        axis.bar(range(len(values)), values, color="#1f77b4", alpha=.8)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_xticks(range(len(values)), labels, rotation=60, ha="right", fontsize=7)
        axis.set_ylabel(label); axis.grid(axis="y", alpha=.2)
    figure.suptitle("Paired seed differences: no_fixed - historical_1p25")
    path = output / "paired_seed_differences.png"
    figure.savefig(path, dpi=150); plt.close(figure); made.append(path)
    return made


def japanese_report(payload: dict) -> str:
    lines = [
        "# Codex7 21時 最終レポート", "",
        f"- 生成時刻（UTC）: {payload['created_utc']}",
        f"- manifest fingerprint: `{payload['manifest_fingerprint']}`", "",
        "## 解釈上の注意", "",
        "atomic endpoint が4096未満のrunはearly evidenceであり、正式なarm比較とは解釈しません。",
        "未起動runは右打切りとして保持し、成功・失敗のどちらにも数えません。",
        "crossing terminal episodeはstrict/jointの正の証拠から除外しています。", "",
        "jointはnoncrossingかつ terminal F≤0.1 かつ spread error≤0.1 と定義しています。", "",
        "cohort待機中に4096を超えたscreened rowはpost_gate_overrun_descriptiveとして保存しますが、primary/formal集計とarm選択比較から除外します。", "",
        "## Task / arm / seed", "",
        "| task | arm | seed | launched | endpoint | phase | integrity | selected | reward(last1024) | F(last1024) | spread err(last1024) | strict | joint | crossing rate | censor |",
        "|---|---|---:|:---:|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["jobs"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['seed']} | "
            f"{'yes' if row['launched'] else 'no'} | {row['atomic_endpoint']} | "
            f"{row['evidence_phase']} | {row['integrity_status']} | "
            f"{'yes' if row['selected'] else 'no'} | "
            f"{_format(row.get('decision_reward_mean_last1024'))} | "
            f"{_format(row.get('decision_terminal_flatness_median_last1024'))} | "
            f"{_format(row.get('decision_spread_error_median_last1024'))} | "
            f"{row.get('decision_strict_count_last1024', '')} | "
            f"{row.get('decision_joint_count_last1024', '')} | "
            f"{_format(row.get('decision_crossing_rate_last1024'))} | "
            f"{row.get('censor_reason', '')} |"
        )
    lines.extend(["", "## Paired seed差（no_fixed − historical_1p25）", "",
                  "| task | seed | decision cap | observed common | phase | Δreward | ΔF | Δspread error | Δstrict rate | Δjoint rate | Δcrossing rate |",
                  "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["paired_seed_differences"]:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['decision_common_endpoint']} | "
            f"{row['observed_common_atomic_endpoint']} | "
            f"{row['comparison_phase']} | "
            + " | ".join(_format(row.get(field)) for field in (
                "no_fixed_minus_historical_reward_mean_last1024",
                "no_fixed_minus_historical_terminal_flatness_median_last1024",
                "no_fixed_minus_historical_spread_error_median_last1024",
                "no_fixed_minus_historical_strict_rate_last1024",
                "no_fixed_minus_historical_joint_rate_last1024",
                "no_fixed_minus_historical_crossing_rate_last1024",
            )) + " |"
        )
    lines.extend(["", "## Integrity", "",
                  f"入力ファイル {len(payload['input_sha256'])} 件をSHA-256で固定しました。",
                  "報酬式・transition式・Actor–Critic設定・crossing reward分解・parent/spectral分解を各atomic historyで再検証しています。",
                  ""])
    return "\n".join(lines)


def _format(value) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.6g}" if math.isfinite(value) else ""


def visual_qa(paths: Sequence[Path]) -> list[dict]:
    import matplotlib.image as mpimg
    result = []
    for path in paths:
        image = mpimg.imread(path)
        if image.ndim < 2 or image.shape[0] < 600 or image.shape[1] < 900:
            raise RuntimeError(f"Plot dimensions too small: {path} {image.shape}")
        if not bool((image == image).all()) or float(image.std()) < .01:
            raise RuntimeError(f"Plot is blank/nonfinite: {path}")
        result.append({"path": path.name, "shape": list(image.shape),
                       "pixel_std": float(image.std()), "sha256": sha256_file(path)})
    return result


def validate_output_authority(output: Path, campaign_root: Path) -> Path:
    output = output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("Production output is fixed exactly to C7TC2A_2100")
    campaign_root = campaign_root.resolve()
    if output == campaign_root or campaign_root in output.parents:
        raise RuntimeError("Final output must not be a campaign descendant")
    for model in (SCRIPT_DIRECTORY / "model").glob("*"):
        resolved = model.resolve()
        if output == resolved or resolved in output.parents:
            raise RuntimeError("Final output must not be a model descendant")
    if any(output == (SCRIPT_DIRECTORY / name).resolve() for name in SOURCE_NAMES):
        raise RuntimeError("Final output must not overwrite a bound source")
    return output


def write_bundle(payload: dict, output: Path, enforce_authority=True,
                 publish_active_check=True) -> dict:
    output = output.resolve()
    if enforce_authority:
        output = validate_output_authority(output, Path(payload["root"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building.", dir=output.parent) as temporary:
        build = Path(temporary)
        compact = {key: value for key, value in payload.items()
                   if key not in {"episode_curves", "optimizer_curves"}}
        (build / "final_report.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (build / "input_hashes.json").write_text(json.dumps({
            "manifest_fingerprint": payload["manifest_fingerprint"],
            "source_sha256": payload["source_sha256"],
            "finalizer_source_sha256": payload["finalizer_source_sha256"],
            "input_sha256": payload["input_sha256"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(build / "job_summary.csv", payload["jobs"])
        write_csv(build / "episode_curves.csv", payload["episode_curves"])
        write_csv(build / "optimizer_curves.csv", payload["optimizer_curves"])
        write_csv(build / "paired_seed_differences.csv", payload["paired_seed_differences"])
        markdown = japanese_report(payload)
        if "�" in markdown:
            raise RuntimeError("Japanese report contains Unicode replacement characters")
        (build / "report_ja.md").write_text(markdown, encoding="utf-8")
        plots = render_plots(payload, build)
        qa = visual_qa(plots)
        (build / "visual_qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        inventory = {}
        for path in sorted(build.iterdir()):
            if path.is_file():
                inventory[path.name] = sha256_file(path)
        (build / "output_hashes.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if publish_active_check:
            active = active_codex7_pids()
            if active:
                raise RuntimeError(
                    f"Active Codex7 PID(s) appeared before atomic publish: {active}"
                )
        os.replace(build, output)
    return {"output": str(output), "files": sorted(path.name for path in output.iterdir())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.root.resolve() != DEFAULT_ROOT.resolve():
        raise RuntimeError("Production root is fixed exactly to experiments_ori7/C7TC2A")
    validate_output_authority(args.output, args.root)
    active = active_codex7_pids()
    if active:
        raise RuntimeError(f"Active Codex7 PID(s); refusing finalization: {active}")
    payload = analyze_campaign(args.root, args.manifest, active_check=False)
    result = write_bundle(payload, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
