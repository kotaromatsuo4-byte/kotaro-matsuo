"""Fail-closed combined finalizer for the two zero-reject experiment phases.

Inputs are read only.  The finalizer waits for both staged runners and every
trainer to exit, requires both runner final summaries, recovers natural episode
10000 completion from the final policy/history/update/summary set, and writes a
fresh standalone analysis directory.  It never imports or edits an active
trainer/runner state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import analyze_ori_codex6_zero_reject_staged as staged
import run_ori_codex6_zero_reject_water_fallback as fallback


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
R1_ROOT = SCRIPT_DIRECTORY / (
    "experiments_ori6/staged/zero_reject_reward/P0X2RCS05ZR0R1"
)
FALLBACK_ROOT = SCRIPT_DIRECTORY / (
    "experiments_ori6/staged/zero_reject_water_fallback/P0X2RCS05ZR0WF12"
)
OLD_ROOT = SCRIPT_DIRECTORY / (
    "experiments_ori6/full/two_task_full_diagnostic/P0X2RCS05"
)
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / (
    "experiments_ori6/analysis/zero_reject_reward_final"
)
OLD_ENDPOINT = 6144
R1_SEEDS = (4201, 4202, 4203)
FALLBACK_SEEDS = (4301, 4302, 4303)
TASKS = ("torus2", "water1")
PROFILES = {"torus2": "A", "water1": "B"}
RUNNER_NAMES = (
    "run_ori_codex6_zero_reject_staged.py",
    "run_ori_codex6_zero_reject_water_fallback.py",
)
ROOT_SLUGS = ("P0X2RCS05ZR0R1", "P0X2RCS05ZR0WF12")


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    task: str
    seed: int
    run_directory: Path
    endpoint_mode: str
    role: str
    label: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def serialisable(value):
    if isinstance(value, dict):
        return {
            key: serialisable(item) for key, item in value.items()
            if key not in {"records", "updates", "raw_rows"}
        }
    if isinstance(value, list):
        return [serialisable(item) for item in value]
    if isinstance(value, tuple):
        return [serialisable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        query_limited_information, False, int(pid)
    )
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        ok = ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        )
        return bool(ok and exit_code.value == still_active)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def matching_python_processes() -> list[dict]:
    """Return only live Python runner/trainers; ignore read-only monitors."""
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, check=True,
        )
        text = completed.stdout.strip()
        payload = json.loads(text) if text else []
        processes = payload if isinstance(payload, list) else [payload]
        result = []
        for item in processes:
            name = str(item.get("Name") or "").lower()
            cmdline = str(item.get("CommandLine") or "")
            if not name.startswith("python"):
                continue
            lowered = cmdline.lower()
            if any(value.lower() in lowered for value in (*ROOT_SLUGS, *RUNNER_NAMES)):
                result.append({
                    "pid": int(item.get("ProcessId") or 0),
                    "name": item.get("Name"),
                    "command_line": cmdline,
                })
        return result
    result = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Cannot enumerate runner processes on this platform")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        lowered = cmdline.lower()
        if "python" in lowered and any(
            value.lower() in lowered for value in (*ROOT_SLUGS, *RUNNER_NAMES)
        ):
            result.append({"pid": int(entry.name), "name": "python", "command_line": cmdline})
    return result


def status_records(root: Path, tasks_and_seeds: Sequence[tuple[str, int]]) -> list[dict]:
    records = []
    for task, seed in tasks_and_seeds:
        path = root / "_runner_logs" / task / f"seed{seed}" / "status.json"
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise FileNotFoundError(f"missing runner status: {path}")
        pid = int(payload.get("pid", 0) or 0)
        records.append({
            "root": str(root.resolve()),
            "task": task,
            "seed": seed,
            "path": str(path.resolve()),
            "status": payload.get("status"),
            "checkpoint_episode": payload.get("checkpoint_episode"),
            "reason": payload.get("reason"),
            "pid": pid,
            "pid_alive": process_is_alive(pid),
            "payload": payload,
        })
    return records


def assert_ready() -> dict:
    r1_expected = [
        (task, seed) for task in TASKS for seed in R1_SEEDS
    ]
    fallback_expected = [("water1", seed) for seed in FALLBACK_SEEDS]
    r1_status = status_records(R1_ROOT, r1_expected)
    fallback_status = status_records(FALLBACK_ROOT, fallback_expected)
    live_status = [item for item in [*r1_status, *fallback_status] if item["pid_alive"]]
    live_scan = matching_python_processes()
    if live_status or live_scan:
        labels = [f"status pid={item['pid']} {item['task']}/{item['seed']}" for item in live_status]
        labels.extend(f"process pid={item['pid']}" for item in live_scan)
        raise RuntimeError("runner/trainer still alive: " + ", ".join(labels))
    unfinished = [
        item for item in [*r1_status, *fallback_status]
        if item["status"] not in {"complete", "stopped_at_last_atomic_checkpoint"}
    ]
    if unfinished:
        raise RuntimeError(
            "runner status not terminal: "
            + ", ".join(f"{x['task']}/seed{x['seed']}={x['status']}" for x in unfinished)
        )
    r1_final_path = R1_ROOT / "analysis/final_summary.json"
    fallback_final_path = FALLBACK_ROOT / "analysis/final_summary.json"
    r1_final = read_json(r1_final_path)
    fallback_final = read_json(fallback_final_path)
    if not isinstance(r1_final, dict) or not isinstance(fallback_final, dict):
        missing = [
            str(path) for path, payload in (
                (r1_final_path, r1_final), (fallback_final_path, fallback_final)
            ) if not isinstance(payload, dict)
        ]
        raise FileNotFoundError("runner final summary absent: " + ", ".join(missing))
    return {
        "ready": True,
        "r1_statuses": r1_status,
        "fallback_statuses": fallback_status,
        "r1_final": r1_final,
        "fallback_final": fallback_final,
        "live_processes": [],
    }


def torch_episode(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episode = int(payload.get("episode", 0))
        return episode if episode > 0 else None
    except (OSError, RuntimeError, EOFError, ValueError, TypeError):
        return None


def csv_last_episode(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = [
                line for line in handle.read().decode("utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
        return int(float(next(csv.reader([lines[-1]]))[-1])) if lines else None
    except (OSError, csv.Error, ValueError, IndexError):
        return None


def authoritative_endpoint(run_directory: Path) -> dict:
    checkpoint = torch_episode(run_directory / "checkpoint.pt")
    policy = torch_episode(run_directory / "policy.pt")
    history = csv_last_episode(run_directory / "training_history.csv")
    updates = csv_last_episode(run_directory / "update_history.csv")
    complete = (
        policy == 10000 and history == 10000 and updates == 10000
        and (run_directory / "summary.json").is_file()
        and (run_directory / "run_config.json").is_file()
    )
    if complete:
        return {
            "episode": 10000,
            "authority": "complete_final_policy_history_update_summary_config",
            "checkpoint_episode": checkpoint,
            "policy_episode": policy,
            "history_episode": history,
            "update_episode": updates,
            "natural_10000_recovered": True,
            "periodic_checkpoint_gap": checkpoint != 10000,
        }
    endpoint = checkpoint or 0
    return {
        "episode": endpoint,
        "authority": "periodic_atomic_checkpoint",
        "checkpoint_episode": checkpoint,
        "policy_episode": policy,
        "history_episode": history,
        "update_episode": updates,
        "natural_10000_recovered": False,
        "periodic_checkpoint_gap": False,
    }


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else math.nan


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else math.nan


def enrich_record(record: dict, raw: dict, task: str) -> dict:
    result = dict(record)
    flatness = result["flatness"]
    spread_error = result["spread_error"]
    result.update({
        "success": bool(result["strict"]),
        "wrong_plane_basin": bool(
            task == "water1" and not result["invalid"]
            and flatness <= 0.5 and spread_error >= 0.30
        ),
        "boundary_plane_basin": bool(
            task == "water1" and not result["invalid"]
            and flatness <= 0.5 and 0.10 < spread_error <= 0.15
        ),
        "torus_collapse_basin": bool(
            task == "torus2" and not result["invalid"]
            and flatness >= 0.9 and spread_error >= 0.30
        ),
        "zero_reject_count": staged._int(raw, ("ori6_zero_reject_count",), 0),
        "zero_reject_ordinary_reward": staged._float(
            raw, ("ori6_zero_reject_total_reward",), 0.0
        ),
        "zero_reject_timeout_reward": staged._float(
            raw, ("ori6_zero_reject_timeout_reward",), 0.0
        ),
        "reject_streak_max": staged._int(
            raw, ("ori6_max_consecutive_rejected_crossings",), 0
        ),
        "termination_reason": raw.get("termination_reason", ""),
    })
    return result


def window_summary(
    records: Sequence[dict], updates: Sequence[dict], start: int, end: int,
) -> dict:
    rows = [row for row in records if start <= row["episode"] <= end]
    n = len(rows)
    base = staged.window_summary(rows, start, end)
    return {
        **base,
        **staged.update_window_summary(updates, start, end),
        "success_count": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / n if n else 0.0,
        "wrong_plane_basin_count": sum(row["wrong_plane_basin"] for row in rows),
        "wrong_plane_basin_rate": sum(row["wrong_plane_basin"] for row in rows) / n if n else 0.0,
        "boundary_plane_basin_count": sum(row["boundary_plane_basin"] for row in rows),
        "boundary_plane_basin_rate": sum(row["boundary_plane_basin"] for row in rows) / n if n else 0.0,
        "torus_collapse_basin_count": sum(row["torus_collapse_basin"] for row in rows),
        "torus_collapse_basin_rate": sum(row["torus_collapse_basin"] for row in rows) / n if n else 0.0,
        "reject_episode_count": sum(row["rejects"] > 0 for row in rows),
        "reject_episode_rate": sum(row["rejects"] > 0 for row in rows) / n if n else 0.0,
        "zero_reject_action_count": sum(row["zero_reject_count"] for row in rows),
        "zero_reject_ordinary_reward_sum": sum(
            row["zero_reject_ordinary_reward"] for row in rows
        ),
        "max_abs_zero_reject_ordinary_reward": max(
            [abs(row["zero_reject_ordinary_reward"]) for row in rows] or [0.0]
        ),
        "zero_reject_timeout_reward_sum": sum(
            row["zero_reject_timeout_reward"] for row in rows
        ),
        "max_reject_streak": max([row["reject_streak_max"] for row in rows] or [0]),
    }


def _last_csv_row(path: Path, endpoint: int) -> dict | None:
    if not path.is_file():
        return None
    latest = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = staged._int(row, ("episode",), -1)
            if 0 < episode <= endpoint:
                latest = row
    return latest


def _evaluation_from_stochastic(data: dict, source: str, heldout: bool) -> dict:
    return {
        "evaluation_mode": "stochastic_heldout" if heldout else "periodic_stochastic",
        "source": source,
        "heldout_final": heldout,
        "sample_count": int(data.get("evaluation_episodes", 0) or 0),
        "episode": int(data.get("episode", 0) or 0),
        "success_rate": float(data.get("success_rate", math.nan)),
        "mean_return": float(data.get("mean_return", math.nan)),
        "mean_flatness": float(data.get("mean_terminal_ratio", math.nan)),
        "median_flatness": float(data.get("median_terminal_ratio", math.nan)),
        "mean_spread_error": float(data.get("mean_spread_error", math.nan)),
        "median_spread_error": float(data.get("median_spread_error", math.nan)),
        "mean_reject_count": float(data.get("mean_ori6_rejected_crossing_count", math.nan)),
        "mean_zero_reject_reward": float(data.get("mean_ori6_zero_reject_total_reward", math.nan)),
        "interpretation": (
            "64-rollout final held-out stochastic estimate"
            if heldout else "latest periodic stochastic diagnostic; not final held-out"
        ),
    }


def _evaluation_from_deterministic(data: dict, source: str, final: bool) -> dict:
    flatness = staged._float(data, ("ori5_last_valid_flatness_ratio", "flatness_ratio"))
    spread_error = staged._float(data, ("ori5_last_valid_spread_error", "spread_error"))
    return {
        "evaluation_mode": "deterministic_final_single" if final else "periodic_deterministic_single",
        "source": source,
        "heldout_final": False,
        "sample_count": 1,
        "episode": staged._int(data, ("episode",), 0),
        "success_rate": float(staged._int(data, ("success", "strict_compact_success"), 0)),
        "mean_return": staged._float(data, ("total_reward",)),
        "mean_flatness": flatness,
        "median_flatness": flatness,
        "mean_spread_error": spread_error,
        "median_spread_error": spread_error,
        "mean_reject_count": float(staged._int(data, ("ori6_rejected_crossing_count",), 0)),
        "mean_zero_reject_reward": staged._float(
            data, ("ori6_zero_reject_total_reward",), 0.0
        ),
        "interpretation": (
            "one deterministic final rollout; not a success-rate estimate"
            if final else "one latest periodic deterministic rollout; not held-out rate evidence"
        ),
    }


def extract_evaluations(run_directory: Path, endpoint: int, natural_10000: bool) -> list[dict]:
    summary = read_json(run_directory / "summary.json", {})
    result = []
    if natural_10000 and isinstance(summary, dict):
        stochastic = summary.get("last_policy_held_out_evaluation")
        deterministic = summary.get("last_policy_deterministic_evaluation")
        if isinstance(stochastic, dict):
            result.append(_evaluation_from_stochastic(
                stochastic, "summary:last_policy_held_out_evaluation", True
            ))
        if isinstance(deterministic, dict):
            result.append(_evaluation_from_deterministic(
                deterministic, "summary:last_policy_deterministic_evaluation", True
            ))
    if not result:
        stochastic = _last_csv_row(
            run_directory / "stochastic_evaluation_history.csv", endpoint
        )
        deterministic = _last_csv_row(
            run_directory / "deterministic_evaluation_history.csv", endpoint
        )
        if stochastic:
            result.append(_evaluation_from_stochastic(
                stochastic, "stochastic_evaluation_history.csv:last<=endpoint", False
            ))
        if deterministic:
            result.append(_evaluation_from_deterministic(
                deterministic, "deterministic_evaluation_history.csv:last<=endpoint", False
            ))
    return result


def make_run_specs(readiness: dict) -> tuple[list[RunSpec], dict]:
    r1_selected = {
        key: int(value) for key, value in readiness["r1_final"].get("selected_seeds", {}).items()
    }
    fallback_selected = readiness["fallback_final"].get("selected_seed")
    if fallback_selected is None:
        fallback_selected = readiness["fallback_final"].get("selected_seeds", {}).get("water1")
    fallback_selected = int(fallback_selected) if fallback_selected is not None else None
    specs: list[RunSpec] = []
    for task in TASKS:
        for seed in R1_SEEDS:
            selected = r1_selected.get(task) == seed
            if task == "water1":
                role = "formal_rejected_screen"
                mode = "formal_4096"
                label = f"R1 far=0.40 seed{seed} rejected@4096"
            elif selected:
                role = "selected_production"
                mode = "authoritative"
                label = f"R1 zero-reject seed{seed} selected"
            else:
                role = "screen_not_selected"
                mode = "formal_4096"
                label = f"R1 zero-reject seed{seed} screen@4096"
            specs.append(RunSpec(
                "r1_zero_reject", task, seed,
                R1_ROOT / task / PROFILES[task] / f"seed{seed}",
                mode, role, label,
            ))
    for seed in FALLBACK_SEEDS:
        selected = fallback_selected == seed
        specs.append(RunSpec(
            "water_fallback_far_0p12", "water1", seed,
            FALLBACK_ROOT / "water1/B" / f"seed{seed}",
            "authoritative" if selected else "formal_4096",
            "selected_production" if selected else "screen_not_selected",
            f"fallback far=0.12 seed{seed}" + (" selected" if selected else " screen@4096"),
        ))
    for task in TASKS:
        specs.append(RunSpec(
            "old_P0X2RCS05", task, 4101,
            OLD_ROOT / task / PROFILES[task] / "seed4101",
            "old_6144", "old_baseline", "old P0X2RCS05 seed4101@6144",
        ))
    return specs, {
        "r1_selected": r1_selected,
        "fallback_selected": fallback_selected,
    }


def validate_config(dataset: str, task: str, seed: int, path: Path) -> dict | None:
    if dataset == "r1_zero_reject":
        return staged.validate_config(path, task, seed)
    if dataset == "water_fallback_far_0p12":
        return fallback.validate_fallback_config(path, seed)
    return None


def load_run(spec: RunSpec) -> dict:
    authority = authoritative_endpoint(spec.run_directory)
    if spec.endpoint_mode == "formal_4096":
        endpoint = 4096
        endpoint_authority = {
            **authority,
            "episode": endpoint,
            "authority": "formal_gate_prefix_4096",
            "underlying_atomic_authority": authority["authority"],
            "underlying_atomic_episode": authority["episode"],
        }
        if authority["episode"] < endpoint:
            raise RuntimeError(
                f"{spec.label}: atomic endpoint {authority['episode']} below formal 4096"
            )
    elif spec.endpoint_mode == "old_6144":
        endpoint = OLD_ENDPOINT
        endpoint_authority = {
            **authority,
            "episode": endpoint,
            "authority": "preserved_old_atomic_prefix_6144",
        }
        if authority["episode"] < endpoint:
            raise RuntimeError(f"{spec.label}: old prefix does not reach 6144")
    else:
        endpoint = int(authority["episode"])
        endpoint_authority = authority
        if endpoint <= 0:
            raise RuntimeError(f"{spec.label}: no authoritative endpoint")

    raw_rows, history_errors = staged.read_history(
        spec.run_directory / "training_history.csv", endpoint
    )
    updates, update_errors = staged.read_updates(
        spec.run_directory / "update_history.csv", endpoint
    )
    records = [
        enrich_record(staged.normalise(raw), raw, spec.task) for raw in raw_rows
    ]
    windows = []
    for start in range(1, endpoint + 1, 512):
        end = min(endpoint, start + 511)
        windows.append(window_summary(records, updates, start, end))
    first = window_summary(records, updates, 1, min(1024, endpoint))
    last = window_summary(records, updates, max(1, endpoint - 1023), endpoint)
    last_512 = window_summary(records, updates, max(1, endpoint - 511), endpoint)
    config = validate_config(
        spec.dataset, spec.task, spec.seed, spec.run_directory / "run_config.json"
    )
    errors = [*history_errors, *update_errors]
    if config is not None and not config.get("pass"):
        errors.extend(f"config:{item}" for item in config.get("errors", []))
    global_decomposition_max = max(
        [record["decomposition_max"] for record in records] or [math.inf]
    )
    global_nonfinite_count = sum(record["nonfinite_count"] for record in records)
    if global_decomposition_max > 1.0e-8:
        errors.append("reward decomposition error exceeds 1e-8")
    if global_nonfinite_count:
        errors.append("nonfinite reward component")
    if spec.dataset != "old_P0X2RCS05" and (
        max(abs(row["zero_reject_ordinary_reward"]) for row in records) > 1.0e-12
    ):
        errors.append("ordinary rejected no-op reward is nonzero")
    evaluations = extract_evaluations(
        spec.run_directory, endpoint, bool(endpoint_authority["natural_10000_recovered"])
    )
    for evaluation in evaluations:
        evaluation.update({
            "dataset": spec.dataset,
            "task": spec.task,
            "seed": spec.seed,
            "role": spec.role,
            "run_label": spec.label,
            "run_endpoint": endpoint,
        })
    return {
        "dataset": spec.dataset,
        "task": spec.task,
        "seed": spec.seed,
        "role": spec.role,
        "label": spec.label,
        "run_directory": str(spec.run_directory.resolve()),
        "endpoint": endpoint,
        "endpoint_authority": endpoint_authority,
        "integrity_pass": not errors,
        "integrity_errors": errors,
        "global_max_abs_reward_decomposition_error": global_decomposition_max,
        "global_nonfinite_reward_component_count": global_nonfinite_count,
        "config_validation": config,
        "first_1024": first,
        "last_1024": last,
        "last_512": last_512,
        "windows_512": windows,
        "evaluations": evaluations,
        "records": records,
        "updates": updates,
    }


def decision_rows(root: Path, phase: str) -> list[dict]:
    payload = read_json(root / "analysis/decision_history.json", {})
    rows = []
    for index, decision in enumerate(payload.get("decisions", []), start=1):
        warning = decision.get("collapse_warning", {})
        rows.append({
            "phase": phase,
            "sequence_in_phase": index,
            "task": decision.get("task"),
            "checkpoint_episode": decision.get("checkpoint_episode"),
            "rule": decision.get("rule"),
            "promote": decision.get("promote"),
            "stop": decision.get("stop"),
            "selected_seed": decision.get("selected_seed"),
            "promising_seed_count": decision.get("promising_seed_count"),
            "promising_seeds": json.dumps(decision.get("promising_seeds", [])),
            "selection_order": json.dumps(decision.get("selection_order", [])),
            "collapse_warning": warning.get("warning"),
            "sustained_collapse": decision.get("sustained_collapse"),
            "reasons": "; ".join(decision.get("reasons", [])),
        })
    return rows


def checkpoint_json(root: Path, endpoint: int, task: str) -> tuple[Path, dict]:
    path = root / f"analysis/checkpoint_{endpoint}_{task}/checkpoint_analysis.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise FileNotFoundError(path)
    return path, payload


def gate_integrity(readiness: dict) -> tuple[dict, list[dict]]:
    r1_torus_path, r1_torus = checkpoint_json(R1_ROOT, 4096, "torus2")
    r1_water_path, r1_water = checkpoint_json(R1_ROOT, 4096, "water1")
    fallback_path, fallback_water = checkpoint_json(FALLBACK_ROOT, 4096, "water1")
    checks = []

    def add(name: str, passed: bool, evidence: str, detail: str):
        checks.append({
            "category": "gate", "check": name, "pass": bool(passed),
            "evidence": evidence, "detail": detail,
        })

    td = r1_torus["tasks"]["torus2"]["decision"]
    wd = r1_water["tasks"]["water1"]["decision"]
    fd = fallback_water["tasks"]["water1"]["decision"]
    add(
        "R1 torus formal gate promoted 2/3 and selected 4203",
        td.get("promote") is True and td.get("stop") is False
        and int(td.get("promising_seed_count", 0)) >= 2
        and td.get("selected_seed") == 4203,
        str(r1_torus_path.resolve()), json.dumps(td, ensure_ascii=False),
    )
    add(
        "R1 water far=0.40 formal gate rejected",
        wd.get("promote") is False and wd.get("stop") is True
        and int(wd.get("promising_seed_count", 99)) < 2
        and wd.get("selected_seed") is None,
        str(r1_water_path.resolve()), json.dumps(wd, ensure_ascii=False),
    )
    add(
        "fallback water far=0.12 formal gate promoted 2/3 and selected 4301",
        fd.get("promote") is True and fd.get("stop") is False
        and int(fd.get("promising_seed_count", 0)) >= 2
        and fd.get("selected_seed") == 4301,
        str(fallback_path.resolve()), json.dumps(fd, ensure_ascii=False),
    )
    add(
        "R1 runner final selection agrees with torus gate and excludes water",
        readiness["r1_final"].get("selected_seeds") == {"torus2": 4203},
        str((R1_ROOT / "analysis/final_summary.json").resolve()),
        json.dumps(readiness["r1_final"].get("selected_seeds"), ensure_ascii=False),
    )
    fallback_selected = readiness["fallback_final"].get("selected_seed")
    add(
        "fallback runner final selection agrees with gate",
        fallback_selected == 4301,
        str((FALLBACK_ROOT / "analysis/final_summary.json").resolve()),
        f"selected_seed={fallback_selected}",
    )
    return {
        "r1_torus": td,
        "r1_water": wd,
        "fallback_water": fd,
        "seed_reports": {
            "r1_torus": r1_torus["tasks"]["torus2"]["seeds"],
            "r1_water": r1_water["tasks"]["water1"]["seeds"],
            "fallback_water": fallback_water["tasks"]["water1"]["seeds"],
        },
        "evidence_paths": [
            str(r1_torus_path.resolve()), str(r1_water_path.resolve()),
            str(fallback_path.resolve()),
        ],
    }, checks


def chronological_evidence(readiness: dict, runs: Sequence[dict]) -> list[dict]:
    r1_torus_path, r1_torus = checkpoint_json(R1_ROOT, 4096, "torus2")
    r1_water_path, r1_water = checkpoint_json(R1_ROOT, 4096, "water1")
    fallback_gate_path, fallback_gate = checkpoint_json(FALLBACK_ROOT, 4096, "water1")
    counterfactual_path = SCRIPT_DIRECTORY / (
        "experiments_ori6/analysis/water_cutoff_counterfactual_20260827/"
        "water_cutoff_counterfactual.json"
    )
    counterfactual = read_json(counterfactual_path, {})
    r1_binding_path = R1_ROOT / "execution_manifest_binding.json"
    r1_binding = read_json(r1_binding_path, {})
    fallback_binding_path = FALLBACK_ROOT / "execution_manifest_binding.json"
    fallback_binding = read_json(fallback_binding_path, {})
    selected_torus = next(
        run for run in runs if run["dataset"] == "r1_zero_reject"
        and run["task"] == "torus2" and run["seed"] == 4203
    )
    selected_water = next(
        run for run in runs if run["dataset"] == "water_fallback_far_0p12"
        and run["seed"] == 4301
    )
    return [
        {
            "sequence": 1,
            "stage": "old_reference",
            "description_ja": "旧 P0X2RCS05 seed4101 の6144 episodeを記述的基準として固定。",
            "evidence": str(OLD_ROOT.resolve()),
        },
        {
            "sequence": 2,
            "stage": "r1_preregistered_design",
            "description_ja": (
                "通常crossing rejectをreward=0のno-op継続に変更。R1はtorusの"
                "actor LR初回milestone=1528、waterのnear-miss far=0.40を"
                "task-specific設定として開始し、run途中では変更しなかった。"
            ),
            "started_utc": r1_binding.get("started_utc"),
            "evidence": str(r1_binding_path.resolve()),
        },
        {
            "sequence": 3,
            "stage": "r1_formal_4096",
            "description_ja": (
                "zero-reject R1では torus が4096 gateを通過してseed4203を選択し、"
                "water far=0.40は2/3条件を満たさず正式棄却。"
            ),
            "generated_utc": r1_water.get("generated_utc"),
            "evidence": [str(r1_torus_path.resolve()), str(r1_water_path.resolve())],
        },
        {
            "sequence": 4,
            "stage": "water_counterfactual",
            "description_ja": (
                "保存trajectory上の反実仮想で境界 basin の過大報酬を確認し、"
                "最小変更 far=0.12を事前選択。これは再学習結果ではない。"
            ),
            "generated_utc": counterfactual.get("generated_utc"),
            "evidence": str(counterfactual_path.resolve()),
        },
        {
            "sequence": 5,
            "stage": "fallback_authorization",
            "description_ja": (
                "R1 water正式棄却、CSV再計算、反実仮想成果物・source prefix hashを"
                "fail-closed確認してfresh fallbackを開始。"
            ),
            "started_utc": fallback_binding.get("started_utc"),
            "evidence": str(fallback_binding_path.resolve()),
        },
        {
            "sequence": 6,
            "stage": "fallback_formal_4096",
            "description_ja": (
                "water far=0.12は4096で2/3 seedがpromisingとなりseed4301を選択。"
            ),
            "generated_utc": fallback_gate.get("generated_utc"),
            "evidence": str(fallback_gate_path.resolve()),
        },
        {
            "sequence": 7,
            "stage": "selected_production_endpoints",
            "description_ja": (
                f"torus seed4203はep{selected_torus['endpoint']}、water seed4301は"
                f"ep{selected_water['endpoint']}を最終authorityとして解析。"
            ),
            "evidence": [
                selected_torus["run_directory"], selected_water["run_directory"],
            ],
        },
    ]


def collect_input_paths(runs: Sequence[dict], gate_info: dict) -> list[Path]:
    paths: set[Path] = set()
    for run in runs:
        directory = Path(run["run_directory"])
        for name in (
            "training_history.csv", "update_history.csv", "run_config.json",
            "checkpoint.pt", "policy.pt", "summary.json",
            "stochastic_evaluation_history.csv", "deterministic_evaluation_history.csv",
            "best_policy_metrics.json",
        ):
            path = directory / name
            if path.is_file():
                paths.add(path.resolve())
    for root, tasks_and_seeds in (
        (R1_ROOT, ((task, seed) for task in TASKS for seed in R1_SEEDS)),
        (FALLBACK_ROOT, (("water1", seed) for seed in FALLBACK_SEEDS)),
    ):
        for task, seed in tasks_and_seeds:
            status = root / "_runner_logs" / task / f"seed{seed}" / "status.json"
            if status.is_file():
                paths.add(status.resolve())
    for root in (R1_ROOT, FALLBACK_ROOT):
        for relative in (
            "analysis/final_summary.json", "analysis/decision_history.json",
            "execution_manifest_binding.json",
        ):
            path = (root / relative).resolve()
            if path.is_file():
                paths.add(path)
        binding = read_json(root / "execution_manifest_binding.json", {})
        manifest_path = Path(str(binding.get("manifest_path", "")))
        if manifest_path.is_file():
            paths.add(manifest_path.resolve())
    for value in gate_info.get("evidence_paths", []):
        path = Path(value)
        if path.is_file():
            paths.add(path.resolve())
    counterfactual_directory = (
        SCRIPT_DIRECTORY / "experiments_ori6/analysis/water_cutoff_counterfactual_20260827"
    )
    if counterfactual_directory.is_dir():
        for path in counterfactual_directory.iterdir():
            if path.is_file():
                paths.add(path.resolve())
    for path in (
        SCRIPT_DIRECTORY / "analyze_ori_codex6_zero_reject_staged.py",
        SCRIPT_DIRECTORY / "run_ori_codex6_zero_reject_staged.py",
        SCRIPT_DIRECTORY / "run_ori_codex6_zero_reject_water_fallback.py",
        SCRIPT_DIRECTORY / "analyze_ori_codex6_water_cutoff_counterfactual.py",
        SCRIPT_DIRECTORY / "experiments_ori6/analysis/water_cutoff_counterfactual_20260827/water_cutoff_counterfactual.json",
        Path(__file__).resolve(),
        SCRIPT_DIRECTORY / "test_finalize_ori_codex6_zero_reject_combined.py",
    ):
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths, key=str)


def hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths}


def add_check(
    checks: list[dict], category: str, name: str, passed: bool,
    evidence: str = "", detail: str = "",
) -> None:
    checks.append({
        "category": category,
        "check": name,
        "pass": bool(passed),
        "evidence": evidence,
        "detail": detail,
    })


def run_map(runs: Sequence[dict]) -> dict[tuple[str, str, int], dict]:
    return {(run["dataset"], run["task"], run["seed"]): run for run in runs}


def build_payload() -> dict:
    readiness = assert_ready()
    specs, selections = make_run_specs(readiness)
    if selections["r1_selected"] != {"torus2": 4203}:
        raise RuntimeError(f"unexpected R1 selections: {selections['r1_selected']}")
    if selections["fallback_selected"] != 4301:
        raise RuntimeError(f"unexpected fallback selection: {selections['fallback_selected']}")
    runs = [load_run(spec) for spec in specs]
    by_key = run_map(runs)
    gate_info, gate_checks = gate_integrity(readiness)
    checks = list(gate_checks)

    for run in runs:
        add_check(
            checks, "run_integrity", f"{run['label']} contiguous/config/reward integrity",
            run["integrity_pass"], run["run_directory"],
            "; ".join(run["integrity_errors"]) or "PASS",
        )
    current_runs = [run for run in runs if run["dataset"] != "old_P0X2RCS05"]
    max_ordinary = max(
        abs(record["zero_reject_ordinary_reward"])
        for run in current_runs for record in run["records"]
    )
    rejected_actions = sum(
        record["zero_reject_count"] for run in current_runs for record in run["records"]
    )
    add_check(
        checks, "zero_reject",
        "ordinary rejected actions are exact zero-reward no-ops",
        max_ordinary <= 1.0e-12 and rejected_actions > 0,
        "training_history.csv fields ori6_zero_reject_count/total_reward",
        f"rejected_actions={rejected_actions}; max_abs_ordinary_reward={max_ordinary:.17g}",
    )
    add_check(
        checks, "process", "both runners and all trainers exited",
        not readiness["live_processes"],
        "runner statuses plus Win32 process command lines", "no live writer process",
    )

    torus = by_key[("r1_zero_reject", "torus2", 4203)]
    water = by_key[("water_fallback_far_0p12", "water1", 4301)]
    add_check(
        checks, "endpoint", "R1 torus selected endpoint is natural 10000",
        torus["endpoint"] == 10000
        and torus["endpoint_authority"]["natural_10000_recovered"],
        torus["run_directory"], json.dumps(torus["endpoint_authority"]),
    )
    expected_fallback_endpoint = int(readiness["fallback_final"].get("endpoint", 0) or 0)
    add_check(
        checks, "endpoint", "fallback water endpoint agrees with runner final authority",
        water["endpoint"] == expected_fallback_endpoint,
        water["run_directory"],
        f"postprocessor={water['endpoint']}; runner_final={expected_fallback_endpoint}",
    )
    if water["endpoint"] == 10000:
        add_check(
            checks, "endpoint", "fallback water natural 10000 complete set recovered",
            water["endpoint_authority"]["natural_10000_recovered"],
            water["run_directory"], json.dumps(water["endpoint_authority"]),
        )

    torus_modes = {item["evaluation_mode"] for item in torus["evaluations"]}
    water_modes = {item["evaluation_mode"] for item in water["evaluations"]}
    add_check(
        checks, "evaluation",
        "torus final stochastic held-out and deterministic-single are separated",
        torus_modes == {"stochastic_heldout", "deterministic_final_single"},
        torus["run_directory"], ",".join(sorted(torus_modes)),
    )
    if water["endpoint"] == 10000:
        add_check(
            checks, "evaluation",
            "water final stochastic held-out and deterministic-single are separated",
            water_modes == {"stochastic_heldout", "deterministic_final_single"},
            water["run_directory"], ",".join(sorted(water_modes)),
        )
    else:
        add_check(
            checks, "evaluation",
            "water non-10000 endpoint is explicitly periodic, not mislabeled held-out",
            "stochastic_heldout" not in water_modes
            and "deterministic_final_single" not in water_modes,
            water["run_directory"], ",".join(sorted(water_modes)),
        )

    chronology = chronological_evidence(readiness, runs)
    comparisons = {}
    for task, selected_run in (("torus2", torus), ("water1", water)):
        old = by_key[("old_P0X2RCS05", task, 4101)]
        first = selected_run["first_1024"]
        last = selected_run["last_1024"]
        old_last = old["last_1024"]
        comparisons[task] = {
            "scope_note": (
                "descriptive unmatched-seed comparison; current task-specific tuning "
                "and selected-seed production are not a paired causal estimate"
            ),
            "current_dataset": selected_run["dataset"],
            "current_seed": selected_run["seed"],
            "current_endpoint": selected_run["endpoint"],
            "old_seed": old["seed"],
            "old_endpoint": old["endpoint"],
            "training_change_first_to_last": {
                "mean_return_delta": last["mean_return"] - first["mean_return"],
                "median_flatness_delta": last["median_flatness"] - first["median_flatness"],
                "median_spread_error_delta": last["median_spread_error"] - first["median_spread_error"],
                "success_rate_delta": last["success_rate"] - first["success_rate"],
                "constructive_rate_delta": last["constructive_rate"] - first["constructive_rate"],
            },
            "current_minus_old_last1024": {
                "mean_return_delta": last["mean_return"] - old_last["mean_return"],
                "median_flatness_delta": last["median_flatness"] - old_last["median_flatness"],
                "median_spread_error_delta": last["median_spread_error"] - old_last["median_spread_error"],
                "success_rate_delta": last["success_rate"] - old_last["success_rate"],
                "constructive_rate_delta": last["constructive_rate"] - old_last["constructive_rate"],
            },
        }

    payload = {
        "schema_version": 1,
        "analysis_type": "ori6_zero_reject_two_phase_combined_final",
        "generated_utc": utc_now(),
        "roots": {
            "r1": str(R1_ROOT.resolve()),
            "water_fallback": str(FALLBACK_ROOT.resolve()),
            "old_reference": str(OLD_ROOT.resolve()),
        },
        "selection": selections,
        "runner_final_summaries": {
            "r1": readiness["r1_final"],
            "water_fallback": readiness["fallback_final"],
        },
        "runner_statuses": {
            "r1": readiness["r1_statuses"],
            "water_fallback": readiness["fallback_statuses"],
        },
        "gates": gate_info,
        "causal_chronology": chronology,
        "runs": runs,
        "evaluations": [
            evaluation for run in runs for evaluation in run["evaluations"]
        ],
        "comparisons": comparisons,
        "integrity": {
            "pass": all(item["pass"] for item in checks),
            "checks": checks,
            "ordinary_zero_reject_max_abs_reward": max_ordinary,
            "ordinary_zero_reject_action_count": rejected_actions,
            "natural_10000_recoveries": {
                run["label"]: run["endpoint_authority"]
                for run in runs if run["endpoint_authority"]["natural_10000_recovered"]
            },
        },
    }
    input_paths = collect_input_paths(runs, gate_info)
    payload["input_sha256"] = hash_paths(input_paths)
    payload["input_hash_set_sha256"] = canonical_digest(payload["input_sha256"])
    r1_binding = read_json(R1_ROOT / "execution_manifest_binding.json", {})
    fallback_binding = read_json(FALLBACK_ROOT / "execution_manifest_binding.json", {})
    authority_paths = {
        "r1_execution_manifest": Path(str(r1_binding.get("manifest_path", ""))),
        "fallback_execution_manifest": Path(str(fallback_binding.get("manifest_path", ""))),
        "r1_torus_formal_gate": Path(gate_info["evidence_paths"][0]),
        "r1_water_formal_gate": Path(gate_info["evidence_paths"][1]),
        "fallback_water_formal_gate": Path(gate_info["evidence_paths"][2]),
        "combined_postprocessor_source": Path(__file__).resolve(),
        "combined_postprocessor_test": (
            SCRIPT_DIRECTORY / "test_finalize_ori_codex6_zero_reject_combined.py"
        ),
    }
    payload["authority_sha256"] = {
        label: {
            "path": str(path.resolve()),
            "sha256": payload["input_sha256"].get(str(path.resolve())),
        }
        for label, path in authority_paths.items()
    }
    missing_authority_hashes = [
        label for label, item in payload["authority_sha256"].items()
        if not item["sha256"]
    ]
    if missing_authority_hashes:
        raise RuntimeError(
            "authority file missing from frozen input hash set: "
            + ", ".join(missing_authority_hashes)
        )
    return payload


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_windows_csv(payload: dict, path: Path) -> None:
    rows = []
    for run in payload["runs"]:
        for window in run["windows_512"]:
            rows.append({
                "window_kind": (
                    "nonoverlapping_512" if window["episode_count"] == 512
                    else "nonoverlapping_partial"
                ),
                "dataset": run["dataset"], "task": run["task"],
                "seed": run["seed"], "role": run["role"],
                "label": run["label"], "run_endpoint": run["endpoint"],
                **window,
            })
        trailing = run["last_512"]
        final_nonoverlap = run["windows_512"][-1]
        if (
            trailing["start_episode"] != final_nonoverlap["start_episode"]
            or trailing["end_episode"] != final_nonoverlap["end_episode"]
        ):
            rows.append({
                "window_kind": "trailing_512_overlap",
                "dataset": run["dataset"], "task": run["task"],
                "seed": run["seed"], "role": run["role"],
                "label": run["label"], "run_endpoint": run["endpoint"],
                **trailing,
            })
    write_csv(path, rows)


def write_evaluations_csv(payload: dict, path: Path) -> None:
    write_csv(path, payload["evaluations"])


def write_integrity_csv(payload: dict, path: Path) -> None:
    write_csv(path, payload["integrity"]["checks"])


def write_gates_csv(payload: dict, path: Path) -> None:
    rows = []
    for row in decision_rows(R1_ROOT, "R1 zero-reject"):
        rows.append({"row_type": "decision", **row})
    for row in decision_rows(FALLBACK_ROOT, "water fallback far=0.12"):
        rows.append({"row_type": "decision", **row})
    groups = (
        ("R1 zero-reject", "torus2", "r1_torus"),
        ("R1 far=0.40", "water1", "r1_water"),
        ("water fallback far=0.12", "water1", "fallback_water"),
    )
    for phase, task, key in groups:
        for report in payload["gates"]["seed_reports"][key]:
            recent = report["last_1024"]
            late = report["last_512"]
            components = report.get("promising_components", {})
            rows.append({
                "row_type": "seed_at_formal_4096",
                "phase": phase,
                "task": task,
                "seed": report["seed"],
                "checkpoint_episode": 4096,
                "promising_at_4096": report.get("promising_at_4096"),
                "evidence": components.get("evidence"),
                "persistence": components.get("persistence"),
                "quality": components.get("quality"),
                "trend": components.get("trend"),
                "last_1024_mean_return": recent["mean_return"],
                "last_1024_median_flatness": recent["median_flatness"],
                "last_1024_median_spread_error": recent["median_spread_error"],
                "last_1024_constructive_rate": recent["constructive_rate"],
                "last_1024_terminal_joint_rate": recent["valid_joint_rate"],
                "last_1024_strict_count": recent["strict_count"],
                "last_512_constructive_rate": late["constructive_rate"],
                "last_512_terminal_joint_rate": late["valid_joint_rate"],
                "last_512_strict_count": late["strict_count"],
            })
    write_csv(path, rows)


def rolling(values: Sequence[float], window: int = 200):
    import numpy as np
    array = np.asarray(values, dtype=float)
    if not len(array):
        return array
    width = min(window, len(array))
    return np.convolve(array, np.ones(width) / width, mode="valid")


def run_style(run: dict) -> dict:
    if run["dataset"] == "old_P0X2RCS05":
        return {"color": "#111111", "linestyle": "--", "alpha": 0.90, "linewidth": 1.8}
    if run["dataset"] == "water_fallback_far_0p12":
        if run["role"] == "selected_production":
            return {"color": "#16853b", "linestyle": "-", "alpha": 1.0, "linewidth": 2.2}
        colors = {4301: "#67b77a", 4302: "#91c79e", 4303: "#b2d9ba"}
        return {"color": colors[run["seed"]], "linestyle": "-", "alpha": 0.55, "linewidth": 1.0}
    if run["task"] == "water1":
        colors = {4201: "#c45a17", 4202: "#df7f35", 4203: "#eba56e"}
        return {"color": colors[run["seed"]], "linestyle": "-", "alpha": 0.58, "linewidth": 1.1}
    if run["role"] == "selected_production":
        return {"color": "#2468b4", "linestyle": "-", "alpha": 1.0, "linewidth": 2.2}
    colors = {4201: "#79a7d8", 4202: "#9cbbe0", 4203: "#2468b4"}
    return {"color": colors[run["seed"]], "linestyle": "-", "alpha": 0.48, "linewidth": 1.0}


def _plot_label(run: dict) -> str:
    if run["dataset"] == "old_P0X2RCS05":
        return "old seed4101 (6144)"
    if run["dataset"] == "water_fallback_far_0p12":
        suffix = "selected" if run["role"] == "selected_production" else "screen"
        return f"fallback .12 s{run['seed']} {suffix}"
    if run["task"] == "water1":
        return f"R1 .40 s{run['seed']} rejected"
    suffix = "selected" if run["role"] == "selected_production" else "screen"
    return f"R1 ZR s{run['seed']} {suffix}"


def plot_learning(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(6, 2, figsize=(15.5, 18.5), constrained_layout=True)
    metric_labels = (
        "episodic return (MA200)",
        "terminal flatness F (MA200)",
        "terminal spread error eD (MA200)",
        "strict success / terminal joint / constructive (solid/dashed/dotted), MA200",
        "wrong-plane or collapse (solid) / boundary-plane (dotted), MA200",
        "rejected-action rate (MA200)",
    )
    for column, task in enumerate(TASKS):
        task_runs = [run for run in payload["runs"] if run["task"] == task]
        for run in task_runs:
            records = run["records"]
            width = min(200, len(records))
            x = np.asarray([row["episode"] for row in records], dtype=float)[width - 1:]
            style = run_style(run)
            label = _plot_label(run)
            metrics = (
                [row["reward"] for row in records],
                [max(1.0e-8, row["flatness"]) for row in records],
                [max(1.0e-8, row["spread_error"]) for row in records],
                [float(row["success"]) for row in records],
                [float(
                    row["torus_collapse_basin"] if task == "torus2"
                    else row["wrong_plane_basin"]
                ) for row in records],
                [row["reject_rate"] for row in records],
            )
            for index, values in enumerate(metrics):
                axes[index, column].plot(
                    x, rolling(values, width), label=label if index == 0 else None,
                    **style,
                )
            axes[3, column].plot(
                x, rolling([float(row["joint"]) for row in records], width),
                color=style["color"], linestyle="--", alpha=style["alpha"],
                linewidth=style["linewidth"],
            )
            axes[3, column].plot(
                x, rolling([float(row["constructive"]) for row in records], width),
                color=style["color"], linestyle=":", alpha=style["alpha"],
                linewidth=style["linewidth"],
            )
            if task == "water1":
                axes[4, column].plot(
                    x, rolling([float(row["boundary_plane_basin"]) for row in records], width),
                    color=style["color"], linestyle=":", alpha=style["alpha"],
                    linewidth=style["linewidth"],
                )
            if run["role"] == "selected_production":
                axes[0, column].annotate(
                    f"selected endpoint {run['endpoint']}",
                    xy=(run["endpoint"], rolling(metrics[0], width)[-1]),
                    xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=8,
                    color=style["color"],
                )
        for row, label in enumerate(metric_labels):
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlabel("episode")
            axes[row, column].grid(alpha=0.20)
            axes[row, column].axvline(4096, color="#777777", linewidth=0.8, linestyle=":")
        axes[1, column].set_yscale("log")
        axes[2, column].set_yscale("log")
        axes[2, column].axhline(0.05, color="#555555", linewidth=0.8, linestyle="--")
        axes[2, column].axhline(0.10, color="#888888", linewidth=0.8, linestyle=":")
        axes[3, column].set_ylim(-0.03, 1.05)
        axes[4, column].set_ylim(-0.03, 1.05)
        axes[0, column].set_title(
            "torus D=2: R1 selected production" if task == "torus2"
            else "water D=1: R1 .40 reject -> fallback .12"
        )
        axes[0, column].legend(fontsize=7.5, loc="best", ncol=1)
    figure.suptitle(
        "Zero-reject learning curves: reward, geometry, success/joint persistence, basin occupancy, and rejects"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_optimizer(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(6, 2, figsize=(15.5, 18), constrained_layout=True)
    labels = (
        "critic loss (MA8 updates)",
        "critic gradient pre-clip (MA8)",
        "actor gradient pre-clip (MA8)",
        "entropy (MA8 updates)",
        "applied actor learning rate",
        "rejected-action rate (MA200)",
    )
    for column, task in enumerate(TASKS):
        task_runs = [run for run in payload["runs"] if run["task"] == task]
        for run in task_runs:
            style = run_style(run)
            label = _plot_label(run)
            updates = run["updates"]
            episode = np.asarray([staged._int(row, ("episode",)) for row in updates])
            width = min(8, len(updates))
            x = episode[width - 1:]
            update_metrics = (
                [max(1.0e-12, staged._float(row, ("critic_loss",))) for row in updates],
                [staged._float(row, ("critic_gradient_norm",)) for row in updates],
                [staged._float(row, ("actor_gradient_norm",)) for row in updates],
                [staged._float(row, ("entropy",)) for row in updates],
            )
            for index, values in enumerate(update_metrics):
                axes[index, column].plot(
                    x, rolling(values, width), label=label if index == 0 else None,
                    **style,
                )
            axes[4, column].plot(
                episode,
                [staged._float(row, ("actor_learning_rate",)) for row in updates],
                label=None, **style,
            )
            records = run["records"]
            record_width = min(200, len(records))
            record_x = np.asarray([row["episode"] for row in records])[record_width - 1:]
            axes[5, column].plot(
                record_x, rolling([row["reject_rate"] for row in records], record_width),
                label=None, **style,
            )
        for row, label in enumerate(labels):
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlabel("episode")
            axes[row, column].grid(alpha=0.20)
            axes[row, column].axvline(4096, color="#777777", linewidth=0.8, linestyle=":")
        axes[0, column].set_yscale("log")
        axes[1, column].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        axes[2, column].axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
        axes[0, column].set_title(task)
        axes[0, column].legend(fontsize=7.5, loc="best")
    no_op = payload["integrity"]["ordinary_zero_reject_max_abs_reward"]
    count = payload["integrity"]["ordinary_zero_reject_action_count"]
    figure.suptitle(
        f"Optimizer/reject diagnostics; {count:,} rejected no-op actions, "
        f"max |ordinary reject reward|={no_op:.1e}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_evaluations(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    metrics = (
        ("success_rate", "success (stochastic rate / deterministic 0 or 1)"),
        ("mean_flatness", "terminal flatness F"),
        ("mean_spread_error", "terminal spread error eD"),
        ("mean_return", "episodic return"),
    )
    for column, task in enumerate(TASKS):
        selected = next(
            run for run in payload["runs"]
            if run["task"] == task and run["role"] == "selected_production"
        )
        old = next(
            run for run in payload["runs"]
            if run["task"] == task and run["dataset"] == "old_P0X2RCS05"
        )
        groups = [(0.0, old), (1.0, selected)]
        for x_base, run in groups:
            for evaluation in run["evaluations"]:
                mode = evaluation["evaluation_mode"]
                if mode == "stochastic_heldout":
                    marker, offset, face = "o", -0.10, run_style(run)["color"]
                    mode_label = "final held-out stochastic (n=64)"
                elif mode == "periodic_stochastic":
                    marker, offset, face = "^", -0.10, "none"
                    mode_label = "latest periodic stochastic"
                else:
                    marker, offset, face = "X", 0.10, run_style(run)["color"]
                    mode_label = "single deterministic rollout"
                for row, (key, ylabel) in enumerate(metrics):
                    value = evaluation[key]
                    axes[row, column].scatter(
                        [x_base + offset], [max(value, 1.0e-9) if row in (1, 2) else value],
                        marker=marker, s=70, edgecolor=run_style(run)["color"],
                        facecolor=face, linewidth=1.2,
                        label=mode_label if row == 0 and x_base == 1.0 else None,
                        zorder=3,
                    )
                    if row == 0:
                        axes[row, column].annotate(
                            f"n={evaluation['sample_count']}",
                            (x_base + offset, value), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8,
                        )
        for row, (_, ylabel) in enumerate(metrics):
            axes[row, column].set_ylabel(ylabel)
            axes[row, column].set_xticks([0, 1], ["old@6144", f"selected@{selected['endpoint']}"])
            axes[row, column].grid(axis="y", alpha=0.22)
        axes[1, column].set_yscale("log")
        axes[2, column].set_yscale("log")
        axes[2, column].axhline(0.05, color="#555555", linewidth=0.8, linestyle="--")
        axes[0, column].set_title(task)
        axes[0, column].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Policy evaluation: stochastic estimates and single deterministic rollouts are not interchangeable"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"


def pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: dict, path: Path) -> None:
    by_key = run_map(payload["runs"])
    torus = by_key[("r1_zero_reject", "torus2", 4203)]
    water = by_key[("water_fallback_far_0p12", "water1", 4301)]
    torus_old = by_key[("old_P0X2RCS05", "torus2", 4101)]
    water_old = by_key[("old_P0X2RCS05", "water1", 4101)]
    lines = [
        "# zero-reject二段階実験 最終報告",
        "",
        f"生成時刻（UTC）: {payload['generated_utc']}",
        "",
        "## 結論",
        "",
        (
            f"- torus D=2 はR1の4096 episode・3 seed gateを通過してseed4203を選び、"
            f"最終ep{torus['endpoint']}まで到達した。最終1024 episodeの成功率は"
            f"{pct(torus['last_1024']['success_rate'])}、平均returnは"
            f"{fmt(torus['last_1024']['mean_return'])}、terminal F中央値は"
            f"{fmt(torus['last_1024']['median_flatness'])}、spread error中央値は"
            f"{fmt(torus['last_1024']['median_spread_error'])}。"
        ),
        (
            f"- water D=1 はR1（near-miss spread far=0.40）が4096 gateで正式棄却された。"
            f"その後、保存trajectoryの反実仮想だけを根拠にfar=0.12へ一項目変更したfresh fallbackを実行し、"
            f"2/3 seedでgateを通過してseed4301を選択、最終ep{water['endpoint']}まで解析した。"
            f"最終1024 episodeの成功率は{pct(water['last_1024']['success_rate'])}、"
            f"平均returnは{fmt(water['last_1024']['mean_return'])}、terminal F中央値は"
            f"{fmt(water['last_1024']['median_flatness'])}、spread error中央値は"
            f"{fmt(water['last_1024']['median_spread_error'])}。"
        ),
        "- 4096までは3 seedのscreen evidence、4096以降は選抜1 seedのproduction diagnosticであり、3 seed安定性の確認ではない。",
        "- 旧P0X2RCS05 seed4101@6144との比較はseedも設定も一致しない記述的比較であり、因果効果量として扱わない。",
        "",
        "## 因果的な検討順序",
        "",
    ]
    for item in payload["causal_chronology"]:
        lines.append(f"{item['sequence']}. {item['description_ja']}")
    lines.extend([
        "",
        "この順序により、water far=0.12はR1 far=0.40の4096正式棄却後にのみ導入されており、"
        "同じrun内の途中変更ではない。したがって両者はfresh cohortとして分離して読む。",
        "",
        "## 学習曲線の数値要約",
        "",
        "| task / run | endpoint | first1024 return / F / eD | last1024 return / F / eD | success | constructive | terminal joint | strict hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in (torus, water, torus_old, water_old):
        first, last = run["first_1024"], run["last_1024"]
        lines.append(
            f"| {run['task']} / {run['label']} | {run['endpoint']} | "
            f"{fmt(first['mean_return'])} / {fmt(first['median_flatness'])} / {fmt(first['median_spread_error'])} | "
            f"{fmt(last['mean_return'])} / {fmt(last['median_flatness'])} / {fmt(last['median_spread_error'])} | "
            f"{pct(last['success_rate'])} | {pct(last['constructive_rate'])} | "
            f"{pct(last['valid_joint_rate'])} | {last['strict_count']} |"
        )
    lines.extend([
        "",
        "### basin占有率（最終1024 episode）",
        "",
        "| task / run | wrong-plane | boundary-plane 0.10<eD<=0.15 | torus collapse F>=0.9,eD>=0.30 |",
        "|---|---:|---:|---:|",
    ])
    for run in (torus, water, torus_old, water_old):
        last = run["last_1024"]
        lines.append(
            f"| {run['task']} / {run['label']} | "
            f"{pct(last['wrong_plane_basin_rate'])} | "
            f"{pct(last['boundary_plane_basin_rate'])} | "
            f"{pct(last['torus_collapse_basin_rate'])} |"
        )
    lines.extend([
        "",
        "wrong-planeはwaterでF<=0.5かつeD>=0.30、boundary-planeはwaterで"
        "F<=0.5かつ0.10<eD<=0.15、torus collapseはF>=0.9かつeD>=0.30として集計した。",
        "",
        "FとeDは小さいほどよい。learning curveではreturn、F、eD、success/constructive、reject率を"
        "同じepisode軸で表示し、terminal jointとbasin占有率も追加し、4096 gateを縦線で示した。",
        "",
        "## 学習前半→後半と旧runとの差（記述的）",
        "",
        "| task | first1024→last1024 Δreturn / ΔF / ΔeD / Δsuccess | current last1024 - old last1024 Δreturn / ΔF / ΔeD / Δsuccess |",
        "|---|---:|---:|",
    ])
    for task in TASKS:
        comparison = payload["comparisons"][task]
        training = comparison["training_change_first_to_last"]
        old_delta = comparison["current_minus_old_last1024"]
        lines.append(
            f"| {task} | {fmt(training['mean_return_delta'])} / "
            f"{fmt(training['median_flatness_delta'])} / "
            f"{fmt(training['median_spread_error_delta'])} / "
            f"{pct(training['success_rate_delta'])} | "
            f"{fmt(old_delta['mean_return_delta'])} / "
            f"{fmt(old_delta['median_flatness_delta'])} / "
            f"{fmt(old_delta['median_spread_error_delta'])} / "
            f"{pct(old_delta['success_rate_delta'])} |"
        )
    lines.extend([
        "",
        "returnとsuccessは正、FとeDは負が改善方向。ただしoldとの差はseed4101対選抜seedで、"
        "task-specific設定も異なるため、あくまで記述的比較である。",
        "",
        "## 4096 seed gateと選択",
        "",
        "| phase | task | seed | promising | last1024 return | F median | eD median | constructive | terminal joint | strict |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ])
    gate_groups = (
        ("R1", "torus2", R1_SEEDS, "r1_torus"),
        ("R1 far=0.40", "water1", R1_SEEDS, "r1_water"),
        ("fallback far=0.12", "water1", FALLBACK_SEEDS, "fallback_water"),
    )
    for phase, task, seeds, gate_key in gate_groups:
        gate_payload = (
            payload["gates"]["r1_torus"] if task == "torus2" else
            payload["gates"]["r1_water"] if gate_key == "r1_water" else
            payload["gates"]["fallback_water"]
        )
        promising = set(gate_payload.get("promising_seeds", []))
        reports = {
            report["seed"]: report
            for report in payload["gates"]["seed_reports"][gate_key]
        }
        for seed in seeds:
            last = reports[seed]["last_1024"]
            lines.append(
                f"| {phase} | {task} | {seed} | {seed in promising} | "
                f"{fmt(last['mean_return'])} | {fmt(last['median_flatness'])} | "
                f"{fmt(last['median_spread_error'])} | {pct(last['constructive_rate'])} | "
                f"{pct(last['valid_joint_rate'])} | {last['strict_count']} |"
            )
    lines.extend([
        "",
        f"R1 torusはseed4203、fallback waterはseed4301が、promising seed内で"
        "last512/last1024 persistenceを優先する事前登録順序により選択された。"
        "R1 waterはpromising seedが2未満だったため選択せず停止した。",
        "",
        "## zero-reject no-opとreward整合性",
        "",
        f"- 記録された通常reject action数: {payload['integrity']['ordinary_zero_reject_action_count']:,}",
        f"- `ori6_zero_reject_total_reward` の最大絶対値: {payload['integrity']['ordinary_zero_reject_max_abs_reward']:.3e}",
        "- 通常crossing候補はlast-safe stateを維持し、accepted width=0、displacement=0、reward=0としてstepを消費する。",
        "- horizonでrejectした場合だけ通常timeout `-1 + q(last_safe)` を別成分として一度与える。通常no-op reward=0とは区別した。",
        "",
        "## optimizer診断（各runの最終512 episode）",
        "",
        "| run | critic loss mean/p95/max | critic grad mean/p95/max; >1 | actor grad mean/p95/max; >0.5 | entropy | actor LR | reject rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in [item for item in payload["runs"] if item["dataset"] != "old_P0X2RCS05"]:
        u = run["last_512"]
        lines.append(
            f"| {run['label']} | {fmt(u['critic_loss_mean'])}/{fmt(u['critic_loss_p95'])}/{fmt(u['critic_loss_max'])} | "
            f"{fmt(u['critic_gradient_norm_mean'])}/{fmt(u['critic_gradient_norm_p95'])}/{fmt(u['critic_gradient_norm_max'])}; "
            f"{pct(u['critic_gradient_over_clip_fraction'])} | "
            f"{fmt(u['actor_gradient_norm_mean'])}/{fmt(u['actor_gradient_norm_p95'])}/{fmt(u['actor_gradient_norm_max'])}; "
            f"{pct(u['actor_gradient_over_clip_fraction'])} | {fmt(u['entropy_mean'])} | "
            f"{fmt(u['actor_learning_rate_last'])} | {pct(u['mean_reject_rate'])} |"
        )
    lines.extend([
        "",
        "gradient normはclip前の記録であり、`>1`（critic）と`>0.5`（actor）は閾値超過更新の割合。"
        "critic lossの一時的shock、actor更新量、entropy、適用LRの変化を別々に確認できる。",
        "",
        "## 最終policy評価：stochasticとdeterministicの区別",
        "",
        "| run | mode | episode | n | success | return | F | eD | 解釈 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    evaluation_runs = {torus["label"], water["label"], torus_old["label"], water_old["label"]}
    for evaluation in payload["evaluations"]:
        if evaluation["run_label"] not in evaluation_runs:
            continue
        lines.append(
            f"| {evaluation['run_label']} | {evaluation['evaluation_mode']} | "
            f"{evaluation['episode']} | {evaluation['sample_count']} | "
            f"{pct(evaluation['success_rate'])} | {fmt(evaluation['mean_return'])} | "
            f"{fmt(evaluation['mean_flatness'])} | {fmt(evaluation['mean_spread_error'])} | "
            f"{evaluation['interpretation']} |"
        )
    lines.extend([
        "",
        "`stochastic_heldout`はfinal evaluation seedでの64 rollout平均である。"
        "`deterministic_final_single`は平均成功率ではなく、mean actionによる1 trajectoryの結果である。"
        "したがって、両者が異なっても矛盾ではなく、探索分散を含むpolicyとmean policyの挙動差を示す。"
        "10000未満で停止したrunについてはfinal held-outと呼ばず、最新periodic diagnosticと明記した。",
        "",
        "## endpoint authority",
        "",
    ])
    for run in (torus, water):
        authority = run["endpoint_authority"]
        lines.append(
            f"- {run['label']}: ep{run['endpoint']} / `{authority['authority']}`; "
            f"checkpoint={authority.get('checkpoint_episode')}, policy={authority.get('policy_episode')}, "
            f"history={authority.get('history_episode')}, update={authority.get('update_episode')}。"
        )
        if authority.get("periodic_checkpoint_gap"):
            lines.append(
                f"  - checkpoint.pt={authority.get('checkpoint_episode')}でも、"
                "policy.pt・training_history・update_history・summary・configが"
                "すべて10000で完結しているため、自然終了10000を優先して回収した。"
            )
    lines.extend([
        "",
        "## 整合性・再現可能性",
        "",
        f"- 全integrity check: {'PASS' if payload['integrity']['pass'] else 'FAIL'}",
        f"- input hash set SHA-256: `{payload['input_hash_set_sha256']}`",
        "- `final_metrics.json`に全input fileの絶対pathとSHA-256、endpoint authority、gate、selection、window統計を保存した。",
        "- `integrity.json` / `integrity_checks.csv`に個別checkを保存した。",
        "- `artifact_sha256.json`にこの報告・CSV・JSON・PNGとpostprocessor/test sourceの最終SHA-256を保存する。",
        "",
        "主要authority fileのSHA-256:",
        "",
    ])
    for label, item in payload.get("authority_sha256", {}).items():
        lines.append(f"- {label}: `{item['sha256']}` — `{item['path']}`")
    lines.extend([
        "",
        "## 出力",
        "",
        "- `learning_curve_episode_comparison.png`: return、F、eD、strict/joint/constructive、basin占有率、reject率",
        "- `optimizer_reject_diagnostics.png`: critic loss/grad、actor grad、entropy、LR、reject率",
        "- `policy_evaluation_comparison.png`: stochastic estimateとdeterministic single trajectory",
        "- `final_windows_512.csv`: 全runの非重複512 window統計（端数endpointでは真の最終512 overlap行も追加）",
        "- `final_seed_gates.csv`: gate・selection・collapse判定履歴",
        "- `final_evaluations.csv`: evaluation modeとsample数を分離したpolicy評価",
        "- `causal_chronology.csv`: 二段階実験の検討順序とauthority",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_input_hashes(payload: dict) -> None:
    current = hash_paths([Path(path) for path in payload["input_sha256"]])
    if current != payload["input_sha256"]:
        changed = sorted(
            path for path in set(current) | set(payload["input_sha256"])
            if current.get(path) != payload["input_sha256"].get(path)
        )
        raise RuntimeError("input changed during finalization: " + ", ".join(changed))


def write_all(payload: dict, output: Path) -> dict:
    resolved = output.resolve()
    for active_root in (R1_ROOT.resolve(), FALLBACK_ROOT.resolve()):
        if resolved == active_root or active_root in resolved.parents:
            raise RuntimeError("combined finalizer output must stay outside active roots")
    if output.exists():
        raise FileExistsError(f"fresh final output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "metrics": output / "final_metrics.json",
        "windows": output / "final_windows_512.csv",
        "gates": output / "final_seed_gates.csv",
        "evaluations": output / "final_evaluations.csv",
        "chronology": output / "causal_chronology.csv",
        "integrity": output / "integrity.json",
        "integrity_csv": output / "integrity_checks.csv",
        "report": output / "final_report_ja.md",
        "learning_plot": output / "learning_curve_episode_comparison.png",
        "optimizer_plot": output / "optimizer_reject_diagnostics.png",
        "evaluation_plot": output / "policy_evaluation_comparison.png",
    }
    atomic_json(paths["metrics"], serialisable(payload))
    write_windows_csv(payload, paths["windows"])
    write_gates_csv(payload, paths["gates"])
    write_evaluations_csv(payload, paths["evaluations"])
    write_csv(paths["chronology"], payload["causal_chronology"])
    atomic_json(paths["integrity"], {
        "schema_version": 1,
        "generated_utc": payload["generated_utc"],
        **payload["integrity"],
        "input_hash_set_sha256": payload["input_hash_set_sha256"],
        "input_sha256": payload["input_sha256"],
    })
    write_integrity_csv(payload, paths["integrity_csv"])
    plot_learning(payload, paths["learning_plot"])
    plot_optimizer(payload, paths["optimizer_plot"])
    plot_evaluations(payload, paths["evaluation_plot"])
    write_report(payload, paths["report"])
    verify_input_hashes(payload)
    artifact_hashes = {
        str(path.resolve()): file_sha256(path) for path in paths.values()
    }
    source_hashes = {}
    for source in (
        Path(__file__).resolve(),
        SCRIPT_DIRECTORY / "test_finalize_ori_codex6_zero_reject_combined.py",
    ):
        if source.is_file():
            source_hashes[str(source.resolve())] = file_sha256(source)
    artifact_manifest = output / "artifact_sha256.json"
    atomic_json(artifact_manifest, {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "input_hash_set_sha256": payload["input_hash_set_sha256"],
        "artifact_sha256": artifact_hashes,
        "postprocessor_source_sha256": source_hashes,
    })
    return {
        **{key: str(value.resolve()) for key, value in paths.items()},
        "artifact_manifest": str(artifact_manifest.resolve()),
        "artifact_manifest_sha256": file_sha256(artifact_manifest),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--check-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    if options.check_ready:
        try:
            ready = assert_ready()
            if options.output.exists():
                raise RuntimeError(
                    f"fresh final output already exists: {options.output.resolve()}"
                )
            print(json.dumps({
                "ready": True,
                "r1_selected": ready["r1_final"].get("selected_seeds"),
                "fallback_selected": ready["fallback_final"].get("selected_seed"),
                "output_absent": True,
            }, ensure_ascii=False, indent=2))
            return 0
        except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as error:
            print(json.dumps({"ready": False, "reason": str(error)}, ensure_ascii=False, indent=2))
            return 3
    payload = build_payload()
    if not payload["integrity"]["pass"]:
        failed = [item["check"] for item in payload["integrity"]["checks"] if not item["pass"]]
        raise RuntimeError("integrity checks failed; refusing final artifacts: " + "; ".join(failed))
    artifacts = write_all(payload, options.output.resolve())
    print(json.dumps({
        "integrity_pass": payload["integrity"]["pass"],
        "torus_endpoint": next(
            run["endpoint"] for run in payload["runs"]
            if run["dataset"] == "r1_zero_reject" and run["task"] == "torus2"
            and run["seed"] == 4203
        ),
        "water_endpoint": next(
            run["endpoint"] for run in payload["runs"]
            if run["dataset"] == "water_fallback_far_0p12" and run["seed"] == 4301
        ),
        "artifacts": artifacts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
