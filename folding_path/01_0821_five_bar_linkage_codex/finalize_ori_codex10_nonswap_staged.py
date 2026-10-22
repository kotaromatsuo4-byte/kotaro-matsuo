"""Fail-closed finalizer for the immutable C10NS1 non-swap campaign.

The finalizer is strictly additive: it never launches, stops, resumes, or
modifies a training run.  Publication is allowed only after the supervisor has
committed ``runner_report.json`` and no bound runner/trainer PID remains.  All
screen decisions are independently recomputed from their exact checkpoint
prefixes; rows written after a causal gate are recorded as overrun and excluded
from formal metrics and plots.
"""

from __future__ import annotations

import argparse
import bisect
from collections import deque
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Iterable, Sequence

import analyze_ori_codex10_nonswap_staged as staged
import run_ori_codex10_nonswap_staged as bound_runner


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY / "experiments_ori10" / "C10NS1"
DEFAULT_OUTPUT = SCRIPT_DIRECTORY / "experiments_ori10" / "analysis" / "C10NS1_final"
MANIFEST_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori10" / "manifests"
EXPECTED_MANIFEST_FINGERPRINT = (
    "7f1a9303048ba683d01f8d8539ed30444e8800ff5952ec498890e694410165ab"
)
EXPECTED_PROTOCOL = "codex6_p0x2rcs05_nonswap_optimizer_screen_to_10000"
TASKS = ("five_d2", "five_d3", "watersqrt2")
TASK_ARMS = {"five_d2": ("B", "W"), "five_d3": ("B", "W"),
             "watersqrt2": ("B", "E")}
SEEDS = (9101, 9102, 9103)
SCREEN_CHECKPOINTS = (512, 1024, 2048, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
ROLLING_WINDOWS = (256, 512)
FINALIZER_NAMES = (
    "finalize_ori_codex10_nonswap_staged.py",
    "test_finalize_ori_codex10_nonswap_staged.py",
)
SOURCE_NAMES = (
    "analyze_ori_codex10_nonswap_staged.py", "collision_3d.py",
    "ori_codex.py", "ori_codex2.py", "ori_codex3.py", "ori_codex4.py",
    "ori_codex4_3d.py", "ori_codex5.py", "ori_codex5_3d.py",
    "ori_codex6.py", "ori_codex6_3d.py", "ori_codex6_rejected_crossing.py",
    "ori_codex6_rejected_crossing_3d.py", "ori_codex6_spectral_subspace.py",
    "ori_codex6_spectral_subspace_3d.py", "run_ori_codex10_nonswap_staged.py",
    "single_solution_common.py",
)
JOB_INPUT_NAMES = (
    "run_config.json", "training_history.csv", "update_history.csv",
    "stochastic_evaluation_history.csv", "checkpoint.pt", "summary.json",
    "policy.pt", "evaluation_summary.csv",
)
REQUIRED_JOB_INPUT_NAMES = (
    "run_config.json", "training_history.csv", "update_history.csv",
    "stochastic_evaluation_history.csv", "checkpoint.pt",
)
HISTORICAL_ROOT = SCRIPT_DIRECTORY / "experiments_ori5" / "full"
EXPECTED_BINDING_KEYS = {
    "manifest_fingerprint", "manifest_path", "manifest_sha256",
    "source_sha256", "started_utc", "thread_contract",
}
EXPECTED_REPORT_KEYS = {
    "schema_version", "protocol", "manifest_fingerprint", "finished_utc",
    "deadline_stop", "failure", "selections", "production_warning_streak",
    "gate_history", "jobs", "report_deadline_met",
}
EXPECTED_RUNNER_JOB_KEYS = {
    "task", "arm", "seed", "status", "selected", "endpoint",
    "returncode", "output_directory",
}
HISTORICAL_C5_SHA256 = {
    "five_d2": {
        "run_config.json": "646f69a769eaaca71947db005e2188175b5e9dd56a7f0b6a64e4701fb7c34601",
        "training_history.csv": "4d204e162e2a6869af9fd1d792ce975622c7a84ac237cfa5f24f7a465e1f46c7",
        "summary.json": "e2837e6e494a6e2dd4d71b7603e4cf09e77dae8d5b2866d3b62a1d341867ed51",
        "evaluation_summary.csv": "93458708d0670b92fc1b2252139d649afc8736b5906e25f12ec06ed5794ff25f",
    },
    "five_d3": {
        "run_config.json": "221d73031f2d25401640ca144152f7ce8ace9e8d19f15283c51adfd515267ca4",
        "training_history.csv": "1d03eab6d45eb20876f5b014be938969be93bed6c84db9148df43ef1330e7b02",
        "summary.json": "a5e987572878b40d385235f98692afbc3248c0b462c498128ec61d2c94ba8dc9",
        "evaluation_summary.csv": "658306bc83504100d1a9e6cf736e2511d345cb49bf521cb089fc17866a43215f",
    },
    "watersqrt2": {
        "run_config.json": "db58dc8629c2240d672bbfc1a2f74360af5b92d96d77f199d2c9d8bedef237de",
        "training_history.csv": "5aff693251fcc10b166132082c66323c77643d582610dd8e66bb085f0c1bea53",
        "summary.json": "42a7097fd60e9a95f777f0ee6d1bbe7a0d980387c6a0c1751812305f52b26ec1",
        "evaluation_summary.csv": "3790b6a80076242cd14ef9c215f8572742e6d5a1d2b618e4c586a1fe11d8d109",
    },
}


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
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being read: {path}")
    return data


def record_stable_input(path: Path, inputs: dict[str, str] | None = None) -> bytes:
    path = path.resolve()
    data = stable_bytes(path)
    digest = sha256_bytes(data)
    if inputs is not None:
        key = str(path)
        previous = inputs.get(key)
        if previous is not None and previous != digest:
            raise RuntimeError(f"Input changed between reads: {path}")
        inputs[key] = digest
    return data


def load_json(path: Path, inputs: dict[str, str] | None = None) -> dict:
    return json.loads(record_stable_input(path, inputs).decode("utf-8-sig"))


def load_csv(path: Path, inputs: dict[str, str] | None = None) -> list[dict]:
    data = record_stable_input(path, inputs).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(data)))


def verify_inputs_unchanged(inputs: dict[str, str]) -> None:
    for raw_path, expected in sorted(inputs.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Consumed input disappeared before publish: {path}")
        observed = sha256_bytes(stable_bytes(path))
        if observed != expected:
            raise RuntimeError(f"Consumed input changed before publish: {path}")


def require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise RuntimeError(f"{label} is not an exact SHA-256 digest")
    return value.lower()


def exact_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} is not boolean")
    return value


def parse_utc(value, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is missing")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} is not ISO-8601: {value!r}") from error
    if result.tzinfo is None:
        raise RuntimeError(f"{label} has no timezone: {value!r}")
    return result


def require_below(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Path escaped authority root: {path}")
    return path


def validate_finalizer_approval(
    approved_finalizer_sha256: str,
    approved_test_sha256: str,
    inputs: dict[str, str] | None = None,
) -> dict[str, str]:
    approved = {
        FINALIZER_NAMES[0]: require_sha256(
            approved_finalizer_sha256, "approved finalizer SHA-256"
        ),
        FINALIZER_NAMES[1]: require_sha256(
            approved_test_sha256, "approved finalizer-test SHA-256"
        ),
    }
    observed = {}
    for name, expected in approved.items():
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise RuntimeError(f"Finalizer authority missing: {path}")
        digest = sha256_bytes(record_stable_input(path, inputs))
        if digest != expected:
            raise RuntimeError(f"Externally approved SHA-256 mismatch: {name}")
        observed[name] = digest
    return observed


def exact_int(value, label: str, minimum: int | None = None,
              maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(f"{label} is not an exact integer: {value!r}")
    result = int(number)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{label} is below {minimum}: {result}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{label} is above {maximum}: {result}")
    return result


def exact_json_int(value, label: str, minimum: int | None = None,
                   maximum: int | None = None) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} is not a JSON integer: {value!r}")
    return exact_int(value, label, minimum, maximum)


def finite_float(row: dict, *names: str, default=math.nan) -> float:
    for name in names:
        raw = row.get(name)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float(default)


def required_finite_float(row: dict, label: str, *names: str) -> float:
    value = finite_float(row, *names)
    if not math.isfinite(value):
        raise RuntimeError(f"{label} is missing or nonfinite")
    return value


def finite_mean(values: Iterable[float]) -> float:
    chosen = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(chosen)) if chosen else math.nan


def finite_median(values: Iterable[float]) -> float:
    chosen = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(chosen)) if chosen else math.nan


def rolling(values: Sequence[float], window: int, kind="mean") -> list[float]:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    queue: deque[float | None] = deque()
    result: list[float] = []
    if kind == "median":
        ordered: list[float] = []
        for raw in values:
            value = float(raw)
            item = value if math.isfinite(value) else None
            queue.append(item)
            if item is not None:
                bisect.insort(ordered, item)
            if len(queue) > window:
                expired = queue.popleft()
                if expired is not None:
                    ordered.pop(bisect.bisect_left(ordered, expired))
            n = len(ordered)
            result.append(
                math.nan if not n else float(ordered[n // 2]) if n % 2
                else float((ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
            )
        return result
    total, count = 0.0, 0
    for raw in values:
        value = float(raw)
        item = value if math.isfinite(value) else None
        queue.append(item)
        if item is not None:
            total += item
            count += 1
        if len(queue) > window:
            expired = queue.popleft()
            if expired is not None:
                total -= expired
                count -= 1
        result.append(float(total / count) if count else math.nan)
    return result


def active_c10_pids(root: Path = DEFAULT_ROOT) -> list[dict]:
    needle = str(root.resolve()).replace("'", "''")
    names = (
        "run_ori_codex10_nonswap_staged.py",
        "ori_codex6_spectral_subspace.py",
        "ori_codex6_spectral_subspace_3d.py",
    )
    clauses = " -or ".join(
        f"($_.CommandLine -like '*{name}*')" for name in names
    )
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and $_.CommandLine -like '*" + needle
        + "*' -and (" + clauses + ")}; "
        "$p | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    return [
        {"pid": exact_int(row["ProcessId"], "active PID", 1),
         "command": str(row.get("CommandLine", ""))}
        for row in payload
    ]


def wait_until_ready(root: Path, timeout_seconds: float, poll_seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        active = active_c10_pids(root)
        report_exists = (root / "runner_report.json").is_file()
        if report_exists and not active:
            return
        if time.monotonic() >= deadline:
            state = "runner_report absent" if not report_exists else f"active={active}"
            raise RuntimeError(f"C10NS1 is not finalizable: {state}")
        time.sleep(max(1.0, min(30.0, poll_seconds)))


def _strip_generated(value):
    if isinstance(value, dict):
        return {key: _strip_generated(item) for key, item in value.items()
                if key != "generated_utc"}
    if isinstance(value, list):
        return [_strip_generated(item) for item in value]
    return value


def _semantic_equal(left, right) -> bool:
    return _strip_generated(left) == _strip_generated(right)


def find_manifest(binding: dict, explicit: Path | None) -> Path:
    recorded_raw = binding.get("manifest_path")
    if not isinstance(recorded_raw, str) or not recorded_raw:
        raise RuntimeError("Execution binding has no exact manifest path")
    if not Path(recorded_raw).is_absolute():
        raise RuntimeError("Execution binding manifest path is not absolute")
    recorded = require_below(Path(recorded_raw), MANIFEST_DIRECTORY)
    if explicit is not None and explicit.resolve() != recorded:
        raise RuntimeError("Explicit manifest disagrees with execution binding")
    fingerprint = binding.get("manifest_fingerprint")
    expected_name = f"codex10_nonswap_{str(fingerprint)[:12]}.json"
    if recorded.name != expected_name or not recorded.is_file():
        raise RuntimeError(f"Bound manifest path/name is unavailable: {recorded}")
    return recorded


def _manifest_content(manifest: dict) -> dict:
    content = dict(manifest)
    content.pop("manifest_fingerprint", None)
    content.pop("generated_utc", None)
    return content


def validate_control_plane(root: Path, manifest_path: Path | None = None):
    binding_path = root / "execution_binding.json"
    report_path = root / "runner_report.json"
    if not report_path.is_file():
        raise RuntimeError("runner_report.json is absent; supervisor has not committed")
    inputs: dict[str, str] = {}
    binding = load_json(binding_path, inputs)
    if set(binding) != EXPECTED_BINDING_KEYS:
        raise RuntimeError(f"Execution binding key mismatch: {sorted(binding)}")
    manifest_file = find_manifest(binding, manifest_path)
    manifest = load_json(manifest_file, inputs)
    report = load_json(report_path, inputs)
    fingerprint = binding.get("manifest_fingerprint")
    if fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise RuntimeError("Finalizer is bound only to authorized C10NS1")
    if manifest.get("manifest_fingerprint") != fingerprint \
            or report.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError("Manifest/binding/report fingerprint mismatch")
    if canonical_digest(_manifest_content(manifest)) != fingerprint:
        raise RuntimeError("Manifest self-fingerprint mismatch")
    if inputs[str(manifest_file.resolve())] != require_sha256(
        binding.get("manifest_sha256"), "binding manifest SHA-256"
    ):
        raise RuntimeError("Manifest file hash disagrees with execution binding")
    if root.resolve() != DEFAULT_ROOT.resolve() \
            or Path(manifest.get("experiment_root", "")).resolve() != root.resolve():
        raise RuntimeError("Finalizer/root authority mismatch")
    expected_top = {
        "schema_version": 1, "protocol": EXPECTED_PROTOCOL,
        "seeds": list(SEEDS), "task_order": list(TASKS),
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "max_parallel": 18, "training_deadline_local": "22:30",
        "report_deadline_local": "23:00",
        "all_18_jobs_fresh_from_episode_zero": True,
        "selected_arm_all_three_seeds_continue": True,
    }
    changed = [key for key, value in expected_top.items()
               if manifest.get(key) != value]
    if changed:
        raise RuntimeError(f"Manifest protocol mismatch: {changed}")
    runner_validated = bound_runner.validate_manifest(manifest_file, root, 18)
    if not _semantic_equal(runner_validated, manifest):
        raise RuntimeError("Bound runner manifest reconstruction mismatch")
    if exact_json_int(report.get("schema_version"), "runner schema version", 1, 1) != 1 \
            or report.get("protocol") != EXPECTED_PROTOCOL:
        raise RuntimeError("Runner report schema/protocol mismatch")
    if set(report) != EXPECTED_REPORT_KEYS:
        raise RuntimeError(f"Runner report key mismatch: {sorted(report)}")
    started = parse_utc(binding.get("started_utc"), "binding start time")
    finished = parse_utc(report.get("finished_utc"), "runner finish time")
    if finished < started:
        raise RuntimeError("Runner finish timestamp precedes execution start")
    if report.get("failure") is not None:
        raise RuntimeError(f"Supervisor failure: {report['failure']}")
    deadline_stop = exact_bool(report.get("deadline_stop"), "runner deadline_stop")
    report_deadline_met = exact_bool(
        report.get("report_deadline_met"), "runner report_deadline_met"
    )
    # The campaign host is fixed to Japan Standard Time (UTC+09:00, no DST).
    # Bind the two runner deadline booleans to its committed finish timestamp;
    # a two-second boundary allowance covers the runner's timestamp-then-bool
    # write order without accepting a materially false deadline claim.
    jst = timezone(timedelta(hours=9))
    started_local, finished_local = started.astimezone(jst), finished.astimezone(jst)
    training_deadline = datetime.combine(
        started_local.date(), datetime.strptime("22:30", "%H:%M").time(), jst
    )
    report_deadline = datetime.combine(
        finished_local.date(), datetime.strptime("23:00", "%H:%M").time(), jst
    )
    boundary_allowance = timedelta(seconds=2)
    if deadline_stop and finished + boundary_allowance < training_deadline:
        raise RuntimeError("Runner deadline_stop precedes the bound 22:30 JST deadline")
    if report_deadline_met and finished - boundary_allowance >= report_deadline:
        raise RuntimeError("Runner report_deadline_met is false by committed finish time")
    if not report_deadline_met and finished + boundary_allowance < report_deadline:
        raise RuntimeError("Runner report_deadline_met is inconsistent with finish time")
    selections = report.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(TASKS):
        raise RuntimeError("Runner selections are incomplete")
    for task, arm in selections.items():
        if arm is not None and arm not in TASK_ARMS[task]:
            raise RuntimeError(f"Runner selection is invalid: {task}/{arm}")
    thread = binding.get("thread_contract")
    expected_threads = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    if not isinstance(thread, dict) or thread.get("env") != expected_threads \
            or exact_json_int(thread.get("torch"), "bound torch threads", 1, 1) != 1 \
            or exact_json_int(thread.get("cpu"), "bound CPU count", 1) < 1:
        raise RuntimeError("Execution thread contract mismatch")
    if binding.get("source_sha256") != manifest.get("source_sha256"):
        raise RuntimeError("Execution/source binding mismatch")
    if set(manifest.get("source_sha256", {})) != set(SOURCE_NAMES):
        raise RuntimeError("Unexpected source authority closure")
    for name, expected in manifest["source_sha256"].items():
        path = SCRIPT_DIRECTORY / name
        if not path.is_file() or sha256_bytes(record_stable_input(path, inputs)) != expected:
            raise RuntimeError(f"Bound source changed: {name}")
    matrix = {
        (task, arm, seed)
        for task in TASKS for arm in TASK_ARMS[task] for seed in SEEDS
    }
    jobs = manifest.get("jobs", [])
    observed = {
        (job.get("task", {}).get("name"), job.get("arm"),
          exact_json_int(job.get("seed"), "manifest seed")) for job in jobs
    }
    if len(jobs) != 18 or observed != matrix:
        raise RuntimeError("Manifest job matrix is not the exact 18-job factorial")
    for job in jobs:
        identity = dict(job)
        fingerprint_job = identity.pop("job_fingerprint", None)
        if canonical_digest(identity) != fingerprint_job:
            raise RuntimeError(
                f"Job self-fingerprint mismatch: {job['task']['name']}/"
                f"{job['arm']}/s{job['seed']}"
            )
    runner_jobs = report.get("jobs", [])
    if any(set(row) != EXPECTED_RUNNER_JOB_KEYS for row in runner_jobs):
        raise RuntimeError("Runner job row key mismatch")
    runner_map = {
        (row.get("task"), row.get("arm"),
         exact_json_int(row.get("seed"), "runner seed")): row
        for row in runner_jobs
    }
    if len(runner_jobs) != 18 or set(runner_map) != matrix:
        raise RuntimeError("Runner report job table is incomplete or duplicated")
    return manifest, binding, report, runner_map, inputs


def checkpoint_episode(path: Path, inputs: dict[str, str] | None = None) -> int:
    if not path.is_file():
        return 0
    before = sha256_bytes(record_stable_input(path, inputs))
    import torch
    # This is the frozen runner's freshly produced local checkpoint.  Its numpy
    # RNG state is incompatible with weights_only=True; source/root/PID and the
    # exact bytes are bound before trusting this local pickle format.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = exact_int(payload.get("episode", 0), "checkpoint episode", 0, 10000)
    if sha256_bytes(stable_bytes(path)) != before:
        raise RuntimeError(f"Checkpoint changed while loading: {path}")
    return result


def contiguous_history(rows: Sequence[dict]) -> int:
    episodes = [exact_int(row.get("episode"), "history episode", 1, 10000)
                for row in rows]
    if episodes != list(range(1, len(rows) + 1)):
        raise RuntimeError("training_history.csv is not contiguous and unique from episode 1")
    return len(rows)


def atomic_endpoint(
    directory: Path, rows: Sequence[dict], inputs: dict[str, str] | None = None,
) -> tuple[int, dict]:
    history_endpoint = contiguous_history(rows)
    checkpoint = checkpoint_episode(directory / "checkpoint.pt", inputs)
    summary_endpoint = 0
    if (directory / "summary.json").is_file():
        summary = load_json(directory / "summary.json", inputs)
        summary_endpoint = exact_int(
            summary.get("training_episodes", 0), "summary training episodes", 0, 10000
        )
    if summary_endpoint == 10000:
        required = ("policy.pt", "run_config.json", "training_history.csv")
        if history_endpoint != 10000 or any(
            not (directory / name).is_file() for name in required
        ):
            raise RuntimeError(f"Incomplete natural 10000 commit: {directory}")
        endpoint, authority = 10000, "summary_and_history"
    elif checkpoint > 0:
        if checkpoint > history_endpoint:
            raise RuntimeError(f"Checkpoint ahead of history: {directory}")
        endpoint, authority = checkpoint, "checkpoint_and_history"
    else:
        raise RuntimeError(f"No positive atomic endpoint: {directory}")
    return endpoint, {
        "authority": authority, "history_endpoint": history_endpoint,
        "checkpoint_endpoint": checkpoint, "summary_endpoint": summary_endpoint,
        "post_atomic_history_overrun_rows": max(0, history_endpoint - endpoint),
    }


def gate_artifact_path(root: Path, gate: dict) -> Path:
    task, endpoint = gate["task"], exact_json_int(gate["endpoint"], "gate endpoint")
    if "arm_summaries" in gate:
        name = f"checkpoint_{endpoint}_{task}.json"
    else:
        arm = gate["arm"]
        prefix = "production" if endpoint in PRODUCTION_CHECKPOINTS else "checkpoint"
        name = f"{prefix}_{endpoint}_{task}_{arm}_selected.json"
    return root / "analysis" / name


def recompute_gates(manifest: dict, report: dict, root: Path, inputs: dict):
    formal = {(task, arm, seed): 0 for task in TASKS
              for arm in TASK_ARMS[task] for seed in SEEDS}
    prior_warning = {task: 0 for task in TASKS}
    recomputed_rows, screen_science = [], {}
    derived_final: dict[str, str | None] = {}
    ever_selected: dict[str, str] = {}
    latest_selected: dict[str, dict] = {}
    states = {
        task: {
            "phase": "screening", "screen_index": 0,
            "production_index": 0, "arm": None,
        }
        for task in TASKS
    }
    history = report.get("gate_history")
    if not isinstance(history, list):
        raise RuntimeError("Runner gate_history is not a list")
    seen_artifacts = set()
    for index, recorded in enumerate(history):
        if not isinstance(recorded, dict):
            raise RuntimeError(f"Gate {index} is not an object")
        task = recorded.get("task")
        endpoint = exact_json_int(recorded.get("endpoint"), "gate endpoint", 1, 10000)
        if task not in TASKS:
            raise RuntimeError(f"Unknown gate task: {task}")
        state = states[task]
        if "arm_summaries" in recorded:
            if state["phase"] != "screening":
                raise RuntimeError(f"Task-level gate after task resolution: {task}")
            screen_index = state["screen_index"]
            if screen_index >= len(SCREEN_CHECKPOINTS) \
                    or endpoint != SCREEN_CHECKPOINTS[screen_index]:
                raise RuntimeError(
                    f"Non-causal screen gate for {task}: {endpoint}, index={screen_index}"
                )
            state["screen_index"] += 1
            fresh = staged.analyse_task(manifest["jobs"], task, endpoint)
            for arm in TASK_ARMS[task]:
                for seed in SEEDS:
                    formal[(task, arm, seed)] = max(formal[(task, arm, seed)], endpoint)
            screen_science[task] = fresh
            decision = fresh["decision"]
            if decision["selected_arm"] is not None:
                arm = decision["selected_arm"]
                if arm not in TASK_ARMS[task]:
                    raise RuntimeError(f"Invalid selected arm: {task}/{arm}")
                ever_selected[task] = arm
                derived_final[task] = arm
                state.update({"phase": "selected", "arm": arm})
            elif decision["stop_task"]:
                derived_final[task] = None
                state["phase"] = "stopped"
            elif endpoint == SCREEN_CHECKPOINTS[-1]:
                raise RuntimeError(f"Unresolved task after final screen gate: {task}")
        else:
            arm = recorded.get("arm")
            if state["phase"] != "selected" or arm != state["arm"] \
                    or arm not in TASK_ARMS[task]:
                raise RuntimeError(f"Selected gate without matching arm selection: {task}/{arm}")
            if endpoint in SCREEN_CHECKPOINTS:
                screen_index = state["screen_index"]
                if screen_index >= len(SCREEN_CHECKPOINTS) \
                        or endpoint != SCREEN_CHECKPOINTS[screen_index]:
                    raise RuntimeError(
                        f"Non-causal selected screen gate for {task}: {endpoint}"
                    )
                state["screen_index"] += 1
            elif endpoint in PRODUCTION_CHECKPOINTS:
                if state["screen_index"] != len(SCREEN_CHECKPOINTS):
                    raise RuntimeError(f"Production gate before formal 4096: {task}")
                production_index = state["production_index"]
                if production_index >= len(PRODUCTION_CHECKPOINTS) \
                        or endpoint != PRODUCTION_CHECKPOINTS[production_index]:
                    raise RuntimeError(
                        f"Non-causal production gate for {task}: {endpoint}"
                    )
                state["production_index"] += 1
            else:
                raise RuntimeError(f"Selected gate has unknown endpoint: {endpoint}")
            fresh = staged.analyse_selected(
                manifest["jobs"], task, arm, endpoint,
                prior_production_warning_streak=prior_warning[task],
            )
            latest_selected[task] = fresh
            prior_warning[task] = fresh["production_monitor"][
                "consecutive_warning_count"
            ]
            for seed in SEEDS:
                formal[(task, arm, seed)] = max(formal[(task, arm, seed)], endpoint)
            if fresh["decision"]["revoke"]:
                derived_final[task] = None
                state["phase"] = "revoked"
        artifact = gate_artifact_path(root, recorded)
        resolved_artifact = str(artifact.resolve())
        if resolved_artifact in seen_artifacts:
            raise RuntimeError(f"Duplicate gate artifact in causal history: {artifact.name}")
        seen_artifacts.add(resolved_artifact)
        if not artifact.is_file():
            raise RuntimeError(f"Missing runner gate artifact: {artifact}")
        artifact_payload = load_json(artifact, inputs)
        if not _semantic_equal(recorded, artifact_payload) \
                or not _semantic_equal(recorded, fresh):
            raise RuntimeError(f"Gate recomputation mismatch: {artifact.name}")
        recomputed_rows.append({
            "index": index, "task": task, "arm": recorded.get("arm"),
            "endpoint": endpoint,
            "causal_phase_after_gate": state["phase"],
            "decision": fresh["decision"], "artifact": str(artifact.resolve()),
            "artifact_sha256": inputs[str(artifact.resolve())],
        })
    if set(derived_final) != set(TASKS):
        raise RuntimeError(f"Gate history did not resolve every task: {derived_final}")
    if derived_final != report.get("selections"):
        raise RuntimeError(
            f"Recomputed final selections disagree with runner: {derived_final} != "
            f"{report.get('selections')}"
        )
    observed_warning = report.get("production_warning_streak")
    if not isinstance(observed_warning, dict) or set(observed_warning) != set(TASKS):
        raise RuntimeError("Runner production-warning state is incomplete")
    exact_warning = {
        task: exact_json_int(observed_warning[task], f"{task} warning streak", 0)
        for task in TASKS
    }
    if exact_warning != prior_warning:
        raise RuntimeError(
            f"Production warning streak mismatch: {exact_warning} != {prior_warning}"
        )
    return formal, ever_selected, screen_science, recomputed_rows, latest_selected


def build_episode_curves(
    rows: Sequence[dict], endpoint: int, identity: dict, batch_size: int,
    expected_flatness_tolerance: float, expected_spread_tolerance: float,
    expected_target_spread: float,
) -> tuple[list[dict], dict]:
    prefix = list(rows[:endpoint])
    cumulative, first_episode, first_proposals = 0, None, None
    records = []
    for row in prefix:
        episode = exact_int(row.get("episode"), "curve episode", 1, endpoint)
        steps = exact_int(row.get("steps"), "mechanical proposals", 1, 400)
        cumulative += steps
        invalid_flags = {
            name: exact_int(row.get(name), f"episode {episode} {name}", 0, 1)
            for name in ("self_crossing", "correction_failed", "no_modes")
        }
        invalid = any(invalid_flags.values())
        strict = exact_int(
            row.get("strict_compact_success"), f"episode {episode} strict", 0, 1
        )
        success = exact_int(row.get("success"), f"episode {episode} success", 0, 1)
        flatness = required_finite_float(
            row, f"episode {episode} terminal flatness",
            "flatness_ratio", "ori5_last_valid_flatness_ratio",
        )
        spread_error = required_finite_float(
            row, f"episode {episode} spread error",
            "spread_error", "ori5_last_valid_spread_error",
        )
        if flatness < 0.0 or spread_error < 0.0:
            raise RuntimeError(f"Negative geometry metric at episode {episode}")
        flatness_tolerance = required_finite_float(
            row, f"episode {episode} evaluation flatness tolerance",
            "evaluation_flatness_tolerance",
        )
        spread_tolerance = required_finite_float(
            row, f"episode {episode} spread tolerance", "spread_tolerance"
        )
        target_spread = required_finite_float(
            row, f"episode {episode} target spread", "target_spread"
        )
        spread = required_finite_float(row, f"episode {episode} spread", "spread")
        expected_values = (
            (flatness_tolerance, expected_flatness_tolerance, "flatness tolerance"),
            (spread_tolerance, expected_spread_tolerance, "spread tolerance"),
            (target_spread, expected_target_spread, "target spread"),
        )
        if any(
            not math.isfinite(float(expected)) or not math.isclose(
                observed, float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
            for observed, expected, _ in expected_values
        ):
            mismatches = [
                f"{label}={observed!r}!={expected!r}"
                for observed, expected, label in expected_values
                if not math.isfinite(float(expected)) or not math.isclose(
                    observed, float(expected), rel_tol=1e-12, abs_tol=1e-12
                )
            ]
            raise RuntimeError(
                f"Bound geometry authority mismatch at episode {episode}: {mismatches}"
            )
        if min(flatness_tolerance, spread_tolerance, target_spread, spread) < 0.0:
            raise RuntimeError(f"Negative success tolerance at episode {episode}")
        if not math.isclose(
            spread_error, abs(spread - target_spread), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError(f"Spread/error identity mismatch at episode {episode}")
        strict_from_geometry = int(
            not invalid and flatness <= flatness_tolerance
            and spread_error <= spread_tolerance
        )
        if strict != success or strict != strict_from_geometry:
            raise RuntimeError(
                f"Strict/success/geometry mismatch at episode {episode}: "
                f"strict={strict}, success={success}, geometry={strict_from_geometry}"
            )
        rejected_count = exact_int(
            row.get("ori6_rejected_crossing_count"),
            f"episode {episode} rejected crossing count", 0, steps,
        )
        rejected_rate = required_finite_float(
            row, f"episode {episode} rejected crossing rate",
            "ori6_rejected_crossing_rate",
        )
        if not 0.0 <= rejected_rate <= 1.0 or not math.isclose(
            rejected_rate, rejected_count / steps, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError(f"Rejected-crossing rate mismatch at episode {episode}")
        spectral_potential = required_finite_float(
            row, f"episode {episode} spectral potential",
            "ori6_spectral_terminal_potential",
        )
        if not -1e-12 <= spectral_potential <= 1.0 + 1e-12:
            raise RuntimeError(f"Spectral potential outside [0,1] at episode {episode}")
        spectral_reward = required_finite_float(
            row, f"episode {episode} spectral reward", "ori6_spectral_total_reward"
        )
        episode_return = required_finite_float(
            row, f"episode {episode} return", "total_reward"
        )
        constructive = int(
            not invalid and math.isfinite(flatness) and math.isfinite(spread_error)
            and flatness <= 0.5 and spread_error <= 0.1
        )
        joint = int(
            not invalid and math.isfinite(flatness) and math.isfinite(spread_error)
            and flatness <= 0.1 and spread_error <= 0.1
        )
        if strict and first_episode is None:
            first_episode, first_proposals = episode, cumulative
        records.append({
            **identity, "episode": episode, "cumulative_proposals": cumulative,
            "steps": steps, "strict": strict, "constructive_C": constructive,
            "joint_J": joint, "terminal_flatness_F": flatness,
            "spread_error_e": spread_error,
            "return_diagnostic": episode_return,
            "rejected_crossing_rate": rejected_rate,
            "spectral_terminal_potential": spectral_potential,
            "spectral_reward": spectral_reward,
        })
    rolling_fields = (
        ("strict", "mean", "strict_rate"),
        ("constructive_C", "mean", "constructive_C_rate"),
        ("joint_J", "mean", "joint_J_rate"),
        ("terminal_flatness_F", "median", "terminal_flatness_F_median"),
        ("spread_error_e", "median", "spread_error_e_median"),
        ("return_diagnostic", "mean", "return_diagnostic_mean"),
        ("rejected_crossing_rate", "mean", "rejected_crossing_rate_mean"),
        ("spectral_terminal_potential", "mean", "spectral_potential_mean"),
        ("spectral_reward", "mean", "spectral_reward_mean"),
    )
    for window in ROLLING_WINDOWS:
        for source, kind, destination in rolling_fields:
            values = rolling([record[source] for record in records], window, kind)
            for record, value in zip(records, values):
                record[f"{destination}_{window}"] = value
    recent = records[-min(1024, len(records)):]
    metrics = {
        "first_hit_episode": first_episode,
        "first_hit_proposals": first_proposals,
        "first_hit_before_or_at_first_update": bool(
            first_episode is not None and first_episode <= batch_size
        ),
        "proposal_censor": cumulative,
        "strict_hits": sum(record["strict"] for record in records),
        "strict_rate_last1024": finite_mean(record["strict"] for record in recent),
        "constructive_C_rate_last1024": finite_mean(
            record["constructive_C"] for record in recent
        ),
        "joint_J_rate_last1024": finite_mean(record["joint_J"] for record in recent),
        "median_F_last1024": finite_median(
            record["terminal_flatness_F"] for record in recent
        ),
        "median_e_last1024": finite_median(record["spread_error_e"] for record in recent),
        "mean_return_last512_diagnostic": finite_mean(
            record["return_diagnostic"] for record in records[-min(512, len(records)):]
        ),
    }
    return records, metrics


def fixed_evaluations(directory: Path, endpoint: int, curves: Sequence[dict],
                      identity: dict, inputs: dict[str, str] | None = None) -> list[dict]:
    path = directory / "stochastic_evaluation_history.csv"
    rows = load_csv(path, inputs)
    proposal_by_episode = {row["episode"]: row["cumulative_proposals"] for row in curves}
    result = []
    seen = set()
    for row in rows:
        episode = exact_int(row.get("episode"), "fixed evaluation episode", 1, 10000)
        if episode > endpoint:
            continue
        if episode in seen or episode not in proposal_by_episode:
            raise RuntimeError(f"Invalid fixed evaluation episode {episode}: {path}")
        seen.add(episode)
        trials = exact_int(row.get("evaluation_episodes"), "fixed trials", 1)
        rate = finite_float(row, "strict_compact_success_rate")
        if trials != 16 or not 0.0 <= rate <= 1.0 \
                or not math.isclose(rate * trials, round(rate * trials), abs_tol=1e-9):
            raise RuntimeError(f"Invalid fixed stochastic evaluation: {path} ep{episode}")
        result.append({
            **identity, "episode": episode,
            "cumulative_proposals": proposal_by_episode[episode],
            "evaluation_episodes": trials,
            "strict_compact_success_rate": rate,
        })
    if endpoint not in seen:
        raise RuntimeError(f"No fixed stochastic evaluation at formal endpoint {endpoint}")
    return result


def validate_final_summary(
    directory: Path, endpoint: int, expected_flatness_tolerance: float,
    expected_spread_tolerance: float, expected_target_spread: float,
    inputs: dict[str, str] | None = None,
) -> dict | None:
    if endpoint != 10000:
        return None
    summary = load_json(directory / "summary.json", inputs)
    held = summary.get("last_policy_held_out_evaluation")
    if not isinstance(held, dict):
        raise RuntimeError(f"Missing final held-out evaluation: {directory}")
    trials = exact_int(held.get("evaluation_episodes"), "final evaluation episodes", 1)
    rate = finite_float(held, "strict_compact_success_rate")
    if trials != 64 or not 0.0 <= rate <= 1.0 \
            or not math.isclose(rate * trials, round(rate * trials), abs_tol=1e-9):
        raise RuntimeError(f"Invalid 64-trial final evaluation: {directory}")
    raw_path = directory / "evaluation_summary.csv"
    raw = load_csv(raw_path, inputs)
    if len(raw) != trials:
        raise RuntimeError(f"Final evaluation row count is not 64: {raw_path}")
    flags = []
    for index, row in enumerate(raw, start=1):
        strict = exact_int(
            row.get("strict_compact_success"),
            f"final evaluation strict row {index}", 0, 1,
        )
        success = exact_int(
            row.get("success"), f"final evaluation success row {index}", 0, 1,
        )
        if strict != success:
            raise RuntimeError(f"Final evaluation success mismatch: {raw_path} row {index}")
        invalid = any(
            exact_int(
                row.get(name), f"final evaluation {name} row {index}", 0, 1
            )
            for name in ("self_crossing", "correction_failed", "no_modes")
        )
        flatness = required_finite_float(
            row, f"final evaluation flatness row {index}", "flatness_ratio"
        )
        error = required_finite_float(
            row, f"final evaluation spread error row {index}", "spread_error"
        )
        flatness_tolerance = required_finite_float(
            row, f"final evaluation flatness tolerance row {index}",
            "evaluation_flatness_tolerance",
        )
        spread_tolerance = required_finite_float(
            row, f"final evaluation spread tolerance row {index}", "spread_tolerance"
        )
        target_spread = required_finite_float(
            row, f"final evaluation target spread row {index}", "target_spread"
        )
        spread = required_finite_float(
            row, f"final evaluation spread row {index}", "spread"
        )
        for observed, expected, label in (
            (flatness_tolerance, expected_flatness_tolerance, "flatness tolerance"),
            (spread_tolerance, expected_spread_tolerance, "spread tolerance"),
            (target_spread, expected_target_spread, "target spread"),
        ):
            if not math.isfinite(float(expected)) or not math.isclose(
                observed, float(expected), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"Final evaluation bound {label} mismatch: {raw_path} row {index}"
                )
        if min(flatness, error, flatness_tolerance, spread_tolerance,
               target_spread, spread) < 0.0:
            raise RuntimeError(f"Negative final evaluation geometry value: {raw_path} row {index}")
        if not math.isclose(
            error, abs(spread - target_spread), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Final evaluation spread/error mismatch: {raw_path} row {index}"
            )
        expected_strict = int(
            not invalid and flatness <= flatness_tolerance
            and error <= spread_tolerance
        )
        if strict != expected_strict:
            raise RuntimeError(
                f"Final evaluation strict/geometry mismatch: {raw_path} row {index}"
            )
        flags.append(strict)
    raw_rate = sum(flags) / trials
    if not math.isclose(raw_rate, rate, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Final evaluation raw/summary mismatch: {raw_rate} != {rate}"
        )
    return {"evaluation_episodes": trials, "strict_compact_success_rate": rate}


def historical_c5_anchors(inputs: dict) -> list[dict]:
    anchors = []
    for task in TASKS:
        directory = HISTORICAL_ROOT / task / "B" / "seed2101"
        config_path, history_path, summary_path, evaluation_path = (
            directory / "run_config.json", directory / "training_history.csv",
            directory / "summary.json", directory / "evaluation_summary.csv",
        )
        for name, expected in HISTORICAL_C5_SHA256[task].items():
            path = directory / name
            observed = sha256_bytes(record_stable_input(path, inputs))
            if observed != expected:
                raise RuntimeError(f"Historical Codex5 authority changed: {path}")
        config = load_json(config_path, inputs)
        rows = load_csv(history_path, inputs)
        summary = load_json(summary_path, inputs)
        if contiguous_history(rows) != 10000 \
                or exact_int(config.get("seed"), "C5 seed") != 2101 \
                or exact_int(config.get("episodes"), "C5 episodes") != 10000 \
                or config.get("reward_design") != "ori5_endpoint_potential_fixed_milestones" \
                or exact_int(config.get("max_steps"), "C5 max steps") != 400:
            raise RuntimeError(f"Invalid historical Codex5 anchor: {directory}")
        cumulative, first_episode, first_proposals = 0, None, None
        flags = []
        for row in rows:
            cumulative += exact_int(row.get("steps"), "C5 steps", 0, 400)
            strict = exact_int(row.get("strict_compact_success"), "C5 strict", 0, 1)
            flags.append(strict)
            if strict and first_episode is None:
                first_episode = exact_int(row["episode"], "C5 first episode")
                first_proposals = cumulative
        if first_episode != exact_int(
            summary.get("training_first_success_episode"), "C5 summary first episode"
        ) or first_proposals != exact_int(
            summary.get("training_first_success_mechanical_steps"),
            "C5 summary first proposals",
        ):
            raise RuntimeError(f"C5 first-discovery summary mismatch: {task}")
        held = summary.get("last_policy_held_out_evaluation", {})
        trials = exact_int(held.get("evaluation_episodes"), "C5 held-out trials")
        rate = finite_float(held, "strict_compact_success_rate")
        raw = load_csv(evaluation_path, inputs)
        raw_flags = [
            exact_int(row.get("strict_compact_success"), "C5 raw strict", 0, 1)
            for row in raw
        ]
        if trials != 100 or len(raw_flags) != trials \
                or not math.isclose(sum(raw_flags) / trials, rate, abs_tol=1e-12):
            raise RuntimeError(f"Invalid historical Codex5 held-out evaluation: {task}")
        anchors.append({
            "task": task, "arm": "B", "seed": 2101, "episodes": 10000,
            "first_hit_episode": first_episode,
            "first_hit_proposals": first_proposals,
            "first_hit_before_or_at_first_update": first_episode <= int(config["batch_size"]),
            "strict_rate_last1024": sum(flags[-1024:]) / 1024.0,
            "final_stochastic_episodes": trials,
            "final_stochastic_strict_rate": rate,
            "reward_design": config.get("reward_design"),
            "crossing_penalty": config.get("ori5_crossing_penalty"),
            "comparison_scope": (
                "descriptive_only_unmatched_seed_and_protocol; Codex5 crossing terminates "
                "the episode, while C10 uses Codex6 reject-and-continue"
            ),
        })
    return anchors


def _arm_summary_rows(screen_science: dict) -> list[dict]:
    rows = []
    for task in TASKS:
        screen = screen_science[task]
        selected = screen["decision"]["selected_arm"]
        for arm in TASK_ARMS[task]:
            summary = screen["arm_summaries"][arm]
            rows.append({
                "task": task, "arm": arm, "screen_endpoint": screen["endpoint"],
                "selected_at_screen": arm == selected,
                "selection_reason": screen["decision"]["reason"],
                "common_proposal_cap": screen["common_proposal_cap"],
                "hit_seed_count": summary["hit_seed_count"],
                "pre_update_hit_seed_count": summary["pre_update_hit_seed_count"],
                "post_update_hit_seed_count": summary["post_update_hit_seed_count"],
                "stable_seed_count": summary["stable_seed_count"],
                "common_cap_proposal_rmst": summary["common_cap_proposal_rmst"],
                "mean_tail_strict_rate": summary["mean_tail_strict_rate"],
                "mean_fixed_stochastic_strict_rate": summary[
                    "mean_fixed_stochastic_strict_rate"
                ],
                "learning_evidence_beyond_initial_policy": screen[
                    "learning_evidence_beyond_initial_policy"
                ][arm],
                "production_ready_at_4096": screen[
                    "production_ready_at_4096"
                ][arm],
            })
    return rows


def _paired_seed_rows(screen_science: dict) -> list[dict]:
    result = []
    for task in TASKS:
        screen = screen_science[task]
        candidate = screen["paired_candidate_vs_B"]["candidate"]
        cap = exact_int(screen["common_proposal_cap"], "paired common cap", 1)
        reports = {
            (arm, exact_int(row["seed"], "paired report seed")): row
            for arm, summary in screen["arm_summaries"].items()
            for row in summary["reports"]
        }
        for paired in screen["paired_candidate_vs_B"]["per_seed"]:
            seed = exact_int(paired["seed"], "paired seed")
            control_report = reports[("B", seed)]
            candidate_report = reports[(candidate, seed)]
            candidate_capped = exact_int(
                paired["candidate"], "candidate capped proposals", 1, cap
            )
            control_capped = exact_int(
                paired["control"], "control capped proposals", 1, cap
            )
            outcome = paired.get("outcome")
            if outcome not in {"win", "loss", "tie"}:
                raise RuntimeError(f"Invalid paired outcome: {task}/s{seed}/{outcome!r}")
            candidate_preupdate = exact_bool(
                candidate_report["first_hit_before_or_at_first_update"],
                "candidate pre-update first-hit flag",
            )
            control_preupdate = exact_bool(
                control_report["first_hit_before_or_at_first_update"],
                "control pre-update first-hit flag",
            )
            result.append({
                "task": task, "screen_endpoint": screen["endpoint"],
                "common_proposal_cap": cap, "candidate_arm": candidate,
                "control_arm": "B", "seed": seed,
                "candidate_event_by_cap": exact_bool(
                    paired["candidate_event_by_cap"], "candidate event"
                ),
                "control_event_by_cap": exact_bool(
                    paired["control_event_by_cap"], "control event"
                ),
                "candidate_first_hit_proposals_capped": candidate_capped,
                "control_first_hit_proposals_capped": control_capped,
                "proposal_delta_candidate_minus_control": (
                    candidate_capped - control_capped
                ),
                "candidate_first_hit_episode": candidate_report["first_hit_episode"],
                "control_first_hit_episode": control_report["first_hit_episode"],
                "candidate_first_hit_before_or_at_first_update": candidate_preupdate,
                "control_first_hit_before_or_at_first_update": control_preupdate,
                "candidate_tail_strict_rate": candidate_report["tail_strict_rate"],
                "control_tail_strict_rate": control_report["tail_strict_rate"],
                "tail_strict_rate_delta_candidate_minus_control": (
                    candidate_report["tail_strict_rate"]
                    - control_report["tail_strict_rate"]
                ),
                "candidate_fixed_stochastic_strict_rate": candidate_report[
                    "fixed_stochastic_strict_rate"
                ],
                "control_fixed_stochastic_strict_rate": control_report[
                    "fixed_stochastic_strict_rate"
                ],
                "fixed_rate_delta_candidate_minus_control": (
                    candidate_report["fixed_stochastic_strict_rate"]
                    - control_report["fixed_stochastic_strict_rate"]
                ),
                "outcome_for_candidate": outcome,
            })
    if len(result) != len(TASKS) * len(SEEDS):
        raise RuntimeError("Paired seed comparison is not the exact 3-task x 3-seed set")
    return result


def validate_runner_job_state(
    runner_row: dict,
    task: str,
    arm: str,
    ever_selected: dict[str, str],
    final_selections: dict[str, str | None],
    deadline_stop: bool,
    atomic_endpoint_value: int,
    latest_selected_gate: dict | None,
) -> tuple[bool, int]:
    selected = exact_bool(runner_row.get("selected"), "runner selected")
    expected_selected = ever_selected.get(task) == arm
    if selected != expected_selected:
        raise RuntimeError(f"Selected flag mismatch: {(task, arm, runner_row.get('seed'))}")
    status = runner_row.get("status")
    if not isinstance(status, str):
        raise RuntimeError("Runner status is not a string")
    returncode = exact_json_int(runner_row.get("returncode"), "runner returncode")
    final = final_selections[task]
    if not expected_selected:
        expected_status = "not_selected" if task in ever_selected else "not_promising"
        natural_before_gate = (
            status == "complete" and atomic_endpoint_value == 10000 and returncode == 0
        )
        if status != expected_status and not natural_before_gate:
            raise RuntimeError(
                f"Unexpected unselected status {task}/{arm}: {status} != {expected_status}"
            )
    else:
        natural_completion = (
            status == "complete" and atomic_endpoint_value == 10000 and returncode == 0
        )
        if status == "complete":
            if not natural_completion:
                raise RuntimeError(f"Natural completion authority mismatch: {task}/{arm}")
        elif final is None:
            if not isinstance(latest_selected_gate, dict) or not exact_bool(
                latest_selected_gate.get("decision", {}).get("revoke"),
                f"{task} latest revoke decision",
            ):
                raise RuntimeError(f"Final selection vanished without a revoke gate: {task}")
            revoke_endpoint = exact_json_int(
                latest_selected_gate.get("endpoint"), f"{task} revoke endpoint", 1, 10000
            )
            expected_status = (
                "selected_arm_revoked" if revoke_endpoint <= 4096
                else "production_stability_revoked"
            )
            if status != expected_status:
                raise RuntimeError(
                    f"Revocation status mismatch {task}/{arm}: "
                    f"{status} != {expected_status}"
                )
        elif status == "deadline_stop":
            if not deadline_stop or final != arm:
                raise RuntimeError(f"Deadline-stop state mismatch: {task}/{arm}")
        else:
            raise RuntimeError(f"Unexpected selected status {task}/{arm}: {status}")
    if status == "deadline_stop" and not deadline_stop:
        raise RuntimeError(f"Job deadline_stop without report deadline: {task}/{arm}")
    return selected, returncode


def prehash_job_and_model_inputs(
    manifest: dict, root: Path, inputs: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Capture every mutable input before gate recomputation reads any job."""
    model_authority: dict[str, dict[str, str]] = {}
    for job in manifest["jobs"]:
        task, arm = job["task"]["name"], job["arm"]
        seed = exact_json_int(job["seed"], "prehash job seed")
        directory = require_below(Path(job["output_directory"]), root)
        for name, expected in job.get("model_sha256", {}).items():
            expected_digest = require_sha256(
                expected, f"model SHA-256 {task}/{arm}/s{seed}/{name}"
            )
            model = SCRIPT_DIRECTORY / job["task"]["model_directory"] / name
            if not model.is_file():
                raise RuntimeError(f"Bound model is missing: {model}")
            observed = sha256_bytes(record_stable_input(model, inputs))
            if observed != expected_digest:
                raise RuntimeError(f"Bound model changed: {model}")
            bucket = model_authority.setdefault(job["task"]["model_directory"], {})
            previous = bucket.get(name)
            if previous is not None and previous != expected_digest:
                raise RuntimeError(f"Inconsistent repeated model authority: {model}")
            bucket[name] = expected_digest
        for name in JOB_INPUT_NAMES:
            path = directory / name
            if not path.is_file():
                if name in REQUIRED_JOB_INPUT_NAMES:
                    raise RuntimeError(f"Required committed job input is missing: {path}")
                continue
            record_stable_input(path, inputs)
    return model_authority


def first_hit_censor_reason(
    metrics: dict,
    task: str,
    arm: str,
    formal_endpoint: int,
    runner_status: str,
    ever_selected: dict[str, str],
    final_selections: dict[str, str | None],
    latest_selected_gate: dict | None,
    deadline_stop: bool,
) -> str | None:
    if metrics["first_hit_episode"] is not None:
        return None
    if formal_endpoint == 10000 and runner_status == "complete":
        return "natural_completion_at_episode_10000_without_strict_hit"
    if runner_status == "deadline_stop" or (
        deadline_stop and formal_endpoint < 10000 and runner_status == "complete"
    ):
        return f"deadline_right_censor_at_episode_{formal_endpoint}"
    if ever_selected.get(task) != arm:
        if task in ever_selected:
            return f"competing_arm_not_selected_at_episode_{formal_endpoint}"
        return f"task_stopped_no_promising_arm_at_episode_{formal_endpoint}"
    if final_selections[task] is None:
        reason = None
        if isinstance(latest_selected_gate, dict):
            reason = latest_selected_gate.get("decision", {}).get("reason")
        if not isinstance(reason, str) or not reason:
            raise RuntimeError(f"Selected censor lacks a causal revoke reason: {task}/{arm}")
        return f"selected_arm_revoked_at_episode_{formal_endpoint}:{reason}"
    raise RuntimeError(f"Unclassified first-hit censor: {task}/{arm}/ep{formal_endpoint}")


def analyze_campaign(
    root: Path = DEFAULT_ROOT,
    manifest_path: Path | None = None,
    active_check=True,
    approved_finalizer_sha256: str | None = None,
    approved_test_sha256: str | None = None,
) -> dict:
    root = root.resolve()
    if active_check:
        active = active_c10_pids(root)
        if active:
            raise RuntimeError(f"Active C10NS1 PID(s); refusing analysis: {active}")
    manifest, binding, report, runner_map, inputs = validate_control_plane(
        root, manifest_path
    )
    if approved_finalizer_sha256 is None or approved_test_sha256 is None:
        raise RuntimeError("External finalizer/test SHA-256 approval is required")
    finalizer_sources = validate_finalizer_approval(
        approved_finalizer_sha256, approved_test_sha256, inputs
    )
    model_authority = prehash_job_and_model_inputs(manifest, root, inputs)
    formal, ever_selected, screen_science, gate_rows, latest_selected = recompute_gates(
        manifest, report, root, inputs
    )
    jobs, curves, fixed_rows = [], [], []
    selected_curves, selected_fixed = [], []
    deadline_stop = exact_bool(report["deadline_stop"], "runner deadline_stop")
    any_deadline_job = False
    for job in manifest["jobs"]:
        task, arm = job["task"]["name"], job["arm"]
        seed = exact_json_int(job["seed"], "job seed")
        identity_key = (task, arm, seed)
        runner = runner_map[identity_key]
        directory = require_below(Path(job["output_directory"]), root)
        if Path(runner.get("output_directory", "")).resolve() != directory:
            raise RuntimeError(f"Runner output path mismatch: {identity_key}")
        for name, expected in job.get("model_sha256", {}).items():
            model = SCRIPT_DIRECTORY / job["task"]["model_directory"] / name
            if sha256_bytes(record_stable_input(model, inputs)) != expected:
                raise RuntimeError(f"Bound model changed during analysis: {model}")
        for name in JOB_INPUT_NAMES:
            path = directory / name
            if path.is_file():
                record_stable_input(path, inputs)
        history_path = directory / "training_history.csv"
        rows = load_csv(history_path, inputs)
        atomic, authority = atomic_endpoint(directory, rows, inputs)
        runner_endpoint = exact_json_int(
            runner.get("endpoint"), "runner endpoint", 1, 10000
        )
        if atomic != runner_endpoint:
            raise RuntimeError(
                f"Runner/atomic endpoint mismatch {identity_key}: {runner_endpoint} != {atomic}"
            )
        formal_endpoint = formal[identity_key]
        if formal_endpoint <= 0 or formal_endpoint > atomic:
            raise RuntimeError(
                f"Invalid formal endpoint {formal_endpoint}/{atomic}: {identity_key}"
            )
        verify_inputs_unchanged({
            key: value for key, value in inputs.items()
            if str(directory.resolve()) in key
        })
        integrity = staged.seed_report(job, formal_endpoint)
        verify_inputs_unchanged({
            key: value for key, value in inputs.items()
            if str(directory.resolve()) in key
        })
        if not integrity["integrity_pass"]:
            raise RuntimeError(
                f"Scientific integrity failure {identity_key}: {integrity['integrity_errors']}"
            )
        identity = {
            "task": task, "arm": arm, "seed": seed,
            "selected": ever_selected.get(task) == arm,
        }
        run_curves, metrics = build_episode_curves(
            rows, formal_endpoint, identity,
            exact_json_int(job["expected_config"]["batch_size"], "batch size", 1),
            float(job["expected_config"]["flatness_tolerance"]),
            float(job["expected_config"]["spread_tolerance"]),
            float(job["expected_config"]["target_spread"]),
        )
        evaluations = fixed_evaluations(
            directory, formal_endpoint, run_curves, identity, inputs
        )
        final_eval = validate_final_summary(
            directory, formal_endpoint,
            float(job["expected_config"]["flatness_tolerance"]),
            float(job["expected_config"]["spread_tolerance"]),
            float(job["expected_config"]["target_spread"]), inputs,
        )
        if metrics["first_hit_episode"] != integrity["first_hit_episode"] \
                or metrics["first_hit_proposals"] != integrity["first_hit_proposals"]:
            raise RuntimeError(f"Independent first-hit mismatch: {identity_key}")
        selected, runner_returncode = validate_runner_job_state(
            runner, task, arm, ever_selected, report["selections"],
            deadline_stop, atomic, latest_selected.get(task),
        )
        any_deadline_job = any_deadline_job or runner.get("status") == "deadline_stop"
        censor_reason = first_hit_censor_reason(
            metrics, task, arm, formal_endpoint, runner.get("status"),
            ever_selected, report["selections"], latest_selected.get(task),
            deadline_stop,
        )
        entry = {
            **identity, "runner_status": runner.get("status"),
            "runner_selected": selected,
            "runner_returncode": runner_returncode,
            "job_fingerprint": job["job_fingerprint"],
            "model_sha256": job["model_sha256"],
            "optimizer_name": job["optimizer"]["name"],
            "batch_size": job["expected_config"]["batch_size"],
            "actor_initial_learning_rate": job["expected_config"][
                "actor_learning_rate"
            ],
            "critic_learning_rate": job["expected_config"][
                "critic_learning_rate"
            ],
            "entropy_coefficient": job["expected_config"]["entropy_coefficient"],
            "gamma": job["expected_config"]["gamma"],
            "actor_lr_milestones": json.dumps(
                job["expected_config"]["actor_lr_milestones"], separators=(",", ":")
            ),
            "actor_lr_factors": json.dumps(
                job["expected_config"]["actor_lr_factors"], separators=(",", ":")
            ),
            "atomic_endpoint": atomic, "formal_endpoint": formal_endpoint,
            "post_formal_atomic_overrun_episodes": atomic - formal_endpoint,
            "post_atomic_history_overrun_rows": authority[
                "post_atomic_history_overrun_rows"
            ],
            "total_ignored_post_formal_rows": authority["history_endpoint"] - formal_endpoint,
            **authority, **metrics,
            "first_hit_event": metrics["first_hit_episode"] is not None,
            "first_hit_censor_reason": censor_reason,
            "fixed_stochastic_rate_at_formal_endpoint": integrity[
                "fixed_stochastic_strict_rate"
            ],
            "optimizer_update_count_at_formal_endpoint": integrity["update_count"],
            "last_actor_learning_rate_at_formal_endpoint": integrity[
                "last_actor_learning_rate"
            ],
            "last_critic_learning_rate_at_formal_endpoint": integrity[
                "last_critic_learning_rate"
            ],
            "final_64_stochastic_strict_rate": (
                final_eval["strict_compact_success_rate"] if final_eval else None
            ),
            "final_64_stochastic_episodes": (
                final_eval["evaluation_episodes"] if final_eval else None
            ),
            "mean_return_is_diagnostic_not_selection": True,
        }
        jobs.append(entry)
        curves.extend(run_curves)
        fixed_rows.extend(evaluations)
        if selected:
            selected_curves.extend(run_curves)
            selected_fixed.extend(evaluations)
    if any_deadline_job and not deadline_stop:
        raise RuntimeError(
            f"Runner deadline flag/job status mismatch: {deadline_stop}/{any_deadline_job}"
        )
    selected_science = {}
    for task, arm in ever_selected.items():
        endpoint = min(
            formal[(task, arm, seed)] for seed in SEEDS
        )
        report_selected = latest_selected.get(task)
        if report_selected is None or report_selected.get("endpoint") != endpoint:
            report_selected = staged.analyse_selected(
                manifest["jobs"], task, arm, endpoint
            )
        if not report_selected["arm_summary"]["integrity_pass"]:
            raise RuntimeError(f"Selected final science failed: {task}/{arm}")
        selected_science[task] = report_selected
    anchors = historical_c5_anchors(inputs)
    paired_rows = _paired_seed_rows(screen_science)
    verify_inputs_unchanged(inputs)
    return {
        "schema_version": 1,
        "analysis": "ori_codex10_nonswap_staged_finalizer",
        "created_utc": utc_now(), "root": str(root),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "protocol": manifest["protocol"],
        "reward_authority": (
            "literal Codex6 P0X2RCS05: spectral=0.05, frontier=0.9, rejected "
            "crossing reward=-2+q(last_safe), reject and continue"
        ),
        "selection_metric_note": (
            "First strict discovery in cumulative mechanical proposals and stability; "
            "return is diagnostic only."
        ),
        "proposal_clock_definition": (
            "sum(training_history.steps); each rejected crossing candidate increments "
            "step_count and is therefore counted as one raw action proposal"
        ),
        "overrun_note": (
            "All metrics and curves stop at each causal formal gate; atomic/history rows "
            "after that gate are inventoried but excluded."
        ),
        "historical_c5_caveat": (
            "Codex5 anchors are one unmatched seed (2101), use crossing termination and a "
            "different reward; they are descriptive context, not causal controls."
        ),
        "manifest": manifest,
        "binding": binding,
        "runner_report": report,
        "ever_selected_arms": ever_selected,
        "final_runner_selections": report["selections"],
        "gate_recomputation": gate_rows,
        "screen_science": screen_science,
        "screen_comparison": _arm_summary_rows(screen_science),
        "paired_seed_comparisons": paired_rows,
        "selected_science": selected_science,
        "jobs": jobs, "selected_episode_curves": selected_curves,
        "selected_fixed_evaluations": selected_fixed,
        "all_episode_curves": curves, "all_fixed_evaluations": fixed_rows,
        "historical_c5_anchors": anchors,
        "source_sha256": manifest["source_sha256"],
        "model_sha256": model_authority,
        "finalizer_source_sha256": finalizer_sources,
        "externally_approved_finalizer_source_sha256": finalizer_sources,
        "input_sha256": dict(sorted(inputs.items())),
    }


def serialisable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialisable(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(serialisable(payload), ensure_ascii=False, indent=2,
                   sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    rows = [serialisable(row) for row in rows]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _selected_rows(payload: dict, task: str, seed: int) -> list[dict]:
    return [row for row in payload["selected_episode_curves"]
            if row["task"] == task and row["seed"] == seed]


def _fixed_rows(payload: dict, task: str, seed: int) -> list[dict]:
    return [row for row in payload["selected_fixed_evaluations"]
            if row["task"] == task and row["seed"] == seed]


def render_selected_plot(payload: dict, task: str, path: Path,
                         proposals_axis=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arm = payload["ever_selected_arms"][task]
    colours = {9101: "#2563eb", 9102: "#d97706", 9103: "#059669"}
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    xfield = "cumulative_proposals" if proposals_axis else "episode"
    xlabel = "Cumulative mechanical proposals" if proposals_axis else "Episode"
    job_map = {(row["task"], row["arm"], row["seed"]): row for row in payload["jobs"]}
    for seed in SEEDS:
        rows = _selected_rows(payload, task, seed)
        job_key = (task, arm, seed)
        if job_key not in job_map or not rows:
            raise RuntimeError(f"Selected plot is missing seed {task}/{arm}/s{seed}")
        expected_endpoint = exact_int(
            job_map[job_key].get("formal_endpoint"),
            f"selected plot formal endpoint {task}/{arm}/s{seed}", 1, 10000,
        )
        if len(rows) != expected_endpoint or rows[-1]["episode"] != expected_endpoint:
            raise RuntimeError(f"Selected plot has a truncated seed {task}/{arm}/s{seed}")
        x = [row[xfield] for row in rows]
        colour = colours[seed]
        axes[0, 0].plot(x, [row["strict_rate_256"] for row in rows],
                        color=colour, linewidth=1.0, alpha=.88)
        axes[0, 0].plot(x, [row["strict_rate_512"] for row in rows],
                        color=colour, linewidth=.9, linestyle="--", alpha=.70)
        fixed = _fixed_rows(payload, task, seed)
        if not fixed or fixed[-1]["episode"] != expected_endpoint:
            raise RuntimeError(f"Selected plot lacks formal fixed eval {task}/{arm}/s{seed}")
        axes[0, 0].scatter(
            [row[xfield] for row in fixed],
            [row["strict_compact_success_rate"] for row in fixed],
            color=colour, marker="o", s=18, facecolors="none", linewidths=.8,
        )
        axes[0, 1].plot(
            x, [max(row["terminal_flatness_F_median_256"], 1e-12) for row in rows],
            color=colour, linewidth=.9, alpha=.80,
        )
        axes[1, 0].plot(
            x, [max(row["spread_error_e_median_256"], 1e-12) for row in rows],
            color=colour, linewidth=.9, alpha=.80,
        )
        axes[1, 1].plot(x, [row["constructive_C_rate_256"] for row in rows],
                        color=colour, linewidth=.9, alpha=.82)
        axes[1, 1].plot(x, [row["joint_J_rate_256"] for row in rows],
                        color=colour, linewidth=.85, linestyle="--", alpha=.72)
        axes[2, 0].plot(x, [row["spectral_potential_mean_256"] for row in rows],
                        color=colour, linewidth=.85, alpha=.78)
        axes[2, 0].plot(x, [row["rejected_crossing_rate_mean_256"] for row in rows],
                        color=colour, linewidth=.75, linestyle=":", alpha=.65)
        axes[2, 1].plot(x, [row["return_diagnostic_mean_256"] for row in rows],
                        color=colour, linewidth=.85, alpha=.78)
        job = job_map[job_key]
        if job["first_hit_episode"] is not None:
            hit_x = (job["first_hit_proposals"] if proposals_axis
                     else job["first_hit_episode"])
            index = job["first_hit_episode"] - 1
            marker = "D" if job["first_hit_before_or_at_first_update"] else "*"
            axes[0, 0].scatter(
                [hit_x], [rows[index]["strict_rate_256"]], color=colour,
                marker=marker, s=54 if marker == "*" else 34,
                edgecolors="black", linewidths=.45, zorder=5,
            )
        else:
            axes[0, 0].scatter(
                [x[-1]], [rows[-1]["strict_rate_256"]], color=colour,
                marker="X", s=34, edgecolors="black", linewidths=.45, zorder=5,
            )
    titles = (
        "Strict success: MA256 (solid), MA512 (dashed), fixed-16 (circles)",
        "Terminal flatness F: rolling median 256",
        "Spread error e: rolling median 256",
        "Constructive C (solid) / joint J (dashed): rolling rate 256",
        "Spectral potential (solid) / rejected-crossing rate (dotted)",
        "Return: rolling mean 256 (diagnostic only)",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=.20)
    axes[0, 0].set_ylim(-.03, 1.03)
    axes[1, 1].set_ylim(-.03, 1.03)
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    handles = [Line2D([0], [0], color=colours[s], label=f"seed {s}") for s in SEEDS]
    handles.extend([
        Line2D([0], [0], marker="D", color="black", linestyle="none",
               markersize=5, label="first hit before/at first update"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=7, label="first hit after first update"),
        Line2D([0], [0], marker="X", color="black", linestyle="none",
               markersize=5, label="right-censored without strict hit"),
    ])
    axes[0, 0].legend(handles=handles, fontsize=7, ncol=2, loc="best")
    final_state = (
        "retained" if payload["final_runner_selections"].get(task) == arm else "revoked"
    )
    figure.suptitle(
        f"C10NS1 {task}: screen-selected arm {arm}, final={final_state} "
        "— formal prefixes only"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def render_screen_plot(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["screen_comparison"]
    if len(rows) != 6 or {
        (row["task"], row["arm"]) for row in rows
    } != {
        (task, arm) for task in TASKS for arm in TASK_ARMS[task]
    }:
        raise RuntimeError("Screen plot does not contain the exact six task/arm cells")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = [
        f"{row['task']}\n{row['arm']}@ep{row['screen_endpoint']}"
        for row in rows
    ]
    x = list(range(len(rows)))
    colours = ["#4b5563" if row["arm"] == "B" else "#2563eb" for row in rows]
    selected_edges = ["#dc2626" if row["selected_at_screen"] else "none" for row in rows]
    axes[0, 0].bar(x, [row["common_cap_proposal_rmst"] for row in rows],
                   color=colours, edgecolor=selected_edges, linewidth=1.8)
    axes[0, 0].set_ylabel("Proposal RMST at common cap (lower is better)")
    axes[0, 1].bar(x, [row["hit_seed_count"] for row in rows], color=colours,
                   edgecolor=selected_edges, linewidth=1.8, label="hit seeds")
    axes[0, 1].scatter(x, [row["stable_seed_count"] for row in rows],
                       color="black", marker="x", label="stable seeds")
    axes[0, 1].set_ylim(0, 3.3)
    axes[0, 1].set_ylabel("Seeds (of 3)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(x, [row["mean_tail_strict_rate"] for row in rows],
                   color=colours, edgecolor=selected_edges, linewidth=1.8)
    axes[1, 0].set_ylabel("Mean tail strict rate")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 1].bar(x, [row["mean_fixed_stochastic_strict_rate"] for row in rows],
                   color=colours, edgecolor=selected_edges, linewidth=1.8)
    axes[1, 1].set_ylabel("Mean fixed-16 strict rate")
    axes[1, 1].set_ylim(0, 1.05)
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle(
        "C10NS1 same-seed optimizer screen (red outline = selected at screen)"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def render_plots(payload: dict, output: Path) -> list[Path]:
    paths = []
    for task in TASKS:
        if task not in payload["ever_selected_arms"]:
            continue
        for proposals in (False, True):
            suffix = "proposals" if proposals else "episode"
            path = output / f"learning_curve_{task}_{suffix}.png"
            render_selected_plot(payload, task, path, proposals_axis=proposals)
            paths.append(path)
    screen = output / "screen_optimizer_comparison.png"
    render_screen_plot(payload, screen)
    paths.append(screen)
    return paths


def visual_qa(paths: Sequence[Path]) -> list[dict]:
    import matplotlib.image as mpimg
    result = []
    for path in paths:
        image = mpimg.imread(path)
        if image.ndim < 2 or image.shape[0] < 900 or image.shape[1] < 1200:
            raise RuntimeError(f"Plot dimensions too small: {path} {image.shape}")
        if not bool((image == image).all()) or float(image.std()) < .01:
            raise RuntimeError(f"Plot is blank/nonfinite: {path}")
        result.append({
            "path": path.name, "shape": list(image.shape),
            "pixel_std": float(image.std()), "sha256": sha256_file(path),
        })
    return result


def fmt(value, digits=5) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}g}" if math.isfinite(number) else "—"


def japanese_report(payload: dict) -> str:
    job_map = {(row["task"], row["arm"], row["seed"]): row
               for row in payload["jobs"]}
    lines = [
        "# Codex6 非 plane-swap 課題：optimizer 調整結果", "",
        f"- 作成時刻（UTC）: {payload['created_utc']}",
        f"- manifest fingerprint: `{payload['manifest_fingerprint']}`", "",
        "## 結論", "",
    ]
    for task in TASKS:
        arm = payload["ever_selected_arms"].get(task)
        final = payload["final_runner_selections"].get(task)
        if arm is None:
            lines.append(f"- `{task}`: 4096 episode までに有望な arm を選べず停止しました。")
            continue
        endpoints = [job_map[(task, arm, seed)]["formal_endpoint"] for seed in SEEDS]
        state = ("3 seed とも 10000 episode を完走" if min(endpoints) == 10000
                 else f"arm {arm} を選択後、formal endpoint {min(endpoints)} で停止")
        if final is None:
            state += "（安定性基準で revoke）"
        elif min(endpoints) < 10000 and payload["runner_report"]["deadline_stop"]:
            state += "（実行期限による右打切り）"
        lines.append(f"- `{task}`: `{arm}` を選択し、{state}。")
    lines.extend([
        "", "arm 選択には return を使っていません。初回 strict 成功までの累積 mechanical proposals、同一 seed の比較、strict の持続性、固定乱数 stochastic 評価で判定しています。proposal clock は `training_history.steps` の累積であり、reject された crossing candidate も1 proposalとして数えます。", "",
        "## 選択された arm の結果", "",
        "| task | arm | seed | formal ep | first hit ep | proposals/censor | first update前/同時 | strict(last1024) | fixed-16@formal | final-64 | overrun除外 |",
        "|---|---|---:|---:|---:|---|:---:|---:|---:|---:|---:|",
    ])
    for task, arm in payload["ever_selected_arms"].items():
        for seed in SEEDS:
            row = job_map[(task, arm, seed)]
            lines.append(
                f"| {task} | {arm} | {seed} | {row['formal_endpoint']} | "
                f"{fmt(row['first_hit_episode'], 8)} | "
                f"{fmt(row['first_hit_proposals'], 10) if row['first_hit_event'] else row['first_hit_censor_reason']} | "
                f"{'—' if not row['first_hit_event'] else ('yes' if row['first_hit_before_or_at_first_update'] else 'no')} | "
                f"{fmt(row['strict_rate_last1024'])} | "
                f"{fmt(row['fixed_stochastic_rate_at_formal_endpoint'])} | "
                f"{fmt(row['final_64_stochastic_strict_rate'])} | "
                f"{row['total_ignored_post_formal_rows']} |"
            )
    if payload["jobs"]:
        lines.extend([
            "", "## optimizer arm の固定条件", "",
            "| task | arm | batch | actor LR | critic LR | entropy | actor milestones | factors |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ])
        for task in TASKS:
            for arm in TASK_ARMS[task]:
                row = job_map[(task, arm, SEEDS[0])]
                lines.append(
                    f"| {task} | {arm} | {row['batch_size']} | "
                    f"{fmt(row['actor_initial_learning_rate'], 8)} | "
                    f"{fmt(row['critic_learning_rate'], 8)} | "
                    f"{fmt(row['entropy_coefficient'], 8)} | "
                    f"{row['actor_lr_milestones']} | {row['actor_lr_factors']} |"
                )
    lines.extend([
        "", "`first update前/同時=yes` の初回成功は初期方策による発見を含むため、学習が初回発見を早めた証拠とは扱いません。", "",
        "## B 対 candidate の screen", "",
        "| task | arm | gate ep | selected | hit seed | pre/post update hit | stable seed | proposal RMST | tail strict | fixed-16 | reason |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload["screen_comparison"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['screen_endpoint']} | "
            f"{'yes' if row['selected_at_screen'] else 'no'} | {row['hit_seed_count']} | "
            f"{row['pre_update_hit_seed_count']}/{row['post_update_hit_seed_count']} | "
            f"{row['stable_seed_count']} | {fmt(row['common_cap_proposal_rmst'], 9)} | "
            f"{fmt(row['mean_tail_strict_rate'])} | "
            f"{fmt(row['mean_fixed_stochastic_strict_rate'])} | "
            f"{row['selection_reason']} |"
        )
    lines.extend([
        "", "## 同一 seed の proposal-clock 比較", "",
        "| task | candidate | seed | candidate hit@cap | B hit@cap | pre-update hit candidate/B | candidate proposals | B proposals | proposal Δ | outcome | tail Δ | fixed Δ |",
        "|---|---|---:|:---:|:---:|:---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in payload["paired_seed_comparisons"]:
        lines.append(
            f"| {row['task']} | {row['candidate_arm']} | {row['seed']} | "
            f"{'yes' if row['candidate_event_by_cap'] else 'no'} | "
            f"{'yes' if row['control_event_by_cap'] else 'no'} | "
            f"{'yes' if row['candidate_first_hit_before_or_at_first_update'] else 'no'}/"
            f"{'yes' if row['control_first_hit_before_or_at_first_update'] else 'no'} | "
            f"{row['candidate_first_hit_proposals_capped']} | "
            f"{row['control_first_hit_proposals_capped']} | "
            f"{row['proposal_delta_candidate_minus_control']} | "
            f"{row['outcome_for_candidate']} | "
            f"{fmt(row['tail_strict_rate_delta_candidate_minus_control'])} | "
            f"{fmt(row['fixed_rate_delta_candidate_minus_control'])} |"
        )
    lines.extend([
        "", "## Codex5 の歴史的 anchor（記述的比較のみ）", "",
        "| task | seed | first hit ep | proposals | 初期方策hit | strict(last1024) | final stochastic |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ])
    for row in payload["historical_c5_anchors"]:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['first_hit_episode']} | "
            f"{row['first_hit_proposals']} | "
            f"{'yes' if row['first_hit_before_or_at_first_update'] else 'no'} | "
            f"{fmt(row['strict_rate_last1024'])} | "
            f"{fmt(row['final_stochastic_strict_rate'])} ({row['final_stochastic_episodes']} trials) |"
        )
    lines.extend([
        "", "注意: Codex5 anchor は seed2101 の1本だけで、seed が一致せず、報酬も異なり、crossing 時に episode を終了します。今回の Codex6 は crossing candidate を reject して同じ episode を継続します。したがって差を optimizer の因果効果とは解釈できません。", "",
        "## 曲線と指標", "",
        "選択 arm について episode 軸と累積 proposal 軸の両方を描きました。strict は MA256（実線）と MA512（破線）、固定乱数16試行は白抜き丸です。F は terminal flatness、e は spread error、C は `F<=0.5 and e<=0.1`、J は `F<=0.1 and e<=0.1`（いずれも invalid episode を除外）です。菱形は最初の update 前または同時の初回hit、星形は update 後の初回hit、X は formal endpoint での右打切りです。return は診断表示のみです。", "",
        "## 完全性", "",
        f"manifest、execution binding、runner report、全18 jobのsource/model/config/history、gate artifactを検証し、入力 {len(payload['input_sha256'])} ファイルを SHA-256 で固定しました。", 
        "各 gate の判定は bound analyzer で再計算し、runner report および gate artifact と一致することを確認しています。formal gate 後に書かれた atomic/history overrun は件数を記録し、表・曲線・選択指標から除外しています。", "",
    ])
    report = "\n".join(lines)
    if "\ufffd" in report:
        raise RuntimeError("Japanese report contains Unicode replacement character")
    return report


def validate_output_authority(output: Path, root: Path) -> Path:
    output, root = output.resolve(), root.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("Production output is fixed exactly to C10NS1_final")
    if output == root or root in output.parents:
        raise RuntimeError("Final output must stay outside the campaign root")
    for model in (SCRIPT_DIRECTORY / "model").glob("*"):
        resolved = model.resolve()
        if output == resolved or resolved in output.parents:
            raise RuntimeError("Final output must not be a model descendant")
    return output


def write_bundle(payload: dict, output: Path = DEFAULT_OUTPUT,
                  enforce_authority=True, publish_active_check=True,
                  render=True) -> dict:
    output = output.resolve()
    if enforce_authority:
        output = validate_output_authority(output, Path(payload["root"]))
        approved = payload.get("externally_approved_finalizer_source_sha256")
        if not isinstance(approved, dict) or set(approved) != set(FINALIZER_NAMES):
            raise RuntimeError("External finalizer/test SHA-256 approval is absent")
        observed = validate_finalizer_approval(
            approved[FINALIZER_NAMES[0]], approved[FINALIZER_NAMES[1]], None
        )
        if observed != payload.get("finalizer_source_sha256"):
            raise RuntimeError("Finalizer source authority changed before bundle build")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output.name + ".building.", dir=output.parent
    ) as temporary:
        build = Path(temporary)
        compact = {key: value for key, value in payload.items()
                   if key not in {"selected_episode_curves", "all_episode_curves",
                                  "selected_fixed_evaluations", "all_fixed_evaluations"}}
        write_json(build / "final_report.json", compact)
        write_json(build / "input_hashes.json", {
            "manifest_fingerprint": payload["manifest_fingerprint"],
            "source_sha256": payload["source_sha256"],
            "model_sha256": payload.get("model_sha256", {}),
            "finalizer_source_sha256": payload["finalizer_source_sha256"],
            "externally_approved_finalizer_source_sha256": payload.get(
                "externally_approved_finalizer_source_sha256", {}
            ),
            "input_sha256": payload["input_sha256"],
        })
        write_csv(build / "job_summary.csv", payload["jobs"])
        write_csv(build / "screen_comparison.csv", payload["screen_comparison"])
        write_csv(
            build / "paired_seed_comparisons.csv",
            payload.get("paired_seed_comparisons", []),
        )
        write_csv(build / "selected_learning_curves.csv",
                  payload["selected_episode_curves"])
        write_csv(build / "selected_fixed_evaluations.csv",
                  payload["selected_fixed_evaluations"])
        write_csv(build / "historical_c5_anchors.csv",
                  payload["historical_c5_anchors"])
        (build / "report_ja.md").write_text(
            japanese_report(payload), encoding="utf-8", newline="\n"
        )
        qa = []
        if render:
            qa = visual_qa(render_plots(payload, build))
        write_json(build / "visual_qa.json", qa)
        inventory = {
            path.name: sha256_file(path) for path in sorted(build.iterdir())
            if path.is_file()
        }
        write_json(build / "output_hashes.json", inventory)
        verify_inputs_unchanged(payload["input_sha256"])
        if publish_active_check:
            if not (Path(payload["root"]) / "runner_report.json").is_file():
                raise RuntimeError("runner_report disappeared before publish")
            active = active_c10_pids(Path(payload["root"]))
            if active:
                raise RuntimeError(
                    f"Active C10NS1 PID(s) appeared before publish: {active}"
                )
        os.replace(build, output)
    return {
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
        "output_hashes_sha256": sha256_file(output / "output_hashes.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approved-finalizer-sha256", required=True)
    parser.add_argument("--approved-test-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=8 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.root.resolve() != DEFAULT_ROOT.resolve():
        raise RuntimeError("Production root is fixed exactly to experiments_ori10/C10NS1")
    validate_output_authority(args.output, args.root)
    wait_until_ready(args.root.resolve(), args.wait_seconds, args.poll_seconds)
    payload = analyze_campaign(
        args.root, args.manifest, active_check=True,
        approved_finalizer_sha256=args.approved_finalizer_sha256,
        approved_test_sha256=args.approved_test_sha256,
    )
    result = write_bundle(payload, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
