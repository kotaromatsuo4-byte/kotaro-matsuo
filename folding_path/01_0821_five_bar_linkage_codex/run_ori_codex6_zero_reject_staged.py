"""Fresh, deadline-bounded staged runner for the ori6 zero-reject candidate.

The six 10k-cap children start together.  Checkpoints through episode 4096 are
three-seed screening evidence; no task may be positively adopted before the
formal 4096 gate.  A passing task keeps only its preregistered best seed as a
production diagnostic.  The runner never resumes or overwrites a run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import analyze_ori_codex6_zero_reject_staged as analysis


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori6"
FAILED_ATTEMPT_ROOT = (
    EXPERIMENT_DIRECTORY / "staged" / "zero_reject_reward" / "P0X2RCS05ZR0"
)
DEFAULT_ROOT = (
    EXPERIMENT_DIRECTORY / "staged" / "zero_reject_reward" / "P0X2RCS05ZR0R1"
)
MANIFEST_DIRECTORY = EXPERIMENT_DIRECTORY / "manifests"
CANDIDATE = analysis.CANDIDATE
CANDIDATE_SLUG = "P0X2RCS05ZR0R1"
FAILED_ATTEMPT_MANIFEST = (
    MANIFEST_DIRECTORY / "zero_reject_staged_c66413d76779.json"
)
FAILED_ATTEMPT_EXPECTED_SHA256 = {
    "manifest": "09b3bbad94bbf039b46fc661cf2a05d92a3f3a96ffb41c35ed48e4ff401d3ff3",
    "execution_binding": "c9eaef5e42c1b7e0f6831061bd704d6a156e54197877fa8123370920dc45d8c3",
    "checkpoint_analysis": "0ab159c00d5881458ae266eb4d8848bfa1987815a960a122fc2a22551fddddf6",
}
SEEDS = analysis.SEEDS
SCREEN_CHECKPOINTS = (512, 1024, 2048, 3072, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
EXPECTED_CORE_HASHES = {
    "ori_codex6_zero_reject.py": "c113dfeedb8a5e7252e5bac01de9b35d2ccff4d0517392d797bbd5e0a835898d",
    "ori_codex6_zero_reject_3d.py": "b995787a4764e9b464ab98b8fd3d34621a30d10f1ed58dc22027b123143ace42",
}
SOURCE_FILES = (
    "ori_codex6_zero_reject.py",
    "ori_codex6_zero_reject_3d.py",
    "ori_codex6_spectral_subspace.py",
    "ori_codex6_spectral_subspace_3d.py",
    "ori_codex6_rejected_crossing.py",
    "ori_codex6_rejected_crossing_3d.py",
    "ori_codex6.py",
    "ori_codex6_3d.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "collision_3d.py",
    "analyze_ori_codex6_zero_reject_staged.py",
    "run_ori_codex6_zero_reject_staged.py",
)
MODEL_FILES = ("node_initial.dat", "member.dat", "fix.dat")


@dataclass(frozen=True)
class Task:
    name: str
    profile: str
    model_directory: str
    target_spread: float
    max_steps: int
    max_step_width: float
    actor_learning_rate: float
    critic_learning_rate: float
    actor_lr_milestones: tuple[int, int, int]
    near_miss_spread_error_far: float


TASKS = {
    "torus2": Task(
        "torus2", "A", "model/4unit_torus", 2.0, 200, 0.2,
        1.0e-3, 3.0e-4, (1528, 4000, 7000), 1.0,
    ),
    "water1": Task(
        "water1", "B", "model/waterbomb", 1.0, 400, 0.3,
        5.0e-4, 1.5e-4, (2000, 4000, 7000), 0.40,
    ),
}


COMMON_ARGUMENTS = (
    "--ori5-flatness-clip", "2",
    "--ori5-spread-milestone-initial-factor", "2",
    "--ori5-spread-milestone-bonus", "0",
    "--ori5-joint-flatness-thresholds", "0.5", "0.1", "0.01",
    "--ori5-joint-milestone-bonuses", "0.2", "0.3", "0.5",
    "--ori5-crossing-penalty", "2",
    "--ori5-wrong-flat-penalty", "2",
    "--ori5-timeout-penalty", "1",
    "--ori5-success-bonus", "10",
    "--ori5-quality-weight", "1",
    "--ori5-success-speed-bonus-weight", "0.5",
    "--near-miss-flatness-far", "1",
    "--gamma", "1",
    "--curriculum-waypoint-bonus", "0",
    "--spread-tolerance", "0.05",
    "--flatness-tolerance", "0.001",
    "--numerical-failure-penalty", "10",
    "--no-mode-penalty", "5",
    "--strain-tolerance", "1e-8",
    "--crossing-tolerance", "1e-10",
    "--min-step-width", "1e-4",
    "--projection-shrink", "0.5",
    "--max-correction-ratio", "1",
    "--max-actual-step-ratio", "1.25",
    "--terminate-wrong-flat",
    "--ori6-spread-frontier-mode", "state_potential",
    "--ori6-spread-frontier-bonus", "0.9",
    "--ori6-spread-frontier-flatness-cap", "2",
    "--ori6-spectral-subspace",
    "--ori6-spectral-weight", "0.05",
    "--ori6-spectral-relative-epsilon", "1e-12",
    "--ori6-zero-reject-reward",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bound source: {path}")
        hashes[name] = file_sha256(path)
    for name, expected in EXPECTED_CORE_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"Core hash changed for {name}: {hashes[name]} != {expected}"
            )
    return hashes


def model_hashes(task: Task) -> dict[str, str]:
    directory = (SCRIPT_DIRECTORY / task.model_directory).resolve()
    result = {}
    for name in MODEL_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        result[name] = file_sha256(path)
    return result


def output_directory(root: Path, task: Task, seed: int) -> Path:
    destination = (root / task.name / task.profile / f"seed{seed}").resolve()
    if root.resolve() not in destination.parents:
        raise RuntimeError("Output escaped experiment root")
    return destination


def build_command(root: Path, task: Task, seed: int) -> list[str]:
    return [
        sys.executable, "-u", str(SCRIPT_DIRECTORY / "ori_codex6_zero_reject_3d.py"),
        "--episodes", "10000",
        "--batch-size", "8",
        "--max-steps", str(task.max_steps),
        "--max-step-width", format(task.max_step_width, ".17g"),
        "--flatness-subspace-dimension", "2",
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(task.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(task.critic_learning_rate, ".17g"),
        "--entropy-coefficient", "0.001",
        "--actor-lr-milestones", *[str(value) for value in task.actor_lr_milestones],
        "--actor-lr-factors", "0.3", "0.1", "0.03",
        "--eval-interval", "512",
        "--periodic-stochastic-eval-episodes", "16",
        "--eval-episodes", "64",
        "--checkpoint-every", "512",
        "--print-every", "128",
        "--seed", str(seed),
        "--selection-evaluation-seed", "9320000",
        "--final-evaluation-seed", "9420000",
        "--no-random-search",
        "--output-dir", str(output_directory(root, task, seed)),
        "--near-miss-spread-error-far",
        format(task.near_miss_spread_error_far, ".17g"),
        "--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve()),
        "--no-rigid-diagnostics",
        *COMMON_ARGUMENTS,
    ]


def jobs(root: Path) -> list[dict]:
    result = []
    sources = source_hashes()
    for task in TASKS.values():
        models = model_hashes(task)
        task_identity = asdict(task)
        task_identity["actor_lr_milestones"] = list(task.actor_lr_milestones)
        for seed in SEEDS:
            destination = output_directory(root, task, seed)
            command = build_command(root, task, seed)
            identity = {
                "task": task_identity,
                "seed": seed,
                "candidate": CANDIDATE,
                "candidate_slug": CANDIDATE_SLUG,
                "output_directory": str(destination),
                "command": command,
                "source_sha256": sources,
                "model_sha256": models,
            }
            identity["job_fingerprint"] = canonical_digest(identity)
            result.append(identity)
    return result


def gate_specification() -> dict:
    return {
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "checkpoint_512": "snapshot only",
        "checkpoint_1024": (
            "catastrophic stop only; no positive adoption: integrity failure, "
            "or all-seed no-signal plus reject/invalid/extreme-return lock"
        ),
        "checkpoint_2048_3072": "snapshot only; no positive adoption",
        "checkpoint_4096": {
            "formal_gate": "at least 2/3 seeds pass task-specific persistence gate",
            "constructive": "safe last-valid F<=0.5 and spread_error<=0.1",
            "terminal_joint": "same endpoint F<=0.1 and spread_error<=0.1",
            "path_joint_discovery": (
                "milestone ledger is reported separately and is not terminal persistence"
            ),
            "torus_thresholds": asdict(analysis.TASK_GATES["torus2"]),
            "water_thresholds": asdict(analysis.TASK_GATES["water1"]),
            "selection": (
                "among promising seeds only: lexicographic maximum of last512 "
                "terminal-joint rate, last512 constructive rate, last1024 "
                "terminal-joint rate, last1024 constructive rate, full strict "
                "hits, negative normalized terminal joint cost, return, negative seed"
            ),
            "post_gate_scope": (
                "one selected-seed production diagnostic, not 3-seed confirmation"
            ),
        },
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "production_stop": (
            "selected seed stops after two consecutive checkpoint collapse "
            "warnings; integrity failure stops immediately"
        ),
        "reward_integrity": {
            "ordinary_reject_total": 0.0,
            "horizon_reject": "one normal timeout -1 + q(last_safe)",
            "nonfinite_count": 0,
            "maximum_abs_decomposition_error": 1.0e-8,
        },
    }


def failed_attempt_authority() -> dict:
    """Bind the preserved R0 failure without modifying or reusing its root."""
    binding = FAILED_ATTEMPT_ROOT / "execution_manifest_binding.json"
    checkpoint_analysis = (
        FAILED_ATTEMPT_ROOT / "analysis" / "checkpoint_512_torus2" /
        "checkpoint_analysis.json"
    )
    expected_paths = {
        "manifest": FAILED_ATTEMPT_MANIFEST,
        "execution_binding": binding,
        "checkpoint_analysis": checkpoint_analysis,
    }
    observed_hashes = {}
    for label, path in expected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing preserved failed-attempt authority: {path}")
        observed_hashes[label] = file_sha256(path)
        if observed_hashes[label] != FAILED_ATTEMPT_EXPECTED_SHA256[label]:
            raise RuntimeError(f"Preserved failed-attempt authority changed: {label}")
    statuses = []
    for task in TASKS:
        for seed in SEEDS:
            path = (
                FAILED_ATTEMPT_ROOT / "_runner_logs" / task /
                f"seed{seed}" / "status.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing failed-attempt status: {path}")
            with open(path, encoding="utf-8") as handle:
                status = json.load(handle)
            statuses.append({
                "task": task,
                "seed": seed,
                "status": status.get("status"),
                "checkpoint_episode": status.get("checkpoint_episode"),
                "reason": status.get("reason"),
                "sha256": file_sha256(path),
            })
    if [row["checkpoint_episode"] for row in statuses[:3]] != [512, 512, 512]:
        raise RuntimeError("Preserved torus failed-attempt endpoints changed")
    if any(row["checkpoint_episode"] is not None for row in statuses[3:]):
        raise RuntimeError("Preserved water failed-attempt endpoints changed")
    return {
        "attempt_slug": "P0X2RCS05ZR0",
        "root": str(FAILED_ATTEMPT_ROOT.resolve()),
        "root_preserved_and_never_reused": True,
        "manifest_path": str(FAILED_ATTEMPT_MANIFEST.resolve()),
        "authority_sha256": observed_hashes,
        "manifest_fingerprint": (
            "c66413d767791da603eca5c51ab55aa9b407b644c6c6e4dec393f5d556aa117a"
        ),
        "failure_checkpoint": 512,
        "failure_scope": "torus cohort reached atomic 512; water had no atomic checkpoint",
        "failure_reason": (
            "checkpoint analyzer CSV DictWriter had a hand-maintained field list "
            "that omitted newly added window diagnostics: update_count, "
            "valid_joint_rate, median_return, strict_rate, mean_flatness, "
            "invalid_count, and mean_spread_error"
        ),
        "partial_csv_tmp_preserved": str((
            FAILED_ATTEMPT_ROOT / "analysis" / "checkpoint_512_torus2" /
            "windows_512.csv.tmp"
        ).resolve()),
        "statuses": statuses,
    }


def manifest_payload(root: Path, deadline: datetime, stop_at: datetime) -> dict:
    payload = {
        "schema_version": 1,
        "protocol": "ori6_zero_reject_staged_to_10000",
        "candidate": CANDIDATE,
        "candidate_slug": CANDIDATE_SLUG,
        "retry_of_failed_attempt": failed_attempt_authority(),
        "fresh_fixed_seeds": list(SEEDS),
        "episodes_cap": 10000,
        "experiment_root": str(root.resolve()),
        "deadline": deadline.isoformat(),
        "deadline_stop_at": stop_at.isoformat(),
        "task_adjustments": {
            "torus2": (
                "zero reject; evidence-based first actor-LR milestone 2000->1528 "
                "because the scheduler applies only after the milestone, thereby "
                "lowering LR for the batch ending at ep1536 where prior collapse began"
            ),
            "water1": (
                "zero reject; near_miss_spread_error_far 1.0->0.40 so the "
                "eD approximately 0.414 wrong-plane basin receives no positive qD"
            ),
        },
        "gate_specification": gate_specification(),
        "jobs": jobs(root),
        "fresh_start_no_resume": True,
        "failed_attempt_output_excluded_from_discovery_and_reuse": True,
        "all_six_children_start_concurrently": True,
        "generated_utc": utc_now(),
    }
    content = dict(payload)
    content.pop("generated_utc", None)
    payload["manifest_fingerprint"] = canonical_digest(content)
    return payload


def manifest_path(payload: dict) -> Path:
    return MANIFEST_DIRECTORY / (
        "zero_reject_staged_" + payload["manifest_fingerprint"][:12] + ".json"
    )


def write_or_validate_manifest(payload: dict) -> Path:
    path = manifest_path(payload)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            observed = json.load(handle)
        comparable_expected = dict(payload)
        comparable_observed = dict(observed)
        comparable_expected.pop("generated_utc", None)
        comparable_observed.pop("generated_utc", None)
        if comparable_expected != comparable_observed:
            raise RuntimeError(f"Immutable manifest collision: {path}")
    else:
        atomic_json(path, payload)
    return path.resolve()


def write_stale_manifest_exclusions(current_manifest: Path, payload: dict) -> Path:
    records = []
    for path in sorted(MANIFEST_DIRECTORY.glob("zero_reject_staged_*.json")):
        if path.name.endswith("stale_manifest_exclusions.json") or path.resolve() == current_manifest.resolve():
            continue
        reason = (
            "executed_attempt_failed_at_analyzer_csv_checkpoint512"
            if path.resolve() == FAILED_ATTEMPT_MANIFEST.resolve()
            else "superseded_prelaunch_or_dry_run_manifest"
        )
        records.append({
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "reason": reason,
            "excluded_from_current_execution": True,
        })
    record = {
        "schema_version": 1,
        "current_manifest": str(current_manifest.resolve()),
        "current_manifest_fingerprint": payload["manifest_fingerprint"],
        "current_output_root": payload["experiment_root"],
        "preserved_failed_attempt_root": str(FAILED_ATTEMPT_ROOT.resolve()),
        "excluded_manifests": records,
        "generated_utc": utc_now(),
    }
    path = MANIFEST_DIRECTORY / "zero_reject_staged_stale_manifest_exclusions.json"
    atomic_json(path, record)
    return path.resolve()


def checkpoint_episode(destination: Path) -> int | None:
    # A natural 10,000-episode completion is not divisible by the 512 periodic
    # checkpoint cadence.  Treat the complete final artifact set as atomic
    # authority so the 10,000 gate is reachable; otherwise use checkpoint.pt.
    policy_path = destination / "policy.pt"
    if (
        policy_path.is_file()
        and (destination / "summary.json").is_file()
        and (destination / "run_config.json").is_file()
        and history_has_endpoint(destination, 10000)
        and csv_reaches_endpoint(destination / "update_history.csv", 10000)
    ):
        try:
            import torch
            policy = torch.load(
                policy_path, map_location="cpu", weights_only=False
            )
            if int(policy.get("episode", 0)) == 10000:
                return 10000
        except (OSError, RuntimeError, EOFError, ValueError):
            pass
    path = destination / "checkpoint.pt"
    if not path.is_file():
        return None
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = int(payload.get("episode", 0))
        return value if value > 0 else None
    except (OSError, RuntimeError, EOFError, ValueError):
        return None


def csv_reaches_endpoint(path: Path, endpoint: int) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            tail = handle.read().decode("utf-8", errors="ignore")
        lines = [line for line in tail.splitlines() if line.strip()]
        if not lines:
            return False
        # The checkpoint is written only after the full CSV rewrite.  The last
        # complete row therefore certifies every earlier contiguous prefix;
        # the analyzer subsequently verifies exact 1..endpoint continuity.
        last = next(csv.reader([lines[-1]]))
        return int(float(last[-1])) >= endpoint
    except (OSError, csv.Error, ValueError, IndexError):
        return False


def history_has_endpoint(destination: Path, endpoint: int) -> bool:
    return csv_reaches_endpoint(
        destination / "training_history.csv", endpoint
    )


def status_path(root: Path, task: str, seed: int) -> Path:
    return root / "_runner_logs" / task / f"seed{seed}" / "status.json"


def update_status(root: Path, job: dict, **updates) -> None:
    path = status_path(root, job["task"]["name"], job["seed"])
    payload = {
        "schema_version": 1,
        "job_fingerprint": job["job_fingerprint"],
        "task": job["task"]["name"],
        "seed": job["seed"],
        "output_directory": job["output_directory"],
        "updated_utc": utc_now(),
        **updates,
    }
    atomic_json(path, payload)


def launch(root: Path, job: dict) -> dict:
    task = job["task"]["name"]
    seed = job["seed"]
    log_dir = root / "_runner_logs" / task / f"seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    stdout = open(stdout_path, "w", encoding="utf-8", buffering=1)
    stderr = open(stderr_path, "w", encoding="utf-8", buffering=1)
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(seed),
    })
    process = subprocess.Popen(
        job["command"], cwd=SCRIPT_DIRECTORY, env=environment,
        stdout=stdout, stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    update_status(
        root, job, status="running", pid=process.pid,
        started_utc=utc_now(), stdout=str(stdout_path.resolve()),
        stderr=str(stderr_path.resolve()), checkpoint_episode=None,
    )
    return {
        "job": job,
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "active": True,
    }


def stop_item(root: Path, item: dict, reason: str, gate: dict | None = None) -> None:
    process = item["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    item["stdout"].close()
    item["stderr"].close()
    item["active"] = False
    job = item["job"]
    destination = Path(job["output_directory"])
    update_status(
        root, job, status="stopped_at_last_atomic_checkpoint",
        reason=reason, returncode=process.returncode,
        checkpoint_episode=checkpoint_episode(destination), gate=gate,
        stopped_utc=utc_now(),
    )


def finish_item(root: Path, item: dict) -> None:
    process = item["process"]
    returncode = process.poll()
    if returncode is None:
        return
    item["stdout"].close()
    item["stderr"].close()
    item["active"] = False
    job = item["job"]
    destination = Path(job["output_directory"])
    endpoint = checkpoint_episode(destination)
    update_status(
        root, job,
        status="complete" if returncode == 0 and endpoint == 10000 else "failed",
        returncode=returncode, checkpoint_episode=endpoint,
        finished_utc=utc_now(),
    )


def cohort(items: Sequence[dict], task: str) -> list[dict]:
    return [item for item in items if item["job"]["task"]["name"] == task]


def gate_cohort(
    items: Sequence[dict], task: str, selected_seed: int | None = None,
) -> list[dict]:
    """Keep completed screen members; completion must not shrink a cohort."""
    members = cohort(items, task)
    if selected_seed is not None:
        members = [
            item for item in members if item["job"]["seed"] == selected_seed
        ]
    return members


def checkpoint_ready(items: Sequence[dict], endpoint: int) -> bool:
    return bool(items) and all(
        (checkpoint_episode(Path(item["job"]["output_directory"])) or 0) >= endpoint
        and history_has_endpoint(Path(item["job"]["output_directory"]), endpoint)
        for item in items
    )


def write_checkpoint_artifacts(root: Path, payload: dict, endpoint: int, suffix="") -> dict:
    name = f"checkpoint_{endpoint}" + (f"_{suffix}" if suffix else "")
    output = root / "analysis" / name
    # Artifact rendering must never terminate otherwise valid training.  Keep a
    # machine-readable payload/error fallback, then let later checkpoints retry.
    try:
        artifacts = analysis.write_artifacts(payload, output)
        return {"pass": True, "artifacts": artifacts}
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "checkpoint_analysis_fallback.json",
            analysis._serialisable(payload),
        )
        failure = {
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "generated_utc": utc_now(),
        }
        atomic_json(output / "artifact_generation_error.json", failure)
        return failure


def execute(root: Path, manifest: Path, payload: dict, stop_at: datetime, poll_seconds: float) -> int:
    if root.exists():
        raise FileExistsError(f"Fresh experiment root already exists: {root}")
    # Rebuild immediately before launch; source/model changes invalidate execution.
    rebuilt = manifest_payload(root, datetime.fromisoformat(payload["deadline"]), stop_at)
    comparable_a, comparable_b = dict(payload), dict(rebuilt)
    comparable_a.pop("generated_utc", None)
    comparable_b.pop("generated_utc", None)
    if comparable_a != comparable_b:
        raise RuntimeError("Manifest/source binding changed before execution")
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "execution_manifest_binding.json", {
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "started_utc": utc_now(),
    })
    items = []
    try:
        for job in payload["jobs"]:
            items.append(launch(root, job))
    except Exception:
        for item in items:
            if item["active"]:
                stop_item(root, item, "partial_launch_failure")
        raise
    checked: dict[str, set[int]] = {task: set() for task in TASKS}
    selected: dict[str, int] = {}
    baseline_4096: dict[str, dict] = {}
    previous_warnings: dict[str, bool] = {}
    task_active = set(TASKS)
    decisions = []
    try:
        while task_active and datetime.now(stop_at.tzinfo) < stop_at:
            for item in items:
                if item["active"] and item["process"].poll() is not None:
                    finish_item(root, item)
            for task in list(task_active):
                members = gate_cohort(items, task, selected.get(task))
                active_members = [item for item in members if item["active"]]
                schedule = SCREEN_CHECKPOINTS if task not in selected else PRODUCTION_CHECKPOINTS
                processed_checkpoint = False
                for endpoint in schedule:
                    if endpoint in checked[task] or not checkpoint_ready(members, endpoint):
                        continue
                    if task in selected:
                        task_payload = analysis.analyse_checkpoint(
                            root, endpoint, tasks=(task,),
                            selected_seeds={task: selected[task]},
                            baseline_4096=baseline_4096,
                            previous_warnings=previous_warnings,
                        )
                        decision = task_payload["tasks"][task]["decision"]
                        previous_warnings[task] = bool(
                            decision.get("collapse_warning", {}).get("warning", False)
                        )
                    else:
                        task_payload = analysis.analyse_checkpoint(
                            root, endpoint, tasks=(task,), seeds=SEEDS
                        )
                        decision = task_payload["tasks"][task]["decision"]
                    artifact_status = write_checkpoint_artifacts(
                        root, task_payload, endpoint, task
                    )
                    decision["artifact_generation"] = artifact_status
                    processed_checkpoint = True
                    checked[task].add(endpoint)
                    decisions.append(decision)
                    atomic_json(root / "analysis" / "decision_history.json", {
                        "updated_utc": utc_now(), "decisions": decisions,
                    })
                    if endpoint == 1024 and decision["stop"]:
                        for item in active_members:
                            stop_item(root, item, "catastrophic_1024_gate", decision)
                        task_active.discard(task)
                    elif endpoint == 4096:
                        if not decision["promote"]:
                            for item in active_members:
                                stop_item(root, item, "not_promising_at_4096", decision)
                            task_active.discard(task)
                        else:
                            chosen = int(decision["selected_seed"])
                            selected[task] = chosen
                            selected_report = next(
                                report for report in task_payload["tasks"][task]["seeds"]
                                if report["seed"] == chosen
                            )
                            baseline_4096[task] = selected_report
                            for item in active_members:
                                if item["job"]["seed"] != chosen:
                                    stop_item(root, item, "not_selected_after_4096", decision)
                    elif endpoint > 4096 and decision["stop"]:
                        for item in active_members:
                            stop_item(root, item, "sustained_production_collapse", decision)
                        task_active.discard(task)
                    break
                if not active_members and not processed_checkpoint:
                    task_active.discard(task)
            if task_active:
                time.sleep(min(30.0, max(1.0, poll_seconds)))
    finally:
        for item in items:
            if item["active"]:
                stop_item(root, item, "deadline_or_runner_finalization")
    final_tasks = {}
    for task in TASKS:
        task_items = cohort(items, task)
        task_seeds = (selected[task],) if task in selected else SEEDS
        endpoints = [
            checkpoint_episode(Path(item["job"]["output_directory"])) or 0
            for item in task_items if item["job"]["seed"] in task_seeds
        ]
        endpoint = min(endpoints) if endpoints else 0
        if endpoint:
            task_payload = analysis.analyse_checkpoint(
                root, endpoint, tasks=(task,), seeds=task_seeds,
                selected_seeds=({task: selected[task]} if task in selected else None),
                baseline_4096=(baseline_4096 if endpoint > 4096 else None),
                previous_warnings=previous_warnings,
            )
            artifact_status = write_checkpoint_artifacts(
                root, task_payload, endpoint, f"final_{task}"
            )
            final_tasks[task] = {
                "endpoint": endpoint,
                "selected_seed": selected.get(task),
                "analysis": analysis._serialisable(task_payload),
                "artifacts": artifact_status,
            }
    final = {
        "schema_version": 1,
        "protocol": "ori6_zero_reject_staged_to_10000",
        "candidate": CANDIDATE,
        "generated_utc": utc_now(),
        "deadline_stop_at": stop_at.isoformat(),
        "selected_seeds": selected,
        "decisions": decisions,
        "tasks": final_tasks,
        "scope_note": (
            "4096 is a 3-seed formal screen; post-4096 output is a selected-seed "
            "production diagnostic, not 3-seed confirmation"
        ),
    }
    atomic_json(root / "analysis" / "final_summary.json", final)
    return 0


def parse_jst(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=9)))
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--deadline-jst", default="2026-08-27T09:00:00+09:00")
    parser.add_argument("--deadline-buffer-minutes", type=float, default=35.0)
    parser.add_argument("--poll-seconds", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    root = options.experiment_root.resolve()
    deadline = parse_jst(options.deadline_jst)
    stop_at = deadline - timedelta(minutes=options.deadline_buffer_minutes)
    payload = manifest_payload(root, deadline, stop_at)
    manifest = write_or_validate_manifest(payload)
    exclusions = write_stale_manifest_exclusions(manifest, payload)
    print(f"Manifest: {manifest}")
    print(f"Stale exclusions: {exclusions}")
    print(f"Experiment root: {root}")
    print(f"Stop-at: {stop_at.isoformat()}")
    for job in payload["jobs"]:
        print("DRY RUN " + subprocess.list2cmdline(job["command"]))
    if not options.execute:
        return 0
    if datetime.now(stop_at.tzinfo) >= stop_at:
        raise RuntimeError("Refusing to launch after the deadline safety stop time")
    return execute(root, manifest, payload, stop_at, options.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
