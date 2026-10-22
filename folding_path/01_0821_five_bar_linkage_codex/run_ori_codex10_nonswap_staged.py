"""Manifest-bound Codex6 non-swap optimizer screen and 10k supervisor.

This additive harness keeps the literal Codex6 P0X2RCS05 authority fixed:
rejected crossing candidates are not committed, ordinary reject reward is
``-2 + q(last_safe)``, the episode continues, spectral weight is 0.05, and
the spread-frontier weight is 0.9.  Only optimizer settings differ by arm.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Sequence

import analyze_ori_codex10_nonswap_staged as analysis


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori10"
DEFAULT_ROOT = EXPERIMENT_DIRECTORY / "C10NS1"
MANIFEST_DIRECTORY = EXPERIMENT_DIRECTORY / "manifests"
RUNNER_NAME = "run_ori_codex10_nonswap_staged.py"
ANALYZER_NAME = "analyze_ori_codex10_nonswap_staged.py"
SEEDS = analysis.SEEDS
TASK_ORDER = analysis.TASK_ORDER
SCREEN_CHECKPOINTS = analysis.SCREEN_CHECKPOINTS
PRODUCTION_CHECKPOINTS = analysis.PRODUCTION_CHECKPOINTS
TRAINING_DEADLINE_LOCAL = "22:30"
REPORT_DEADLINE_LOCAL = "23:00"
MAX_PARALLEL = 18
MODEL_FILES = ("node_initial.dat", "member.dat", "fix.dat")

EXPECTED_CORE_HASHES = {
    "ori_codex6_spectral_subspace.py": "4e309f92a412742b6889d44708409be21dd46afbf1066338d56e231e186bd998",
    "ori_codex6_spectral_subspace_3d.py": "83a3d1cf69e1e0f4efbd0b3177e37d8d3e690b2713e906da01d756e3ec1aa953",
    "ori_codex6_rejected_crossing.py": "0040a552693135ae7d48655865f2535d5fa5f10321a4b235ab006cd34f6f5a94",
    "ori_codex6_rejected_crossing_3d.py": "7e9d78bb0cf73b123b57f458dd29b1c7a94baca1414a2451fdedfbec24942ea2",
    "ori_codex6.py": "38a1f6ac1466088b3e208fe9000b32f52cecb4c2537d0da591458bafea851c63",
    "ori_codex6_3d.py": "2f065459de5022248ab1bb7fada04cf317bb875151bb5128d88208152ab8e46d",
    "ori_codex5.py": "bfd9eff2b3c5d835088deb8bd2b1cc020c2aeb702a96538ef3bdda6167b54d5e",
    "ori_codex5_3d.py": "e085f4d651669f3f934a1e12155358666eb91d1df3336724dba0bc7316ae0a00",
    "collision_3d.py": "0733a4eba57e163797698e7d6688fe5efb33ab22ffe3311af223a863bee67a79",
    "ori_codex4.py": "4fb7a16357728e8086ca4973a6bdb31ad07b7724be998893f956e1fae94e9ca6",
    "ori_codex4_3d.py": "4a41aa423d5bae85cb0ba95f84947a0ba35cbd5564600afd7bb94d7cb1c30c4c",
    "ori_codex3.py": "8201569b66de14997e2668929b95301feffe28232befb5bea922329c0dcb87f8",
    "ori_codex2.py": "bb9ee82cb3eb628ddaf31dc1bdc40c8e46ec7fa854990fdc28ee0f307f14f2e7",
    "ori_codex.py": "2d7c1b7b187ef33c0da3f01bb92c2d06934b94e9dd0cd8b53cfa4601057ace3f",
    "single_solution_common.py": "26a266990dcbff7bb6d6fb4f00a09e6ed9dd5a1318e2803067d542f05157158f",
}

EXPECTED_MODEL_HASHES = {
    "model/five_bar_linkage": {
        "node_initial.dat": "0a492f90e618b451d9f8782d3274aa35fd76528b5152de49f533b2dd2a5692fe",
        "member.dat": "aa87857409cf08b575be0ea1c1092884859eeb313b27ad43e4f0ceacb3c608b9",
        "fix.dat": "7e170fed204e26eb76fd7181ff7cdee7fab9389c13574cb64395d1e578d19898",
    },
    "model/waterbomb": {
        "node_initial.dat": "ba5880f3c34252836912c7973f9cb9dcd33f988507010e4af8bccf36425539a5",
        "member.dat": "de02624cf8085b0cc27901b961608704f2fe1ab92f158540e3464eecd7d0507e",
        "fix.dat": "f3ee434ca07c0a2d9d12ca2a6096a177d3a18e147eff87980b064dd9936c3b88",
    },
}

SOURCE_FILES = (
    RUNNER_NAME,
    ANALYZER_NAME,
    "ori_codex6_spectral_subspace.py",
    "ori_codex6_spectral_subspace_3d.py",
    "ori_codex6_rejected_crossing.py",
    "ori_codex6_rejected_crossing_3d.py",
    "ori_codex6.py",
    "ori_codex6_3d.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex3.py",
    "ori_codex2.py",
    "ori_codex.py",
    "collision_3d.py",
    "single_solution_common.py",
)

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Task:
    name: str
    model_directory: str
    target_spread: float
    max_step_width: float
    flatness_subspace_dimension: int
    arms: tuple[str, str]

    @property
    def spatial(self) -> bool:
        return self.flatness_subspace_dimension == 2


@dataclass(frozen=True)
class Optimizer:
    name: str
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    actor_lr_milestones: tuple[int, ...]
    actor_lr_factors: tuple[float, ...]


TASKS = {
    "five_d2": Task("five_d2", "model/five_bar_linkage", 2.0, 0.1, 1, ("B", "W")),
    "five_d3": Task("five_d3", "model/five_bar_linkage", 3.0, 0.1, 1, ("B", "W")),
    "watersqrt2": Task(
        "watersqrt2", "model/waterbomb", math.sqrt(2.0), 0.3, 2, ("B", "E")
    ),
}

OPTIMIZERS = {
    "B": Optimizer("B", 8, 5.0e-4, 1.5e-4, (2000, 4000, 7000), (0.3, 0.1, 0.03)),
    # Actor-only warm start.  The scheduler applies a factor only when
    # episode > milestone, so ep512 itself is the final 1e-3 update.
    "W": Optimizer(
        "W", 8, 1.0e-3, 1.5e-4,
        (512, 2000, 4000, 7000), (0.5, 0.15, 0.05, 0.015),
    ),
    "E": Optimizer("E", 4, 2.5e-4, 7.5e-5, (2000, 4000, 7000), (0.3, 0.1, 0.03)),
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
    "--near-miss-spread-error-far", "1",
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
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic temporary: {temporary}")
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path.resolve()


def ensure_below(path: Path, root: Path) -> Path:
    result, authority = path.resolve(), root.resolve()
    if result != authority and authority not in result.parents:
        raise RuntimeError(f"path escaped root: {result}")
    return result


def source_hashes() -> dict[str, str]:
    result = {}
    for name in SOURCE_FILES:
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source: {path}")
        result[name] = file_sha256(path)
    for name, expected in EXPECTED_CORE_HASHES.items():
        if result[name] != expected:
            raise RuntimeError(f"frozen core hash changed: {name}")
    return result


def model_hashes(task: Task) -> dict[str, str]:
    directory = (SCRIPT_DIRECTORY / task.model_directory).resolve()
    result = {name: file_sha256(directory / name) for name in MODEL_FILES}
    if result != EXPECTED_MODEL_HASHES[task.model_directory]:
        raise RuntimeError(f"frozen model hash changed: {task.model_directory}")
    return result


def output_directory(root: Path, task: Task, arm: str, seed: int) -> Path:
    return ensure_below(root / task.name / arm / f"s{seed}", root)


def entrypoint(task: Task) -> Path:
    name = (
        "ori_codex6_spectral_subspace_3d.py"
        if task.spatial else "ori_codex6_spectral_subspace.py"
    )
    return (SCRIPT_DIRECTORY / name).resolve()


def expected_config(task: Task, command: Sequence[str]) -> dict:
    """Bind every parser field, including model/output paths and all defaults."""
    module_name = (
        "ori_codex6_spectral_subspace_3d"
        if task.spatial else "ori_codex6_spectral_subspace"
    )
    module = importlib.import_module(module_name)
    parsed = vars(module.parse_args(list(command[3:])))
    # Fail now rather than discovering a non-JSON parser value after launch.
    json.dumps(parsed, allow_nan=False)
    return parsed


def build_command(root: Path, task: Task, arm: str, seed: int) -> list[str]:
    optimizer = OPTIMIZERS[arm]
    command = [
        sys.executable, "-u", str(entrypoint(task)),
        "--ori6-spectral-subspace",
        "--episodes", "10000",
        "--batch-size", str(optimizer.batch_size),
        "--max-steps", "400",
        "--max-step-width", format(task.max_step_width, ".17g"),
        "--flatness-subspace-dimension", str(task.flatness_subspace_dimension),
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(optimizer.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(optimizer.critic_learning_rate, ".17g"),
        "--actor-lr-milestones", *[str(x) for x in optimizer.actor_lr_milestones],
        "--actor-lr-factors", *[format(x, ".17g") for x in optimizer.actor_lr_factors],
        "--entropy-coefficient", "0.001",
        "--eval-interval", "512",
        "--periodic-stochastic-eval-episodes", "16",
        "--eval-episodes", "64",
        "--checkpoint-every", "512",
        "--print-every", "128",
        "--seed", str(seed),
        "--selection-evaluation-seed", "9510000",
        "--final-evaluation-seed", "9610000",
        "--no-random-search",
        "--output-dir", str(output_directory(root, task, arm, seed)),
        "--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve()),
        *COMMON_ARGUMENTS,
    ]
    if task.spatial:
        command.append("--no-rigid-diagnostics")
    return command


def build_jobs(root: Path) -> list[dict]:
    sources = source_hashes()
    result = []
    for task_name in TASK_ORDER:
        task = TASKS[task_name]
        models = model_hashes(task)
        for arm in task.arms:
            optimizer = OPTIMIZERS[arm]
            for seed in SEEDS:
                command = build_command(root, task, arm, seed)
                identity = {
                    "task": {**asdict(task), "arms": list(task.arms)},
                    "arm": arm,
                    "optimizer": {
                        **asdict(optimizer),
                        "actor_lr_milestones": list(optimizer.actor_lr_milestones),
                        "actor_lr_factors": list(optimizer.actor_lr_factors),
                    },
                    "seed": seed,
                    "entrypoint": str(entrypoint(task)),
                    "output_directory": str(output_directory(root, task, arm, seed)),
                    "expected_config": expected_config(task, command),
                    "source_sha256": sources,
                    "model_sha256": models,
                    "literal_codex6_control": {
                        "candidate": "P0X2RCS05",
                        "crossing_policy": "reject_candidate_and_continue",
                        "ordinary_reject_reward": "-2 + q(last_safe)",
                        "spectral_weight": 0.05,
                        "frontier_weight": 0.9,
                    },
                }
                identity["command"] = command
                identity["job_fingerprint"] = canonical_digest(identity)
                result.append(identity)
    return result


def manifest_payload(root: Path, max_parallel: int) -> dict:
    if not 6 <= max_parallel <= MAX_PARALLEL:
        raise ValueError(f"max_parallel must be 6..{MAX_PARALLEL}")
    payload = {
        "schema_version": 1,
        "protocol": "codex6_p0x2rcs05_nonswap_optimizer_screen_to_10000",
        "experiment_root": str(root.resolve()),
        "seeds": list(SEEDS),
        "task_order": list(TASK_ORDER),
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "max_parallel": max_parallel,
        "training_deadline_local": TRAINING_DEADLINE_LOCAL,
        "report_deadline_local": REPORT_DEADLINE_LOCAL,
        "all_18_jobs_fresh_from_episode_zero": True,
        "selected_arm_all_three_seeds_continue": True,
        "jobs": build_jobs(root),
        "source_sha256": source_hashes(),
        "generated_utc": utc_now(),
    }
    content = dict(payload)
    content.pop("generated_utc")
    payload["manifest_fingerprint"] = canonical_digest(content)
    return payload


def manifest_path(payload: dict) -> Path:
    return MANIFEST_DIRECTORY / (
        "codex10_nonswap_" + payload["manifest_fingerprint"][:12] + ".json"
    )


def write_manifest(payload: dict) -> Path:
    path = manifest_path(payload)
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        a, b = dict(existing), dict(payload)
        a.pop("generated_utc", None)
        b.pop("generated_utc", None)
        if a != b:
            raise RuntimeError(f"immutable manifest collision: {path}")
        return path.resolve()
    return atomic_json(path, payload)


def validate_manifest(path: Path, root: Path, max_parallel: int) -> dict:
    with open(path, encoding="utf-8") as handle:
        observed = json.load(handle)
    if Path(observed["experiment_root"]).resolve() != root.resolve():
        raise RuntimeError("CLI experiment root disagrees with manifest")
    if int(observed["max_parallel"]) != max_parallel:
        raise RuntimeError("CLI max_parallel disagrees with manifest")
    rebuilt = manifest_payload(root, max_parallel)
    a, b = dict(observed), dict(rebuilt)
    a.pop("generated_utc", None)
    b.pop("generated_utc", None)
    if a != b:
        raise RuntimeError("manifest/source/model/command binding is stale")
    content = dict(observed)
    fingerprint = content.pop("manifest_fingerprint")
    content.pop("generated_utc", None)
    if canonical_digest(content) != fingerprint:
        raise RuntimeError("manifest self-fingerprint mismatch")
    return observed


def exact_nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{label} is not an exact nonnegative integer: {value!r}")
    return int(numeric)


def checkpoint_episode(destination: Path) -> int:
    checkpoint = destination / "checkpoint.pt"
    if checkpoint.is_file():
        try:
            import torch
            # Checkpoints are freshly produced under the manifest-bound local
            # trainer.  Its frozen numpy RNG state is not accepted by
            # weights_only=True, so use the trusted local format and still
            # validate the episode value exactly before scheduling on it.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return exact_nonnegative_integer(payload.get("episode", 0), "checkpoint episode")
        except (
            OSError, RuntimeError, EOFError, TypeError, ValueError,
            pickle.UnpicklingError,
        ):
            return 0
    return 0


def completed_endpoint(destination: Path) -> int:
    if not all((destination / name).is_file() for name in (
        "policy.pt", "summary.json", "run_config.json", "training_history.csv"
    )):
        return checkpoint_episode(destination)
    try:
        with open(destination / "training_history.csv", newline="", encoding="utf-8-sig") as handle:
            endpoint = max((
                exact_nonnegative_integer(row["episode"], "history episode")
                for row in csv.DictReader(handle)
            ), default=0)
        return endpoint
    except (OSError, csv.Error, ValueError, KeyError):
        return checkpoint_episode(destination)


def verify_thread_contract() -> dict:
    environment = {**os.environ, **THREAD_ENVIRONMENT}
    code = (
        "import json,os,torch;print(json.dumps({'env':{k:os.environ[k] for k in "
        "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},"
        "'torch':torch.get_num_threads(),'cpu':os.cpu_count()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=SCRIPT_DIRECTORY, env=environment,
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout)
    if set(payload["env"].values()) != {"1"} or payload["torch"] != 1:
        raise RuntimeError(f"thread contract failed: {payload}")
    return payload


def active_duplicates(root: Path) -> list[int]:
    needle = str(root.resolve()).replace("'", "''")
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and $_.CommandLine -like '*"
        + needle + "*'}; @($p | Select-Object ProcessId,CommandLine)"
        "|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    records = payload if isinstance(payload, list) else [payload]
    authorities = (
        RUNNER_NAME,
        "ori_codex6_spectral_subspace.py",
        "ori_codex6_spectral_subspace_3d.py",
    )
    return [
        int(record["ProcessId"])
        for record in records
        if isinstance(record, dict)
        and int(record.get("ProcessId", -1)) != os.getpid()
        and any(name in str(record.get("CommandLine", "")) for name in authorities)
    ]


def verify_cohort_binding(payload: dict, task_name: str) -> None:
    """Re-hash authorities immediately before each potentially delayed cohort."""
    observed_sources = source_hashes()
    if observed_sources != payload["source_sha256"]:
        raise RuntimeError(f"source binding changed before launching {task_name}")
    task = TASKS[task_name]
    observed_models = model_hashes(task)
    members = [
        job for job in payload["jobs"] if job["task"]["name"] == task_name
    ]
    if len(members) != 6 or any(
        job.get("model_sha256") != observed_models for job in members
    ):
        raise RuntimeError(f"model binding changed before launching {task_name}")


def launch(root: Path, job: dict) -> dict:
    log = root / "_logs" / job["task"]["name"] / job["arm"] / f"s{job['seed']}"
    log.mkdir(parents=True, exist_ok=False)
    stdout = open(log / "stdout.log", "x", encoding="utf-8", buffering=1)
    stderr = open(log / "stderr.log", "x", encoding="utf-8", buffering=1)
    environment = {
        **os.environ, **THREAD_ENVIRONMENT, "PYTHONHASHSEED": str(job["seed"])
    }
    process = subprocess.Popen(
        job["command"], cwd=SCRIPT_DIRECTORY, env=environment,
        stdout=stdout, stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return {
        "job": job, "process": process, "stdout": stdout, "stderr": stderr,
        "selected": False, "stopped": False, "status": "running",
        "checked_selected": set(),
    }


def close_item(item: dict) -> None:
    for key in ("stdout", "stderr"):
        handle = item.get(key)
        if handle and not handle.closed:
            handle.close()


def stop_item(item: dict, reason: str) -> None:
    process = item["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    close_item(item)
    item["stopped"] = True
    item["status"] = reason


def task_items(items: Sequence[dict], task: str, arm: str | None = None) -> list[dict]:
    return [
        item for item in items
        if item["job"]["task"]["name"] == task
        and (arm is None or item["job"]["arm"] == arm)
    ]


def apply_arm_selection(members: Sequence[dict], active: list[dict], arm: str) -> None:
    """Mark one three-seed arm selected and stop only the competing arm."""
    selected = [item for item in members if item["job"]["arm"] == arm]
    rejected = [item for item in members if item["job"]["arm"] != arm]
    if len(selected) != len(SEEDS) or len(rejected) != len(SEEDS):
        raise RuntimeError(
            f"selection cohort must be 3+3 jobs, got {len(selected)}+{len(rejected)}"
        )
    for item in members:
        item["selected"] = item in selected
    for item in rejected:
        if item in active:
            stop_item(item, "not_selected")
            active.remove(item)


def _ready(items: Sequence[dict], endpoint: int) -> bool:
    if not items:
        return False
    if endpoint == 10000 and not all(
        item["process"].poll() == 0 for item in items
    ):
        # Natural completion rewrites config/summary through the rejected,
        # spectral, and 3-D wrappers after the base history reaches 10000.
        # Process exit is the only commit marker for those non-atomic writes.
        return False
    return all(
        completed_endpoint(Path(item["job"]["output_directory"])) >= endpoint
        for item in items
    )


def execute(payload: dict, manifest: Path, poll_seconds: float) -> dict:
    root = Path(payload["experiment_root"]).resolve()
    if root.exists():
        raise FileExistsError(f"fresh experiment root exists: {root}")
    duplicates = active_duplicates(root)
    if duplicates:
        raise RuntimeError(f"duplicate active jobs: {duplicates}")
    thread_contract = verify_thread_contract()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "execution_binding.json", {
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "source_sha256": payload["source_sha256"],
        "thread_contract": thread_contract,
        "started_utc": utc_now(),
    })
    jobs_by_task = {
        task: [job for job in payload["jobs"] if job["task"]["name"] == task]
        for task in TASK_ORDER
    }
    active: list[dict] = []
    all_items: list[dict] = []
    launched: set[str] = set()
    selections: dict[str, str | None] = {}
    screen_checked: dict[str, set[int]] = {task: set() for task in TASK_ORDER}
    production_checked: dict[str, set[int]] = {task: set() for task in TASK_ORDER}
    production_warning_streak: dict[str, int] = {task: 0 for task in TASK_ORDER}
    gate_history: list[dict] = []
    deadline = datetime.combine(
        datetime.now().date(),
        datetime.strptime(payload["training_deadline_local"], "%H:%M").time(),
    )
    deadline_stop = False
    failure = None
    try:
        while True:
            # A task cohort is indivisible.  With max_parallel=12 the third
            # cohort starts after prior loser arms free three slots; with 18
            # all three cohorts launch together.
            for task in TASK_ORDER:
                if task in launched:
                    continue
                if len(active) + 6 <= int(payload["max_parallel"]):
                    verify_cohort_binding(payload, task)
                    for job in jobs_by_task[task]:
                        item = launch(root, job)
                        active.append(item)
                        all_items.append(item)
                    launched.add(task)
                else:
                    break

            if datetime.now() >= deadline:
                deadline_stop = True
                for item in list(active):
                    stop_item(item, "deadline_stop")
                active.clear()
                break

            for item in list(active):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                close_item(item)
                active.remove(item)
                item["status"] = "complete" if returncode == 0 else "failed"
                if returncode != 0 and not item["stopped"]:
                    raise RuntimeError(
                        f"child failed {item['job']['task']['name']}/{item['job']['arm']}/"
                        f"{item['job']['seed']} rc={returncode}"
                    )

            for task in list(launched):
                members = task_items(all_items, task)
                if task not in selections:
                    for endpoint in SCREEN_CHECKPOINTS:
                        if endpoint in screen_checked[task] or not _ready(members, endpoint):
                            continue
                        report = analysis.analyse_task(payload["jobs"], task, endpoint)
                        artifact = root / "analysis" / f"checkpoint_{endpoint}_{task}.json"
                        analysis.write_json(artifact, report)
                        screen_checked[task].add(endpoint)
                        gate_history.append(report)
                        decision = report["decision"]
                        if decision["selected_arm"] is not None:
                            arm = decision["selected_arm"]
                            selections[task] = arm
                            apply_arm_selection(members, active, arm)
                        elif decision["stop_task"]:
                            selections[task] = None
                            for item in members:
                                if item in active:
                                    stop_item(item, "not_promising")
                                    active.remove(item)
                        break
                else:
                    arm = selections[task]
                    if arm is None:
                        continue
                    selected_items = task_items(all_items, task, arm)
                    for endpoint in SCREEN_CHECKPOINTS:
                        if endpoint <= min(screen_checked[task] or {0}):
                            continue
                        if endpoint in screen_checked[task] or not _ready(selected_items, endpoint):
                            continue
                        report = analysis.analyse_selected(
                            payload["jobs"], task, arm, endpoint
                        )
                        artifact = (
                            root / "analysis" / f"checkpoint_{endpoint}_{task}_{arm}_selected.json"
                        )
                        analysis.write_json(artifact, report)
                        screen_checked[task].add(endpoint)
                        gate_history.append(report)
                        if report["decision"]["revoke"]:
                            selections[task] = None
                            for item in selected_items:
                                if item in active:
                                    stop_item(item, "selected_arm_revoked")
                                    active.remove(item)
                        break
                    if selections[task] is None or max(screen_checked[task] or {0}) < 4096:
                        continue
                    for endpoint in PRODUCTION_CHECKPOINTS:
                        if endpoint in production_checked[task]:
                            continue
                        if not _ready(selected_items, endpoint):
                            continue
                        report = analysis.analyse_selected(
                            payload["jobs"], task, arm, endpoint,
                            prior_production_warning_streak=production_warning_streak[task],
                        )
                        production_warning_streak[task] = report[
                            "production_monitor"
                        ]["consecutive_warning_count"]
                        artifact = (
                            root / "analysis"
                            / f"production_{endpoint}_{task}_{arm}_selected.json"
                        )
                        analysis.write_json(artifact, report)
                        production_checked[task].add(endpoint)
                        gate_history.append(report)
                        if report["decision"]["revoke"]:
                            selections[task] = None
                            for item in selected_items:
                                if item in active:
                                    stop_item(item, "production_stability_revoked")
                                    active.remove(item)
                        break
            pending_gate_work = any(
                task in launched and (
                    task not in selections
                    or (
                        selections.get(task) is not None
                        and (
                            max(screen_checked[task] or {0}) < 4096
                            or max(production_checked[task] or {0}) < 10000
                        )
                    )
                )
                for task in TASK_ORDER
            )
            if not active and len(launched) == len(TASK_ORDER) and not pending_gate_work:
                break
            time.sleep(max(1.0, min(30.0, poll_seconds)))
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        for item in list(active):
            stop_item(item, "supervisor_failure")
        active.clear()
    finally:
        for item in list(active):
            stop_item(item, "supervisor_finalization")
        active.clear()

    records = []
    for item in all_items:
        destination = Path(item["job"]["output_directory"])
        records.append({
            "task": item["job"]["task"]["name"],
            "arm": item["job"]["arm"],
            "seed": item["job"]["seed"],
            "status": item["status"],
            "selected": item["selected"],
            "endpoint": completed_endpoint(destination),
            "returncode": item["process"].poll(),
            "output_directory": str(destination),
        })
    report = {
        "schema_version": 1,
        "protocol": payload["protocol"],
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "finished_utc": utc_now(),
        "deadline_stop": deadline_stop,
        "failure": failure,
        "selections": selections,
        "production_warning_streak": production_warning_streak,
        "gate_history": gate_history,
        "jobs": records,
        "report_deadline_met": datetime.now() < datetime.combine(
            datetime.now().date(),
            datetime.strptime(payload["report_deadline_local"], "%H:%M").time(),
        ),
    }
    atomic_json(root / "runner_report.json", report)
    if failure:
        raise RuntimeError(failure)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--authorize-manifest", action="store_true")
    mode.add_argument("--execute-manifest", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-parallel", type=int, default=12)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)
    if not 6 <= args.max_parallel <= MAX_PARALLEL:
        parser.error(f"--max-parallel must be 6..{MAX_PARALLEL}")
    root = ensure_below(args.experiment_root, SCRIPT_DIRECTORY)
    if args.execute_manifest:
        payload = validate_manifest(args.execute_manifest.resolve(), root, args.max_parallel)
        result = execute(payload, args.execute_manifest.resolve(), args.poll_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    payload = manifest_payload(root, args.max_parallel)
    if args.draft:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return payload
    path = write_manifest(payload)
    print(path)
    return str(path)


if __name__ == "__main__":
    main()
