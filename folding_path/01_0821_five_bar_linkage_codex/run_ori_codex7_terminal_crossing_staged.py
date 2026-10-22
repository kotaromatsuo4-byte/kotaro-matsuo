"""Immutable-manifest staged runner for the two-arm Codex7 experiment.

Dry-run is side-effect free.  Manifest authorization requires explicit hashes
for both stable Codex7 entrypoints.  Execution refuses changed sources/models,
pre-existing output roots, active duplicate jobs, or an unaudited manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import analyze_ori_codex7_terminal_crossing_staged as analysis


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori7"
DEFAULT_ROOT = EXPERIMENT_DIRECTORY / "C7TC2A"
MANIFEST_DIRECTORY = EXPERIMENT_DIRECTORY / "manifests"
CORE_PLANAR = "ori_codex7.py"
CORE_SPATIAL = "ori_codex7_3d.py"
ARMS = analysis.ARMS
SEEDS = analysis.SEEDS
SCREEN_CHECKPOINTS = (512, 1024, 2048, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
MODEL_FILES = ("node_initial.dat", "member.dat", "fix.dat")
SOURCE_FILES = (
    CORE_PLANAR, CORE_SPATIAL,
    "ori_codex6_spectral_subspace.py", "ori_codex6_spectral_subspace_3d.py",
    "ori_codex6_rejected_crossing.py", "ori_codex6_rejected_crossing_3d.py",
    "ori_codex6.py", "ori_codex6_3d.py", "ori_codex5.py", "ori_codex5_3d.py",
    "ori_codex4.py", "ori_codex4_3d.py", "ori_codex3.py",
    "ori_codex2.py", "ori_codex.py", "collision_3d.py", "single_solution_common.py",
    "analyze_ori_codex7_terminal_crossing_staged.py",
    "run_ori_codex7_terminal_crossing_staged.py",
)


@dataclass(frozen=True)
class Task:
    name: str
    profile: str
    model_directory: str
    target_spread: float
    max_steps: int
    max_step_width: float
    flatness_dimension: int
    actor_learning_rate: float
    critic_learning_rate: float
    actor_lr_milestones: tuple[int, int, int]
    near_miss_spread_error_far: float

    @property
    def spatial(self) -> bool:
        return self.flatness_dimension == 2


TASKS = {
    "torus2": Task("torus2", "A", "model/4unit_torus", 2.0,
                   200, 0.2, 2, 1e-3, 3e-4, (1528, 4000, 7000), 1.0),
    "water1": Task("water1", "B", "model/waterbomb", 1.0,
                   400, 0.3, 2, 5e-4, 1.5e-4, (2000, 4000, 7000), 0.12),
    "five_d2": Task("five_d2", "B", "model/five_bar_linkage", 2.0,
                    400, 0.1, 1, 5e-4, 1.5e-4, (2000, 4000, 7000), 1.0),
    "five_d3": Task("five_d3", "B", "model/five_bar_linkage", 3.0,
                    400, 0.1, 1, 5e-4, 1.5e-4, (2000, 4000, 7000), 1.0),
    "watersqrt2": Task("watersqrt2", "B", "model/waterbomb", math.sqrt(2),
                       400, 0.3, 2, 5e-4, 1.5e-4,
                       (2000, 4000, 7000), 1.0),
}


COMMON_ARGUMENTS = (
    "--ori5-flatness-clip", "2",
    "--ori5-spread-milestone-initial-factor", "2",
    "--ori5-spread-milestone-bonus", "0",
    "--ori5-joint-flatness-thresholds", "0.5", "0.1", "0.01",
    "--ori5-joint-milestone-bonuses", "0.2", "0.3", "0.5",
    "--ori5-crossing-penalty", "1.25",
    "--ori5-wrong-flat-penalty", "2", "--ori5-timeout-penalty", "1",
    "--ori5-success-bonus", "10", "--ori5-quality-weight", "1",
    "--ori5-success-speed-bonus-weight", "0.5",
    "--near-miss-flatness-far", "1", "--gamma", "1",
    "--curriculum-waypoint-bonus", "0", "--spread-tolerance", "0.05",
    "--flatness-tolerance", "0.001", "--numerical-failure-penalty", "10",
    "--no-mode-penalty", "5", "--strain-tolerance", "1e-8",
    "--crossing-tolerance", "1e-10", "--min-step-width", "1e-4",
    "--projection-shrink", "0.5", "--max-correction-ratio", "1",
    "--max-actual-step-ratio", "1.25", "--terminate-wrong-flat",
    "--ori6-spread-frontier-mode", "state_potential",
    "--ori6-spread-frontier-bonus", "0.9",
    "--ori6-spread-frontier-flatness-cap", "2",
    "--ori6-spectral-subspace", "--ori6-spectral-weight", "0.05",
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
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def ensure_below(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Path escaped root: {path}")
    return path


def source_hashes() -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bound source: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def model_hashes(task: Task) -> dict[str, str]:
    directory = SCRIPT_DIRECTORY / task.model_directory
    return {name: file_sha256(directory / name) for name in MODEL_FILES}


def output_directory(root: Path, arm: str, task: Task, seed: int) -> Path:
    return ensure_below(root / arm / task.name / task.profile / f"seed{seed}", root)


def build_command(root: Path, arm: str, task: Task, seed: int) -> list[str]:
    entrypoint = CORE_SPATIAL if task.spatial else CORE_PLANAR
    command = [
        sys.executable, "-u", str((SCRIPT_DIRECTORY / entrypoint).resolve()),
        "--episodes", "10000", "--batch-size", "8",
        "--max-steps", str(task.max_steps),
        "--max-step-width", format(task.max_step_width, ".17g"),
        "--flatness-subspace-dimension", str(task.flatness_dimension),
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(task.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(task.critic_learning_rate, ".17g"),
        "--entropy-coefficient", "0.001", "--actor-lr-milestones",
        *[str(value) for value in task.actor_lr_milestones],
        "--actor-lr-factors", "0.3", "0.1", "0.03",
        "--eval-interval", "512", "--periodic-stochastic-eval-episodes", "16",
        "--eval-episodes", "64", "--checkpoint-every", "512",
        "--print-every", "128", "--seed", str(seed),
        "--selection-evaluation-seed", "9350000",
        "--final-evaluation-seed", "9450000", "--no-random-search",
        "--output-dir", str(output_directory(root, arm, task, seed)),
        "--near-miss-spread-error-far",
        format(task.near_miss_spread_error_far, ".17g"),
        "--ori7-crossing-terminal-mode", arm, *COMMON_ARGUMENTS,
    ]
    command.extend(["--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve())])
    if task.spatial:
        command.append("--no-rigid-diagnostics")
    return command


def build_jobs(root: Path) -> list[dict]:
    result = []
    for task in TASKS.values():
        models = model_hashes(task)
        for arm in ARMS:
            for seed in SEEDS:
                command = build_command(root, arm, task, seed)
                task_identity = asdict(task)
                task_identity["actor_lr_milestones"] = list(task.actor_lr_milestones)
                identity = {
                    "task": task_identity, "arm": arm, "seed": seed,
                    "output_directory": str(output_directory(root, arm, task, seed)),
                    "command": command, "model_sha256": models,
                }
                result.append({**identity, "job_fingerprint": canonical_digest(identity)})
    return result


def path_budget(root: Path, jobs: Sequence[dict]) -> dict:
    """Fail before root creation if any known artifact approaches MAX_PATH."""

    paths = [root / "execution_binding.json.tmp", root / "runner_report.json.tmp"]
    core_names = (
        "best_policy_evaluation_summary.csv", "deterministic_evaluation_history.csv",
        "stochastic_evaluation_history.csv", "ori5_milestone_learning_curve.png",
        "near_miss_learning_curve.png", "training_history.csv", "checkpoint.pt.tmp",
        "mixture_distribution.log", "best_policy_metrics.json",
    )
    for job in jobs:
        destination = Path(job["output_directory"])
        paths.extend(destination / name for name in core_names)
        log = root / "_runner_logs" / job["arm"] / job["task"]["name"] / f"seed{job['seed']}"
        paths.extend((log / "stdout.log", log / "stderr.log"))
    for task in TASKS:
        for endpoint in SCREEN_CHECKPOINTS:
            for label in (*ARMS, "_".join(ARMS)):
                base = root / "analysis" / f"checkpoint_{endpoint}_{task}_{label}"
                paths.extend((base / "checkpoint_analysis.json", base / f"learning_curve_{task}.png"))
        for endpoint in PRODUCTION_CHECKPOINTS:
            for arm in ARMS:
                for seed in SEEDS:
                    base = root / "analysis" / f"checkpoint_{endpoint}_{task}_{arm}_seed{seed}"
                    paths.extend((base / "production_checkpoint.json",
                                  base / "learning_curve_selected_seed.png"))
    longest = max(paths, key=lambda path: len(str(path.resolve())))
    result = {"limit": 240, "maximum_known_characters": len(str(longest.resolve())),
              "longest_known_path": str(longest.resolve())}
    if result["maximum_known_characters"] >= result["limit"]:
        raise RuntimeError(f"Known artifact path exceeds safe budget: {result}")
    return result


def manifest_payload(root: Path, approved_planar: str, approved_spatial: str) -> dict:
    sources = source_hashes()
    approved = {CORE_PLANAR: approved_planar.lower(), CORE_SPATIAL: approved_spatial.lower()}
    for name, expected in approved.items():
        if sources[name] != expected:
            raise RuntimeError(f"Approved stable hash mismatch for {name}")
    jobs = build_jobs(root)
    payload = {
        "schema_version": 1,
        "protocol": "codex7_terminal_crossing_two_arm_staged_to_10000",
        "experiment_root": str(root.resolve()), "source_sha256": sources,
        "approved_core_sha256": approved, "tasks": list(TASKS),
        "arms": list(ARMS), "seeds": list(SEEDS), "episodes_cap": 10000,
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "job_order": list(TASKS), "max_parallel": 12,
        "training_stop_local": "20:45", "report_deadline_local": "21:00",
        "wall_time_is_comparison_metric": False,
        "child_thread_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "torch_intraop_expected": 1,
            "torch_interop_descriptive_only": True,
        },
        "crossing_rate_caps": {
            name: analysis.GATES[name].crossing_rate_cap for name in TASKS
        },
        "screen_rule": "fixed cohorts; only integrity/catastrophic stop before 4096",
        "promotion_rule": "at 4096 select an arm only after >=2/3 persistent evidence; continue selected seed only",
        "paired_arm_rule": "if both arms pass, require >=2/3 same-seed wins; ties/inconclusive do not auto-promote",
        "production_stop_rule": "selected seed: integrity immediate; otherwise two consecutive warnings where primary signal is below max(absolute floor, 0.2*4096 baseline) and quality/crossing is degraded",
        "arm_semantics": {
            "no_fixed": "terminal uncommitted crossing r=q_safe",
            "historical_1p25": "terminal uncommitted crossing r=-1.25+q_safe",
        },
        "path_budget": path_budget(root, jobs),
        "jobs": jobs,
    }
    payload["manifest_fingerprint"] = canonical_digest(payload)
    return payload


def validate_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = dict(payload); recorded = content.pop("manifest_fingerprint", None)
    if canonical_digest(content) != recorded:
        raise RuntimeError("Manifest self-fingerprint invalid")
    actual = source_hashes()
    if payload.get("source_sha256") != actual:
        raise RuntimeError("Bound source hashes changed")
    approved = payload.get("approved_core_sha256", {})
    if set(approved) != {CORE_PLANAR, CORE_SPATIAL}:
        raise RuntimeError("Approved core keyset changed")
    if any(actual.get(name) != expected for name, expected in approved.items()):
        raise RuntimeError("Stable core approval changed")
    root = Path(payload["experiment_root"]).resolve()
    if len(payload.get("jobs", [])) != 30:
        raise RuntimeError("Manifest must contain 30 jobs")
    if payload.get("tasks") != list(TASKS) or payload.get("arms") != list(ARMS):
        raise RuntimeError("Manifest cohort changed")
    if payload.get("seeds") != list(SEEDS):
        raise RuntimeError("Manifest seeds changed")
    expected_protocol = {
        "job_order": list(TASKS), "max_parallel": 12,
        "training_stop_local": "20:45", "report_deadline_local": "21:00",
        "wall_time_is_comparison_metric": False,
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "crossing_rate_caps": {
            name: analysis.GATES[name].crossing_rate_cap for name in TASKS
        },
    }
    for field, expected in expected_protocol.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"Manifest protocol field changed: {field}")
    expected_threads = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "torch_intraop_expected": 1,
        "torch_interop_descriptive_only": True,
    }
    if payload.get("child_thread_environment") != expected_threads:
        raise RuntimeError("Manifest child thread contract changed")
    rebuilt = build_jobs(root)
    if payload["jobs"] != rebuilt:
        raise RuntimeError("Manifest commands/models changed")
    if payload.get("path_budget") != path_budget(root, rebuilt):
        raise RuntimeError("Manifest path budget changed")
    return payload


def checkpoint_episode(destination: Path) -> int:
    summary = destination / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if int(payload.get(
                "training_episodes", payload.get("episodes", payload.get("episode", 0))
            )) == 10000:
                return 10000
        except Exception:
            pass
    checkpoint = destination / "checkpoint.pt"
    if checkpoint.is_file():
        try:
            import torch
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return int(value.get("episode", 0))
        except Exception:
            return 0
    history = destination / "training_history.csv"
    if history.is_file():
        try:
            with open(history, encoding="utf-8-sig", newline="") as handle:
                return max((_int_csv(row.get("episode")) for row in csv.DictReader(handle)), default=0)
        except Exception:
            return 0
    return 0


def _int_csv(value) -> int:
    try: return int(float(value))
    except (TypeError, ValueError): return 0


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def active_duplicates() -> list[int]:
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and "
        "($_.CommandLine -like '*ori_codex7.py*' -or "
        "$_.CommandLine -like '*ori_codex7_3d.py*') -and "
        "$_.CommandLine -like '*--ori7-crossing-terminal-mode*'}; "
        "@($p.ProcessId) | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    return [int(value) for value in (payload if isinstance(payload, list) else [payload])]


def launch_job(root: Path, job: dict) -> dict:
    log_dir = root / "_runner_logs" / job["arm"] / job["task"]["name"] / f"seed{job['seed']}"
    log_dir.mkdir(parents=True, exist_ok=False)
    stdout = open(log_dir / "stdout.log", "x", encoding="utf-8")
    stderr = open(log_dir / "stderr.log", "x", encoding="utf-8")
    child_env = {**os.environ, "PYTHONHASHSEED": str(job["seed"])}
    child_env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )})
    process = subprocess.Popen(
        job["command"], cwd=SCRIPT_DIRECTORY, stdout=stdout, stderr=stderr,
        env=child_env,
    )
    return {"job": job, "process": process, "stdout": stdout, "stderr": stderr,
            "checked": set(), "selected": False, "stopped": False,
            "production_warning_count": 0}


def close_item(item: dict) -> None:
    for name in ("stdout", "stderr"):
        handle = item.get(name)
        if handle and not handle.closed: handle.close()


def cohort(items: Sequence[dict], task: str, arm: str) -> list[dict]:
    return [item for item in items if item["job"]["task"]["name"] == task and item["job"]["arm"] == arm]


def generate_analysis(root: Path, endpoint: int, task: str, arms: Sequence[str]):
    payload = analysis.analyze(root, endpoint, tasks=(task,), arms=tuple(arms))
    label = "_".join(arms)
    output = root / "analysis" / f"checkpoint_{endpoint}_{task}_{label}"
    artifacts = analysis.write_artifacts(payload, output)
    return payload, artifacts


def launchable_task_names(task_selected: dict) -> set[str]:
    """First two tasks form one wave; remaining tasks enter one at a time."""

    first_wave = tuple(TASKS)[:2]
    if not all(task in task_selected for task in first_wave):
        return set(first_wave)
    for task in tuple(TASKS)[2:]:
        if task not in task_selected:
            return {task}
    return set()


def reached_unchecked(checkpoints: Sequence[int], checked: set[int], reached: int) -> list[int]:
    """Drain all reached endpoints in order, including after a fast child exit."""
    return [value for value in checkpoints if value <= reached and value not in checked]


def verify_thread_contract() -> dict:
    env = {**os.environ, **{name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )}}
    code = "import json,os,torch; print(json.dumps({'env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},'torch_intraop':torch.get_num_threads(),'torch_interop_descriptive':torch.get_num_interop_threads(),'logical_cpu_count':os.cpu_count()}))"
    completed = subprocess.run([sys.executable, "-c", code], cwd=SCRIPT_DIRECTORY,
                               env=env, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout.strip())
    if set(result["env"].values()) != {"1"} or result["torch_intraop"] != 1:
        raise RuntimeError(f"Child thread contract failed: {result}")
    return result


def execute(payload: dict, max_parallel: int, poll_seconds: float,
            training_deadline_local: str, report_deadline_local: str) -> dict:
    root = Path(payload["experiment_root"]).resolve()
    if root.exists():
        raise FileExistsError(f"Fresh experiment root already exists: {root}")
    if active_duplicates():
        raise RuntimeError("Active Codex7 duplicate processes exist")
    if max_parallel != payload["max_parallel"]:
        raise RuntimeError("CLI max_parallel disagrees with immutable manifest")
    if training_deadline_local != payload["training_stop_local"] or report_deadline_local != payload["report_deadline_local"]:
        raise RuntimeError("CLI deadlines disagree with immutable manifest")
    thread_contract = verify_thread_contract()
    jobs = list(payload["jobs"])
    active: list[dict] = []; all_items: list[dict] = []
    group_eliminated = set(); task_selected = {}; artifacts = []
    deadline = datetime.combine(datetime.now().date(), datetime.strptime(training_deadline_local, "%H:%M").time())
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "execution_binding.json", {
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "source_sha256": payload["source_sha256"], "started_utc": utc_now(),
        "job_order": payload["job_order"], "max_parallel": max_parallel,
        "training_stop_local": training_deadline_local,
        "report_deadline_local": report_deadline_local,
        "child_thread_contract": thread_contract,
    })
    deadline_stop = False
    try:
        while jobs or active:
            while jobs and len(active) < max_parallel:
                allowed = launchable_task_names(task_selected)
                index = next((i for i, job in enumerate(jobs)
                              if job["task"]["name"] in allowed), None)
                if index is None:
                    break
                item = launch_job(root, jobs.pop(index)); active.append(item); all_items.append(item)
            if not deadline_stop and datetime.now() >= deadline:
                deadline_stop = True
                for item in active:
                    terminate_tree(item["process"]); item["stopped"] = True; close_item(item)
                active.clear(); jobs.clear()
            for item in list(active):
                code = item["process"].poll()
                if code is not None:
                    close_item(item); active.remove(item)
                    if code != 0 and not item["stopped"]:
                        raise RuntimeError(f"Child failed: {item['job']['job_fingerprint']} rc={code}")
            # Fixed checkpoint evidence; no positive decision before 4096.
            for task in TASKS:
                for arm in ARMS:
                    members = cohort(all_items, task, arm)
                    if len(members) != 3 or (task, arm) in group_eliminated:
                        continue
                    reached = min(checkpoint_episode(Path(item["job"]["output_directory"])) for item in members)
                    for endpoint in reached_unchecked(SCREEN_CHECKPOINTS, members[0]["checked"], reached):
                        if all(checkpoint_episode(Path(item["job"]["output_directory"])) >= endpoint for item in members):
                            payload_a, made = generate_analysis(root, endpoint, task, (arm,))
                            artifacts.append(made)
                            for item in members: item["checked"].add(endpoint)
                            decision = payload_a["arm_decisions"][task][arm]
                            if endpoint < 4096 and decision["stop"]:
                                group_eliminated.add((task, arm))
                                for item in members:
                                    if item in active:
                                        terminate_tree(item["process"]); item["stopped"] = True
                                        close_item(item); active.remove(item)
                                break
                if task in task_selected:
                    continue
                groups = {arm: cohort(all_items, task, arm) for arm in ARMS}
                ready = all(
                    (task, arm) in group_eliminated
                    or (
                        len(groups[arm]) == 3
                        and all(4096 in item["checked"] for item in groups[arm])
                    )
                    for arm in ARMS
                )
                surviving_arms = tuple(
                    arm for arm in ARMS if (task, arm) not in group_eliminated
                )
                if ready and not surviving_arms:
                    task_selected[task] = {
                        "task": task, "passing_arms": [],
                        "selected_arm": None, "selected_seed": None,
                        "promote_selected_to_10000": False,
                        "reason": "both arms eliminated by integrity/catastrophic gate",
                    }
                elif ready and surviving_arms:
                    payload_t, made = generate_analysis(
                        root, 4096, task, surviving_arms
                    )
                    artifacts.append(made)
                    decision = payload_t["task_decisions"][task]
                    task_selected[task] = decision
                    chosen = (decision["selected_arm"], decision["selected_seed"])
                    for group in groups.values():
                        for item in group:
                            is_chosen = (item["job"]["arm"], item["job"]["seed"]) == chosen
                            item["selected"] = is_chosen
                            if not is_chosen and item in active:
                                terminate_tree(item["process"]); item["stopped"] = True
                                close_item(item); active.remove(item)
            # Continue only the selected seed and inspect every declared production endpoint.
            for task, selection in task_selected.items():
                arm, seed = selection.get("selected_arm"), selection.get("selected_seed")
                if arm is None or seed is None:
                    continue
                selected = next((item for item in all_items
                                 if item["job"]["task"]["name"] == task
                                 and item["job"]["arm"] == arm
                                 and item["job"]["seed"] == seed), None)
                if selected is None:
                    continue
                reached = checkpoint_episode(Path(selected["job"]["output_directory"]))
                for endpoint in reached_unchecked(PRODUCTION_CHECKPOINTS, selected["checked"], reached):
                    report = analysis.seed_report(root, arm, task, seed, endpoint)
                    baseline = analysis.seed_report(root, arm, task, seed, 4096)
                    decision = analysis.production_decision(
                        report, baseline, selected["production_warning_count"]
                    )
                    selected["production_warning_count"] = decision["consecutive_warning_count"]
                    output = root / "analysis" / f"checkpoint_{endpoint}_{task}_{arm}_seed{seed}"
                    made = analysis.write_production_artifacts(report, decision, output)
                    artifacts.append(made); selected["checked"].add(endpoint)
                    if decision["stop"] and selected in active:
                        terminate_tree(selected["process"]); selected["stopped"] = True
                        close_item(selected); active.remove(selected)
                    if decision["stop"]:
                        break
            time.sleep(poll_seconds)
    finally:
        for item in active:
            terminate_tree(item["process"]); close_item(item)
    result = {
        "schema_version": 1, "protocol": payload["protocol"],
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "deadline_stop": deadline_stop, "training_deadline_local": training_deadline_local,
        "report_deadline_local": report_deadline_local,
        "task_selections": task_selected, "artifacts": artifacts,
        "finished_utc": utc_now(),
        "report_deadline_met": datetime.now() < datetime.combine(
            datetime.now().date(), datetime.strptime(report_deadline_local, "%H:%M").time()
        ),
        "endpoints": [],
    }
    launched = {item["job"]["job_fingerprint"]: item for item in all_items}
    for job in payload["jobs"]:
        item = launched.get(job["job_fingerprint"])
        result["endpoints"].append({
            "task": job["task"]["name"], "arm": job["arm"], "seed": job["seed"],
            "launched": item is not None,
            "status": ("queued_censored_at_training_deadline" if item is None and deadline_stop
                       else ("not_launched_by_staged_protocol" if item is None
                             else ("selected" if item["selected"] else "screened"))),
            "selected": bool(item and item["selected"]),
            "endpoint": (checkpoint_episode(Path(job["output_directory"])) if item else 0),
        })
    atomic_json(root / "runner_report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--authorize-manifest", action="store_true")
    mode.add_argument("--execute-manifest", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--approved-core-planar-sha256")
    parser.add_argument("--approved-core-spatial-sha256")
    parser.add_argument("--max-parallel", type=int, default=12)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    parser.add_argument("--training-deadline-local", default="20:45")
    parser.add_argument("--report-deadline-local", default="21:00")
    args = parser.parse_args(argv)
    if not 1 <= args.max_parallel <= 12:
        parser.error("--max-parallel must be between 1 and 12")
    root = ensure_below(args.experiment_root, SCRIPT_DIRECTORY)
    if args.execute_manifest:
        payload = validate_manifest(args.execute_manifest)
        if Path(payload["experiment_root"]).resolve() != root:
            raise RuntimeError("CLI root disagrees with manifest")
        result = execute(payload, args.max_parallel, args.poll_seconds,
                         args.training_deadline_local, args.report_deadline_local)
        print(json.dumps(result, indent=2)); return result
    if not args.approved_core_planar_sha256 or not args.approved_core_spatial_sha256:
        parser.error("draft/authorization requires both approved stable core hashes")
    payload = manifest_payload(root, args.approved_core_planar_sha256,
                               args.approved_core_spatial_sha256)
    if args.draft:
        print(json.dumps(payload, indent=2)); return payload
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIRECTORY / f"codex7_terminal_crossing_staged_{payload['manifest_fingerprint'][:12]}.json"
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, payload); print(path); return payload


if __name__ == "__main__":
    main()
