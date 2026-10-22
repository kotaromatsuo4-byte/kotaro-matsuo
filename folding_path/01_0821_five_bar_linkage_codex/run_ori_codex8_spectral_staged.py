"""Fail-closed staged runner for the Codex8 spectral reward comparison.

This is an additive experiment harness.  It does not edit or resume a parent
run.  A dry run is side-effect free, and execution requires an immutable
manifest bound to approved hashes for both Codex8 entrypoints and every local
runtime dependency.

The formal screen contains three fresh, same-seed arms for torus D=2 and
water D=1.  Cohorts normally remain through episode 4096; integrity or
catastrophic evidence may stop them, and the separately bound strong-evidence
rule may select a new arm at 2048.  Only a clear paired winner may continue,
and all three seeds of that winning arm (not a selected best seed) remain
eligible through episode 10000.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

try:
    import analyze_ori_codex8_spectral_staged as analysis
except ImportError:  # The runner may be inspected while the analyzer is frozen.
    analysis = None


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori8"
DEFAULT_ROOT = EXPERIMENT_DIRECTORY / "C8SP2R1"
MANIFEST_DIRECTORY = EXPERIMENT_DIRECTORY / "manifests"
FORENSIC_PARENT_ROOT = EXPERIMENT_DIRECTORY / "C8SP2"
CONTAMINATED_MANIFEST_FINGERPRINT = (
    "56a0da899663735a1be5768c56c4e6a8f0138389492992a51d71516909fd3620"
)
REPLAY_REASON = (
    "C8SP2 torus cohorts were valid through atomic episode 512, but frozen "
    "analyzer 23cdf444 incorrectly classified a separate horizon-boundary "
    "timeout reward as the ordinary zero-reject total and eliminated all "
    "three arms; the supervisor was stopped before water reached a formal "
    "checkpoint"
)

CORE_PLANAR = "ori_codex8.py"
CORE_SPATIAL = "ori_codex8_3d.py"
ANALYZER = "analyze_ori_codex8_spectral_staged.py"
RUNNER = "run_ori_codex8_spectral_staged.py"

ARMS = ("legacy", "scale", "phase_v2")
SEEDS = (8101, 8102, 8103)
SCREEN_CHECKPOINTS = (512, 1024, 2048, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
DESCRIPTIVE_PAIRED_CHECKPOINTS = (2048, 4096)
SELECTION_RULE_VERSION = "codex8_same_seed_lexicographic_v1"
MODEL_FILES = ("node_initial.dat", "member.dat", "fix.dat")
MAX_PARALLEL = 12
TRAINING_STOP_LOCAL = "09:20"
REPORT_DEADLINE_LOCAL = "10:00"

# The transitive local import closure is intentionally explicit.  A manifest
# becomes invalid if any bound parent implementation or experiment code moves.
SOURCE_FILES = (
    CORE_PLANAR,
    CORE_SPATIAL,
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
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex3.py",
    "ori_codex2.py",
    "ori_codex.py",
    "collision_3d.py",
    "single_solution_common.py",
    ANALYZER,
    RUNNER,
)

# These files determine the parent neural-network observation and architecture.
# Their independent hash set makes the reward-only/POMDP scope auditable.
PARENT_ARCHITECTURE_FILES = (
    "ori_codex.py",
    "ori_codex2.py",
    "ori_codex3.py",
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "ori_codex6.py",
    "ori_codex6_3d.py",
    "ori_codex6_zero_reject.py",
    "ori_codex6_zero_reject_3d.py",
)


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
    scale_lambda: float
    phase_lambda: float


TASKS = {
    "torus2": Task(
        "torus2", "A", "model/4unit_torus", 2.0, 200, 0.2,
        1.0e-3, 3.0e-4, (1528, 4000, 7000), 1.0, 0.05, 1.0,
    ),
    "water1": Task(
        "water1", "B", "model/waterbomb", 1.0, 400, 0.3,
        5.0e-4, 1.5e-4, (2000, 4000, 7000), 0.12, 0.025, 0.5,
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
    "--ori6-spectral-weight", "0",
    "--ori6-spectral-relative-epsilon", "1e-12",
    "--ori6-zero-reject-reward",
    "--ori8-spectral-ablation",
    "--ori8-scale-multiplier", "10",
    "--ori8-close-rhobar-threshold", "0.1",
    "--ori8-turn-delta-threshold", "0.7",
)

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _analysis_module():
    if analysis is None:
        raise RuntimeError(
            f"{ANALYZER} is not available; this draft is not launchable"
        )
    return analysis


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
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def ensure_below(path: Path, root: Path) -> Path:
    resolved, resolved_root = path.resolve(), root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"Path escaped root: {resolved}")
    return resolved


def source_hashes() -> dict[str, str]:
    result = {}
    for name in SOURCE_FILES:
        path = SCRIPT_DIRECTORY / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bound source: {path}")
        result[name] = file_sha256(path)
    return result


def architecture_hashes(sources: dict[str, str] | None = None) -> dict[str, str]:
    sources = source_hashes() if sources is None else sources
    return {name: sources[name] for name in PARENT_ARCHITECTURE_FILES}


def model_hashes(task: Task) -> dict[str, str]:
    directory = SCRIPT_DIRECTORY / task.model_directory
    result = {}
    for name in MODEL_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        result[name] = file_sha256(path)
    return result


def forensic_parent_binding() -> dict:
    """Bind, but never consume or modify, the preserved contaminated root."""
    root = FORENSIC_PARENT_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing preserved forensic root: {root}")
    if (root / "runner_report.json").exists():
        raise RuntimeError("Forensic root unexpectedly contains runner_report.json")
    binding = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        binding[relative] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    required = [
        "execution_binding.json",
        *[
            f"analysis/checkpoint_512_torus2_{arm}/checkpoint_analysis.json"
            for arm in ARMS
        ],
        *[
            f"torus2/{arm}/seed{seed}/checkpoint.pt"
            for arm in ARMS for seed in SEEDS
        ],
    ]
    missing = [name for name in required if name not in binding]
    if missing:
        raise RuntimeError(f"Forensic root is incomplete: {missing}")
    execution = json.loads(
        (root / "execution_binding.json").read_text(encoding="utf-8")
    )
    if execution.get("manifest_fingerprint") != CONTAMINATED_MANIFEST_FINGERPRINT:
        raise RuntimeError("Forensic parent manifest fingerprint changed")
    manifest_path = (
        MANIFEST_DIRECTORY
        / f"codex8_spectral_staged_{CONTAMINATED_MANIFEST_FINGERPRINT[:12]}.json"
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing contaminated manifest: {manifest_path}")
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_content = dict(old_manifest)
    old_recorded = old_content.pop("manifest_fingerprint", None)
    if (
        old_recorded != CONTAMINATED_MANIFEST_FINGERPRINT
        or canonical_digest(old_content) != old_recorded
    ):
        raise RuntimeError("Contaminated manifest self-fingerprint invalid")
    return {
        "root": str(root),
        "manifest_fingerprint": CONTAMINATED_MANIFEST_FINGERPRINT,
        "reason": REPLAY_REASON,
        "file_count": len(binding),
        "files": binding,
        "manifest_file": {
            "path": str(manifest_path),
            "size": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        },
        "runner_report_present": False,
        "reuse_policy": (
            "preserve read-only; copy/resume/reuse no files or optimizer state"
        ),
    }


def arm_lambda(arm: str, task: Task) -> float:
    if arm == "legacy":
        return 0.05
    if arm == "scale":
        return task.scale_lambda
    if arm == "phase_v2":
        return task.phase_lambda
    raise ValueError(f"Unknown arm: {arm}")


def effective_legacy_scale(arm: str, task: Task) -> float | None:
    if arm == "legacy":
        return arm_lambda(arm, task)
    if arm == "scale":
        return 10.0 * arm_lambda(arm, task)
    return None


def effective_parent_spectral_weight(arm: str, task: Task) -> float:
    """Return the exact Codex6 spectral weight used by each runtime arm."""
    if arm == "legacy":
        return 0.05
    if arm == "scale":
        return 10.0 * task.scale_lambda
    if arm == "phase_v2":
        return 0.0
    raise ValueError(f"Unknown arm: {arm}")


def output_directory(root: Path, arm: str, task: Task, seed: int) -> Path:
    # This layout is a public contract shared with the frozen analyzer.  Keep
    # task before arm and do not insert the descriptive profile component.
    return ensure_below(root / task.name / arm / f"seed{seed}", root)


def build_command(root: Path, arm: str, task: Task, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str((SCRIPT_DIRECTORY / CORE_SPATIAL).resolve()),
        "--episodes", "10000",
        "--batch-size", "8",
        "--max-steps", str(task.max_steps),
        "--max-step-width", format(task.max_step_width, ".17g"),
        "--flatness-subspace-dimension", "2",
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(task.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(task.critic_learning_rate, ".17g"),
        "--entropy-coefficient", "0.001",
        "--actor-lr-milestones",
        *[str(value) for value in task.actor_lr_milestones],
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
        "--output-dir", str(output_directory(root, arm, task, seed)),
        "--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve()),
        "--near-miss-spread-error-far",
        format(task.near_miss_spread_error_far, ".17g"),
        "--ori8-spectral-arm", arm,
        "--ori8-lambda", format(arm_lambda(arm, task), ".17g"),
        *COMMON_ARGUMENTS,
        "--no-rigid-diagnostics",
    ]
    return command


def build_jobs(root: Path) -> list[dict]:
    result = []
    for task in TASKS.values():
        models = model_hashes(task)
        for arm in ARMS:
            for seed in SEEDS:
                task_identity = asdict(task)
                task_identity["actor_lr_milestones"] = list(
                    task.actor_lr_milestones
                )
                identity = {
                    "task": task_identity,
                    "arm": arm,
                    "seed": seed,
                    "ori8_lambda": arm_lambda(arm, task),
                    "effective_legacy_scale": effective_legacy_scale(arm, task),
                    "effective_parent_ori6_spectral_weight": (
                        effective_parent_spectral_weight(arm, task)
                    ),
                    "output_directory": str(
                        output_directory(root, arm, task, seed)
                    ),
                    "command": build_command(root, arm, task, seed),
                    "model_sha256": models,
                }
                result.append({
                    **identity,
                    "job_fingerprint": canonical_digest(identity),
                })
    return result


def path_budget(root: Path, jobs: Sequence[dict]) -> dict:
    paths = [
        root / "execution_binding.json.tmp",
        root / "runner_report.json.tmp",
    ]
    core_names = (
        "checkpoint.pt.tmp",
        "training_history.csv",
        "update_history.csv",
        "deterministic_evaluation_history.csv",
        "stochastic_evaluation_history.csv",
        "best_policy_evaluation_summary.csv",
        "best_policy_metrics.json",
        "ori8_spectral_learning_curve.png",
    )
    for job in jobs:
        destination = Path(job["output_directory"])
        paths.extend(destination / name for name in core_names)
        log = (
            root / "_runner_logs" / job["arm"] / job["task"]["name"]
            / f"seed{job['seed']}"
        )
        paths.extend((log / "stdout.log", log / "stderr.log"))
    for task in TASKS:
        for endpoint in SCREEN_CHECKPOINTS:
            for label in (*ARMS, "_".join(ARMS)):
                base = (
                    root / "analysis"
                    / f"checkpoint_{endpoint}_{task}_{label}"
                )
                paths.extend((
                    base / "checkpoint_analysis.json",
                    base / f"learning_curve_{task}.png",
                ))
        for endpoint in PRODUCTION_CHECKPOINTS:
            for arm in ARMS:
                for seed in SEEDS:
                    base = (
                        root / "analysis"
                        / f"checkpoint_{endpoint}_{task}_{arm}_seed{seed}"
                    )
                    paths.extend((
                        base / "production_checkpoint.json",
                        base / "learning_curve_selected_seed.png",
                    ))
    longest = max(paths, key=lambda value: len(str(value.resolve())))
    result = {
        "limit": 240,
        "maximum_known_characters": len(str(longest.resolve())),
        "longest_known_path": str(longest.resolve()),
    }
    if result["maximum_known_characters"] >= result["limit"]:
        raise RuntimeError(f"Known artifact path exceeds safe budget: {result}")
    return result


def arm_semantics() -> dict:
    return {
        "legacy": {
            "potential": "rho*delta",
            "task_lambda": {name: 0.05 for name in TASKS},
            "effective_max_coefficient": {
                name: 0.05 for name in TASKS
            },
        },
        "scale": {
            "potential": "10*rho*delta",
            "scale_multiplier": 10.0,
            "task_lambda": {
                name: task.scale_lambda for name, task in TASKS.items()
            },
            "effective_max_coefficient": {
                name: 10.0 * task.scale_lambda
                for name, task in TASKS.items()
            },
        },
        "phase_v2": {
            "rhobar": "clip(rho/rho0,0,1)",
            "ordered_hidden_ledger": (
                "accepted-valid close rhobar<=0.1; only a later accepted-valid "
                "state with delta>=0.7 sets z=1; no same-step cascade"
            ),
            "potential": (
                "0.1*(1-rhobar) if z=0 else "
                "0.1+0.4*rhobar*delta"
            ),
            "task_lambda": {
                name: task.phase_lambda for name, task in TASKS.items()
            },
        },
    }


def manifest_payload(
    root: Path, approved_planar: str, approved_spatial: str
) -> dict:
    _analysis_module()
    sources = source_hashes()
    approved = {
        CORE_PLANAR: approved_planar.lower(),
        CORE_SPATIAL: approved_spatial.lower(),
    }
    for name, expected in approved.items():
        if len(expected) != 64 or sources[name] != expected:
            raise RuntimeError(f"Approved stable hash mismatch for {name}")
    jobs = build_jobs(root)
    payload = {
        "schema_version": 1,
        "protocol": "codex8_spectral_reward_only_three_arm_staged_replay1",
        "experiment_root": str(root.resolve()),
        "source_sha256": sources,
        "approved_core_sha256": approved,
        "parent_architecture_sha256": architecture_hashes(sources),
        "observation_contract": "exact_Codex6_parent_observation_unchanged",
        "reward_only_change": True,
        "fresh_replay_from_episode_zero": True,
        "forensic_parent": forensic_parent_binding(),
        "replay_policy": (
            "same seeds, frozen core and commands restart from episode zero "
            "in a new root; no checkpoint, history, RNG, model, or optimizer "
            "state is copied from the forensic parent"
        ),
        "hidden_ledger_pomdp_caveat": (
            "phase_v2 z is environment-only and is not appended to actor or "
            "critic observations; a phase-v2 failure does not isolate reward "
            "quality from hidden-ledger critic aliasing"
        ),
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "arm_semantics": arm_semantics(),
        "seeds": list(SEEDS),
        "episodes_cap": 10000,
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "descriptive_paired_checkpoints": list(
            DESCRIPTIVE_PAIRED_CHECKPOINTS
        ),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "job_order": list(TASKS),
        "task_launch_policy": (
            "screen tasks sequentially in job_order; after a task reaches an "
            "authorized strong-early 2048 selection or its formal 4096 "
            "decision, selected production seeds may overlap the next task's "
            "complete nine-job screen cohort"
        ),
        "max_parallel": MAX_PARALLEL,
        "training_stop_local": TRAINING_STOP_LOCAL,
        "report_deadline_local": REPORT_DEADLINE_LOCAL,
        "wall_time_is_comparison_metric": False,
        "child_thread_environment": {
            **THREAD_ENVIRONMENT,
            "torch_intraop_expected": 1,
            "torch_interop_descriptive_only": True,
        },
        "screen_rule": (
            "fixed three-seed cohorts through 2048 and normally through 4096; "
            "before 2048 only integrity or catastrophic stops are permitted; "
            "at 2048 the separately bound strong-early rule is the sole "
            "positive-selection exception"
        ),
        "paired_rule": (
            "same-seed lexicographic comparisons begin at 2048 using: hit by "
            "endpoint; conditional smaller cumulative raw-proposal T_hit; "
            "last-window strict, joint, constructive; lower spread-error "
            "median; lower flatness median; lower reject rate. Return and "
            "wall time are never tie-breakers. At 4096, a new arm must win "
            "at least two of three same-seed comparisons against each other "
            "arm"
        ),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "strong_early_promotion_rule": (
            "at 2048 only: scale or phase_v2 must win at least two of three "
            "same-seed comparisons against each other arm; every task row "
            "and candidate seed passes integrity, candidate farming gate "
            "passes; and at least two candidate seeds either have a strict "
            "first hit by 2048 or satisfy last512 joint>=2 and constructive>=6 "
            "plus previous512 joint>=1 and constructive>=3. Return and phase "
            "events alone cannot qualify. The selected arm remains subject to "
            "an integrity/catastrophic revocation gate at 4096"
        ),
        "promotion_rule": (
            "only a strong-early-qualified winner at 2048 or a clear scale "
            "or phase_v2 task-level winner at 4096 may continue; all three "
            "seeds of that arm, never a selected best seed, continue. A "
            "legacy win is a non-improvement conclusion and receives no "
            "automatic 10000 continuation"
        ),
        "production_stop_rule": (
            "per seed: integrity failure stops immediately; otherwise require "
            "two consecutive collapse warnings relative to its 4096 baseline"
        ),
        "formal_cap_rule": (
            "all analyses slice immutable histories at the declared endpoint; "
            "raw process overrun is reported separately and excluded"
        ),
        "path_budget": path_budget(root, jobs),
        "jobs": jobs,
    }
    payload["manifest_fingerprint"] = canonical_digest(payload)
    return payload


def validate_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = dict(payload)
    recorded = content.pop("manifest_fingerprint", None)
    if canonical_digest(content) != recorded:
        raise RuntimeError("Manifest self-fingerprint invalid")
    actual = source_hashes()
    if payload.get("source_sha256") != actual:
        raise RuntimeError("Bound source hashes changed")
    if payload.get("parent_architecture_sha256") != architecture_hashes(actual):
        raise RuntimeError("Parent architecture hashes changed")
    approved = payload.get("approved_core_sha256", {})
    if set(approved) != {CORE_PLANAR, CORE_SPATIAL}:
        raise RuntimeError("Approved core keyset changed")
    if any(actual.get(name) != expected for name, expected in approved.items()):
        raise RuntimeError("Stable core approval changed")
    root = Path(payload["experiment_root"]).resolve()
    expected_constants = {
        "protocol": "codex8_spectral_reward_only_three_arm_staged_replay1",
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "arm_semantics": arm_semantics(),
        "seeds": list(SEEDS),
        "episodes_cap": 10000,
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "descriptive_paired_checkpoints": list(
            DESCRIPTIVE_PAIRED_CHECKPOINTS
        ),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "job_order": list(TASKS),
        "task_launch_policy": (
            "screen tasks sequentially in job_order; after a task reaches an "
            "authorized strong-early 2048 selection or its formal 4096 "
            "decision, selected production seeds may overlap the next task's "
            "complete nine-job screen cohort"
        ),
        "max_parallel": MAX_PARALLEL,
        "training_stop_local": TRAINING_STOP_LOCAL,
        "report_deadline_local": REPORT_DEADLINE_LOCAL,
        "wall_time_is_comparison_metric": False,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "observation_contract": "exact_Codex6_parent_observation_unchanged",
        "reward_only_change": True,
        "fresh_replay_from_episode_zero": True,
    }
    for field, expected in expected_constants.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"Manifest protocol field changed: {field}")
    expected_threads = {
        **THREAD_ENVIRONMENT,
        "torch_intraop_expected": 1,
        "torch_interop_descriptive_only": True,
    }
    if payload.get("child_thread_environment") != expected_threads:
        raise RuntimeError("Manifest child thread contract changed")
    if len(payload.get("jobs", [])) != 18:
        raise RuntimeError("Manifest must contain exactly 18 jobs")
    rebuilt = build_jobs(root)
    if payload["jobs"] != rebuilt:
        raise RuntimeError("Manifest commands/models changed")
    if payload.get("path_budget") != path_budget(root, rebuilt):
        raise RuntimeError("Manifest path budget changed")
    expected_payload = manifest_payload(
        root, approved[CORE_PLANAR], approved[CORE_SPATIAL]
    )
    if payload != expected_payload:
        raise RuntimeError("Manifest differs from the exact frozen protocol")
    return payload


def _int_csv(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def checkpoint_episode(destination: Path) -> int:
    summary = destination / "summary.json"
    if summary.is_file():
        try:
            value = json.loads(summary.read_text(encoding="utf-8"))
            endpoint = int(value.get(
                "training_episodes",
                value.get("episodes", value.get("episode", 0)),
            ))
            if endpoint == 10000:
                return endpoint
        except Exception:
            pass
    checkpoint = destination / "checkpoint.pt"
    if checkpoint.is_file():
        try:
            import torch
            value = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            return int(value.get("episode", 0))
        except Exception:
            return 0
    history = destination / "training_history.csv"
    if history.is_file():
        try:
            with open(history, encoding="utf-8-sig", newline="") as handle:
                return max(
                    (_int_csv(row.get("episode")) for row in csv.DictReader(handle)),
                    default=0,
                )
        except Exception:
            return 0
    return 0


def reached_unchecked(
    checkpoints: Sequence[int], checked: set[int], reached: int
) -> list[int]:
    return [
        endpoint for endpoint in checkpoints
        if endpoint <= reached and endpoint not in checked
    ]


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
        "($_.CommandLine -like '*ori_codex8.py*' -or "
        "$_.CommandLine -like '*ori_codex8_3d.py*') -and "
        "$_.CommandLine -like '*--ori8-spectral-arm*'}; "
        "@($p.ProcessId) | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    values = payload if isinstance(payload, list) else [payload]
    return [int(value) for value in values]


def verify_thread_contract() -> dict:
    environment = {**os.environ, **THREAD_ENVIRONMENT}
    code = (
        "import json,os,torch; print(json.dumps({"
        "'env':{k:os.environ[k] for k in "
        "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',"
        "'NUMEXPR_NUM_THREADS')},"
        "'torch_intraop':torch.get_num_threads(),"
        "'torch_interop_descriptive':torch.get_num_interop_threads(),"
        "'logical_cpu_count':os.cpu_count()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SCRIPT_DIRECTORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip())
    if set(result["env"].values()) != {"1"}:
        raise RuntimeError(f"Child environment thread contract failed: {result}")
    if result["torch_intraop"] != 1:
        raise RuntimeError(f"Torch intra-op contract failed: {result}")
    return result


def launch_job(root: Path, job: dict) -> dict:
    log_directory = (
        root / "_runner_logs" / job["arm"] / job["task"]["name"]
        / f"seed{job['seed']}"
    )
    log_directory.mkdir(parents=True, exist_ok=False)
    stdout = open(log_directory / "stdout.log", "x", encoding="utf-8")
    stderr = open(log_directory / "stderr.log", "x", encoding="utf-8")
    environment = {
        **os.environ,
        **THREAD_ENVIRONMENT,
        "PYTHONHASHSEED": str(job["seed"]),
    }
    process = subprocess.Popen(
        job["command"],
        cwd=SCRIPT_DIRECTORY,
        stdout=stdout,
        stderr=stderr,
        env=environment,
    )
    return {
        "job": job,
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "checked": set(),
        "selected": False,
        "stopped": False,
        "production_warning_count": 0,
    }


def close_item(item: dict) -> None:
    for name in ("stdout", "stderr"):
        handle = item.get(name)
        if handle and not handle.closed:
            handle.close()


def cohort(
    items: Sequence[dict], task: str, arm: str
) -> list[dict]:
    return [
        item for item in items
        if item["job"]["task"]["name"] == task
        and item["job"]["arm"] == arm
    ]


def generate_analysis(
    root: Path, endpoint: int, task: str, arms: Sequence[str]
):
    module = _analysis_module()
    payload = module.analyze(
        root, endpoint, tasks=(task,), arms=tuple(arms)
    )
    label = "_".join(arms)
    output = root / "analysis" / f"checkpoint_{endpoint}_{task}_{label}"
    artifacts = module.write_artifacts(payload, output)
    return payload, artifacts


def early_stop_kind(decision: dict) -> str | None:
    if not decision.get("stop", False):
        return None
    value = decision.get("stop_kind", decision.get("stop_class"))
    if value not in {"integrity", "catastrophic"}:
        raise RuntimeError(
            "Screen/revocation stop was not integrity/catastrophic"
        )
    return str(value)


def selected_arm_from_decision(decision: dict) -> str | None:
    arm = decision.get("selected_arm")
    promote = bool(decision.get(
        "promote_selected_to_10000", decision.get("promote", False)
    ))
    if arm is None:
        if promote:
            raise RuntimeError("Inconclusive task decision claimed promotion")
        return None
    if arm not in ARMS:
        raise RuntimeError(f"Analyzer selected an unknown arm: {arm}")
    if arm == "legacy":
        raise RuntimeError(
            "Legacy is a non-improvement control and cannot be promoted"
        )
    if not promote:
        raise RuntimeError("Selected arm lacks clear comparative promotion")
    if decision.get("selection_rule_version") != SELECTION_RULE_VERSION:
        raise RuntimeError("Analyzer selection-rule authority changed")
    wins = decision.get("paired_wins")
    if not isinstance(wins, dict):
        raise RuntimeError("Selected arm lacks paired-win evidence")
    selected_wins = wins.get(arm)
    if not isinstance(selected_wins, dict):
        raise RuntimeError("Selected arm lacks nested paired-win evidence")
    for opponent in ARMS:
        if opponent == arm:
            continue
        comparison = selected_wins.get(opponent)
        if not isinstance(comparison, dict):
            raise RuntimeError(f"Missing paired evidence against {opponent}")
        if int(comparison.get("wins", -1)) < 2:
            raise RuntimeError(
                f"Selected arm lacks two of three wins against {opponent}"
            )
    selected_seed = decision.get("selected_seed")
    if selected_seed is not None:
        raise RuntimeError("Codex8 must continue all three seeds, not a best seed")
    return arm


def strong_early_arm_from_decision(decision: dict) -> str | None:
    """Validate the predeclared, deliberately strict episode-2048 shortcut."""
    strong = decision.get("strong_early_promotion")
    if strong is not True:
        if decision.get("selected_arm") is not None or decision.get(
            "promote_selected_to_10000", decision.get("promote", False)
        ):
            raise RuntimeError("Unqualified episode-2048 selection")
        return None
    arm = selected_arm_from_decision(decision)
    if arm is None:
        raise RuntimeError("Strong early promotion lacks a selected arm")
    if decision.get("all_task_integrity") is not True:
        raise RuntimeError("Strong early promotion lacks all-task integrity")
    if decision.get("all_candidate_seeds_integrity_pass") is not True:
        raise RuntimeError("Strong early promotion lacks candidate integrity")
    if decision.get("candidate_farming_gate_pass") is not True:
        raise RuntimeError("Strong early promotion failed the farming gate")
    if int(decision.get("early_evidence_seed_count", -1)) < 2:
        raise RuntimeError("Strong early promotion lacks two evidence seeds")
    return arm


def record_task_selection(
    task: str,
    decision: dict,
    selected_arm: str | None,
    groups: dict[str, list[dict]],
    active: list[dict],
    selection_endpoint: int,
) -> dict:
    recorded = dict(decision)
    recorded["selection_endpoint"] = selection_endpoint
    recorded["continued_seed_policy"] = (
        "all_three_paired_seeds" if selected_arm else "none"
    )
    recorded["continued_seeds"] = list(SEEDS) if selected_arm else []
    for arm, members in groups.items():
        for item in members:
            item["selected"] = arm == selected_arm
            if not item["selected"] and item in active:
                terminate_tree(item["process"])
                item["stopped"] = True
                close_item(item)
                active.remove(item)
    return recorded


def task_screen_launchable(job: dict, selected: dict[str, dict]) -> bool:
    """Keep each nine-job screen cohort coherent under the 12-child cap."""
    order = list(TASKS)
    task = job["task"]["name"]
    index = order.index(task)
    return all(previous in selected for previous in order[:index])


def formal_endpoint(checked: set[int]) -> int:
    return max(checked, default=0)


def endpoint_record(job: dict, item: dict | None, deadline_stop: bool) -> dict:
    if item is None:
        return {
            "task": job["task"]["name"],
            "arm": job["arm"],
            "seed": job["seed"],
            "launched": False,
            "status": (
                "queued_censored_at_training_deadline"
                if deadline_stop else "not_launched_by_staged_protocol"
            ),
            "selected": False,
            "raw_endpoint": 0,
            "formal_analyzed_endpoint": 0,
            "overrun_episode_count": 0,
            "overrun_excluded_from_formal_analysis": False,
        }
    raw = checkpoint_episode(Path(job["output_directory"]))
    formal = formal_endpoint(item["checked"])
    return {
        "task": job["task"]["name"],
        "arm": job["arm"],
        "seed": job["seed"],
        "launched": True,
        "status": "selected" if item["selected"] else "screened",
        "selected": bool(item["selected"]),
        "raw_endpoint": raw,
        "formal_analyzed_endpoint": formal,
        "overrun_episode_count": max(raw - formal, 0),
        "overrun_excluded_from_formal_analysis": raw > formal,
    }


def execute(
    payload: dict,
    max_parallel: int,
    poll_seconds: float,
    training_deadline_local: str,
    report_deadline_local: str,
) -> dict:
    root = Path(payload["experiment_root"]).resolve()
    if root.exists():
        raise FileExistsError(f"Fresh experiment root already exists: {root}")
    duplicates = active_duplicates()
    if duplicates:
        raise RuntimeError(f"Active Codex8 duplicate processes exist: {duplicates}")
    if max_parallel != payload["max_parallel"]:
        raise RuntimeError("CLI max_parallel disagrees with immutable manifest")
    if training_deadline_local != payload["training_stop_local"]:
        raise RuntimeError("CLI training deadline disagrees with manifest")
    if report_deadline_local != payload["report_deadline_local"]:
        raise RuntimeError("CLI report deadline disagrees with manifest")
    thread_contract = verify_thread_contract()
    jobs = list(payload["jobs"])
    active: list[dict] = []
    all_items: list[dict] = []
    group_eliminated: set[tuple[str, str]] = set()
    combined_checked: set[tuple[str, int]] = set()
    task_selected: dict[str, dict] = {}
    artifacts = []
    deadline = datetime.combine(
        datetime.now().date(),
        datetime.strptime(training_deadline_local, "%H:%M").time(),
    )
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "execution_binding.json", {
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "source_sha256": payload["source_sha256"],
        "parent_architecture_sha256": payload["parent_architecture_sha256"],
        "started_utc": utc_now(),
        "job_order": payload["job_order"],
        "max_parallel": max_parallel,
        "training_stop_local": training_deadline_local,
        "report_deadline_local": report_deadline_local,
        "reward_only_change": True,
        "fresh_replay_from_episode_zero": bool(
            payload.get("fresh_replay_from_episode_zero", False)
        ),
        "forensic_parent_summary": (
            {
                "root": payload["forensic_parent"]["root"],
                "manifest_fingerprint": payload["forensic_parent"][
                    "manifest_fingerprint"
                ],
                "reason": payload["forensic_parent"]["reason"],
                "binding_sha256": canonical_digest(
                    payload["forensic_parent"]
                ),
            }
            if payload.get("forensic_parent") else None
        ),
        "hidden_ledger_pomdp_caveat": payload[
            "hidden_ledger_pomdp_caveat"
        ],
        "child_thread_contract": thread_contract,
    })
    deadline_stop = False
    try:
        while jobs or active:
            while (
                jobs
                and len(active) < max_parallel
                and task_screen_launchable(jobs[0], task_selected)
            ):
                item = launch_job(root, jobs.pop(0))
                active.append(item)
                all_items.append(item)

            if not deadline_stop and datetime.now() >= deadline:
                deadline_stop = True
                for item in list(active):
                    terminate_tree(item["process"])
                    item["stopped"] = True
                    close_item(item)
                active.clear()
                jobs.clear()

            for item in list(active):
                return_code = item["process"].poll()
                if return_code is None:
                    continue
                close_item(item)
                active.remove(item)
                if return_code != 0 and not item["stopped"]:
                    raise RuntimeError(
                        "Child failed: "
                        f"{item['job']['job_fingerprint']} rc={return_code}"
                    )

            # Drain every reached endpoint.  Positive selection normally waits
            # for 4096; the manifest-bound strong 2048 rule is the sole exception.
            for task in TASKS:
                for arm in ARMS:
                    members = cohort(all_items, task, arm)
                    if len(members) != len(SEEDS):
                        continue
                    if (task, arm) in group_eliminated:
                        continue
                    reached = min(
                        checkpoint_episode(Path(item["job"]["output_directory"]))
                        for item in members
                    )
                    for endpoint in reached_unchecked(
                        SCREEN_CHECKPOINTS, members[0]["checked"], reached
                    ):
                        payload_arm, made = generate_analysis(
                            root, endpoint, task, (arm,)
                        )
                        artifacts.append(made)
                        for item in members:
                            item["checked"].add(endpoint)
                        early_selected_here = bool(
                            endpoint == 4096
                            and task in task_selected
                            and task_selected[task].get("selection_endpoint") == 2048
                            and task_selected[task].get("selected_arm") == arm
                        )
                        if endpoint < 4096 or early_selected_here:
                            decision = payload_arm["arm_decisions"][task][arm]
                            if early_stop_kind(decision) is not None:
                                group_eliminated.add((task, arm))
                                for item in members:
                                    if item in active:
                                        terminate_tree(item["process"])
                                        item["stopped"] = True
                                        close_item(item)
                                        active.remove(item)
                                if early_selected_here:
                                    selection = task_selected[task]
                                    selection["comparative_winner_arm"] = arm
                                    selection["selected_arm"] = None
                                    selection["promote_selected_to_10000"] = False
                                    selection["continued_seed_policy"] = (
                                        "revoked_at_4096_integrity_or_catastrophic_gate"
                                    )
                                    selection["continued_seeds"] = []
                                    selection["promotion_revoked_endpoint"] = 4096
                                    selection["promotion_revoked_kind"] = (
                                        decision.get("stop_kind")
                                    )
                                    for item in members:
                                        item["selected"] = False
                                break

                groups = {
                    arm: cohort(all_items, task, arm) for arm in ARMS
                }
                for endpoint in DESCRIPTIVE_PAIRED_CHECKPOINTS:
                    # The formal selection block below owns the 4096 combined
                    # artifact.  Avoid a timing-dependent duplicate write.
                    if endpoint == 4096:
                        continue
                    key = (task, endpoint)
                    if key in combined_checked:
                        continue
                    surviving = tuple(
                        arm for arm in ARMS
                        if (task, arm) not in group_eliminated
                    )
                    ready = bool(surviving) and all(
                        len(groups[arm]) == len(SEEDS)
                        and all(endpoint in item["checked"] for item in groups[arm])
                        for arm in surviving
                    )
                    if ready:
                        payload_paired, made = generate_analysis(
                            root, endpoint, task, surviving
                        )
                        artifacts.append(made)
                        combined_checked.add(key)
                        if endpoint == 2048:
                            decision_2048 = payload_paired.get(
                                "task_decisions", {}
                            ).get(task)
                            if decision_2048 is None:
                                raise RuntimeError(
                                    "Analyzer omitted episode-2048 task decision"
                                )
                            early_arm = strong_early_arm_from_decision(
                                decision_2048
                            )
                            if early_arm is not None and any(
                                4096 in item["checked"]
                                for arm_members in groups.values()
                                for item in arm_members
                            ):
                                # If polling already observed 4096, use the
                                # stronger formal decision and its integrity
                                # gate instead of retrospectively calling it
                                # an early promotion.
                                early_arm = None
                            if early_arm is not None:
                                task_selected[task] = record_task_selection(
                                    task,
                                    decision_2048,
                                    early_arm,
                                    groups,
                                    active,
                                    2048,
                                )

                if task in task_selected:
                    continue
                surviving = tuple(
                    arm for arm in ARMS
                    if (task, arm) not in group_eliminated
                )
                ready_4096 = all(
                    (task, arm) in group_eliminated
                    or (
                        len(groups[arm]) == len(SEEDS)
                        and all(4096 in item["checked"] for item in groups[arm])
                    )
                    for arm in ARMS
                )
                if not ready_4096:
                    continue
                if not surviving:
                    decision = {
                        "task": task,
                        "selected_arm": None,
                        "promote_selected_to_10000": False,
                        "comparative_conclusion": False,
                        "reason": "all arms eliminated by integrity/catastrophic gate",
                    }
                else:
                    payload_task, made = generate_analysis(
                        root, 4096, task, surviving
                    )
                    artifacts.append(made)
                    combined_checked.add((task, 4096))
                    decision = payload_task["task_decisions"][task]
                selected_arm = selected_arm_from_decision(decision)
                task_selected[task] = record_task_selection(
                    task,
                    decision,
                    selected_arm,
                    groups,
                    active,
                    4096,
                )

            # Every seed in a winning arm is an independently monitored
            # production continuation.  No best-seed selection is permitted.
            module = _analysis_module()
            for task, selection in task_selected.items():
                selected_arm = selection.get("selected_arm")
                if selected_arm is None:
                    continue
                for selected in cohort(all_items, task, selected_arm):
                    seed = selected["job"]["seed"]
                    reached = checkpoint_episode(
                        Path(selected["job"]["output_directory"])
                    )
                    for endpoint in reached_unchecked(
                        PRODUCTION_CHECKPOINTS,
                        selected["checked"],
                        reached,
                    ):
                        report = module.seed_report(
                            root, task, selected_arm, seed, endpoint
                        )
                        baseline = module.seed_report(
                            root, task, selected_arm, seed, 4096
                        )
                        production = module.production_decision(
                            report,
                            baseline,
                            selected["production_warning_count"],
                        )
                        selected["production_warning_count"] = int(
                            production["consecutive_warning_count"]
                        )
                        output = (
                            root / "analysis"
                            / f"checkpoint_{endpoint}_{task}_{selected_arm}_seed{seed}"
                        )
                        made = module.write_production_artifacts(
                            report, production, output
                        )
                        artifacts.append(made)
                        selected["checked"].add(endpoint)
                        if production["stop"]:
                            if selected in active:
                                terminate_tree(selected["process"])
                                selected["stopped"] = True
                                close_item(selected)
                                active.remove(selected)
                            break

            if jobs or active:
                time.sleep(poll_seconds)
    finally:
        for item in list(active):
            terminate_tree(item["process"])
            close_item(item)

    launched = {
        item["job"]["job_fingerprint"]: item for item in all_items
    }
    result = {
        "schema_version": 1,
        "protocol": payload["protocol"],
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "fresh_replay_from_episode_zero": bool(
            payload.get("fresh_replay_from_episode_zero", False)
        ),
        "forensic_parent_summary": (
            {
                "root": payload["forensic_parent"]["root"],
                "manifest_fingerprint": payload["forensic_parent"][
                    "manifest_fingerprint"
                ],
                "reason": payload["forensic_parent"]["reason"],
                "binding_sha256": canonical_digest(
                    payload["forensic_parent"]
                ),
            }
            if payload.get("forensic_parent") else None
        ),
        "deadline_stop": deadline_stop,
        "training_deadline_local": training_deadline_local,
        "report_deadline_local": report_deadline_local,
        "task_selections": task_selected,
        "artifacts": artifacts,
        "finished_utc": utc_now(),
        "report_deadline_met": datetime.now() < datetime.combine(
            datetime.now().date(),
            datetime.strptime(report_deadline_local, "%H:%M").time(),
        ),
        "endpoints": [
            endpoint_record(
                job,
                launched.get(job["job_fingerprint"]),
                deadline_stop,
            )
            for job in payload["jobs"]
        ],
    }
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
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    parser.add_argument(
        "--training-deadline-local", default=TRAINING_STOP_LOCAL
    )
    parser.add_argument(
        "--report-deadline-local", default=REPORT_DEADLINE_LOCAL
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_parallel <= MAX_PARALLEL:
        parser.error(f"--max-parallel must be between 1 and {MAX_PARALLEL}")
    root = ensure_below(args.experiment_root, SCRIPT_DIRECTORY)
    if args.execute_manifest:
        payload = validate_manifest(args.execute_manifest)
        if Path(payload["experiment_root"]).resolve() != root:
            raise RuntimeError("CLI root disagrees with manifest")
        result = execute(
            payload,
            args.max_parallel,
            args.poll_seconds,
            args.training_deadline_local,
            args.report_deadline_local,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if (
        not args.approved_core_planar_sha256
        or not args.approved_core_spatial_sha256
    ):
        parser.error(
            "draft/authorization requires both approved stable core hashes"
        )
    payload = manifest_payload(
        root,
        args.approved_core_planar_sha256,
        args.approved_core_spatial_sha256,
    )
    if args.draft:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = (
        MANIFEST_DIRECTORY
        / f"codex8_spectral_staged_replay1_{payload['manifest_fingerprint'][:12]}.json"
    )
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, payload)
    print(path)
    return payload


if __name__ == "__main__":
    main()
