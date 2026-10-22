"""Safe, restart-aware orchestration for the ori_codex5 experiment matrix.

The default action is a dry run: it writes an immutable manifest and prints
commands, but starts no training process.  Pass ``--execute`` explicitly to
run pending jobs.  Completed jobs are validated and skipped.  An interrupted
output is never overwritten; ``--rerun-incomplete`` creates a numbered retry
attempt below the original job directory.

All generated files are constrained to ``experiments_ori5`` below this file.
The reward is intentionally not a tuning dimension.  Every command receives
the same explicit ori5 reward arguments and every completed artifact is
checked against that common specification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_ROOT = (SCRIPT_DIRECTORY / "experiments_ori5").resolve()
REWARD_DESIGN = "ori5_endpoint_potential_fixed_milestones"
REWARD_SCHEMA_VERSION = "1.0.0"
SOURCE_FILES = (
    "ori_codex2.py",
    "ori_codex3.py",
    "ori_codex4.py",
    "ori_codex5.py",
    "collision_3d.py",
    "ori_codex4_3d.py",
    "ori_codex5_3d.py",
)


# This is the only permitted reward specification in the experiment matrix.
# Keep it JSON-identical to ori_codex5.reward_parameters(parse_args([])).
COMMON_REWARD_PARAMETERS = {
    "schema_version": REWARD_SCHEMA_VERSION,
    "gamma": 1.0,
    "flatness_clip": 2.0,
    "spread_milestone_initial_factor": 2.0,
    "spread_milestone_bonus": 0.9,
    "joint_flatness_thresholds": [0.5, 0.1, 0.01],
    "joint_milestone_bonuses": [0.2, 0.3, 0.5],
    "crossing_penalty": 1.25,
    "wrong_flat_penalty": 2.0,
    "timeout_penalty": 1.0,
    "success_bonus": 10.0,
    "quality_weight": 1.0,
    "success_speed_bonus_weight": 0.5,
    "near_miss_flatness_far": 1.0,
    "near_miss_spread_error_far": 1.0,
    "path_integrated_cost_weight": 0.0,
    "milestone_after_state_semantics": (
        "all valid running/success/timeout/no_modes/wrong_flat states; "
        "wrong_flat cancels the full episode milestone ledger"
    ),
}


COMMON_REWARD_ARGUMENTS = (
    "--ori5-flatness-clip", "2",
    "--ori5-spread-milestone-initial-factor", "2",
    "--ori5-spread-milestone-bonus", "0.9",
    "--ori5-joint-flatness-thresholds", "0.5", "0.1", "0.01",
    "--ori5-joint-milestone-bonuses", "0.2", "0.3", "0.5",
    "--ori5-crossing-penalty", "1.25",
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
)

FINAL_RUNTIME_SCHEMA_VERSION = "1.0.0"
FINAL_FIXED_RUNTIME_PARAMETERS = {
    "schema_version": FINAL_RUNTIME_SCHEMA_VERSION,
    "numerical_failure_penalty": 10.0,
    "no_mode_penalty": 5.0,
    "flatness_tolerance": 1.0e-3,
    "spread_tolerance": 0.05,
    "strain_tolerance": 1.0e-8,
    "crossing_tolerance": 1.0e-10,
    "min_step_width": 1.0e-4,
    "projection_shrink": 0.5,
    "max_correction_ratio": 1.0,
    "max_actual_step_ratio": 1.25,
    "terminate_wrong_flat": True,
    "quality_combination": "flatness_only",
    "target_deviation_cost_weight": 0.0,
    "joint_success_bonus": 0.0,
    "curriculum_waypoint_bonus": 0.0,
    "provenance": (
        "Stage-2 run_config recorded these effective defaults; the full-stage "
        "CLI makes every externally configurable numerical/termination value "
        "explicit without changing reward semantics"
    ),
}
FINAL_FIXED_RUNTIME_ARGUMENTS = (
    "--numerical-failure-penalty", "10",
    "--no-mode-penalty", "5",
    "--strain-tolerance", "1e-8",
    "--crossing-tolerance", "1e-10",
    "--min-step-width", "1e-4",
    "--projection-shrink", "0.5",
    "--max-correction-ratio", "1",
    "--max-actual-step-ratio", "1.25",
    "--terminate-wrong-flat",
)


@dataclass(frozen=True)
class Task:
    name: str
    entrypoint: str
    target_spread: float
    max_steps: int
    max_step_width: float
    flatness_subspace_dimension: int
    model_directory: str | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    entropy_coefficient: float


@dataclass(frozen=True)
class Stage:
    name: str
    episodes: int
    seed: int
    eval_interval: int
    periodic_evaluation_episodes: int
    final_evaluation_episodes: int
    checkpoint_every: int
    print_every: int
    selection_evaluation_seed: int
    final_evaluation_seed: int


TASKS = {
    "five_d2": Task(
        "five_d2", "ori_codex5.py", 2.0, 400, 0.1, 1,
    ),
    "five_d3": Task(
        "five_d3", "ori_codex5.py", 3.0, 400, 0.1, 1,
    ),
    "torus2": Task(
        "torus2", "ori_codex5_3d.py", 2.0, 200, 0.2, 2,
        "model/4unit_torus",
    ),
    "water1": Task(
        "water1", "ori_codex5_3d.py", 1.0, 400, 0.3, 2,
        "model/waterbomb",
    ),
    "watersqrt2": Task(
        "watersqrt2", "ori_codex5_3d.py", math.sqrt(2.0), 400, 0.3, 2,
        "model/waterbomb",
    ),
}


PROFILES = {
    "A": Profile("A", 8, 1.0e-3, 3.0e-4, 1.0e-3),
    "B": Profile("B", 8, 5.0e-4, 1.5e-4, 1.0e-3),
    "C": Profile("C", 32, 1.0e-3, 3.0e-4, 1.0e-3),
    "D": Profile("D", 32, 5.0e-4, 1.5e-4, 1.0e-3),
    "E": Profile("E", 4, 2.5e-4, 7.5e-5, 1.0e-3),
    "F": Profile("F", 8, 5.0e-4, 1.5e-4, 3.0e-3),
}

STAGE1_PROFILE_NAMES = ("A", "B", "C", "D")
WATER_RESCUE_PROFILE_NAMES = ("E", "F")
STAGE2_MATRIX = {
    "five_d2": ("A", "B"),
    "five_d3": ("A", "B"),
    "torus2": ("A", "C"),
    "water1": ("B", "E", "F"),
    "watersqrt2": ("B", "E", "F"),
}
SELECTION_GROUPS = {
    "five": ("five_d2", "five_d3"),
    "torus": ("torus2",),
    "waterbomb": ("water1", "watersqrt2"),
}
STAGE2_GROUP_PROFILE_CANDIDATES = {
    "five": ("A", "B"),
    "torus": ("A", "C"),
    "waterbomb": ("B", "E", "F"),
}


STAGES = {
    "smoke": Stage(
        "smoke", 8, 9001, 8, 8, 8, 8, 4, 1000003, 2000003,
    ),
    "stage1": Stage(
        "stage1", 512, 1101, 256, 16, 32, 512, 64, 5310000, 5410000,
    ),
    "water_rescue": Stage(
        "water_rescue", 512, 1101, 256, 16, 32, 512, 64,
        5310000, 5410000,
    ),
    "stage2": Stage(
        "stage2", 2048, 1102, 512, 16, 64, 1024, 128,
        5320000, 5420000,
    ),
    "full": Stage(
        "full", 10000, 2101, 500, 50, 100, 1024, 100, 6310000, 6410000,
    ),
}

SMOKE_TASK_SEEDS = {
    "five_d2": 9001,
    "five_d3": 9002,
    "torus2": 9003,
    "water1": 9004,
    "watersqrt2": 9005,
}


def stage_for_task(stage: Stage, task: Task) -> Stage:
    if stage.name == "smoke":
        return replace(stage, seed=SMOKE_TASK_SEEDS[task.name])
    return stage


def profile_names_for_stage_task(stage_name: str, task_name: str) -> tuple[str, ...]:
    """Return the exact profile cells belonging to one stage/task matrix."""

    if stage_name == "smoke":
        return ("A",)
    if stage_name == "stage1":
        return STAGE1_PROFILE_NAMES
    if stage_name == "water_rescue":
        return WATER_RESCUE_PROFILE_NAMES if task_name.startswith("water") else ()
    if stage_name == "stage2":
        return STAGE2_MATRIX.get(task_name, ())
    if stage_name == "full":
        # The effective profile is supplied by selected_profiles.json.
        return ()
    raise ValueError(f"Unknown stage: {stage_name}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def canonical_digest(value) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_within_experiment_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Experiment output must stay below {EXPERIMENT_ROOT}: {resolved}"
        ) from error
    return resolved


def values_equal(actual, expected, *, atol=1.0e-12) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and values_equal(actual[key], value, atol=atol)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            values_equal(left, right, atol=atol)
            for left, right in zip(actual, expected)
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
        except (TypeError, ValueError):
            return False
    return actual == expected


def expected_config(task: Task, profile: Profile, stage: Stage) -> dict:
    return {
        "episodes": stage.episodes,
        "eval_episodes": stage.final_evaluation_episodes,
        "eval_interval": stage.eval_interval,
        "periodic_stochastic_eval_episodes": stage.periodic_evaluation_episodes,
        "batch_size": profile.batch_size,
        "max_steps": task.max_steps,
        "max_step_width": task.max_step_width,
        "flatness_subspace_dimension": task.flatness_subspace_dimension,
        "flatness_tolerance": 1.0e-3,
        "spread_tolerance": 0.05,
        "target_spread": task.target_spread,
        "actor_learning_rate": profile.actor_learning_rate,
        "critic_learning_rate": profile.critic_learning_rate,
        "entropy_coefficient": profile.entropy_coefficient,
        "actor_lr_milestones": [2000, 4000, 7000],
        "actor_lr_factors": [0.3, 0.1, 0.03],
        "checkpoint_every": stage.checkpoint_every,
        "seed": stage.seed,
        "selection_evaluation_seed": stage.selection_evaluation_seed,
        "final_evaluation_seed": stage.final_evaluation_seed,
        "random_search": False,
        "gamma": 1.0,
        "min_step_width": 1.0e-4,
        "projection_shrink": 0.5,
        "max_correction_ratio": 1.0,
        "max_actual_step_ratio": 1.25,
        "strain_tolerance": 1.0e-8,
        "crossing_tolerance": 1.0e-10,
        "terminate_wrong_flat": True,
        "reward_design": REWARD_DESIGN,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "ori5_reward_parameters": COMMON_REWARD_PARAMETERS,
    }


def output_directory(stage: Stage, task: Task, profile: Profile) -> Path:
    # Adopt the already-produced smoke layout.  Stage 1 and full retain the
    # profile and seed in their paths, so every matrix cell is unambiguous.
    if stage.name == "smoke":
        return ensure_within_experiment_root(
            EXPERIMENT_ROOT / "smoke" / task.name
        )
    if stage.name == "stage1":
        stage_root = EXPERIMENT_ROOT / "tuning" / "r1"
    elif stage.name == "stage2":
        stage_root = EXPERIMENT_ROOT / "tuning" / "r2"
    elif stage.name == "water_rescue":
        stage_root = EXPERIMENT_ROOT / "tuning" / "r2_water_rescue"
    else:
        stage_root = EXPERIMENT_ROOT / stage.name
    return ensure_within_experiment_root(
        stage_root / task.name / profile.name
        / f"seed{stage.seed}"
    )


def build_command(
    task: Task, profile: Profile, stage: Stage, destination: Path,
) -> list[str]:
    destination = ensure_within_experiment_root(destination)
    command = [
        sys.executable,
        str(SCRIPT_DIRECTORY / task.entrypoint),
        "--episodes", str(stage.episodes),
        "--batch-size", str(profile.batch_size),
        "--max-steps", str(task.max_steps),
        "--max-step-width", format(task.max_step_width, ".17g"),
        "--flatness-subspace-dimension",
        str(task.flatness_subspace_dimension),
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(profile.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(profile.critic_learning_rate, ".17g"),
        "--entropy-coefficient", format(profile.entropy_coefficient, ".17g"),
        "--actor-lr-milestones", "2000", "4000", "7000",
        "--actor-lr-factors", "0.3", "0.1", "0.03",
        "--eval-interval", str(stage.eval_interval),
        "--periodic-stochastic-eval-episodes",
        str(stage.periodic_evaluation_episodes),
        "--eval-episodes", str(stage.final_evaluation_episodes),
        "--checkpoint-every", str(stage.checkpoint_every),
        "--print-every", str(stage.print_every),
        "--seed", str(stage.seed),
        "--selection-evaluation-seed", str(stage.selection_evaluation_seed),
        "--final-evaluation-seed", str(stage.final_evaluation_seed),
        "--no-random-search",
        "--output-dir", str(destination),
        *COMMON_REWARD_ARGUMENTS,
    ]
    if task.model_directory is not None:
        command.extend([
            "--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve()),
            "--no-rigid-diagnostics",
        ])
    if stage.name == "full":
        command.extend(FINAL_FIXED_RUNTIME_ARGUMENTS)
    return command


def read_csv_count(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_artifact(
    destination: Path, task: Task, profile: Profile, stage: Stage,
) -> dict:
    errors: list[str] = []
    required = (
        "run_config.json", "summary.json", "training_history.csv",
        "update_history.csv", "deterministic_evaluation_history.csv",
        "stochastic_evaluation_history.csv", "evaluation_summary.csv",
        "best_policy_evaluation_summary.csv", "policy.pt", "best_policy.pt",
        "best_policy_metrics.json", "checkpoint.pt", "learning_curves.png",
        "ori5_milestone_learning_curve.png", "near_miss_learning_curve.png",
    )
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        errors.append("missing files: " + ", ".join(missing))
        return {"valid": False, "errors": errors, "path": str(destination)}
    try:
        with open(destination / "run_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        with open(destination / "summary.json", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "errors": [f"unreadable JSON: {error}"],
            "path": str(destination),
        }
    for key, expected in expected_config(task, profile, stage).items():
        if key not in config:
            errors.append(f"run_config missing {key}")
        elif not values_equal(config[key], expected):
            errors.append(
                f"run_config {key} mismatch: {config[key]!r} != {expected!r}"
            )
    configured_output = config.get("output_directory")
    if configured_output is None:
        errors.append("run_config missing output_directory")
    elif Path(configured_output).resolve() != destination.resolve():
        errors.append(
            "run_config output_directory mismatch: "
            f"{configured_output!r} != {str(destination.resolve())!r}"
        )
    if task.model_directory is not None:
        adapter = config.get("three_dimensional_adapter", {})
        expected_model = Path(task.model_directory).name
        if adapter.get("adapter") != "ori_codex5_3d":
            errors.append("missing ori_codex5_3d adapter metadata")
        if adapter.get("model_name") != expected_model:
            errors.append(
                f"3-D model mismatch: {adapter.get('model_name')} != {expected_model}"
            )
        expected_model_hashes = task_model_file_hashes(task)
        if adapter.get("model_file_sha256") != expected_model_hashes:
            errors.append(
                "3-D model-file hash mismatch: "
                f"{adapter.get('model_file_sha256')} != {expected_model_hashes}"
            )
        if adapter.get("rigid_motion_diagnostics") is not False:
            errors.append("rigid-motion diagnostics must be disabled")
        if adapter.get("path_integrated_spread_cost") is not False:
            errors.append("path-integrated spread cost must be disabled")
    try:
        history_count = read_csv_count(destination / "training_history.csv")
    except OSError as error:
        errors.append(f"cannot read training history: {error}")
        history_count = -1
    if history_count != stage.episodes:
        errors.append(
            f"training_history rows {history_count} != {stage.episodes}"
        )
    expected_periodic_rows = 1 + stage.episodes // stage.eval_interval
    if stage.episodes % stage.eval_interval == 0 and stage.eval_interval == 1:
        expected_periodic_rows = stage.episodes
    csv_row_expectations = {
        "update_history.csv": math.ceil(stage.episodes / profile.batch_size),
        "deterministic_evaluation_history.csv": expected_periodic_rows,
        "stochastic_evaluation_history.csv": expected_periodic_rows,
        "evaluation_summary.csv": stage.final_evaluation_episodes,
        "best_policy_evaluation_summary.csv": stage.final_evaluation_episodes,
    }
    for filename, expected_rows in csv_row_expectations.items():
        try:
            actual_rows = read_csv_count(destination / filename)
        except OSError as error:
            errors.append(f"cannot read {filename}: {error}")
            continue
        if actual_rows != expected_rows:
            errors.append(
                f"{filename} rows {actual_rows} != {expected_rows}"
            )
    if int(summary.get("training_episodes", -1)) != stage.episodes:
        errors.append("summary training_episodes mismatch")
    if summary.get("reward_design") != REWARD_DESIGN:
        errors.append("summary reward design mismatch")
    if not values_equal(
        summary.get("ori5_reward_parameters"), COMMON_REWARD_PARAMETERS
    ):
        errors.append("summary reward parameters mismatch")
    if int(summary.get("ori5_nonfinite_reward_component_count", -1)) != 0:
        errors.append("non-finite reward component detected")
    for key in (
        "maximum_abs_ori5_reward_decomposition_error",
        "maximum_abs_ori5_step_reward_decomposition_error",
    ):
        try:
            if abs(float(summary[key])) > 1.0e-8:
                errors.append(f"{key} exceeds 1e-8")
        except (KeyError, TypeError, ValueError):
            errors.append(f"summary missing finite {key}")
    return {
        "valid": not errors,
        "errors": errors,
        "path": str(destination),
        "training_episodes": summary.get("training_episodes"),
        "training_successes": summary.get("training_successes"),
        "first_success_episode": summary.get("training_first_success_episode"),
        "first_success_mechanical_steps": summary.get(
            "training_first_success_mechanical_steps"
        ),
    }


def config_matches_job(
    candidate: Path, task: Task, profile: Profile, stage: Stage,
) -> tuple[bool, list[str]]:
    """Match a live/staged output using immutable effective configuration."""

    config_path = candidate / "run_config.json"
    if not config_path.is_file():
        return False, ["missing run_config.json"]
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return False, [f"unreadable run_config.json: {error}"]
    errors = []
    for key, expected in expected_config(task, profile, stage).items():
        if key not in config:
            errors.append(f"run_config missing {key}")
        elif not values_equal(config[key], expected):
            errors.append(f"run_config {key} mismatch")
    configured_output = config.get("output_directory")
    if (
        configured_output is None
        or Path(configured_output).resolve() != candidate.resolve()
    ):
        errors.append("run_config effective output_directory mismatch")
    if task.model_directory is not None:
        adapter = config.get("three_dimensional_adapter", {})
        if adapter.get("adapter") != "ori_codex5_3d":
            errors.append("missing ori_codex5_3d adapter metadata")
        if adapter.get("model_name") != Path(task.model_directory).name:
            errors.append("3-D model name mismatch")
        if adapter.get("model_file_sha256") != task_model_file_hashes(task):
            errors.append("3-D model-file hash mismatch")
    return not errors, errors


def effective_output_candidates(requested: Path) -> list[Path]:
    """Include timestamp siblings created by resolve_output_directory()."""

    candidates = [requested]
    if requested.parent.is_dir():
        candidates.extend(sorted(
            path for path in requested.parent.glob(f"{requested.name}_*")
            if path.is_dir()
        ))
    return candidates


def discover_effective_output(
    requested: Path, task: Task, profile: Profile, stage: Stage,
    *, status: dict | None = None, require_complete: bool = False,
) -> tuple[Path | None, dict | None]:
    """Adopt exactly one config-matching canonical or timestamp output."""

    matches = []
    started_timestamp = None
    if status and status.get("started_utc"):
        try:
            started_timestamp = datetime.fromisoformat(
                str(status["started_utc"])
            ).timestamp()
        except ValueError:
            started_timestamp = None
    for candidate in effective_output_candidates(requested):
        config_path = candidate / "run_config.json"
        matches_job, _ = config_matches_job(candidate, task, profile, stage)
        if not matches_job:
            continue
        if (
            started_timestamp is not None
            and config_path.stat().st_mtime < started_timestamp - 5.0
        ):
            continue
        validation = None
        if require_complete:
            validation = validate_artifact(candidate, task, profile, stage)
            if not validation["valid"]:
                continue
        matches.append((candidate, validation))
    if status and status.get("effective_output_directory"):
        recorded = Path(status["effective_output_directory"]).resolve()
        recorded_matches = [
            item for item in matches if item[0].resolve() == recorded
        ]
        if len(recorded_matches) == 1:
            return recorded_matches[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None, None
    return None, {
        "valid": False,
        "errors": [
            "ambiguous effective outputs: "
            + ", ".join(str(item[0]) for item in matches)
        ],
    }


def next_retry_directory(job_directory: Path) -> Path:
    retry_root = job_directory / "_attempts"
    existing = []
    if retry_root.is_dir():
        for path in retry_root.glob("attempt_*" ):
            try:
                existing.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
    number = max(existing, default=0) + 1
    return ensure_within_experiment_root(
        retry_root / f"attempt_{number:03d}" / "output"
    )


def process_is_alive(pid: int) -> bool:
    """Return whether a recorded child PID is still active."""

    if int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes

        query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def active_python_processes() -> list[dict]:
    """Read Python command lines once, including externally launched jobs."""

    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return []
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return []
        rows = value if isinstance(value, list) else [value]
        return [{
            "pid": int(row.get("ProcessId", -1)),
            "command_line": str(row.get("CommandLine") or ""),
        } for row in rows]
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True,
        timeout=10, check=False,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and "python" in fields[1].lower():
            rows.append({"pid": int(fields[0]), "command_line": fields[1]})
    return rows


def external_process_for_destination(
    destination: Path, processes: list[dict],
) -> dict | None:
    normalised_absolute = str(destination.resolve()).lower().replace("\\", "/")
    try:
        normalised_relative = str(
            destination.resolve().relative_to(SCRIPT_DIRECTORY)
        ).lower().replace("\\", "/")
    except ValueError:
        normalised_relative = ""
    for row in processes:
        command_line = row["command_line"].lower().replace("\\", "/")
        if (
            normalised_absolute in command_line
            or (normalised_relative and normalised_relative in command_line)
        ):
            return row
    return None


def source_hashes() -> dict[str, str]:
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in SOURCE_FILES
    }


def task_model_file_hashes(task: Task) -> dict[str, str]:
    data_directory = (
        (SCRIPT_DIRECTORY / task.model_directory).resolve()
        if task.model_directory is not None else SCRIPT_DIRECTORY
    )
    return {
        "node": file_sha256(data_directory / "node_initial.dat"),
        "member": file_sha256(data_directory / "member.dat"),
        "fix": file_sha256(data_directory / "fix.dat"),
    }


def job_record(task: Task, profile: Profile, stage: Stage) -> dict:
    destination = output_directory(stage, task, profile)
    identity = {
        "task": asdict(task),
        "profile": asdict(profile),
        "stage": asdict(stage),
        "reward_parameters": COMMON_REWARD_PARAMETERS,
        "source_sha256": source_hashes(),
    }
    if stage.name == "full":
        identity["model_file_sha256"] = task_model_file_hashes(task)
        identity["fixed_runtime_parameters"] = FINAL_FIXED_RUNTIME_PARAMETERS
    return {
        "job_id": f"{stage.name}:{task.name}:{profile.name}:seed{stage.seed}",
        "fingerprint": canonical_digest(identity),
        **identity,
        "output_directory": str(destination),
        "command": build_command(task, profile, stage, destination),
    }


def selected_profiles_from_file(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    mapping = value.get("selected_profiles", value)
    result = {}
    for task_name in TASKS:
        profile_name = mapping.get(task_name)
        if profile_name not in PROFILES:
            raise ValueError(
                f"Selection file must map {task_name} to one of {tuple(PROFILES)}"
            )
        result[task_name] = profile_name
    return result


def stage_artifact_candidates(base: Path) -> list[Path]:
    """Return requested canonical and isolated retry output names."""

    candidates = [base]
    retry_root = base / "_attempts"
    if retry_root.is_dir():
        candidates.extend(sorted(retry_root.glob("attempt_*/output")))
    return candidates


def find_valid_stage_artifact(
    task: Task, profile: Profile, stage: Stage,
) -> tuple[Path | None, dict | None]:
    valid_matches = []
    last_validation = None
    for requested in reversed(stage_artifact_candidates(
        output_directory(stage, task, profile)
    )):
        candidate, validation = discover_effective_output(
            requested, task, profile, stage, require_complete=True
        )
        if candidate is not None:
            valid_matches.append((candidate, validation))
        elif validation is not None:
            last_validation = validation
    if len(valid_matches) == 1:
        return valid_matches[0]
    if len(valid_matches) > 1:
        return None, {
            "valid": False,
            "errors": [
                "ambiguous valid attempts: "
                + ", ".join(str(item[0]) for item in valid_matches)
            ],
        }
    return None, last_validation


def validated_stage2_selection(path: Path) -> dict[str, str]:
    """Reject a full run unless Stage 2 and its ranking are complete."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("source_stage") != "stage2":
        raise ValueError(
            "Full stage requires selected_profiles.json with "
            "source_stage='stage2'. Run the complete Stage-2 analyzer first."
        )
    manifest_path, manifest = current_stage_manifest("stage2")
    recorded_manifest = payload.get("stage2_manifest", {})
    if not (
        recorded_manifest.get("manifest_fingerprint")
        == manifest.get("manifest_fingerprint")
        and Path(recorded_manifest.get("path", "")).resolve()
        == manifest_path.resolve()
        and recorded_manifest.get("source_sha256") == source_hashes()
        and manifest.get("source_sha256") == source_hashes()
    ):
        raise ValueError(
            "Stage-2 selection provenance does not match the unique current "
            "source-authority manifest."
        )
    current_model_hashes = {
        task_name: task_model_file_hashes(task)
        for task_name, task in TASKS.items()
    }
    if payload.get("model_file_sha256_by_task") != current_model_hashes:
        raise ValueError(
            "Stage-2 selection model-file hashes are stale or incomplete."
        )
    if current_model_hashes["water1"] != current_model_hashes["watersqrt2"]:
        raise ValueError("Waterbomb D=1 and D=sqrt(2) model hashes differ.")
    mapping = selected_profiles_from_file(path)
    for group_name, task_names in SELECTION_GROUPS.items():
        selected = {mapping[task_name] for task_name in task_names}
        if len(selected) != 1:
            raise ValueError(
                f"Stage-2 selection must use one common profile for "
                f"{group_name}: {selected}"
            )
        winner = next(iter(selected))
        if winner not in STAGE2_GROUP_PROFILE_CANDIDATES[group_name]:
            raise ValueError(
                f"Selected {group_name} profile {winner} is outside the "
                "Stage-2 candidate matrix."
            )

    analysis_directory = path.parent
    report_path = analysis_directory / "stage2_analysis_report.json"
    ranking_path = analysis_directory / "stage2_group_ranking.csv"
    if not report_path.is_file() or not ranking_path.is_file():
        raise ValueError(
            "Full stage requires Stage-2 analysis report and group ranking."
        )
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected_count = sum(len(names) for names in STAGE2_MATRIX.values())
    if not (
        report.get("stage") == "stage2"
        and report.get("complete") is True
        and int(report.get("expected_artifacts", -1)) == expected_count
        and int(report.get("valid_complete_artifacts", -1)) == expected_count
        and report.get("reward_invariant") is True
        and report.get("random_baselines_compatible") is True
        and report.get("png_artifacts_valid") is True
        and report.get("selected_profiles") == mapping
    ):
        raise ValueError(
            "Stage-2 analysis report is incomplete, invalid, or stale "
            "relative to selected_profiles.json."
        )

    with open(ranking_path, newline="", encoding="utf-8") as handle:
        ranking_rows = list(csv.DictReader(handle))
    rank_one = {
        row["group"]: row["profile"]
        for row in ranking_rows if str(row.get("rank", "")) == "1"
    }
    if set(rank_one) != set(SELECTION_GROUPS):
        raise ValueError("Stage-2 ranking does not have exactly one winner per group.")
    for group_name, task_names in SELECTION_GROUPS.items():
        if any(mapping[task_name] != rank_one[group_name] for task_name in task_names):
            raise ValueError(
                f"Selected profile does not match Stage-2 rank 1 for {group_name}."
            )

    stage = STAGES["stage2"]
    failures = []
    validated_count = 0
    for task_name, profile_names in STAGE2_MATRIX.items():
        for profile_name in profile_names:
            artifact, validation = find_valid_stage_artifact(
                TASKS[task_name], PROFILES[profile_name], stage
            )
            if artifact is None:
                failures.append(
                    f"{task_name}/{profile_name}: "
                    + " | ".join((validation or {}).get(
                        "errors", ["missing artifact"]
                    ))
                )
            else:
                validated_count += 1
    if validated_count != expected_count:
        raise ValueError(
            "Full stage blocked: Stage-2 artifacts failed live validation: "
            + "; ".join(failures)
        )
    return mapping


def build_jobs(args) -> list[dict]:
    stage = STAGES[args.stage]
    selected_tasks = [TASKS[name] for name in args.tasks]
    if stage.name == "stage1":
        invalid = set(args.profiles) - set(STAGE1_PROFILE_NAMES)
        if invalid:
            raise ValueError(
                f"Stage 1 permits only profiles {STAGE1_PROFILE_NAMES}: "
                f"{sorted(invalid)}"
            )
        selected_profiles = [PROFILES[name] for name in args.profiles]
        return [
            job_record(task, profile, stage)
            for task in selected_tasks for profile in selected_profiles
        ]
    if stage.name == "smoke":
        return [
            job_record(task, PROFILES["A"], stage_for_task(stage, task))
            for task in selected_tasks
        ]
    if stage.name == "water_rescue":
        return [
            job_record(TASKS[task_name], PROFILES[profile_name], stage)
            for task_name in args.tasks
            for profile_name in profile_names_for_stage_task(
                stage.name, task_name
            )
        ]
    if stage.name == "stage2":
        return [
            job_record(TASKS[task_name], PROFILES[profile_name], stage)
            for task_name in args.tasks
            for profile_name in profile_names_for_stage_task(
                stage.name, task_name
            )
        ]
    selection_path = Path(args.selection_json).resolve()
    mapping = validated_stage2_selection(selection_path)
    return [
        job_record(task, PROFILES[mapping[task.name]], stage)
        for task in selected_tasks
    ]


def write_manifest(stage: Stage, jobs: list[dict]) -> Path:
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": asdict(stage),
        "reward_design": REWARD_DESIGN,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "common_reward_parameters": COMMON_REWARD_PARAMETERS,
        "reward_is_not_a_tuning_dimension": True,
        "source_sha256": source_hashes(),
        "jobs": jobs,
    }
    if stage.name == "full":
        manifest["runner_recovery_semantics"] = (
            "valid completed artifacts are skipped; an incomplete run is "
            "preserved and any retry is a fresh restart in an isolated output"
        )
        manifest["exact_training_continuation_supported"] = False
        manifest["model_file_sha256_by_task"] = {
            task_name: task_model_file_hashes(TASKS[task_name])
            for task_name in TASKS
        }
        manifest["fixed_runtime_parameter_schema_version"] = (
            FINAL_RUNTIME_SCHEMA_VERSION
        )
        manifest["fixed_runtime_parameters"] = FINAL_FIXED_RUNTIME_PARAMETERS
    content_fingerprint = canonical_digest({
        key: value for key, value in manifest.items() if key != "created_utc"
    })
    manifest["manifest_fingerprint"] = content_fingerprint
    path = ensure_within_experiment_root(
        EXPERIMENT_ROOT / "manifests"
        / f"{stage.name}_seed{stage.seed}_{content_fingerprint[:12]}.json"
    )
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            old = json.load(handle)
        old_comparable = dict(old)
        old_comparable.pop("created_utc", None)
        new_comparable = dict(manifest)
        new_comparable.pop("created_utc", None)
        if canonical_digest(old_comparable) != canonical_digest(new_comparable):
            raise RuntimeError(f"Manifest collision at {path}")
    else:
        atomic_json(path, manifest)
    return path


def current_stage_manifest(stage_name: str) -> tuple[Path, dict]:
    """Find the unique intact manifest matching current stage job identities."""

    if stage_name != "stage2":
        raise ValueError("Current manifest lookup is implemented for Stage 2.")
    stage = STAGES[stage_name]
    expected_jobs = [
        job_record(TASKS[task_name], PROFILES[profile_name], stage)
        for task_name, profile_names in STAGE2_MATRIX.items()
        for profile_name in profile_names
    ]
    expected_fingerprints = {
        row["job_id"]: row["fingerprint"] for row in expected_jobs
    }
    matches = []
    for path in sorted((EXPERIMENT_ROOT / "manifests").glob(
        f"{stage_name}_seed{stage.seed}_*.json"
    )):
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        recorded_fingerprint = manifest.get("manifest_fingerprint")
        fingerprint_payload = dict(manifest)
        fingerprint_payload.pop("created_utc", None)
        fingerprint_payload.pop("manifest_fingerprint", None)
        if canonical_digest(fingerprint_payload) != recorded_fingerprint:
            continue
        recorded_jobs = {
            row.get("job_id"): row.get("fingerprint")
            for row in manifest.get("jobs", [])
        }
        if not (
            values_equal(manifest.get("stage"), asdict(stage))
            and values_equal(
                manifest.get("common_reward_parameters"),
                COMMON_REWARD_PARAMETERS,
            )
            and manifest.get("source_sha256") == source_hashes()
            and recorded_jobs == expected_fingerprints
        ):
            continue
        matches.append((path.resolve(), manifest))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one intact Stage-2 manifest matching current "
            f"source/job identities; found {len(matches)}."
        )
    return matches[0]


def command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def runner_log_directory(job: dict, destination: Path) -> Path:
    """Keep orchestration files outside the child's requested output."""

    stage = Stage(**job["stage"])
    task = Task(**job["task"])
    profile = Profile(**job["profile"])
    destination_tag = canonical_digest(str(destination.resolve()))[:12]
    return ensure_within_experiment_root(
        EXPERIMENT_ROOT / "_runner_logs" / stage.name / task.name
        / profile.name / f"seed{stage.seed}" / destination_tag
    )


def read_job_status(job: dict, destination: Path) -> tuple[dict | None, Path | None]:
    """Read new external logs first, then the legacy in-output status."""

    paths = (
        runner_log_directory(job, destination) / "status.json",
        destination / "_runner" / "status.json",
    )
    existing = [path for path in paths if path.is_file()]
    for path in sorted(existing, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle), path
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def record_effective_output_adoption(
    job: dict, requested: Path, effective: Path, validation: dict,
) -> Path:
    provenance = {
        "schema_version": 1,
        "recorded_utc": utc_now(),
        "job_id": job["job_id"],
        "job_fingerprint": job["fingerprint"],
        "requested_output_directory": str(requested.resolve()),
        "effective_output_directory": str(effective.resolve()),
        "run_config_sha256": file_sha256(effective / "run_config.json"),
        "source_sha256": job["source_sha256"],
        "artifact_validation": validation,
    }
    if job["stage"]["name"] == "stage2":
        manifest_path, manifest = current_stage_manifest("stage2")
        provenance.update({
            "manifest_path": str(manifest_path),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        })
    path = runner_log_directory(job, requested) / "adoption.json"
    atomic_json(path, provenance)
    return path


def classify_job(
    job: dict, rerun_incomplete: bool, processes: list[dict] | None = None,
) -> tuple[str, Path, dict | None]:
    task = Task(**job["task"])
    profile = Profile(**job["profile"])
    stage = Stage(**job["stage"])
    destination = Path(job["output_directory"])
    external = external_process_for_destination(destination, processes or [])
    if external is not None and process_is_alive(external["pid"]):
        return "external_active", destination, external
    status, _ = read_job_status(job, destination)
    effective, validation = discover_effective_output(
        destination, task, profile, stage, status=status,
        require_complete=True,
    )
    if effective is not None:
        if effective.resolve() != destination.resolve():
            record_effective_output_adoption(
                job, destination, effective, validation
            )
        return "complete", effective, validation
    if validation is not None and not rerun_incomplete:
        return "invalid", destination, validation
    if status is not None:
        try:
            if (
                status.get("status") in ("launching", "running")
                and process_is_alive(int(status.get("pid", -1)))
            ):
                return "running", destination, status
        except (ValueError, TypeError):
            pass
    effective_incomplete, ambiguity = discover_effective_output(
        destination, task, profile, stage, status=status,
        require_complete=False,
    )
    if ambiguity is not None and not rerun_incomplete:
        return "invalid", destination, ambiguity
    if effective_incomplete is not None:
        if not rerun_incomplete:
            return "incomplete", effective_incomplete, None
        return "pending_retry", next_retry_directory(destination), None
    if destination.exists() and any(destination.iterdir()):
        if not rerun_incomplete:
            return "incomplete", destination, None
        return "pending_retry", next_retry_directory(destination), None
    return "pending", destination, None


def launch_job(job: dict, destination: Path) -> dict:
    ensure_within_experiment_root(destination)
    # Do not create destination: the training CLI interprets an existing
    # directory as a collision and redirects to a timestamp sibling.
    runner_directory = runner_log_directory(job, destination)
    runner_directory.mkdir(parents=True, exist_ok=True)
    task = Task(**job["task"])
    profile = Profile(**job["profile"])
    stage = Stage(**job["stage"])
    command = build_command(task, profile, stage, destination)
    attempt = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_fingerprint": job["fingerprint"],
        "status": "launching",
        "started_utc": utc_now(),
        "command": command,
        "requested_output_directory": str(destination),
        "effective_output_directory": None,
    }
    status_path = runner_directory / "status.json"
    atomic_json(status_path, attempt)
    stdout_handle = open(runner_directory / "stdout.log", "a", encoding="utf-8")
    stderr_handle = open(runner_directory / "stderr.log", "a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(stage.seed),
    })
    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIRECTORY,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=environment,
        creationflags=creationflags,
    )
    attempt.update({"status": "running", "pid": process.pid})
    atomic_json(status_path, attempt)
    return {
        "job": job,
        "task": task,
        "profile": profile,
        "stage": stage,
        "destination": destination,
        "runner_directory": runner_directory,
        "status_path": status_path,
        "attempt": attempt,
        "process": process,
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
        "start_monotonic": time.monotonic(),
    }


def finish_process(active: dict) -> dict:
    process = active["process"]
    return_code = int(process.returncode)
    active["stdout_handle"].close()
    active["stderr_handle"].close()
    effective, validation = discover_effective_output(
        active["destination"], active["task"], active["profile"],
        active["stage"], status=active["attempt"],
        require_complete=True,
    ) if return_code == 0 else (None, None)
    successful = return_code == 0 and validation is not None and validation["valid"]
    status = dict(active["attempt"])
    status.update({
        "status": "complete" if successful else "failed",
        "finished_utc": utc_now(),
        "return_code": return_code,
        "runner_wall_seconds": time.monotonic() - active["start_monotonic"],
        "requested_output_directory": str(active["destination"]),
        "effective_output_directory": str(effective) if effective else None,
        "artifact_validation": validation,
    })
    atomic_json(active["status_path"], status)
    return status


def execute_jobs(pending: list[tuple[dict, Path]], max_parallel: int) -> int:
    queue = list(pending)
    active: list[dict] = []
    failed = 0
    last_report = 0.0
    while queue or active:
        while queue and len(active) < max_parallel:
            job, destination = queue.pop(0)
            print(f"LAUNCH {job['job_id']} -> {destination}", flush=True)
            active.append(launch_job(job, destination))
        for item in list(active):
            if item["process"].poll() is None:
                continue
            result = finish_process(item)
            active.remove(item)
            if result["status"] != "complete":
                failed += 1
            print(
                f"{result['status'].upper()} {result['job_id']} "
                f"rc={result['return_code']} "
                f"wall={result['runner_wall_seconds']:.1f}s",
                flush=True,
            )
        now = time.monotonic()
        if active and now - last_report >= 30.0:
            labels = ", ".join(
                f"{item['job']['job_id']}({now-item['start_monotonic']:.0f}s)"
                for item in active
            )
            print(f"RUNNING {labels}", flush=True)
            last_report = now
        if active:
            time.sleep(1.0)
    return failed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="stage1")
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS),
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES),
        default=list(STAGE1_PROFILE_NAMES),
        help="Stage-1 profiles. Ignored by smoke/full.",
    )
    parser.add_argument(
        "--selection-json",
        default=str(EXPERIMENT_ROOT / "analysis" / "selected_profiles.json"),
        help="Task-to-profile mapping required by the full stage.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually launch pending jobs. Default is manifest-only dry run.",
    )
    parser.add_argument(
        "--rerun-incomplete", action="store_true",
        help="Preserve incomplete output and start a numbered retry attempt.",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 1 <= int(args.max_parallel) <= 12:
        raise ValueError("--max-parallel must be in [1, 12] on this workstation.")
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest_path = write_manifest(STAGES[args.stage], jobs)
    print(f"Manifest: {manifest_path}")
    processes = active_python_processes()
    counts: dict[str, int] = {}
    pending: list[tuple[dict, Path]] = []
    invalid_messages = []
    for job in jobs:
        state, destination, validation = classify_job(
            job, args.rerun_incomplete, processes
        )
        counts[state] = counts.get(state, 0) + 1
        print(f"{state.upper():13s} {job['job_id']}")
        if state.startswith("pending"):
            task = Task(**job["task"])
            profile = Profile(**job["profile"])
            stage = Stage(**job["stage"])
            command = build_command(task, profile, stage, destination)
            print("  " + command_string(command))
            pending.append((job, destination))
        elif state in ("invalid", "incomplete"):
            invalid_messages.append((job["job_id"], state, validation))
    print("Status counts: " + json.dumps(counts, sort_keys=True))
    if invalid_messages:
        print(
            "Incomplete/invalid outputs were preserved. Re-run with "
            "--rerun-incomplete to create isolated retry attempts.",
            file=sys.stderr,
        )
        if not args.execute:
            return 2
    if not args.execute:
        print(f"DRY RUN: {len(pending)} process(es) would be launched.")
        return 0
    if not pending:
        print("No pending jobs.")
        return 0
    failed = execute_jobs(pending, int(args.max_parallel))
    print(f"Finished {len(pending)} job(s); failed={failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
