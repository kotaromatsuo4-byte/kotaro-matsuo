"""Safe staged orchestration for the ori_codex6 transfer experiment.

The runner intentionally separates four questions.

``rescue``
    Screen the selected ori6 state-potential mode on torus2 and water1 with
    three independent seeds.  Checkpoint 512/1024 decisions may stop only a
    complete task cohort and are recorded as derived orchestration metadata.
``regression``
    Compare the selected ori6 mode with a fresh, paired ori5 reference on the
    three tasks that ori5 solved (five_d2, five_d3, watersqrt2).
``full_primary``
    Run one 10,000-episode seed on all five tasks, but only after the rescue
    and regression analyzer has emitted a source- and artifact-bound
    ``selected_candidate.json``.
``full_replication``
    Add two 10,000-episode seeds only after the primary analyzer authorises it.

Dry-run is the default.  Every write stays below ``experiments_ori6``.  Child
stdout/stderr/status files are outside the requested training output, because
the trainer redirects to a timestamp sibling when its output directory already
exists.  Checkpoints provide completed-skip/fresh-restart semantics only; the
base trainer has no exact continuation loader.
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
import uuid


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_ROOT = (SCRIPT_DIRECTORY / "experiments_ori6").resolve()
ORI5_ANALYSIS = SCRIPT_DIRECTORY / "experiments_ori5" / "analysis"
RANDOM_COMPATIBILITY_PATH = ORI5_ANALYSIS / "random_baseline_compatibility.json"

ORI6_REWARD_DESIGN = "ori6_continuous_spread_state_potential_rollback"
ORI6_REWARD_SCHEMA_VERSION = "1.0.0"
ORI5_REWARD_DESIGN = "ori5_endpoint_potential_fixed_milestones"
ORI5_REWARD_SCHEMA_VERSION = "1.0.0"

SOURCE_FILES = (
    "ori_codex2.py",
    "ori_codex3.py",
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "ori_codex6.py",
    "ori_codex6_3d.py",
    "collision_3d.py",
)
REJECTED_CROSSING_ADDITIVE_SOURCE_FILES = (
    "ori_codex6_rejected_crossing.py",
    "ori_codex6_rejected_crossing_3d.py",
)
SPECTRAL_SUBSPACE_ADDITIVE_SOURCE_FILES = (
    "ori_codex6_spectral_subspace.py",
    "ori_codex6_spectral_subspace_3d.py",
)
REJECTED_CROSSING_ORCHESTRATION_SOURCE_FILES = (
    "run_ori_codex6_experiments.py",
    "analyze_ori_codex6_experiments.py",
)
MODEL_FILE_NAMES = ("node_initial.dat", "member.dat", "fix.dat")

P0_MODE = "state_potential"
P1_MODE = "flatness_gated_state_potential"
P0_CROSS2_MODE = "state_potential_crossing_penalty_2"
P0_CROSS2_FRONTIER12_MODE = (
    "state_potential_crossing_penalty_2_frontier_1p2"
)
P0_CROSS2_ENTROPY3_MODE = (
    "state_potential_crossing_penalty_2_entropy_3e3"
)
P0_CROSS2_WIDTH2_MODE = (
    "state_potential_crossing_penalty_2_max_step_width_0p2"
)
P0_CROSS2_REJECTED_CONTINUATION_MODE = (
    "state_potential_crossing_penalty_2_rejected_crossing_continuation"
)
P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE = (
    "state_potential_crossing_penalty_2_rejected_crossing_continuation_"
    "spectral_subspace_0p05"
)
CANDIDATE_MODES = (
    P0_MODE, P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
    P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
    P0_CROSS2_REJECTED_CONTINUATION_MODE,
    P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE, P1_MODE,
)
MODE_SLUGS = {
    P0_MODE: "P0",
    P0_CROSS2_MODE: "P0X2",
    P0_CROSS2_FRONTIER12_MODE: "P0X2B12",
    P0_CROSS2_ENTROPY3_MODE: "P0X2E3",
    P0_CROSS2_WIDTH2_MODE: "P0X2W2",
    P0_CROSS2_REJECTED_CONTINUATION_MODE: "P0X2RC",
    P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE: "P0X2RCS05",
    P1_MODE: "P1",
}
P0_CROSS2_OFFLINE_AUDIT = (
    EXPERIMENT_ROOT / "analysis" / "offline_P0X2_saved_path_ordering.json"
)
P0_RECLASSIFICATION_REPORT = (
    EXPERIMENT_ROOT / "analysis" / "rescue_P0_reclassification.json"
)
P0_CROSS2_FRONTIER12_OFFLINE_AUDIT = (
    EXPERIMENT_ROOT / "analysis"
    / "offline_P0X2B12_saved_path_ordering.json"
)
P0_CROSS2_REJECT_REPORT = (
    EXPERIMENT_ROOT / "analysis" / "rescue_P0X2_report.json"
)
P0_CROSS2_FRONTIER12_REJECT_REPORT = (
    EXPERIMENT_ROOT / "analysis"
    / "rescue_P0X2B12_checkpoint1024_reject_report.json"
)
P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT = (
    EXPERIMENT_ROOT / "analysis" / "preflight_P0X2E3_entropy_screen.json"
)
P0_CROSS2_ENTROPY3_REJECT_REPORT = (
    EXPERIMENT_ROOT / "analysis"
    / "entropy_diagnostic_P0X2E3_reject_report.json"
)
P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT = (
    EXPERIMENT_ROOT / "analysis"
    / "offline_P0X2W2_water1_width_feasibility.json"
)
P0_CROSS2_WIDTH2_REJECT_REPORT = (
    EXPERIMENT_ROOT / "analysis"
    / "width_diagnostic_P0X2W2_reject_report.json"
)
P0_CROSS2_WIDTH2_REJECT_REPORT_SHA256 = (
    "578bd1c448871917b7b2df212d1907426e29fffac2ed52cadf9d5b29774d38e0"
)
REJECTED_CROSSING_CANDIDATE_NAME = (
    "ori6_P0X2_rejected_crossing_continuation"
)
REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION = "1.0.0"
REJECTED_CROSSING_POLICY = "reject_candidate_and_continue"
SPECTRAL_SUBSPACE_CANDIDATE_NAME = "ori6_RC_spectral_subspace_potential"
SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION = "1.0.0"
SPECTRAL_SUBSPACE_WEIGHT = 0.05
SPECTRAL_SUBSPACE_RELATIVE_EPSILON = 1.0e-12
SPECTRAL_SUBSPACE_FORMULA_SHA256 = (
    "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6"
)
P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT = (
    EXPERIMENT_ROOT / "analysis"
    / "crossing_continuation_diagnostic_P0X2RC_reject_report.json"
)
P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT_SHA256 = (
    "f520049575f0eb54b582107688bd9cada51e38f57aa64a9a28b8a1a7127d0489"
)
SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY = (
    EXPERIMENT_ROOT / "analysis"
    / "spectral_subspace_offline_design_authority.json"
)
SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY_SHA256 = (
    "52b1388038b58e42f32fd1036dcf08a3db4833a45a6338e9d78e5c642f5a0856"
)
SPECTRAL_SUBSPACE_SCREEN_REPORT = (
    EXPERIMENT_ROOT / "analysis"
    / "spectral_subspace_diagnostic_P0X2RCS05_formal_report.json"
)
SPECTRAL_SUBSPACE_SCREEN_REPORT_SHA256 = (
    "fa68e83201dd2028141c12f90d200dcdb10b080c7a02ccf5ebf8bc5e3a02af30"
)
SPECTRAL_SUBSPACE_SCREEN_REPORT_FINGERPRINT = (
    "fc30d7fb675d188b3d8ba69bb74f9f2369d1a4c63315d88b2bd1f87188de01fd"
)

# Baseline P0/P1 values.  Orchestration-only candidates override explicitly
# named leaves through the candidate helper functions below.
COMMON_ORI6_PARAMETERS = {
    "maximum_bonus": 0.9,
    "flatness_gate_cap": 2.0,
    "initial_eligibility_factor": 2.0,
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
    "gamma": 1.0,
}

COMMON_RUNTIME = {
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
}

COMMON_ARGUMENTS = (
    "--ori5-flatness-clip", "2",
    "--ori5-spread-milestone-initial-factor", "2",
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
    planar_entrypoint: str
    spatial_entrypoint: str
    model_directory: str
    target_spread: float
    max_steps: int
    max_step_width: float
    flatness_subspace_dimension: int
    profile: str

    @property
    def is_three_dimensional(self) -> bool:
        return self.flatness_subspace_dimension == 2


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    entropy_coefficient: float


@dataclass(frozen=True)
class Protocol:
    name: str
    episodes: int
    seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    implementations: tuple[str, ...]
    eval_interval: int
    periodic_evaluation_episodes: int
    final_evaluation_episodes: int
    checkpoint_every: int
    print_every: int
    selection_evaluation_seed: int
    final_evaluation_seed: int


TASKS = {
    "five_d2": Task(
        "five_d2", "ori_codex6.py", "", "model/five_bar_linkage",
        2.0, 400, 0.1, 1, "B",
    ),
    "five_d3": Task(
        "five_d3", "ori_codex6.py", "", "model/five_bar_linkage",
        3.0, 400, 0.1, 1, "B",
    ),
    "torus2": Task(
        "torus2", "", "ori_codex6_3d.py", "model/4unit_torus",
        2.0, 200, 0.2, 2, "A",
    ),
    "water1": Task(
        "water1", "", "ori_codex6_3d.py", "model/waterbomb",
        1.0, 400, 0.3, 2, "B",
    ),
    "watersqrt2": Task(
        "watersqrt2", "", "ori_codex6_3d.py", "model/waterbomb",
        math.sqrt(2.0), 400, 0.3, 2, "B",
    ),
}

PROFILES = {
    "A": Profile("A", 8, 1.0e-3, 3.0e-4, 1.0e-3),
    "B": Profile("B", 8, 5.0e-4, 1.5e-4, 1.0e-3),
}

PROTOCOLS = {
    "smoke": Protocol(
        "smoke", 16, (3901,), tuple(TASKS), ("ori6",),
        8, 4, 8, 8, 4, 7300000, 7400000,
    ),
    "rescue": Protocol(
        "rescue", 2048, (3101, 3102, 3103), ("torus2", "water1"),
        ("ori6",), 512, 16, 64, 512, 128, 7310000, 7410000,
    ),
    "entropy_diagnostic": Protocol(
        "entropy_diagnostic", 1024, (3101, 3102, 3103), ("water1",),
        ("ori6",), 512, 16, 64, 512, 128, 7310000, 7410000,
    ),
    "width_diagnostic": Protocol(
        "width_diagnostic", 2048, (3101, 3102, 3103), ("water1",),
        ("ori6",), 512, 16, 64, 512, 128, 7310000, 7410000,
    ),
    "crossing_continuation_diagnostic": Protocol(
        "crossing_continuation_diagnostic", 2048,
        (3201, 3202, 3203), ("torus2", "water1"),
        ("ori6",), 512, 16, 64, 512, 128, 7320000, 7420000,
    ),
    "spectral_subspace_diagnostic": Protocol(
        "spectral_subspace_diagnostic", 2048,
        (3201, 3202, 3203), ("torus2", "water1"),
        ("ori6",), 512, 16, 64, 512, 128, 7320000, 7420000,
    ),
    "spectral_nonregression": Protocol(
        "spectral_nonregression", 2048,
        (3301, 3302, 3303), ("five_d2", "five_d3", "watersqrt2"),
        ("ori6", "ori5_reference"),
        512, 16, 64, 512, 128, 7330000, 7430000,
    ),
    "regression": Protocol(
        "regression", 2048, (3101, 3102, 3103),
        ("five_d2", "five_d3", "watersqrt2"),
        ("ori6", "ori5_reference"),
        512, 16, 64, 512, 128, 7310000, 7410000,
    ),
    "full_primary": Protocol(
        "full_primary", 10000, (4101,), tuple(TASKS), ("ori6",),
        500, 25, 100, 1024, 100, 8310000, 8410000,
    ),
    "full_replication": Protocol(
        "full_replication", 10000, (4102, 4103), tuple(TASKS), ("ori6",),
        500, 25, 100, 1024, 100, 8310000, 8410000,
    ),
}


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
    encoded = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the same-directory temporary name short.  The workspace path is
    # already close to the legacy Windows MAX_PATH limit; including the full
    # destination filename plus a 32-hex UUID made runner status temporaries
    # exceed 260 characters even though their parent directory existed.
    temporary = None
    handle = None
    for _ in range(16):
        candidate = path.with_name(f".tmp-{uuid.uuid4().hex[:16]}")
        try:
            handle = open(candidate, "x", encoding="utf-8")
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary is None or handle is None:
        raise FileExistsError(
            f"Could not reserve an atomic JSON temporary beside {path}"
        )
    try:
        with handle:
            json.dump(jsonable(value), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_below_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"ori6 experiment output must stay below {EXPERIMENT_ROOT}: {resolved}"
        ) from error
    return resolved


def source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (SCRIPT_DIRECTORY / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing ori6 source files: " + ", ".join(missing))
    return {name: file_sha256(SCRIPT_DIRECTORY / name) for name in SOURCE_FILES}


def rejected_crossing_additive_source_hashes() -> dict[str, str]:
    missing = [
        name for name in REJECTED_CROSSING_ADDITIVE_SOURCE_FILES
        if not (SCRIPT_DIRECTORY / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing rejected-crossing additive source files: "
            + ", ".join(missing)
        )
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in REJECTED_CROSSING_ADDITIVE_SOURCE_FILES
    }


def spectral_subspace_additive_source_hashes() -> dict[str, str]:
    missing = [
        name for name in SPECTRAL_SUBSPACE_ADDITIVE_SOURCE_FILES
        if not (SCRIPT_DIRECTORY / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing spectral-subspace additive source files: "
            + ", ".join(missing)
        )
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in SPECTRAL_SUBSPACE_ADDITIVE_SOURCE_FILES
    }


def spectral_subspace_runtime_identity() -> dict:
    """Read the additive implementation identity, including its formula."""

    import ori_codex6_spectral_subspace as spectral

    identity = {
        "candidate": spectral.CANDIDATE_NAME,
        "schema_version": spectral.CANDIDATE_SCHEMA_VERSION,
        "weight": float(spectral.DEFAULT_SPECTRAL_WEIGHT),
        "relative_epsilon": float(spectral.DEFAULT_RELATIVE_EPSILON),
        "formula": spectral.SPECTRAL_FORMULA,
        "formula_sha256": spectral.SPECTRAL_FORMULA_SHA256,
    }
    if identity != {
        "candidate": SPECTRAL_SUBSPACE_CANDIDATE_NAME,
        "schema_version": SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION,
        "weight": SPECTRAL_SUBSPACE_WEIGHT,
        "relative_epsilon": SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
        "formula": spectral.SPECTRAL_FORMULA,
        "formula_sha256": SPECTRAL_SUBSPACE_FORMULA_SHA256,
    }:
        raise RuntimeError("Spectral-subspace implementation identity changed")
    if canonical_digest(identity["formula"]) != identity["formula_sha256"]:
        raise RuntimeError("Spectral-subspace formula digest mismatch")
    return identity


def rejected_crossing_orchestration_source_hashes() -> dict[str, str]:
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in REJECTED_CROSSING_ORCHESTRATION_SOURCE_FILES
    }


def model_hashes(task: Task) -> dict[str, str]:
    directory = (SCRIPT_DIRECTORY / task.model_directory).resolve()
    result = {}
    for name in MODEL_FILE_NAMES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        key = {"node_initial.dat": "node", "member.dat": "member", "fix.dat": "fix"}[name]
        result[key] = file_sha256(path)
    return result


def values_equal(actual, expected, atol=1.0e-12) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and values_equal(actual[key], value, atol)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            values_equal(left, right, atol) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
        except (TypeError, ValueError):
            return False
    return actual == expected


def random_baseline_authority() -> dict:
    if not RANDOM_COMPATIBILITY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing ori5 random compatibility authority: {RANDOM_COMPATIBILITY_PATH}"
        )
    with open(RANDOM_COMPATIBILITY_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    if payload.get("all_compatible") is not True:
        errors.append("all_compatible is not true")
    for name, task in TASKS.items():
        row = payload.get("tasks", {}).get(name, {})
        if row.get("compatible") is not True:
            errors.append(f"{name}: incompatible")
            continue
        expected = {
            "max_steps": task.max_steps,
            "max_step_width": task.max_step_width,
            "min_step_width": COMMON_RUNTIME["min_step_width"],
            "projection_shrink": COMMON_RUNTIME["projection_shrink"],
            "max_correction_ratio": COMMON_RUNTIME["max_correction_ratio"],
            "max_actual_step_ratio": COMMON_RUNTIME["max_actual_step_ratio"],
            "flatness_subspace_dimension": task.flatness_subspace_dimension,
            "flatness_tolerance": COMMON_RUNTIME["flatness_tolerance"],
            "strain_tolerance": COMMON_RUNTIME["strain_tolerance"],
            "crossing_tolerance": COMMON_RUNTIME["crossing_tolerance"],
            "target_spread": task.target_spread,
            "spread_tolerance": COMMON_RUNTIME["spread_tolerance"],
            "terminate_wrong_flat": True,
        }
        signature = row.get("mechanics_signature_expected", {})
        if not values_equal(signature, expected):
            errors.append(f"{name}: mechanics signature mismatch")
        recorded_models = row.get("provenance", {}).get("current_model_sha256")
        if task.is_three_dimensional and not values_equal(recorded_models, model_hashes(task)):
            errors.append(f"{name}: current 3-D model hashes changed")
    if errors:
        raise RuntimeError("Random baseline reuse rejected: " + "; ".join(errors))
    return {
        "path": str(RANDOM_COMPATIBILITY_PATH.resolve()),
        "sha256": file_sha256(RANDOM_COMPATIBILITY_PATH),
        "all_compatible": True,
        "task_random_summaries": {
            name: payload["tasks"][name]["random_summary"] for name in TASKS
        },
    }


def rejected_crossing_random_baseline_disposition() -> dict:
    return {
        "status": "incompatible_with_changed_crossing_transition_semantics",
        "reason": (
            "existing uniform trajectories terminate on self-crossing, "
            "whereas P0X2RC rejects the candidate and continues"
        ),
        "existing_random_baseline_compatible": False,
        "new_random_baseline_required_if_adopted": True,
        "required_tasks": list(TASKS),
    }


def spectral_subspace_gate_specification() -> dict:
    """Immutable predicates for the bounded RC+spectral screen."""

    return {
        "cohort_seeds": [3201, 3202, 3203],
        "task_cohorts_are_independent_and_fixed": True,
        "constructive_proxy_by_task": {
            "torus2": (
                "valid endpoint flatness_ratio + spread_error^2 <= 0.5"
            ),
            "water1": (
                "valid endpoint flatness_ratio <= 0.5 and spread_error <= 0.1"
            ),
        },
        "spectral_peak_threshold": 1.0e-12,
        "spectral_delta_threshold": 1.0e-6,
        "checkpoint_512": {
            "phase": "pathology_and_integrity_only",
            "performance_predicate_count": 0,
            "all_three_seed_histories_valid_required": True,
            "parent_rejected_crossing_integrity_required": True,
            "spectral_reward_identity_and_integrity_required": True,
            "valid_cohort_always_continues": True,
        },
        "checkpoint_1024": {
            "constructive_prefix_seed_count_min": 2,
            "valid_F_le_0_1_e_le_0_05_or_strict_seed_count_min": 1,
            "positive_spectral_peak_seed_count_min": 2,
            "spectral_delta_gt_1e_6_seed_count_min": 2,
            "reward_spectral_and_transition_integrity_required": True,
            "parent_common_prefix_comparison_is_diagnostic_only": True,
            "missing_parent_1024_torus_does_not_invalidate": True,
        },
        "checkpoint_2048": {
            "strict_hit_seed_count_min": 1,
            "constructive_prefix_seed_count_min": 2,
            "reward_spectral_and_transition_integrity_required": True,
            "heldout_inspected_during_training_gate": False,
            "pass_action": "allow_all_jobs_to_finalize_for_heldout_analysis",
        },
        "post_finalization": {
            "complete_2048_with_heldout_seed_count": 3,
            "heldout_constructive_or_strict_seed_count_min": 2,
            "fresh_confirmation_seeds_required_if_screen_passes": True,
            "screen_pass_does_not_authorize_adoption_or_full_training": True,
        },
    }


def spectral_nonregression_gate_specification() -> dict:
    """Preregister the paired successful-task nonregression screen."""

    return {
        "candidate": "P0X2RCS05",
        "reference": "ori5_reference",
        "tasks": ["five_d2", "five_d3", "watersqrt2"],
        "paired_seeds": [3301, 3302, 3303],
        "episodes": 2048,
        "checkpoint_every": 512,
        "final_heldout_episodes": 64,
        "task_cohorts_are_independent_and_fixed": True,
        "jobs_per_task": 6,
        "total_jobs": 18,
        "checkpoint_512": {
            "phase": "catastrophic_regression_and_integrity_only",
            "candidate_seed_count": 3,
            "reference_seed_count": 3,
            "performance_noninferiority_tested": False,
            "stop_if_all_candidate_seeds_have_zero_success_and_"
            "rejected_action_rate_ge_0_95": True,
            "reward_transition_and_spectral_integrity_required": True,
            "reference_reward_integrity_required": True,
        },
        "post_finalization_2048": {
            "all_18_jobs_complete_with_heldout_required": True,
            "paired_one_sided_success_difference_lower_bound_strictly_gt": -0.10,
            "paired_one_sided_crossing_difference_upper_bound_strictly_lt": 0.10,
            "paired_bootstrap_confidence_level": 0.95,
            "tail512_success_difference_min": -0.15,
            "successful_spread_error_max": 0.05,
            "successful_spread_error_worsening_max": 0.01,
            "discovery_mechanics_candidate_over_reference_max": 1.5,
            "discovery_mechanics_tasks": ["five_d2", "watersqrt2"],
            "five_d3_speed_claim_excluded_if_hit_precedes_first_update": True,
            "reward_transition_and_spectral_integrity_required": True,
            "reference_reward_integrity_required": True,
        },
        "promotion": {
            "all_three_task_gates_must_pass": True,
            "regression_pass_is_required_before_full_manifest": True,
            "fresh_random_baseline_still_required": True,
        },
    }


def crossing_continuation_gate_specification() -> dict:
    """Immutable decision predicates for the bounded P0X2RC screen."""

    return {
        "cohort_seeds": [3201, 3202, 3203],
        "task_cohorts_are_independent_and_fixed": True,
        "tail_window_episodes": 256,
        "constructive_proxy_by_task": {
            "torus2": (
                "valid endpoint flatness_ratio + spread_error^2 <= 0.5"
            ),
            "water1": (
                "valid endpoint flatness_ratio <= 0.5 and spread_error <= 0.1"
            ),
        },
        "checkpoint_512": {
            "all_seed_pooled_rejected_actions_over_steps_max": 0.25,
            "escape_after_reject_positive_seed_count_min": 2,
            "torus_no_reject_substitute": (
                "zero rejected actions AND positive frontier episode AND "
                "minimum terminal F+e^2 <= 0.5"
            ),
            "reward_and_transition_integrity_required": True,
        },
        "checkpoint_1024": {
            "tail_rejected_actions_over_steps_max": 0.10,
            "tail_escape_count_over_streak_count_min": 0.25,
            "stable_seed_count_min": 2,
            "torus_no_reject_substitute_same_as_checkpoint_512": True,
            "constructive_prefix_seed_count_min": 2,
            "valid_F_le_0_1_e_le_0_05_or_strict_seed_count_min": 1,
            "reward_and_transition_integrity_required": True,
        },
        "checkpoint_2048": {
            "strict_hit_seed_count_min": 1,
            "constructive_prefix_seed_count_min": 2,
            "reward_and_transition_integrity_required": True,
            "heldout_inspected_during_training_gate": False,
            "pass_action": "allow_all_jobs_to_finalize_for_heldout_analysis",
        },
    }


def core_frontier_mode(mode: str) -> str:
    """Map orchestration-only variants onto a core-supported frontier mode."""

    if mode in (
        P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
        P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        return P0_MODE
    if mode not in (P0_MODE, P1_MODE):
        raise ValueError(f"Unknown ori6 candidate mode: {mode}")
    return mode


def candidate_crossing_penalty(mode: str) -> float:
    return (
        2.0
        if mode in (
            P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
            P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        )
        else 1.25
    )


def candidate_frontier_bonus(mode: str) -> float:
    return 1.2 if mode == P0_CROSS2_FRONTIER12_MODE else 0.9


def candidate_entropy_coefficient(mode: str, profile: Profile) -> float:
    """Return the effective optimizer entropy without changing the task profile."""

    return 3.0e-3 if mode == P0_CROSS2_ENTROPY3_MODE else profile.entropy_coefficient


def candidate_max_step_width(mode: str, task: Task) -> float:
    """Return the sole mechanics change in the water1-only W2 diagnostic."""

    if mode == P0_CROSS2_WIDTH2_MODE:
        if task.name != "water1":
            raise ValueError("P0X2W2 changes max_step_width only for water1")
        return 0.2
    return task.max_step_width


def expected_mechanics(mode: str, task: Task) -> dict:
    return {
        "max_steps": task.max_steps,
        "max_step_width": candidate_max_step_width(mode, task),
        "min_step_width": COMMON_RUNTIME["min_step_width"],
        "projection_shrink": COMMON_RUNTIME["projection_shrink"],
        "max_correction_ratio": COMMON_RUNTIME["max_correction_ratio"],
        "max_actual_step_ratio": COMMON_RUNTIME["max_actual_step_ratio"],
        "strain_tolerance": COMMON_RUNTIME["strain_tolerance"],
        "crossing_tolerance": COMMON_RUNTIME["crossing_tolerance"],
        "flatness_subspace_dimension": task.flatness_subspace_dimension,
    }


def candidate_mechanics_sha256(mode: str, task: Task) -> str:
    return canonical_digest(expected_mechanics(mode, task))


def expected_optimizer(mode: str, profile: Profile) -> dict:
    """Hashable optimizer identity, kept separate from the reward identity."""

    return {
        "batch_size": profile.batch_size,
        "actor_learning_rate": profile.actor_learning_rate,
        "critic_learning_rate": profile.critic_learning_rate,
        "entropy_coefficient": candidate_entropy_coefficient(mode, profile),
        "actor_lr_milestones": [2000, 4000, 7000],
        "actor_lr_factors": [0.3, 0.1, 0.03],
    }


def candidate_optimizer_sha256(mode: str, profile: Profile) -> str:
    return canonical_digest(expected_optimizer(mode, profile))


def expected_crossing_transition(mode: str) -> dict:
    if mode not in (
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        return {
            "policy": "terminate_on_self_crossing",
            "candidate_committed": True,
            "step_count_increments": True,
            "episode_continues_before_horizon": False,
        }
    return {
        "policy": REJECTED_CROSSING_POLICY,
        "candidate_committed": False,
        "step_count_increments": True,
        "ordinary_rejected_reward_formula": (
            "-crossing_penalty + quality_weight * q(last_safe)"
        ),
        "horizon_rejected_reward_formula": (
            "-crossing_penalty + quality_weight * q(last_safe) "
            "- timeout_penalty"
        ),
        "horizon_priority": (
            "correction_failed; else reject crossing candidate; if "
            "step_count >= max_steps classify terminal as timeout; crossed "
            "candidate cannot be success or wrong_flat"
        ),
        "rejected_candidate_endpoint_reward": 0.0,
        "rejected_candidate_spread_frontier_reward": 0.0,
        "rejected_candidate_new_milestone_reward": 0.0,
        "past_frontier_and_joint_ledgers_retained": True,
        "escape_definition": (
            "one escape per maximal consecutive rejected-crossing streak "
            "when its immediately following non-rejected action is a valid "
            "accepted transition; unresolved terminal counts as zero"
        ),
    }


def candidate_transition_semantics_sha256(mode: str) -> str:
    return canonical_digest(expected_crossing_transition(mode))


def expected_ori6_reward(mode: str) -> dict:
    frontier_mode = core_frontier_mode(mode)
    result = {
        "schema_version": ORI6_REWARD_SCHEMA_VERSION,
        "gamma": 1.0,
        "endpoint_potential": {
            "flatness_clip": 2.0,
            "flatness_weight": 1.0,
            "squared_spread_error_weight": 1.0,
            "path_integrated_cost_weight": 0.0,
        },
        "spread_frontier": {
            "mode": frontier_mode,
            "maximum_bonus": candidate_frontier_bonus(mode),
            "initial_eligibility_factor": 2.0,
            "flatness_gate_cap": 2.0,
            "hard_spread_milestone_bonus_effective": 0.0,
            "is_running_best": False,
            "is_markov_in_existing_coordinate_observation": True,
            "actor_critic_frontier_scalar_input_added": False,
        },
        "joint_flatness_thresholds": [0.5, 0.1, 0.01],
        "joint_milestone_bonuses": [0.2, 0.3, 0.5],
        "crossing_penalty": candidate_crossing_penalty(mode),
        "wrong_flat_penalty": 2.0,
        "timeout_penalty": 1.0,
        "success_bonus": 10.0,
        "quality_weight": 1.0,
        "success_speed_bonus_weight": 0.5,
        "near_miss_flatness_far": 1.0,
        "near_miss_spread_error_far": 1.0,
        "rollback": {
            "self_crossing": ["spread_frontier", "joint_milestones"],
            "wrong_flat": ["flatness_progress", "spread_frontier", "joint_milestones"],
            "correction_failed": ["spread_frontier", "joint_milestones"],
            "no_modes": ["spread_frontier"],
            "timeout": [],
            "success": [],
        },
        "collision_quality_reference": "last_valid_before_crossing",
    }
    if mode in (
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        result.update({
            "rollback": {
                **result["rollback"],
                "self_crossing": [],
            },
            "collision_quality_reference": (
                "last_safe_before_rejected_crossing"
            ),
            "candidate": REJECTED_CROSSING_CANDIDATE_NAME,
            "candidate_schema_version": (
                REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            ),
            "crossing_transition": expected_crossing_transition(mode),
        })
    if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
        runtime = spectral_subspace_runtime_identity()
        result.update({
            "candidate": SPECTRAL_SUBSPACE_CANDIDATE_NAME,
            "candidate_schema_version": (
                SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
            ),
            "parent_candidate": REJECTED_CROSSING_CANDIDATE_NAME,
            "spectral_subspace": {
                "weight": SPECTRAL_SUBSPACE_WEIGHT,
                "relative_epsilon": SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
                "formula": runtime["formula"],
                "formula_sha256": SPECTRAL_SUBSPACE_FORMULA_SHA256,
                "current_state_markov": True,
                "reference_reset_exact": True,
                "accepted_valid_transitions_only": True,
                "weight_zero_parent_bit_exact": True,
            },
        })
    return result


def candidate_reward_sha256(mode: str) -> str:
    """Hash the effective reward, independently of optimizer/task metadata."""

    return canonical_digest({
        "reward_design": ORI6_REWARD_DESIGN,
        "reward_schema_version": ORI6_REWARD_SCHEMA_VERSION,
        "effective_parameters": expected_ori6_reward(mode),
    })


def expected_ori5_reward() -> dict:
    return {
        "schema_version": ORI5_REWARD_SCHEMA_VERSION,
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
    }


def output_directory(
    protocol: Protocol, mode: str, implementation: str, task: Task, seed: int,
) -> Path:
    mode_slug = MODE_SLUGS[mode]
    if protocol.name == "rescue":
        parts = ("screen", "rescue", mode_slug, task.name, task.profile, f"seed{seed}")
    elif protocol.name == "entropy_diagnostic":
        parts = (
            "screen", "entropy_diagnostic", mode_slug,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "width_diagnostic":
        parts = (
            "screen", "width_diagnostic", mode_slug,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "crossing_continuation_diagnostic":
        parts = (
            "screen", "crossing_continuation_diagnostic", mode_slug,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "spectral_subspace_diagnostic":
        parts = (
            "screen", "spectral_subspace_diagnostic", mode_slug,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "spectral_nonregression":
        parts = (
            "screen", "spectral_nonregression", mode_slug, implementation,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "regression":
        parts = (
            "screen", "regression", mode_slug, implementation,
            task.name, task.profile, f"seed{seed}",
        )
    elif protocol.name == "smoke":
        parts = ("smoke", mode_slug, task.name, task.profile, f"seed{seed}")
    else:
        phase = "primary" if protocol.name == "full_primary" else "replication"
        parts = ("full", phase, mode_slug, task.name, task.profile, f"seed{seed}")
    return ensure_below_root(EXPERIMENT_ROOT.joinpath(*parts))


def entrypoint(task: Task, implementation: str) -> str:
    if implementation == "ori6":
        return task.spatial_entrypoint if task.is_three_dimensional else task.planar_entrypoint
    if implementation != "ori5_reference":
        raise ValueError(f"Unknown implementation: {implementation}")
    return "ori_codex5_3d.py" if task.is_three_dimensional else "ori_codex5.py"


def candidate_entrypoint(task: Task, implementation: str, mode: str) -> str:
    if (
        implementation == "ori6"
        and mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        return (
            "ori_codex6_spectral_subspace_3d.py"
            if task.is_three_dimensional
            else "ori_codex6_spectral_subspace.py"
        )
    if (
        implementation == "ori6"
        and mode == P0_CROSS2_REJECTED_CONTINUATION_MODE
    ):
        return (
            "ori_codex6_rejected_crossing_3d.py"
            if task.is_three_dimensional
            else "ori_codex6_rejected_crossing.py"
        )
    return entrypoint(task, implementation)


def expected_config(
    task: Task, profile: Profile, protocol: Protocol, seed: int,
    implementation: str, mode: str,
) -> dict:
    result = {
        "episodes": protocol.episodes,
        "eval_episodes": protocol.final_evaluation_episodes,
        "eval_interval": protocol.eval_interval,
        "periodic_stochastic_eval_episodes": protocol.periodic_evaluation_episodes,
        "batch_size": profile.batch_size,
        "max_steps": task.max_steps,
        "max_step_width": candidate_max_step_width(mode, task),
        "flatness_subspace_dimension": task.flatness_subspace_dimension,
        "target_spread": task.target_spread,
        "actor_learning_rate": profile.actor_learning_rate,
        "critic_learning_rate": profile.critic_learning_rate,
        "entropy_coefficient": candidate_entropy_coefficient(mode, profile),
        "actor_lr_milestones": [2000, 4000, 7000],
        "actor_lr_factors": [0.3, 0.1, 0.03],
        "checkpoint_every": protocol.checkpoint_every,
        "seed": seed,
        "selection_evaluation_seed": protocol.selection_evaluation_seed,
        "final_evaluation_seed": protocol.final_evaluation_seed,
        "random_search": False,
        **COMMON_RUNTIME,
        "gamma": 1.0,
    }
    if implementation == "ori6":
        result.update({
            "reward_design": ORI6_REWARD_DESIGN,
            "reward_schema_version": ORI6_REWARD_SCHEMA_VERSION,
            "ori6_spread_frontier_mode": core_frontier_mode(mode),
            "ori6_spread_frontier_bonus": candidate_frontier_bonus(mode),
            "ori6_spread_frontier_flatness_cap": 2.0,
            "ori5_spread_milestone_bonus": 0.0,
            "ori5_reward_parameters": expected_ori6_reward(mode),
        })
        if mode in (
            P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
            P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        ):
            # Keep the legacy P0/P1 job identities byte-for-byte stable while
            # binding this orchestration-only variant to its one changed knob.
            result["ori5_crossing_penalty"] = candidate_crossing_penalty(mode)
        if mode in (
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        ):
            result.update({
                "ori6_rejected_crossing_continuation": True,
                "ori6_candidate": (
                    SPECTRAL_SUBSPACE_CANDIDATE_NAME
                    if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                    else REJECTED_CROSSING_CANDIDATE_NAME
                ),
                "ori6_candidate_schema_version": (
                    SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
                    if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                    else REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                ),
                "ori6_crossing_policy": REJECTED_CROSSING_POLICY,
                "ori6_reward_parameters": expected_ori6_reward(mode),
                "crossing_transition_policy": (
                    expected_crossing_transition(mode)
                ),
                "candidate_entrypoint": candidate_entrypoint(
                    task, implementation, mode
                ),
                "base_ori6_source_is_unmodified": True,
            })
        if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
            runtime = spectral_subspace_runtime_identity()
            result.update({
                "ori6_spectral_subspace": True,
                "ori6_spectral_weight": SPECTRAL_SUBSPACE_WEIGHT,
                "ori6_spectral_relative_epsilon": (
                    SPECTRAL_SUBSPACE_RELATIVE_EPSILON
                ),
                "ori6_parent_candidate": REJECTED_CROSSING_CANDIDATE_NAME,
                "spectral_subspace_formula": runtime["formula"],
                "spectral_subspace_formula_sha256": (
                    SPECTRAL_SUBSPACE_FORMULA_SHA256
                ),
                "base_and_rejected_authorities_unmodified": True,
            })
    else:
        result.update({
            "reward_design": ORI5_REWARD_DESIGN,
            "reward_schema_version": ORI5_REWARD_SCHEMA_VERSION,
            "ori5_spread_milestone_bonus": 0.9,
            "ori5_reward_parameters": expected_ori5_reward(),
        })
    return result


def build_command(job: dict, destination: Path) -> list[str]:
    task = Task(**job["task"])
    profile = Profile(**job["profile"])
    protocol = Protocol(**{
        **job["protocol"],
        "seeds": tuple(job["protocol"]["seeds"]),
        "tasks": tuple(job["protocol"]["tasks"]),
        "implementations": tuple(job["protocol"]["implementations"]),
    })
    seed = int(job["seed"])
    implementation = job["implementation"]
    mode = job.get("candidate_mode")
    common_arguments = list(COMMON_ARGUMENTS)
    if implementation == "ori6" and mode in (
        P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
        P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        penalty_index = common_arguments.index("--ori5-crossing-penalty")
        common_arguments[penalty_index + 1] = format(
            candidate_crossing_penalty(mode), ".17g"
        )
    command = [
        sys.executable, str(
            SCRIPT_DIRECTORY / candidate_entrypoint(
                task, implementation, mode
            )
        ),
        "--episodes", str(protocol.episodes),
        "--batch-size", str(profile.batch_size),
        "--max-steps", str(task.max_steps),
        "--max-step-width", format(
            candidate_max_step_width(mode, task), ".17g"
        ),
        "--flatness-subspace-dimension", str(task.flatness_subspace_dimension),
        "--target-spread", format(task.target_spread, ".17g"),
        "--actor-learning-rate", format(profile.actor_learning_rate, ".17g"),
        "--critic-learning-rate", format(profile.critic_learning_rate, ".17g"),
        "--entropy-coefficient",
        format(candidate_entropy_coefficient(mode, profile), ".17g"),
        "--actor-lr-milestones", "2000", "4000", "7000",
        "--actor-lr-factors", "0.3", "0.1", "0.03",
        "--eval-interval", str(protocol.eval_interval),
        "--periodic-stochastic-eval-episodes", str(protocol.periodic_evaluation_episodes),
        "--eval-episodes", str(protocol.final_evaluation_episodes),
        "--checkpoint-every", str(protocol.checkpoint_every),
        "--print-every", str(protocol.print_every),
        "--seed", str(seed),
        "--selection-evaluation-seed", str(protocol.selection_evaluation_seed),
        "--final-evaluation-seed", str(protocol.final_evaluation_seed),
        "--no-random-search",
        "--output-dir", str(ensure_below_root(destination)),
        *common_arguments,
    ]
    if implementation == "ori6":
        command.extend([
            "--ori5-spread-milestone-bonus", "0",
            "--ori6-spread-frontier-mode", core_frontier_mode(mode),
            "--ori6-spread-frontier-bonus",
            format(candidate_frontier_bonus(mode), ".17g"),
            "--ori6-spread-frontier-flatness-cap", "2",
        ])
        if mode in (
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        ):
            command.append("--ori6-rejected-crossing-continuation")
        if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
            command.extend([
                "--ori6-spectral-subspace",
                "--ori6-spectral-weight",
                format(SPECTRAL_SUBSPACE_WEIGHT, ".17g"),
                "--ori6-spectral-relative-epsilon",
                format(SPECTRAL_SUBSPACE_RELATIVE_EPSILON, ".17g"),
            ])
    else:
        command.extend(["--ori5-spread-milestone-bonus", "0.9"])
    if task.is_three_dimensional:
        command.extend([
            "--data-dir", str((SCRIPT_DIRECTORY / task.model_directory).resolve()),
            "--no-rigid-diagnostics",
        ])
    return command


def build_jobs(protocol: Protocol, mode: str, selected_tasks: list[str]) -> list[dict]:
    if protocol.name == "entropy_diagnostic" and mode != P0_CROSS2_ENTROPY3_MODE:
        raise ValueError("entropy_diagnostic is reserved for P0X2E3")
    if protocol.name == "width_diagnostic" and mode != P0_CROSS2_WIDTH2_MODE:
        raise ValueError("width_diagnostic is reserved for P0X2W2")
    if (
        protocol.name == "crossing_continuation_diagnostic"
        and mode != P0_CROSS2_REJECTED_CONTINUATION_MODE
    ):
        raise ValueError(
            "crossing_continuation_diagnostic is reserved for P0X2RC"
        )
    if (
        protocol.name == "spectral_subspace_diagnostic"
        and mode != P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        raise ValueError(
            "spectral_subspace_diagnostic is reserved for P0X2RCS05"
        )
    if (
        protocol.name == "spectral_nonregression"
        and mode != P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        raise ValueError(
            "spectral_nonregression is reserved for P0X2RCS05"
        )
    if mode == P0_CROSS2_ENTROPY3_MODE and not (
        protocol.name == "entropy_diagnostic"
        and selected_tasks == ["water1"]
    ):
        raise ValueError(
            "P0X2E3 is a preregistered water1-only 1024-episode diagnostic; "
            "use --stage entropy_diagnostic --tasks water1"
        )
    if mode == P0_CROSS2_WIDTH2_MODE and not (
        protocol.name == "width_diagnostic"
        and selected_tasks == ["water1"]
    ):
        raise ValueError(
            "P0X2W2 is a preregistered water1-only 2048-episode diagnostic; "
            "use --stage width_diagnostic --tasks water1"
        )
    if mode == P0_CROSS2_REJECTED_CONTINUATION_MODE and not (
        protocol.name == "crossing_continuation_diagnostic"
        and selected_tasks == ["torus2", "water1"]
    ):
        raise ValueError(
            "P0X2RC is a fixed torus2+water1 2048-episode diagnostic; "
            "use --stage crossing_continuation_diagnostic "
            "--tasks torus2 water1"
        )
    spectral_scope_valid = (
        protocol.name == "spectral_subspace_diagnostic"
        and selected_tasks == ["torus2", "water1"]
    ) or (
        protocol.name == "spectral_nonregression"
        and selected_tasks == ["five_d2", "five_d3", "watersqrt2"]
    )
    if (
        mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        and not spectral_scope_valid
    ):
        raise ValueError(
            "P0X2RCS05 is authorized only for its fixed torus2+water1 "
            "diagnostic or fixed paired nonregression protocol"
        )
    hashes = source_hashes()
    jobs = []
    for task_name in protocol.tasks:
        if task_name not in selected_tasks:
            continue
        task = TASKS[task_name]
        profile = PROFILES[task.profile]
        for implementation in protocol.implementations:
            for seed in protocol.seeds:
                destination = output_directory(protocol, mode, implementation, task, seed)
                identity = {
                    "schema_version": 1,
                    "protocol": asdict(protocol),
                    "task": asdict(task),
                    "profile": asdict(profile),
                    "seed": seed,
                    "implementation": implementation,
                    "candidate_mode": mode,
                    "output_directory": str(destination),
                    "expected_config": expected_config(
                        task, profile, protocol, seed, implementation, mode
                    ),
                    "source_sha256": hashes,
                    "model_sha256": model_hashes(task),
                }
                if mode in (
                    P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
                    P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
                    P0_CROSS2_REJECTED_CONTINUATION_MODE,
                    P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
                ):
                    identity.update({
                        "core_frontier_mode": core_frontier_mode(mode),
                        "candidate_reward_sha256": candidate_reward_sha256(mode),
                    })
                    if mode == P0_CROSS2_ENTROPY3_MODE:
                        identity["candidate_optimizer_sha256"] = (
                            candidate_optimizer_sha256(mode, profile)
                        )
                    elif mode == P0_CROSS2_WIDTH2_MODE:
                        identity.update({
                            "candidate_optimizer_sha256": (
                                candidate_optimizer_sha256(mode, profile)
                            ),
                            "candidate_mechanics_sha256": (
                                candidate_mechanics_sha256(mode, task)
                            ),
                        })
                    elif mode in (
                        P0_CROSS2_REJECTED_CONTINUATION_MODE,
                        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
                    ):
                        identity.update({
                            "candidate_optimizer_sha256": (
                                candidate_optimizer_sha256(mode, profile)
                            ),
                            "candidate_mechanics_sha256": (
                                candidate_mechanics_sha256(mode, task)
                            ),
                            "candidate_transition_semantics_sha256": (
                                candidate_transition_semantics_sha256(mode)
                            ),
                            "candidate_additive_source_sha256": (
                                rejected_crossing_additive_source_hashes()
                            ),
                            "ori6_candidate": (
                                REJECTED_CROSSING_CANDIDATE_NAME
                            ),
                            "ori6_crossing_policy": (
                                REJECTED_CROSSING_POLICY
                            ),
                        })
                        if mode == (
                            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                        ):
                            identity.update({
                                "candidate_additive_source_sha256": {
                                    **rejected_crossing_additive_source_hashes(),
                                    **spectral_subspace_additive_source_hashes(),
                                },
                                "spectral_additive_source_sha256": (
                                    spectral_subspace_additive_source_hashes()
                                ),
                                "ori6_candidate": (
                                    SPECTRAL_SUBSPACE_CANDIDATE_NAME
                                ),
                                "ori6_parent_candidate": (
                                    REJECTED_CROSSING_CANDIDATE_NAME
                                ),
                                "spectral_formula_sha256": (
                                    SPECTRAL_SUBSPACE_FORMULA_SHA256
                                ),
                                "spectral_weight": SPECTRAL_SUBSPACE_WEIGHT,
                            })
                fingerprint = canonical_digest(identity)
                identity.update({
                    "job_id": (
                        f"{protocol.name}:{MODE_SLUGS[mode]}:{implementation}:"
                        f"{task.name}:{profile.name}:seed{seed}"
                    ),
                    "fingerprint": fingerprint,
                })
                identity["command"] = build_command(identity, destination)
                jobs.append(identity)
    return jobs


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_evaluation_episodes(total: int, interval: int) -> list[int]:
    values = [1, *range(interval, total + 1, interval)]
    if total not in values:
        values.append(total)
    return sorted(set(values))


def validate_artifact(job: dict, destination: Path, endpoint: int | None = None) -> dict:
    errors = []
    mode = job.get("candidate_mode")
    protocol = job["protocol"]
    expected_rows = int(protocol["episodes"] if endpoint is None else endpoint)
    # Passing ``endpoint`` explicitly means "validate an atomic training
    # checkpoint", even when that checkpoint happens to equal the configured
    # episode budget.  A child can be stopped after the final training flush
    # but before held-out evaluation/plots; treating that as a complete run
    # would incorrectly require artifacts which were never started.
    complete = endpoint is None
    required = [
        "run_config.json", "training_history.csv", "update_history.csv",
        "deterministic_evaluation_history.csv", "stochastic_evaluation_history.csv",
        "checkpoint.pt", "best_policy.pt", "best_policy_metrics.json",
    ]
    if complete:
        required.extend([
            "summary.json", "evaluation_summary.csv",
            "best_policy_evaluation_summary.csv", "policy.pt",
            "learning_curves.png",
        ])
        if job["implementation"] == "ori6":
            required.extend([
                "ori6_frontier_learning_curve.png",
                "ori6_near_miss_learning_curve.png",
            ])
            if mode in (
                P0_CROSS2_REJECTED_CONTINUATION_MODE,
                P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
            ):
                required.append(
                    "ori6_rejected_crossing_learning_curve.png"
                )
            if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
                required.append(
                    "ori6_spectral_subspace_learning_curve.png"
                )
        elif mode == P0_CROSS2_ENTROPY3_MODE:
            required.extend([
                "ori5_milestone_learning_curve.png",
                "near_miss_learning_curve.png",
            ])
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        return {"valid": False, "errors": ["missing: " + ", ".join(missing)], "path": str(destination)}
    try:
        with open(destination / "run_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        history = read_csv(destination / "training_history.csv")
        updates = read_csv(destination / "update_history.csv")
        deterministic = read_csv(destination / "deterministic_evaluation_history.csv")
        stochastic = read_csv(destination / "stochastic_evaluation_history.csv")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": [repr(error)], "path": str(destination)}
    if not values_equal(
        config, config_expectation(job, complete=complete)
    ):
        errors.append("run_config does not match expected mechanics/optimizer/reward")
    if len(history) != expected_rows:
        errors.append(f"training rows {len(history)} != {expected_rows}")
    episodes = [int(float(row.get("episode", -1))) for row in history]
    if episodes != list(range(1, expected_rows + 1)):
        errors.append("training episode sequence is not contiguous")
    batch = int(job["profile"]["batch_size"])
    expected_update_rows = math.ceil(expected_rows / batch)
    if len(updates) != expected_update_rows:
        errors.append(f"update rows {len(updates)} != {expected_update_rows}")
    expected_eval = expected_evaluation_episodes(expected_rows, int(protocol["eval_interval"]))
    for name, rows in (("deterministic", deterministic), ("stochastic", stochastic)):
        actual = [int(float(row.get("episode", -1))) for row in rows]
        if actual != expected_eval:
            errors.append(f"{name} evaluation episodes {actual} != {expected_eval}")
    if job["implementation"] == "ori6" and history:
        required_columns = {
            "ori6_maximum_spread_progress", "ori6_maximum_spread_potential",
            "ori6_total_spread_frontier_reward",
            "ori6_total_frontier_rollback_reward",
            "ori6_total_joint_milestone_rollback_reward",
            "ori6_reward_decomposition_error",
            "ori6_nonfinite_reward_component_count",
        }
        absent = sorted(required_columns - set(history[0]))
        if absent:
            errors.append("missing ori6 columns: " + ", ".join(absent))
        else:
            max_error = max(abs(float(row["ori6_reward_decomposition_error"])) for row in history)
            nonfinite = sum(int(float(row["ori6_nonfinite_reward_component_count"])) for row in history)
            if max_error > 1.0e-10:
                errors.append(f"reward decomposition error {max_error}")
            if nonfinite:
                errors.append(f"nonfinite reward components {nonfinite}")
        if mode in (
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        ):
            candidate_columns = {
                "ori6_candidate",
                "ori6_candidate_schema_version",
                "ori6_crossing_policy",
                "ori6_rejected_crossing_count",
                "ori6_rejected_crossing_rate",
                "ori6_rejected_crossing_streak_count",
                "ori6_escape_after_rejected_crossing_count",
                "ori6_rejected_crossing_escape_rate",
                "ori6_max_consecutive_rejected_crossings",
                "ori6_success_after_rejected_crossing",
                "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
            }
            missing_candidate = sorted(candidate_columns - set(history[0]))
            if missing_candidate:
                errors.append(
                    "missing rejected-crossing columns: "
                    + ", ".join(missing_candidate)
                )
            elif any(
                row["ori6_candidate"] != REJECTED_CROSSING_CANDIDATE_NAME
                or row["ori6_candidate_schema_version"]
                != REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                or row["ori6_crossing_policy"] != REJECTED_CROSSING_POLICY
                for row in history
            ):
                errors.append("rejected-crossing candidate identity mismatch")
        if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
            spectral_columns = {
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
            }
            missing_spectral = sorted(spectral_columns - set(history[0]))
            if missing_spectral:
                errors.append(
                    "missing spectral-subspace columns: "
                    + ", ".join(missing_spectral)
                )
            else:
                identity_ok = all(
                    row["ori6_spectral_candidate"]
                    == SPECTRAL_SUBSPACE_CANDIDATE_NAME
                    and row["ori6_spectral_candidate_schema_version"]
                    == SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
                    and row["ori6_spectral_formula_sha256"]
                    == SPECTRAL_SUBSPACE_FORMULA_SHA256
                    and row["ori6_spectral_parent_candidate"]
                    == REJECTED_CROSSING_CANDIDATE_NAME
                    and values_equal(
                        float(row["ori6_spectral_weight"]),
                        SPECTRAL_SUBSPACE_WEIGHT,
                    )
                    and values_equal(
                        float(row["ori6_spectral_relative_epsilon"]),
                        SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
                    )
                    for row in history
                )
                if not identity_ok:
                    errors.append("spectral-subspace candidate identity mismatch")
                maximum_error = max(
                    max(
                        abs(float(row[
                            "ori6_spectral_reward_decomposition_error"
                        ])),
                        abs(float(row[
                            "ori6_spectral_maximum_abs_step_reward_"
                            "decomposition_error"
                        ])),
                    )
                    for row in history
                )
                if maximum_error > 1.0e-10:
                    errors.append(
                        f"spectral reward decomposition error {maximum_error}"
                    )
                spectral_nonfinite = sum(
                    int(float(row[
                        "ori6_spectral_nonfinite_component_count"
                    ]))
                    for row in history
                )
                if spectral_nonfinite:
                    errors.append(
                        "spectral nonfinite reward components "
                        f"{spectral_nonfinite}"
                    )
                telescope_error = max(
                    max(
                        abs(
                            float(row["ori6_spectral_total_reward"])
                            - float(row[
                                "ori6_spectral_total_progress_reward"
                            ])
                            - float(row[
                                "ori6_spectral_total_rollback_reward"
                            ])
                        ),
                        abs(
                            float(row["ori6_spectral_total_reward"])
                            - float(row[
                                "ori6_spectral_cumulative_reward"
                            ])
                        ),
                    )
                    for row in history
                )
                if telescope_error > 1.0e-10:
                    errors.append(
                        f"spectral telescope error {telescope_error}"
                    )
    elif job["implementation"] == "ori5_reference" and history:
        required_columns = {
            "ori5_reward_decomposition_error",
            "ori5_maximum_abs_step_reward_decomposition_error",
            "ori5_nonfinite_reward_component_count",
        }
        absent = sorted(required_columns - set(history[0]))
        if absent:
            errors.append("missing ori5 reward columns: " + ", ".join(absent))
        else:
            maximum_error = max(
                max(
                    abs(float(row["ori5_reward_decomposition_error"])),
                    abs(float(row[
                        "ori5_maximum_abs_step_reward_decomposition_error"
                    ])),
                )
                for row in history
            )
            nonfinite = sum(
                int(float(row["ori5_nonfinite_reward_component_count"]))
                for row in history
            )
            if maximum_error > 1.0e-10:
                errors.append(
                    f"ori5 reward decomposition error {maximum_error}"
                )
            if nonfinite:
                errors.append(f"ori5 nonfinite reward components {nonfinite}")
    if task := TASKS.get(job["task"]["name"]):
        if task.is_three_dimensional:
            adapter = config.get("three_dimensional_adapter", {})
            if not values_equal(adapter.get("model_file_sha256"), job["model_sha256"]):
                errors.append("3-D model hash mismatch")
    if complete:
        final_rows = read_csv(destination / "evaluation_summary.csv")
        best_rows = read_csv(destination / "best_policy_evaluation_summary.csv")
        n = int(protocol["final_evaluation_episodes"])
        if len(final_rows) != n or len(best_rows) != n:
            errors.append(f"held-out rows final/best={len(final_rows)}/{len(best_rows)} != {n}")
        try:
            with open(destination / "summary.json", encoding="utf-8") as handle:
                summary = json.load(handle)
            if job["implementation"] == "ori6":
                if not values_equal(summary.get("ori6_reward_parameters"), expected_ori6_reward(job["candidate_mode"])):
                    errors.append("summary ori6 reward parameters mismatch")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid summary: {error}")
    return {
        "valid": not errors,
        "errors": errors,
        "path": str(destination.resolve()),
        "endpoint_episode": expected_rows,
        "complete": complete,
        "training_rows": len(history),
        "update_rows": len(updates),
    }


def runner_directory(job: dict, destination: Path) -> Path:
    tag = canonical_digest(str(destination.resolve()))[:12]
    return ensure_below_root(
        EXPERIMENT_ROOT / "_runner_logs" / job["protocol"]["name"]
        / MODE_SLUGS[job["candidate_mode"]] / job["task"]["name"]
        / job["implementation"] / f"seed{job['seed']}" / tag
    )


def status_path(job: dict, destination: Path) -> Path:
    return runner_directory(job, destination) / "status.json"


def load_status(job: dict, destination: Path) -> dict | None:
    path = status_path(job, destination)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def active_python_processes() -> list[dict]:
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode or not result.stdout.strip():
            return []
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        rows = payload if isinstance(payload, list) else [payload]
        return [{
            "pid": int(row.get("ProcessId", -1)),
            "command_line": str(row.get("CommandLine") or ""),
        } for row in rows]
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True,
        timeout=10, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and "python" in fields[1].lower():
            rows.append({"pid": int(fields[0]), "command_line": fields[1]})
    return rows


def external_process(destination: Path, processes: list[dict]) -> dict | None:
    absolute = str(destination.resolve()).lower().replace("\\", "/")
    relative = str(destination.resolve().relative_to(SCRIPT_DIRECTORY)).lower().replace("\\", "/")
    for process in processes:
        command = process["command_line"].lower().replace("\\", "/")
        if absolute in command or relative in command:
            return process
    return None


def effective_candidates(destination: Path) -> list[Path]:
    parent = destination.parent
    if not parent.is_dir():
        return []
    return [path for path in parent.glob(destination.name + "*") if path.is_dir()]


REJECTED_CROSSING_POST_FINAL_CONFIG_KEYS = frozenset({
    "ori6_candidate",
    "ori6_candidate_schema_version",
    "ori6_reward_parameters",
    "crossing_transition_policy",
    "candidate_entrypoint",
    "base_ori6_source_is_unmodified",
})
SPECTRAL_SUBSPACE_POST_FINAL_CONFIG_KEYS = frozenset({
    *REJECTED_CROSSING_POST_FINAL_CONFIG_KEYS,
    "ori6_parent_candidate",
    "spectral_subspace_formula",
    "spectral_subspace_formula_sha256",
    "base_and_rejected_authorities_unmodified",
})


def config_expectation(job: dict, *, complete: bool) -> dict:
    """Return the config fields available at an atomic checkpoint.

    The additive wrapper deliberately appends provenance-only metadata after
    the unchanged ori6 trainer has completed held-out evaluation.  Its
    opt-in flag, crossing policy, and effective ``ori5_reward_parameters``
    are present from episode one and remain sufficient to identify a live
    checkpoint; the post-final fields are required for a complete artifact.
    """

    expected = dict(job["expected_config"])
    if (
        not complete
        and job.get("candidate_mode")
        in (
            P0_CROSS2_REJECTED_CONTINUATION_MODE,
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
        )
    ):
        post_keys = (
            SPECTRAL_SUBSPACE_POST_FINAL_CONFIG_KEYS
            if job.get("candidate_mode")
            == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
            else REJECTED_CROSSING_POST_FINAL_CONFIG_KEYS
        )
        for key in post_keys:
            expected.pop(key, None)
    return expected


def config_matches_job(path: Path, job: dict) -> bool:
    config_path = path / "run_config.json"
    if not config_path.is_file():
        return False
    try:
        with open(config_path, encoding="utf-8") as handle:
            return values_equal(
                json.load(handle), config_expectation(job, complete=False)
            )
    except (OSError, json.JSONDecodeError):
        return False


def discover_output(job: dict, destination: Path) -> tuple[Path | None, dict | None]:
    matches = [path for path in effective_candidates(destination) if config_matches_job(path, job)]
    if not matches:
        return None, None
    complete = []
    invalid = []
    for path in matches:
        validation = validate_artifact(job, path)
        (complete if validation["valid"] else invalid).append((path, validation))
    if len(complete) == 1:
        return complete[0]
    if len(complete) > 1:
        return None, {"valid": False, "errors": ["multiple complete matching outputs"], "paths": [str(p) for p, _ in complete]}
    return None, invalid[0][1] if len(invalid) == 1 else {
        "valid": False, "errors": ["multiple incomplete matching outputs"],
        "paths": [str(p) for p, _ in invalid],
    }


def next_retry(destination: Path) -> Path:
    root = destination.parent / (destination.name + "_attempts")
    numbers = []
    if root.is_dir():
        for path in root.glob("attempt_*"):
            try:
                numbers.append(int(path.name.split("_")[-1]))
            except ValueError:
                pass
    return ensure_below_root(root / f"attempt_{max(numbers, default=0) + 1:03d}")


def classify(job: dict, rerun_incomplete: bool, processes: list[dict]) -> tuple[str, Path, dict | None]:
    destination = Path(job["output_directory"])
    active = external_process(destination, processes)
    if active and process_is_alive(active["pid"]):
        return "external_active", destination, active
    status = load_status(job, destination)
    if status and status.get("status") in ("launching", "running"):
        try:
            if process_is_alive(int(status.get("pid", -1))):
                return "running", destination, status
        except (TypeError, ValueError):
            pass
    effective, validation = discover_output(job, destination)
    if effective is not None:
        return "complete", effective, validation
    if status and status.get("status") == "early_stopped_at_checkpoint":
        endpoint = int(status.get("checkpoint_episode", 0))
        partial_path = Path(status.get("effective_output_directory") or destination)
        partial = validate_artifact(job, partial_path, endpoint=endpoint)
        if partial["valid"]:
            return "early_stopped", partial_path, partial
    matching = [path for path in effective_candidates(destination) if config_matches_job(path, job)]
    if matching:
        if rerun_incomplete:
            return "pending_retry", next_retry(destination), validation
        return "incomplete", matching[0], validation
    if destination.exists():
        if rerun_incomplete:
            return "pending_retry", next_retry(destination), validation
        return "incomplete", destination, validation
    return "pending", destination, None


def manifest_path(protocol: Protocol, mode: str, jobs: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "protocol": asdict(protocol),
        "candidate_mode": mode,
        "candidate_mode_slug": MODE_SLUGS[mode],
        "jobs": jobs,
        "source_sha256": source_hashes(),
        "model_sha256_by_task": {name: model_hashes(task) for name, task in TASKS.items()},
        "random_baseline_authority": (
            rejected_crossing_random_baseline_disposition()
            if mode in (
                P0_CROSS2_REJECTED_CONTINUATION_MODE,
                P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
            )
            else {
                "status": "not_used_for_bounded_mechanics_diagnostic",
                "reason": "max_step_width differs from the existing random baseline",
                "new_random_baseline_required_only_if_adopted": True,
            }
            if mode == P0_CROSS2_WIDTH2_MODE
            else random_baseline_authority()
        ),
        "offline_rollback_only": True,
        "training_candidates": [MODE_SLUGS[mode]],
        "exact_resume_supported": False,
        "recovery_semantics": "completed-skip or preserved-incomplete fresh restart",
    }
    if mode == P0_CROSS2_MODE:
        if not P0_CROSS2_OFFLINE_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_OFFLINE_AUDIT)
        if not P0_RECLASSIFICATION_REPORT.is_file():
            raise FileNotFoundError(P0_RECLASSIFICATION_REPORT)
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "single_changed_reward_parameter": {
                "name": "ori5_crossing_penalty",
                "P0_value": candidate_crossing_penalty(P0_MODE),
                "candidate_value": candidate_crossing_penalty(mode),
            },
            "offline_saved_path_ordering_authority": (
                str(P0_CROSS2_OFFLINE_AUDIT.resolve())
            ),
            "offline_saved_path_ordering_authority_sha256": (
                file_sha256(P0_CROSS2_OFFLINE_AUDIT)
            ),
            "P0_crossing_attractor_authority": str(
                P0_RECLASSIFICATION_REPORT.resolve()
            ),
            "P0_crossing_attractor_authority_sha256": file_sha256(
                P0_RECLASSIFICATION_REPORT
            ),
        })
    elif mode == P0_CROSS2_FRONTIER12_MODE:
        if not P0_CROSS2_FRONTIER12_OFFLINE_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_FRONTIER12_OFFLINE_AUDIT)
        if not P0_CROSS2_REJECT_REPORT.is_file():
            raise FileNotFoundError(P0_CROSS2_REJECT_REPORT)
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "parent_candidate_mode": P0_CROSS2_MODE,
            "parent_candidate_reward_sha256": candidate_reward_sha256(
                P0_CROSS2_MODE
            ),
            "single_changed_reward_parameter": {
                "name": "ori6_spread_frontier_bonus",
                "P0X2_value": candidate_frontier_bonus(P0_CROSS2_MODE),
                "candidate_value": candidate_frontier_bonus(mode),
            },
            "offline_saved_path_ordering_authority": str(
                P0_CROSS2_FRONTIER12_OFFLINE_AUDIT.resolve()
            ),
            "offline_saved_path_ordering_authority_sha256": file_sha256(
                P0_CROSS2_FRONTIER12_OFFLINE_AUDIT
            ),
            "parent_candidate_reject_authority": str(
                P0_CROSS2_REJECT_REPORT.resolve()
            ),
            "parent_candidate_reject_authority_sha256": file_sha256(
                P0_CROSS2_REJECT_REPORT
            ),
        })
    elif mode == P0_CROSS2_ENTROPY3_MODE:
        if not P0_CROSS2_OFFLINE_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_OFFLINE_AUDIT)
        if not P0_CROSS2_FRONTIER12_REJECT_REPORT.is_file():
            raise FileNotFoundError(P0_CROSS2_FRONTIER12_REJECT_REPORT)
        if not P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT.is_file():
            raise FileNotFoundError(P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT)
        profile = PROFILES[TASKS["water1"].profile]
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "parent_candidate_mode": P0_CROSS2_MODE,
            "parent_candidate_reward_sha256": candidate_reward_sha256(
                P0_CROSS2_MODE
            ),
            "reward_identity_reused_from_parent": True,
            "candidate_optimizer_sha256": candidate_optimizer_sha256(
                mode, profile
            ),
            "parent_candidate_optimizer_sha256": candidate_optimizer_sha256(
                P0_CROSS2_MODE, profile
            ),
            "single_changed_training_parameter": {
                "name": "entropy_coefficient",
                "P0X2_value": candidate_entropy_coefficient(
                    P0_CROSS2_MODE, profile
                ),
                "candidate_value": candidate_entropy_coefficient(mode, profile),
            },
            "screen_scope": {
                "stage": "entropy_diagnostic",
                "tasks": ["water1"],
                "seeds": list(PROTOCOLS["entropy_diagnostic"].seeds),
                "episodes": PROTOCOLS["entropy_diagnostic"].episodes,
                "checkpoint_every": PROTOCOLS["entropy_diagnostic"].checkpoint_every,
                "early_gate_checkpoints": [512, 1024],
            },
            "offline_saved_path_ordering_authority": str(
                P0_CROSS2_OFFLINE_AUDIT.resolve()
            ),
            "offline_saved_path_ordering_authority_sha256": file_sha256(
                P0_CROSS2_OFFLINE_AUDIT
            ),
            "preceding_candidate_reject_authority": str(
                P0_CROSS2_FRONTIER12_REJECT_REPORT.resolve()
            ),
            "preceding_candidate_reject_authority_sha256": file_sha256(
                P0_CROSS2_FRONTIER12_REJECT_REPORT
            ),
            "entropy_preflight_authority": str(
                P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT.resolve()
            ),
            "entropy_preflight_authority_sha256": file_sha256(
                P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT
            ),
        })
    elif mode == P0_CROSS2_WIDTH2_MODE:
        if not P0_CROSS2_OFFLINE_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_OFFLINE_AUDIT)
        if not P0_CROSS2_ENTROPY3_REJECT_REPORT.is_file():
            raise FileNotFoundError(P0_CROSS2_ENTROPY3_REJECT_REPORT)
        if not P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT)
        task = TASKS["water1"]
        profile = PROFILES[task.profile]
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "parent_candidate_mode": P0_CROSS2_MODE,
            "parent_candidate_reward_sha256": candidate_reward_sha256(
                P0_CROSS2_MODE
            ),
            "reward_identity_reused_from_parent": True,
            "candidate_optimizer_sha256": candidate_optimizer_sha256(
                mode, profile
            ),
            "parent_candidate_optimizer_sha256": candidate_optimizer_sha256(
                P0_CROSS2_MODE, profile
            ),
            "optimizer_identity_reused_from_parent": True,
            "candidate_mechanics_sha256": candidate_mechanics_sha256(
                mode, task
            ),
            "parent_candidate_mechanics_sha256": candidate_mechanics_sha256(
                P0_CROSS2_MODE, task
            ),
            "single_changed_mechanics_parameter": {
                "name": "max_step_width",
                "P0X2_value": candidate_max_step_width(
                    P0_CROSS2_MODE, task
                ),
                "candidate_value": candidate_max_step_width(mode, task),
            },
            "screen_scope": {
                "stage": "width_diagnostic",
                "tasks": ["water1"],
                "seeds": list(PROTOCOLS["width_diagnostic"].seeds),
                "episodes": PROTOCOLS["width_diagnostic"].episodes,
                "checkpoint_every": PROTOCOLS["width_diagnostic"].checkpoint_every,
                "early_gate_checkpoints": [512, 1024, 2048],
                "budget_rationale": (
                    "Continue the same atomic cohort to 2048 only when the "
                    "1024 constructive-orientation gate passes."
                ),
            },
            "offline_saved_path_ordering_authority": str(
                P0_CROSS2_OFFLINE_AUDIT.resolve()
            ),
            "offline_saved_path_ordering_authority_sha256": file_sha256(
                P0_CROSS2_OFFLINE_AUDIT
            ),
            "preceding_candidate_reject_authority": str(
                P0_CROSS2_ENTROPY3_REJECT_REPORT.resolve()
            ),
            "preceding_candidate_reject_authority_sha256": file_sha256(
                P0_CROSS2_ENTROPY3_REJECT_REPORT
            ),
            "width_feasibility_authority": str(
                P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT.resolve()
            ),
            "width_feasibility_authority_sha256": file_sha256(
                P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT
            ),
        })
    elif mode == P0_CROSS2_REJECTED_CONTINUATION_MODE:
        if not P0_CROSS2_OFFLINE_AUDIT.is_file():
            raise FileNotFoundError(P0_CROSS2_OFFLINE_AUDIT)
        if not P0_CROSS2_WIDTH2_REJECT_REPORT.is_file():
            raise FileNotFoundError(P0_CROSS2_WIDTH2_REJECT_REPORT)
        selected_task_names = list(
            PROTOCOLS["crossing_continuation_diagnostic"].tasks
        )
        profiles = {
            TASKS[name].profile: PROFILES[TASKS[name].profile]
            for name in selected_task_names
        }
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "ori6_candidate": REJECTED_CROSSING_CANDIDATE_NAME,
            "ori6_candidate_schema_version": (
                REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            ),
            "candidate_additive_source_sha256": (
                rejected_crossing_additive_source_hashes()
            ),
            "candidate_orchestration_source_sha256": (
                rejected_crossing_orchestration_source_hashes()
            ),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "parent_candidate_mode": P0_CROSS2_MODE,
            "parent_candidate_reward_sha256": candidate_reward_sha256(
                P0_CROSS2_MODE
            ),
            "numeric_reward_hyperparameters_reused_from_parent": True,
            "candidate_optimizer_sha256_by_profile": {
                name: candidate_optimizer_sha256(mode, profile)
                for name, profile in profiles.items()
            },
            "parent_candidate_optimizer_sha256_by_profile": {
                name: candidate_optimizer_sha256(
                    P0_CROSS2_MODE, profile
                )
                for name, profile in profiles.items()
            },
            "optimizer_identity_reused_from_parent": True,
            "candidate_mechanics_sha256_by_task": {
                name: candidate_mechanics_sha256(mode, TASKS[name])
                for name in selected_task_names
            },
            "parent_candidate_mechanics_sha256_by_task": {
                name: candidate_mechanics_sha256(
                    P0_CROSS2_MODE, TASKS[name]
                )
                for name in selected_task_names
            },
            "physical_mechanics_identity_reused_from_parent": True,
            "candidate_transition_semantics_sha256": (
                candidate_transition_semantics_sha256(mode)
            ),
            "parent_transition_semantics_sha256": (
                candidate_transition_semantics_sha256(P0_CROSS2_MODE)
            ),
            "single_changed_environment_semantics": {
                "name": "self_crossing_transition_policy",
                "P0X2_value": "terminate_on_self_crossing",
                "candidate_value": REJECTED_CROSSING_POLICY,
                "P0X2_crossed_candidate_committed": True,
                "candidate_crossed_candidate_committed": False,
                "rejected_attempt_consumes_step": True,
                "last_safe_state_restored": True,
                "past_frontier_and_joint_ledgers_retained": True,
            },
            "screen_scope": {
                "stage": "crossing_continuation_diagnostic",
                "tasks": selected_task_names,
                "seeds": list(
                    PROTOCOLS[
                        "crossing_continuation_diagnostic"
                    ].seeds
                ),
                "episodes": PROTOCOLS[
                    "crossing_continuation_diagnostic"
                ].episodes,
                "checkpoint_every": PROTOCOLS[
                    "crossing_continuation_diagnostic"
                ].checkpoint_every,
                "early_gate_checkpoints": [512, 1024, 2048],
                "task_cohorts_are_independent_and_fixed": True,
            },
            "early_gate_specification": (
                crossing_continuation_gate_specification()
            ),
            "parent_saved_path_ordering_authority": str(
                P0_CROSS2_OFFLINE_AUDIT.resolve()
            ),
            "parent_saved_path_ordering_authority_sha256": file_sha256(
                P0_CROSS2_OFFLINE_AUDIT
            ),
            "parent_saved_path_scope": (
                "numeric P0X2 reward and collision-free path evidence only; "
                "terminal self-crossing path returns are not reused"
            ),
            "preceding_candidate_reject_authority": str(
                P0_CROSS2_WIDTH2_REJECT_REPORT.resolve()
            ),
            "preceding_candidate_reject_authority_sha256": file_sha256(
                P0_CROSS2_WIDTH2_REJECT_REPORT
            ),
            "adoption_requirements": {
                "fresh_random_baseline_all_five_tasks": True,
                "fresh_successful_task_regression": True,
                "screen_pass_does_not_authorize_full_training": True,
            },
        })
    elif mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
        if not P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT.is_file():
            raise FileNotFoundError(
                P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
            )
        if not SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY.is_file():
            raise FileNotFoundError(SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY)
        validate_P0X2RC_reject_authority(
            P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
        )
        validate_spectral_subspace_offline_authority(
            SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
        )
        if protocol.name not in (
            "spectral_subspace_diagnostic", "spectral_nonregression",
        ):
            raise RuntimeError(
                "P0X2RCS05 manifest is restricted to its bounded diagnostic "
                "and paired nonregression protocols"
            )
        protocol_expected = PROTOCOLS[protocol.name]
        if protocol != protocol_expected:
            raise RuntimeError("P0X2RCS05 protocol identity mismatch")
        selected_task_names = list(protocol_expected.tasks)
        profiles = {
            TASKS[name].profile: PROFILES[TASKS[name].profile]
            for name in selected_task_names
        }
        runtime = spectral_subspace_runtime_identity()
        parent_mode = P0_CROSS2_REJECTED_CONTINUATION_MODE
        if protocol.name == "spectral_subspace_diagnostic":
            screen_scope = {
                "stage": "spectral_subspace_diagnostic",
                "tasks": selected_task_names,
                "seeds": list(protocol_expected.seeds),
                "episodes": protocol_expected.episodes,
                "checkpoint_every": protocol_expected.checkpoint_every,
                "early_gate_checkpoints": [512, 1024, 2048],
                "task_cohorts_are_independent_and_fixed": True,
                "paired_parent_seeds": [3201, 3202, 3203],
                "parent_torus_endpoint": 512,
                "parent_water1_endpoint": 1024,
            }
            early_gate_specification = spectral_subspace_gate_specification()
            stage_specific = {
                "parent_comparison_scope": {
                    "same_seeds": [3201, 3202, 3203],
                    "common_prefix_only": True,
                    "performance_comparison_is_diagnostic_not_a_gate": True,
                    "missing_parent_1024_torus_does_not_invalidate": True,
                    "parent_artifacts_are_immutable_reject_evidence": True,
                },
                "adoption_requirements": {
                    "fresh_confirmation_seeds": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "fresh_successful_task_regression": True,
                    "screen_pass_does_not_authorize_full_training": True,
                },
            }
        else:
            screen_scope = {
                "stage": "spectral_nonregression",
                "tasks": selected_task_names,
                "seeds": list(protocol_expected.seeds),
                "episodes": protocol_expected.episodes,
                "checkpoint_every": protocol_expected.checkpoint_every,
                "live_early_gate_checkpoints": [512],
                "implementations": list(protocol_expected.implementations),
                "jobs_per_task": 6,
                "total_jobs": 18,
                "task_cohorts_are_independent_and_fixed": True,
            }
            early_gate_specification = (
                spectral_nonregression_gate_specification()
            )
            report_binding = spectral_screen_report_binding()
            stage_specific = {
                "spectral_screen_report": report_binding["path"],
                "spectral_screen_report_sha256": report_binding["sha256"],
                "spectral_screen_report_status": report_binding["status"],
                "paired_reference": {
                    "implementation": "ori5_reference",
                    "reward_sha256": canonical_digest({
                        "reward_design": ORI5_REWARD_DESIGN,
                        "reward_schema_version": ORI5_REWARD_SCHEMA_VERSION,
                        "effective_parameters": expected_ori5_reward(),
                    }),
                    "mechanics_and_optimizer_profiles_are_exactly_paired": True,
                },
                "promotion_requirements": {
                    "formal_spectral_report_explicitly_permits_"
                    "preregistered_regression": True,
                    "formal_spectral_report_does_not_authorize_"
                    "adoption_or_full": True,
                    "all_three_successful_task_nonregression_gates_must_pass": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "regression_pass_is_necessary_but_not_sufficient_for_full": True,
                },
            }
        payload.update({
            "core_frontier_mode": core_frontier_mode(mode),
            "ori6_candidate": SPECTRAL_SUBSPACE_CANDIDATE_NAME,
            "ori6_candidate_schema_version": (
                SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
            ),
            "parent_candidate_mode": parent_mode,
            "parent_ori6_candidate": REJECTED_CROSSING_CANDIDATE_NAME,
            "candidate_additive_source_sha256": {
                **rejected_crossing_additive_source_hashes(),
                **spectral_subspace_additive_source_hashes(),
            },
            "parent_additive_source_sha256": (
                rejected_crossing_additive_source_hashes()
            ),
            "spectral_additive_source_sha256": (
                spectral_subspace_additive_source_hashes()
            ),
            "candidate_orchestration_source_sha256": (
                rejected_crossing_orchestration_source_hashes()
            ),
            "candidate_reward_sha256": candidate_reward_sha256(mode),
            "parent_candidate_reward_sha256": candidate_reward_sha256(
                parent_mode
            ),
            "single_changed_reward_parameter": {
                "name": "ori6_spectral_subspace.weight",
                "parent_value": 0.0,
                "candidate_value": SPECTRAL_SUBSPACE_WEIGHT,
                "parent_bit_exact_at_zero": True,
            },
            "spectral_formula": runtime["formula"],
            "spectral_formula_sha256": SPECTRAL_SUBSPACE_FORMULA_SHA256,
            "spectral_relative_epsilon": (
                SPECTRAL_SUBSPACE_RELATIVE_EPSILON
            ),
            "candidate_optimizer_sha256_by_profile": {
                name: candidate_optimizer_sha256(mode, profile)
                for name, profile in profiles.items()
            },
            "parent_candidate_optimizer_sha256_by_profile": {
                name: candidate_optimizer_sha256(parent_mode, profile)
                for name, profile in profiles.items()
            },
            "optimizer_identity_reused_from_parent": True,
            "candidate_mechanics_sha256_by_task": {
                name: candidate_mechanics_sha256(mode, TASKS[name])
                for name in selected_task_names
            },
            "parent_candidate_mechanics_sha256_by_task": {
                name: candidate_mechanics_sha256(parent_mode, TASKS[name])
                for name in selected_task_names
            },
            "physical_mechanics_identity_reused_from_parent": True,
            "candidate_transition_semantics_sha256": (
                candidate_transition_semantics_sha256(mode)
            ),
            "parent_transition_semantics_sha256": (
                candidate_transition_semantics_sha256(parent_mode)
            ),
            "crossing_transition_semantics_reused_from_parent": True,
            "screen_scope": screen_scope,
            "early_gate_specification": early_gate_specification,
            "parent_RC_reject_authority": str(
                P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT.resolve()
            ),
            "parent_RC_reject_authority_sha256": file_sha256(
                P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
            ),
            "spectral_offline_design_authority": str(
                SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY.resolve()
            ),
            "spectral_offline_design_authority_sha256": file_sha256(
                SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
            ),
            **stage_specific,
        })
    fingerprint = canonical_digest(payload)
    payload["manifest_fingerprint"] = fingerprint
    payload["created_utc"] = utc_now()
    path = ensure_below_root(
        EXPERIMENT_ROOT / "manifests"
        / f"{protocol.name}_{MODE_SLUGS[mode]}_{fingerprint[:12]}.json"
    )
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            prior = json.load(handle)
        comparable = jsonable(prior)
        comparable.pop("created_utc", None)
        new = jsonable(payload)
        new.pop("created_utc", None)
        if comparable != new:
            raise RuntimeError(f"Immutable manifest collision: {path}")
    else:
        atomic_json(path, payload)
    record_stale_manifests(protocol, mode, path, payload)
    return path


def record_stale_manifests(
    protocol: Protocol, mode: str, current_path: Path, current_payload: dict,
) -> None:
    """Keep old immutable manifests, but make their exclusion machine-readable."""

    pattern = f"{protocol.name}_{MODE_SLUGS[mode]}_*.json"
    excluded = []
    for candidate in sorted(current_path.parent.glob(pattern)):
        if candidate.resolve() == current_path.resolve():
            continue
        if candidate.name.endswith("_stale_manifest_exclusions.json"):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            excluded.append({
                "path": str(candidate.resolve()),
                "reason": "unreadable_manifest",
                "error": repr(error),
            })
            continue
        reasons = []
        if payload.get("source_sha256") != current_payload.get("source_sha256"):
            reasons.append("source_sha256_stale")
        if payload.get("candidate_additive_source_sha256") != current_payload.get(
            "candidate_additive_source_sha256"
        ):
            reasons.append("candidate_additive_source_sha256_stale")
        if payload.get(
            "candidate_orchestration_source_sha256"
        ) != current_payload.get("candidate_orchestration_source_sha256"):
            reasons.append("candidate_orchestration_source_sha256_stale")
        if payload.get("model_sha256_by_task") != current_payload.get("model_sha256_by_task"):
            reasons.append("model_sha256_stale")
        if payload.get("random_baseline_authority", {}).get("sha256") != (
            current_payload.get("random_baseline_authority", {}).get("sha256")
        ):
            reasons.append("random_baseline_authority_stale")
        if payload.get("candidate_reward_sha256") != current_payload.get(
            "candidate_reward_sha256"
        ):
            reasons.append("candidate_reward_identity_changed")
        if payload.get("offline_saved_path_ordering_authority_sha256") != (
            current_payload.get("offline_saved_path_ordering_authority_sha256")
        ):
            reasons.append("offline_saved_path_ordering_authority_changed")
        if payload.get("P0_crossing_attractor_authority_sha256") != (
            current_payload.get("P0_crossing_attractor_authority_sha256")
        ):
            reasons.append("P0_crossing_attractor_authority_changed")
        if payload.get("parent_candidate_reject_authority_sha256") != (
            current_payload.get("parent_candidate_reject_authority_sha256")
        ):
            reasons.append("parent_candidate_reject_authority_changed")
        if payload.get("preceding_candidate_reject_authority_sha256") != (
            current_payload.get("preceding_candidate_reject_authority_sha256")
        ):
            reasons.append("preceding_candidate_reject_authority_changed")
        if payload.get("candidate_optimizer_sha256") != current_payload.get(
            "candidate_optimizer_sha256"
        ):
            reasons.append("candidate_optimizer_identity_changed")
        if payload.get("candidate_mechanics_sha256") != current_payload.get(
            "candidate_mechanics_sha256"
        ):
            reasons.append("candidate_mechanics_identity_changed")
        if payload.get("candidate_transition_semantics_sha256") != (
            current_payload.get("candidate_transition_semantics_sha256")
        ):
            reasons.append("candidate_transition_semantics_changed")
        if payload.get("entropy_preflight_authority_sha256") != (
            current_payload.get("entropy_preflight_authority_sha256")
        ):
            reasons.append("entropy_preflight_authority_changed")
        if payload.get("width_feasibility_authority_sha256") != (
            current_payload.get("width_feasibility_authority_sha256")
        ):
            reasons.append("width_feasibility_authority_changed")
        if payload.get("parent_RC_reject_authority_sha256") != (
            current_payload.get("parent_RC_reject_authority_sha256")
        ):
            reasons.append("parent_RC_reject_authority_changed")
        if payload.get("spectral_offline_design_authority_sha256") != (
            current_payload.get("spectral_offline_design_authority_sha256")
        ):
            reasons.append("spectral_offline_design_authority_changed")
        if payload.get("spectral_formula_sha256") != current_payload.get(
            "spectral_formula_sha256"
        ):
            reasons.append("spectral_formula_identity_changed")
        if payload.get("spectral_screen_report_sha256") != current_payload.get(
            "spectral_screen_report_sha256"
        ):
            reasons.append("spectral_screen_report_authority_changed")
        if payload.get("spectral_screen_report_status") != current_payload.get(
            "spectral_screen_report_status"
        ):
            reasons.append("spectral_screen_report_status_changed")
        if (
            payload.get("manifest_fingerprint")
            != current_payload.get("manifest_fingerprint")
            and not reasons
        ):
            reasons.append("job_or_protocol_identity_changed")
        if reasons:
            excluded.append({
                "path": str(candidate.resolve()),
                "manifest_fingerprint": payload.get("manifest_fingerprint"),
                "reasons": reasons,
            })
    index = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "protocol": protocol.name,
        "candidate_mode": mode,
        "current_manifest": str(current_path.resolve()),
        "current_manifest_fingerprint": current_payload["manifest_fingerprint"],
        "excluded_manifests": excluded,
    }
    atomic_json(
        current_path.parent
        / f"{protocol.name}_{MODE_SLUGS[mode]}_stale_manifest_exclusions.json",
        index,
    )


def validate_manifest_execution_binding(
    path: Path, protocol: Protocol, mode: str, jobs: list[dict],
) -> dict:
    """Re-hash immutable manifest authorities immediately before execution."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    content = dict(payload)
    recorded_fingerprint = content.pop("manifest_fingerprint", None)
    content.pop("created_utc", None)
    errors = []
    if recorded_fingerprint != canonical_digest(content):
        errors.append("manifest self-digest mismatch")
    if payload.get("protocol", {}).get("name") != protocol.name:
        errors.append("manifest protocol mismatch")
    if payload.get("candidate_mode") != mode:
        errors.append("manifest candidate mismatch")
    if sorted(row.get("fingerprint") for row in payload.get("jobs", [])) != sorted(
        job["fingerprint"] for job in jobs
    ):
        errors.append("manifest jobs mismatch")
    if mode in (
        P0_CROSS2_MODE, P0_CROSS2_FRONTIER12_MODE,
        P0_CROSS2_ENTROPY3_MODE, P0_CROSS2_WIDTH2_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        if payload.get("candidate_reward_sha256") != candidate_reward_sha256(mode):
            errors.append("manifest reward hash mismatch")
        if mode == P0_CROSS2_MODE:
            if payload.get("single_changed_reward_parameter") != {
                "name": "ori5_crossing_penalty",
                "P0_value": candidate_crossing_penalty(P0_MODE),
                "candidate_value": candidate_crossing_penalty(mode),
            }:
                errors.append("manifest single changed parameter mismatch")
            authorities = (
                (
                    "offline_saved_path_ordering_authority",
                    "offline_saved_path_ordering_authority_sha256",
                    P0_CROSS2_OFFLINE_AUDIT,
                ),
                (
                    "P0_crossing_attractor_authority",
                    "P0_crossing_attractor_authority_sha256",
                    P0_RECLASSIFICATION_REPORT,
                ),
            )
        elif mode == P0_CROSS2_FRONTIER12_MODE:
            if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
                errors.append("manifest parent candidate mismatch")
            if payload.get("parent_candidate_reward_sha256") != (
                candidate_reward_sha256(P0_CROSS2_MODE)
            ):
                errors.append("manifest parent reward hash mismatch")
            if payload.get("single_changed_reward_parameter") != {
                "name": "ori6_spread_frontier_bonus",
                "P0X2_value": candidate_frontier_bonus(P0_CROSS2_MODE),
                "candidate_value": candidate_frontier_bonus(mode),
            }:
                errors.append("manifest single changed parameter mismatch")
            authorities = (
                (
                    "offline_saved_path_ordering_authority",
                    "offline_saved_path_ordering_authority_sha256",
                    P0_CROSS2_FRONTIER12_OFFLINE_AUDIT,
                ),
                (
                    "parent_candidate_reject_authority",
                    "parent_candidate_reject_authority_sha256",
                    P0_CROSS2_REJECT_REPORT,
                ),
            )
        elif mode == P0_CROSS2_ENTROPY3_MODE:
            profile = PROFILES[TASKS["water1"].profile]
            if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
                errors.append("manifest parent candidate mismatch")
            if payload.get("parent_candidate_reward_sha256") != (
                candidate_reward_sha256(P0_CROSS2_MODE)
            ):
                errors.append("manifest parent reward hash mismatch")
            if payload.get("reward_identity_reused_from_parent") is not True:
                errors.append("manifest unchanged reward declaration mismatch")
            if candidate_reward_sha256(mode) != candidate_reward_sha256(
                P0_CROSS2_MODE
            ):
                errors.append("candidate reward differs from P0X2")
            if payload.get("candidate_optimizer_sha256") != (
                candidate_optimizer_sha256(mode, profile)
            ):
                errors.append("manifest optimizer hash mismatch")
            if payload.get("parent_candidate_optimizer_sha256") != (
                candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
            ):
                errors.append("manifest parent optimizer hash mismatch")
            if payload.get("single_changed_training_parameter") != {
                "name": "entropy_coefficient",
                "P0X2_value": candidate_entropy_coefficient(
                    P0_CROSS2_MODE, profile
                ),
                "candidate_value": candidate_entropy_coefficient(mode, profile),
            }:
                errors.append("manifest single changed optimizer mismatch")
            if payload.get("screen_scope") != {
                "stage": "entropy_diagnostic",
                "tasks": ["water1"],
                "seeds": list(PROTOCOLS["entropy_diagnostic"].seeds),
                "episodes": PROTOCOLS["entropy_diagnostic"].episodes,
                "checkpoint_every": PROTOCOLS["entropy_diagnostic"].checkpoint_every,
                "early_gate_checkpoints": [512, 1024],
            }:
                errors.append("manifest screen scope mismatch")
            authorities = (
                (
                    "offline_saved_path_ordering_authority",
                    "offline_saved_path_ordering_authority_sha256",
                    P0_CROSS2_OFFLINE_AUDIT,
                ),
                (
                    "preceding_candidate_reject_authority",
                    "preceding_candidate_reject_authority_sha256",
                    P0_CROSS2_FRONTIER12_REJECT_REPORT,
                ),
                (
                    "entropy_preflight_authority",
                    "entropy_preflight_authority_sha256",
                    P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT,
                ),
            )
        elif mode == P0_CROSS2_WIDTH2_MODE:
            task = TASKS["water1"]
            profile = PROFILES[task.profile]
            if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
                errors.append("manifest parent candidate mismatch")
            if payload.get("parent_candidate_reward_sha256") != (
                candidate_reward_sha256(P0_CROSS2_MODE)
            ):
                errors.append("manifest parent reward hash mismatch")
            if payload.get("reward_identity_reused_from_parent") is not True:
                errors.append("manifest unchanged reward declaration mismatch")
            if candidate_reward_sha256(mode) != candidate_reward_sha256(
                P0_CROSS2_MODE
            ):
                errors.append("candidate reward differs from P0X2")
            if payload.get("candidate_optimizer_sha256") != (
                candidate_optimizer_sha256(mode, profile)
            ):
                errors.append("manifest optimizer hash mismatch")
            if payload.get("parent_candidate_optimizer_sha256") != (
                candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
            ):
                errors.append("manifest parent optimizer hash mismatch")
            if payload.get("optimizer_identity_reused_from_parent") is not True:
                errors.append("manifest unchanged optimizer declaration mismatch")
            if candidate_optimizer_sha256(mode, profile) != (
                candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
            ):
                errors.append("candidate optimizer differs from P0X2")
            if payload.get("candidate_mechanics_sha256") != (
                candidate_mechanics_sha256(mode, task)
            ):
                errors.append("manifest mechanics hash mismatch")
            if payload.get("parent_candidate_mechanics_sha256") != (
                candidate_mechanics_sha256(P0_CROSS2_MODE, task)
            ):
                errors.append("manifest parent mechanics hash mismatch")
            if payload.get("single_changed_mechanics_parameter") != {
                "name": "max_step_width",
                "P0X2_value": candidate_max_step_width(P0_CROSS2_MODE, task),
                "candidate_value": candidate_max_step_width(mode, task),
            }:
                errors.append("manifest single changed mechanics mismatch")
            scope = payload.get("screen_scope", {})
            if any((
                scope.get("stage") != "width_diagnostic",
                scope.get("tasks") != ["water1"],
                scope.get("seeds") != list(PROTOCOLS["width_diagnostic"].seeds),
                scope.get("episodes") != PROTOCOLS["width_diagnostic"].episodes,
                scope.get("checkpoint_every")
                != PROTOCOLS["width_diagnostic"].checkpoint_every,
                scope.get("early_gate_checkpoints") != [512, 1024, 2048],
            )):
                errors.append("manifest screen scope mismatch")
            if payload.get("random_baseline_authority") != {
                "status": "not_used_for_bounded_mechanics_diagnostic",
                "reason": "max_step_width differs from the existing random baseline",
                "new_random_baseline_required_only_if_adopted": True,
            }:
                errors.append("manifest random-baseline scope mismatch")
            authorities = (
                (
                    "offline_saved_path_ordering_authority",
                    "offline_saved_path_ordering_authority_sha256",
                    P0_CROSS2_OFFLINE_AUDIT,
                ),
                (
                    "preceding_candidate_reject_authority",
                    "preceding_candidate_reject_authority_sha256",
                    P0_CROSS2_ENTROPY3_REJECT_REPORT,
                ),
                (
                    "width_feasibility_authority",
                    "width_feasibility_authority_sha256",
                    P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT,
                ),
            )
        elif mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
            parent_mode = P0_CROSS2_REJECTED_CONTINUATION_MODE
            if protocol.name not in (
                "spectral_subspace_diagnostic", "spectral_nonregression",
            ):
                errors.append("manifest spectral protocol is unauthorized")
                protocol_expected = protocol
            else:
                protocol_expected = PROTOCOLS[protocol.name]
            selected_task_names = list(protocol_expected.tasks)
            profiles = {
                TASKS[name].profile: PROFILES[TASKS[name].profile]
                for name in selected_task_names
            }
            runtime = spectral_subspace_runtime_identity()
            expected_optimizer = {
                name: candidate_optimizer_sha256(mode, profile)
                for name, profile in profiles.items()
            }
            parent_optimizer = {
                name: candidate_optimizer_sha256(parent_mode, profile)
                for name, profile in profiles.items()
            }
            expected_mechanics = {
                name: candidate_mechanics_sha256(mode, TASKS[name])
                for name in selected_task_names
            }
            parent_mechanics = {
                name: candidate_mechanics_sha256(parent_mode, TASKS[name])
                for name in selected_task_names
            }
            if protocol.name == "spectral_subspace_diagnostic":
                expected_scope = {
                    "stage": "spectral_subspace_diagnostic",
                    "tasks": selected_task_names,
                    "seeds": list(protocol_expected.seeds),
                    "episodes": protocol_expected.episodes,
                    "checkpoint_every": protocol_expected.checkpoint_every,
                    "early_gate_checkpoints": [512, 1024, 2048],
                    "task_cohorts_are_independent_and_fixed": True,
                    "paired_parent_seeds": [3201, 3202, 3203],
                    "parent_torus_endpoint": 512,
                    "parent_water1_endpoint": 1024,
                }
                expected_gate = spectral_subspace_gate_specification()
            else:
                expected_scope = {
                    "stage": "spectral_nonregression",
                    "tasks": selected_task_names,
                    "seeds": list(protocol_expected.seeds),
                    "episodes": protocol_expected.episodes,
                    "checkpoint_every": protocol_expected.checkpoint_every,
                    "live_early_gate_checkpoints": [512],
                    "implementations": list(protocol_expected.implementations),
                    "jobs_per_task": 6,
                    "total_jobs": 18,
                    "task_cohorts_are_independent_and_fixed": True,
                }
                expected_gate = spectral_nonregression_gate_specification()
            if payload.get("parent_candidate_mode") != parent_mode:
                errors.append("manifest parent candidate mismatch")
            if payload.get("ori6_candidate") != (
                SPECTRAL_SUBSPACE_CANDIDATE_NAME
            ):
                errors.append("manifest spectral candidate mismatch")
            if payload.get("ori6_candidate_schema_version") != (
                SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
            ):
                errors.append("manifest spectral schema mismatch")
            if payload.get("parent_ori6_candidate") != (
                REJECTED_CROSSING_CANDIDATE_NAME
            ):
                errors.append("manifest RC parent identity mismatch")
            if payload.get("candidate_additive_source_sha256") != {
                **rejected_crossing_additive_source_hashes(),
                **spectral_subspace_additive_source_hashes(),
            }:
                errors.append("manifest combined additive source mismatch")
            if payload.get("parent_additive_source_sha256") != (
                rejected_crossing_additive_source_hashes()
            ):
                errors.append("manifest parent additive source mismatch")
            if payload.get("spectral_additive_source_sha256") != (
                spectral_subspace_additive_source_hashes()
            ):
                errors.append("manifest spectral additive source mismatch")
            if payload.get("candidate_orchestration_source_sha256") != (
                rejected_crossing_orchestration_source_hashes()
            ):
                errors.append("manifest orchestration source mismatch")
            if payload.get("source_sha256") != source_hashes():
                errors.append("manifest base source hashes mismatch")
            if payload.get("model_sha256_by_task") != {
                name: model_hashes(task) for name, task in TASKS.items()
            }:
                errors.append("manifest model hashes mismatch")
            if payload.get("parent_candidate_reward_sha256") != (
                candidate_reward_sha256(parent_mode)
            ):
                errors.append("manifest parent reward hash mismatch")
            if payload.get("single_changed_reward_parameter") != {
                "name": "ori6_spectral_subspace.weight",
                "parent_value": 0.0,
                "candidate_value": SPECTRAL_SUBSPACE_WEIGHT,
                "parent_bit_exact_at_zero": True,
            }:
                errors.append("manifest sole spectral reward change mismatch")
            if (
                payload.get("spectral_formula") != runtime["formula"]
                or payload.get("spectral_formula_sha256")
                != SPECTRAL_SUBSPACE_FORMULA_SHA256
                or payload.get("spectral_relative_epsilon")
                != SPECTRAL_SUBSPACE_RELATIVE_EPSILON
            ):
                errors.append("manifest spectral formula identity mismatch")
            if (
                payload.get("candidate_optimizer_sha256_by_profile")
                != expected_optimizer
                or payload.get(
                    "parent_candidate_optimizer_sha256_by_profile"
                ) != parent_optimizer
                or payload.get("optimizer_identity_reused_from_parent")
                is not True
                or expected_optimizer != parent_optimizer
            ):
                errors.append("manifest optimizer identity changed")
            if (
                payload.get("candidate_mechanics_sha256_by_task")
                != expected_mechanics
                or payload.get("parent_candidate_mechanics_sha256_by_task")
                != parent_mechanics
                or payload.get("physical_mechanics_identity_reused_from_parent")
                is not True
                or expected_mechanics != parent_mechanics
            ):
                errors.append("manifest mechanics identity changed")
            if (
                payload.get("candidate_transition_semantics_sha256")
                != candidate_transition_semantics_sha256(mode)
                or payload.get("parent_transition_semantics_sha256")
                != candidate_transition_semantics_sha256(parent_mode)
                or payload.get(
                    "crossing_transition_semantics_reused_from_parent"
                ) is not True
                or candidate_transition_semantics_sha256(mode)
                != candidate_transition_semantics_sha256(parent_mode)
            ):
                errors.append("manifest crossing transition identity changed")
            if payload.get("screen_scope") != expected_scope:
                errors.append("manifest spectral screen scope mismatch")
            if payload.get("early_gate_specification") != expected_gate:
                errors.append("manifest spectral early-gate mismatch")
            if payload.get("random_baseline_authority") != (
                rejected_crossing_random_baseline_disposition()
            ):
                errors.append("manifest random-baseline incompatibility mismatch")
            authorities = [
                (
                    "parent_RC_reject_authority",
                    "parent_RC_reject_authority_sha256",
                    P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT,
                ),
                (
                    "spectral_offline_design_authority",
                    "spectral_offline_design_authority_sha256",
                    SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY,
                ),
            ]
            if protocol.name == "spectral_subspace_diagnostic":
                if payload.get("parent_comparison_scope") != {
                    "same_seeds": [3201, 3202, 3203],
                    "common_prefix_only": True,
                    "performance_comparison_is_diagnostic_not_a_gate": True,
                    "missing_parent_1024_torus_does_not_invalidate": True,
                    "parent_artifacts_are_immutable_reject_evidence": True,
                }:
                    errors.append("manifest parent comparison scope mismatch")
                if payload.get("adoption_requirements") != {
                    "fresh_confirmation_seeds": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "fresh_successful_task_regression": True,
                    "screen_pass_does_not_authorize_full_training": True,
                }:
                    errors.append(
                        "manifest spectral adoption requirements mismatch"
                    )
            else:
                expected_reference = {
                    "implementation": "ori5_reference",
                    "reward_sha256": canonical_digest({
                        "reward_design": ORI5_REWARD_DESIGN,
                        "reward_schema_version": ORI5_REWARD_SCHEMA_VERSION,
                        "effective_parameters": expected_ori5_reward(),
                    }),
                    "mechanics_and_optimizer_profiles_are_exactly_paired": True,
                }
                if payload.get("paired_reference") != expected_reference:
                    errors.append("manifest paired reference identity mismatch")
                if payload.get("promotion_requirements") != {
                    "formal_spectral_report_explicitly_permits_"
                    "preregistered_regression": True,
                    "formal_spectral_report_does_not_authorize_"
                    "adoption_or_full": True,
                    "all_three_successful_task_nonregression_gates_must_pass": True,
                    "fresh_random_baseline_all_five_tasks": True,
                    "regression_pass_is_necessary_but_not_sufficient_for_full": True,
                }:
                    errors.append("manifest promotion requirements mismatch")
                if payload.get("spectral_screen_report_status") != (
                    "bound_regression_permission_authority"
                ):
                    errors.append("manifest spectral screen authority is pending")
                authorities.append((
                    "spectral_screen_report",
                    "spectral_screen_report_sha256",
                    SPECTRAL_SUBSPACE_SCREEN_REPORT,
                ))
                try:
                    validate_spectral_screen_report_authority(
                        SPECTRAL_SUBSPACE_SCREEN_REPORT
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append(f"spectral screen authority invalid: {error}")
        else:
            protocol_expected = PROTOCOLS[
                "crossing_continuation_diagnostic"
            ]
            selected_task_names = list(protocol_expected.tasks)
            profiles = {
                TASKS[name].profile: PROFILES[TASKS[name].profile]
                for name in selected_task_names
            }
            if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
                errors.append("manifest parent candidate mismatch")
            if payload.get("ori6_candidate") != (
                REJECTED_CROSSING_CANDIDATE_NAME
            ):
                errors.append("manifest additive candidate mismatch")
            if payload.get("ori6_candidate_schema_version") != (
                REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            ):
                errors.append("manifest additive candidate schema mismatch")
            if payload.get("candidate_additive_source_sha256") != (
                rejected_crossing_additive_source_hashes()
            ):
                errors.append("manifest additive source hashes mismatch")
            if payload.get("candidate_orchestration_source_sha256") != (
                rejected_crossing_orchestration_source_hashes()
            ):
                errors.append("manifest orchestration source hashes mismatch")
            if payload.get("source_sha256") != source_hashes():
                errors.append("manifest base source hashes mismatch")
            if payload.get("model_sha256_by_task") != {
                name: model_hashes(task) for name, task in TASKS.items()
            }:
                errors.append("manifest model hashes mismatch")
            if payload.get("parent_candidate_reward_sha256") != (
                candidate_reward_sha256(P0_CROSS2_MODE)
            ):
                errors.append("manifest parent reward hash mismatch")
            if payload.get(
                "numeric_reward_hyperparameters_reused_from_parent"
            ) is not True:
                errors.append("manifest numeric reward identity mismatch")
            expected_optimizers = {
                name: candidate_optimizer_sha256(mode, profile)
                for name, profile in profiles.items()
            }
            parent_optimizers = {
                name: candidate_optimizer_sha256(
                    P0_CROSS2_MODE, profile
                )
                for name, profile in profiles.items()
            }
            if payload.get("candidate_optimizer_sha256_by_profile") != (
                expected_optimizers
            ):
                errors.append("manifest optimizer hashes mismatch")
            if payload.get(
                "parent_candidate_optimizer_sha256_by_profile"
            ) != parent_optimizers:
                errors.append("manifest parent optimizer hashes mismatch")
            if (
                payload.get("optimizer_identity_reused_from_parent")
                is not True
                or expected_optimizers != parent_optimizers
            ):
                errors.append("manifest optimizer identity changed")
            expected_mechanics_by_task = {
                name: candidate_mechanics_sha256(mode, TASKS[name])
                for name in selected_task_names
            }
            parent_mechanics_by_task = {
                name: candidate_mechanics_sha256(
                    P0_CROSS2_MODE, TASKS[name]
                )
                for name in selected_task_names
            }
            if payload.get("candidate_mechanics_sha256_by_task") != (
                expected_mechanics_by_task
            ):
                errors.append("manifest physical mechanics hashes mismatch")
            if payload.get(
                "parent_candidate_mechanics_sha256_by_task"
            ) != parent_mechanics_by_task:
                errors.append("manifest parent mechanics hashes mismatch")
            if (
                payload.get("physical_mechanics_identity_reused_from_parent")
                is not True
                or expected_mechanics_by_task != parent_mechanics_by_task
            ):
                errors.append("manifest physical mechanics identity changed")
            if payload.get("candidate_transition_semantics_sha256") != (
                candidate_transition_semantics_sha256(mode)
            ):
                errors.append("manifest transition semantics hash mismatch")
            if payload.get("parent_transition_semantics_sha256") != (
                candidate_transition_semantics_sha256(P0_CROSS2_MODE)
            ):
                errors.append("manifest parent transition hash mismatch")
            if payload.get("single_changed_environment_semantics") != {
                "name": "self_crossing_transition_policy",
                "P0X2_value": "terminate_on_self_crossing",
                "candidate_value": REJECTED_CROSSING_POLICY,
                "P0X2_crossed_candidate_committed": True,
                "candidate_crossed_candidate_committed": False,
                "rejected_attempt_consumes_step": True,
                "last_safe_state_restored": True,
                "past_frontier_and_joint_ledgers_retained": True,
            }:
                errors.append("manifest changed transition declaration mismatch")
            if payload.get("screen_scope") != {
                "stage": "crossing_continuation_diagnostic",
                "tasks": selected_task_names,
                "seeds": list(protocol_expected.seeds),
                "episodes": protocol_expected.episodes,
                "checkpoint_every": protocol_expected.checkpoint_every,
                "early_gate_checkpoints": [512, 1024, 2048],
                "task_cohorts_are_independent_and_fixed": True,
            }:
                errors.append("manifest screen scope mismatch")
            if payload.get("early_gate_specification") != (
                crossing_continuation_gate_specification()
            ):
                errors.append("manifest early-gate specification mismatch")
            if payload.get("parent_saved_path_scope") != (
                "numeric P0X2 reward and collision-free path evidence only; "
                "terminal self-crossing path returns are not reused"
            ):
                errors.append("manifest parent saved-path scope mismatch")
            if payload.get("random_baseline_authority") != (
                rejected_crossing_random_baseline_disposition()
            ):
                errors.append("manifest random-baseline incompatibility mismatch")
            if payload.get("adoption_requirements") != {
                "fresh_random_baseline_all_five_tasks": True,
                "fresh_successful_task_regression": True,
                "screen_pass_does_not_authorize_full_training": True,
            }:
                errors.append("manifest adoption requirements mismatch")
            authorities = (
                (
                    "parent_saved_path_ordering_authority",
                    "parent_saved_path_ordering_authority_sha256",
                    P0_CROSS2_OFFLINE_AUDIT,
                ),
                (
                    "preceding_candidate_reject_authority",
                    "preceding_candidate_reject_authority_sha256",
                    P0_CROSS2_WIDTH2_REJECT_REPORT,
                ),
            )
        for path_field, digest_field, expected_path in authorities:
            recorded_path = payload.get(path_field)
            try:
                matches = (
                    bool(recorded_path)
                    and Path(recorded_path).resolve() == expected_path.resolve()
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"manifest {path_field} mismatch")
            elif not expected_path.is_file():
                errors.append(f"manifest authority missing: {path_field}")
            elif payload.get(digest_field) != file_sha256(expected_path):
                errors.append(f"manifest authority hash mismatch: {path_field}")
    if errors:
        raise RuntimeError(
            "Execution manifest authority rejected: " + "; ".join(errors)
        )
    return payload


def resolve_authority_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else SCRIPT_DIRECTORY / path).resolve()


def validate_P0X2_saved_path_authority(path: Path, reward_hash: str) -> dict:
    if not path.is_file():
        raise RuntimeError("P0X2 saved-path ordering authority is missing")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    if payload.get("candidate_mode") != P0_CROSS2_MODE:
        errors.append("candidate mode mismatch")
    if payload.get("candidate_reward_sha256") != reward_hash:
        errors.append("reward hash mismatch")
    if payload.get("all_required_orderings_pass") is not True:
        errors.append("saved path ordering did not pass")
    orderings = payload.get("orderings")
    if (
        not isinstance(orderings, dict)
        or set(orderings) != set(TASKS)
        or not all(
            isinstance(row, dict) and row.get("pass") is True
            for row in orderings.values()
        )
    ):
        errors.append("per-task saved path orderings did not pass")
    if payload.get("source_sha256") != source_hashes():
        errors.append("source hashes are stale")
    if payload.get("model_sha256_by_task") != {
        name: model_hashes(task) for name, task in TASKS.items()
    }:
        errors.append("model hashes are stale")
    if errors:
        raise RuntimeError(
            "P0X2 saved-path authority rejected: " + "; ".join(errors)
        )
    return payload


def validate_P0X2B12_checkpoint_reject_authority(path: Path) -> dict:
    """Validate the immutable episode-1024 B12 failure and its raw bindings."""

    if not path.is_file():
        raise RuntimeError("P0X2B12 checkpoint reject authority is missing")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    scope = payload.get("scope", {})
    binding = payload.get("historical_binding", {})
    integrity = payload.get("atomic_checkpoint_integrity", {})
    predicates = payload.get("reclassification_predicates", {})
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != "ori6_rescue_atomic_checkpoint_reclassification":
        errors.append("authority type mismatch")
    if payload.get("candidate_mode") != P0_CROSS2_FRONTIER12_MODE:
        errors.append("candidate mode mismatch")
    if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
        errors.append("parent candidate mismatch")
    if payload.get("task") != "water1":
        errors.append("task mismatch")
    if payload.get("decision_endpoint_episode") != 1024:
        errors.append("checkpoint endpoint mismatch")
    if payload.get("decision") != "reject_candidate":
        errors.append("reject decision is absent")
    if not (
        scope.get("is_atomic_checkpoint_authority") is True
        and scope.get("is_post_checkpoint_reclassification") is True
        and scope.get("is_full_2048_rescue_report") is False
        and scope.get("heldout_evaluation_available") is False
        and scope.get("raw_artifacts_modified") is False
    ):
        errors.append("checkpoint-only scope mismatch")
    if binding.get("candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_FRONTIER12_MODE
    ):
        errors.append("candidate reward hash mismatch")
    if binding.get("parent_candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_MODE
    ):
        errors.append("parent reward hash mismatch")
    if binding.get("source_sha256") != source_hashes():
        errors.append("historical source hashes are stale")
    if binding.get("model_sha256") != model_hashes(TASKS["water1"]):
        errors.append("historical water1 model hashes are stale")
    required_predicates = (
        "atomic_checkpoint_integrity_pass",
        "reward_integrity_pass",
        "return_order_pass",
        "all_seed_zero_strict_success",
        "all_seed_tail_median_spread_error_at_least_0_35",
        "tail_good_spread_endpoint_count_at_most_1_of_768",
        "good_spread_endpoint_rate_drop_first_to_second_half_at_least_0_90",
        "reject_candidate",
    )
    if not all(predicates.get(name) is True for name in required_predicates):
        errors.append("formal reject predicates are incomplete")
    if not (
        integrity.get("all_three_valid") is True
        and integrity.get("all_three_config_match_manifest") is True
        and integrity.get("all_three_checkpoint_episode") == 1024
        and integrity.get("all_three_training_rows") == 1024
        and integrity.get("all_three_update_rows") == 128
        and integrity.get("temporary_or_partial_file_count") == 0
    ):
        errors.append("atomic checkpoint aggregate mismatch")

    manifest = None
    manifest_path = None
    try:
        manifest_path = resolve_authority_path(binding.get("manifest_path", ""))
    except (OSError, RuntimeError, TypeError, ValueError):
        errors.append("historical manifest path is invalid")
    if manifest_path is not None:
        expected_manifest_directory = (EXPERIMENT_ROOT / "manifests").resolve()
        if manifest_path.parent != expected_manifest_directory:
            errors.append("historical manifest path is outside the manifest directory")
        elif not manifest_path.is_file():
            errors.append("historical manifest is missing")
        elif binding.get("manifest_sha256") != file_sha256(manifest_path):
            errors.append("historical manifest hash mismatch")
        else:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("manifest_fingerprint") != binding.get(
                "manifest_fingerprint"
            ):
                errors.append("historical manifest fingerprint mismatch")
            if manifest.get("candidate_mode") != P0_CROSS2_FRONTIER12_MODE:
                errors.append("historical manifest candidate mismatch")
            if manifest.get("source_sha256") != binding.get("source_sha256"):
                errors.append("historical manifest source binding mismatch")

    expected_jobs = build_jobs(
        PROTOCOLS["rescue"], P0_CROSS2_FRONTIER12_MODE,
        list(PROTOCOLS["rescue"].tasks),
    )
    if manifest_path is not None and manifest is not None:
        try:
            validate_manifest_execution_binding(
                manifest_path, PROTOCOLS["rescue"],
                P0_CROSS2_FRONTIER12_MODE, expected_jobs,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"historical manifest authority rejected: {error}")

    expected_water_jobs = {
        int(job["seed"]): job for job in expected_jobs
        if job["task"]["name"] == "water1" and job["implementation"] == "ori6"
    }
    per_seed = integrity.get("per_seed")
    if not isinstance(per_seed, list) or {
        row.get("seed") for row in per_seed if isinstance(row, dict)
    } != set(expected_water_jobs):
        errors.append("atomic checkpoint seed set mismatch")
    else:
        for row in per_seed:
            seed = int(row["seed"])
            job = expected_water_jobs[seed]
            try:
                artifact = resolve_authority_path(row.get("artifact_path", ""))
            except (OSError, RuntimeError, TypeError, ValueError):
                errors.append(f"seed{seed} artifact path is invalid")
                continue
            if artifact != Path(job["output_directory"]).resolve():
                errors.append(f"seed{seed} artifact path mismatch")
                continue
            for field, name in (
                ("checkpoint_sha256", "checkpoint.pt"),
                ("training_history_sha256", "training_history.csv"),
                ("update_history_sha256", "update_history.csv"),
                ("run_config_sha256", "run_config.json"),
            ):
                target = artifact / name
                if not target.is_file() or row.get(field) != file_sha256(target):
                    errors.append(f"seed{seed} {name} hash mismatch")
            if not config_matches_job(artifact, job):
                errors.append(f"seed{seed} run config does not match manifest")
            validation = validate_artifact(job, artifact, endpoint=1024)
            if validation.get("valid") is not True:
                errors.append(f"seed{seed} atomic artifact validation failed")
            if checkpoint_episode(artifact) != 1024:
                errors.append(f"seed{seed} checkpoint episode mismatch")

    if errors:
        raise RuntimeError(
            "P0X2B12 checkpoint reject authority rejected: "
            + "; ".join(errors)
        )
    return payload


def validate_P0X2E3_preflight_authority(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError("P0X2E3 preflight authority is missing")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    identity = payload.get("execution_identity", {})
    stage2 = payload.get("ori5_stage2_prior", {})
    entropy_prior = payload.get("P0X2_policy_entropy_prior", {})
    saved_path = payload.get("saved_path_reward_authority", {})
    B12 = payload.get("preceding_B12_reject_authority", {})
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("status") != "preflight_complete_no_training":
        errors.append("preflight status mismatch")
    if payload.get("candidate_mode") != P0_CROSS2_ENTROPY3_MODE:
        errors.append("candidate mode mismatch")
    if payload.get("candidate_mode_slug") != "P0X2E3":
        errors.append("candidate slug mismatch")
    if payload.get("recommendation") != "bounded_1024_diagnostic_only_negative_prior":
        errors.append("negative-prior recommendation mismatch")
    if payload.get("training_launched") is not False:
        errors.append("preflight claims training was launched")
    if payload.get("preflight_pass") is not True:
        errors.append("preflight did not pass")
    if payload.get("screen_scope") != {
        "task": "water1", "seeds": [3101, 3102, 3103],
        "episodes": 1024, "checkpoints": [512, 1024],
    }:
        errors.append("screen scope mismatch")
    if not (
        identity.get("parent_candidate_mode") == P0_CROSS2_MODE
        and identity.get("candidate_mode") == P0_CROSS2_ENTROPY3_MODE
        and identity.get("parent_reward_sha256")
        == candidate_reward_sha256(P0_CROSS2_MODE)
        and identity.get("candidate_reward_sha256")
        == candidate_reward_sha256(P0_CROSS2_ENTROPY3_MODE)
        and identity.get("reward_hash_unchanged") is True
        and identity.get("execution_difference_is_entropy_only") is True
        and identity.get("expected_config_differences") == [{
            "path": "entropy_coefficient",
            "P0X2_value": 1.0e-3,
            "candidate_value": 3.0e-3,
        }]
        and identity.get("duplicate_long_flags") == []
    ):
        errors.append("entropy-only execution identity mismatch")
    command_differences = identity.get("command_differences")
    if not (
        isinstance(command_differences, list)
        and len(command_differences) == 1
        and command_differences[0].get("flag") == "--entropy-coefficient"
        and float(command_differences[0].get("P0X2_value", math.nan)) == 1.0e-3
        and float(command_differences[0].get("candidate_value", math.nan)) == 3.0e-3
    ):
        errors.append("entropy-only command difference mismatch")
    if payload.get("source_sha256") != source_hashes():
        errors.append("source hashes are stale")
    if payload.get("model_sha256_by_task") != {
        name: model_hashes(task) for name, task in TASKS.items()
    }:
        errors.append("model hashes are stale")

    for label, authority, expected in (
        ("saved path", saved_path, P0_CROSS2_OFFLINE_AUDIT),
        ("B12 reject", B12, P0_CROSS2_FRONTIER12_REJECT_REPORT),
    ):
        try:
            recorded = resolve_authority_path(authority.get("path", ""))
        except (OSError, RuntimeError, TypeError, ValueError):
            recorded = None
        if recorded != expected.resolve():
            errors.append(f"{label} authority path mismatch")
        elif not expected.is_file() or authority.get("sha256") != file_sha256(expected):
            errors.append(f"{label} authority hash mismatch")
    if saved_path.get("candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_ENTROPY3_MODE
    ):
        errors.append("saved-path reward identity mismatch")

    stage2_path = None
    try:
        stage2_path = resolve_authority_path(stage2.get("authority_path", ""))
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    expected_stage2 = (
        SCRIPT_DIRECTORY / "experiments_ori5" / "analysis"
        / "stage2_run_summary.csv"
    ).resolve()
    if stage2_path != expected_stage2:
        errors.append("ori5 Stage2 authority path mismatch")
    elif not expected_stage2.is_file() or stage2.get(
        "authority_sha256"
    ) != file_sha256(expected_stage2):
        errors.append("ori5 Stage2 authority hash mismatch")
    profiles = stage2.get("profiles", {})
    if not (
        profiles.get("B", {}).get("entropy_coefficient") == 1.0e-3
        and profiles.get("B", {}).get("training_successes") == 1
        and profiles.get("F", {}).get("entropy_coefficient") == 3.0e-3
        and profiles.get("F", {}).get("training_successes") == 0
        and profiles.get("E", {}).get("entropy_coefficient") == 1.0e-3
        and profiles.get("E", {}).get("batch_size") == 4
    ):
        errors.append("ori5 Stage2 negative prior mismatch")

    entropy_rows = entropy_prior.get("per_seed")
    if not isinstance(entropy_rows, dict) or set(entropy_rows) != {
        "3101", "3102", "3103"
    }:
        errors.append("P0X2 entropy prior seed set mismatch")
    else:
        for seed, row in entropy_rows.items():
            expected = (
                EXPERIMENT_ROOT / "screen" / "rescue" / "P0X2"
                / "water1" / "B" / f"seed{seed}" / "update_history.csv"
            ).resolve()
            try:
                recorded = resolve_authority_path(row.get("path", ""))
            except (OSError, RuntimeError, TypeError, ValueError):
                recorded = None
            if recorded != expected:
                errors.append(f"P0X2 seed{seed} entropy authority path mismatch")
            elif not expected.is_file() or row.get("sha256") != file_sha256(expected):
                errors.append(f"P0X2 seed{seed} entropy authority hash mismatch")
            if row.get("update_count", 0) < 64 or row.get("minimum", 0.0) < 1.0:
                errors.append(f"P0X2 seed{seed} entropy-collapse claim mismatch")
    if errors:
        raise RuntimeError(
            "P0X2E3 preflight authority rejected: " + "; ".join(errors)
        )
    return payload


def validate_P0X2E3_reject_authority(path: Path) -> dict:
    """Validate the immutable atomic episode-1024 E3 rejection evidence."""

    if not path.is_file():
        raise RuntimeError("P0X2E3 reject authority is missing")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    scope = payload.get("scope", {})
    binding = payload.get("historical_binding", {})
    integrity = payload.get("atomic_checkpoint_integrity", {})
    gate = payload.get("preregistered_gate", {}).get("checkpoint_1024", {})
    formal = gate.get("formal_entropy_diagnostic_gate", {})
    reward = payload.get("reward_integrity", {})
    orientation = payload.get("orientation_inferred_plane_switch_proxy", {})
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != (
        "ori6_entropy_diagnostic_atomic_checkpoint_reject"
    ):
        errors.append("authority type mismatch")
    if payload.get("candidate_mode") != P0_CROSS2_ENTROPY3_MODE:
        errors.append("candidate mode mismatch")
    if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
        errors.append("parent candidate mismatch")
    if payload.get("candidate_mode_slug") != "P0X2E3":
        errors.append("candidate slug mismatch")
    if payload.get("task") != "water1":
        errors.append("task mismatch")
    if payload.get("decision_endpoint_episode") != 1024:
        errors.append("decision endpoint mismatch")
    if payload.get("decision") != "reject_candidate":
        errors.append("reject decision is absent")
    if payload.get("decision_reason") != "wide_spread_zero_success_reject":
        errors.append("reject reason mismatch")
    if not (
        scope.get("protocol") == "entropy_diagnostic"
        and scope.get("seeds") == [3101, 3102, 3103]
        and scope.get("episodes") == 1024
        and scope.get("checkpoint_every") == 512
        and scope.get("is_preregistered_gate_decision") is True
        and scope.get("is_full_2048_rescue_report") is False
        and scope.get("all_seed_heldout_evaluation_available") is False
        and scope.get("raw_artifacts_modified") is False
    ):
        errors.append("diagnostic scope mismatch")
    profile = PROFILES[TASKS["water1"].profile]
    if binding.get("candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_ENTROPY3_MODE
    ):
        errors.append("candidate reward hash mismatch")
    if binding.get("parent_candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_MODE
    ):
        errors.append("parent reward hash mismatch")
    if binding.get("reward_identity_reused_from_parent") is not True:
        errors.append("unchanged reward declaration mismatch")
    if binding.get("candidate_optimizer_sha256") != candidate_optimizer_sha256(
        P0_CROSS2_ENTROPY3_MODE, profile
    ):
        errors.append("candidate optimizer hash mismatch")
    if binding.get("parent_candidate_optimizer_sha256") != (
        candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
    ):
        errors.append("parent optimizer hash mismatch")
    if binding.get("single_changed_training_parameter") != {
        "name": "entropy_coefficient",
        "P0X2_value": 0.001,
        "candidate_value": 0.003,
    }:
        errors.append("entropy-only change declaration mismatch")
    if binding.get("source_sha256") != source_hashes():
        errors.append("historical core source hashes are stale")
    if binding.get("water1_model_sha256") != model_hashes(TASKS["water1"]):
        errors.append("historical water1 model hashes are stale")
    if not (
        integrity.get("all_three_valid_at_episode_1024") is True
        and integrity.get("all_three_config_match_manifest") is True
        and integrity.get("all_three_training_rows") == 1024
        and integrity.get("all_three_update_rows") == 128
    ):
        errors.append("atomic checkpoint aggregate mismatch")
    if not (
        gate.get("stop") is True
        and gate.get("reason") == "wide_spread_zero_success_reject"
        and formal.get("pass") is False
        and formal.get("reject") is True
        and formal.get("wide_spread_zero_success_seed_count") >= 2
        and formal.get("reward_integrity_pass") is True
    ):
        errors.append("formal entropy reject gate mismatch")
    if not (
        reward.get("training_episode_count") == 3072
        and reward.get("success_count") == 0
        and reward.get("nonfinite_reward_component_count") == 0
        and reward.get("maximum_abs_episode_reward_decomposition_error", math.inf)
        <= 1.0e-10
        and reward.get("maximum_abs_step_reward_decomposition_error", math.inf)
        <= 1.0e-10
    ):
        errors.append("reward integrity mismatch")
    if not (
        orientation.get("tail256_proxy_positive_seed_count") == 1
        and orientation.get("tail256_proxy_episode_count_by_seed")
        == {"3101": 2, "3102": 0, "3103": 0}
    ):
        errors.append("orientation proxy evidence mismatch")

    bound_files = (
        ("manifest_path", "manifest_sha256", EXPERIMENT_ROOT / "manifests"
         / "entropy_diagnostic_P0X2E3_9b65ea2ce286.json"),
        ("preflight_path", "preflight_sha256", P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT),
        ("preceding_B12_reject_path", "preceding_B12_reject_sha256",
         P0_CROSS2_FRONTIER12_REJECT_REPORT),
        ("checkpoint_512_gate_path", "checkpoint_512_gate_sha256",
         EXPERIMENT_ROOT / "early_gates" / "entropy_diagnostic" / "P0X2E3"
         / "water1" / "checkpoint_512.json"),
        ("checkpoint_1024_gate_path", "checkpoint_1024_gate_sha256",
         EXPERIMENT_ROOT / "early_gates" / "entropy_diagnostic" / "P0X2E3"
         / "water1" / "checkpoint_1024.json"),
    )
    historical_manifest = None
    historical_manifest_path = None
    for path_field, digest_field, expected in bound_files:
        try:
            recorded = resolve_authority_path(binding.get(path_field, ""))
        except (OSError, RuntimeError, TypeError, ValueError):
            errors.append(f"{path_field} is invalid")
            continue
        if recorded != expected.resolve():
            errors.append(f"{path_field} mismatch")
        elif not expected.is_file() or binding.get(digest_field) != file_sha256(expected):
            errors.append(f"{path_field} hash mismatch")
        elif path_field == "manifest_path":
            historical_manifest_path = expected.resolve()
            with open(expected, encoding="utf-8") as handle:
                historical_manifest = json.load(handle)
    if historical_manifest is not None:
        if historical_manifest.get("manifest_fingerprint") != binding.get(
            "manifest_fingerprint"
        ):
            errors.append("historical manifest fingerprint mismatch")
        expected_jobs = build_jobs(
            PROTOCOLS["entropy_diagnostic"], P0_CROSS2_ENTROPY3_MODE,
            ["water1"],
        )
        try:
            validate_manifest_execution_binding(
                historical_manifest_path, PROTOCOLS["entropy_diagnostic"],
                P0_CROSS2_ENTROPY3_MODE, expected_jobs,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"historical manifest authority rejected: {error}")
    else:
        expected_jobs = []

    jobs_by_seed = {int(job["seed"]): job for job in expected_jobs}
    per_seed = integrity.get("per_seed")
    if not isinstance(per_seed, list) or {
        row.get("seed") for row in per_seed if isinstance(row, dict)
    } != {3101, 3102, 3103}:
        errors.append("atomic checkpoint seed set mismatch")
    elif jobs_by_seed:
        for row in per_seed:
            seed = int(row["seed"])
            job = jobs_by_seed[seed]
            try:
                artifact = resolve_authority_path(row.get("artifact_path", ""))
            except (OSError, RuntimeError, TypeError, ValueError):
                errors.append(f"seed{seed} artifact path is invalid")
                continue
            if artifact != Path(job["output_directory"]).resolve():
                errors.append(f"seed{seed} artifact path mismatch")
                continue
            for field, filename in (
                ("checkpoint_sha256", "checkpoint.pt"),
                ("training_history_sha256", "training_history.csv"),
                ("update_history_sha256", "update_history.csv"),
                ("run_config_sha256", "run_config.json"),
            ):
                target = artifact / filename
                if not target.is_file() or row.get(field) != file_sha256(target):
                    errors.append(f"seed{seed} {filename} hash mismatch")
            if not config_matches_job(artifact, job):
                errors.append(f"seed{seed} run config does not match manifest")
            if validate_artifact(job, artifact, endpoint=1024).get("valid") is not True:
                errors.append(f"seed{seed} atomic artifact validation failed")
            if checkpoint_episode(artifact) != 1024:
                errors.append(f"seed{seed} checkpoint episode mismatch")
    if errors:
        raise RuntimeError(
            "P0X2E3 reject authority rejected: " + "; ".join(errors)
        )
    return payload


def validate_P0X2W2_feasibility_authority(path: Path) -> dict:
    """Validate the h=0.2 policy-representable saved-success replay."""

    if not path.is_file():
        raise RuntimeError("P0X2W2 width feasibility authority is missing")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    replay = payload.get("subdivided_replay", {})
    ordering = payload.get("return_ordering", {})
    gate = payload.get("feasibility_gate", {})
    task = TASKS["water1"]
    profile = PROFILES[task.profile]
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != (
        "ori6_water1_subdivided_mechanics_feasibility"
    ):
        errors.append("authority type mismatch")
    if payload.get("candidate_mode_slug") != "P0X2W2":
        errors.append("candidate slug mismatch")
    if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
        errors.append("parent candidate mismatch")
    if payload.get("task") != "water1":
        errors.append("task mismatch")
    if payload.get("single_changed_mechanics_parameter") != {
        "name": "max_step_width", "P0X2_value": 0.3, "candidate_value": 0.2,
    }:
        errors.append("single mechanics change mismatch")
    if not (
        payload.get("reward_identity_reused_from_parent") is True
        and payload.get("candidate_reward_sha256")
        == candidate_reward_sha256(P0_CROSS2_MODE)
        and payload.get("parent_candidate_reward_sha256")
        == candidate_reward_sha256(P0_CROSS2_MODE)
    ):
        errors.append("reward identity mismatch")
    if not (
        payload.get("optimizer_identity_reused_from_parent") is True
        and payload.get("candidate_optimizer_sha256")
        == candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
        and payload.get("parent_candidate_optimizer_sha256")
        == candidate_optimizer_sha256(P0_CROSS2_MODE, profile)
    ):
        errors.append("optimizer identity mismatch")
    if payload.get("source_sha256") != source_hashes():
        errors.append("source hashes are stale")
    if payload.get("model_sha256_by_task") != {
        name: model_hashes(item) for name, item in TASKS.items()
    }:
        errors.append("model hashes are stale")
    if not (
        payload.get("all_required_checks_pass") is True
        and gate.get("pass") is True
        and gate.get("strict_success") is True
        and gate.get("collision_free") is True
        and gate.get("correction_failure_free") is True
        and gate.get("transition_count_le_400") is True
        and gate.get("all_requested_widths_le_0_2") is True
        and gate.get("reward_ordering_unchanged") is True
        and gate.get("reward_integrity") is True
    ):
        errors.append("feasibility gate mismatch")
    if not (
        replay.get("success") is True
        and replay.get("termination_reason") == "success"
        and 0 < replay.get("transition_count", 0) <= 400
        and replay.get("horizon_limit") == 400
        and replay.get("maximum_absolute_requested_width", math.inf)
        <= 0.2 + 1.0e-12
        and replay.get("self_crossing_count") == 0
        and replay.get("correction_failure_count") == 0
        and replay.get("all_after_states_valid") is True
        and replay.get("final_flatness_ratio", math.inf) <= 0.001
        and replay.get("final_spread_error", math.inf) <= 0.05
        and replay.get("step_reward_identity_max_abs_error", math.inf) <= 1.0e-10
        and replay.get("maximum_mode_reconstruction_error", math.inf) <= 1.0e-10
    ):
        errors.append("subdivided mechanics replay mismatch")
    environment = replay.get("environment", {})
    if not (
        environment.get("max_steps") == 400
        and values_equal(environment.get("max_step_width"), 0.2)
        and values_equal(environment.get("target_spread"), 1.0)
        and values_equal(environment.get("crossing_tolerance"), 1.0e-10)
    ):
        errors.append("replay environment mismatch")
    transitions = replay.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != replay.get(
        "transition_count"
    ):
        errors.append("transition ledger length mismatch")
    elif not all(
        abs(float(row.get("requested_width", math.inf))) <= 0.2 + 1.0e-12
        and row.get("after_state_valid") == 1
        and row.get("self_crossing") == 0
        and row.get("correction_failed") == 0
        for row in transitions
    ):
        errors.append("transition ledger mechanics mismatch")
    returns = ordering.get("P0X2_returns", {})
    required = ["subdivided_ideal_success", "near_cross", "wide_wrong"]
    if not (
        ordering.get("required_order") == required
        and ordering.get("pass") is True
        and returns.get(required[0], -math.inf) > returns.get(required[1], math.inf)
        > returns.get(required[2], math.inf)
        and ordering.get("parent_water1_pass") is True
        and ordering.get("other_four_tasks_mechanics_unchanged_by_water1_only_screen")
        is True
    ):
        errors.append("saved-path return ordering mismatch")
    recorded_parent = ordering.get(
        "unchanged_reward_all_five_task_ordering_authority"
    )
    try:
        parent_path = Path(recorded_parent).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        parent_path = None
    if parent_path != P0_CROSS2_OFFLINE_AUDIT.resolve():
        errors.append("parent saved-path authority path mismatch")
    elif ordering.get(
        "unchanged_reward_all_five_task_ordering_authority_sha256"
    ) != file_sha256(P0_CROSS2_OFFLINE_AUDIT):
        errors.append("parent saved-path authority hash mismatch")
    else:
        try:
            validate_P0X2_saved_path_authority(
                P0_CROSS2_OFFLINE_AUDIT,
                candidate_reward_sha256(P0_CROSS2_MODE),
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"parent saved-path authority rejected: {error}")
    source = payload.get("source_saved_success", {})
    archive = (
        SCRIPT_DIRECTORY / "experiments_3d" / "feasibility"
        / "waterbomb_spread1_collision_free_path"
    )
    raw_files = (
        ("actions_path", "actions_sha256", archive / "actions.json"),
        ("summary_path", "summary_sha256", archive / "summary.json"),
    )
    for path_field, digest_field, expected in raw_files:
        try:
            recorded = Path(source.get(path_field, "")).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            recorded = None
        if recorded != expected.resolve():
            errors.append(f"saved success {path_field} mismatch")
        elif not expected.is_file() or source.get(digest_field) != file_sha256(expected):
            errors.append(f"saved success {path_field} hash mismatch")
    step_hashes = source.get("step_file_sha256", {})
    if set(step_hashes) != {f"step{index}.dat" for index in range(21)}:
        errors.append("saved success step-file set mismatch")
    else:
        for name, digest in step_hashes.items():
            target = archive / name
            if not target.is_file() or digest != file_sha256(target):
                errors.append(f"saved success {name} hash mismatch")
    if errors:
        raise RuntimeError(
            "P0X2W2 feasibility authority rejected: " + "; ".join(errors)
        )
    return payload


def validate_P0X2W2_reject_authority(path: Path) -> dict:
    """Validate immutable W2 episode-1024 rejection and every raw binding."""

    if not path.is_file():
        raise RuntimeError("P0X2W2 reject authority is missing")
    if file_sha256(path) != P0_CROSS2_WIDTH2_REJECT_REPORT_SHA256:
        raise RuntimeError("P0X2W2 reject authority fixed SHA mismatch")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    content = dict(payload)
    recorded_fingerprint = content.pop("authority_fingerprint", None)
    if recorded_fingerprint != canonical_digest(content):
        errors.append("authority self-digest mismatch")
    scope = payload.get("scope", {})
    binding = payload.get("historical_binding", {})
    integrity = payload.get("atomic_checkpoint_integrity", {})
    gate = payload.get("preregistered_gate", {}).get(
        "checkpoint_1024", {}
    )
    formal = gate.get("formal_width_diagnostic_gate", {})
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != (
        "ori6_width_diagnostic_atomic_checkpoint_reject"
    ):
        errors.append("authority type mismatch")
    if payload.get("candidate_mode") != P0_CROSS2_WIDTH2_MODE:
        errors.append("candidate mode mismatch")
    if payload.get("candidate_mode_slug") != "P0X2W2":
        errors.append("candidate slug mismatch")
    if payload.get("parent_candidate_mode") != P0_CROSS2_MODE:
        errors.append("parent candidate mismatch")
    if payload.get("task") != "water1":
        errors.append("task mismatch")
    if payload.get("decision_endpoint_episode") != 1024:
        errors.append("decision endpoint mismatch")
    if payload.get("decision") != "reject_candidate":
        errors.append("reject decision absent")
    if payload.get("decision_reason") != (
        "width_diagnostic_wide_spread_reject"
    ):
        errors.append("reject reason mismatch")
    if not (
        scope.get("protocol") == "width_diagnostic"
        and scope.get("historically_preregistered_episode_cap") == 2048
        and scope.get("actual_atomic_decision_endpoint") == 1024
        and scope.get("checkpoint_every") == 512
        and scope.get("seeds") == [3101, 3102, 3103]
        and scope.get("tasks") == ["water1"]
        and scope.get("is_atomic_checkpoint_authority") is True
        and scope.get("episode_2048_claimed") is False
        and scope.get("heldout_claimed") is False
        and scope.get("random_baseline_claimed") is False
    ):
        errors.append("authority scope mismatch")
    if not (
        gate.get("stop") is True
        and gate.get("reason") == "width_diagnostic_wide_spread_reject"
        and gate.get("checkpoint_episode") == 1024
        and formal.get("pass") is False
        and formal.get("reject") is True
        and formal.get("reward_integrity_pass") is True
        and formal.get(
            "tail_exact_median_spread_error_ge_0_30_seed_count"
        ) == 2
        and formal.get("tail_plane_switch_proxy_positive_seed_count") == 0
        and formal.get(
            "tail_valid_F_le_0_1_e_le_0_05_or_strict_seed_count"
        ) == 0
    ):
        errors.append("formal checkpoint reject mismatch")
    pooled = payload.get("reward_integrity", {}).get("pooled", {})
    if not (
        pooled.get("pass") is True
        and pooled.get("episode_count") == 3072
        and pooled.get("termination_counts", {}).get("success") == 0
        and pooled.get("nonfinite_reward_component_count") == 0
        and pooled.get(
            "maximum_abs_saved_episode_reward_decomposition_error",
            math.inf,
        ) <= 1.0e-10
        and pooled.get(
            "maximum_abs_recomputed_episode_reward_decomposition_error",
            math.inf,
        ) <= 1.0e-10
        and pooled.get(
            "maximum_abs_step_reward_decomposition_error", math.inf
        ) <= 1.0e-10
    ):
        errors.append("pooled reward integrity mismatch")

    historical_manifest = (
        SCRIPT_DIRECTORY
        / "experiments_ori6/manifests/"
        "width_diagnostic_P0X2W2_d1dc9b798e14.json"
    ).resolve()
    gate512 = (
        EXPERIMENT_ROOT
        / "early_gates/width_diagnostic/P0X2W2/water1/"
        "checkpoint_512.json"
    ).resolve()
    gate1024 = (
        EXPERIMENT_ROOT
        / "early_gates/width_diagnostic/P0X2W2/water1/"
        "checkpoint_1024.json"
    ).resolve()
    bound_files = (
        ("manifest_path", "manifest_sha256", historical_manifest),
        ("checkpoint_512_gate_path", "checkpoint_512_gate_sha256", gate512),
        ("checkpoint_1024_gate_path", "checkpoint_1024_gate_sha256", gate1024),
        (
            "offline_saved_path_ordering_authority_path",
            "offline_saved_path_ordering_authority_sha256",
            P0_CROSS2_OFFLINE_AUDIT.resolve(),
        ),
        (
            "preceding_E3_reject_authority_path",
            "preceding_E3_reject_authority_sha256",
            P0_CROSS2_ENTROPY3_REJECT_REPORT.resolve(),
        ),
        (
            "width_feasibility_authority_path",
            "width_feasibility_authority_sha256",
            P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT.resolve(),
        ),
    )
    for path_field, digest_field, expected in bound_files:
        try:
            recorded = resolve_authority_path(binding.get(path_field, ""))
        except (OSError, RuntimeError, TypeError, ValueError):
            recorded = None
        if recorded != expected:
            errors.append(f"historical binding path mismatch: {path_field}")
        elif not expected.is_file() or binding.get(digest_field) != file_sha256(
            expected
        ):
            errors.append(f"historical binding SHA mismatch: {path_field}")

    historical_payload = None
    if historical_manifest.is_file():
        with open(historical_manifest, encoding="utf-8") as handle:
            historical_payload = json.load(handle)
        historical_content = dict(historical_payload)
        manifest_fingerprint = historical_content.pop(
            "manifest_fingerprint", None
        )
        historical_content.pop("created_utc", None)
        if manifest_fingerprint != canonical_digest(historical_content):
            errors.append("historical manifest self-digest mismatch")
        if binding.get("manifest_fingerprint") != manifest_fingerprint:
            errors.append("historical manifest fingerprint mismatch")
        if binding.get("source_sha256") != historical_payload.get(
            "source_sha256"
        ):
            errors.append("historical source binding mismatch")
        if binding.get("model_sha256_by_task") != historical_payload.get(
            "model_sha256_by_task"
        ):
            errors.append("historical model binding mismatch")
        for field in (
            "candidate_reward_sha256",
            "parent_candidate_reward_sha256",
            "candidate_optimizer_sha256",
            "parent_candidate_optimizer_sha256",
            "candidate_mechanics_sha256",
            "parent_candidate_mechanics_sha256",
            "single_changed_mechanics_parameter",
        ):
            if binding.get(field) != historical_payload.get(field):
                errors.append(f"historical manifest field mismatch: {field}")
    else:
        errors.append("historical manifest missing")

    per_seed = integrity.get("per_seed")
    if (
        not isinstance(per_seed, list)
        or {row.get("seed") for row in per_seed if isinstance(row, dict)}
        != {3101, 3102, 3103}
    ):
        errors.append("atomic checkpoint seed set mismatch")
    elif historical_payload is not None:
        jobs_by_seed = {
            int(job["seed"]): job for job in historical_payload["jobs"]
        }
        for row in per_seed:
            seed = int(row["seed"])
            job = jobs_by_seed.get(seed)
            if job is None:
                errors.append(f"seed{seed} historical job missing")
                continue
            try:
                artifact = resolve_authority_path(row["artifact_path"])
                recorded_status = resolve_authority_path(
                    row["runner_status_path"]
                )
            except (OSError, RuntimeError, KeyError, TypeError, ValueError):
                errors.append(f"seed{seed} artifact/status path invalid")
                continue
            if artifact != Path(job["output_directory"]).resolve():
                errors.append(f"seed{seed} artifact path mismatch")
                continue
            if (
                not recorded_status.is_file()
                or row.get("runner_status_sha256")
                != file_sha256(recorded_status)
            ):
                errors.append(f"seed{seed} runner status SHA mismatch")
            inventory = row.get("artifact_inventory", {})
            for filename in (
                "checkpoint.pt", "training_history.csv",
                "update_history.csv", "run_config.json",
                "best_policy.pt", "best_policy_metrics.json",
                "deterministic_evaluation_history.csv",
                "stochastic_evaluation_history.csv",
            ):
                target = artifact / filename
                recorded = inventory.get(filename, {})
                if (
                    not target.is_file()
                    or recorded.get("sha256") != file_sha256(target)
                    or recorded.get("size_bytes") != target.stat().st_size
                ):
                    errors.append(f"seed{seed} {filename} binding mismatch")
            if config_matches_job(artifact, job) is not True:
                errors.append(f"seed{seed} config/manifest mismatch")
            validation = validate_artifact(job, artifact, endpoint=1024)
            if validation.get("valid") is not True:
                errors.append(f"seed{seed} atomic artifact invalid")
            if checkpoint_episode(artifact) != 1024:
                errors.append(f"seed{seed} checkpoint episode mismatch")
    if not (
        integrity.get("all_three_valid_at_episode_1024") is True
        and integrity.get("all_three_config_match_historical_manifest")
        is True
        and integrity.get("all_three_training_rows") == 1024
        and integrity.get("all_three_update_rows") == 128
    ):
        errors.append("atomic checkpoint aggregate mismatch")
    if errors:
        raise RuntimeError(
            "P0X2W2 reject authority rejected: " + "; ".join(errors)
        )
    return payload


def _authority_self_digest_valid(payload: dict) -> bool:
    content = dict(payload)
    recorded = content.pop("authority_fingerprint", None)
    return recorded == canonical_digest(content)


def validate_P0X2RC_reject_authority(path: Path) -> dict:
    """Validate the immutable RC rejection and all six atomic endpoints."""

    if not path.is_file():
        raise RuntimeError("P0X2RC reject authority is missing")
    if file_sha256(path) != (
        P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT_SHA256
    ):
        raise RuntimeError("P0X2RC reject authority fixed SHA mismatch")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    if not _authority_self_digest_valid(payload):
        errors.append("authority self-digest mismatch")
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != (
        "ori6_crossing_continuation_atomic_checkpoint_reject"
    ):
        errors.append("authority type mismatch")
    if payload.get("candidate_mode") != (
        P0_CROSS2_REJECTED_CONTINUATION_MODE
    ):
        errors.append("candidate mode mismatch")
    if payload.get("decision") != "reject_candidate":
        errors.append("reject decision absent")
    if payload.get("decision_endpoint_episode") != 1024:
        errors.append("decision endpoint mismatch")
    if payload.get("task_atomic_endpoints") != {
        "torus2": 512, "water1": 1024,
    }:
        errors.append("task atomic endpoints mismatch")
    if payload.get("formal_reject_predicates", {}).get(
        "reject_candidate"
    ) is not True:
        errors.append("formal rejection predicate absent")
    if payload.get("reward_and_transition_integrity", {}).get(
        "pass"
    ) is not True:
        errors.append("parent reward/transition integrity failed")
    integrity = payload.get("atomic_checkpoint_integrity", {})
    if not (
        integrity.get("all_six_valid") is True
        and integrity.get("pass") is True
    ):
        errors.append("atomic endpoint aggregate failed")
    rows = integrity.get("per_seed")
    expected_keys = {
        (task_name, seed)
        for task_name in ("torus2", "water1")
        for seed in (3201, 3202, 3203)
    }
    if not isinstance(rows, list) or {
        (row.get("task"), row.get("seed"))
        for row in rows if isinstance(row, dict)
    } != expected_keys:
        errors.append("atomic endpoint cohort mismatch")
    else:
        for row in rows:
            task_name = row["task"]
            seed = int(row["seed"])
            expected_endpoint = 512 if task_name == "torus2" else 1024
            expected_path = (
                EXPERIMENT_ROOT / "screen"
                / "crossing_continuation_diagnostic" / "P0X2RC"
                / task_name / TASKS[task_name].profile / f"seed{seed}"
            ).resolve()
            try:
                artifact = resolve_authority_path(row["artifact_path"])
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                artifact = None
            if artifact != expected_path:
                errors.append(f"{task_name}/seed{seed} artifact path mismatch")
                continue
            if row.get("checkpoint_episode") != expected_endpoint:
                errors.append(f"{task_name}/seed{seed} endpoint mismatch")
            inventory = row.get("artifact_inventory", {})
            required_files = {
                "checkpoint.pt", "training_history.csv", "update_history.csv",
                "run_config.json", "best_policy.pt", "best_policy_metrics.json",
                "deterministic_evaluation_history.csv",
                "stochastic_evaluation_history.csv",
            }
            if set(inventory) != required_files:
                errors.append(f"{task_name}/seed{seed} inventory mismatch")
            else:
                for filename, binding in inventory.items():
                    target = artifact / filename
                    if (
                        not target.is_file()
                        or binding.get("sha256") != file_sha256(target)
                        or binding.get("size_bytes") != target.stat().st_size
                    ):
                        errors.append(
                            f"{task_name}/seed{seed} {filename} binding mismatch"
                        )
            status_binding = row.get("runner_status", {})
            try:
                status = resolve_authority_path(status_binding.get("path", ""))
            except (OSError, RuntimeError, TypeError, ValueError):
                status = None
            if (
                status is None or not status.is_file()
                or status_binding.get("sha256") != file_sha256(status)
                or status_binding.get("size_bytes") != status.stat().st_size
            ):
                errors.append(f"{task_name}/seed{seed} status binding mismatch")
            stderr_binding = row.get("runner_stderr", {})
            try:
                stderr = resolve_authority_path(
                    stderr_binding.get("path", "")
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                stderr = None
            if (
                stderr is None or not stderr.is_file()
                or stderr_binding.get("sha256") != file_sha256(stderr)
                or stderr_binding.get("size_bytes") != stderr.stat().st_size
                or stderr_binding.get("empty") is not True
            ):
                errors.append(f"{task_name}/seed{seed} stderr binding mismatch")
            try:
                with open(artifact / "run_config.json", encoding="utf-8") as handle:
                    config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                config = {}
            if not (
                config.get("ori6_rejected_crossing_continuation") is True
                and config.get("ori6_crossing_policy")
                == REJECTED_CROSSING_POLICY
                and values_equal(
                    config.get("ori5_reward_parameters"),
                    expected_ori6_reward(
                        P0_CROSS2_REJECTED_CONTINUATION_MODE
                    ),
                )
            ):
                errors.append(f"{task_name}/seed{seed} config identity mismatch")
    binding = payload.get("historical_binding", {})
    manifest_binding = binding.get("manifest", {})
    try:
        historical_manifest = resolve_authority_path(
            manifest_binding.get("path", "")
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        historical_manifest = None
    if (
        historical_manifest is None or not historical_manifest.is_file()
        or manifest_binding.get("sha256")
        != file_sha256(historical_manifest)
        or manifest_binding.get("size_bytes")
        != historical_manifest.stat().st_size
    ):
        errors.append("historical RC manifest binding mismatch")
    else:
        with open(historical_manifest, encoding="utf-8") as handle:
            historical_payload = json.load(handle)
        if historical_payload.get("manifest_fingerprint") != (
            manifest_binding.get("manifest_fingerprint")
        ):
            errors.append("historical RC manifest fingerprint mismatch")
    gates = payload.get("preregistered_gates", {})
    if set(gates) != {"torus2_512", "water1_512", "water1_1024"}:
        errors.append("historical RC gate inventory mismatch")
    else:
        for label, gate in gates.items():
            try:
                gate_path = resolve_authority_path(gate.get("path", ""))
            except (OSError, RuntimeError, TypeError, ValueError):
                gate_path = None
            if (
                gate_path is None or not gate_path.is_file()
                or gate.get("sha256") != file_sha256(gate_path)
                or gate.get("size_bytes") != gate_path.stat().st_size
            ):
                errors.append(f"historical RC gate binding mismatch: {label}")
    if errors:
        raise RuntimeError(
            "P0X2RC reject authority rejected: " + "; ".join(errors)
        )
    return payload


def validate_spectral_subspace_offline_authority(path: Path) -> dict:
    """Validate the fixed offline design and executable formula identity."""

    if not path.is_file():
        raise RuntimeError("Spectral-subspace offline authority is missing")
    if file_sha256(path) != SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY_SHA256:
        raise RuntimeError("Spectral-subspace offline authority fixed SHA mismatch")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    if not _authority_self_digest_valid(payload):
        errors.append("authority self-digest mismatch")
    if payload.get("schema_version") != 1:
        errors.append("schema version mismatch")
    if payload.get("authority_type") != "ori6_spectral_subspace_offline_design":
        errors.append("authority type mismatch")
    if payload.get("decision") != "pilot_worthy_not_adopted":
        errors.append("bounded-pilot decision mismatch")
    if payload.get("candidate") != {
        "default_weight": SPECTRAL_SUBSPACE_WEIGHT,
        "name": SPECTRAL_SUBSPACE_CANDIDATE_NAME,
        "parent": "ori6_rejected_crossing_continuation",
        "relative_epsilon": SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
        "schema_version": SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION,
    }:
        errors.append("offline candidate identity mismatch")
    formulation = payload.get("spectral_formulation", {})
    runtime = spectral_subspace_runtime_identity()
    if not (
        formulation.get("formula") == runtime["formula"]
        and formulation.get("formula_sha256")
        == SPECTRAL_SUBSPACE_FORMULA_SHA256
        and formulation.get("offline_screen_parameters", {}).get("lambda")
        == SPECTRAL_SUBSPACE_WEIGHT
        and formulation.get("offline_screen_parameters", {}).get(
            "relative_epsilon"
        ) == SPECTRAL_SUBSPACE_RELATIVE_EPSILON
    ):
        errors.append("offline formula identity mismatch")
    scope = payload.get("scope", {})
    if not (
        scope.get("tasks") == list(TASKS)
        and scope.get("training_executed_by_finalizer") is False
        and scope.get("adoption_or_nonregression_claim") is False
    ):
        errors.append("offline authority scope mismatch")
    parent = payload.get("historical_binding", {}).get(
        "parent_rc_reject_authority", {}
    )
    try:
        parent_path = resolve_authority_path(parent.get("path", ""))
    except (OSError, RuntimeError, TypeError, ValueError):
        parent_path = None
    if not (
        parent_path == P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT.resolve()
        and parent.get("sha256")
        == P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT_SHA256
        and parent.get("decision") == "reject_candidate"
    ):
        errors.append("offline parent RC authority mismatch")
    sources = payload.get("historical_binding", {}).get(
        "source", {}
    ).get("files", {})
    expected_source_names = {
        "ori_codex6.py", "ori_codex6_3d.py",
        *REJECTED_CROSSING_ADDITIVE_SOURCE_FILES,
        *SPECTRAL_SUBSPACE_ADDITIVE_SOURCE_FILES,
        "test_ori_codex6_spectral_subspace.py",
    }
    if set(sources) != expected_source_names:
        errors.append("offline source inventory mismatch")
    else:
        for name, binding in sources.items():
            target = (SCRIPT_DIRECTORY / name).resolve()
            if (
                not target.is_file()
                or binding.get("sha256") != file_sha256(target)
                or binding.get("size_bytes") != target.stat().st_size
            ):
                errors.append(f"offline source binding mismatch: {name}")
    generator = payload.get("historical_binding", {}).get(
        "source", {}
    ).get("generator", {})
    generator_path = SCRIPT_DIRECTORY / "finalize_ori_codex6_spectral_subspace_offline.py"
    if (
        not generator_path.is_file()
        or generator.get("sha256") != file_sha256(generator_path)
        or generator.get("size_bytes") != generator_path.stat().st_size
    ):
        errors.append("offline generator binding mismatch")
    if payload.get("weight_zero_contract", {}).get("weight") != 0.0:
        errors.append("weight-zero parent contract missing")
    if errors:
        raise RuntimeError(
            "Spectral-subspace offline authority rejected: "
            + "; ".join(errors)
        )
    return payload


def validate_spectral_screen_report_authority(path: Path) -> dict:
    """Validate the immutable formal screen report that permits regression.

    The bounded screen formally rejected adoption, but its signed conclusion
    explicitly permits a separately preregistered fresh-seed regression.  This
    validator therefore checks that narrow authorization; it must never turn
    the report into evidence for adoption or a 10,000-episode production run.
    """

    def recorded_path(value) -> Path | None:
        try:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = SCRIPT_DIRECTORY / candidate
            return candidate.resolve()
        except (OSError, TypeError, ValueError):
            return None

    def file_binding_is_current(binding: dict) -> bool:
        if not isinstance(binding, dict):
            return False
        target = recorded_path(binding.get("path"))
        return bool(
            target is not None
            and target.is_file()
            and binding.get("sha256") == file_sha256(target)
            and binding.get("size_bytes") == target.stat().st_size
        )

    if path.resolve() != SPECTRAL_SUBSPACE_SCREEN_REPORT.resolve():
        raise RuntimeError("Spectral screen report path mismatch")
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != SPECTRAL_SUBSPACE_SCREEN_REPORT_SHA256:
        raise RuntimeError("Spectral screen report fixed SHA256 mismatch")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    errors = []
    fingerprint_payload = dict(payload)
    recorded_fingerprint = fingerprint_payload.pop(
        "authority_fingerprint", None
    )
    if not (
        recorded_fingerprint == SPECTRAL_SUBSPACE_SCREEN_REPORT_FINGERPRINT
        and canonical_digest(fingerprint_payload) == recorded_fingerprint
    ):
        errors.append("formal report self-digest mismatch")
    if not (
        payload.get("schema_version") == 1
        and payload.get("authority_type")
        == "ori6_spectral_subspace_atomic_screen_formal_report"
        and payload.get("stage") == "spectral_subspace_diagnostic"
        and payload.get("candidate_mode")
        == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        and payload.get("candidate_mode_slug") == "P0X2RCS05"
    ):
        errors.append("formal report identity mismatch")
    if not (
        payload.get("decision")
        == "formal_gate_reject_but_promising_signal_for_regression"
        and payload.get("spectral_subspace_screen_pass") is False
        and payload.get("candidate_adopted") is False
        and payload.get("fresh_confirmation_required") is True
        and payload.get("full_10000_episode_training_authorized_by_this_report")
        is False
    ):
        errors.append("formal reject/regression-only disposition mismatch")
    rationale = payload.get("user_defined_possibility_rationale", {})
    if not (
        rationale.get("classification")
        == "possibility_visible_but_not_confirmed"
        and rationale.get("permitted_next_claim")
        == (
            "A separately preregistered fresh-seed regression/confirmation "
            "may be considered; this report does not itself authorize "
            "adoption or a 10000-episode production run."
        )
    ):
        errors.append("fresh regression permission mismatch")
    expected_requirements = [
        "fresh preregistered confirmation seeds",
        "fresh random baseline on all five tasks",
        "fresh successful-task regression on five_d2, five_d3, and watersqrt2",
        "a separate authorization decision before any 10000-episode training",
    ]
    if payload.get("fresh_confirmation_requirements") != expected_requirements:
        errors.append("fresh confirmation requirements mismatch")
    scope = payload.get("scope", {})
    if not (
        scope.get("protocol") == "spectral_subspace_diagnostic"
        and scope.get("tasks") == ["torus2", "water1"]
        and scope.get("seeds") == [3201, 3202, 3203]
        and scope.get("checkpoint_every") == 512
        and scope.get("preregistered_episode_cap") == 2048
        and scope.get("task_atomic_endpoints")
        == {"torus2": 2048, "water1": 1024}
        and scope.get("fixed_independent_task_cohorts") is True
        and scope.get("heldout_metrics_used_for_decision") is False
        and scope.get("fresh_random_baseline_executed_or_claimed") is False
        and scope.get("raw_artifacts_modified_by_finalizer") is False
        and scope.get("runner_or_core_modified_by_finalizer") is False
    ):
        errors.append("formal report scope mismatch")

    historical = payload.get("historical_binding", {})
    runtime = spectral_subspace_runtime_identity()
    if historical.get("candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        errors.append("candidate reward identity mismatch")
    if historical.get("parent_candidate_reward_sha256") != candidate_reward_sha256(
        P0_CROSS2_REJECTED_CONTINUATION_MODE
    ):
        errors.append("parent reward identity mismatch")
    if not (
        historical.get("spectral_formula_sha256")
        == SPECTRAL_SUBSPACE_FORMULA_SHA256
        and historical.get("spectral_formula") == runtime["formula"]
        and historical.get("spectral_weight") == SPECTRAL_SUBSPACE_WEIGHT
        and historical.get("spectral_relative_epsilon")
        == SPECTRAL_SUBSPACE_RELATIVE_EPSILON
    ):
        errors.append("spectral formula/runtime identity mismatch")
    if historical.get("parent_additive_source_sha256") != (
        rejected_crossing_additive_source_hashes()
    ):
        errors.append("parent additive source mismatch")
    if historical.get("spectral_additive_source_sha256") != (
        spectral_subspace_additive_source_hashes()
    ):
        errors.append("spectral additive source mismatch")
    if historical.get("source_sha256") != source_hashes():
        errors.append("base source binding mismatch")
    if historical.get("model_sha256_by_task") != {
        name: model_hashes(TASKS[name]) for name in ("torus2", "water1")
    }:
        errors.append("screen model binding mismatch")
    expected_profiles = {
        name: PROFILES[name] for name in ("A", "B")
    }
    expected_optimizer = {
        name: candidate_optimizer_sha256(
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE, profile
        )
        for name, profile in expected_profiles.items()
    }
    parent_optimizer = {
        name: candidate_optimizer_sha256(
            P0_CROSS2_REJECTED_CONTINUATION_MODE, profile
        )
        for name, profile in expected_profiles.items()
    }
    if historical.get("candidate_optimizer_sha256_by_profile") != (
        expected_optimizer
    ):
        errors.append("candidate optimizer binding mismatch")
    if historical.get("parent_optimizer_sha256_by_profile") != parent_optimizer:
        errors.append("parent optimizer binding mismatch")
    expected_mechanics = {
        name: candidate_mechanics_sha256(
            P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE, TASKS[name]
        )
        for name in ("torus2", "water1")
    }
    parent_mechanics = {
        name: candidate_mechanics_sha256(
            P0_CROSS2_REJECTED_CONTINUATION_MODE, TASKS[name]
        )
        for name in ("torus2", "water1")
    }
    if historical.get("candidate_mechanics_sha256_by_task") != expected_mechanics:
        errors.append("candidate mechanics binding mismatch")
    if historical.get("parent_mechanics_sha256_by_task") != parent_mechanics:
        errors.append("parent mechanics binding mismatch")
    transition_hash = canonical_digest(expected_crossing_transition(
        P0_CROSS2_REJECTED_CONTINUATION_MODE
    ))
    if not (
        historical.get("candidate_transition_semantics_sha256")
        == transition_hash
        and historical.get("parent_transition_semantics_sha256")
        == transition_hash
    ):
        errors.append("transition semantics binding mismatch")

    for field, expected_path, fixed_digest in (
        (
            "parent_RC_reject_authority",
            P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT,
            P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT_SHA256,
        ),
        (
            "spectral_offline_design_authority",
            SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY,
            SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY_SHA256,
        ),
    ):
        binding = historical.get(field, {})
        if not (
            recorded_path(binding.get("path")) == expected_path.resolve()
            and expected_path.is_file()
            and binding.get("sha256") == fixed_digest
            and file_sha256(expected_path) == fixed_digest
            and binding.get("size_bytes") == expected_path.stat().st_size
        ):
            errors.append(f"{field} binding mismatch")
    finalizer = historical.get("finalizer", {})
    if not file_binding_is_current(finalizer):
        errors.append("formal report finalizer binding mismatch")

    expected_endpoints = {
        (task_name, seed): 2048 if task_name == "torus2" else 1024
        for task_name in ("torus2", "water1")
        for seed in (3201, 3202, 3203)
    }
    atomic = payload.get("atomic_endpoint_integrity", {})
    records = atomic.get("per_seed")
    if not (
        atomic.get("pass") is True
        and isinstance(records, list)
        and {
            (row.get("task"), row.get("seed"))
            for row in records if isinstance(row, dict)
        } == set(expected_endpoints)
        and len(records) == len(expected_endpoints)
    ):
        errors.append("spectral atomic endpoint cohort mismatch")
    else:
        for row in records:
            key = (row["task"], row["seed"])
            endpoint = expected_endpoints[key]
            label = f"{row['task']}/seed{row['seed']}"
            artifact = recorded_path(row.get("artifact_path"))
            validation = row.get("artifact_validation", {})
            status = row.get("runner_status_value")
            expected_complete = status == "complete"
            expected_return_code = 0 if expected_complete else 1
            if not (
                row.get("checkpoint_episode") == endpoint
                and row.get("training_rows") == endpoint
                and validation.get("valid") is True
                and validation.get("complete") is expected_complete
                and validation.get("endpoint_episode") == endpoint
                and validation.get("training_rows") == endpoint
                and artifact is not None
                and artifact.is_dir()
                and row.get("checkpoint_config_training_subset_matches_manifest_job")
                is True
                and row.get("run_config_training_subset_matches_manifest_job")
                is True
                and status in {"complete", "early_stopped_at_checkpoint"}
                and row.get("runner_return_code") == expected_return_code
            ):
                errors.append(f"{label} endpoint integrity mismatch")
                continue
            inventory = row.get("direct_file_inventory")
            if not isinstance(inventory, dict) or not inventory:
                errors.append(f"{label} direct inventory missing")
            else:
                for filename, binding in inventory.items():
                    if not (
                        file_binding_is_current(binding)
                        and recorded_path(binding.get("path"))
                        == (artifact / filename).resolve()
                    ):
                        errors.append(f"{label} artifact binding mismatch: {filename}")
            for field in ("runner_status", "runner_stdout", "runner_stderr"):
                if not file_binding_is_current(row.get(field, {})):
                    errors.append(f"{label} {field} binding mismatch")

    reward_integrity = payload.get(
        "reward_transition_and_spectral_integrity", {}
    )
    integrity_rows = reward_integrity.get("per_seed")
    if not (
        reward_integrity.get("all_six_pass") is True
        and isinstance(integrity_rows, list)
        and {
            (row.get("task"), row.get("seed"))
            for row in integrity_rows if isinstance(row, dict)
        } == set(expected_endpoints)
        and len(integrity_rows) == len(expected_endpoints)
        and all(
            row.get("pass") is True
            and row.get("episode_count")
            == expected_endpoints[(row.get("task"), row.get("seed"))]
            and row.get("nonfinite_reward_component_count") == 0
            for row in integrity_rows
        )
    ):
        errors.append("reward/transition/spectral integrity mismatch")

    expected_gates = {
        ("torus2", 512), ("torus2", 1024), ("torus2", 2048),
        ("water1", 512), ("water1", 1024),
    }
    gates = payload.get("preregistered_gates")
    if not (
        isinstance(gates, list)
        and {(row.get("task"), row.get("checkpoint_episode")) for row in gates}
        == expected_gates
        and len(gates) == len(expected_gates)
    ):
        errors.append("preregistered gate cohort mismatch")
    else:
        for row in gates:
            label = f"{row['task']}/checkpoint{row['checkpoint_episode']}"
            target = recorded_path(row.get("path"))
            if not (
                target is not None
                and target.is_file()
                and row.get("sha256") == file_sha256(target)
                and row.get("size_bytes") == target.stat().st_size
                and row.get("critical_semantic_recomputation_match") is True
            ):
                errors.append(f"{label} gate binding mismatch")
                continue
            with open(target, encoding="utf-8") as handle:
                saved_gate = json.load(handle)
            if not (
                saved_gate == row.get("payload")
                and saved_gate.get("protocol") == "spectral_subspace_diagnostic"
                and saved_gate.get("task") == row["task"]
                and saved_gate.get("checkpoint_episode")
                == row["checkpoint_episode"]
                and saved_gate.get("candidate_mode")
                == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
                and saved_gate.get("preregistered_gate_specification")
                == spectral_subspace_gate_specification()
            ):
                errors.append(f"{label} gate semantic mismatch")

    manifest_binding = historical.get("manifest", {})
    manifest_path = recorded_path(manifest_binding.get("path"))
    if not (
        manifest_path is not None
        and manifest_path.is_file()
        and manifest_binding.get("sha256") == file_sha256(manifest_path)
        and manifest_binding.get("size_bytes") == manifest_path.stat().st_size
    ):
        errors.append("historical spectral manifest binding mismatch")
    else:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        content = dict(manifest)
        manifest_fingerprint = content.pop("manifest_fingerprint", None)
        content.pop("created_utc", None)
        if not (
            manifest_fingerprint == canonical_digest(content)
            and manifest_fingerprint
            == manifest_binding.get("manifest_fingerprint")
            and manifest.get("protocol", {}).get("name")
            == "spectral_subspace_diagnostic"
            and manifest.get("candidate_mode")
            == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
            and len(manifest.get("jobs", [])) == 6
            and manifest.get("source_sha256") == historical.get("source_sha256")
            and manifest.get("candidate_additive_source_sha256") == {
                **historical.get("parent_additive_source_sha256", {}),
                **historical.get("spectral_additive_source_sha256", {}),
            }
            and manifest.get("candidate_reward_sha256")
            == historical.get("candidate_reward_sha256")
        ):
            errors.append("historical spectral manifest identity mismatch")
    if errors:
        raise RuntimeError(
            "Spectral screen report authority rejected: " + "; ".join(errors)
        )
    return payload


def spectral_screen_report_binding() -> dict:
    """Return a pending dry-run binding or a validated immutable report SHA."""

    path = SPECTRAL_SUBSPACE_SCREEN_REPORT.resolve()
    if not path.is_file():
        return {
            "path": str(path),
            "sha256": None,
            "status": "pending_not_launchable",
        }
    validate_spectral_screen_report_authority(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": "bound_regression_permission_authority",
    }


def validate_execution_gate(protocol: Protocol, mode: str, selection_path: Path) -> dict | None:
    if protocol.name in (
        "smoke", "rescue", "entropy_diagnostic", "width_diagnostic",
        "crossing_continuation_diagnostic", "spectral_subspace_diagnostic",
        "spectral_nonregression",
    ):
        if mode == P0_CROSS2_MODE:
            if not P0_CROSS2_OFFLINE_AUDIT.is_file():
                raise RuntimeError(
                    "P0X2 rescue requires offline_P0X2_saved_path_ordering.json"
                )
            with open(P0_CROSS2_OFFLINE_AUDIT, encoding="utf-8") as handle:
                offline = json.load(handle)
            offline_errors = []
            if offline.get("candidate_mode") != P0_CROSS2_MODE:
                offline_errors.append("candidate mode mismatch")
            if offline.get("candidate_reward_sha256") != candidate_reward_sha256(mode):
                offline_errors.append("reward hash mismatch")
            if offline.get("all_required_orderings_pass") is not True:
                offline_errors.append("saved path ordering did not pass")
            orderings = offline.get("orderings")
            if (
                not isinstance(orderings, dict)
                or set(orderings) != set(TASKS)
                or not all(
                    isinstance(row, dict) and row.get("pass") is True
                    for row in orderings.values()
                )
            ):
                offline_errors.append("per-task saved path orderings did not pass")
            if offline.get("source_sha256") != source_hashes():
                offline_errors.append("source hashes are stale")
            if offline.get("model_sha256_by_task") != {
                name: model_hashes(task) for name, task in TASKS.items()
            }:
                offline_errors.append("model hashes are stale")
            if offline_errors:
                raise RuntimeError(
                    "P0X2 offline authority rejected: " + "; ".join(offline_errors)
                )
            if not P0_RECLASSIFICATION_REPORT.is_file():
                raise RuntimeError(
                    "P0X2 rescue requires rescue_P0_reclassification.json"
                )
            with open(P0_RECLASSIFICATION_REPORT, encoding="utf-8") as handle:
                P0_evidence = json.load(handle)
            formal = P0_evidence.get("corrected_decision", {}).get(
                "formal_return_order_gate", {}
            )
            evidence_errors = []
            if P0_evidence.get("candidate_mode") != P0_MODE:
                evidence_errors.append("P0 candidate identity mismatch")
            if P0_evidence.get("source_sha256") != source_hashes():
                evidence_errors.append("P0 source hashes are stale")
            if formal.get("observed_crossing_attractor_seed_count", 0) < 2:
                evidence_errors.append("fewer than two P0 crossing attractor seeds")
            if formal.get("pass_possible_from_fixed_checkpoint_data") is not False:
                evidence_errors.append("P0 return-order failure is not established")
            if evidence_errors:
                raise RuntimeError(
                    "P0X2 P0-failure authority rejected: "
                    + "; ".join(evidence_errors)
                )
        if mode == P0_CROSS2_FRONTIER12_MODE:
            if not P0_CROSS2_FRONTIER12_OFFLINE_AUDIT.is_file():
                raise RuntimeError(
                    "P0X2B12 rescue requires "
                    "offline_P0X2B12_saved_path_ordering.json"
                )
            if not P0_CROSS2_REJECT_REPORT.is_file():
                raise RuntimeError(
                    "P0X2B12 rescue requires rescue_P0X2_report.json"
                )
            with open(
                P0_CROSS2_FRONTIER12_OFFLINE_AUDIT, encoding="utf-8"
            ) as handle:
                offline = json.load(handle)
            offline_errors = []
            if offline.get("candidate_mode") != mode:
                offline_errors.append("candidate mode mismatch")
            if offline.get("parent_candidate_mode") != P0_CROSS2_MODE:
                offline_errors.append("parent candidate mismatch")
            if offline.get("candidate_reward_sha256") != candidate_reward_sha256(mode):
                offline_errors.append("reward hash mismatch")
            if offline.get("parent_candidate_reward_sha256") != candidate_reward_sha256(
                P0_CROSS2_MODE
            ):
                offline_errors.append("parent reward hash mismatch")
            if offline.get("single_changed_reward_parameter") != {
                "name": "ori6_spread_frontier_bonus",
                "P0X2_value": candidate_frontier_bonus(P0_CROSS2_MODE),
                "candidate_value": candidate_frontier_bonus(mode),
            }:
                offline_errors.append("single changed parameter mismatch")
            if offline.get("all_required_orderings_pass") is not True:
                offline_errors.append("saved path ordering did not pass")
            orderings = offline.get("orderings")
            if (
                not isinstance(orderings, dict)
                or set(orderings) != set(TASKS)
                or not all(
                    isinstance(row, dict) and row.get("pass") is True
                    for row in orderings.values()
                )
            ):
                offline_errors.append("per-task saved path orderings did not pass")
            if offline.get("source_sha256") != source_hashes():
                offline_errors.append("source hashes are stale")
            if offline.get("model_sha256_by_task") != {
                name: model_hashes(task) for name, task in TASKS.items()
            }:
                offline_errors.append("model hashes are stale")
            offline_parent_path = offline.get(
                "parent_candidate_reject_authority"
            )
            try:
                offline_parent_path_matches = (
                    bool(offline_parent_path)
                    and Path(offline_parent_path).resolve()
                    == P0_CROSS2_REJECT_REPORT.resolve()
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                offline_parent_path_matches = False
            if not offline_parent_path_matches:
                offline_errors.append("parent reject authority path mismatch")
            elif offline.get(
                "parent_candidate_reject_authority_sha256"
            ) != file_sha256(P0_CROSS2_REJECT_REPORT):
                offline_errors.append("parent reject authority hash mismatch")
            if offline_errors:
                raise RuntimeError(
                    "P0X2B12 offline authority rejected: "
                    + "; ".join(offline_errors)
                )
            with open(P0_CROSS2_REJECT_REPORT, encoding="utf-8") as handle:
                parent_evidence = json.load(handle)
            water_gate = parent_evidence.get("water1_gate", {})
            torus_gate = parent_evidence.get("torus_gate", {})
            parent_manifest = parent_evidence.get("manifest_authority", {})
            evidence_errors = []
            if parent_evidence.get("candidate_mode") != P0_CROSS2_MODE:
                evidence_errors.append("P0X2 candidate identity mismatch")
            if parent_evidence.get("decision") != "reject_candidate":
                evidence_errors.append("P0X2 reject decision is absent")
            if parent_evidence.get("rescue_gate_pass") is not False:
                evidence_errors.append("P0X2 rescue rejection is not established")
            if water_gate.get("pass") is not False:
                evidence_errors.append("P0X2 water1 failure is not established")
            if torus_gate.get("pass") is not True:
                evidence_errors.append("P0X2 torus rescue is not established")
            if parent_evidence.get("source_sha256") != source_hashes():
                evidence_errors.append("P0X2 source hashes are stale")
            if parent_evidence.get("model_sha256_by_task") != {
                name: model_hashes(task) for name, task in TASKS.items()
            }:
                evidence_errors.append("P0X2 model hashes are stale")
            parent_manifest_path = parent_manifest.get("path")
            if not parent_manifest_path or not Path(parent_manifest_path).is_file():
                evidence_errors.append("P0X2 manifest authority is missing")
            elif parent_manifest.get("sha256") != file_sha256(
                Path(parent_manifest_path)
            ):
                evidence_errors.append("P0X2 manifest authority hash mismatch")
            if evidence_errors:
                raise RuntimeError(
                    "P0X2B12 P0X2-failure authority rejected: "
                    + "; ".join(evidence_errors)
                )
        if mode == P0_CROSS2_ENTROPY3_MODE:
            if (
                protocol.name != "entropy_diagnostic"
                or protocol.tasks != ("water1",)
                or protocol.episodes != 1024
                or protocol.seeds != (3101, 3102, 3103)
                or protocol.checkpoint_every != 512
            ):
                raise RuntimeError(
                    "P0X2E3 is authorized only for the fixed water1 "
                    "1024-episode diagnostic protocol"
                )
            if candidate_reward_sha256(mode) != candidate_reward_sha256(
                P0_CROSS2_MODE
            ):
                raise RuntimeError("P0X2E3 reward identity differs from P0X2")
            profile = PROFILES[TASKS["water1"].profile]
            parent_optimizer = expected_optimizer(P0_CROSS2_MODE, profile)
            candidate_optimizer = expected_optimizer(mode, profile)
            changed = {
                key for key in candidate_optimizer
                if candidate_optimizer[key] != parent_optimizer[key]
            }
            if changed != {"entropy_coefficient"}:
                raise RuntimeError(
                    "P0X2E3 optimizer differs from P0X2 beyond entropy_coefficient"
                )
            if (
                parent_optimizer["entropy_coefficient"] != 1.0e-3
                or candidate_optimizer["entropy_coefficient"] != 3.0e-3
            ):
                raise RuntimeError("P0X2E3 entropy transition is not 0.001 -> 0.003")
            validate_P0X2_saved_path_authority(
                P0_CROSS2_OFFLINE_AUDIT, candidate_reward_sha256(mode)
            )
            validate_P0X2B12_checkpoint_reject_authority(
                P0_CROSS2_FRONTIER12_REJECT_REPORT
            )
            validate_P0X2E3_preflight_authority(
                P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT
            )
        if mode == P0_CROSS2_WIDTH2_MODE:
            if (
                protocol.name != "width_diagnostic"
                or protocol.tasks != ("water1",)
                or protocol.episodes != 2048
                or protocol.seeds != (3101, 3102, 3103)
                or protocol.checkpoint_every != 512
            ):
                raise RuntimeError(
                    "P0X2W2 is authorized only for the fixed water1 "
                    "2048-episode diagnostic protocol"
                )
            if candidate_reward_sha256(mode) != candidate_reward_sha256(
                P0_CROSS2_MODE
            ):
                raise RuntimeError("P0X2W2 reward identity differs from P0X2")
            task = TASKS["water1"]
            profile = PROFILES[task.profile]
            if expected_optimizer(mode, profile) != expected_optimizer(
                P0_CROSS2_MODE, profile
            ):
                raise RuntimeError("P0X2W2 optimizer identity differs from P0X2")
            parent_mechanics = expected_mechanics(P0_CROSS2_MODE, task)
            candidate_mechanics = expected_mechanics(mode, task)
            changed = {
                key for key in candidate_mechanics
                if candidate_mechanics[key] != parent_mechanics[key]
            }
            if changed != {"max_step_width"}:
                raise RuntimeError(
                    "P0X2W2 mechanics differ from P0X2 beyond max_step_width"
                )
            if not (
                values_equal(parent_mechanics["max_step_width"], 0.3)
                and values_equal(candidate_mechanics["max_step_width"], 0.2)
            ):
                raise RuntimeError("P0X2W2 width transition is not 0.3 -> 0.2")
            validate_P0X2_saved_path_authority(
                P0_CROSS2_OFFLINE_AUDIT, candidate_reward_sha256(mode)
            )
            validate_P0X2E3_reject_authority(
                P0_CROSS2_ENTROPY3_REJECT_REPORT
            )
            validate_P0X2W2_feasibility_authority(
                P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT
            )
        if mode == P0_CROSS2_REJECTED_CONTINUATION_MODE:
            expected_protocol = PROTOCOLS[
                "crossing_continuation_diagnostic"
            ]
            if protocol != expected_protocol:
                raise RuntimeError(
                    "P0X2RC is authorized only for the fixed torus2+water1 "
                    "seed3201-3203 2048-episode diagnostic protocol"
                )
            if candidate_frontier_bonus(mode) != 0.9:
                raise RuntimeError("P0X2RC frontier bonus is not 0.9")
            if candidate_crossing_penalty(mode) != 2.0:
                raise RuntimeError("P0X2RC crossing penalty is not 2.0")
            if rejected_crossing_additive_source_hashes() != {
                name: file_sha256(SCRIPT_DIRECTORY / name)
                for name in REJECTED_CROSSING_ADDITIVE_SOURCE_FILES
            }:
                raise RuntimeError("P0X2RC additive source identity mismatch")
            for task_name in expected_protocol.tasks:
                task = TASKS[task_name]
                profile = PROFILES[task.profile]
                if expected_optimizer(mode, profile) != expected_optimizer(
                    P0_CROSS2_MODE, profile
                ):
                    raise RuntimeError(
                        f"P0X2RC {task_name} optimizer differs from P0X2"
                    )
                if expected_mechanics(mode, task) != expected_mechanics(
                    P0_CROSS2_MODE, task
                ):
                    raise RuntimeError(
                        f"P0X2RC {task_name} physical mechanics differ "
                        "from P0X2"
                    )
            transition = expected_crossing_transition(mode)
            if not (
                transition.get("policy") == REJECTED_CROSSING_POLICY
                and transition.get("candidate_committed") is False
                and transition.get("step_count_increments") is True
                and transition.get(
                    "past_frontier_and_joint_ledgers_retained"
                ) is True
                and transition.get(
                    "rejected_candidate_endpoint_reward"
                ) == 0.0
                and transition.get(
                    "rejected_candidate_spread_frontier_reward"
                ) == 0.0
                and transition.get(
                    "rejected_candidate_new_milestone_reward"
                ) == 0.0
            ):
                raise RuntimeError("P0X2RC transition semantics mismatch")
            validate_P0X2_saved_path_authority(
                P0_CROSS2_OFFLINE_AUDIT,
                candidate_reward_sha256(P0_CROSS2_MODE),
            )
            validate_P0X2W2_reject_authority(
                P0_CROSS2_WIDTH2_REJECT_REPORT
            )
        if mode == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE:
            if protocol.name not in (
                "spectral_subspace_diagnostic", "spectral_nonregression",
            ):
                raise RuntimeError(
                    "P0X2RCS05 is restricted to its diagnostic and paired "
                    "nonregression protocols"
                )
            expected_protocol = PROTOCOLS[protocol.name]
            if protocol != expected_protocol:
                raise RuntimeError(
                    "P0X2RCS05 protocol identity mismatch"
                )
            parent_mode = P0_CROSS2_REJECTED_CONTINUATION_MODE
            runtime = spectral_subspace_runtime_identity()
            if not (
                runtime["candidate"] == SPECTRAL_SUBSPACE_CANDIDATE_NAME
                and runtime["weight"] == SPECTRAL_SUBSPACE_WEIGHT
                and runtime["relative_epsilon"]
                == SPECTRAL_SUBSPACE_RELATIVE_EPSILON
                and runtime["formula_sha256"]
                == SPECTRAL_SUBSPACE_FORMULA_SHA256
            ):
                raise RuntimeError("P0X2RCS05 runtime identity mismatch")
            for task_name in expected_protocol.tasks:
                task = TASKS[task_name]
                profile = PROFILES[task.profile]
                if expected_optimizer(mode, profile) != expected_optimizer(
                    parent_mode, profile
                ):
                    raise RuntimeError(
                        f"P0X2RCS05 {task_name} optimizer differs from RC"
                    )
                if expected_mechanics(mode, task) != expected_mechanics(
                    parent_mode, task
                ):
                    raise RuntimeError(
                        f"P0X2RCS05 {task_name} mechanics differ from RC"
                    )
            if expected_crossing_transition(mode) != (
                expected_crossing_transition(parent_mode)
            ):
                raise RuntimeError(
                    "P0X2RCS05 crossing transition differs from RC"
                )
            validate_P0X2RC_reject_authority(
                P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
            )
            validate_spectral_subspace_offline_authority(
                SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
            )
            if protocol.name == "spectral_nonregression":
                if protocol.implementations != (
                    "ori6", "ori5_reference",
                ):
                    raise RuntimeError(
                        "Spectral nonregression requires candidate/reference pairs"
                    )
                validate_spectral_screen_report_authority(
                    SPECTRAL_SUBSPACE_SCREEN_REPORT
                )
        if mode == P1_MODE:
            trigger = EXPERIMENT_ROOT / "analysis" / "rescue_P0_report.json"
            if not trigger.is_file():
                raise RuntimeError("P1 rescue requires rescue_P0_report.json")
            with open(trigger, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("decision") != "trigger_P1":
                raise RuntimeError("P1 is allowed only after P0 safe-timeout/frontier failure")
        return None
    if protocol.name == "regression":
        path = EXPERIMENT_ROOT / "analysis" / f"rescue_{MODE_SLUGS[mode]}_report.json"
        if not path.is_file():
            raise RuntimeError(f"Regression requires {path.name}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("rescue_gate_pass") is not True:
            raise RuntimeError("Regression requires a passing rescue gate")
        if payload.get("source_sha256") != source_hashes():
            raise RuntimeError("Rescue report source hashes are stale")
        return payload
    if not selection_path.is_file():
        raise RuntimeError(f"Full stage requires analyzer selection: {selection_path}")
    with open(selection_path, encoding="utf-8") as handle:
        selection = json.load(handle)
    errors = []
    if selection.get("selected_candidate_mode") != mode:
        errors.append("candidate mode mismatch")
    if selection.get("rescue_gate_pass") is not True:
        errors.append("rescue gate did not pass")
    if selection.get("successful_task_regression_gate_pass") is not True:
        errors.append("successful-task regression gate did not pass")
    if selection.get("source_sha256") != source_hashes():
        errors.append("source hashes are stale")
    if selection.get("model_sha256_by_task") != {
        name: model_hashes(task) for name, task in TASKS.items()
    }:
        errors.append("model hashes are stale")
    authority = random_baseline_authority()
    if selection.get("random_baseline_authority_sha256") != authority["sha256"]:
        errors.append("random baseline authority changed")
    report_authorities = (
        (
            "rescue",
            EXPERIMENT_ROOT / "analysis"
            / f"rescue_{MODE_SLUGS[mode]}_report.json",
        ),
        (
            "regression",
            EXPERIMENT_ROOT / "analysis"
            / f"regression_{MODE_SLUGS[mode]}_report.json",
        ),
    )
    for label, expected_path in report_authorities:
        recorded_path = selection.get(f"{label}_report")
        recorded_sha256 = selection.get(f"{label}_report_sha256")
        if not recorded_path:
            errors.append(f"{label} report path is missing")
            continue
        try:
            path_matches = Path(recorded_path).resolve() == expected_path.resolve()
        except (OSError, TypeError, ValueError):
            path_matches = False
        if not path_matches:
            errors.append(f"{label} report path mismatch")
            continue
        if not expected_path.is_file():
            errors.append(f"{label} report is missing")
            continue
        if not recorded_sha256:
            errors.append(f"{label} report sha256 is missing")
        elif recorded_sha256 != file_sha256(expected_path):
            errors.append(f"{label} report sha256 mismatch")
    if protocol.name == "full_replication" and selection.get("primary_full_gate_pass") is not True:
        errors.append("replication requires a passing primary full gate")
    if errors:
        raise RuntimeError("Full-stage selection rejected: " + "; ".join(errors))
    return selection


def launch(job: dict, destination: Path) -> dict:
    if destination.exists():
        raise RuntimeError(f"Refusing to precreate/reuse child output: {destination}")
    log_directory = runner_directory(job, destination)
    log_directory.mkdir(parents=True, exist_ok=True)
    command = build_command(job, destination)
    status = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_fingerprint": job["fingerprint"],
        "status": "launching",
        "started_utc": utc_now(),
        "command": command,
        "requested_output_directory": str(destination),
        "effective_output_directory": None,
    }
    status_file = log_directory / "status.json"
    atomic_json(status_file, status)
    stdout = open(log_directory / "stdout.log", "a", encoding="utf-8")
    stderr = open(log_directory / "stderr.log", "a", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(job["seed"]),
    })
    process = subprocess.Popen(
        command, cwd=SCRIPT_DIRECTORY, stdout=stdout, stderr=stderr,
        env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    status.update({"status": "running", "pid": process.pid})
    atomic_json(status_file, status)
    return {
        "job": job, "destination": destination, "log_directory": log_directory,
        "status_file": status_file, "status": status, "process": process,
        "stdout": stdout, "stderr": stderr, "started": time.monotonic(),
    }


def current_output(item: dict) -> Path | None:
    destination = item["destination"]
    matches = [path for path in effective_candidates(destination) if config_matches_job(path, item["job"])]
    return matches[0] if len(matches) == 1 else None


def checkpoint_episode(path: Path) -> int | None:
    checkpoint = path / "checkpoint.pt"
    if not checkpoint.is_file():
        return None
    try:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(payload["episode"])
    except Exception:
        return None


def float_field(row: dict, name: str, default=0.0) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def spectral_parent_common_prefix_comparison(
    rows_by_seed: dict[int, list[dict]], task_name: str, episode: int,
) -> dict:
    """Compare paired seeds only where the rejected RC parent exists.

    This is deliberately diagnostic.  In particular, the rejected parent has
    only 512 torus episodes, so a missing torus episode-1024 endpoint can never
    veto the spectral candidate's episode-1024 gate.
    """

    result = {
        "gating": False,
        "requested_candidate_endpoint": int(episode),
        "parent_requested_endpoint_available": False,
        "missing_parent_requested_endpoint_invalidates": False,
        "per_seed": {},
    }
    try:
        with open(
            P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT,
            encoding="utf-8",
        ) as handle:
            authority = json.load(handle)
        bound = {
            (row["task"], int(row["seed"])): row
            for row in authority.get(
                "atomic_checkpoint_integrity", {}
            ).get("per_seed", [])
        }
        for seed in sorted(rows_by_seed):
            record = bound.get((task_name, int(seed)))
            if record is None:
                result["per_seed"][str(seed)] = {
                    "available": False,
                    "reason": "parent_atomic_record_missing",
                }
                continue
            parent_endpoint = int(record.get("checkpoint_episode", 0))
            artifact = resolve_authority_path(record.get("artifact_path", ""))
            history_path = artifact / "training_history.csv"
            if not history_path.is_file():
                result["per_seed"][str(seed)] = {
                    "available": False,
                    "reason": "parent_history_missing",
                }
                continue
            parent_rows = read_csv(history_path)
            common = min(int(episode), parent_endpoint, len(parent_rows))
            candidate_rows = rows_by_seed[seed][:common]
            parent_prefix = parent_rows[:common]

            def constructive(row: dict) -> bool:
                if not (
                    float_field(row, "self_crossing") == 0.0
                    and float_field(row, "correction_failed") == 0.0
                    and float_field(row, "no_modes") == 0.0
                ):
                    return False
                flatness = float_field(row, "flatness_ratio", math.inf)
                spread = float_field(row, "spread_error", math.inf)
                return (
                    flatness + spread ** 2 <= 0.5
                    if task_name == "torus2"
                    else flatness <= 0.5 and spread <= 0.1
                )

            def metrics(rows: list[dict]) -> dict:
                return {
                    "episodes": len(rows),
                    "constructive_proxy_episodes": sum(
                        constructive(row) for row in rows
                    ),
                    "valid_joint_or_strict_episodes": sum(
                        (
                            float_field(row, "flatness_ratio", math.inf) <= 0.1
                            and float_field(row, "spread_error", math.inf) <= 0.05
                            and float_field(row, "self_crossing") == 0.0
                            and float_field(row, "correction_failed") == 0.0
                            and float_field(row, "no_modes") == 0.0
                        )
                        or float_field(row, "success") > 0.0
                        for row in rows
                    ),
                    "strict_success_episodes": sum(
                        float_field(row, "success") > 0.0 for row in rows
                    ),
                    "maximum_spread_progress": max(
                        [float_field(
                            row, "ori6_maximum_spread_progress"
                        ) for row in rows] or [0.0]
                    ),
                }

            candidate_metrics = metrics(candidate_rows)
            parent_metrics = metrics(parent_prefix)
            result["per_seed"][str(seed)] = {
                "available": bool(common),
                "parent_atomic_endpoint": parent_endpoint,
                "common_prefix_episode": common,
                "candidate": candidate_metrics,
                "parent_RC": parent_metrics,
                "candidate_minus_parent_RC": {
                    field: candidate_metrics[field] - parent_metrics[field]
                    for field in (
                        "constructive_proxy_episodes",
                        "valid_joint_or_strict_episodes",
                        "strict_success_episodes",
                        "maximum_spread_progress",
                    )
                },
            }
        result["parent_requested_endpoint_available"] = all(
            row.get("available") is True
            and row.get("parent_atomic_endpoint", 0) >= episode
            for row in result["per_seed"].values()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["diagnostic_error"] = repr(error)
    return result


def early_gate_decision(
    protocol_name: str, task_name: str, rows_by_seed: dict[int, list[dict]],
    episode: int, candidate_mode: str | None = None,
    reference_rows_by_seed: dict[int, list[dict]] | None = None,
) -> dict:
    decision = {
        "schema_version": 1,
        "protocol": protocol_name,
        "task": task_name,
        "checkpoint_episode": episode,
        "candidate_mode": candidate_mode,
        "evaluated_utc": utc_now(),
        "stop": False,
        "reason": "continue",
        "per_seed": {},
    }
    rejected_crossing_cohort_shape_valid = True
    rejected_family_gate = (
        (
            protocol_name == "crossing_continuation_diagnostic"
            and candidate_mode == P0_CROSS2_REJECTED_CONTINUATION_MODE
        )
        or (
            protocol_name == "spectral_subspace_diagnostic"
            and candidate_mode
            == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        )
        or (
            protocol_name == "spectral_nonregression"
            and candidate_mode
            == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        )
    )
    if rejected_family_gate:
        expected_seed_set = (
            {3301, 3302, 3303}
            if protocol_name == "spectral_nonregression"
            else {3201, 3202, 3203}
        )
        rejected_crossing_cohort_shape_valid = bool(
            set(rows_by_seed) == expected_seed_set
            and all(len(rows) >= episode for rows in rows_by_seed.values())
            and all(
                [
                    int(float(row.get("episode", -1)))
                    for row in rows[:episode]
                ] == list(range(1, episode + 1))
                for rows in rows_by_seed.values()
            )
        )
        decision["cohort_shape_valid"] = (
            rejected_crossing_cohort_shape_valid
        )
        decision["preregistered_gate_specification"] = (
            spectral_nonregression_gate_specification()
            if protocol_name == "spectral_nonregression"
            else spectral_subspace_gate_specification()
            if candidate_mode
            == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
            else crossing_continuation_gate_specification()
        )
    for seed, rows in rows_by_seed.items():
        prefix = rows[:episode]
        tail = prefix[-min(256, len(prefix)):]
        first512 = prefix[:512]
        second512 = prefix[512:1024]
        tail_spread_errors = sorted(
            float_field(row, "spread_error", math.inf) for row in tail
        )
        tail_exact_median_spread_error = (
            0.5 * (
                tail_spread_errors[(len(tail_spread_errors) - 1) // 2]
                + tail_spread_errors[len(tail_spread_errors) // 2]
            )
            if tail_spread_errors else math.inf
        )
        def plane_switch_proxy(row: dict) -> bool:
            return (
                float_field(row, "flatness_ratio", math.inf) <= 0.5
                and float_field(row, "spread_error", math.inf) <= 0.1
            )

        def valid_joint_target(row: dict) -> bool:
            return (
                float_field(row, "flatness_ratio", math.inf) <= 0.1
                and float_field(row, "spread_error", math.inf) <= 0.05
                and float_field(row, "self_crossing") == 0.0
                and float_field(row, "correction_failed") == 0.0
                and float_field(row, "no_modes") == 0.0
            )
        decision["per_seed"][str(seed)] = {
            "maximum_spread_progress": max(
                [float_field(row, "ori6_maximum_spread_progress") for row in prefix] or [0.0]
            ),
            "positive_frontier_episodes": sum(
                float_field(row, "ori6_total_spread_frontier_reward") > 1.0e-12
                for row in prefix
            ),
            "minimum_terminal_flatness_ratio": min(
                [float_field(row, "flatness_ratio", math.inf) for row in prefix] or [math.inf]
            ),
            "minimum_terminal_joint_cost": min(
                [
                    float_field(row, "flatness_ratio", math.inf)
                    + float_field(row, "spread_error", math.inf) ** 2
                    for row in prefix
                ] or [math.inf]
            ),
            "tail_success_rate": sum(float_field(row, "success") for row in tail) / max(len(tail), 1),
            "success_episodes": sum(
                float_field(row, "success") > 0.0 for row in prefix
            ),
            "tail_crossing_rate": sum(float_field(row, "self_crossing") for row in tail) / max(len(tail), 1),
            "tail_wrong_flat_rate": sum(
                float_field(row, "wrong_flat") for row in tail
            ) / max(len(tail), 1),
            "tail_timeout_rate": sum(float_field(row, "timeout") for row in tail) / max(len(tail), 1),
            "tail_median_flatness_ratio": sorted(
                float_field(row, "flatness_ratio", math.inf) for row in tail
            )[len(tail) // 2] if tail else math.inf,
            "tail_median_spread_error": sorted(
                float_field(row, "spread_error", math.inf) for row in tail
            )[len(tail) // 2] if tail else math.inf,
            "tail_exact_median_spread_error": tail_exact_median_spread_error,
            "tail_plane_switch_endpoint_proxy_episodes": sum(
                plane_switch_proxy(row) for row in tail
            ),
            "prefix_plane_switch_endpoint_proxy_episodes": sum(
                plane_switch_proxy(row) for row in prefix
            ),
            "tail_valid_joint_target_episodes": sum(
                valid_joint_target(row) for row in tail
            ),
            "joint_milestone_episodes": sum(
                float_field(row, "ori6_total_joint_milestone_reward") > 1.0e-12
                for row in prefix
            ),
            "first512_good_spread_endpoint_episodes": sum(
                float_field(row, "spread_error", math.inf) <= 0.1
                for row in first512
            ),
            "second512_good_spread_endpoint_episodes": sum(
                float_field(row, "spread_error", math.inf) <= 0.1
                for row in second512
            ),
            "tail_good_spread_endpoint_episodes": sum(
                float_field(row, "spread_error", math.inf) <= 0.1
                for row in tail
            ),
            "tail_good_spread_endpoint_rate": sum(
                float_field(row, "spread_error", math.inf) <= 0.1
                for row in tail
            ) / max(len(tail), 1),
            "tail_strong_spread_progress_rate": sum(
                float_field(row, "ori6_maximum_spread_progress") >= 0.8
                for row in tail
            ) / max(len(tail), 1),
            "reward_integrity": (
                bool(prefix)
                and all(
                    all(name in row for name in (
                        "ori6_reward_decomposition_error",
                        "ori6_maximum_abs_step_reward_decomposition_error",
                        "ori6_nonfinite_reward_component_count",
                    ))
                    for row in prefix
                )
                and max(
                    abs(float_field(row, "ori6_reward_decomposition_error"))
                    for row in prefix
                ) <= 1.0e-10
                and max(
                    abs(float_field(
                        row, "ori6_maximum_abs_step_reward_decomposition_error"
                    ))
                    for row in prefix
                ) <= 1.0e-10
                and sum(
                    float_field(row, "ori6_nonfinite_reward_component_count")
                    for row in prefix
                ) == 0.0
            ),
        }
        if rejected_family_gate:
            metric = decision["per_seed"][str(seed)]

            def rejected_counts(sample: list[dict]) -> dict:
                rejected = int(sum(
                    float_field(row, "ori6_rejected_crossing_count")
                    for row in sample
                ))
                streaks = int(sum(
                    float_field(
                        row, "ori6_rejected_crossing_streak_count"
                    )
                    for row in sample
                ))
                escapes = int(sum(
                    float_field(
                        row,
                        "ori6_escape_after_rejected_crossing_count",
                    )
                    for row in sample
                ))
                steps = int(sum(float_field(row, "steps") for row in sample))
                return {
                    "rejected_count": rejected,
                    "streak_count": streaks,
                    "escape_count": escapes,
                    "step_count": steps,
                    "rejected_action_rate": float(
                        rejected / max(steps, 1)
                    ),
                    "escape_rate": (
                        float(escapes / streaks) if streaks else None
                    ),
                }

            def candidate_constructive(row: dict) -> bool:
                if not (
                    float_field(row, "self_crossing") == 0.0
                    and float_field(row, "correction_failed") == 0.0
                    and float_field(row, "no_modes") == 0.0
                ):
                    return False
                flatness = float_field(
                    row, "flatness_ratio", math.inf
                )
                spread_error = float_field(
                    row, "spread_error", math.inf
                )
                if task_name == "torus2":
                    return flatness + spread_error ** 2 <= 0.5
                return flatness <= 0.5 and spread_error <= 0.1

            required_candidate_fields = (
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
                "ori6_maximum_abs_rejected_crossing_reward_"
                "decomposition_error",
            )
            rejected_integrity = bool(prefix) and all(
                all(field in row for field in required_candidate_fields)
                and row.get("ori6_candidate")
                == REJECTED_CROSSING_CANDIDATE_NAME
                and row.get("ori6_candidate_schema_version")
                == REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                and row.get("ori6_crossing_policy")
                == REJECTED_CROSSING_POLICY
                and all(
                    float_field(row, field).is_integer()
                    for field in (
                        "ori6_rejected_crossing_count",
                        "ori6_rejected_crossing_streak_count",
                        "ori6_escape_after_rejected_crossing_count",
                        "ori6_unescaped_rejected_crossing_streak_count",
                        "ori6_max_consecutive_rejected_crossings",
                        "ori6_success_after_rejected_crossing",
                    )
                )
                and 0.0 <= float_field(
                    row, "ori6_rejected_crossing_count"
                ) <= float_field(row, "steps")
                and 0.0 <= float_field(
                    row, "ori6_escape_after_rejected_crossing_count"
                ) <= float_field(
                    row, "ori6_rejected_crossing_streak_count"
                ) <= float_field(row, "ori6_rejected_crossing_count")
                and math.isclose(
                    float_field(row, "ori6_rejected_crossing_rate"),
                    float_field(row, "ori6_rejected_crossing_count")
                    / max(float_field(row, "steps"), 1.0),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    float_field(
                        row, "ori6_rejected_crossing_escape_rate"
                    ),
                    float_field(
                        row,
                        "ori6_escape_after_rejected_crossing_count",
                    )
                    / max(
                        float_field(
                            row, "ori6_rejected_crossing_streak_count"
                        ),
                        1.0,
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    float_field(
                        row,
                        "ori6_unescaped_rejected_crossing_streak_count",
                    ),
                    float_field(
                        row, "ori6_rejected_crossing_streak_count"
                    )
                    - float_field(
                        row,
                        "ori6_escape_after_rejected_crossing_count",
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and 0.0 <= float_field(
                    row, "ori6_max_consecutive_rejected_crossings"
                ) <= float_field(row, "ori6_rejected_crossing_count")
                and float_field(
                    row, "ori6_success_after_rejected_crossing"
                ) in (0.0, 1.0)
                and abs(float_field(
                    row,
                    "ori6_maximum_abs_rejected_crossing_reward_"
                    "decomposition_error",
                    math.inf,
                )) <= 1.0e-10
                and math.isfinite(float_field(
                    row, "ori6_total_rejected_crossing_reward", math.inf
                ))
                and float_field(row, "self_crossing") == 0.0
                for row in prefix
            )
            prefix_counts = rejected_counts(prefix)
            tail_counts = rejected_counts(tail)
            metric.update({
                "prefix_rejected_crossing": prefix_counts,
                "tail256_rejected_crossing": tail_counts,
                "prefix_constructive_proxy_episodes": sum(
                    candidate_constructive(row) for row in prefix
                ),
                "tail256_constructive_proxy_episodes": sum(
                    candidate_constructive(row) for row in tail
                ),
                "prefix_valid_joint_target_episodes": sum(
                    valid_joint_target(row) for row in prefix
                ),
                "no_reject_torus_frontier_joint_eligible": bool(
                    task_name == "torus2"
                    and prefix_counts["rejected_count"] == 0
                    and metric["positive_frontier_episodes"] > 0
                    and metric["minimum_terminal_joint_cost"] <= 0.5
                ),
                "rejected_crossing_integrity": rejected_integrity,
            })
            if candidate_mode == (
                P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
            ):
                spectral_fields = (
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
                    "ori6_spectral_maximum_abs_step_reward_"
                    "decomposition_error",
                    "ori6_spectral_nonfinite_component_count",
                )

                def spectral_row_integrity(row: dict) -> bool:
                    if not all(field in row for field in spectral_fields):
                        return False
                    numeric_fields = (
                        "ori6_spectral_total_progress_reward",
                        "ori6_spectral_total_rollback_reward",
                        "ori6_spectral_total_reward",
                        "ori6_spectral_cumulative_reward",
                        "ori6_spectral_terminal_potential",
                        "ori6_spectral_terminal_delta",
                        "ori6_spectral_maximum_potential",
                        "ori6_spectral_maximum_delta",
                    )
                    return bool(
                        row.get("ori6_spectral_candidate")
                        == SPECTRAL_SUBSPACE_CANDIDATE_NAME
                        and row.get(
                            "ori6_spectral_candidate_schema_version"
                        ) == SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
                        and row.get("ori6_spectral_formula_sha256")
                        == SPECTRAL_SUBSPACE_FORMULA_SHA256
                        and row.get("ori6_spectral_parent_candidate")
                        == REJECTED_CROSSING_CANDIDATE_NAME
                        and values_equal(
                            float_field(row, "ori6_spectral_weight"),
                            SPECTRAL_SUBSPACE_WEIGHT,
                        )
                        and values_equal(
                            float_field(
                                row, "ori6_spectral_relative_epsilon"
                            ),
                            SPECTRAL_SUBSPACE_RELATIVE_EPSILON,
                        )
                        and all(
                            math.isfinite(float_field(row, field, math.inf))
                            for field in numeric_fields
                        )
                        and abs(float_field(
                            row,
                            "ori6_spectral_reward_decomposition_error",
                            math.inf,
                        )) <= 1.0e-10
                        and abs(float_field(
                            row,
                            "ori6_spectral_maximum_abs_step_reward_"
                            "decomposition_error",
                            math.inf,
                        )) <= 1.0e-10
                        and float_field(
                            row,
                            "ori6_spectral_nonfinite_component_count",
                            math.inf,
                        ) == 0.0
                        and math.isclose(
                            float_field(row, "ori6_spectral_total_reward"),
                            float_field(
                                row, "ori6_spectral_total_progress_reward"
                            ) + float_field(
                                row, "ori6_spectral_total_rollback_reward"
                            ),
                            rel_tol=0.0,
                            abs_tol=1.0e-10,
                        )
                        and math.isclose(
                            float_field(row, "ori6_spectral_total_reward"),
                            float_field(
                                row, "ori6_spectral_cumulative_reward"
                            ),
                            rel_tol=0.0,
                            abs_tol=1.0e-10,
                        )
                        and (
                            row.get("termination_reason") not in {
                                "wrong_flat", "correction_failed",
                                "no_modes", "self_crossing",
                            }
                            or abs(float_field(
                                row, "ori6_spectral_cumulative_reward"
                            )) <= 1.0e-10
                        )
                        and 0.0 <= float_field(
                            row, "ori6_spectral_maximum_potential"
                        ) <= 1.0 + 1.0e-12
                        and 0.0 <= float_field(
                            row, "ori6_spectral_maximum_delta"
                        ) <= 1.0 + 1.0e-12
                    )

                spectral_integrity = bool(prefix) and all(
                    spectral_row_integrity(row) for row in prefix
                )
                metric.update({
                    "spectral_integrity": spectral_integrity,
                    "maximum_spectral_potential": max(
                        [float_field(
                            row, "ori6_spectral_maximum_potential"
                        ) for row in prefix] or [0.0]
                    ),
                    "maximum_spectral_delta": max(
                        [float_field(
                            row, "ori6_spectral_maximum_delta"
                        ) for row in prefix] or [0.0]
                    ),
                    "positive_spectral_peak_episodes": sum(
                        float_field(
                            row, "ori6_spectral_maximum_potential"
                        ) > 1.0e-12
                        for row in prefix
                    ),
                    "spectral_delta_gt_1e_6_episodes": sum(
                        float_field(
                            row, "ori6_spectral_maximum_delta"
                        ) > 1.0e-6
                        for row in prefix
                    ),
                })
    metrics = list(decision["per_seed"].values())
    if protocol_name == "rescue" and task_name == "torus2":
        if episode == 512 and all(
            row["maximum_spread_progress"] <= 1.0e-12
            and row["positive_frontier_episodes"] == 0 for row in metrics
        ):
            decision.update({"stop": True, "reason": "torus_zero_frontier_all_seeds"})
        elif episode == 1024 and all(
            row["minimum_terminal_flatness_ratio"] >= 1.0
            and row["minimum_terminal_joint_cost"] >= 0.95 for row in metrics
        ):
            decision.update({"stop": True, "reason": "torus_no_flatness_or_joint_frontier"})
    if protocol_name == "rescue" and task_name == "water1" and episode == 1024:
        safe_timeout = all(
            row["tail_timeout_rate"] >= 0.8
            and row["tail_median_flatness_ratio"] >= 1.0
            and row["tail_median_spread_error"] <= 0.05
            and row["joint_milestone_episodes"] == 0
            for row in metrics
        )
        if safe_timeout:
            decision.update({"stop": True, "reason": "water1_safe_timeout_attractor_trigger_P1"})
    if (
        protocol_name == "entropy_diagnostic"
        and candidate_mode == P0_CROSS2_ENTROPY3_MODE
        and task_name == "water1"
        and episode == 512
        and all(
            row["success_episodes"] == 0
            and row["tail_crossing_rate"] >= 0.95
            for row in metrics
        )
    ):
        decision.update({"stop": True, "reason": "all_seed_crossing_collapse"})
    if (
        protocol_name == "entropy_diagnostic"
        and candidate_mode == P0_CROSS2_ENTROPY3_MODE
        and task_name == "water1"
        and episode == 1024
    ):
        median_spread_pass_count = sum(
            row["tail_median_spread_error"] <= 0.2 for row in metrics
        )
        constructive_rate_pass_count = sum(
            row["tail_good_spread_endpoint_rate"] >= 0.05
            or row["tail_strong_spread_progress_rate"] >= 0.25
            for row in metrics
        )
        crossing_pass_count = sum(
            row["tail_crossing_rate"] <= 0.75 for row in metrics
        )
        reward_integrity_pass = all(
            row["reward_integrity"] is True for row in metrics
        )
        wide_spread_zero_success_count = sum(
            row["success_episodes"] == 0
            and row["tail_median_spread_error"] >= 0.30
            for row in metrics
        )
        screen_pass = (
            median_spread_pass_count >= 2
            and constructive_rate_pass_count >= 2
            and crossing_pass_count >= 2
            and reward_integrity_pass
        )
        formal = {
            "pass": screen_pass,
            "median_spread_error_le_0_20_seed_count": median_spread_pass_count,
            "constructive_rate_seed_count": constructive_rate_pass_count,
            "constructive_rate_definition": (
                "tail256 spread_error<=0.1 rate>=0.05 OR "
                "ori6_maximum_spread_progress>=0.8 rate>=0.25"
            ),
            "tail_crossing_rate_le_0_75_seed_count": crossing_pass_count,
            "reward_integrity_pass": reward_integrity_pass,
            "wide_spread_zero_success_seed_count": wide_spread_zero_success_count,
            "reject": wide_spread_zero_success_count >= 2,
        }
        decision["formal_entropy_diagnostic_gate"] = formal
        if not reward_integrity_pass:
            decision.update({"stop": True, "reason": "reward_integrity_failure"})
        elif formal["reject"]:
            decision.update({
                "stop": True,
                "reason": "wide_spread_zero_success_reject",
            })
        elif screen_pass and not decision["stop"]:
            decision["reason"] = "entropy_diagnostic_gate_pass"
        elif not decision["stop"]:
            decision["reason"] = "entropy_diagnostic_inconclusive"
    if (
        protocol_name == "width_diagnostic"
        and candidate_mode == P0_CROSS2_WIDTH2_MODE
        and task_name == "water1"
        and episode in (512, 1024, 2048)
    ):
        reward_integrity_pass = bool(metrics) and all(
            row["reward_integrity"] is True for row in metrics
        )
        all_seed_crossing_collapse = bool(metrics) and all(
            row["tail_crossing_rate"] >= 0.95 for row in metrics
        )
        decision["orientation_proxy"] = {
            "definition": "endpoint flatness_ratio<=0.5 AND spread_error<=0.1",
            "tail_window": 256,
            "interpretation": "orientation-inferred water1 yz/plane-switch surrogate",
            "is_direct_PCA_plane_normal": False,
        }
        if episode == 512:
            decision["formal_width_diagnostic_gate"] = {
                "phase": "catastrophe_only",
                "reward_integrity_pass": reward_integrity_pass,
                "all_seed_tail_crossing_rate_ge_0_95": all_seed_crossing_collapse,
            }
            if not reward_integrity_pass:
                decision.update({"stop": True, "reason": "reward_integrity_failure"})
            elif all_seed_crossing_collapse:
                decision.update({
                    "stop": True,
                    "reason": "width_diagnostic_all_seed_crossing_collapse",
                })
        elif episode == 1024:
            plane_proxy_seed_count = sum(
                row["tail_plane_switch_endpoint_proxy_episodes"] >= 1
                for row in metrics
            )
            valid_joint_or_strict_seed_count = sum(
                row["tail_valid_joint_target_episodes"] >= 1
                or row["success_episodes"] >= 1
                for row in metrics
            )
            wide_spread_seed_count = sum(
                row["tail_exact_median_spread_error"] >= 0.30
                for row in metrics
            )
            screen_pass = bool(
                len(metrics) == 3
                and plane_proxy_seed_count >= 2
                and valid_joint_or_strict_seed_count >= 1
                and reward_integrity_pass
            )
            formal = {
                "phase": "continue_to_2048_decision",
                "pass": screen_pass,
                "tail_plane_switch_proxy_positive_seed_count": (
                    plane_proxy_seed_count
                ),
                "tail_plane_switch_proxy_required_seed_count": 2,
                "tail_valid_F_le_0_1_e_le_0_05_or_strict_seed_count": (
                    valid_joint_or_strict_seed_count
                ),
                "valid_joint_or_strict_required_seed_count": 1,
                "all_seed_tail_crossing_rate_ge_0_95": (
                    all_seed_crossing_collapse
                ),
                "tail_exact_median_spread_error_ge_0_30_seed_count": (
                    wide_spread_seed_count
                ),
                "reward_integrity_pass": reward_integrity_pass,
                "reject": bool(
                    all_seed_crossing_collapse or wide_spread_seed_count >= 2
                ),
            }
            decision["formal_width_diagnostic_gate"] = formal
            if not reward_integrity_pass:
                decision.update({"stop": True, "reason": "reward_integrity_failure"})
            elif all_seed_crossing_collapse:
                decision.update({
                    "stop": True,
                    "reason": "width_diagnostic_all_seed_crossing_collapse",
                })
            elif wide_spread_seed_count >= 2:
                decision.update({
                    "stop": True,
                    "reason": "width_diagnostic_wide_spread_reject",
                })
            elif screen_pass:
                decision["reason"] = "width_diagnostic_continue_to_2048"
            else:
                decision.update({
                    "stop": True,
                    "reason": "width_diagnostic_1024_gate_not_met",
                })
        else:
            strict_hit_count = sum(
                row["success_episodes"] >= 1 for row in metrics
            )
            constructive_prefix_seed_count = sum(
                row["prefix_plane_switch_endpoint_proxy_episodes"] >= 1
                for row in metrics
            )
            formal_pass = bool(
                len(metrics) == 3
                and strict_hit_count >= 1
                and constructive_prefix_seed_count >= 2
                and reward_integrity_pass
            )
            decision["formal_width_diagnostic_gate"] = {
                "phase": "adoption_decision",
                "pass": formal_pass,
                "strict_hit_seed_count": strict_hit_count,
                "strict_hit_required_seed_count": 1,
                "constructive_prefix_seed_count": constructive_prefix_seed_count,
                "constructive_prefix_required_seed_count": 2,
                "constructive_prefix_definition": (
                    "at least one endpoint with flatness_ratio<=0.5 and "
                    "spread_error<=0.1 by episode 2048"
                ),
                "reward_integrity_pass": reward_integrity_pass,
            }
            if not reward_integrity_pass:
                decision.update({"stop": True, "reason": "reward_integrity_failure"})
            elif formal_pass:
                decision["reason"] = "width_diagnostic_adoption_gate_pass"
            else:
                decision.update({
                    "stop": True,
                    "reason": "width_diagnostic_adoption_gate_reject",
                })
    if (
        protocol_name == "crossing_continuation_diagnostic"
        and candidate_mode == P0_CROSS2_REJECTED_CONTINUATION_MODE
        and task_name in ("torus2", "water1")
        and episode in (512, 1024, 2048)
    ):
        reward_integrity_pass = bool(metrics) and all(
            row["reward_integrity"] is True
            and row["rejected_crossing_integrity"] is True
            for row in metrics
        )
        constructive_seed_count = sum(
            row["prefix_constructive_proxy_episodes"] >= 1
            for row in metrics
        )
        valid_joint_or_strict_seed_count = sum(
            row["prefix_valid_joint_target_episodes"] >= 1
            or row["success_episodes"] >= 1
            for row in metrics
        )
        strict_hit_seed_count = sum(
            row["success_episodes"] >= 1 for row in metrics
        )
        common = {
            "cohort_shape_valid": rejected_crossing_cohort_shape_valid,
            "reward_and_transition_integrity_pass": (
                reward_integrity_pass
            ),
            "constructive_proxy": {
                "torus2": (
                    "valid endpoint flatness_ratio + spread_error^2 <= 0.5"
                ),
                "water1": (
                    "valid endpoint flatness_ratio <= 0.5 and "
                    "spread_error <= 0.1"
                ),
            }[task_name],
            "constructive_prefix_seed_count": constructive_seed_count,
            "valid_F_le_0_1_e_le_0_05_or_strict_seed_count": (
                valid_joint_or_strict_seed_count
            ),
            "strict_hit_seed_count": strict_hit_seed_count,
        }
        if episode == 512:
            reject_rate_pass = bool(metrics) and all(
                row["prefix_rejected_crossing"][
                    "rejected_action_rate"
                ] <= 0.25
                for row in metrics
            )
            escape_or_torus_frontier_seed_count = sum(
                row["prefix_rejected_crossing"]["escape_count"] > 0
                or row["no_reject_torus_frontier_joint_eligible"]
                for row in metrics
            )
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and reject_rate_pass
                and escape_or_torus_frontier_seed_count >= 2
                and reward_integrity_pass
            )
            formal = {
                **common,
                "phase": "bounded_safety_and_escape_screen",
                "pass": formal_pass,
                "all_seed_rejected_action_rate_le_0_25": (
                    reject_rate_pass
                ),
                "escape_or_torus_no_reject_frontier_seed_count": (
                    escape_or_torus_frontier_seed_count
                ),
                "escape_or_torus_frontier_required_seed_count": 2,
            }
            decision["formal_crossing_continuation_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "crossing_continuation_continue_to_1024"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "crossing_continuation_512_gate_not_met",
                })
        elif episode == 1024:
            stable_seed_count = sum(
                row["tail256_rejected_crossing"][
                    "rejected_action_rate"
                ] <= 0.10
                and (
                    (
                        row["tail256_rejected_crossing"]["escape_rate"]
                        is not None
                        and row["tail256_rejected_crossing"][
                            "escape_rate"
                        ] >= 0.25
                    )
                    or row["no_reject_torus_frontier_joint_eligible"]
                )
                for row in metrics
            )
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and stable_seed_count >= 2
                and constructive_seed_count >= 2
                and valid_joint_or_strict_seed_count >= 1
                and reward_integrity_pass
            )
            formal = {
                **common,
                "phase": "constructive_continuation_screen",
                "pass": formal_pass,
                "tail256_reject_rate_le_0_10_and_escape_rate_ge_0_25_"
                "or_torus_no_reject_frontier_seed_count": stable_seed_count,
                "stable_continuation_required_seed_count": 2,
                "constructive_prefix_required_seed_count": 2,
                "valid_joint_or_strict_required_seed_count": 1,
            }
            decision["formal_crossing_continuation_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "crossing_continuation_continue_to_2048"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "crossing_continuation_1024_gate_not_met",
                })
        else:
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and strict_hit_seed_count >= 1
                and constructive_seed_count >= 2
                and reward_integrity_pass
            )
            formal = {
                **common,
                "phase": "finalize_for_heldout_analysis",
                "pass": formal_pass,
                "strict_hit_required_seed_count": 1,
                "constructive_prefix_required_seed_count": 2,
                "heldout_not_inspected_during_training_gate": True,
            }
            decision["formal_crossing_continuation_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "crossing_continuation_finalize_heldout"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "crossing_continuation_2048_gate_reject",
                })
    if (
        protocol_name == "spectral_subspace_diagnostic"
        and candidate_mode
        == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        and task_name in ("torus2", "water1")
        and episode in (512, 1024, 2048)
    ):
        integrity_pass = bool(metrics) and all(
            row["reward_integrity"] is True
            and row["rejected_crossing_integrity"] is True
            and row.get("spectral_integrity") is True
            for row in metrics
        )
        constructive_seed_count = sum(
            row["prefix_constructive_proxy_episodes"] >= 1
            for row in metrics
        )
        valid_joint_or_strict_seed_count = sum(
            row["prefix_valid_joint_target_episodes"] >= 1
            or row["success_episodes"] >= 1
            for row in metrics
        )
        strict_hit_seed_count = sum(
            row["success_episodes"] >= 1 for row in metrics
        )
        positive_peak_seed_count = sum(
            row.get("maximum_spectral_potential", 0.0) > 1.0e-12
            for row in metrics
        )
        delta_evidence_seed_count = sum(
            row.get("maximum_spectral_delta", 0.0) > 1.0e-6
            for row in metrics
        )
        common = {
            "cohort_shape_valid": rejected_crossing_cohort_shape_valid,
            "reward_spectral_and_transition_integrity_pass": integrity_pass,
            "constructive_proxy": (
                spectral_subspace_gate_specification()[
                    "constructive_proxy_by_task"
                ][task_name]
            ),
            "constructive_prefix_seed_count": constructive_seed_count,
            "valid_F_le_0_1_e_le_0_05_or_strict_seed_count": (
                valid_joint_or_strict_seed_count
            ),
            "strict_hit_seed_count": strict_hit_seed_count,
            "positive_spectral_peak_seed_count": positive_peak_seed_count,
            "spectral_delta_gt_1e_6_seed_count": delta_evidence_seed_count,
        }
        if episode == 512:
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and integrity_pass
            )
            formal = {
                **common,
                "phase": "pathology_and_integrity_only",
                "pass": formal_pass,
                "performance_predicate_count": 0,
                "valid_cohort_always_continues": True,
            }
            decision["formal_spectral_subspace_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "spectral_subspace_integrity_continue_to_1024"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "spectral_subspace_512_integrity_failure",
                })
        elif episode == 1024:
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and integrity_pass
                and constructive_seed_count >= 2
                and valid_joint_or_strict_seed_count >= 1
                and positive_peak_seed_count >= 2
                and delta_evidence_seed_count >= 2
            )
            formal = {
                **common,
                "phase": "constructive_spectral_screen",
                "pass": formal_pass,
                "constructive_prefix_required_seed_count": 2,
                "valid_joint_or_strict_required_seed_count": 1,
                "positive_spectral_peak_required_seed_count": 2,
                "spectral_peak_threshold": 1.0e-12,
                "spectral_delta_required_seed_count": 2,
                "spectral_delta_threshold": 1.0e-6,
                "parent_common_prefix_comparison": (
                    spectral_parent_common_prefix_comparison(
                        rows_by_seed, task_name, episode
                    )
                ),
                "parent_comparison_gating": False,
            }
            decision["formal_spectral_subspace_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "spectral_subspace_continue_to_2048"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "spectral_subspace_1024_gate_not_met",
                })
        else:
            formal_pass = bool(
                len(metrics) == 3
                and rejected_crossing_cohort_shape_valid
                and integrity_pass
                and strict_hit_seed_count >= 1
                and constructive_seed_count >= 2
            )
            formal = {
                **common,
                "phase": "finalize_for_heldout_analysis",
                "pass": formal_pass,
                "strict_hit_required_seed_count": 1,
                "constructive_prefix_required_seed_count": 2,
                "heldout_not_inspected_during_training_gate": True,
            }
            decision["formal_spectral_subspace_gate"] = formal
            if formal_pass:
                decision["reason"] = (
                    "spectral_subspace_finalize_heldout"
                )
            else:
                decision.update({
                    "stop": True,
                    "reason": "spectral_subspace_2048_gate_reject",
                })
    if (
        protocol_name == "spectral_nonregression"
        and candidate_mode
        == P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        and task_name in ("five_d2", "five_d3", "watersqrt2")
        and episode == 512
    ):
        expected_seeds = {3301, 3302, 3303}
        reference_rows_by_seed = reference_rows_by_seed or {}
        reference_shape_valid = bool(
            set(reference_rows_by_seed) == expected_seeds
            and all(
                len(rows) >= episode
                and [
                    int(float(row.get("episode", -1)))
                    for row in rows[:episode]
                ] == list(range(1, episode + 1))
                for rows in reference_rows_by_seed.values()
            )
        )
        reference_metrics = {}
        for seed, rows in reference_rows_by_seed.items():
            prefix = rows[:episode]
            tail = prefix[-min(256, len(prefix)):]
            required_fields = (
                "ori5_reward_decomposition_error",
                "ori5_maximum_abs_step_reward_decomposition_error",
                "ori5_nonfinite_reward_component_count",
            )
            integrity = bool(prefix) and all(
                all(field in row for field in required_fields)
                and abs(float_field(
                    row, "ori5_reward_decomposition_error", math.inf
                )) <= 1.0e-10
                and abs(float_field(
                    row,
                    "ori5_maximum_abs_step_reward_decomposition_error",
                    math.inf,
                )) <= 1.0e-10
                and float_field(
                    row, "ori5_nonfinite_reward_component_count", math.inf
                ) == 0.0
                for row in prefix
            )
            reference_metrics[str(seed)] = {
                "success_episodes": sum(
                    float_field(row, "success") > 0.0 for row in prefix
                ),
                "tail256_success_rate": sum(
                    float_field(row, "success") for row in tail
                ) / max(len(tail), 1),
                "tail256_crossing_rate": sum(
                    float_field(row, "self_crossing") for row in tail
                ) / max(len(tail), 1),
                "reward_integrity": integrity,
            }
        decision["reference_per_seed"] = reference_metrics
        candidate_integrity = bool(
            rejected_crossing_cohort_shape_valid
            and len(decision["per_seed"]) == 3
            and all(
                row["reward_integrity"] is True
                and row.get("rejected_crossing_integrity") is True
                and row.get("spectral_integrity") is True
                for row in decision["per_seed"].values()
            )
        )
        reference_integrity = bool(
            reference_shape_valid
            and len(reference_metrics) == 3
            and all(
                row["reward_integrity"] is True
                for row in reference_metrics.values()
            )
        )
        catastrophic = bool(
            len(decision["per_seed"]) == 3
            and all(
                row["success_episodes"] == 0
                and row["tail256_rejected_crossing"][
                    "rejected_action_rate"
                ] >= 0.95
                for row in decision["per_seed"].values()
            )
        )
        formal_pass = bool(
            candidate_integrity and reference_integrity and not catastrophic
        )
        decision["formal_spectral_nonregression_gate"] = {
            "phase": "catastrophic_regression_and_integrity_only",
            "pass": formal_pass,
            "candidate_cohort_shape_valid": (
                rejected_crossing_cohort_shape_valid
            ),
            "reference_cohort_shape_valid": reference_shape_valid,
            "candidate_reward_transition_spectral_integrity_pass": (
                candidate_integrity
            ),
            "reference_reward_integrity_pass": reference_integrity,
            "all_candidate_seed_zero_success_and_tail256_rejected_action_"
            "rate_ge_0_95": catastrophic,
            "performance_noninferiority_tested": False,
        }
        if formal_pass:
            decision["reason"] = (
                "spectral_nonregression_continue_to_final_2048"
            )
        else:
            decision.update({
                "stop": True,
                "reason": (
                    "spectral_nonregression_integrity_failure"
                    if not (candidate_integrity and reference_integrity)
                    else "spectral_nonregression_catastrophic_reject"
                ),
            })
    # Legacy regression uses a deliberately conservative all-seed stop.
    if protocol_name == "regression" and episode in (512, 1024):
        if all(
            row["tail_success_rate"] == 0.0 and row["tail_crossing_rate"] >= 0.95
            for row in metrics
        ):
            decision.update({"stop": True, "reason": "all_seed_crossing_collapse"})
    return decision


def cohort_checkpoint_snapshot(
    cohort_items: list[dict], checkpoint: int,
) -> dict | None:
    """Read a gate snapshot from the immutable, all-seed cohort."""

    outputs = [current_output(item) for item in cohort_items]
    endpoints = [
        checkpoint_episode(output) if output is not None else None
        for output in outputs
    ]
    if not all(
        endpoint is not None and endpoint >= checkpoint
        for endpoint in endpoints
    ):
        return None
    rows_by_seed = {
        int(item["job"]["seed"]): read_csv(
            output / "training_history.csv"
        )
        for item, output in zip(cohort_items, outputs)
        if item["job"]["implementation"] == "ori6"
    }
    expected_seeds = {
        int(item["job"]["seed"])
        for item in cohort_items
        if item["job"]["implementation"] == "ori6"
    }
    if set(rows_by_seed) != expected_seeds:
        raise RuntimeError("All-cohort checkpoint snapshot lost an ori6 seed")
    return {
        "outputs": outputs,
        "checkpoint_endpoints": endpoints,
        "rows_by_seed": rows_by_seed,
    }


def terminate_at_checkpoint(items: list[dict], episode: int, decision: dict) -> None:
    for item in items:
        process = item["process"]
        if process.poll() is None:
            process.terminate()
    for item in items:
        process = item["process"]
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        item["stdout"].close()
        item["stderr"].close()
        effective = current_output(item)
        actual_endpoint = (
            checkpoint_episode(effective) if effective is not None else None
        )
        validation = (
            validate_artifact(
                item["job"], effective, endpoint=actual_endpoint,
            )
            if effective is not None and actual_endpoint is not None else None
        )
        status = dict(item["status"])
        status.update({
            "status": "early_stopped_at_checkpoint",
            "finished_utc": utc_now(),
            "return_code": int(process.returncode),
            "checkpoint_episode": actual_endpoint,
            "cohort_decision_checkpoint_episode": episode,
            "early_gate_decision": decision,
            "effective_output_directory": str(effective) if effective else None,
            "partial_artifact_validation": validation,
        })
        atomic_json(item["status_file"], status)


def finish(item: dict) -> dict:
    process = item["process"]
    item["stdout"].close()
    item["stderr"].close()
    effective, validation = discover_output(item["job"], item["destination"])
    successful = process.returncode == 0 and effective is not None and validation and validation["valid"]
    status = dict(item["status"])
    status.update({
        "status": "complete" if successful else "failed",
        "finished_utc": utc_now(),
        "return_code": int(process.returncode),
        "runner_wall_seconds": time.monotonic() - item["started"],
        "effective_output_directory": str(effective) if effective else None,
        "artifact_validation": validation,
    })
    atomic_json(item["status_file"], status)
    return status


def early_gate_checkpoints(protocol_name: str) -> tuple[int, ...]:
    if protocol_name == "spectral_nonregression":
        return (512,)
    return (
        (512, 1024, 2048)
        if protocol_name in (
            "width_diagnostic",
            "crossing_continuation_diagnostic",
            "spectral_subspace_diagnostic",
        )
        else (512, 1024)
    )


def execute_cohort(jobs: list[tuple[dict, Path]], early_stop: bool) -> int:
    items = []
    for job, destination in jobs:
        print(f"LAUNCH {job['job_id']} -> {destination}", flush=True)
        items.append(launch(job, destination))
    # Keep the immutable cohort membership for checkpoint gates.  A fast job
    # may finish (and therefore leave ``items``) before a slower peer reaches
    # the common checkpoint; dropping it here would silently turn a three-seed
    # gate into a two-seed gate.  ``items`` remains the active-process list.
    cohort_items = list(items)
    failed = 0
    checked = set()
    last_report = 0.0

    def evaluate_available_gates(*, terminate_active: bool) -> bool:
        if not early_stop:
            return False
        checkpoint_schedule = early_gate_checkpoints(
            cohort_items[0]["job"]["protocol"]["name"]
        )
        for checkpoint in checkpoint_schedule:
            if checkpoint in checked:
                continue
            outputs = [current_output(item) for item in cohort_items]
            if not all(
                output is not None
                and checkpoint_episode(output) is not None
                and checkpoint_episode(output) >= checkpoint
                for output in outputs
            ):
                continue
            rows_by_seed = {
                int(item["job"]["seed"]): read_csv(
                    output / "training_history.csv"
                )
                for item, output in zip(cohort_items, outputs)
                if item["job"]["implementation"] == "ori6"
            }
            reference_rows_by_seed = {
                int(item["job"]["seed"]): read_csv(
                    output / "training_history.csv"
                )
                for item, output in zip(cohort_items, outputs)
                if item["job"]["implementation"] == "ori5_reference"
            }
            if cohort_items[0]["job"]["protocol"]["name"] == (
                "spectral_nonregression"
            ):
                decision = early_gate_decision(
                    cohort_items[0]["job"]["protocol"]["name"],
                    cohort_items[0]["job"]["task"]["name"], rows_by_seed,
                    checkpoint,
                    cohort_items[0]["job"]["candidate_mode"],
                    reference_rows_by_seed=reference_rows_by_seed,
                )
            else:
                decision = early_gate_decision(
                    cohort_items[0]["job"]["protocol"]["name"],
                    cohort_items[0]["job"]["task"]["name"], rows_by_seed,
                    checkpoint,
                    cohort_items[0]["job"]["candidate_mode"],
                )
            gate_path = ensure_below_root(
                EXPERIMENT_ROOT / "early_gates"
                / cohort_items[0]["job"]["protocol"]["name"]
                / MODE_SLUGS[
                    cohort_items[0]["job"]["candidate_mode"]
                ]
                / cohort_items[0]["job"]["task"]["name"]
                / f"checkpoint_{checkpoint}.json"
            )
            atomic_json(gate_path, decision)
            checked.add(checkpoint)
            print(
                f"EARLY GATE {gate_path}: {decision['reason']}",
                flush=True,
            )
            if decision["stop"]:
                if terminate_active and items:
                    terminate_at_checkpoint(items, checkpoint, decision)
                return True
        return False

    while items:
        if evaluate_available_gates(terminate_active=True):
            return failed
        for item in list(items):
            if item["process"].poll() is None:
                continue
            result = finish(item)
            items.remove(item)
            if result["status"] != "complete":
                failed += 1
            print(
                f"{result['status'].upper()} {result['job_id']} "
                f"rc={result['return_code']}", flush=True,
            )
        # A fast cohort can finish between the gate check above and process
        # polling.  Re-read finalized checkpoints once before leaving the
        # loop so checkpoint_2048 authority is never lost to that race.
        if not items:
            evaluate_available_gates(terminate_active=False)
        now = time.monotonic()
        if items and now - last_report >= 60:
            print("RUNNING " + ", ".join(item["job"]["job_id"] for item in items), flush=True)
            last_report = now
        if items:
            time.sleep(1)
    return failed


def execute(pending: list[tuple[dict, Path]], early_stop: bool, max_parallel: int) -> int:
    if not pending:
        return 0
    protocol_name = pending[0][0]["protocol"]["name"]
    if early_stop and protocol_name in (
        "rescue", "regression", "entropy_diagnostic", "width_diagnostic",
        "crossing_continuation_diagnostic", "spectral_subspace_diagnostic",
        "spectral_nonregression",
    ):
        cohorts = []
        for task_name in pending[0][0]["protocol"]["tasks"]:
            cohort = [item for item in pending if item[0]["task"]["name"] == task_name]
            if cohort:
                if protocol_name in (
                    "crossing_continuation_diagnostic",
                    "spectral_subspace_diagnostic",
                    "spectral_nonregression",
                ):
                    expected_keys = {
                        (implementation, int(seed))
                        for implementation in pending[0][0]["protocol"][
                            "implementations"
                        ]
                        for seed in pending[0][0]["protocol"]["seeds"]
                    }
                    actual_keys = {
                        (item[0]["implementation"], int(item[0]["seed"]))
                        for item in cohort
                    }
                else:
                    expected_keys = actual_keys = set()
                if expected_keys != actual_keys:
                    raise RuntimeError(
                        "Rejected-crossing family protocols refuse a partial "
                        "pending seed cohort (task/implementation/seed)"
                    )
                cohorts.append(cohort)
        failed = 0
        for cohort in cohorts:
            if len(cohort) > max_parallel:
                raise RuntimeError(
                    f"Early-stop cohort size {len(cohort)} exceeds --max-parallel {max_parallel}"
                )
            failed += execute_cohort(cohort, early_stop=True)
        return failed
    queue = list(pending)
    failed = 0
    while queue:
        cohort = queue[:max_parallel]
        queue = queue[max_parallel:]
        failed += execute_cohort(cohort, early_stop=False)
    return failed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(PROTOCOLS), default="rescue")
    parser.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default=P0_MODE)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-cohort-early-stop", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument(
        "--selection-json", type=Path,
        default=EXPERIMENT_ROOT / "analysis" / "selected_candidate.json",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_parallel < 1 or args.max_parallel > 12:
        raise ValueError("--max-parallel must be between 1 and 12")
    if args.candidate_mode == P0_CROSS2_ENTROPY3_MODE:
        if not args.enable_cohort_early_stop:
            raise ValueError("P0X2E3 requires --enable-cohort-early-stop")
        if args.max_parallel < 3:
            raise ValueError("P0X2E3 requires --max-parallel >= 3 for its fixed cohort")
    if args.candidate_mode == P0_CROSS2_WIDTH2_MODE:
        if not args.enable_cohort_early_stop:
            raise ValueError("P0X2W2 requires --enable-cohort-early-stop")
        if args.max_parallel < 3:
            raise ValueError("P0X2W2 requires --max-parallel >= 3 for its fixed cohort")
    if args.candidate_mode == P0_CROSS2_REJECTED_CONTINUATION_MODE:
        if not args.enable_cohort_early_stop:
            raise ValueError("P0X2RC requires --enable-cohort-early-stop")
        if args.max_parallel < 3:
            raise ValueError(
                "P0X2RC requires --max-parallel >= 3 for its fixed cohort"
            )
    if args.candidate_mode == (
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
    ):
        if not args.enable_cohort_early_stop:
            raise ValueError("P0X2RCS05 requires --enable-cohort-early-stop")
        minimum_parallel = 6 if args.stage == "spectral_nonregression" else 3
        if args.max_parallel < minimum_parallel:
            raise ValueError(
                "P0X2RCS05 requires --max-parallel >= "
                f"{minimum_parallel} for its fixed cohort"
            )
    protocol = PROTOCOLS[args.stage]
    selected_tasks = [name for name in protocol.tasks if name in args.tasks]
    if not selected_tasks:
        raise ValueError("No task remains after applying --tasks")
    jobs = build_jobs(protocol, args.candidate_mode, selected_tasks)
    if args.candidate_mode in (
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        expected_job_keys = {
            (task_name, implementation, seed)
            for task_name in protocol.tasks
            for implementation in protocol.implementations
            for seed in protocol.seeds
        }
        actual_job_keys = {
            (
                job["task"]["name"], job["implementation"],
                int(job["seed"]),
            )
            for job in jobs
        }
        expected_count = len(expected_job_keys)
        if actual_job_keys != expected_job_keys or len(jobs) != expected_count:
            raise RuntimeError(
                "Rejected-crossing family protocol requires its exact fixed "
                "task/implementation/seed cohorts"
            )
    manifest = manifest_path(protocol, args.candidate_mode, jobs)
    print(f"Manifest: {manifest}")
    processes = active_python_processes()
    pending = []
    counts = {}
    states_by_task = {
        task_name: [] for task_name in protocol.tasks
    }
    for job in jobs:
        state, destination, detail = classify(job, args.rerun_incomplete, processes)
        states_by_task[job["task"]["name"]].append(state)
        counts[state] = counts.get(state, 0) + 1
        print(f"{state.upper():16s} {job['job_id']} -> {destination}")
        if state in ("pending", "pending_retry"):
            pending.append((job, destination))
        elif state in ("invalid", "incomplete", "external_active", "running"):
            if state not in ("external_active", "running") and detail:
                print("  " + "; ".join(detail.get("errors", [])))
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if args.candidate_mode in (
        P0_CROSS2_REJECTED_CONTINUATION_MODE,
        P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE,
    ):
        expected_states_per_task = (
            len(protocol.seeds) * len(protocol.implementations)
        )
        for task_name, states in states_by_task.items():
            all_complete = (
                len(states) == expected_states_per_task
                and set(states) == {"complete"}
            )
            all_launchable = (
                len(states) == expected_states_per_task
                and set(states).issubset({
                "pending", "pending_retry",
                })
            )
            if not (all_complete or all_launchable):
                raise RuntimeError(
                    "Rejected-crossing diagnostic refuses mixed "
                    "complete/pending/incomplete state "
                    f"inside fixed cohort {task_name}: {states}"
                )
    if not args.execute:
        print(f"DRY RUN: {len(pending)} job(s) would launch; no training started.")
        return 0
    validate_execution_gate(protocol, args.candidate_mode, args.selection_json.resolve())
    validate_manifest_execution_binding(
        manifest, protocol, args.candidate_mode, jobs
    )
    blockers = sum(counts.get(name, 0) for name in ("invalid", "incomplete", "running", "external_active"))
    if blockers:
        raise RuntimeError("Execution refused while invalid/incomplete/active jobs exist")
    failed = execute(pending, args.enable_cohort_early_stop, args.max_parallel)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
