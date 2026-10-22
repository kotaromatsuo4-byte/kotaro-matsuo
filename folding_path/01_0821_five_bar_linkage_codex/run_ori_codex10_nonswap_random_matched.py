"""Fail-closed launcher for the C10NS1 exact-mechanics uniform controls.

The underlying official harness is intentionally not modified.  Each process
runs exactly one seed with ``--no-resume``.  Its checkpoint interval is fixed
to ``mechanics_budget + 1``: because every episode consumes at least one raw
action proposal, the checkpoint branch is unreachable.  Only an rc=0 trial
with a complete summary and a contiguous exact proposal clock is admissible.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import ori_codex6_spectral_random_baseline as baseline


SCHEMA_VERSION = 1
PROTOCOL = "c10ns1_exact_uniform_selected_endpoint_caps_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE = (SCRIPT_DIR / "ori_codex6_spectral_random_baseline.py").resolve()
C10_MANIFEST = (
    SCRIPT_DIR / "experiments_ori10" / "manifests"
    / "codex10_nonswap_7f1a9303048b.json"
).resolve()
TASK_BUDGETS = {
    "five_d2": 819_200,
    "five_d3": 409_600,
    "watersqrt2": 409_600,
}
SEEDS = tuple(range(6101, 6111))
MAX_RANDOM_PARALLEL = 10
MAX_TOTAL_TRAINERS = 12
DEADLINE_LOCAL = "2026-08-28T22:15:00+09:00"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
TRANSITIVE_SOURCE_NAMES = (
    "ori_codex.py",
    "ori_codex2.py",
    "ori_codex3.py",
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex4_3d_first_discovery.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "ori_codex6.py",
    "ori_codex6_3d.py",
    "ori_codex6_rejected_crossing.py",
    "ori_codex6_rejected_crossing_3d.py",
    "ori_codex6_spectral_subspace.py",
    "ori_codex6_spectral_subspace_3d.py",
    "collision_3d.py",
    "single_solution_common.py",
    "run_ori_codex6_experiments.py",
    "ori_codex6_spectral_random_baseline.py",
    Path(__file__).name,
)
ENVIRONMENT_KEYS = (
    "max_steps", "max_step_width", "min_step_width",
    "flatness_subspace_dimension", "target_spread", "spread_tolerance",
    "flatness_tolerance", "strain_tolerance", "crossing_tolerance",
    "projection_shrink", "max_correction_ratio",
    "max_actual_step_ratio", "terminate_wrong_flat",
    "ori6_crossing_policy", "ori6_rejected_crossing_continuation",
    "ori5_crossing_penalty", "ori5_quality_weight",
    "ori6_spectral_weight", "ori6_spectral_relative_epsilon",
    "ori6_spread_frontier_mode", "ori6_spread_frontier_bonus",
    "ori6_spread_frontier_flatness_cap",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_hashed_json(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON authority is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"tmp-{uuid.uuid4().hex}.json")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def exact_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an exact integer")
    return value


def exact_binary(value, name: str) -> int:
    number = exact_int(value, name)
    if number not in (0, 1):
        raise RuntimeError(f"{name} must be binary")
    return number


def finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} must be finite")
    return number


def lowercase_sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def source_hashes() -> dict[str, str]:
    result = {}
    for name in TRANSITIVE_SOURCE_NAMES:
        path = SCRIPT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha256(path)
    return result


def c10_authority() -> tuple[dict, str]:
    manifest = load_json(C10_MANIFEST)
    identity = dict(manifest)
    fingerprint = identity.pop("manifest_fingerprint", None)
    identity.pop("generated_utc", None)
    if canonical_digest(identity) != fingerprint:
        raise RuntimeError("C10 manifest fingerprint identity mismatch")
    if manifest.get("manifest_fingerprint") != (
        "7f1a9303048ba683d01f8d8539ed30444e8800ff5952ec498890e694410165ab"
    ):
        raise RuntimeError("Unexpected C10 manifest fingerprint")
    if manifest.get("protocol") != (
        "codex6_p0x2rcs05_nonswap_optimizer_screen_to_10000"
    ):
        raise RuntimeError("Unexpected C10 protocol")
    return manifest, file_sha256(C10_MANIFEST)


def task_c10_config(c10: dict, task: str) -> dict:
    matches = [
        job for job in c10["jobs"]
        if job["task"]["name"] == task and job["arm"] == "B"
    ]
    if len(matches) != 3:
        raise RuntimeError(f"C10 B authority is not three seeds for {task}")
    first = matches[0]["expected_config"]
    for job in matches[1:]:
        for key in ENVIRONMENT_KEYS:
            if job["expected_config"].get(key) != first.get(key):
                raise RuntimeError(f"C10 environment differs by seed: {task}/{key}")
    return {key: first[key] for key in ENVIRONMENT_KEYS}


def job_output(experiment_root: Path, task: str, seed: int) -> Path:
    return (
        experiment_root / task / f"budget{TASK_BUDGETS[task]}"
        / f"seed{seed}"
    ).resolve()


def job_command(output: Path, task: str, seed: int) -> list[str]:
    budget = TASK_BUDGETS[task]
    return [
        str(Path(sys.executable).resolve()), "-u", str(BASELINE),
        "--task", task,
        "--mechanics-budget", str(budget),
        "--trial-seeds", str(seed),
        "--output-root", str(output),
        "--no-resume",
        "--checkpoint-every-episodes", str(budget + 1),
        "--print-every", "100",
    ]


def build_manifest(experiment_root: Path) -> dict:
    root = Path(experiment_root).resolve()
    if root.exists():
        raise FileExistsError(f"Fresh experiment root already exists: {root}")
    c10, c10_sha = c10_authority()
    sources = source_hashes()
    # The C10 manifest must bind the same current transitive sources it names.
    for name, digest in c10["source_sha256"].items():
        path = SCRIPT_DIR / name
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"C10 source authority changed: {name}")
    jobs = []
    # Long tasks are interleaved by seed so work-conserving scheduling does not
    # strand all spatial trials at the end.
    for seed in SEEDS:
        for task in ("five_d2", "watersqrt2", "five_d3"):
            output = job_output(root, task, seed)
            args = baseline.candidate_args(task, seed)
            environment = {key: getattr(args, key) for key in ENVIRONMENT_KEYS}
            expected = task_c10_config(c10, task)
            if environment != expected:
                raise RuntimeError(
                    f"Official random environment differs from C10: {task}"
                )
            random_identity = baseline.root_identity(
                task, TASK_BUDGETS[task], [seed]
            )
            c10_job = next(
                item for item in c10["jobs"]
                if item["task"]["name"] == task and item["arm"] == "B"
            )
            # Names differ, values must be identical.
            model = baseline.model_hashes(task)
            if model != {
                "node": c10_job["model_sha256"]["node_initial.dat"],
                "member": c10_job["model_sha256"]["member.dat"],
                "fix": c10_job["model_sha256"]["fix.dat"],
            }:
                raise RuntimeError(f"Model identity differs from C10: {task}")
            body = {
                "task": task,
                "seed": seed,
                "mechanics_budget": TASK_BUDGETS[task],
                "output_root": str(output),
                "command": job_command(output, task, seed),
                "thread_environment": THREAD_ENV,
                "checkpoint_interval": TASK_BUDGETS[task] + 1,
                "checkpoint_branch_unreachable": True,
                "resume": False,
                "c10_environment": expected,
                "model_sha256": model,
                "random_root_fingerprint": canonical_digest(random_identity),
                "random_fingerprint_identity": random_identity,
            }
            jobs.append({**body, "job_fingerprint": canonical_digest(body)})
    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "experiment_root": str(root),
        "tasks": TASK_BUDGETS,
        "seeds": list(SEEDS),
        "job_count": 30,
        "max_random_parallel": MAX_RANDOM_PARALLEL,
        "max_total_trainers": MAX_TOTAL_TRAINERS,
        "deadline_local": DEADLINE_LOCAL,
        "thread_environment": THREAD_ENV,
        "resume_forbidden": True,
        "partial_trials_excluded": True,
        "checkpoint_authority": "forbidden; interval=budget+1",
        "success": "strict_compact_success == 1",
        "proposal_clock": (
            "every raw action proposal, including rejected crossings"
        ),
        "crossing_policy": "reject_candidate_and_continue",
        "c10_manifest_path": str(C10_MANIFEST),
        "c10_manifest_sha256": c10_sha,
        "c10_manifest_fingerprint": c10["manifest_fingerprint"],
        "source_sha256": sources,
        "jobs": jobs,
    }
    return {
        **identity,
        "fingerprint_identity": identity,
        "manifest_fingerprint": canonical_digest(identity),
        "generated_utc": utc_now(),
    }


def validate_manifest(manifest: dict, path: Path | None = None) -> None:
    identity = manifest.get("fingerprint_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Manifest lacks fingerprint_identity")
    if canonical_digest(identity) != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Manifest fingerprint mismatch")
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Manifest identity mismatch: {key}")
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Protocol mismatch")
    if manifest.get("tasks") != TASK_BUDGETS:
        raise RuntimeError("Task budget mismatch")
    if manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Seed mismatch")
    if exact_int(manifest.get("job_count"), "job_count") != 30:
        raise RuntimeError("Job count mismatch")
    if manifest.get("thread_environment") != THREAD_ENV:
        raise RuntimeError("Thread environment mismatch")
    if manifest.get("resume_forbidden") is not True:
        raise RuntimeError("Resume must be forbidden")
    if manifest.get("partial_trials_excluded") is not True:
        raise RuntimeError("Partial trials must be excluded")
    if exact_int(
        manifest.get("max_random_parallel"), "max_random_parallel"
    ) != MAX_RANDOM_PARALLEL:
        raise RuntimeError("Random parallelism mismatch")
    if exact_int(
        manifest.get("max_total_trainers"), "max_total_trainers"
    ) != MAX_TOTAL_TRAINERS:
        raise RuntimeError("Physical trainer cap mismatch")
    if manifest.get("deadline_local") != DEADLINE_LOCAL:
        raise RuntimeError("Deadline mismatch")
    if manifest.get("checkpoint_authority") != (
        "forbidden; interval=budget+1"
    ):
        raise RuntimeError("Checkpoint authority mismatch")
    c10, c10_sha = c10_authority()
    if manifest.get("c10_manifest_path") != str(C10_MANIFEST):
        raise RuntimeError("C10 manifest path mismatch")
    if manifest.get("c10_manifest_fingerprint") != c10["manifest_fingerprint"]:
        raise RuntimeError("C10 manifest fingerprint mismatch")
    if c10_sha != manifest.get("c10_manifest_sha256"):
        raise RuntimeError("C10 manifest changed")
    for name, expected in c10.get("source_sha256", {}).items():
        path = SCRIPT_DIR / name
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"C10 source authority changed: {name}")
    current_sources = source_hashes()
    if manifest.get("source_sha256") != current_sources:
        raise RuntimeError("Source closure changed")
    root = Path(manifest["experiment_root"]).resolve()
    seen = set()
    if len(manifest.get("jobs", [])) != 30:
        raise RuntimeError("Manifest job matrix is incomplete")
    for job in manifest["jobs"]:
        task = job.get("task")
        seed = exact_int(job.get("seed"), "job.seed")
        if task not in TASK_BUDGETS or seed not in SEEDS:
            raise RuntimeError("Unexpected task/seed")
        if (task, seed) in seen:
            raise RuntimeError("Duplicate task/seed")
        seen.add((task, seed))
        budget = TASK_BUDGETS[task]
        output = job_output(root, task, seed)
        body = {key: value for key, value in job.items() if key != "job_fingerprint"}
        if canonical_digest(body) != job.get("job_fingerprint"):
            raise RuntimeError("Job fingerprint mismatch")
        if job.get("output_root") != str(output):
            raise RuntimeError("Job output mismatch")
        if job.get("command") != job_command(output, task, seed):
            raise RuntimeError("Job command mismatch")
        if job.get("thread_environment") != THREAD_ENV:
            raise RuntimeError("Job thread environment mismatch")
        if exact_int(job.get("mechanics_budget"), "budget") != budget:
            raise RuntimeError("Job budget mismatch")
        if exact_int(job.get("checkpoint_interval"), "checkpoint") != budget + 1:
            raise RuntimeError("Checkpoint workaround mismatch")
        if job.get("checkpoint_branch_unreachable") is not True:
            raise RuntimeError("Checkpoint branch must be unreachable")
        if job.get("resume") is not False:
            raise RuntimeError("Resume flag mismatch")
        expected_environment = task_c10_config(c10, task)
        if job.get("c10_environment") != expected_environment:
            raise RuntimeError("C10 environment binding mismatch")
        args = baseline.candidate_args(task, seed)
        random_environment = {
            key: getattr(args, key) for key in ENVIRONMENT_KEYS
        }
        if random_environment != expected_environment:
            raise RuntimeError("Random/C10 environment mismatch")
        expected_model = baseline.model_hashes(task)
        if job.get("model_sha256") != expected_model:
            raise RuntimeError("Model binding mismatch")
        expected_identity = baseline.root_identity(task, budget, [seed])
        if job.get("random_fingerprint_identity") != expected_identity:
            raise RuntimeError("Random root identity mismatch")
        if job.get("random_root_fingerprint") != canonical_digest(expected_identity):
            raise RuntimeError("Random root fingerprint mismatch")
    if seen != {(task, seed) for task in TASK_BUDGETS for seed in SEEDS}:
        raise RuntimeError("Task/seed matrix mismatch")
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(path)


def exact_csv_int(value: str, name: str) -> int:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"{name} is not an exact CSV integer")
    if value[0] == "-":
        digits = value[1:]
    else:
        digits = value
    if not digits.isdigit():
        raise RuntimeError(f"{name} is not an exact CSV integer")
    return int(value)


def exact_csv_binary(value: str, name: str) -> int:
    number = exact_csv_int(value, name)
    if number not in (0, 1):
        raise RuntimeError(f"{name} is not binary")
    return number


def finite_csv_float(value: str, name: str) -> float:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"{name} is not an exact CSV number")
    try:
        number = float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} is not numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite")
    return number


def validate_completed_job(job: dict) -> dict:
    output = Path(job["output_root"])
    budget = exact_int(job["mechanics_budget"], "budget")
    seed = exact_int(job["seed"], "seed")
    paths = {
        "root_config": output / "run_config.json",
        "aggregate": output / "aggregate.json",
        "trial_config": output / "trials" / "uniform" / f"seed{seed}"
        / "run_config.json",
        "summary": output / "trials" / "uniform" / f"seed{seed}"
        / "summary.json",
        "history": output / "trials" / "uniform" / f"seed{seed}"
        / "episode_history.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"Missing formal artifact: {path}")
    checkpoint = paths["summary"].parent / "checkpoint.pt"
    if checkpoint.exists():
        raise RuntimeError("Checkpoint artifact is forbidden")
    root_config = load_json(paths["root_config"])
    baseline.validate_root_config(root_config)
    if root_config.get("fingerprint_identity") != job["random_fingerprint_identity"]:
        raise RuntimeError("Persisted random identity mismatch")
    if root_config.get("root_fingerprint") != job["random_root_fingerprint"]:
        raise RuntimeError("Persisted random fingerprint mismatch")
    trial_config = load_json(paths["trial_config"])
    summary = load_json(paths["summary"])
    aggregate = load_json(paths["aggregate"])
    expected_trial_config = {
        "schema_version": baseline.SCHEMA_VERSION,
        "harness_version": baseline.HARNESS_VERSION,
        "root_fingerprint": job["random_root_fingerprint"],
        "method": baseline.METHOD,
        "seed": seed,
        "task": job["task"],
        "mechanics_budget": budget,
        "episode_cap": None,
        "max_steps_per_episode": job["c10_environment"]["max_steps"],
        "candidate": baseline.spectral.CANDIDATE_NAME,
        "crossing_policy": baseline.rejected.CROSSING_POLICY,
        "spectral_formula_sha256": (
            baseline.spectral.SPECTRAL_FORMULA_SHA256
        ),
        "budget_stop_reward": (
            "unchanged candidate reward from final ordinary valid transition"
        ),
    }
    if trial_config != expected_trial_config:
        raise RuntimeError("Persisted trial configuration mismatch")
    for name, value in expected_trial_config.items():
        if summary.get(name) != value:
            raise RuntimeError(f"Trial summary authority mismatch: {name}")
    if summary.get("complete") is not True:
        raise RuntimeError("Trial is partial")
    if summary.get("reward_integrity_pass") is not True:
        raise RuntimeError("Reward integrity failed")
    audit = summary.get("reward_integrity")
    if not isinstance(audit, dict) or audit.get("candidate_identity_all_rows") is not True:
        raise RuntimeError("Candidate row identity failed")
    if exact_int(audit.get("committed_self_crossing_count"), "committed crossings") != 0:
        raise RuntimeError("A crossing candidate was committed")
    for name in (
        "ori6_nonfinite_reward_component_count",
        "spectral_nonfinite_component_count",
    ):
        if exact_int(audit.get(name), name) != 0:
            raise RuntimeError(f"Nonfinite reward component: {name}")
    for name in (
        "maximum_abs_ori6_reward_decomposition_error",
        "maximum_abs_ori6_step_reward_decomposition_error",
        "maximum_abs_rejected_crossing_decomposition_error",
        "maximum_abs_spectral_reward_decomposition_error",
        "maximum_abs_spectral_step_reward_decomposition_error",
    ):
        if abs(finite_float(audit.get(name), name)) > 1e-10:
            raise RuntimeError(f"Reward decomposition error: {name}")
    success = exact_binary(summary.get("success"), "success")
    censored = exact_binary(summary.get("censored"), "censored")
    if success + censored != 1:
        raise RuntimeError("Success/censor identity failed")
    mechanics = exact_int(summary.get("mechanical_steps"), "mechanical_steps")
    if not 1 <= mechanics <= budget:
        raise RuntimeError("Proposal clock is out of range")
    episodes = exact_int(summary.get("episodes"), "episodes")
    if episodes < 1:
        raise RuntimeError("Episode count is invalid")
    projector_calls = exact_int(summary.get("projector_calls"), "projector_calls")
    if projector_calls < 0:
        raise RuntimeError("Projector clock is out of range")
    if summary.get("action_trace_chain") != baseline.ACTION_TRACE_CHAIN:
        raise RuntimeError("Action trace-chain definition mismatch")
    action_chain = lowercase_sha256(
        summary.get("action_chain_sha256"), "action_chain_sha256"
    )
    prefixes = summary.get("action_prefix_sha256_by_mechanics")
    if not isinstance(prefixes, dict):
        raise RuntimeError("Action-prefix authority is not a dictionary")
    audit_mechanics = min(baseline.INITIAL_AUDIT_MECHANICS, budget)
    expected_prefix_keys = {str(audit_mechanics)} if mechanics >= audit_mechanics else set()
    if set(prefixes) != expected_prefix_keys:
        raise RuntimeError("Action-prefix checkpoint set mismatch")
    for key, digest in prefixes.items():
        lowercase_sha256(digest, f"action_prefix_sha256_by_mechanics[{key}]")
    if mechanics == audit_mechanics and prefixes[str(audit_mechanics)] != action_chain:
        raise RuntimeError("Action chain/prefix boundary mismatch")
    if censored:
        if mechanics != budget or summary.get("censor_reason") != "mechanics_budget":
            raise RuntimeError("Censored trial missed exact proposal cap")
        if any(summary.get(name) is not None for name in (
            "first_success_episode", "first_success_mechanical_steps",
            "first_success_projector_calls", "first_hit_path",
            "terminal_flatness_ratio", "terminal_spread",
        )):
            raise RuntimeError("Censored trial has a first hit")
    else:
        if summary.get("censor_reason") is not None:
            raise RuntimeError("Successful trial has a censor reason")
        if exact_int(
            summary.get("first_success_episode"), "first success episode"
        ) != episodes:
            raise RuntimeError("First-hit episode mismatch")
        if exact_int(
            summary.get("first_success_mechanical_steps"), "first hit"
        ) != mechanics:
            raise RuntimeError("First-hit proposal clock mismatch")
        if exact_int(
            summary.get("first_success_projector_calls"),
            "first success projector calls",
        ) != projector_calls:
            raise RuntimeError("First-hit projector clock mismatch")
        if summary.get("first_hit_path") != "first_hit_path":
            raise RuntimeError("First-hit path authority mismatch")
        if finite_float(
            summary.get("terminal_flatness_ratio"), "flatness"
        ) > job["c10_environment"]["flatness_tolerance"]:
            raise RuntimeError("Successful trial missed flatness tolerance")
        target = job["c10_environment"]["target_spread"]
        if abs(
            finite_float(summary.get("terminal_spread"), "spread") - target
        ) > job["c10_environment"]["spread_tolerance"]:
            raise RuntimeError("Successful trial missed spread tolerance")
    with open(paths["history"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Episode history is empty")
    cumulative = 0
    cumulative_projectors = 0
    strict_count = 0
    for index, row in enumerate(rows, start=1):
        if row.get("method") != baseline.METHOD:
            raise RuntimeError("History method mismatch")
        if exact_csv_int(row.get("seed"), "row seed") != seed:
            raise RuntimeError("History seed mismatch")
        if exact_csv_int(row.get("episode"), "episode") != index:
            raise RuntimeError("Episode history is not contiguous")
        steps = exact_csv_int(row.get("mechanical_steps"), "episode steps")
        if not 1 <= steps <= job["c10_environment"]["max_steps"]:
            raise RuntimeError("Episode step count is invalid")
        cumulative += steps
        if exact_csv_int(
            row.get("cumulative_mechanical_steps"), "cumulative proposals"
        ) != cumulative:
            raise RuntimeError("Cumulative proposal clock mismatch")
        row_projectors = exact_csv_int(row.get("projector_calls"), "projector calls")
        if row_projectors < 0:
            raise RuntimeError("Episode projector count is invalid")
        cumulative_projectors += row_projectors
        if exact_csv_int(
            row.get("cumulative_projector_calls"), "cumulative projector calls"
        ) != cumulative_projectors:
            raise RuntimeError("Cumulative projector clock mismatch")
        strict = exact_csv_binary(
            row.get("strict_compact_success"), "strict_compact_success"
        )
        if exact_csv_binary(row.get("success"), "row success") != strict:
            raise RuntimeError("Row success/strict identity failed")
        strict_count += strict
        if exact_csv_binary(row.get("self_crossing"), "self_crossing") != 0:
            raise RuntimeError("Committed self-crossing row")
        for name in ("correction_failed", "timeout", "no_modes", "wrong_flat"):
            exact_csv_binary(row.get(name), name)
        if row.get("ori6_crossing_policy") != baseline.rejected.CROSSING_POLICY:
            raise RuntimeError("Row crossing policy mismatch")
        if row.get("ori6_candidate") != baseline.rejected.CANDIDATE_NAME:
            raise RuntimeError("Parent candidate row mismatch")
        if row.get("ori6_spectral_candidate") != baseline.spectral.CANDIDATE_NAME:
            raise RuntimeError("Spectral candidate row mismatch")
        if row.get("ori6_spectral_parent_candidate") != baseline.rejected.CANDIDATE_NAME:
            raise RuntimeError("Spectral parent row mismatch")
        if row.get("ori6_spectral_formula_sha256") != (
            baseline.spectral.SPECTRAL_FORMULA_SHA256
        ):
            raise RuntimeError("Spectral formula row mismatch")
        lowercase_sha256(row.get("action_trace_sha256"), "row action trace")
        rejected_count = exact_csv_int(
            row.get("ori6_rejected_crossing_count"), "rejected crossings"
        )
        if not 0 <= rejected_count <= steps:
            raise RuntimeError("Rejected crossing count is invalid")
        rejected_rate = finite_csv_float(
            row.get("ori6_rejected_crossing_rate"), "rejected crossing rate"
        )
        if abs(rejected_rate - rejected_count / steps) > 1e-12:
            raise RuntimeError("Rejected crossing count/rate mismatch")
        for name in (
            "ori6_nonfinite_reward_component_count",
            "ori6_spectral_nonfinite_component_count",
        ):
            if exact_csv_int(row.get(name), name) != 0:
                raise RuntimeError(f"Nonfinite row component: {name}")
        for name in (
            "ori6_reward_decomposition_error",
            "ori6_maximum_abs_step_reward_decomposition_error",
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
            "ori6_spectral_reward_decomposition_error",
            "ori6_spectral_maximum_abs_step_reward_decomposition_error",
        ):
            if abs(finite_csv_float(row.get(name), name)) > 1e-10:
                raise RuntimeError(f"Row reward decomposition error: {name}")
        if abs(finite_csv_float(
            row.get("ori6_spectral_weight"), "spectral weight"
        ) - job["c10_environment"]["ori6_spectral_weight"]) > 1e-15:
            raise RuntimeError("Spectral weight row mismatch")
        if abs(finite_csv_float(
            row.get("ori6_spectral_relative_epsilon"), "spectral epsilon"
        ) - job["c10_environment"]["ori6_spectral_relative_epsilon"]) > 1e-24:
            raise RuntimeError("Spectral epsilon row mismatch")
    if cumulative != mechanics or len(rows) != episodes:
        raise RuntimeError("History/summary proposal clock mismatch")
    if cumulative_projectors != projector_calls:
        raise RuntimeError("History/summary projector clock mismatch")
    if strict_count != success or exact_csv_int(
        rows[-1]["strict_compact_success"], "final strict"
    ) != success:
        raise RuntimeError("Strict first-hit stopping identity failed")
    expected_aggregate = {
        "schema_version": baseline.SCHEMA_VERSION,
        "harness_version": baseline.HARNESS_VERSION,
        "candidate": baseline.spectral.CANDIDATE_NAME,
        "task": job["task"],
        "method": baseline.METHOD,
        "root_fingerprint": job["random_root_fingerprint"],
        "mechanics_budget": budget,
        "seeds": [seed],
        "complete": True,
        "missing_seeds": [],
        "successes": success,
        "censored": censored,
        "first_success_mechanical_steps": [mechanics] if success else [],
        "rmst_censor_assignment": (
            "all censored trials assigned the common mechanics budget"
        ),
        "all_reward_integrity_pass": True,
    }
    for name, value in expected_aggregate.items():
        if aggregate.get(name) != value:
            raise RuntimeError(f"Aggregate authority mismatch: {name}")
    if exact_int(aggregate.get("expected_trials"), "expected_trials") != 1:
        raise RuntimeError("Aggregate expected count mismatch")
    if exact_int(aggregate.get("completed_trials"), "completed_trials") != 1:
        raise RuntimeError("Aggregate completed count mismatch")
    if abs(finite_float(
        aggregate.get("success_at_budget"), "aggregate success rate"
    ) - success) > 1e-15:
        raise RuntimeError("Aggregate success rate mismatch")
    if abs(finite_float(
        aggregate.get("restricted_mean_first_hit_mechanical_steps"),
        "aggregate RMST",
    ) - mechanics) > 1e-9:
        raise RuntimeError("Aggregate RMST mismatch")
    if aggregate.get("trial_summaries") != [summary]:
        raise RuntimeError("Aggregate trial-summary copy mismatch")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    # Hash every saved hit-path file as part of the formal input closure.
    hit_dir = paths["summary"].parent / "first_hit_path"
    if success:
        if not hit_dir.is_dir():
            raise RuntimeError("Successful trial lacks first_hit_path")
        expected_names = {"actions.json"} | {
            f"step{index}.dat" for index in range(steps + 1)
        }
        actual_names = {path.name for path in hit_dir.iterdir()}
        if actual_names != expected_names:
            raise RuntimeError("First-hit path file set mismatch")
        saved_actions = load_json(hit_dir / "actions.json")
        if (
            not isinstance(saved_actions, list)
            or len(saved_actions) != steps
            or any(
                not isinstance(action, dict)
                or exact_int(action.get("step"), "saved action step") != index
                for index, action in enumerate(saved_actions, start=1)
            )
        ):
            raise RuntimeError("First-hit action sequence mismatch")
        for path in sorted(hit_dir.iterdir()):
            if not path.is_file():
                raise RuntimeError("Unexpected non-file in first_hit_path")
            hashes[f"first_hit_path/{path.name}"] = file_sha256(path)
    elif hit_dir.exists():
        raise RuntimeError("Censored trial has first_hit_path")
    return {
        "success": success,
        "censored": censored,
        "first_success_mechanical_steps": (
            mechanics if success else None
        ),
        "mechanical_steps": mechanics,
        "episodes": len(rows),
        "artifact_sha256": hashes,
    }


def process_records() -> list[dict]:
    """Read Windows process command lines without an optional dependency."""
    if os.name != "nt":
        raise RuntimeError("This frozen launcher requires Windows process authority")
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="strict",
        timeout=30, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(
            f"Process authority failed rc={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("Process authority returned a non-list")
    result = []
    for row in payload:
        if not isinstance(row, dict):
            raise RuntimeError("Malformed process authority row")
        pid = row.get("ProcessId")
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise RuntimeError("Malformed process id")
        result.append({
            "pid": pid,
            "name": row.get("Name"),
            "command_line": row.get("CommandLine"),
        })
    return result


def option_from_command_line(command_line: str, option: str) -> str | None:
    pattern = re.compile(
        rf"(?:^|\s){re.escape(option)}\s+(?:\"([^\"]+)\"|(\S+))",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(command_line)
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Duplicate command option: {option}")
    return matches[0][0] or matches[0][1]


def command_key(command_line: str) -> tuple[str, str] | None:
    lowered = command_line.lower()
    random_output = option_from_command_line(command_line, "--output-root")
    if random_output is not None and str(BASELINE).lower() in lowered:
        return "random", str(Path(random_output).resolve()).lower()
    c10_output = option_from_command_line(command_line, "--output-dir")
    if c10_output is not None and "ori_codex6_spectral_subspace" in lowered:
        return "c10", str(Path(c10_output).resolve()).lower()
    return None


def active_trainer_count(manifest: dict) -> tuple[int, list[dict], list[int]]:
    random_outputs = {job["output_root"].lower() for job in manifest["jobs"]}
    c10 = load_json(C10_MANIFEST)
    c10_outputs = {job["output_directory"].lower() for job in c10["jobs"]}
    workers = []
    unknown_python = []
    for process in process_records():
        if process["pid"] == os.getpid():
            continue
        name = (process.get("name") or "").lower()
        if "python" not in name:
            continue
        command_line = process.get("command_line")
        if not isinstance(command_line, str) or not command_line:
            unknown_python.append(process["pid"])
            continue
        try:
            key = command_key(command_line)
        except Exception:
            unknown_python.append(process["pid"])
            continue
        if key is None:
            lowered = command_line.lower()
            if (
                str(BASELINE).lower() in lowered
                or "ori_codex6_spectral_subspace" in lowered
            ):
                unknown_python.append(process["pid"])
            continue
        kind, output = key
        if (kind == "random" and output in random_outputs) or (
            kind == "c10" and output in c10_outputs
        ):
            workers.append({
                "pid": process["pid"], "kind": kind, "output": output,
            })
    return len(workers), workers, unknown_python


def duplicate_supervisor() -> list[int]:
    result = []
    marker = str(Path(__file__).resolve()).lower()
    for process in process_records():
        if process["pid"] == os.getpid():
            continue
        command_line = process.get("command_line")
        if not isinstance(command_line, str):
            continue
        lowered = command_line.lower()
        if marker in lowered and "--execute" in lowered:
            result.append(process["pid"])
    return result


def reconcile_process_authority(
    workers: list[dict], unknown: list[int], owned_processes: dict[int, object],
) -> tuple[int, list[int]]:
    """Combine CIM rows with child handles without an exit-transition race."""
    owned_pids = set(owned_processes)
    external_unknown = [pid for pid in unknown if pid not in owned_pids]
    live_handle_pids = {
        pid for pid, process in owned_processes.items()
        if process.poll() is None
    }
    observed_pids = {
        item["pid"] for item in workers if item["pid"] not in owned_pids
    } | live_handle_pids
    return len(observed_pids), external_unknown


def final_pid_zero_errors(
    workers: list[dict], unknown: list[int], owned_processes: dict[int, object],
) -> dict[str, list[int]]:
    owned = set(owned_processes)
    owned_live = sorted(
        pid for pid, process in owned_processes.items()
        if process.poll() is None
    )
    external_unknown = sorted(pid for pid in unknown if pid not in owned)
    external_random = sorted(
        item["pid"] for item in workers
        if item["kind"] == "random" and item["pid"] not in owned
    )
    return {
        "owned_live": owned_live,
        "external_unknown": external_unknown,
        "external_random": external_random,
    }


def verify_manifest_file(path: Path, expected_sha256: str) -> None:
    if file_sha256(Path(path)) != expected_sha256:
        raise RuntimeError("Random campaign manifest changed after binding")


def verify_live_authority(
    manifest: dict,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> None:
    if (manifest_path is None) != (manifest_sha256 is None):
        raise RuntimeError("Manifest path/hash authority must be paired")
    if manifest_path is not None:
        verify_manifest_file(manifest_path, manifest_sha256)
    validate_manifest(manifest)
    for job in manifest["jobs"]:
        for name, expected in job["model_sha256"].items():
            task = job["task"]
            current = baseline.model_hashes(task)[name]
            if current != expected:
                raise RuntimeError(f"Model changed: {task}/{name}")
    if manifest_path is not None:
        verify_manifest_file(manifest_path, manifest_sha256)


def deadline_epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise RuntimeError("Deadline must include timezone")
    return parsed.timestamp()


def launch_before_deadline(deadline: float) -> bool:
    return time.time() < deadline


def run_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest, bound_manifest_sha256 = load_hashed_json(manifest_path)
    validate_manifest(manifest, manifest_path)
    root = Path(manifest["experiment_root"])
    if root.exists():
        raise FileExistsError(f"Fresh experiment root already exists: {root}")
    duplicates = duplicate_supervisor()
    if duplicates:
        raise RuntimeError(f"Duplicate random supervisor: {duplicates}")
    # No bound worker may pre-exist, even if its output root has not appeared.
    count, workers, unknown = active_trainer_count(manifest)
    if any(item["kind"] == "random" for item in workers):
        raise RuntimeError(f"Bound random worker already active: {workers}")
    if unknown:
        raise RuntimeError(f"Cannot inspect Python processes: {unknown}")
    if count > MAX_TOTAL_TRAINERS:
        raise RuntimeError("C10 trainer count already exceeds physical cap")
    deadline = deadline_epoch(manifest["deadline_local"])
    if time.time() >= deadline:
        raise RuntimeError("Campaign deadline has already passed")
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "_logs"
    logs.mkdir()
    binding = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "manifest_path": str(manifest_path),
        "manifest_sha256": bound_manifest_sha256,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "supervisor_pid": os.getpid(),
        "started_utc": utc_now(),
        "thread_environment": THREAD_ENV,
        "deadline_local": DEADLINE_LOCAL,
        "status": "running",
    }
    atomic_json(root / "execution_binding.json", binding)
    pending = list(manifest["jobs"])
    running = {}
    owned_processes = {}
    completed = []
    launch_audit = []
    fatal_error = None
    env = os.environ.copy()
    env.update(THREAD_ENV)
    while (pending and fatal_error is None) or running:
        now = time.time()
        if now >= deadline:
            for state in running.values():
                if state["process"].poll() is None:
                    state["process"].terminate()
            for state in running.values():
                try:
                    state["process"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    state["process"].kill()
                    state["process"].wait(timeout=10)
            break
        finished = []
        for fingerprint, state in running.items():
            rc = state["process"].poll()
            if rc is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            result = {
                "job_fingerprint": fingerprint,
                "task": state["job"]["task"],
                "seed": state["job"]["seed"],
                "return_code": rc,
                "launched_utc": state["launched_utc"],
                "finished_utc": utc_now(),
                "stdout": state["stdout_path"],
                "stderr": state["stderr_path"],
                "admissible": False,
            }
            if rc == 0:
                try:
                    result.update(validate_completed_job(state["job"]))
                    result["admissible"] = True
                except Exception as error:  # recorded, never selected
                    result["integrity_error"] = repr(error)
            completed.append(result)
            finished.append(fingerprint)
        for fingerprint in finished:
            del running[fingerprint]
        # Re-scan immediately before every launch.  Known child handles are
        # unioned with the process table so process-enumeration lag can never
        # create an extra physical slot.
        while pending and fatal_error is None:
            try:
                total, workers, unknown = active_trainer_count(manifest)
            except Exception as error:
                fatal_error = f"process_authority: {error!r}"
                launch_audit.append({
                    "utc": utc_now(), "launch_failed_closed": fatal_error,
                })
                break
            total, external_unknown = reconcile_process_authority(
                workers, unknown, owned_processes
            )
            if external_unknown:
                fatal_error = (
                    "uninspectable_external_python_processes: "
                    f"{external_unknown}"
                )
                launch_audit.append({
                    "utc": utc_now(), "launch_failed_closed": fatal_error,
                })
                break
            if (
                len(running) >= MAX_RANDOM_PARALLEL
                or total >= MAX_TOTAL_TRAINERS
            ):
                break
            try:
                verify_live_authority(
                    manifest, manifest_path, bound_manifest_sha256
                )
                if not launch_before_deadline(deadline):
                    launch_audit.append({
                        "utc": utc_now(),
                        "launch_blocked_at_deadline": True,
                    })
                    break
                job = pending[0]
                output = Path(job["output_root"])
                if output.exists():
                    raise FileExistsError(f"Job output is not fresh: {output}")
                log_stem = f"{job['task']}_seed{job['seed']}"
                stdout_path = logs / f"{log_stem}.stdout.log"
                stderr_path = logs / f"{log_stem}.stderr.log"
                stdout = open(stdout_path, "xb")
                stderr = open(stderr_path, "xb")
                try:
                    # File authority and deadline are checked again at the
                    # last possible point before process creation.
                    verify_manifest_file(
                        manifest_path, bound_manifest_sha256
                    )
                    if not launch_before_deadline(deadline):
                        stdout.close()
                        stderr.close()
                        launch_audit.append({
                            "utc": utc_now(),
                            "launch_blocked_at_deadline": True,
                        })
                        break
                    process = subprocess.Popen(
                        job["command"], cwd=SCRIPT_DIR, env=env,
                        stdout=stdout, stderr=stderr,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW
                            if os.name == "nt" else 0
                        ),
                    )
                except Exception:
                    stdout.close()
                    stderr.close()
                    raise
                pending.pop(0)
                running[job["job_fingerprint"]] = {
                    "job": job, "process": process,
                    "stdout": stdout, "stderr": stderr,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "launched_utc": utc_now(),
                }
                owned_processes[process.pid] = process
                launch_audit.append({
                    "utc": utc_now(), "pid": process.pid,
                    "task": job["task"], "seed": job["seed"],
                    "active_before_launch": total,
                    "physical_cap": MAX_TOTAL_TRAINERS,
                })
            except Exception as error:
                fatal_error = f"launch_authority: {error!r}"
                launch_audit.append({
                    "utc": utc_now(), "launch_failed_closed": fatal_error,
                })
                break
        status = "running"
        atomic_json(root / "runner_progress.json", {
            **binding, "status": status, "updated_utc": utc_now(),
            "pending": len(pending), "running": len(running),
            "completed": len(completed),
            "admissible": sum(item["admissible"] for item in completed),
            "launch_audit": launch_audit,
        })
        time.sleep(1.0)
    # Close handles after deadline termination and record every partial as such.
    for state in running.values():
        state["stdout"].close()
        state["stderr"].close()
        completed.append({
            "job_fingerprint": state["job"]["job_fingerprint"],
            "task": state["job"]["task"], "seed": state["job"]["seed"],
            "return_code": state["process"].poll(),
            "admissible": False, "exclusion_reason": "deadline_partial",
            "stdout": state["stdout_path"], "stderr": state["stderr_path"],
        })
    # Revalidate and rehash every admissible result immediately before report.
    try:
        verify_live_authority(
            manifest, manifest_path, bound_manifest_sha256
        )
        _, final_workers, final_unknown = active_trainer_count(manifest)
        pid_errors = final_pid_zero_errors(
            final_workers, final_unknown, owned_processes
        )
        if any(pid_errors.values()):
            raise RuntimeError(
                f"Random PID-zero authority failed: {pid_errors}"
            )
    except Exception as error:
        fatal_error = fatal_error or f"final_authority: {error!r}"
        for result in completed:
            if result.get("admissible"):
                result["admissible"] = False
                result["integrity_error"] = fatal_error
    by_fp = {job["job_fingerprint"]: job for job in manifest["jobs"]}
    for result in completed:
        if result.get("admissible"):
            try:
                result.update(validate_completed_job(
                    by_fp[result["job_fingerprint"]]
                ))
            except Exception as error:
                result["admissible"] = False
                result["integrity_error"] = f"final_revalidation: {error!r}"
    # Recheck the complete control plane after potentially slow artifact
    # hashing and immediately before constructing the atomic report.
    try:
        verify_live_authority(
            manifest, manifest_path, bound_manifest_sha256
        )
    except Exception as error:
        fatal_error = fatal_error or f"pre_report_authority: {error!r}"
        for result in completed:
            if result.get("admissible"):
                result["admissible"] = False
                result["integrity_error"] = fatal_error
    admissible = sum(item.get("admissible") is True for item in completed)
    report = {
        **binding,
        "status": "complete" if admissible == 30 else "incomplete",
        "finished_utc": utc_now(),
        "expected_jobs": 30,
        "admissible_jobs": admissible,
        "partial_or_failed_jobs": 30 - admissible,
        "all_formal_jobs_complete": admissible == 30,
        "fatal_error": fatal_error,
        "results": completed,
        "launch_audit": launch_audit,
        "pending_unlaunched": [
            {"task": job["task"], "seed": job["seed"]} for job in pending
        ],
    }
    atomic_json(root / "runner_report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.prepare:
        if options.experiment_root is None or options.manifest is not None:
            raise ValueError("--prepare requires only --experiment-root")
        manifest = build_manifest(options.experiment_root)
        directory = SCRIPT_DIR / "experiments_ori10" / "manifests"
        path = directory / (
            f"codex10_nonswap_random_matched_"
            f"{manifest['manifest_fingerprint'][:12]}.json"
        )
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, manifest)
        print(json.dumps({
            "manifest": str(path),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "experiment_root": manifest["experiment_root"],
            "jobs": manifest["job_count"],
        }, indent=2))
        return manifest
    if options.manifest is None or options.experiment_root is not None:
        raise ValueError("--execute requires only --manifest")
    report = run_manifest(options.manifest)
    print(json.dumps({
        "status": report["status"],
        "admissible_jobs": report["admissible_jobs"],
        "report": str(Path(report["manifest_path"]).parent),
    }, indent=2))
    if report["status"] != "complete":
        raise SystemExit(2)
    return report


if __name__ == "__main__":
    main()
