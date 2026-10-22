"""Calibrate ori5 uniform-random rollout wall time on fixed prefixes.

This utility is intentionally separate from training and first-discovery
experiments.  It replays a stored set of uniform-random seed prefixes in one
process using the current ori5 environment, reward, projector, and collision
backend.  It measures rollout calls only, bootstraps timing blocks by seed,
and converts the already-audited random mechanics-step RMST to an estimated
wall-time RMST.

The converted wall time is *not* a direct measurement of Actor--Critic wall
time and must never be labelled as one.  Its uncertainty interval combines
the stored random first-hit sample with calibration-prefix timing variability;
it does not account for machine drift outside this calibration run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import gc
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch


_TOOLS_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _TOOLS_DIRECTORY.parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import collision_3d  # noqa: E402
import ori_codex3 as ori3  # noqa: E402
import ori_codex5 as ori5  # noqa: E402
import ori_codex5_first_discovery as discovery  # noqa: E402


SCHEMA_VERSION = 1
CALIBRATOR_VERSION = "1.0.0"
DEFAULT_PROTOCOL = _TOOLS_DIRECTORY / "ori5_uniform_walltime_protocol.json"
DEFAULT_BASELINE_SUMMARY = (
    _TOOLS_DIRECTORY.parent / "random_baseline_compatibility.json"
)
DEFAULT_OUTPUT_PARENT = _TOOLS_DIRECTORY.parent / "walltime_calibration"

_SOURCE_NAMES = (
    "ori_codex2.py",
    "ori_codex3.py",
    "ori_codex4.py",
    "ori_codex4_3d.py",
    "ori_codex4_3d_first_discovery.py",
    "ori_codex5.py",
    "ori_codex5_3d.py",
    "ori_codex5_first_discovery.py",
    "collision_3d.py",
)

_MECHANICS_FIELDS = (
    "max_steps",
    "max_step_width",
    "min_step_width",
    "projection_shrink",
    "max_correction_ratio",
    "max_actual_step_ratio",
    "flatness_subspace_dimension",
    "flatness_tolerance",
    "strain_tolerance",
    "crossing_tolerance",
    "target_spread",
    "spread_tolerance",
    "terminate_wrong_flat",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _finite_float(value, name, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _source_provenance():
    result = {
        name: discovery._file_sha256(_REPOSITORY_ROOT / name)
        for name in _SOURCE_NAMES
        if (_REPOSITORY_ROOT / name).is_file()
    }
    result[str(Path(__file__).relative_to(_REPOSITORY_ROOT))] = (
        discovery._file_sha256(Path(__file__))
    )
    return result


def _collision_identity(problem):
    if problem == "five_bar":
        return {
            "backend": "single_solution_common.planar_proper_crossing",
            "dimension": 2,
        }
    return {
        "backend": collision_3d.COLLISION_BACKEND_NAME,
        "version": collision_3d.COLLISION_BACKEND_VERSION,
        "dimension": 3,
    }


def _mechanics_signature(args):
    return {
        name: discovery._jsonable(getattr(args, name))
        for name in _MECHANICS_FIELDS
    }


def _values_match(actual, expected):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) is bool(expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    return actual == expected


def _validate_mechanics_against_baseline(args, baseline_task):
    expected = baseline_task.get("mechanics_signature_expected")
    if not isinstance(expected, dict):
        raise ValueError("Baseline task lacks mechanics_signature_expected.")
    current = _mechanics_signature(args)
    errors = []
    for name in _MECHANICS_FIELDS:
        if name not in expected:
            errors.append(f"missing baseline field {name}")
        elif not _values_match(current[name], expected[name]):
            errors.append(
                f"{name}: current={current[name]!r}, baseline={expected[name]!r}"
            )
    if errors:
        raise ValueError(
            "Current ori5 mechanics do not match the audited random baseline: "
            + "; ".join(errors)
        )
    return current


def validate_protocol(protocol):
    if int(protocol.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported wall-time protocol schema.")
    if not protocol.get("single_process_required"):
        raise ValueError("The protocol must require a single process.")
    if int(protocol.get("torch_threads", -1)) != 1:
        raise ValueError("The protocol must fix torch_threads to one.")
    if not protocol.get("continue_after_strict_success"):
        raise ValueError(
            "Fixed-prefix timing must continue after strict success."
        )
    confidence = _finite_float(
        protocol.get("confidence_level"), "confidence_level"
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    if int(protocol.get("bootstrap_replicates", 0)) < 2:
        raise ValueError("bootstrap_replicates must be at least two.")
    int(protocol.get("bootstrap_seed"))

    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Protocol tasks must be a non-empty list.")
    identifiers = []
    for task in tasks:
        identifier = str(task.get("task", ""))
        if not identifier:
            raise ValueError("Every protocol task requires an identifier.")
        identifiers.append(identifier)
        problem = task.get("problem")
        if problem not in discovery.PROBLEMS:
            raise ValueError(f"Unknown problem for {identifier}: {problem!r}")
        core_args = task.get("core_args")
        if not isinstance(core_args, list) or not all(
            isinstance(value, str) for value in core_args
        ):
            raise ValueError(f"{identifier} core_args must be strings.")
        discovery._reject_reward_overrides(core_args)
        if any(value in ("--output-dir", "--data-dir") for value in core_args):
            raise ValueError(
                f"{identifier} may not redirect output or model data."
            )
        seeds = task.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{identifier} seeds must be non-empty.")
        converted_seeds = [int(seed) for seed in seeds]
        if len(converted_seeds) != len(set(converted_seeds)):
            raise ValueError(f"{identifier} contains duplicate timing seeds.")
        if int(task.get("prefix_mechanics_per_seed", 0)) < 1:
            raise ValueError(f"{identifier} prefix mechanics must be positive.")
        if int(task.get("warmup_mechanics", 0)) < 1:
            raise ValueError(f"{identifier} warm-up mechanics must be positive.")
        if int(task.get("warmup_seed")) in set(converted_seeds):
            raise ValueError(
                f"{identifier} warm-up seed must not be a measured seed."
            )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Protocol task identifiers must be unique.")
    return protocol


def _reconstruct_baseline_assigned_mechanics(random_summary):
    hits = [
        int(value)
        for value in random_summary.get("first_success_mechanical_steps", [])
    ]
    censored_count = int(random_summary.get("censored_count", -1))
    censor_cap = int(random_summary.get("common_mechanical_censor_cap", 0))
    search_count = int(random_summary.get("search_count", -1))
    if censored_count < 0 or censor_cap < 1 or search_count < 1:
        raise ValueError("Baseline random summary has invalid censor metadata.")
    assigned = np.asarray(
        hits + [censor_cap] * censored_count, dtype=np.float64
    )
    if len(assigned) != search_count:
        raise ValueError(
            "Baseline successes plus censored trials do not equal search_count."
        )
    reported = _finite_float(
        random_summary.get("restricted_mean_first_success_mechanical_steps"),
        "baseline mechanics RMST",
        positive=True,
    )
    reconstructed = float(np.mean(assigned))
    if not math.isclose(reconstructed, reported, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(
            "Baseline mechanics RMST cannot be reconstructed at its common "
            f"censor: reconstructed={reconstructed}, reported={reported}."
        )
    if not random_summary.get("rmst_is_exact_common_administrative_censor"):
        raise ValueError("Baseline RMST lacks exact common-censor provenance.")
    return assigned, reported


def prepare_tasks(protocol, baseline, selected_task_ids=None):
    validate_protocol(protocol)
    if not baseline.get("all_compatible"):
        raise ValueError("Random baseline compatibility audit is not complete.")
    baseline_tasks = baseline.get("tasks")
    if not isinstance(baseline_tasks, dict):
        raise ValueError("Baseline compatibility report lacks tasks.")

    available = [str(task["task"]) for task in protocol["tasks"]]
    if selected_task_ids is None:
        selected = set(available)
    else:
        selected_list = [str(value) for value in selected_task_ids]
        if len(selected_list) != len(set(selected_list)):
            raise ValueError("Selected task identifiers must be unique.")
        unknown = sorted(set(selected_list) - set(available))
        if unknown:
            raise ValueError("Unknown selected tasks: " + ", ".join(unknown))
        selected = set(selected_list)

    prepared = []
    for task_index, task in enumerate(protocol["tasks"]):
        task_id = str(task["task"])
        if task_id not in selected:
            continue
        baseline_task = baseline_tasks.get(task_id)
        if not isinstance(baseline_task, dict) or not baseline_task.get(
            "compatible"
        ):
            raise ValueError(f"Baseline task {task_id} is absent or incompatible.")
        args = discovery._parse_core_args(task["problem"], task["core_args"])
        current_signature = _validate_mechanics_against_baseline(
            args, baseline_task
        )
        adapter = discovery._adapter_for_problem(task["problem"])
        node, member, fix, files = adapter.load_problem_data(args.data_dir)
        model_hashes = {
            name: discovery._file_sha256(path) for name, path in files.items()
        }
        assigned, reported_rmst = _reconstruct_baseline_assigned_mechanics(
            baseline_task["random_summary"]
        )
        prepared.append({
            "task_index": int(task_index),
            "task_id": task_id,
            "protocol_task": task,
            "problem": task["problem"],
            "args": args,
            "adapter": adapter,
            "node": node,
            "member": member,
            "fix": fix,
            "model_file_sha256": model_hashes,
            "effective_args_sha256": discovery._canonical_digest({
                key: discovery._jsonable(value)
                for key, value in vars(args).items()
                if key not in ("output_dir", "three_dimensional_adapter")
            }),
            "mechanics_signature": current_signature,
            "collision_identity": _collision_identity(task["problem"]),
            "baseline_task": baseline_task,
            "baseline_assigned_mechanics": assigned,
            "baseline_reported_rmst": reported_rmst,
        })
    if not prepared:
        raise ValueError("No calibration tasks were selected.")
    return prepared


def _episode_trace_row(summary, actions, cumulative_steps):
    return {
        "steps": int(summary["steps"]),
        "cumulative_steps": int(cumulative_steps),
        "projection_attempts": int(summary["projection_attempts"]),
        "termination_reason": summary.get("termination_reason"),
        "strict_compact_success": int(summary.get(
            "strict_compact_success", summary.get("success", 0)
        )),
        "self_crossing": int(summary.get("self_crossing", 0)),
        "wrong_flat": int(summary.get("wrong_flat", 0)),
        "timeout": int(summary.get("timeout", 0)),
        "correction_failed": int(summary.get("correction_failed", 0)),
        "no_modes": int(summary.get("no_modes", 0)),
        "total_reward": float(summary.get("total_reward", 0.0)),
        "action_trace_sha256": discovery.legacy._episode_action_digest(actions),
    }


def run_fixed_prefix(prepared_task, seed, mechanics_budget):
    """Run one exact mechanics prefix and return timing and trace evidence."""

    seed = int(seed)
    mechanics_budget = int(mechanics_budget)
    if mechanics_budget < 1:
        raise ValueError("mechanics_budget must be positive.")
    args = prepared_task["args"]
    adapter = prepared_task["adapter"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = adapter.make_environment(
        args,
        prepared_task["node"],
        prepared_task["member"],
        prepared_task["fix"],
    )
    uniform_rng = np.random.default_rng(seed)
    sampler = ori3.UniformRandomAgent(args.max_step_width, uniform_rng)
    agent = discovery._NoUpdateAgent(sampler)

    mechanical_steps = 0
    projector_calls = 0
    rollout_wall_seconds = 0.0
    rollout_cpu_seconds = 0.0
    first_success_step = None
    episodes = []
    terminations = Counter()
    maximum_decomposition_error = 0.0
    nonfinite_reward_components = 0

    while mechanical_steps < mechanics_budget:
        remaining = mechanics_budget - mechanical_steps
        budgeted = discovery.BudgetedEnvironment(environment, remaining)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        summary, _, actions = adapter.run_episode(
            budgeted, agent, deterministic=False, training=True
        )
        rollout_cpu_seconds += time.process_time() - cpu_start
        rollout_wall_seconds += time.perf_counter() - wall_start
        steps = int(summary["steps"])
        if not 1 <= steps <= remaining:
            raise RuntimeError(
                f"Prefix rollout returned {steps} steps with {remaining} left."
            )
        mechanical_steps += steps
        projector_calls += int(summary["projection_attempts"])
        trace_row = _episode_trace_row(summary, actions, mechanical_steps)
        episodes.append(trace_row)
        terminations[str(summary.get("termination_reason", "unknown"))] += 1
        strict_success = bool(trace_row["strict_compact_success"])
        if strict_success and first_success_step is None:
            first_success_step = int(mechanical_steps)
        maximum_decomposition_error = max(
            maximum_decomposition_error,
            abs(float(summary.get("ori5_reward_decomposition_error", 0.0))),
        )
        nonfinite_reward_components += int(
            summary.get("ori5_nonfinite_reward_component_count", 0)
        )

    if mechanical_steps != mechanics_budget:
        raise RuntimeError("Fixed-prefix mechanics budget was not exact.")
    if not math.isfinite(rollout_wall_seconds) or rollout_wall_seconds <= 0.0:
        raise RuntimeError("Measured rollout wall time is not positive and finite.")
    return {
        "task": prepared_task["task_id"],
        "seed": seed,
        "mechanical_steps": int(mechanical_steps),
        "projector_calls": int(projector_calls),
        "episodes": int(len(episodes)),
        "rollout_wall_seconds": float(rollout_wall_seconds),
        "rollout_cpu_seconds": float(rollout_cpu_seconds),
        "mechanical_steps_per_wall_second": float(
            mechanical_steps / rollout_wall_seconds
        ),
        "projector_calls_per_wall_second": float(
            projector_calls / rollout_wall_seconds
        ),
        "first_strict_success_step_within_prefix": first_success_step,
        "strict_successes_within_prefix": int(sum(
            row["strict_compact_success"] for row in episodes
        )),
        "termination_counts": dict(sorted(terminations.items())),
        "maximum_abs_reward_decomposition_error": float(
            maximum_decomposition_error
        ),
        "nonfinite_reward_component_count": int(nonfinite_reward_components),
        "prefix_trace_sha256": discovery._canonical_digest(episodes),
    }


def _percentile_interval(values, confidence_level):
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return [float(lower), float(upper)]


def bootstrap_task(blocks, baseline_assigned_mechanics, *, replicates,
                   confidence_level, seed):
    """Cluster-bootstrap timing blocks and stored first-hit trials."""

    steps = np.asarray([row["mechanical_steps"] for row in blocks], dtype=float)
    seconds = np.asarray(
        [row["rollout_wall_seconds"] for row in blocks], dtype=float
    )
    assigned = np.asarray(baseline_assigned_mechanics, dtype=float)
    if len(steps) < 2 or len(assigned) < 2:
        raise ValueError("Bootstrap requires at least two timing and RMST trials.")
    if np.any(steps <= 0.0) or np.any(seconds <= 0.0):
        raise ValueError("Bootstrap timing blocks must be positive.")
    replicates = int(replicates)
    rng = np.random.default_rng(int(seed))
    timing_indices = rng.integers(
        0, len(steps), size=(replicates, len(steps))
    )
    baseline_indices = rng.integers(
        0, len(assigned), size=(replicates, len(assigned))
    )
    sampled_step_totals = steps[timing_indices].sum(axis=1)
    sampled_second_totals = seconds[timing_indices].sum(axis=1)
    sampled_seconds_per_step = sampled_second_totals / sampled_step_totals
    sampled_throughput = 1.0 / sampled_seconds_per_step
    sampled_mechanics_rmst = assigned[baseline_indices].mean(axis=1)
    sampled_wall_rmst = sampled_mechanics_rmst * sampled_seconds_per_step

    total_steps = float(steps.sum())
    total_seconds = float(seconds.sum())
    point_seconds_per_step = total_seconds / total_steps
    point_throughput = 1.0 / point_seconds_per_step
    return {
        "method": (
            "nonparametric cluster bootstrap; timing blocks resampled by seed "
            "and stored common-censor first-hit trials resampled independently"
        ),
        "replicates": replicates,
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(seed),
        "mechanical_steps_per_wall_second": {
            "estimate": float(point_throughput),
            "bootstrap_ci": _percentile_interval(
                sampled_throughput, confidence_level
            ),
        },
        "wall_seconds_per_mechanical_step": {
            "estimate": float(point_seconds_per_step),
            "bootstrap_ci": _percentile_interval(
                sampled_seconds_per_step, confidence_level
            ),
        },
        "baseline_mechanics_rmst": {
            "estimate": float(np.mean(assigned)),
            "bootstrap_ci": _percentile_interval(
                sampled_mechanics_rmst, confidence_level
            ),
        },
        "joint_calibrated_wall_rmst_seconds": {
            "estimate": float(np.mean(assigned) * point_seconds_per_step),
            "bootstrap_ci": _percentile_interval(
                sampled_wall_rmst, confidence_level
            ),
        },
        "timing_only_calibrated_wall_rmst_seconds_ci": _percentile_interval(
            float(np.mean(assigned)) * sampled_seconds_per_step,
            confidence_level,
        ),
    }


@contextmanager
def _single_thread_context():
    torch.set_num_threads(1)
    limiter = nullcontext()
    backend = "torch_only; threadpoolctl unavailable"
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
        backend = "torch.set_num_threads(1) + threadpoolctl(limits=1)"
    except ImportError:
        pass
    with limiter:
        yield backend


def _system_identity(thread_limiter):
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "process_id": os.getpid(),
        "process_count_used_by_calibrator": 1,
        "thread_limiter": thread_limiter,
        "torch_num_threads": torch.get_num_threads(),
    }


def _task_identity(prepared_task):
    task = prepared_task["protocol_task"]
    return {
        "task": prepared_task["task_id"],
        "problem": prepared_task["problem"],
        "core_args": list(task["core_args"]),
        "seeds": [int(seed) for seed in task["seeds"]],
        "prefix_mechanics_per_seed": int(task["prefix_mechanics_per_seed"]),
        "warmup_seed": int(task["warmup_seed"]),
        "warmup_mechanics": int(task["warmup_mechanics"]),
        "effective_args_sha256": prepared_task["effective_args_sha256"],
        "mechanics_signature": prepared_task["mechanics_signature"],
        "model_file_sha256": prepared_task["model_file_sha256"],
        "collision_identity": prepared_task["collision_identity"],
    }


def build_plan(protocol_path=DEFAULT_PROTOCOL,
               baseline_path=DEFAULT_BASELINE_SUMMARY,
               selected_task_ids=None):
    protocol_path = Path(protocol_path).resolve()
    baseline_path = Path(baseline_path).resolve()
    protocol = validate_protocol(_load_json(protocol_path))
    baseline = _load_json(baseline_path)
    prepared = prepare_tasks(protocol, baseline, selected_task_ids)
    measured_steps = sum(
        len(row["protocol_task"]["seeds"])
        * int(row["protocol_task"]["prefix_mechanics_per_seed"])
        for row in prepared
    )
    warmup_steps = sum(
        int(row["protocol_task"]["warmup_mechanics"]) for row in prepared
    )
    return {
        "protocol_path": protocol_path,
        "baseline_path": baseline_path,
        "protocol": protocol,
        "baseline": baseline,
        "prepared": prepared,
        "selected_tasks": [row["task_id"] for row in prepared],
        "all_protocol_tasks_selected": len(prepared) == len(protocol["tasks"]),
        "planned_measured_mechanics": int(measured_steps),
        "planned_excluded_warmup_mechanics": int(warmup_steps),
        "protocol_file_sha256": discovery._file_sha256(protocol_path),
        "protocol_content_sha256": discovery._canonical_digest(protocol),
        "baseline_file_sha256": discovery._file_sha256(baseline_path),
        "source_sha256": _source_provenance(),
    }


def _write_markdown_report(path, result):
    lines = [
        "# ori5 uniform-random wall-time calibration",
        "",
        "> This report contains calibrated random wall-time estimates. It does "
        "**not** contain directly measured Actor--Critic/RL wall time.",
        "",
        f"Protocol: `{result['protocol_id']}`  ",
        f"Run fingerprint: `{result['run_fingerprint']}`  ",
        f"Generated: `{result['completed_utc']}`",
        "",
        "| Task | Timing blocks | steps/s (bootstrap CI) | random mechanics RMST | calibrated random wall RMST (joint bootstrap CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in result["tasks"]:
        statistics = task["statistics"]
        throughput = statistics["mechanical_steps_per_wall_second"]
        wall = statistics["joint_calibrated_wall_rmst_seconds"]
        wall_hours = wall["estimate"] / 3600.0
        wall_ci_hours = [value / 3600.0 for value in wall["bootstrap_ci"]]
        lines.append(
            f"| {task['task']} | {len(task['blocks'])} | "
            f"{throughput['estimate']:.3f} "
            f"[{throughput['bootstrap_ci'][0]:.3f}, "
            f"{throughput['bootstrap_ci'][1]:.3f}] | "
            f"{statistics['baseline_mechanics_rmst']['estimate']:.3f} | "
            f"{wall_hours:.4f} h "
            f"[{wall_ci_hours[0]:.4f}, {wall_ci_hours[1]:.4f}] |"
        )
    lines.extend([
        "",
        "The joint interval resamples both fixed-seed timing blocks and the "
        "stored common-censor random first-hit trials. The conversion assumes "
        "that seconds per mechanics step observed on the fixed prefixes is "
        "representative of the longer random search. Machine-load drift and "
        "RL forward/backward/update overhead are outside this interval.",
        "",
        f"Measured rollout mechanics: {result['totals']['measured_mechanics']}  ",
        f"Measured rollout wall seconds: {result['totals']['measured_rollout_wall_seconds']:.6f}  ",
        f"Excluded warm-up mechanics: {result['totals']['excluded_warmup_mechanics']}",
        "",
    ])
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(lines))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_calibration(protocol_path=DEFAULT_PROTOCOL,
                    baseline_path=DEFAULT_BASELINE_SUMMARY,
                    output_directory=None, selected_task_ids=None,
                    progress=True):
    plan = build_plan(protocol_path, baseline_path, selected_task_ids)
    protocol = plan["protocol"]
    prepared_tasks = plan["prepared"]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "calibrator_version": CALIBRATOR_VERSION,
        "protocol_content_sha256": plan["protocol_content_sha256"],
        "protocol_file_sha256": plan["protocol_file_sha256"],
        "baseline_file_sha256": plan["baseline_file_sha256"],
        "reward_constants_sha256": discovery.FIXED_REWARD_SHA256,
        "selected_tasks": plan["selected_tasks"],
        "task_identity": [_task_identity(row) for row in prepared_tasks],
        "source_sha256": plan["source_sha256"],
    }
    run_fingerprint = discovery._canonical_digest(identity)
    if output_directory is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = (
            DEFAULT_OUTPUT_PARENT / f"{stamp}_{run_fingerprint[:12]}"
        )
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Calibration output directory is not empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    started_utc = _utc_now()
    run_config = {
        **identity,
        "run_fingerprint": run_fingerprint,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(plan["protocol_path"]),
        "baseline_path": str(plan["baseline_path"]),
        "protocol_snapshot": protocol,
        "all_protocol_tasks_selected": plan["all_protocol_tasks_selected"],
        "planned_measured_mechanics": plan["planned_measured_mechanics"],
        "planned_excluded_warmup_mechanics": (
            plan["planned_excluded_warmup_mechanics"]
        ),
        "started_utc": started_utc,
        "command_line": [str(value) for value in sys.argv],
        "measurement_class": "calibrated_uniform_random_wall_equivalent",
        "direct_rl_wall_measurement": False,
    }
    discovery._atomic_json(output_directory / "run_config.json", run_config)

    task_results = []
    all_blocks = []
    total_warmup = 0
    with _single_thread_context() as limiter:
        system_identity = _system_identity(limiter)
        for prepared in prepared_tasks:
            task = prepared["protocol_task"]
            if progress:
                print(
                    f"warm-up {prepared['task_id']}: "
                    f"{int(task['warmup_mechanics'])} mechanics steps",
                    flush=True,
                )
            with prepared["adapter"].patched_three_dimensional_runtime(
                prepared["args"], prepared["member"]
            ):
                warmup = run_fixed_prefix(
                    prepared, task["warmup_seed"], task["warmup_mechanics"]
                )
                total_warmup += int(warmup["mechanical_steps"])
                blocks = []
                for block_index, seed in enumerate(task["seeds"], start=1):
                    gc.collect()
                    block = run_fixed_prefix(
                        prepared, seed, task["prefix_mechanics_per_seed"]
                    )
                    block["block_index"] = int(block_index)
                    block["model_file_sha256"] = prepared[
                        "model_file_sha256"
                    ]
                    blocks.append(block)
                    all_blocks.append(block)
                    if progress:
                        print(
                            f"{prepared['task_id']} block {block_index}/"
                            f"{len(task['seeds'])}: seed={int(seed)} "
                            f"{block['mechanical_steps_per_wall_second']:.2f} "
                            "mechanics steps/s",
                            flush=True,
                        )
            statistics = bootstrap_task(
                blocks,
                prepared["baseline_assigned_mechanics"],
                replicates=protocol["bootstrap_replicates"],
                confidence_level=protocol["confidence_level"],
                seed=int(protocol["bootstrap_seed"])
                + int(prepared["task_index"]),
            )
            if not math.isclose(
                statistics["baseline_mechanics_rmst"]["estimate"],
                prepared["baseline_reported_rmst"],
                rel_tol=1e-12,
                abs_tol=1e-8,
            ):
                raise RuntimeError("Bootstrap baseline RMST point mismatch.")
            task_results.append({
                **_task_identity(prepared),
                "timing_definition": protocol["timing_scope"],
                "warmup": {
                    "seed": int(task["warmup_seed"]),
                    "mechanical_steps": int(warmup["mechanical_steps"]),
                    "prefix_trace_sha256": warmup["prefix_trace_sha256"],
                    "timing_excluded": True,
                },
                "blocks": blocks,
                "statistics": statistics,
                "baseline_source": {
                    "baseline_kind": prepared["baseline_task"].get(
                        "baseline_kind"
                    ),
                    "search_count": int(len(
                        prepared["baseline_assigned_mechanics"]
                    )),
                    "common_censor_exact": True,
                },
            })

    completed_utc = _utc_now()
    total_mechanics = int(sum(row["mechanical_steps"] for row in all_blocks))
    total_wall = float(sum(row["rollout_wall_seconds"] for row in all_blocks))
    total_cpu = float(sum(row["rollout_cpu_seconds"] for row in all_blocks))
    result = {
        **identity,
        "run_fingerprint": run_fingerprint,
        "protocol_id": protocol["protocol_id"],
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "complete": True,
        "all_protocol_tasks_selected": plan["all_protocol_tasks_selected"],
        "system_identity": system_identity,
        "measurement_definition": {
            "class": "calibrated_uniform_random_wall_equivalent",
            "rollout_scope": protocol["timing_scope"],
            "fixed_seed_prefixes": True,
            "single_process": True,
            "continue_after_strict_success": True,
            "direct_uniform_first_discovery_wall_measurement": False,
            "direct_rl_wall_measurement": False,
            "rl_update_time_included": False,
            "interpretation": (
                "Existing random mechanics RMST multiplied by current-machine "
                "seconds per mechanics step measured on stored prefixes."
            ),
        },
        "tasks": task_results,
        "totals": {
            "measured_mechanics": total_mechanics,
            "measured_rollout_wall_seconds": total_wall,
            "measured_rollout_cpu_seconds": total_cpu,
            "excluded_warmup_mechanics": int(total_warmup),
            "timing_blocks": int(len(all_blocks)),
        },
        "limitations": [
            "Calibrated random wall RMST is not a direct RL wall-time measurement.",
            "The interval assumes fixed-prefix seconds per mechanics step "
            "represent longer uniform searches.",
            "The interval does not include machine-load drift outside this run.",
            "RL neural inference, backpropagation, optimizer, checkpoint, and "
            "evaluation overhead are excluded.",
        ],
    }
    discovery._atomic_json(output_directory / "calibration_results.json", result)

    block_fieldnames = (
        "task", "block_index", "seed", "mechanical_steps", "projector_calls",
        "episodes", "rollout_wall_seconds", "rollout_cpu_seconds",
        "mechanical_steps_per_wall_second",
        "projector_calls_per_wall_second",
        "first_strict_success_step_within_prefix",
        "strict_successes_within_prefix",
        "maximum_abs_reward_decomposition_error",
        "nonfinite_reward_component_count", "prefix_trace_sha256",
    )
    discovery._atomic_csv(
        output_directory / "timing_blocks.csv", all_blocks, block_fieldnames
    )
    summary_rows = []
    for task in task_results:
        statistics = task["statistics"]
        throughput = statistics["mechanical_steps_per_wall_second"]
        wall = statistics["joint_calibrated_wall_rmst_seconds"]
        summary_rows.append({
            "task": task["task"],
            "timing_blocks": len(task["blocks"]),
            "mechanical_steps_per_wall_second": throughput["estimate"],
            "throughput_ci_lower": throughput["bootstrap_ci"][0],
            "throughput_ci_upper": throughput["bootstrap_ci"][1],
            "baseline_mechanics_rmst": statistics[
                "baseline_mechanics_rmst"
            ]["estimate"],
            "calibrated_random_wall_rmst_seconds": wall["estimate"],
            "calibrated_random_wall_rmst_ci_lower_seconds": (
                wall["bootstrap_ci"][0]
            ),
            "calibrated_random_wall_rmst_ci_upper_seconds": (
                wall["bootstrap_ci"][1]
            ),
            "direct_rl_wall_measurement": False,
        })
    discovery._atomic_csv(
        output_directory / "task_summary.csv",
        summary_rows,
        tuple(summary_rows[0]),
    )
    _write_markdown_report(output_directory / "REPORT.md", result)
    return output_directory, result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Single-process fixed-prefix wall-time calibration for ori5 "
            "uniform-random baselines."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Optional subset of stored task identifiers; seeds/prefixes stay fixed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate hashes/configuration and print the mechanics plan only.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.dry_run:
        plan = build_plan(
            args.protocol, args.baseline_summary, args.tasks
        )
        public_plan = {
            "protocol_id": plan["protocol"]["protocol_id"],
            "selected_tasks": plan["selected_tasks"],
            "all_protocol_tasks_selected": plan[
                "all_protocol_tasks_selected"
            ],
            "planned_measured_mechanics": plan[
                "planned_measured_mechanics"
            ],
            "planned_excluded_warmup_mechanics": plan[
                "planned_excluded_warmup_mechanics"
            ],
            "timing_blocks": int(sum(
                len(row["protocol_task"]["seeds"])
                for row in plan["prepared"]
            )),
            "protocol_file_sha256": plan["protocol_file_sha256"],
            "baseline_file_sha256": plan["baseline_file_sha256"],
            "reward_constants_sha256": discovery.FIXED_REWARD_SHA256,
            "direct_rl_wall_measurement": False,
        }
        print(json.dumps(public_plan, indent=2))
        return public_plan
    output_directory, result = run_calibration(
        protocol_path=args.protocol,
        baseline_path=args.baseline_summary,
        output_directory=args.output_dir,
        selected_task_ids=args.tasks,
        progress=not args.quiet,
    )
    print(json.dumps({
        "complete": result["complete"],
        "run_fingerprint": result["run_fingerprint"],
        "output_directory": str(output_directory),
        "direct_rl_wall_measurement": False,
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
