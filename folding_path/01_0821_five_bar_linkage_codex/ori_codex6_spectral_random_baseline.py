"""Exact-mechanics uniform baseline for the promoted ori6 spectral candidate.

This is intentionally a uniform-only harness.  It uses the same mechanics,
termination semantics, and spectral/rejected-crossing environment as the RL
candidate, but actions are sampled by ``UniformRandomAgent``.  An unsuccessful
trial is administratively censored at exactly the requested mechanics budget;
the final ordinary transition reward is not replaced by an environment timeout
reward.

The production orchestrator fixes the initial protocol to all five tasks,
seeds 5101--5110, and 102400 mechanics per trial.  A larger-budget extension
uses the same ten seeds in a new, fingerprinted output root so that the initial
prefix can be checked rather than silently mixed with a different experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import torch

import ori_codex3 as ori3
import ori_codex4_3d_first_discovery as legacy
import ori_codex6_rejected_crossing as rejected
import ori_codex6_spectral_subspace as spectral
import ori_codex6_spectral_subspace_3d as spectral3d
import run_ori_codex6_experiments as base_runner


SCHEMA_VERSION = 1
HARNESS_VERSION = "1.0.0"
METHOD = "uniform"
ACTION_LAW = "isotropic current-mode direction and uniform signed width"
ACTION_TRACE_CHAIN = (
    "h_0=64 zero hex digits; h_i=sha256(bytes.fromhex(h_(i-1)) || "
    "canonical_json(action_i))"
)
TASK_NAMES = tuple(base_runner.TASKS)
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:16]}")
        try:
            with open(temporary, "x", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return
        except FileExistsError:
            continue
        finally:
            if temporary.exists():
                temporary.unlink()
    raise RuntimeError(f"Could not allocate atomic temporary near {path}")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = [
        "method", "seed", "episode", "success", "strict_compact_success",
        "termination_reason", "mechanical_steps", "projector_calls",
        "cumulative_mechanical_steps", "cumulative_projector_calls",
        "rollout_seconds", "cumulative_rollout_seconds", "total_reward",
        "flatness_ratio", "spread", "spread_error", "self_crossing",
        "correction_failed", "timeout", "near_miss_q", "near_miss_bonus",
        "action_trace_sha256",
    ]
    extras = sorted({key for row in rows for key in row} - set(fixed))
    fields = fixed + extras
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:16]}")
    try:
        with open(temporary, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:16]}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_planar_problem_data(data_directory):
    directory = (
        Path(data_directory).resolve()
        if data_directory is not None else SCRIPT_DIRECTORY
    )
    paths = {
        "node": directory / "node_initial.dat",
        "member": directory / "member.dat",
        "fix": directory / "fix.dat",
    }
    node = np.atleast_2d(np.loadtxt(paths["node"], delimiter=","))
    member = np.atleast_2d(np.loadtxt(
        paths["member"], delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(paths["fix"], delimiter=",", dtype=int))
    return node, member, fix, paths


class PlanarSpectralAdapter:
    load_problem_data = staticmethod(load_planar_problem_data)
    make_environment = staticmethod(spectral.make_environment)
    run_episode = staticmethod(spectral.run_episode)

    @staticmethod
    def patched_three_dimensional_runtime(args, member):
        del args, member
        return nullcontext(None)


PLANAR_ADAPTER = PlanarSpectralAdapter()


def adapter_for_task(task_name: str):
    task = base_runner.TASKS[task_name]
    return spectral3d if task.is_three_dimensional else PLANAR_ADAPTER


def candidate_args(task_name: str, seed: int):
    """Parse the exact production candidate command without running training."""

    task = base_runner.TASKS[task_name]
    profile = base_runner.PROFILES[task.profile]
    protocol = base_runner.Protocol(
        "spectral_uniform_argument_identity", 1, (int(seed),),
        (task_name,), ("ori6",), 1, 1, 1, 1, 1, 8510000, 8520000,
    )
    identity = {
        "protocol": {
            **base_runner.asdict(protocol),
        },
        "task": base_runner.asdict(task),
        "profile": base_runner.asdict(profile),
        "seed": int(seed),
        "implementation": "ori6",
        "candidate_mode": (
            base_runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        ),
    }
    unused = base_runner.EXPERIMENT_ROOT / "_unused_spectral_uniform_parse"
    command = base_runner.build_command(identity, unused)
    arguments = command[2:]
    args = (
        spectral3d.parse_args(arguments)
        if task.is_three_dimensional else spectral.parse_args(arguments)
    )
    # The random harness has no evaluation and never writes through train().
    args.random_search = False
    return args


def source_hashes() -> dict[str, str]:
    names = (
        "ori_codex2.py",
        "ori_codex3.py",
        "ori_codex4_3d_first_discovery.py",
        "ori_codex5.py",
        "ori_codex6.py",
        "ori_codex6_3d.py",
        "ori_codex6_rejected_crossing.py",
        "ori_codex6_rejected_crossing_3d.py",
        "ori_codex6_spectral_subspace.py",
        "ori_codex6_spectral_subspace_3d.py",
        "collision_3d.py",
        Path(__file__).name,
    )
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in names
    }


def model_hashes(task_name: str) -> dict[str, str]:
    return base_runner.model_hashes(base_runner.TASKS[task_name])


def effective_args_identity(args) -> dict:
    excluded = {
        "output_dir", "three_dimensional_adapter",
        "selection_evaluation_seed", "final_evaluation_seed",
        "seed",
    }
    return {
        key: jsonable(value) for key, value in vars(args).items()
        if key not in excluded
    }


def root_identity(task_name: str, mechanics_budget: int,
                  seeds: list[int]) -> dict:
    task = base_runner.TASKS[task_name]
    args = candidate_args(task_name, seeds[0])
    reward = spectral.reward_parameters(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "candidate": spectral.CANDIDATE_NAME,
        "candidate_schema_version": spectral.CANDIDATE_SCHEMA_VERSION,
        "parent_candidate": rejected.CANDIDATE_NAME,
        "crossing_policy": rejected.CROSSING_POLICY,
        "task": task_name,
        "task_specification": base_runner.asdict(task),
        "profile": base_runner.asdict(base_runner.PROFILES[task.profile]),
        "method": METHOD,
        "action_law": ACTION_LAW,
        "action_trace_chain": ACTION_TRACE_CHAIN,
        "action_trace_audit_mechanics": [
            min(INITIAL_AUDIT_MECHANICS, int(mechanics_budget))
        ],
        "seeds": [int(seed) for seed in seeds],
        "mechanics_budget": int(mechanics_budget),
        "episode_cap": None,
        "evaluation_rollouts": 0,
        "effective_candidate_args": effective_args_identity(args),
        "reward_parameters": reward,
        "reward_sha256": base_runner.candidate_reward_sha256(
            base_runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        ),
        "transition_semantics_sha256": (
            base_runner.candidate_transition_semantics_sha256(
                base_runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
            )
        ),
        "spectral_formula_sha256": spectral.SPECTRAL_FORMULA_SHA256,
        "model_sha256": model_hashes(task_name),
        "source_sha256": source_hashes(),
        "budget_stop_reward": (
            "unchanged candidate reward from final ordinary valid transition"
        ),
    }


def make_root_config(task_name: str, mechanics_budget: int,
                     seeds: list[int]) -> dict:
    identity = root_identity(task_name, mechanics_budget, seeds)
    return {
        **identity,
        "fingerprint_identity": identity,
        "root_fingerprint": canonical_digest(identity),
        "created_utc": utc_now(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "comparison_rules": {
            "success": "strict_compact_success == 1",
            "stop": "first strict success or exact mechanics budget",
            "censored_cost": "common mechanics budget",
            "episode_cap": "forbidden",
            "random_action_law": ACTION_LAW,
        },
    }


def validate_root_config(config: dict) -> None:
    identity = config.get("fingerprint_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Random root lacks fingerprint_identity")
    if canonical_digest(identity) != config.get("root_fingerprint"):
        raise RuntimeError("Random root fingerprint mismatch")
    for key, value in identity.items():
        if config.get(key) != value:
            raise RuntimeError(f"Random root identity mismatch for {key}")
    if config.get("candidate") != spectral.CANDIDATE_NAME:
        raise RuntimeError("Random root candidate mismatch")
    if config.get("method") != METHOD:
        raise RuntimeError("Random root is not uniform-only")
    if config.get("task") not in TASK_NAMES:
        raise RuntimeError("Random root task mismatch")
    if config.get("episode_cap") is not None:
        raise RuntimeError("Episode cap is forbidden")
    if int(config.get("mechanics_budget", 0)) < 1:
        raise RuntimeError("Mechanics budget must be positive")
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("Random root seeds must be unique and non-empty")


def action_digest(actions: list[dict]) -> str:
    compact = [{
        "physical_direction": action.get("physical_direction"),
        "mode_coefficients": action.get("mode_coefficients"),
        "requested_width": action.get("requested_width"),
    } for action in actions]
    return canonical_digest(compact)


INITIAL_AUDIT_MECHANICS = 102400


def chained_action_trace(state: str, actions: list[dict],
                         mechanics_before: int,
                         checkpoints: set[int],
                         recorded: dict[str, str]) -> str:
    if len(state) != 64:
        raise RuntimeError("Invalid chained action-trace state")
    for offset, action in enumerate(actions, start=1):
        compact = {
            "physical_direction": action.get("physical_direction"),
            "mode_coefficients": action.get("mode_coefficients"),
            "requested_width": action.get("requested_width"),
        }
        encoded = json.dumps(
            compact, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        state = hashlib.sha256(bytes.fromhex(state) + encoded).hexdigest()
        cumulative = int(mechanics_before + offset)
        if cumulative in checkpoints:
            recorded[str(cumulative)] = state
    return state


SUMMARY_FIELDS = (
    "reward_design", "reward_schema_version",
    "ori6_reward_decomposition_error",
    "ori6_maximum_abs_step_reward_decomposition_error",
    "ori6_nonfinite_reward_component_count",
    "ori6_candidate", "ori6_candidate_schema_version",
    "ori6_crossing_policy", "ori6_rejected_crossing_count",
    "ori6_rejected_crossing_rate", "ori6_rejected_crossing_streak_count",
    "ori6_max_consecutive_rejected_crossings",
    "ori6_escape_after_rejected_crossing_count",
    "ori6_rejected_crossing_escape_rate",
    "ori6_unescaped_rejected_crossing_streak_count",
    "ori6_success_after_rejected_crossing",
    "ori6_total_rejected_crossing_reward",
    "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
    "ori6_spectral_candidate", "ori6_spectral_candidate_schema_version",
    "ori6_spectral_formula_sha256", "ori6_spectral_parent_candidate",
    "ori6_spectral_weight", "ori6_spectral_relative_epsilon",
    "ori6_spectral_total_progress_reward",
    "ori6_spectral_total_rollback_reward", "ori6_spectral_total_reward",
    "ori6_spectral_cumulative_reward",
    "ori6_spectral_terminal_potential", "ori6_spectral_terminal_delta",
    "ori6_spectral_terminal_rho", "ori6_spectral_maximum_potential",
    "ori6_spectral_maximum_delta", "ori6_spectral_maximum_rho",
    "ori6_spectral_rollback_count",
    "ori6_spectral_reward_decomposition_error",
    "ori6_spectral_maximum_abs_step_reward_decomposition_error",
    "ori6_spectral_nonfinite_component_count",
    "wrong_flat", "no_modes",
)


def history_row(seed: int, episode: int, summary: dict,
                cumulative_steps: int, cumulative_calls: int,
                episode_seconds: float, rollout_seconds: float,
                actions: list[dict]) -> dict:
    row = {
        "method": METHOD,
        "seed": int(seed),
        "episode": int(episode),
        "success": int(summary.get("success", 0)),
        "strict_compact_success": int(summary.get(
            "strict_compact_success", summary.get("success", 0)
        )),
        "termination_reason": summary.get("termination_reason", "unknown"),
        "mechanical_steps": int(summary["steps"]),
        "projector_calls": int(summary["projection_attempts"]),
        "cumulative_mechanical_steps": int(cumulative_steps),
        "cumulative_projector_calls": int(cumulative_calls),
        "rollout_seconds": float(episode_seconds),
        "cumulative_rollout_seconds": float(rollout_seconds),
        "total_reward": float(summary["total_reward"]),
        "flatness_ratio": float(summary["flatness_ratio"]),
        "spread": float(summary.get("spread", math.nan)),
        "spread_error": float(summary.get("spread_error", math.nan)),
        "self_crossing": int(summary.get("self_crossing", 0)),
        "correction_failed": int(summary.get("correction_failed", 0)),
        "timeout": int(summary.get("timeout", 0)),
        "near_miss_q": float(summary.get("near_miss_q", 0.0)),
        "near_miss_bonus": float(summary.get("near_miss_bonus", 0.0)),
        "action_trace_sha256": action_digest(actions),
    }
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field)
    return row


def trial_directory(output_root: Path, seed: int) -> Path:
    return Path(output_root) / "trials" / METHOD / f"seed{int(seed)}"


def write_checkpoint(path: Path, *, fingerprint: str, seed: int,
                     episodes: int, mechanics: int, projector_calls: int,
                     rollout_seconds: float, history: list[dict],
                     uniform_rng, action_chain_sha256: str,
                     action_prefix_sha256_by_mechanics: dict[str, str]) -> None:
    atomic_torch_save(path, {
        "schema_version": SCHEMA_VERSION,
        "root_fingerprint": fingerprint,
        "seed": int(seed),
        "episodes": int(episodes),
        "mechanical_steps": int(mechanics),
        "projector_calls": int(projector_calls),
        "rollout_seconds": float(rollout_seconds),
        "history": history,
        "numpy_rng_state": np.random.get_state(),
        "uniform_rng_state": copy.deepcopy(uniform_rng.bit_generator.state),
        "action_chain_sha256": action_chain_sha256,
        "action_prefix_sha256_by_mechanics": dict(
            action_prefix_sha256_by_mechanics
        ),
    })


def reward_integrity(history: list[dict]) -> dict:
    def maximum_abs(field: str) -> float:
        return float(max([
            abs(float(row.get(field) or 0.0)) for row in history
        ] or [0.0]))

    return {
        "maximum_abs_ori6_reward_decomposition_error": maximum_abs(
            "ori6_reward_decomposition_error"
        ),
        "maximum_abs_ori6_step_reward_decomposition_error": maximum_abs(
            "ori6_maximum_abs_step_reward_decomposition_error"
        ),
        "ori6_nonfinite_reward_component_count": int(sum(
            int(float(row.get("ori6_nonfinite_reward_component_count") or 0))
            for row in history
        )),
        "maximum_abs_rejected_crossing_decomposition_error": maximum_abs(
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error"
        ),
        "maximum_abs_spectral_reward_decomposition_error": maximum_abs(
            "ori6_spectral_reward_decomposition_error"
        ),
        "maximum_abs_spectral_step_reward_decomposition_error": maximum_abs(
            "ori6_spectral_maximum_abs_step_reward_decomposition_error"
        ),
        "spectral_nonfinite_component_count": int(sum(
            int(float(row.get("ori6_spectral_nonfinite_component_count") or 0))
            for row in history
        )),
        "candidate_identity_all_rows": bool(history) and all(
            row.get("ori6_spectral_candidate") == spectral.CANDIDATE_NAME
            and row.get("ori6_spectral_formula_sha256")
            == spectral.SPECTRAL_FORMULA_SHA256
            and row.get("ori6_crossing_policy") == rejected.CROSSING_POLICY
            for row in history
        ),
        "committed_self_crossing_count": int(sum(
            int(float(row.get("self_crossing") or 0)) for row in history
        )),
    }


def validate_reward_integrity(audit: dict) -> bool:
    return bool(
        audit.get("candidate_identity_all_rows") is True
        and audit.get("committed_self_crossing_count") == 0
        and audit.get("ori6_nonfinite_reward_component_count") == 0
        and audit.get("spectral_nonfinite_component_count") == 0
        and all(float(audit.get(field, math.inf)) <= 1.0e-10 for field in (
            "maximum_abs_ori6_reward_decomposition_error",
            "maximum_abs_ori6_step_reward_decomposition_error",
            "maximum_abs_rejected_crossing_decomposition_error",
            "maximum_abs_spectral_reward_decomposition_error",
            "maximum_abs_spectral_step_reward_decomposition_error",
        ))
    )


def run_trial(args, *, task_name: str, seed: int, mechanics_budget: int,
              output_root: Path, root_fingerprint: str, resume: bool,
              checkpoint_every_episodes: int, print_every: int) -> dict:
    directory = trial_directory(output_root, seed)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as handle:
            completed = json.load(handle)
        if completed.get("root_fingerprint") != root_fingerprint:
            raise RuntimeError(f"Completed trial fingerprint mismatch: {directory}")
        if not resume:
            raise FileExistsError(f"Completed trial exists: {directory}")
        return completed

    trial_args = copy.deepcopy(args)
    trial_args.seed = int(seed)
    adapter = adapter_for_task(task_name)
    node, member, fix, _ = adapter.load_problem_data(trial_args.data_dir)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    environment = adapter.make_environment(trial_args, node, member, fix)
    uniform_rng = np.random.default_rng(int(seed))
    agent = ori3.UniformRandomAgent(trial_args.max_step_width, uniform_rng)
    rollout_agent = legacy._NoUpdateAgent(agent)

    episodes = mechanics = projector_calls = 0
    rollout_seconds = 0.0
    history: list[dict] = []
    action_chain_sha256 = "0" * 64
    audit_checkpoints = {
        min(INITIAL_AUDIT_MECHANICS, int(mechanics_budget))
    }
    action_prefix_sha256_by_mechanics: dict[str, str] = {}
    checkpoint_path = directory / "checkpoint.pt"
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("root_fingerprint") != root_fingerprint
            or int(checkpoint.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {directory}")
        episodes = int(checkpoint["episodes"])
        mechanics = int(checkpoint["mechanical_steps"])
        projector_calls = int(checkpoint["projector_calls"])
        rollout_seconds = float(checkpoint["rollout_seconds"])
        history = list(checkpoint["history"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        uniform_rng.bit_generator.state = checkpoint["uniform_rng_state"]
        action_chain_sha256 = checkpoint["action_chain_sha256"]
        action_prefix_sha256_by_mechanics = dict(
            checkpoint["action_prefix_sha256_by_mechanics"]
        )
        environment.training_episode_count = episodes

    trial_config = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "root_fingerprint": root_fingerprint,
        "method": METHOD,
        "seed": int(seed),
        "task": task_name,
        "mechanics_budget": int(mechanics_budget),
        "episode_cap": None,
        "max_steps_per_episode": int(trial_args.max_steps),
        "candidate": spectral.CANDIDATE_NAME,
        "crossing_policy": rejected.CROSSING_POLICY,
        "spectral_formula_sha256": spectral.SPECTRAL_FORMULA_SHA256,
        "budget_stop_reward": (
            "unchanged candidate reward from final ordinary valid transition"
        ),
    }
    atomic_json(directory / "run_config.json", trial_config)

    success_summary = success_nodes = success_actions = None
    with adapter.patched_three_dimensional_runtime(trial_args, member):
        while mechanics < mechanics_budget:
            remaining = mechanics_budget - mechanics
            budgeted = legacy.BudgetedEnvironment(environment, remaining)
            started = time.perf_counter()
            episode_summary, nodes, actions = adapter.run_episode(
                budgeted, rollout_agent, deterministic=False, training=True
            )
            elapsed = time.perf_counter() - started
            steps = int(episode_summary["steps"])
            if not 1 <= steps <= remaining:
                raise RuntimeError("Candidate violated exact mechanics budget")
            if len(actions) != steps:
                raise RuntimeError("Candidate action trace/step count mismatch")
            mechanics_before = mechanics
            action_chain_sha256 = chained_action_trace(
                action_chain_sha256, actions, mechanics_before,
                audit_checkpoints, action_prefix_sha256_by_mechanics,
            )
            episodes += 1
            mechanics += steps
            projector_calls += int(episode_summary["projection_attempts"])
            rollout_seconds += elapsed
            history.append(history_row(
                seed, episodes, episode_summary, mechanics,
                projector_calls, elapsed, rollout_seconds, actions,
            ))
            strict = bool(episode_summary.get(
                "strict_compact_success", episode_summary.get("success", 0)
            ))
            if strict:
                success_summary, success_nodes, success_actions = (
                    episode_summary, nodes, actions
                )
                break
            if (
                mechanics < mechanics_budget
                and episodes % int(checkpoint_every_episodes) == 0
            ):
                write_checkpoint(
                    checkpoint_path, fingerprint=root_fingerprint, seed=seed,
                    episodes=episodes, mechanics=mechanics,
                    projector_calls=projector_calls,
                    rollout_seconds=rollout_seconds, history=history,
                    uniform_rng=uniform_rng,
                    action_chain_sha256=action_chain_sha256,
                    action_prefix_sha256_by_mechanics=(
                        action_prefix_sha256_by_mechanics
                    ),
                )
                atomic_csv(directory / "episode_history.csv", history)
            if print_every and episodes % int(print_every) == 0:
                print(
                    f"uniform task={task_name} seed={seed} episode={episodes} "
                    f"mechanics={mechanics}/{mechanics_budget}", flush=True,
                )

    success = success_summary is not None
    if success:
        rejected.ori6.mechanics.save_rollout(
            directory / "first_hit_path", success_nodes, success_actions
        )
    atomic_csv(directory / "episode_history.csv", history)
    audit = reward_integrity(history)
    result = {
        **trial_config,
        "complete": True,
        "completed_utc": utc_now(),
        "success": int(success),
        "censored": int(not success),
        "censor_reason": None if success else "mechanics_budget",
        "episodes": int(episodes),
        "mechanical_steps": int(mechanics),
        "projector_calls": int(projector_calls),
        "first_success_episode": int(episodes) if success else None,
        "first_success_mechanical_steps": int(mechanics) if success else None,
        "first_success_projector_calls": (
            int(projector_calls) if success else None
        ),
        "rollout_wall_seconds": float(rollout_seconds),
        "compute_wall_seconds": float(rollout_seconds),
        "first_hit_path": "first_hit_path" if success else None,
        "terminal_flatness_ratio": (
            float(success_summary["flatness_ratio"]) if success else None
        ),
        "terminal_spread": (
            float(success_summary["spread"]) if success else None
        ),
        "reward_integrity": audit,
        "reward_integrity_pass": validate_reward_integrity(audit),
        "action_trace_chain": ACTION_TRACE_CHAIN,
        "action_chain_sha256": action_chain_sha256,
        "action_prefix_sha256_by_mechanics": (
            action_prefix_sha256_by_mechanics
        ),
    }
    if not result["reward_integrity_pass"]:
        raise RuntimeError(f"Uniform trial reward audit failed: {directory}")
    atomic_json(summary_path, result)
    return result


def load_root(output_root: Path) -> dict:
    path = Path(output_root) / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    validate_root_config(config)
    return config


def aggregate_results(output_root: Path, require_complete: bool = False) -> dict:
    output_root = Path(output_root)
    config = load_root(output_root)
    budget = int(config["mechanics_budget"])
    expected = {int(seed) for seed in config["seeds"]}
    rows = []
    seen = set()
    for path in sorted((output_root / "trials" / METHOD).glob(
        "seed*/summary.json"
    )):
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        seed = int(row.get("seed", -1))
        if seed in seen or seed not in expected:
            raise RuntimeError(f"Duplicate or unexpected random seed: {path}")
        if (
            row.get("root_fingerprint") != config["root_fingerprint"]
            or not row.get("complete")
            or row.get("method") != METHOD
            or int(row.get("mechanics_budget", -1)) != budget
            or row.get("reward_integrity_pass") is not True
        ):
            raise RuntimeError(f"Invalid random trial summary: {path}")
        mechanics = int(row.get("mechanical_steps", -1))
        if not row.get("success") and mechanics != budget:
            raise RuntimeError(f"Censored trial missed exact budget: {path}")
        if row.get("success") and not 1 <= mechanics <= budget:
            raise RuntimeError(f"Successful trial mechanics invalid: {path}")
        trace = row.get("action_chain_sha256")
        prefixes = row.get("action_prefix_sha256_by_mechanics")
        checkpoint = min(INITIAL_AUDIT_MECHANICS, budget)
        if (
            not isinstance(trace, str) or len(trace) != 64
            or not isinstance(prefixes, dict)
            or (
                mechanics >= checkpoint
                and prefixes.get(str(checkpoint)) is None
            )
        ):
            raise RuntimeError(f"Missing chained action-trace audit: {path}")
        seen.add(seed)
        rows.append(row)
    missing = sorted(expected - seen)
    if require_complete and missing:
        raise RuntimeError(f"Missing random seeds: {missing}")
    assigned = [
        int(row["first_success_mechanical_steps"])
        if row["success"] else budget for row in rows
    ]
    successes = [row for row in rows if row["success"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "candidate": spectral.CANDIDATE_NAME,
        "task": config["task"],
        "method": METHOD,
        "root_fingerprint": config["root_fingerprint"],
        "mechanics_budget": budget,
        "seeds": config["seeds"],
        "expected_trials": len(expected),
        "completed_trials": len(rows),
        "complete": not missing,
        "missing_seeds": missing,
        "successes": len(successes),
        "censored": len(rows) - len(successes),
        "success_at_budget": (
            float(len(successes) / len(rows)) if rows else None
        ),
        "first_success_mechanical_steps": [
            int(row["first_success_mechanical_steps"])
            for row in successes
        ],
        "restricted_mean_first_hit_mechanical_steps": (
            float(np.mean(assigned)) if assigned else None
        ),
        "rmst_censor_assignment": (
            "all censored trials assigned the common mechanics budget"
        ),
        "trial_summaries": rows,
        "all_reward_integrity_pass": bool(rows) and all(
            row["reward_integrity_pass"] is True for row in rows
        ),
        "generated_utc": utc_now(),
    }
    atomic_json(output_root / "aggregate.json", result)
    return result


def prepare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--mechanics-budget", type=int, required=True)
    parser.add_argument("--trial-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-every-episodes", type=int, default=128)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if options.mechanics_budget < 1:
        raise ValueError("--mechanics-budget must be positive")
    if len(options.trial_seeds) != len(set(options.trial_seeds)):
        raise ValueError("--trial-seeds must be unique")
    if options.checkpoint_every_episodes < 1:
        raise ValueError("--checkpoint-every-episodes must be positive")
    output_root = options.output_root
    if not output_root.is_absolute():
        output_root = SCRIPT_DIRECTORY / output_root
    options.output_root = output_root.resolve()
    return options


def main(argv=None):
    torch.set_num_threads(1)
    options = prepare(argv)
    if options.aggregate_only:
        result = aggregate_results(options.output_root, require_complete=True)
        print(json.dumps(result, indent=2))
        return result
    config = make_root_config(
        options.task, options.mechanics_budget, options.trial_seeds
    )
    if options.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "output_root": str(options.output_root),
            "root_fingerprint": config["root_fingerprint"],
            "task": options.task,
            "mechanics_budget": options.mechanics_budget,
            "seeds": options.trial_seeds,
        }, indent=2))
        return config
    root_path = options.output_root / "run_config.json"
    if root_path.is_file():
        existing = load_root(options.output_root)
        if existing["root_fingerprint"] != config["root_fingerprint"]:
            raise RuntimeError(
                f"Output root contains another experiment: {options.output_root}"
            )
        config = existing
    else:
        atomic_json(root_path, config)
    args = candidate_args(options.task, options.trial_seeds[0])
    for seed in options.trial_seeds:
        result = run_trial(
            args, task_name=options.task, seed=seed,
            mechanics_budget=options.mechanics_budget,
            output_root=options.output_root,
            root_fingerprint=config["root_fingerprint"],
            resume=options.resume,
            checkpoint_every_episodes=options.checkpoint_every_episodes,
            print_every=options.print_every,
        )
        print(
            f"uniform task={options.task} seed={seed} "
            f"success={bool(result['success'])} "
            f"mechanics={result['mechanical_steps']}", flush=True,
        )
    aggregate = aggregate_results(options.output_root, require_complete=True)
    print(json.dumps(aggregate, indent=2))
    print(f"Results: {options.output_root}")
    return aggregate


if __name__ == "__main__":
    main()
