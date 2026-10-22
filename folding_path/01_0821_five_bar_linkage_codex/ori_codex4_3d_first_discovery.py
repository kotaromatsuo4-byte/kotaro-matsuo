"""First-discovery benchmark for the three-dimensional ori_codex4 task.

This is intentionally separate from the training/evaluation driver.  A trial
contains only on-policy discovery episodes and stops at the first *strict*
compact-flat success, or exactly at a common cumulative mechanics-step
budget.  There are no deterministic, periodic, held-out, or final rollouts.

The three compared samplers are:

``learning``
    The unmodified ori_codex4 Monte-Carlo Actor--Critic, updated after every
    complete batch and never on a final partial batch.
``frozen``
    The same freshly initialised neural policy, with the same seed and action
    distribution, but with transition storage and updates disabled.
``uniform``
    A uniform signed width and an isotropic direction in the current modal
    coefficient space (ori_codex3's existing random baseline).

Examples
--------
Run paired, shardable trials::

    python ori_codex4_3d_first_discovery.py --model waterbomb \
        --target-spread 1 --mechanics-budget 200000 \
        --trial-seed-start 100 --trial-count 30 \
        --output-root first_discovery/waterbomb_spread1

Resume one of four deterministic task shards by changing only shard-index::

    python ori_codex4_3d_first_discovery.py ... \
        --shard-count 4 --shard-index 0

Every remaining option is forwarded to :func:`ori_codex4_3d.parse_args`.
Relative output paths are resolved beside this file.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import torch

import collision_3d
import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex4 as ori4
import ori_codex4_3d as adapter


SCHEMA_VERSION = 1
HARNESS_VERSION = "1.0.0"
METHODS = ("learning", "frozen", "uniform")
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_digest(value):
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(_jsonable(value), file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class BudgetedEnvironment:
    """Logging-only exact step-budget stop around an ori4 environment.

    The wrapped ``step`` is always evaluated first.  If that ordinary ori4
    transition is nonterminal and consumes the final budgeted mechanics step,
    only ``done`` and the diagnostic termination label are changed.  No
    timeout penalty, terminal bonus, or other synthetic reward is introduced.
    """

    def __init__(self, environment, remaining_steps):
        self.environment = environment
        self.remaining_steps = int(remaining_steps)
        if self.remaining_steps < 1:
            raise ValueError("remaining_steps must be positive.")
        self.budget_exhausted = False

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def step(self, *args, **kwargs):
        if self.remaining_steps <= 0:
            raise RuntimeError("A mechanics step was requested after the budget.")
        next_state, reward, done, info = self.environment.step(*args, **kwargs)
        self.remaining_steps -= 1
        info = dict(info)
        info["benchmark_budget_exhausted"] = 0
        if self.remaining_steps == 0:
            self.budget_exhausted = True
            info["benchmark_budget_exhausted"] = 1
            if not done:
                done = True
                info["termination_reason"] = "benchmark_budget"
                # This is administrative censoring, not an environment timeout.
                info["timeout"] = False
        return next_state, reward, done, info


class _NoUpdateAgent:
    """Expose a sampler while discarding Actor--Critic transitions."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def add_transition(self, transition):
        del transition

    def finish_episode(self):
        return None


def _model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _episode_action_digest(actions):
    compact = [{
        "physical_direction": action["physical_direction"],
        "mode_coefficients": action["mode_coefficients"],
        "requested_width": action["requested_width"],
    } for action in actions]
    return _canonical_digest(compact)


def _make_neural_agent(args, coordinate_size):
    model = mechanics.VariableModeActorCritic(
        coordinate_size=coordinate_size,
        hidden_size=args.hidden_size,
        variance_bias=args.variance_bias,
        width_mean_limit=args.width_mean_limit,
    )
    agent = mechanics.ActorCriticAgent(
        model=model,
        max_step_width=args.max_step_width,
        gamma=args.gamma,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        actor_gradient_clip=args.actor_gradient_clip,
        critic_gradient_clip=args.critic_gradient_clip,
    )
    return model, agent


def _normalise_and_validate(args):
    """Apply the same reward-side normalisation as ori_codex3.train."""

    ori4._validate_args(args)
    if int(args.flatness_subspace_dimension) != 2:
        raise ValueError("First-discovery 3-D runs require planar flatness (k=2).")
    if args.update_method != "actor_critic":
        raise ValueError("This benchmark compares ori4 Actor--Critic, not PPO.")
    # ori_codex3.train fixes this legacy parser option before constructing the
    # ori4 environment; reproduce that effective configuration here.
    args.crossing_penalty = 0.0
    if args.quality_combination == "path_integrated_target_deviation":
        args.terminate_wrong_flat = True
        args.wrong_flat_penalty = 0.0
    if len(args.actor_lr_milestones) != len(args.actor_lr_factors):
        raise ValueError("actor_lr_milestones and factors must have equal length.")
    if list(args.actor_lr_milestones) != sorted(args.actor_lr_milestones):
        raise ValueError("actor_lr_milestones must be sorted.")
    if any(float(factor) <= 0.0 for factor in args.actor_lr_factors):
        raise ValueError("actor_lr_factors must be positive.")
    if any(
        int(milestone) % int(args.batch_size) != 0
        for milestone in args.training_flatness_milestones
    ):
        raise ValueError("Curriculum milestones must fall on batch boundaries.")
    if (
        args.curriculum_waypoint_bonus > 0.0
        and args.curriculum_waypoint_start_episode > 0
        and args.curriculum_waypoint_start_episode % args.batch_size != 0
    ):
        raise ValueError("The curriculum waypoint must fall on a batch boundary.")
    if args.reset_critic_at_waypoint_start:
        raise ValueError("Critic reset is unavailable for Actor--Critic.")
    return args


def _checkpoint_payload(
    *, root_fingerprint, method, seed, model, agent, uniform_rng,
    episodes, mechanical_steps, projector_calls, rollout_seconds,
    update_seconds, history, updates, initial_model_sha256,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "root_fingerprint": root_fingerprint,
        "method": method,
        "seed": int(seed),
        "episodes": int(episodes),
        "mechanical_steps": int(mechanical_steps),
        "projector_calls": int(projector_calls),
        "rollout_seconds": float(rollout_seconds),
        "update_seconds": float(update_seconds),
        "history": history,
        "updates": updates,
        "initial_model_sha256": initial_model_sha256,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "uniform_rng_state": (
            copy.deepcopy(uniform_rng.bit_generator.state)
            if uniform_rng is not None else None
        ),
        "model_state_dict": model.state_dict() if model is not None else None,
        "actor_optimizer_state_dict": (
            agent.actor_optimizer.state_dict()
            if method == "learning" else None
        ),
        "critic_optimizer_state_dict": (
            agent.critic_optimizer.state_dict()
            if method == "learning" else None
        ),
    }
    return payload


_HISTORY_FIELDS = (
    "method", "seed", "episode", "success", "strict_compact_success",
    "termination_reason", "mechanical_steps", "projector_calls",
    "cumulative_mechanical_steps", "cumulative_projector_calls",
    "rollout_seconds", "cumulative_rollout_seconds", "total_reward",
    "flatness_ratio", "spread", "spread_error", "self_crossing",
    "correction_failed", "timeout", "near_miss_q", "near_miss_bonus",
    "mean_commanded_rigid_energy_fraction", "action_trace_sha256",
)


def _write_trial_history(trial_directory, history, updates):
    _atomic_csv(
        trial_directory / "episode_history.csv", history, _HISTORY_FIELDS
    )
    update_fields = (
        "episode", "loss", "actor_loss", "critic_loss", "entropy",
        "actor_gradient_norm", "critic_gradient_norm",
        "actor_learning_rate", "critic_learning_rate", "update_seconds",
    )
    _atomic_csv(trial_directory / "update_history.csv", updates, update_fields)


def _history_row(method, seed, episode, summary, cumulative_steps,
                 cumulative_calls, episode_seconds, rollout_seconds, actions):
    return {
        "method": method,
        "seed": int(seed),
        "episode": int(episode),
        "success": int(summary.get("success", 0)),
        "strict_compact_success": int(
            summary.get("strict_compact_success", summary.get("success", 0))
        ),
        "termination_reason": summary.get("termination_reason", "unknown"),
        "mechanical_steps": int(summary["steps"]),
        "projector_calls": int(summary["projection_attempts"]),
        "cumulative_mechanical_steps": int(cumulative_steps),
        "cumulative_projector_calls": int(cumulative_calls),
        "rollout_seconds": float(episode_seconds),
        "cumulative_rollout_seconds": float(rollout_seconds),
        "total_reward": float(summary["total_reward"]),
        "flatness_ratio": float(summary["flatness_ratio"]),
        "spread": float(summary.get("spread", np.nan)),
        "spread_error": float(summary.get("spread_error", np.nan)),
        "self_crossing": int(summary.get("self_crossing", 0)),
        "correction_failed": int(summary.get("correction_failed", 0)),
        "timeout": int(summary.get("timeout", 0)),
        "near_miss_q": float(summary.get("near_miss_q", 0.0)),
        "near_miss_bonus": float(summary.get("near_miss_bonus", 0.0)),
        "mean_commanded_rigid_energy_fraction": summary.get(
            "mean_commanded_rigid_energy_fraction"
        ),
        "action_trace_sha256": _episode_action_digest(actions),
    }


def run_trial(
    args, method, seed, mechanics_budget, episode_cap, output_root,
    root_fingerprint, resume=True, checkpoint_every_batches=1,
    progress_every=0,
):
    """Run or resume one method/seed pair and return its terminal record."""

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    mechanics_budget = int(mechanics_budget)
    if mechanics_budget < 1:
        raise ValueError("mechanics_budget must be positive.")
    if episode_cap is not None and int(episode_cap) < 1:
        raise ValueError("episode_cap must be positive when supplied.")
    if checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive.")

    trial_directory = (
        Path(output_root) / "trials" / method / f"seed{int(seed)}"
    )
    trial_directory.mkdir(parents=True, exist_ok=True)
    summary_path = trial_directory / "summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as file:
            completed = json.load(file)
        if completed.get("root_fingerprint") != root_fingerprint:
            raise RuntimeError(f"Configuration mismatch in {trial_directory}")
        if not resume:
            raise FileExistsError(f"Completed trial exists: {trial_directory}")
        return completed

    trial_args = copy.deepcopy(args)
    trial_args.seed = int(seed)
    node, member, fix, _ = adapter.load_problem_data(trial_args.data_dir)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    environment = adapter.make_environment(trial_args, node, member, fix)
    coordinate_size = int(environment.coordinate_size)

    model = agent = uniform_rng = None
    if method in ("learning", "frozen"):
        model, agent = _make_neural_agent(trial_args, coordinate_size)
        initial_model_sha256 = _model_digest(model)
        rollout_agent = agent if method == "learning" else _NoUpdateAgent(agent)
    else:
        uniform_rng = np.random.default_rng(int(seed))
        agent = ori3.UniformRandomAgent(trial_args.max_step_width, uniform_rng)
        rollout_agent = _NoUpdateAgent(agent)
        initial_model_sha256 = None

    episodes = mechanical_steps = projector_calls = 0
    rollout_seconds = update_seconds = 0.0
    history, updates = [], []
    checkpoint_path = trial_directory / "checkpoint.pt"
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = (root_fingerprint, method, int(seed))
        actual = (
            checkpoint.get("root_fingerprint"), checkpoint.get("method"),
            int(checkpoint.get("seed", -1)),
        )
        if actual != expected:
            raise RuntimeError(f"Checkpoint identity mismatch in {trial_directory}")
        episodes = int(checkpoint["episodes"])
        if episodes % int(trial_args.batch_size):
            raise RuntimeError("A resumable checkpoint is not on a batch boundary.")
        mechanical_steps = int(checkpoint["mechanical_steps"])
        projector_calls = int(checkpoint["projector_calls"])
        rollout_seconds = float(checkpoint["rollout_seconds"])
        update_seconds = float(checkpoint["update_seconds"])
        history = list(checkpoint["history"])
        updates = list(checkpoint["updates"])
        initial_model_sha256 = checkpoint.get("initial_model_sha256")
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        if method == "learning":
            agent.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            agent.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
        if uniform_rng is not None:
            uniform_rng.bit_generator.state = checkpoint["uniform_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        environment.training_episode_count = episodes

    trial_config = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "root_fingerprint": root_fingerprint,
        "method": method,
        "seed": int(seed),
        "mechanics_budget": mechanics_budget,
        "episode_cap": episode_cap,
        "max_steps_per_episode": int(trial_args.max_steps),
        "batch_size": int(trial_args.batch_size),
        "strict_success_field": "strict_compact_success",
        "budget_stop_reward": "unchanged ori4 nonterminal transition reward",
        "initial_model_sha256": initial_model_sha256,
    }
    _atomic_json(trial_directory / "run_config.json", trial_config)

    success_summary = success_nodes = success_actions = None
    censor_reason = None
    batches_since_checkpoint = 0
    with adapter.patched_three_dimensional_runtime(trial_args, member):
        while mechanical_steps < mechanics_budget:
            if episode_cap is not None and episodes >= int(episode_cap):
                censor_reason = "episode_cap"
                break
            remaining = mechanics_budget - mechanical_steps
            budgeted = BudgetedEnvironment(environment, remaining)
            episode_start = time.perf_counter()
            if model is not None:
                model.train()
            episode_summary, nodes, actions = adapter.run_episode(
                budgeted, rollout_agent, deterministic=False, training=True
            )
            episode_seconds = time.perf_counter() - episode_start
            steps = int(episode_summary["steps"])
            if not 0 <= steps <= remaining:
                raise RuntimeError("The environment violated the mechanics budget.")
            episodes += 1
            mechanical_steps += steps
            projector_calls += int(episode_summary["projection_attempts"])
            rollout_seconds += episode_seconds
            history.append(_history_row(
                method, seed, episodes, episode_summary, mechanical_steps,
                projector_calls, episode_seconds, rollout_seconds, actions,
            ))
            strict_success = bool(episode_summary.get(
                "strict_compact_success", episode_summary.get("success", 0)
            ))
            if strict_success:
                success_summary, success_nodes, success_actions = (
                    episode_summary, nodes, actions
                )
                break

            trial_ended = bool(
                mechanical_steps >= mechanics_budget
                or (episode_cap is not None and episodes >= int(episode_cap))
            )
            administratively_truncated = (
                episode_summary.get("termination_reason") == "benchmark_budget"
            )
            on_batch_boundary = episodes % int(trial_args.batch_size) == 0
            if (
                method == "learning"
                and on_batch_boundary
                and not administratively_truncated
                and not trial_ended
            ):
                learning_rate = mechanics.actor_learning_rate_at_episode(
                    trial_args.actor_learning_rate, episodes,
                    trial_args.actor_lr_milestones,
                    trial_args.actor_lr_factors,
                )
                agent.set_actor_learning_rate(learning_rate)
                update_start = time.perf_counter()
                update = agent.update()
                elapsed_update = time.perf_counter() - update_start
                update_seconds += elapsed_update
                if update is None:
                    raise RuntimeError("A complete Actor--Critic batch was empty.")
                update.update({
                    "episode": episodes,
                    "update_seconds": elapsed_update,
                })
                updates.append(update)
            if on_batch_boundary and not trial_ended:
                batches_since_checkpoint += 1
                if batches_since_checkpoint >= int(checkpoint_every_batches):
                    payload = _checkpoint_payload(
                        root_fingerprint=root_fingerprint, method=method,
                        seed=seed, model=model, agent=agent,
                        uniform_rng=uniform_rng, episodes=episodes,
                        mechanical_steps=mechanical_steps,
                        projector_calls=projector_calls,
                        rollout_seconds=rollout_seconds,
                        update_seconds=update_seconds, history=history,
                        updates=updates,
                        initial_model_sha256=initial_model_sha256,
                    )
                    _atomic_torch_save(checkpoint_path, payload)
                    _write_trial_history(trial_directory, history, updates)
                    batches_since_checkpoint = 0
            if progress_every and episodes % int(progress_every) == 0:
                print(
                    f"{method} seed={seed} episode={episodes} "
                    f"steps={mechanical_steps}/{mechanics_budget}",
                    flush=True,
                )

    success = success_summary is not None
    if success:
        mechanics.save_rollout(
            trial_directory / "first_hit_path", success_nodes, success_actions
        )
    elif censor_reason is None:
        censor_reason = (
            "mechanics_budget"
            if mechanical_steps >= mechanics_budget else "unknown"
        )

    _write_trial_history(trial_directory, history, updates)
    prefix_rows = history[:min(int(trial_args.batch_size), len(history))]
    first_batch_trace = _canonical_digest([
        row["action_trace_sha256"] for row in prefix_rows
    ]) if prefix_rows else None
    result = {
        **trial_config,
        "complete": True,
        "completed_utc": _utc_now(),
        "success": int(success),
        "censored": int(not success),
        "censor_reason": None if success else censor_reason,
        "episodes": int(episodes),
        "mechanical_steps": int(mechanical_steps),
        "projector_calls": int(projector_calls),
        "first_success_episode": int(episodes) if success else None,
        "first_success_mechanical_steps": (
            int(mechanical_steps) if success else None
        ),
        "first_success_projector_calls": (
            int(projector_calls) if success else None
        ),
        "rollout_wall_seconds": float(rollout_seconds),
        "update_wall_seconds": float(update_seconds),
        "compute_wall_seconds": float(rollout_seconds + update_seconds),
        "policy_updates": int(len(updates)),
        "full_batches_completed_before_stop": int(
            episodes // trial_args.batch_size
        ),
        "learning_episodes_in_unupdated_final_batch": (
            int(episodes - len(updates) * trial_args.batch_size)
            if method == "learning" else None
        ),
        "first_batch_action_trace_sha256": first_batch_trace,
        "terminal_flatness_ratio": (
            float(success_summary["flatness_ratio"]) if success else None
        ),
        "terminal_spread": (
            float(success_summary["spread"]) if success else None
        ),
        "first_hit_path": "first_hit_path" if success else None,
    }
    _atomic_json(summary_path, result)
    return result


def aggregate_results(
    output_root, mechanics_budget=None, episode_cap=None,
    require_complete=False,
):
    """Aggregate all completed trials currently visible below output_root."""

    output_root = Path(output_root)
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    with open(config_path, encoding="utf-8") as file:
        root_config = json.load(file)
    expected_fingerprint = root_config["root_fingerprint"]
    expected_budget = int(root_config["mechanics_budget"])
    expected_horizon = int(
        root_config["effective_ori4_args"]["max_steps"]
    )
    expected_pairs = {
        (str(method), int(seed))
        for seed in root_config["seeds"]
        for method in root_config["methods"]
    }
    summaries = []
    seen_pairs = set()
    for path in sorted((output_root / "trials").glob("*/seed*/summary.json")):
        with open(path, encoding="utf-8") as file:
            row = json.load(file)
        if row.get("complete"):
            pair = (str(row.get("method")), int(row.get("seed", -1)))
            if pair in seen_pairs:
                raise RuntimeError(f"Duplicate completed trial: {pair}")
            if pair not in expected_pairs:
                raise RuntimeError(f"Unexpected completed trial: {pair}")
            if row.get("root_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    f"Fingerprint mismatch in {path}: "
                    f"{row.get('root_fingerprint')} != {expected_fingerprint}"
                )
            if int(row.get("mechanics_budget", -1)) != expected_budget:
                raise RuntimeError(f"Mechanics-budget mismatch in {path}")
            if int(row.get("max_steps_per_episode", -1)) != expected_horizon:
                raise RuntimeError(f"Episode-horizon mismatch in {path}")
            seen_pairs.add(pair)
            row["summary_path"] = str(path.relative_to(output_root))
            summaries.append(row)
    missing_pairs = sorted(expected_pairs - seen_pairs)
    if require_complete and missing_pairs:
        raise RuntimeError(
            "Incomplete benchmark; missing method/seed pairs: "
            + ", ".join(f"{method}:{seed}" for method, seed in missing_pairs)
        )
    if mechanics_budget is None:
        budgets = {int(row["mechanics_budget"]) for row in summaries}
        if len(budgets) > 1:
            raise ValueError("Completed trials do not share one mechanics budget.")
        mechanics_budget = next(iter(budgets), expected_budget)
    mechanics_budget = int(mechanics_budget) if mechanics_budget is not None else None
    if mechanics_budget != expected_budget:
        raise ValueError(
            f"Requested budget {mechanics_budget} does not match "
            f"run budget {expected_budget}."
        )
    horizons = {int(row["max_steps_per_episode"]) for row in summaries}
    if len(horizons) > 1:
        raise ValueError("Completed trials do not share one episode horizon.")
    max_steps_per_episode = next(iter(horizons), None)
    common_episode_horizon = (
        mechanics_budget // max_steps_per_episode
        if mechanics_budget is not None and max_steps_per_episode else None
    )
    if episode_cap is not None:
        common_episode_horizon = (
            min(common_episode_horizon, int(episode_cap))
            if common_episode_horizon is not None else int(episode_cap)
        )

    by_method = {}
    for method in METHODS:
        rows = [row for row in summaries if row["method"] == method]
        if not rows:
            continue
        successes = [row for row in rows if row["success"]]
        assigned_costs = [
            int(row["first_success_mechanical_steps"])
            if row["success"] else mechanics_budget
            for row in rows
        ]
        episode_events = [
            int(row["first_success_episode"])
            if row["success"] else None
            for row in rows
        ]
        metrics = {
            "trials": len(rows),
            "successes": len(successes),
            "censored": len(rows) - len(successes),
            "success_at_budget": float(len(successes) / len(rows)),
            "mechanics_budget": mechanics_budget,
            # With common administrative censoring this is exactly the
            # empirical Kaplan--Meier RMST.  An optional earlier episode cap is
            # conservatively assigned the common mechanics budget.
            "restricted_mean_first_hit_mechanical_steps": float(
                np.mean(assigned_costs)
            ),
            "rmst_censor_assignment": (
                "all censored trials assigned the common mechanics budget"
            ),
            "rmst_is_exact_common_administrative_censor": bool(
                episode_cap is None
            ),
            "success_at_budget_is_lower_bound_due_episode_cap": bool(
                episode_cap is not None
            ),
            "mean_first_hit_mechanical_steps_among_successes": (
                float(np.mean([
                    row["first_success_mechanical_steps"] for row in successes
                ])) if successes else None
            ),
            "median_first_hit_mechanical_steps_among_successes": (
                float(np.median([
                    row["first_success_mechanical_steps"] for row in successes
                ])) if successes else None
            ),
            "mean_first_hit_episode_among_successes": (
                float(np.mean([
                    row["first_success_episode"] for row in successes
                ])) if successes else None
            ),
            "mean_first_hit_projector_calls_among_successes": (
                float(np.mean([
                    row["first_success_projector_calls"] for row in successes
                ])) if successes else None
            ),
            "mean_compute_wall_seconds": float(np.mean([
                row["compute_wall_seconds"] for row in rows
            ])),
            "censor_reasons": {
                reason: sum(row.get("censor_reason") == reason for row in rows)
                for reason in sorted({
                    row.get("censor_reason") for row in rows
                    if row.get("censor_reason") is not None
                })
            },
        }
        if common_episode_horizon is not None and common_episode_horizon >= 1:
            episode_costs = [
                min(event, common_episode_horizon)
                if event is not None else common_episode_horizon
                for event in episode_events
            ]
            metrics["restricted_mean_first_hit_episode"] = float(
                np.mean(episode_costs)
            )
            metrics["success_at_episode_horizon"] = float(sum(
                event is not None and event <= common_episode_horizon
                for event in episode_events
            ) / len(rows))
            metrics["common_episode_horizon"] = int(common_episode_horizon)
            metrics["episode_horizon_rule"] = (
                "min(optional episode cap, floor(mechanics budget / "
                "max steps per episode))"
            )
        by_method[method] = metrics

    paired = {}
    indexed = {
        (row["method"], int(row["seed"])): row for row in summaries
    }
    for baseline in ("frozen", "uniform"):
        seeds = sorted({
            seed for method, seed in indexed
            if method == "learning" and (baseline, seed) in indexed
        })
        if not seeds or mechanics_budget is None:
            continue
        differences = []
        learning_earlier = ties = 0
        actor_init_matches = []
        first_batch_trace_matches = []
        for seed in seeds:
            learning = indexed[("learning", seed)]
            other = indexed[(baseline, seed)]
            learning_cost = (
                learning["first_success_mechanical_steps"]
                if learning["success"] else mechanics_budget
            )
            other_cost = (
                other["first_success_mechanical_steps"]
                if other["success"] else mechanics_budget
            )
            differences.append(float(learning_cost - other_cost))
            learning_earlier += learning_cost < other_cost
            ties += learning_cost == other_cost
            if baseline == "frozen":
                actor_init_matches.append(
                    learning.get("initial_model_sha256")
                    == other.get("initial_model_sha256")
                )
                first_batch_trace_matches.append(
                    learning.get("first_batch_action_trace_sha256")
                    == other.get("first_batch_action_trace_sha256")
                )
        paired[baseline] = {
            "paired_seeds": len(seeds),
            "mean_learning_minus_baseline_assigned_cost": float(
                np.mean(differences)
            ),
            "learning_earlier_fraction": float(learning_earlier / len(seeds)),
            "assigned_cost_tie_fraction": float(ties / len(seeds)),
            "identical_initial_actor_all_pairs": (
                bool(all(actor_init_matches)) if actor_init_matches else None
            ),
            "identical_preupdate_action_trace_all_pairs": (
                bool(all(first_batch_trace_matches))
                if first_batch_trace_matches else None
            ),
        }

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "generated_utc": _utc_now(),
        "completed_trials": len(summaries),
        "expected_trials": len(expected_pairs),
        "complete": not missing_pairs,
        "missing_method_seed_pairs": [
            {"method": method, "seed": seed}
            for method, seed in missing_pairs
        ],
        "mechanics_budget": mechanics_budget,
        "max_steps_per_episode": max_steps_per_episode,
        "common_episode_horizon": common_episode_horizon,
        "methods": by_method,
        "paired_learning_comparisons": paired,
    }
    _atomic_json(output_root / "aggregate.json", aggregate)
    fields = (
        "method", "seed", "success", "censored", "censor_reason",
        "episodes", "mechanical_steps", "projector_calls",
        "first_success_episode", "first_success_mechanical_steps",
        "first_success_projector_calls", "rollout_wall_seconds",
        "update_wall_seconds", "compute_wall_seconds", "policy_updates",
        "initial_model_sha256", "first_batch_action_trace_sha256",
        "summary_path",
    )
    _atomic_csv(output_root / "trials.csv", summaries, fields)
    return aggregate


def _parse_harness_args(argv):
    parser = argparse.ArgumentParser(
        add_help=False,
        description="No-evaluation first-discovery benchmark for ori_codex4_3d.",
    )
    parser.add_argument("--mechanics-budget", type=int, default=None)
    parser.add_argument("--episode-cap", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--trial-seed-start", type=int, default=None)
    parser.add_argument("--trial-count", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--benchmark-print-every", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("-h", "--help", action="store_true", dest="harness_help")
    options, remaining = parser.parse_known_args(argv)
    if options.harness_help:
        parser.print_help()
        print("\nAll unrecognised options are forwarded to ori_codex4_3d.py.")
        raise SystemExit(0)
    return options, remaining


def _source_provenance():
    names = (
        "ori_codex2.py", "ori_codex3.py", "ori_codex4.py",
        "ori_codex4_3d.py", "collision_3d.py",
        Path(__file__).name,
    )
    return {
        name: _file_sha256(_SCRIPT_DIRECTORY / name)
        for name in names if (_SCRIPT_DIRECTORY / name).is_file()
    }


def _prepare_run(argv=None):
    harness, core_argv = _parse_harness_args(
        sys.argv[1:] if argv is None else list(argv)
    )
    if harness.aggregate_only:
        if harness.output_root is None:
            raise ValueError("--aggregate-only requires --output-root.")
        output_root = Path(harness.output_root)
        if not output_root.is_absolute():
            output_root = _SCRIPT_DIRECTORY / output_root
        return harness, None, output_root, [], None
    if harness.mechanics_budget is None or harness.mechanics_budget < 1:
        raise ValueError("A positive --mechanics-budget is required.")
    if harness.trial_count < 1:
        raise ValueError("trial_count must be positive.")
    if harness.shard_count < 1 or not 0 <= harness.shard_index < harness.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count.")
    if harness.checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive.")

    # The adapter's output directory is deliberately unused here; pin it so
    # its unique timestamp does not contaminate a reproducibility fingerprint.
    core_args = adapter.parse_args(["--output-dir", "__unused_by_discovery__", *core_argv])
    _normalise_and_validate(core_args)
    if harness.trial_seeds is not None:
        seeds = list(dict.fromkeys(int(seed) for seed in harness.trial_seeds))
    else:
        first_seed = (
            int(core_args.seed)
            if harness.trial_seed_start is None
            else int(harness.trial_seed_start)
        )
        seeds = list(range(first_seed, first_seed + int(harness.trial_count)))

    node, member, fix, files = adapter.load_problem_data(core_args.data_dir)
    model_hashes = {name: _file_sha256(path) for name, path in files.items()}
    effective_args = {
        key: _jsonable(value) for key, value in vars(core_args).items()
        if key != "output_dir"
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "effective_ori4_args": effective_args,
        "methods": list(harness.methods),
        "seeds": seeds,
        "mechanics_budget": int(harness.mechanics_budget),
        "episode_cap": harness.episode_cap,
        "model_file_sha256": model_hashes,
        "source_sha256": _source_provenance(),
    }
    root_fingerprint = _canonical_digest(identity)
    if harness.output_root is None:
        slug = (
            f"{core_args.selected_model}_spread{float(core_args.target_spread):.8g}_"
            f"budget{int(harness.mechanics_budget)}_{root_fingerprint[:12]}"
        ).replace(".", "p")
        output_root = _SCRIPT_DIRECTORY / "first_discovery_results" / slug
    else:
        output_root = Path(harness.output_root)
        if not output_root.is_absolute():
            output_root = _SCRIPT_DIRECTORY / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = output_root / "run_config.json"
    run_config = {
        **identity,
        "root_fingerprint": root_fingerprint,
        "created_utc": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "collision_backend": collision_3d.COLLISION_BACKEND_NAME,
        "collision_backend_version": collision_3d.COLLISION_BACKEND_VERSION,
        "comparison_rules": {
            "strict_success": "strict_compact_success == 1",
            "stop": "first strict success or exact cumulative mechanics budget",
            "evaluation_rollouts": 0,
            "learning_update": "one update after each full batch before discovery",
            "final_partial_batch_update": False,
            "administratively_truncated_episode_update": False,
            "frozen_actor": "same seed/model/action distribution, no updates",
            "common_random_numbers_window": (
                "learning and frozen share initial weights and sampled-action "
                "stream through the first learning batch only; both hashes "
                "are recorded, with no post-update equality claim"
            ),
            "uniform_random": "isotropic current-mode direction and uniform signed width",
            "episode_censor": (
                "E=floor(mechanics_budget/max_steps); if an optional episode "
                "cap is smaller use that. A mechanics hit after E is an event "
                "for step analysis but censored for episode RMST."
            ),
        },
        "node_count": int(len(node)),
        "member_count": int(len(member)),
        "fixed_scalar_dof_count": int(len(fix)),
    }
    if run_config_path.is_file():
        with open(run_config_path, encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("root_fingerprint") != root_fingerprint:
            raise RuntimeError(
                f"Output root contains a different experiment: {output_root}"
            )
    else:
        _atomic_json(run_config_path, run_config)

    tasks = [
        (method, seed) for seed in seeds for method in harness.methods
    ]
    tasks = [
        task for index, task in enumerate(tasks)
        if index % harness.shard_count == harness.shard_index
    ]
    return harness, core_args, output_root, tasks, root_fingerprint


def main(argv=None):
    torch.set_num_threads(1)
    harness, args, output_root, tasks, fingerprint = _prepare_run(argv)
    if harness.aggregate_only:
        with open(output_root / "run_config.json", encoding="utf-8") as file:
            config = json.load(file)
        aggregate = aggregate_results(
            output_root,
            mechanics_budget=config["mechanics_budget"],
            episode_cap=config.get("episode_cap"),
            require_complete=True,
        )
        print(json.dumps(aggregate, indent=2))
        return aggregate

    for method, seed in tasks:
        result = run_trial(
            args=args,
            method=method,
            seed=seed,
            mechanics_budget=harness.mechanics_budget,
            episode_cap=harness.episode_cap,
            output_root=output_root,
            root_fingerprint=fingerprint,
            resume=harness.resume,
            checkpoint_every_batches=harness.checkpoint_every_batches,
            progress_every=harness.benchmark_print_every,
        )
        print(
            f"{method} seed={seed}: success={bool(result['success'])} "
            f"episodes={result['episodes']} steps={result['mechanical_steps']} "
            f"projector_calls={result['projector_calls']}",
            flush=True,
        )
    aggregate = aggregate_results(
        output_root, harness.mechanics_budget, harness.episode_cap
    )
    print(json.dumps(aggregate, indent=2))
    print(f"Results: {output_root}")
    return aggregate


if __name__ == "__main__":
    main()
