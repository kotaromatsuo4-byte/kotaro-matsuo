"""Audit five-task first discovery of learning against uniform random search.

The primary clock is the number of action proposals consumed by the mechanics.
Every task is right-censored at the proposal count of its designated primary
learning first hit.  The script is intentionally additive/read-only with respect
to experiment artifacts: it only writes a new analysis directory.

It supports an incomplete random campaign (``status=pending``) for monitoring.
Use ``--require-complete`` for the final, ten-seed report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
EXPECTED_RANDOM_SEEDS = tuple(range(6101, 6111))
TASK_ORDER = ("five_d2", "five_d3", "watersqrt2", "torus2", "water1")
TASK_LABELS = {
    "five_d2": "Five-bar D=2",
    "five_d3": "Five-bar D=3",
    "watersqrt2": "Water D=sqrt(2)",
    "torus2": "Torus D=2",
    "water1": "Water D=1",
}


@dataclass(frozen=True)
class LearningRun:
    seed: int
    history: str
    primary: bool = False


@dataclass(frozen=True)
class TaskSpec:
    task: str
    cap: int
    learning_family: str
    learning_runs: tuple[LearningRun, ...]
    comparison_class: str
    expected_random_seeds: tuple[int, ...]


def default_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            "five_d2",
            233824,
            "historical_ori5_actor_critic",
            (LearningRun(2101, "experiments_ori5/full/five_d2/B/seed2101/training_history.csv", True),),
            "historical_transition_matched_single_learning_run",
            tuple(range(3000003, 3000033)),
        ),
        TaskSpec(
            "five_d3",
            1033,
            "historical_ori5_actor_critic",
            (LearningRun(2101, "experiments_ori5/full/five_d3/B/seed2101/training_history.csv", True),),
            "historical_transition_matched_single_learning_run",
            tuple(range(3000003, 3000033)),
        ),
        TaskSpec(
            "watersqrt2",
            17324,
            "historical_ori5_actor_critic",
            (LearningRun(2101, "experiments_ori5/full/watersqrt2/B/seed2101/training_history.csv", True),),
            "historical_transition_matched_single_learning_run",
            tuple(range(1001, 1011)),
        ),
        TaskSpec(
            "torus2",
            636198,
            "ori6_zero_reject_actor_critic",
            tuple(
                LearningRun(
                    seed,
                    f"experiments_ori6/staged/zero_reject_reward/P0X2RCS05ZR0R1/torus2/A/seed{seed}/training_history.csv",
                    seed == 4203,
                )
                for seed in (4201, 4202, 4203)
            ),
            "geometry_matched_exploratory_reward_not_identical",
            EXPECTED_RANDOM_SEEDS,
        ),
        TaskSpec(
            "water1",
            126319,
            "ori6_zero_reject_water_fallback_actor_critic",
            tuple(
                LearningRun(
                    seed,
                    f"experiments_ori6/staged/zero_reject_water_fallback/P0X2RCS05ZR0WF12/water1/B/seed{seed}/training_history.csv",
                    seed == 4301,
                )
                for seed in (4301, 4302, 4303)
            ),
            "geometry_matched_exploratory_reward_not_identical",
            EXPECTED_RANDOM_SEEDS,
        ),
    )


HISTORICAL_RANDOM_CSV = {
    "five_d2": "ori_codex4_h010_valid_terminal_near_miss_actor_critic_10000_results/uniform_random_search/summary.csv",
    "five_d3": "ori_codex4_2_target_spread_3_h010_valid_terminal_near_miss_actor_critic_10000_results/uniform_random_search/summary.csv",
    "watersqrt2": "experiments_3d/first_discovery/screen_n10_b102400/watersqrt2/trials.csv",
}

SOURCE_COMMON_CAP = {
    "five_d2": 3_999_760,
    "five_d3": 3_999_980,
    "watersqrt2": 102_400,
    "torus2": 636_198,
    "water1": 126_319,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int_value(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_learning_history(
    root: Path,
    spec: TaskSpec,
    run: LearningRun,
    input_manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    path = root / run.history
    errors: list[str] = []
    if not path.is_file():
        return ({
            "task": spec.task,
            "arm": "learning",
            "method": spec.learning_family,
            "seed": run.seed,
            "primary": run.primary,
            "available": False,
            "cap": spec.cap,
        }, [f"missing learning history: {path}"])

    file_hash = sha256_file(path)
    input_manifest.append({"path": str(path.resolve()), "sha256": file_hash, "role": "learning_history"})
    cumulative_steps = 0
    cumulative_projectors = 0
    first_hit: int | None = None
    first_hit_projectors: int | None = None
    first_hit_episode: int | None = None
    row_count = 0
    policies: set[str] = set()
    max_abs_zero_reward = 0.0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"steps", "strict_compact_success"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            errors.append(f"{path}: missing columns {sorted(missing)}")
        for ordinal, row in enumerate(reader, 1):
            row_count += 1
            cumulative_steps += int_value(row.get("steps"))
            cumulative_projectors += int_value(row.get("projection_attempts"))
            policy = str(row.get("ori6_crossing_policy", "")).strip()
            if policy:
                policies.add(policy)
            if row.get("ori6_zero_reject_total_reward") not in (None, ""):
                max_abs_zero_reward = max(max_abs_zero_reward, abs(float(row["ori6_zero_reject_total_reward"])))
            if first_hit is None and bool_value(row.get("strict_compact_success")):
                first_hit = cumulative_steps
                first_hit_projectors = cumulative_projectors
                first_hit_episode = int_value(row.get("episode"), ordinal)

    event = first_hit is not None and first_hit <= spec.cap
    adequate_followup = event or cumulative_steps >= spec.cap
    observed = first_hit if event else min(cumulative_steps, spec.cap)
    if not adequate_followup:
        errors.append(
            f"{path}: only {cumulative_steps} proposals without a hit; cannot censor at cap {spec.cap}"
        )
    if run.primary and first_hit != spec.cap:
        errors.append(f"{path}: primary first hit {first_hit}, expected exactly {spec.cap}")

    expected_zero_policy = "reject_candidate_as_zero_reward_noop_and_continue"
    if spec.learning_family.startswith("ori6_zero"):
        if policies != {expected_zero_policy}:
            errors.append(f"{path}: zero-reject policy mismatch: {sorted(policies)}")
        if max_abs_zero_reward > 1e-12:
            errors.append(f"{path}: nonzero zero-reject reward component {max_abs_zero_reward}")

    return ({
        "task": spec.task,
        "arm": "learning",
        "method": spec.learning_family,
        "seed": run.seed,
        "primary": run.primary,
        "available": True,
        "cap": spec.cap,
        "event": int(event),
        "censored": int(not event),
        "first_hit_proposals": first_hit if event else None,
        "observed_proposals": observed,
        "first_hit_episode": first_hit_episode if event else None,
        "first_hit_projector_calls": first_hit_projectors if event else None,
        "history_proposals": cumulative_steps,
        "history_episodes": row_count,
        "adequate_followup": adequate_followup,
        "crossing_policies": sorted(policies),
        "max_abs_zero_reject_reward": max_abs_zero_reward,
        "source_path": str(path.resolve()),
        "source_sha256": file_hash,
    }, errors)


def load_json_stable(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def discover_random_summaries(random_root: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    found: list[tuple[Path, Mapping[str, Any]]] = []
    if not random_root.is_dir():
        return found
    for path in random_root.rglob("summary.json"):
        payload = load_json_stable(path)
        if payload is None:
            continue
        if payload.get("method") == "uniform" and "mechanics_budget" in payload and "seed" in payload:
            found.append((path, payload))
    return found


def read_historical_random_trials(
    root: Path,
    specs: Sequence[TaskSpec],
    input_manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the transition-matched historical random authorities.

    The original searches ran beyond the learning hit thresholds.  They are
    administratively re-censored here at each task's primary learning hit.
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec in specs:
        relative = HISTORICAL_RANDOM_CSV.get(spec.task)
        if relative is None:
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"missing historical random authority: {path}")
            continue
        file_hash = sha256_file(path)
        input_manifest.append({
            "path": str(path.resolve()),
            "sha256": file_hash,
            "role": "historical_random_authority",
        })
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if spec.task == "watersqrt2" and row.get("method") != "uniform":
                    continue
                seed = int_value(row.get("search_seed", row.get("seed")), -1)
                success = bool_value(row.get("success"))
                first_raw = row.get("first_success_mechanical_steps")
                first_hit = int_value(first_raw) if first_raw not in (None, "") else None
                mechanical_steps = int_value(row.get("mechanical_steps"))
                event = success and first_hit is not None and first_hit <= spec.cap
                adequate = event or mechanical_steps >= spec.cap
                source_cap = SOURCE_COMMON_CAP[spec.task]
                source_event = success and first_hit is not None and first_hit <= source_cap
                source_adequate = source_event or mechanical_steps >= source_cap
                if not adequate:
                    errors.append(
                        f"{path}: seed {seed} stops at {mechanical_steps} before cap {spec.cap} without an event"
                    )
                if not source_adequate:
                    errors.append(
                        f"{path}: seed {seed} stops at {mechanical_steps} before source cap {source_cap} without an event"
                    )
                records.append({
                    "task": spec.task,
                    "arm": "random",
                    "method": "uniform",
                    "seed": seed,
                    "primary": False,
                    "available": True,
                    "cap": spec.cap,
                    "event": int(event),
                    "censored": int(not event),
                    "first_hit_proposals": first_hit if event else None,
                    "observed_proposals": first_hit if event else spec.cap,
                    "source_common_cap": source_cap,
                    "source_event": int(source_event),
                    "source_first_hit_proposals": first_hit if source_event else None,
                    "source_observed_proposals": first_hit if source_event else source_cap,
                    "source_adequate_followup": source_adequate,
                    "first_hit_episode": row.get("first_success_episode") if event else None,
                    "first_hit_projector_calls": row.get("first_success_projector_calls") if event else None,
                    "adequate_followup": adequate,
                    "complete": adequate,
                    "crossing_policy": "historical_task_matched",
                    "source_candidate_reward_integrity_pass": None,
                    "random_source": "historical_transition_matched",
                    "source_path": str(path.resolve()),
                    "source_sha256": file_hash,
                    "config_path": None,
                    "config_sha256": None,
                })
    return records, errors


def authoritative_random_config(summary_path: Path, random_root: Path) -> Path | None:
    """Return the richest ancestor config, falling back to the trial config.

    Trial directories contain a compact identity config.  The invocation root
    contains the transition/reward specification needed by this audit.
    """
    parent = summary_path.parent
    fallback: Path | None = None
    while True:
        candidate = parent / "run_config.json"
        if candidate.is_file():
            fallback = fallback or candidate
            payload = load_json_stable(candidate) or {}
            if "reward_parameters" in payload and "action_law" in payload:
                return candidate
        if parent == random_root or random_root not in parent.parents:
            return fallback
        parent = parent.parent


def read_random_trials(
    random_root: Path,
    specs: Sequence[TaskSpec],
    input_manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    spec_by_task = {spec.task: spec for spec in specs}
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    policies: set[str] = set()
    methods: set[str] = set()
    action_laws: set[str] = set()
    reward_hashes: set[str] = set()
    transition_hashes: set[str] = set()
    all_state_restore = True
    all_step_increment = True
    all_uniform_reward_blind = True

    for path, payload in discover_random_summaries(random_root):
        task = str(payload.get("task", ""))
        if task not in spec_by_task:
            continue
        spec = spec_by_task[task]
        seed = int_value(payload.get("seed"), -1)
        key = (task, seed)
        if key in seen:
            errors.append(f"duplicate random summary for {task} seed {seed}: {path}")
            continue
        seen.add(key)
        summary_hash = sha256_file(path)
        input_manifest.append({"path": str(path.resolve()), "sha256": summary_hash, "role": "random_trial_summary"})

        budget = int_value(payload.get("mechanics_budget"), -1)
        complete = bool_value(payload.get("complete"))
        success = bool_value(payload.get("success"))
        first_hit = payload.get("first_success_mechanical_steps")
        first_hit = int_value(first_hit) if first_hit not in (None, "") else None
        mechanical_steps = int_value(payload.get("mechanical_steps"))
        event = success and first_hit is not None and first_hit <= spec.cap
        adequate = event or (complete and mechanical_steps >= spec.cap)

        if budget != spec.cap:
            errors.append(f"{path}: budget {budget}, expected {spec.cap}")
        if complete and not success and mechanical_steps != spec.cap:
            errors.append(f"{path}: complete censor used {mechanical_steps}, expected exact cap {spec.cap}")
        if success and first_hit != mechanical_steps:
            errors.append(f"{path}: success first hit {first_hit} != stopped mechanics {mechanical_steps}")
        if payload.get("reward_integrity_pass") is not True:
            errors.append(f"{path}: legacy source-candidate reward decomposition integrity is not true")
        if not adequate:
            errors.append(f"{path}: incomplete before cap and has no event")

        policy = str(payload.get("crossing_policy", ""))
        policies.add(policy)
        methods.add(str(payload.get("method", "")))
        if policy != "reject_candidate_and_continue":
            errors.append(f"{path}: random crossing policy {policy!r} is not reject-and-continue")

        config_path = authoritative_random_config(path, random_root)
        config_hash = None
        if config_path is not None:
            config_hash = sha256_file(config_path)
            input_manifest.append({"path": str(config_path.resolve()), "sha256": config_hash, "role": "random_trial_config"})
            config = load_json_stable(config_path) or {}
            action_laws.add(str(config.get("action_law", "")))
            if config.get("reward_sha256"):
                reward_hashes.add(str(config["reward_sha256"]))
            if config.get("transition_semantics_sha256"):
                transition_hashes.add(str(config["transition_semantics_sha256"]))
            crossing = ((config.get("reward_parameters") or {}).get("crossing_transition") or {})
            all_state_restore = all_state_restore and crossing.get("candidate_committed") is False
            all_step_increment = all_step_increment and crossing.get("step_count_increments") is True
            # UniformRandomAgent has a fixed action law and no update method; the
            # harness summary additionally identifies method=uniform.
            all_uniform_reward_blind = all_uniform_reward_blind and config.get("method") == "uniform"
        else:
            errors.append(f"{path}: no run_config.json found")

        records.append({
            "task": task,
            "arm": "random",
            "method": "uniform",
            "seed": seed,
            "primary": False,
            "available": True,
            "cap": spec.cap,
            "event": int(event),
            "censored": int(not event),
            "first_hit_proposals": first_hit if event else None,
            "observed_proposals": first_hit if event else spec.cap,
            "source_common_cap": SOURCE_COMMON_CAP[task],
            "source_event": int(event),
            "source_first_hit_proposals": first_hit if event else None,
            "source_observed_proposals": first_hit if event else SOURCE_COMMON_CAP[task],
            "source_adequate_followup": adequate,
            "first_hit_episode": payload.get("first_success_episode") if event else None,
            "first_hit_projector_calls": payload.get("first_success_projector_calls") if event else None,
            "adequate_followup": adequate,
            "complete": complete,
            "crossing_policy": policy,
            "source_candidate_reward_integrity_pass": payload.get("reward_integrity_pass"),
            "random_source": "fresh_geometry_matched_legacy_reward",
            "source_path": str(path.resolve()),
            "source_sha256": summary_hash,
            "config_path": str(config_path.resolve()) if config_path else None,
            "config_sha256": config_hash,
        })

    invariance = {
        "checked": bool(records),
        "random_method_uniform_only": methods == {"uniform"},
        "action_laws": sorted(action_laws),
        "crossing_policies": sorted(policies),
        "candidate_never_committed_on_crossing": all_state_restore if records else None,
        "crossing_consumes_one_step": all_step_increment if records else None,
        "uniform_agent_reward_blind": all_uniform_reward_blind if records else None,
        "random_reward_sha256_values": sorted(reward_hashes),
        "random_transition_semantics_sha256_values": sorted(transition_hashes),
        "paired_semantics_audit": {
            "water_seed6101_400_steps": "exact nodes/actions/RNG/termination across 20 rejected crossings",
            "forced_horizon_torus_and_water": "exact",
            "water_saved_first_hit_replay": (
                "257 actions and 9 rejected crossings: exact geometry and terminal F/D; only rejected reward differs"
            ),
            "scope": "campaign protocol audit supplied independently of this artifact analyzer",
        },
        "formal_statement": (
            "The uniform agent samples from its fixed action law and performs no parameter update, so rewards do not "
            "influence any later random action. Paired audits established exact nodes, actions, RNG and termination "
            "for water seed6101 over 400 steps and 20 rejections; exact forced-horizon behavior for torus and water; "
            "and exact geometry plus terminal F/D for the 257-action, 9-rejection saved water first hit. The random "
            "harness restores the last safe state and consumes one action proposal before the next sample. Its legacy "
            "rejected reward is intentionally not identical to the learned zero-reward no-op, and this difference "
            "cannot affect a reward-blind uniform agent. Torus2/water1 are therefore geometry/action-proposal matched "
            "exploratory comparisons with reward identity intentionally excluded."
        ),
    }
    if records and not all((
        invariance["random_method_uniform_only"],
        invariance["candidate_never_committed_on_crossing"],
        invariance["crossing_consumes_one_step"],
        invariance["uniform_agent_reward_blind"],
    )):
        errors.append("formal reward-invariance prerequisites failed")
    return records, errors, invariance


def crossing_semantics_ablation(
    root: Path,
    fresh_records: Sequence[Mapping[str, Any]],
    input_manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Descriptive legacy terminal-vs-continuation random comparison at 102400."""
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    common_cap = 102_400
    for task in ("water1", "torus2"):
        legacy_root = root / "experiments_3d" / "first_discovery" / "screen_n10_b102400" / task
        trials_path = legacy_root / "trials.csv"
        legacy_rows: list[dict[str, str]] = []
        if trials_path.is_file():
            digest = sha256_file(trials_path)
            input_manifest.append({
                "path": str(trials_path.resolve()), "sha256": digest,
                "role": "legacy_crossing_ablation_trials",
            })
            with trials_path.open("r", encoding="utf-8-sig", newline="") as handle:
                legacy_rows = [row for row in csv.DictReader(handle) if row.get("method") == "uniform"]
        else:
            errors.append(f"missing crossing ablation authority: {trials_path}")

        crossing_terminals = 0
        history_episodes = 0
        for history_path in sorted((legacy_root / "trials" / "uniform").glob("seed*/episode_history.csv")):
            digest = sha256_file(history_path)
            input_manifest.append({
                "path": str(history_path.resolve()), "sha256": digest,
                "role": "legacy_crossing_ablation_episode_history",
            })
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    history_episodes += 1
                    crossing_terminals += int(bool_value(row.get("self_crossing")))

        legacy_hits = 0
        legacy_episodes = 0
        for row in legacy_rows:
            first = row.get("first_success_mechanical_steps")
            first = int_value(first) if first not in (None, "") else None
            legacy_hits += int(bool_value(row.get("success")) and first is not None and first <= common_cap)
            legacy_episodes += int_value(row.get("episodes"))
        if history_episodes != legacy_episodes:
            errors.append(
                f"{task}: legacy ablation episode count mismatch history={history_episodes}, trials={legacy_episodes}"
            )

        fresh = [row for row in fresh_records if row["task"] == task and row.get("adequate_followup")]
        fresh_hits = [
            int(row["first_hit_proposals"])
            for row in fresh
            if row.get("event") and int(row["first_hit_proposals"]) <= common_cap
        ]
        tasks.append({
            "task": task,
            "common_action_proposal_cap": common_cap,
            "legacy_terminal_crossing": {
                "seeds": sorted(int_value(row.get("seed")) for row in legacy_rows),
                "n": len(legacy_rows),
                "hits_by_cap": legacy_hits,
                "episodes": legacy_episodes,
                "crossing_terminal_episodes": crossing_terminals,
            },
            "fresh_reject_and_continue": {
                "seeds_completed": sorted(int(row["seed"]) for row in fresh),
                "n_completed": len(fresh),
                "n_expected": 10,
                "hits_by_cap": len(fresh_hits),
                "hit_proposals": sorted(fresh_hits),
                "maximum_hit_proposals": max(fresh_hits) if fresh_hits else None,
            },
        })
    return ({
        "design": "unpaired_descriptive_ablation",
        "causal_claim": False,
        "caveat": (
            "Legacy and fresh cohorts use different, unpaired seeds and differ in more than crossing handling; "
            "this is descriptive evidence, not a causal ablation test."
        ),
        "tasks": tasks,
    }, errors)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def arm_stats(records: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    usable = [row for row in records if row.get("available") and row.get("adequate_followup")]
    n = len(usable)
    hits = sum(int(row.get("event", 0)) for row in usable)
    observed = [int(row["observed_proposals"]) for row in usable]
    return {
        "n": n,
        "hits": hits,
        "censored": n - hits,
        "hit_fraction": hits / n if n else None,
        "hit_fraction_wilson95": wilson_interval(hits, n),
        "capped_rmst_proposals": mean(observed) if observed else None,
        "cap": cap,
    }


def source_window_stats(records: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    usable = [row for row in records if row.get("source_adequate_followup")]
    hits = [int(row["source_first_hit_proposals"]) for row in usable if row.get("source_event")]
    observed = [int(row["source_observed_proposals"]) for row in usable]
    n = len(usable)
    return {
        "n": n,
        "hits": len(hits),
        "censored": n - len(hits),
        "hit_fraction": len(hits) / n if n else None,
        "source_common_cap": cap,
        "fastest_hit_proposals": min(hits) if hits else None,
        "median_successful_hit_proposals": median(hits) if hits else None,
        "capped_rmst_proposals": mean(observed) if observed else None,
    }


def summarize_task(spec: TaskSpec, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    learning = [row for row in records if row["task"] == spec.task and row["arm"] == "learning"]
    random = [row for row in records if row["task"] == spec.task and row["arm"] == "random"]
    primary = [row for row in learning if row.get("primary")]
    primary_ok = len(primary) == 1 and primary[0].get("first_hit_proposals") == spec.cap
    random_seeds = sorted(int(row["seed"]) for row in random if row.get("adequate_followup"))
    missing = sorted(set(spec.expected_random_seeds).difference(random_seeds))
    extra = sorted(set(random_seeds).difference(spec.expected_random_seeds))
    random_faster = sum(
        bool_value(row.get("event")) and int(row["first_hit_proposals"]) < spec.cap for row in random
    )
    random_ties = sum(
        bool_value(row.get("event")) and int(row["first_hit_proposals"]) == spec.cap for row in random
    )
    random_stats = arm_stats(random, spec.cap)
    source_stats = source_window_stats(random, SOURCE_COMMON_CAP[spec.task])
    learning_stats = arm_stats(learning, spec.cap)
    n_random = random_stats["n"]
    faster_ci = wilson_interval(random_faster, n_random)
    if n_random == 0:
        verdict = "pending_random"
    elif missing:
        verdict = "partial_random"
    elif random_faster == 0:
        verdict = "no_random_trial_hit_before_primary_learning"
    elif random_faster < n_random / 2:
        verdict = "fewer_than_half_random_trials_hit_before_primary_learning"
    else:
        verdict = "at_least_half_random_trials_hit_before_primary_learning"
    return {
        "task": spec.task,
        "label": TASK_LABELS[spec.task],
        "cap_primary_learning_hit_proposals": spec.cap,
        "learning_family": spec.learning_family,
        "comparison_class": spec.comparison_class,
        "primary_learning_integrity_pass": primary_ok,
        "learning": learning_stats,
        "random": random_stats,
        "random_source_window": source_stats,
        "random_faster_count": random_faster,
        "random_tie_count": random_ties,
        "random_faster_fraction": random_faster / n_random if n_random else None,
        "random_faster_fraction_wilson95": faster_ci,
        "random_rmst_over_primary_learning_hit": (
            random_stats["capped_rmst_proposals"] / spec.cap
            if random_stats["capped_rmst_proposals"] is not None else None
        ),
        "random_source_rmst_over_primary_learning_hit": (
            source_stats["capped_rmst_proposals"] / spec.cap
            if source_stats["capped_rmst_proposals"] is not None else None
        ),
        "expected_random_seeds": list(spec.expected_random_seeds),
        "observed_random_seeds": random_seeds,
        "missing_random_seeds": missing,
        "unexpected_random_seeds": extra,
        "complete": primary_ok and not missing and not extra,
        "descriptive_verdict": verdict,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_utf8_checked(path: Path, text: str) -> None:
    if "\ufffd" in text:
        raise ValueError(f"replacement character found before writing {path}")
    path.write_text(text, encoding="utf-8")
    decoded = path.read_bytes().decode("utf-8")
    if "\ufffd" in decoded:
        raise ValueError(f"replacement character found after reading {path}")


def plot_survival(output: Path, summaries: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes_flat = list(axes.flat)
    for axis, summary in zip(axes_flat, summaries):
        task = str(summary["task"])
        cap = int(summary["cap_primary_learning_hit_proposals"])
        for arm, colour, linestyle in (("random", "#E67E22", "-"), ("learning", "#2878B5", "--")):
            arm_rows = [row for row in records if row["task"] == task and row["arm"] == arm and row.get("adequate_followup")]
            n = len(arm_rows)
            if not n:
                continue
            event_times = sorted(int(row["first_hit_proposals"]) for row in arm_rows if row.get("event"))
            x = [0]
            y = [1.0]
            survival = 1.0
            at_risk = n
            for time in sorted(set(event_times)):
                deaths = event_times.count(time)
                x.extend([time, time])
                y.extend([survival, survival * (1.0 - deaths / at_risk)])
                survival = y[-1]
                at_risk -= deaths
            x.append(cap)
            y.append(survival)
            axis.plot(x, y, color=colour, linestyle=linestyle, linewidth=2, label=f"{arm} (n={n})")
        axis.axvline(cap, color="#333333", linewidth=1, alpha=0.55)
        axis.set_title(TASK_LABELS[task])
        axis.set_xlim(0, cap)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Cumulative action proposals")
        axis.set_ylabel("Not yet found")
        axis.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle("First strict-success survival (task-specific censoring cap)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_summary(output: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [TASK_LABELS[str(row["task"])] for row in summaries]
    primary_rmst = [
        float(row["random_rmst_over_primary_learning_hit"])
        if row["random_rmst_over_primary_learning_hit"] is not None else math.nan
        for row in summaries
    ]
    source_rmst = [
        float(row["random_source_rmst_over_primary_learning_hit"])
        if row["random_source_rmst_over_primary_learning_hit"] is not None else math.nan
        for row in summaries
    ]
    faster = [
        float(row["random_faster_fraction"])
        if row["random_faster_fraction"] is not None else math.nan
        for row in summaries
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = list(range(len(labels)))
    width = 0.38
    axes[0].bar([value - width / 2 for value in positions], primary_rmst, width=width,
                color="#E67E22", label="primary learning-hit cap")
    axes[0].bar([value + width / 2 for value in positions], source_rmst, width=width,
                color="#2878B5", label="source common cap")
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Random capped RMST / primary learning hit")
    axes[0].set_xticks(positions, labels, rotation=25)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, faster, color="#8E5EA2")
    axes[1].set_xticks(positions, labels, rotation=25)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of random trials faster than learning")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    for axis, values in zip(axes, (source_rmst, faster)):
        for index, value in enumerate(values):
            if math.isnan(value):
                axis.text(index, 0.03, "pending", ha="center", va="bottom", rotation=90, color="#777777")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return f"{int(value):,}" if isinstance(value, int) else str(value)


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# 5 task：学習とランダム探索の初回発見比較",
        "",
        f"生成時刻（UTC）: `{report['generated_utc']}`  ",
        f"状態: **{report['status']}**",
        "",
        "## 判定表",
        "",
        "一次時計は累積 action proposal 数です。各 task の打切り上限は、指定した primary learning run の初回 strict success と完全に同じ値です。",
        "",
        "| Task | learning初回 | random hit | randomが先 | random capped RMST | RMST/learning | 状態 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["tasks"]:
        rnd = item["random"]
        lines.append(
            f"| {item['label']} | {fmt_number(item['cap_primary_learning_hit_proposals'])} | "
            f"{rnd['hits']}/{rnd['n']} | {item['random_faster_count']}/{rnd['n']} | "
            f"{fmt_number(rnd['capped_rmst_proposals'])} | "
            f"{fmt_number(item['random_rmst_over_primary_learning_hit'])} | {item['descriptive_verdict']} |"
        )
    lines.extend([
        "",
        "`randomが先` は random trial の first hit が primary learning hit より**厳密に小さい**場合です。同値は tie として別計数しています。capped RMST は未発見trialをtask固有capに置いた観測時間の平均です。",
        "",
        "## Source共通観測窓での二次統計",
        "",
        "一次表のcapではrandom RMST/learningが必ず1以下になるため、historical authorityが持つ共通観測窓まで延ばした二次統計も示します。fresh Torus/Waterは一次capと同じです。",
        "",
        "| Task | source cap | hit | fastest | successful median | capped RMST | RMST/learning |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in report["tasks"]:
        source = item["random_source_window"]
        lines.append(
            f"| {item['label']} | {fmt_number(source['source_common_cap'])} | {source['hits']}/{source['n']} | "
            f"{fmt_number(source['fastest_hit_proposals'])} | "
            f"{fmt_number(source['median_successful_hit_proposals'])} | "
            f"{fmt_number(source['capped_rmst_proposals'])} | "
            f"{fmt_number(item['random_source_rmst_over_primary_learning_hit'])} |"
        )
    lines.extend([
        "",
        "## 解釈上の制約",
        "",
        "- Five-bar D=2/D=3 と Water D=sqrt(2) はtask transitionと整合するhistorical uniform authorityを使います。ただしlearningはhistorical ori5の各1 runだけなので、結果は記述的比較であり、学習法一般の統計的優越を示す検定ではありません。",
        "- Torus D=2 と Water D=1 は3 learning seedsをcohortとして併記しますが、primary run/capはscreening後に選ばれています。post-selection biasがあるので探索的結果です。",
        "- random actionは現在mode上の等方方向とsigned widthの一様サンプルです。方策NNの初期分布そのものとの同一性は主張しません。",
        "",
        "## crossing と報酬の不変性監査",
        "",
        report["reward_invariance"]["formal_statement"],
        "",
        f"監査結果: `{json.dumps({k: v for k, v in report['reward_invariance'].items() if k != 'formal_statement'}, ensure_ascii=False)}`",
        "",
        "## Crossing semanticsの記述的比較（102,400 proposals）",
        "",
        "| Task | legacy terminal crossing | crossing terminal/episodes | fresh reject-and-continue | fresh hit proposals |",
        "|---|---:|---:|---:|---|",
    ])
    for item in report["crossing_semantics_descriptive_ablation"]["tasks"]:
        legacy = item["legacy_terminal_crossing"]
        fresh = item["fresh_reject_and_continue"]
        lines.append(
            f"| {TASK_LABELS[item['task']]} | {legacy['hits_by_cap']}/{legacy['n']} | "
            f"{legacy['crossing_terminal_episodes']}/{legacy['episodes']} | "
            f"{fresh['hits_by_cap']}/{fresh['n_completed']} completed "
            f"(expected {fresh['n_expected']}) | {fresh['hit_proposals']} |"
        )
    lines.extend([
        "",
        report["crossing_semantics_descriptive_ablation"]["caveat"],
        "",
        "## 完全性",
        "",
    ])
    for item in report["tasks"]:
        lines.append(
            f"- {item['label']}: random seeds={item['observed_random_seeds']}; missing={item['missing_random_seeds']}; "
            f"primary learning integrity={item['primary_learning_integrity_pass']}"
        )
    if report["integrity_errors"]:
        lines.extend(["", "## Integrity errors", ""])
        lines.extend(f"- {message}" for message in report["integrity_errors"])
    lines.extend([
        "",
        "## 成果物",
        "",
        "- `task_summary.csv`: task単位の主要統計",
        "- `trial_level.csv`: learning/randomのseed単位event/censor記録",
        "- `first_discovery_survival.png`: task別の未発見率曲線",
        "- `normalized_summary.png`: normalized RMST と random-faster率",
        "- `input_manifest.json`: 読み込んだauthority fileのSHA-256",
    ])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--random-root",
        type=Path,
        default=Path("experiments_ori6/random/first_discovery_matched_20260827"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments_ori6/analysis/first_discovery_vs_random_20260827"),
    )
    parser.add_argument(
        "--fresh-random-seeds",
        default="6101-6110",
        help="Expected seeds for fresh torus2/water1 only; historical task seed sets are fixed authorities.",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def parse_seed_spec(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = token.split("-", 1)
            seeds.extend(range(int(start), int(stop) + 1))
        else:
            seeds.append(int(token))
    return tuple(sorted(set(seeds)))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    random_root = args.random_root if args.random_root.is_absolute() else root / args.random_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    fresh_seeds = parse_seed_spec(args.fresh_random_seeds)
    specs = tuple(
        TaskSpec(
            spec.task,
            spec.cap,
            spec.learning_family,
            spec.learning_runs,
            spec.comparison_class,
            fresh_seeds if spec.task in {"torus2", "water1"} else spec.expected_random_seeds,
        )
        for spec in default_specs()
    )
    input_manifest: list[dict[str, Any]] = []
    integrity_errors: list[str] = []
    records: list[dict[str, Any]] = []

    for spec in specs:
        for learning_run in spec.learning_runs:
            record, errors = read_learning_history(root, spec, learning_run, input_manifest)
            records.append(record)
            integrity_errors.extend(errors)

    historical_records, historical_errors = read_historical_random_trials(root, specs, input_manifest)
    records.extend(historical_records)
    integrity_errors.extend(historical_errors)
    random_records, random_errors, invariance = read_random_trials(random_root, specs, input_manifest)
    records.extend(random_records)
    integrity_errors.extend(random_errors)
    crossing_ablation, crossing_errors = crossing_semantics_ablation(root, random_records, input_manifest)
    integrity_errors.extend(crossing_errors)
    summaries = [summarize_task(spec, records) for spec in specs]
    complete = all(item["complete"] for item in summaries) and not integrity_errors
    status = "complete" if complete else "pending"

    # Deduplicate manifest entries (the nearest config can be shared by summaries).
    manifest_map = {(item["path"], item["role"]): item for item in input_manifest}
    input_manifest = sorted(manifest_map.values(), key=lambda item: (item["role"], item["path"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "primary_clock": "cumulative_action_proposals",
        "strict_event": "strict_compact_success == 1",
        "censoring": "task-specific exact common cap equal to designated primary learning first hit",
        "expected_random_seeds_by_task": {
            spec.task: list(spec.expected_random_seeds) for spec in specs
        },
        "random_root": str(random_root.resolve()),
        "tasks": summaries,
        "reward_invariance": invariance,
        "crossing_semantics_descriptive_ablation": crossing_ablation,
        "selection_and_transition_caveats": [
            "Historical ori5 tasks use transition-matched historical uniform authorities but have one learning seed each.",
            "Torus/water primary runs and caps were selected after a three-seed screen.",
            "Legacy random rewards differ on rejected crossing, but uniform actions are reward-blind.",
            "Torus/water comparisons are geometry/action-proposal matched exploratory comparisons; reward identity is intentionally excluded.",
        ],
        "integrity_errors": integrity_errors,
        "input_manifest_sha256": hashlib.sha256(
            json.dumps(input_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

    trial_columns = (
        "task", "arm", "method", "seed", "primary", "cap", "event", "censored",
        "first_hit_proposals", "observed_proposals", "first_hit_episode",
        "first_hit_projector_calls", "adequate_followup", "complete", "crossing_policy",
        "random_source", "source_candidate_reward_integrity_pass",
        "source_common_cap", "source_event", "source_first_hit_proposals",
        "source_observed_proposals", "source_adequate_followup",
        "source_path", "source_sha256", "config_path", "config_sha256",
    )
    write_csv(output / "trial_level.csv", records, trial_columns)
    flat_summaries = []
    for item in summaries:
        flat_summaries.append({
            "task": item["task"],
            "label": item["label"],
            "cap_primary_learning_hit_proposals": item["cap_primary_learning_hit_proposals"],
            "learning_n": item["learning"]["n"],
            "learning_hits": item["learning"]["hits"],
            "learning_capped_rmst_proposals": item["learning"]["capped_rmst_proposals"],
            "random_n": item["random"]["n"],
            "random_hits": item["random"]["hits"],
            "random_censored": item["random"]["censored"],
            "random_hit_fraction": item["random"]["hit_fraction"],
            "random_faster_count": item["random_faster_count"],
            "random_faster_fraction": item["random_faster_fraction"],
            "random_tie_count": item["random_tie_count"],
            "random_capped_rmst_proposals": item["random"]["capped_rmst_proposals"],
            "random_rmst_over_primary_learning_hit": item["random_rmst_over_primary_learning_hit"],
            "source_common_cap": item["random_source_window"]["source_common_cap"],
            "source_random_hits": item["random_source_window"]["hits"],
            "source_random_fastest_hit_proposals": item["random_source_window"]["fastest_hit_proposals"],
            "source_random_median_successful_hit_proposals": item["random_source_window"]["median_successful_hit_proposals"],
            "source_random_capped_rmst_proposals": item["random_source_window"]["capped_rmst_proposals"],
            "source_random_rmst_over_primary_learning_hit": item["random_source_rmst_over_primary_learning_hit"],
            "complete": item["complete"],
            "descriptive_verdict": item["descriptive_verdict"],
            "comparison_class": item["comparison_class"],
        })
    write_csv(output / "task_summary.csv", flat_summaries, tuple(flat_summaries[0]))
    write_utf8_checked(output / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_utf8_checked(
        output / "input_manifest.json",
        json.dumps({"schema_version": SCHEMA_VERSION, "files": input_manifest}, ensure_ascii=False, indent=2) + "\n",
    )
    write_utf8_checked(output / "report_ja.md", markdown_report(report))
    plot_survival(output / "first_discovery_survival.png", summaries, records)
    plot_summary(output / "normalized_summary.png", summaries)

    print(json.dumps({
        "status": status,
        "output": str(output.resolve()),
        "random_trials": len(historical_records) + len(random_records),
        "integrity_errors": len(integrity_errors),
        "missing_by_task": {item["task"]: item["missing_random_seeds"] for item in summaries},
    }, ensure_ascii=False, indent=2))
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
