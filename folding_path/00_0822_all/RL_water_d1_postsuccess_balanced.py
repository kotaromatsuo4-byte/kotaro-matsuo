"""Exploratory paired follow-up for water D1 post-success adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import RL_water_alpha_experiment as baseline_runner
from RL_main import strict_success_retention_statistics


BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_NAME = "20260831_water_d1_postsuccess_balanced_v1"
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
BASELINE_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments"
    / "20260831_water_alpha0p05_tuning_v1"
)
TASK = "water_d1"
BASE_CANDIDATE = "C_conservative"
VARIANT = "C_balanced_post"
SEEDS = tuple(range(8101, 8106))
EPISODES = 10_000
POST_OVERRIDES = {
    "post_success_batch_size": 16,
    "post_success_actor_lr_factor": 0.5,
}
SOURCE_FILES = (
    "main.py", "env.py", "policy.py", "agent.py", "RL_main.py",
    "RL_water_alpha_experiment.py", Path(__file__).name,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


BASE_SETTINGS = dict(baseline_runner.CANDIDATES[BASE_CANDIDATE])
VARIANT_SETTINGS = {**BASE_SETTINGS, **POST_OVERRIDES}
baseline_runner.EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY
baseline_runner.TASKS = (TASK,)
baseline_runner.EVALUATION_SEEDS = SEEDS
baseline_runner.CANDIDATES[VARIANT] = VARIANT_SETTINGS
_base_experiment_config = baseline_runner.experiment_config


def variant_experiment_config(
    task_name: str, candidate: str, seed: int, episodes: int
) -> dict:
    if task_name != TASK or candidate != VARIANT:
        raise ValueError((task_name, candidate))
    config = _base_experiment_config(
        task_name, BASE_CANDIDATE, seed, episodes
    )
    before = {
        key: config[key] for key in POST_OVERRIDES
    }
    if before != {
        "post_success_batch_size": 32,
        "post_success_actor_lr_factor": 0.25,
    }:
        raise RuntimeError(f"unexpected baseline post-success settings: {before}")
    config.update(POST_OVERRIDES)
    output = (
        EXPERIMENT_DIRECTORY / "full" / TASK / VARIANT / f"seed{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable(config), file, indent=2, ensure_ascii=False)
    return config


baseline_runner.experiment_config = variant_experiment_config


def write_manifest() -> Path:
    source_hashes = {
        str(name): file_hash(BASE_DIRECTORY / name) for name in SOURCE_FILES
    }
    baseline_manifest = BASELINE_DIRECTORY / "manifest.json"
    data = {
        "experiment": EXPERIMENT_NAME,
        "status": "exploratory follow-up selected after inspecting baseline",
        "task": TASK,
        "seeds": list(SEEDS),
        "episodes": EPISODES,
        "baseline_experiment": str(BASELINE_DIRECTORY),
        "baseline_manifest_sha256": file_hash(baseline_manifest),
        "base_candidate": BASE_CANDIDATE,
        "base_settings": BASE_SETTINGS,
        "variant": VARIANT,
        "variant_settings": VARIANT_SETTINGS,
        "only_intended_changes": POST_OVERRIDES,
        "interpretation": (
            "Discovery is paired and unchanged; this tests amplification and "
            "retention after first strict success, not faster discovery."
        ),
        "source_sha256": source_hashes,
    }
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_DIRECTORY / "manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    selection = {
        TASK: {
            "candidate": VARIANT,
            "promising_pilot": True,
            "selection_rule": "exploratory paired post-success follow-up",
        }
    }
    selection_path = EXPERIMENT_DIRECTORY / "tuning" / "selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2, ensure_ascii=False)
    print(path, flush=True)
    return path


def run_worker(seed: int) -> dict:
    return baseline_runner.full(TASK, VARIANT, seed, EPISODES, smoke=False)


def read_episode_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def first_success(successes: list[int]) -> int | None:
    return next(
        (index for index, value in enumerate(successes, start=1) if value),
        None,
    )


def first_rolling_episode(successes: list[int], required: int) -> int | None:
    window = 200
    values = np.asarray(successes, dtype=np.int64)
    if values.size < window:
        return None
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(values)))
    counts = cumulative[window:] - cumulative[:-window]
    indices = np.flatnonzero(counts >= required)
    return None if not indices.size else int(indices[0] + window)


def paired_comparison() -> Path:
    rows = []
    exact_columns = (
        "episode", "reward", "terminal_flatness", "terminal_spread",
        "spread_error", "strict_success", "termination_reason", "steps",
        "cumulative_mechanical_steps", "rejected_crossings",
        "rollout_phase", "actor_lr_used", "batch_target_used",
        "buffer_episode_count", "optimizer_updates_before_episode_update",
    )
    for seed in SEEDS:
        baseline_output = (
            BASELINE_DIRECTORY / "full" / TASK / BASE_CANDIDATE / f"seed{seed}"
        )
        variant_output = (
            EXPERIMENT_DIRECTORY / "full" / TASK / VARIANT / f"seed{seed}"
        )
        baseline_metrics = read_episode_metrics(
            baseline_output / "episode_metrics.csv"
        )
        variant_metrics = read_episode_metrics(
            variant_output / "episode_metrics.csv"
        )
        if len(baseline_metrics) != EPISODES or len(variant_metrics) != EPISODES:
            raise RuntimeError(f"seed {seed}: incomplete paired metrics")
        baseline_success = [int(row["strict_success"]) for row in baseline_metrics]
        variant_success = [int(row["strict_success"]) for row in variant_metrics]
        baseline_first = first_success(baseline_success)
        variant_first = first_success(variant_success)
        if baseline_first != variant_first:
            raise RuntimeError(
                f"seed {seed}: first success differs "
                f"({baseline_first}, {variant_first})"
            )
        comparison_stop = EPISODES if baseline_first is None else baseline_first
        for index in range(comparison_stop):
            for column in exact_columns:
                if baseline_metrics[index][column] != variant_metrics[index][column]:
                    raise RuntimeError(
                        f"seed {seed} episode {index + 1}: {column} differs"
                    )
        for arm, successes in (
            ("conservative", baseline_success),
            ("balanced", variant_success),
        ):
            retention = strict_success_retention_statistics(successes)
            row = {
                "seed": seed,
                "arm": arm,
                "first_strict_success_episode": baseline_first,
            }
            for horizon in (200, 500, 1000):
                if baseline_first is None:
                    rate = ""
                else:
                    start = baseline_first - 1
                    stop = min(start + horizon, EPISODES)
                    rate = float(np.mean(successes[start:stop]))
                row[f"success_rate_first_{horizon}"] = rate
            first_50 = first_rolling_episode(successes, 100)
            first_90 = first_rolling_episode(successes, 180)
            row.update({
                "first_rolling_200_50pct_episode": first_50,
                "episodes_from_first_to_rolling_50pct": (
                    "" if first_50 is None or baseline_first is None
                    else first_50 - baseline_first
                ),
                "first_rolling_200_90pct_episode": first_90,
                "episodes_from_first_to_rolling_90pct": (
                    "" if first_90 is None or baseline_first is None
                    else first_90 - baseline_first
                ),
                "last_1000_strict_success_rate": retention[
                    "last_1000_strict_success_rate"
                ],
                "confirmation_episode": retention[
                    "rolling_200_confirmation_episode_90pct"
                ],
                "collapsed_after_confirmation": int(retention[
                    "dropped_below_80pct_after_confirmation"
                ]),
                "retention_confirmed": int(retention["retention_confirmed"]),
            })
            rows.append(row)
    path = EXPERIMENT_DIRECTORY / "full" / "paired_comparison.csv"
    baseline_runner.write_csv(path, tuple(rows[0]), rows)
    return path


def merge() -> Path:
    output = baseline_runner.merge_full()
    comparison = paired_comparison()
    print(comparison, flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("merge")
    worker = subparsers.add_parser("full")
    worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        write_manifest()
    elif args.command == "full":
        run_worker(args.seed)
    else:
        merge()


if __name__ == "__main__":
    main()
