"""Fixed-horizon actor loss and success-adaptive five-bar experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from main import task_config
from RL_main import run_rl


EXPERIMENT_NAME = "20260830_fixednorm_adaptive_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
TASKS = ("five_d2", "five_d3")
SCREEN_SEEDS = tuple(range(9101, 9106))
FULL_SEED = 9103
FULL_EPISODES = {"five_d2": 10_000, "five_d3": 5_000}

SCREEN_FIELDS = (
    "task",
    "seed",
    "success",
    "episodes_completed",
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
    "cumulative_mechanical_steps",
    "actor_reference_horizon",
    "initial_batch_size",
    "post_success_batch_size",
    "initial_actor_learning_rate",
    "post_success_actor_learning_rate",
)


def experiment_config(
    task_name: str,
    episodes: int,
    seed: int,
) -> dict:
    config = task_config(task_name, episodes)
    config.update({
        "seed": int(seed),
        "actor_reference_horizon": int(config["max_steps"]),
        "adaptive_after_first_success": True,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
    })
    return config


def screen(task_name: str, seed: int, max_episodes: int) -> dict:
    config = experiment_config(task_name, max_episodes, seed)
    config.update({
        "save_outputs": False,
        "save_diagnostics": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
    })
    result = run_rl(config)
    row = {
        "task": task_name,
        "seed": int(seed),
        "success": int(result["first_strict_success_episode"] is not None),
        "episodes_completed": int(result["episodes"]),
        "first_strict_success_episode": result[
            "first_strict_success_episode"
        ],
        "first_strict_success_episode_steps": result[
            "first_strict_success_episode_steps"
        ],
        "first_strict_success_mechanical_steps": result[
            "first_strict_success_mechanical_steps"
        ],
        "optimizer_updates_before_first_success": result[
            "optimizer_updates_before_first_success"
        ],
        "cumulative_mechanical_steps": int(
            result["cumulative_mechanical_steps"]
        ),
        "actor_reference_horizon": int(config["actor_reference_horizon"]),
        "initial_batch_size": int(config["batch_size"]),
        "post_success_batch_size": int(config["post_success_batch_size"]),
        "initial_actor_learning_rate": float(
            config["actor_learning_rate"]
        ),
        "post_success_actor_learning_rate": float(
            config["actor_learning_rate"]
            * config["post_success_actor_lr_factor"]
        ),
    }
    output = (
        EXPERIMENT_DIRECTORY
        / "screening"
        / "workers"
        / f"{task_name}_seed{seed}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCREEN_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)
    return row


def merge_screening() -> Path:
    worker_directory = EXPERIMENT_DIRECTORY / "screening" / "workers"
    rows = []
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    rows.sort(key=lambda row: (row["task"], int(row["seed"])))
    output = (
        EXPERIMENT_DIRECTORY
        / "screening"
        / "first_success_summary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCREEN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows into {output}", flush=True)
    return output


def run_full(task_name: str) -> dict:
    episodes = FULL_EPISODES[task_name]
    config = experiment_config(task_name, episodes, FULL_SEED)
    config.update({
        "save_outputs": True,
        "save_diagnostics": True,
        "stop_on_first_strict_success": False,
        "print_progress": True,
        "print_result": True,
        "rl_output_directory": (
            EXPERIMENT_DIRECTORY
            / "long"
            / task_name
            / f"seed{FULL_SEED}"
        ),
    })
    return run_rl(config)


def write_manifest() -> Path:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "tasks": list(TASKS),
        "screen_seeds": list(SCREEN_SEEDS),
        "pre_registered_full_seed": FULL_SEED,
        "full_episodes": FULL_EPISODES,
        "method": {
            "actor_and_entropy_reduction": (
                "sum / (actual_batch_episodes * max_steps)"
            ),
            "critic_reduction": "transition mean",
            "advantage_normalization": "all transitions in batch",
            "discovery_batch_size": 8,
            "post_first_strict_batch_size": 16,
            "post_first_strict_actor_lr_factor": 0.5,
            "critic_lr_change": "none",
            "fixed_episode_actor_lr_schedule": "disabled in adaptive mode",
        },
        "stability_rule": {
            "confirmation": "rolling-200 strict success rate >= 0.90",
            "collapse": "rolling-200 < 0.80 after confirmation",
            "final_window": (
                "last-1000 success >= 0.95 and every rolling-200 "
                "within last 1000 >= 0.90"
            ),
            "retention": (
                "confirmation observed for >=1000 episodes, no collapse, "
                "and final-window rule holds"
            ),
        },
    }
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = EXPERIMENT_DIRECTORY / "manifest.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(output, flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("manifest")
    subparsers.add_parser("merge")

    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--task", choices=TASKS, required=True)
    screen_parser.add_argument("--seed", type=int, required=True)
    screen_parser.add_argument("--max-episodes", type=int, default=10_000)

    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--task", choices=TASKS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        write_manifest()
    elif args.command == "merge":
        merge_screening()
    elif args.command == "screen":
        if args.seed not in SCREEN_SEEDS:
            raise ValueError(f"seed {args.seed} is not assigned")
        screen(args.task, args.seed, args.max_episodes)
    elif args.command == "full":
        run_full(args.task)


if __name__ == "__main__":
    main()
