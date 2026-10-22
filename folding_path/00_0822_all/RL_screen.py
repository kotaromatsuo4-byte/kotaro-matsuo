"""First-strict-discovery screening without learning curves or paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from main import task_config
from RL_main import run_rl


SCREEN_SEEDS = {
    "five_d2": tuple(range(9101, 9106)),
    "five_d3": tuple(range(9101, 9106)),
    "torus_d2": tuple(range(8101, 8106)),
}

SUMMARY_FIELDS = (
    "task",
    "seed",
    "max_episodes",
    "max_steps_per_episode",
    "max_step_width",
    "success",
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
    "optimizer_updates_completed",
    "episodes_completed",
    "cumulative_mechanical_steps",
    "terminal_flatness",
    "terminal_spread",
    "spread_error",
)


def screen_seed(
    task_name: str,
    seed: int,
    max_episodes: int = 10_000,
) -> dict:
    config = task_config(task_name, max_episodes)
    config.update({
        "seed": int(seed),
        "save_outputs": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
    })
    result = run_rl(config)
    success = result["first_strict_success_episode"] is not None
    return {
        "task": task_name,
        "seed": int(seed),
        "max_episodes": int(max_episodes),
        "max_steps_per_episode": int(config["max_steps"]),
        "max_step_width": float(config["max_step_width"]),
        "success": int(success),
        "first_strict_success_episode": (
            result["first_strict_success_episode"]
        ),
        "first_strict_success_episode_steps": (
            result["first_strict_success_episode_steps"]
        ),
        "first_strict_success_mechanical_steps": (
            result["first_strict_success_mechanical_steps"]
        ),
        "optimizer_updates_before_first_success": (
            result["optimizer_updates_before_first_success"]
        ),
        "optimizer_updates_completed": (
            result["optimizer_updates_completed"]
        ),
        "episodes_completed": result["episodes"],
        "cumulative_mechanical_steps": (
            result["cumulative_mechanical_steps"]
        ),
        "terminal_flatness": result["last_terminal_flatness"],
        "terminal_spread": result["last_terminal_spread"],
        "spread_error": result["last_spread_error"],
    }


def save_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(SCREEN_SEEDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-episodes", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed not in SCREEN_SEEDS[args.task]:
        raise ValueError(f"seed {args.seed} is not assigned to {args.task}")
    row = screen_seed(args.task, args.seed, args.max_episodes)
    save_summary(args.output, row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
