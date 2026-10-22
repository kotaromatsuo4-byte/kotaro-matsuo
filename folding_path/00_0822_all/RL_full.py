"""Run the pre-registered long RL trajectory for one selected task."""

from __future__ import annotations

import argparse
from pathlib import Path

from main import task_config
from RL_main import run_rl


FULL_SEEDS = {
    "five_d2": 9103,
    "five_d3": 9103,
    "torus_d2": 8103,
}


def run_full(task_name: str, episodes: int = 10_000) -> dict:
    seed = FULL_SEEDS[task_name]
    config = task_config(task_name, episodes)
    config.update({
        "seed": seed,
        "save_outputs": True,
        "stop_on_first_strict_success": False,
        "print_progress": True,
        "print_result": True,
        "rl_output_directory": (
            Path(config["result_directory"]).parent
            / f"rl_{episodes}"
            / task_name
            / f"seed{seed}"
        ),
    })
    return run_rl(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(FULL_SEEDS), required=True)
    parser.add_argument("--episodes", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full(args.task, args.episodes)


if __name__ == "__main__":
    main()
