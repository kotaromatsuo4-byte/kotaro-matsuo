"""Screen torus first discovery for three target-free spectral weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from main import task_config
from RL_main import run_rl


EXPERIMENT_NAME = "20260830_torus_spectral_screen_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
TASK_NAME = "torus_d2"
ALPHAS = ("0.05", "0.25", "0.5")
SEEDS = tuple(range(8101, 8106))
MAX_EPISODES = 10_000

FIELDS = (
    "task",
    "spectral_alpha",
    "seed",
    "success",
    "censored_at_max_episodes",
    "episodes_completed",
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
    "optimizer_updates_at_stop",
    "mechanical_steps_at_stop",
    "max_episodes",
    "max_steps_per_episode",
    "max_step_width",
    "actor_reference_horizon",
    "batch_size",
    "actor_learning_rate",
    "critic_learning_rate",
)


def alpha_label(alpha: str) -> str:
    return f"alpha_{alpha.replace('.', 'p')}"


def experiment_config(alpha: str, seed: int, episodes: int) -> dict:
    config = task_config(TASK_NAME, episodes)
    config.update({
        "seed": int(seed),
        "spectral_weight": float(alpha),
        "actor_reference_horizon": int(config["max_steps"]),
        "adaptive_after_first_success": True,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
        "save_outputs": False,
        "save_diagnostics": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
    })
    return config


def screen(alpha: str, seed: int, max_episodes: int) -> dict:
    config = experiment_config(alpha, seed, max_episodes)
    result = run_rl(config)
    success = result["first_strict_success_episode"] is not None
    updates_at_stop = (
        result["optimizer_updates_before_first_success"]
        if success
        else result["optimizer_updates_completed"]
    )
    row = {
        "task": TASK_NAME,
        "spectral_alpha": alpha,
        "seed": int(seed),
        "success": int(success),
        "censored_at_max_episodes": int(not success),
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
        "optimizer_updates_at_stop": int(updates_at_stop),
        "mechanical_steps_at_stop": int(
            result["cumulative_mechanical_steps"]
        ),
        "max_episodes": int(max_episodes),
        "max_steps_per_episode": int(config["max_steps"]),
        "max_step_width": float(config["max_step_width"]),
        "actor_reference_horizon": int(
            config["actor_reference_horizon"]
        ),
        "batch_size": int(config["batch_size"]),
        "actor_learning_rate": float(config["actor_learning_rate"]),
        "critic_learning_rate": float(config["critic_learning_rate"]),
    }
    output = (
        EXPERIMENT_DIRECTORY
        / "screening"
        / "workers"
        / f"{alpha_label(alpha)}_seed{seed}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)
    return row


def merge() -> Path:
    rows = []
    worker_directory = EXPERIMENT_DIRECTORY / "screening" / "workers"
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    alpha_order = {alpha: index for index, alpha in enumerate(ALPHAS)}
    rows.sort(key=lambda row: (
        alpha_order[row["spectral_alpha"]], int(row["seed"])
    ))
    output = (
        EXPERIMENT_DIRECTORY / "screening" / "first_success_summary.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows into {output}", flush=True)
    return output


def write_manifest() -> Path:
    base = task_config(TASK_NAME, MAX_EPISODES)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "task": TASK_NAME,
        "spectral_alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "max_episodes": MAX_EPISODES,
        "stop_rule": "first strict success, otherwise censor at max episodes",
        "comparison_metrics": [
            "first strict-success episode",
            "optimizer updates completed before first strict success",
            "cumulative mechanical steps through first strict success",
        ],
        "controlled_settings": {
            "actor_and_entropy_reduction": (
                "sum / (actual_batch_episodes * max_steps)"
            ),
            "actor_reference_horizon": int(base["max_steps"]),
            "batch_size": int(base["batch_size"]),
            "actor_learning_rate": float(base["actor_learning_rate"]),
            "critic_learning_rate": float(base["critic_learning_rate"]),
            "fixed_episode_actor_lr_schedule": (
                "disabled because adaptive mode is active"
            ),
            "post_success_adaptation": (
                "configured as batch 16 and actor LR x0.5, but not executed "
                "because screening stops on first strict success"
            ),
            "max_steps_per_episode": int(base["max_steps"]),
            "max_step_width": float(base["max_step_width"]),
            "crossing_policy": (
                "reject candidate as a zero-reward no-op; continue episode"
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
    screen_parser.add_argument("--alpha", choices=ALPHAS, required=True)
    screen_parser.add_argument("--seed", type=int, required=True)
    screen_parser.add_argument(
        "--max-episodes", type=int, default=MAX_EPISODES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        write_manifest()
    elif args.command == "merge":
        merge()
    elif args.command == "screen":
        if args.seed not in SEEDS:
            raise ValueError(f"seed {args.seed} is not assigned")
        screen(args.alpha, args.seed, args.max_episodes)


if __name__ == "__main__":
    main()
