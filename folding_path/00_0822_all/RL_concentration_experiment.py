"""First-discovery concentration diagnostics for the five selected tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")


from main import task_config
from RL_main import run_rl


EXPERIMENT_NAME = "20260901_first_discovery_concentration_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
TASKS = (
    "five_d2",
    "five_d3",
    "water_sqrt2",
    "torus_d2",
    "water_d1",
)
MAX_EPISODES = 10_000

# Earliest first-success seed in each completed five-seed screen.  five_d3
# had a tie at episode 9, so its already pre-registered full-run seed wins.
SEEDS = {
    "five_d2": 9103,
    "five_d3": 9103,
    "water_sqrt2": 8105,
    "torus_d2": 8105,
    "water_d1": 8102,
}

# These are the latest selected alpha=0.05 discovery settings.  In
# particular, main.py still contains older torus and water-D1 presets, so the
# overrides are explicit here.
TASK_OVERRIDES = {
    "five_d2": {},
    "five_d3": {},
    "water_sqrt2": {
        "batch_size": 8,
        "actor_learning_rate": 5.0e-4,
        "critic_learning_rate": 1.5e-4,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
    },
    "torus_d2": {},
    "water_d1": {
        "batch_size": 16,
        "actor_learning_rate": 2.5e-4,
        "critic_learning_rate": 7.5e-5,
        "post_success_batch_size": 32,
        "post_success_actor_lr_factor": 0.25,
    },
}

SUMMARY_FIELDS = (
    "task",
    "seed",
    "success",
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
    "episodes_completed",
    "cumulative_mechanical_steps",
    "max_episodes",
    "max_steps_per_episode",
    "max_step_width",
    "batch_size",
    "actor_learning_rate",
    "critic_learning_rate",
    "spectral_alpha",
    "near_miss_spread_error_far",
    "output_directory",
)


def experiment_config(task_name: str) -> dict:
    config = task_config(task_name, MAX_EPISODES)
    config.update({
        "spectral_weight": 0.05,
        "near_miss_spread_error_far": 1.0,
        "seed": int(SEEDS[task_name]),
        "actor_reference_horizon": int(config["max_steps"]),
        "adaptive_after_first_success": True,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
    })
    config.update(TASK_OVERRIDES[task_name])
    config.update({
        "save_outputs": True,
        "save_diagnostics": True,
        "save_concentration_diagnostics": True,
        "stop_on_first_strict_success": True,
        "print_progress": True,
        "print_result": False,
        "rl_output_directory": (
            EXPERIMENT_DIRECTORY / "runs" / task_name
            / f"seed{SEEDS[task_name]}"
        ),
    })
    return config


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_task(task_name: str) -> dict:
    config = experiment_config(task_name)
    result = run_rl(config)
    row = {
        "task": task_name,
        "seed": int(config["seed"]),
        "success": int(result["first_strict_success_episode"] is not None),
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
        "episodes_completed": int(result["episodes"]),
        "cumulative_mechanical_steps": int(
            result["cumulative_mechanical_steps"]
        ),
        "max_episodes": MAX_EPISODES,
        "max_steps_per_episode": int(config["max_steps"]),
        "max_step_width": float(config["max_step_width"]),
        "batch_size": int(config["batch_size"]),
        "actor_learning_rate": float(config["actor_learning_rate"]),
        "critic_learning_rate": float(config["critic_learning_rate"]),
        "spectral_alpha": float(config["spectral_weight"]),
        "near_miss_spread_error_far": float(
            config["near_miss_spread_error_far"]
        ),
        "output_directory": str(config["rl_output_directory"]),
    }
    worker_path = (
        EXPERIMENT_DIRECTORY / "workers" / f"{task_name}.csv"
    )
    write_csv(worker_path, [row])
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def merge() -> Path:
    rows = []
    for task_name in TASKS:
        path = EXPERIMENT_DIRECTORY / "workers" / f"{task_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    rows.sort(key=lambda row: TASKS.index(row["task"]))
    output = EXPERIMENT_DIRECTORY / "first_discovery_summary.csv"
    write_csv(output, rows)
    print(output, flush=True)
    return output


def serialisable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialisable(item) for item in value]
    if isinstance(value, dict):
        return {key: serialisable(item) for key, item in value.items()}
    return value


def source_hashes() -> dict[str, str]:
    names = (
        "RL_concentration_experiment.py",
        "RL_main.py",
        "agent.py",
        "policy.py",
        "env.py",
        "mechanics.py",
        "main.py",
    )
    return {
        name: hashlib.sha256(
            (BASE_DIRECTORY / name).read_bytes()
        ).hexdigest()
        for name in names
    }


def write_manifest() -> Path:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "tasks": list(TASKS),
        "max_episodes": MAX_EPISODES,
        "stop_rule": "first strict success, otherwise censor at 10000",
        "strict_success": (
            "flatness_ratio <= 1e-3 and spread_error <= 0.05"
        ),
        "crossing_policy": (
            "reject as zero-reward no-op; consume a step; continue episode"
        ),
        "concentration": "L2(mean) / sqrt(sum(per-mode variance))",
        "seed_selection_rule": (
            "earliest first strict success in the completed five-seed "
            "screen; pre-registered seed 9103 breaks the five_d3 tie"
        ),
        "seeds": SEEDS,
        "resolved_configs": {
            task_name: serialisable(experiment_config(task_name))
            for task_name in TASKS
        },
        "source_sha256": source_hashes(),
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
    subparsers.add_parser("run-all")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", choices=TASKS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        write_manifest()
    elif args.command == "merge":
        merge()
    elif args.command == "run-all":
        write_manifest()
        for task_name in TASKS:
            run_task(task_name)
        merge()
    elif args.command == "run":
        run_task(args.task)


if __name__ == "__main__":
    main()
