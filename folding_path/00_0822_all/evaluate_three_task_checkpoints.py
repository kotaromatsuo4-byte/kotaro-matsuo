"""Evaluate three selected training runs at every frozen policy checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

from agent import ActorCriticAgent
from env import OrigamiEnv
from policy import VariableModeActorCritic
from RL_main import load_problem, rollout
from RL_three_task_checkpoint_experiment import (
    EXPECTED,
    EXPERIMENT_DIRECTORY,
    SEEDS,
    experiment_config,
)


TASKS = tuple(SEEDS)
ROLLOUTS_PER_CHECKPOINT = 100
EVALUATION_SEED_BASES = {
    "five_d2": 750_000,
    "five_d3": 751_000,
    "torus_d2": 752_000,
}
EVALUATION_DIRECTORY = EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
RAW_FIELDS = (
    "task",
    "training_seed",
    "update",
    "training_episode",
    "training_mechanical_steps",
    "replicate",
    "evaluation_seed",
    "reach_near_miss",
    "best_quality",
    "strict_success",
    "steps",
    "termination_reason",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_directory(task_name: str) -> Path:
    return Path(experiment_config(task_name)["rl_output_directory"])


def checkpoint_rows(task_name: str) -> list[dict]:
    rows = sorted(
        read_csv(run_directory(task_name) / "policy_checkpoint_index.csv"),
        key=lambda row: int(row["update"]),
    )
    expected_updates = list(range(int(EXPECTED[task_name]["updates"]) + 1))
    actual_updates = [int(row["update"]) for row in rows]
    if actual_updates != expected_updates:
        raise RuntimeError(
            f"{task_name}: expected checkpoints {expected_updates}, "
            f"got {actual_updates}"
        )
    return rows


def make_agent(
    config: dict,
) -> tuple[OrigamiEnv, VariableModeActorCritic, ActorCriticAgent]:
    node, member, fix = load_problem(Path(config["model_directory"]))
    env = OrigamiEnv(config, data=(node, member, fix))
    model = VariableModeActorCritic(
        coordinate_size=env.coordinate_size,
        hidden_size=int(config["hidden_size"]),
        variance_bias=float(config["variance_bias"]),
        width_mean_limit=float(config["width_mean_limit"]),
    )
    agent = ActorCriticAgent(
        model=model,
        max_step_width=float(config["max_step_width"]),
        gamma=float(config["gamma"]),
        actor_learning_rate=float(config["actor_learning_rate"]),
        critic_learning_rate=float(config["critic_learning_rate"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        value_coefficient=float(config["value_coefficient"]),
        actor_gradient_clip=float(config["actor_gradient_clip"]),
        critic_gradient_clip=float(config["critic_gradient_clip"]),
        actor_reference_horizon=config.get("actor_reference_horizon"),
    )
    return env, model, agent


def evaluate_one(
    task_name: str,
    env: OrigamiEnv,
    agent: ActorCriticAgent,
    *,
    checkpoint_row: dict,
    replicate: int,
    evaluation_seed: int,
) -> dict:
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    with torch.inference_mode():
        summary, _ = rollout(env, agent, training=False)

    quality_trace = []
    reach_near_miss = False
    for item in env.reward_trace:
        q_flatness, q_spread, quality = env.near_miss_quality(
            item["flatness_ratio"], item["spread_error"]
        )
        quality_trace.append(quality)
        reach_near_miss = reach_near_miss or bool(
            float(item["flatness_ratio"]) <= 0.1
            and float(item["spread_error"]) <= env.spread_tolerance
        )

    return {
        "task": task_name,
        "training_seed": int(SEEDS[task_name]),
        "update": int(checkpoint_row["update"]),
        "training_episode": int(checkpoint_row["training_episode"]),
        "training_mechanical_steps": int(
            checkpoint_row["cumulative_mechanical_steps"]
        ),
        "replicate": int(replicate),
        "evaluation_seed": int(evaluation_seed),
        "reach_near_miss": int(reach_near_miss),
        "best_quality": float(max(quality_trace)),
        "strict_success": int(summary["strict_success"]),
        "steps": int(summary["steps"]),
        "termination_reason": summary["termination_reason"],
    }


def write_design(worker_count: int) -> None:
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIRECTORY / "evaluation_design.csv"
    fields = (
        "task",
        "training_seed",
        "max_steps",
        "updates",
        "rollouts_per_checkpoint",
        "first_evaluation_seed",
        "last_evaluation_seed",
        "workers",
        "checkpoint_comparison_design",
        "near_miss_definition",
        "quality_definition",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task_name in TASKS:
            config = experiment_config(task_name)
            base = EVALUATION_SEED_BASES[task_name]
            writer.writerow({
                "task": task_name,
                "training_seed": SEEDS[task_name],
                "max_steps": config["max_steps"],
                "updates": EXPECTED[task_name]["updates"],
                "rollouts_per_checkpoint": ROLLOUTS_PER_CHECKPOINT,
                "first_evaluation_seed": base + 1,
                "last_evaluation_seed": base + ROLLOUTS_PER_CHECKPOINT,
                "workers": worker_count,
                "checkpoint_comparison_design": (
                    "within each task, the same 100 independent rollout "
                    "seeds are reused at every checkpoint for a paired "
                    "common-random-numbers comparison"
                ),
                "near_miss_definition": (
                    "exists t: flatness_ratio <= 0.1 and "
                    "spread_error <= 0.05; includes strict success"
                ),
                "quality_definition": "max_t qF(t) * qD(t)",
            })


def write_checkpoint_inventory() -> Path:
    output = EVALUATION_DIRECTORY / "checkpoint_inventory.csv"
    fields = (
        "task", "update", "training_episode",
        "training_mechanical_steps", "filename", "sha256",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task_name in TASKS:
            directory = run_directory(task_name) / "policy_checkpoints"
            for row in checkpoint_rows(task_name):
                path = directory / row["filename"]
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=True
                )
                expected_metadata = {
                    "update": int(row["update"]),
                    "training_episode": int(row["training_episode"]),
                    "cumulative_mechanical_steps": int(
                        row["cumulative_mechanical_steps"]
                    ),
                }
                actual_metadata = {
                    key: int(checkpoint[key]) for key in expected_metadata
                }
                if actual_metadata != expected_metadata:
                    raise RuntimeError(
                        f"checkpoint metadata mismatch: {path}"
                    )
                writer.writerow({
                    "task": task_name,
                    "update": row["update"],
                    "training_episode": row["training_episode"],
                    "training_mechanical_steps": (
                        row["cumulative_mechanical_steps"]
                    ),
                    "filename": row["filename"],
                    "sha256": sha256(path),
                })
    return output


def prepare(worker_count: int) -> None:
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    worker_directory = EVALUATION_DIRECTORY / "workers"
    worker_directory.mkdir(parents=True, exist_ok=True)
    for path in worker_directory.glob("worker*_of_*.csv"):
        path.unlink()
    for name in (
        "checkpoint_rollouts.csv",
        "checkpoint_summary.csv",
        "checkpoint_inventory.csv",
    ):
        (EVALUATION_DIRECTORY / name).unlink(missing_ok=True)
    write_design(worker_count)
    write_checkpoint_inventory()


def work_items() -> list[tuple[str, dict]]:
    return [
        (task_name, row)
        for task_name in TASKS
        for row in checkpoint_rows(task_name)
    ]


def run_worker(worker_index: int, worker_count: int) -> Path:
    torch.set_num_threads(1)
    assigned = [
        item for index, item in enumerate(work_items())
        if index % worker_count == worker_index
    ]
    agents = {}
    for task_name in TASKS:
        agents[task_name] = make_agent(experiment_config(task_name))

    output = (
        EVALUATION_DIRECTORY / "workers"
        / f"worker{worker_index:02d}_of_{worker_count:02d}.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for task_name, checkpoint_row in assigned:
            env, model, agent = agents[task_name]
            path = (
                run_directory(task_name) / "policy_checkpoints"
                / checkpoint_row["filename"]
            )
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            for replicate in range(1, ROLLOUTS_PER_CHECKPOINT + 1):
                evaluation_seed = (
                    EVALUATION_SEED_BASES[task_name] + replicate
                )
                writer.writerow(evaluate_one(
                    task_name,
                    env,
                    agent,
                    checkpoint_row=checkpoint_row,
                    replicate=replicate,
                    evaluation_seed=evaluation_seed,
                ))
            file.flush()
            print(
                f"worker {worker_index}/{worker_count}: {task_name} "
                f"update {checkpoint_row['update']} complete",
                flush=True,
            )
    print(output, flush=True)
    return output


def wilson_interval(successes: int, observations: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / observations
    denominator = 1.0 + z * z / observations
    centre = (
        proportion + z * z / (2.0 * observations)
    ) / denominator
    half_width = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / observations
        + z * z / (4.0 * observations * observations)
    )
    lower = 0.0 if successes == 0 else centre - half_width
    upper = 1.0 if successes == observations else centre + half_width
    return max(0.0, lower), min(1.0, upper)


def merge(worker_count: int) -> tuple[Path, Path]:
    rows = []
    for worker_index in range(worker_count):
        path = (
            EVALUATION_DIRECTORY / "workers"
            / f"worker{worker_index:02d}_of_{worker_count:02d}.csv"
        )
        rows.extend(read_csv(path))
    task_order = {task_name: index for index, task_name in enumerate(TASKS)}
    rows.sort(key=lambda row: (
        task_order[row["task"]],
        int(row["update"]),
        int(row["replicate"]),
    ))

    expected_keys = {
        (task_name, update, replicate)
        for task_name in TASKS
        for update in range(int(EXPECTED[task_name]["updates"]) + 1)
        for replicate in range(1, ROLLOUTS_PER_CHECKPOINT + 1)
    }
    observed_keys = {
        (row["task"], int(row["update"]), int(row["replicate"]))
        for row in rows
    }
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise RuntimeError("task/update/replicate grid has gaps or duplicates")
    for row in rows:
        expected_seed = (
            EVALUATION_SEED_BASES[row["task"]] + int(row["replicate"])
        )
        if (
            int(row["evaluation_seed"]) != expected_seed
            or int(row["training_seed"]) != SEEDS[row["task"]]
        ):
            raise RuntimeError("unexpected evaluation or training seed")

    raw_path = EVALUATION_DIRECTORY / "checkpoint_rollouts.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for task_name in TASKS:
        for update in range(int(EXPECTED[task_name]["updates"]) + 1):
            selected = [
                row for row in rows
                if row["task"] == task_name and int(row["update"]) == update
            ]
            near_count = sum(int(row["reach_near_miss"]) for row in selected)
            near_low, near_high = wilson_interval(
                near_count, len(selected)
            )
            qualities = np.asarray(
                [float(row["best_quality"]) for row in selected],
                dtype=float,
            )
            q25, median, q75 = np.quantile(qualities, (0.25, 0.5, 0.75))
            summary_rows.append({
                "task": task_name,
                "training_seed": SEEDS[task_name],
                "update": update,
                "training_episode": int(selected[0]["training_episode"]),
                "training_mechanical_steps": int(
                    selected[0]["training_mechanical_steps"]
                ),
                "rollouts": len(selected),
                "near_miss_count": near_count,
                "near_miss_rate": near_count / len(selected),
                "near_miss_ci_low": near_low,
                "near_miss_ci_high": near_high,
                "best_quality_q25": float(q25),
                "best_quality_median": float(median),
                "best_quality_q75": float(q75),
            })

    summary_path = EVALUATION_DIRECTORY / "checkpoint_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=tuple(summary_rows[0])
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    write_design(worker_count)
    print(raw_path)
    print(summary_path)
    return raw_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--workers", type=int, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.workers)
    elif args.command == "worker":
        run_worker(args.index, args.workers)
    elif args.command == "merge":
        merge(args.workers)


if __name__ == "__main__":
    main()
