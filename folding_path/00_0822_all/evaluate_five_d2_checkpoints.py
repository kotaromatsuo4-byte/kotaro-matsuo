"""Evaluate every five-D2 policy checkpoint with 100 frozen rollouts."""

from __future__ import annotations

import argparse
import csv
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
from RL_discovery_evidence_experiment import (
    EXPERIMENT_DIRECTORY,
    experiment_config,
)
from RL_main import load_problem, rollout


TASK_NAME = "five_d2"
ROLLOUTS_PER_CHECKPOINT = 100
EVALUATION_SEEDS = tuple(range(730_001, 730_101))
EVALUATION_DIRECTORY = EXPERIMENT_DIRECTORY / "checkpoint_evaluation"

RAW_FIELDS = (
    "update",
    "training_episode",
    "training_mechanical_steps",
    "replicate",
    "evaluation_seed",
    "strict_success",
    "reach_spread",
    "reach_f0p1",
    "reach_f0p01",
    "near_miss_f0p1_only",
    "best_quality",
    "best_q_flatness",
    "best_q_spread",
    "terminal_quality",
    "terminal_q_flatness",
    "terminal_q_spread",
    "terminal_flatness",
    "spread_error",
    "reward",
    "steps",
    "termination_reason",
    "rejected_crossings",
)


def checkpoint_index_path() -> Path:
    config = experiment_config(TASK_NAME)
    return Path(config["rl_output_directory"]) / "policy_checkpoint_index.csv"


def checkpoint_directory() -> Path:
    config = experiment_config(TASK_NAME)
    return Path(config["rl_output_directory"]) / "policy_checkpoints"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_agent(config: dict) -> tuple[OrigamiEnv, VariableModeActorCritic, ActorCriticAgent]:
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


def quality_trace(env: OrigamiEnv) -> list[tuple[float, float, float]]:
    output = []
    for item in env.reward_trace:
        q_flatness, q_spread, quality = env.near_miss_quality(
            item["flatness_ratio"], item["spread_error"]
        )
        output.append((q_flatness, q_spread, quality))
    return output


def evaluate_one(
    env: OrigamiEnv,
    model: VariableModeActorCritic,
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

    trace = env.reward_trace
    qualities = quality_trace(env)
    best_index = int(np.argmax([item[2] for item in qualities]))
    best_q_flatness, best_q_spread, best_quality = qualities[best_index]
    terminal_q_flatness, terminal_q_spread, terminal_quality = (
        env.near_miss_quality(
            summary["terminal_flatness"], summary["spread_error"]
        )
    )
    spread_tolerance = float(env.spread_tolerance)
    reach_spread = any(
        float(item["spread_error"]) <= spread_tolerance
        for item in trace
    )
    reach_f0p1 = any(
        float(item["spread_error"]) <= spread_tolerance
        and float(item["flatness_ratio"]) <= 0.1
        for item in trace
    )
    reach_f0p01 = any(
        float(item["spread_error"]) <= spread_tolerance
        and float(item["flatness_ratio"]) <= 0.01
        for item in trace
    )
    strict_success = bool(summary["strict_success"])
    return {
        "update": int(checkpoint_row["update"]),
        "training_episode": int(checkpoint_row["training_episode"]),
        "training_mechanical_steps": int(
            checkpoint_row["cumulative_mechanical_steps"]
        ),
        "replicate": int(replicate),
        "evaluation_seed": int(evaluation_seed),
        "strict_success": int(strict_success),
        "reach_spread": int(reach_spread),
        "reach_f0p1": int(reach_f0p1),
        "reach_f0p01": int(reach_f0p01),
        "near_miss_f0p1_only": int(reach_f0p1 and not strict_success),
        "best_quality": float(best_quality),
        "best_q_flatness": float(best_q_flatness),
        "best_q_spread": float(best_q_spread),
        "terminal_quality": float(terminal_quality),
        "terminal_q_flatness": float(terminal_q_flatness),
        "terminal_q_spread": float(terminal_q_spread),
        "terminal_flatness": float(summary["terminal_flatness"]),
        "spread_error": float(summary["spread_error"]),
        "reward": float(summary["total_reward"]),
        "steps": int(summary["steps"]),
        "termination_reason": summary["termination_reason"],
        "rejected_crossings": int(summary["rejected_crossings"]),
    }


def write_design(worker_count: int) -> None:
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metadata = EVALUATION_DIRECTORY / "evaluation_design.csv"
    with metadata.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "task", "training_seed", "rollouts_per_checkpoint",
            "first_evaluation_seed", "last_evaluation_seed", "workers",
            "checkpoint_comparison_design",
            "near_miss_definition", "close_miss_definition",
            "strict_success_definition",
        ))
        writer.writeheader()
        writer.writerow({
            "task": TASK_NAME,
            "training_seed": 9105,
            "rollouts_per_checkpoint": ROLLOUTS_PER_CHECKPOINT,
            "first_evaluation_seed": EVALUATION_SEEDS[0],
            "last_evaluation_seed": EVALUATION_SEEDS[-1],
            "workers": int(worker_count),
            "checkpoint_comparison_design": (
                "the same 100 independent rollout seeds are reused at every "
                "checkpoint for a common-random-numbers paired comparison"
            ),
            "near_miss_definition": (
                "exists t: flatness <= 0.1 and spread_error <= 0.05; "
                "includes strict success"
            ),
            "close_miss_definition": (
                "exists t: flatness <= 0.01 and spread_error <= 0.05; "
                "includes strict success"
            ),
            "strict_success_definition": (
                "flatness <= 0.001 and spread_error <= 0.05"
            ),
        })


def prepare(worker_count: int) -> None:
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    worker_directory = EVALUATION_DIRECTORY / "workers"
    worker_directory.mkdir(parents=True, exist_ok=True)
    for path in worker_directory.glob("worker*_of_*.csv"):
        path.unlink()
    for name in ("checkpoint_rollouts.csv", "checkpoint_summary.csv"):
        (EVALUATION_DIRECTORY / name).unlink(missing_ok=True)
    write_design(worker_count)


def run_worker(worker_index: int, worker_count: int) -> Path:
    torch.set_num_threads(1)
    checkpoints = sorted(
        read_csv(checkpoint_index_path()),
        key=lambda row: int(row["update"]),
    )
    expected_updates = list(range(101))
    actual_updates = [int(row["update"]) for row in checkpoints]
    if actual_updates != expected_updates:
        raise RuntimeError(
            f"expected checkpoints 0..100, got {actual_updates}"
        )
    assigned = [
        row for index, row in enumerate(checkpoints)
        if index % worker_count == worker_index
    ]

    config = experiment_config(TASK_NAME)
    env, model, agent = make_agent(config)
    output = (
        EVALUATION_DIRECTORY / "workers"
        / f"worker{worker_index:02d}_of_{worker_count:02d}.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for checkpoint_row in assigned:
            path = checkpoint_directory() / checkpoint_row["filename"]
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            for replicate, evaluation_seed in enumerate(EVALUATION_SEEDS, 1):
                writer.writerow(evaluate_one(
                    env,
                    model,
                    agent,
                    checkpoint_row=checkpoint_row,
                    replicate=replicate,
                    evaluation_seed=evaluation_seed,
                ))
            file.flush()
            print(
                f"worker {worker_index}/{worker_count}: "
                f"update {checkpoint_row['update']} complete",
                flush=True,
            )
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
    return centre - half_width, centre + half_width


def bootstrap_mean_interval(
    values: np.ndarray, *, seed: int
) -> tuple[float, float]:
    random = np.random.default_rng(seed)
    indices = random.integers(
        0, len(values), size=(2000, len(values))
    )
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def merge(worker_count: int) -> tuple[Path, Path]:
    rows = []
    for worker_index in range(worker_count):
        path = (
            EVALUATION_DIRECTORY / "workers"
            / f"worker{worker_index:02d}_of_{worker_count:02d}.csv"
        )
        rows.extend(read_csv(path))
    rows.sort(key=lambda row: (int(row["update"]), int(row["replicate"])))
    expected_count = 101 * ROLLOUTS_PER_CHECKPOINT
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} rows, got {len(rows)}")
    observed_keys = {
        (int(row["update"]), int(row["replicate"])) for row in rows
    }
    expected_keys = {
        (update, replicate)
        for update in range(101)
        for replicate in range(1, ROLLOUTS_PER_CHECKPOINT + 1)
    }
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise RuntimeError("checkpoint/replicate grid has gaps or duplicates")

    raw_path = EVALUATION_DIRECTORY / "checkpoint_rollouts.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    binary_metrics = (
        "reach_spread", "reach_f0p1", "reach_f0p01", "strict_success",
        "near_miss_f0p1_only",
    )
    for update in range(101):
        selected = [row for row in rows if int(row["update"]) == update]
        summary = {
            "update": update,
            "training_episode": int(selected[0]["training_episode"]),
            "training_mechanical_steps": int(
                selected[0]["training_mechanical_steps"]
            ),
            "rollouts": len(selected),
        }
        for metric in binary_metrics:
            count = sum(int(row[metric]) for row in selected)
            lower, upper = wilson_interval(count, len(selected))
            summary[f"{metric}_count"] = count
            summary[f"{metric}_rate"] = count / len(selected)
            summary[f"{metric}_ci_low"] = lower
            summary[f"{metric}_ci_high"] = upper
        for metric, seed_offset in (
            ("best_quality", 0), ("terminal_quality", 10_000)
        ):
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=float
            )
            lower, upper = bootstrap_mean_interval(
                values, seed=740_000 + seed_offset + update
            )
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"median_{metric}"] = float(np.median(values))
            summary[f"{metric}_ci_low"] = lower
            summary[f"{metric}_ci_high"] = upper
        summary_rows.append(summary)

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
        print(run_worker(args.index, args.workers))
    elif args.command == "merge":
        merge(args.workers)


if __name__ == "__main__":
    main()
