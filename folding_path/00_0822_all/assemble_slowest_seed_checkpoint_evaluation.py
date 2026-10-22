"""Assemble one consistent three-task evaluation from audited raw sources."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_three_task_checkpoints as evaluator
from RL_slowest_seed_checkpoint_experiment import (
    BASE_DIRECTORY,
    EXPECTED,
    EXPERIMENT_DIRECTORY,
    SEEDS,
    experiment_config,
)


TASKS = tuple(SEEDS)
ROLLOUTS_PER_CHECKPOINT = 100
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
SOURCE_PATHS = {
    "five_d2": (
        BASE_DIRECTORY / "result" / "experiments"
        / "20260902_slowest_discovery_evidence_v1"
        / "checkpoint_evaluation" / "checkpoint_rollouts.csv"
    ),
    "five_d3": (
        BASE_DIRECTORY / "result" / "experiments"
        / "20260902_three_task_checkpoint_evidence_v1"
        / "checkpoint_evaluation" / "checkpoint_rollouts.csv"
    ),
    "torus_d2": (
        EXPERIMENT_DIRECTORY / "torus_checkpoint_evaluation"
        / "checkpoint_rollouts.csv"
    ),
}
EVALUATION_SEED_BASES = {
    "five_d2": 730_000,
    "five_d3": 751_000,
    "torus_d2": 752_000,
}
RAW_FIELDS = evaluator.RAW_FIELDS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standard_row(task_name: str, row: dict) -> dict:
    if task_name == "five_d2":
        near_value = row["reach_f0p1"]
    else:
        near_value = row["reach_near_miss"]
    return {
        "task": task_name,
        "training_seed": SEEDS[task_name],
        "update": int(row["update"]),
        "training_episode": int(row["training_episode"]),
        "training_mechanical_steps": int(row["training_mechanical_steps"]),
        "replicate": int(row["replicate"]),
        "evaluation_seed": int(row["evaluation_seed"]),
        "reach_near_miss": int(near_value),
        "best_quality": float(row["best_quality"]),
        "strict_success": int(row["strict_success"]),
        "steps": int(row["steps"]),
        "termination_reason": row["termination_reason"],
    }


def source_rows() -> list[dict]:
    rows = []
    for task_name in TASKS:
        selected = read_csv(SOURCE_PATHS[task_name])
        if task_name == "five_d3":
            selected = [row for row in selected if row["task"] == task_name]
        if task_name == "torus_d2":
            selected = [row for row in selected if row["task"] == task_name]
        for row in selected:
            if row.get("task") and row["task"] != task_name:
                raise RuntimeError(f"unexpected source task for {task_name}")
            if (
                row.get("training_seed")
                and int(row["training_seed"]) != int(SEEDS[task_name])
            ):
                raise RuntimeError(
                    f"unexpected source training seed for {task_name}"
                )
        rows.extend(standard_row(task_name, row) for row in selected)
    task_order = {task_name: index for index, task_name in enumerate(TASKS)}
    rows.sort(key=lambda row: (
        task_order[row["task"]], row["update"], row["replicate"]
    ))
    return rows


def validate_rows(rows: list[dict]) -> None:
    checkpoint_metadata = {}
    for task_name in TASKS:
        index_path = (
            Path(experiment_config(task_name)["rl_output_directory"])
            / "policy_checkpoint_index.csv"
        )
        checkpoint_metadata[task_name] = {
            int(row["update"]): (
                int(row["training_episode"]),
                int(row["cumulative_mechanical_steps"]),
            )
            for row in read_csv(index_path)
        }
    expected_keys = {
        (task_name, update, replicate)
        for task_name in TASKS
        for update in range(int(EXPECTED[task_name]["updates"]) + 1)
        for replicate in range(1, ROLLOUTS_PER_CHECKPOINT + 1)
    }
    observed_keys = {
        (row["task"], row["update"], row["replicate"])
        for row in rows
    }
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise RuntimeError("assembled task/update/replicate grid is invalid")
    for row in rows:
        expected_seed = (
            EVALUATION_SEED_BASES[row["task"]] + row["replicate"]
        )
        if row["evaluation_seed"] != expected_seed:
            raise RuntimeError("unexpected evaluation seed")
        if row["training_seed"] != SEEDS[row["task"]]:
            raise RuntimeError("unexpected training seed")
        expected_episode, expected_steps = checkpoint_metadata[
            row["task"]
        ][row["update"]]
        if (
            row["training_episode"] != expected_episode
            or row["training_mechanical_steps"] != expected_steps
        ):
            raise RuntimeError("source/checkpoint metadata mismatch")
        if row["reach_near_miss"] not in (0, 1):
            raise RuntimeError("near-miss indicator is not binary")
        if not np.isfinite(row["best_quality"]):
            raise RuntimeError("non-finite quality")
        if not 0.0 <= row["best_quality"] <= 1.0:
            raise RuntimeError("quality outside [0, 1]")


def write_raw(rows: list[dict]) -> Path:
    path = OUTPUT_DIRECTORY / "checkpoint_rollouts.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(rows: list[dict]) -> Path:
    summary_rows = []
    for task_name in TASKS:
        for update in range(int(EXPECTED[task_name]["updates"]) + 1):
            selected = [
                row for row in rows
                if row["task"] == task_name and row["update"] == update
            ]
            near_count = sum(row["reach_near_miss"] for row in selected)
            near_low, near_high = evaluator.wilson_interval(
                near_count, len(selected)
            )
            qualities = np.asarray(
                [row["best_quality"] for row in selected], dtype=float
            )
            q25, median, q75 = np.quantile(qualities, (0.25, 0.5, 0.75))
            summary_rows.append({
                "task": task_name,
                "training_seed": SEEDS[task_name],
                "update": update,
                "training_episode": selected[0]["training_episode"],
                "training_mechanical_steps": selected[0][
                    "training_mechanical_steps"
                ],
                "rollouts": len(selected),
                "near_miss_count": near_count,
                "near_miss_rate": near_count / len(selected),
                "near_miss_ci_low": near_low,
                "near_miss_ci_high": near_high,
                "best_quality_q25": float(q25),
                "best_quality_median": float(median),
                "best_quality_q75": float(q75),
            })
    path = OUTPUT_DIRECTORY / "checkpoint_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return path


def write_design() -> Path:
    fields = (
        "task", "training_seed", "max_steps", "updates",
        "rollouts_per_checkpoint", "first_evaluation_seed",
        "last_evaluation_seed", "data_source",
        "checkpoint_comparison_design", "near_miss_definition",
        "quality_definition",
    )
    path = OUTPUT_DIRECTORY / "evaluation_design.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task_name in TASKS:
            base = EVALUATION_SEED_BASES[task_name]
            writer.writerow({
                "task": task_name,
                "training_seed": SEEDS[task_name],
                "max_steps": experiment_config(task_name)["max_steps"],
                "updates": EXPECTED[task_name]["updates"],
                "rollouts_per_checkpoint": ROLLOUTS_PER_CHECKPOINT,
                "first_evaluation_seed": base + 1,
                "last_evaluation_seed": base + ROLLOUTS_PER_CHECKPOINT,
                "data_source": str(SOURCE_PATHS[task_name]),
                "checkpoint_comparison_design": (
                    "within each task, the same 100 independent rollout "
                    "seeds are reused at every checkpoint"
                ),
                "near_miss_definition": (
                    "exists t: flatness_ratio <= 0.1 and "
                    "spread_error <= 0.05; includes strict success"
                ),
                "quality_definition": "max_t qF(t) * qD(t)",
            })
    return path


def write_inventory() -> Path:
    evaluator.EXPECTED = EXPECTED
    evaluator.EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY
    evaluator.SEEDS = SEEDS
    evaluator.TASKS = TASKS
    evaluator.experiment_config = experiment_config
    evaluator.EVALUATION_DIRECTORY = OUTPUT_DIRECTORY
    return evaluator.write_checkpoint_inventory()


def write_sources() -> Path:
    payload = {
        task_name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for task_name, path in SOURCE_PATHS.items()
    }
    path = OUTPUT_DIRECTORY / "assembly_sources.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = source_rows()
    validate_rows(rows)
    outputs = (
        write_raw(rows),
        write_summary(rows),
        write_design(),
        write_inventory(),
        write_sources(),
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
