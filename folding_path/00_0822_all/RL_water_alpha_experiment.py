"""Water-task alpha=0.05 tuning and five-seed retention experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import task_config
from RL_main import (
    moving_average,
    run_rl,
    strict_success_retention_statistics,
)


EXPERIMENT_NAME = "20260831_water_alpha0p05_tuning_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
TASKS = ("water_d1", "water_sqrt2")
TUNING_SEEDS = (7101, 7102, 7103)
EVALUATION_SEEDS = tuple(range(8101, 8106))
TUNING_EPISODES = 1_500
FULL_EPISODES = 10_000
SPECTRAL_ALPHA = 0.05

# Reward settings are identical between tasks. Task geometry, target D,
# maximum step count, and maximum mechanical step width remain task settings.
COMMON_REWARD_OVERRIDES = {
    "spectral_weight": SPECTRAL_ALPHA,
    "near_miss_spread_error_far": 1.0,
}

CANDIDATES = {
    "A_base": {
        "batch_size": 8,
        "actor_learning_rate": 5.0e-4,
        "critic_learning_rate": 1.5e-4,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
    },
    "B_low_batch4": {
        "batch_size": 4,
        "actor_learning_rate": 2.5e-4,
        "critic_learning_rate": 7.5e-5,
        "post_success_batch_size": 32,
        "post_success_actor_lr_factor": 0.25,
    },
    "C_conservative": {
        "batch_size": 16,
        "actor_learning_rate": 2.5e-4,
        "critic_learning_rate": 7.5e-5,
        "post_success_batch_size": 32,
        "post_success_actor_lr_factor": 0.25,
    },
}

TUNE_FIELDS = (
    "task", "candidate", "seed", "success", "censored_at_max_episodes",
    "episodes_completed", "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success", "optimizer_updates_at_stop",
    "mechanical_steps_at_stop", "max_episodes", "max_steps_per_episode",
    "max_step_width", "spectral_alpha", "near_miss_spread_error_far",
    "actor_reference_horizon", "batch_size", "actor_learning_rate",
    "critic_learning_rate", "post_success_batch_size",
    "post_success_actor_lr_factor",
)

FULL_FIELDS = (
    "task", "candidate", "seed", "spectral_alpha", "episodes_completed",
    "first_strict_success_episode", "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success", "optimizer_updates_completed",
    "cumulative_mechanical_steps", "adaptation_episode",
    "initial_batch_size", "final_batch_size",
    "initial_actor_learning_rate", "final_actor_learning_rate",
    "last_200_strict_success_rate", "last_1000_strict_success_rate",
    "rolling_200_confirmation_episode_90pct",
    "minimum_rolling_200_after_confirmation",
    "dropped_below_80pct_after_confirmation",
    "last_1000_minimum_rolling_200_strict_rate", "final_window_stable",
    "retention_confirmed", "maximum_consecutive_failures",
    "maximum_consecutive_failures_after_first_success",
    "maximum_consecutive_failures_after_confirmation", "best_reward",
    "output_directory",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def experiment_config(
    task_name: str, candidate: str, seed: int, episodes: int
) -> dict:
    config = task_config(task_name, episodes)
    config.update(COMMON_REWARD_OVERRIDES)
    config.update(CANDIDATES[candidate])
    config.update({
        "seed": int(seed),
        "actor_reference_horizon": int(config["max_steps"]),
        "adaptive_after_first_success": True,
    })
    return config


def tune(task_name: str, candidate: str, seed: int, episodes: int) -> dict:
    config = experiment_config(task_name, candidate, seed, episodes)
    config.update({
        "save_outputs": False,
        "save_diagnostics": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
    })
    result = run_rl(config)
    success = result["first_strict_success_episode"] is not None
    row = {
        "task": task_name,
        "candidate": candidate,
        "seed": seed,
        "success": int(success),
        "censored_at_max_episodes": int(not success),
        "episodes_completed": int(result["episodes"]),
        "first_strict_success_episode": result["first_strict_success_episode"],
        "first_strict_success_episode_steps": result[
            "first_strict_success_episode_steps"
        ],
        "first_strict_success_mechanical_steps": result[
            "first_strict_success_mechanical_steps"
        ],
        "optimizer_updates_before_first_success": result[
            "optimizer_updates_before_first_success"
        ],
        "optimizer_updates_at_stop": int(result["optimizer_updates_completed"]),
        "mechanical_steps_at_stop": int(result["cumulative_mechanical_steps"]),
        "max_episodes": episodes,
        "max_steps_per_episode": int(config["max_steps"]),
        "max_step_width": float(config["max_step_width"]),
        "spectral_alpha": SPECTRAL_ALPHA,
        "near_miss_spread_error_far": float(
            config["near_miss_spread_error_far"]
        ),
        "actor_reference_horizon": int(config["actor_reference_horizon"]),
        "batch_size": int(config["batch_size"]),
        "actor_learning_rate": float(config["actor_learning_rate"]),
        "critic_learning_rate": float(config["critic_learning_rate"]),
        "post_success_batch_size": int(config["post_success_batch_size"]),
        "post_success_actor_lr_factor": float(
            config["post_success_actor_lr_factor"]
        ),
    }
    path = (
        EXPERIMENT_DIRECTORY / "tuning" / "workers"
        / f"{task_name}_{candidate}_seed{seed}.csv"
    )
    write_csv(path, TUNE_FIELDS, [row])
    print(row, flush=True)
    return row


def merge_tuning() -> Path:
    rows = []
    worker_directory = EXPERIMENT_DIRECTORY / "tuning" / "workers"
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    expected = {
        (task, candidate, seed)
        for task in TASKS for candidate in CANDIDATES for seed in TUNING_SEEDS
    }
    observed = {
        (row["task"], row["candidate"], int(row["seed"])) for row in rows
    }
    errors = []
    if observed != expected or len(rows) != len(expected):
        errors.append(
            f"missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}, rows={len(rows)}"
        )
    for row in rows:
        identity = (row["task"], row["candidate"], row["seed"])
        if int(row["max_episodes"]) != TUNING_EPISODES:
            errors.append(f"{identity}: stale horizon {row['max_episodes']}")
        if not np.isclose(float(row["spectral_alpha"]), SPECTRAL_ALPHA):
            errors.append(f"{identity}: wrong alpha")
        if not np.isclose(float(row["near_miss_spread_error_far"]), 1.0):
            errors.append(f"{identity}: reward settings are not common")
        expected_candidate = CANDIDATES[row["candidate"]]
        for key, expected_value in expected_candidate.items():
            observed_value = float(row[key])
            if not np.isclose(observed_value, float(expected_value)):
                errors.append(f"{identity}: wrong {key}={observed_value}")
        if int(row["max_steps_per_episode"]) != 400:
            errors.append(f"{identity}: wrong max steps")
        if not np.isclose(float(row["max_step_width"]), 0.3):
            errors.append(f"{identity}: wrong max step width")
        if int(row["actor_reference_horizon"]) != 400:
            errors.append(f"{identity}: wrong actor reference horizon")
        success = int(row["success"])
        if success:
            first = int(row["first_strict_success_episode"])
            if int(row["episodes_completed"]) != first:
                errors.append(f"{identity}: successful stop mismatch")
        elif int(row["episodes_completed"]) != TUNING_EPISODES:
            errors.append(f"{identity}: censored stop mismatch")
    if errors:
        raise RuntimeError("invalid tuning outputs:\n" + "\n".join(errors))

    order = {name: index for index, name in enumerate(CANDIDATES)}
    summaries = []
    selection = {}
    for task in TASKS:
        task_summaries = []
        for candidate in CANDIDATES:
            group = [
                row for row in rows
                if row["task"] == task and row["candidate"] == candidate
            ]
            summary = {
                "task": task,
                "candidate": candidate,
                "seeds": len(group),
                "successes": sum(int(row["success"]) for row in group),
                "restricted_mean_episode": float(np.mean([
                    int(row["episodes_completed"]) for row in group
                ])),
                "restricted_mean_mechanical_steps": float(np.mean([
                    int(row["mechanical_steps_at_stop"]) for row in group
                ])),
            }
            summaries.append(summary)
            task_summaries.append(summary)
        chosen = min(task_summaries, key=lambda item: (
            -item["successes"], item["restricted_mean_mechanical_steps"],
            item["restricted_mean_episode"], order[item["candidate"]],
        ))
        selection[task] = {
            "candidate": chosen["candidate"],
            "pilot_successes": chosen["successes"],
            "pilot_seeds": len(TUNING_SEEDS),
            "promising_pilot": bool(chosen["successes"] >= 2),
            "selection_rule": (
                "max successes, then min restricted mean mechanical steps, "
                "then min restricted mean episodes, then preregistered order"
            ),
        }
    rows.sort(key=lambda row: (
        TASKS.index(row["task"]), order[row["candidate"]], int(row["seed"])
    ))
    output = EXPERIMENT_DIRECTORY / "tuning" / "tuning_results.csv"
    write_csv(output, TUNE_FIELDS, rows)
    summary_path = EXPERIMENT_DIRECTORY / "tuning" / "candidate_summary.csv"
    write_csv(summary_path, tuple(summaries[0]), summaries)
    selection_path = EXPERIMENT_DIRECTORY / "tuning" / "selection.json"
    with selection_path.open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2, ensure_ascii=False)
    print(json.dumps(selection, ensure_ascii=False), flush=True)
    return output


def selected_candidate(task_name: str) -> str:
    path = EXPERIMENT_DIRECTORY / "tuning" / "selection.json"
    with path.open(encoding="utf-8") as file:
        selection = json.load(file)[task_name]
    if not selection["promising_pilot"]:
        raise RuntimeError(f"{task_name}: pilot is not promising")
    return selection["candidate"]


def consecutive_failures(values: list[int]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def full(
    task_name: str, candidate: str, seed: int, episodes: int, smoke: bool
) -> dict:
    config = experiment_config(task_name, candidate, seed, episodes)
    root = "smoke" if smoke else "full"
    output_directory = (
        EXPERIMENT_DIRECTORY / root / task_name / candidate / f"seed{seed}"
    )
    config.update({
        "save_outputs": True,
        "save_diagnostics": True,
        "stop_on_first_strict_success": False,
        "print_progress": not smoke,
        "print_result": False,
        "rl_output_directory": output_directory,
    })
    result = run_rl(config)
    metrics_path = output_directory / "episode_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as file:
        episode_rows = list(csv.DictReader(file))
    successes = [int(row["strict_success"]) for row in episode_rows]
    first = result["first_strict_success_episode"]
    confirmation = result["rolling_200_confirmation_episode_90pct"]
    row = {
        "task": task_name,
        "candidate": candidate,
        "seed": seed,
        "spectral_alpha": SPECTRAL_ALPHA,
        "episodes_completed": int(result["episodes"]),
        "first_strict_success_episode": first,
        "first_strict_success_episode_steps": result[
            "first_strict_success_episode_steps"
        ],
        "first_strict_success_mechanical_steps": result[
            "first_strict_success_mechanical_steps"
        ],
        "optimizer_updates_before_first_success": result[
            "optimizer_updates_before_first_success"
        ],
        "optimizer_updates_completed": int(result["optimizer_updates_completed"]),
        "cumulative_mechanical_steps": int(result["cumulative_mechanical_steps"]),
        "adaptation_episode": result["adaptation_episode"],
        "initial_batch_size": int(result["initial_batch_size"]),
        "final_batch_size": int(result["final_batch_size"]),
        "initial_actor_learning_rate": float(
            result["initial_actor_learning_rate"]
        ),
        "final_actor_learning_rate": float(result["final_actor_learning_rate"]),
        "last_200_strict_success_rate": float(
            result["last_200_strict_success_rate"]
        ),
        "last_1000_strict_success_rate": float(
            result["last_1000_strict_success_rate"]
        ),
        "rolling_200_confirmation_episode_90pct": confirmation,
        "minimum_rolling_200_after_confirmation": result[
            "minimum_rolling_200_after_confirmation"
        ],
        "dropped_below_80pct_after_confirmation": int(
            result["dropped_below_80pct_after_confirmation"]
        ),
        "last_1000_minimum_rolling_200_strict_rate": result[
            "last_1000_minimum_rolling_200_strict_rate"
        ],
        "final_window_stable": int(result["final_window_stable"]),
        "retention_confirmed": int(result["retention_confirmed"]),
        "maximum_consecutive_failures": int(
            result["maximum_consecutive_failures"]
        ),
        "maximum_consecutive_failures_after_first_success": (
            "" if first is None else consecutive_failures(successes[first - 1:])
        ),
        "maximum_consecutive_failures_after_confirmation": (
            "" if confirmation is None else consecutive_failures(
                successes[confirmation - 1:]
            )
        ),
        "best_reward": float(result["best_reward"]),
        "output_directory": str(output_directory),
    }
    worker = (
        EXPERIMENT_DIRECTORY / root / "workers"
        / f"{task_name}_{candidate}_seed{seed}.csv"
    )
    write_csv(worker, FULL_FIELDS, [row])
    print(row, flush=True)
    return row


def read_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return (
        np.asarray([float(row["reward"]) for row in rows]),
        np.asarray([float(row["terminal_flatness"]) for row in rows]),
    )


def combined_curve(task_name: str, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(len(rows), 2, figsize=(13, 3.1 * len(rows)))
    for index, row in enumerate(rows):
        reward, flatness = read_metrics(
            Path(row["output_directory"]) / "episode_metrics.csv"
        )
        for column, (values, label) in enumerate((
            (reward, "Episode reward"),
            (flatness, "Terminal flatness F"),
        )):
            axis = axes[index, column]
            x = np.arange(1, len(values) + 1)
            axis.plot(x, values, color="0.78", linewidth=0.45, label="raw")
            average_x, average = moving_average(values.tolist(), 200)
            axis.plot(average_x, average, color="black", linewidth=1.5,
                      label="moving average (200)")
            first = row["first_strict_success_episode"]
            if first not in ("", None, "None"):
                axis.axvline(int(first), color="#d62728", linestyle="--",
                             linewidth=1.0, label="first strict success")
            axis.set_title(f"seed {row['seed']}")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if column == 1:
                axis.set_yscale("log")
            if index == len(rows) - 1:
                axis.set_xlabel("Episode")
            if index == 0:
                axis.legend(fontsize=8)
    figure.suptitle(f"{task_name}, spectral alpha=0.05", y=0.998)
    figure.tight_layout()
    output = EXPERIMENT_DIRECTORY / "full" / task_name / "learning_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def merge_full() -> Path:
    rows = []
    worker_directory = EXPERIMENT_DIRECTORY / "full" / "workers"
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    expected_candidates = {task: selected_candidate(task) for task in TASKS}
    expected = {
        (task, expected_candidates[task], seed)
        for task in TASKS for seed in EVALUATION_SEEDS
    }
    observed = {
        (row["task"], row["candidate"], int(row["seed"])) for row in rows
    }
    errors = []
    if observed != expected or len(rows) != len(expected):
        errors.append(
            f"missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}, rows={len(rows)}"
        )
    for row in rows:
        identity = (row["task"], row["candidate"], row["seed"])
        if int(row["episodes_completed"]) != FULL_EPISODES:
            errors.append(f"{identity}: incomplete episodes")
            continue
        output = Path(row["output_directory"])
        expected_output = (
            EXPERIMENT_DIRECTORY / "full" / row["task"] / row["candidate"]
            / f"seed{row['seed']}"
        )
        if output.resolve() != expected_output.resolve():
            errors.append(f"{identity}: unexpected output directory")
            continue
        required = (
            output / "episode_metrics.csv", output / "update_metrics.csv",
            output / "summary.json", output / "learning_curve.png",
        )
        if not all(path.exists() for path in required):
            errors.append(f"{identity}: missing diagnostics")
            continue
        with required[2].open(encoding="utf-8") as file:
            summary = json.load(file)
        candidate_config = CANDIDATES[row["candidate"]]
        summary_checks = {
            "task": row["task"],
            "seed": int(row["seed"]),
            "episodes": FULL_EPISODES,
            "episodes_requested": FULL_EPISODES,
            "actor_reference_horizon": 400,
            "initial_batch_size": candidate_config["batch_size"],
            "initial_actor_learning_rate": candidate_config[
                "actor_learning_rate"
            ],
            "post_success_actor_lr_factor": candidate_config[
                "post_success_actor_lr_factor"
            ],
        }
        for key, expected_value in summary_checks.items():
            observed_value = summary.get(key)
            if isinstance(expected_value, float):
                matches = np.isclose(float(observed_value), expected_value)
            else:
                matches = observed_value == expected_value
            if not matches:
                errors.append(
                    f"{identity}: summary {key}={observed_value}, "
                    f"expected {expected_value}"
                )
        with required[0].open(newline="", encoding="utf-8") as file:
            metrics = list(csv.DictReader(file))
        if len(metrics) != FULL_EPISODES:
            errors.append(f"{identity}: metrics rows={len(metrics)}")
        episodes = [int(item["episode"]) for item in metrics]
        if episodes != list(range(1, FULL_EPISODES + 1)):
            errors.append(f"{identity}: non-contiguous episodes")
        strict_values = [item["strict_success"] for item in metrics]
        if any(value not in ("0", "1") for value in strict_values):
            errors.append(f"{identity}: strict_success is not binary")
            continue
        successes = [int(value) for value in strict_values]
        first = next(
            (index for index, value in enumerate(successes, start=1) if value),
            None,
        )
        recorded_first = row["first_strict_success_episode"]
        recorded_first = (
            None if recorded_first in ("", "None", None)
            else int(recorded_first)
        )
        if recorded_first != first:
            errors.append(
                f"{identity}: first success {recorded_first} != {first}"
            )
        retention = strict_success_retention_statistics(successes)
        for key in (
            "last_200_strict_success_rate",
            "last_1000_strict_success_rate",
            "rolling_200_confirmation_episode_90pct",
            "minimum_rolling_200_after_confirmation",
            "dropped_below_80pct_after_confirmation",
            "last_1000_minimum_rolling_200_strict_rate",
            "final_window_stable",
            "retention_confirmed",
        ):
            value = retention[key]
            row[key] = "" if value is None else value
        row["dropped_below_80pct_after_confirmation"] = int(
            retention["dropped_below_80pct_after_confirmation"]
        )
        row["final_window_stable"] = int(retention["final_window_stable"])
        row["retention_confirmed"] = int(retention["retention_confirmed"])
        row["maximum_consecutive_failures"] = consecutive_failures(successes)
        row["maximum_consecutive_failures_after_first_success"] = (
            "" if first is None else consecutive_failures(successes[first - 1:])
        )
        confirmation = retention["rolling_200_confirmation_episode_90pct"]
        row["maximum_consecutive_failures_after_confirmation"] = (
            "" if confirmation is None
            else consecutive_failures(successes[confirmation - 1:])
        )
        values = [
            float(item[key]) for item in metrics
            for key in ("reward", "terminal_flatness")
        ]
        if not np.all(np.isfinite(values)):
            errors.append(f"{identity}: non-finite metrics")
    if errors:
        raise RuntimeError("invalid full outputs:\n" + "\n".join(errors))
    rows.sort(key=lambda row: (TASKS.index(row["task"]), int(row["seed"])))
    output = EXPERIMENT_DIRECTORY / "full" / "stability_summary.csv"
    write_csv(output, FULL_FIELDS, rows)
    task_summaries = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        discoveries = sum(
            row["first_strict_success_episode"] not in ("", "None", None)
            for row in task_rows
        )
        task_summaries.append({
            "task": task,
            "candidate": expected_candidates[task],
            "evaluation_seeds": len(task_rows),
            "strict_success_discoveries": discoveries,
            "evaluation_promising": int(discoveries >= 4),
            "rolling_90pct_confirmations": sum(
                row["rolling_200_confirmation_episode_90pct"]
                not in ("", "None", None) for row in task_rows
            ),
            "retention_confirmed_seeds": sum(
                int(row["retention_confirmed"]) for row in task_rows
            ),
            "collapsed_after_confirmation_seeds": sum(
                int(row["dropped_below_80pct_after_confirmation"])
                for row in task_rows
            ),
        })
    write_csv(
        EXPERIMENT_DIRECTORY / "full" / "task_summary.csv",
        tuple(task_summaries[0]),
        task_summaries,
    )
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        print(combined_curve(task, task_rows), flush=True)
    print(output, flush=True)
    return output


def manifest() -> Path:
    data = {
        "experiment": EXPERIMENT_NAME,
        "tasks": list(TASKS),
        "spectral_alpha": SPECTRAL_ALPHA,
        "common_reward_overrides": COMMON_REWARD_OVERRIDES,
        "tuning_seeds": list(TUNING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "tuning_episodes": TUNING_EPISODES,
        "full_episodes": FULL_EPISODES,
        "candidates": CANDIDATES,
        "crossing_policy": "zero-reward reject and continue",
        "selection_rule": (
            "per task: max pilot successes; min restricted mean mechanical "
            "steps; min restricted mean episodes; preregistered candidate order"
        ),
        "pilot_promising_rule": "at least 2/3 tuning seeds succeed",
        "evaluation_promising_rule": "at least 4/5 seeds find strict success",
        "retention_rule": {
            "confirmation": "rolling-200 strict success >= 0.90",
            "collapse": "rolling-200 < 0.80 after confirmation",
            "observation": "at least 1000 episodes after confirmation",
            "final_window": (
                "last-1000 success >=0.95 and all rolling-200 in it >=0.90"
            ),
            "retained": (
                "confirmation and observation and no collapse and final window"
            ),
        },
    }
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_DIRECTORY / "manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(path, flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("merge-tuning")
    sub.add_parser("merge-full")
    tune_parser = sub.add_parser("tune")
    tune_parser.add_argument("--task", choices=TASKS, required=True)
    tune_parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    tune_parser.add_argument("--seed", type=int, required=True)
    tune_parser.add_argument("--episodes", type=int, default=TUNING_EPISODES)
    full_parser = sub.add_parser("full")
    full_parser.add_argument("--task", choices=TASKS, required=True)
    full_parser.add_argument("--candidate", choices=tuple(CANDIDATES), default=None)
    full_parser.add_argument("--seed", type=int, required=True)
    full_parser.add_argument("--episodes", type=int, default=FULL_EPISODES)
    full_parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        manifest()
    elif args.command == "merge-tuning":
        merge_tuning()
    elif args.command == "merge-full":
        merge_full()
    elif args.command == "tune":
        if args.seed not in TUNING_SEEDS:
            raise ValueError(f"unassigned tuning seed {args.seed}")
        tune(args.task, args.candidate, args.seed, args.episodes)
    elif args.command == "full":
        if args.seed not in EVALUATION_SEEDS:
            raise ValueError(f"unassigned evaluation seed {args.seed}")
        candidate = args.candidate or selected_candidate(args.task)
        full(args.task, candidate, args.seed, args.episodes, args.smoke)


if __name__ == "__main__":
    main()
