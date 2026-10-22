"""Reward ablation and torus retention experiment.

This runner performs two preregistered checks without changing the shared task
presets in :mod:`main`:

* alpha=0 first-discovery screens for torus_d2 and water_d1;
* 10,000-episode torus_d2 alpha=0.05 runs for four previously successful
  seeds, including post-discovery batch/LR adaptation and learning curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Set thread limits before NumPy/PyTorch are imported. Each experiment worker
# is a separate process, so this keeps the 14-way overnight run predictable.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import task_config
from RL_main import moving_average, run_rl


EXPERIMENT_NAME = "20260831_reward_ablation_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)

ALPHA_ZERO_TASKS = ("torus_d2", "water_d1")
ALPHA_ZERO_SEEDS = tuple(range(8101, 8106))
ALPHA_ZERO_MAX_EPISODES = 10_000

STABILITY_TASK = "torus_d2"
STABILITY_ALPHA = 0.05
STABILITY_SEEDS = (8101, 8102, 8103, 8105)
STABILITY_EPISODES = 10_000

SCREEN_FIELDS = (
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

STABILITY_FIELDS = (
    "task",
    "spectral_alpha",
    "seed",
    "episodes_completed",
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
    "optimizer_updates_completed",
    "cumulative_mechanical_steps",
    "adaptation_episode",
    "initial_batch_size",
    "final_batch_size",
    "initial_actor_learning_rate",
    "final_actor_learning_rate",
    "last_200_strict_success_rate",
    "last_1000_strict_success_rate",
    "rolling_200_confirmation_episode_90pct",
    "minimum_rolling_200_after_confirmation",
    "dropped_below_80pct_after_confirmation",
    "confirmation_observed_for_1000_episodes",
    "last_1000_minimum_rolling_200_strict_rate",
    "final_window_stable",
    "retention_confirmed",
    "maximum_consecutive_failures",
    "best_reward",
    "output_directory",
)

STABILITY_AGGREGATE_FIELDS = STABILITY_FIELDS + (
    "maximum_consecutive_failures_all_training",
    "maximum_consecutive_failures_after_first_success",
    "maximum_consecutive_failures_after_confirmation",
)


def experiment_config(
    task_name: str,
    episodes: int,
    seed: int,
    spectral_alpha: float,
) -> dict:
    """Return the current fixed-horizon/adaptive configuration."""

    config = task_config(task_name, episodes)
    config.update({
        "seed": int(seed),
        "spectral_weight": float(spectral_alpha),
        "actor_reference_horizon": int(config["max_steps"]),
        "adaptive_after_first_success": True,
        "post_success_batch_size": 16,
        "post_success_actor_lr_factor": 0.5,
    })
    return config


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def alpha_zero_screen(
    task_name: str,
    seed: int,
    max_episodes: int,
) -> dict:
    config = experiment_config(task_name, max_episodes, seed, 0.0)
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
        "spectral_alpha": 0.0,
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
        "optimizer_updates_at_stop": int(
            result["optimizer_updates_completed"]
        ),
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
        / "alpha_zero_screen"
        / "workers"
        / f"{task_name}_seed{seed}.csv"
    )
    write_csv(output, SCREEN_FIELDS, [row])
    print(row, flush=True)
    return row


def merge_alpha_zero_screen() -> Path:
    worker_directory = (
        EXPERIMENT_DIRECTORY / "alpha_zero_screen" / "workers"
    )
    rows = []
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    expected = {
        (task_name, seed)
        for task_name in ALPHA_ZERO_TASKS
        for seed in ALPHA_ZERO_SEEDS
    }
    observed = {(row["task"], int(row["seed"])) for row in rows}
    bad_horizons = [
        (row["task"], int(row["seed"]), int(row["max_episodes"]))
        for row in rows
        if int(row["max_episodes"]) != ALPHA_ZERO_MAX_EPISODES
    ]
    if observed != expected or len(rows) != len(expected) or bad_horizons:
        raise RuntimeError(
            "alpha-zero screen is incomplete or contains smoke/stale rows: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}, "
            f"bad_horizons={bad_horizons}, row_count={len(rows)}"
        )
    validation_errors = []
    for row in rows:
        task_name = row["task"]
        config = task_config(task_name, ALPHA_ZERO_MAX_EPISODES)
        identity = f"{task_name}/seed{row['seed']}"
        expected_settings = {
            "spectral_alpha": 0.0,
            "max_steps_per_episode": int(config["max_steps"]),
            "max_step_width": float(config["max_step_width"]),
            "actor_reference_horizon": int(config["max_steps"]),
            "batch_size": int(config["batch_size"]),
            "actor_learning_rate": float(config["actor_learning_rate"]),
            "critic_learning_rate": float(config["critic_learning_rate"]),
        }
        for key, expected_value in expected_settings.items():
            actual = float(row[key])
            if not np.isclose(actual, float(expected_value), rtol=0, atol=1e-15):
                validation_errors.append(
                    f"{identity}: {key}={actual}, expected={expected_value}"
                )
        success = int(row["success"])
        censored = int(row["censored_at_max_episodes"])
        episodes = int(row["episodes_completed"])
        updates = int(row["optimizer_updates_at_stop"])
        if success not in (0, 1) or censored != 1 - success:
            validation_errors.append(
                f"{identity}: inconsistent success/censor flags"
            )
        if success:
            first = int(row["first_strict_success_episode"])
            updates_before = int(
                row["optimizer_updates_before_first_success"]
            )
            expected_updates = (first - 1) // int(config["batch_size"])
            if episodes != first or not 1 <= first <= ALPHA_ZERO_MAX_EPISODES:
                validation_errors.append(
                    f"{identity}: invalid successful stop episode"
                )
            if updates != updates_before or updates != expected_updates:
                validation_errors.append(
                    f"{identity}: invalid success update count {updates}"
                )
            if int(row["mechanical_steps_at_stop"]) != int(
                row["first_strict_success_mechanical_steps"]
            ):
                validation_errors.append(
                    f"{identity}: success mechanical-step totals disagree"
                )
        else:
            if episodes != ALPHA_ZERO_MAX_EPISODES:
                validation_errors.append(
                    f"{identity}: censored run ended at {episodes}"
                )
            expected_updates = (
                ALPHA_ZERO_MAX_EPISODES // int(config["batch_size"])
            )
            if updates != expected_updates:
                validation_errors.append(
                    f"{identity}: censored update count {updates}, "
                    f"expected {expected_updates}"
                )
            for key in (
                "first_strict_success_episode",
                "first_strict_success_episode_steps",
                "first_strict_success_mechanical_steps",
                "optimizer_updates_before_first_success",
            ):
                if row[key] not in ("", "None"):
                    validation_errors.append(
                        f"{identity}: censored row has {key}={row[key]}"
                    )
    if validation_errors:
        raise RuntimeError(
            "invalid alpha-zero worker rows:\n" +
            "\n".join(validation_errors)
        )
    task_order = {
        task_name: index for index, task_name in enumerate(ALPHA_ZERO_TASKS)
    }
    rows.sort(key=lambda row: (
        task_order[row["task"]], int(row["seed"])
    ))
    output = (
        EXPERIMENT_DIRECTORY
        / "alpha_zero_screen"
        / "first_success_summary.csv"
    )
    write_csv(output, SCREEN_FIELDS, rows)
    print(f"merged {len(rows)} rows into {output}", flush=True)
    return output


def stability(seed: int, episodes: int) -> dict:
    config = experiment_config(
        STABILITY_TASK, episodes, seed, STABILITY_ALPHA
    )
    output_directory = (
        EXPERIMENT_DIRECTORY
        / "torus_alpha_0p05_stability"
        / f"seed{seed}"
    )
    config.update({
        "save_outputs": True,
        "save_diagnostics": True,
        "stop_on_first_strict_success": False,
        "print_progress": True,
        "print_result": False,
        "rl_output_directory": output_directory,
    })
    result = run_rl(config)
    row = {
        "task": STABILITY_TASK,
        "spectral_alpha": STABILITY_ALPHA,
        "seed": int(seed),
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
        "optimizer_updates_completed": int(
            result["optimizer_updates_completed"]
        ),
        "cumulative_mechanical_steps": int(
            result["cumulative_mechanical_steps"]
        ),
        "adaptation_episode": result["adaptation_episode"],
        "initial_batch_size": int(result["initial_batch_size"]),
        "final_batch_size": int(result["final_batch_size"]),
        "initial_actor_learning_rate": float(
            result["initial_actor_learning_rate"]
        ),
        "final_actor_learning_rate": float(
            result["final_actor_learning_rate"]
        ),
        "last_200_strict_success_rate": float(
            result["last_200_strict_success_rate"]
        ),
        "last_1000_strict_success_rate": float(
            result["last_1000_strict_success_rate"]
        ),
        "rolling_200_confirmation_episode_90pct": result[
            "rolling_200_confirmation_episode_90pct"
        ],
        "minimum_rolling_200_after_confirmation": result[
            "minimum_rolling_200_after_confirmation"
        ],
        "dropped_below_80pct_after_confirmation": int(
            result["dropped_below_80pct_after_confirmation"]
        ),
        "confirmation_observed_for_1000_episodes": int(
            result["confirmation_observed_for_1000_episodes"]
        ),
        "last_1000_minimum_rolling_200_strict_rate": result[
            "last_1000_minimum_rolling_200_strict_rate"
        ],
        "final_window_stable": int(result["final_window_stable"]),
        "retention_confirmed": int(result["retention_confirmed"]),
        "maximum_consecutive_failures": int(
            result["maximum_consecutive_failures"]
        ),
        "best_reward": float(result["best_reward"]),
        "output_directory": str(output_directory),
    }
    worker_output = (
        EXPERIMENT_DIRECTORY
        / "torus_alpha_0p05_stability"
        / "workers"
        / f"seed{seed}.csv"
    )
    write_csv(worker_output, STABILITY_FIELDS, [row])
    print(row, flush=True)
    return row


def read_episode_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        "episode": np.asarray([int(row["episode"]) for row in rows]),
        "reward": np.asarray([float(row["reward"]) for row in rows]),
        "terminal_flatness": np.asarray([
            float(row["terminal_flatness"]) for row in rows
        ]),
        "strict_success": np.asarray([
            int(row["strict_success"]) for row in rows
        ]),
    }


def consecutive_failures(successes: np.ndarray) -> int:
    longest = current = 0
    for success in successes:
        if int(success):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def save_combined_learning_curves(rows: list[dict]) -> Path | None:
    complete_rows = []
    for row in rows:
        metrics_path = (
            Path(row["output_directory"]) / "episode_metrics.csv"
        )
        if not metrics_path.exists():
            raise RuntimeError(f"missing episode metrics: {metrics_path}")
        complete_rows.append((row, read_episode_metrics(metrics_path)))
    if not complete_rows:
        return None

    figure, axes = plt.subplots(
        len(complete_rows), 2,
        figsize=(13, 3.1 * len(complete_rows)),
        squeeze=False,
    )
    window = 200
    for row_index, (row, metrics) in enumerate(complete_rows):
        episodes = metrics["episode"]
        first_success = row["first_strict_success_episode"]
        for column, (key, ylabel) in enumerate((
            ("reward", "Episode reward"),
            ("terminal_flatness", "Terminal flatness F"),
        )):
            axis = axes[row_index, column]
            values = metrics[key]
            axis.plot(
                episodes, values, color="0.78", linewidth=0.45,
                label="raw",
            )
            average_episode, average = moving_average(
                values.tolist(), window
            )
            if average.size:
                axis.plot(
                    average_episode, average, color="black",
                    linewidth=1.5, label=f"moving average ({window})",
                )
            if first_success not in (None, "", "None"):
                axis.axvline(
                    int(first_success), color="#d62728", linestyle="--",
                    linewidth=1.0, label="first strict success",
                )
            axis.set_ylabel(ylabel)
            axis.set_title(f"seed {row['seed']}")
            axis.grid(alpha=0.25)
            if key == "terminal_flatness":
                axis.set_yscale("log")
            if row_index == len(complete_rows) - 1:
                axis.set_xlabel("Episode")
            if row_index == 0:
                axis.legend(fontsize=8)
    figure.suptitle("torus D2, spectral alpha=0.05", y=0.998)
    figure.tight_layout()
    output = (
        EXPERIMENT_DIRECTORY
        / "torus_alpha_0p05_stability"
        / "learning_curves_combined.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def merge_stability() -> Path:
    worker_directory = (
        EXPERIMENT_DIRECTORY
        / "torus_alpha_0p05_stability"
        / "workers"
    )
    rows = []
    for path in sorted(worker_directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    expected = set(STABILITY_SEEDS)
    observed = {int(row["seed"]) for row in rows}
    bad_lengths = [
        (int(row["seed"]), int(row["episodes_completed"]))
        for row in rows
        if int(row["episodes_completed"]) != STABILITY_EPISODES
    ]
    if observed != expected or len(rows) != len(expected) or bad_lengths:
        raise RuntimeError(
            "stability runs are incomplete or contain smoke/stale rows: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}, "
            f"bad_lengths={bad_lengths}, row_count={len(rows)}"
        )
    expected_prefix = {
        8101: (2632, 197, 526397, 328),
        8102: (1569, 198, 313798, 196),
        8103: (1446, 199, 289199, 180),
        8105: (1259, 188, 251788, 157),
    }
    validation_errors = []
    for row in rows:
        seed = int(row["seed"])
        identity = f"torus_d2/seed{seed}"
        if row["task"] != STABILITY_TASK:
            validation_errors.append(f"{identity}: wrong task {row['task']}")
        if not np.isclose(
            float(row["spectral_alpha"]), STABILITY_ALPHA,
            rtol=0, atol=1e-15,
        ):
            validation_errors.append(
                f"{identity}: wrong alpha {row['spectral_alpha']}"
            )
        actual_prefix = (
            int(row["first_strict_success_episode"]),
            int(row["first_strict_success_episode_steps"]),
            int(row["first_strict_success_mechanical_steps"]),
            int(row["optimizer_updates_before_first_success"]),
        )
        if actual_prefix != expected_prefix[seed]:
            validation_errors.append(
                f"{identity}: discovery prefix {actual_prefix}, "
                f"expected {expected_prefix[seed]}"
            )
        output_directory = Path(row["output_directory"])
        metrics_path = output_directory / "episode_metrics.csv"
        updates_path = output_directory / "update_metrics.csv"
        summary_path = output_directory / "summary.json"
        curve_path = output_directory / "learning_curve.png"
        for required_path in (
            metrics_path, updates_path, summary_path, curve_path
        ):
            if not required_path.exists():
                validation_errors.append(
                    f"{identity}: missing {required_path.name}"
                )
        if not all(path.exists() for path in (
            metrics_path, updates_path, summary_path, curve_path
        )):
            continue
        metrics = read_episode_metrics(metrics_path)
        if len(metrics["episode"]) != STABILITY_EPISODES:
            validation_errors.append(
                f"{identity}: metrics has {len(metrics['episode'])} rows"
            )
        elif not np.array_equal(
            metrics["episode"], np.arange(1, STABILITY_EPISODES + 1)
        ):
            validation_errors.append(
                f"{identity}: episode column is not contiguous 1..10000"
            )
        if not np.all(np.isfinite(metrics["reward"])):
            validation_errors.append(f"{identity}: non-finite reward")
        if not np.all(np.isfinite(metrics["terminal_flatness"])):
            validation_errors.append(
                f"{identity}: non-finite terminal flatness"
            )
        if not np.all(np.isin(metrics["strict_success"], (0, 1))):
            validation_errors.append(
                f"{identity}: strict-success column is not binary"
            )
        with updates_path.open(newline="", encoding="utf-8") as file:
            update_count = sum(1 for _ in csv.DictReader(file))
        if update_count != int(row["optimizer_updates_completed"]):
            validation_errors.append(
                f"{identity}: update rows={update_count}, worker says "
                f"{row['optimizer_updates_completed']}"
            )
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
        summary_checks = {
            "task": row["task"],
            "seed": seed,
            "episodes": int(row["episodes_completed"]),
            "first_strict_success_episode": int(
                row["first_strict_success_episode"]
            ),
            "optimizer_updates_completed": int(
                row["optimizer_updates_completed"]
            ),
            "cumulative_mechanical_steps": int(
                row["cumulative_mechanical_steps"]
            ),
        }
        for key, expected_value in summary_checks.items():
            if summary[key] != expected_value:
                validation_errors.append(
                    f"{identity}: summary {key}={summary[key]}, "
                    f"worker={expected_value}"
                )
        first = int(row["first_strict_success_episode"])
        confirmation = row["rolling_200_confirmation_episode_90pct"]
        row["maximum_consecutive_failures_all_training"] = int(
            row["maximum_consecutive_failures"]
        )
        row["maximum_consecutive_failures_after_first_success"] = (
            consecutive_failures(metrics["strict_success"][first - 1:])
        )
        row["maximum_consecutive_failures_after_confirmation"] = (
            "" if confirmation in ("", "None") else consecutive_failures(
                metrics["strict_success"][int(confirmation) - 1:]
            )
        )
    if validation_errors:
        raise RuntimeError(
            "invalid stability outputs:\n" + "\n".join(validation_errors)
        )
    rows.sort(key=lambda row: int(row["seed"]))
    output = (
        EXPERIMENT_DIRECTORY
        / "torus_alpha_0p05_stability"
        / "stability_summary.csv"
    )
    write_csv(output, STABILITY_AGGREGATE_FIELDS, rows)
    combined = save_combined_learning_curves(rows)
    print(f"merged {len(rows)} rows into {output}", flush=True)
    if combined is not None:
        print(f"saved {combined}", flush=True)
    return output


def write_manifest() -> Path:
    torus = task_config("torus_d2", STABILITY_EPISODES)
    water = task_config("water_d1", ALPHA_ZERO_MAX_EPISODES)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "scientific_question": (
            "Can the plane-swap/spectral reward term be omitted while "
            "retaining first-discovery ability?"
        ),
        "reward_ablation": {
            "change": "spectral_weight=0.0; all other reward terms unchanged",
            "tasks": list(ALPHA_ZERO_TASKS),
            "seeds": list(ALPHA_ZERO_SEEDS),
            "max_episodes": ALPHA_ZERO_MAX_EPISODES,
            "stop_rule": (
                "stop at first strict success, otherwise censor at 10000"
            ),
        },
        "torus_stability": {
            "task": STABILITY_TASK,
            "spectral_weight": STABILITY_ALPHA,
            "seeds": list(STABILITY_SEEDS),
            "episodes_from_scratch": STABILITY_EPISODES,
            "reason_from_scratch": (
                "the discovery screen did not save model, optimizer, and RNG "
                "checkpoints"
            ),
        },
        "controlled_learning_method": {
            "actor_and_entropy_reduction": (
                "sum / (actual_batch_episodes * max_steps)"
            ),
            "critic_reduction": "transition mean",
            "advantage_normalization": "all transitions in batch",
            "discovery_batch_size": int(torus["batch_size"]),
            "post_first_strict_batch_size": 16,
            "post_first_strict_actor_lr_factor": 0.5,
            "critic_lr_change": "none",
            "fixed_episode_actor_lr_schedule": (
                "disabled because adaptive mode is active"
            ),
            "crossing_policy": (
                "reject candidate as a zero-reward no-op; continue episode"
            ),
        },
        "task_settings": {
            "torus_d2": {
                "max_steps": int(torus["max_steps"]),
                "max_step_width": float(torus["max_step_width"]),
                "actor_learning_rate": float(
                    torus["actor_learning_rate"]
                ),
                "critic_learning_rate": float(
                    torus["critic_learning_rate"]
                ),
            },
            "water_d1": {
                "max_steps": int(water["max_steps"]),
                "max_step_width": float(water["max_step_width"]),
                "actor_learning_rate": float(
                    water["actor_learning_rate"]
                ),
                "critic_learning_rate": float(
                    water["critic_learning_rate"]
                ),
                "near_miss_spread_error_far": float(
                    water["near_miss_spread_error_far"]
                ),
            },
        },
        "strict_success": {
            "flatness_tolerance": float(torus["flatness_tolerance"]),
            "spread_tolerance": float(torus["spread_tolerance"]),
        },
        "stability_rule": {
            "confirmation": "rolling-200 strict success rate >= 0.90",
            "collapse": "rolling-200 < 0.80 after confirmation",
            "final_window": (
                "last-1000 success >= 0.95 and every rolling-200 within "
                "the last 1000 >= 0.90"
            ),
            "retention": (
                "confirmation observed for >=1000 episodes, no collapse, "
                "and final-window rule holds"
            ),
        },
        "interpretation_caveat": (
            "The torus alpha=0 screen is paired with the latest torus alpha "
            "screen. The water_d1 alpha=0 screen instead tests omission under "
            "the current fixed-horizon/adaptive method and is not an exact "
            "one-variable replication of an older water_d1 run."
        ),
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
    subparsers.add_parser("merge-screen")
    subparsers.add_parser("merge-stability")

    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument(
        "--task", choices=ALPHA_ZERO_TASKS, required=True
    )
    screen_parser.add_argument("--seed", type=int, required=True)
    screen_parser.add_argument(
        "--max-episodes", type=int, default=ALPHA_ZERO_MAX_EPISODES
    )

    stability_parser = subparsers.add_parser("stability")
    stability_parser.add_argument("--seed", type=int, required=True)
    stability_parser.add_argument(
        "--episodes", type=int, default=STABILITY_EPISODES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        write_manifest()
    elif args.command == "merge-screen":
        merge_alpha_zero_screen()
    elif args.command == "merge-stability":
        merge_stability()
    elif args.command == "screen":
        if args.seed not in ALPHA_ZERO_SEEDS:
            raise ValueError(f"seed {args.seed} is not assigned")
        alpha_zero_screen(args.task, args.seed, args.max_episodes)
    elif args.command == "stability":
        if args.seed not in STABILITY_SEEDS:
            raise ValueError(f"seed {args.seed} is not assigned")
        stability(args.seed, args.episodes)


if __name__ == "__main__":
    main()
