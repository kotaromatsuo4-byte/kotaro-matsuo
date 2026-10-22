"""Plot pre-discovery learning curves and frozen-checkpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from RL_discovery_evidence_experiment import EXPERIMENT_DIRECTORY


OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "analysis"
MOVING_AVERAGE_WINDOW = 40
TASK_LABELS = {
    "five_d2": "Five-bar, D=2",
    "five_d3": "Five-bar, D=3",
    "torus_d2": "Torus, D=2",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.empty(0), np.empty(0)
    result = np.convolve(
        values, np.ones(window, dtype=float) / window, mode="valid"
    )
    return np.arange(window, len(values) + 1), result


def near_miss_quality(
    flatness: float,
    spread_error: float,
    config: dict,
) -> tuple[float, float, float]:
    flatness_tolerance = float(config["flatness_tolerance"])
    spread_tolerance = float(config["spread_tolerance"])
    flatness_far = float(config["near_miss_flatness_far"])
    spread_far = float(config["near_miss_spread_error_far"])
    q_flatness = np.log(
        flatness_far / max(float(flatness), flatness_tolerance)
    ) / np.log(flatness_far / flatness_tolerance)
    q_spread = (
        spread_far - abs(float(spread_error))
    ) / (spread_far - spread_tolerance)
    q_flatness = float(np.clip(q_flatness, 0.0, 1.0))
    q_spread = float(np.clip(q_spread, 0.0, 1.0))
    return q_flatness, q_spread, q_flatness * q_spread


def task_metric_path(task_name: str, manifest: dict) -> Path:
    if task_name == "torus_d2":
        return Path(manifest["torus_learning_curve_source"])
    config = manifest["resolved_configs"][task_name]
    return Path(config["rl_output_directory"]) / "episode_metrics.csv"


def discovery_rows(task_name: str, manifest: dict) -> list[dict]:
    rows = read_csv(task_metric_path(task_name, manifest))
    successes = [
        index for index, row in enumerate(rows)
        if int(row["strict_success"])
    ]
    if not successes:
        raise ValueError(f"no strict success in {task_name}")
    rows = rows[:successes[0] + 1]
    expected = manifest["expected_screen_results"][task_name]
    if int(rows[-1]["episode"]) != int(expected["episode"]):
        raise ValueError(f"unexpected first success for {task_name}")
    return rows


def derived_learning_rows(manifest: dict) -> list[dict]:
    output = []
    for task_name in manifest["tasks"]:
        config = manifest["resolved_configs"][task_name]
        for row in discovery_rows(task_name, manifest):
            q_flatness, q_spread, quality = near_miss_quality(
                float(row["terminal_flatness"]),
                float(row["spread_error"]),
                config,
            )
            output.append({
                "task": task_name,
                "seed": int(manifest["seeds"][task_name]),
                "episode": int(row["episode"]),
                "reward": float(row["reward"]),
                "q_flatness": q_flatness,
                "q_spread": q_spread,
                "quality": quality,
                "strict_success": int(row["strict_success"]),
                "terminal_flatness": float(row["terminal_flatness"]),
                "spread_error": float(row["spread_error"]),
                "cumulative_mechanical_steps": int(
                    row["cumulative_mechanical_steps"]
                ),
                "optimizer_updates_before_episode_update": int(
                    row["optimizer_updates_before_episode_update"]
                ),
            })
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curves(manifest: dict) -> tuple[Path, Path, Path]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = derived_learning_rows(manifest)
    csv_path = OUTPUT_DIRECTORY / "pre_discovery_learning_values.csv"
    write_rows(csv_path, rows)

    figure, axes = plt.subplots(
        2, 3, figsize=(15.0, 7.4), constrained_layout=True
    )
    quality_styles = {
        "q_flatness": ("tab:blue", r"$q_F$"),
        "q_spread": ("tab:orange", r"$q_D$"),
        "quality": ("tab:green", r"$Q=q_Fq_D$"),
    }

    for column, task_name in enumerate(manifest["tasks"]):
        task_rows = [row for row in rows if row["task"] == task_name]
        failures = [row for row in task_rows if not row["strict_success"]]
        success = task_rows[-1]
        episodes = np.asarray(
            [row["episode"] for row in failures], dtype=float
        )
        rewards = np.asarray(
            [row["reward"] for row in failures], dtype=float
        )

        reward_axis = axes[0, column]
        reward_axis.scatter(
            episodes,
            rewards,
            color="0.72",
            s=5,
            alpha=0.55,
            linewidths=0,
            label="Failed episodes",
        )
        average_x, reward_average = moving_average(
            rewards, MOVING_AVERAGE_WINDOW
        )
        reward_axis.plot(
            average_x,
            reward_average,
            color="black",
            linewidth=1.8,
            label=f"Trailing mean ({MOVING_AVERAGE_WINDOW})",
        )
        reward_axis.axvline(
            success["episode"], color="tab:red", linestyle="--",
            linewidth=1.0, alpha=0.7,
        )
        reward_axis.scatter(
            [success["episode"]],
            [0.97],
            transform=reward_axis.get_xaxis_transform(),
            marker="*",
            s=135,
            color="tab:red",
            edgecolor="black",
            linewidth=0.7,
            clip_on=False,
            label="First strict success (timing only)",
            zorder=5,
        )
        expected = manifest["expected_screen_results"][task_name]
        reward_axis.set_title(
            f"{TASK_LABELS[task_name]}, seed {manifest['seeds'][task_name]}\n"
            f"E={expected['episode']}, U={expected['updates']}, "
            f"steps={expected['mechanical_steps']:,}"
        )
        reward_axis.set_ylabel("Episode reward" if column == 0 else "")
        reward_axis.grid(alpha=0.22)
        reward_axis.set_xlim(1, success["episode"])

        quality_axis = axes[1, column]
        for metric, (colour, label) in quality_styles.items():
            values = np.asarray(
                [row[metric] for row in failures], dtype=float
            )
            quality_axis.scatter(
                episodes,
                values,
                color=colour,
                s=4,
                alpha=0.13,
                linewidths=0,
            )
            average_x, average = moving_average(
                values, MOVING_AVERAGE_WINDOW
            )
            quality_axis.plot(
                average_x,
                average,
                color=colour,
                linewidth=1.8,
                label=label,
            )
        quality_axis.axvline(
            success["episode"], color="tab:red", linestyle="--",
            linewidth=1.0, alpha=0.7,
        )
        quality_axis.scatter(
            success["episode"],
            1.0,
            marker="*",
            s=135,
            color="tab:red",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
        quality_axis.set_ylim(-0.025, 1.025)
        quality_axis.set_xlim(1, success["episode"])
        quality_axis.set_xlabel("Episode")
        quality_axis.set_ylabel("Terminal quality" if column == 0 else "")
        quality_axis.grid(alpha=0.22)

    axes[0, 0].legend(loc="best", frameon=False)
    axes[1, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Learning up to first strict discovery: slowest successful seed\n"
        "Success episode is excluded from every moving average; reward star "
        "marks timing, not reward magnitude"
    )
    png_path = OUTPUT_DIRECTORY / "pre_discovery_learning_curves.png"
    pdf_path = OUTPUT_DIRECTORY / "pre_discovery_learning_curves.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path, csv_path


def save_checkpoint_figure() -> tuple[Path, Path]:
    summary_path = (
        EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
        / "checkpoint_summary.csv"
    )
    rows = read_csv(summary_path)
    updates = np.asarray([int(row["update"]) for row in rows], dtype=float)
    training_episodes = np.asarray(
        [int(row["training_episode"]) for row in rows], dtype=float
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(9.4, 7.4), sharex=True, constrained_layout=True
    )
    rate_styles = {
        "reach_f0p1": (
            "tab:blue", r"Near-miss or better: $F\leq0.1$"
        ),
        "reach_f0p01": (
            "tab:orange", r"Close-miss or better: $F\leq0.01$"
        ),
        "strict_success": (
            "tab:red", r"Strict success: $F\leq0.001$"
        ),
    }
    for metric, (colour, label) in rate_styles.items():
        rate = np.asarray(
            [float(row[f"{metric}_rate"]) for row in rows]
        )
        low = np.asarray(
            [float(row[f"{metric}_ci_low"]) for row in rows]
        )
        high = np.asarray(
            [float(row[f"{metric}_ci_high"]) for row in rows]
        )
        axes[0].fill_between(
            updates, low, high, color=colour, alpha=0.12, linewidth=0
        )
        axes[0].plot(
            updates, rate, color=colour, marker="o", markersize=2.6,
            linewidth=1.25, label=label,
        )
    axes[0].set_ylabel("Episode reach rate")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="best", frameon=False)
    axes[0].set_title(
        "Five-bar D=2, training seed 9105: initial policy and 100 frozen "
        "rollouts after every update\n"
        r"All reach events also require $|D-D^*|\leq0.05$; top bands are "
        "pointwise Wilson 95% intervals"
    )
    axes[0].axvline(100, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].text(
        99,
        0.98,
        "U100 generated\ntraining success E803",
        transform=axes[0].get_xaxis_transform(),
        ha="right",
        va="top",
    )

    quality_styles = {
        "best_quality": (
            "tab:green", r"Episode-best $Q^*=\max_t q_F(t)q_D(t)$"
        ),
        "terminal_quality": (
            "tab:purple", r"Terminal $Q=q_Fq_D$"
        ),
    }
    for metric, (colour, label) in quality_styles.items():
        mean = np.asarray(
            [float(row[f"mean_{metric}"]) for row in rows]
        )
        low = np.asarray(
            [float(row[f"{metric}_ci_low"]) for row in rows]
        )
        high = np.asarray(
            [float(row[f"{metric}_ci_high"]) for row in rows]
        )
        axes[1].fill_between(
            updates, low, high, color=colour, alpha=0.12, linewidth=0
        )
        axes[1].plot(
            updates, mean, color=colour, marker="o", markersize=2.6,
            linewidth=1.25, label=label,
        )
    axes[1].set_xlabel("Optimizer update")
    axes[1].set_ylabel("Mean quality over 100 rollouts")
    axes[1].set_ylim(-0.01, 1.01)
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="best", frameon=False)
    axes[1].set_title(
        "Bottom bands are pointwise percentile-bootstrap 95% intervals "
        "for means (2,000 resamples)"
    )

    secondary = axes[0].secondary_xaxis("top")
    secondary_updates = np.asarray((0, 25, 50, 75, 100), dtype=float)
    secondary.set_xticks(
        secondary_updates,
        labels=[
            str(int(8 * value))
            for value in secondary_updates
        ],
    )
    secondary.set_xlabel("Training episode at checkpoint")

    png_path = OUTPUT_DIRECTORY / "five_d2_checkpoint_evaluation.png"
    pdf_path = OUTPUT_DIRECTORY / "five_d2_checkpoint_evaluation.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def load_manifest() -> dict:
    with (EXPERIMENT_DIRECTORY / "manifest.json").open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_analysis_provenance() -> Path:
    manifest_path = EXPERIMENT_DIRECTORY / "manifest.json"
    manifest = load_manifest()
    input_paths = {
        "execution_manifest": manifest_path,
        "five_d2_episode_metrics": task_metric_path("five_d2", manifest),
        "five_d3_episode_metrics": task_metric_path("five_d3", manifest),
        "torus_episode_metrics": task_metric_path("torus_d2", manifest),
        "checkpoint_rollouts": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "checkpoint_rollouts.csv"
        ),
        "checkpoint_summary": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "checkpoint_summary.csv"
        ),
        "evaluation_design": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "evaluation_design.csv"
        ),
    }
    artifact_paths = {
        "learning_values": OUTPUT_DIRECTORY / "pre_discovery_learning_values.csv",
        "learning_png": OUTPUT_DIRECTORY / "pre_discovery_learning_curves.png",
        "learning_pdf": OUTPUT_DIRECTORY / "pre_discovery_learning_curves.pdf",
        "checkpoint_png": OUTPUT_DIRECTORY / "five_d2_checkpoint_evaluation.png",
        "checkpoint_pdf": OUTPUT_DIRECTORY / "five_d2_checkpoint_evaluation.pdf",
    }
    provenance = {
        "note": (
            "manifest.json preserves source hashes captured before training "
            "and frozen-rollout evaluation; the hashes below describe the "
            "final merge and plotting stage"
        ),
        "postprocessing_source_sha256": {
            "evaluate_five_d2_checkpoints.py": sha256(
                Path(__file__).with_name("evaluate_five_d2_checkpoints.py")
            ),
            "plot_discovery_evidence.py": sha256(Path(__file__)),
        },
        "input_sha256": {
            name: sha256(path) for name, path in input_paths.items()
        },
        "artifact_sha256": {
            name: sha256(path) for name, path in artifact_paths.items()
        },
    }
    output = OUTPUT_DIRECTORY / "analysis_provenance.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2, ensure_ascii=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("learning-curves", "checkpoint", "all")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.command in ("learning-curves", "all"):
        print(*save_learning_curves(load_manifest()), sep="\n")
    if args.command in ("checkpoint", "all"):
        print(*save_checkpoint_figure(), sep="\n")
    if args.command == "all":
        print(write_analysis_provenance())


if __name__ == "__main__":
    main()
