"""Plot near-miss reach probability and episode-best quality for three tasks."""

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

from RL_three_task_checkpoint_experiment import (
    EXPECTED,
    EXPERIMENT_DIRECTORY,
    SEEDS,
)


TASKS = tuple(SEEDS)
TASK_LABELS = {
    "five_d2": "Five-bar, D=2",
    "five_d3": "Five-bar, D=3",
    "torus_d2": "Torus, D=2",
}
SELECTION_LABELS = {
    "five_d2": "median discovery seed",
    "five_d3": "slowest discovery seed",
    "torus_d2": "median-ranked screen seed",
}
FIGURE_SELECTION_LABELS = {
    "five_d2": "Median successful seed",
    "five_d3": "Slowest successful seed",
    "torus_d2": "Median rank of 5 runs (1 censored)",
}
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "analysis"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_change_summary(rows: list[dict]) -> Path:
    output = OUTPUT_DIRECTORY / "checkpoint_change_summary.csv"
    fields = (
        "task", "seed", "selection", "first_success_episode",
        "final_update", "initial_near_miss_rate", "final_near_miss_rate",
        "initial_qstar_q25", "initial_qstar_median", "initial_qstar_q75",
        "final_qstar_q25", "final_qstar_median", "final_qstar_q75",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task_name in TASKS:
            selected = [row for row in rows if row["task"] == task_name]
            initial, final = selected[0], selected[-1]
            writer.writerow({
                "task": task_name,
                "seed": SEEDS[task_name],
                "selection": SELECTION_LABELS[task_name],
                "first_success_episode": EXPECTED[task_name]["episode"],
                "final_update": EXPECTED[task_name]["updates"],
                "initial_near_miss_rate": initial["near_miss_rate"],
                "final_near_miss_rate": final["near_miss_rate"],
                "initial_qstar_q25": initial["best_quality_q25"],
                "initial_qstar_median": initial["best_quality_median"],
                "initial_qstar_q75": initial["best_quality_q75"],
                "final_qstar_q25": final["best_quality_q25"],
                "final_qstar_median": final["best_quality_median"],
                "final_qstar_q75": final["best_quality_q75"],
            })
    return output


def save_figure(rows: list[dict]) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 7.6),
        sharey="row",
        constrained_layout=True,
    )

    for column, task_name in enumerate(TASKS):
        selected = [row for row in rows if row["task"] == task_name]
        updates = np.asarray([int(row["update"]) for row in selected])
        near = np.asarray([float(row["near_miss_rate"]) for row in selected])
        near_low = np.asarray(
            [float(row["near_miss_ci_low"]) for row in selected]
        )
        near_high = np.asarray(
            [float(row["near_miss_ci_high"]) for row in selected]
        )
        q25 = np.asarray(
            [float(row["best_quality_q25"]) for row in selected]
        )
        median = np.asarray(
            [float(row["best_quality_median"]) for row in selected]
        )
        q75 = np.asarray(
            [float(row["best_quality_q75"]) for row in selected]
        )

        near_axis = axes[0, column]
        near_axis.fill_between(
            updates, near_low, near_high,
            color="tab:blue", alpha=0.16, linewidth=0,
            label="Pointwise Wilson 95% interval",
        )
        near_axis.plot(
            updates, near,
            color="tab:blue", marker="o", markersize=2.6,
            linewidth=1.3, label="Near-miss-or-better reach rate",
        )
        near_axis.axvline(
            EXPECTED[task_name]["updates"],
            color="0.35", linestyle="--", linewidth=1.0,
        )
        near_axis.set_title(
            f"{TASK_LABELS[task_name]} | seed {SEEDS[task_name]}\n"
            f"{FIGURE_SELECTION_LABELS[task_name]}\n"
            f"First success: E{EXPECTED[task_name]['episode']} "
            f"(policy after U{EXPECTED[task_name]['updates']})",
            fontsize=10.0,
            linespacing=1.12,
        )
        near_axis.set_ylim(-0.02, 1.02)
        near_axis.set_xlim(0, EXPECTED[task_name]["updates"])
        near_axis.grid(alpha=0.22)

        quality_axis = axes[1, column]
        quality_axis.fill_between(
            updates, q25, q75,
            color="tab:green", alpha=0.18, linewidth=0,
            label="Interquartile range (25th–75th percentile)",
        )
        quality_axis.plot(
            updates, median,
            color="tab:green", marker="o", markersize=2.6,
            linewidth=1.3, label=r"Median episode-best $Q^*$",
        )
        quality_axis.axvline(
            EXPECTED[task_name]["updates"],
            color="0.35", linestyle="--", linewidth=1.0,
        )
        quality_axis.set_ylim(-0.02, 1.02)
        quality_axis.set_xlim(0, EXPECTED[task_name]["updates"])
        quality_axis.set_xlabel("Optimizer update")
        quality_axis.grid(alpha=0.22)

    axes[0, 0].set_ylabel("Rollout reach probability")
    axes[1, 0].set_ylabel(r"Episode-best quality $Q^*$")
    axes[0, 0].legend(loc="best", frameon=False)
    axes[1, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Frozen-policy progress toward first discovery "
        "(100 rollouts per checkpoint)\n"
        r"Near-miss-or-better: $\exists t, F_t\leq0.1$ and "
        r"$|D_t-D^*|\leq0.05$; "
        r"$Q^*=\max_t q_F(t)q_D(t)$"
    )

    png_path = OUTPUT_DIRECTORY / "three_task_checkpoint_evaluation.png"
    pdf_path = OUTPUT_DIRECTORY / "three_task_checkpoint_evaluation.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def write_provenance(artifacts: list[Path]) -> Path:
    inputs = {
        "experiment_manifest": EXPERIMENT_DIRECTORY / "manifest.json",
        "evaluation_design": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "evaluation_design.csv"
        ),
        "checkpoint_inventory": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "checkpoint_inventory.csv"
        ),
        "checkpoint_rollouts": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "checkpoint_rollouts.csv"
        ),
        "checkpoint_summary": (
            EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
            / "checkpoint_summary.csv"
        ),
    }
    sources = {
        "evaluate_three_task_checkpoints.py": (
            Path(__file__).with_name("evaluate_three_task_checkpoints.py")
        ),
        "plot_three_task_checkpoint_evaluation.py": Path(__file__),
    }
    payload = {
        "source_sha256": {
            name: sha256(path) for name, path in sources.items()
        },
        "input_sha256": {
            name: sha256(path) for name, path in inputs.items()
        },
        "artifact_sha256": {
            path.name: sha256(path) for path in artifacts
        },
    }
    output = OUTPUT_DIRECTORY / "analysis_provenance.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = (
        EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
        / "checkpoint_summary.csv"
    )
    rows = read_csv(summary_path)
    png_path, pdf_path = save_figure(rows)
    change_path = write_change_summary(rows)
    provenance_path = write_provenance([png_path, pdf_path, change_path])
    print(png_path)
    print(pdf_path)
    print(change_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
