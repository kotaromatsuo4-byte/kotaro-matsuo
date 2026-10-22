"""Plot terminal near-miss qualities up to the first strict success.

The quality definitions and tolerances are read from the saved experiment
manifest so that this post-processing remains tied to the configuration that
produced the episode logs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260902_qf_qd_median_seed_v1"
)
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "analysis"

TASK_LABELS = {
    "five_d2": "Five-bar, D=2",
    "five_d3": "Five-bar, D=3",
    "torus_d2": "Torus, D=2",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def near_miss_quality(
    flatness: float,
    spread_error: float,
    *,
    flatness_tolerance: float,
    spread_tolerance: float,
    flatness_far: float,
    spread_error_far: float,
) -> tuple[float, float]:
    q_flatness = np.log(
        flatness_far / max(float(flatness), flatness_tolerance)
    ) / np.log(flatness_far / flatness_tolerance)
    q_spread = (
        spread_error_far - abs(float(spread_error))
    ) / (spread_error_far - spread_tolerance)
    return (
        float(np.clip(q_flatness, 0.0, 1.0)),
        float(np.clip(q_spread, 0.0, 1.0)),
    )


def terminal_quality_rows() -> list[dict]:
    with (EXPERIMENT_DIRECTORY / "manifest.json").open(
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    output = []
    for task_name in manifest["tasks"]:
        task_label = TASK_LABELS[task_name]
        seed = int(manifest["seeds"][task_name])
        config = manifest["resolved_configs"][task_name]
        metrics_path = (
            EXPERIMENT_DIRECTORY
            / "runs"
            / task_name
            / f"seed{seed}"
            / "episode_metrics.csv"
        )
        rows = read_rows(metrics_path)
        episodes = [int(row["episode"]) for row in rows]
        successes = [int(row["strict_success"]) for row in rows]
        if episodes != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous episode log: {metrics_path}")
        if sum(successes) != 1 or successes[-1] != 1:
            raise ValueError(
                f"run does not stop at its first strict success: {metrics_path}"
            )
        final_episode = int(rows[-1]["episode"])
        for row in rows:
            episode = int(row["episode"])
            flatness = float(row["terminal_flatness"])
            spread_error = float(row["spread_error"])
            q_flatness, q_spread = near_miss_quality(
                flatness,
                spread_error,
                flatness_tolerance=float(config["flatness_tolerance"]),
                spread_tolerance=float(config["spread_tolerance"]),
                flatness_far=float(config["near_miss_flatness_far"]),
                spread_error_far=float(
                    config["near_miss_spread_error_far"]
                ),
            )
            output.append({
                "task": task_name,
                "task_label": task_label,
                "seed": seed,
                "episode": episode,
                "episode_progress": (
                    (episode - 1) / (final_episode - 1)
                    if final_episode > 1 else 1.0
                ),
                "terminal_flatness": flatness,
                "spread_error": spread_error,
                "q_flatness": q_flatness,
                "q_spread": q_spread,
                "quality": q_flatness * q_spread,
                "strict_success": int(row["strict_success"]),
                "termination_reason": row["termination_reason"],
            })
    return output


def stage_median(rows: list[dict], *, late: bool) -> tuple[float, float]:
    failures = [row for row in rows if not row["strict_success"]]
    if not failures:
        raise ValueError("stage medians require at least one failed episode")
    count = max(1, int(np.ceil(len(failures) / 4)))
    stage = failures[-count:] if late else failures[:count]
    return (
        float(np.median([row["q_flatness"] for row in stage])),
        float(np.median([row["q_spread"] for row in stage])),
    )


def write_derived_csv(rows: list[dict]) -> Path:
    path = OUTPUT_DIRECTORY / "qf_qd_terminal_values.csv"
    fields = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(rows: list[dict]) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        1,
        len(TASK_LABELS),
        figsize=(13.2, 4.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    normalise = colors.Normalize(vmin=0.0, vmax=1.0)
    colour_map = plt.get_cmap("viridis")

    for axis, task_name in zip(axes, TASK_LABELS):
        task_rows = [row for row in rows if row["task"] == task_name]
        task_label = TASK_LABELS[task_name]
        seed = int(task_rows[0]["seed"])
        q_flatness = np.asarray(
            [row["q_flatness"] for row in task_rows], dtype=float
        )
        q_spread = np.asarray(
            [row["q_spread"] for row in task_rows], dtype=float
        )
        progress = np.asarray(
            [row["episode_progress"] for row in task_rows], dtype=float
        )

        axis.scatter(
            q_flatness,
            q_spread,
            c=progress,
            norm=normalise,
            cmap=colour_map,
            s=13,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )

        early = stage_median(task_rows, late=False)
        late = stage_median(task_rows, late=True)
        axis.annotate(
            "",
            xy=late,
            xytext=early,
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 1.5,
            },
        )
        axis.scatter(
            *early,
            marker="s",
            s=62,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="First quarter of failed episodes",
            zorder=4,
        )
        axis.scatter(
            *late,
            marker="D",
            s=62,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="Last quarter of failed episodes",
            zorder=4,
        )

        successes = [row for row in task_rows if row["strict_success"]]
        if successes:
            success = successes[0]
            axis.scatter(
                success["q_flatness"],
                success["q_spread"],
                marker="*",
                s=180,
                facecolor="tab:red",
                edgecolor="black",
                linewidth=0.9,
                label="First strict success",
                zorder=5,
            )

        axis.set_title(
            f"{task_label}\nseed {seed}, success at episode "
            f"{task_rows[-1]['episode']}"
        )
        axis.set_xlim(-0.025, 1.025)
        axis.set_ylim(-0.025, 1.025)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"Flatness quality $q_F$")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel(r"Spread quality $q_D$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    scalar_map = plt.cm.ScalarMappable(norm=normalise, cmap=colour_map)
    colour_bar = figure.colorbar(
        scalar_map,
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    colour_bar.set_label("Episode progress to first success")
    figure.suptitle(
        "Terminal near-miss quality up to first strict discovery\n"
        "Geometry-based quality before termination-specific reward overrides"
    )

    png_path = OUTPUT_DIRECTORY / "qf_qd_terminal_distribution.png"
    pdf_path = OUTPUT_DIRECTORY / "qf_qd_terminal_distribution.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = terminal_quality_rows()
    csv_path = write_derived_csv(rows)
    png_path, pdf_path = save_figure(rows)
    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
