"""Create publication-style checkpoint figures for the slowest successful seeds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, ticker
import matplotlib.pyplot as plt
import numpy as np

import RL_slowest_seed_checkpoint_experiment as experiment


EXPECTED = experiment.EXPECTED
EXPERIMENT_DIRECTORY = experiment.EXPERIMENT_DIRECTORY
SEEDS = experiment.SEEDS
TASKS = tuple(SEEDS)
ROLLOUTS_PER_CHECKPOINT = 100
FONT_FAMILY = "Times New Roman"
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


def configure_style() -> tuple[Path, str]:
    properties = font_manager.FontProperties(family=FONT_FAMILY)
    font_path = Path(font_manager.findfont(
        properties,
        fallback_to_default=False,
    )).resolve()
    resolved_family = font_manager.get_font(str(font_path)).family_name
    if resolved_family.casefold() != FONT_FAMILY.casefold():
        raise RuntimeError(
            f"requested {FONT_FAMILY}, resolved {resolved_family}: {font_path}"
        )

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY],
        "font.size": 11.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.linewidth": 0.8,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return font_path, resolved_family


def validated_task_rows(rows: list[dict], task_name: str) -> list[dict]:
    selected = sorted(
        (row for row in rows if row["task"] == task_name),
        key=lambda row: int(row["update"]),
    )
    expected_updates = list(range(int(EXPECTED[task_name]["updates"]) + 1))
    actual_updates = [int(row["update"]) for row in selected]
    if actual_updates != expected_updates:
        raise RuntimeError(
            f"{task_name}: expected updates {expected_updates}, "
            f"got {actual_updates}"
        )

    fields = (
        "near_miss_rate",
        "near_miss_ci_low",
        "near_miss_ci_high",
        "best_quality_q25",
        "best_quality_median",
        "best_quality_q75",
    )
    for row in selected:
        if int(row["training_seed"]) != int(SEEDS[task_name]):
            raise RuntimeError(f"unexpected training seed in {task_name}")
        if int(row["rollouts"]) != ROLLOUTS_PER_CHECKPOINT:
            raise RuntimeError(f"unexpected rollout count in {task_name}")
        values = np.asarray([float(row[field]) for field in fields])
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite summary value in {task_name}")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"summary value outside [0, 1] in {task_name}")
        near, near_low, near_high, q25, median, q75 = values
        if not near_low <= near <= near_high:
            raise RuntimeError(f"invalid Wilson interval in {task_name}")
        if not q25 <= median <= q75:
            raise RuntimeError(f"invalid interquartile range in {task_name}")
    return selected


def save_one_figure(
    task_name: str,
    metric_name: str,
    updates: np.ndarray,
    centre: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ylabel: str,
) -> Path:
    figure, axis = plt.subplots(
        figsize=(5.0, 3.5),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.fill_between(
        updates,
        lower,
        upper,
        color="black",
        alpha=0.15,
        linewidth=0,
        zorder=1,
    )
    axis.plot(
        updates,
        centre,
        color="black",
        linewidth=1.25,
        zorder=2,
    )
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel(ylabel)
    axis.set_xlim(float(updates[0]), float(updates[-1]))
    axis.set_ylim(0.0, 1.0)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    axis.grid(False)
    axis.tick_params(direction="out")

    png_path = OUTPUT_DIRECTORY / f"{task_name}_{metric_name}.png"
    figure.savefig(
        png_path,
        dpi=400,
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)
    return png_path


def save_figures(rows: list[dict]) -> list[Path]:
    observed_tasks = {row["task"] for row in rows}
    if observed_tasks != set(TASKS):
        raise RuntimeError(
            f"expected tasks {set(TASKS)}, got {observed_tasks}"
        )

    artifacts = []
    for task_name in TASKS:
        selected = validated_task_rows(rows, task_name)
        updates = np.asarray(
            [int(row["update"]) for row in selected],
            dtype=float,
        )

        artifacts.append(save_one_figure(
            task_name,
            "near_miss",
            updates,
            np.asarray([
                float(row["near_miss_rate"]) for row in selected
            ]),
            np.asarray([
                float(row["near_miss_ci_low"]) for row in selected
            ]),
            np.asarray([
                float(row["near_miss_ci_high"]) for row in selected
            ]),
            "Near-miss-region reach probability",
        ))
        artifacts.append(save_one_figure(
            task_name,
            "qstar",
            updates,
            np.asarray([
                float(row["best_quality_median"]) for row in selected
            ]),
            np.asarray([
                float(row["best_quality_q25"]) for row in selected
            ]),
            np.asarray([
                float(row["best_quality_q75"]) for row in selected
            ]),
            "Episode-best quality, Q*",
        ))
    if len(artifacts) != 6:
        raise RuntimeError(f"expected 6 figure files, got {len(artifacts)}")
    return artifacts


def write_endpoint_summary(rows: list[dict]) -> Path:
    output = OUTPUT_DIRECTORY / "checkpoint_endpoint_summary.csv"
    fields = (
        "task",
        "seed",
        "selection",
        "first_success_episode",
        "final_update",
        "rollouts_per_checkpoint",
        "initial_near_miss_rate",
        "initial_near_miss_ci_low",
        "initial_near_miss_ci_high",
        "final_near_miss_rate",
        "final_near_miss_ci_low",
        "final_near_miss_ci_high",
        "initial_qstar_q25",
        "initial_qstar_median",
        "initial_qstar_q75",
        "final_qstar_q25",
        "final_qstar_median",
        "final_qstar_q75",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task_name in TASKS:
            selected = validated_task_rows(rows, task_name)
            initial, final = selected[0], selected[-1]
            writer.writerow({
                "task": task_name,
                "seed": SEEDS[task_name],
                "selection": "slowest successful seed",
                "first_success_episode": EXPECTED[task_name]["episode"],
                "final_update": EXPECTED[task_name]["updates"],
                "rollouts_per_checkpoint": ROLLOUTS_PER_CHECKPOINT,
                "initial_near_miss_rate": initial["near_miss_rate"],
                "initial_near_miss_ci_low": initial["near_miss_ci_low"],
                "initial_near_miss_ci_high": initial["near_miss_ci_high"],
                "final_near_miss_rate": final["near_miss_rate"],
                "final_near_miss_ci_low": final["near_miss_ci_low"],
                "final_near_miss_ci_high": final["near_miss_ci_high"],
                "initial_qstar_q25": initial["best_quality_q25"],
                "initial_qstar_median": initial["best_quality_median"],
                "initial_qstar_q75": initial["best_quality_q75"],
                "final_qstar_q25": final["best_quality_q25"],
                "final_qstar_median": final["best_quality_median"],
                "final_qstar_q75": final["best_quality_q75"],
            })
    return output


def write_provenance(
    font_path: Path,
    resolved_family: str,
    figure_artifacts: list[Path],
    endpoint_summary: Path,
) -> Path:
    evaluation_directory = EXPERIMENT_DIRECTORY / "checkpoint_evaluation"
    inputs = {
        "experiment_manifest": EXPERIMENT_DIRECTORY / "manifest.json",
        "evaluation_design": evaluation_directory / "evaluation_design.csv",
        "checkpoint_inventory": (
            evaluation_directory / "checkpoint_inventory.csv"
        ),
        "checkpoint_rollouts": (
            evaluation_directory / "checkpoint_rollouts.csv"
        ),
        "checkpoint_summary": evaluation_directory / "checkpoint_summary.csv",
        "assembly_sources": evaluation_directory / "assembly_sources.json",
    }
    sources = {
        Path(experiment.__file__).name: Path(experiment.__file__).resolve(),
        "evaluate_three_task_checkpoints.py": (
            Path(__file__).with_name("evaluate_three_task_checkpoints.py")
        ),
        "evaluate_torus_slowest_checkpoints.py": (
            Path(__file__).with_name("evaluate_torus_slowest_checkpoints.py")
        ),
        "assemble_slowest_seed_checkpoint_evaluation.py": (
            Path(__file__).with_name(
                "assemble_slowest_seed_checkpoint_evaluation.py"
            )
        ),
        Path(__file__).name: Path(__file__).resolve(),
    }
    artifacts = [*figure_artifacts, endpoint_summary]
    payload = {
        "font": {
            "requested_family": FONT_FAMILY,
            "resolved_family": resolved_family,
            "resolved_path": str(font_path),
            "sha256": sha256(font_path),
        },
        "figure_artifact_count": len(figure_artifacts),
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
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    font_path, resolved_family = configure_style()
    summary_path = (
        EXPERIMENT_DIRECTORY
        / "checkpoint_evaluation"
        / "checkpoint_summary.csv"
    )
    rows = read_csv(summary_path)
    figure_artifacts = save_figures(rows)
    endpoint_summary = write_endpoint_summary(rows)
    provenance = write_provenance(
        font_path,
        resolved_family,
        figure_artifacts,
        endpoint_summary,
    )
    for path in [*figure_artifacts, endpoint_summary, provenance]:
        print(path)


if __name__ == "__main__":
    main()
