"""Plot the saved main9 model-learning and filter evaluations.

The script is intentionally read-only with respect to the evaluation outputs.
It creates two figures in ``result/main9_model_analysis``:

* ``learning_curves_t6.png``
* ``filter_tradeoff.png``

Only Python's standard library and matplotlib are required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_STRICT_DIR = HERE / "result" / "main9_forest_eval_b20000"
DEFAULT_LOOSE_DIR = HERE / "result" / "main9_forest_eval_b20000_loose_loco"
DEFAULT_FEATURE_ABLATION_DIR = HERE / "result" / "main9_feature_ablation_knn_loco"
DEFAULT_OUTPUT_DIR = HERE / "result" / "main9_model_analysis"

T6_MACROS = (11, 14, 17)  # minimum, median, maximum across seeds 800--802

MODEL_ORDER = ("random_forest", "extra_trees", "knn", "prevalence")
MODEL_LABELS = {
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "knn": "kNN",
    "prevalence": "Prevalence",
}
MODEL_COLORS = {
    "random_forest": "#0072B2",
    "extra_trees": "#009E73",
    "knn": "#D55E00",
    "prevalence": "#5E5E5E",
}
MODEL_MARKERS = {
    "random_forest": "o",
    "extra_trees": "s",
    "knn": "^",
    "prevalence": "D",
}


def _float(value: str | None) -> float:
    """Parse a possibly empty CSV value as a float or NaN."""

    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluation CSV has no rows: {path}")
    return rows


def _model_rows(
    rows: Iterable[dict[str, str]], model: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("model_kind") == model]
    return sorted(selected, key=lambda row: int(row["train_macros"]))


def _set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
        }
    )


def plot_learning_curves(strict_rows: list[dict[str, str]], output: Path) -> None:
    """Plot held-out accuracy and macro-F1 with the T6 teacher-count band."""

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    metrics = (
        ("known_accuracy_mean", "known_accuracy_std", "Accuracy"),
        ("known_macro_f1_mean", "known_macro_f1_std", "Macro-F1"),
    )

    for axis, (mean_key, std_key, title) in zip(axes, metrics):
        for model in MODEL_ORDER:
            rows = _model_rows(strict_rows, model)
            if not rows:
                continue
            x = np.asarray([int(row["train_macros"]) for row in rows], dtype=float)
            y = np.asarray([_float(row.get(mean_key)) for row in rows])
            std = np.asarray([_float(row.get(std_key)) for row in rows])
            valid = np.isfinite(x) & np.isfinite(y)
            x, y, std = x[valid], y[valid], std[valid]
            linestyle = "--" if model == "prevalence" else "-"
            axis.plot(
                x,
                y,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=linestyle,
                label=MODEL_LABELS[model],
                zorder=3,
            )
            finite_std = np.isfinite(std)
            if np.any(finite_std):
                axis.fill_between(
                    x[finite_std],
                    np.clip(y[finite_std] - std[finite_std], 0.0, 1.0),
                    np.clip(y[finite_std] + std[finite_std], 0.0, 1.0),
                    color=MODEL_COLORS[model],
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

        t6_min, t6_median, t6_max = T6_MACROS
        axis.axvspan(t6_min, t6_max, color="#CC79A7", alpha=0.14, zorder=0)
        axis.axvline(t6_median, color="#A64D79", linewidth=1.4, zorder=2)
        axis.set_xscale("log", base=2)
        axis.set_xlim(3.5, 365)
        axis.set_ylim(0.0, 0.56 if title == "Accuracy" else 0.50)
        axis.set_title(title)
        axis.set_xlabel("Independent training macros")
        axis.set_ylabel(f"Held-out {title.lower()}")
        # 256 and 320 are too close on a base-2 log axis to label both clearly.
        axis.set_xticks([4, 8, 16, 32, 64, 128, 320])
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.minorticks_off()

    axes[0].annotate(
        "T6 reached with 11--17 macros\n(median 14)",
        xy=(14, 0.515),
        xytext=(26, 0.535),
        ha="left",
        va="top",
        fontsize=9,
        color="#7A315D",
        arrowprops={"arrowstyle": "->", "color": "#7A315D", "lw": 1.0},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#CC79A7", alpha=0.20, edgecolor="none"))
    labels.append("T6 teacher-count range")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.suptitle("Main9 basin-classification learning curves", y=1.12, fontsize=14)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    fig.savefig(
        output,
        metadata={
            "Title": "Main9 basin-classification learning curves with T6 teacher count",
            "Description": (
                "Strict chronological evaluation; curves are means over seeds and "
                "bands are plus/minus one seed standard deviation."
            ),
        },
    )
    plt.close(fig)


def plot_filter_tradeoff(
    strict_rows: list[dict[str, str]],
    loose_rows: list[dict[str, str]],
    feature_ablation_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """Plot the measured LOCO trade-off and the strict/loose rejection profiles.

    The base strict saved evaluation explicitly used ``skip_loco=true``.  Its
    rejection profile is shown on the right.  A later targeted feature ablation
    measured LOCO for context-full strict group-kNN, so that measured point is
    overlaid on both panels instead of fabricating other strict coordinates.
    """

    fig = plt.figure(figsize=(11.5, 5.4))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.42)
    trade_axis = fig.add_subplot(grid[0, 0])
    reject_axis = fig.add_subplot(grid[0, 1])

    training_sizes = sorted(
        {
            int(row["train_macros"])
            for row in loose_rows
            if row.get("model_kind") in MODEL_ORDER[:-1]
        }
    )
    norm = mpl.colors.LogNorm(vmin=min(training_sizes), vmax=max(training_sizes))
    cmap = mpl.colormaps["viridis"]

    for model in MODEL_ORDER[:-1]:
        rows = _model_rows(loose_rows, model)
        points: list[tuple[int, float, float]] = []
        for row in rows:
            x = _float(row.get("pooled_duplicate_rejection_rate"))
            y = _float(row.get("pooled_loco_unknown_retention"))
            if math.isfinite(x) and math.isfinite(y):
                points.append((int(row["train_macros"]), x, y))
        if not points:
            continue

        trade_axis.plot(
            [point[1] for point in points],
            [point[2] for point in points],
            color=MODEL_COLORS[model],
            alpha=0.58,
            linewidth=1.4,
            zorder=1,
        )
        for size, x, y in points:
            trade_axis.scatter(
                x,
                y,
                s=64,
                marker=MODEL_MARKERS[model],
                facecolor=cmap(norm(size)),
                edgecolor=MODEL_COLORS[model],
                linewidth=1.2,
                zorder=3,
            )

        # Label the trajectory endpoints.  Degenerate trajectories (all points
        # identical) get a compact range label rather than overlapping labels.
        first, last = points[0], points[-1]
        if np.allclose(
            [[point[1], point[2]] for point in points],
            [[first[1], first[2]]] * len(points),
        ):
            label = f"{MODEL_LABELS[model]}: {first[0]}--{last[0]}"
            xy = (first[1], first[2])
        else:
            label = f"{MODEL_LABELS[model]}: {last[0]}"
            xy = (last[1], last[2])
        offset = {
            "random_forest": (7, -12),
            # Keep the degenerate Extra Trees label clear of the subplot title.
            "extra_trees": (36, -6),
            "knn": (-7, -13),
        }[model]
        trade_axis.annotate(
            label,
            xy=xy,
            xytext=offset,
            textcoords="offset points",
            ha="right" if model == "knn" else "left",
            va="top" if model == "extra_trees" else "bottom",
            color=MODEL_COLORS[model],
            fontsize=8.5,
        )

    # The desired operating region is the *intersection* of both requirements.
    trade_axis.add_patch(
        Rectangle(
            (0.05, 0.95),
            0.255 - 0.05,
            1.012 - 0.95,
            facecolor="#009E73",
            edgecolor="none",
            alpha=0.08,
            zorder=0,
        )
    )
    trade_axis.axvline(0.05, color="#777777", linestyle=":", linewidth=1.0)
    trade_axis.axhline(0.95, color="#777777", linestyle=":", linewidth=1.0)
    trade_axis.text(
        0.174,
        0.996,
        "target region",
        fontsize=8.5,
        color="#555555",
        ha="center",
        va="top",
    )

    targeted_rows = [
        row
        for row in feature_ablation_rows
        if row.get("feature_set") == "q_context_v_k"
        and row.get("threshold_profile") == "strict"
        and row.get("model_kind") == "knn"
        and int(row.get("train_macros", "-1")) == 320
    ]
    targeted_point: tuple[float, float] | None = None
    if targeted_rows:
        targeted_x = _float(targeted_rows[0].get("pooled_duplicate_rejection_rate"))
        targeted_y = _float(targeted_rows[0].get("pooled_loco_unknown_retention"))
        useful_fraction = _float(targeted_rows[0].get("useful_seed_fraction"))
        if math.isfinite(targeted_x) and math.isfinite(targeted_y):
            targeted_point = (targeted_x, targeted_y)
            trade_axis.scatter(
                targeted_x,
                targeted_y,
                s=210,
                marker="*",
                facecolor="#CC79A7",
                edgecolor="#6B2F54",
                linewidth=1.1,
                zorder=7,
            )
            useful_text = (
                f"; gate {round(useful_fraction * 3):d}/3 seeds"
                if math.isfinite(useful_fraction)
                else ""
            )
            trade_axis.annotate(
                "Context-full strict group-kNN\n"
                f"320 macros: {targeted_x:.1%} / {targeted_y:.1%}{useful_text}",
                xy=(targeted_x, targeted_y),
                xytext=(0.085, 0.944),
                textcoords="data",
                ha="left",
                va="top",
                fontsize=8.3,
                color="#6B2F54",
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#6B2F54",
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 5,
                },
                zorder=8,
            )
    trade_axis.set_xlim(-0.012, 0.255)
    trade_axis.set_ylim(0.74, 1.012)
    trade_axis.set_xlabel("Duplicate rejection rate")
    trade_axis.set_ylabel("LOCO unknown retention")
    trade_axis.set_title("LOCO trade-off + targeted strict point")
    trade_axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    trade_axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=trade_axis,
        pad=0.025,
        fraction=0.055,
    )
    colorbar.ax.set_title("Macros", fontsize=9, pad=6)
    colorbar.set_ticks(training_sizes)
    colorbar.set_ticklabels([str(size) for size in training_sizes])

    for model in MODEL_ORDER[:-1]:
        for policy, rows, linestyle, alpha in (
            ("strict", strict_rows, "--", 0.72),
            ("loose", loose_rows, "-", 1.0),
        ):
            selected = _model_rows(rows, model)
            if not selected:
                continue
            x = np.asarray([int(row["train_macros"]) for row in selected])
            y = np.asarray(
                [_float(row.get("pooled_duplicate_rejection_rate")) for row in selected]
            )
            valid = np.isfinite(y)
            reject_axis.plot(
                x[valid],
                y[valid],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model] if policy == "loose" else None,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=1.6,
                markersize=4.5,
            )

    reject_axis.set_xscale("log", base=2)
    reject_axis.set_xlim(3.5, 365)
    reject_axis.set_ylim(-0.012, 0.255)
    reject_axis.set_xticks([4, 8, 16, 32, 64, 128, 320])
    reject_axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    reject_axis.minorticks_off()
    reject_axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    reject_axis.set_xlabel("Independent training macros")
    reject_axis.set_ylabel("Duplicate rejection rate")
    reject_axis.set_title("Strict vs loose rejection")
    reject_axis.text(
        0.02,
        0.97,
        "Base strict run used skip_loco=true;\nstar has targeted context-full LOCO.",
        transform=reject_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    if targeted_point is not None:
        reject_axis.scatter(
            320,
            targeted_point[0],
            s=125,
            marker="*",
            facecolor="#CC79A7",
            edgecolor="#6B2F54",
            linewidth=1.0,
            zorder=7,
        )

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle="-",
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER[:-1]
    ]
    policy_handles = [
        Line2D([0], [0], color="#555555", linestyle="--", label="Strict"),
        Line2D([0], [0], color="#555555", linestyle="-", label="Loose + LOCO"),
    ]
    fig.legend(
        handles=model_handles + policy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.suptitle("Main9 conservative-filter trade-off", y=1.09, fontsize=14)
    fig.subplots_adjust(top=0.84)
    fig.savefig(
        output,
        metadata={
            "Title": "Main9 duplicate rejection versus LOCO unknown retention",
            "Description": (
                "Loose evaluation supplies the main LOCO trajectory. A targeted "
                "context-full strict group-kNN evaluation at 320 macros is overlaid; "
                "the base strict evaluation otherwise skipped LOCO."
            ),
        },
    )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-dir",
        type=Path,
        default=DEFAULT_STRICT_DIR,
        help="Directory containing the strict aggregate_metrics.csv",
    )
    parser.add_argument(
        "--loose-dir",
        type=Path,
        default=DEFAULT_LOOSE_DIR,
        help="Directory containing the loose+LOCO aggregate_metrics.csv",
    )
    parser.add_argument(
        "--feature-ablation-dir",
        type=Path,
        default=DEFAULT_FEATURE_ABLATION_DIR,
        help="Directory containing the targeted group-kNN LOCO aggregate CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for generated PNG files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    strict_rows = _read_rows(args.strict_dir / "aggregate_metrics.csv")
    loose_rows = _read_rows(args.loose_dir / "aggregate_metrics.csv")
    feature_ablation_csv = args.feature_ablation_dir / "aggregate_metrics.csv"
    feature_ablation_rows = (
        _read_rows(feature_ablation_csv) if feature_ablation_csv.is_file() else []
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    learning_path = args.output_dir / "learning_curves_t6.png"
    tradeoff_path = args.output_dir / "filter_tradeoff.png"
    _set_plot_style()
    plot_learning_curves(strict_rows, learning_path)
    plot_filter_tradeoff(
        strict_rows,
        loose_rows,
        feature_ablation_rows,
        tradeoff_path,
    )
    print(learning_path.resolve())
    print(tradeoff_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
