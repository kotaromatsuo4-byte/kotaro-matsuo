"""Generate the eight compact figures used by the five-bar meeting deck.

Every plotted number is read from an existing CSV/JSON result.  The figures
deliberately keep experiments with different mechanics/candidate definitions in
separate panels; they are not intended as one cross-version leaderboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"

# Stable semantic palette for the entire deck.
RANDOM = "#7A7A7A"
HEURISTIC = "#E67E22"
LEARNED = "#277DA1"
LEARNED_LIGHT = "#78B7D0"
SHUFFLED = "#7B2CBF"
ORACLE = "#161616"
GRID = "#D9D9D9"
MUTED = "#5B5B5B"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Yu Gothic", "Meiryo", "BIZ UDGothic", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "figure.titlesize": 19,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#333333")
    ax.set_axisbelow(True)


def source_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.995, 0.012, f"出典: {text}", ha="right", va="bottom", fontsize=8.5, color=MUTED)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def add_bar_values(ax: plt.Axes, bars, *, fmt: str = ".2f", dy: float | None = None) -> None:
    top = max((float(bar.get_height()) for bar in bars), default=1.0)
    offset = dy if dy is not None else max(0.015 * top, 0.012)
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )


def bars_with_seed_dots(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str,
    ylim: tuple[float, float],
    value_fmt: str = ".2f",
) -> None:
    x = np.arange(len(groups), dtype=float)
    means = np.array([np.mean(values) for values in groups], dtype=float)
    bars = ax.bar(x, means, width=0.62, color=colors, alpha=0.85, zorder=2)
    for xpos, values in zip(x, groups):
        offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            xpos + offsets,
            values,
            s=42,
            facecolor=WHITE,
            edgecolor="#222222",
            linewidth=1.1,
            zorder=4,
        )
    add_bar_values(ax, bars, fmt=value_fmt)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    style_axis(ax)


def figure_main5(outdir: Path) -> None:
    path = RESULT / "heldout_scale020_comparison.csv"
    data = pd.read_csv(path)
    seeds = data[data["row_type"] == "seed"].copy()
    order = [
        "heldout_scale020_random",
        "heldout_scale020_gp_ucb",
        "heldout_scale020_gp_posterior",
    ]
    labels = ["random", "GP-UCB", "GP posterior"]
    colors = [RANDOM, LEARNED, LEARNED_LIGHT]
    groups = {metric: [seeds.loc[seeds.result_name == name, metric].to_numpy(float) for name in order]
              for metric in ["final_cells", "discovery_auc", "walltime_seconds"]}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2))
    fig.suptitle("単一摂動GPは held-out で random を上回らなかった", y=0.98, fontweight="bold")
    bars_with_seed_dots(
        axes[0], groups["final_cells"], labels, colors,
        ylabel="最終elite cell数", ylim=(0, 2.45), value_fmt=".1f",
    )
    axes[0].set_title("探索結果")
    bars_with_seed_dots(
        axes[1], groups["discovery_auc"], labels, colors,
        ylabel="Discovery AUC [cell]", ylim=(0.8, 2.0), value_fmt=".3f",
    )
    axes[1].set_title("予算内の発見速度")
    bars_with_seed_dots(
        axes[2], groups["walltime_seconds"], labels, colors,
        ylabel="wall time [s]", ylim=(0, 125), value_fmt=".1f",
    )
    axes[2].set_title("追加計算コスト")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
    fig.text(0.02, 0.035, "B=1200, held-out seeds 100–104 (n=5); 同一設定内の比較", fontsize=10.5, color=MUTED)
    source_footer(fig, path.name)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_main5")


def figure_reachability(outdir: Path) -> None:
    coverage_path = RESULT / "canonical_coverage_v31.csv"
    coverage = pd.read_csv(coverage_path)
    random_name = "large_random_transport_pool_b10000_v31"
    heuristic_name = "large_transport_heuristic_b10000_v31"
    random_values = coverage.loc[coverage.result_name == random_name, "canonical_coverage_count"].to_numpy(float)
    heuristic_values = coverage.loc[coverage.result_name == heuristic_name, "canonical_coverage_count"].to_numpy(float)

    event_path = RESULT / "large_transport_heuristic_b10000_v31" / "seed_401" / "discovery_events.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    event_rows = []
    for event in events:
        if not event["perturbation_child_state_ids"]:
            continue
        start_depth = event["path_step_types"].index("perturbation")
        event_rows.append((event["cell_index"], start_depth, event["path_perturbation_count"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), gridspec_kw={"width_ratios": [0.95, 1.25]})
    fig.suptitle("6 cell到達には、経路途中からの長い持続摂動列が効いた", y=0.98, fontweight="bold")

    ax = axes[0]
    for i in range(len(random_values)):
        ax.plot([0, 1], [random_values[i], heuristic_values[i]], color="#BDBDBD", linewidth=1.4, zorder=1)
    ax.scatter(np.zeros_like(random_values), random_values, s=75, color=RANDOM, edgecolor=WHITE, linewidth=0.8, zorder=3)
    ax.scatter(np.ones_like(heuristic_values), heuristic_values, s=75, color=HEURISTIC, edgecolor=WHITE, linewidth=0.8, zorder=3)
    ax.set_xticks([0, 1], ["random pool", "80/20 persistent"])
    ax.set_ylabel("canonical coverage / 6")
    ax.set_ylim(0, 6.6)
    ax.set_yticks(range(0, 7))
    ax.set_title("同一seed・B=10000 (n=3)")
    ax.text(0, np.mean(random_values) + 0.25, f"mean {np.mean(random_values):.1f}", ha="center", color=RANDOM, fontweight="bold")
    ax.text(1, np.mean(heuristic_values) + 0.25, f"mean {np.mean(heuristic_values):.1f}", ha="center", color=HEURISTIC, fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    ax.axvspan(-2.5, 2.5, color="#EEEEEE", zorder=0)
    for cell, start, count in event_rows:
        color = HEURISTIC if start > 0 else RANDOM
        ax.scatter(start, count, s=95, color=color, edgecolor=WHITE, linewidth=1.0, zorder=3)
        ax.annotate(f"raw c{cell}", (start, count), xytext=(7, 5), textcoords="offset points", fontsize=10.5)
    ax.text(0, 26.5, "root-start", ha="center", va="top", fontsize=10, color=MUTED)
    ax.annotate(
        "19–25回の経路は\ndepth 41から開始",
        xy=(41, 21), xytext=(22, 25.5),
        arrowprops={"arrowstyle": "->", "color": HEURISTIC, "lw": 1.5},
        color=HEURISTIC, fontsize=11, ha="center",
    )
    ax.set_xlabel("最初の摂動までの経路depth")
    ax.set_ylabel("発見経路中の摂動回数")
    ax.set_xlim(-3, 48)
    ax.set_ylim(0, 28)
    ax.set_title("seed 401: 各新規raw cellの発見経路")
    style_axis(ax)

    fig.text(0.02, 0.035, "左: canonical 6への到達性　右: root摂動だけでなくtrunk途中の連続摂動が主要因", fontsize=10.5, color=MUTED)
    source_footer(fig, f"{coverage_path.name}; seed_401/discovery_events.json")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_reachability")


def figure_angle_offline(outdir: Path) -> None:
    path = RESULT / "macro2_angle_panel_p18_seed623_625_aggregate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    truth = data["group_conditioned_truth"]
    models = data["models"]
    labels = [
        "uniform\nangle", "fixed\nθ=0", "fixed\nθ=π", "raw GP",
        "angle-aware\nGP", "score\nshuffled", "oracle",
    ]
    values = np.array(
        [
            truth["uniform_random_new_rate"],
            truth["fixed_theta_zero_new_rate"],
            truth["fixed_theta_pi_new_rate"],
            models["raw"]["group_top1_new_rate"],
            models["angle_aware"]["group_top1_new_rate"],
            models["feature_shuffled"]["group_top1_new_rate"],
            truth["oracle_top1_new_rate"],
        ]
    )
    colors = [RANDOM, HEURISTIC, HEURISTIC, LEARNED_LIGHT, LEARNED, SHUFFLED, ORACLE]
    hatches = ["", "//", "xx", "", "", "", ""]

    fig, ax = plt.subplots(figsize=(12.6, 5.25))
    fig.suptitle("K=2角度: 類似方向 θ=0 は最良ではなく、GP信号も限定的", y=0.98, fontweight="bold")
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.68, color=colors, alpha=0.88, zorder=2)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_edgecolor("#4A4A4A")
    add_bar_values(ax, bars, fmt=".3f", dy=0.010)
    ax.set_xticks(x, labels)
    ax.set_ylabel("top-1候補の新規cell率")
    ax.set_ylim(0, 0.43)
    ax.set_title("offline counterfactual angle panel")
    style_axis(ax)
    ax.annotate(
        "θ=0よりθ=πが高い",
        xy=(2, values[2]), xytext=(1.15, 0.30),
        arrowprops={"arrowstyle": "->", "color": HEURISTIC, "lw": 1.5},
        color=HEURISTIC, fontsize=11.5,
    )
    fig.text(0.02, 0.035, "3 seeds × 18 q₁ groups; 同一group内で角度候補を反実仮想評価", fontsize=10.5, color=MUTED)
    source_footer(fig, path.name)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_angle_offline")


def figure_main6_main7(outdir: Path) -> None:
    main6_path = RESULT / "main6_paired16_b10000_canonical.csv"
    main7_path = RESULT / "main7_b10000_aggregate.csv"
    d6 = pd.read_csv(main6_path)
    d7 = pd.read_csv(main7_path)

    order6 = [
        "main6_paired16_b10000_independent_random",
        "main6_paired16_b10000_fixed_transport",
        "main6_paired16_b10000_learned_angle_gp",
        "main6_paired16_b10000_learned_angle_gp_shuffled",
    ]
    order7 = ["independent_random", "fixed_transport", "joint_macro_gp", "joint_macro_gp_shuffled"]
    labels = ["independent", "fixed", "GP", "shuffled"]
    colors = [RANDOM, HEURISTIC, LEARNED, SHUFFLED]
    groups6 = [d6.loc[d6.result_name == name, "canonical_coverage_count"].to_numpy(float) for name in order6]
    groups7 = [d7.loc[d7.selector == name, "canonical_cells"].to_numpy(float) for name in order7]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), sharey=True)
    fig.suptitle("K=2のonline最適化は再現的な優位を示さなかった", y=0.98, fontweight="bold")
    bars_with_seed_dots(
        axes[0], groups6, labels, colors,
        ylabel="最終canonical cell数", ylim=(0, 4.5), value_fmt=".2f",
    )
    axes[0].set_title("main6: 第2角度のみ選択")
    bars_with_seed_dots(
        axes[1], groups7, labels, colors,
        ylabel="", ylim=(0, 4.5), value_fmt=".2f",
    )
    axes[1].set_title("main7: 親・第1方向・角度をjoint選択")
    axes[0].text(1.5, 4.15, "全方式で同一", ha="center", color=MUTED, fontsize=11)
    axes[1].text(2.5, 4.15, "GP優位はseed依存", ha="center", color=MUTED, fontsize=11)
    for ax in axes:
        ax.tick_params(axis="x", rotation=10)
    fig.text(0.02, 0.035, "各panel内: B=10000, seeds 600–602 (n=3)。main6とmain7の絶対値は直接比較しない", fontsize=10.5, color=MUTED)
    source_footer(fig, f"{main6_path.name}; {main7_path.name}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_main6_main7")


def figure_main8_ablation(outdir: Path) -> None:
    path = RESULT / "main8_ablation_canonical_audit.csv"
    data = pd.read_csv(path)
    k1_name = "main8_ablation_k1_random_b3000"
    vk_name = "main8_ablation_variable_k_random_b3000"
    k1 = data.loc[data.result_name == k1_name, "canonical_coverage_count"].to_numpy(float)
    variable = data.loc[data.result_name == vk_name, "canonical_coverage_count"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.8, 5.45))
    fig.suptitle("可変K action spaceだけで6 cell到達が 0/3 → 3/3", y=0.98, fontweight="bold")
    for i, (before, after) in enumerate(zip(k1, variable)):
        offset = (i - 1) * 0.035
        ax.plot([0 + offset, 1 + offset], [before, after], color="#B8B8B8", linewidth=2.0, zorder=1)
        ax.scatter(0 + offset, before, s=100, color=RANDOM, edgecolor=WHITE, linewidth=1.0, zorder=3)
        ax.scatter(1 + offset, after, s=100, color=HEURISTIC, edgecolor=WHITE, linewidth=1.0, zorder=3)
        ax.text(1.07, after + (i - 1) * 0.20, f"seed {700+i}", fontsize=10.5, va="center")
    ax.set_xticks([0, 1], ["K=1 random", "K∼U[1,24]\npersistent random"])
    ax.set_ylabel("最終canonical coverage / 6")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 6.65)
    ax.set_yticks(range(0, 7))
    ax.text(0, np.mean(k1) + 0.35, f"mean {np.mean(k1):.2f}", ha="center", color=RANDOM, fontweight="bold")
    ax.text(1, 6.26, "3/3 reached", ha="center", color=HEURISTIC, fontweight="bold")
    style_axis(ax)
    fig.text(0.02, 0.035, "B=3000, paired seeds 700–702; 変えたのは持続回数Kの探索範囲", fontsize=10.5, color=MUTED)
    source_footer(fig, path.name)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_main8_ablation")


def figure_main8_offline_online(outdir: Path) -> None:
    offline_path = RESULT / "persistent_k_panel_seed700_702" / "aggregate_summary.json"
    online_path = RESULT / "main8_group_heldout_b3000_runs.csv"
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    truth = offline["group_truth"]
    models = offline["models"]
    offline_labels = ["uniform K", "basis-free GP", "shuffled", "oracle"]
    offline_values = np.array(
        [
            truth["uniform_random_new_rate"],
            models["basis_free_gp"]["group_top1_new_rate"],
            models["basis_free_gp_score_shuffled"]["group_top1_new_rate"],
            truth["oracle_top1_new_rate"],
        ]
    )
    colors = [RANDOM, LEARNED, SHUFFLED, ORACLE]

    online = pd.read_csv(online_path)
    online_order = ["variable_k_random", "variable_k_gp_group_argmax", "variable_k_gp_group_argmax_shuffled"]
    online_labels = ["random", "group GP", "shuffled"]
    success = np.array(
        [online.loc[online.selector == name, "canonical_t6_reached"].astype(float).mean() for name in online_order]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    fig.suptitle("Kの順位はofflineで学べても、online高速化にはつながらない", y=0.98, fontweight="bold")

    x = np.arange(len(offline_values))
    bars = axes[0].bar(x, offline_values, color=colors, width=0.68, alpha=0.88, zorder=2)
    add_bar_values(axes[0], bars, fmt=".3f", dy=0.016)
    axes[0].set_xticks(x, offline_labels)
    axes[0].set_ylabel("top-1新規cell率")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("offline: 同一(q,v)で全K評価 (18 groups)")
    style_axis(axes[0])

    x = np.arange(len(success))
    bars = axes[1].bar(x, success, color=[RANDOM, LEARNED, SHUFFLED], width=0.68, alpha=0.88, zorder=2)
    for bar, count in zip(bars, (success * 10).astype(int)):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035,
            f"{count}/10", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    axes[1].set_xticks(x, online_labels)
    axes[1].set_ylabel("B=3000でcanonical 6到達率")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("online held-out: group-K selector (10 seeds)")
    style_axis(axes[1])
    axes[1].annotate(
        "learned < random",
        xy=(1, success[1]), xytext=(0.38, 0.77),
        arrowprops={"arrowstyle": "->", "color": LEARNED, "lw": 1.5},
        color=LEARNED, fontsize=11.5,
    )
    fig.text(0.02, 0.035, "左右は異なる評価量。offline learnability と閉ループ探索性能を区別", fontsize=10.5, color=MUTED)
    source_footer(fig, f"persistent_k_panel_seed700_702/aggregate_summary.json; {online_path.name}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_main8_offline_online")


def figure_main9_timing(outdir: Path) -> None:
    path = RESULT / "main9_model_analysis" / "analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    macro_steps = {int(key): float(value) for key, value in agg["median_step_at_macro_count"].items()}
    t6 = float(agg["median_canonical_t6_step"])
    stable_macro = int(agg["first_sustained_prevalence_gain_macros"]["knn"])
    reject_macro = int(agg["strict_first_rejection_macros"]["knn"])
    safety_macro = 320
    points = [
        (0, t6, "random T6中央値", f"{t6:.0f} step\n14 macros", RANDOM),
        (1, macro_steps[stable_macro], "予測改善が持続", f"{macro_steps[stable_macro]:.0f} step\n{stable_macro} macros", LEARNED),
        (1, macro_steps[reject_macro], "strict kNN初棄却", f"{macro_steps[reject_macro]:.0f} step\n{reject_macro} macros", LEARNED),
        (1, macro_steps[safety_macro], "安全filterの兆候*", f"{macro_steps[safety_macro]:.0f} step\n{safety_macro} macros", LEARNED),
    ]

    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    fig.suptitle("random探索が、分類filterを安全に使える時期より先に解く", y=0.98, fontweight="bold")
    ax.hlines([0, 1], xmin=400, xmax=22000, colors="#BDBDBD", linewidth=2.0, zorder=0)
    for lane, x, title, value, color in points:
        ax.scatter(x, lane, s=125, color=color, edgecolor=WHITE, linewidth=1.2, zorder=3)
        if lane == 0:
            ax.annotate(f"{title}\n{value}", (x, lane), xytext=(0, -47), textcoords="offset points", ha="center", va="top", fontsize=11.5)
        else:
            ax.annotate(f"{title}\n{value}", (x, lane), xytext=(0, 18), textcoords="offset points", ha="center", va="bottom", fontsize=11.0)
    ax.axvline(t6, color=RANDOM, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(400, 22000)
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000], ["500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xlabel("mechanics step（log scale）")
    ax.set_yticks([0, 1], ["online探索", "offline model診断"])
    ax.set_ylim(-0.65, 1.72)
    style_axis(ax, grid_axis="x")
    fig.text(0.02, 0.035, "* targeted LOCOで初めて弱い安全信号。3-seed random corpus; 時刻はmacro数の中央値対応", fontsize=10.5, color=MUTED)
    source_footer(fig, "main9_model_analysis/analysis.json")
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    save_figure(fig, outdir, "fig_main9_timing")


def figure_main9_snapshot(outdir: Path) -> None:
    metrics_path = RESULT / "main9_snapshot_filter_evaluation" / "aggregate" / "snapshot_metrics_aggregate.csv"
    filter_path = RESULT / "main9_snapshot_filter_evaluation" / "aggregate" / "filter_metrics_aggregate.csv"
    metrics = pd.read_csv(metrics_path)
    filters = pd.read_csv(filter_path)
    targets = np.array(sorted(metrics.target_cells.unique()), dtype=int)

    support = np.array(
        [metrics.loc[metrics.target_cells == target, "train_macros_mean"].iloc[0] for target in targets], dtype=float
    )
    known_best = np.array(
        [metrics.loc[metrics.target_cells == target, "known_top1_accuracy_mean"].max() for target in targets], dtype=float
    )
    open_best = np.array(
        [metrics.loc[metrics.target_cells == target, "open_world_top1_accuracy_mean"].max() for target in targets], dtype=float
    )
    best_model = []
    for target in targets:
        rows = metrics.loc[metrics.target_cells == target]
        best = rows["known_top1_accuracy_mean"].max()
        tied = rows.loc[np.isclose(rows["known_top1_accuracy_mean"], best), "model_kind"].tolist()
        if len(tied) > 1:
            best_model.append("all tied")
        else:
            best_model.append({"knn": "kNN", "extra_trees": "ExtraTrees", "random_forest": "RF"}.get(tied[0], tied[0]))
    strict = filters.loc[filters.policy == "strict_all_known"].groupby("target_cells")["actual_reject_fraction_mean"].max()
    strict_values = np.array([strict.loc[target] for target in targets], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.2))
    fig.suptitle("2–4 cell時点では教師が少なく、安全filterは一度も発火しない", y=0.98, fontweight="bold")

    bars = axes[0].bar(targets, support, color=RANDOM, width=0.58, alpha=0.88, zorder=2)
    add_bar_values(axes[0], bars, fmt=".1f", dy=0.12)
    axes[0].set_xlabel("発見済みcell数")
    axes[0].set_ylabel("独立教師macro数")
    axes[0].set_xticks(targets)
    axes[0].set_ylim(0, 6.2)
    axes[0].set_title("online training support")
    style_axis(axes[0])

    axes[1].plot(targets, known_best, marker="o", markersize=8, linewidth=2.2, color=LEARNED, label="known-cell top-1")
    axes[1].plot(targets, open_best, marker="s", markersize=7, linewidth=2.0, linestyle="--", color=LEARNED_LIGHT, label="open-world top-1")
    for x, y, model in zip(targets, known_best, best_model):
        axes[1].text(x, y + 0.038, f"{y:.3f}\n({model})", ha="center", fontsize=9.5)
    axes[1].set_xlabel("発見済みcell数")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xticks(targets)
    axes[1].set_ylim(0, 0.72)
    axes[1].set_title("凍結snapshotの反実仮想panel")
    axes[1].legend(frameon=False, loc="lower center")
    style_axis(axes[1])

    axes[2].plot(targets, strict_values, marker="X", markersize=11, linewidth=1.8, color=LEARNED)
    axes[2].axhline(0, color="#888888", linewidth=1.0)
    axes[2].set_xlabel("発見済みcell数")
    axes[2].set_ylabel("strict reject fraction")
    axes[2].set_xticks(targets)
    axes[2].set_ylim(-0.005, 0.105)
    axes[2].set_title("production safety gate")
    axes[2].text(3, 0.055, "0 / all candidates\n→ 選択圧なし", ha="center", color=LEARNED, fontsize=12, fontweight="bold")
    style_axis(axes[2])

    fig.text(0.02, 0.035, "3 seeds × 3 snapshots × 128候補; 候補rolloutは学習/archiveへ戻さない", fontsize=10.5, color=MUTED)
    source_footer(fig, "main9_snapshot_filter_evaluation/aggregate/*.csv")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save_figure(fig, outdir, "fig_main9_snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Destination for PNG and PDF files (default: meeting_materials/figures).",
    )
    args = parser.parse_args()
    configure_matplotlib()

    builders = [
        figure_main5,
        figure_reachability,
        figure_angle_offline,
        figure_main6_main7,
        figure_main8_ablation,
        figure_main8_offline_online,
        figure_main9_timing,
        figure_main9_snapshot,
    ]
    for builder in builders:
        builder(args.output_directory)
    print(f"Generated {len(builders)} PNG/PDF figure pairs in {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
