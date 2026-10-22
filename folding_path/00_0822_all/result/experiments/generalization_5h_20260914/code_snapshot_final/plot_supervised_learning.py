"""Plot saved supervised folding logs, without retraining or modifying checkpoints."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_run(run):
    run = Path(run)
    training = sorted(read_rows(run / "training.csv"), key=lambda row: int(row["round"]))
    evaluation = read_rows(run / "evaluation.csv")
    epochs = read_rows(run / "training_epochs.csv")
    if not training or not evaluation:
        raise ValueError("training.csv and evaluation.csv need completed records")
    update_at_round = {-1: 0}
    cumulative = 0
    for row in training:
        cumulative += int(row["updates"])
        update_at_round[int(row["round"])] = cumulative
    rounds = sorted({int(row["round"]) for row in evaluation} & update_at_round.keys())
    x = [update_at_round[r] for r in rounds]
    goals = sorted({float(row["goal"]) for row in evaluation})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    loss_ax, success_ax, data_ax, flat_ax, spread_ax, step_ax = axes.flat
    fig.suptitle("Supervised folding: imitation and held-out rollouts", fontsize=17, weight="bold")
    if epochs:
        loss_ax.plot([int(row["cumulative_updates"]) for row in epochs],
                     [float(row["loss"]) for row in epochs], color="#354e74", lw=1.6,
                     label="Epoch mean training loss")
    else:
        loss_ax.plot([update_at_round[int(row["round"])] for row in training],
                     [float(row["loss"]) for row in training], "o-", color="#354e74",
                     label="Round mean training loss")
        loss_ax.text(.97, .96, "Older run: epoch logs unavailable", transform=loss_ax.transAxes,
                     ha="right", va="top", fontsize=8, color="#555555")
    loss_ax.set(title="Direction regression (lower is better)", ylabel="Mean squared unit-direction distance")
    loss_ax.set_yscale("symlog", linthresh=1e-4)
    loss_ax.legend(fontsize=8, loc="lower left")
    data_x = [update_at_round[int(row["round"])] for row in training]
    for key, label, color in (("states", "Stored states", "#354e74"),
                              ("labels", "Valid direction labels", "#b75b20")):
        data_ax.plot(data_x, [int(row[key]) for row in training], "o-", color=color, label=label)
    data_ax.set(title="Accumulated training dataset", ylabel="Count")
    data_ax.legend(fontsize=8)
    colors = plt.get_cmap("tab10")
    summary = []
    for goal_id, goal in enumerate(goals):
        color = colors(goal_id % 10)
        gx, rates, medians = [], [], {key: [] for key in ("flatness", "spread_error", "steps")}
        lows, highs = {k: [] for k in medians}, {k: [] for k in medians}
        for r in rounds:
            rows = [row for row in evaluation if int(row["round"]) == r and float(row["goal"]) == goal]
            if not rows:
                continue
            gx.append(update_at_round[r])
            successes = sum(int(row["success"]) for row in rows)
            rates.append(successes / len(rows) * 100)
            record = dict(round=r, cumulative_updates=update_at_round[r], goal=goal,
                          rollouts=len(rows), successes=successes, success_rate=successes / len(rows))
            for key in medians:
                values = [float(row[key]) for row in rows]
                lo, median, hi = np.quantile(values, [.25, .5, .75])
                lows[key].append(lo)
                medians[key].append(median)
                highs[key].append(hi)
                record[f"{key}_median"] = median
                record[f"{key}_q25"] = lo
                record[f"{key}_q75"] = hi
            summary.append(record)
            # Offset the labels of different goals so identical curves remain legible.
            success_ax.annotate(f"{successes}/{len(rows)}", (gx[-1], rates[-1]),
                                xytext=(0, 8 + goal_id * 11), textcoords="offset points",
                                ha="center", fontsize=8, color=color)
        success_ax.plot(gx, rates, "o-", label=f"D={goal:g}", color=color)
        for axis, key in ((flat_ax, "flatness"), (spread_ax, "spread_error"), (step_ax, "steps")):
            axis.plot(gx, medians[key], "o-", color=color, label=f"D={goal:g}")
            axis.fill_between(gx, lows[key], highs[key], color=color, alpha=.14)
    success_ax.set(title="Success from held-out initial shapes", ylabel="Success rate (%)", ylim=(-6, 133))
    success_ax.set_yticks([0, 25, 50, 75, 100])
    success_ax.legend(fontsize=8, loc="center right")
    flat_ax.set(title="Terminal flatness: median and IQR", ylabel="F (fixed reference normalization)")
    flat_ax.set_yscale("symlog", linthresh=1e-5)
    flat_ax.axhline(1e-3, color="black", ls="--", lw=1, label="Success threshold")
    spread_ax.set(title="Terminal spread error: median and IQR", ylabel="|D - target D|")
    spread_ax.set_yscale("symlog", linthresh=1e-5)
    spread_ax.axhline(.05, color="black", ls="--", lw=1, label="Success threshold")
    step_ax.set(title="Rollout steps: median and IQR", ylabel="Accepted steps (failures included)")
    step_ax.text(.97, .96, "Early failure can also shorten a rollout", transform=step_ax.transAxes,
                 ha="right", va="top", fontsize=8, color="#555555")
    flat_ax.legend(fontsize=8)
    spread_ax.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Cumulative optimizer updates")
        axis.grid(True, alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=.12)
        for r in rounds:
            if r >= 0:
                axis.axvline(update_at_round[r], color="#999999", lw=.7, alpha=.25)
    fig.supxlabel("Shading: rollout IQR, not confidence intervals. Training data change between rounds; loss alone does not establish generalization.", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(run / f"learning_curve.{suffix}", dpi=180)
    plt.close(fig)
    with (run / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return run / "learning_curve.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(plot_run(parser.parse_args().run))
