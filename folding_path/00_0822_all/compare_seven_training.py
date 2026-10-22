"""Compare previous and expanded seven-bar training on identical held-out starts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare(previous, expanded):
    previous, expanded = Path(previous), Path(expanded)
    np.testing.assert_array_equal(np.load(previous / "starts.npz")["test"],
                                  np.load(expanded / "starts.npz")["test"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#386da1", "#d67529"]
    results = []
    for run, label, style in ((previous, "16 starts / 3 rounds", "--"),
                              (expanded, "64 starts / 10 rounds", "-")):
        summary = read(run / "evaluation_summary.csv")
        training = read(run / "training_epochs.csv")
        for i, goal in enumerate((2., 3.)):
            rows = [r for r in summary if float(r["goal"]) == goal]
            axes[0, i].plot([int(r["cumulative_updates"]) for r in rows],
                            [100 * float(r["success_rate"]) for r in rows],
                            marker="o", ls=style, label=label, color=colors[0 if run == previous else 1])
        axes[1, 0].plot([int(r["cumulative_updates"]) for r in training],
                       [float(r["loss"]) for r in training], ls=style, label=label,
                       color=colors[0 if run == previous else 1], lw=.9)
        final = [r for r in read(run / "d4_transfer/summary.csv") if r["method"] == "NN"]
        offset = -.18 if run == previous else .18
        heights = [100 * float(r["success_rate"]) for r in final]
        axes[1, 1].bar(np.arange(3) + offset, heights, .35, label=label,
                       color=colors[0 if run == previous else 1])
        for i, row in enumerate(final):
            axes[1, 1].text(i + offset, heights[i] + 2, f'{row["successes"]}/{row["trials"]}',
                            ha="center", fontsize=9)
            results.append(dict(run=run.name, **row))
    for i, goal in enumerate((2, 3)):
        axes[0, i].set(title=f"D{goal}: fixed held-out initial shapes", ylabel="Success rate (%)",
                       xlabel="Cumulative optimizer updates", ylim=(-3, 105))
    axes[1, 0].set(title="Epoch mean direction regression loss", xlabel="Cumulative optimizer updates",
                   ylabel="Squared unit-direction distance", yscale="log")
    axes[1, 1].set(title="Final frozen NN: D4 remains untrained", ylabel="Success rate (%)",
                   xticks=np.arange(3), xticklabels=["D2", "D3", "D4"], ylim=(0, 120))
    for axis in axes.flat:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.suptitle("Seven-bar: does more training improve D2/D3 and transfer to D4?", weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(expanded / f"comparison_previous.{suffix}", dpi=180)
    plt.close(fig)
    with (expanded / "comparison_previous.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    old = read(previous / "d4_transfer/rollouts.csv")
    new = read(expanded / "d4_transfer/rollouts.csv")
    paired = []
    for goal in (2., 3., 4.):
        a = {int(r["start_id"]): int(r["success"]) for r in old if r["method"] == "NN" and float(r["goal"]) == goal}
        b = {int(r["start_id"]): int(r["success"]) for r in new if r["method"] == "NN" and float(r["goal"]) == goal}
        assert a.keys() == b.keys()
        paired.append(dict(goal=goal, both_success=sum(a[k] and b[k] for k in a),
                           improved=sum(not a[k] and b[k] for k in a),
                           regressed=sum(a[k] and not b[k] for k in a),
                           both_failure=sum(not a[k] and not b[k] for k in a)))
    with (expanded / "paired_success_changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    return results, paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("expanded", type=Path)
    args = parser.parse_args()
    print(compare(args.previous, args.expanded))
