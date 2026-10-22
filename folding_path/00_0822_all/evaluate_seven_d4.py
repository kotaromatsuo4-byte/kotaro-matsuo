"""Evaluate a frozen D2/D3 seven-bar policy on D4, with a separate teacher reference.

No optimizer or dataset update is performed. D4 target coordinates are used
ONLY by the teacher reference, never by the NN. All starts come from the held-out split.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from supervised_folding import DirectionNet, Geometry, load_targets, rollout, write_csv


def evaluate_run(run):
    run = Path(run).resolve()
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    if design["training_goals"] != [2.0, 3.0] or design["coordinate_size"] != 21:
        raise ValueError("Expected a seven-bar run trained only on D2 and D3")
    output = run / "d4_transfer"
    output.mkdir(exist_ok=False)
    checkpoint_path = run / "checkpoint_final.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = DirectionNet(checkpoint["hidden"], checkpoint["coordinate_size"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    geometry = Geometry(run / "model", design["step_width"])
    starts = np.load(run / "starts.npz")["test"]
    targets = load_targets(geometry)
    # Six unit edges with fixed endpoints -1 and +1; diameter = 4.
    reference_d4 = np.zeros((7, 3))
    reference_d4[:, 0] = [-1, 0, 1, 2, 3, 2, 1]
    np.savetxt(output / "teacher_reference_d4.dat", reference_d4, delimiter=",")
    targets[4.0] = reference_d4
    rows, student_d4 = [], []
    for method in ("NN", "teacher_reference"):
        for start_id, initial in enumerate(starts):
            for goal in (2.0, 3.0, 4.0):
                controller = {"model": model} if method == "NN" else {"target": targets[goal]}
                path, metrics = rollout(geometry, initial, goal, design["max_steps"], **controller)
                rows.append(dict(method=method, start_id=start_id, goal=goal, **metrics))
                np.savez_compressed(output / f"{method}_start{start_id:03d}_d{goal:g}.npz", nodes=path)
                if method == "NN" and goal == 4.0:
                    student_d4.append(path)
            print(f"{method}: {start_id + 1}/{len(starts)} starts", flush=True)
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    write_csv(output / "rollouts.csv", rows)
    summaries = []
    for method in ("NN", "teacher_reference"):
        for goal in (2.0, 3.0, 4.0):
            group = [r for r in rows if r["method"] == method and r["goal"] == goal]
            summaries.append(dict(method=method, goal=goal, trials=len(group),
                                  successes=sum(r["success"] for r in group),
                                  success_rate=np.mean([r["success"] for r in group]),
                                  median_flatness=np.median([r["flatness"] for r in group]),
                                  median_spread=np.median([r["spread"] for r in group]),
                                  median_spread_error=np.median([r["spread_error"] for r in group]),
                                  median_steps=np.median([r["steps"] for r in group]),
                                  termination_reasons=json.dumps(dict(Counter(r["reason"] for r in group)))))
    write_csv(output / "summary.csv", summaries)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    x = np.arange(3)
    for offset, method, color in ((-.18, "NN", "#356aa0"), (.18, "teacher_reference", "#aaaaaa")):
        group = [r for r in summaries if r["method"] == method]
        rates = [r["success_rate"] * 100 for r in group]
        axes[0].bar(x + offset, rates, width=.35, color=color, label=method.replace("_", " "))
        for i, r in enumerate(group):
            axes[0].text(i + offset, rates[i] + 2, f'{r["successes"]}/{r["trials"]}', ha="center", fontsize=9)
    axes[0].set(xticks=x, xticklabels=["D2 (trained)", "D3 (trained)", "D4 (unseen)"],
                ylim=(0, 120), yticks=[0, 25, 50, 75, 100], ylabel="Success rate (%)", title="Frozen NN vs coordinate-target teacher")
    axes[0].legend(fontsize=8)
    colors = plt.get_cmap("tab10")
    for i, goal in enumerate((2.0, 3.0, 4.0)):
        group = [r for r in rows if r["method"] == "NN" and r["goal"] == goal]
        axes[1].scatter([r["spread"] for r in group], [r["flatness"] for r in group],
                        color=colors(i), alpha=.8, label=f"Requested D={goal:g}", s=28)
    axes[1].axhline(1e-3, color="black", ls="--", lw=1)
    axes[1].axvspan(3.95, 4.05, color=colors(2), alpha=.12)
    axes[1].set(yscale="log", xlabel="Terminal spread", ylabel="Terminal flatness F", title="NN terminal states")
    axes[1].legend(fontsize=8)
    for path in student_d4:
        spread = [geometry.metrics(node, 4)["spread"] for node in path]
        axes[2].plot(np.arange(len(path)), spread, color=colors(2), alpha=.4, lw=.8)
    axes[2].axhline(4, color="black", ls="--", label="Requested D=4")
    axes[2].set(xlabel="Accepted steps", ylabel="Spread", title="D4: every held-out NN trajectory")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Seven-bar: train on D2/D3, evaluate D4 without additional training", weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"transfer_result.{suffix}", dpi=180)
    plt.close(fig)
    # Representative chosen by median final spread error, not by success.
    d4_rows = [r for r in rows if r["method"] == "NN" and r["goal"] == 4.0]
    representative = sorted(d4_rows, key=lambda r: (r["spread_error"], r["start_id"]))[len(d4_rows) // 2]
    path = student_d4[representative["start_id"]]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3), constrained_layout=True)
    for axis, step in zip(axes, np.linspace(0, len(path) - 1, 5).astype(int)):
        node = path[step]
        axis.plot(node[:, 0], node[:, 2], "o-", color="#356aa0")
        axis.set_aspect("equal")
        axis.set(xlim=(path[:, :, 0].min() - .2, path[:, :, 0].max() + .2),
                 ylim=(path[:, :, 2].min() - .2, path[:, :, 2].max() + .2), title=f"step {step}", xlabel="x", ylabel="z")
        axis.grid(alpha=.2)
    fig.suptitle(f'D4 NN trajectory: held-out start {representative["start_id"]} (median final spread error)')
    fig.savefig(output / "d4_representative_path.png", dpi=180)
    plt.close(fig)
    lines = ["# Seven-bar D2/D3 → D4 evaluation", "",
             f"Training seed: {design['seed']}; training starts: {design['train_starts']}; held-out starts: {len(starts)}.",
             f"Training goals: D2, D3 only. Max steps: {design['max_steps']}; epochs per round: {design['epochs']}; aggregation rounds: {design['rounds']}.",
             "D4 is evaluated only after all training. NN receives current coordinates and requested spread; never target coordinates.",
             "The teacher reference receives full target coordinates and is a feasibility/reference check, not a learned baseline.", "",
             "| Method | Goal | Success | Median flatness | Median spread | Median steps |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f'| {r["method"]} | {r["goal"]:g} | {r["successes"]}/{r["trials"]} | {r["median_flatness"]:.6g} | {r["median_spread"]:.6g} | {r["median_steps"]:g} |')
    lines += ["", f"Checkpoint SHA256 (unchanged after evaluation): `{checkpoint_hash}`.",
              "Success requires F <= 0.001, spread error <= 0.05 and length-error sum <= 1e-8; crossing candidates are rejected.",
              "One training seed, same seven-bar structure and randomized starts around one reference shape. This is target extrapolation, not transfer to an unseen structure.",
              "The D2/D3 target configurations are one chosen pair of overlapping straight shapes; results may depend on that choice.",
              "Failure here does not prove that D4 transfer is impossible. No hyperparameter choice was made using D4 outcomes."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    return output


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(evaluate_run(parser.parse_args().run))
