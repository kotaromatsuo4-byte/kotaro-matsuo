"""Read-only checkpoint analysis for the water-D=1 spread-alpha screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE / "experiments_ori9" / "C9SA1"
CONTROL_ROOT = BASE / "experiments_ori8" / "C8SP2R1" / "water1" / "scale"
SEEDS = (8101, 8102, 8103)
ALPHAS = (1, 2, 3)
ATOMIC = 512


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    episodes = [int(float(row["episode"])) for row in rows]
    if episodes != list(range(1, len(rows) + 1)):
        raise ValueError(f"Non-contiguous episode history: {path}")
    return rows


def value(row, name, default=0.0):
    raw = row.get(name, "")
    return float(raw) if raw not in (None, "") else float(default)


def summarize(rows, endpoint, window):
    rows = rows[:endpoint]
    tail = rows[-min(window, len(rows)):]
    strict = [value(row, "strict_compact_success") > 0.5 for row in rows]
    first_index = next((i for i, hit in enumerate(strict) if hit), None)
    cumulative = 0
    first_proposals = None
    for i, row in enumerate(rows):
        cumulative += int(value(row, "steps"))
        if first_index == i:
            first_proposals = cumulative
            break
    exposure = sum(int(value(row, "steps")) for row in rows)

    def c(row):
        return value(row, "flatness_ratio") <= 0.5 and value(row, "spread_error") <= 0.1

    def j(row):
        return value(row, "flatness_ratio") <= 0.1 and value(row, "spread_error") <= 0.1

    decomp_fields = (
        "ori5_reward_decomposition_error",
        "ori6_reward_decomposition_error",
        "ori6_spectral_reward_decomposition_error",
        "ori6_maximum_abs_zero_reject_reward_decomposition_error",
        "ori9_reward_decomposition_error",
    )
    max_decomp = max(
        [abs(value(row, name)) for row in rows for name in decomp_fields
         if row.get(name, "") not in (None, "")] or [0.0]
    )
    nonfinite = sum(
        int(value(row, name)) for row in rows for name in (
            "ori5_nonfinite_reward_component_count",
            "ori6_nonfinite_reward_component_count",
            "ori6_spectral_nonfinite_component_count",
            "ori9_nonfinite_component_count",
        ) if row.get(name, "") not in (None, "")
    )
    rejects = sum(int(value(row, "ori6_rejected_crossing_count")) for row in tail)
    tail_steps = sum(int(value(row, "steps")) for row in tail)
    return {
        "endpoint": endpoint,
        "window": len(tail),
        "first_hit_episode": None if first_index is None else first_index + 1,
        "first_hit_proposals": first_proposals,
        "proposal_exposure": exposure,
        "strict_total": sum(strict),
        "tail_strict": sum(value(row, "strict_compact_success") > 0.5 for row in tail),
        "tail_joint": sum(j(row) for row in tail),
        "tail_constructive": sum(c(row) for row in tail),
        "tail_median_F": median(value(row, "flatness_ratio") for row in tail),
        "tail_median_e": median(value(row, "spread_error") for row in tail),
        "tail_mean_return": sum(value(row, "total_reward") for row in tail) / len(tail),
        "tail_wrong_flat": sum(value(row, "wrong_flat") > 0.5 for row in tail),
        "tail_reject_rate": rejects / tail_steps if tail_steps else 0.0,
        "max_abs_decomposition_error": max_decomp,
        "nonfinite_component_count": nonfinite,
    }


def history_path(root, alpha, seed):
    if alpha == 1:
        return CONTROL_ROOT / f"seed{seed}" / "training_history.csv"
    return root / "water1" / f"alpha{alpha}" / f"seed{seed}" / "training_history.csv"


def choose_endpoint(root, requested):
    lengths = []
    for alpha in (2, 3):
        for seed in SEEDS:
            path = history_path(root, alpha, seed)
            if not path.is_file():
                return 0
            lengths.append(len(read_rows(path)))
    available = (min(lengths) // ATOMIC) * ATOMIC
    return min(available, requested) if requested else available


def analyze(root, endpoint, window):
    records = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            rows = read_rows(history_path(root, alpha, seed))
            if len(rows) < endpoint:
                raise ValueError(f"alpha{alpha}/seed{seed} lacks endpoint {endpoint}")
            row = summarize(rows, endpoint, window)
            row.update({"alpha": alpha, "seed": seed})
            records.append(row)

    common_cap = min(row["proposal_exposure"] for row in records)
    arms = {}
    for alpha in ALPHAS:
        selected = [row for row in records if row["alpha"] == alpha]
        arm = {
            "hit_count": sum(
                row["first_hit_proposals"] is not None
                and row["first_hit_proposals"] <= common_cap
                for row in selected
            ),
            "common_cap_rmst": sum(
                min(row["first_hit_proposals"] or common_cap, common_cap)
                for row in selected
            ) / len(selected),
            "tail_strict": sum(row["tail_strict"] for row in selected),
            "tail_joint": sum(row["tail_joint"] for row in selected),
            "tail_constructive": sum(row["tail_constructive"] for row in selected),
        }
        arms[str(alpha)] = arm

    paired = {}
    controls = {row["seed"]: row for row in records if row["alpha"] == 1}
    for alpha in (2, 3):
        wins = losses = ties = 0
        details = []
        for row in (record for record in records if record["alpha"] == alpha):
            control = controls[row["seed"]]
            candidate_t = row["first_hit_proposals"]
            control_t = control["first_hit_proposals"]
            if candidate_t is not None and (control_t is None or candidate_t < control_t):
                result = "win"; wins += 1
            elif control_t is not None and (candidate_t is None or candidate_t > control_t):
                result = "loss"; losses += 1
            else:
                result = "tie"; ties += 1
            details.append({"seed": row["seed"], "result": result,
                            "candidate_T": candidate_t, "control_T": control_t})
        paired[str(alpha)] = {"wins": wins, "losses": losses, "ties": ties,
                              "details": details}
    return {"endpoint": endpoint, "window": window, "common_cap": common_cap,
            "records": records, "arms": arms, "paired_vs_alpha1": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--endpoint", type=int, default=0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    endpoint = choose_endpoint(root, args.endpoint)
    if endpoint <= 0:
        raise RuntimeError("No common atomic checkpoint is available")
    result = analyze(root, endpoint, args.window)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
