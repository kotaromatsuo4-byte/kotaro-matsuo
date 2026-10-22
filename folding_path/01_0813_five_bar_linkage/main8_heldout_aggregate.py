"""Aggregate the frozen main8 held-out comparison and paired confidence intervals."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from main8_results_aggregate import (
    canonical_lookup,
    mean_std,
    one_run,
    warmup_audit,
)


SELECTORS = (
    "variable_k_random",
    "variable_k_gp_filter",
    "variable_k_gp_filter_shuffled",
)
PAIRS = (
    ("variable_k_gp_filter", "variable_k_random"),
    ("variable_k_gp_filter", "variable_k_gp_filter_shuffled"),
)
METRICS = (
    "final_raw_cells",
    "final_canonical_cells",
    "raw_auc",
    "canonical_auc",
    "postwarm_raw_auc",
    "postwarm_canonical_auc",
    "canonical_t6_capped",
    "canonical_t6_reached",
    "guided_rollouts_per_1000_postwarm_steps",
    "mean_guided_requested_k",
    "mean_guided_executed_k",
    "wall_time_seconds",
)


def bootstrap_ci(values, seed=813, samples=100_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
        axis=1
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def exact_sign_flip_p(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    permutations = np.asarray(
        [
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    return float(np.mean(permutations >= observed - 1.0e-15))


def aggregate(rows):
    output = []
    for selector in SELECTORS:
        group = [row for row in rows if row["selector"] == selector]
        item = {"selector": selector, "seeds": len(group)}
        for metric in METRICS:
            stats = mean_std(row[metric] for row in group)
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
        reached = [row["canonical_t6"] for row in group if row["canonical_t6"]]
        item["canonical_t6_mean_reached"] = statistics.fmean(reached) if reached else None
        output.append(item)
    return output


def paired(rows):
    lookup = {(row["selector"], row["seed"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    output = []
    for pair_index, (left, right) in enumerate(PAIRS):
        item = {"left": left, "right": right, "seeds": seeds, "metrics": {}}
        for metric_index, metric in enumerate(METRICS):
            differences = [
                float(lookup[(left, seed)][metric])
                - float(lookup[(right, seed)][metric])
                for seed in seeds
            ]
            stats = mean_std(differences)
            item["metrics"][metric] = {
                "by_seed": dict(zip(map(str, seeds), differences)),
                **stats,
                "bootstrap_95_ci": bootstrap_ci(
                    differences, seed=813 + 100 * pair_index + metric_index
                ),
                "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
            }
        output.append(item)
    return output


def fmt(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"


def markdown(payload):
    lines = [
        "# main8 variable-K held-out comparison (B=3000)",
        "",
        "Frozen `main8_variable_k_transport_macro_gp_v2`, seeds 703--712.",
        "All methods used the same paired warm-up and mechanics budget.",
        "",
        "## Warm-up audit",
        "",
        "All ten seeds passed normalized-configuration, prefix-hash, and "
        "training-hash equality across the three selectors.",
        "",
        "## Aggregate",
        "",
        "| selector | raw/canonical final | raw/canonical AUC | postwarm raw/canonical AUC | canonical T6 | rollouts/1000 | requested/executed K | wall [s] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['selector']} | "
            f"{fmt(row['final_raw_cells_mean'], row['final_raw_cells_std'])} / "
            f"{fmt(row['final_canonical_cells_mean'], row['final_canonical_cells_std'])} | "
            f"{fmt(row['raw_auc_mean'], row['raw_auc_std'])} / "
            f"{fmt(row['canonical_auc_mean'], row['canonical_auc_std'])} | "
            f"{fmt(row['postwarm_raw_auc_mean'], row['postwarm_raw_auc_std'])} / "
            f"{fmt(row['postwarm_canonical_auc_mean'], row['postwarm_canonical_auc_std'])} | "
            f"{row['canonical_t6_mean_reached']:.1f} "
            f"({int(row['canonical_t6_reached_mean'] * row['seeds'])}/{row['seeds']}) | "
            f"{fmt(row['guided_rollouts_per_1000_postwarm_steps_mean'], row['guided_rollouts_per_1000_postwarm_steps_std'])} | "
            f"{fmt(row['mean_guided_requested_k_mean'], row['mean_guided_requested_k_std'])} / "
            f"{fmt(row['mean_guided_executed_k_mean'], row['mean_guided_executed_k_std'])} | "
            f"{fmt(row['wall_time_seconds_mean'], row['wall_time_seconds_std'])} |"
        )
    lines += [
        "",
        "## Paired differences (left - right)",
        "",
        "For AUC and coverage, positive favors GP. For capped T6, negative favors GP.",
        "",
        "| left | right | raw AUC [95% CI] | canonical AUC [95% CI] | postwarm canonical AUC [95% CI] | capped T6 [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["paired"]:
        def cell(metric):
            value = row["metrics"][metric]
            lo, hi = value["bootstrap_95_ci"]
            return f"{value['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"

        lines.append(
            f"| {row['left']} | {row['right']} | {cell('raw_auc')} | "
            f"{cell('canonical_auc')} | {cell('postwarm_canonical_auc')} | "
            f"{cell('canonical_t6_capped')} |"
        )
    lines += [
        "",
        "## Per-seed AUC and T6",
        "",
        "| seed | selector | raw/canonical | raw/canonical AUC | postwarm canonical AUC | T6 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['seed']} | {row['selector']} | {row['final_raw_cells']}/"
            f"{row['final_canonical_cells']} | {row['raw_auc']:.3f}/"
            f"{row['canonical_auc']:.3f} | {row['postwarm_canonical_auc']:.3f} | "
            f"{row['canonical_t6']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="result")
    parser.add_argument(
        "--coverage", default="result/main8_heldout_b3000_canonical_coverage.json"
    )
    parser.add_argument("--json", default="result/main8_heldout_b3000_aggregate.json")
    parser.add_argument("--csv", default="result/main8_heldout_b3000_runs.csv")
    parser.add_argument("--markdown", default="MAIN8_HELDOUT_RESULTS.md")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    cell_lookup = canonical_lookup(json.loads(Path(args.coverage).read_text(encoding="utf-8")))
    rows = []
    for selector in SELECTORS:
        result_name = f"main8_heldout_b3000_{selector}"
        for seed in range(703, 713):
            seed_dir = result_dir / result_name / f"seed_{seed}"
            row = one_run(seed_dir, selector, cell_lookup[(result_name, seed)])
            row["canonical_t6_reached"] = int(row["canonical_t6"] is not None)
            row["canonical_t6_capped"] = (
                row["canonical_t6"] if row["canonical_t6"] is not None else row["budget"]
            )
            rows.append(row)
    rows.sort(key=lambda row: (row["seed"], SELECTORS.index(row["selector"])))
    audits = warmup_audit(rows)
    for audit in audits:
        audit["passed"] = (
            audit["config_match"]
            and len(audit["warmup_steps"]) == 1
            and len(audit["warmup_raw_cells"]) == 1
            and len(audit["warmup_canonical_cells"]) == 1
            and len(audit["prefix_hashes"]) == 1
            and len(audit["training_hashes"]) == 1
        )
    payload = {
        "runs": rows,
        "aggregate": aggregate(rows),
        "paired": paired(rows),
        "warmup_audit": audits,
        "all_warmup_audits_passed": all(row["passed"] for row in audits),
    }
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "seed", "selector", "final_raw_cells", "final_canonical_cells",
        "raw_auc", "canonical_auc", "postwarm_raw_auc",
        "postwarm_canonical_auc", "canonical_t6", "rollouts",
        "guided_rollouts_per_1000_postwarm_steps", "mean_guided_requested_k",
        "mean_guided_executed_k", "wall_time_seconds",
    ]
    with Path(args.csv).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    report = markdown(payload)
    Path(args.markdown).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
