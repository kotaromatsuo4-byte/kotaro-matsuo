"""Aggregate the frozen hierarchical-K held-out experiment.

This script never reruns mechanics.  It combines the saved discovery events
with the frozen five-bar canonical-cell audit, then reports paired bootstrap
intervals and exact sign-flip tests for seeds 703--712.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from main8_results_aggregate import canonical_lookup, mean_std, one_run


SELECTORS = (
    (
        "variable_k_random",
        "main8_heldout_b3000_variable_k_random",
    ),
    (
        "variable_k_gp_group_argmax",
        "main8_group_heldout_b3000_variable_k_gp_group_argmax",
    ),
    (
        "variable_k_gp_group_argmax_shuffled",
        "main8_group_heldout_b3000_variable_k_gp_group_argmax_shuffled",
    ),
)

PAIRS = (
    ("variable_k_gp_group_argmax", "variable_k_random"),
    (
        "variable_k_gp_group_argmax",
        "variable_k_gp_group_argmax_shuffled",
    ),
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


def bootstrap_ci(values, seed, samples=100_000):
    """Percentile interval for the paired mean difference."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def exact_sign_flip_p(values):
    """Exact two-sided randomization p-value for paired differences."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    randomized = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(randomized) >= observed - 1.0e-15))


def aggregate(rows):
    """Mean and sample standard deviation for every selector."""
    output = []
    for selector, _ in SELECTORS:
        group = [row for row in rows if row["selector"] == selector]
        item = {"selector": selector, "seeds": len(group)}
        for metric in METRICS:
            stats = mean_std(row[metric] for row in group)
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
        reached = [
            row["canonical_t6"]
            for row in group
            if row["canonical_t6"] is not None
        ]
        item["canonical_t6_mean_reached"] = (
            statistics.fmean(reached) if reached else None
        )
        item["canonical_t6_reach_count"] = len(reached)
        output.append(item)
    return output


def paired(rows):
    """Paired differences with bootstrap intervals and sign-flip tests."""
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
                    differences,
                    seed=813 + 100 * pair_index + metric_index,
                ),
                "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
            }
        output.append(item)
    return output


def fair_configuration(config):
    """Remove the three fields that must differ between compared selectors."""
    return {
        key: value
        for key, value in config.items()
        if key not in {"algorithm_version", "selector", "result_name"}
    }


def warmup_audit(rows):
    """Check the paired prefix and all non-selector settings seed by seed."""
    output = []
    for seed in sorted({row["seed"] for row in rows}):
        group = [row for row in rows if row["seed"] == seed]
        configurations = {
            json.dumps(fair_configuration(row["config_normalized"]), sort_keys=True)
            for row in group
        }
        item = {
            "seed": seed,
            "selector_count": len(group),
            "configuration_match": len(configurations) == 1,
            "warmup_end_steps": sorted({row["warmup_end_step"] for row in group}),
            "warmup_raw_cells": sorted({row["warmup_raw_cells"] for row in group}),
            "warmup_canonical_cells": sorted(
                {row["warmup_canonical_cells"] for row in group}
            ),
            "prefix_hashes": sorted(
                {row["warmup_prefix_sha256"] for row in group}
            ),
            "training_hashes": sorted(
                {row["warmup_training_sha256"] for row in group}
            ),
        }
        item["passed"] = (
            item["selector_count"] == len(SELECTORS)
            and item["configuration_match"]
            and len(item["warmup_end_steps"]) == 1
            and len(item["warmup_raw_cells"]) == 1
            and len(item["warmup_canonical_cells"]) == 1
            and len(item["prefix_hashes"]) == 1
            and len(item["training_hashes"]) == 1
        )
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="result")
    parser.add_argument(
        "--coverage",
        default="result/main8_group_heldout_b3000_canonical_coverage.json",
    )
    parser.add_argument(
        "--json",
        default="result/main8_group_heldout_b3000_aggregate.json",
    )
    parser.add_argument(
        "--csv",
        default="result/main8_group_heldout_b3000_runs.csv",
    )
    arguments = parser.parse_args()

    result_directory = Path(arguments.result_dir)
    coverage = json.loads(Path(arguments.coverage).read_text(encoding="utf-8"))
    cell_lookup = canonical_lookup(coverage)
    rows = []
    for selector, result_name in SELECTORS:
        for seed in range(703, 713):
            seed_directory = result_directory / result_name / f"seed_{seed}"
            row = one_run(
                seed_directory,
                selector,
                cell_lookup[(result_name, seed)],
            )
            row["canonical_t6_reached"] = int(row["canonical_t6"] is not None)
            row["canonical_t6_capped"] = (
                row["canonical_t6"]
                if row["canonical_t6"] is not None
                else row["budget"]
            )
            rows.append(row)
    selector_order = [selector for selector, _ in SELECTORS]
    rows.sort(key=lambda row: (row["seed"], selector_order.index(row["selector"])))

    audit = warmup_audit(rows)
    payload = {
        "runs": rows,
        "aggregate": aggregate(rows),
        "paired": paired(rows),
        "warmup_audit": audit,
        "all_warmup_audits_passed": all(item["passed"] for item in audit),
    }
    Path(arguments.json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = (
        "seed",
        "selector",
        "final_raw_cells",
        "final_canonical_cells",
        "raw_auc",
        "canonical_auc",
        "postwarm_raw_auc",
        "postwarm_canonical_auc",
        "canonical_t6",
        "canonical_t6_reached",
        "canonical_t6_capped",
        "guided_rollouts_per_1000_postwarm_steps",
        "mean_guided_requested_k",
        "mean_guided_executed_k",
        "wall_time_seconds",
    )
    with Path(arguments.csv).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fields}
            for row in rows
        )

    for row in payload["aggregate"]:
        print(
            row["selector"],
            f"canonical={row['final_canonical_cells_mean']:.3f}",
            f"AUC={row['canonical_auc_mean']:.3f}",
            f"reach6={row['canonical_t6_reach_count']}/{row['seeds']}",
            f"capped_T6={row['canonical_t6_capped_mean']:.1f}",
        )
    print(f"warmup audit: {payload['all_warmup_audits_passed']}")
    print(f"json: {Path(arguments.json)}")
    print(f"csv: {Path(arguments.csv)}")


if __name__ == "__main__":
    main()
