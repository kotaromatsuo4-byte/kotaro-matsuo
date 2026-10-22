"""Summarize online perturbation-selection experiments.

This script complements ``compare_results.py``.  It emphasizes quantities
needed for a filter/ranker study: cell counts at fixed mechanics-step
prefixes, time to each archive size, discoveries per mechanics step, and
optional filter diagnostics.  The optional columns are read only when a
future selector writes them to ``rollout_diagnostics.dat``; therefore the
script also works with the existing main5-v3.1 results.

Example
-------
python online_policy_metrics.py \
    large_random_transport_pool_b10000_v31 \
    large_transport_heuristic_b10000_v31 \
    large_gp_transport_bonus1_b10000_v31
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path

from compare_results import (
    cell_count_at,
    read_discovery_curve,
    read_json,
    read_rollout_rows,
)


OPTIONAL_FILTER_FIELDS = (
    "candidate_count",
    "retained_count",
    "filter_rejected_count",
    "filter_bypass",
    "selected_rank_fraction",
    "audit_rejected",
    "audit_would_be_new",
)


def _finite_mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def _first_step_to_count(curve, target):
    for step, count in curve:
        if int(count) >= int(target):
            return int(step)
    return math.nan


def _optional_mean(rows, field):
    return _finite_mean(
        row[field]
        for row in rows
        if field in row and math.isfinite(float(row[field]))
    )


def summarize_seed(result_name, seed_directory, prefixes, target_cells):
    summary = read_json(seed_directory / "summary.json")
    curve = read_discovery_curve(seed_directory / "discovery_curve.dat")
    rows = read_rollout_rows(seed_directory / "rollout_diagnostics.dat")
    selected = [
        row for row in rows if int(round(row.get("selection_code", 0.0))) != 0
    ]
    spent = sum(row.get("spent_steps", 0.0) for row in selected)
    discoveries = sum(row.get("new_cell", 0.0) for row in selected)
    parent_counts = Counter(
        int(round(row.get("parent_id", -1.0))) for row in selected
    )
    parent_probabilities = [
        count / len(selected) for count in parent_counts.values()
    ] if selected else []
    parent_entropy = -sum(
        probability * math.log(probability)
        for probability in parent_probabilities
        if probability > 0.0
    )
    maximum_parent_entropy = (
        math.log(len(parent_counts)) if len(parent_counts) > 1 else 0.0
    )

    row = {
        "result_name": result_name,
        "row_type": "seed",
        "seed": int(summary["seed"]),
        "step_budget": int(summary["step_budget"]),
        "final_cells": int(summary["elite_cells"]),
        "discovery_auc": float(summary["discovery_auc"]),
        "walltime_seconds": float(summary["wall_time_seconds"]),
        "selection_count": len(selected),
        "unique_parent_fraction": (
            len(parent_counts) / len(selected) if selected else math.nan
        ),
        "maximum_parent_reuse": (
            max(parent_counts.values()) if parent_counts else math.nan
        ),
        "normalized_parent_entropy": (
            parent_entropy / maximum_parent_entropy
            if maximum_parent_entropy > 0.0
            else math.nan
        ),
        "mean_rollout_steps": (
            spent / len(selected) if selected else math.nan
        ),
        "flat_fraction": _finite_mean(
            int(round(item.get("stop_code", -1.0))) == 0 for item in selected
        ),
        "collision_fraction": _finite_mean(
            int(round(item.get("stop_code", -1.0))) == 1 for item in selected
        ),
        "discoveries_per_1000_steps": (
            1000.0 * discoveries / spent if spent > 0.0 else math.nan
        ),
        "direct_parent_fraction": _finite_mean(
            item.get("parent_is_direct_perturbation", math.nan)
            for item in selected
        ),
        "exact_transport_fraction": _finite_mean(
            item.get("candidate_is_transported", math.nan) for item in selected
        ),
        "mean_transport_alignment": _finite_mean(
            item.get("transport_alignment", math.nan) for item in selected
        ),
    }
    for prefix in prefixes:
        row[f"cells_at_{prefix}"] = cell_count_at(curve, int(prefix))
    for target in range(2, int(target_cells) + 1):
        step = _first_step_to_count(curve, target)
        row[f"step_to_{target}"] = step
        row[f"reached_{target}"] = int(math.isfinite(step))
    for field in OPTIONAL_FILTER_FIELDS:
        row[f"mean_{field}"] = _optional_mean(selected, field)

    candidate_count = row["mean_candidate_count"]
    retained_count = row["mean_retained_count"]
    row["mean_retained_fraction"] = (
        retained_count / candidate_count
        if math.isfinite(candidate_count)
        and math.isfinite(retained_count)
        and candidate_count > 0.0
        else math.nan
    )
    audited_rejected = sum(
        float(item["audit_rejected"])
        for item in selected
        if "audit_rejected" in item
        and math.isfinite(float(item["audit_rejected"]))
    )
    audited_new = sum(
        float(item["audit_would_be_new"])
        for item in selected
        if "audit_would_be_new" in item
        and math.isfinite(float(item["audit_would_be_new"]))
    )
    row["audited_false_reject_fraction"] = (
        audited_new / audited_rejected
        if audited_rejected > 0.0
        else math.nan
    )
    return row


def aggregate_rows(result_name, rows, fields):
    aggregate = {
        "result_name": result_name,
        "row_type": "mean",
        "seed": "mean",
    }
    for field in fields:
        if field in aggregate or field in ("result_name", "row_type", "seed"):
            continue
        aggregate[field] = _finite_mean(row[field] for row in rows)
    return aggregate


def parse_arguments():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_names", nargs="+")
    parser.add_argument("--result-root", type=Path, default=base / "result")
    parser.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=(1000, 3000, 5000, 10000),
    )
    parser.add_argument("--target-cells", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "result" / "online_policy_metrics.csv",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    seed_rows = []
    grouped = {}
    for result_name in arguments.result_names:
        directory = arguments.result_root / result_name
        rows = [
            summarize_seed(
                result_name,
                seed_directory,
                arguments.prefixes,
                arguments.target_cells,
            )
            for seed_directory in sorted(directory.glob("seed_*"))
            if (seed_directory / "summary.json").is_file()
        ]
        if not rows:
            raise FileNotFoundError(f"no seed results found in {directory}")
        grouped[result_name] = rows
        seed_rows.extend(rows)

    fields = list(seed_rows[0])
    output_rows = []
    for result_name in arguments.result_names:
        output_rows.extend(grouped[result_name])
        output_rows.append(aggregate_rows(result_name, grouped[result_name], fields))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"CSV saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
