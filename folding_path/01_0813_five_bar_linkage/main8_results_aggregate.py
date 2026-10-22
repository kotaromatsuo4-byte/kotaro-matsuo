"""Aggregate the frozen main8 variable-K pilot without rerunning mechanics.

The five-bar archive can contain valid layer variants beyond the six
canonical witnesses.  This script therefore reconstructs both the raw archive
curve and a canonical-coverage curve.  Cell indices are stable: ``cell[i]``
is the cell created by the discovery event whose ``cell_index`` is ``i``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SELECTORS = (
    "variable_k_random",
    "variable_k_gp_argmax",
    "variable_k_gp_argmax_shuffled",
    "variable_k_gp_filter",
    "variable_k_gp_filter_shuffled",
)

PAIRS = (
    ("variable_k_gp_argmax", "variable_k_random"),
    ("variable_k_gp_argmax", "variable_k_gp_argmax_shuffled"),
    ("variable_k_gp_filter", "variable_k_random"),
    ("variable_k_gp_filter", "variable_k_gp_filter_shuffled"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_raw_curve(path):
    curve = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        step, count = line.split()[:2]
        curve.append((int(float(step)), int(float(count))))
    return sorted(dict(curve).items())


def count_at(curve, step):
    value = 0
    for event_step, count in curve:
        if event_step > step:
            break
        value = count
    return int(value)


def interval_auc(curve, start, end, baseline=0):
    if end <= start:
        return 0.0
    cursor = int(start)
    value = count_at(curve, cursor)
    area = 0.0
    for event_step, next_value in curve:
        if event_step <= start:
            continue
        if event_step > end:
            break
        area += (event_step - cursor) * (value - baseline)
        cursor = int(event_step)
        value = int(next_value)
    area += (end - cursor) * (value - baseline)
    return float(area / (end - start))


def canonical_lookup(coverage):
    lookup = {}
    for result in coverage["results"]:
        result_name = result["result_name"]
        for seed_report in result["seeds"]:
            seed_name = Path(seed_report["seed_directory"]).name
            seed = int(seed_name.split("_")[-1])
            lookup[(result_name, seed)] = {
                int(cell["cell"].split("[")[1].split("]")[0]): cell[
                    "canonical_target_id"
                ]
                for cell in seed_report["cells"]
            }
    return lookup


def canonical_curve(discovery_events, cell_targets, budget):
    covered = set()
    curve = [(0, 0)]
    discovery_steps = {}
    for event in sorted(discovery_events, key=lambda row: row["discovery_step"]):
        target = cell_targets.get(int(event["cell_index"]))
        if target is None or int(target) in covered:
            continue
        target = int(target)
        covered.add(target)
        discovery_steps[target] = int(event["discovery_step"])
        curve.append((int(event["discovery_step"]), len(covered)))
    if curve[-1][0] != budget:
        curve.append((int(budget), len(covered)))
    return curve, discovery_steps


def normalized_config(config):
    return {
        key: value
        for key, value in config.items()
        if key not in {"selector", "result_name"}
    }


def load_rollout_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def one_run(seed_dir, selector, cell_targets):
    summary = read_json(seed_dir / "summary.json")
    config = read_json(seed_dir / "configuration.json")
    events = read_json(seed_dir / "discovery_events.json")
    raw_curve = read_raw_curve(seed_dir / "discovery_curve.dat")
    rollout_rows = load_rollout_rows(seed_dir / "rollout_diagnostics.csv")
    budget = int(summary["step_budget"])
    warm_step = int(summary["warmup_end_step"])
    warm_raw = int(summary["warmup_end_cells"])
    canonical, canonical_steps = canonical_curve(events, cell_targets, budget)
    warm_canonical = count_at(canonical, warm_step)
    guided = [row for row in rollout_rows if int(row["warmup_group"]) < 0]
    requested = [int(row["requested_k"]) for row in guided]
    executed = [int(row["executed_k"]) for row in guided]
    new_rows = [row for row in guided if int(row["new_cell"]) == 1]
    raw_steps = [int(event["discovery_step"]) for event in events]
    raw_t6 = raw_steps[5] if len(raw_steps) >= 6 else None
    canonical_t6 = max(canonical_steps.values()) if len(canonical_steps) == 6 else None
    return {
        "seed": int(summary["seed"]),
        "selector": selector,
        "result_name": config["result_name"],
        "algorithm_version": summary["algorithm_version"],
        "budget": budget,
        "warmup_end_step": warm_step,
        "warmup_raw_cells": warm_raw,
        "warmup_canonical_cells": warm_canonical,
        "warmup_prefix_sha256": summary["warmup_prefix_sha256"],
        "warmup_training_sha256": summary["warmup_training_sha256"],
        "config_normalized": normalized_config(config),
        "final_raw_cells": int(summary["elite_cells"]),
        "final_canonical_cells": count_at(canonical, budget),
        "extra_cells": int(summary["elite_cells"]) - count_at(canonical, budget),
        "raw_auc": interval_auc(raw_curve, 0, budget),
        "canonical_auc": interval_auc(canonical, 0, budget),
        "postwarm_raw_auc": interval_auc(raw_curve, warm_step, budget, warm_raw),
        "postwarm_canonical_auc": interval_auc(
            canonical, warm_step, budget, warm_canonical
        ),
        "raw_discovery_steps": raw_steps,
        "canonical_discovery_steps": canonical_steps,
        "raw_t6": raw_t6,
        "canonical_t6": canonical_t6,
        "rollouts": int(summary["num_rollouts"]),
        "guided_rollouts": len(guided),
        "guided_rollouts_per_1000_postwarm_steps": (
            1000.0 * len(guided) / max(1, budget - warm_step)
        ),
        "mean_guided_requested_k": statistics.fmean(requested),
        "mean_guided_executed_k": statistics.fmean(executed),
        "guided_requested_k_counts": dict(sorted(Counter(requested).items())),
        "guided_executed_k_counts": dict(sorted(Counter(executed).items())),
        "new_raw_cell_requested_k": [int(row["requested_k"]) for row in new_rows],
        "wall_time_seconds": float(summary["wall_time_seconds"]),
        "fit_time_seconds": float(summary["fit_time_seconds"]),
        "score_time_seconds": float(summary["score_time_seconds"]),
    }


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(rows):
    metrics = (
        "final_raw_cells",
        "final_canonical_cells",
        "extra_cells",
        "raw_auc",
        "canonical_auc",
        "postwarm_raw_auc",
        "postwarm_canonical_auc",
        "rollouts",
        "guided_rollouts_per_1000_postwarm_steps",
        "mean_guided_requested_k",
        "mean_guided_executed_k",
        "wall_time_seconds",
        "fit_time_seconds",
        "score_time_seconds",
    )
    result = []
    for selector in SELECTORS:
        group = [row for row in rows if row["selector"] == selector]
        item = {"selector": selector, "seeds": len(group)}
        for metric in metrics:
            stats = mean_std(row[metric] for row in group)
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
        t6 = [row["canonical_t6"] for row in group if row["canonical_t6"] is not None]
        item["canonical_t6_successes"] = len(t6)
        item["canonical_t6_mean_reached"] = statistics.fmean(t6) if t6 else None
        pooled_requested = Counter()
        for row in group:
            pooled_requested.update(
                {int(k): int(v) for k, v in row["guided_requested_k_counts"].items()}
            )
        item["guided_requested_k_counts"] = dict(sorted(pooled_requested.items()))
        result.append(item)
    return result


def paired(rows):
    lookup = {(row["selector"], row["seed"]): row for row in rows}
    output = []
    metrics = (
        "final_raw_cells",
        "final_canonical_cells",
        "raw_auc",
        "canonical_auc",
        "postwarm_raw_auc",
        "postwarm_canonical_auc",
        "guided_rollouts_per_1000_postwarm_steps",
        "mean_guided_requested_k",
        "wall_time_seconds",
    )
    seeds = sorted({row["seed"] for row in rows})
    for left, right in PAIRS:
        item = {"left": left, "right": right, "seed_differences": {}}
        for metric in metrics:
            differences = [
                lookup[(left, seed)][metric] - lookup[(right, seed)][metric]
                for seed in seeds
            ]
            stats = mean_std(differences)
            item[f"delta_{metric}"] = {
                "by_seed": dict(zip(map(str, seeds), differences)),
                **stats,
            }
        output.append(item)
    return output


def warmup_audit(rows):
    output = []
    for seed in sorted({row["seed"] for row in rows}):
        group = [row for row in rows if row["seed"] == seed]
        configs = {
            json.dumps(row["config_normalized"], sort_keys=True) for row in group
        }
        output.append(
            {
                "seed": seed,
                "selectors": len(group),
                "config_match": len(configs) == 1,
                "warmup_steps": sorted({row["warmup_end_step"] for row in group}),
                "warmup_raw_cells": sorted({row["warmup_raw_cells"] for row in group}),
                "warmup_canonical_cells": sorted(
                    {row["warmup_canonical_cells"] for row in group}
                ),
                "prefix_hashes": sorted({row["warmup_prefix_sha256"] for row in group}),
                "training_hashes": sorted(
                    {row["warmup_training_sha256"] for row in group}
                ),
            }
        )
    return output


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown(payload):
    lines = [
        "# main8 variable-K pilot (B=3000)",
        "",
        "## Audit",
        "",
        "| seed | config | warmup step | raw/canonical cells | prefix/training hash count |",
        "|---:|---|---:|---|---|",
    ]
    for row in payload["warmup_audit"]:
        lines.append(
            f"| {row['seed']} | {'match' if row['config_match'] else 'MISMATCH'} | "
            f"{row['warmup_steps']} | {row['warmup_raw_cells']}/"
            f"{row['warmup_canonical_cells']} | {len(row['prefix_hashes'])}/"
            f"{len(row['training_hashes'])} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "| selector | raw/canonical final | raw/canonical AUC | postwarm raw/canonical AUC | T6 | rollouts/1000 | requested/executed K | wall [s] |",
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
            f"{row['canonical_t6_successes']}/{row['seeds']} "
            f"({row['canonical_t6_mean_reached']}) | "
            f"{fmt(row['guided_rollouts_per_1000_postwarm_steps_mean'], row['guided_rollouts_per_1000_postwarm_steps_std'])} | "
            f"{fmt(row['mean_guided_requested_k_mean'], row['mean_guided_requested_k_std'])} / "
            f"{fmt(row['mean_guided_executed_k_mean'], row['mean_guided_executed_k_std'])} | "
            f"{fmt(row['wall_time_seconds_mean'], row['wall_time_seconds_std'])} |"
        )
    lines += ["", "## Per seed", ""]
    lines += [
        "| seed | selector | raw/canonical | raw/canonical AUC | postwarm raw/canonical AUC | T6 | discovery K |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['seed']} | {row['selector']} | {row['final_raw_cells']}/"
            f"{row['final_canonical_cells']} | {row['raw_auc']:.3f}/"
            f"{row['canonical_auc']:.3f} | {row['postwarm_raw_auc']:.3f}/"
            f"{row['postwarm_canonical_auc']:.3f} | "
            f"{row['canonical_t6'] if row['canonical_t6'] is not None else '--'} | "
            f"{row['new_raw_cell_requested_k']} |"
        )
    lines += ["", "## Paired deltas (left - right)", ""]
    lines += [
        "| left | right | raw AUC | canonical AUC | postwarm canonical AUC | final canonical |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["paired"]:
        lines.append(
            f"| {row['left']} | {row['right']} | "
            f"{row['delta_raw_auc']['mean']:+.3f} | "
            f"{row['delta_canonical_auc']['mean']:+.3f} | "
            f"{row['delta_postwarm_canonical_auc']['mean']:+.3f} | "
            f"{row['delta_final_canonical_cells']['mean']:+.3f} |"
        )
    lines += [
        "",
        "All steps include perturbation micro-steps, deterministic suffix steps, "
        "and failed mechanics steps.  Canonical curves ignore valid extra layer variants.",
    ]
    return "\n".join(lines) + "\n"


def write_runs_csv(rows, path):
    fields = [
        "seed", "selector", "final_raw_cells", "final_canonical_cells",
        "extra_cells", "raw_auc", "canonical_auc", "postwarm_raw_auc",
        "postwarm_canonical_auc", "canonical_t6", "rollouts",
        "guided_rollouts_per_1000_postwarm_steps", "mean_guided_requested_k",
        "mean_guided_executed_k", "wall_time_seconds",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="result")
    parser.add_argument(
        "--coverage", default="result/main8_pilot_b3000_canonical_coverage.json"
    )
    parser.add_argument("--json", default="result/main8_pilot_b3000_aggregate.json")
    parser.add_argument("--csv", default="result/main8_pilot_b3000_runs.csv")
    parser.add_argument("--markdown", default="MAIN8_PILOT_RESULTS.md")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    lookup = canonical_lookup(read_json(args.coverage))
    rows = []
    for selector in SELECTORS:
        result_name = f"main8_pilot_b3000_{selector}"
        for seed_dir in sorted((result_dir / result_name).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            rows.append(one_run(seed_dir, selector, lookup[(result_name, seed)]))
    rows.sort(key=lambda row: (row["seed"], SELECTORS.index(row["selector"])))
    payload = {
        "runs": rows,
        "aggregate": aggregate(rows),
        "paired": paired(rows),
        "warmup_audit": warmup_audit(rows),
    }
    Path(args.json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_runs_csv(rows, args.csv)
    Path(args.markdown).write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))


if __name__ == "__main__":
    main()
