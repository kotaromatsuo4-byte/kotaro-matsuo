"""Aggregate paired ``main6.py`` experiments without rerunning mechanics.

The script only reads the result artifacts written by ``main6.py``.  Its
calculation routines do not depend on module-level state, so the same
functions can also be imported from a notebook or a test.

Default usage
-------------
Run this file from ``01_0813_five_bar_linkage``::

    python main6_results_aggregate.py

Only directories matching ``result/main6_pilot_b*`` are read by default.
Incomplete seed directories are reported and skipped.  Re-running the command
after a long experiment finishes automatically incorporates the new result.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path


DEFAULT_CHECKPOINTS = (1000, 3000, 5000, 10000)
DEFAULT_TARGET_CELLS = (1, 2, 3, 4, 5, 6)
SELECTOR_ORDER = (
    "independent_random",
    "fixed_transport",
    "fixed_reverse",
    "learned_angle_gp",
    "learned_angle_gp_shuffled",
    "learned_angle_gp_cost_aware",
    "learned_angle_gp_cost_aware_shuffled",
)


def read_json(path: Path):
    """Return the decoded JSON value at ``path``."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_discovery_curve(path: Path):
    """Read a two-column discovery curve and return sorted ``(step, count)``.

    Duplicate steps are collapsed by retaining the last count.  A malformed,
    decreasing step/count sequence raises ``ValueError`` instead of silently
    producing a misleading AUC.
    """

    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number}: expected two columns")
        rows.append((int(float(parts[0])), int(float(parts[1]))))

    if not rows:
        raise ValueError(f"{path}: empty discovery curve")

    collapsed = {}
    for step, count in rows:
        if step < 0 or count < 0:
            raise ValueError(f"{path}: negative step or cell count")
        collapsed[step] = count
    curve = sorted(collapsed.items())
    if any(right[1] < left[1] for left, right in zip(curve, curve[1:])):
        raise ValueError(f"{path}: cell count is not cumulative")
    return curve


def cell_count_at(curve, step):
    """Evaluate the right-continuous cumulative discovery curve at ``step``."""

    count = 0
    for event_step, event_count in curve:
        if event_step > step:
            break
        count = event_count
    return int(count)


def interval_auc(curve, start_step, end_step, baseline_count=0):
    """Average ``cell_count - baseline_count`` over ``[start_step,end_step]``.

    The returned quantity has units of "mean number of cells", matching the
    ``discovery_auc`` definition in ``main6.py``.  It is not divided by the
    theoretical maximum of six cells.
    """

    start = int(start_step)
    end = int(end_step)
    if end <= start:
        return 0.0

    current_step = start
    current_count = cell_count_at(curve, start)
    area = 0.0
    for event_step, event_count in curve:
        clipped_step = min(int(event_step), end)
        if clipped_step <= start:
            continue
        area += (clipped_step - current_step) * (
            current_count - baseline_count
        )
        current_step = clipped_step
        current_count = int(event_count)
        if current_step >= end:
            break
    area += (end - current_step) * (current_count - baseline_count)
    return float(area / (end - start))


def first_step_to_cell(curve, target_cell_count):
    """Return the first mechanics step reaching ``target_cell_count`` or None."""

    target = int(target_cell_count)
    for step, count in curve:
        if count >= target:
            return int(step)
    return None


def finite_mean(values):
    """Return the arithmetic mean of finite numeric entries, else ``None``."""

    selected = [float(value) for value in values if is_finite_number(value)]
    return statistics.fmean(selected) if selected else None


def sample_std(values):
    """Return sample standard deviation of finite entries (0 for one entry)."""

    selected = [float(value) for value in values if is_finite_number(value)]
    if not selected:
        return None
    return statistics.stdev(selected) if len(selected) >= 2 else 0.0


def is_finite_number(value):
    """True for finite int/float values, excluding booleans."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def run_metrics(result_name, seed_directory, checkpoints, target_cells):
    """Compute all metrics for one completed seed directory.

    ``None`` is returned when either of the two required result files is still
    absent.  This is the only incompleteness expected while a long run is being
    written.
    """

    summary_path = seed_directory / "summary.json"
    curve_path = seed_directory / "discovery_curve.dat"
    if not summary_path.is_file() or not curve_path.is_file():
        return None

    summary = read_json(summary_path)
    curve = read_discovery_curve(curve_path)
    budget = int(summary.get("step_budget", summary.get("used_steps", 0)))
    used_steps = int(summary.get("used_steps", budget))
    selector = str(summary.get("selector", result_name))
    seed = int(summary.get("seed", seed_directory.name.removeprefix("seed_")))
    warm_step = int(summary.get("warmup_end_step", 0))
    warm_cells = int(
        summary.get("warmup_end_cells", cell_count_at(curve, warm_step))
    )

    full_auc = interval_auc(curve, 0, budget, baseline_count=0)
    post_total_auc = interval_auc(curve, warm_step, budget, baseline_count=0)
    post_incremental_auc = interval_auc(
        curve, warm_step, budget, baseline_count=warm_cells
    )
    final_cells = cell_count_at(curve, budget)

    row = {
        "result_name": result_name,
        "seed_directory": str(seed_directory),
        "algorithm_version": summary.get("algorithm_version"),
        "selector": selector,
        "seed": seed,
        "budget": budget,
        "used_steps": used_steps,
        "complete": used_steps >= budget,
        "final_cells": final_cells,
        "additional_cells_after_warmup": final_cells - warm_cells,
        "discovery_auc": full_auc,
        "reported_discovery_auc": float(
            summary.get("discovery_auc", full_auc)
        ),
        "auc_absolute_error": abs(
            full_auc - float(summary.get("discovery_auc", full_auc))
        ),
        "post_warmup_total_auc": post_total_auc,
        "post_warmup_incremental_auc": post_incremental_auc,
        "warmup_end_step": warm_step,
        "warmup_end_cells": warm_cells,
        "warmup_prefix_sha256": summary.get("warmup_prefix_sha256"),
        "num_rollouts": summary.get("num_rollouts"),
        "completed_second_actions": summary.get("completed_second_actions"),
        "training_rollouts": summary.get("training_rollouts"),
        "cost_training_rollouts": summary.get("cost_training_rollouts"),
        "flat_rollouts": summary.get("flat_rollouts"),
        "collision_rollouts": summary.get("collision_rollouts"),
        "fit_time_seconds": summary.get("fit_time_seconds"),
        "score_time_seconds": summary.get("score_time_seconds"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
    }
    if is_finite_number(row["num_rollouts"]):
        row["rollouts_per_1000_steps"] = (
            1000.0 * float(row["num_rollouts"]) / max(float(budget), 1.0)
        )
    if is_finite_number(row["completed_second_actions"]):
        row["completed_second_actions_per_1000_steps"] = (
            1000.0
            * float(row["completed_second_actions"])
            / max(float(budget), 1.0)
        )
    for checkpoint in checkpoints:
        key = f"cells_at_{int(checkpoint)}"
        row[key] = (
            cell_count_at(curve, checkpoint)
            if int(checkpoint) <= budget
            else None
        )
    for target in target_cells:
        step = first_step_to_cell(curve, target)
        row[f"step_to_{int(target)}_cells"] = (
            step if step is not None and step <= budget else None
        )
    return row


def discover_run_metrics(result_directory, pattern, checkpoints, target_cells):
    """Return ``(completed_rows, skipped_paths)`` for matching result roots."""

    rows = []
    skipped = []
    for result_root in sorted(result_directory.glob(pattern)):
        if not result_root.is_dir():
            continue
        for seed_directory in sorted(result_root.glob("seed_*")):
            if not seed_directory.is_dir():
                continue
            row = run_metrics(
                result_root.name,
                seed_directory,
                checkpoints,
                target_cells,
            )
            if row is None:
                skipped.append(str(seed_directory))
            else:
                rows.append(row)
    return rows, skipped


def read_canonical_coverage(result_directory, pattern):
    """Read optional five-bar canonical-coverage audit CSV files.

    Coverage audits are deliberately optional because ``main6.py`` itself is
    structure-agnostic.  Later audit files override earlier duplicate keys,
    making it safe to regenerate a more recent audit before rerunning this
    aggregation script.
    """

    coverage = {}
    source_files = []
    for path in sorted(result_directory.glob(pattern)):
        if not path.is_file():
            continue
        source_files.append(str(path))
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                result_name = row.get("result_name", "").strip()
                seed_text = row.get("seed", "").strip().removeprefix("seed_")
                if not result_name or not seed_text:
                    continue
                key = (result_name, int(seed_text))
                coverage[key] = {
                    "canonical_coverage_count": int(
                        row["canonical_coverage_count"]
                    ),
                    "canonical_extra_cell_count": int(row["extra_cell_count"]),
                    "canonical_covered_ids": row.get(
                        "canonical_covered_ids", ""
                    ).strip(),
                }
    return coverage, source_files


def merge_canonical_coverage(rows, coverage):
    """Return copies of run rows enriched with optional canonical coverage."""

    merged = []
    for row in rows:
        enriched = dict(row)
        enriched.update(coverage.get((row["result_name"], row["seed"]), {}))
        merged.append(enriched)
    return merged


def selector_sort_key(selector):
    """Stable, human-oriented selector ordering."""

    try:
        return (SELECTOR_ORDER.index(selector), selector)
    except ValueError:
        return (len(SELECTOR_ORDER), selector)


def aggregate_metric_rows(rows, metric_names):
    """Aggregate run rows by ``(budget, selector)``."""

    groups = {}
    for row in rows:
        groups.setdefault((row["budget"], row["selector"]), []).append(row)

    aggregates = []
    for (budget, selector), group in sorted(
        groups.items(), key=lambda item: (item[0][0], selector_sort_key(item[0][1]))
    ):
        aggregate = {
            "budget": budget,
            "selector": selector,
            "num_seeds": len(group),
            "seeds": sorted(int(row["seed"]) for row in group),
        }
        for metric in metric_names:
            values = [row.get(metric) for row in group]
            finite = [value for value in values if is_finite_number(value)]
            aggregate[f"{metric}_mean"] = finite_mean(finite)
            aggregate[f"{metric}_std"] = sample_std(finite)
            aggregate[f"{metric}_n"] = len(finite)
        aggregates.append(aggregate)
    return aggregates


def paired_comparisons(rows, metric_names):
    """Build seed-matched selector differences and their means.

    Every difference is defined as ``selector_a - selector_b``.  Selector order
    follows ``SELECTOR_ORDER`` so the sign remains stable between reruns.
    """

    by_budget_seed = {}
    for row in rows:
        by_budget_seed.setdefault((row["budget"], row["seed"]), {})[
            row["selector"]
        ] = row

    paired_rows = []
    for (budget, seed), selector_rows in sorted(by_budget_seed.items()):
        selectors = sorted(selector_rows, key=selector_sort_key)
        for selector_a, selector_b in itertools.combinations(selectors, 2):
            pair = {
                "budget": budget,
                "seed": seed,
                "selector_a": selector_a,
                "selector_b": selector_b,
            }
            for metric in metric_names:
                value_a = selector_rows[selector_a].get(metric)
                value_b = selector_rows[selector_b].get(metric)
                pair[f"delta_{metric}"] = (
                    float(value_a) - float(value_b)
                    if is_finite_number(value_a) and is_finite_number(value_b)
                    else None
                )
            paired_rows.append(pair)

    grouped = {}
    for row in paired_rows:
        key = (row["budget"], row["selector_a"], row["selector_b"])
        grouped.setdefault(key, []).append(row)
    paired_aggregates = []
    for (budget, selector_a, selector_b), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            selector_sort_key(item[0][1]),
            selector_sort_key(item[0][2]),
        ),
    ):
        aggregate = {
            "budget": budget,
            "selector_a": selector_a,
            "selector_b": selector_b,
            "difference_definition": "selector_a - selector_b",
            "num_paired_seeds": len(group),
            "seeds": sorted(int(row["seed"]) for row in group),
        }
        for metric in metric_names:
            key = f"delta_{metric}"
            values = [row.get(key) for row in group]
            aggregate[f"{key}_mean"] = finite_mean(values)
            aggregate[f"{key}_std"] = sample_std(values)
            aggregate[f"{key}_n"] = sum(
                is_finite_number(value) for value in values
            )
        paired_aggregates.append(aggregate)
    return paired_rows, paired_aggregates


def warmup_consistency(rows):
    """Check that all policies share exactly the same warm-up for each seed."""

    groups = {}
    for row in rows:
        groups.setdefault(row["seed"], []).append(row)

    checks = []
    for seed, group in sorted(groups.items()):
        hashes = sorted(
            {row["warmup_prefix_sha256"] for row in group if row["warmup_prefix_sha256"]}
        )
        steps = sorted({row["warmup_end_step"] for row in group})
        cells = sorted({row["warmup_end_cells"] for row in group})
        checks.append(
            {
                "seed": int(seed),
                "num_runs": len(group),
                "consistent": len(hashes) == len(steps) == len(cells) == 1,
                "hashes": hashes,
                "warmup_end_steps": steps,
                "warmup_end_cells": cells,
            }
        )
    return checks


def csv_value(value):
    """Represent ``None`` as an empty CSV field."""

    return "" if value is None else value


def write_csv(path, rows):
    """Write a union of row keys to UTF-8 CSV."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def format_mean_std(mean, std, digits=3):
    """Format a mean and sample standard deviation for Markdown."""

    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_report(payload, checkpoints, target_cells):
    """Return a compact, self-contained Markdown report."""

    has_canonical = any(
        is_finite_number(row.get("canonical_coverage_count"))
        for row in payload["runs"]
    )
    lines = [
        "# main6 K=2 online aggregate",
        "",
        "`discovery_auc` は全予算における累積 cell 数の時間平均、",
        "`post-warmup incremental AUC` は共通 warm-up 終了時の cell 数を",
        "差し引いた、その後の平均追加 cell 数である。いずれも最大6で",
        "正規化した値ではない。標準偏差は seed 間の標本標準偏差。",
        "",
        "## Aggregate",
        "",
        "| budget | selector | seeds | final cells | canonical coverage | discovery AUC | post-warm-up incremental AUC |"
        if has_canonical
        else "| budget | selector | seeds | final cells | discovery AUC | post-warm-up incremental AUC |",
        "|---:|---|---:|---:|---:|---:|---:|"
        if has_canonical
        else "|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["aggregates"]:
        canonical = format_mean_std(
            row.get("canonical_coverage_count_mean"),
            row.get("canonical_coverage_count_std"),
        )
        template = (
            "| {budget} | {selector} | {num_seeds} | {cells} | {canonical} | {auc} | {post} |"
            if has_canonical
            else "| {budget} | {selector} | {num_seeds} | {cells} | {auc} | {post} |"
        )
        lines.append(
            template.format(
                budget=row["budget"],
                selector=row["selector"],
                num_seeds=row["num_seeds"],
                cells=format_mean_std(
                    row.get("final_cells_mean"), row.get("final_cells_std")
                ),
                canonical=canonical,
                auc=format_mean_std(
                    row.get("discovery_auc_mean"),
                    row.get("discovery_auc_std"),
                    digits=4,
                ),
                post=format_mean_std(
                    row.get("post_warmup_incremental_auc_mean"),
                    row.get("post_warmup_incremental_auc_std"),
                    digits=4,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Mechanics-budget throughput",
            "",
            "| budget | selector | rollouts / 1000 steps | completed second actions / 1000 steps |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in payload["aggregates"]:
        lines.append(
            "| {budget} | {selector} | {rollouts} | {completed} |".format(
                budget=row["budget"],
                selector=row["selector"],
                rollouts=format_mean_std(
                    row.get("rollouts_per_1000_steps_mean"),
                    row.get("rollouts_per_1000_steps_std"),
                    digits=2,
                ),
                completed=format_mean_std(
                    row.get("completed_second_actions_per_1000_steps_mean"),
                    row.get("completed_second_actions_per_1000_steps_std"),
                    digits=2,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Per-seed checkpoints",
            "",
            "| budget | selector | seed | final | post-warm-up AUC | "
            + " | ".join(f"C@{step}" for step in checkpoints)
            + " | "
            + " | ".join(f"T{target}" for target in target_cells)
            + " |",
            "|---:|---|---:|---:|---:|"
            + "---:|" * (len(checkpoints) + len(target_cells)),
        ]
    )
    for row in sorted(
        payload["runs"],
        key=lambda item: (
            item["budget"],
            selector_sort_key(item["selector"]),
            item["seed"],
        ),
    ):
        checkpoint_values = [
            "--" if row.get(f"cells_at_{step}") is None else str(row[f"cells_at_{step}"])
            for step in checkpoints
        ]
        target_values = [
            "--" if row.get(f"step_to_{target}_cells") is None else str(row[f"step_to_{target}_cells"])
            for target in target_cells
        ]
        lines.append(
            "| {budget} | {selector} | {seed} | {final} | {post:.4f} | {rest} |".format(
                budget=row["budget"],
                selector=row["selector"],
                seed=row["seed"],
                final=row["final_cells"],
                post=row["post_warmup_incremental_auc"],
                rest=" | ".join(checkpoint_values + target_values),
            )
        )

    control_pairs = (
        ("learned_angle_gp", "learned_angle_gp_shuffled"),
        (
            "learned_angle_gp_cost_aware",
            "learned_angle_gp_cost_aware_shuffled",
        ),
    )
    for selector, shuffled in control_pairs:
        matching = [
            row
            for row in payload["paired_aggregates"]
            if row["selector_a"] == selector and row["selector_b"] == shuffled
        ]
        if not matching:
            continue
        lines.extend(
            [
                "",
                f"## {selector} versus shuffled-score control",
                "",
                f"差は `{selector} - {shuffled}`。",
                "",
                "| budget | paired seeds | Δ final cells | Δ discovery AUC | Δ post-warm-up incremental AUC |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in matching:
            lines.append(
                "| {budget} | {n} | {cells} | {auc} | {post} |".format(
                    budget=row["budget"],
                    n=row["num_paired_seeds"],
                    cells=format_mean_std(
                        row.get("delta_final_cells_mean"),
                        row.get("delta_final_cells_std"),
                    ),
                    auc=format_mean_std(
                        row.get("delta_discovery_auc_mean"),
                        row.get("delta_discovery_auc_std"),
                        digits=4,
                    ),
                    post=format_mean_std(
                        row.get("delta_post_warmup_incremental_auc_mean"),
                        row.get("delta_post_warmup_incremental_auc_std"),
                        digits=4,
                    ),
                )
            )

    lines.extend(["", "## Warm-up identity check", ""])
    for check in payload["warmup_checks"]:
        status = "PASS" if check["consistent"] else "FAIL"
        lines.append(
            f"- seed {check['seed']}: {status}; "
            f"step={check['warmup_end_steps']}, cells={check['warmup_end_cells']}, "
            f"hash={check['hashes']}"
        )
    if payload["skipped_seed_directories"]:
        lines.extend(["", "## Skipped incomplete directories", ""])
        lines.extend(
            f"- `{path}`" for path in payload["skipped_seed_directories"]
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(
    rows,
    skipped,
    checkpoints,
    target_cells,
    canonical_coverage_sources=(),
):
    """Build all aggregate and paired tables from immutable run rows."""

    primary_metrics = [
        "final_cells",
        "additional_cells_after_warmup",
        "discovery_auc",
        "post_warmup_total_auc",
        "post_warmup_incremental_auc",
        "num_rollouts",
        "completed_second_actions",
        "training_rollouts",
        "cost_training_rollouts",
        "rollouts_per_1000_steps",
        "completed_second_actions_per_1000_steps",
        "fit_time_seconds",
        "score_time_seconds",
        "wall_time_seconds",
        "canonical_coverage_count",
        "canonical_extra_cell_count",
    ]
    primary_metrics.extend(f"cells_at_{step}" for step in checkpoints)
    primary_metrics.extend(f"step_to_{target}_cells" for target in target_cells)
    paired_rows, paired_aggregates = paired_comparisons(rows, primary_metrics)
    return {
        "definitions": {
            "discovery_auc": "mean cumulative cell count over [0,budget]",
            "post_warmup_total_auc": (
                "mean cumulative cell count over [warmup_end_step,budget]"
            ),
            "post_warmup_incremental_auc": (
                "mean cells beyond warmup_end_cells over "
                "[warmup_end_step,budget]"
            ),
            "paired_delta": "selector_a - selector_b on the same seed",
        },
        "checkpoints": list(checkpoints),
        "target_cells": list(target_cells),
        "runs": rows,
        "aggregates": aggregate_metric_rows(rows, primary_metrics),
        "paired_runs": paired_rows,
        "paired_aggregates": paired_aggregates,
        "warmup_checks": warmup_consistency(rows),
        "canonical_coverage_sources": list(canonical_coverage_sources),
        "skipped_seed_directories": skipped,
    }


def parse_integer_list(text):
    """Parse a comma-separated list of integers."""

    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--pattern", default="main6_pilot_b*")
    parser.add_argument(
        "--coverage-pattern",
        default="main6_b*_canonical*.csv",
        help="optional five-bar canonical coverage audit CSV glob",
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_integer_list,
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument(
        "--target-cells",
        type=parse_integer_list,
        default=DEFAULT_TARGET_CELLS,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("result/main6_online_aggregate.json"),
    )
    parser.add_argument(
        "--runs-csv-output",
        type=Path,
        default=Path("result/main6_online_aggregate.csv"),
    )
    parser.add_argument(
        "--paired-csv-output",
        type=Path,
        default=Path("result/main6_online_paired.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("MAIN6_ONLINE_AGGREGATE.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = tuple(sorted(set(args.checkpoints)))
    target_cells = tuple(sorted(set(args.target_cells)))
    rows, skipped = discover_run_metrics(
        args.result_dir,
        args.pattern,
        checkpoints,
        target_cells,
    )
    coverage, coverage_sources = read_canonical_coverage(
        args.result_dir, args.coverage_pattern
    )
    rows = merge_canonical_coverage(rows, coverage)
    if not rows:
        raise SystemExit(
            f"No complete seed results matched {args.result_dir / args.pattern}"
        )
    payload = build_payload(
        rows,
        skipped,
        checkpoints,
        target_cells,
        canonical_coverage_sources=coverage_sources,
    )

    for path in (
        args.json_output,
        args.runs_csv_output,
        args.paired_csv_output,
        args.markdown_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.runs_csv_output, payload["runs"])
    write_csv(args.paired_csv_output, payload["paired_runs"])
    args.markdown_output.write_text(
        markdown_report(payload, checkpoints, target_cells),
        encoding="utf-8",
    )

    print(f"complete runs: {len(rows)}")
    print(f"skipped incomplete directories: {len(skipped)}")
    print(f"JSON: {args.json_output}")
    print(f"runs CSV: {args.runs_csv_output}")
    print(f"paired CSV: {args.paired_csv_output}")
    print(f"Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
