"""Aggregate the paired-warmup main6 experiment and its legacy control.

This script is intentionally read-only with respect to mechanics results.  It
reuses ``main6_results_aggregate.py`` for discovery-curve metrics and
``five_bar_coverage_audit.py`` for the canonical-six audit, then adds checks
specific to the paired warmup:

* all selectors must have an identical warmup prefix for each seed;
* every warmup group must contain the same-q1 transport, orthogonal, and
  reverse branches exactly once;
* the summary's requested/completed group and branch counts must agree with
  ``rollout_diagnostics.csv``;
* paired-warmup runs are compared both with one another and with the legacy
  one-angle-per-q1 warmup on the same selector and seed.

The command exits before writing a final report if an expected run is still
missing, so it is safe to rerun after long jobs finish::

    python main6_paired_warmup_aggregate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from five_bar_coverage_audit import (
    inspect_results,
    load_five_bar_model,
    write_summary_csv,
)
from main6_results_aggregate import (
    build_payload,
    discover_run_metrics,
    finite_mean,
    format_mean_std,
    merge_canonical_coverage,
    read_canonical_coverage,
    sample_std,
    selector_sort_key,
    write_csv,
)


CHECKPOINTS = (1000, 3000, 5000, 10000)
TARGET_CELLS = (1, 2, 3, 4, 5, 6)
EXPECTED_SELECTORS = (
    "independent_random",
    "fixed_transport",
    "learned_angle_gp",
    "learned_angle_gp_shuffled",
)
EXPECTED_SEEDS = (600, 601, 602)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_int(row, key):
    return int(float(row[key]))


def _as_float(row, key):
    return float(row[key])


def _close(left, right, tolerance=1.0e-12):
    return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)


def audit_paired_warmup(seed_directory):
    """Return a detailed structural audit for one paired-warmup run."""

    seed_directory = Path(seed_directory)
    summary = _read_json(seed_directory / "summary.json")
    diagnostics_path = seed_directory / "rollout_diagnostics.csv"
    with diagnostics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    warm_rows = [
        row for row in all_rows if _as_int(row, "paired_warmup_group") >= 0
    ]

    requested = int(summary["paired_warmup_requested_groups"])
    grouped = {}
    for row in warm_rows:
        grouped.setdefault(_as_int(row, "paired_warmup_group"), []).append(row)

    expected_roles = ("transport", "orthogonal", "reverse")
    expected_codes = (14, 15, 16)
    group_checks = []
    for group_index in range(requested):
        rows = sorted(
            grouped.get(group_index, []),
            key=lambda row: _as_int(row, "paired_warmup_branch"),
        )
        branch_indices = [_as_int(row, "paired_warmup_branch") for row in rows]
        codes = [_as_int(row, "selection_code") for row in rows]
        roles = [row["paired_angle_role"] for row in rows]
        q1_ids = sorted({_as_int(row, "q1_state_id") for row in rows})
        parent_ids = sorted({_as_int(row, "parent_id") for row in rows})
        shared_cost = [_as_int(row, "shared_q1_cost_included") for row in rows]
        theta_ok = False
        if len(rows) == 3:
            thetas = [_as_float(row, "theta") for row in rows]
            expected_orthogonal = math.pi / 2.0 if group_index % 2 == 0 else -math.pi / 2.0
            theta_ok = (
                _close(thetas[0], 0.0)
                and _close(thetas[1], expected_orthogonal)
                and _close(thetas[2], math.pi)
            )
        ok = (
            len(rows) == 3
            and branch_indices == [0, 1, 2]
            and codes == list(expected_codes)
            and roles == list(expected_roles)
            and len(q1_ids) == 1
            and len(parent_ids) == 1
            and shared_cost == [1, 0, 0]
            and theta_ok
        )
        group_checks.append(
            {
                "group": group_index,
                "ok": ok,
                "row_count": len(rows),
                "branches": branch_indices,
                "selection_codes": codes,
                "roles": roles,
                "q1_state_ids": q1_ids,
                "parent_ids": parent_ids,
                "shared_q1_cost_flags": shared_cost,
                "theta_ok": theta_ok,
            }
        )

    summary_counts = summary.get("selection_code_counts", {})
    expected_branches = 3 * requested
    summary_ok = (
        bool(summary.get("paired_warmup"))
        and bool(summary.get("warmup_completed"))
        and int(summary.get("paired_warmup_completed_groups", -1)) == requested
        and int(summary.get("paired_warmup_completed_branches", -1))
        == expected_branches
        and len(warm_rows) == expected_branches
        and all(int(summary_counts.get(str(code), 0)) == requested for code in expected_codes)
    )
    all_groups_ok = len(grouped) == requested and all(
        check["ok"] for check in group_checks
    )
    return {
        "result_name": seed_directory.parent.name,
        "seed": int(summary["seed"]),
        "seed_directory": str(seed_directory.resolve()),
        "requested_groups": requested,
        "completed_groups": int(summary["paired_warmup_completed_groups"]),
        "completed_branches": int(summary["paired_warmup_completed_branches"]),
        "diagnostic_warmup_rows": len(warm_rows),
        "warmup_end_step": int(summary["warmup_end_step"]),
        "warmup_end_cells": int(summary["warmup_end_cells"]),
        "warmup_prefix_sha256": summary["warmup_prefix_sha256"],
        "summary_counts_ok": summary_ok,
        "all_groups_ok": all_groups_ok,
        "passed": summary_ok and all_groups_ok,
        "group_checks": group_checks,
    }


def cross_selector_prefix_checks(audits):
    """Verify identical paired prefixes across selectors for every seed."""

    by_seed = {}
    for audit in audits:
        by_seed.setdefault(audit["seed"], []).append(audit)
    checks = []
    for seed, group in sorted(by_seed.items()):
        hashes = sorted({row["warmup_prefix_sha256"] for row in group})
        steps = sorted({row["warmup_end_step"] for row in group})
        cells = sorted({row["warmup_end_cells"] for row in group})
        selectors = sorted(row["result_name"] for row in group)
        passed = (
            len(group) == len(EXPECTED_SELECTORS)
            and len(hashes) == len(steps) == len(cells) == 1
            and all(row["passed"] for row in group)
        )
        checks.append(
            {
                "seed": seed,
                "num_selectors": len(group),
                "result_names": selectors,
                "hashes": hashes,
                "warmup_end_steps": steps,
                "warmup_end_cells": cells,
                "passed": passed,
            }
        )
    return checks


def expected_result_names(prefix):
    return tuple(f"{prefix}_{selector}" for selector in EXPECTED_SELECTORS)


def find_missing(result_directory, prefix):
    missing = []
    for result_name in expected_result_names(prefix):
        for seed in EXPECTED_SEEDS:
            seed_directory = result_directory / result_name / f"seed_{seed:03d}"
            required = (
                seed_directory / "summary.json",
                seed_directory / "discovery_curve.dat",
                seed_directory / "rollout_diagnostics.csv",
            )
            if not all(path.is_file() for path in required):
                missing.append(str(seed_directory))
    return missing


def generate_coverage(base_directory, result_directory, result_names, output_path):
    """Run the canonical-six audit through its importable implementation."""

    reference_path = (
        result_directory / "reachability_audit" / "reachability_report.json"
    )
    reference_report = _read_json(reference_path)
    model = load_five_bar_model(base_directory / "model")
    roots = [result_directory / name for name in result_names]
    report = inspect_results(roots, reference_report, model)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(report, output_path.with_suffix(".csv"))
    return report


def paired_minus_legacy(paired_rows, legacy_rows):
    """Build same-selector, same-seed deltas (paired warmup minus legacy)."""

    metrics = (
        "final_cells",
        "canonical_coverage_count",
        "discovery_auc",
        "post_warmup_incremental_auc",
        "warmup_end_step",
        "warmup_end_cells",
    )
    legacy_index = {
        (row["selector"], row["seed"], row["budget"]): row for row in legacy_rows
    }
    comparisons = []
    for paired in paired_rows:
        legacy = legacy_index.get(
            (paired["selector"], paired["seed"], paired["budget"])
        )
        if legacy is None:
            continue
        row = {
            "selector": paired["selector"],
            "seed": paired["seed"],
            "budget": paired["budget"],
            "difference_definition": "paired_warmup - legacy_warmup",
        }
        for metric in metrics:
            left = paired.get(metric)
            right = legacy.get(metric)
            row[f"paired_{metric}"] = left
            row[f"legacy_{metric}"] = right
            row[f"delta_{metric}"] = (
                float(left) - float(right)
                if isinstance(left, (int, float))
                and isinstance(right, (int, float))
                else None
            )
        comparisons.append(row)
    return comparisons


def aggregate_legacy_comparisons(rows):
    metrics = (
        "final_cells",
        "canonical_coverage_count",
        "discovery_auc",
        "post_warmup_incremental_auc",
        "warmup_end_step",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["selector"], []).append(row)
    output = []
    for selector, group in sorted(grouped.items(), key=lambda item: selector_sort_key(item[0])):
        aggregate = {
            "selector": selector,
            "num_paired_seeds": len(group),
            "seeds": sorted(row["seed"] for row in group),
            "difference_definition": "paired_warmup - legacy_warmup",
        }
        for metric in metrics:
            values = [row.get(f"delta_{metric}") for row in group]
            aggregate[f"delta_{metric}_mean"] = finite_mean(values)
            aggregate[f"delta_{metric}_std"] = sample_std(values)
        output.append(aggregate)
    return output


def _format_bool(value):
    return "PASS" if value else "FAIL"


def markdown_report(payload):
    paired = payload["paired_experiment"]
    lines = [
        "# main6 paired-warmup results",
        "",
        "同じ `q1` から `theta=0`, `theta=+/-pi/2`, `theta=pi` の3分岐を評価する",
        "paired warmupを16組（48終端教師）与えた後、4つの選択則を10000 mechanics stepで比較した。",
        "`discovery AUC` は全予算上の累積cell数の時間平均、`post-warmup incremental AUC` は",
        "warmup終了時のcell数を差し引いた後半区間の時間平均であり、最大6に正規化した値ではない。",
        "",
        "## Paired-warmup aggregate",
        "",
        "| selector | seeds | final cells | canonical coverage | discovery AUC | post-warmup incremental AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired["aggregates"]:
        lines.append(
            "| {selector} | {n} | {cells} | {canonical} | {auc} | {post} |".format(
                selector=row["selector"],
                n=row["num_seeds"],
                cells=format_mean_std(row.get("final_cells_mean"), row.get("final_cells_std")),
                canonical=format_mean_std(
                    row.get("canonical_coverage_count_mean"),
                    row.get("canonical_coverage_count_std"),
                ),
                auc=format_mean_std(
                    row.get("discovery_auc_mean"), row.get("discovery_auc_std"), digits=4
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
            "## Per-seed checkpoints",
            "",
            "| selector | seed | final | canonical | C@1000 | C@3000 | C@5000 | C@10000 | T3 | T4 | T5 | T6 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        paired["runs"], key=lambda item: (selector_sort_key(item["selector"]), item["seed"])
    ):
        value = lambda key: "--" if row.get(key) is None else str(row[key])
        lines.append(
            "| {selector} | {seed} | {final} | {canonical} | {c1} | {c3} | {c5} | {c10} | {t3} | {t4} | {t5} | {t6} |".format(
                selector=row["selector"], seed=row["seed"], final=row["final_cells"],
                canonical=value("canonical_coverage_count"), c1=value("cells_at_1000"),
                c3=value("cells_at_3000"), c5=value("cells_at_5000"),
                c10=value("cells_at_10000"), t3=value("step_to_3_cells"),
                t4=value("step_to_4_cells"), t5=value("step_to_5_cells"),
                t6=value("step_to_6_cells"),
            )
        )

    lines.extend(
        [
            "",
            "## Paired warmup versus legacy warmup",
            "",
            "差は `paired - legacy`。両者は同じselector・seed・10000-step予算だが、",
            "paired warmupは48分岐、legacy warmupは16個の異なるq1で1分岐ずつなので、",
            "この表は教師データの情報量とwarmupコストを合わせた実用上の差を表す。",
            "",
            "| selector | paired seeds | delta final cells | delta canonical | delta discovery AUC | delta post-warmup AUC | delta warmup steps |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["paired_minus_legacy_aggregate"]:
        lines.append(
            "| {selector} | {n} | {cells} | {canonical} | {auc} | {post} | {steps} |".format(
                selector=row["selector"], n=row["num_paired_seeds"],
                cells=format_mean_std(row.get("delta_final_cells_mean"), row.get("delta_final_cells_std")),
                canonical=format_mean_std(row.get("delta_canonical_coverage_count_mean"), row.get("delta_canonical_coverage_count_std")),
                auc=format_mean_std(row.get("delta_discovery_auc_mean"), row.get("delta_discovery_auc_std"), digits=4),
                post=format_mean_std(row.get("delta_post_warmup_incremental_auc_mean"), row.get("delta_post_warmup_incremental_auc_std"), digits=4),
                steps=format_mean_std(row.get("delta_warmup_end_step_mean"), row.get("delta_warmup_end_step_std"), digits=1),
            )
        )

    lines.extend(["", "## Warmup audit", ""])
    for check in payload["cross_selector_prefix_checks"]:
        lines.append(
            "- seed {seed}: {status}; selectors={n}, step={steps}, cells={cells}, hash={hashes}".format(
                seed=check["seed"], status=_format_bool(check["passed"]),
                n=check["num_selectors"], steps=check["warmup_end_steps"],
                cells=check["warmup_end_cells"], hashes=check["hashes"],
            )
        )
    all_branch_audits = all(row["passed"] for row in payload["paired_warmup_audits"])
    lines.extend(
        [
            "",
            f"- 16 groups x 3 branches structural audit: {_format_bool(all_branch_audits)}",
            "- 各組でbranch index `[0,1,2]`、selection code `[14,15,16]`、",
            "  role `[transport,orthogonal,reverse]`、同一 `q1_state_id`、",
            "  `theta=[0,+/-pi/2,pi]`、q1コストの一回だけの計上を確認した。",
            "",
            "## Interpretation",
            "",
        ]
    )

    aggregates = {row["selector"]: row for row in paired["aggregates"]}
    gp = aggregates.get("learned_angle_gp")
    shuffled = aggregates.get("learned_angle_gp_shuffled")
    independent = aggregates.get("independent_random")
    if gp and shuffled:
        gp_delta = gp.get("final_cells_mean", 0.0) - shuffled.get("final_cells_mean", 0.0)
        gp_auc_delta = gp.get("discovery_auc_mean", 0.0) - shuffled.get("discovery_auc_mean", 0.0)
        lines.append(
            f"- learned-angle GP minus shuffled controlは平均final cellで {gp_delta:+.3f}、"
            f"discovery AUCで {gp_auc_delta:+.4f}。scoreと結果の因果的対応はこの差で判断する。"
        )
    if gp and independent:
        cells_delta = gp.get("final_cells_mean", 0.0) - independent.get("final_cells_mean", 0.0)
        auc_delta = gp.get("discovery_auc_mean", 0.0) - independent.get("discovery_auc_mean", 0.0)
        lines.append(
            f"- learned-angle GP minus independent randomは平均final cellで {cells_delta:+.3f}、"
            f"discovery AUCで {auc_delta:+.4f}。"
        )
    lines.extend(
        [
            "- 3 seedsだけなので標準偏差とseed別結果を併記し、汎用性や有意差は主張しない。",
            "- paired warmupでGPが改善しない場合、単なる教師不足だけではなく、",
            "  theta選択より前段のparent/q1選択が探索性能を支配している可能性が残る。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--paired-prefix", default="main6_paired16_b10000"
    )
    parser.add_argument(
        "--legacy-pattern", default="main6_pilot_b10000_*"
    )
    parser.add_argument(
        "--output-prefix", default="main6_paired_warmup"
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("MAIN6_PAIRED_WARMUP_AUDIT.md"),
        help="structural-audit report (kept separate from the generic aggregate)",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="aggregate completed runs instead of exiting on missing expected runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_directory = Path(__file__).resolve().parent
    result_directory = args.result_dir
    if not result_directory.is_absolute():
        result_directory = base_directory / result_directory
    result_directory.mkdir(parents=True, exist_ok=True)

    missing = find_missing(result_directory, args.paired_prefix)
    if missing and not args.allow_incomplete:
        print("Paired experiment is incomplete; rerun this command later.")
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit(2)

    result_names = [
        name for name in expected_result_names(args.paired_prefix)
        if (result_directory / name).is_dir()
    ]
    coverage_path = result_directory / f"{args.output_prefix}_canonical.json"
    coverage_report = generate_coverage(
        base_directory, result_directory, result_names, coverage_path
    )

    paired_rows, skipped = discover_run_metrics(
        result_directory,
        f"{args.paired_prefix}_*",
        CHECKPOINTS,
        TARGET_CELLS,
    )
    paired_rows = [
        row for row in paired_rows if row["result_name"] in result_names
    ]
    coverage, coverage_sources = read_canonical_coverage(
        result_directory, f"{args.output_prefix}_canonical.csv"
    )
    paired_rows = merge_canonical_coverage(paired_rows, coverage)
    paired_payload = build_payload(
        paired_rows,
        skipped,
        CHECKPOINTS,
        TARGET_CELLS,
        canonical_coverage_sources=coverage_sources,
    )

    legacy_rows, _ = discover_run_metrics(
        result_directory, args.legacy_pattern, CHECKPOINTS, TARGET_CELLS
    )
    legacy_coverage, _ = read_canonical_coverage(
        result_directory, "main6_b10000_canonical*.csv"
    )
    legacy_rows = merge_canonical_coverage(legacy_rows, legacy_coverage)
    selected_keys = {
        (row["selector"], row["seed"], row["budget"]) for row in paired_rows
    }
    legacy_rows = [
        row for row in legacy_rows
        if (row["selector"], row["seed"], row["budget"]) in selected_keys
    ]

    audits = []
    for result_name in result_names:
        for seed in EXPECTED_SEEDS:
            seed_directory = result_directory / result_name / f"seed_{seed:03d}"
            if (seed_directory / "summary.json").is_file():
                audits.append(audit_paired_warmup(seed_directory))
    prefix_checks = cross_selector_prefix_checks(audits)
    comparisons = paired_minus_legacy(paired_rows, legacy_rows)
    comparison_aggregates = aggregate_legacy_comparisons(comparisons)

    payload = {
        "definitions": {
            "paired_warmup": "16 shared-q1 groups with theta 0, +/-pi/2, pi",
            "paired_minus_legacy": "paired warmup metric - legacy warmup metric on same selector and seed",
            "canonical_coverage": "coverage of the six frozen five-bar canonical witnesses",
        },
        "expected_result_names": result_names,
        "expected_seeds": list(EXPECTED_SEEDS),
        "missing_seed_directories": missing,
        "paired_experiment": paired_payload,
        "legacy_runs": legacy_rows,
        "paired_minus_legacy_runs": comparisons,
        "paired_minus_legacy_aggregate": comparison_aggregates,
        "paired_warmup_audits": audits,
        "cross_selector_prefix_checks": prefix_checks,
        "canonical_coverage_report": coverage_report,
    }

    json_path = result_directory / f"{args.output_prefix}_aggregate.json"
    runs_csv_path = result_directory / f"{args.output_prefix}_runs.csv"
    pairwise_csv_path = result_directory / f"{args.output_prefix}_pairwise.csv"
    legacy_csv_path = result_directory / f"{args.output_prefix}_vs_legacy.csv"
    markdown_path = args.markdown_output
    if not markdown_path.is_absolute():
        markdown_path = base_directory / markdown_path
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(runs_csv_path, paired_payload["runs"])
    write_csv(pairwise_csv_path, paired_payload["paired_runs"])
    write_csv(legacy_csv_path, comparisons)
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")

    print(f"complete paired runs: {len(paired_rows)}")
    print(f"all branch audits pass: {all(row['passed'] for row in audits)}")
    print(f"all prefix audits pass: {all(row['passed'] for row in prefix_checks)}")
    print(f"JSON: {json_path}")
    print(f"runs CSV: {runs_csv_path}")
    print(f"pairwise CSV: {pairwise_csv_path}")
    print(f"legacy CSV: {legacy_csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
