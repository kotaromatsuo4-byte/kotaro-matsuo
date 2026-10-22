"""Aggregate ``gp_paired_panel.py`` summaries over independent seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
METRICS = (
    "mean_transport_rank",
    "transport_top1_rate",
    "transport_mrr",
    "new_cell_success_at_1",
    "new_cell_success_at_2",
    "new_cell_roc_auc",
    "new_cell_average_precision",
)


def resolve_result_directory(value):
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    local = SCRIPT_DIRECTORY / "result" / value
    if local.is_dir():
        return local.resolve()
    raise FileNotFoundError(value)


def load_seed_summaries(result_directory):
    summaries = []
    for seed_directory in sorted(Path(result_directory).glob("seed_*")):
        path = seed_directory / "summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError("no paired-panel summary.json files were found")
    return summaries


def aggregate(summaries, regime="early_one_cell"):
    methods = sorted(
        {
            method
            for summary in summaries
            for method in summary["regimes"][regime]["models"]
        }
    )
    rows = []
    for method in methods:
        row = {
            "method": method,
            "num_seeds": len(summaries),
            "panel_rollouts_mean": mean(
                summary["panel_rollouts"] for summary in summaries
            ),
        }
        for metric in METRICS:
            values = [
                summary["regimes"][regime]["models"][method].get(metric)
                for summary in summaries
            ]
            finite = [float(value) for value in values if value is not None]
            row[f"{metric}_mean"] = mean(finite) if finite else None
            row[f"{metric}_sd"] = (
                stdev(finite) if len(finite) >= 2 else None
            )
        rows.append(row)
    return rows


def write_csv(path, rows):
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    result_directory = resolve_result_directory(arguments.result)
    summaries = load_seed_summaries(result_directory)
    rows = aggregate(summaries)
    output = (
        Path(arguments.output)
        if arguments.output
        else result_directory / "paired_panel_aggregate.csv"
    )
    write_csv(output, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"CSV saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
