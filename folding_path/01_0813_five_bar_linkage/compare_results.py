"""Compare multiple ``main5`` result directories.

Example
-------
Run from any working directory::

    python compare_results.py \
        pilot_default_random pilot_default_gp_ucb pilot_default_gp_posterior

The script reads only existing result files.  It prints seed-wise rows and one
mean row per result to stdout, and writes the same table to CSV.  By default the
CSV is ``result/main5_comparison.csv`` beside this script.

``post_trunk_auc`` measures only discoveries made after the common initial
deterministic trunk:

    1 / (B - b0) * integral[b0, B] (C(b) - C(b0)) db

where ``b0`` is the trunk end step and ``C`` is the cumulative elite-cell
count.  Thus a cell already found by the common trunk does not obscure the
effect of perturbation selection.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path


SELECTION_TRUNK = 0
SELECTION_RANDOM = 1
SELECTION_EPSILON_RANDOM = 2
SELECTION_GP_UCB = 3
SELECTION_GP_POSTERIOR = 4
SELECTION_COLD_START = 5
SELECTION_TRANSPORT_HEURISTIC = 6
SELECTION_TRANSPORT_RANDOM = 7

NUMERIC_FIELDS = (
    "step_budget",
    "used_steps",
    "final_cells",
    "discovery_auc",
    "trunk_end_step",
    "trunk_cells",
    "post_trunk_new_cells",
    "post_trunk_auc",
    "walltime_seconds",
    "gp_selection_count",
    "random_selection_count",
    "random_selector_count",
    "cold_start_count",
    "epsilon_random_count",
    "gp_ucb_count",
    "gp_posterior_count",
    "transport_heuristic_count",
    "transport_random_count",
    "direct_parent_count",
    "transported_direction_count",
    "mean_transport_alignment",
    "maximum_consecutive_perturbations",
    "selection_count",
)

CSV_FIELDS = (
    "result_name",
    "row_type",
    "seed",
    "n_seeds",
    *NUMERIC_FIELDS,
)


def read_json(path: Path) -> dict:
    """Read one JSON object without changing any result file."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_discovery_curve(path: Path) -> list[tuple[int, int]]:
    """Read ``(mechanics_step, cumulative_cells)`` pairs."""

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        step, count = stripped.split()[:2]
        rows.append((int(float(step)), int(float(count))))
    if not rows:
        return [(0, 0)]
    return sorted(rows)


def read_rollout_rows(path: Path) -> list[dict[str, float]]:
    """Read the numeric ``rollout_diagnostics.dat`` CSV."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]


def cell_count_at(curve: list[tuple[int, int]], step: int) -> int:
    """Return the right-continuous cumulative cell count at ``step``."""

    count = 0
    for event_step, event_count in curve:
        if event_step > step:
            break
        count = event_count
    return int(count)


def interval_auc(
    curve: list[tuple[int, int]],
    start_step: int,
    end_step: int,
    baseline_count: int = 0,
) -> float:
    """Average a cumulative step function over ``[start_step, end_step]``."""

    start = int(start_step)
    end = int(end_step)
    if end <= start:
        return 0.0

    current_step = start
    current_count = cell_count_at(curve, start)
    area = 0.0
    for event_step, event_count in curve:
        if event_step <= start:
            continue
        if event_step > end:
            break
        area += (event_step - current_step) * (
            current_count - baseline_count
        )
        current_step = event_step
        current_count = event_count
    area += (end - current_step) * (current_count - baseline_count)
    return float(area / (end - start))


def selection_counts(rollout_rows: list[dict[str, float]]) -> dict[str, float]:
    """Count actual perturbation selections, excluding the initial trunk."""

    codes = [int(round(row["selection_code"])) for row in rollout_rows]
    random_selector = sum(code == SELECTION_RANDOM for code in codes)
    epsilon_random = sum(code == SELECTION_EPSILON_RANDOM for code in codes)
    gp_ucb = sum(code == SELECTION_GP_UCB for code in codes)
    gp_posterior = sum(code == SELECTION_GP_POSTERIOR for code in codes)
    cold_start = sum(code == SELECTION_COLD_START for code in codes)
    transport_heuristic = sum(
        code == SELECTION_TRANSPORT_HEURISTIC for code in codes
    )
    transport_random = sum(code == SELECTION_TRANSPORT_RANDOM for code in codes)
    nontrunk_rows = [
        row
        for row, code in zip(rollout_rows, codes)
        if code != SELECTION_TRUNK
    ]
    direct_parent = sum(
        row.get("parent_is_direct_perturbation", 0.0) >= 0.5
        for row in nontrunk_rows
    )
    transported_direction = sum(
        row.get("candidate_is_transported", 0.0) >= 0.5
        for row in nontrunk_rows
    )
    alignments = [
        row.get("transport_alignment", 0.0)
        for row in nontrunk_rows
    ]
    return {
        "gp_selection_count": gp_ucb + gp_posterior,
        "random_selection_count": (
            random_selector + cold_start + epsilon_random + transport_random
        ),
        "random_selector_count": random_selector,
        "cold_start_count": cold_start,
        "epsilon_random_count": epsilon_random,
        "gp_ucb_count": gp_ucb,
        "gp_posterior_count": gp_posterior,
        "transport_heuristic_count": transport_heuristic,
        "transport_random_count": transport_random,
        "direct_parent_count": direct_parent,
        "transported_direction_count": transported_direction,
        "mean_transport_alignment": (
            sum(alignments) / len(alignments) if alignments else 0.0
        ),
        "selection_count": (
            random_selector
            + cold_start
            + epsilon_random
            + gp_ucb
            + gp_posterior
            + transport_heuristic
            + transport_random
        ),
    }


def summarize_seed(result_name: str, seed_directory: Path) -> dict:
    """Build all comparison metrics for one result/seed directory."""

    summary = read_json(seed_directory / "summary.json")
    curve = read_discovery_curve(seed_directory / "discovery_curve.dat")
    rollouts = read_rollout_rows(seed_directory / "rollout_diagnostics.dat")

    trunk_rows = [
        row
        for row in rollouts
        if int(round(row["selection_code"])) == SELECTION_TRUNK
    ]
    trunk_end_step = int(round(trunk_rows[0]["end_step"])) if trunk_rows else 0
    trunk_cells = cell_count_at(curve, trunk_end_step)
    budget = int(summary["step_budget"])
    final_cells = int(summary["elite_cells"])

    row = {
        "result_name": result_name,
        "row_type": "seed",
        "seed": int(summary["seed"]),
        "n_seeds": 1,
        "step_budget": budget,
        "used_steps": int(summary["used_steps"]),
        "final_cells": final_cells,
        "discovery_auc": float(summary["discovery_auc"]),
        "trunk_end_step": trunk_end_step,
        "trunk_cells": trunk_cells,
        "post_trunk_new_cells": final_cells - trunk_cells,
        "post_trunk_auc": interval_auc(
            curve,
            trunk_end_step,
            budget,
            baseline_count=trunk_cells,
        ),
        "walltime_seconds": float(summary["wall_time_seconds"]),
        "maximum_consecutive_perturbations": float(
            summary.get("maximum_consecutive_perturbation_count", 0)
        ),
    }
    row.update(selection_counts(rollouts))
    return row


def mean_row(result_name: str, seed_rows: list[dict]) -> dict:
    """Return the arithmetic mean of every numeric comparison metric."""

    if not seed_rows:
        raise ValueError(f"result '{result_name}' has no seed rows")
    row = {
        "result_name": result_name,
        "row_type": "mean",
        "seed": "mean",
        "n_seeds": len(seed_rows),
    }
    for field in NUMERIC_FIELDS:
        row[field] = sum(float(seed[field]) for seed in seed_rows) / len(seed_rows)
    return row


def collect_result(result_root: Path, result_name: str) -> list[dict]:
    """Collect sorted seed rows followed by their mean row."""

    result_directory = result_root / result_name
    if not result_directory.is_dir():
        raise FileNotFoundError(f"result directory not found: {result_directory}")
    seed_directories = sorted(
        directory
        for directory in result_directory.iterdir()
        if directory.is_dir() and (directory / "summary.json").is_file()
    )
    seed_rows = [
        summarize_seed(result_name, directory)
        for directory in seed_directories
    ]
    return [*seed_rows, mean_row(result_name, seed_rows)]


def collect_comparison(result_root: Path, result_names: list[str]) -> list[dict]:
    """Collect comparison rows for all requested result names."""

    rows = []
    for result_name in result_names:
        rows.extend(collect_result(result_root, result_name))
    return rows


def csv_text(rows: list[dict]) -> str:
    """Serialize comparison rows in one stable column order."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parse_arguments(argv=None):
    base_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare main5 result directories by seed and mean.",
    )
    parser.add_argument(
        "result_names",
        nargs="+",
        help="directory names below the result root",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=base_directory / "result",
        help="directory containing result-name directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_directory / "result" / "main5_comparison.csv",
        help="output CSV path",
    )
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    rows = collect_comparison(arguments.result_root, arguments.result_names)
    text = csv_text(rows)

    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    sys.stdout.write(text)
    print(f"CSV saved to: {output_path}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
