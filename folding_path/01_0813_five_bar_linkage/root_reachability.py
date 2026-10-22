"""Diagnose terminal cells reachable by one perturbation at the root.

Each rollout is deliberately restricted to

    root -- one random tangent perturbation -- deterministic rollout.

The driver calls the current ``roll.random_roll``,
``roll.deterministic_roll``, and ``archive.update_elite_archive`` functions.
Consequently it uses the same corrector, planar ``collision2`` check, terminal
descriptor, layer ordering, and archive tolerances as the current programs.

Results are written only below ``result/root_reachability`` by default, so no
main1/main5 result directory is replaced.  The computational functions do not
write files; all output is confined to ``save_run`` and the guarded ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from archive import update_elite_archive
from mechanics import compute_gravity_potential
from result import reset_output_directory, save_discovery_curve, save_elite_paths
from roll import deterministic_roll, random_roll
from state import create_history


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_number_list(text, conversion=float):
    """Parse comma- or space-separated CLI/environment values."""
    if isinstance(text, (list, tuple)):
        parts = []
        for value in text:
            parts.extend(str(value).split(","))
    else:
        parts = str(text).replace(" ", ",").split(",")
    values = [conversion(part) for part in parts if str(part).strip()]
    if not values:
        raise ValueError("at least one value is required")
    return values


def load_model(model_directory):
    """Load the same node/member/fix representation used by main5."""
    directory = Path(model_directory)
    node = np.atleast_2d(
        np.loadtxt(directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(directory / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(directory / "fix.dat", delimiter=",", dtype=int)
    )
    fixed_indices = 3 * fix[:, 0] + fix[:, 1]
    index_free = np.delete(np.arange(3 * len(node)), fixed_indices)
    member_length = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    return {
        "node_initial": node,
        "member": member,
        "index_free": index_free,
        "member_length": member_length,
        "num_node": int(len(node)),
        "num_member": int(len(member)),
        "length_scale": float(np.mean(member_length)),
    }


def default_configuration(seed, perturbation_scale, step_budget):
    """Return parameters matching the current main5 physical calculation."""
    return {
        "seed": int(seed),
        "perturbation_scale": float(perturbation_scale),
        "total_step_budget": int(step_budget),
        "max_rollout_steps": 100,
        "reference_point": 0.0,
        "constraint_tolerance": 1.0e-10,
        "beta0": 1.0,
        "corrector_iterations": 10,
        "line_search_iterations": 10,
        "line_search_tau": 0.5,
        "cma_population": 6,
        "cma_generations": 10,
        "cma_seed": 0,
        "target_norm_max": 0.10,
        "flatness_tolerance": 0.05,
        "dist_tol": 0.05,
        "normal_tol": 0.10,
        "cos_tol": 0.99,
        "collision_model": "planar",
        "layer_threshold": 0.10,
        "overlap_tolerance": 0.05,
        "separation_tolerance": 1.0e-5,
        "collinearity_tolerance": 0.02,
        "parallel_tolerance": 0.02,
        "rms_tolerance": 0.02,
        "max_tolerance": 0.05,
    }


def validate_configuration(config):
    if config["total_step_budget"] <= 0:
        raise ValueError("total_step_budget must be positive")
    if config["max_rollout_steps"] <= 0:
        raise ValueError("max_rollout_steps must be positive")
    if config["perturbation_scale"] <= 0.0:
        raise ValueError("perturbation_scale must be positive")
    if config["collision_model"] != "planar":
        raise ValueError("this diagnostic must use the planar collision model")


def deterministic_step_from_state(history, parent_id, model, config):
    """Execute exactly the current deterministic_roll implementation."""
    return deterministic_roll(
        history,
        parent_id,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        config["target_norm_max"],
        config["constraint_tolerance"],
        config["beta0"],
        config["corrector_iterations"],
        config["line_search_iterations"],
        config["line_search_tau"],
        config["cma_population"],
        config["cma_generations"],
        config["cma_seed"],
        config["reference_point"],
        config["flatness_tolerance"],
        config["dist_tol"],
        config["normal_tol"],
        config["cos_tol"],
        collision_model=config["collision_model"],
    )


def root_perturbation_step(history, model, config):
    """Create one child of root using the current random_roll implementation."""
    return random_roll(
        history,
        0,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        config["perturbation_scale"],
        config["constraint_tolerance"],
        config["beta0"],
        config["corrector_iterations"],
        config["line_search_iterations"],
        config["line_search_tau"],
        config["reference_point"],
        config["flatness_tolerance"],
        config["dist_tol"],
        config["normal_tol"],
        config["cos_tol"],
        collision_model=config["collision_model"],
    )


def update_archive(elite_archive, history, terminal_state_id, model, config):
    """Use the current terminal descriptor and elite archive implementation."""
    return update_elite_archive(
        elite_archive,
        history,
        terminal_state_id,
        model["member"],
        model["length_scale"],
        config["layer_threshold"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
        config["rms_tolerance"],
        config["max_tolerance"],
        config["reference_point"],
    )


def execute_root_rollout(history, remaining_budget, model, config):
    """Run one root perturbation followed only by deterministic transitions."""
    if remaining_budget <= 0:
        return None, [], 0, "budget"

    current = root_perturbation_step(history, model, config)
    state_ids = [int(current.state_id)]
    used_steps = 1
    while (
        current.terminal_reason is None
        and len(state_ids) < config["max_rollout_steps"]
        and used_steps < remaining_budget
    ):
        current = deterministic_step_from_state(
            history, current.state_id, model, config
        )
        state_ids.append(int(current.state_id))
        used_steps += 1

    if current.terminal_reason is not None:
        stop_reason = current.terminal_reason
    elif len(state_ids) >= config["max_rollout_steps"]:
        stop_reason = "timeout"
    else:
        stop_reason = "budget"
    return current, state_ids, used_steps, stop_reason


def run_root_reachability(model, config, progress_interval=0):
    """Return an in-memory root-only search result without writing files."""
    validate_configuration(config)
    np.random.seed(config["seed"])
    initial_potential = compute_gravity_potential(
        model["node_initial"], config["reference_point"]
    )
    history = create_history(model["node_initial"], initial_potential)
    elite_archive = []
    discovery_curve = [(0, 0)]
    rollout_rows = []
    used_steps = 0
    rollout_id = 0
    start_time = time.perf_counter()

    while used_steps < config["total_step_budget"]:
        start_step = used_steps
        current, state_ids, spent, stop_reason = execute_root_rollout(
            history,
            config["total_step_budget"] - used_steps,
            model,
            config,
        )
        if current is None or spent == 0:
            break
        used_steps += spent

        is_new_cell = False
        new_cell_index = -1
        if stop_reason == "flat":
            is_new_cell = update_archive(
                elite_archive,
                history,
                current.state_id,
                model,
                config,
            )
            if is_new_cell:
                new_cell_index = len(elite_archive) - 1
                discovery_curve.append((used_steps, len(elite_archive)))

        rollout_rows.append(
            {
                "rollout_id": int(rollout_id),
                "start_step": int(start_step),
                "end_step": int(used_steps),
                "spent_steps": int(spent),
                "terminal_state_id": int(current.state_id),
                "stop_reason": stop_reason,
                "new_cell": int(is_new_cell),
                "new_cell_index": int(new_cell_index),
                "cell_count": int(len(elite_archive)),
            }
        )
        rollout_id += 1
        if progress_interval and rollout_id % int(progress_interval) == 0:
            print(
                f"seed={config['seed']} scale={config['perturbation_scale']:.3f} "
                f"rollouts={rollout_id} steps={used_steps} "
                f"cells={len(elite_archive)}"
            )

    if discovery_curve[-1][0] != used_steps:
        discovery_curve.append((used_steps, len(elite_archive)))

    stop_counts = {}
    for row in rollout_rows:
        reason = row["stop_reason"]
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    rollout_count = len(rollout_rows)
    discovery_steps = []
    previous_cell_count = 0
    for step, cell_count in discovery_curve:
        if int(cell_count) > previous_cell_count:
            discovery_steps.append(int(step))
        previous_cell_count = max(previous_cell_count, int(cell_count))
    summary = {
        "seed": int(config["seed"]),
        "perturbation_scale": float(config["perturbation_scale"]),
        "step_budget": int(config["total_step_budget"]),
        "used_steps": int(used_steps),
        "rollout_count": int(rollout_count),
        "elite_cells": int(len(elite_archive)),
        "flat_rollouts": int(stop_counts.get("flat", 0)),
        "collision_rollouts": int(stop_counts.get("collision", 0)),
        "corrector_failures": int(stop_counts.get("corrector_failure", 0)),
        "timeout_rollouts": int(stop_counts.get("timeout", 0)),
        "budget_censored_rollouts": int(stop_counts.get("budget", 0)),
        "flat_rate": (
            float(stop_counts.get("flat", 0) / rollout_count)
            if rollout_count
            else 0.0
        ),
        "collision_rate": (
            float(stop_counts.get("collision", 0) / rollout_count)
            if rollout_count
            else 0.0
        ),
        "discovery_steps": discovery_steps,
        "wall_time_seconds": float(time.perf_counter() - start_time),
    }
    return {
        "history": history,
        "elite_archive": elite_archive,
        "discovery_curve": discovery_curve,
        "rollout_rows": rollout_rows,
        "summary": summary,
    }


def serializable_cell_rows(result):
    """Return shape and layer descriptors for every final archive cell."""
    history = result["history"]
    discovery_steps = result["summary"]["discovery_steps"]
    rows = []
    for cell_index, cell in enumerate(result["elite_archive"]):
        terminal_state_id = int(cell["terminal_state_id"])
        rows.append(
            {
                "cell_index": int(cell_index),
                "discovery_step": int(discovery_steps[cell_index]),
                "terminal_state_id": terminal_state_id,
                "path_depth": int(history[terminal_state_id].depth),
                "shape_descriptor": np.asarray(
                    cell["shape_descriptor"], dtype=float
                ).tolist(),
                "layer_key": [list(item) for item in cell["layer_key"]],
                "continuous_descriptor": np.asarray(
                    cell["continuous_descriptor"], dtype=float
                ).tolist(),
            }
        )
    return rows


def scale_directory_name(scale):
    """Create a filesystem-safe, stable scale label."""
    return f"scale_{float(scale):.3f}".replace(".", "p")


def write_dictionary_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_run(result, config, output_directory):
    """Save one diagnostic run without touching any main1/main5 directory."""
    directory = Path(output_directory)
    reset_output_directory(directory)
    save_elite_paths(result["history"], result["elite_archive"], directory)
    save_discovery_curve(result["discovery_curve"], directory)
    write_dictionary_csv(directory / "rollouts.csv", result["rollout_rows"])
    (directory / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    (directory / "cells.json").write_text(
        json.dumps(serializable_cell_rows(result), indent=2), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def save_aggregate(rows, output_directory):
    """Save summaries for the seeds/scales evaluated by one invocation."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_dictionary_csv(directory / "aggregate.csv", rows)
    (directory / "aggregate.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=[os.environ.get("FOLDING_ROOT_SCALES", "0.30")],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[os.environ.get("FOLDING_ROOT_SEEDS", "0,1,2,3,4")],
    )
    parser.add_argument(
        "--step-budget",
        type=int,
        default=int(os.environ.get("FOLDING_STEP_BUDGET", "5000")),
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--target-norm-max",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=SCRIPT_DIRECTORY / "model",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "root_reachability",
    )
    parser.add_argument("--progress-interval", type=int, default=20)
    return parser


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    scales = parse_number_list(args.scales, float)
    seeds = parse_number_list(args.seeds, int)
    model = load_model(args.model_directory)
    aggregate_rows = []

    for scale in scales:
        for seed in seeds:
            config = default_configuration(seed, scale, args.step_budget)
            config["max_rollout_steps"] = int(args.max_rollout_steps)
            config["target_norm_max"] = float(args.target_norm_max)
            result = run_root_reachability(
                model, config, progress_interval=args.progress_interval
            )
            output_directory = (
                args.output_root
                / scale_directory_name(scale)
                / f"seed_{int(seed):03d}"
            )
            save_run(result, config, output_directory)
            aggregate_rows.append(result["summary"])
            print(json.dumps(result["summary"], ensure_ascii=False))

    save_aggregate(aggregate_rows, args.output_root)
    return aggregate_rows


if __name__ == "__main__":
    main()
