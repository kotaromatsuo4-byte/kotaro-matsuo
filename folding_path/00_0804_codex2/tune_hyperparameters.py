"""Small, reproducible pilot sweep for search-specific hyperparameters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np

from qd_folding.config import ExperimentConfig
from qd_folding.mechanics import generate_bootstrap_paths, load_model
from qd_folding.output import runtime_metadata, write_csv, write_json
from qd_folding.sequence_search import SEQUENCE_METHOD, run_sequence_map_elites
from qd_folding.state_search import (
    BOUNDED_METHOD,
    EVENT_METHOD,
    run_bounded_archive_search,
    run_event_frontier_search,
)


PILOT_CANDIDATES: dict[str, list[tuple[str, dict]]] = {
    EVENT_METHOD: [
        (
            "event_c050",
            {
                "ucb_exploration": 0.50,
                "event_novelty_tolerance": 0.045,
                "event_merge_tolerance": 0.018,
            },
        ),
        (
            "event_c100",
            {
                "ucb_exploration": 1.00,
                "event_novelty_tolerance": 0.045,
                "event_merge_tolerance": 0.018,
            },
        ),
        (
            "event_c200",
            {
                "ucb_exploration": 2.00,
                "event_novelty_tolerance": 0.045,
                "event_merge_tolerance": 0.018,
            },
        ),
        (
            "event_eps030",
            {
                "ucb_exploration": 1.00,
                "event_novelty_tolerance": 0.030,
                "event_merge_tolerance": 0.012,
            },
        ),
        (
            "event_eps070",
            {
                "ucb_exploration": 1.00,
                "event_novelty_tolerance": 0.070,
                "event_merge_tolerance": 0.028,
            },
        ),
    ],
    BOUNDED_METHOD: [
        (
            "bounded_c050_w010",
            {"ucb_exploration": 0.50, "state_bin_width": 0.10},
        ),
        (
            "bounded_c100_w010",
            {"ucb_exploration": 1.00, "state_bin_width": 0.10},
        ),
        (
            "bounded_c200_w010",
            {"ucb_exploration": 2.00, "state_bin_width": 0.10},
        ),
        (
            "bounded_c100_w007",
            {"ucb_exploration": 1.00, "state_bin_width": 0.07},
        ),
        (
            "bounded_c100_w015",
            {"ucb_exploration": 1.00, "state_bin_width": 0.15},
        ),
    ],
    SEQUENCE_METHOD: [
        (
            "sequence_s005_i035",
            {
                "sequence_coefficient_sigma": 0.05,
                "sequence_insert_probability": 0.35,
                "sequence_max_wait": 8,
            },
        ),
        (
            "sequence_s008_i035",
            {
                "sequence_coefficient_sigma": 0.08,
                "sequence_insert_probability": 0.35,
                "sequence_max_wait": 8,
            },
        ),
        (
            "sequence_s012_i035",
            {
                "sequence_coefficient_sigma": 0.12,
                "sequence_insert_probability": 0.35,
                "sequence_max_wait": 8,
            },
        ),
        (
            "sequence_s008_i055",
            {
                "sequence_coefficient_sigma": 0.08,
                "sequence_insert_probability": 0.55,
                "sequence_max_wait": 8,
            },
        ),
        (
            "sequence_s008_wait4",
            {
                "sequence_coefficient_sigma": 0.08,
                "sequence_insert_probability": 0.35,
                "sequence_max_wait": 4,
            },
        ),
    ],
}


RUNNERS = {
    EVENT_METHOD: run_event_frontier_search,
    BOUNDED_METHOD: run_bounded_archive_search,
    SEQUENCE_METHOD: run_sequence_map_elites,
}


def _candidate_config(
    base: ExperimentConfig,
    *,
    seed: int,
    budget: int,
    overrides: dict,
) -> ExperimentConfig:
    values = base.to_dict()
    values.update(
        {
            "seed": seed,
            "search_step_budget": budget,
            "save_all_histories": False,
        }
    )
    values.update(overrides)
    return ExperimentConfig.from_dict(values)


def run_pilot(
    *,
    project_root: Path,
    output: Path,
    seeds: list[int],
    budget: int,
) -> tuple[list[dict], list[dict], dict]:
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(project_root / "data")
    base = ExperimentConfig()
    trials: list[dict] = []

    for seed in seeds:
        bootstrap_config = _candidate_config(
            base, seed=seed, budget=budget, overrides={}
        )
        bootstrap_seed = np.random.SeedSequence(seed).spawn(1)[0]
        bootstrap = generate_bootstrap_paths(
            model,
            bootstrap_config,
            np.random.default_rng(bootstrap_seed),
        )
        for method_index, (method, candidates) in enumerate(
            PILOT_CANDIDATES.items()
        ):
            for candidate_id, overrides in candidates:
                config = _candidate_config(
                    base,
                    seed=seed,
                    budget=budget,
                    overrides=overrides,
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, method_index, 7919])
                )
                started = time.perf_counter()
                result = RUNNERS[method](
                    model, config, copy.deepcopy(bootstrap), rng
                )
                summary = result["summary"]
                trials.append(
                    {
                        "method": method,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "budget": budget,
                        "parameters": overrides,
                        "effective_config": config.to_dict(),
                        "outcome_cells": summary["outcome_cells"],
                        "new_cells": (
                            summary["outcome_cells"]
                            - summary["bootstrap_outcome_cells"]
                        ),
                        "coverage_auc": summary["coverage_auc"],
                        "shape_count": summary["shape_count"],
                        "layer_count": summary["layer_count"],
                        "median_elite_steps": summary[
                            "median_elite_steps"
                        ],
                        "rollouts": summary["rollouts"],
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"{method:29s} {candidate_id:22s} seed={seed:3d} "
                    f"cells={summary['outcome_cells']:2d} "
                    f"auc={summary['coverage_auc']:.3f}"
                )

    summaries: list[dict] = []
    selected: dict[str, dict] = {}
    for method, candidates in PILOT_CANDIDATES.items():
        method_rows: list[dict] = []
        for candidate_id, overrides in candidates:
            rows = [
                row
                for row in trials
                if row["method"] == method
                and row["candidate_id"] == candidate_id
            ]
            summary = {
                "method": method,
                "candidate_id": candidate_id,
                "parameters": overrides,
                "median_outcome_cells": statistics.median(
                    row["outcome_cells"] for row in rows
                ),
                "median_new_cells": statistics.median(
                    row["new_cells"] for row in rows
                ),
                "mean_coverage_auc": statistics.fmean(
                    row["coverage_auc"] for row in rows
                ),
                "median_elite_steps": statistics.median(
                    row["median_elite_steps"] for row in rows
                ),
                "mean_wall_seconds": statistics.fmean(
                    row["wall_seconds"] for row in rows
                ),
            }
            summaries.append(summary)
            method_rows.append(summary)
        best = max(
            method_rows,
            key=lambda row: (
                row["median_outcome_cells"],
                row["mean_coverage_auc"],
                -row["median_elite_steps"],
            ),
        )
        selected[method] = {
            "candidate_id": best["candidate_id"],
            "parameters": best["parameters"],
            "selection_rule": (
                "lexicographic maximum of median terminal-cell count, "
                "mean discovery-curve AUC, and negative median elite length"
            ),
        }

    write_csv(trials, output / "tuning_trials.csv")
    write_csv(summaries, output / "tuning_summary.csv")
    write_json(
        {
            "seeds": seeds,
            "search_step_budget_per_trial": budget,
            "base_config": base.to_dict(),
            "candidate_definitions": {
                method: [
                    {
                        "candidate_id": candidate_id,
                        "overrides": overrides,
                    }
                    for candidate_id, overrides in candidates
                ]
                for method, candidates in PILOT_CANDIDATES.items()
            },
            "runtime": runtime_metadata(),
        },
        output / "pilot_metadata.json",
    )
    write_json(
        {
            "seeds": seeds,
            "search_step_budget_per_trial": budget,
            "selected": selected,
        },
        output / "selected_hyperparameters.json",
    )
    return trials, summaries, selected


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument(
        "--output", type=Path, default=Path("output/tuning")
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parent
    output = arguments.output
    if not output.is_absolute():
        output = project_root / output
    _, _, selected = run_pilot(
        project_root=project_root,
        output=output,
        seeds=arguments.seeds,
        budget=arguments.budget,
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print(f"Tuning results: {output.resolve()}")


if __name__ == "__main__":
    main()
