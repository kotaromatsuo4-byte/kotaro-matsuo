"""Experiment orchestration with one common bootstrap for all methods."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np

from .mechanics import generate_bootstrap_paths, load_model
from .output import (
    save_comparison,
    save_method_result,
    save_trajectory,
    write_json,
)
from .sequence_search import SEQUENCE_METHOD, run_sequence_map_elites
from .state_search import (
    BOUNDED_METHOD,
    EVENT_METHOD,
    run_bounded_archive_search,
    run_event_frontier_search,
)


METHOD_ALIASES = {
    "1": EVENT_METHOD,
    "event": EVENT_METHOD,
    EVENT_METHOD: EVENT_METHOD,
    "2": BOUNDED_METHOD,
    "bounded": BOUNDED_METHOD,
    BOUNDED_METHOD: BOUNDED_METHOD,
    "3": SEQUENCE_METHOD,
    "sequence": SEQUENCE_METHOD,
    SEQUENCE_METHOD: SEQUENCE_METHOD,
}


def normalize_methods(methods: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in methods:
        key = value.strip().lower()
        if key == "all":
            requested = [EVENT_METHOD, BOUNDED_METHOD, SEQUENCE_METHOD]
        elif key in METHOD_ALIASES:
            requested = [METHOD_ALIASES[key]]
        else:
            raise ValueError(f"Unknown method: {value}")
        for method in requested:
            if method not in result:
                result.append(method)
    return result


def run_experiment(
    *,
    config,
    methods: list[str],
    project_root: Path,
    output: Path,
    save_outputs: bool = True,
    clean_output: bool = False,
) -> dict[str, dict]:
    config.validate()
    selected = normalize_methods(methods)
    if clean_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    model = load_model(project_root / "data")
    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = seed_sequence.spawn(1 + len(selected))
    bootstrap_rng = np.random.default_rng(child_seeds[0])
    bootstrap_paths = generate_bootstrap_paths(model, config, bootstrap_rng)

    if save_outputs:
        write_json(config.to_dict(), output / "config.json")
        bootstrap_folder = output / "bootstrap"
        for index, trajectory in enumerate(bootstrap_paths):
            save_trajectory(
                trajectory,
                bootstrap_folder / f"path_{index:03d}",
                model,
            )

    runners = {
        EVENT_METHOD: run_event_frontier_search,
        BOUNDED_METHOD: run_bounded_archive_search,
        SEQUENCE_METHOD: run_sequence_map_elites,
    }
    results: dict[str, dict] = {}
    for index, method in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] {method}")
        rng = np.random.default_rng(child_seeds[index + 1])
        result = runners[method](
            model,
            copy.deepcopy(config),
            bootstrap_paths,
            rng,
        )
        results[method] = result
        summary = result["summary"]
        print(
            "  "
            f"cells={summary['outcome_cells']} "
            f"shapes={summary['shape_count']} "
            f"layers={summary['layer_count']} "
            f"steps={summary['search_steps']} "
            f"rollouts={summary['rollouts']}"
        )
        if save_outputs:
            save_method_result(
                result,
                model,
                config,
                output / method,
            )

    if save_outputs:
        save_comparison(results, output)
    return results
