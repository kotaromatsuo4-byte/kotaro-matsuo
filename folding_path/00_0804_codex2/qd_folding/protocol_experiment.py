"""Two-stage experiment orchestration with held-out selector seeds."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .mechanics import generate_bootstrap_paths, load_model
from .output import save_method_result, save_trajectory, write_csv, write_json
from .protocol_search import (
    ROOT_POOL_KINDS,
    method_name,
    run_pool_search,
    run_root_search,
)


STAGE1_POOLS = ("frontier", "event", "history")
# Root conditions are diagnostic controls.  They are shown beside Stage 1 but
# are not eligible to become the Stage-2 winner because a one-state root set
# makes a selector comparison meaningless.
STAGE1_ROOT_BASELINES = ("root", "root_bootstrap")
STAGE2_SELECTORS = ("random", "ucb", "smucb")
TIE_ORDER = {name: index for index, name in enumerate(STAGE1_POOLS)}


def stable_seed(*parts: object) -> int:
    """Order-independent named RNG seed used throughout the protocol."""

    payload = "|".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _config_for_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    values = config.to_dict()
    values["seed"] = int(seed)
    return ExperimentConfig.from_dict(values)


def _step_value(
    search_log: list[dict],
    step: int,
    field: str,
    initial: float,
) -> float:
    value = float(initial)
    for row in search_log:
        if int(row["search_steps"]) > step:
            break
        value = float(row[field])
    return value


def _aggregate_rows(rows: list[dict], condition_field: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[condition_field])].append(row)
    metrics = (
        "new_outcome_cells",
        "outcome_cells",
        "shape_count",
        "layer_count",
        "bootstrap_outcome_cells",
        "new_cell_auc",
        "coverage_auc",
        "median_elite_steps",
        "search_steps",
        "rollouts",
        "wall_seconds",
        "mean_candidate_size",
    )
    output: list[dict] = []
    for condition, values in grouped.items():
        aggregate = {
            condition_field: condition,
            "replicates": len(values),
        }
        for metric in metrics:
            samples = np.asarray([float(row[metric]) for row in values], dtype=float)
            finite = samples[np.isfinite(samples)]
            if finite.size == 0:
                aggregate[f"mean_{metric}"] = float("nan")
                aggregate[f"median_{metric}"] = float("nan")
                aggregate[f"std_{metric}"] = float("nan")
            else:
                aggregate[f"mean_{metric}"] = float(np.mean(finite))
                aggregate[f"median_{metric}"] = float(np.median(finite))
                aggregate[f"std_{metric}"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                )
        output.append(aggregate)
    return output


def _save_stage_curves(
    runs: dict[tuple[int, str], dict],
    conditions: Iterable[str],
    budget: int,
    output: Path,
) -> None:
    grid = np.unique(
        np.linspace(0, budget, num=min(budget + 1, 101), dtype=int)
    )
    rows: list[dict] = []
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for condition in conditions:
        matching = [
            result
            for (_, name), result in runs.items()
            if name == condition
        ]
        if not matching:
            continue
        matrix = np.asarray(
            [
                [
                    _step_value(
                        result["search_log"],
                        int(step),
                        "outcome_cells",
                        result["summary"]["bootstrap_outcome_cells"],
                    )
                    for step in grid
                ]
                for result in matching
            ],
            dtype=float,
        )
        mean = np.mean(matrix, axis=0)
        lower = np.min(matrix, axis=0)
        upper = np.max(matrix, axis=0)
        for index, step in enumerate(grid):
            rows.append(
                {
                    "condition": condition,
                    "search_steps": int(step),
                    "mean_outcome_cells": float(mean[index]),
                    "min_outcome_cells": float(lower[index]),
                    "max_outcome_cells": float(upper[index]),
                    "replicates": len(matching),
                }
            )
        axis.step(grid, mean, where="post", label=condition)
        if len(matching) > 1:
            axis.fill_between(grid, lower, upper, step="post", alpha=0.13)
    axis.set_xlabel("Newly simulated mechanics steps")
    axis.set_ylabel("Discovered terminal cells")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "discovery_curves.png", dpi=180)
    plt.close(figure)
    write_csv(rows, output / "discovery_curves.csv")


def _run_stage(
    *,
    stage_name: str,
    seeds: list[int],
    conditions: list[tuple[str, str]],
    config: ExperimentConfig,
    project_root: Path,
    output: Path,
    save_outputs: bool,
) -> tuple[dict[tuple[int, str], dict], list[dict]]:
    model = load_model(project_root / "data")
    runs: dict[tuple[int, str], dict] = {}
    summary_rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        run_config = _config_for_seed(config, seed)
        bootstrap_rng = np.random.default_rng(
            stable_seed(seed, stage_name, "bootstrap")
        )
        bootstrap_paths = generate_bootstrap_paths(
            model,
            run_config,
            bootstrap_rng,
        )
        seed_folder = output / "runs" / f"seed_{seed:06d}"
        if save_outputs:
            bootstrap_folder = seed_folder / "bootstrap"
            for index, trajectory in enumerate(bootstrap_paths):
                save_trajectory(
                    trajectory,
                    bootstrap_folder / f"path_{index:03d}",
                    model,
                )

        for condition_index, (pool_kind, selector_kind) in enumerate(conditions):
            name = method_name(pool_kind, selector_kind)
            print(
                f"[{stage_name} seed {seed_index + 1}/{len(seeds)} "
                f"condition {condition_index + 1}/{len(conditions)}] {name}"
            )
            # Each condition receives a fresh generator with the same named
            # seed.  Candidate-set/selector internals therefore cannot shift
            # the random perturbation coefficient stream of another method.
            emitter_rng = np.random.default_rng(
                stable_seed(seed, stage_name, "emitter")
            )
            if pool_kind in ROOT_POOL_KINDS:
                result = run_root_search(
                    pool_kind=pool_kind,
                    model=model,
                    config=run_config,
                    bootstrap_paths=bootstrap_paths,
                    emitter_rng=emitter_rng,
                )
            else:
                result = run_pool_search(
                    pool_kind=pool_kind,
                    selector_kind=selector_kind,
                    model=model,
                    config=run_config,
                    bootstrap_paths=bootstrap_paths,
                    priority_seed=stable_seed(seed, stage_name, "priority"),
                    selector_rng=np.random.default_rng(
                        stable_seed(seed, stage_name, "selector")
                    ),
                    emitter_rng=emitter_rng,
                )
            runs[(seed, name)] = result
            row = dict(result["summary"])
            row.update(
                {
                    "stage": stage_name,
                    "replicate_seed": seed,
                    "condition": name,
                }
            )
            summary_rows.append(row)
            print(
                "  "
                f"new_cells={row['new_outcome_cells']} "
                f"total_cells={row['outcome_cells']} "
                f"steps={row['search_steps']} "
                f"rollouts={row['rollouts']}"
            )
            if save_outputs:
                save_method_result(
                    result,
                    model,
                    run_config,
                    seed_folder / name,
                )

    return runs, summary_rows


def _select_stage1_winner(aggregate_rows: list[dict]) -> dict:
    """Rank only frontier/event/history by the pre-registered winner rule."""

    eligible_rows = [
        row for row in aggregate_rows if str(row["pool_kind"]) in STAGE1_POOLS
    ]
    if not eligible_rows:
        raise ValueError("stage 1 produced no aggregate rows")
    ranked = sorted(
        eligible_rows,
        key=lambda row: (
            -float(row["median_new_outcome_cells"]),
            -float(row["mean_new_cell_auc"]),
            -float(row["mean_new_outcome_cells"]),
            TIE_ORDER[str(row["pool_kind"])],
        ),
    )
    winner = dict(ranked[0])
    return {
        "pool_kind": winner["pool_kind"],
        "criterion": [
            "maximum median new outcome cells",
            "maximum mean new-cell AUC",
            "maximum mean new outcome cells",
            "fixed tie order: frontier, event, history",
        ],
        "winning_aggregate": winner,
        "complete_ranking": ranked,
    }


def run_two_stage_protocol(
    *,
    config: ExperimentConfig,
    stage1_seeds: list[int],
    stage2_seeds: list[int],
    project_root: Path,
    output: Path,
    stages: Iterable[int] = (1, 2),
    winner_pool: str | None = None,
    save_outputs: bool = True,
    clean_output: bool = False,
) -> dict:
    """Run root controls and set screening, then held-out selector comparison."""

    config.validate()
    selected_stages = tuple(dict.fromkeys(int(value) for value in stages))
    if not selected_stages or any(value not in {1, 2} for value in selected_stages):
        raise ValueError("stages must contain 1 and/or 2")
    if 1 in selected_stages and not stage1_seeds:
        raise ValueError("stage 1 needs at least one seed")
    if 2 in selected_stages and not stage2_seeds:
        raise ValueError("stage 2 needs at least one seed")
    output = Path(output)
    if clean_output and output.exists():
        shutil.rmtree(output)
    if save_outputs:
        output.mkdir(parents=True, exist_ok=True)
        write_json(config.to_dict(), output / "effective_config.json")
        write_json(
            {
                "stage1_seeds": stage1_seeds,
                "stage2_seeds": stage2_seeds,
                "stage1_conditions": [
                    method_name(pool, "random")
                    for pool in STAGE1_ROOT_BASELINES + STAGE1_POOLS
                ],
                "stage1_winner_eligible_pools": list(STAGE1_POOLS),
                "root_baseline_note": (
                    "root uses perturb_max_norm; root_bootstrap uses "
                    "initial_imperfection and is a radius sensitivity check"
                ),
                "stage2_selectors": list(STAGE2_SELECTORS),
                "reward": "1[new terminal cell]",
                "budget_unit": "newly simulated mechanics steps",
                "pool_refresh": "fixed within candidate_epoch_rollouts",
            },
            output / "protocol.json",
        )

    returned: dict[str, object] = {}
    winner: dict | None = None
    if 1 in selected_stages:
        stage1_output = output / "stage1_candidate_sets"
        stage1_conditions = [
            (pool, "random")
            for pool in STAGE1_ROOT_BASELINES + STAGE1_POOLS
        ]
        stage1_runs, stage1_rows = _run_stage(
            stage_name="stage1",
            seeds=[int(value) for value in stage1_seeds],
            conditions=stage1_conditions,
            config=config,
            project_root=project_root,
            output=stage1_output,
            save_outputs=save_outputs,
        )
        stage1_aggregate = _aggregate_rows(stage1_rows, "pool_kind")
        winner = _select_stage1_winner(stage1_aggregate)
        returned["stage1"] = {
            "runs": stage1_runs,
            "per_run": stage1_rows,
            "aggregate": stage1_aggregate,
            "winner": winner,
        }
        if save_outputs:
            write_csv(stage1_rows, stage1_output / "per_run_summary.csv")
            write_csv(stage1_aggregate, stage1_output / "aggregate_summary.csv")
            write_json(winner, stage1_output / "winner.json")
            _save_stage_curves(
                stage1_runs,
                [
                    method_name(pool, "random")
                    for pool in STAGE1_ROOT_BASELINES + STAGE1_POOLS
                ],
                config.search_step_budget,
                stage1_output,
            )

    if 2 in selected_stages:
        if winner_pool is None and winner is not None:
            winner_pool = str(winner["pool_kind"])
        if winner_pool is None:
            winner_path = output / "stage1_candidate_sets" / "winner.json"
            if winner_path.exists():
                winner_pool = str(
                    json.loads(winner_path.read_text(encoding="utf-8"))["pool_kind"]
                )
        if winner_pool not in STAGE1_POOLS:
            raise ValueError(
                "stage 2 requires --winner-pool or an existing stage 1 winner"
            )
        stage2_output = output / "stage2_selectors"
        stage2_conditions = [
            (winner_pool, selector) for selector in STAGE2_SELECTORS
        ]
        stage2_runs, stage2_rows = _run_stage(
            stage_name="stage2",
            seeds=[int(value) for value in stage2_seeds],
            conditions=stage2_conditions,
            config=config,
            project_root=project_root,
            output=stage2_output,
            save_outputs=save_outputs,
        )
        stage2_aggregate = _aggregate_rows(stage2_rows, "selector_kind")
        returned["stage2"] = {
            "winner_pool": winner_pool,
            "runs": stage2_runs,
            "per_run": stage2_rows,
            "aggregate": stage2_aggregate,
        }
        if save_outputs:
            write_csv(stage2_rows, stage2_output / "per_run_summary.csv")
            write_csv(stage2_aggregate, stage2_output / "aggregate_summary.csv")
            write_json(
                {
                    "winner_pool": winner_pool,
                    "stage1_and_stage2_seeds_are_disjoint": not bool(
                        set(stage1_seeds) & set(stage2_seeds)
                    ),
                },
                stage2_output / "design.json",
            )
            _save_stage_curves(
                stage2_runs,
                [method_name(winner_pool, value) for value in STAGE2_SELECTORS],
                config.search_step_budget,
                stage2_output,
            )
    return returned


__all__ = [
    "STAGE1_POOLS",
    "STAGE1_ROOT_BASELINES",
    "STAGE2_SELECTORS",
    "run_two_stage_protocol",
    "stable_seed",
]
