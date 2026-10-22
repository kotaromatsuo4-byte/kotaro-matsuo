"""Shared accounting for all search methods."""

from __future__ import annotations

import copy
import time

import numpy as np

from .archives import ArchiveUpdate, HistoryStore, OutcomeArchive
from .mechanics import Model


def clone_trajectory(trajectory: dict) -> dict:
    """Deep-copy one trajectory so method-specific metadata cannot leak."""

    return copy.deepcopy(trajectory)


def initialize_from_bootstrap(
    bootstrap_paths: list[dict],
    model: Model,
    config,
    method: str,
) -> tuple[HistoryStore, OutcomeArchive, list[dict]]:
    history = HistoryStore()
    archive = OutcomeArchive()
    copied_paths = [clone_trajectory(value) for value in bootstrap_paths]
    for index, trajectory in enumerate(copied_paths):
        history.append(trajectory)
        archive.update(
            trajectory,
            model,
            config,
            evaluation_index=-(len(copied_paths) - index),
            method=method,
            candidate_metadata={"bootstrap_index": index},
        )
    return history, archive, copied_paths


def archive_reward(
    update: ArchiveUpdate,
    exploration_gain: int,
    compute_steps: int,
    config,
) -> float:
    gain = 0.0
    if update.is_new:
        gain += config.reward_new_cell
    if update.is_improvement:
        gain += (
            config.reward_elite_improvement * update.relative_improvement
        )
    gain += config.reward_exploration_gain * min(exploration_gain, 1)
    return (
        config.ucb_reward_step_scale
        * gain
        / max(int(compute_steps), 1)
    )


def curve_auc(search_log: list[dict], budget: int, initial_cells: int) -> float:
    """Normalized area under the stepwise discovery curve."""

    if budget <= 0:
        return 0.0
    previous_step = 0
    cells = initial_cells
    area = 0.0
    for row in search_log:
        step = min(int(row["search_steps"]), budget)
        area += (step - previous_step) * cells
        previous_step = step
        cells = int(row["outcome_cells"])
        if previous_step >= budget:
            break
    area += max(budget - previous_step, 0) * cells
    return area / budget


def make_summary(
    *,
    method: str,
    history: HistoryStore,
    archive: OutcomeArchive,
    search_log: list[dict],
    bootstrap_outcome_cells: int,
    bootstrap_compute_steps: int,
    search_steps: int,
    rollouts: int,
    started_at: float,
    active_units: int,
    config,
) -> dict:
    terminal_counts: dict[str, int] = {}
    for trajectory in history.trajectories:
        reason = trajectory["terminal_reason"]
        terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
    shortest = [elite["logical_steps"] for elite in archive.cells.values()]
    return {
        "method": method,
        "seed": config.seed,
        "bootstrap_outcome_cells": bootstrap_outcome_cells,
        "outcome_cells": len(archive),
        "shape_count": archive.shape_count,
        "layer_count": archive.layer_count,
        "median_elite_steps": (
            float(np.median(shortest)) if shortest else float("nan")
        ),
        "bootstrap_compute_steps": bootstrap_compute_steps,
        "search_steps": search_steps,
        "search_step_budget": config.search_step_budget,
        "rollouts": rollouts,
        "history_trajectories": len(history.trajectories),
        "history_states": history.state_count,
        "active_search_units": active_units,
        "coverage_auc": curve_auc(
            search_log, config.search_step_budget, bootstrap_outcome_cells
        ),
        "terminal_counts": terminal_counts,
        "wall_seconds": time.perf_counter() - started_at,
    }
