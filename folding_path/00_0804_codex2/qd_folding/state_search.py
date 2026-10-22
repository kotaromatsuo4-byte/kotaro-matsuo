"""The two state-restart methods with UCB and a random emitter."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np

from .archives import OutcomeArchive
from .descriptors import configuration_descriptor
from .mechanics import (
    Model,
    gravity_potential,
    mechanism_modes,
    rollout_random_perturbation,
    vector_to_node,
)
from .search_common import (
    archive_reward,
    initialize_from_bootstrap,
    make_summary,
)
from .state_repositories import (
    BoundedStateArchive,
    EventFrontier,
    StateRecord,
)


EVENT_METHOD = "event_frontier_ucb_random"
BOUNDED_METHOD = "bounded_archive_ucb_random"


def _descriptor_callback(model: Model):
    return lambda state: configuration_descriptor(state, model)


def _gravity_callback(model: Model):
    return lambda state: gravity_potential(vector_to_node(state, model))


def _modes_callback(model: Model):
    return lambda state: mechanism_modes(vector_to_node(state, model), model)


def make_event_frontier(
    model: Model, config, rng: np.random.Generator
) -> EventFrontier:
    return EventFrontier(
        config,
        descriptor_fn=_descriptor_callback(model),
        gravity_fn=_gravity_callback(model),
        modes_fn=_modes_callback(model),
        rng=rng,
    )


def make_bounded_archive(
    model: Model, config, rng: np.random.Generator
) -> BoundedStateArchive:
    return BoundedStateArchive(
        config,
        descriptor_fn=_descriptor_callback(model),
        rng=rng,
    )


def _prefix_from_record(record: StateRecord) -> dict:
    trajectory = record.trajectory
    step = int(record.step)
    return {
        "path": trajectory["path"][:, : step + 1].copy(),
        "actions": [
            np.asarray(value, dtype=float).copy()
            for value in trajectory["actions"][:step]
        ],
        "action_kinds": list(trajectory["action_kinds"][:step]),
        "intervention_steps": [
            int(value)
            for value in trajectory.get("intervention_steps", [])
            if int(value) <= step
        ],
    }


def _insertion_row(
    *,
    method: str,
    evaluation: int,
    source: str,
    report,
    repository_size: int,
) -> dict:
    row = {
        "method": method,
        "evaluation": evaluation,
        "source": source,
        "repository_size": repository_size,
    }
    row.update(asdict(report))
    return row


def _add_bootstrap_to_repository(
    repository,
    bootstrap_paths: list[dict],
    method: str,
) -> list[dict]:
    rows: list[dict] = []
    for index, trajectory in enumerate(bootstrap_paths):
        # The common root is useful once; novelty/bin merging removes repeats.
        report = repository.add_trajectory(
            trajectory,
            new_start_step=0,
            source_rollout=f"bootstrap_{index:03d}",
        )
        rows.append(
            _insertion_row(
                method=method,
                evaluation=-(len(bootstrap_paths) - index),
                source=trajectory["source"],
                report=report,
                repository_size=len(repository),
            )
        )
    return rows


def run_state_restart_search(
    *,
    method: str,
    model: Model,
    config,
    bootstrap_paths: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Run either event-frontier UCB or bounded-cell UCB."""

    if method == EVENT_METHOD:
        repository = make_event_frontier(model, config, rng)
    elif method == BOUNDED_METHOD:
        repository = make_bounded_archive(model, config, rng)
    else:
        raise ValueError(f"Unknown state-restart method: {method}")

    started_at = time.perf_counter()
    history, archive, copied_bootstrap = initialize_from_bootstrap(
        bootstrap_paths, model, config, method
    )
    bootstrap_outcome_cells = len(archive)
    bootstrap_compute_steps = sum(
        int(value["compute_steps"]) for value in copied_bootstrap
    )
    repository_log = _add_bootstrap_to_repository(
        repository, copied_bootstrap, method
    )
    if len(repository) == 0:
        raise RuntimeError(f"{method}: bootstrap produced no eligible states")

    search_log: list[dict] = []
    search_steps = 0
    rollouts = 0
    while search_steps < config.search_step_budget:
        try:
            if method == EVENT_METHOD:
                selected_record = repository.select(
                    total_visits=rollouts
                )
                selected_arm = selected_record
                selected_arm_id = selected_record.state_id
            else:
                selected_cell, selected_record = repository.select(
                    total_visits=rollouts
                )
                selected_arm = selected_cell
                selected_arm_id = selected_cell.cell_id
        except LookupError:
            break

        prefix = _prefix_from_record(selected_record)
        remaining = config.search_step_budget - search_steps
        trajectory = rollout_random_perturbation(
            prefix["path"],
            prefix["actions"],
            model,
            config,
            rng,
            source=f"{method}_rollout_{rollouts:05d}",
            remaining_compute_steps=remaining,
            prefix_action_kinds=prefix["action_kinds"],
            prefix_intervention_count=selected_record.intervention_count,
            prefix_intervention_steps=prefix["intervention_steps"],
        )
        consumed = int(trajectory["compute_steps"])
        if consumed <= 0:
            repository.update(
                selected_arm,
                0.0,
                simulated_steps=0,
                success=False,
            )
            break
        search_steps += consumed
        history.append(trajectory)

        archive_update = archive.update(
            trajectory,
            model,
            config,
            evaluation_index=rollouts,
            method=method,
            candidate_metadata={
                "selected_state_id": selected_record.state_id,
                "selected_arm_id": selected_arm_id,
            },
        )

        # The restart state is an already known prefix endpoint.  Only newly
        # generated columns are offered to the active repository.
        first_new_column = min(
            int(trajectory["new_start_step"]) + 1,
            trajectory["path"].shape[1],
        )
        insertion = repository.add_trajectory(
            trajectory,
            new_start_step=first_new_column,
            source_rollout=rollouts,
        )
        reward = archive_reward(
            archive_update,
            insertion.exploration_gain,
            consumed,
            config,
        )
        repository.update(
            selected_arm,
            reward,
            simulated_steps=consumed,
            success=archive_update.is_new or archive_update.is_improvement,
        )

        repository_log.append(
            _insertion_row(
                method=method,
                evaluation=rollouts,
                source=trajectory["source"],
                report=insertion,
                repository_size=len(repository),
            )
        )
        search_log.append(
            {
                "method": method,
                "evaluation": rollouts,
                "search_steps": search_steps,
                "compute_steps": consumed,
                "logical_steps": trajectory["path"].shape[1] - 1,
                "selected_arm_id": selected_arm_id,
                "selected_state_id": selected_record.state_id,
                "selected_state_step": selected_record.step,
                "selected_state_interventions": (
                    selected_record.intervention_count
                ),
                "arm_visits": selected_arm.statistics.visits,
                "arm_mean_reward": selected_arm.statistics.mean_reward,
                "reward": reward,
                "archive_update": archive_update.status,
                "terminal_reason": trajectory["terminal_reason"],
                "final_flatness": trajectory["final_flatness"],
                "outcome_cells": len(archive),
                "shape_count": archive.shape_count,
                "layer_count": archive.layer_count,
                "repository_size": len(repository),
                "repository_exploration_gain": insertion.exploration_gain,
            }
        )
        rollouts += 1

    summary = make_summary(
        method=method,
        history=history,
        archive=archive,
        search_log=search_log,
        bootstrap_outcome_cells=bootstrap_outcome_cells,
        bootstrap_compute_steps=bootstrap_compute_steps,
        search_steps=search_steps,
        rollouts=rollouts,
        started_at=started_at,
        active_units=len(repository),
        config=config,
    )
    return {
        "method": method,
        "history": history,
        "outcome_archive": archive,
        "repository": repository,
        "search_log": search_log,
        "repository_log": repository_log,
        "summary": summary,
    }


def run_event_frontier_search(
    model: Model,
    config,
    bootstrap_paths: list[dict],
    rng: np.random.Generator,
) -> dict:
    return run_state_restart_search(
        method=EVENT_METHOD,
        model=model,
        config=config,
        bootstrap_paths=bootstrap_paths,
        rng=rng,
    )


def run_bounded_archive_search(
    model: Model,
    config,
    bootstrap_paths: list[dict],
    rng: np.random.Generator,
) -> dict:
    return run_state_restart_search(
        method=BOUNDED_METHOD,
        model=model,
        config=config,
        bootstrap_paths=bootstrap_paths,
        rng=rng,
    )
