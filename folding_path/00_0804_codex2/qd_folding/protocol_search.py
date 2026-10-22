"""Common state-restart loop for the two-stage comparison protocol."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np

from .candidate_sets import CandidatePools, CandidateState, POOL_KINDS
from .mechanics import (
    Model,
    mechanism_modes,
    rollout_mode_perturbation,
    rollout_random_perturbation,
    sample_uniform_ball,
    start_path,
)
from .search_common import initialize_from_bootstrap, make_summary
from .selectors import SELECTOR_KINDS, make_selector


# ``root`` is the fair location-only baseline: it uses exactly the same
# random-emitter radius as the intermediate-state conditions.  The
# ``root_bootstrap`` sensitivity condition repeats the smaller initial
# imperfection used to create the three common bootstrap paths.
ROOT_POOL_KINDS = ("root", "root_bootstrap")


def method_name(pool_kind: str, selector_kind: str) -> str:
    return f"{pool_kind}_{selector_kind}"


def _prefix_from_state(record: CandidateState) -> dict:
    """Copy the complete root prefix represented by one candidate state."""

    trajectory = record.trajectory
    step = int(record.step)
    return {
        "path": np.asarray(trajectory["path"], dtype=float)[:, : step + 1].copy(),
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


def _new_cell_auc(
    search_log: list[dict],
    budget: int,
    bootstrap_cells: int,
) -> float:
    """Area under discoveries-after-bootstrap versus mechanics-step budget."""

    if budget <= 0:
        return 0.0
    previous = 0
    discoveries = 0
    area = 0.0
    for row in search_log:
        current = min(int(row["search_steps"]), budget)
        area += (current - previous) * discoveries
        previous = current
        discoveries = int(row["outcome_cells"]) - int(bootstrap_cells)
        if current >= budget:
            break
    area += max(budget - previous, 0) * discoveries
    return area / budget


def run_pool_search(
    *,
    pool_kind: str,
    selector_kind: str,
    model: Model,
    config,
    bootstrap_paths: list[dict],
    priority_seed: int,
    selector_rng: np.random.Generator,
    emitter_rng: np.random.Generator,
) -> dict:
    """Run one fixed ``candidate-set x selector`` condition.

    The terminal archive is the only source of reward.  A new outcome cell
    gives one and every other outcome (including an elite shortening) gives
    zero.  Elite replacement is still performed and reported independently.
    """

    pool_kind = str(pool_kind).strip().lower()
    selector_kind = str(selector_kind).strip().lower()
    if pool_kind not in POOL_KINDS:
        raise ValueError(f"unknown pool kind: {pool_kind}")
    if selector_kind not in SELECTOR_KINDS:
        raise ValueError(f"unknown selector kind: {selector_kind}")
    config.validate()

    method = method_name(pool_kind, selector_kind)
    started_at = time.perf_counter()
    history, archive, copied_bootstrap = initialize_from_bootstrap(
        bootstrap_paths,
        model,
        config,
        method,
    )
    bootstrap_outcome_cells = len(archive)
    bootstrap_compute_steps = sum(
        int(value["compute_steps"]) for value in copied_bootstrap
    )

    pools = CandidatePools(model, config, priority_seed=priority_seed)
    repository_log: list[dict] = []
    for index, trajectory in enumerate(copied_bootstrap):
        # Step zero is the common root and is excluded for all three rules.
        report = pools.ingest_trajectory(
            trajectory,
            first_new_column=1,
            source_rollout=f"bootstrap_{index:03d}",
        )
        repository_log.append(
            {
                "method": method,
                "evaluation": -(len(copied_bootstrap) - index),
                "source": trajectory["source"],
                **asdict(report),
                **{f"raw_{key}": value for key, value in pools.raw_sizes.items()},
                "common_candidate_size": pools.common_size,
            }
        )

    selector = make_selector(selector_kind, config, selector_rng)
    search_log: list[dict] = []
    search_steps = 0
    rollouts = 0
    epoch_index = -1
    epoch_candidates: list[CandidateState] = []
    epoch_pool_sizes: dict[str, int] = {}
    size_violations = 0
    reward_sum = 0.0
    elite_improvements = 0

    while search_steps < config.search_step_budget:
        if not epoch_candidates or rollouts % config.candidate_epoch_rollouts == 0:
            try:
                snapshots = pools.active_sets()
            except LookupError:
                break
            epoch_index += 1
            epoch_candidates = list(snapshots[pool_kind])
            epoch_pool_sizes = {
                key: len(values) for key, values in snapshots.items()
            }
            if len(set(epoch_pool_sizes.values())) != 1:
                size_violations += 1

        archive_context = frozenset(archive.cells.keys())
        try:
            selected = selector.select(epoch_candidates, archive_context)
        except LookupError:
            break
        diagnostic = selector.last_diagnostic
        prefix = _prefix_from_state(selected)
        remaining = config.search_step_budget - search_steps
        trajectory = rollout_random_perturbation(
            prefix["path"],
            prefix["actions"],
            model,
            config,
            emitter_rng,
            source=f"{method}_rollout_{rollouts:05d}",
            remaining_compute_steps=remaining,
            prefix_action_kinds=prefix["action_kinds"],
            prefix_intervention_count=selected.intervention_count,
            prefix_intervention_steps=prefix["intervention_steps"],
        )
        consumed = int(trajectory["compute_steps"])
        if consumed <= 0:
            selector.update(selected, archive_context, 0.0)
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
                "pool_kind": pool_kind,
                "selector_kind": selector_kind,
                "selected_state_id": selected.state_id,
                "epoch_index": epoch_index,
            },
        )
        if archive_update.is_improvement:
            elite_improvements += 1

        first_new_column = min(
            int(trajectory["new_start_step"]) + 1,
            trajectory["path"].shape[1],
        )
        insertion = pools.ingest_trajectory(
            trajectory,
            first_new_column=first_new_column,
            source_rollout=rollouts,
        )

        reward = float(archive_update.is_new)
        reward_sum += reward
        selector.update(selected, archive_context, reward)
        raw_sizes = pools.raw_sizes
        repository_log.append(
            {
                "method": method,
                "evaluation": rollouts,
                "source": trajectory["source"],
                **asdict(insertion),
                **{f"raw_{key}": value for key, value in raw_sizes.items()},
                "common_candidate_size": pools.common_size,
            }
        )
        search_log.append(
            {
                "method": method,
                "pool_kind": pool_kind,
                "selector_kind": selector_kind,
                "evaluation": rollouts,
                "epoch_index": epoch_index,
                "search_steps": search_steps,
                "compute_steps": consumed,
                "logical_steps": trajectory["path"].shape[1] - 1,
                "new_start_step": int(trajectory["new_start_step"]),
                "selected_state_id": selected.state_id,
                "selected_state_step": selected.step,
                "selected_state_depth": selected.depth,
                "selected_state_interventions": selected.intervention_count,
                "selected_state_event_mask": selected.event_mask,
                "selection_score": (
                    diagnostic.score if diagnostic is not None else None
                ),
                "posterior_mean": (
                    diagnostic.posterior_mean if diagnostic is not None else None
                ),
                "posterior_std": (
                    diagnostic.posterior_std if diagnostic is not None else None
                ),
                "state_visits_before": (
                    diagnostic.visits if diagnostic is not None else None
                ),
                "reward": reward,
                "archive_update": archive_update.status,
                "terminal_reason": trajectory["terminal_reason"],
                "final_flatness": trajectory["final_flatness"],
                "outcome_cells": len(archive),
                "shape_count": archive.shape_count,
                "layer_count": archive.layer_count,
                "epoch_candidate_size": len(epoch_candidates),
                **{
                    f"epoch_{key}_size": value
                    for key, value in epoch_pool_sizes.items()
                },
                **{f"raw_{key}": value for key, value in raw_sizes.items()},
            }
        )
        rollouts += 1

    active_units = len(epoch_candidates)
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
        active_units=active_units,
        config=config,
    )
    candidate_sizes = [
        int(row["epoch_candidate_size"]) for row in search_log
    ]
    summary.update(
        {
            "pool_kind": pool_kind,
            "selector_kind": selector_kind,
            "new_outcome_cells": len(archive) - bootstrap_outcome_cells,
            "new_cell_auc": _new_cell_auc(
                search_log,
                config.search_step_budget,
                bootstrap_outcome_cells,
            ),
            "selector_reward_sum": reward_sum,
            "elite_improvements": elite_improvements,
            "candidate_pool_capacity": config.candidate_pool_capacity,
            "candidate_epoch_rollouts": config.candidate_epoch_rollouts,
            "mean_candidate_size": (
                float(np.mean(candidate_sizes)) if candidate_sizes else 0.0
            ),
            "max_candidate_size": max(candidate_sizes, default=0),
            "candidate_size_violations": size_violations,
            **{f"final_raw_{key}": value for key, value in pools.raw_sizes.items()},
        }
    )
    return {
        "method": method,
        "history": history,
        "outcome_archive": archive,
        "repository": pools,
        "selector": selector,
        "search_log": search_log,
        "repository_log": repository_log,
        "summary": summary,
    }


def run_root_search(
    *,
    pool_kind: str,
    model: Model,
    config,
    bootstrap_paths: list[dict],
    emitter_rng: np.random.Generator,
) -> dict:
    """Spend the complete search budget on independent root restarts.

    Both root conditions receive the same three common bootstrap trajectories
    as the ordinary Stage-1 conditions.  They then repeatedly perturb the
    undeformed root and run the same deterministic rollout until the common
    post-bootstrap mechanics-step budget is exhausted.

    ``root`` uses ``perturb_max_norm`` and is therefore the primary control for
    perturbation *location*.  ``root_bootstrap`` uses
    ``initial_imperfection`` and is the literal "keep generating bootstrap
    paths" sensitivity condition; it changes location and perturbation radius
    together and must not be interpreted as a location-only comparison.
    """

    pool_kind = str(pool_kind).strip().lower()
    if pool_kind not in ROOT_POOL_KINDS:
        raise ValueError(f"unknown root pool kind: {pool_kind}")
    config.validate()

    selector_kind = "random"
    method = method_name(pool_kind, selector_kind)
    started_at = time.perf_counter()
    history, archive, copied_bootstrap = initialize_from_bootstrap(
        bootstrap_paths,
        model,
        config,
        method,
    )
    bootstrap_outcome_cells = len(archive)
    bootstrap_compute_steps = sum(
        int(value["compute_steps"]) for value in copied_bootstrap
    )

    root_path = start_path(model.node_initial)
    root_dimension = mechanism_modes(model.node_initial, model).shape[1]
    perturbation_radius = (
        config.perturb_max_norm
        if pool_kind == "root"
        else config.initial_imperfection
    )

    search_log: list[dict] = []
    search_steps = 0
    rollouts = 0
    reward_sum = 0.0
    elite_improvements = 0

    while search_steps < config.search_step_budget:
        remaining = config.search_step_budget - search_steps
        if pool_kind == "root":
            # This is exactly the emitter used by frontier/event/history.
            trajectory = rollout_random_perturbation(
                root_path,
                [],
                model,
                config,
                emitter_rng,
                source=f"{method}_rollout_{rollouts:05d}",
                remaining_compute_steps=remaining,
            )
        else:
            # Literal repeated-bootstrap condition with the smaller radius.
            coefficients = sample_uniform_ball(
                root_dimension,
                config.initial_imperfection,
                emitter_rng,
            )
            trajectory = rollout_mode_perturbation(
                root_path,
                [],
                coefficients,
                model,
                config,
                source=f"{method}_rollout_{rollouts:05d}",
                remaining_compute_steps=remaining,
                maximum_norm=config.initial_imperfection,
            )

        consumed = int(trajectory["compute_steps"])
        if consumed <= 0:
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
                "pool_kind": pool_kind,
                "selector_kind": selector_kind,
                "selected_state_id": "root",
                "perturbation_radius": perturbation_radius,
            },
        )
        if archive_update.is_improvement:
            elite_improvements += 1
        reward = float(archive_update.is_new)
        reward_sum += reward
        search_log.append(
            {
                "method": method,
                "pool_kind": pool_kind,
                "selector_kind": selector_kind,
                "evaluation": rollouts,
                "epoch_index": 0,
                "search_steps": search_steps,
                "compute_steps": consumed,
                "logical_steps": trajectory["path"].shape[1] - 1,
                "new_start_step": int(trajectory["new_start_step"]),
                "selected_state_id": "root",
                "selected_state_step": 0,
                "selected_state_depth": 0,
                "selected_state_interventions": 0,
                "selected_state_event_mask": 0,
                "selection_score": None,
                "posterior_mean": None,
                "posterior_std": None,
                "state_visits_before": rollouts,
                "reward": reward,
                "archive_update": archive_update.status,
                "terminal_reason": trajectory["terminal_reason"],
                "final_flatness": trajectory["final_flatness"],
                "outcome_cells": len(archive),
                "shape_count": archive.shape_count,
                "layer_count": archive.layer_count,
                "epoch_candidate_size": 1,
                "raw_root": 1,
                "perturbation_radius": perturbation_radius,
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
        active_units=1,
        config=config,
    )
    summary.update(
        {
            "pool_kind": pool_kind,
            "selector_kind": selector_kind,
            "new_outcome_cells": len(archive) - bootstrap_outcome_cells,
            "new_cell_auc": _new_cell_auc(
                search_log,
                config.search_step_budget,
                bootstrap_outcome_cells,
            ),
            "selector_reward_sum": reward_sum,
            "elite_improvements": elite_improvements,
            "candidate_pool_capacity": 1,
            "candidate_epoch_rollouts": 0,
            "mean_candidate_size": 1.0 if search_log else 0.0,
            "max_candidate_size": 1 if search_log else 0,
            "candidate_size_violations": 0,
            "final_raw_root": 1,
            "perturbation_radius": perturbation_radius,
            "root_baseline_kind": (
                "same_random_emitter"
                if pool_kind == "root"
                else "literal_bootstrap_radius"
            ),
        }
    )
    return {
        "method": method,
        "history": history,
        "outcome_archive": archive,
        "repository": None,
        "selector": None,
        "search_log": search_log,
        "repository_log": [],
        "summary": summary,
    }


__all__ = [
    "ROOT_POOL_KINDS",
    "method_name",
    "run_pool_search",
    "run_root_search",
]
