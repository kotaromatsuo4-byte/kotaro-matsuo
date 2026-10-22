"""Direct intervention-sequence MAP-Elites."""

from __future__ import annotations

import time

import numpy as np

from .mechanics import Model, rollout_with_interventions
from .search_common import initialize_from_bootstrap, make_summary
from .sequence_genome import (
    bootstrap_genome,
    materialize_genome,
    mutate_genome,
    random_genome,
    serializable_genome,
)


SEQUENCE_METHOD = "sequence_map_elites"


def _attach_bootstrap_genomes(archive) -> None:
    for elite in archive.cells.values():
        genome = bootstrap_genome(elite["trajectory"])
        elite["candidate_metadata"] = {
            **elite.get("candidate_metadata", {}),
            "origin": "bootstrap",
            "genome": serializable_genome(genome),
        }


def _parent_elites(archive) -> list[tuple[tuple[str, str], dict]]:
    return [
        (key, elite)
        for key, elite in archive.cells.items()
        if "genome" in elite.get("candidate_metadata", {})
    ]


def _propose_genome(
    archive,
    evaluation: int,
    model: Model,
    config,
    rng: np.random.Generator,
) -> tuple[dict, str, tuple[str, str] | None]:
    parents = _parent_elites(archive)
    force_random = evaluation < config.sequence_initial_random
    fresh_random = rng.random() < config.sequence_fresh_random_probability
    choose_parent = (
        parents
        and not force_random
        and not fresh_random
        and rng.random() < config.sequence_parent_probability
    )
    if not choose_parent:
        return (
            random_genome(
                rng,
                latent_dimension=model.vector_size,
                config=config,
            ),
            "fresh_random",
            None,
        )

    parent_key, parent_elite = parents[int(rng.integers(len(parents)))]
    parent = materialize_genome(
        parent_elite["candidate_metadata"]["genome"]
    )
    return (
        mutate_genome(
            parent,
            rng,
            latent_dimension=model.vector_size,
            config=config,
        ),
        "elite_mutation",
        parent_key,
    )


def run_sequence_map_elites(
    model: Model,
    config,
    bootstrap_paths: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Search complete wait-and-perturb programs and map terminal outcomes."""

    started_at = time.perf_counter()
    history, archive, copied_bootstrap = initialize_from_bootstrap(
        bootstrap_paths, model, config, SEQUENCE_METHOD
    )
    _attach_bootstrap_genomes(archive)
    bootstrap_outcome_cells = len(archive)
    bootstrap_compute_steps = sum(
        int(value["compute_steps"]) for value in copied_bootstrap
    )

    search_log: list[dict] = []
    search_steps = 0
    evaluations = 0
    while search_steps < config.search_step_budget:
        genome, origin, parent_key = _propose_genome(
            archive, evaluations, model, config, rng
        )
        remaining = config.search_step_budget - search_steps
        trajectory = rollout_with_interventions(
            model.node_initial,
            genome["interventions"],
            model,
            config,
            source=f"{SEQUENCE_METHOD}_candidate_{evaluations:05d}",
            remaining_compute_steps=remaining,
        )
        consumed = int(trajectory["compute_steps"])
        if consumed <= 0:
            break
        search_steps += consumed
        history.append(trajectory)

        stored_genome = serializable_genome(genome)
        update = archive.update(
            trajectory,
            model,
            config,
            evaluation_index=evaluations,
            method=SEQUENCE_METHOD,
            candidate_metadata={
                "origin": origin,
                "parent_cell": parent_key,
                "genome": stored_genome,
            },
        )
        search_log.append(
            {
                "method": SEQUENCE_METHOD,
                "evaluation": evaluations,
                "search_steps": search_steps,
                "compute_steps": consumed,
                "logical_steps": trajectory["path"].shape[1] - 1,
                "origin": origin,
                "parent_cell": parent_key,
                "intervention_count": len(genome["interventions"]),
                "waits": [
                    int(value["wait"])
                    for value in genome["interventions"]
                ],
                "archive_update": update.status,
                "terminal_reason": trajectory["terminal_reason"],
                "final_flatness": trajectory["final_flatness"],
                "outcome_cells": len(archive),
                "shape_count": archive.shape_count,
                "layer_count": archive.layer_count,
                "repository_size": len(archive),
            }
        )
        evaluations += 1

    summary = make_summary(
        method=SEQUENCE_METHOD,
        history=history,
        archive=archive,
        search_log=search_log,
        bootstrap_outcome_cells=bootstrap_outcome_cells,
        bootstrap_compute_steps=bootstrap_compute_steps,
        search_steps=search_steps,
        rollouts=evaluations,
        started_at=started_at,
        active_units=len(archive),
        config=config,
    )
    return {
        "method": SEQUENCE_METHOD,
        "history": history,
        "outcome_archive": archive,
        "repository": archive,
        "search_log": search_log,
        "repository_log": [],
        "summary": summary,
    }
