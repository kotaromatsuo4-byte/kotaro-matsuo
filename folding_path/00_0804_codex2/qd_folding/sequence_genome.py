"""Variable-length intervention genomes used by direct MAP-Elites."""

from __future__ import annotations

import copy

import numpy as np

from .mechanics import sample_uniform_ball


def bootstrap_genome(trajectory: dict) -> dict:
    """Recover the initial-imperfection intervention from a bootstrap path."""

    if not trajectory["intervention_steps"]:
        return {"interventions": []}
    first_step = int(trajectory["intervention_steps"][0])
    action = np.asarray(trajectory["actions"][first_step - 1], dtype=float)
    return {
        "interventions": [{"wait": 0, "latent": action.copy()}],
    }


def random_genome(
    rng: np.random.Generator,
    *,
    latent_dimension: int,
    config,
) -> dict:
    count = int(rng.integers(1, config.max_interventions + 1))
    interventions = [
        {
            "wait": int(rng.integers(0, config.sequence_max_wait + 1)),
            "latent": sample_uniform_ball(
                latent_dimension, config.perturb_max_norm, rng
            ),
        }
        for _ in range(count)
    ]
    return {"interventions": interventions}


def _clamp_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= radius:
        return vector
    return radius * vector / max(norm, 1.0e-14)


def mutate_genome(
    parent: dict,
    rng: np.random.Generator,
    *,
    latent_dimension: int,
    config,
) -> dict:
    """Mutation with coefficient, wait, insertion, and deletion operators."""

    child = copy.deepcopy(parent)

    for intervention in child["interventions"]:
        latent = np.asarray(intervention["latent"], dtype=float)
        latent = latent + config.sequence_coefficient_sigma * rng.normal(
            size=latent_dimension
        )
        intervention["latent"] = _clamp_ball(
            latent, config.perturb_max_norm
        )
        wait_shift = int(
            np.rint(config.sequence_wait_sigma * rng.normal())
        )
        intervention["wait"] = int(
            np.clip(
                int(intervention["wait"]) + wait_shift,
                0,
                config.sequence_max_wait,
            )
        )

    if (
        len(child["interventions"]) < config.max_interventions
        and rng.random() < config.sequence_insert_probability
    ):
        insertion = {
            "wait": int(rng.integers(0, config.sequence_max_wait + 1)),
            "latent": sample_uniform_ball(
                latent_dimension, config.perturb_max_norm, rng
            ),
        }
        position = int(rng.integers(len(child["interventions"]) + 1))
        child["interventions"].insert(position, insertion)

    if (
        child["interventions"]
        and rng.random() < config.sequence_delete_probability
    ):
        position = int(rng.integers(len(child["interventions"])))
        del child["interventions"][position]

    if not child["interventions"]:
        child["interventions"].append(
            {
                "wait": int(rng.integers(0, config.sequence_max_wait + 1)),
                "latent": sample_uniform_ball(
                    latent_dimension, config.perturb_max_norm, rng
                ),
            }
        )
    return child


def serializable_genome(genome: dict) -> dict:
    return {
        "interventions": [
            {
                "wait": int(value["wait"]),
                "latent": np.asarray(value["latent"], dtype=float).tolist(),
            }
            for value in genome["interventions"]
        ],
    }


def materialize_genome(genome: dict) -> dict:
    """Convert a JSON-compatible stored genome back to NumPy arrays."""

    return {
        "interventions": [
            {
                "wait": int(value["wait"]),
                "latent": np.asarray(value["latent"], dtype=float),
            }
            for value in genome["interventions"]
        ],
    }
