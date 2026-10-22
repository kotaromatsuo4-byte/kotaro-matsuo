"""Integration tests for terminal archives, descriptors, genomes, and methods."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qd_folding.archives import OutcomeArchive
from qd_folding.config import ExperimentConfig
from qd_folding.descriptors import (
    fold_shape_signature,
    last_nonflat_index,
    layer_order_signature,
    terminal_cell,
)
from qd_folding.experiment import run_experiment
from qd_folding.mechanics import load_model, node_to_vector
from qd_folding.sequence_genome import (
    materialize_genome,
    mutate_genome,
    random_genome,
    serializable_genome,
)
from qd_folding.sequence_search import SEQUENCE_METHOD
from qd_folding.state_search import BOUNDED_METHOD, EVENT_METHOD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAJECTORY_KEYS = {
    "path",
    "actions",
    "source",
    "terminal_reason",
    "is_flat",
    "collided",
    "failed",
    "new_start_step",
    "intervention_count",
    "compute_steps",
    "final_flatness",
}


def synthetic_flat_trajectory(model, *, duplicate_root: bool = False) -> dict:
    """Make a deterministic path with shape ``+-++`` and layer ``0>2``."""

    root = model.node_initial.copy()
    final = np.zeros_like(root)
    final[:, 0] = np.asarray([-1.0, 0.0, -1.0, 0.0, 1.0])

    observation = final.copy()
    observation[:, 2] = np.asarray([0.0, 0.2, -0.2, 0.0, 0.0])
    near_flat = final.copy()
    near_flat[:, 2] = 0.05 * observation[:, 2]

    nodes = [root]
    if duplicate_root:
        nodes.append(root.copy())
    nodes.extend([observation, near_flat, final])
    path = np.column_stack([node_to_vector(node) for node in nodes])
    actions = [
        np.zeros(model.vector_size, dtype=float)
        for _ in range(path.shape[1] - 1)
    ]
    return {
        "path": path,
        "actions": actions,
        "source": "synthetic",
        "terminal_reason": "flat",
        "is_flat": True,
        "collided": False,
        "failed": False,
        "new_start_step": 0,
        "intervention_count": 1,
        "compute_steps": path.shape[1] - 1,
        "final_flatness": 0.0,
        "action_kinds": ["synthetic"] * len(actions),
        "intervention_steps": [1],
    }


class OutcomeArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(DATA_DIR)
        cls.config = ExperimentConfig()

    def test_only_shortest_path_replaces_elite_in_same_cell(self) -> None:
        archive = OutcomeArchive()
        longer = synthetic_flat_trajectory(
            self.model, duplicate_root=True
        )
        shorter = synthetic_flat_trajectory(self.model)

        first = archive.update(
            longer,
            self.model,
            self.config,
            evaluation_index=0,
            method="test",
        )
        second = archive.update(
            shorter,
            self.model,
            self.config,
            evaluation_index=1,
            method="test",
        )
        third = archive.update(
            copy.deepcopy(longer),
            self.model,
            self.config,
            evaluation_index=2,
            method="test",
        )

        self.assertEqual(first.status, "new_cell")
        self.assertEqual(second.status, "elite_improved")
        self.assertGreater(second.relative_improvement, 0.0)
        self.assertEqual(third.status, "duplicate")
        self.assertEqual(len(archive), 1)
        elite = next(iter(archive.cells.values()))
        self.assertEqual(
            elite["logical_steps"], shorter["path"].shape[1] - 1
        )
        self.assertIs(elite["trajectory"], shorter)


class DescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(DATA_DIR)
        cls.config = ExperimentConfig()
        cls.trajectory = synthetic_flat_trajectory(cls.model)

    def test_shape_and_common_last_nonflat_layer_frame(self) -> None:
        path = self.trajectory["path"]
        self.assertEqual(
            fold_shape_signature(path[:, -1], self.model),
            "+-++",
        )
        observation_step = last_nonflat_index(
            path, self.model, self.config.layer_threshold
        )
        self.assertEqual(observation_step, 1)

        layer, reported_step = layer_order_signature(
            path,
            self.model,
            self.config.layer_threshold,
            self.config.overlap_tolerance,
            self.config.layer_separation_tolerance,
        )
        self.assertEqual(reported_step, observation_step)
        self.assertEqual(layer, "0>2")

        key, metadata = terminal_cell(
            self.trajectory, self.model, self.config
        )
        self.assertEqual(key, ("+-++", "0>2"))
        self.assertEqual(
            metadata["layer_observation_step"], observation_step
        )


class SequenceGenomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(DATA_DIR)

    def test_random_and_mutated_genomes_respect_all_bounds(self) -> None:
        config = ExperimentConfig(
            max_interventions=3,
            sequence_max_wait=5,
            sequence_insert_probability=0.8,
            sequence_delete_probability=0.8,
        )
        rng = np.random.default_rng(208)
        genome = random_genome(
            rng,
            latent_dimension=self.model.vector_size,
            config=config,
        )

        for _ in range(100):
            genome = mutate_genome(
                genome,
                rng,
                latent_dimension=self.model.vector_size,
                config=config,
            )
            interventions = genome["interventions"]
            self.assertGreaterEqual(len(interventions), 1)
            self.assertLessEqual(
                len(interventions), config.max_interventions
            )
            for intervention in interventions:
                self.assertGreaterEqual(int(intervention["wait"]), 0)
                self.assertLessEqual(
                    int(intervention["wait"]), config.sequence_max_wait
                )
                latent = np.asarray(intervention["latent"], dtype=float)
                self.assertEqual(latent.shape, (self.model.vector_size,))
                self.assertLessEqual(
                    float(np.linalg.norm(latent)),
                    config.perturb_max_norm + 1.0e-12,
                )

        encoded = serializable_genome(genome)
        decoded = materialize_genome(encoded)
        self.assertEqual(
            len(decoded["interventions"]), len(genome["interventions"])
        )
        for original, restored in zip(
            genome["interventions"], decoded["interventions"]
        ):
            self.assertEqual(
                int(original["wait"]), int(restored["wait"])
            )
            np.testing.assert_allclose(
                original["latent"], restored["latent"]
            )


class SmallExperimentContractTests(unittest.TestCase):
    def test_all_three_methods_obey_common_small_run_contract(self) -> None:
        config = ExperimentConfig(
            seed=101,
            bootstrap_count=3,
            max_path_steps=8,
            search_step_budget=6,
            local_cma_generations=1,
            local_cma_population=4,
            no_progress_patience=3,
            event_frontier_capacity=24,
            event_max_add_per_rollout=4,
            event_stride=2,
            event_random_points=1,
            state_archive_max_cells=24,
            state_representatives_per_cell=1,
            sequence_initial_random=2,
            max_interventions=2,
            sequence_max_wait=2,
        )
        expected_methods = {
            EVENT_METHOD,
            BOUNDED_METHOD,
            SEQUENCE_METHOD,
        }

        with TemporaryDirectory() as temporary:
            results = run_experiment(
                config=config,
                methods=["all"],
                project_root=PROJECT_ROOT,
                output=Path(temporary) / "output",
                save_outputs=False,
            )

        self.assertEqual(set(results), expected_methods)
        for method, result in results.items():
            self.assertEqual(result["method"], method)
            self.assertTrue(
                {
                    "history",
                    "outcome_archive",
                    "repository",
                    "search_log",
                    "repository_log",
                    "summary",
                }
                <= set(result)
            )
            summary = result["summary"]
            self.assertEqual(summary["method"], method)
            self.assertGreater(summary["rollouts"], 0)
            self.assertGreater(summary["search_steps"], 0)
            self.assertLessEqual(
                summary["search_steps"], config.search_step_budget
            )
            self.assertEqual(
                len(result["history"].trajectories),
                config.bootstrap_count + summary["rollouts"],
            )
            self.assertEqual(
                summary["outcome_cells"], len(result["outcome_archive"])
            )

            cumulative_steps = [
                int(row["search_steps"]) for row in result["search_log"]
            ]
            self.assertEqual(cumulative_steps, sorted(cumulative_steps))
            self.assertEqual(cumulative_steps[-1], summary["search_steps"])
            for trajectory in result["history"].trajectories:
                self.assertTrue(TRAJECTORY_KEYS <= set(trajectory))
                self.assertEqual(
                    trajectory["path"].shape[1] - 1,
                    len(trajectory["actions"]),
                )
                self.assertGreaterEqual(trajectory["compute_steps"], 0)


if __name__ == "__main__":
    unittest.main()
