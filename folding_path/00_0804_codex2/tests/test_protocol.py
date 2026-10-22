"""Tests for equal candidate sets and the nonstationarity-aware protocol."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qd_folding.candidate_sets import CandidatePools, CandidateState, POOL_KINDS
from qd_folding.config import ExperimentConfig
from qd_folding.mechanics import generate_bootstrap_paths, load_model
from qd_folding.protocol_experiment import (
    STAGE1_ROOT_BASELINES,
    run_two_stage_protocol,
)
from qd_folding.selectors import LifetimeUCBSelector, SMUCBSelector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def small_config(**overrides) -> ExperimentConfig:
    values = ExperimentConfig().to_dict()
    values.update(
        {
            "bootstrap_count": 3,
            "max_path_steps": 18,
            "search_step_budget": 24,
            "local_cma_generations": 1,
            "local_cma_population": 4,
            "candidate_pool_capacity": 6,
            "candidate_epoch_rollouts": 2,
            "save_all_histories": False,
        }
    )
    values.update(overrides)
    return ExperimentConfig.from_dict(values)


class CandidatePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(PROJECT_ROOT / "data")

    def test_all_active_sets_have_exactly_the_same_size(self) -> None:
        config = small_config()
        bootstrap = generate_bootstrap_paths(
            self.model,
            config,
            np.random.default_rng(12),
        )
        pools = CandidatePools(self.model, config, priority_seed=44)
        for index, trajectory in enumerate(bootstrap):
            pools.ingest_trajectory(
                trajectory,
                first_new_column=1,
                source_rollout=f"bootstrap-{index}",
            )

        active = pools.active_sets()
        self.assertEqual(set(active), set(POOL_KINDS))
        self.assertEqual({len(value) for value in active.values()}, {pools.common_size})
        self.assertGreater(pools.common_size, 0)
        self.assertTrue(
            pools.universes["frontier"] <= pools.universes["event"]
        )
        self.assertTrue(
            pools.universes["event"] <= pools.universes["history"]
        )


def dummy_state(state_id: int, feature: list[float]) -> CandidateState:
    trajectory = {
        "path": np.zeros((1, 1)),
        "actions": [],
        "action_kinds": [],
        "intervention_steps": [],
    }
    values = np.asarray(feature, dtype=float)
    return CandidateState(
        state_id=state_id,
        trajectory=trajectory,
        step=0,
        descriptor=values.copy(),
        feature=values.copy(),
        depth=0,
        intervention_count=0,
        potential=1.0,
        mechanism_dimension=1,
        priority=0.1 * (state_id + 1),
        created_order=state_id,
    )


class SelectorTests(unittest.TestCase):
    def test_lifetime_ucb_visits_every_new_state_before_reuse(self) -> None:
        first = dummy_state(0, [0.0, 0.0])
        second = dummy_state(1, [1.0, 1.0])
        selector = LifetimeUCBSelector(
            np.random.default_rng(1), exploration=1.0
        )
        context = frozenset()
        chosen = selector.select([first, second], context)
        selector.update(chosen, context, 1.0)
        other = selector.select([first, second], context)
        self.assertNotEqual(chosen.state_id, other.state_id)

    def test_smucb_accepts_archive_context_and_binary_feedback(self) -> None:
        first = dummy_state(0, [0.0, 0.0])
        second = dummy_state(1, [1.0, 1.0])
        selector = SMUCBSelector(
            np.random.default_rng(2),
            state_lengthscale=0.5,
            archive_lengthscale=2.0,
            noise_variance=0.1,
            delta=0.1,
            beta_scale=1.0,
        )
        empty = frozenset()
        selected = selector.select([first, second], empty)
        selector.update(selected, empty, 1.0)
        occupied = frozenset({("shape-a", "layer-a")})
        next_state = selector.select([first, second], occupied)
        selector.update(next_state, occupied, 0.0)
        self.assertEqual(len(selector.observations), 2)
        self.assertIsNotNone(selector.last_diagnostic)


class TwoStageProtocolTests(unittest.TestCase):
    def test_tiny_protocol_includes_root_controls_and_six_main_conditions(self) -> None:
        config = small_config(search_step_budget=12)
        with TemporaryDirectory() as temporary:
            result = run_two_stage_protocol(
                config=config,
                stage1_seeds=[7],
                stage2_seeds=[107],
                project_root=PROJECT_ROOT,
                output=Path(temporary) / "protocol",
                stages=(1, 2),
                save_outputs=False,
            )
        self.assertEqual(len(result["stage1"]["runs"]), 5)
        self.assertEqual(len(result["stage2"]["runs"]), 3)
        self.assertIn(
            result["stage1"]["winner"]["pool_kind"],
            POOL_KINDS,
        )
        for stage in ("stage1", "stage2"):
            for run in result[stage]["runs"].values():
                self.assertEqual(
                    run["summary"]["candidate_size_violations"], 0
                )

        root_runs = {
            name: run
            for (_, name), run in result["stage1"]["runs"].items()
            if run["summary"]["pool_kind"] in STAGE1_ROOT_BASELINES
        }
        self.assertEqual(len(root_runs), 2)
        for run in root_runs.values():
            self.assertEqual(run["summary"]["search_steps"], 12)
            self.assertEqual(run["summary"]["mean_candidate_size"], 1.0)
            self.assertTrue(run["search_log"])
            self.assertTrue(
                all(row["selected_state_step"] == 0 for row in run["search_log"])
            )
            self.assertTrue(
                all(row["selected_state_depth"] == 0 for row in run["search_log"])
            )
            self.assertTrue(
                all(row["new_start_step"] == 0 for row in run["search_log"])
            )

        bootstrap_counts = {
            run["summary"]["bootstrap_outcome_cells"]
            for run in result["stage1"]["runs"].values()
        }
        self.assertEqual(len(bootstrap_counts), 1)


if __name__ == "__main__":
    unittest.main()
