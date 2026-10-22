"""Tests for qd_folding.state_repositories using only the standard library."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_folding.config import ExperimentConfig
from qd_folding.state_repositories import (
    ArmStatistics,
    BoundedStateArchive,
    EventFrontier,
    EventType,
    select_ucb_arm,
    ucb_score,
    update_arm_statistics,
)


def trajectory(
    path: np.ndarray,
    *,
    terminal_reason: str = "running",
    intervention_counts: list[int] | None = None,
    valid_state_mask: list[bool] | None = None,
) -> dict:
    result = {
        "path": np.asarray(path, dtype=float),
        "terminal_reason": terminal_reason,
    }
    if intervention_counts is not None:
        result["intervention_counts"] = intervention_counts
    if valid_state_mask is not None:
        result["valid_state_mask"] = valid_state_mask
    return result


class UcbTests(unittest.TestCase):
    def test_score_update_and_random_tie_break(self) -> None:
        first = ArmStatistics()
        second = ArmStatistics()
        self.assertTrue(math.isinf(ucb_score(first, 0, 1.0)))
        update_arm_statistics(
            first, 0.5, simulated_steps=7, success=True
        )
        self.assertEqual(first.visits, 1)
        self.assertEqual(first.simulated_steps, 7)
        self.assertEqual(first.successes, 1)
        self.assertAlmostEqual(first.mean_reward, 0.5)

        rng = np.random.default_rng(4)
        selected = select_ucb_arm(
            [first, second],
            total_visits=1,
            exploration=1.0,
            rng=rng,
        )
        self.assertIs(selected, second)

        # Equal visited arms are not resolved by list order alone.
        update_arm_statistics(second, 0.5)
        choices = {
            id(
                select_ucb_arm(
                    [first, second],
                    total_visits=2,
                    exploration=1.0,
                    rng=np.random.default_rng(seed),
                )
            )
            for seed in range(10)
        }
        self.assertEqual(len(choices), 2)


class EventFrontierTests(unittest.TestCase):
    def config(self) -> ExperimentConfig:
        return replace(
            ExperimentConfig(),
            max_path_steps=20,
            max_interventions=3,
            event_frontier_capacity=12,
            event_max_add_per_rollout=8,
            event_stride=2,
            event_potential_prominence=0.05,
            event_basis_change_threshold=0.05,
            event_curvature_threshold=0.2,
            event_novelty_tolerance=0.01,
            event_merge_tolerance=0.002,
            event_random_points=1,
            event_peak_window=0,
        )

    def test_all_event_types_and_new_start(self) -> None:
        config = self.config()
        path = np.asarray(
            [
                [1.00, 0.80, 0.60, 0.90, 0.50, 0.11, 0.04, 0.01],
                [0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.40],
            ]
        )

        def modes(state: np.ndarray) -> np.ndarray:
            if state[1] < 0.2:
                return np.asarray([[1.0], [0.0]])
            return np.asarray([[0.0], [1.0]])

        frontier = EventFrontier(
            config,
            descriptor_fn=lambda state: state,
            gravity_fn=lambda state: abs(float(state[0])),
            modes_fn=modes,
            rng=np.random.default_rng(2),
        )
        report = frontier.add_trajectory(
            trajectory(
                path,
                terminal_reason="flat",
                intervention_counts=[0, 0, 1, 1, 1, 1, 1, 1],
            ),
            new_start_step=2,
            source_rollout="flat-0",
        )
        self.assertGreater(report.added, 0)
        self.assertLessEqual(report.added, config.event_max_add_per_rollout)
        self.assertTrue(all(record.step >= 2 for record in frontier.records))
        combined = EventType(0)
        for record in frontier.records:
            combined |= EventType(record.event_mask)
        self.assertTrue(combined & EventType.STRIDE)
        self.assertTrue(combined & EventType.POTENTIAL_EXTREMUM)
        self.assertTrue(combined & EventType.BASIS_CHANGE)
        self.assertTrue(combined & EventType.DESCRIPTOR_CURVATURE)
        self.assertTrue(combined & EventType.RANDOM)
        self.assertTrue(combined & EventType.LAST_NONFLAT)

    def test_collision_predecessor_and_merge(self) -> None:
        config = self.config()
        frontier = EventFrontier(
            config,
            descriptor_fn=lambda state: state,
            gravity_fn=lambda state: abs(float(state[0])),
            rng=np.random.default_rng(3),
        )
        value = trajectory(
            np.asarray([[0.8, 0.5, 0.25, 0.0]]),
            terminal_reason="collision",
            valid_state_mask=[True, True, True, False],
            intervention_counts=[0, 0, 1, 1],
        )
        first = frontier.add_trajectory(value, new_start_step=1)
        collision_records = [
            record
            for record in frontier.records
            if EventType(record.event_mask) & EventType.COLLISION_PREDECESSOR
        ]
        self.assertEqual(len(collision_records), 1)
        self.assertEqual(collision_records[0].step, 2)

        second = frontier.add_trajectory(value, new_start_step=1)
        self.assertGreater(second.merged, 0)
        self.assertEqual(len(frontier), first.added)

    def test_capacity_and_eligibility_are_strict(self) -> None:
        config = replace(
            self.config(),
            event_frontier_capacity=2,
            event_max_add_per_rollout=2,
            event_novelty_tolerance=0.001,
            event_merge_tolerance=0.0001,
        )
        frontier = EventFrontier(
            config,
            descriptor_fn=lambda state: state,
            rng=np.random.default_rng(1),
        )
        value = trajectory(
            np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]]),
            intervention_counts=[0, 0, 0, 0, 0],
        )
        frontier.add_trajectory(value)
        self.assertLessEqual(len(frontier), 2)
        for record in frontier.records:
            record.intervention_count = config.max_interventions
        self.assertEqual(frontier.eligible_records(), [])
        with self.assertRaises(LookupError):
            frontier.select()


class BoundedStateArchiveTests(unittest.TestCase):
    def test_bins_reservoir_capacity_and_stable_statistics(self) -> None:
        config = replace(
            ExperimentConfig(),
            state_bin_width=0.1,
            state_archive_max_cells=2,
            state_representatives_per_cell=2,
            max_path_steps=20,
            max_interventions=3,
        )
        archive = BoundedStateArchive(
            config,
            descriptor_fn=lambda state: state,
            rng=np.random.default_rng(5),
        )
        value = trajectory(
            np.asarray([[0.01, 0.02, 0.03, 0.15, 0.16, 0.26]]),
            intervention_counts=[0, 0, 0, 1, 1, 1],
        )
        report = archive.add_trajectory(value)
        self.assertEqual(report.considered, 6)
        self.assertLessEqual(len(archive), 2)
        self.assertTrue(
            all(
                len(cell.representative_ids)
                <= config.state_representatives_per_cell
                for cell in archive.cells
            )
        )
        all_ids = [
            state_id
            for cell in archive.all_cells
            for state_id in cell.representative_ids
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))

        cell, state = archive.select()
        original_id = cell.cell_id
        archive.update(cell, 0.75, simulated_steps=9, success=True)
        self.assertEqual(cell.statistics.visits, 1)
        self.assertEqual(cell.statistics.simulated_steps, 9)

        repeated = trajectory(
            np.asarray([[state.state[0], state.state[0] + 0.001]]),
            intervention_counts=[0, 0],
        )
        archive.add_trajectory(repeated)
        key = archive.bin_key(np.asarray([state.state[0]]))
        self.assertEqual(archive._bin_to_cell_id[key], original_id)
        self.assertEqual(archive._cells[original_id].statistics.visits, 1)

    def test_intervention_limit_filters_representatives(self) -> None:
        config = replace(
            ExperimentConfig(),
            state_archive_max_cells=8,
            state_representatives_per_cell=2,
            max_interventions=2,
        )
        archive = BoundedStateArchive(
            config,
            descriptor_fn=lambda state: state,
            rng=np.random.default_rng(8),
        )
        archive.add_trajectory(
            trajectory(
                np.asarray([[0.0, 0.2, 0.4]]),
                intervention_counts=[2, 2, 2],
            )
        )
        self.assertEqual(archive.eligible_cells(), [])
        with self.assertRaises(LookupError):
            archive.select()


if __name__ == "__main__":
    unittest.main()
