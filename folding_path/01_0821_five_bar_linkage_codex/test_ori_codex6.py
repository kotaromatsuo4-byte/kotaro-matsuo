"""Focused reward-ledger tests for :mod:`ori_codex6`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import ori_codex3 as ori3
import ori_codex5 as ori5
import ori_codex6 as ori6


def metrics(flatness, spread, target=2.0):
    error = abs(float(spread) - float(target))
    return {
        "flatness_ratio": float(flatness),
        "spread": float(spread),
        "spread_error": error,
        "target_deviation_energy": error * error,
    }


class ScriptedRewardEnv(ori6.CommonRewardTargetFoldingEnv):
    def compactness_metrics(self, node=None):
        return dict(self.scripted_metrics[self.scripted_index])


def scripted_environment(
    states,
    reasons,
    valid=None,
    max_steps=10,
    frontier_mode="state_potential",
):
    """Build only the reward state needed by the scripted step tests."""

    env = object.__new__(ScriptedRewardEnv)
    env.scripted_metrics = list(states)
    env.scripted_index = 0
    env.scripted_reasons = list(reasons)
    env.scripted_valid = (
        [True] * len(reasons) if valid is None else list(valid)
    )
    env.target_spread = 2.0
    env.spread_tolerance = 0.05
    env.evaluation_flatness_tolerance = 0.001
    env.near_miss_flatness_far = 1.0
    env.near_miss_spread_error_far = 1.0
    env.ori5_flatness_clip = 2.0
    env.ori5_spread_milestone_initial_factor = 2.0
    env.ori5_spread_milestone_bonus = 0.0
    env.ori5_joint_flatness_thresholds = (0.5, 0.1, 0.01)
    env.ori5_joint_milestone_bonuses = (0.2, 0.3, 0.5)
    env.ori5_crossing_penalty = 1.25
    env.ori5_wrong_flat_penalty = 2.0
    env.ori5_timeout_penalty = 1.0
    env.ori5_success_bonus = 10.0
    env.ori5_quality_weight = 1.0
    env.ori5_success_speed_bonus_weight = 0.5
    env.numerical_failure_penalty = 10.0
    env.no_mode_penalty = 5.0
    env.max_steps = int(max_steps)
    env.step_count = 0
    env.ori6_spread_frontier_bonus = 0.9
    env.ori6_spread_frontier_mode = str(frontier_mode)
    env.ori6_spread_frontier_flatness_cap = 2.0
    env._reset_ori5_episode_state()
    env._validate_ori6_parameters()
    env._reset_ori6_episode_state()
    return env


def scripted_parent_step(
    env, physical_direction, requested_width, mode_coefficients
):
    index = env.scripted_index
    reason = env.scripted_reasons[index]
    is_valid = env.scripted_valid[index]
    env.scripted_index += 1
    env.step_count += 1
    return None, 12345.0, reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(is_valid),
    }


def take_scripted_step(env):
    with mock.patch.object(
        ori3.CompactTargetFoldingEnv,
        "step",
        scripted_parent_step,
    ):
        return ori6.CommonRewardTargetFoldingEnv.step(env, None, 0.0, [])


def component_sum(info):
    return sum(float(info[name]) for name in (
        "ori6_flatness_progress_reward",
        "ori6_target_progress_reward",
        "ori6_spread_frontier_reward",
        "ori6_joint_milestone_reward",
        "ori6_terminal_fixed_reward",
        "ori6_terminal_quality_reward",
        "ori6_terminal_speed_reward",
        "ori6_terminal_correction_reward",
    ))


class OriCodex6RewardTests(unittest.TestCase):
    def test_default_frontier_is_continuous_bounded_and_telescoping(self):
        env = scripted_environment(
            [
                metrics(1.0, 0.0),
                metrics(1.0, 1.0),
                metrics(1.0, 2.0),
                metrics(1.0, 1.5),
                metrics(1.0, 2.0),
            ],
            ["running"] * 4,
        )
        observed = []
        for _ in range(4):
            _, reward, _, info = take_scripted_step(env)
            observed.append(info["ori6_spread_frontier_reward"])
            self.assertAlmostEqual(reward, component_sum(info), places=12)
            self.assertEqual(info["ori6_hard_spread_milestone_reward"], 0.0)

        first = 0.9 * (1.0 / 1.95)
        expected = [first, 0.9 - first, -(0.9 - 0.9 * 1.5 / 1.95),
                    0.9 - 0.9 * 1.5 / 1.95]
        np.testing.assert_allclose(observed, expected)
        self.assertAlmostEqual(sum(observed), 0.9)
        self.assertTrue(all(value != 0.0 for value in observed))
        self.assertAlmostEqual(env.ori6_maximum_spread_potential, 0.9)

    def test_initially_near_target_episode_is_not_frontier_eligible(self):
        env = scripted_environment(
            [metrics(1.0, 1.92), metrics(1.0, 2.0)],
            ["running"],
        )
        _, _, _, info = take_scripted_step(env)
        self.assertFalse(env.ori6_spread_frontier_eligible)
        self.assertEqual(info["ori6_spread_progress_after"], 0.0)
        self.assertEqual(info["ori6_spread_frontier_reward"], 0.0)

    def test_flatness_gated_ablation_is_markov_and_configurable(self):
        a0 = scripted_environment(
            [metrics(2.0, 0.0), metrics(1.0, 1.0)],
            ["running"],
        )
        a1 = scripted_environment(
            [metrics(2.0, 0.0), metrics(1.0, 1.0)],
            ["running"],
            frontier_mode="flatness_gated_state_potential",
        )
        _, _, _, a0_info = take_scripted_step(a0)
        _, _, _, a1_info = take_scripted_step(a1)
        expected_gate = (2.0 - 1.0) / (2.0 - 0.001)
        self.assertAlmostEqual(a0_info["ori6_flatness_gate_after"], 1.0)
        self.assertAlmostEqual(
            a1_info["ori6_flatness_gate_after"], expected_gate
        )
        self.assertAlmostEqual(
            a1_info["ori6_spread_frontier_reward"],
            a0_info["ori6_spread_frontier_reward"] * expected_gate,
        )

    def test_crossing_rolls_back_frontier_and_joint_ledgers(self):
        env = scripted_environment(
            [
                metrics(1.0, 0.0),
                metrics(0.4, 2.0),
                metrics(0.0001, 2.0),
            ],
            ["running", "self_crossing"],
            valid=[True, False],
        )
        _, first_reward, _, first = take_scripted_step(env)
        _, second_reward, done, second = take_scripted_step(env)
        self.assertTrue(done)
        self.assertAlmostEqual(first["ori6_spread_frontier_reward"], 0.9)
        self.assertAlmostEqual(first["ori6_joint_milestone_reward"], 0.2)
        self.assertAlmostEqual(second["ori6_frontier_rollback_reward"], -0.9)
        self.assertAlmostEqual(
            second["ori6_joint_milestone_rollback_reward"], -0.2
        )
        self.assertEqual(
            second["ori6_quality_source"], "last_valid_before_crossing"
        )
        self.assertAlmostEqual(
            env.ori5_last_valid_metrics["flatness_ratio"], 0.4
        )
        self.assertAlmostEqual(
            first["ori6_spread_frontier_reward"]
            + first["ori6_joint_milestone_reward"]
            + second["ori6_frontier_rollback_reward"]
            + second["ori6_joint_milestone_rollback_reward"],
            0.0,
        )
        self.assertAlmostEqual(first_reward, component_sum(first), places=12)
        self.assertAlmostEqual(second_reward, component_sum(second), places=12)

    def test_wrong_flat_cancels_flatness_frontier_and_joint_credit(self):
        env = scripted_environment(
            [
                metrics(1.0, 0.0),
                metrics(0.4, 2.0),
                metrics(0.0005, 1.0),
            ],
            ["running", "wrong_flat"],
        )
        _, first_reward, _, first = take_scripted_step(env)
        _, second_reward, done, second = take_scripted_step(env)
        self.assertTrue(done)
        episode = first_reward + second_reward
        cancelled = sum(
            row[name]
            for row in (first, second)
            for name in (
                "ori6_flatness_progress_reward",
                "ori6_spread_frontier_reward",
                "ori6_joint_milestone_reward",
            )
        ) + second["ori6_terminal_correction_reward"]
        self.assertAlmostEqual(cancelled, 0.0, places=12)
        # Only target endpoint progress (4 -> 1) and -2 wrong-flat remain.
        self.assertAlmostEqual(episode, 1.0, places=12)

    def test_timeout_retains_frontier_and_joint_credit(self):
        env = scripted_environment(
            [metrics(1.0, 0.0), metrics(0.4, 2.0)],
            ["timeout"],
            max_steps=1,
        )
        _, reward, done, info = take_scripted_step(env)
        self.assertTrue(done)
        self.assertAlmostEqual(info["ori6_spread_frontier_reward"], 0.9)
        self.assertAlmostEqual(info["ori6_joint_milestone_reward"], 0.2)
        self.assertEqual(info["ori6_terminal_correction_reward"], 0.0)
        self.assertAlmostEqual(reward, component_sum(info), places=12)

    def test_no_modes_rolls_back_only_frontier_as_documented(self):
        env = scripted_environment(
            [metrics(1.0, 0.0), metrics(0.4, 2.0)],
            ["no_modes"],
        )
        _, reward, done, info = take_scripted_step(env)
        self.assertTrue(done)
        self.assertAlmostEqual(info["ori6_frontier_rollback_reward"], -0.9)
        self.assertEqual(info["ori6_joint_milestone_rollback_reward"], 0.0)
        self.assertAlmostEqual(reward, component_sum(info), places=12)

    def test_parser_schema_and_ori5_defaults_are_isolated(self):
        ori5_design_before = ori5.REWARD_DESIGN_NAME
        ori5_before = ori5.parse_args(["--episodes", "1"])
        args = ori6.parse_args([
            "--episodes", "1",
            "--ori5-spread-milestone-bonus", "17",
        ])
        self.assertAlmostEqual(ori5_before.ori5_spread_milestone_bonus, 0.9)
        self.assertEqual(args.ori5_spread_milestone_bonus, 0.0)
        self.assertEqual(
            args.ori6_spread_frontier_mode, "state_potential"
        )
        parameters = ori6.reward_parameters(args)
        frontier = parameters["spread_frontier"]
        self.assertEqual(frontier["maximum_bonus"], 0.9)
        self.assertFalse(frontier["is_running_best"])
        self.assertTrue(
            frontier["is_markov_in_existing_coordinate_observation"]
        )
        self.assertFalse(frontier["actor_critic_frontier_scalar_input_added"])
        self.assertEqual(
            parameters["rollback"]["self_crossing"],
            ["spread_frontier", "joint_milestones"],
        )
        # Importing and parsing ori6 must not change the standalone ori5 CLI.
        ori5_after = ori5.parse_args(["--episodes", "1"])
        self.assertAlmostEqual(ori5_after.ori5_spread_milestone_bonus, 0.9)
        self.assertEqual(ori5.REWARD_DESIGN_NAME, ori5_design_before)

    def test_training_wrapper_records_full_schema_and_restores_ori5(self):
        args = ori6.parse_args([
            "--episodes", "1",
            "--moving-average-window", "1",
            "--no-random-search",
        ])
        row = {
            "episode": 1,
            "ori6_total_spread_frontier_reward": 0.6,
            "ori6_total_frontier_rollback_reward": -0.2,
            "ori6_reward_decomposition_error": 0.0,
            "ori6_maximum_abs_step_reward_decomposition_error": 0.0,
            "ori6_nonfinite_reward_component_count": 0,
        }
        originals = (
            ori5.REWARD_DESIGN_NAME,
            ori5.REWARD_SCHEMA_VERSION,
            ori5.make_environment,
            ori5.run_episode,
            ori5.aggregate_stochastic_evaluations,
            ori5.reward_parameters,
            ori5.save_ori5_learning_curves,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def fake_train(selected_args):
                self.assertEqual(
                    ori5.REWARD_DESIGN_NAME, ori6.REWARD_DESIGN_NAME
                )
                self.assertIs(ori5.make_environment, ori6.make_environment)
                with open(
                    output / "training_history.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                with open(
                    output / "run_config.json", "w", encoding="utf-8"
                ) as file:
                    json.dump({}, file)
                return output, {}

            with mock.patch.object(ori5, "train", fake_train):
                returned, summary = ori6.train(args)

            self.assertEqual(Path(returned), output)
            with open(
                output / "run_config.json", encoding="utf-8"
            ) as file:
                config = json.load(file)
            self.assertEqual(config["reward_design"], ori6.REWARD_DESIGN_NAME)
            self.assertEqual(
                config["reward_schema_version"],
                ori6.REWARD_SCHEMA_VERSION,
            )
            frontier = config["ori6_reward_parameters"]["spread_frontier"]
            self.assertEqual(frontier["mode"], "state_potential")
            self.assertFalse(frontier["is_running_best"])
            self.assertFalse(
                frontier["actor_critic_frontier_scalar_input_added"]
            )
            self.assertEqual(
                config["ori6_reward_parameters"]["rollback"][
                    "self_crossing"
                ],
                ["spread_frontier", "joint_milestones"],
            )
            self.assertEqual(
                summary["ori6_nonfinite_reward_component_count"], 0
            )

        restored = (
            ori5.REWARD_DESIGN_NAME,
            ori5.REWARD_SCHEMA_VERSION,
            ori5.make_environment,
            ori5.run_episode,
            ori5.aggregate_stochastic_evaluations,
            ori5.reward_parameters,
            ori5.save_ori5_learning_curves,
        )
        for before, after in zip(originals, restored):
            self.assertIs(before, after)

    def test_untraced_initial_no_modes_is_in_terminal_fixed_ledger(self):
        class DummyEnvironment:
            no_mode_penalty = 5.0
            ori6_initial_spread_error = 0.4
            ori6_spread_frontier_eligible = True
            ori6_spread_frontier_mode = "state_potential"
            ori6_maximum_spread_progress = 0.0
            ori6_maximum_spread_potential = 0.0
            ori5_joint_milestones_awarded = [False, False, False]

        parent_result = (
            {
                "termination_reason": "no_modes",
                "total_reward": -5.0,
            },
            [np.zeros((1, 2), dtype=float)],
            [],
        )
        with mock.patch.object(
            ori6,
            "_ORIGINAL_ORI5_RUN_EPISODE",
            return_value=parent_result,
        ):
            summary, nodes, actions = ori6.run_episode(
                DummyEnvironment(), agent=None
            )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(actions, [])
        self.assertAlmostEqual(
            summary["ori6_total_terminal_fixed_reward"], -5.0
        )
        self.assertAlmostEqual(summary["ori6_reward_decomposition_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
