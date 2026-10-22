"""Focused mechanics, reward, trace, and hook tests for Codex7."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

import ori_codex2 as mechanics
import ori_codex6_spectral_subspace as spectral
import ori_codex6_spectral_subspace_3d as spectral3d
import ori_codex7 as ori7
import ori_codex7_3d as adapter
import test_ori_codex6_zero_reject as zero_tests


ROOT = Path(__file__).resolve().parent
WATER_ACTIONS = ROOT / (
    "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
    "final_training_path/actions.json"
)


def actions():
    with WATER_ACTIONS.open(encoding="utf-8") as handle:
        return json.load(handle)


def water_args(mode, max_steps=206):
    return adapter.parse_args([
        ori7.CLI_FLAG, mode,
        "--model", "waterbomb",
        "--target-spread", "1",
        "--max-steps", str(max_steps),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
        "--no-rigid-diagnostics",
    ])


def make_water(mode, max_steps=206):
    args = water_args(mode, max_steps=max_steps)
    node, member, fix, _ = adapter.load_problem_data(args.data_dir)
    return args, node, member, fix


def replay_to_crossing(mode, max_steps=206, forced_q=None):
    saved = actions()
    args, node, member, fix = make_water(mode, max_steps=max_steps)
    with adapter.patched_three_dimensional_runtime(args, member):
        env = adapter.make_environment(args, node, member, fix)
        env.reset()
        for row in saved[:-1]:
            _, _, done, info = env.step(
                np.asarray(row["physical_direction"], dtype=float),
                float(row["requested_width"]),
                np.asarray(row["mode_coefficients"], dtype=float),
            )
            if done:
                raise AssertionError(f"unexpected pre-crossing terminal: {info}")
        env.spectral_subspace_metrics()
        mechanics_before = zero_tests._mechanical_snapshot(env)
        ledgers_before = zero_tests._ledger_snapshot(env)
        if forced_q is not None:
            env.near_miss_quality = lambda *_: (
                float(forced_q), float(forced_q), float(forced_q)
            )
        row = saved[-1]
        state, reward, done, info = env.step(
            np.asarray(row["physical_direction"], dtype=float),
            float(row["requested_width"]),
            np.asarray(row["mode_coefficients"], dtype=float),
        )
    return env, state, reward, done, info, mechanics_before, ledgers_before


class OriCodex7Tests(unittest.TestCase):
    def test_crossing_priority_over_all_competing_terminal_flags(self):
        for competing in (
            "correction_failed", "success", "wrong_flat", "no_modes", "timeout"
        ):
            with self.subTest(competing=competing):
                self.assertTrue(ori7.crossing_terminal_selected({
                    "self_crossing": True,
                    competing: True,
                }))
        self.assertFalse(ori7.crossing_terminal_selected({
            "self_crossing": False,
            "timeout": True,
        }))

    def test_cli_formula_hashes_and_actor_critic_freeze(self):
        for mode in ori7.CROSSING_MODES:
            args = ori7.parse_args([
                ori7.CLI_FLAG, mode, "--episodes", "1"
            ])
            self.assertEqual(args.ori7_crossing_terminal_mode, mode)
            self.assertEqual(args.ori6_spectral_weight, 0.05)
            self.assertEqual(args.ori5_quality_weight, 1.0)
            self.assertEqual(args.ori5_crossing_penalty, 1.25)
            self.assertEqual(args.ori7_candidate, ori7.CANDIDATE_NAME)
            self.assertEqual(
                args.ori7_reward_formula_sha256, ori7.REWARD_FORMULA_SHA256
            )
            self.assertEqual(
                args.ori7_transition_formula_sha256,
                ori7.TRANSITION_FORMULA_SHA256,
            )
            self.assertEqual(
                args.ori7_reward_parameters["crossing_terminal_mode"], mode
            )
            self.assertEqual(args.update_method, "actor_critic")
            parameters = ori7.reward_parameters(args)
            self.assertEqual(parameters["reward_formula_sha256"], ori7.REWARD_FORMULA_SHA256)
            self.assertEqual(parameters["transition_formula_sha256"], ori7.TRANSITION_FORMULA_SHA256)
            self.assertEqual(
                parameters["update_method"],
                "episode_separated_monte_carlo_actor_critic",
            )
        with self.assertRaises(ValueError):
            ori7.parse_args([
                ori7.CLI_FLAG, ori7.MODE_NO_FIXED,
                "--ori6-spectral-weight", "0.1",
            ])
        with self.assertRaises(ValueError):
            ori7.parse_args([
                ori7.CLI_FLAG, ori7.MODE_NO_FIXED,
                "--ori5-quality-weight", "0.9",
            ])
        with self.assertRaises(ValueError):
            ori7.parse_args([
                ori7.CLI_FLAG, ori7.MODE_NO_FIXED,
                "--ori5-crossing-penalty", "2",
            ])
        with self.assertRaises(SystemExit):
            ori7.parse_args(["--episodes", "1"])

    def test_q_endpoints_modes_horizon_state_ledgers_and_components(self):
        for mode, fixed in (
            (ori7.MODE_NO_FIXED, 0.0),
            (ori7.MODE_HISTORICAL_1P25, -1.25),
        ):
            for q in (0.0, 1.0):
                with self.subTest(mode=mode, q=q):
                    env, _, reward, done, info, mechanics_before, ledgers_before = replay_to_crossing(
                        mode, max_steps=len(actions()), forced_q=q
                    )
                    self.assertTrue(done)
                    self.assertEqual(env.step_count, len(actions()))
                    self.assertEqual(info["termination_reason"], "self_crossing")
                    self.assertTrue(info["self_crossing"])
                    self.assertTrue(info["self_crossing_candidate"])
                    self.assertFalse(info["timeout"])
                    self.assertEqual(info["ori7_terminal_crossing"], 1)
                    self.assertEqual(info["ori6_rejected_crossing_timeout_boundary"], 0)
                    self.assertEqual(reward.hex(), float(fixed + q).hex())
                    self.assertEqual(info["ori7_crossing_fixed_reward"], fixed)
                    self.assertEqual(info["ori7_crossing_quality_reward"], q)
                    self.assertEqual(info["ori7_crossing_timeout_reward"], 0.0)
                    self.assertEqual(info["ori7_crossing_reward"].hex(), reward.hex())
                    self.assertEqual(info["ori7_crossing_reward_decomposition_error"], 0.0)
                    self.assertEqual(info["ori6_spectral_reward"], 0.0)
                    self.assertEqual(info["ori6_spectral_rollback_reward"], 0.0)
                    self.assertEqual(info["ori6_spectral_rollback_reason"], "none")
                    for field in (
                        "ori7_candidate_flatness_reward",
                        "ori7_candidate_target_progress_reward",
                        "ori7_candidate_spread_frontier_reward",
                        "ori7_candidate_joint_milestone_reward",
                        "ori7_candidate_spectral_reward",
                    ):
                        self.assertEqual(info[field], 0.0, field)
                    zero_tests._assert_nested_equal(
                        self, mechanics_before, zero_tests._mechanical_snapshot(env)
                    )
                    zero_tests._assert_nested_equal(
                        self, ledgers_before, zero_tests._ledger_snapshot(env)
                    )
                    with self.assertRaises(RuntimeError):
                        env.step(np.zeros_like(env.node), 0.0, np.zeros(1))

    def test_nonfinite_q_is_explicitly_counted(self):
        _, _, reward, done, info, _, _ = replay_to_crossing(
            ori7.MODE_NO_FIXED, forced_q=np.nan
        )
        self.assertTrue(done)
        self.assertTrue(np.isnan(reward))
        self.assertGreaterEqual(info["ori7_crossing_nonfinite_component_count"], 2)
        self.assertTrue(np.isnan(info["ori7_crossing_reward_decomposition_error"]))

    def test_sampled_action_and_log_probability_are_retained(self):
        saved = actions()
        args, node, member, fix = make_water(
            ori7.MODE_HISTORICAL_1P25, max_steps=len(saved)
        )
        agent = zero_tests._TraceAgent(saved)
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            summary, nodes, trace = adapter.run_episode(
                env, agent, deterministic=True, training=True
            )
        self.assertTrue(agent.finished)
        self.assertEqual(len(agent.transitions), len(saved))
        self.assertEqual(
            float(agent.transitions[-1].log_probability), 1000.0 + len(saved)
        )
        self.assertEqual(summary["termination_reason"], "self_crossing")
        self.assertEqual(summary["ori7_terminal_crossing_count"], 1)
        self.assertEqual(trace[-1]["ori7_terminal_crossing"], 1)
        self.assertEqual(trace[-1]["ori7_sampled_action_retained"], 1)
        np.testing.assert_array_equal(
            trace[-1]["physical_direction"], saved[-1]["physical_direction"]
        )
        np.testing.assert_array_equal(nodes[-1], nodes[-2])

    def test_first_32_accepted_steps_are_parent_bit_exact(self):
        saved = actions()[:32]
        parent_args = spectral3d.parse_args([
            spectral.CLI_FLAG,
            "--model", "waterbomb",
            "--target-spread", "1",
            "--max-steps", "400",
            "--ori6-spectral-weight", "0.05",
            "--episodes", "1",
            "--periodic-stochastic-eval-episodes", "0",
            "--no-random-search",
            "--no-rigid-diagnostics",
        ])
        node, member, fix, _ = spectral3d.load_problem_data(parent_args.data_dir)
        with spectral3d.patched_three_dimensional_runtime(parent_args, member):
            parent = spectral3d.make_environment(parent_args, node, member, fix)
            parent.reset()
            expected = []
            for row in saved:
                expected.append(parent.step(
                    np.asarray(row["physical_direction"], dtype=float),
                    float(row["requested_width"]),
                    np.asarray(row["mode_coefficients"], dtype=float),
                ))

        args, _, _, _ = make_water(ori7.MODE_NO_FIXED, max_steps=400)
        with adapter.patched_three_dimensional_runtime(args, member):
            candidate = adapter.make_environment(args, node, member, fix)
            candidate.reset()
            for row, (state, reward, done, parent_info) in zip(saved, expected):
                observed_state, observed_reward, observed_done, info = candidate.step(
                    np.asarray(row["physical_direction"], dtype=float),
                    float(row["requested_width"]),
                    np.asarray(row["mode_coefficients"], dtype=float),
                )
                self.assertEqual(observed_done, done)
                np.testing.assert_array_equal(observed_state, state)
                self.assertEqual(observed_reward.hex(), reward.hex())
                self.assertEqual(
                    info["ori6_spectral_reward"].hex(),
                    parent_info["ori6_spectral_reward"].hex(),
                )

    def test_planar_and_3d_hooks_restore(self):
        args3d, _, member, _ = make_water(ori7.MODE_NO_FIXED)
        originals = (
            mechanics.has_proper_self_crossing,
            ori7.make_environment,
            ori7.run_episode,
            ori7.aggregate_stochastic_evaluations,
        )
        with adapter.patched_three_dimensional_runtime(args3d, member):
            self.assertIsNot(mechanics.has_proper_self_crossing, originals[0])
            self.assertIs(ori7.make_environment, adapter.make_environment)
        self.assertEqual(
            (
                mechanics.has_proper_self_crossing,
                ori7.make_environment,
                ori7.run_episode,
                ori7.aggregate_stochastic_evaluations,
            ),
            originals,
        )

        spectral_originals = (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
            spectral.reward_parameters,
        )
        with ori7._patched_spectral_training_runtime():
            self.assertIs(spectral.make_environment, ori7.make_environment)
        self.assertEqual(
            (
                spectral.make_environment,
                spectral.run_episode,
                spectral.aggregate_stochastic_evaluations,
                spectral.reward_parameters,
            ),
            spectral_originals,
        )


if __name__ == "__main__":
    unittest.main()
