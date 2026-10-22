"""Focused smoke tests for the ori_codex5 three-dimensional adapter."""

from __future__ import annotations

import unittest

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex5 as ori5
import ori_codex5_3d as adapter


class OriCodex5ThreeDimensionalTests(unittest.TestCase):
    def _args(self, model, *extra):
        return adapter.parse_args([
            "--model", model,
            "--episodes", "1",
            "--eval-episodes", "1",
            "--max-steps", "2",
            "--periodic-stochastic-eval-episodes", "0",
            "--no-random-search",
            *extra,
        ])

    def test_model_and_common_reward_defaults(self):
        expected = {
            "4unit_torus": (0.2, 2.0, True),
            "waterbomb": (0.3, 1.0, False),
        }
        for model, (width, target, rigid) in expected.items():
            with self.subTest(model=model):
                args = self._args(model)
                self.assertEqual(args.flatness_subspace_dimension, 2)
                self.assertAlmostEqual(args.max_step_width, width, places=6)
                self.assertAlmostEqual(args.target_spread, target)
                self.assertEqual(args.rigid_motion_diagnostics, rigid)
                self.assertAlmostEqual(args.ori5_crossing_penalty, 1.25)
                self.assertAlmostEqual(
                    args.ori5_success_speed_bonus_weight, 0.5
                )

    def test_ori5_cli_and_core_values_override_adapter_defaults(self):
        args = self._args(
            "waterbomb",
            "--max-step-width", "0.125",
            "--target-spread", "1.4142135623730951",
            "--ori5-crossing-penalty", "1.75",
            "--output-dir", "explicit_ori5_test_output",
            "--rigid-diagnostics",
        )
        self.assertAlmostEqual(args.max_step_width, 0.125)
        self.assertAlmostEqual(args.target_spread, np.sqrt(2.0))
        self.assertAlmostEqual(args.ori5_crossing_penalty, 1.75)
        self.assertEqual(args.output_dir, "explicit_ori5_test_output")
        self.assertTrue(args.rigid_motion_diagnostics)
        self.assertEqual(args.step_width_rule, "user_supplied")

    def test_short_rollouts_use_3d_collision_and_ori5_reward(self):
        original_collision = mechanics.has_proper_self_crossing
        for model in ("4unit_torus", "waterbomb"):
            with self.subTest(model=model):
                args = self._args(model)
                node, member, fix, _ = adapter.load_problem_data(args.data_dir)
                agent = ori3.UniformRandomAgent(
                    args.max_step_width, np.random.default_rng(123)
                )
                with adapter.patched_three_dimensional_runtime(
                    args, member
                ) as callback:
                    self.assertIs(mechanics.has_proper_self_crossing, callback)
                    self.assertEqual(
                        callback.checker.pair_indices.shape[0],
                        collision_3d.nonadjacent_member_pairs(member).shape[0],
                    )
                    environment = adapter.make_environment(
                        args, node, member, fix
                    )
                    self.assertIsInstance(
                        environment, ori5.CommonRewardTargetFoldingEnv
                    )
                    summary, nodes, actions = adapter.run_episode(
                        environment,
                        agent,
                        deterministic=False,
                        training=False,
                    )
                self.assertIs(
                    mechanics.has_proper_self_crossing, original_collision
                )
                self.assertEqual(
                    summary["reward_design"], ori5.REWARD_DESIGN_NAME
                )
                self.assertLessEqual(
                    abs(summary["ori5_reward_decomposition_error"]), 1.0e-10
                )
                self.assertEqual(
                    summary["ori5_nonfinite_reward_component_count"], 0
                )
                self.assertEqual(len(nodes), len(actions) + 1)
                self.assertTrue(actions)
                self.assertIn("ori5_base_reward", actions[-1])
                if args.rigid_motion_diagnostics:
                    self.assertIn(
                        "rigid_diagnostic_valid_transition", actions[-1]
                    )
                else:
                    self.assertEqual(
                        summary["rigid_motion_diagnostics_enabled"], 0
                    )

    def test_runtime_hooks_restore_after_an_exception(self):
        args = self._args("4unit_torus")
        _, member, _, _ = adapter.load_problem_data(args.data_dir)
        originals = (
            mechanics.has_proper_self_crossing,
            ori5.make_environment,
            ori5.run_episode,
            ori5.aggregate_stochastic_evaluations,
        )
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with adapter.patched_three_dimensional_runtime(args, member):
                raise RuntimeError("deliberate test exception")
        self.assertIs(mechanics.has_proper_self_crossing, originals[0])
        self.assertIs(ori5.make_environment, originals[1])
        self.assertIs(ori5.run_episode, originals[2])
        self.assertIs(ori5.aggregate_stochastic_evaluations, originals[3])


if __name__ == "__main__":
    unittest.main()
