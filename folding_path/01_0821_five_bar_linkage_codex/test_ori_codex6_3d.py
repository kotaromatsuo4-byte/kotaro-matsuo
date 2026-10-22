"""Focused smoke tests for the ori_codex6 three-dimensional adapter."""

from __future__ import annotations

import unittest

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex6 as ori6
import ori_codex6_3d as adapter


class OriCodex6ThreeDimensionalTests(unittest.TestCase):
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

    def test_model_and_reward_defaults(self):
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
                self.assertEqual(
                    args.ori6_spread_frontier_mode, "state_potential"
                )
                self.assertAlmostEqual(args.ori6_spread_frontier_bonus, 0.9)
                self.assertEqual(args.ori5_spread_milestone_bonus, 0.0)

    def test_short_rollout_uses_3d_collision_and_ori6_reward(self):
        args = self._args("4unit_torus")
        node, member, fix, _ = adapter.load_problem_data(args.data_dir)
        agent = ori3.UniformRandomAgent(
            args.max_step_width, np.random.default_rng(123)
        )
        original_collision = mechanics.has_proper_self_crossing
        with adapter.patched_three_dimensional_runtime(args, member) as callback:
            self.assertIs(mechanics.has_proper_self_crossing, callback)
            self.assertEqual(
                callback.checker.pair_indices.shape[0],
                collision_3d.nonadjacent_member_pairs(member).shape[0],
            )
            environment = adapter.make_environment(args, node, member, fix)
            self.assertIsInstance(environment, ori6.CommonRewardTargetFoldingEnv)
            summary, nodes, actions = adapter.run_episode(
                environment,
                agent,
                deterministic=False,
                training=False,
            )
        self.assertIs(mechanics.has_proper_self_crossing, original_collision)
        self.assertEqual(summary["reward_design"], ori6.REWARD_DESIGN_NAME)
        self.assertLessEqual(
            abs(summary["ori6_reward_decomposition_error"]), 1.0e-10
        )
        self.assertEqual(summary["ori6_nonfinite_reward_component_count"], 0)
        self.assertEqual(len(nodes), len(actions) + 1)
        self.assertTrue(actions)
        self.assertIn("ori6_spread_frontier_reward", actions[-1])
        self.assertIn("rigid_diagnostic_valid_transition", actions[-1])

    def test_runtime_hooks_restore_after_exception(self):
        args = self._args("4unit_torus")
        _, member, _, _ = adapter.load_problem_data(args.data_dir)
        originals = (
            mechanics.has_proper_self_crossing,
            ori6.make_environment,
            ori6.run_episode,
            ori6.aggregate_stochastic_evaluations,
        )
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with adapter.patched_three_dimensional_runtime(args, member):
                raise RuntimeError("deliberate test exception")
        self.assertIs(mechanics.has_proper_self_crossing, originals[0])
        self.assertIs(ori6.make_environment, originals[1])
        self.assertIs(ori6.run_episode, originals[2])
        self.assertIs(ori6.aggregate_stochastic_evaluations, originals[3])


if __name__ == "__main__":
    unittest.main()
