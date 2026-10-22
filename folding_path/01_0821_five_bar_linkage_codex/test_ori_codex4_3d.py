"""Focused smoke tests for the ori_codex4 three-dimensional adapter."""

from __future__ import annotations

import unittest

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex4 as ori4
import ori_codex4_3d as adapter


class OriCodex4ThreeDimensionalTests(unittest.TestCase):
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

    def test_model_defaults_and_support_compatible_rigid_ranks(self):
        expected = {
            "4unit_torus": (0.2, 2.0, True, 3),
            "waterbomb": (0.3, 1.0, False, 0),
        }
        for model, (width, target, enabled, rank) in expected.items():
            with self.subTest(model=model):
                args = self._args(model)
                node, member, fix, _ = adapter.load_problem_data(args.data_dir)
                self.assertEqual(args.flatness_subspace_dimension, 2)
                self.assertAlmostEqual(args.max_step_width, width, places=6)
                self.assertAlmostEqual(args.target_spread, target)
                self.assertEqual(args.rigid_motion_diagnostics, enabled)
                self.assertEqual(
                    adapter.support_compatible_rigid_basis(node, fix).shape[1],
                    rank,
                )
                self.assertEqual(member.shape[1], 2)

    def test_core_cli_values_override_adapter_defaults(self):
        args = self._args(
            "waterbomb",
            "--max-step-width", "0.125",
            "--target-spread", "1.4142135623730951",
            "--output-dir", "explicit_test_output",
            "--rigid-diagnostics",
        )
        self.assertAlmostEqual(args.max_step_width, 0.125)
        self.assertAlmostEqual(args.target_spread, np.sqrt(2.0))
        self.assertEqual(args.output_dir, "explicit_test_output")
        self.assertTrue(args.rigid_motion_diagnostics)
        self.assertEqual(args.step_width_rule, "user_supplied")

    def test_short_rollouts_use_3d_checker_and_log_without_reward_hooks(self):
        original_collision = mechanics.has_proper_self_crossing
        for model in ("4unit_torus", "waterbomb"):
            with self.subTest(model=model):
                args = self._args(model)
                node, member, fix, _ = adapter.load_problem_data(args.data_dir)
                agent = ori3.UniformRandomAgent(
                    args.max_step_width, np.random.default_rng(123)
                )
                with adapter.patched_three_dimensional_runtime(args, member) as callback:
                    self.assertIs(mechanics.has_proper_self_crossing, callback)
                    self.assertEqual(
                        callback.checker.pair_indices.shape[0],
                        collision_3d.nonadjacent_member_pairs(member).shape[0],
                    )
                    environment = adapter.make_environment(args, node, member, fix)
                    summary, nodes, actions = adapter.run_episode(
                        environment, agent, deterministic=False, training=False
                    )
                self.assertIs(mechanics.has_proper_self_crossing, original_collision)
                self.assertEqual(summary["steps"], len(actions))
                self.assertEqual(len(nodes), len(actions) + 1)
                if model == "4unit_torus":
                    self.assertEqual(
                        summary["maximum_support_compatible_rigid_rank"], 3
                    )
                    self.assertIn(
                        "commanded_rigid_energy_fraction", actions[0]
                    )
                    self.assertGreaterEqual(
                        actions[0]["commanded_rigid_energy_fraction"], 0.0
                    )
                    self.assertLessEqual(
                        actions[0]["commanded_rigid_energy_fraction"], 1.0
                    )
                else:
                    self.assertEqual(
                        summary["rigid_motion_diagnostics_enabled"], 0
                    )

    def test_all_runtime_hooks_restore_when_training_raises(self):
        args = self._args("4unit_torus")
        originals = (
            mechanics.has_proper_self_crossing,
            ori4.make_environment,
            ori4.run_episode,
            ori4.aggregate_stochastic_evaluations,
            ori4.train,
        )

        def deliberate_failure(_):
            raise RuntimeError("deliberate test failure")

        ori4.train = deliberate_failure
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                adapter.train(args)
        finally:
            ori4.train = originals[4]
        self.assertIs(mechanics.has_proper_self_crossing, originals[0])
        self.assertIs(ori4.make_environment, originals[1])
        self.assertIs(ori4.run_episode, originals[2])
        self.assertIs(ori4.aggregate_stochastic_evaluations, originals[3])


if __name__ == "__main__":
    unittest.main()
