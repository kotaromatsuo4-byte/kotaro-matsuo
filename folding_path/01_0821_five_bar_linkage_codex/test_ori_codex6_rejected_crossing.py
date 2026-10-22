"""Mechanics and ledger tests for ori6 rejected-crossing continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import collision_3d
import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex5 as ori5
import ori_codex6 as ori6
import ori_codex6_rejected_crossing as rejected
import ori_codex6_rejected_crossing_3d as adapter


ROOT = Path(__file__).resolve().parent
WATER_NEAR_CROSS = ROOT / (
    "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
    "final_training_path/actions.json"
)


def _saved_water_actions():
    with open(WATER_NEAR_CROSS, encoding="utf-8") as file:
        return json.load(file)


def _water_args(max_steps):
    return adapter.parse_args([
        "--model", "waterbomb",
        "--target-spread", "1",
        "--ori6-rejected-crossing-continuation",
        "--max-steps", str(int(max_steps)),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
    ])


def _replay_direct(max_steps=400):
    actions = _saved_water_actions()
    args = _water_args(max_steps)
    node, member, fix, _ = adapter.load_problem_data(args.data_dir)
    rewards = []
    infos = []
    states_before = []
    caches_before = []
    with adapter.patched_three_dimensional_runtime(args, member) as callback:
        env = adapter.make_environment(args, node, member, fix)
        env.reset()
        for action in actions:
            states_before.append(env.node.copy())
            caches_before.append(copy.deepcopy({
                name: getattr(env, name)
                for name in (
                    "_ori5_flatness_metrics_cache",
                    "_ori5_compactness_metrics_cache",
                    "_ori5_mode_cache_key",
                    "_ori5_mode_cache",
                )
            }))
            _, reward, done, info = env.step(
                np.asarray(action["physical_direction"], dtype=float),
                float(action["requested_width"]),
                np.asarray(action["mode_coefficients"], dtype=float),
            )
            rewards.append(float(reward))
            infos.append(info)
            if done:
                break
    return (
        env,
        callback,
        actions,
        rewards,
        infos,
        states_before,
        caches_before,
    )


class _SavedActionAgent:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def get_action(self, state, modes, deterministic=False):
        row = self.actions[self.index]
        self.index += 1
        zero = torch.zeros(())
        return mechanics.PolicyAction(
            physical_direction=np.asarray(
                row["physical_direction"], dtype=float
            ),
            mode_coefficients=np.asarray(
                row["mode_coefficients"], dtype=float
            ),
            requested_width=float(row["requested_width"]),
            log_probability=zero,
            value=zero,
            entropy=zero,
            mean_mixture_variance=0.0,
            width_mean=0.0,
            width_variance=0.0,
        )


class OriCodex6RejectedCrossingTests(unittest.TestCase):
    def test_cli_is_opt_in_p0x2_and_does_not_mutate_ori5(self):
        before = ori5.parse_args(["--episodes", "1"])
        ordinary = ori6.parse_args(["--episodes", "1"])
        candidate = rejected.parse_args([
            "--ori6-rejected-crossing-continuation",
            "--episodes", "1",
        ])
        explicit = rejected.parse_args([
            "--ori6-rejected-crossing-continuation",
            "--ori5-crossing-penalty", "2.5",
            "--episodes", "1",
        ])
        after = ori5.parse_args(["--episodes", "1"])

        self.assertFalse(hasattr(
            ordinary, "ori6_rejected_crossing_continuation"
        ))
        self.assertAlmostEqual(ordinary.ori5_crossing_penalty, 1.25)
        self.assertTrue(candidate.ori6_rejected_crossing_continuation)
        self.assertEqual(
            candidate.ori6_crossing_policy,
            rejected.CROSSING_POLICY,
        )
        self.assertAlmostEqual(candidate.ori6_spread_frontier_bonus, 0.9)
        self.assertAlmostEqual(candidate.ori5_crossing_penalty, 2.0)
        self.assertAlmostEqual(explicit.ori5_crossing_penalty, 2.5)
        self.assertAlmostEqual(before.ori5_crossing_penalty, 1.25)
        self.assertAlmostEqual(after.ori5_crossing_penalty, 1.25)

        ordinary_parameters = ori6.reward_parameters(ordinary)
        candidate_parameters = rejected.reward_parameters(candidate)
        self.assertEqual(
            ordinary_parameters["rollback"]["self_crossing"],
            ["spread_frontier", "joint_milestones"],
        )
        self.assertEqual(
            candidate_parameters["rollback"]["self_crossing"], []
        )
        self.assertTrue(candidate_parameters["crossing_transition"][
            "past_frontier_and_joint_ledgers_retained"
        ])

    def test_saved_path_counterfactual_rejects_and_repeat_is_costly(self):
        (
            env,
            callback,
            actions,
            rewards,
            infos,
            states_before,
            caches_before,
        ) = _replay_direct(max_steps=400)
        final = infos[-1]
        safe = states_before[-1]

        self.assertEqual(len(infos), len(actions))
        self.assertEqual(
            callback.checker.pair_indices.shape[0],
            collision_3d.nonadjacent_member_pairs(env.member).shape[0],
        )
        self.assertEqual(final["ori6_rejected_crossing"], 1)
        self.assertEqual(final["termination_reason"], "running")
        self.assertFalse(final["self_crossing"])
        self.assertFalse(final["timeout"])
        self.assertTrue(np.array_equal(env.node, safe))
        cache_before = caches_before[-1]
        self.assertIsNot(
            env._ori5_flatness_metrics_cache,
            cache_before["_ori5_flatness_metrics_cache"],
        )
        self.assertEqual(
            env._ori5_flatness_metrics_cache,
            cache_before["_ori5_flatness_metrics_cache"],
        )
        self.assertIsNot(
            env._ori5_compactness_metrics_cache,
            cache_before["_ori5_compactness_metrics_cache"],
        )
        self.assertEqual(
            env._ori5_compactness_metrics_cache,
            cache_before["_ori5_compactness_metrics_cache"],
        )
        self.assertEqual(
            env._ori5_mode_cache_key,
            cache_before["_ori5_mode_cache_key"],
        )
        np.testing.assert_array_equal(
            env._ori5_mode_cache,
            cache_before["_ori5_mode_cache"],
        )
        if env._ori5_mode_cache is not None:
            self.assertIsNot(
                env._ori5_mode_cache,
                cache_before["_ori5_mode_cache"],
            )
        np.testing.assert_allclose(env.get_state(), (
            safe / env.coordinate_scale
        ).flatten().astype(np.float32))
        self.assertNotEqual(
            final["ori6_rejected_candidate_accepted_width"], 0.0
        )
        self.assertEqual(final["accepted_width"], 0.0)
        self.assertEqual(final["actual_displacement"], 0.0)
        self.assertAlmostEqual(
            final["ori6_rejected_candidate_base_reward"], 0.0
        )
        self.assertAlmostEqual(
            final["ori6_rejected_candidate_spread_frontier_reward"], 0.0
        )
        self.assertAlmostEqual(
            final["ori6_rejected_candidate_joint_milestone_reward"], 0.0
        )
        self.assertAlmostEqual(final["ori6_frontier_rollback_reward"], 0.0)
        self.assertAlmostEqual(
            final["ori6_joint_milestone_rollback_reward"], 0.0
        )
        self.assertAlmostEqual(
            env.ori6_cumulative_spread_frontier_reward, 0.9, places=12
        )
        self.assertAlmostEqual(
            env.ori6_cumulative_joint_milestone_reward, 0.2, places=12
        )
        self.assertAlmostEqual(
            float(sum(rewards)), 0.0235206299776327, places=11
        )
        self.assertAlmostEqual(
            final["ori6_reward_decomposition_error"], 0.0, places=12
        )

        # The state was restored, so issuing the identical unsafe command is
        # rejected again and cannot cheaply advance the path.
        last_action = actions[-1]
        args = _water_args(max_steps=400)
        _, member, _, _ = adapter.load_problem_data(args.data_dir)
        with adapter.patched_three_dimensional_runtime(args, member):
            _, repeat_reward, repeat_done, repeat = env.step(
                np.asarray(last_action["physical_direction"], dtype=float),
                float(last_action["requested_width"]),
                np.asarray(last_action["mode_coefficients"], dtype=float),
            )
            self.assertFalse(repeat_done)
            self.assertEqual(repeat["ori6_rejected_crossing"], 1)
            self.assertEqual(
                repeat["ori6_consecutive_rejected_crossings"], 2
            )
            self.assertEqual(env.ori6_rejected_crossing_streak_count, 1)
            self.assertAlmostEqual(repeat_reward, rewards[-1], places=12)
            self.assertTrue(np.array_equal(env.node, safe))

            # A valid accepted move immediately after that reject streak is
            # one escape, counted per streak rather than per rejected attempt.
            direction = -np.asarray(
                last_action["physical_direction"], dtype=float
            )
            modes = env.get_modes()
            coefficients = modes.T @ direction.reshape(-1)
            _, _, escape_done, escape = env.step(
                direction, -0.05, coefficients
            )
        self.assertFalse(escape_done)
        self.assertEqual(escape["ori6_rejected_crossing"], 0)
        self.assertEqual(escape["ori6_escape_after_rejected_crossing"], 1)
        self.assertEqual(env.ori6_escape_after_rejected_crossing_count, 1)

    def test_crossing_on_horizon_is_timeout_with_q_only_once(self):
        (
            env,
            _,
            actions,
            rewards,
            infos,
            states_before,
            _,
        ) = _replay_direct(max_steps=206)
        terminal = infos[-1]
        q = float(terminal["ori6_q"])

        self.assertEqual(len(infos), len(actions))
        self.assertEqual(env.step_count, 206)
        self.assertTrue(np.array_equal(env.node, states_before[-1]))
        self.assertEqual(terminal["ori6_rejected_crossing"], 1)
        self.assertEqual(terminal["ori6_rejected_crossing_timeout_boundary"], 1)
        self.assertEqual(terminal["termination_reason"], "timeout")
        self.assertTrue(terminal["timeout"])
        self.assertFalse(terminal["self_crossing"])
        self.assertAlmostEqual(rewards[-1], -3.0 + q, places=12)
        self.assertAlmostEqual(
            terminal["ori6_terminal_fixed_reward"], -3.0, places=12
        )
        self.assertAlmostEqual(
            terminal["ori6_terminal_quality_reward"], q, places=12
        )
        self.assertAlmostEqual(
            terminal["ori6_rejected_crossing_timeout_reward"],
            -1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(sum(rewards)), -0.9764793700223673, places=11
        )

    def test_rollout_summary_and_action_trace_report_rejection(self):
        actions = _saved_water_actions()
        args = _water_args(max_steps=206)
        node, member, fix, _ = adapter.load_problem_data(args.data_dir)
        agent = _SavedActionAgent(actions)
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            summary, nodes, saved_actions = adapter.run_episode(
                env, agent, deterministic=True, training=False
            )

        self.assertEqual(summary["termination_reason"], "timeout")
        self.assertEqual(summary["self_crossing"], 0)
        self.assertEqual(summary["ori6_rejected_crossing_count"], 1)
        self.assertEqual(
            summary["ori6_rejected_crossing_streak_count"], 1
        )
        self.assertEqual(
            summary["ori6_unescaped_rejected_crossing_streak_count"], 1
        )
        self.assertEqual(
            summary["ori6_escape_after_rejected_crossing_count"], 0
        )
        self.assertAlmostEqual(
            summary["ori6_rejected_crossing_rate"], 1.0 / 206.0
        )
        self.assertAlmostEqual(
            summary["ori6_reward_decomposition_error"], 0.0, places=10
        )
        self.assertEqual(len(nodes), len(saved_actions) + 1)
        self.assertTrue(np.array_equal(nodes[-1], nodes[-2]))
        self.assertEqual(saved_actions[-1]["ori6_rejected_crossing"], 1)
        self.assertEqual(saved_actions[-1]["after_state_valid"], 0)

    def test_planar_and_3d_training_hooks_do_not_recurse_and_restore(self):
        ori6_originals = (
            ori6.make_environment,
            ori6.run_episode,
            ori6.aggregate_stochastic_evaluations,
            ori6.reward_parameters,
            ori6.save_ori6_learning_curves,
        )
        with tempfile.TemporaryDirectory() as directory:
            planar_output = Path(directory) / "planar"
            args = rejected.parse_args([
                "--data-dir", str(ROOT / "model/five_bar_linkage"),
                "--target-spread", "2",
                "--ori6-rejected-crossing-continuation",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--max-steps", "1",
                "--batch-size", "1",
                "--moving-average-window", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--no-random-search",
                "--output-dir", str(planar_output),
            ])
            output, summary = rejected.train(args)
            self.assertEqual(Path(output), planar_output)
            self.assertEqual(
                summary["ori6_crossing_policy"], rejected.CROSSING_POLICY
            )
            self.assertTrue((
                planar_output / "ori6_rejected_crossing_learning_curve.png"
            ).exists())

            spatial_output = Path(directory) / "spatial"
            spatial_args = adapter.parse_args([
                "--model", "waterbomb",
                "--target-spread", "1",
                "--ori6-rejected-crossing-continuation",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--max-steps", "1",
                "--batch-size", "1",
                "--moving-average-window", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--no-random-search",
                "--output-dir", str(spatial_output),
            ])
            output, summary = adapter.train(spatial_args)
            self.assertEqual(Path(output), spatial_output)
            self.assertEqual(
                summary["three_dimensional_adapter"]["adapter"],
                adapter.ADAPTER_NAME,
            )

        self.assertEqual(
            (
                ori6.make_environment,
                ori6.run_episode,
                ori6.aggregate_stochastic_evaluations,
                ori6.reward_parameters,
                ori6.save_ori6_learning_curves,
            ),
            ori6_originals,
        )

    def test_all_five_task_configurations_smoke(self):
        two_dimensional = (
            ("five_spread2", 2.0),
            ("five_spread3", 3.0),
        )
        for name, target in two_dimensional:
            with self.subTest(task=name):
                args = rejected.parse_args([
                    "--data-dir", str(ROOT / "model/five_bar_linkage"),
                    "--target-spread", str(target),
                    "--ori6-rejected-crossing-continuation",
                    "--max-steps", "1",
                    "--episodes", "1",
                    "--no-random-search",
                ])
                node, member, fix = ori3._load_problem_data(args)
                env = rejected.make_environment(args, node, member, fix)
                agent = ori3.UniformRandomAgent(
                    args.max_step_width, np.random.default_rng(11)
                )
                summary, nodes, actions = rejected.run_episode(env, agent)
                self.assertIsInstance(
                    env,
                    rejected.RejectedCrossingContinuationTargetFoldingEnv,
                )
                self.assertEqual(
                    summary["ori6_crossing_policy"],
                    rejected.CROSSING_POLICY,
                )
                self.assertEqual(len(nodes), len(actions) + 1)

        three_dimensional = (
            ("torus_spread2", "4unit_torus", 2.0),
            ("water_spread1", "waterbomb", 1.0),
            ("water_spread_sqrt2", "waterbomb", np.sqrt(2.0)),
        )
        for name, model, target in three_dimensional:
            with self.subTest(task=name):
                args = adapter.parse_args([
                    "--model", model,
                    "--target-spread", str(float(target)),
                    "--ori6-rejected-crossing-continuation",
                    "--max-steps", "1",
                    "--episodes", "1",
                    "--periodic-stochastic-eval-episodes", "0",
                    "--no-random-search",
                ])
                node, member, fix, _ = adapter.load_problem_data(
                    args.data_dir
                )
                agent = ori3.UniformRandomAgent(
                    args.max_step_width, np.random.default_rng(13)
                )
                with adapter.patched_three_dimensional_runtime(args, member):
                    env = adapter.make_environment(args, node, member, fix)
                    summary, nodes, actions = adapter.run_episode(
                        env, agent
                    )
                self.assertIsInstance(
                    env,
                    rejected.RejectedCrossingContinuationTargetFoldingEnv,
                )
                self.assertEqual(
                    summary["ori6_crossing_policy"],
                    rejected.CROSSING_POLICY,
                )
                self.assertEqual(len(nodes), len(actions) + 1)


if __name__ == "__main__":
    unittest.main()
