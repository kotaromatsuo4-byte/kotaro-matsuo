"""Focused contract tests for the additive Codex8 reward ablation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex6_spectral_subspace as spectral
import ori_codex6_zero_reject as zero_reject
import ori_codex6_zero_reject_3d as zero3d
import ori_codex8 as ori8
import ori_codex8_3d as ori8_3d
import test_ori_codex6_rejected_crossing as rejected_tests
import test_ori_codex6_spectral_subspace as spectral_tests
import test_ori_codex6_zero_reject as zero_tests


ROOT = Path(__file__).resolve().parent


def _args(arm="phase_v2", coefficient=1.0, max_steps=206):
    return ori8_3d.parse_args([
        ori8.CLI_FLAG,
        ori8.ARM_FLAG, arm,
        ori8.LAMBDA_FLAG, str(float(coefficient)),
        "--model", "waterbomb",
        "--target-spread", "1",
        "--max-steps", str(int(max_steps)),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
        "--no-rigid-diagnostics",
    ])


def _problem(args):
    return (*zero3d.load_problem_data(args.data_dir)[:3],)


def _phase_snapshot(env):
    return copy.deepcopy({
        name: getattr(env, name)
        for name in (
            "ori8_close_latched",
            "ori8_z_latched",
            "ori8_current_psi",
            "ori8_cumulative_reward",
            "ori8_accepted_transition_count",
            "ori8_noncommit_count",
            "ori8_close_event_count",
            "ori8_turn_event_count",
            "ori8_rollback_count",
        )
    })


def _mock_phase(script, coefficient=1.0):
    env = object.__new__(ori8.Codex8SpectralTargetFoldingEnv)
    env.ori6_spectral_weight = 0.0
    env.ori8_spectral_arm = "phase_v2"
    env.ori8_lambda = float(coefficient)
    env.ori8_scale_multiplier = 10.0
    env.ori8_close_rhobar_threshold = 0.1
    env.ori8_turn_delta_threshold = 0.7
    env.ori8_rho0 = 1.0
    env.ori8_close_latched = False
    env.ori8_z_latched = False
    env.ori8_current_psi = 0.0
    env.ori8_initial_psi = 0.0
    env.ori8_cumulative_reward = 0.0
    env.ori8_accepted_transition_count = 0
    env.ori8_noncommit_count = 0
    env.ori8_close_event_count = 0
    env.ori8_turn_event_count = 0
    env.ori8_rollback_count = 0
    env.ori5_reward_trace = []
    env.mock_script = list(script)
    return env


def _mock_zero_step(env, *_):
    row = env.mock_script.pop(0)
    env.ori5_reward_trace.append({})
    reason = str(row.get("reason", "running"))
    valid = bool(row.get("valid", True))
    rejected = bool(row.get("rejected", False))
    before = row["before"]
    after = row.get("after", before)
    return None, float(row.get("core", 0.75)), reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(valid),
        "accepted_width": 0.1 if valid and not rejected else 0.0,
        "ori6_rejected_crossing": int(rejected),
        "ori6_spectral_weight": 0.0,
        "ori6_spectral_reward": 0.0,
        "ori6_spectral_rho_before": float(before[0]),
        "ori6_spectral_delta_before": float(before[1]),
        "ori6_spectral_rho_after": float(after[0]),
        "ori6_spectral_delta_after": float(after[1]),
    }


class OriCodex8Tests(unittest.TestCase):
    def test_cli_hashes_effective_weights_and_fail_closed(self):
        cases = (
            ("legacy", 0.05, 0.05),
            ("scale", 0.05, 0.5),
            ("scale", 0.025, 0.25),
            ("phase_v2", 1.0, 0.0),
            ("phase_v2", 0.5, 0.0),
        )
        expected_hash = hashlib.sha256(json.dumps(
            ori8.REWARD_FORMULA,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(ori8.REWARD_FORMULA_SHA256, expected_hash)
        for arm, coefficient, effective in cases:
            with self.subTest(arm=arm, coefficient=coefficient):
                args = _args(arm, coefficient)
                self.assertEqual(args.ori6_spectral_weight, effective)
                self.assertEqual(
                    args.ori8_effective_parent_spectral_weight, effective
                )
                self.assertEqual(args.update_method, "actor_critic")
                self.assertEqual(
                    args.ori8_reward_formula_sha256, expected_hash
                )
                parameters = ori8.reward_parameters(args)
                self.assertEqual(
                    parameters["parent_spectral_weight_effective"],
                    effective,
                )
        with self.assertRaises((ValueError, SystemExit)):
            ori8.parse_args([
                ori8.CLI_FLAG, ori8.ARM_FLAG, "legacy",
                ori8.LAMBDA_FLAG, ".05",
                "--ori6-spectral-weight", ".2",
            ])
        with self.assertRaises(ValueError):
            ori8.parse_args([
                ori8.CLI_FLAG, ori8.ARM_FLAG, "scale",
                ori8.LAMBDA_FLAG, ".05",
                "--ori8-scale-multiplier", "9",
            ])

    def test_legacy_and_scale_are_literal_codex6_parent(self):
        for arm, coefficient, effective in (
            ("legacy", 0.05, 0.05),
            ("scale", 0.05, 0.5),
            ("scale", 0.025, 0.25),
        ):
            with self.subTest(arm=arm, coefficient=coefficient):
                args = _args(arm, coefficient, max_steps=8)
                node, member, fix = _problem(args)
                with ori8_3d.patched_three_dimensional_runtime(args, member):
                    env = ori8_3d.make_environment(
                        args, node, member, fix
                    )
                self.assertIs(
                    type(env),
                    zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
                )
                self.assertFalse(hasattr(env, "ori8_spectral_arm"))
                self.assertEqual(env.ori6_spectral_weight, effective)
                self.assertEqual(
                    spectral.SPECTRAL_ROLLBACK_REASONS,
                    frozenset({
                        "wrong_flat", "correction_failed", "no_modes",
                        "self_crossing",
                    }),
                )

    def test_legacy_horizon_episode_is_bit_exact_to_codex6(self):
        actions = zero_tests._actions()
        args = _args("legacy", 0.05, max_steps=len(actions))
        node, member, fix = _problem(args)
        parent_args = zero_tests._water_args(
            max_steps=len(actions), weight=0.05
        )
        with zero3d.patched_three_dimensional_runtime(parent_args, member):
            parent = zero3d.make_environment(
                parent_args, node, member, fix
            )
            parent_result = zero3d.run_episode(
                parent,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        with ori8_3d.patched_three_dimensional_runtime(args, member):
            candidate = ori8_3d.make_environment(
                args, node, member, fix
            )
            candidate_result = ori8_3d.run_episode(
                candidate,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        parent_summary, parent_nodes, parent_trace = parent_result
        summary, nodes, trace = candidate_result
        for key, value in parent_summary.items():
            self.assertEqual(value, summary[key], key)
        self.assertEqual(len(parent_nodes), len(nodes))
        for left, right in zip(parent_nodes, nodes):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            [row["reward"].hex() for row in parent_trace],
            [row["reward"].hex() for row in trace],
        )
        self.assertEqual(trace[-1]["termination_reason"], "timeout")
        self.assertEqual(trace[-1]["ori6_rejected_crossing"], 1)

    def test_legacy_wrong_flat_and_correction_steps_are_parent_exact(self):
        rng = np.random.default_rng(8028)
        first = rng.normal(size=(6, 2)) @ np.diag([2.0, 0.3])
        second = rng.normal(size=(6, 2)) @ np.diag([2.5, 0.1])
        for terminal, valid in (
            ("wrong_flat", True),
            ("correction_failed", False),
        ):
            script = [
                {"node": first, "reason": "running"},
                {
                    "node": second if valid else None,
                    "reason": terminal,
                    "valid": valid,
                },
            ]
            parent = spectral_tests._mock_spectral_environment(
                copy.deepcopy(script), weight=0.05
            )
            candidate = spectral_tests._mock_spectral_environment(
                copy.deepcopy(script), weight=0.05
            )
            with mock.patch.object(
                spectral_tests.rejected.RejectedCrossingContinuationTargetFoldingEnv,
                "step", spectral_tests._mock_parent_step,
            ):
                parent_rows = [parent.step(None, 0.1, []) for _ in script]
                candidate_rows = [
                    candidate.step(None, 0.1, []) for _ in script
                ]
            with self.subTest(terminal=terminal):
                for left, right in zip(parent_rows, candidate_rows):
                    self.assertEqual(left[1].hex(), right[1].hex())
                    zero_tests._assert_nested_equal(self, left[3], right[3])
                self.assertEqual(
                    candidate_rows[-1][3]["ori6_spectral_rollback_reason"],
                    terminal,
                )
                self.assertEqual(
                    candidate.ori6_spectral_cumulative_reward, 0.0
                )

    def test_phase_ordered_events_and_potential(self):
        env = _mock_phase([
            {"before": (1.0, 0.0), "after": (0.05, 0.8)},
            {"before": (0.05, 0.8), "after": (0.2, 0.8)},
        ])
        with mock.patch.object(
            zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
            "step", _mock_zero_step,
        ):
            _, reward1, _, first = env.step(None, 0.1, [])
            _, reward2, _, second = env.step(None, 0.1, [])
        self.assertEqual(first["ori8_close_event"], 1)
        self.assertEqual(first["ori8_turn_event"], 0)
        self.assertEqual(first["ori8_z_after"], 0)
        self.assertAlmostEqual(first["ori8_psi_after"], 0.095, places=15)
        self.assertAlmostEqual(reward1, 0.75 + 0.095, places=15)
        self.assertEqual(second["ori8_close_before"], 1)
        self.assertEqual(second["ori8_turn_event"], 1)
        self.assertEqual(second["ori8_z_after"], 1)
        self.assertAlmostEqual(second["ori8_psi_after"], 0.164, places=15)
        self.assertAlmostEqual(reward2, 0.75 + 0.069, places=15)

    def test_phase_terminal_rollback_and_timeout_retention(self):
        for terminal, valid in (
            ("wrong_flat", True),
            ("no_modes", True),
            ("correction_failed", False),
            ("self_crossing", False),
        ):
            env = _mock_phase([
                {"before": (1.0, 0.0), "after": (0.5, 0.2)},
                {
                    "before": (0.5, 0.2), "after": (0.4, 0.3),
                    "reason": terminal, "valid": valid,
                    "rejected": terminal == "self_crossing",
                },
            ])
            with mock.patch.object(
                zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
                "step", _mock_zero_step,
            ):
                env.step(None, 0.1, [])
                _, _, _, info = env.step(None, 0.1, [])
            with self.subTest(terminal=terminal):
                self.assertEqual(info["ori8_rollback_reason"], terminal)
                self.assertEqual(env.ori8_cumulative_reward, 0.0)
                self.assertEqual(env.ori8_rollback_count, 1)
                self.assertAlmostEqual(
                    info["ori8_progress_reward"]
                    + info["ori8_rollback_reward"],
                    -0.05,
                    places=15,
                )

        env = _mock_phase([
            {"before": (1.0, 0.0), "after": (0.5, 0.2)},
            {
                "before": (0.5, 0.2), "after": (0.4, 0.3),
                "reason": "timeout",
            },
        ])
        with mock.patch.object(
            zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
            "step", _mock_zero_step,
        ):
            env.step(None, 0.1, [])
            env.step(None, 0.1, [])
        self.assertEqual(env.ori8_rollback_count, 0)
        self.assertNotEqual(env.ori8_cumulative_reward, 0.0)

    def test_phase_crossing_noncommit_and_horizon_prefix_retention(self):
        env = _mock_phase([
            {"before": (1.0, 0.0), "after": (0.5, 0.2)},
            {
                "before": (0.5, 0.2), "after": (0.01, 0.95),
                "rejected": True, "valid": False, "core": 0.0,
            },
            {
                "before": (0.5, 0.2), "after": (0.01, 0.95),
                "rejected": True, "valid": False, "reason": "timeout",
                "core": -0.73,
            },
        ])
        with mock.patch.object(
            zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
            "step", _mock_zero_step,
        ):
            env.step(None, 0.1, [])
            prefix = _phase_snapshot(env)
            _, ordinary_reward, _, ordinary = env.step(None, 0.1, [])
            after_ordinary = _phase_snapshot(env)
            _, horizon_reward, done, horizon = env.step(None, 0.1, [])
        self.assertEqual(ordinary_reward.hex(), 0.0.hex())
        self.assertEqual(horizon_reward.hex(), (-0.73).hex())
        self.assertTrue(done)
        for key in (
            "ori8_close_latched", "ori8_z_latched", "ori8_current_psi",
            "ori8_cumulative_reward", "ori8_accepted_transition_count",
            "ori8_close_event_count", "ori8_turn_event_count",
            "ori8_rollback_count",
        ):
            self.assertEqual(prefix[key], after_ordinary[key], key)
            self.assertEqual(prefix[key], _phase_snapshot(env)[key], key)
        self.assertEqual(ordinary["ori8_reward"], 0.0)
        self.assertEqual(horizon["ori8_reward"], 0.0)
        self.assertEqual(horizon["ori8_rollback_reason"], "none")

    def test_lambda_zero_parent_exact_observation_and_hook_restoration(self):
        args = _args("phase_v2", 0.0, max_steps=32)
        node, member, fix = _problem(args)
        original_hooks = (
            mechanics.has_proper_self_crossing,
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
        )
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with ori8_3d.patched_three_dimensional_runtime(args, member):
                raise RuntimeError("deliberate")
        self.assertEqual(original_hooks, (
            mechanics.has_proper_self_crossing,
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
        ))
        with ori8_3d.patched_three_dimensional_runtime(args, member):
            env = ori8_3d.make_environment(args, node, member, fix)
            observation = env.reset().copy()
            self.assertFalse(hasattr(env, "ori8_spectral_arm"))
            self.assertEqual(env.ori6_spectral_weight, 0.0)
        self.assertEqual(
            observation.shape,
            (node.size,),
        )
        np.testing.assert_array_equal(
            observation,
            (node / env.coordinate_scale).flatten().astype(np.float32),
        )
        self.assertIn("partially observable", ori8.POMDP_CAVEAT)

    def test_lambda_zero_episode_is_bit_exact_parent(self):
        actions = zero_tests._actions()[:32]
        args = _args("phase_v2", 0.0, max_steps=len(actions))
        node, member, fix = _problem(args)
        parent_args = zero_tests._water_args(
            max_steps=len(actions), weight=0.0
        )
        with zero3d.patched_three_dimensional_runtime(parent_args, member):
            parent = zero3d.make_environment(
                parent_args, node, member, fix
            )
            expected = zero3d.run_episode(
                parent,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        with ori8_3d.patched_three_dimensional_runtime(args, member):
            candidate = ori8_3d.make_environment(
                args, node, member, fix
            )
            observed = ori8_3d.run_episode(
                candidate,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        expected_summary, expected_nodes, expected_actions = expected
        summary, nodes, trace = observed
        zero_tests._assert_nested_equal(
            self, expected_summary, {
                key: summary[key] for key in expected_summary
            }
        )
        for left, right in zip(expected_nodes, nodes):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            [row["reward"].hex() for row in expected_actions],
            [row["reward"].hex() for row in trace],
        )
        self.assertFalse(any(
            key.startswith("ori8_") for key in trace[-1]
        ))

    def test_planar_and_spatial_factories_smoke(self):
        planar_args = ori8.parse_args([
            ori8.CLI_FLAG, ori8.ARM_FLAG, "phase_v2",
            ori8.LAMBDA_FLAG, "1", "--episodes", "1",
        ])
        node, member, fix = ori3._load_problem_data(planar_args)
        planar = ori8.make_environment(planar_args, node, member, fix)
        self.assertIsInstance(planar, ori8.Codex8SpectralTargetFoldingEnv)
        spatial_args = _args("phase_v2", 0.5, max_steps=4)
        node3, member3, fix3 = _problem(spatial_args)
        with ori8_3d.patched_three_dimensional_runtime(
            spatial_args, member3
        ):
            spatial = ori8_3d.make_environment(
                spatial_args, node3, member3, fix3
            )
        self.assertEqual(spatial.node.shape[1], 3)
        self.assertEqual(spatial.flatness_subspace_dimension, 2)


if __name__ == "__main__":
    unittest.main()
