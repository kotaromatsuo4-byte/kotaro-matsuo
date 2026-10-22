"""Focused contract tests for the water-D=1 spread-alpha candidate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import ori_codex6_zero_reject as zero_reject
import ori_codex8 as ori8
import ori_codex9_spread_alpha as ori9


def _args(alpha=2.0, *, target=1.0, arm="scale", coefficient=0.025):
    return ori9.parse_args([
        ori9.CLI_FLAG,
        ori9.ALPHA_FLAG, str(float(alpha)),
        ori8.CLI_FLAG,
        ori8.ARM_FLAG, arm,
        ori8.LAMBDA_FLAG, str(float(coefficient)),
        "--data-dir", str(Path(__file__).resolve().parent / "model" / "waterbomb"),
        "--target-spread", str(float(target)),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
    ])


def _mock_environment(script, alpha):
    environment = object.__new__(ori9.SpreadAlphaTargetFoldingEnv)
    environment.ori9_spread_alpha = float(alpha)
    environment.ori5_reward_trace = []
    environment.mock_script = copy.deepcopy(script)
    environment._reset_ori9_episode_state()
    return environment


def _mock_parent_step(environment, *_):
    row = environment.mock_script.pop(0)
    reason = str(row.get("reason", "running"))
    valid = int(row.get("valid", True))
    info = {
        "termination_reason": reason,
        "after_state_valid": valid,
        "ori6_target_progress_reward": float(row.get("target", 0.0)),
        "ori6_spread_frontier_reward": float(row.get("frontier", 0.0)),
        "ori6_frontier_rollback_reward": float(row.get("rollback", 0.0)),
        # Unaffected parent fields are sentinels for the non-mutation check.
        "ori6_flatness_progress_reward": float(row.get("flatness", -0.17)),
        "ori6_joint_milestone_reward": float(row.get("joint", 0.3)),
        "ori6_spectral_reward": float(row.get("spectral", 0.07)),
        "ori6_terminal_fixed_reward": float(row.get("terminal", 0.0)),
        "ori6_rejected_crossing": int(row.get("rejected", False)),
    }
    environment.ori5_reward_trace.append({
        "termination_reason": reason,
        "after_state_valid": valid,
        **info,
    })
    return (
        None,
        float(row.get("core", 0.75)),
        reason != "running",
        info,
    )


class OriCodex9SpreadAlphaTests(unittest.TestCase):
    def test_formula_hash_cli_scope_and_fail_closed_validation(self):
        expected_hash = hashlib.sha256(json.dumps(
            ori9.REWARD_FORMULA,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(ori9.REWARD_FORMULA_SHA256, expected_hash)
        for alpha in ori9.ALPHAS:
            with self.subTest(alpha=alpha):
                args = _args(alpha)
                self.assertEqual(args.ori9_spread_alpha, alpha)
                self.assertEqual(args.ori8_spectral_arm, "scale")
                self.assertEqual(args.ori6_spectral_weight, 0.25)
                self.assertEqual(args.update_method, "actor_critic")
                parameters = ori9.reward_parameters(args)
                self.assertEqual(parameters["spread_alpha"], alpha)
                self.assertEqual(
                    parameters["reward_formula_sha256"], expected_hash
                )

        bad = (
            (1.0, 1.0, "scale", 0.025),
            (4.0, 1.0, "scale", 0.025),
            (2.0, np.sqrt(2.0), "scale", 0.025),
            (2.0, 1.0, "legacy", 0.05),
            (2.0, 1.0, "scale", 0.05),
        )
        for alpha, target, arm, coefficient in bad:
            with self.subTest(
                alpha=alpha, target=target, arm=arm,
                coefficient=coefficient,
            ):
                with self.assertRaises((ValueError, SystemExit)):
                    _args(alpha, target=target, arm=arm,
                          coefficient=coefficient)

    def test_alpha_one_formula_is_literal_identity_but_not_an_experiment_arm(self):
        components = ori9.alpha_increment_components(1.0, 0.2, -0.3, 0.1)
        self.assertEqual(components, {
            "target": 0.0,
            "frontier": -0.0,
            "frontier_rollback": 0.0,
            "total": 0.0,
        })
        with self.assertRaises((ValueError, SystemExit)):
            _args(1.0)

    def test_alpha_two_and_three_scale_only_the_three_bound_components(self):
        script = [{
            "target": 0.1,
            "frontier": 0.2,
            "rollback": -0.05,
            "core": 0.75,
            "flatness": -0.17,
            "joint": 0.3,
            "spectral": 0.07,
            "terminal": -1.0,
        }]
        for alpha, expected_increment in ((2.0, 0.25), (3.0, 0.5)):
            environment = _mock_environment(script, alpha)
            with mock.patch.object(
                zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
                "step", _mock_parent_step,
            ):
                _, reward, _, info = environment.step(None, 0.1, [])
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(reward, 0.75 + expected_increment)
                self.assertAlmostEqual(info["ori9_reward"], expected_increment)
                self.assertEqual(info["ori6_flatness_progress_reward"], -0.17)
                self.assertEqual(info["ori6_joint_milestone_reward"], 0.3)
                self.assertEqual(info["ori6_spectral_reward"], 0.07)
                self.assertEqual(info["ori6_terminal_fixed_reward"], -1.0)
                self.assertAlmostEqual(
                    info["ori9_reward_decomposition_error"], 0.0,
                    places=15,
                )
                self.assertEqual(
                    environment.ori5_reward_trace[-1]["ori9_reward"],
                    info["ori9_reward"],
                )

    def test_frontier_rollback_is_scaled_and_target_credit_is_retained(self):
        for reason in ("wrong_flat", "correction_failed", "no_modes"):
            script = [
                {
                    "target": 0.10,
                    "frontier": 0.40,
                    "rollback": 0.0,
                    "core": 0.8,
                },
                {
                    "target": 0.02 if reason == "wrong_flat" else 0.0,
                    "frontier": 0.10 if reason == "wrong_flat" else 0.0,
                    "rollback": -0.50 if reason == "wrong_flat" else -0.40,
                    "core": -1.2,
                    "reason": reason,
                    "valid": reason == "wrong_flat",
                },
            ]
            environment = _mock_environment(script, 3.0)
            with mock.patch.object(
                zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
                "step", _mock_parent_step,
            ):
                environment.step(None, 0.1, [])
                _, _, done, terminal = environment.step(None, 0.1, [])
            with self.subTest(reason=reason):
                self.assertTrue(done)
                # The parent terminal rollback cancels its complete frontier
                # prefix.  The alpha ledger must cancel by the same factor.
                expected_frontier = 0.0
                self.assertAlmostEqual(
                    environment.ori9_cumulative_frontier_increment_reward
                    + environment.ori9_cumulative_frontier_rollback_increment_reward,
                    expected_frontier,
                )
                expected_target = 0.24 if reason == "wrong_flat" else 0.20
                self.assertAlmostEqual(
                    environment.ori9_cumulative_target_increment_reward,
                    expected_target,
                )
                expected_rollback = -1.0 if reason == "wrong_flat" else -0.8
                self.assertAlmostEqual(
                    terminal["ori9_frontier_rollback_increment_reward"],
                    expected_rollback,
                )

    def test_zero_reject_ordinary_and_horizon_rewards_remain_bit_exact(self):
        script = [
            {
                "target": 0.0, "frontier": 0.0, "rollback": 0.0,
                "core": 0.0, "rejected": True,
            },
            {
                "target": 0.0, "frontier": 0.0, "rollback": 0.0,
                "core": -0.73, "rejected": True, "reason": "timeout",
                "valid": False,
            },
        ]
        environment = _mock_environment(script, 3.0)
        with mock.patch.object(
            zero_reject.ZeroRejectRewardSpectralTargetFoldingEnv,
            "step", _mock_parent_step,
        ):
            _, ordinary_reward, ordinary_done, ordinary = environment.step(
                None, 0.1, []
            )
            _, horizon_reward, horizon_done, horizon = environment.step(
                None, 0.1, []
            )
        self.assertFalse(ordinary_done)
        self.assertTrue(horizon_done)
        self.assertEqual(ordinary_reward.hex(), 0.0.hex())
        self.assertEqual(horizon_reward.hex(), (-0.73).hex())
        self.assertEqual(ordinary["ori9_reward"], 0.0)
        self.assertEqual(horizon["ori9_reward"], 0.0)

    def test_episode_accounting_corrects_parent_shells_once(self):
        environment = SimpleNamespace(
            ori9_spread_alpha=2.0,
            ori9_cumulative_reward=0.5,
        )
        actions = [
            {
                "reward": 1.3,
                "ori9_core_reward": 1.0,
                "ori9_reward": 0.3,
                "ori9_target_increment_reward": 0.1,
                "ori9_frontier_increment_reward": 0.2,
                "ori9_frontier_rollback_increment_reward": 0.0,
                "ori9_reward_decomposition_error": 0.0,
                "ori9_nonfinite_component_count": 0,
            },
            {
                "reward": 0.2,
                "ori9_core_reward": 0.0,
                "ori9_reward": 0.2,
                "ori9_target_increment_reward": 0.1,
                "ori9_frontier_increment_reward": 0.2,
                "ori9_frontier_rollback_increment_reward": -0.1,
                "ori9_reward_decomposition_error": 0.0,
                "ori9_nonfinite_component_count": 0,
            },
        ]
        parent_summary = {
            "total_reward": 1.5,
            "ori6_reward_decomposition_error": 0.5,
            "ori5_reward_decomposition_error": 0.5,
            "ori6_spectral_reward_decomposition_error": 0.5,
        }
        with mock.patch.object(
            ori9,
            "_BASE_RUN_EPISODE",
            return_value=(parent_summary, [np.zeros((1, 3))], actions),
        ):
            summary, _, returned_actions = ori9.run_episode(
                environment, object()
            )
        self.assertIs(returned_actions, actions)
        self.assertEqual(summary["ori9_total_reward"], 0.5)
        self.assertEqual(summary["ori9_total_target_increment_reward"], 0.2)
        self.assertEqual(summary["ori9_total_frontier_increment_reward"], 0.4)
        self.assertEqual(
            summary["ori9_total_frontier_rollback_increment_reward"], -0.1
        )
        self.assertAlmostEqual(summary["ori9_reward_decomposition_error"], 0.0)
        self.assertAlmostEqual(summary["ori6_reward_decomposition_error"], 0.0)
        self.assertAlmostEqual(summary["ori5_reward_decomposition_error"], 0.0)
        self.assertAlmostEqual(
            summary["ori6_spectral_reward_decomposition_error"], 0.0
        )

    def test_patch_context_restores_every_parent_hook(self):
        original = (
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
            ori8.reward_parameters,
        )
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with ori9.patched_ori8_training_runtime():
                self.assertIs(ori8.make_environment, ori9.make_environment)
                raise RuntimeError("deliberate")
        self.assertEqual(original, (
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
            ori8.reward_parameters,
        ))


if __name__ == "__main__":
    unittest.main()
