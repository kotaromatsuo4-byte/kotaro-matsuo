"""Exact-equivalence and boundary tests for ori5 geometry memoisation."""

from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex5 as ori5
import ori_codex5_3d as ori5_3d


_DIRECTORY = Path(ori5.__file__).resolve().parent


class _UncachedOri5Environment(ori5.CommonRewardTargetFoldingEnv):
    """The pre-optimisation implementations used as an exact oracle."""

    def flatness_metrics(self, node=None):
        return mechanics.FixedStartFoldingEnv.flatness_metrics(self, node)

    def get_modes(self):
        return mechanics.FixedStartFoldingEnv.get_modes(self)

    def compactness_metrics(self, node=None):
        metrics = ori3.CompactTargetFoldingEnv.compactness_metrics(self, node)
        flatness_clip = float(getattr(
            self, "ori5_flatness_clip", self.flatness_reward_clip
        ))
        metrics["combined_quality"] = float(-(
            np.clip(metrics["flatness_ratio"], 0.0, flatness_clip)
            + metrics["target_deviation_energy"]
        ))
        return metrics


def _five_data():
    node = np.atleast_2d(np.loadtxt(
        _DIRECTORY / "node_initial.dat", delimiter=","
    ))
    member = np.atleast_2d(np.loadtxt(
        _DIRECTORY / "member.dat", delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(
        _DIRECTORY / "fix.dat", delimiter=",", dtype=int
    ))
    return node, member, fix


def _five_args(max_steps=8):
    return ori5.parse_args([
        "--episodes", "1",
        "--max-steps", str(max_steps),
        "--max-step-width", "0.1",
        "--flatness-subspace-dimension", "1",
        "--target-spread", "2",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
    ])


def _assert_nested_exact(testcase, left, right, path="root"):
    if isinstance(left, dict):
        testcase.assertIsInstance(right, dict, path)
        testcase.assertEqual(set(left), set(right), path)
        for key in left:
            _assert_nested_exact(
                testcase, left[key], right[key], f"{path}.{key}"
            )
        return
    if isinstance(left, (list, tuple)):
        testcase.assertIsInstance(right, type(left), path)
        testcase.assertEqual(len(left), len(right), path)
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_nested_exact(
                testcase, first, second, f"{path}[{index}]"
            )
        return
    if isinstance(left, np.ndarray):
        testcase.assertTrue(
            np.array_equal(left, right, equal_nan=True), path
        )
        return
    if isinstance(left, (float, np.floating)) and math.isnan(float(left)):
        testcase.assertTrue(math.isnan(float(right)), path)
        return
    testcase.assertEqual(left, right, path)


class OriCodex5CacheTests(unittest.TestCase):
    def test_cached_and_uncached_planar_rollouts_are_bit_exact(self):
        args = _five_args(max_steps=12)
        node, member, fix = _five_data()
        cached = ori5.make_environment(args, node, member, fix)
        with mock.patch.object(
            ori5, "CommonRewardTargetFoldingEnv", _UncachedOri5Environment
        ):
            uncached = ori5.make_environment(args, node, member, fix)

        reference_agent = ori3.UniformRandomAgent(
            args.max_step_width, np.random.default_rng(731)
        )
        cached_agent = ori3.UniformRandomAgent(
            args.max_step_width, np.random.default_rng(731)
        )
        reference = ori5.run_episode(
            uncached, reference_agent, deterministic=False, training=False
        )
        actual = ori5.run_episode(
            cached, cached_agent, deterministic=False, training=False
        )
        _assert_nested_exact(self, reference, actual)

    def test_cached_and_uncached_3d_rollouts_are_bit_exact(self):
        for model in ("4unit_torus", "waterbomb"):
            with self.subTest(model=model):
                args = ori5_3d.parse_args([
                    "--model", model,
                    "--episodes", "1",
                    "--max-steps", "4",
                    "--periodic-stochastic-eval-episodes", "0",
                    "--no-random-search",
                    "--no-rigid-diagnostics",
                ])
                node, member, fix, _ = ori5_3d.load_problem_data(
                    args.data_dir
                )
                with ori5_3d.patched_three_dimensional_runtime(args, member):
                    with mock.patch.object(
                        ori5,
                        "CommonRewardTargetFoldingEnv",
                        _UncachedOri5Environment,
                    ):
                        uncached = ori5_3d.make_environment(
                            args, node, member, fix
                        )
                    cached = ori5_3d.make_environment(
                        args, node, member, fix
                    )
                    reference = ori5_3d.run_episode(
                        uncached,
                        ori3.UniformRandomAgent(
                            args.max_step_width, np.random.default_rng(811)
                        ),
                        deterministic=False,
                        training=False,
                    )
                    actual = ori5_3d.run_episode(
                        cached,
                        ori3.UniformRandomAgent(
                            args.max_step_width, np.random.default_rng(811)
                        ),
                        deterministic=False,
                        training=False,
                    )
                _assert_nested_exact(self, reference, actual)

    def test_cache_returns_defensive_values_and_reset_starts_new_generation(self):
        args = _five_args()
        node, member, fix = _five_data()
        environment = ori5.make_environment(args, node, member, fix)
        environment.reset()

        expected = environment.compactness_metrics()
        exposed = environment.compactness_metrics()
        exposed["spread"] = -12345.0
        self.assertEqual(environment.compactness_metrics(), expected)
        self.assertEqual(len(environment._ori5_compactness_metrics_cache), 1)

        moved = environment.node.copy()
        moved[0, 0] += 1.0e-6
        environment.compactness_metrics(moved)
        self.assertEqual(len(environment._ori5_compactness_metrics_cache), 2)

        real_mode = mechanics.mode
        with mock.patch.object(mechanics, "mode", wraps=real_mode) as solver:
            first = environment.get_modes()
            self.assertEqual(solver.call_count, 1)
            untouched = first.copy()
            first.flat[0] += 10.0
            np.testing.assert_array_equal(environment.get_modes(), untouched)
            self.assertEqual(solver.call_count, 1)
            environment.reset()
            environment.get_modes()
            self.assertEqual(solver.call_count, 2)

        self.assertEqual(len(environment._ori5_compactness_metrics_cache), 1)
        self.assertNotIn(
            environment._ori5_geometry_cache_key(moved),
            environment._ori5_compactness_metrics_cache,
        )

    def test_crossed_and_correction_failed_candidates_do_not_survive_reset(self):
        args = _five_args()
        node, member, fix = _five_data()

        crossed = ori5.make_environment(args, node, member, fix)
        crossed.reset()
        initial_metrics = dict(crossed.ori5_last_valid_metrics)
        modes = crossed.get_modes()
        coefficients = np.zeros(modes.shape[1])
        coefficients[0] = 1.0
        with mock.patch.object(
            mechanics, "has_proper_self_crossing", return_value=True
        ):
            _, _, done, info = crossed.step(
                modes[:, 0], 0.01, coefficients
            )
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "self_crossing")
        self.assertEqual(crossed.ori5_last_valid_metrics, initial_metrics)
        crossed_key = crossed._ori5_geometry_cache_key(crossed.node)
        self.assertIn(crossed_key, crossed._ori5_compactness_metrics_cache)
        crossed.reset()
        self.assertNotIn(crossed_key, crossed._ori5_compactness_metrics_cache)
        self.assertEqual(len(crossed._ori5_compactness_metrics_cache), 1)

        failed = ori5.make_environment(args, node, member, fix)
        failed.reset()
        initial_metrics = dict(failed.ori5_last_valid_metrics)
        modes = failed.get_modes()
        with mock.patch.object(
            failed,
            "_project_candidate",
            return_value=(
                failed.node.copy(), 0.0, True, 1,
                float("inf"), float("inf"), float("inf"),
            ),
        ):
            _, _, done, info = failed.step(
                modes[:, 0], 0.01, np.ones(modes.shape[1])
            )
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "correction_failed")
        self.assertEqual(failed.ori5_last_valid_metrics, initial_metrics)
        failed.reset()
        self.assertEqual(len(failed._ori5_compactness_metrics_cache), 1)

    def test_step_internal_no_modes_cache_is_replaced_on_reset(self):
        args = _five_args()
        node, member, fix = _five_data()
        environment = ori5.make_environment(args, node, member, fix)
        environment.reset()
        real_mode = mechanics.mode
        call_count = 0

        def scripted_mode(*mode_args, **mode_kwargs):
            nonlocal call_count
            call_count += 1
            result = real_mode(*mode_args, **mode_kwargs)
            if call_count == 2:
                return np.zeros((result.shape[0], 0))
            return result

        with mock.patch.object(mechanics, "mode", side_effect=scripted_mode):
            modes = environment.get_modes()
            self.assertGreater(modes.shape[1], 0)
            with mock.patch.object(
                mechanics, "has_proper_self_crossing", return_value=False
            ):
                _, _, done, info = environment.step(
                    modes[:, 0], 0.01, np.ones(modes.shape[1])
                )
            self.assertTrue(done)
            self.assertEqual(info["termination_reason"], "no_modes")
            self.assertEqual(environment.get_modes().shape[1], 0)
            self.assertEqual(call_count, 2)

            environment.reset()
            self.assertGreater(environment.get_modes().shape[1], 0)
            self.assertEqual(call_count, 3)


if __name__ == "__main__":
    unittest.main()
