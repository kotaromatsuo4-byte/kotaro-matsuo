"""Focused tests for the additive ori6 spectral-subspace candidate."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex6_rejected_crossing as rejected
import ori_codex6_rejected_crossing_3d as rejected3d
import ori_codex6_spectral_subspace as spectral
import ori_codex6_spectral_subspace_3d as adapter
import test_ori_codex6_rejected_crossing as rejected_tests


ROOT = Path(__file__).resolve().parent


def _spectral_water_args(max_steps=206, weight=0.05):
    return adapter.parse_args([
        "--ori6-spectral-subspace",
        "--model", "waterbomb",
        "--target-spread", "1",
        "--max-steps", str(int(max_steps)),
        "--ori6-spectral-weight", str(float(weight)),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
        "--no-rigid-diagnostics",
    ])


def _numeric_steps(path):
    files = sorted(
        Path(path).glob("step*.dat"),
        key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1)),
    )
    return [np.loadtxt(file) for file in files]


def _spectral_peak(path, k, reference=None):
    nodes = _numeric_steps(path)
    if not nodes:
        raise AssertionError(f"saved path contains no states: {path}")
    if reference is None:
        reference = spectral.spectral_subspace_geometry(
            nodes[0], k
        )["projector"]
    metrics = [
        spectral.spectral_subspace_geometry(
            node,
            k,
            reference_projector=reference,
        )
        for node in nodes
    ]
    return max(row["potential"] for row in metrics), metrics, nodes[0]


def _assert_cache_equal(test, left, right):
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertEqual(set(left[key]), set(right[key]))
        for name, value in left[key].items():
            other = right[key][name]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
                test.assertIsNot(value, other)
            else:
                test.assertEqual(value, other)


def _mock_spectral_environment(script, weight=0.05):
    rng = np.random.default_rng(1)
    initial = rng.normal(size=(6, 2)) @ np.diag([3.0, 0.2])
    env = object.__new__(spectral.SpectralSubspaceTargetFoldingEnv)
    env.node = initial
    env.flatness_subspace_dimension = 1
    env.ori5_reward_trace = []
    env.mock_script = list(script)
    env._enable_spectral_subspace(weight, 1e-12)
    return env


def _mock_parent_step(env, *_):
    row = env.mock_script.pop(0)
    if row.get("node") is not None:
        env.node = np.array(row["node"], copy=True)
    reason = row["reason"]
    valid = bool(row.get("valid", True))
    rejected_crossing = bool(row.get("rejected", False))
    env.ori5_reward_trace.append({})
    return None, float(row.get("core_reward", 0.75)), reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(valid),
        "accepted_width": 0.1 if valid and not rejected_crossing else 0.0,
        "ori6_rejected_crossing": int(rejected_crossing),
    }


class OriCodex6SpectralSubspaceTests(unittest.TestCase):
    def test_cli_formula_hash_and_parent_defaults(self):
        parent_before = rejected.parse_args([
            rejected.CLI_FLAG,
            "--episodes", "1",
        ])
        args = spectral.parse_args([
            spectral.CLI_FLAG,
            "--episodes", "1",
        ])
        parent_after = rejected.parse_args([
            rejected.CLI_FLAG,
            "--episodes", "1",
        ])

        self.assertTrue(args.ori6_spectral_subspace)
        self.assertTrue(args.ori6_rejected_crossing_continuation)
        self.assertAlmostEqual(args.ori6_spectral_weight, 0.05)
        self.assertAlmostEqual(args.ori6_spectral_relative_epsilon, 1e-12)
        self.assertAlmostEqual(args.ori5_crossing_penalty, 2.0)
        self.assertAlmostEqual(args.ori6_spread_frontier_bonus, 0.9)
        self.assertEqual(
            parent_before.ori5_crossing_penalty,
            parent_after.ori5_crossing_penalty,
        )
        expected_hash = hashlib.sha256(json.dumps(
            spectral.SPECTRAL_FORMULA,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(spectral.SPECTRAL_FORMULA_SHA256, expected_hash)
        parameters = spectral.reward_parameters(args)
        self.assertEqual(parameters["candidate"], spectral.CANDIDATE_NAME)
        self.assertEqual(
            parameters["parent_candidate"], rejected.CANDIDATE_NAME
        )
        self.assertEqual(
            parameters["spectral_subspace"]["formula_sha256"],
            expected_hash,
        )

    def test_rigid_reflection_scale_sign_and_basis_invariance(self):
        rng = np.random.default_rng(20260826)
        initial = rng.normal(size=(9, 3)) @ np.diag([3.0, 1.5, 0.2])
        current = rng.normal(size=(9, 3)) @ np.diag([2.5, 1.1, 0.3])
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        reflection = rotation @ np.diag([1.0, 1.0, -1.0])
        translation = np.array([7.0, -11.0, 5.0])

        reference = spectral.spectral_subspace_geometry(initial, 2)[
            "projector"
        ]
        baseline = spectral.spectral_subspace_geometry(
            current,
            2,
            reference_projector=reference,
        )
        for transform in (rotation, reflection):
            for scale in (1e-12, 1e-6, 1.0, 1e6, 1e12):
                # Scale the complete coordinate system, including its origin;
                # retaining an O(1) translation beside O(1e-12) structure
                # would intentionally destroy the structure by cancellation.
                transformed_initial = scale * (
                    initial @ transform + translation
                )
                transformed_current = scale * (
                    current @ transform + translation
                )
                transformed_reference = spectral.spectral_subspace_geometry(
                    transformed_initial,
                    2,
                )["projector"]
                observed = spectral.spectral_subspace_geometry(
                    transformed_current,
                    2,
                    reference_projector=transformed_reference,
                )
                self.assertAlmostEqual(observed["rho"], baseline["rho"], 10)
                self.assertAlmostEqual(observed["delta"], baseline["delta"], 10)
                self.assertAlmostEqual(
                    observed["potential"], baseline["potential"], 10
                )

        translated_reference = spectral.spectral_subspace_geometry(
            initial + translation,
            2,
        )["projector"]
        translated = spectral.spectral_subspace_geometry(
            current + translation,
            2,
            reference_projector=translated_reference,
        )
        self.assertAlmostEqual(translated["rho"], baseline["rho"], 12)
        self.assertAlmostEqual(translated["delta"], baseline["delta"], 12)

        orthogonal, _ = np.linalg.qr(rng.normal(size=(12, 3)))
        within, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        changed = np.array(orthogonal, copy=True)
        changed[:, :2] = changed[:, :2] @ within
        changed[:, 0] *= -1.0
        np.testing.assert_allclose(
            spectral.subspace_projector(orthogonal, 2),
            spectral.subspace_projector(changed, 2),
            atol=2e-15,
            rtol=0.0,
        )

    def test_gap_degeneracy_reference_reset_and_defensive_cache(self):
        rng = np.random.default_rng(77)
        raw = rng.normal(size=(8, 3))
        raw -= np.mean(raw, axis=0, keepdims=True)
        left, _ = np.linalg.qr(raw)
        degenerate = left[:, :3] @ np.diag([3.0, 1.0, 1.0])
        alternate = np.eye(len(degenerate))[:, :2]
        observed = spectral.spectral_subspace_geometry(
            degenerate,
            2,
            reference_projector=alternate @ alternate.T,
        )
        self.assertEqual(observed["gap_degenerate"], 1)
        self.assertEqual(observed["rho"], 0.0)
        self.assertEqual(observed["potential"], 0.0)

        args = spectral.parse_args([
            spectral.CLI_FLAG,
            "--data-dir", str(ROOT / "model/five_bar_linkage"),
            "--episodes", "1",
            "--max-steps", "1",
            "--no-random-search",
        ])
        node, member, fix = ori3._load_problem_data(args)
        env = spectral.make_environment(args, node, member, fix)
        reference = env.ori6_spectral_reference_projector.copy()
        reference_hash = env.ori6_spectral_reference_projector_sha256
        first = env.spectral_subspace_metrics()
        first["projector"][0, 0] += 100.0
        second = env.spectral_subspace_metrics()
        self.assertNotEqual(first["projector"][0, 0], second["projector"][0, 0])
        env.reset()
        np.testing.assert_array_equal(
            env.ori6_spectral_reference_projector,
            reference,
        )
        self.assertEqual(
            env.ori6_spectral_reference_projector_sha256,
            reference_hash,
        )
        self.assertEqual(env.ori6_spectral_current_potential, 0.0)
        self.assertEqual(env.spectral_subspace_metrics()["potential"], 0.0)
        self.assertEqual(len(env._ori6_spectral_metrics_cache), 1)

    def test_saved_success_paths_outscore_failure_paths_for_all_five_tasks(self):
        pairs = {
            "five_d2": (
                ROOT / "experiments_ori5/full/five_d2/B/seed2101/first_success_path",
                ROOT / "experiments_ori5/full/five_d2/B/seed2101/best_policy_deterministic_path",
                1,
            ),
            "five_d3": (
                ROOT / "experiments_ori5/full/five_d3/B/seed2101/first_success_path",
                ROOT / "experiments_ori5/full/five_d3/B/seed2101/last_policy_deterministic_path",
                1,
            ),
            "torus2": (
                ROOT / "experiments_ori6/screen/rescue/P0X2/torus2/A/seed3103/first_success_path",
                ROOT / "experiments_ori6/screen/rescue/P0X2/torus2/A/seed3103/best_policy_deterministic_path",
                2,
            ),
            "water1": (
                ROOT / "experiments_ori5/tuning/r1/water1/B/seed1101/first_success_path",
                ROOT / "experiments_ori5/full/water1/B/seed2101/best_policy_deterministic_path",
                2,
            ),
            "watersqrt2": (
                ROOT / "experiments_ori5/full/watersqrt2/B/seed2101/first_success_path",
                ROOT / "experiments_ori5/full/watersqrt2/B/seed2101/best_policy_deterministic_path",
                2,
            ),
        }
        for task, (success_path, failure_path, k) in pairs.items():
            with self.subTest(task=task):
                success_peak, success_metrics, success_initial = _spectral_peak(
                    success_path,
                    k,
                )
                reference = spectral.spectral_subspace_geometry(
                    success_initial,
                    k,
                )["projector"]
                failure_peak, failure_metrics, failure_initial = _spectral_peak(
                    failure_path,
                    k,
                    reference=reference,
                )
                np.testing.assert_array_equal(success_initial, failure_initial)
                self.assertGreater(success_peak, failure_peak)
                for row in success_metrics + failure_metrics:
                    self.assertTrue(all(np.isfinite(row[name]) for name in (
                        "sigma_k",
                        "sigma_k_plus_1",
                        "rho",
                        "delta",
                        "potential",
                    )))

    def test_telescoping_retention_and_exact_failure_rollbacks(self):
        rng = np.random.default_rng(1)
        _ = rng.normal(size=(6, 2)) @ np.diag([3.0, 0.2])
        first = rng.normal(size=(6, 2)) @ np.diag([2.0, 0.3])
        second = rng.normal(size=(6, 2)) @ np.diag([2.5, 0.1])

        for terminal in ("success", "timeout"):
            env = _mock_spectral_environment([
                {"node": first, "reason": "running"},
                {"node": second, "reason": terminal},
            ])
            with mock.patch.object(
                rejected.RejectedCrossingContinuationTargetFoldingEnv,
                "step",
                _mock_parent_step,
            ):
                _, _, _, one = env.step(None, 0.1, [])
                _, _, _, two = env.step(None, 0.1, [])
            expected = 0.05 * two["ori6_spectral_potential_after"]
            self.assertAlmostEqual(
                one["ori6_spectral_reward"]
                + two["ori6_spectral_reward"],
                expected,
                places=14,
            )
            self.assertAlmostEqual(
                env.ori6_spectral_cumulative_reward,
                expected,
                places=14,
            )
            self.assertEqual(two["ori6_spectral_rollback_reward"], 0.0)

        for terminal in sorted(spectral.SPECTRAL_ROLLBACK_REASONS):
            valid = terminal in ("wrong_flat", "no_modes")
            env = _mock_spectral_environment([
                {"node": first, "reason": "running"},
                {
                    "node": second if valid else None,
                    "reason": terminal,
                    "valid": valid,
                },
            ])
            with mock.patch.object(
                rejected.RejectedCrossingContinuationTargetFoldingEnv,
                "step",
                _mock_parent_step,
            ):
                _, _, _, one = env.step(None, 0.1, [])
                _, _, _, two = env.step(None, 0.1, [])
            self.assertAlmostEqual(
                one["ori6_spectral_reward"]
                + two["ori6_spectral_reward"],
                0.0,
                places=14,
            )
            self.assertEqual(env.ori6_spectral_cumulative_reward, 0.0)
            self.assertEqual(two["ori6_spectral_rollback_reason"], terminal)
            self.assertEqual(env.ori6_spectral_rollback_count, 1)

    def test_weight_zero_is_parent_bit_exact_and_rejects_have_zero_delta(self):
        actions = rejected_tests._saved_water_actions()
        parent_args = rejected_tests._water_args(max_steps=206)
        node, member, fix, _ = rejected3d.load_problem_data(
            parent_args.data_dir
        )
        with rejected3d.patched_three_dimensional_runtime(parent_args, member):
            parent_env = rejected3d.make_environment(
                parent_args, node, member, fix
            )
            parent_summary, parent_nodes, parent_actions = (
                rejected3d.run_episode(
                    parent_env,
                    rejected_tests._SavedActionAgent(actions),
                    deterministic=True,
                )
            )

        zero_args = _spectral_water_args(max_steps=206, weight=0.0)
        with adapter.patched_three_dimensional_runtime(zero_args, member):
            zero_env = adapter.make_environment(zero_args, node, member, fix)
            zero_summary, zero_nodes, zero_actions = adapter.run_episode(
                zero_env,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        for key, value in parent_summary.items():
            self.assertEqual(value, zero_summary[key], key)
        self.assertEqual(len(parent_nodes), len(zero_nodes))
        for parent_node, zero_node in zip(parent_nodes, zero_nodes):
            np.testing.assert_array_equal(parent_node, zero_node)
        self.assertTrue(all(
            parent["reward"].hex() == zero["reward"].hex()
            for parent, zero in zip(parent_actions, zero_actions)
        ))

        args = _spectral_water_args(max_steps=400, weight=0.05)
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            for action in actions:
                _, _, done, info = env.step(
                    np.asarray(action["physical_direction"], dtype=float),
                    float(action["requested_width"]),
                    np.asarray(action["mode_coefficients"], dtype=float),
                )
                if done:
                    break
            self.assertEqual(info["ori6_rejected_crossing"], 1)
            self.assertEqual(info["ori6_spectral_reward"], 0.0)
            self.assertEqual(info["ori6_spectral_potential_difference"], 0.0)
            safe_node = env.node.copy()
            safe_potential = env.ori6_spectral_current_potential
            cache = copy.deepcopy(env._ori6_spectral_metrics_cache)
            last = actions[-1]
            _, _, repeat_done, repeat = env.step(
                np.asarray(last["physical_direction"], dtype=float),
                float(last["requested_width"]),
                np.asarray(last["mode_coefficients"], dtype=float),
            )
        self.assertFalse(repeat_done)
        self.assertEqual(repeat["ori6_rejected_crossing"], 1)
        self.assertEqual(repeat["ori6_spectral_reward"], 0.0)
        self.assertEqual(env.ori6_spectral_current_potential, safe_potential)
        np.testing.assert_array_equal(env.node, safe_node)
        _assert_cache_equal(self, env._ori6_spectral_metrics_cache, cache)

        horizon_args = _spectral_water_args(max_steps=206, weight=0.05)
        with adapter.patched_three_dimensional_runtime(horizon_args, member):
            horizon_env = adapter.make_environment(
                horizon_args, node, member, fix
            )
            horizon_summary, _, horizon_actions = adapter.run_episode(
                horizon_env,
                rejected_tests._SavedActionAgent(actions),
                deterministic=True,
            )
        self.assertEqual(horizon_summary["termination_reason"], "timeout")
        self.assertEqual(horizon_actions[-1]["ori6_rejected_crossing"], 1)
        self.assertEqual(horizon_actions[-1]["ori6_spectral_reward"], 0.0)
        self.assertAlmostEqual(
            horizon_summary["ori6_spectral_total_reward"],
            horizon_env.ori6_spectral_cumulative_reward,
            places=14,
        )

    def test_all_five_smoke_and_three_dimensional_hooks_restore(self):
        planar = ((2.0, 17), (3.0, 19))
        for target, seed in planar:
            with self.subTest(task=f"five_{target}"):
                args = spectral.parse_args([
                    spectral.CLI_FLAG,
                    "--data-dir", str(ROOT / "model/five_bar_linkage"),
                    "--target-spread", str(target),
                    "--max-steps", "1",
                    "--episodes", "1",
                    "--no-random-search",
                ])
                node, member, fix = ori3._load_problem_data(args)
                env = spectral.make_environment(args, node, member, fix)
                summary, nodes, actions = spectral.run_episode(
                    env,
                    ori3.UniformRandomAgent(
                        args.max_step_width,
                        np.random.default_rng(seed),
                    ),
                )
                self.assertEqual(
                    summary["ori6_spectral_formula_sha256"],
                    spectral.SPECTRAL_FORMULA_SHA256,
                )
                self.assertEqual(
                    summary["ori6_spectral_nonfinite_component_count"], 0
                )
                self.assertEqual(len(nodes), len(actions) + 1)

        original = (
            mechanics.has_proper_self_crossing,
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
        )
        spatial = (
            ("4unit_torus", 2.0, 23),
            ("waterbomb", 1.0, 29),
            ("waterbomb", np.sqrt(2.0), 31),
        )
        for model, target, seed in spatial:
            with self.subTest(task=f"{model}_{target}"):
                args = adapter.parse_args([
                    spectral.CLI_FLAG,
                    "--model", model,
                    "--target-spread", str(float(target)),
                    "--max-steps", "1",
                    "--episodes", "1",
                    "--periodic-stochastic-eval-episodes", "0",
                    "--no-random-search",
                    "--no-rigid-diagnostics",
                ])
                node, member, fix, _ = adapter.load_problem_data(args.data_dir)
                with adapter.patched_three_dimensional_runtime(args, member):
                    env = adapter.make_environment(args, node, member, fix)
                    summary, nodes, actions = adapter.run_episode(
                        env,
                        ori3.UniformRandomAgent(
                            args.max_step_width,
                            np.random.default_rng(seed),
                        ),
                    )
                self.assertEqual(
                    summary["ori6_spectral_nonfinite_component_count"], 0
                )
                self.assertEqual(len(nodes), len(actions) + 1)
        self.assertEqual(original, (
            mechanics.has_proper_self_crossing,
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
        ))

    def test_planar_and_spatial_training_write_schema_and_restore_hooks(self):
        rejected_original = (
            rejected.make_environment,
            rejected.run_episode,
            rejected.aggregate_stochastic_evaluations,
            rejected.reward_parameters,
            rejected.save_learning_curves,
        )
        spectral_original = (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_args = rejected.parse_args([
                rejected.CLI_FLAG,
                "--episodes", "1",
                "--eval-episodes", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--max-steps", "1",
                "--seed", "37",
                "--output-dir", str(root / "parent"),
                "--no-random-search",
            ])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                parent_output, parent_summary = rejected.train(parent_args)

            planar_args = spectral.parse_args([
                spectral.CLI_FLAG,
                "--ori6-spectral-weight", "0",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--max-steps", "1",
                "--seed", "37",
                "--output-dir", str(root / "planar"),
                "--no-random-search",
            ])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                planar_output, planar_summary = spectral.train(planar_args)

            spatial_args = adapter.parse_args([
                spectral.CLI_FLAG,
                "--model", "waterbomb",
                "--target-spread", "1",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--max-steps", "1",
                "--output-dir", str(root / "spatial"),
                "--no-random-search",
                "--no-rigid-diagnostics",
            ])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                spatial_output, spatial_summary = adapter.train(spatial_args)

            for output, summary in (
                (planar_output, planar_summary),
                (spatial_output, spatial_summary),
            ):
                output = Path(output)
                self.assertTrue(
                    (output / "ori6_spectral_subspace_learning_curve.png").exists()
                )
                with open(output / "run_config.json", encoding="utf-8") as file:
                    config = json.load(file)
                self.assertEqual(
                    config["spectral_subspace_formula_sha256"],
                    spectral.SPECTRAL_FORMULA_SHA256,
                )
                self.assertEqual(
                    summary["ori6_spectral_nonfinite_component_count"], 0
                )

            with open(
                Path(parent_output) / "training_history.csv",
                newline="",
                encoding="utf-8",
            ) as file:
                parent_history = list(csv.DictReader(file))
            with open(
                Path(planar_output) / "training_history.csv",
                newline="",
                encoding="utf-8",
            ) as file:
                planar_history = list(csv.DictReader(file))
            self.assertEqual(len(parent_history), len(planar_history))
            for parent_row, planar_row in zip(parent_history, planar_history):
                for key, value in parent_row.items():
                    self.assertEqual(value, planar_row[key], key)
            def assert_parent_subset(left, right):
                if isinstance(left, dict):
                    for nested_key, nested_value in left.items():
                        self.assertIn(nested_key, right)
                        assert_parent_subset(nested_value, right[nested_key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for left_item, right_item in zip(left, right):
                        assert_parent_subset(left_item, right_item)
                else:
                    self.assertEqual(left, right)

            for key in (
                "training_episodes",
                "training_successes",
                "last_1000_success_rate",
                "last_1000_mean_terminal_ratio",
                "last_policy_held_out_evaluation",
                "last_policy_deterministic_evaluation",
                "best_policy_selection_evaluation",
                "best_policy_held_out_evaluation",
                "best_policy_deterministic_evaluation",
                "best_training_flatness_ratio",
            ):
                assert_parent_subset(parent_summary[key], planar_summary[key])

            parent_checkpoint = torch.load(
                Path(parent_output) / "policy.pt",
                map_location="cpu",
                weights_only=False,
            )
            planar_checkpoint = torch.load(
                Path(planar_output) / "policy.pt",
                map_location="cpu",
                weights_only=False,
            )

            def assert_nested_equal(left, right):
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right))
                elif isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                elif isinstance(left, dict):
                    self.assertEqual(set(left), set(right))
                    for nested_key in left:
                        assert_nested_equal(left[nested_key], right[nested_key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for left_item, right_item in zip(left, right):
                        assert_nested_equal(left_item, right_item)
                else:
                    self.assertEqual(left, right)

            for key in (
                "episode",
                "model_state_dict",
                "actor_optimizer_state_dict",
                "critic_optimizer_state_dict",
                "actor_learning_rate",
                "critic_learning_rate",
                "torch_rng_state",
                "numpy_rng_state",
            ):
                assert_nested_equal(
                    parent_checkpoint[key],
                    planar_checkpoint[key],
                )
        self.assertEqual(rejected_original, (
            rejected.make_environment,
            rejected.run_episode,
            rejected.aggregate_stochastic_evaluations,
            rejected.reward_parameters,
            rejected.save_learning_curves,
        ))
        self.assertEqual(spectral_original, (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
        ))


if __name__ == "__main__":
    unittest.main()
