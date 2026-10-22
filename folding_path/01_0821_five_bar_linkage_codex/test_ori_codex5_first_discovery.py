"""Focused tests for the common-reward first-discovery benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

import ori_codex5_first_discovery as discovery


class _OneStepEnvironment:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1
        return "state", 1.25, False, {
            "termination_reason": "continuing",
            "timeout": False,
        }


class Ori5FirstDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _prepare(
        self, root, problem="five_bar", budget=1, methods=("uniform",),
        seeds=(17,), *extra,
    ):
        arguments = [
            "--problem", problem,
            "--mechanics-budget", str(budget),
            "--trial-seeds", *(str(seed) for seed in seeds),
            "--methods", *methods,
            "--output-root", str(root),
            "--max-steps", "1",
            *extra,
        ]
        return discovery._prepare_run(arguments)

    def test_budget_wrapper_is_exact_and_preserves_final_reward(self):
        underlying = _OneStepEnvironment()
        wrapped = discovery.BudgetedEnvironment(underlying, 1)
        state, reward, done, info = wrapped.step()
        self.assertEqual(state, "state")
        self.assertEqual(reward, 1.25)
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "benchmark_budget")
        self.assertFalse(info["timeout"])
        self.assertTrue(wrapped.budget_exhausted)
        with self.assertRaises(RuntimeError):
            wrapped.step()

    def test_reward_overrides_and_episode_cap_are_forbidden(self):
        cases = (
            ("--episode-cap", "2"),
            ("--gamma", "0.99"),
            ("--flatness-tolerance", "0.01"),
            ("--spread-tolerance", "0.1"),
            ("--spread-sigma", "0.2"),
            ("--ori5-success-bonus", "11"),
            ("--gated-compact-cost-weight", "2"),
        )
        for option, value in cases:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(
                    ValueError, "fixed|no episode cap"
                ):
                    discovery._prepare_run([
                        "--mechanics-budget", "1",
                        "--output-root", td,
                        option, value,
                    ])

    def test_prepare_selects_planar_and_spatial_mechanics(self):
        prepared = {}
        for problem, expected_dimension, expected_k in (
            ("five_bar", 2, 1),
            ("4unit_torus", 3, 2),
            ("waterbomb", 3, 2),
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as td:
                harness, args, root, tasks, fingerprint = self._prepare(
                    td, problem=problem
                )
                self.assertEqual(harness.problem, problem)
                self.assertEqual(args.flatness_subspace_dimension, expected_k)
                self.assertEqual(args.benchmark_spatial_dimension, expected_dimension)
                self.assertEqual(tasks, [("uniform", 17)])
                self.assertEqual(len(fingerprint), 64)
                with open(root / "run_config.json", encoding="utf-8") as file:
                    config = json.load(file)
                self.assertEqual(config["benchmark_problem"], problem)
                self.assertEqual(config["spatial_dimension"], expected_dimension)
                self.assertEqual(config["episode_cap"], None)
                self.assertFalse(config["episode_cap_supported"])
                self.assertEqual(
                    config["reward_constants_sha256"],
                    discovery.FIXED_REWARD_SHA256,
                )
                self.assertIn("node", config["model_file_sha256"])
                self.assertIn("ori_codex5.py", config["source_sha256"])
                prepared[problem] = (
                    fingerprint, config["collision_identity"]["backend"]
                )
        self.assertEqual(len({value[0] for value in prepared.values()}), 3)
        self.assertNotEqual(prepared["five_bar"][1], prepared["waterbomb"][1])

    def test_paired_planar_methods_share_exact_budget_and_initial_trace(self):
        with tempfile.TemporaryDirectory() as td:
            harness, args, root, tasks, fingerprint = self._prepare(
                td, "five_bar", 4, discovery.METHODS, (17,),
                "--batch-size", "2",
            )
            rows = {}
            for method, seed in tasks:
                rows[method] = discovery.run_trial(
                    args=args,
                    method=method,
                    seed=seed,
                    mechanics_budget=harness.mechanics_budget,
                    output_root=root,
                    root_fingerprint=fingerprint,
                )

            self.assertEqual(rows["learning"]["policy_updates"], 1)
            self.assertEqual(rows["frozen"]["policy_updates"], 0)
            self.assertEqual(rows["uniform"]["policy_updates"], 0)
            for row in rows.values():
                self.assertEqual(row["mechanical_steps"], 4)
                self.assertEqual(row["episodes"], 4)
                self.assertEqual(row["censor_reason"], "mechanics_budget")
                self.assertIsNone(row["episode_cap"])
                self.assertEqual(
                    row["reward_constants_sha256"],
                    discovery.FIXED_REWARD_SHA256,
                )
            self.assertEqual(
                rows["learning"]["initial_model_sha256"],
                rows["frozen"]["initial_model_sha256"],
            )
            self.assertEqual(
                rows["learning"]["first_batch_action_trace_sha256"],
                rows["frozen"]["first_batch_action_trace_sha256"],
            )

            history_path = (
                root / "trials" / "learning" / "seed17"
                / "episode_history.csv"
            )
            with open(history_path, newline="", encoding="utf-8") as file:
                history = list(csv.DictReader(file))
            self.assertEqual(len(history), 4)
            self.assertIn("ori5_total_milestone_reward", history[0])
            self.assertIn("ori5_reward_decomposition_error", history[0])

            aggregate = discovery.aggregate_results(
                root, mechanics_budget=4, require_complete=True
            )
            self.assertTrue(aggregate["complete"])
            self.assertEqual(aggregate["completed_trials"], 3)
            self.assertTrue(
                aggregate["aggregate_validation"]
                ["censored_trials_exact_budget_checked"]
            )
            names = {path.name for path in root.rglob("*")}
            self.assertFalse(any("evaluation" in name for name in names))

    def test_each_spatial_problem_executes_through_same_budget_harness(self):
        for problem in ("4unit_torus", "waterbomb"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as td:
                harness, args, root, tasks, fingerprint = self._prepare(
                    td, problem=problem
                )
                row = discovery.run_trial(
                    args, "uniform", 17, harness.mechanics_budget, root,
                    fingerprint,
                )
                self.assertEqual(row["mechanical_steps"], 1)
                self.assertEqual(row["censor_reason"], "mechanics_budget")
                self.assertEqual(
                    row["reward_audit"]
                    ["nonfinite_reward_component_count"],
                    0,
                )
                aggregate = discovery.aggregate_results(
                    root, require_complete=True
                )
                self.assertTrue(aggregate["complete"])

    def test_aggregate_requires_every_configured_method_seed_pair(self):
        with tempfile.TemporaryDirectory() as td:
            self._prepare(
                td, budget=2, methods=("learning", "frozen"), seeds=(5, 6)
            )
            with self.assertRaisesRegex(RuntimeError, "Incomplete benchmark"):
                discovery.aggregate_results(td, require_complete=True)
            partial = discovery.aggregate_results(td, require_complete=False)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["expected_trials"], 4)
            self.assertEqual(partial["completed_trials"], 0)

    def test_aggregate_rejects_root_field_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, root, _, _ = self._prepare(td)
            path = root / "run_config.json"
            with open(path, encoding="utf-8") as file:
                config = json.load(file)
            config["mechanics_budget"] = 2
            discovery._atomic_json(path, config)
            with self.assertRaisesRegex(RuntimeError, "fingerprinted identity"):
                discovery.aggregate_results(root)

        with tempfile.TemporaryDirectory() as td:
            _, _, root, _, _ = self._prepare(td)
            path = root / "run_config.json"
            with open(path, encoding="utf-8") as file:
                config = json.load(file)
            config["root_fingerprint"] = "0" * 64
            discovery._atomic_json(path, config)
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                discovery.aggregate_results(root)

    def test_duplicate_methods_and_seeds_are_rejected(self):
        cases = (
            ["--methods", "uniform", "uniform"],
            ["--trial-seeds", "7", "7"],
        )
        for duplicate_arguments in cases:
            with self.subTest(arguments=duplicate_arguments), tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    discovery._prepare_run([
                        "--mechanics-budget", "1",
                        "--output-root", td,
                        *duplicate_arguments,
                    ])

        with tempfile.TemporaryDirectory() as td:
            _, _, root, _, _ = self._prepare(td)
            path = root / "run_config.json"
            with open(path, encoding="utf-8") as file:
                config = json.load(file)
            config["methods"] = ["uniform", "uniform"]
            config["fingerprint_identity"]["methods"] = [
                "uniform", "uniform"
            ]
            config["root_fingerprint"] = discovery._canonical_digest(
                config["fingerprint_identity"]
            )
            discovery._atomic_json(path, config)
            with self.assertRaisesRegex(RuntimeError, "duplicate methods"):
                discovery.aggregate_results(root)


if __name__ == "__main__":
    unittest.main()
