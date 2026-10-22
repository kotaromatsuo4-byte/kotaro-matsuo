"""Focused tests for the no-evaluation first-discovery harness."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import ori_codex4_3d as adapter
import ori_codex4_3d_first_discovery as discovery


class _OneStepEnvironment:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1
        return "state", 1.25, False, {
            "termination_reason": "continuing", "timeout": False,
        }


class FirstDiscoveryTests(unittest.TestCase):
    def _args(self, *extra):
        args = adapter.parse_args([
            "--model", "waterbomb",
            "--output-dir", "__unused_test_output__",
            "--max-steps", "1",
            "--batch-size", "2",
            "--no-rigid-diagnostics",
            *extra,
        ])
        return discovery._normalise_and_validate(args)

    def test_budget_wrapper_stops_exactly_without_changing_reward(self):
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

    def test_paired_tiny_runs_are_budget_exact_and_update_by_batch(self):
        args = self._args()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {}
            for method in discovery.METHODS:
                rows[method] = discovery.run_trial(
                    args, method, 17, 4, None, root, "paired-test",
                    checkpoint_every_batches=1,
                )
            # The second full batch ends exactly at the administrative
            # benchmark censor, so updating it cannot influence discovery and
            # is deliberately skipped.
            self.assertEqual(rows["learning"]["policy_updates"], 1)
            self.assertEqual(rows["frozen"]["policy_updates"], 0)
            self.assertEqual(rows["uniform"]["policy_updates"], 0)
            for row in rows.values():
                self.assertEqual(row["mechanical_steps"], 4)
                self.assertEqual(row["episodes"], 4)
                self.assertEqual(row["censor_reason"], "mechanics_budget")
            self.assertEqual(
                rows["learning"]["initial_model_sha256"],
                rows["frozen"]["initial_model_sha256"],
            )
            self.assertEqual(
                rows["learning"]["first_batch_action_trace_sha256"],
                rows["frozen"]["first_batch_action_trace_sha256"],
            )
            discovery._atomic_json(root / "run_config.json", {
                "root_fingerprint": "paired-test",
                "mechanics_budget": 4,
                "effective_ori4_args": {"max_steps": 1},
                "methods": list(discovery.METHODS),
                "seeds": [17],
            })
            aggregate = discovery.aggregate_results(root, 4)
            self.assertEqual(aggregate["completed_trials"], 3)
            self.assertEqual(aggregate["expected_trials"], 3)
            self.assertTrue(aggregate["complete"])
            discovery.aggregate_results(root, 4, require_complete=True)
            self.assertEqual(
                aggregate["methods"]["learning"]
                ["restricted_mean_first_hit_mechanical_steps"],
                4.0,
            )
            self.assertEqual(aggregate["common_episode_horizon"], 4)
            self.assertEqual(
                aggregate["methods"]["learning"]
                ["restricted_mean_first_hit_episode"],
                4.0,
            )
            names = {path.name for path in root.rglob("*")}
            self.assertFalse(any("evaluation" in name for name in names))

    def test_aggregate_rejects_incomplete_or_foreign_trials(self):
        config = {
            "root_fingerprint": "aggregate-test",
            "mechanics_budget": 4,
            "effective_ori4_args": {"max_steps": 1},
            "methods": ["learning", "frozen"],
            "seeds": [17],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discovery._atomic_json(root / "run_config.json", config)
            with self.assertRaisesRegex(RuntimeError, "Incomplete benchmark"):
                discovery.aggregate_results(root, require_complete=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discovery._atomic_json(root / "run_config.json", config)
            foreign = root / "trials" / "uniform" / "seed17"
            discovery._atomic_json(foreign / "summary.json", {
                "complete": True,
                "method": "uniform",
                "seed": 17,
                "root_fingerprint": "aggregate-test",
                "mechanics_budget": 4,
                "max_steps_per_episode": 1,
            })
            with self.assertRaisesRegex(RuntimeError, "Unexpected completed"):
                discovery.aggregate_results(root)

    def test_budget_truncated_episode_is_never_used_for_update(self):
        args = self._args("--max-steps", "3", "--batch-size", "1")
        with tempfile.TemporaryDirectory() as temporary:
            row = discovery.run_trial(
                args, "learning", 19, 1, None, Path(temporary),
                "partial-episode-test",
            )
            self.assertEqual(row["mechanical_steps"], 1)
            self.assertEqual(row["policy_updates"], 0)
            self.assertEqual(row["censor_reason"], "mechanics_budget")

    def test_resume_replays_only_from_atomic_batch_checkpoint(self):
        args = self._args("--batch-size", "1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = discovery.run_trial(
                args, "learning", 23, 3, None, root, "resume-test"
            )
            trial = root / "trials" / "learning" / "seed23"
            checkpoint = discovery.torch.load(
                trial / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["episodes"], 2)
            (trial / "summary.json").unlink()
            resumed = discovery.run_trial(
                args, "learning", 23, 3, None, root, "resume-test"
            )
            self.assertEqual(resumed["mechanical_steps"], 3)
            self.assertEqual(
                resumed["first_batch_action_trace_sha256"],
                first["first_batch_action_trace_sha256"],
            )

    def test_first_hit_path_uses_compact_step_names(self):
        args = self._args(
            "--flatness-tolerance", "2",
            "--training-flatness-tolerances", "2",
            "--near-miss-flatness-far", "3",
            "--spread-tolerance", "1",
            "--near-miss-spread-error-far", "2",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = discovery.run_trial(
                args, "uniform", 31, 2, None, root, "path-test"
            )
            self.assertEqual(row["success"], 1)
            hit = root / "trials" / "uniform" / "seed31" / "first_hit_path"
            self.assertTrue((hit / "step0.dat").is_file())
            self.assertTrue((hit / "step1.dat").is_file())
            self.assertFalse((hit / "step_0000.dat").exists())

    def test_prepare_run_shards_method_seed_tasks_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            harness, args, root, tasks, fingerprint = discovery._prepare_run([
                "--model", "waterbomb",
                "--mechanics-budget", "10",
                "--trial-seeds", "5", "6",
                "--methods", "learning", "frozen", "uniform",
                "--shard-count", "2", "--shard-index", "1",
                "--output-root", temporary,
                "--max-steps", "1", "--no-rigid-diagnostics",
            ])
            self.assertIsNotNone(args)
            self.assertEqual(
                tasks, [("frozen", 5), ("learning", 6), ("uniform", 6)]
            )
            self.assertEqual(root, Path(temporary))
            self.assertEqual(len(fingerprint), 64)
            with open(root / "run_config.json", encoding="utf-8") as file:
                config = json.load(file)
            self.assertIn("ori_codex4.py", config["source_sha256"])
            self.assertIn("node", config["model_file_sha256"])
            self.assertEqual(config["comparison_rules"]["evaluation_rollouts"], 0)


if __name__ == "__main__":
    unittest.main()
