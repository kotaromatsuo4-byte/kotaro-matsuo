"""Focused tests for the standalone zero-reject finalizer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import finalize_ori_codex6_zero_reject_staged as finalizer


class FinalizerTests(unittest.TestCase):
    def test_completed_10000_set_overrides_periodic_9728_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.save({"episode": 9728}, run / "checkpoint.pt")
            torch.save({"episode": 10000}, run / "policy.pt")
            for name in ("training_history.csv", "update_history.csv"):
                (run / name).write_text("value,episode\n0,10000\n", encoding="utf-8")
            (run / "summary.json").write_text("{}\n", encoding="utf-8")
            (run / "run_config.json").write_text("{}\n", encoding="utf-8")
            authority = finalizer.authoritative_endpoint(run)
        self.assertEqual(authority["episode"], 10000)
        self.assertEqual(authority["checkpoint_episode"], 9728)
        self.assertTrue(authority["runner_checkpoint_10000_gap"])

    def test_incomplete_set_falls_back_to_periodic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.save({"episode": 6144}, run / "checkpoint.pt")
            authority = finalizer.authoritative_endpoint(run)
        self.assertEqual(authority["episode"], 6144)
        self.assertEqual(authority["authority"], "periodic_atomic_checkpoint")

    def test_wrong_basin_and_torus_collapse_are_separate(self):
        base = {
            "invalid": False, "flatness": 0.3, "spread_error": 0.35,
        }
        water = finalizer.enrich_record(base, "water1")
        torus = finalizer.enrich_record(
            {**base, "flatness": 1.0}, "torus2"
        )
        self.assertTrue(water["wrong_basin"])
        self.assertFalse(water["collapsed_basin"])
        self.assertFalse(torus["wrong_basin"])
        self.assertTrue(torus["collapsed_basin"])

    def test_output_inside_active_root_is_rejected(self):
        with self.assertRaises(RuntimeError):
            finalizer.write_all({}, finalizer.ACTIVE_ROOT / "postprocessor")

    def test_missing_runner_final_summary_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in finalizer.TASKS:
                for seed in finalizer.SEEDS:
                    path = root / "_runner_logs" / task / f"seed{seed}"
                    path.mkdir(parents=True)
                    (path / "status.json").write_text(
                        json.dumps({"pid": 0, "status": "stopped"}),
                        encoding="utf-8",
                    )
            with self.assertRaises(FileNotFoundError):
                finalizer.assert_safe_to_finalize(root)


if __name__ == "__main__":
    unittest.main()
