"""Focused tests for the additive zero-reject staged runner/analyzer."""

from __future__ import annotations

import tempfile
import unittest
import csv
from unittest import mock

import torch
from pathlib import Path

import ori_codex6_zero_reject_3d as core3d
import analyze_ori_codex6_zero_reject_staged as analyzer
import run_ori_codex6_zero_reject_staged as runner


def report(seed, promising, strict, joint, constructive, late_joint=0.0,
           late_constructive=0.0):
    recent = {
        "strict_count": strict,
        "valid_joint_count": joint,
        "constructive_count": constructive,
        "valid_joint_rate": joint / 1024,
        "constructive_rate": constructive / 1024,
        "median_flatness": 0.2,
        "median_spread_error": 0.05,
        "mean_return": 0.1,
        "mean_reject_rate": 0.0,
        "mean_reject_count": 0.0,
        "episode_count": 1024,
        "mechanical_steps": 1024 * 200,
        "invalid_count": 0,
    }
    return {
        "seed": seed,
        "promising_at_4096": promising,
        "selection_key": {
            "strict_hits": strict,
            "last_1024_valid_joint_rate": joint / 1024,
            "last_1024_constructive_rate": constructive / 1024,
            "last_1024_terminal_joint_cost": 0.9,
            "last_1024_mean_return": 0.1,
        },
        "last_1024": recent,
        "last_512": {
            **recent,
            "valid_joint_rate": late_joint,
            "constructive_rate": late_constructive,
        },
        "integrity_pass": True,
    }


class RunnerTests(unittest.TestCase):
    FAILED_ROOT = (
        runner.SCRIPT_DIRECTORY / "experiments_ori6" / "staged" /
        "zero_reject_reward" / "P0X2RCS05ZR0"
    )

    def test_commands_parse_and_bind_task_adjustments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torus = core3d.parse_args(
                runner.build_command(root, runner.TASKS["torus2"], 4201)[3:]
            )
            water = core3d.parse_args(
                runner.build_command(root, runner.TASKS["water1"], 4201)[3:]
            )
        self.assertEqual(torus.episodes, 10000)
        self.assertEqual(torus.checkpoint_every, 512)
        self.assertEqual(torus.actor_lr_milestones, [1528, 4000, 7000])
        self.assertEqual(water.actor_lr_milestones, [2000, 4000, 7000])
        self.assertAlmostEqual(water.near_miss_spread_error_far, 0.40)
        self.assertTrue(torus.ori6_zero_reject_reward)
        self.assertTrue(water.ori6_spectral_subspace)

    def test_core_hashes_are_bound(self):
        hashes = runner.source_hashes()
        for name, expected in runner.EXPECTED_CORE_HASHES.items():
            self.assertEqual(hashes[name], expected)

    def test_retry_root_is_fresh_and_failed_attempt_is_bound(self):
        self.assertNotEqual(runner.DEFAULT_ROOT.resolve(), self.FAILED_ROOT.resolve())
        self.assertFalse(runner.DEFAULT_ROOT.exists())
        authority = runner.failed_attempt_authority()
        self.assertTrue(authority["root_preserved_and_never_reused"])
        self.assertEqual(
            [row["checkpoint_episode"] for row in authority["statuses"][:3]],
            [512, 512, 512],
        )
        self.assertEqual(
            [row["checkpoint_episode"] for row in authority["statuses"][3:]],
            [None, None, None],
        )

    def test_real_checkpoint_window_schema_writes_without_dropped_fields(self):
        payload = analyzer.analyse_checkpoint(
            self.FAILED_ROOT, 512, tasks=("torus2",), seeds=analyzer.SEEDS
        )
        expected_window_keys = set(
            payload["tasks"]["torus2"]["seeds"][0]["windows_512"][0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.csv"
            analyzer.write_csv(payload, path)
            with open(path, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                observed_fields = set(reader.fieldnames or [])
        self.assertEqual(len(rows), 3)
        self.assertTrue(expected_window_keys <= observed_fields)
        for key in (
            "update_count", "valid_joint_rate", "median_return",
            "strict_rate", "mean_flatness", "invalid_count",
            "mean_spread_error",
        ):
            self.assertIn(key, observed_fields)

    def test_completed_seed_is_retained_in_screen_gate_cohort(self):
        items = []
        for seed, active in ((4201, False), (4202, True), (4203, True)):
            items.append({
                "active": active,
                "job": {"seed": seed, "task": {"name": "torus2"}},
            })
        self.assertEqual(
            [item["job"]["seed"] for item in runner.gate_cohort(items, "torus2")],
            [4201, 4202, 4203],
        )
        self.assertEqual(
            [item["job"]["seed"] for item in runner.gate_cohort(items, "torus2", 4201)],
            [4201],
        )

    def test_advanced_history_certifies_older_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            lines = ["value,episode"] + [f"0,{episode}" for episode in range(1, 10001)]
            (destination / "training_history.csv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            self.assertTrue(runner.history_has_endpoint(destination, 512))
            self.assertTrue(runner.history_has_endpoint(destination, 10000))
            self.assertFalse(runner.history_has_endpoint(destination, 10001))

    def test_complete_10000_artifact_set_is_authoritative_over_checkpoint_9728(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            torch.save({"episode": 9728}, destination / "checkpoint.pt")
            torch.save({"episode": 10000}, destination / "policy.pt")
            for name in ("training_history.csv", "update_history.csv"):
                (destination / name).write_text(
                    "value,episode\n0,10000\n", encoding="utf-8"
                )
            (destination / "summary.json").write_text("{}\n", encoding="utf-8")
            (destination / "run_config.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(runner.checkpoint_episode(destination), 10000)

    def test_artifact_failure_is_recorded_without_raising(self):
        payload = {
            "schema_version": 1,
            "analysis_type": "synthetic",
            "generated_utc": "test",
            "tasks": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                runner.analysis, "write_artifacts",
                side_effect=RuntimeError("synthetic render failure"),
            ):
                result = runner.write_checkpoint_artifacts(
                    root, payload, 512, "torus2"
                )
            output = root / "analysis" / "checkpoint_512_torus2"
            self.assertFalse(result["pass"])
            self.assertTrue((output / "checkpoint_analysis_fallback.json").is_file())
            self.assertTrue((output / "artifact_generation_error.json").is_file())

    def test_no_positive_decision_before_4096(self):
        reports = [report(seed, True, 10, 10, 100) for seed in analyzer.SEEDS]
        for endpoint in (512, 1024, 2048, 3072):
            decision = analyzer.task_decision("torus2", reports, endpoint)
            self.assertFalse(decision["promote"])

    def test_4096_selects_only_persistent_promising_seed(self):
        collapsed_early_winner = report(4201, False, 99, 0, 0)
        persistent_a = report(4202, True, 1, 10, 30, 0.02, 0.04)
        persistent_b = report(4203, True, 0, 8, 25, 0.03, 0.03)
        decision = analyzer.task_decision(
            "torus2", [collapsed_early_winner, persistent_a, persistent_b], 4096
        )
        self.assertTrue(decision["promote"])
        self.assertEqual(decision["selected_seed"], 4203)
        self.assertNotIn(4201, decision["selection_order"])

    def test_fewer_than_two_promising_seeds_stops(self):
        reports = [
            report(4201, True, 1, 2, 20),
            report(4202, False, 0, 0, 0),
            report(4203, False, 0, 0, 0),
        ]
        decision = analyzer.task_decision("water1", reports, 4096)
        self.assertTrue(decision["stop"])
        self.assertFalse(decision["promote"])

    def test_production_signal_uses_baseline_floor(self):
        current = report(4202, True, 0, 0, 3)
        current["last_1024"].update({
            "median_flatness": 0.3,
            "median_spread_error": 0.35,
            "mean_return": -0.7,
            "mean_reject_rate": 0.0,
        })
        baseline = report(4202, True, 0, 4, 50)
        baseline["last_1024"].update({
            "median_flatness": 0.1,
            "median_spread_error": 0.04,
            "mean_return": 0.2,
        })
        warning = analyzer.production_warning("water1", current, baseline)
        self.assertEqual(warning["constructive_count_floor_from_4096"], 10)
        self.assertTrue(warning["no_signal"])
        self.assertTrue(warning["warning"])

    def test_terminal_joint_is_not_path_milestone(self):
        row = {
            "episode": "1",
            "total_reward": "1",
            "ori5_last_valid_flatness_ratio": "0.2",
            "ori5_last_valid_spread_error": "0.05",
            "ori6_joint_milestones_awarded": "3",
        }
        record = analyzer.normalise(row)
        self.assertTrue(record["constructive"])
        self.assertFalse(record["joint"])
        self.assertTrue(record["path_joint_discovery"])


if __name__ == "__main__":
    unittest.main()
