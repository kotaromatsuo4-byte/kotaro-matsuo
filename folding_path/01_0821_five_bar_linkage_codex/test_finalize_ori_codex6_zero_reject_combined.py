from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import finalize_ori_codex6_zero_reject_combined as finalizer


class CombinedFinalizerTests(unittest.TestCase):
    def test_natural_10000_overrides_periodic_9728_checkpoint(self):
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
        self.assertTrue(authority["natural_10000_recovered"])
        self.assertTrue(authority["periodic_checkpoint_gap"])

    def test_incomplete_set_uses_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.save({"episode": 6144}, run / "checkpoint.pt")
            authority = finalizer.authoritative_endpoint(run)
        self.assertEqual(authority["episode"], 6144)
        self.assertEqual(authority["authority"], "periodic_atomic_checkpoint")

    def test_process_scan_ignores_monitor_powershell_but_keeps_python(self):
        payload = [
            {
                "ProcessId": 11,
                "Name": "python.exe",
                "CommandLine": "python run_ori_codex6_zero_reject_water_fallback.py --execute-manifest x",
            },
            {
                "ProcessId": 12,
                "Name": "powershell.exe",
                "CommandLine": "monitor P0X2RCS05ZR0WF12",
            },
            {"ProcessId": 13, "Name": "python.exe", "CommandLine": "python unrelated.py"},
        ]
        result = SimpleNamespace(stdout=json.dumps(payload))
        with mock.patch.object(finalizer.subprocess, "run", return_value=result):
            found = finalizer.matching_python_processes()
        self.assertEqual([item["pid"] for item in found], [11])

    def test_final_and_periodic_evaluation_modes_are_not_conflated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            summary = {
                "last_policy_held_out_evaluation": {
                    "episode": 10000, "evaluation_episodes": 64,
                    "success_rate": 0.75, "mean_return": 2.0,
                    "mean_terminal_ratio": 0.02, "median_terminal_ratio": 0.001,
                    "mean_spread_error": 0.03, "median_spread_error": 0.002,
                },
                "last_policy_deterministic_evaluation": {
                    "episode": 10000, "success": 0, "total_reward": -1,
                    "flatness_ratio": 1.0, "spread_error": 0.1,
                },
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            final = finalizer.extract_evaluations(run, 10000, True)
            self.assertEqual(
                {row["evaluation_mode"] for row in final},
                {"stochastic_heldout", "deterministic_final_single"},
            )
            (run / "stochastic_evaluation_history.csv").write_text(
                "episode,evaluation_episodes,success_rate,mean_return,mean_terminal_ratio,"
                "median_terminal_ratio,mean_spread_error,median_spread_error\n"
                "512,16,0.5,1,0.1,0.1,0.2,0.2\n",
                encoding="utf-8",
            )
            (run / "deterministic_evaluation_history.csv").write_text(
                "episode,success,total_reward,flatness_ratio,spread_error\n"
                "512,1,3,0.001,0.01\n",
                encoding="utf-8",
            )
            periodic = finalizer.extract_evaluations(run, 512, False)
            self.assertEqual(
                {row["evaluation_mode"] for row in periodic},
                {"periodic_stochastic", "periodic_deterministic_single"},
            )
            self.assertTrue(all(not row["heldout_final"] for row in periodic))

    def test_enriched_basin_and_noop_fields(self):
        record = {
            "invalid": False, "flatness": 0.2, "spread_error": 0.12,
            "strict": False,
        }
        raw = {
            "ori6_zero_reject_count": "3",
            "ori6_zero_reject_total_reward": "0",
            "ori6_zero_reject_timeout_reward": "-0.5",
        }
        water = finalizer.enrich_record(record, raw, "water1")
        self.assertTrue(water["boundary_plane_basin"])
        self.assertEqual(water["zero_reject_count"], 3)
        self.assertEqual(water["zero_reject_ordinary_reward"], 0.0)

    @staticmethod
    def synthetic_window(endpoint=512):
        return {
            "start_episode": 1, "end_episode": endpoint, "episode_count": endpoint,
            "mean_return": 1.0, "median_return": 1.0,
            "mean_flatness": 0.1, "median_flatness": 0.01,
            "mean_spread_error": 0.02, "median_spread_error": 0.01,
            "strict_count": 5, "strict_rate": 0.01,
            "success_count": 5, "success_rate": 0.01,
            "valid_joint_count": 10, "valid_joint_rate": 0.02,
            "constructive_count": 20, "constructive_rate": 0.04,
            "path_joint_discovery_count": 2, "near_miss_count": 2,
            "wrong_flat_count": 0, "invalid_count": 0,
            "mean_reject_count": 1.0, "mean_reject_rate": 0.01,
            "mechanical_steps": endpoint * 10,
            "max_abs_reward_decomposition_error": 1e-15,
            "nonfinite_component_count": 0,
            "max_abs_zero_reject_ordinary_reward": 0.0,
            "wrong_plane_basin_count": 0, "wrong_plane_basin_rate": 0.0,
            "boundary_plane_basin_count": 0, "boundary_plane_basin_rate": 0.0,
            "torus_collapse_basin_count": 0, "torus_collapse_basin_rate": 0.0,
            "reject_episode_count": 5, "reject_episode_rate": 0.01,
            "zero_reject_action_count": 5, "zero_reject_ordinary_reward_sum": 0.0,
            "zero_reject_timeout_reward_sum": 0.0, "max_reject_streak": 1,
            "update_count": 64,
            "critic_loss_mean": 0.2, "critic_loss_p95": 0.4, "critic_loss_max": 0.8,
            "critic_gradient_norm_mean": 0.5, "critic_gradient_norm_p95": 1.1,
            "critic_gradient_norm_max": 2.0, "critic_gradient_over_clip_fraction": 0.1,
            "actor_gradient_norm_mean": 0.1, "actor_gradient_norm_p95": 0.2,
            "actor_gradient_norm_max": 0.3, "actor_gradient_over_clip_fraction": 0.0,
            "entropy_mean": 2.0, "actor_learning_rate_first": 5e-4,
            "actor_learning_rate_last": 1.5e-4,
            "actor_learning_rate_min": 1.5e-4, "actor_learning_rate_max": 5e-4,
        }

    @classmethod
    def synthetic_run(cls, dataset, task, seed, role, label):
        records = []
        for episode in range(1, 257):
            records.append({
                "episode": episode, "reward": episode / 256,
                "flatness": max(1e-4, 1 - episode / 300),
                "spread_error": max(1e-4, 0.4 - episode / 800),
                "success": episode > 240, "strict": episode > 240,
                "constructive": episode > 180, "joint": episode > 220,
                "wrong_plane_basin": False, "boundary_plane_basin": False,
                "torus_collapse_basin": episode < 32 and task == "torus2",
                "reject_rate": 0.02, "steps": 10,
                "zero_reject_ordinary_reward": 0.0,
            })
        updates = [
            {
                "episode": episode, "critic_loss": 0.2,
                "critic_gradient_norm": 0.5, "actor_gradient_norm": 0.1,
                "entropy": 2.0, "actor_learning_rate": 5e-4,
            }
            for episode in range(8, 257, 8)
        ]
        evaluation = {
            "evaluation_mode": "periodic_stochastic",
            "source": "fixture", "heldout_final": False,
            "sample_count": 16, "episode": 256, "success_rate": 0.5,
            "mean_return": 1.0, "mean_flatness": 0.01, "median_flatness": 0.01,
            "mean_spread_error": 0.02, "median_spread_error": 0.02,
            "mean_reject_count": 1.0, "mean_zero_reject_reward": 0.0,
            "interpretation": "fixture", "dataset": dataset, "task": task,
            "seed": seed, "role": role, "run_label": label, "run_endpoint": 256,
        }
        deterministic = dict(evaluation)
        deterministic.update({
            "evaluation_mode": "periodic_deterministic_single", "sample_count": 1,
            "success_rate": 1.0,
        })
        window = cls.synthetic_window(256)
        return {
            "dataset": dataset, "task": task, "seed": seed, "role": role,
            "label": label, "run_directory": str(Path.cwd()), "endpoint": 256,
            "endpoint_authority": {
                "authority": "fixture", "checkpoint_episode": 256,
                "policy_episode": None, "history_episode": 256, "update_episode": 256,
                "natural_10000_recovered": False, "periodic_checkpoint_gap": False,
            },
            "integrity_pass": True, "integrity_errors": [],
            "first_1024": window, "last_1024": window, "last_512": window,
            "windows_512": [window],
            "evaluations": [evaluation, deterministic],
            "records": records, "updates": updates,
        }

    @classmethod
    def synthetic_payload(cls):
        runs = []
        for task in finalizer.TASKS:
            for seed in finalizer.R1_SEEDS:
                role = "selected_production" if task == "torus2" and seed == 4203 else "screen_not_selected"
                runs.append(cls.synthetic_run(
                    "r1_zero_reject", task, seed, role, f"R1 {task} s{seed}"
                ))
        for seed in finalizer.FALLBACK_SEEDS:
            role = "selected_production" if seed == 4301 else "screen_not_selected"
            runs.append(cls.synthetic_run(
                "water_fallback_far_0p12", "water1", seed, role,
                f"fallback water s{seed}",
            ))
        for task in finalizer.TASKS:
            runs.append(cls.synthetic_run(
                "old_P0X2RCS05", task, 4101, "old_baseline", f"old {task}"
            ))
        evaluations = [item for run in runs for item in run["evaluations"]]
        formal_window = cls.synthetic_window(4096)

        def gate_seed_report(seed, promising):
            return {
                "seed": seed,
                "last_1024": formal_window,
                "last_512": formal_window,
                "promising_at_4096": promising,
                "promising_components": {
                    "evidence": promising,
                    "persistence": promising,
                    "quality": promising,
                    "trend": promising,
                },
            }

        return {
            "generated_utc": "fixture",
            "runs": runs,
            "evaluations": evaluations,
            "comparisons": {
                task: {
                    "training_change_first_to_last": {
                        "mean_return_delta": 1.0,
                        "median_flatness_delta": -0.1,
                        "median_spread_error_delta": -0.1,
                        "success_rate_delta": 0.1,
                    },
                    "current_minus_old_last1024": {
                        "mean_return_delta": 0.5,
                        "median_flatness_delta": -0.05,
                        "median_spread_error_delta": -0.05,
                        "success_rate_delta": 0.05,
                    },
                }
                for task in finalizer.TASKS
            },
            "gates": {
                "r1_torus": {"promising_seeds": [4202, 4203]},
                "r1_water": {"promising_seeds": [4203]},
                "fallback_water": {"promising_seeds": [4301, 4303]},
                "seed_reports": {
                    "r1_torus": [
                        gate_seed_report(4201, False),
                        gate_seed_report(4202, True),
                        gate_seed_report(4203, True),
                    ],
                    "r1_water": [
                        gate_seed_report(4201, False),
                        gate_seed_report(4202, False),
                        gate_seed_report(4203, True),
                    ],
                    "fallback_water": [
                        gate_seed_report(4301, True),
                        gate_seed_report(4302, False),
                        gate_seed_report(4303, True),
                    ],
                },
            },
            "causal_chronology": [
                {"sequence": 1, "stage": "fixture", "description_ja": "検証", "evidence": "fixture"}
            ],
            "integrity": {
                "pass": True, "checks": [{
                    "category": "fixture", "check": "writer", "pass": True,
                    "evidence": "fixture", "detail": "PASS",
                }],
                "ordinary_zero_reject_max_abs_reward": 0.0,
                "ordinary_zero_reject_action_count": 55,
            },
            "input_sha256": {}, "input_hash_set_sha256": finalizer.canonical_digest({}),
        }

    def test_full_synthetic_writer_emits_json_csv_report_and_three_pngs(self):
        payload = self.synthetic_payload()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "combined_final"
            with mock.patch.object(finalizer, "decision_rows", return_value=[]):
                artifacts = finalizer.write_all(payload, output)
            with Path(artifacts["gates"]).open(encoding="utf-8-sig", newline="") as handle:
                gate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(gate_rows), 9)
            self.assertTrue(all(row["row_type"] == "seed_at_formal_4096" for row in gate_rows))
            self.assertGreater(Path(artifacts["report"]).stat().st_size, 0)
            report = Path(artifacts["report"]).read_text(encoding="utf-8")
            self.assertIn("最終報告", report)
            self.assertNotIn("�", report)
            for key in ("learning_plot", "optimizer_plot", "evaluation_plot"):
                self.assertGreater(Path(artifacts[key]).stat().st_size, 1000)
            manifest = json.loads(Path(artifacts["artifact_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["artifact_sha256"]), 11)

    def test_active_output_and_existing_output_are_rejected(self):
        with self.assertRaises(RuntimeError):
            finalizer.write_all({}, finalizer.R1_ROOT / "postprocessor")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                finalizer.write_all({}, Path(directory))


if __name__ == "__main__":
    unittest.main()
