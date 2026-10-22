import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_ori_codex7_terminal_crossing_staged as analyzer
import ori_codex7 as core
import run_ori_codex7_terminal_crossing_staged as runner


class Codex7TerminalCrossingStagedTests(unittest.TestCase):
    def test_fixed_cohort_and_commands(self):
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "fresh"
            jobs = runner.build_jobs(root)
        self.assertEqual(len(jobs), 30)
        identities = {
            (row["task"]["name"], row["arm"], row["seed"]) for row in jobs
        }
        self.assertEqual(len(identities), 30)
        for row in jobs:
            command = row["command"]
            arm = row["arm"]
            self.assertEqual(
                command[command.index("--ori7-crossing-terminal-mode") + 1], arm
            )
            self.assertEqual(
                command[command.index("--ori5-crossing-penalty") + 1], "1.25"
            )
            self.assertEqual(
                command[command.index("--ori6-spectral-weight") + 1], "0.05"
            )
            self.assertIn("--no-random-search", command)
            self.assertIn("--data-dir", command)
            expected_data = str((runner.SCRIPT_DIRECTORY / row["task"]["model_directory"]).resolve())
            self.assertEqual(command[command.index("--data-dir") + 1], expected_data)

    def test_task_specific_frozen_configs(self):
        self.assertEqual(runner.TASKS["torus2"].actor_lr_milestones[0], 1528)
        self.assertEqual(runner.TASKS["torus2"].max_steps, 200)
        self.assertEqual(runner.TASKS["water1"].near_miss_spread_error_far, 0.12)
        self.assertEqual(runner.TASKS["water1"].max_steps, 400)
        for name in ("five_d2", "five_d3", "watersqrt2"):
            self.assertEqual(runner.TASKS[name].profile, "B")
        self.assertEqual(list(runner.TASKS), [
            "torus2", "water1", "five_d2", "five_d3", "watersqrt2"
        ])
        self.assertEqual(runner.launchable_task_names({}), {"torus2", "water1"})
        self.assertEqual(
            runner.launchable_task_names({"torus2": {}, "water1": {}}),
            {"five_d2"},
        )

    def test_child_thread_contract(self):
        result = runner.verify_thread_contract()
        self.assertEqual(result["torch_intraop"], 1)
        self.assertEqual(set(result["env"].values()), {"1"})

    def test_fast_completion_drains_every_reached_checkpoint(self):
        self.assertEqual(
            runner.reached_unchecked(runner.SCREEN_CHECKPOINTS, set(), 10000),
            [512, 1024, 2048, 4096],
        )
        self.assertEqual(
            runner.reached_unchecked(runner.PRODUCTION_CHECKPOINTS, set(), 10000),
            [5120, 6144, 7168, 8192, 9216, 10000],
        )

    def test_manifest_is_exact_and_does_not_create_experiment_root(self):
        hashes = runner.source_hashes()
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "must_not_exist"
            payload = runner.manifest_payload(
                root, hashes[runner.CORE_PLANAR], hashes[runner.CORE_SPATIAL]
            )
            self.assertFalse(root.exists())
        content = dict(payload)
        recorded = content.pop("manifest_fingerprint")
        self.assertEqual(recorded, runner.canonical_digest(content))
        self.assertEqual(len(payload["jobs"]), 30)
        self.assertEqual(payload["max_parallel"], 12)
        self.assertEqual(payload["training_stop_local"], "20:45")
        self.assertEqual(payload["report_deadline_local"], "21:00")
        self.assertFalse(payload["wall_time_is_comparison_metric"])
        self.assertLess(payload["path_budget"]["maximum_known_characters"], 240)
        self.assertEqual(runner.DEFAULT_ROOT, runner.EXPERIMENT_DIRECTORY / "C7TC2A")

    def test_manifest_survives_json_roundtrip_and_rebuild(self):
        hashes = runner.source_hashes()
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            directory = Path(directory)
            root = directory / "fresh"
            payload = runner.manifest_payload(
                root, hashes[runner.CORE_PLANAR], hashes[runner.CORE_SPATIAL]
            )
            path = directory / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            validated = runner.validate_manifest(path)
            self.assertEqual(validated["manifest_fingerprint"], payload["manifest_fingerprint"])
            self.assertFalse(root.exists())

    def test_crossing_contract_fields_are_classified(self):
        base = {
            "episode": "1", "total_reward": "-0.9", "flatness_ratio": "0.2",
            "spread_error": "0.1", "ori7_terminal_crossing_count": "1",
            "ori6_maximum_spread_progress": "0.9",
            "termination_reason": "self_crossing",
            "ori7_crossing_terminal_mode": "historical_1p25",
            "ori7_candidate": core.CANDIDATE_NAME,
            "ori7_candidate_schema_version": core.CANDIDATE_SCHEMA_VERSION,
            "ori7_crossing_policy": core.CROSSING_POLICY,
            "ori7_reward_formula_sha256": core.REWARD_FORMULA_SHA256,
            "ori7_transition_formula_sha256": core.TRANSITION_FORMULA_SHA256,
            "ori7_terminal_crossing_reward": "-0.9",
            "ori7_maximum_abs_crossing_reward_decomposition_error": "0",
            "ori7_crossing_nonfinite_component_count": "0",
            "self_crossing": "1", "ori5_total_terminal_quality_reward": ".35",
            "ori6_reward_decomposition_error": "0",
            "ori6_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_nonfinite_reward_component_count": "0",
            "ori6_spectral_reward_decomposition_error": "0",
            "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_spectral_nonfinite_component_count": "0",
        }
        row = analyzer.classify(base)
        self.assertTrue(row["crossing"])
        self.assertTrue(row["invalid"])
        self.assertFalse(row["strict"])
        self.assertFalse(row["constructive"])
        self.assertFalse(row["joint"])
        self.assertEqual(row["crossing_fixed_reward"], -1.25)
        self.assertAlmostEqual(row["crossing_q_safe"], 0.35)
        self.assertTrue(math.isnan(row["noncrossing_reward"]))
        self.assertEqual(analyzer.validate_history_row(base, "historical_1p25"), [])
        missing = dict(base); missing.pop("ori7_transition_formula_sha256")
        self.assertIn("history_ori7_transition_formula_sha256",
                      analyzer.validate_history_row(missing, "historical_1p25"))
        wrong = dict(base); wrong["ori7_terminal_crossing_reward"] = "-0.8"
        self.assertIn("history_crossing_reward_contract",
                      analyzer.validate_history_row(wrong, "historical_1p25"))
        accepted = dict(base)
        accepted.update({"ori7_terminal_crossing_count": "0",
                         "ori7_terminal_crossing_reward": "0",
                         "self_crossing": "0", "termination_reason": "timeout"})
        self.assertEqual(analyzer.validate_history_row(accepted, "historical_1p25"), [])
        accepted["ori7_terminal_crossing_reward"] = ".1"
        self.assertIn("history_noncrossing_ori7_terminal_crossing_reward_nonzero",
                      analyzer.validate_history_row(accepted, "historical_1p25"))
        missing_parent = dict(base); missing_parent.pop("ori6_spectral_reward_decomposition_error")
        self.assertIn("history_missing_ori6_spectral_reward_decomposition_error",
                      analyzer.validate_history_row(missing_parent, "historical_1p25"))

    def test_two_of_three_promotion_and_selected_seed(self):
        reports = []
        for seed, promising in zip(analyzer.SEEDS, (True, True, False)):
            reports.append({
                "arm": "no_fixed", "task": "five_d2", "seed": seed,
                "integrity_pass": True, "promising_at_4096": promising,
                "full": {"strict_count": 1},
                "last1024": {
                    "strict_count": 2 if seed == 7102 else 1,
                    "joint_count": 2, "constructive_count": 8,
                    "spread_error_median": .1, "flatness_median": .1,
                    "return_mean": 1.0, "noncrossing_return_mean": 1.0,
                    "crossing_rate": .1,
                },
                "last512": {
                    "strict_count": 1, "constructive_count": 4,
                    "invalid_count": 0, "count": 512,
                },
            })
        decision = analyzer.arm_decision(reports, 4096)
        self.assertTrue(decision["promote"])
        self.assertEqual(decision["promising_seed_count"], 2)
        self.assertEqual(decision["selected_seed"], 7102)

    def test_config_validation_binds_core_formulas_and_fixed_reward(self):
        config = analyzer.expected_config("water1", "no_fixed", 7101)
        config.update({
            "ori7_candidate": core.CANDIDATE_NAME,
            "ori7_candidate_schema_version": core.CANDIDATE_SCHEMA_VERSION,
            "ori7_crossing_policy": core.CROSSING_POLICY,
            "ori7_update_method": "episode_separated_monte_carlo_actor_critic",
            "ori7_reward_formula_sha256": core.REWARD_FORMULA_SHA256,
            "ori7_transition_formula_sha256": core.TRANSITION_FORMULA_SHA256,
            "ori7_reward_parameters": {"crossing_fixed_reward": 0.0},
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(
                analyzer.validate_config(path, "no_fixed", "water1", 7101), []
            )
            config["actor_lr_milestones"] = [1, 2, 3]
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertIn("config_actor_lr_milestones",
                          analyzer.validate_config(path, "no_fixed", "water1", 7101))

    def test_source_hash_closure_and_mutation_fail_closed(self):
        for name in ("ori_codex.py", "ori_codex3.py", "ori_codex4.py", "ori_codex4_3d.py",
                     "single_solution_common.py"):
            self.assertIn(name, runner.SOURCE_FILES)
        hashes = runner.source_hashes()
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            directory = Path(directory); root = directory / "fresh"
            payload = runner.manifest_payload(root, hashes[runner.CORE_PLANAR], hashes[runner.CORE_SPATIAL])
            path = directory / "manifest.json"; path.write_text(json.dumps(payload), encoding="utf-8")
            mutated = dict(hashes); mutated["ori_codex4.py"] = "0" * 64
            with mock.patch.object(runner, "source_hashes", return_value=mutated):
                with self.assertRaisesRegex(RuntimeError, "Bound source hashes changed"):
                    runner.validate_manifest(path)

    def test_same_seed_paired_comparison_requires_two_wins(self):
        def report(seed, strict, ret=1.0):
            return {"seed": seed, "last1024": {
                "strict_count": strict, "joint_count": 2,
                "constructive_count": 8, "noncrossing_return_mean": ret,
                "crossing_rate": .05, "spread_error_median": .1,
                "flatness_median": .1,
            }}
        no = [report(7101, 2), report(7102, 1), report(7103, 1, math.nan)]
        hi = [report(7101, 1), report(7102, 1), report(7103, 1, math.nan)]
        result = analyzer.paired_arm_comparison(no, hi)
        self.assertIsNone(result["selected_arm"])
        self.assertEqual(result["wins"]["no_fixed"], 1)
        no[1]["last1024"]["strict_count"] = 2
        result = analyzer.paired_arm_comparison(no, hi)
        self.assertEqual(result["selected_arm"], "no_fixed")

    def test_analyze_both_pass_inconclusive_does_not_select_or_crash(self):
        def fake_report(root, arm, task, seed, endpoint):
            strict = 2 if (arm == "no_fixed" and seed == 7101) else 1
            window = {
                "strict_count": strict, "joint_count": 2,
                "constructive_count": 8, "noncrossing_return_mean": 1.0,
                "return_mean": 1.0, "crossing_rate": .05,
                "spread_error_median": .1, "flatness_median": .1,
                "invalid_count": 0, "count": 512,
            }
            return {"arm": arm, "task": task, "seed": seed,
                    "integrity_pass": True, "promising_at_4096": True,
                    "full": {**window, "strict_count": strict},
                    "last1024": dict(window), "last512": dict(window)}
        with mock.patch.object(analyzer, "seed_report", side_effect=fake_report):
            payload = analyzer.analyze(Path("."), 4096, tasks=("water1",))
        decision = payload["task_decisions"]["water1"]
        self.assertIsNone(decision["selected_arm"])
        self.assertIsNone(decision["selected_seed"])
        self.assertFalse(decision["promote_selected_to_10000"])
        self.assertFalse(decision["comparative_conclusion"])

    def test_production_requires_two_consecutive_baseline_relative_warnings(self):
        def window(signal_constructive, flatness):
            return {"strict_count": 0, "joint_count": 0,
                    "constructive_count": signal_constructive,
                    "flatness_median": flatness, "spread_error_median": .3,
                    "crossing_rate": .3}
        baseline = {"last512": window(20, .1)}
        report = {"task": "water1", "integrity_pass": True,
                  "last512": window(1, 1.0)}
        first = analyzer.production_decision(report, baseline, 0)
        self.assertTrue(first["warning"]); self.assertFalse(first["stop"])
        second = analyzer.production_decision(report, baseline, first["consecutive_warning_count"])
        self.assertTrue(second["stop"])
        recovered = dict(report); recovered["last512"] = window(10, .1)
        reset = analyzer.production_decision(recovered, baseline, first["consecutive_warning_count"])
        self.assertEqual(reset["consecutive_warning_count"], 0)


if __name__ == "__main__":
    unittest.main()
