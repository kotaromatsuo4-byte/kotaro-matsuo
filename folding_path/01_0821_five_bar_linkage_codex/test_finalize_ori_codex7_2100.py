import math
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import finalize_ori_codex7_2100 as finalizer


def authority_row(episode=1, arm="no_fixed", crossing=False):
    fixed = 0.0 if arm == "no_fixed" else -1.25
    quality = .35
    return {
        "episode": str(episode), "total_reward": str(.2 if not crossing else fixed + quality),
        "flatness_ratio": ".05", "spread_error": ".04",
        "strict_compact_success": "1", "self_crossing": str(int(crossing)),
        "termination_reason": "self_crossing" if crossing else "success",
        "ori7_candidate": "ori7_spectral_terminal_crossing_fixed_penalty_ablation",
        "ori7_candidate_schema_version": "1.0.0",
        "ori7_crossing_policy": "reject_candidate_restore_last_safe_and_terminate",
        "ori7_crossing_terminal_mode": arm,
        "ori7_reward_formula_sha256":
            "508f2f185f41bbd31b7edbd0d0248163bec39554bcd00a1351deff7ae158cb3d",
        "ori7_transition_formula_sha256":
            "061539186dc820c791fa4e2c57032d66e05e5232dae881bbf3d9bbb4984ece32",
        "ori7_terminal_crossing_count": str(int(crossing)),
        "ori7_terminal_crossing_reward": str(fixed + quality if crossing else 0),
        "ori7_maximum_abs_crossing_reward_decomposition_error": "0",
        "ori7_crossing_nonfinite_component_count": "0",
        "ori5_total_terminal_quality_reward": str(quality),
        "ori6_reward_decomposition_error": "0",
        "ori6_maximum_abs_step_reward_decomposition_error": "0",
        "ori6_nonfinite_reward_component_count": "0",
        "ori6_spectral_reward_decomposition_error": "0",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
        "ori6_spectral_nonfinite_component_count": "0",
    }


class Codex72100FinalizerTests(unittest.TestCase):
    def exact_contract(self):
        jobs = [{"task": {"name": task}, "arm": arm, "seed": seed}
                for task in finalizer.TASKS for arm in finalizer.ARMS
                for seed in (7101, 7102, 7103)]
        manifest = {
            "manifest_fingerprint": finalizer.EXPECTED_MANIFEST_FINGERPRINT,
            "experiment_root": str(finalizer.DEFAULT_ROOT.resolve()),
            "schema_version": 1, "protocol": finalizer.EXPECTED_PROTOCOL,
            "tasks": list(finalizer.TASKS), "arms": list(finalizer.ARMS),
            "seeds": [7101, 7102, 7103], "episodes_cap": 10000,
            "screen_checkpoints": [512, 1024, 2048, 4096],
            "production_checkpoints": [5120, 6144, 7168, 8192, 9216, 10000],
            "job_order": list(finalizer.TASKS), "max_parallel": 12,
            "training_stop_local": "20:45", "report_deadline_local": "21:00",
            "jobs": jobs,
        }
        binding = {"manifest_fingerprint": finalizer.EXPECTED_MANIFEST_FINGERPRINT,
                   "job_order": list(finalizer.TASKS), "max_parallel": 12,
                   "training_stop_local": "20:45", "report_deadline_local": "21:00"}
        report = {"manifest_fingerprint": finalizer.EXPECTED_MANIFEST_FINGERPRINT,
                  "schema_version": 1, "protocol": finalizer.EXPECTED_PROTOCOL,
                  "training_deadline_local": "20:45", "report_deadline_local": "21:00",
                  "endpoints": [{} for _ in range(30)]}
        return manifest, binding, report

    def test_exact_campaign_fingerprint_matrix_protocol_and_schema(self):
        manifest, binding, report = self.exact_contract()
        finalizer.validate_exact_campaign_contract(
            manifest, binding, report, finalizer.DEFAULT_ROOT
        )
        wrong = copy.deepcopy(binding); wrong["manifest_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "authorized C7TC2A"):
            finalizer.validate_exact_campaign_contract(
                manifest, wrong, report, finalizer.DEFAULT_ROOT
            )
        wrong_manifest = copy.deepcopy(manifest); wrong_manifest["max_parallel"] = 11
        with self.assertRaisesRegex(RuntimeError, "matrix/protocol"):
            finalizer.validate_exact_campaign_contract(
                wrong_manifest, binding, report, finalizer.DEFAULT_ROOT
            )
        wrong_report = copy.deepcopy(report); wrong_report["schema_version"] = 2
        with self.assertRaisesRegex(RuntimeError, "report schema"):
            finalizer.validate_exact_campaign_contract(
                manifest, binding, wrong_report, finalizer.DEFAULT_ROOT
            )
        wrong_binding = copy.deepcopy(binding); wrong_binding["max_parallel"] = 11
        with self.assertRaisesRegex(RuntimeError, "Execution binding"):
            finalizer.validate_exact_campaign_contract(
                manifest, wrong_binding, report, finalizer.DEFAULT_ROOT
            )
        wrong_report = copy.deepcopy(report); wrong_report.pop("training_deadline_local")
        with self.assertRaisesRegex(RuntimeError, "report schema"):
            finalizer.validate_exact_campaign_contract(
                manifest, binding, wrong_report, finalizer.DEFAULT_ROOT
            )

    def test_causal_censor_reasons_are_distinct(self):
        base = {"task": "water1", "arm": "no_fixed", "seed": 7101,
                "launched": True, "runner_status": "screened", "atomic_endpoint": 1024}
        deadline = {"deadline_stop": True, "task_selections": {}}
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "launched": False,
             "runner_status": "queued_censored_at_training_deadline"}, deadline, {}, {}),
            "queued_at_training_deadline")
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 0}, deadline, {}, {}),
            "deadline_before_first_atomic_checkpoint")
        self.assertEqual(finalizer.causal_censor_reason(
            base, deadline, {("water1", "no_fixed"): {
                "endpoint": 1024, "reasons": ["all-seed no-signal invalid lock"]}}, {}),
            "catastrophic_stop_before_4096")
        self.assertEqual(finalizer.causal_censor_reason(
            base, deadline, {("water1", "no_fixed"): {
                "endpoint": 512, "reasons": ["integrity failure"]}}, {}),
            "integrity_failure_before_4096")
        formal_report = {"deadline_stop": False, "task_selections": {"water1": {
            "passing_arms": ["no_fixed", "historical_1p25"],
            "selected_arm": "historical_1p25", "selected_seed": 7102}}}
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 4096}, formal_report, {}, {}),
            "not_selected_by_paired_arm_comparison")
        rejected_report = copy.deepcopy(formal_report)
        rejected_report["task_selections"]["water1"]["passing_arms"] = ["historical_1p25"]
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 4096}, rejected_report, {}, {}),
            "rejected_at_4096_arm_gate")
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 6144}, formal_report, {}, {
                ("water1", "no_fixed", 7101, 6144): ["two consecutive production collapse warnings"]}),
            "production_collapse_stop")
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 5120}, formal_report, {}, {
                ("water1", "no_fixed", 7101, 5120): ["integrity failure"]}),
            "production_integrity_stop")
        self.assertEqual(finalizer.causal_censor_reason(
            {**base, "atomic_endpoint": 10000}, formal_report, {}, {}),
            "natural_completion_10000")

    def test_output_authority_and_publish_toctou(self):
        self.assertEqual(finalizer.validate_output_authority(
            finalizer.DEFAULT_OUTPUT, finalizer.DEFAULT_ROOT), finalizer.DEFAULT_OUTPUT.resolve())
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "arbitrary"
            with self.assertRaisesRegex(RuntimeError, "fixed exactly"):
                finalizer.validate_output_authority(forbidden, finalizer.DEFAULT_ROOT)
            payload = {
                "schema_version": 1, "analysis": "test", "created_utc": "now",
                "root": str(finalizer.DEFAULT_ROOT), "manifest_fingerprint": "abc",
                "formal_endpoint": 4096, "formal_caveat": "early",
                "source_sha256": {}, "finalizer_source_sha256": {},
                "input_sha256": {}, "runner_report": {}, "jobs": [],
                "paired_seed_differences": [], "episode_curves": [],
                "optimizer_curves": [],
            }
            with mock.patch.object(finalizer, "render_plots", return_value=[]), \
                    mock.patch.object(finalizer, "visual_qa", return_value=[]), \
                    mock.patch.object(finalizer, "active_codex7_pids",
                                      return_value=[{"pid": 9, "command": "trainer"}]):
                with self.assertRaisesRegex(RuntimeError, "before atomic publish"):
                    finalizer.write_bundle(payload, forbidden, enforce_authority=False)
            self.assertFalse(forbidden.exists())

    def test_contiguous_and_atomic_endpoint(self):
        rows = [authority_row(1), authority_row(2)]
        self.assertEqual(finalizer.contiguous_history(rows), 2)
        rows[1]["episode"] = "3"
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            finalizer.contiguous_history(rows)
        rows[1]["episode"] = "2"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with mock.patch.object(finalizer, "checkpoint_episode", return_value=2):
                endpoint, audit = finalizer.atomic_endpoint(directory, rows)
        self.assertEqual(endpoint, 2)
        self.assertEqual(audit["authority"], "checkpoint_and_history")

    def test_crossing_is_not_strict_or_joint_and_reward_is_exact(self):
        row = authority_row(1, "historical_1p25", crossing=True)
        source = {
            "ori_codex7.py": "20fda5bf54a1136713cebf89ab4ebf65166f9929b648316cb965cc771d565887",
            "ori_codex7_3d.py": "d6a2aea943bc2efb1a269e7f640e96e70ee27a2de04c409ec59dd5b9643e208e",
        }
        finalizer.validate_history_authority([row], "historical_1p25", 1, source)
        curves = finalizer.episode_curves(
            [row], 1, {"task": "water1", "arm": "historical_1p25", "seed": 7101}
        )
        self.assertEqual(curves[0]["strict"], 0)
        self.assertEqual(curves[0]["joint"], 0)
        self.assertEqual(curves[0]["crossing"], 1)
        row["ori7_terminal_crossing_reward"] = "-0.8"
        with self.assertRaisesRegex(RuntimeError, "Crossing reward"):
            finalizer.validate_history_authority([row], "historical_1p25", 1, source)

    def test_optimizer_nonfinite_is_rejected(self):
        identity = {"task": "water1", "arm": "no_fixed", "seed": 7101}
        rows = [{"episode": "1", "critic_loss": "1", "critic_gradient_norm": "2",
                 "actor_learning_rate": ".001", "critic_learning_rate": ".0003"}]
        self.assertEqual(len(finalizer.optimizer_curves(rows, 1, identity)), 1)
        rows[0]["critic_loss"] = "nan"
        with self.assertRaisesRegex(RuntimeError, "Nonfinite optimizer"):
            finalizer.optimizer_curves(rows, 1, identity)

    def test_paired_censor_and_common_cap(self):
        jobs = []
        curves = []
        for arm, endpoint, reward in (("no_fixed", 5000, 2.0),
                                      ("historical_1p25", 4608, 1.0)):
            jobs.append({"task": "water1", "arm": arm, "seed": 7101,
                         "atomic_endpoint": endpoint})
            curves.extend({"task": "water1", "arm": arm, "seed": 7101,
                           "episode": index,
                           "reward": (reward if index <= 4096 else
                                      (-100.0 if arm == "no_fixed" else 100.0)),
                           "terminal_flatness": .1, "spread_error": .1,
                           "strict": 1, "joint": 1, "crossing": 0}
                          for index in range(1, endpoint + 1))
        paired = finalizer.paired_differences(jobs, curves)
        row = next(row for row in paired if row["task"] == "water1" and row["seed"] == 7101)
        self.assertEqual(row["decision_common_endpoint"], 4096)
        self.assertEqual(row["observed_common_atomic_endpoint"], 4608)
        self.assertEqual(row["comparison_phase"], "formal_exact_4096")
        self.assertAlmostEqual(row["no_fixed_minus_historical_reward_mean_last1024"], 1.0)
        unavailable = next(row for row in paired if row["task"] == "torus2")
        self.assertEqual(unavailable["comparison_phase"], "unavailable_censored")

    def test_main_fails_closed_for_active_pid(self):
        with mock.patch.object(finalizer, "active_codex7_pids",
                               return_value=[{"pid": 123, "command": "trainer"}]):
            with self.assertRaisesRegex(RuntimeError, "Active Codex7"):
                finalizer.main([])

    def test_bundle_outputs_utf8_csv_json_markdown_and_nonblank_png(self):
        episode = []
        optimizer = []
        jobs = []
        for arm, reward in (("no_fixed", .2), ("historical_1p25", -.1)):
            identity = {"task": "water1", "arm": arm, "seed": 7101}
            rows = []
            for index in range(1, 65):
                rows.append({**identity, "episode": index, "reward": reward + index / 1000,
                             "terminal_flatness": 1 / (index + 1),
                             "spread_error": .2 / (index + 1),
                             "strict": int(index > 50), "joint": int(index > 48),
                             "crossing": int(index % 19 == 0)})
            for field, values in {
                "reward_rolling": finalizer.rolling([row["reward"] for row in rows], 8),
                "terminal_flatness_rolling_median": finalizer.rolling(
                    [row["terminal_flatness"] for row in rows], 8, "median"),
                "spread_error_rolling_median": finalizer.rolling(
                    [row["spread_error"] for row in rows], 8, "median"),
                "strict_rolling_rate": finalizer.rolling([row["strict"] for row in rows], 8),
                "joint_rolling_rate": finalizer.rolling([row["joint"] for row in rows], 8),
                "crossing_rolling_rate": finalizer.rolling([row["crossing"] for row in rows], 8),
            }.items():
                for row, value in zip(rows, values):
                    row[field] = value
            episode.extend(rows)
            optimizer.extend({**identity, "episode": index, "critic_loss": 1 / index,
                              "critic_gradient_norm": 2 / index,
                              "actor_learning_rate": .001,
                              "critic_learning_rate": .0003}
                             for index in range(1, 65))
            jobs.append({**identity, "launched": True, "runner_status": "screened",
                         "selected": False, "runner_endpoint": 64,
                         "atomic_endpoint": 64, "evidence_phase": "early_evidence_only",
                         "integrity_status": "verified",
                         "censor_reason": "screened", "strict_count_last1024": 14,
                         "joint_count_last1024": 16, "crossing_rate_last1024": 3 / 64})
        payload = {
            "schema_version": 1, "analysis": "test", "created_utc": "now",
            "root": "test", "manifest_fingerprint": "abc", "formal_endpoint": 4096,
            "formal_caveat": "early", "source_sha256": {}, "input_sha256": {},
            "finalizer_source_sha256": {},
            "runner_report": {}, "jobs": jobs,
            "paired_seed_differences": finalizer.paired_differences(jobs, episode),
            "episode_curves": episode, "optimizer_curves": optimizer,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            result = finalizer.write_bundle(
                payload, output, enforce_authority=False, publish_active_check=False
            )
            self.assertIn("report_ja.md", result["files"])
            self.assertNotIn("�", (output / "report_ja.md").read_text(encoding="utf-8"))
            qa = (output / "visual_qa.json").read_text(encoding="utf-8")
            self.assertIn("pixel_std", qa)
            self.assertEqual(len(list(output.glob("*.png"))), 11)


if __name__ == "__main__":
    unittest.main()
