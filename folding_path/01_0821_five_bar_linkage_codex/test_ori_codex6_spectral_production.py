import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_ori_codex6_spectral_production as analyzer
import ori_codex6_spectral_random_baseline as random_harness
import run_ori_codex6_experiments as base
import run_ori_codex6_spectral_production as production


class SpectralProductionTest(unittest.TestCase):
    def regression_report_payload(self):
        manifest = production.validate_fixed_regression_manifest()
        records = [
            {
                "task": task,
                "implementation": implementation,
                "seed": seed,
                "endpoint_episode": 2048,
            }
            for task in production.REGRESSION_TASKS
            for implementation in ("ori6", "ori5_reference")
            for seed in (3301, 3302, 3303)
        ]
        task_results = {}
        for task in production.REGRESSION_TASKS:
            task_results[task] = {
                flag: True
                for flag in production.REGRESSION_REQUIRED_TASK_FLAGS
            }
            task_results[task]["checkpoint_512_authority"] = {
                "valid_and_passed": True
            }
        return {
            "schema_version": 1,
            "stage": "spectral_nonregression",
            "candidate_mode": production.MODE,
            "candidate_mode_slug": production.MODE_SLUG,
            "status": "pass",
            "successful_task_regression_gate_pass": True,
            "records": records,
            "task_results": task_results,
            "manifest_authority": {
                "path": str(production.REGRESSION_MANIFEST_PATH.resolve()),
                "sha256": production.REGRESSION_MANIFEST_SHA256,
                "manifest_fingerprint": manifest["manifest_fingerprint"],
            },
            "candidate_reward_sha256": base.candidate_reward_sha256(
                production.MODE
            ),
            "spectral_additive_source_sha256": (
                base.spectral_subspace_additive_source_hashes()
            ),
            "parent_additive_source_sha256": (
                base.rejected_crossing_additive_source_hashes()
            ),
            "source_sha256": base.source_hashes(),
        }

    def test_historical_regression_manifest_is_exact(self):
        manifest = production.validate_fixed_regression_manifest()
        self.assertEqual(
            manifest["manifest_fingerprint"],
            production.REGRESSION_MANIFEST_FINGERPRINT,
        )
        self.assertEqual(len(manifest["jobs"]), 18)

    def test_regression_report_requires_every_task_flag(self):
        payload = self.regression_report_payload()
        validation = production.validate_regression_report_payload(payload)
        self.assertTrue(validation["valid"])
        broken = copy.deepcopy(payload)
        broken["task_results"]["five_d2"]["spread_precision_pass"] = False
        with self.assertRaisesRegex(RuntimeError, "five_d2.spread_precision"):
            production.validate_regression_report_payload(broken)

    def test_regression_report_requires_exact_18_endpoint_cohort(self):
        payload = self.regression_report_payload()
        payload["records"].pop()
        with self.assertRaisesRegex(RuntimeError, "records_18"):
            production.validate_regression_report_payload(payload)

    def test_regression_authority_binds_report_sha_and_self_digest(self):
        payload = self.regression_report_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.json"
            authority_path = root / "authority.json"
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            report_sha = production.file_sha256(report_path)
            authority = {
                "schema_version": 1,
                "authority_type": (
                    "ori6_spectral_production_regression_promotion"
                ),
                "status": "pass",
                "candidate_mode": production.MODE,
                "regression_report": {
                    "path": str(report_path.resolve()),
                    "sha256": report_sha,
                    "canonical_sha256": production.canonical_digest(payload),
                },
                "regression_manifest": {
                    "path": str(
                        production.REGRESSION_MANIFEST_PATH.resolve()
                    ),
                    "sha256": production.REGRESSION_MANIFEST_SHA256,
                    "manifest_fingerprint": (
                        production.REGRESSION_MANIFEST_FINGERPRINT
                    ),
                },
            }
            authority["authority_fingerprint"] = (
                production.canonical_self_fingerprint(
                    authority, "authority_fingerprint"
                )
            )
            with open(authority_path, "w", encoding="utf-8") as handle:
                json.dump(authority, handle, indent=2)
            with mock.patch.object(
                production, "REGRESSION_REPORT_PATH", report_path
            ), mock.patch.object(
                production, "REGRESSION_AUTHORITY_PATH", authority_path
            ):
                self.assertEqual(
                    production.validate_regression_authority(
                        authority_path
                    )["status"],
                    "pass",
                )
                payload["status"] = "reject"
                with open(report_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    production.validate_regression_authority(authority_path)

    def test_random_plan_is_fixed_all_five_by_ten_at_102400(self):
        gate = {"authority_fingerprint": "a" * 64}
        jobs = production.random_screen_jobs(gate)
        self.assertEqual([job["task"] for job in jobs], list(base.TASKS))
        self.assertEqual(len(jobs), 5)
        for job in jobs:
            self.assertEqual(job["seeds"], list(range(5101, 5111)))
            self.assertEqual(job["mechanics_budget"], 102400)
            self.assertEqual(job["method"], "uniform")
            self.assertIn("--trial-seeds", job["command"])

    def test_full_plan_is_fixed_all_five_seed4101_10000(self):
        regression = {"authority_fingerprint": "a" * 64}
        random_gate = {"report_fingerprint": "b" * 64}
        jobs = production.build_full_jobs(regression, random_gate)
        self.assertEqual(len(jobs), 5)
        self.assertEqual([job["task"]["name"] for job in jobs], list(base.TASKS))
        for job in jobs:
            self.assertEqual(job["seed"], 4101)
            self.assertEqual(job["protocol"]["episodes"], 10000)
            self.assertEqual(job["candidate_mode"], production.MODE)
            self.assertIn("--ori6-spectral-subspace", job["command"])
            expected_profile = base.TASKS[job["task"]["name"]].profile
            self.assertEqual(job["profile"]["name"], expected_profile)

    def test_two_task_diagnostic_binds_exact_immutable_screen_and_override(self):
        gate = production.validate_two_task_screen_authority()
        self.assertTrue(gate["pass"])
        self.assertEqual(
            gate["screen_report_sha256"],
            production.TWO_TASK_SCREEN_REPORT_SHA256,
        )
        self.assertEqual(
            gate["screen_report_authority_fingerprint"],
            production.TWO_TASK_SCREEN_REPORT_FINGERPRINT,
        )
        self.assertFalse(gate["screen_report_authorizes_10000_episodes"])
        self.assertEqual(
            gate["user_scope_override_sha256"],
            production.canonical_digest(production.TWO_TASK_SCOPE_OVERRIDE),
        )
        self.assertFalse(
            gate["user_scope_override"]["candidate_adoption_claim_authorized"]
        )

    def test_two_task_diagnostic_is_exact_fresh_two_job_plan(self):
        gate = production.validate_two_task_screen_authority()
        jobs = production.build_two_task_diagnostic_jobs(gate)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            [job["task"]["name"] for job in jobs],
            ["torus2", "water1"],
        )
        self.assertEqual(
            [job["profile"]["name"] for job in jobs], ["A", "B"]
        )
        for job in jobs:
            self.assertEqual(job["seed"], 4101)
            self.assertEqual(job["protocol"]["episodes"], 10000)
            self.assertEqual(
                tuple(job["protocol"]["tasks"]), ("torus2", "water1")
            )
            self.assertEqual(
                tuple(job["protocol"]["implementations"]), ("ori6",)
            )
            self.assertEqual(job["spectral_weight"], 0.05)
            self.assertEqual(
                job["screen_report_sha256"],
                production.TWO_TASK_SCREEN_REPORT_SHA256,
            )
            self.assertEqual(
                job["source_sha256"], base.source_hashes()
            )
            self.assertEqual(
                job["model_sha256"],
                base.model_hashes(base.TASKS[job["task"]["name"]]),
            )
            self.assertEqual(
                job["candidate_reward_sha256"],
                base.candidate_reward_sha256(production.MODE),
            )
            self.assertEqual(
                job["candidate_mechanics_sha256"],
                base.candidate_mechanics_sha256(
                    production.MODE, base.TASKS[job["task"]["name"]]
                ),
            )
            self.assertEqual(
                job["spectral_formula_sha256"],
                base.SPECTRAL_SUBSPACE_FORMULA_SHA256,
            )
            self.assertEqual(
                job["command"].count("--ori6-spectral-subspace"), 1
            )
            weight_index = job["command"].index("--ori6-spectral-weight")
            self.assertEqual(float(job["command"][weight_index + 1]), 0.05)
            self.assertIn(
                str(Path("full") / production.TWO_TASK_DIAGNOSTIC_STAGE),
                job["output_directory"],
            )

    def test_two_task_dry_run_bypasses_old_gates_and_never_launches(self):
        gate = production.validate_two_task_screen_authority()
        jobs = production.build_two_task_diagnostic_jobs(gate)
        states = [{
            "job_id": job["job_id"], "state": "pending",
            "destination": job["output_directory"], "details": None,
        } for job in jobs]
        pending = [
            (job, Path(job["output_directory"])) for job in jobs
        ]
        with mock.patch.object(
            production, "validate_two_task_screen_authority",
            return_value=gate,
        ), mock.patch.object(
            production, "write_immutable_manifest",
            return_value=Path("two_task_manifest.json"),
        ), mock.patch.object(
            production, "classify_full_jobs",
            return_value=(states, pending),
        ), mock.patch.object(
            production, "prerequisite_status",
            side_effect=AssertionError("old regression gate was called"),
        ), mock.patch.object(base, "execute") as execute:
            self.assertEqual(production.main([
                "--stage", production.TWO_TASK_DIAGNOSTIC_STAGE,
                "--max-parallel", "2",
            ]), 0)
            execute.assert_not_called()

    def test_two_task_diagnostic_requires_max_parallel_two(self):
        for value in (1, 3):
            with self.assertRaisesRegex(ValueError, "max-parallel 2"):
                production.main([
                    "--stage", production.TWO_TASK_DIAGNOSTIC_STAGE,
                    "--max-parallel", str(value),
                ])

    def test_two_task_manifest_round_trip_preserves_tuple_identity(self):
        gate = production.validate_two_task_screen_authority()
        gates = {"spectral_screen_user_scope_override": gate}
        jobs = production.build_two_task_diagnostic_jobs(gate)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            production, "MANIFEST_DIRECTORY", Path(temporary)
        ):
            first = production.write_immutable_manifest(
                production.TWO_TASK_DIAGNOSTIC_STAGE, jobs, gates
            )
            second = production.write_immutable_manifest(
                production.TWO_TASK_DIAGNOSTIC_STAGE, jobs, gates
            )
            self.assertEqual(first, second)
            validation = production.validate_immutable_manifest(
                production.TWO_TASK_DIAGNOSTIC_STAGE, jobs, gates
            )
            self.assertEqual(
                validation["manifest_fingerprint"],
                production.load_json(first)["manifest_fingerprint"],
            )

    def test_two_task_curve_plot_has_episode_and_mechanics_axes(self):
        fields = (
            "episode", "steps", "total_reward", "flatness_ratio",
            "spread_error", "success", "timeout", "wrong_flat",
            "correction_failed", "no_modes", "self_crossing",
            "ori6_rejected_crossing_rate",
            "ori6_rejected_crossing_escape_rate",
            "ori6_spectral_total_reward",
            "ori6_spectral_maximum_potential",
            "near_miss_qF", "near_miss_qD", "near_miss_q",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            artifact.mkdir()
            with open(
                artifact / "training_history.csv", "w", newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for episode in range(1, 13):
                    writer.writerow({
                        "episode": episode, "steps": 10,
                        "total_reward": -1.0 + episode / 20,
                        "flatness_ratio": 1.0 / episode,
                        "spread_error": 0.5 / episode,
                        "success": int(episode == 12),
                        "timeout": int(episode != 12),
                        "wrong_flat": 0, "correction_failed": 0,
                        "no_modes": 0, "self_crossing": 0,
                        "ori6_rejected_crossing_rate": 0.02,
                        "ori6_rejected_crossing_escape_rate": 0.1,
                        "ori6_spectral_total_reward": episode / 10000,
                        "ori6_spectral_maximum_potential": episode / 100,
                        "near_miss_qF": episode / 12,
                        "near_miss_qD": 0.5,
                        "near_miss_q": episode / 24,
                    })
            outputs = analyzer.plot_two_task_diagnostic_curves([{
                "task": "torus2", "seed": 4101,
                "artifact_path": str(artifact),
            }], root / "curves")
            self.assertEqual(len(outputs), 1)
            self.assertTrue(Path(outputs[0]["episode_axis_png"]).is_file())
            self.assertTrue(Path(outputs[0][
                "cumulative_mechanics_axis_png"
            ]).is_file())

    def test_two_task_rolling_statistics_include_partial_prefix(self):
        self.assertEqual(
            analyzer.rolling_mean([1, 2, 3, 4], 3).tolist(),
            [1.0, 1.5, 2.0, 3.0],
        )
        self.assertEqual(
            analyzer.rolling_median([1, 4, 2, 8], 3).tolist(),
            [1.0, 2.5, 2.0, 4.0],
        )

    def test_full_execute_is_blocked_until_random_report_passes(self):
        regression = {
            "status": "pass", "pass": True,
            "authority_fingerprint": "a" * 64,
        }
        random_gate = {
            "status": "pending_not_launchable", "pass": False,
            "reason": "random pending",
        }
        with mock.patch.object(
            production, "prerequisite_status", return_value=regression
        ), mock.patch.object(
            production, "random_screen_status", return_value=random_gate
        ), mock.patch.object(base, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "random pending"):
                production.main([
                    "--stage", "full_primary", "--execute"
                ])
            execute.assert_not_called()

    def test_candidate_args_match_candidate_for_planar_and_spatial(self):
        for task in ("five_d2", "torus2"):
            args = random_harness.candidate_args(task, 5101)
            self.assertTrue(args.ori6_spectral_subspace)
            self.assertTrue(args.ori6_rejected_crossing_continuation)
            self.assertEqual(args.ori5_crossing_penalty, 2.0)
            self.assertEqual(args.ori6_spectral_weight, 0.05)
            self.assertEqual(
                args.max_steps, base.TASKS[task].max_steps
            )

    def test_uniform_harness_consumes_exact_one_step_censor(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "uniform"
            result = random_harness.main([
                "--task", "five_d2",
                "--mechanics-budget", "1",
                "--trial-seeds", "9917",
                "--output-root", str(output),
                "--checkpoint-every-episodes", "1",
                "--print-every", "0",
            ])
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_trials"], 1)
            trial = result["trial_summaries"][0]
            self.assertEqual(trial["mechanical_steps"], 1)
            self.assertEqual(trial["censor_reason"], "mechanics_budget")
            self.assertTrue(trial["reward_integrity_pass"])

    def test_chained_action_trace_is_chunk_and_resume_invariant(self):
        actions = [
            {
                "physical_direction": [1.0, 0.0],
                "mode_coefficients": [0.5, -0.5],
                "requested_width": 0.1,
            },
            {
                "physical_direction": [0.0, 1.0],
                "mode_coefficients": [0.25, 0.75],
                "requested_width": -0.2,
            },
        ]
        whole_prefixes = {}
        whole = random_harness.chained_action_trace(
            "0" * 64, actions, 0, {1, 2}, whole_prefixes
        )
        resumed_prefixes = {}
        first = random_harness.chained_action_trace(
            "0" * 64, actions[:1], 0, {1, 2}, resumed_prefixes
        )
        resumed = random_harness.chained_action_trace(
            first, actions[1:], 1, {1, 2}, resumed_prefixes
        )
        self.assertEqual(whole, resumed)
        self.assertEqual(whole_prefixes, resumed_prefixes)
        self.assertEqual(whole_prefixes["2"], whole)

    def test_extension_keeps_same_ten_seeds_and_exact_dynamic_budget(self):
        gate = {"authority_fingerprint": "a" * 64}
        with mock.patch.object(
            production, "validate_random_screen_report", return_value={}
        ), mock.patch.object(
            production, "extension_budget_by_task",
            return_value={"torus2": 345678},
        ):
            jobs = production.random_extension_jobs(gate)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["task"], "torus2")
        self.assertEqual(jobs[0]["mechanics_budget"], 345678)
        self.assertEqual(jobs[0]["seeds"], list(range(5101, 5111)))


if __name__ == "__main__":
    unittest.main()
