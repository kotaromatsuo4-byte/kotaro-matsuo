"""Non-launching contract tests for the additive C11 staged experiment."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import analyze_ori_codex11_zero_nonswap_staged as analysis
import ori_codex6_zero_reject as zero_reject
import run_ori_codex11_zero_nonswap_staged as runner


class C11ZeroNonswapRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = runner.manifest_payload(runner.DEFAULT_ROOT, 18)

    def test_full_c10_matrix_and_paired_seeds(self):
        self.assertEqual(runner.SEEDS, (9101, 9102, 9103))
        self.assertEqual(runner.SCREEN_CHECKPOINTS, (512, 1024, 2048, 4096))
        self.assertEqual(runner.PRODUCTION_CHECKPOINTS[-1], 10000)
        self.assertEqual(runner.MAX_PARALLEL, 18)
        self.assertEqual(
            {task: runner.TASKS[task].arms for task in runner.TASK_ORDER},
            {
                "five_d2": ("B", "W"),
                "five_d3": ("B", "W"),
                "watersqrt2": ("B", "E"),
            },
        )
        self.assertEqual(len(self.payload["jobs"]), 18)
        for task in runner.TASK_ORDER:
            jobs = [
                job for job in self.payload["jobs"]
                if job["task"]["name"] == task
            ]
            self.assertEqual(len(jobs), 6)
            for arm in runner.TASKS[task].arms:
                self.assertEqual(
                    sorted(job["seed"] for job in jobs if job["arm"] == arm),
                    list(runner.SEEDS),
                )

    def test_campaign_root_and_absolute_deadlines_are_bound(self):
        self.assertEqual(runner.DEFAULT_ROOT.name, "C11ZRNS1")
        self.assertEqual(self.payload["campaign"], "C11ZRNS1")
        self.assertEqual(self.payload["training_date_local"], "2026-08-29")
        self.assertEqual(
            self.payload["training_deadline_local_iso"],
            "2026-08-29T09:00:00+09:00",
        )
        self.assertEqual(
            self.payload["report_deadline_local_iso"],
            "2026-08-29T09:30:00+09:00",
        )
        self.assertTrue(self.payload["all_18_jobs_fresh_from_episode_zero"])
        self.assertTrue(self.payload["selected_arm_all_three_seeds_continue"])

    def test_manifest_construction_does_not_create_experiment_root(self):
        self.assertFalse(runner.DEFAULT_ROOT.exists())

    def test_commands_use_only_zero_entrypoints_and_explicit_opt_in(self):
        entrypoints = {
            Path(job["entrypoint"]).name for job in self.payload["jobs"]
        }
        self.assertEqual(
            entrypoints,
            {"ori_codex6_zero_reject.py", "ori_codex6_zero_reject_3d.py"},
        )
        for job in self.payload["jobs"]:
            command = job["command"]
            self.assertEqual(command.count("--ori6-zero-reject-reward"), 1)
            self.assertIn("--no-random-search", command)
            self.assertEqual(job["expected_config"]["episodes"], 10000)
            self.assertEqual(job["expected_config"]["max_steps"], 400)
            self.assertIs(job["expected_config"]["ori6_zero_reject_reward"], True)
            self.assertEqual(
                job["expected_config"]["ori6_crossing_policy"],
                analysis.CROSSING_POLICY,
            )
            self.assertEqual(
                command[command.index("--selection-evaluation-seed") + 1],
                "9510000",
            )
            self.assertEqual(
                command[command.index("--final-evaluation-seed") + 1],
                "9610000",
            )

    def test_commands_are_c10_identical_after_entrypoint_and_zero_flag(self):
        c10 = importlib.import_module("run_ori_codex10_nonswap_staged")
        root = runner.SCRIPT_DIRECTORY / "_nonexistent_c11_command_test_root"
        for task_name in runner.TASK_ORDER:
            for arm in runner.TASKS[task_name].arms:
                zero_command = runner.build_command(
                    root, runner.TASKS[task_name], arm, 9101
                )
                c10_command = c10.build_command(
                    root, c10.TASKS[task_name], arm, 9101
                )
                reduced = list(zero_command)
                reduced.remove("--ori6-zero-reject-reward")
                reduced[2] = c10_command[2]
                self.assertEqual(reduced, c10_command)

    def test_optimizer_values_are_frozen_c10_values(self):
        self.assertEqual(runner.OPTIMIZERS["B"].batch_size, 8)
        self.assertEqual(runner.OPTIMIZERS["B"].actor_learning_rate, 5e-4)
        self.assertEqual(runner.OPTIMIZERS["W"].actor_learning_rate, 1e-3)
        self.assertEqual(
            runner.OPTIMIZERS["W"].actor_lr_milestones,
            (512, 2000, 4000, 7000),
        )
        self.assertEqual(runner.OPTIMIZERS["E"].batch_size, 4)
        self.assertEqual(runner.OPTIMIZERS["E"].actor_learning_rate, 2.5e-4)

    def test_reward_identity_is_bound_per_job(self):
        for job in self.payload["jobs"]:
            identity = job["literal_zero_reject_ablation"]
            self.assertEqual(identity["candidate"], analysis.CANDIDATE)
            self.assertEqual(identity["ordinary_reject_reward"], "0")
            self.assertEqual(
                identity["horizon_reject_reward"],
                "-timeout_penalty + quality_weight * q(last_safe)",
            )
            self.assertEqual(identity["configured_crossing_penalty"], 2.0)
            self.assertIs(identity["crossing_penalty_applied"], False)
            self.assertEqual(identity["formula_sha256"], analysis.FORMULA_SHA256)
            self.assertNotIn("literal_codex6_control", job)

    def test_source_and_model_hash_contracts(self):
        observed = runner.source_hashes()
        for name, required in runner.EXPECTED_CORE_HASHES.items():
            self.assertEqual(observed[name], required)
        self.assertIn(runner.RUNNER_NAME, observed)
        self.assertIn(runner.ANALYZER_NAME, observed)
        for task in runner.TASKS.values():
            self.assertEqual(
                runner.model_hashes(task),
                runner._base.EXPECTED_MODEL_HASHES[task.model_directory],
            )

    def test_job_and_manifest_fingerprints_recompute(self):
        for job in self.payload["jobs"]:
            content = dict(job)
            fingerprint = content.pop("job_fingerprint")
            self.assertEqual(runner.canonical_digest(content), fingerprint)
        content = dict(self.payload)
        fingerprint = content.pop("manifest_fingerprint")
        content.pop("generated_utc")
        self.assertEqual(runner.canonical_digest(content), fingerprint)

    def test_manifest_round_trip_validation_is_source_bound(self):
        with tempfile.TemporaryDirectory(
            prefix="c11_manifest_test_", dir=runner.SCRIPT_DIRECTORY
        ) as temporary:
            path = Path(temporary) / "manifest.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.payload, handle, ensure_ascii=False, allow_nan=False)
            validated = runner.validate_manifest(
                path, runner.DEFAULT_ROOT, runner.MAX_PARALLEL
            )
            self.assertEqual(
                validated["manifest_fingerprint"],
                self.payload["manifest_fingerprint"],
            )

    def test_historical_c10_import_is_not_monkeypatched(self):
        c10 = importlib.import_module("run_ori_codex10_nonswap_staged")
        self.assertEqual(
            c10.entrypoint(c10.TASKS["five_d2"]).name,
            "ori_codex6_spectral_subspace.py",
        )
        self.assertNotIn("--ori6-zero-reject-reward", c10.COMMON_ARGUMENTS)
        self.assertIsNot(c10, runner._base)

    def test_wrong_date_aborts_before_supervisor_or_root_creation(self):
        payload = dict(self.payload)
        payload["training_date_local"] = "1900-01-01"
        nonexistent = runner.SCRIPT_DIRECTORY / "_never_created_by_deadline_test"
        payload["experiment_root"] = str(nonexistent)
        with mock.patch.object(runner, "_ORIGINAL_EXECUTE") as execute:
            with self.assertRaisesRegex(RuntimeError, "campaign is bound"):
                runner.execute(payload, Path("unused.json"), 1.0)
            execute.assert_not_called()
        self.assertFalse(nonexistent.exists())


class C11ZeroNonswapAnalyzerTests(unittest.TestCase):
    @staticmethod
    def _valid_row() -> dict:
        return {
            "episode": "1",
            "success": "0",
            "strict_compact_success": "0",
            "steps": "1",
            "self_crossing": "0",
            "correction_failed": "0",
            "no_modes": "0",
            "ori6_rejected_crossing_count": "1",
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0",
            "ori6_spectral_total_reward": "0",
            "ori6_spectral_reward_decomposition_error": "0",
            "ori6_candidate": analysis.CANDIDATE,
            "ori6_candidate_schema_version": zero_reject.CANDIDATE_SCHEMA_VERSION,
            "ori6_crossing_policy": analysis.CROSSING_POLICY,
            "ori6_spectral_candidate": analysis.PARENT_CANDIDATE,
            "ori6_spectral_parent_candidate": (
                "ori6_P0X2_rejected_crossing_continuation"
            ),
            "ori6_spectral_weight": "0.05",
            "ori5_reward_decomposition_error": "0",
            "ori5_maximum_abs_step_reward_decomposition_error": "0",
            "ori5_nonfinite_reward_component_count": "0",
            "ori6_reward_decomposition_error": "0",
            "ori6_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_nonfinite_reward_component_count": "0",
            "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_spectral_nonfinite_component_count": "0",
            "ori6_parent_candidate": analysis.PARENT_CANDIDATE,
            "ori6_zero_reject_formula_sha256": analysis.FORMULA_SHA256,
            "ori6_zero_reject_count": "1",
            "ori6_zero_reject_total_reward": "0.0",
            "ori6_zero_reject_timeout_reward": "0.0",
            "ori6_maximum_abs_zero_reject_reward_decomposition_error": "0.0",
        }

    def test_zero_history_integrity_accepts_exact_zero(self):
        self.assertEqual(analysis._integrity_errors([self._valid_row()], 400), [])

    def test_zero_history_integrity_rejects_nonzero_ordinary_reward(self):
        row = self._valid_row()
        row["ori6_zero_reject_total_reward"] = "0.0001"
        errors = analysis._integrity_errors([row], 400)
        self.assertTrue(any("not exactly zero" in error for error in errors))

    def test_zero_history_integrity_rejects_identity_or_count_tampering(self):
        row = self._valid_row()
        row["ori6_zero_reject_formula_sha256"] = "tampered"
        row["ori6_zero_reject_count"] = "2"
        errors = analysis._integrity_errors([row], 400)
        self.assertTrue(any("formula_sha256" in error for error in errors))
        self.assertTrue(any("counts" in error for error in errors))

    def test_zero_config_authority_and_formula_tampering(self):
        job = runner.manifest_payload(runner.DEFAULT_ROOT, 18)["jobs"][0]
        expected = job["expected_config"]
        parameters = zero_reject.reward_parameters(
            SimpleNamespace(**copy.deepcopy(expected))
        )
        observed = {
            **copy.deepcopy(expected),
            "output_directory": expected["output_dir"],
            "architecture_name": (
                "variable_mode_ori6_continuous_spread_state_potential_rollback_"
                "actor_critic"
            ),
            "architecture": (
                "variable_mode_ori6_continuous_spread_state_potential_rollback_"
                "actor_critic"
            ),
            "reward_design": (
                "ori6_continuous_spread_state_potential_rollback"
            ),
            "ori5_reward_parameters": parameters,
        }
        with tempfile.TemporaryDirectory(
            prefix="c11_config_test_", dir=runner.SCRIPT_DIRECTORY
        ) as temporary:
            path = Path(temporary) / "run_config.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(observed, handle, allow_nan=False)
            self.assertEqual(analysis.validate_config(path, expected), [])
            observed["ori5_reward_parameters"]["crossing_transition"][
                "ordinary_rejected_reward_formula"
            ] = "-crossing_penalty + quality_weight * q(last_safe)"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(observed, handle, allow_nan=False)
            errors = analysis.validate_config(path, expected)
            self.assertTrue(any("ordinary_reject_formula" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
