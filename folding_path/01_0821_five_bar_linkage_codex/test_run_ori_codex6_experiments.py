import json
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import run_ori_codex6_experiments as runner
import analyze_ori_codex6_width2_feasibility as width2_audit


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class Ori6RunnerTests(unittest.TestCase):
    def test_cross2_command_contains_one_effective_penalty(self):
        jobs = runner.build_jobs(
            runner.PROTOCOLS["rescue"],
            runner.P0_CROSS2_MODE,
            ["torus2", "water1"],
        )
        self.assertEqual(len(jobs), 6)
        for job in jobs:
            command = list(job["command"])
            indices = [
                index for index, value in enumerate(command)
                if value == "--ori5-crossing-penalty"
            ]
            self.assertEqual(len(indices), 1)
            self.assertEqual(float(command[indices[0] + 1]), 2.0)
            self.assertEqual(job["expected_config"]["ori5_crossing_penalty"], 2.0)

    def test_frontier12_is_one_change_from_cross2_with_unique_cli_flags(self):
        parent = runner.expected_ori6_reward(runner.P0_CROSS2_MODE)
        candidate = runner.expected_ori6_reward(
            runner.P0_CROSS2_FRONTIER12_MODE
        )
        normalized = json.loads(json.dumps(candidate))
        self.assertEqual(
            normalized["spread_frontier"]["maximum_bonus"], 1.2
        )
        normalized["spread_frontier"]["maximum_bonus"] = 0.9
        self.assertEqual(normalized, parent)
        self.assertEqual(parent["crossing_penalty"], 2.0)
        self.assertEqual(candidate["crossing_penalty"], 2.0)
        # Protect the already-audited P0X2 reward identity.
        self.assertEqual(
            runner.candidate_reward_sha256(runner.P0_CROSS2_MODE),
            "ba48b5e037202cab32962ae45b8083d459e89c521815907597e5bcc10cef6892",
        )

        jobs = runner.build_jobs(
            runner.PROTOCOLS["rescue"],
            runner.P0_CROSS2_FRONTIER12_MODE,
            ["torus2", "water1"],
        )
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job["seed"] for job in jobs}, {3101, 3102, 3103})
        for job in jobs:
            command = list(job["command"])
            for flag, expected in (
                ("--ori5-crossing-penalty", 2.0),
                ("--ori6-spread-frontier-bonus", 1.2),
            ):
                indices = [
                    index for index, value in enumerate(command)
                    if value == flag
                ]
                self.assertEqual(len(indices), 1)
                self.assertEqual(float(command[indices[0] + 1]), expected)
            self.assertEqual(job["protocol"]["episodes"], 2048)
            self.assertEqual(job["protocol"]["checkpoint_every"], 512)
            self.assertEqual(
                job["expected_config"]["ori6_spread_frontier_bonus"], 1.2
            )
            self.assertEqual(
                job["expected_config"]["ori5_crossing_penalty"], 2.0
            )

        regression_jobs = runner.build_jobs(
            runner.PROTOCOLS["regression"],
            runner.P0_CROSS2_FRONTIER12_MODE,
            list(runner.PROTOCOLS["regression"].tasks),
        )
        for job in [*jobs, *regression_jobs]:
            command = list(job["command"])
            counts = Counter(
                token for token in command if token.startswith("--")
            )
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            if job["implementation"] == "ori5_reference":
                crossing = command.index("--ori5-crossing-penalty")
                milestone = command.index("--ori5-spread-milestone-bonus")
                self.assertEqual(float(command[crossing + 1]), 1.25)
                self.assertEqual(float(command[milestone + 1]), 0.9)
                self.assertNotIn("--ori6-spread-frontier-bonus", command)

    def test_execution_manifest_rehash_rejects_authority_changed_after_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline = root / "offline.json"
            reject = root / "reject.json"
            offline.write_text("offline", encoding="utf-8")
            reject.write_text("reject", encoding="utf-8")
            jobs = [{"fingerprint": "job-fingerprint"}]
            payload = {
                "schema_version": 1,
                "protocol": asdict(runner.PROTOCOLS["rescue"]),
                "candidate_mode": runner.P0_CROSS2_FRONTIER12_MODE,
                "jobs": jobs,
                "candidate_reward_sha256": runner.candidate_reward_sha256(
                    runner.P0_CROSS2_FRONTIER12_MODE
                ),
                "parent_candidate_mode": runner.P0_CROSS2_MODE,
                "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
                    runner.P0_CROSS2_MODE
                ),
                "single_changed_reward_parameter": {
                    "name": "ori6_spread_frontier_bonus",
                    "P0X2_value": 0.9,
                    "candidate_value": 1.2,
                },
                "offline_saved_path_ordering_authority": str(offline.resolve()),
                "offline_saved_path_ordering_authority_sha256": (
                    runner.file_sha256(offline)
                ),
                "parent_candidate_reject_authority": str(reject.resolve()),
                "parent_candidate_reject_authority_sha256": (
                    runner.file_sha256(reject)
                ),
            }
            payload["manifest_fingerprint"] = runner.canonical_digest(payload)
            payload["created_utc"] = "2026-01-01T00:00:00+00:00"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                runner, "P0_CROSS2_FRONTIER12_OFFLINE_AUDIT", offline
            ), mock.patch.object(
                runner, "P0_CROSS2_REJECT_REPORT", reject
            ):
                runner.validate_manifest_execution_binding(
                    manifest, runner.PROTOCOLS["rescue"],
                    runner.P0_CROSS2_FRONTIER12_MODE, jobs,
                )
                offline.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "manifest authority hash mismatch"
                ):
                    runner.validate_manifest_execution_binding(
                        manifest, runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE, jobs,
                    )
                offline.write_text("offline", encoding="utf-8")
                reject.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "manifest authority hash mismatch"
                ):
                    runner.validate_manifest_execution_binding(
                        manifest, runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE, jobs,
                    )

    def test_frontier12_dry_run_never_calls_gate_or_execute(self):
        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify",
            side_effect=lambda job, _rerun, _processes: (
                "pending", Path(job["output_directory"]), None,
            ),
        ), mock.patch.object(
            runner, "validate_execution_gate"
        ) as gate, mock.patch.object(
            runner, "validate_manifest_execution_binding"
        ) as manifest_gate, mock.patch.object(
            runner, "execute"
        ) as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                result = runner.main([
                    "--stage", "rescue",
                    "--candidate-mode",
                    runner.P0_CROSS2_FRONTIER12_MODE,
                    "--tasks", "torus2", "water1",
                    "--enable-cohort-early-stop",
                    "--max-parallel", "6",
                ])
        self.assertEqual(result, 0)
        self.assertIn("DRY RUN: 6 job(s) would launch; no training started.", output.getvalue())
        gate.assert_not_called()
        manifest_gate.assert_not_called()
        execute.assert_not_called()

    def test_frontier12_rescue_gate_accepts_bound_offline_and_parent_evidence(self):
        self.assertIsNone(runner.validate_execution_gate(
            runner.PROTOCOLS["rescue"],
            runner.P0_CROSS2_FRONTIER12_MODE,
            runner.EXPERIMENT_ROOT / "analysis" / "selected_candidate.json",
        ))

    def test_frontier12_rescue_gate_fails_closed_on_mutated_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline = root / "offline.json"
            reject = root / "reject.json"
            source_offline = runner.P0_CROSS2_FRONTIER12_OFFLINE_AUDIT
            source_reject = runner.P0_CROSS2_REJECT_REPORT
            offline_payload = json.loads(source_offline.read_text(encoding="utf-8"))
            reject_payload = json.loads(source_reject.read_text(encoding="utf-8"))
            reject.write_text(json.dumps(reject_payload), encoding="utf-8")
            offline_payload["parent_candidate_reject_authority"] = str(
                reject.resolve()
            )
            offline_payload["parent_candidate_reject_authority_sha256"] = (
                runner.file_sha256(reject)
            )
            offline.write_text(json.dumps(offline_payload), encoding="utf-8")
            with mock.patch.object(
                runner, "P0_CROSS2_FRONTIER12_OFFLINE_AUDIT", offline
            ), mock.patch.object(
                runner, "P0_CROSS2_REJECT_REPORT", reject
            ):
                runner.validate_execution_gate(
                    runner.PROTOCOLS["rescue"],
                    runner.P0_CROSS2_FRONTIER12_MODE,
                    root / "unused-selection.json",
                )
                offline_payload["all_required_orderings_pass"] = False
                offline.write_text(json.dumps(offline_payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "saved path ordering did not pass"
                ):
                    runner.validate_execution_gate(
                        runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE,
                        root / "unused-selection.json",
                    )

                offline_payload["all_required_orderings_pass"] = True
                omitted = offline_payload["orderings"].pop("five_d2")
                offline.write_text(json.dumps(offline_payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "per-task saved path orderings did not pass"
                ):
                    runner.validate_execution_gate(
                        runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE,
                        root / "unused-selection.json",
                    )
                offline_payload["orderings"]["five_d2"] = omitted
                offline.write_text(json.dumps(offline_payload), encoding="utf-8")
                reject_payload["decision"] = "proceed_to_successful_task_regression"
                reject.write_text(json.dumps(reject_payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "parent reject authority hash mismatch"
                ):
                    runner.validate_execution_gate(
                        runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE,
                        root / "unused-selection.json",
                    )
                offline_payload["parent_candidate_reject_authority_sha256"] = (
                    runner.file_sha256(reject)
                )
                offline.write_text(json.dumps(offline_payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "P0X2 reject decision is absent"
                ):
                    runner.validate_execution_gate(
                        runner.PROTOCOLS["rescue"],
                        runner.P0_CROSS2_FRONTIER12_MODE,
                        root / "unused-selection.json",
                    )

    def test_entropy3_is_water1_only_and_changes_only_entropy(self):
        mode = runner.P0_CROSS2_ENTROPY3_MODE
        parent = runner.P0_CROSS2_MODE
        protocol = runner.PROTOCOLS["entropy_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["water1"])
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job["seed"] for job in jobs}, {3101, 3102, 3103})
        self.assertEqual((protocol.episodes, protocol.checkpoint_every), (1024, 512))
        self.assertEqual(
            runner.expected_ori6_reward(mode), runner.expected_ori6_reward(parent)
        )
        self.assertEqual(
            runner.candidate_reward_sha256(mode),
            "ba48b5e037202cab32962ae45b8083d459e89c521815907597e5bcc10cef6892",
        )
        profile = runner.PROFILES["B"]
        old = runner.expected_optimizer(parent, profile)
        new = runner.expected_optimizer(mode, profile)
        self.assertEqual(
            {key for key in new if new[key] != old[key]}, {"entropy_coefficient"}
        )
        self.assertEqual((old["entropy_coefficient"], new["entropy_coefficient"]), (0.001, 0.003))
        for job in jobs:
            command = list(job["command"])
            counts = Counter(token for token in command if token.startswith("--"))
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            for flag, expected in (
                ("--entropy-coefficient", 0.003),
                ("--ori5-crossing-penalty", 2.0),
                ("--ori6-spread-frontier-bonus", 0.9),
                ("--ori5-spread-milestone-bonus", 0.0),
            ):
                index = command.index(flag)
                self.assertEqual(float(command[index + 1]), expected)
        with self.assertRaisesRegex(ValueError, "water1-only"):
            runner.build_jobs(runner.PROTOCOLS["rescue"], mode, ["water1"])
        with self.assertRaisesRegex(ValueError, "reserved for P0X2E3"):
            runner.build_jobs(protocol, parent, ["water1"])

    def test_entropy3_authorities_pass_and_mutations_fail_closed(self):
        self.assertIsNone(runner.validate_execution_gate(
            runner.PROTOCOLS["entropy_diagnostic"],
            runner.P0_CROSS2_ENTROPY3_MODE,
            runner.EXPERIMENT_ROOT / "analysis" / "unused.json",
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            payload = json.loads(
                runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT.read_text(encoding="utf-8")
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            runner.validate_P0X2E3_preflight_authority(path)
            payload["recommendation"] = "launch_10000"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "negative-prior recommendation mismatch"):
                runner.validate_P0X2E3_preflight_authority(path)
            reject_path = Path(directory) / "B12_reject.json"
            reject = json.loads(
                runner.P0_CROSS2_FRONTIER12_REJECT_REPORT.read_text(encoding="utf-8")
            )
            reject_path.write_text(json.dumps(reject), encoding="utf-8")
            runner.validate_P0X2B12_checkpoint_reject_authority(reject_path)
            reject["decision"] = "continue"
            reject_path.write_text(json.dumps(reject), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reject decision is absent"):
                runner.validate_P0X2B12_checkpoint_reject_authority(reject_path)

    def test_entropy3_manifest_rehashes_all_three_authorities(self):
        protocol = runner.PROTOCOLS["entropy_diagnostic"]
        mode = runner.P0_CROSS2_ENTROPY3_MODE
        jobs = runner.build_jobs(protocol, mode, ["water1"])
        source_manifest = (
            runner.EXPERIMENT_ROOT / "manifests"
            / "entropy_diagnostic_P0X2E3_9b65ea2ce286.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorities = {
                "offline": root / "offline.json",
                "reject": root / "reject.json",
                "preflight": root / "preflight.json",
            }
            for key, source in (
                ("offline", runner.P0_CROSS2_OFFLINE_AUDIT),
                ("reject", runner.P0_CROSS2_FRONTIER12_REJECT_REPORT),
                ("preflight", runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT),
            ):
                authorities[key].write_bytes(source.read_bytes())
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            for path_field, digest_field, key in (
                ("offline_saved_path_ordering_authority", "offline_saved_path_ordering_authority_sha256", "offline"),
                ("preceding_candidate_reject_authority", "preceding_candidate_reject_authority_sha256", "reject"),
                ("entropy_preflight_authority", "entropy_preflight_authority_sha256", "preflight"),
            ):
                payload[path_field] = str(authorities[key].resolve())
                payload[digest_field] = runner.file_sha256(authorities[key])
            content = dict(payload)
            content.pop("manifest_fingerprint", None)
            content.pop("created_utc", None)
            payload["manifest_fingerprint"] = runner.canonical_digest(content)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(runner, "P0_CROSS2_OFFLINE_AUDIT", authorities["offline"]), \
                 mock.patch.object(runner, "P0_CROSS2_FRONTIER12_REJECT_REPORT", authorities["reject"]), \
                 mock.patch.object(runner, "P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT", authorities["preflight"]):
                runner.validate_manifest_execution_binding(manifest, protocol, mode, jobs)
                for key in authorities:
                    with self.subTest(authority=key):
                        original = authorities[key].read_bytes()
                        authorities[key].write_bytes(original + b"changed")
                        with self.assertRaisesRegex(RuntimeError, "authority hash mismatch"):
                            runner.validate_manifest_execution_binding(
                                manifest, protocol, mode, jobs
                            )
                        authorities[key].write_bytes(original)

    def test_entropy3_1024_gate_pass_reject_and_reward_integrity(self):
        def history(spread, crossing, good, strong, bad=False):
            result = []
            for index in range(1024):
                tail = index - 768
                is_good = 0 <= tail < round(256 * good)
                is_strong = 0 <= tail < round(256 * strong)
                is_cross = 0 <= tail < round(256 * crossing)
                result.append({
                    "success": "0", "self_crossing": str(int(is_cross)),
                    "wrong_flat": "0", "timeout": str(int(not is_cross)),
                    "flatness_ratio": "0.5",
                    "spread_error": "0.05" if is_good else str(spread),
                    "ori6_maximum_spread_progress": "0.9" if is_strong else "0",
                    "ori6_total_spread_frontier_reward": "0",
                    "ori6_total_joint_milestone_reward": "0",
                    "ori6_reward_decomposition_error": "1e-3" if bad and index == 0 else "0",
                    "ori6_maximum_abs_step_reward_decomposition_error": "0",
                    "ori6_nonfinite_reward_component_count": "0",
                })
            return result
        passing = {
            3101: history(0.15, 0.25, 0.06, 0.0),
            3102: history(0.18, 0.50, 0.0, 0.30),
            3103: history(0.35, 0.90, 0.0, 0.0),
        }
        decision = runner.early_gate_decision(
            "entropy_diagnostic", "water1", passing, 1024,
            runner.P0_CROSS2_ENTROPY3_MODE,
        )
        self.assertEqual((decision["stop"], decision["reason"]), (False, "entropy_diagnostic_gate_pass"))
        rejecting = {
            3101: history(0.35, 0.20, 0.0, 0.0),
            3102: history(0.40, 0.20, 0.0, 0.0),
            3103: history(0.15, 0.20, 0.06, 0.0),
        }
        decision = runner.early_gate_decision(
            "entropy_diagnostic", "water1", rejecting, 1024,
            runner.P0_CROSS2_ENTROPY3_MODE,
        )
        self.assertEqual((decision["stop"], decision["reason"]), (True, "wide_spread_zero_success_reject"))
        passing[3101] = history(0.15, 0.25, 0.06, 0.0, bad=True)
        decision = runner.early_gate_decision(
            "entropy_diagnostic", "water1", passing, 1024,
            runner.P0_CROSS2_ENTROPY3_MODE,
        )
        self.assertEqual((decision["stop"], decision["reason"]), (True, "reward_integrity_failure"))

    def test_entropy3_dry_run_never_launches_training(self):
        with mock.patch.object(runner, "manifest_path", return_value=Path("manifest.json")), \
             mock.patch.object(runner, "active_python_processes", return_value=[]), \
             mock.patch.object(
                 runner, "classify",
                 side_effect=lambda job, _rerun, _processes: (
                     "pending", Path(job["output_directory"]), None,
                 ),
             ), \
             mock.patch.object(runner, "validate_execution_gate") as gate, \
             mock.patch.object(runner, "validate_manifest_execution_binding") as binding, \
             mock.patch.object(runner, "execute") as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                result = runner.main([
                    "--stage", "entropy_diagnostic", "--candidate-mode",
                    runner.P0_CROSS2_ENTROPY3_MODE, "--tasks", "water1",
                    "--enable-cohort-early-stop", "--max-parallel", "3",
                ])
        self.assertEqual(result, 0)
        self.assertIn("DRY RUN: 3 job(s) would launch; no training started.", output.getvalue())
        gate.assert_not_called()
        binding.assert_not_called()
        execute.assert_not_called()
        base = [
            "--stage", "entropy_diagnostic", "--candidate-mode",
            runner.P0_CROSS2_ENTROPY3_MODE, "--tasks", "water1",
        ]
        with self.assertRaisesRegex(ValueError, "enable-cohort-early-stop"):
            runner.main(base)
        with self.assertRaisesRegex(ValueError, "max-parallel >= 3"):
            runner.main([
                *base, "--enable-cohort-early-stop", "--max-parallel", "2",
            ])

    def test_entropy3_execute_dispatches_one_three_seed_early_stop_cohort(self):
        protocol = runner.PROTOCOLS["entropy_diagnostic"]
        jobs = runner.build_jobs(
            protocol, runner.P0_CROSS2_ENTROPY3_MODE, ["water1"]
        )
        pending = [
            (job, Path(job["output_directory"])) for job in jobs
        ]
        with mock.patch.object(
            runner, "execute_cohort", return_value=0
        ) as execute_cohort:
            result = runner.execute(pending, early_stop=True, max_parallel=3)
        self.assertEqual(result, 0)
        execute_cohort.assert_called_once()
        cohort = execute_cohort.call_args.args[0]
        self.assertEqual(len(cohort), 3)
        self.assertEqual(
            {item[0]["seed"] for item in cohort}, {3101, 3102, 3103}
        )
        self.assertIs(execute_cohort.call_args.kwargs["early_stop"], True)

    def test_width2_is_water1_only_and_changes_only_max_step_width(self):
        mode = runner.P0_CROSS2_WIDTH2_MODE
        parent = runner.P0_CROSS2_MODE
        protocol = runner.PROTOCOLS["width_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["water1"])
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job["seed"] for job in jobs}, {3101, 3102, 3103})
        self.assertEqual((protocol.episodes, protocol.checkpoint_every), (2048, 512))
        self.assertEqual(
            runner.early_gate_checkpoints(protocol.name), (512, 1024, 2048)
        )
        self.assertEqual(
            runner.expected_ori6_reward(mode), runner.expected_ori6_reward(parent)
        )
        profile = runner.PROFILES["B"]
        self.assertEqual(
            runner.expected_optimizer(mode, profile),
            runner.expected_optimizer(parent, profile),
        )
        task = runner.TASKS["water1"]
        old = runner.expected_mechanics(parent, task)
        new = runner.expected_mechanics(mode, task)
        self.assertEqual(
            {key for key in new if new[key] != old[key]}, {"max_step_width"}
        )
        self.assertEqual((old["max_step_width"], new["max_step_width"]), (0.3, 0.2))
        parent_jobs = {
            job["seed"]: job for job in runner.build_jobs(
                runner.PROTOCOLS["rescue"], parent, ["water1"]
            )
        }
        for job in jobs:
            command = list(job["command"])
            counts = Counter(token for token in command if token.startswith("--"))
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            for flag, expected in (
                ("--max-step-width", 0.2),
                ("--entropy-coefficient", 0.001),
                ("--ori5-crossing-penalty", 2.0),
                ("--ori6-spread-frontier-bonus", 0.9),
                ("--ori5-spread-milestone-bonus", 0.0),
            ):
                index = command.index(flag)
                self.assertEqual(float(command[index + 1]), expected)
            self.assertEqual(job["expected_config"]["max_step_width"], 0.2)
            parent_job = parent_jobs[job["seed"]]
            normalized_command = list(command)
            parent_command = list(parent_job["command"])
            for flag in ("--max-step-width", "--output-dir"):
                normalized_command[normalized_command.index(flag) + 1] = (
                    parent_command[parent_command.index(flag) + 1]
                )
            self.assertEqual(normalized_command, parent_command)
            normalized_config = json.loads(json.dumps(job["expected_config"]))
            normalized_config["max_step_width"] = 0.3
            self.assertEqual(normalized_config, parent_job["expected_config"])
        with self.assertRaisesRegex(ValueError, "water1-only"):
            runner.build_jobs(runner.PROTOCOLS["rescue"], mode, ["water1"])
        with self.assertRaisesRegex(ValueError, "reserved for P0X2W2"):
            runner.build_jobs(protocol, parent, ["water1"])

    def test_width2_saved_success_replay_is_policy_representable_and_ordered(self):
        report = width2_audit.build_report()
        replay = report["subdivided_replay"]
        self.assertTrue(report["all_required_checks_pass"])
        self.assertEqual(replay["termination_reason"], "success")
        self.assertLessEqual(replay["transition_count"], 400)
        self.assertLessEqual(replay["maximum_absolute_requested_width"], 0.2)
        self.assertEqual(replay["self_crossing_count"], 0)
        self.assertEqual(replay["correction_failure_count"], 0)
        self.assertLessEqual(replay["maximum_mode_reconstruction_error"], 1e-10)
        returns = report["return_ordering"]["P0X2_returns"]
        self.assertGreater(
            returns["subdivided_ideal_success"], returns["near_cross"]
        )
        self.assertGreater(returns["near_cross"], returns["wide_wrong"])

    def test_width2_authorities_pass_and_semantic_mutations_fail_closed(self):
        self.assertIsNone(runner.validate_execution_gate(
            runner.PROTOCOLS["width_diagnostic"],
            runner.P0_CROSS2_WIDTH2_MODE,
            runner.EXPERIMENT_ROOT / "analysis" / "unused.json",
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e3 = root / "e3_reject.json"
            feasibility = root / "width_feasibility.json"
            e3_payload = json.loads(
                runner.P0_CROSS2_ENTROPY3_REJECT_REPORT.read_text(encoding="utf-8")
            )
            feasibility_payload = json.loads(
                runner.P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT.read_text(
                    encoding="utf-8"
                )
            )
            e3.write_text(json.dumps(e3_payload), encoding="utf-8")
            feasibility.write_text(json.dumps(feasibility_payload), encoding="utf-8")
            runner.validate_P0X2E3_reject_authority(e3)
            runner.validate_P0X2W2_feasibility_authority(feasibility)
            e3_payload["decision"] = "continue"
            e3.write_text(json.dumps(e3_payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reject decision is absent"):
                runner.validate_P0X2E3_reject_authority(e3)
            feasibility_payload["all_required_checks_pass"] = False
            feasibility.write_text(
                json.dumps(feasibility_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "feasibility gate mismatch"):
                runner.validate_P0X2W2_feasibility_authority(feasibility)

    def test_width2_manifest_rehash_rejects_each_authority_TOCTOU(self):
        protocol = runner.PROTOCOLS["width_diagnostic"]
        mode = runner.P0_CROSS2_WIDTH2_MODE
        jobs = runner.build_jobs(protocol, mode, ["water1"])
        manifests = [
            path for path in (runner.EXPERIMENT_ROOT / "manifests").glob(
                "width_diagnostic_P0X2W2_*.json"
            )
            if "stale_manifest_exclusions" not in path.name
        ]
        self.assertEqual(len(manifests), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorities = {
                "offline": root / "offline.json",
                "reject": root / "reject.json",
                "feasibility": root / "feasibility.json",
            }
            for key, source in (
                ("offline", runner.P0_CROSS2_OFFLINE_AUDIT),
                ("reject", runner.P0_CROSS2_ENTROPY3_REJECT_REPORT),
                ("feasibility", runner.P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT),
            ):
                authorities[key].write_bytes(source.read_bytes())
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            for path_field, digest_field, key in (
                ("offline_saved_path_ordering_authority",
                 "offline_saved_path_ordering_authority_sha256", "offline"),
                ("preceding_candidate_reject_authority",
                 "preceding_candidate_reject_authority_sha256", "reject"),
                ("width_feasibility_authority",
                 "width_feasibility_authority_sha256", "feasibility"),
            ):
                payload[path_field] = str(authorities[key].resolve())
                payload[digest_field] = runner.file_sha256(authorities[key])
            content = dict(payload)
            content.pop("manifest_fingerprint", None)
            content.pop("created_utc", None)
            payload["manifest_fingerprint"] = runner.canonical_digest(content)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                runner, "P0_CROSS2_OFFLINE_AUDIT", authorities["offline"]
            ), mock.patch.object(
                runner, "P0_CROSS2_ENTROPY3_REJECT_REPORT", authorities["reject"]
            ), mock.patch.object(
                runner, "P0_CROSS2_WIDTH2_FEASIBILITY_AUDIT",
                authorities["feasibility"],
            ):
                runner.validate_manifest_execution_binding(
                    manifest, protocol, mode, jobs
                )
                for key, authority in authorities.items():
                    with self.subTest(authority=key):
                        original = authority.read_bytes()
                        authority.write_bytes(original + b"changed")
                        with self.assertRaisesRegex(
                            RuntimeError, "authority hash mismatch"
                        ):
                            runner.validate_manifest_execution_binding(
                                manifest, protocol, mode, jobs
                            )
                        authority.write_bytes(original)

    def test_width2_1024_and_2048_formal_gates(self):
        def history(
            episodes, *, proxy=False, valid_joint=False, success=False,
            spread=0.15, crossing=0.0, bad_reward=False,
        ):
            result = []
            for index in range(episodes):
                in_tail = index >= episodes - 256
                proxy_row = proxy and in_tail and index == episodes - 1
                valid_row = valid_joint and in_tail and index == episodes - 2
                success_row = success and index == episodes // 2
                is_cross = in_tail and index < episodes - 256 + round(256 * crossing)
                flatness = 0.05 if valid_row else (0.4 if proxy_row else 1.0)
                error = 0.04 if (proxy_row or valid_row or success_row) else spread
                result.append({
                    "success": str(int(success_row)),
                    "self_crossing": str(int(is_cross)),
                    "correction_failed": "0", "no_modes": "0",
                    "wrong_flat": "0", "timeout": str(int(not is_cross)),
                    "flatness_ratio": str(flatness), "spread_error": str(error),
                    "ori6_maximum_spread_progress": "0.9" if proxy_row else "0",
                    "ori6_total_spread_frontier_reward": "0",
                    "ori6_total_joint_milestone_reward": "0",
                    "ori6_reward_decomposition_error": (
                        "1e-3" if bad_reward and index == 0 else "0"
                    ),
                    "ori6_maximum_abs_step_reward_decomposition_error": "0",
                    "ori6_nonfinite_reward_component_count": "0",
                })
            return result

        passing_1024 = {
            3101: history(1024, proxy=True, valid_joint=True),
            3102: history(1024, proxy=True),
            3103: history(1024),
        }
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", passing_1024, 1024,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (False, "width_diagnostic_continue_to_2048"),
        )
        self.assertTrue(decision["formal_width_diagnostic_gate"]["pass"])

        wide = {
            3101: history(1024, spread=0.40),
            3102: history(1024, spread=0.35),
            3103: history(1024, proxy=True, valid_joint=True),
        }
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", wide, 1024,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (True, "width_diagnostic_wide_spread_reject"),
        )
        crossing = {
            seed: history(512, crossing=0.96) for seed in (3101, 3102, 3103)
        }
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", crossing, 512,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (True, "width_diagnostic_all_seed_crossing_collapse"),
        )
        passing_1024[3101] = history(
            1024, proxy=True, valid_joint=True, bad_reward=True
        )
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", passing_1024, 1024,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (True, "reward_integrity_failure"),
        )

        passing_2048 = {
            3101: history(2048, proxy=True, success=True),
            3102: history(2048, proxy=True),
            3103: history(2048),
        }
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", passing_2048, 2048,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (False, "width_diagnostic_adoption_gate_pass"),
        )
        passing_2048[3101] = history(2048, proxy=True)
        decision = runner.early_gate_decision(
            "width_diagnostic", "water1", passing_2048, 2048,
            runner.P0_CROSS2_WIDTH2_MODE,
        )
        self.assertEqual(
            (decision["stop"], decision["reason"]),
            (True, "width_diagnostic_adoption_gate_reject"),
        )

    def test_width2_dry_run_and_execute_dispatch_are_bounded(self):
        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify",
            side_effect=lambda job, _rerun, _processes: (
                "pending", Path(job["output_directory"]), None,
            ),
        ), mock.patch.object(
            runner, "validate_execution_gate"
        ) as gate, mock.patch.object(
            runner, "validate_manifest_execution_binding"
        ) as binding, mock.patch.object(
            runner, "execute"
        ) as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                result = runner.main([
                    "--stage", "width_diagnostic", "--candidate-mode",
                    runner.P0_CROSS2_WIDTH2_MODE, "--tasks", "water1",
                    "--enable-cohort-early-stop", "--max-parallel", "3",
                ])
        self.assertEqual(result, 0)
        self.assertIn(
            "DRY RUN: 3 job(s) would launch; no training started.",
            output.getvalue(),
        )
        gate.assert_not_called()
        binding.assert_not_called()
        execute.assert_not_called()
        base = [
            "--stage", "width_diagnostic", "--candidate-mode",
            runner.P0_CROSS2_WIDTH2_MODE, "--tasks", "water1",
        ]
        with self.assertRaisesRegex(ValueError, "enable-cohort-early-stop"):
            runner.main(base)
        with self.assertRaisesRegex(ValueError, "max-parallel >= 3"):
            runner.main([
                *base, "--enable-cohort-early-stop", "--max-parallel", "2",
            ])

        protocol = runner.PROTOCOLS["width_diagnostic"]
        jobs = runner.build_jobs(
            protocol, runner.P0_CROSS2_WIDTH2_MODE, ["water1"]
        )
        pending = [(job, Path(job["output_directory"])) for job in jobs]
        with mock.patch.object(
            runner, "execute_cohort", return_value=0
        ) as execute_cohort:
            self.assertEqual(
                runner.execute(pending, early_stop=True, max_parallel=3), 0
            )
        execute_cohort.assert_called_once()
        self.assertEqual(len(execute_cohort.call_args.args[0]), 3)
        self.assertIs(execute_cohort.call_args.kwargs["early_stop"], True)

    def test_ori6_validator_requires_ori6_curve_names(self):
        job = {
            "protocol": {"episodes": 1},
            "implementation": "ori6",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = runner.validate_artifact(job, Path(directory))
        self.assertFalse(result["valid"])
        missing = result["errors"][0].removeprefix("missing: ").split(", ")
        self.assertIn("ori6_frontier_learning_curve.png", missing)
        self.assertIn("ori6_near_miss_learning_curve.png", missing)
        self.assertNotIn("near_miss_learning_curve.png", missing)

    def test_rejected_crossing_candidate_identity_commands_and_runtime_config(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        parent = runner.P0_CROSS2_MODE
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job["seed"] for job in jobs}, {3201, 3202, 3203})
        self.assertEqual(
            {job["task"]["name"] for job in jobs}, {"torus2", "water1"}
        )
        self.assertEqual(
            runner.early_gate_checkpoints(protocol.name), (512, 1024, 2048)
        )
        self.assertTrue(
            runner.expected_crossing_transition(parent)["candidate_committed"]
        )
        self.assertFalse(
            runner.expected_crossing_transition(mode)["candidate_committed"]
        )
        for profile_name in {job["profile"]["name"] for job in jobs}:
            profile = runner.PROFILES[profile_name]
            self.assertEqual(
                runner.expected_optimizer(mode, profile),
                runner.expected_optimizer(parent, profile),
            )
        for job in jobs:
            task = runner.TASKS[job["task"]["name"]]
            self.assertEqual(
                runner.expected_mechanics(mode, task),
                runner.expected_mechanics(parent, task),
            )
            command = job["command"]
            counts = Counter(
                token for token in command if token.startswith("--")
            )
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            self.assertEqual(
                counts["--ori6-rejected-crossing-continuation"], 1
            )
            self.assertEqual(
                float(command[command.index("--entropy-coefficient") + 1]),
                0.001,
            )
            self.assertEqual(
                float(command[command.index("--ori5-crossing-penalty") + 1]),
                2.0,
            )
            self.assertEqual(
                float(command[command.index("--ori6-spread-frontier-bonus") + 1]),
                0.9,
            )
            expected_width = 0.2 if task.name == "torus2" else 0.3
            self.assertEqual(
                float(command[command.index("--max-step-width") + 1]),
                expected_width,
            )
            expected_entrypoint = (
                "ori_codex6_rejected_crossing_3d.py"
                if task.is_three_dimensional
                else "ori_codex6_rejected_crossing.py"
            )
            self.assertEqual(Path(command[1]).name, expected_entrypoint)
            self.assertEqual(
                job["expected_config"]["candidate_entrypoint"],
                expected_entrypoint,
            )
            partial = runner.config_expectation(job, complete=False)
            complete = runner.config_expectation(job, complete=True)
            self.assertIn("ori6_rejected_crossing_continuation", partial)
            self.assertIn("ori6_crossing_policy", partial)
            for key in runner.REJECTED_CROSSING_POST_FINAL_CONFIG_KEYS:
                self.assertNotIn(key, partial)
                self.assertIn(key, complete)

    def test_rejected_crossing_formal_gates_and_fail_closed_integrity(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE

        def history(episodes, *, proxy=False, joint=False, success=False,
                    reject_escape=False, torus=False, bad_rate=False):
            rows = []
            for index in range(episodes):
                rejected = int(
                    reject_escape and index in (100, 500, 1010, 2030)
                )
                streaks = rejected
                escapes = rejected
                proxy_row = proxy and index == 120
                joint_row = joint and index == 121
                success_row = success and index == 200
                flatness = (
                    0.05 if joint_row else 0.35 if proxy_row else 1.0
                )
                spread = 0.04 if (proxy_row or joint_row or success_row) else 0.5
                rate = rejected / 10.0
                rows.append({
                    "episode": str(index + 1), "steps": "10",
                    "success": str(int(success_row)), "self_crossing": "0",
                    "correction_failed": "0", "no_modes": "0",
                    "wrong_flat": "0", "timeout": str(int(not success_row)),
                    "flatness_ratio": str(flatness), "spread_error": str(spread),
                    "ori6_maximum_spread_progress": "0.8" if proxy_row else "0",
                    "ori6_total_spread_frontier_reward": (
                        "0.1" if proxy_row else "0"
                    ),
                    "ori6_total_joint_milestone_reward": "0.2" if joint_row else "0",
                    "ori6_reward_decomposition_error": "0",
                    "ori6_maximum_abs_step_reward_decomposition_error": "0",
                    "ori6_nonfinite_reward_component_count": "0",
                    "ori6_candidate": runner.REJECTED_CROSSING_CANDIDATE_NAME,
                    "ori6_candidate_schema_version": (
                        runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                    ),
                    "ori6_crossing_policy": runner.REJECTED_CROSSING_POLICY,
                    "ori6_rejected_crossing_count": str(rejected),
                    "ori6_rejected_crossing_rate": str(
                        0.4 if bad_rate and rejected else rate
                    ),
                    "ori6_rejected_crossing_streak_count": str(streaks),
                    "ori6_escape_after_rejected_crossing_count": str(escapes),
                    "ori6_rejected_crossing_escape_rate": str(
                        escapes / max(streaks, 1)
                    ),
                    "ori6_unescaped_rejected_crossing_streak_count": "0",
                    "ori6_max_consecutive_rejected_crossings": str(rejected),
                    "ori6_success_after_rejected_crossing": "0",
                    "ori6_total_rejected_crossing_reward": "-1" if rejected else "0",
                    "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0",
                })
            return rows

        passing = {
            3201: history(2048, proxy=True, joint=True, success=True,
                          reject_escape=True),
            3202: history(2048, proxy=True, reject_escape=True),
            3203: history(2048),
        }
        for checkpoint, reason in (
            (512, "crossing_continuation_continue_to_1024"),
            (1024, "crossing_continuation_continue_to_2048"),
            (2048, "crossing_continuation_finalize_heldout"),
        ):
            decision = runner.early_gate_decision(
                "crossing_continuation_diagnostic", "water1", passing,
                checkpoint, mode,
            )
            self.assertFalse(decision["stop"])
            self.assertEqual(decision["reason"], reason)
            self.assertTrue(
                decision["formal_crossing_continuation_gate"]["pass"]
            )
        missing_seed = {key: value for key, value in passing.items() if key != 3203}
        decision = runner.early_gate_decision(
            "crossing_continuation_diagnostic", "water1", missing_seed,
            512, mode,
        )
        self.assertTrue(decision["stop"])
        self.assertFalse(decision["cohort_shape_valid"])
        broken = dict(passing)
        broken[3201] = history(
            1024, proxy=True, joint=True, reject_escape=True, bad_rate=True
        )
        decision = runner.early_gate_decision(
            "crossing_continuation_diagnostic", "water1", broken,
            1024, mode,
        )
        self.assertTrue(decision["stop"])
        self.assertFalse(
            decision["formal_crossing_continuation_gate"]
            ["reward_and_transition_integrity_pass"]
        )
        torus_no_reject = {
            3201: history(1024, proxy=True, joint=True, torus=True),
            3202: history(1024, proxy=True, torus=True),
            3203: history(1024, torus=True),
        }
        decision = runner.early_gate_decision(
            "crossing_continuation_diagnostic", "torus2",
            torus_no_reject, 1024, mode,
        )
        self.assertFalse(decision["stop"])
        self.assertTrue(
            decision["formal_crossing_continuation_gate"]["pass"]
        )

    def test_rejected_crossing_dry_run_and_two_fixed_cohorts(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        base = [
            "--stage", "crossing_continuation_diagnostic",
            "--candidate-mode", mode, "--tasks", "torus2", "water1",
        ]
        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify", side_effect=lambda job, _rerun, _processes: (
                "pending", Path(job["output_directory"]), None
            )
        ), mock.patch.object(runner, "validate_execution_gate") as gate, \
             mock.patch.object(runner, "validate_manifest_execution_binding") as binding, \
             mock.patch.object(runner, "execute") as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                result = runner.main([
                    *base, "--enable-cohort-early-stop", "--max-parallel", "3"
                ])
        self.assertEqual(result, 0)
        self.assertIn("DRY RUN: 6 job(s) would launch", output.getvalue())
        gate.assert_not_called()
        binding.assert_not_called()
        execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "enable-cohort-early-stop"):
            runner.main(base)
        with self.assertRaisesRegex(ValueError, "max-parallel >= 3"):
            runner.main([
                *base, "--enable-cohort-early-stop", "--max-parallel", "2"
            ])
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        pending = [(job, Path(job["output_directory"])) for job in jobs]
        with mock.patch.object(
            runner, "execute_cohort", return_value=0
        ) as execute_cohort:
            self.assertEqual(runner.execute(pending, True, 3), 0)
        self.assertEqual(execute_cohort.call_count, 2)
        self.assertEqual(
            [
                {item[0]["task"]["name"] for item in call.args[0]}
                for call in execute_cohort.call_args_list
            ],
            [{"torus2"}, {"water1"}],
        )

    def test_rejected_crossing_manifest_and_W2_authority_fail_closed(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        manifest = runner.manifest_path(protocol, mode, jobs)
        runner.validate_manifest_execution_binding(
            manifest, protocol, mode, jobs
        )
        self.assertIsNone(runner.validate_execution_gate(
            protocol, mode, runner.EXPERIMENT_ROOT / "analysis" / "unused.json"
        ))
        runner.validate_P0X2W2_reject_authority(
            runner.P0_CROSS2_WIDTH2_REJECT_REPORT
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        mutations = (
            ("candidate_additive_source_sha256", {"bad.py": "0" * 64}),
            ("candidate_orchestration_source_sha256", {}),
            ("early_gate_specification", {}),
            ("parent_saved_path_scope", "overclaim crossing-path ordering"),
            ("preceding_candidate_reject_authority_sha256", "0" * 64),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in mutations:
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(payload))
                    changed[field] = value
                    content = dict(changed)
                    content.pop("manifest_fingerprint", None)
                    content.pop("created_utc", None)
                    changed["manifest_fingerprint"] = runner.canonical_digest(
                        content
                    )
                    path = root / f"{field}.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "Execution manifest authority rejected"
                    ):
                        runner.validate_manifest_execution_binding(
                            path, protocol, mode, jobs
                        )
            original_hash = runner.file_sha256

            def changed_W2_hash(path):
                if Path(path).resolve() == (
                    runner.P0_CROSS2_WIDTH2_REJECT_REPORT.resolve()
                ):
                    return "f" * 64
                return original_hash(Path(path))

            with mock.patch.object(
                runner, "file_sha256", side_effect=changed_W2_hash
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "authority hash mismatch"
                ):
                    runner.validate_manifest_execution_binding(
                        manifest, protocol, mode, jobs
                    )

    def test_rejected_crossing_rejects_subset_and_mixed_fixed_cohort(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        with self.assertRaisesRegex(ValueError, "fixed torus2\+water1"):
            runner.build_jobs(protocol, mode, ["torus2"])
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        partial_pending = [
            (job, Path(job["output_directory"]))
            for job in jobs if not (
                job["task"]["name"] == "torus2" and job["seed"] == 3203
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "partial pending seed cohort"):
            runner.execute(partial_pending, early_stop=True, max_parallel=3)

        def mixed_state(job, _rerun, _processes):
            state = (
                "complete"
                if job["task"]["name"] == "torus2" and job["seed"] == 3201
                else "pending"
            )
            return state, Path(job["output_directory"]), None

        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify", side_effect=mixed_state
        ):
            with self.assertRaisesRegex(RuntimeError, "mixed complete/pending"):
                runner.main([
                    "--stage", "crossing_continuation_diagnostic",
                    "--candidate-mode", mode,
                    "--tasks", "torus2", "water1",
                    "--enable-cohort-early-stop", "--max-parallel", "3",
                ])

    def test_rejected_crossing_final_checkpoint_gate_survives_finish_race(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        jobs = [
            job for job in runner.build_jobs(
                protocol, mode, ["torus2", "water1"]
            )
            if job["task"]["name"] == "water1"
        ]
        fake_items = []
        for job in jobs:
            stream = mock.Mock()
            fake_items.append({
                "job": job,
                "destination": Path(job["output_directory"]),
                "process": _FakeProcess(0),
                "stdout": stream,
                "stderr": stream,
            })
        checkpoint_calls = {"count": 0}

        def checkpoint_after_poll(_path):
            checkpoint_calls["count"] += 1
            # Three short-circuited unavailable checks (512/1024/2048), then the
            # final post-finish sweep observes episode 2048.
            return None if checkpoint_calls["count"] <= 3 else 2048

        decisions = []

        def gate_decision(protocol_name, task_name, _rows, episode,
                          candidate_mode):
            decision = {
                "protocol": protocol_name, "task": task_name,
                "candidate_mode": candidate_mode,
                "checkpoint_episode": episode,
                "stop": False, "reason": f"pass_{episode}",
            }
            decisions.append(decision)
            return decision

        launched = iter(fake_items)
        with mock.patch.object(
            runner, "launch", side_effect=lambda *_args: next(launched)
        ), mock.patch.object(
            runner, "current_output", return_value=Path("effective")
        ), mock.patch.object(
            runner, "checkpoint_episode", side_effect=checkpoint_after_poll
        ), mock.patch.object(
            runner, "read_csv", return_value=[]
        ), mock.patch.object(
            runner, "early_gate_decision", side_effect=gate_decision
        ), mock.patch.object(
            runner, "finish", side_effect=lambda item: {
                "status": "complete", "job_id": item["job"]["job_id"],
                "return_code": 0,
            }
        ), mock.patch.object(runner, "atomic_json") as atomic:
            self.assertEqual(
                runner.execute_cohort(
                    [(job, Path(job["output_directory"])) for job in jobs],
                    early_stop=True,
                ),
                0,
            )
        self.assertEqual(
            [row["checkpoint_episode"] for row in decisions],
            [512, 1024, 2048],
        )
        self.assertEqual(atomic.call_count, 3)

    def test_launch_writes_status_below_max_path_without_precreating_output(self):
        with tempfile.TemporaryDirectory(
            dir=runner.SCRIPT_DIRECTORY, prefix="maxpath_regression_"
        ) as directory:
            experiment_root = Path(directory).resolve()
            with mock.patch.object(
                runner, "EXPERIMENT_ROOT", experiment_root
            ):
                jobs = runner.build_jobs(
                    runner.PROTOCOLS["crossing_continuation_diagnostic"],
                    runner.P0_CROSS2_REJECTED_CONTINUATION_MODE,
                    ["torus2", "water1"],
                )
                job = jobs[0]
                destination = Path(job["output_directory"])
                log_directory = runner.runner_directory(job, destination)
                status_file = log_directory / "status.json"
                legacy_temporary = status_file.with_name(
                    f".{status_file.name}.{'a' * 32}.tmp"
                )
                shortened_temporary = status_file.with_name(
                    f".tmp-{'a' * 16}"
                )
                self.assertGreater(len(str(legacy_temporary)), 260)
                self.assertLessEqual(len(str(shortened_temporary)), 254)
                self.assertFalse(destination.exists())
                fake_process = mock.Mock(pid=43210)
                with mock.patch.object(
                    runner.subprocess, "Popen", return_value=fake_process
                ) as popen:
                    item = runner.launch(job, destination)
                try:
                    self.assertTrue(log_directory.is_dir())
                    self.assertTrue(status_file.is_file())
                    status = json.loads(status_file.read_text(encoding="utf-8"))
                    self.assertEqual(status["status"], "running")
                    self.assertEqual(status["pid"], 43210)
                    self.assertFalse(destination.exists())
                    self.assertFalse(list(log_directory.glob(".tmp-*")))
                    popen.assert_called_once()
                finally:
                    item["stdout"].close()
                    item["stderr"].close()
    def test_checkpoint_gate_keeps_a_seed_that_finished_early(self):
        checkpoints = {3101: 2048, 3102: 0, 3103: 0}
        items = []
        for seed in (3101, 3102, 3103):
            process = _FakeProcess(0 if seed == 3101 else None)
            job = {
                "job_id": f"rescue:P0:ori6:water1:B:seed{seed}",
                "seed": seed,
                "implementation": "ori6",
                "candidate_mode": runner.P0_MODE,
                "protocol": {"name": "rescue"},
                "task": {"name": "water1"},
            }
            items.append({
                "job": job,
                "process": process,
                "output": Path(str(seed)),
            })

        seen_seed_sets = []
        terminated_active_seed_sets = []

        def fake_current_output(item):
            return item["output"]

        def fake_checkpoint_episode(output):
            return checkpoints[int(output.name)]

        def fake_read_csv(path):
            return [{"seed": int(path.parent.name)}]

        def fake_decision(protocol, task, rows_by_seed, checkpoint, candidate_mode):
            seen_seed_sets.append(set(rows_by_seed))
            return {"stop": True, "reason": "test", "checkpoint": checkpoint}

        def fake_finish(item):
            return {"status": "complete", "job_id": item["job"]["job_id"], "return_code": 0}

        def fake_sleep(_seconds):
            checkpoints[3102] = 512
            checkpoints[3103] = 512

        def fake_terminate(active_items, _episode, _decision):
            terminated_active_seed_sets.append({item["job"]["seed"] for item in active_items})

        with mock.patch.object(runner, "launch", side_effect=items), \
             mock.patch.object(runner, "current_output", side_effect=fake_current_output), \
             mock.patch.object(runner, "checkpoint_episode", side_effect=fake_checkpoint_episode), \
             mock.patch.object(runner, "read_csv", side_effect=fake_read_csv), \
             mock.patch.object(runner, "early_gate_decision", side_effect=fake_decision), \
             mock.patch.object(runner, "finish", side_effect=fake_finish), \
             mock.patch.object(runner, "terminate_at_checkpoint", side_effect=fake_terminate), \
             mock.patch.object(runner, "atomic_json"), \
             mock.patch.object(runner.time, "sleep", side_effect=fake_sleep):
            failures = runner.execute_cohort(
                [(item["job"], Path(str(item["job"]["seed"]))) for item in items],
                early_stop=True,
            )

        self.assertEqual(failures, 0)
        self.assertEqual(seen_seed_sets, [{3101, 3102, 3103}])
        self.assertEqual(terminated_active_seed_sets, [{3102, 3103}])

    def test_full_gate_rehashes_rescue_and_regression_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_directory = root / "analysis"
            analysis_directory.mkdir()
            rescue = analysis_directory / "rescue_P0_report.json"
            regression = analysis_directory / "regression_P0_report.json"
            rescue.write_text('{"gate": "pass"}', encoding="utf-8")
            regression.write_text('{"gate": "pass"}', encoding="utf-8")
            source_hashes = {"source.py": "source-sha256"}
            model_hashes = {name: {"model": name} for name in runner.TASKS}
            selection = {
                "selected_candidate_mode": runner.P0_MODE,
                "rescue_gate_pass": True,
                "successful_task_regression_gate_pass": True,
                "source_sha256": source_hashes,
                "model_sha256_by_task": model_hashes,
                "random_baseline_authority_sha256": "random-sha256",
                "rescue_report": str(rescue.resolve()),
                "rescue_report_sha256": runner.file_sha256(rescue),
                "regression_report": str(regression.resolve()),
                "regression_report_sha256": runner.file_sha256(regression),
            }
            selection_path = analysis_directory / "selected_candidate.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with mock.patch.object(runner, "EXPERIMENT_ROOT", root), \
                 mock.patch.object(runner, "source_hashes", return_value=source_hashes), \
                 mock.patch.object(
                     runner, "model_hashes",
                     side_effect=lambda task: model_hashes[task.name],
                 ), \
                 mock.patch.object(
                     runner, "random_baseline_authority",
                     return_value={"sha256": "random-sha256"},
                 ):
                accepted = runner.validate_execution_gate(
                    runner.PROTOCOLS["full_primary"], runner.P0_MODE,
                    selection_path,
                )
                self.assertEqual(accepted, selection)
                regression.write_text('{"gate": "changed"}', encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "regression report sha256 mismatch"
                ):
                    runner.validate_execution_gate(
                        runner.PROTOCOLS["full_primary"], runner.P0_MODE,
                        selection_path,
                    )


    @staticmethod
    def _spectral_history(episodes, *, proxy=False, joint=False,
                          success=False, spectral=True, corrupt=False):
        rows = []
        for index in range(episodes):
            proxy_row = proxy and index == 120
            joint_row = joint and index == 121
            success_row = success and index == 200
            evidence = spectral and index == 100
            flatness = 0.05 if joint_row else 0.35 if proxy_row else 1.0
            spread = 0.04 if (proxy_row or joint_row or success_row) else 0.5
            spectral_total = 0.001 if evidence else 0.0
            row = {
                "episode": str(index + 1), "steps": "10",
                "success": str(int(success_row)), "self_crossing": "0",
                "correction_failed": "0", "no_modes": "0",
                "wrong_flat": "0", "timeout": str(int(not success_row)),
                "termination_reason": "success" if success_row else "timeout",
                "flatness_ratio": str(flatness), "spread_error": str(spread),
                "ori6_maximum_spread_progress": "0.8" if proxy_row else "0",
                "ori6_total_spread_frontier_reward": "0.1" if proxy_row else "0",
                "ori6_total_joint_milestone_reward": "0.2" if joint_row else "0",
                "ori6_reward_decomposition_error": "0",
                "ori6_maximum_abs_step_reward_decomposition_error": "0",
                "ori6_nonfinite_reward_component_count": "0",
                "ori6_candidate": runner.REJECTED_CROSSING_CANDIDATE_NAME,
                "ori6_candidate_schema_version": (
                    runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                ),
                "ori6_crossing_policy": runner.REJECTED_CROSSING_POLICY,
                "ori6_rejected_crossing_count": "0",
                "ori6_rejected_crossing_rate": "0",
                "ori6_rejected_crossing_streak_count": "0",
                "ori6_escape_after_rejected_crossing_count": "0",
                "ori6_rejected_crossing_escape_rate": "0",
                "ori6_unescaped_rejected_crossing_streak_count": "0",
                "ori6_max_consecutive_rejected_crossings": "0",
                "ori6_success_after_rejected_crossing": "0",
                "ori6_total_rejected_crossing_reward": "0",
                "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0",
                "ori6_spectral_candidate": runner.SPECTRAL_SUBSPACE_CANDIDATE_NAME,
                "ori6_spectral_candidate_schema_version": (
                    runner.SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
                ),
                "ori6_spectral_formula_sha256": (
                    runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
                ),
                "ori6_spectral_parent_candidate": (
                    runner.REJECTED_CROSSING_CANDIDATE_NAME
                ),
                "ori6_spectral_weight": "0.05",
                "ori6_spectral_relative_epsilon": "1e-12",
                "ori6_spectral_total_progress_reward": str(spectral_total),
                "ori6_spectral_total_rollback_reward": "0",
                "ori6_spectral_total_reward": str(spectral_total),
                "ori6_spectral_cumulative_reward": str(spectral_total),
                "ori6_spectral_terminal_potential": "0.02" if evidence else "0",
                "ori6_spectral_terminal_delta": "0.1" if evidence else "0",
                "ori6_spectral_terminal_rho": "0.2" if evidence else "0",
                "ori6_spectral_maximum_potential": "0.02" if evidence else "0",
                "ori6_spectral_maximum_delta": "0.1" if evidence else "0",
                "ori6_spectral_maximum_rho": "0.2" if evidence else "0",
                "ori6_spectral_reward_decomposition_error": (
                    "1" if corrupt and index == 0 else "0"
                ),
                "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
                "ori6_spectral_nonfinite_component_count": "0",
            }
            rows.append(row)
        return rows

    def test_spectral_candidate_is_exact_bounded_RC_addition(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        parent = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        protocol = runner.PROTOCOLS["spectral_subspace_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job["seed"] for job in jobs}, {3201, 3202, 3203})
        self.assertEqual(
            runner.candidate_reward_sha256(mode),
            "0c148e1eaada0da0d7ac1781abd2157fc36b516a12459e194139e843623794de",
        )
        self.assertNotEqual(
            runner.candidate_reward_sha256(mode),
            runner.candidate_reward_sha256(parent),
        )
        self.assertEqual(
            runner.expected_crossing_transition(mode),
            runner.expected_crossing_transition(parent),
        )
        for job in jobs:
            task = runner.TASKS[job["task"]["name"]]
            profile = runner.PROFILES[job["profile"]["name"]]
            self.assertEqual(
                runner.expected_optimizer(mode, profile),
                runner.expected_optimizer(parent, profile),
            )
            self.assertEqual(
                runner.expected_mechanics(mode, task),
                runner.expected_mechanics(parent, task),
            )
            self.assertEqual(
                Path(job["command"][1]).name,
                "ori_codex6_spectral_subspace_3d.py",
            )
            counts = Counter(
                token for token in job["command"] if token.startswith("--")
            )
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            for flag in (
                "--ori6-rejected-crossing-continuation",
                "--ori6-spectral-subspace",
                "--ori6-spectral-weight",
                "--ori6-spectral-relative-epsilon",
            ):
                self.assertEqual(counts[flag], 1)
            partial = runner.config_expectation(job, complete=False)
            complete = runner.config_expectation(job, complete=True)
            self.assertTrue(partial["ori6_spectral_subspace"])
            self.assertEqual(partial["ori6_spectral_weight"], 0.05)
            for key in runner.SPECTRAL_SUBSPACE_POST_FINAL_CONFIG_KEYS:
                self.assertNotIn(key, partial)
                self.assertIn(key, complete)

    def test_spectral_512_is_integrity_only_then_1024_and_2048_are_constructive(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        no_signal = {
            seed: self._spectral_history(512, spectral=False)
            for seed in (3201, 3202, 3203)
        }
        decision = runner.early_gate_decision(
            "spectral_subspace_diagnostic", "torus2", no_signal, 512, mode,
        )
        self.assertFalse(decision["stop"])
        self.assertEqual(
            decision["reason"],
            "spectral_subspace_integrity_continue_to_1024",
        )
        self.assertEqual(
            decision["formal_spectral_subspace_gate"][
                "performance_predicate_count"
            ], 0,
        )
        passing = {
            3201: self._spectral_history(
                2048, proxy=True, joint=True, success=True
            ),
            3202: self._spectral_history(2048, proxy=True),
            3203: self._spectral_history(2048, spectral=False),
        }
        with mock.patch.object(
            runner, "spectral_parent_common_prefix_comparison",
            return_value={"gating": False, "available": False},
        ):
            decision1024 = runner.early_gate_decision(
                "spectral_subspace_diagnostic", "torus2", passing,
                1024, mode,
            )
        self.assertFalse(decision1024["stop"])
        self.assertTrue(
            decision1024["formal_spectral_subspace_gate"]["pass"]
        )
        self.assertFalse(
            decision1024["formal_spectral_subspace_gate"][
                "parent_comparison_gating"
            ]
        )
        decision2048 = runner.early_gate_decision(
            "spectral_subspace_diagnostic", "torus2", passing, 2048, mode,
        )
        self.assertFalse(decision2048["stop"])
        broken = dict(passing)
        broken[3201] = self._spectral_history(512, corrupt=True)
        decision = runner.early_gate_decision(
            "spectral_subspace_diagnostic", "torus2", broken, 512, mode,
        )
        self.assertTrue(decision["stop"])
        self.assertFalse(
            decision["formal_spectral_subspace_gate"][
                "reward_spectral_and_transition_integrity_pass"
            ]
        )

    def test_spectral_authorities_manifest_and_dry_run_fail_closed(self):
        runner.validate_P0X2RC_reject_authority(
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT
        )
        runner.validate_spectral_subspace_offline_authority(
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY
        )
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_subspace_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        manifest = runner.manifest_path(protocol, mode, jobs)
        runner.validate_manifest_execution_binding(
            manifest, protocol, mode, jobs
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["parent_RC_reject_authority_sha256"],
            runner.P0_CROSS2_REJECTED_CONTINUATION_REJECT_REPORT_SHA256,
        )
        self.assertEqual(
            payload["spectral_offline_design_authority_sha256"],
            runner.SPECTRAL_SUBSPACE_OFFLINE_AUTHORITY_SHA256,
        )
        changed = json.loads(json.dumps(payload))
        changed["early_gate_specification"] = {}
        content = dict(changed)
        content.pop("manifest_fingerprint")
        content.pop("created_utc")
        changed["manifest_fingerprint"] = runner.canonical_digest(content)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "spectral early-gate"):
                runner.validate_manifest_execution_binding(
                    path, protocol, mode, jobs
                )
        base = [
            "--stage", "spectral_subspace_diagnostic",
            "--candidate-mode", mode,
            "--tasks", "torus2", "water1",
            "--enable-cohort-early-stop", "--max-parallel", "3",
        ]
        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify", side_effect=lambda job, _rerun, _processes: (
                "pending", Path(job["output_directory"]), None
            )
        ), mock.patch.object(runner, "validate_execution_gate") as gate, \
             mock.patch.object(runner, "validate_manifest_execution_binding") as binding, \
             mock.patch.object(runner, "execute") as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(runner.main(base), 0)
        self.assertIn("DRY RUN: 6 job(s) would launch", output.getvalue())
        gate.assert_not_called()
        binding.assert_not_called()
        execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "fixed torus2\+water1"):
            runner.build_jobs(protocol, mode, ["torus2"])

    def test_spectral_nonregression_has_exact_18_paired_jobs(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_nonregression"]
        jobs = runner.build_jobs(protocol, mode, list(protocol.tasks))
        self.assertEqual(len(jobs), 18)
        self.assertEqual({job["seed"] for job in jobs}, {3301, 3302, 3303})
        self.assertEqual(
            {job["task"]["name"] for job in jobs},
            {"five_d2", "five_d3", "watersqrt2"},
        )
        for task_name in protocol.tasks:
            selected = [job for job in jobs if job["task"]["name"] == task_name]
            self.assertEqual(len(selected), 6)
            self.assertEqual(
                {(job["implementation"], job["seed"]) for job in selected},
                {
                    (implementation, seed)
                    for implementation in ("ori6", "ori5_reference")
                    for seed in (3301, 3302, 3303)
                },
            )
            task = runner.TASKS[task_name]
            profile = runner.PROFILES[task.profile]
            self.assertEqual(
                runner.expected_optimizer(mode, profile),
                runner.expected_optimizer(
                    runner.P0_CROSS2_REJECTED_CONTINUATION_MODE, profile
                ),
            )
            self.assertEqual(
                runner.expected_mechanics(mode, task),
                runner.expected_mechanics(
                    runner.P0_CROSS2_REJECTED_CONTINUATION_MODE, task
                ),
            )
            for seed in protocol.seeds:
                candidate_job = next(
                    job for job in selected
                    if job["implementation"] == "ori6"
                    and job["seed"] == seed
                )
                reference_job = next(
                    job for job in selected
                    if job["implementation"] == "ori5_reference"
                    and job["seed"] == seed
                )
                for field in (
                    "episodes", "eval_episodes", "eval_interval",
                    "periodic_stochastic_eval_episodes", "batch_size",
                    "max_steps", "max_step_width",
                    "flatness_subspace_dimension", "target_spread",
                    "actor_learning_rate", "critic_learning_rate",
                    "entropy_coefficient", "actor_lr_milestones",
                    "actor_lr_factors", "checkpoint_every", "seed",
                    "selection_evaluation_seed", "final_evaluation_seed",
                    "gamma", "flatness_tolerance", "spread_tolerance",
                    "strain_tolerance", "crossing_tolerance",
                    "min_step_width", "projection_shrink",
                    "max_correction_ratio", "max_actual_step_ratio",
                    "terminate_wrong_flat",
                ):
                    self.assertEqual(
                        candidate_job["expected_config"][field],
                        reference_job["expected_config"][field],
                        f"{task_name}/seed{seed}/{field}",
                    )
                self.assertEqual(
                    candidate_job["candidate_optimizer_sha256"],
                    reference_job["candidate_optimizer_sha256"],
                )
                self.assertEqual(
                    candidate_job["candidate_mechanics_sha256"],
                    reference_job["candidate_mechanics_sha256"],
                )
        for job in jobs:
            command = job["command"]
            counts = Counter(
                token for token in command if token.startswith("--")
            )
            self.assertFalse({flag: n for flag, n in counts.items() if n > 1})
            self.assertEqual(job["protocol"]["episodes"], 2048)
            self.assertEqual(job["protocol"]["checkpoint_every"], 512)
            self.assertEqual(job["protocol"]["final_evaluation_episodes"], 64)
            task_name = job["task"]["name"]
            if job["implementation"] == "ori6":
                expected = (
                    "ori_codex6_spectral_subspace_3d.py"
                    if task_name == "watersqrt2"
                    else "ori_codex6_spectral_subspace.py"
                )
                self.assertEqual(Path(command[1]).name, expected)
                self.assertEqual(
                    counts["--ori6-rejected-crossing-continuation"], 1
                )
                self.assertEqual(counts["--ori6-spectral-subspace"], 1)
            else:
                expected = (
                    "ori_codex5_3d.py"
                    if task_name == "watersqrt2" else "ori_codex5.py"
                )
                self.assertEqual(Path(command[1]).name, expected)
                self.assertNotIn("--ori6-rejected-crossing-continuation", command)
                self.assertNotIn("--ori6-spectral-subspace", command)

    def test_spectral_nonregression_512_gate_is_catastrophe_only(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        candidate = {
            seed: self._spectral_history(512, spectral=True)
            for seed in (3301, 3302, 3303)
        }
        reference = {
            seed: [
                {
                    "episode": str(index + 1), "success": "0",
                    "self_crossing": "0",
                    "ori5_reward_decomposition_error": "0",
                    "ori5_maximum_abs_step_reward_decomposition_error": "0",
                    "ori5_nonfinite_reward_component_count": "0",
                }
                for index in range(512)
            ]
            for seed in (3301, 3302, 3303)
        }
        decision = runner.early_gate_decision(
            "spectral_nonregression", "five_d2", candidate, 512, mode,
            reference_rows_by_seed=reference,
        )
        formal = decision["formal_spectral_nonregression_gate"]
        self.assertTrue(formal["pass"])
        self.assertFalse(formal["performance_noninferiority_tested"])
        self.assertFalse(decision["stop"])

        catastrophic = {
            seed: json.loads(json.dumps(rows))
            for seed, rows in candidate.items()
        }
        for rows in catastrophic.values():
            for row in rows[-256:]:
                row.update({
                    "ori6_rejected_crossing_count": "10",
                    "ori6_rejected_crossing_rate": "1",
                    "ori6_rejected_crossing_streak_count": "1",
                    "ori6_escape_after_rejected_crossing_count": "0",
                    "ori6_rejected_crossing_escape_rate": "0",
                    "ori6_unescaped_rejected_crossing_streak_count": "1",
                    "ori6_max_consecutive_rejected_crossings": "10",
                })
        decision = runner.early_gate_decision(
            "spectral_nonregression", "five_d2", catastrophic, 512, mode,
            reference_rows_by_seed=reference,
        )
        self.assertTrue(decision["stop"])
        self.assertEqual(
            decision["reason"],
            "spectral_nonregression_catastrophic_reject",
        )
        corrupt_reference = {
            seed: json.loads(json.dumps(rows))
            for seed, rows in reference.items()
        }
        corrupt_reference[3301][0]["ori5_nonfinite_reward_component_count"] = "1"
        decision = runner.early_gate_decision(
            "spectral_nonregression", "five_d2", candidate, 512, mode,
            reference_rows_by_seed=corrupt_reference,
        )
        self.assertTrue(decision["stop"])
        self.assertEqual(
            decision["reason"], "spectral_nonregression_integrity_failure"
        )

    def test_spectral_formal_report_binds_regression_but_not_full(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_nonregression"]
        authority = runner.validate_spectral_screen_report_authority(
            runner.SPECTRAL_SUBSPACE_SCREEN_REPORT
        )
        self.assertFalse(authority["spectral_subspace_screen_pass"])
        self.assertFalse(
            authority["full_10000_episode_training_authorized_by_this_report"]
        )
        self.assertEqual(
            authority["decision"],
            "formal_gate_reject_but_promising_signal_for_regression",
        )
        binding = runner.spectral_screen_report_binding()
        self.assertEqual(
            binding["status"], "bound_regression_permission_authority"
        )
        self.assertEqual(
            binding["sha256"],
            runner.SPECTRAL_SUBSPACE_SCREEN_REPORT_SHA256,
        )
        jobs = runner.build_jobs(protocol, mode, list(protocol.tasks))
        with mock.patch.object(runner, "record_stale_manifests"):
            manifest = runner.manifest_path(protocol, mode, jobs)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["spectral_screen_report_status"],
            "bound_regression_permission_authority",
        )
        self.assertTrue(
            payload["promotion_requirements"][
                "formal_spectral_report_explicitly_permits_"
                "preregistered_regression"
            ]
        )
        self.assertTrue(
            payload["promotion_requirements"][
                "formal_spectral_report_does_not_authorize_adoption_or_full"
            ]
        )
        runner.validate_manifest_execution_binding(
            manifest, protocol, mode, jobs
        )
        runner.validate_execution_gate(
            protocol, mode,
            runner.EXPERIMENT_ROOT / "analysis" / "selected_candidate.json",
        )

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "formal_report.json"
            copied.write_bytes(runner.SPECTRAL_SUBSPACE_SCREEN_REPORT.read_bytes())
            with mock.patch.object(
                runner, "SPECTRAL_SUBSPACE_SCREEN_REPORT", copied
            ):
                runner.validate_spectral_screen_report_authority(copied)
                changed = json.loads(copied.read_text(encoding="utf-8"))
                changed["candidate_adopted"] = True
                copied.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fixed SHA256"):
                    runner.validate_spectral_screen_report_authority(copied)

    def test_spectral_nonregression_pending_manifest_is_dry_run_only(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_nonregression"]
        jobs = runner.build_jobs(protocol, mode, list(protocol.tasks))
        with mock.patch.object(
            runner, "SPECTRAL_SUBSPACE_SCREEN_REPORT",
            runner.EXPERIMENT_ROOT / "analysis" / "definitely_missing.json",
        ), mock.patch.object(runner, "record_stale_manifests"):
            manifest = runner.manifest_path(protocol, mode, jobs)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["spectral_screen_report_status"],
                "pending_not_launchable",
            )
            self.assertIsNone(payload["spectral_screen_report_sha256"])
            with self.assertRaisesRegex(
                RuntimeError, "spectral screen authority is pending"
            ):
                runner.validate_manifest_execution_binding(
                    manifest, protocol, mode, jobs
                )
            with self.assertRaises(FileNotFoundError):
                runner.validate_execution_gate(
                    protocol, mode,
                    runner.EXPERIMENT_ROOT / "analysis" / "selected_candidate.json",
                )

        base = [
            "--stage", "spectral_nonregression",
            "--candidate-mode", mode,
            "--tasks", "five_d2", "five_d3", "watersqrt2",
            "--enable-cohort-early-stop", "--max-parallel", "12",
        ]
        with mock.patch.object(
            runner, "manifest_path", return_value=Path("manifest.json")
        ), mock.patch.object(
            runner, "active_python_processes", return_value=[]
        ), mock.patch.object(
            runner, "classify", side_effect=lambda job, _rerun, _processes: (
                "pending", Path(job["output_directory"]), None
            )
        ), mock.patch.object(runner, "validate_execution_gate") as gate, \
             mock.patch.object(runner, "validate_manifest_execution_binding") as binding, \
             mock.patch.object(runner, "execute") as execute:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(runner.main(base), 0)
        self.assertIn("DRY RUN: 18 job(s) would launch", output.getvalue())
        gate.assert_not_called()
        binding.assert_not_called()
        execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "fixed paired"):
            runner.main([
                "--stage", "spectral_nonregression",
                "--candidate-mode", mode,
                "--tasks", "five_d2",
                "--enable-cohort-early-stop", "--max-parallel", "12",
            ])


if __name__ == "__main__":
    unittest.main()
