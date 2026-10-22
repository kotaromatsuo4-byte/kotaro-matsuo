import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_ori_codex6_experiments as analysis
import analyze_ori_codex6_frontier12_counterfactual as frontier12_audit
import run_ori_codex6_experiments as runner


class Ori6AnalyzerTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(directory: Path, name: str, payload: dict) -> Path:
        content = dict(payload)
        content["manifest_fingerprint"] = runner.canonical_digest(payload)
        content["created_utc"] = "2026-01-01T00:00:00+00:00"
        path = directory / name
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _manifest_payload(
        offline: Path, p0_evidence: Path, offline_sha256: str, **extra,
    ) -> dict:
        return {
            "schema_version": 1,
            "protocol": {"name": "rescue"},
            "candidate_mode": runner.P0_CROSS2_MODE,
            "candidate_mode_slug": "P0X2",
            "jobs": [{"fingerprint": "job-fingerprint"}],
            "source_sha256": {"source.py": "source-sha256"},
            "core_frontier_mode": runner.P0_MODE,
            "candidate_reward_sha256": runner.candidate_reward_sha256(
                runner.P0_CROSS2_MODE
            ),
            "single_changed_reward_parameter": {
                "name": "ori5_crossing_penalty",
                "P0_value": 1.25,
                "candidate_value": 2.0,
            },
            "offline_saved_path_ordering_authority": str(offline.resolve()),
            "offline_saved_path_ordering_authority_sha256": offline_sha256,
            "P0_crossing_attractor_authority": str(p0_evidence.resolve()),
            "P0_crossing_attractor_authority_sha256": runner.file_sha256(
                p0_evidence
            ),
            **extra,
        }

    def _authority_patches(self, root: Path, offline: Path, evidence: Path):
        return (
            mock.patch.object(runner, "EXPERIMENT_ROOT", root),
            mock.patch.object(runner, "P0_CROSS2_OFFLINE_AUDIT", offline),
            mock.patch.object(runner, "P0_RECLASSIFICATION_REPORT", evidence),
            mock.patch.object(
                runner, "source_hashes",
                return_value={"source.py": "source-sha256"},
            ),
        )

    def test_current_manifest_selects_current_P0X2_evidence_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            manifests.mkdir()
            offline = root / "offline.json"
            evidence = root / "P0.json"
            offline.write_text("offline-current", encoding="utf-8")
            evidence.write_text("P0-current", encoding="utf-8")
            stale = self._write_manifest(
                manifests, "rescue_P0X2_stale.json",
                self._manifest_payload(offline, evidence, "0" * 64),
            )
            current = self._write_manifest(
                manifests, "rescue_P0X2_current.json",
                self._manifest_payload(
                    offline, evidence, runner.file_sha256(offline)
                ),
            )
            before = {
                path: (runner.file_sha256(path), path.stat().st_mtime_ns)
                for path in (stale, current)
            }
            patches = self._authority_patches(root, offline, evidence)
            with patches[0], patches[1], patches[2], patches[3]:
                authority = analysis.current_manifest_authority(
                    "rescue", runner.P0_CROSS2_MODE,
                    [{"fingerprint": "job-fingerprint"}],
                )
            self.assertEqual(Path(authority["path"]), current.resolve())
            after = {
                path: (runner.file_sha256(path), path.stat().st_mtime_ns)
                for path in (stale, current)
            }
            self.assertEqual(before, after)

    def test_current_manifest_rejects_true_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            manifests.mkdir()
            offline = root / "offline.json"
            evidence = root / "P0.json"
            offline.write_text("offline-current", encoding="utf-8")
            evidence.write_text("P0-current", encoding="utf-8")
            digest = runner.file_sha256(offline)
            self._write_manifest(
                manifests, "rescue_P0X2_first.json",
                self._manifest_payload(offline, evidence, digest, nonce=1),
            )
            self._write_manifest(
                manifests, "rescue_P0X2_second.json",
                self._manifest_payload(offline, evidence, digest, nonce=2),
            )
            patches = self._authority_patches(root, offline, evidence)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(RuntimeError, "found 2"):
                    analysis.current_manifest_authority(
                        "rescue", runner.P0_CROSS2_MODE,
                        [{"fingerprint": "job-fingerprint"}],
                    )

    def test_current_manifest_ignores_tampered_self_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            manifests.mkdir()
            offline = root / "offline.json"
            evidence = root / "P0.json"
            offline.write_text("offline-current", encoding="utf-8")
            evidence.write_text("P0-current", encoding="utf-8")
            path = self._write_manifest(
                manifests, "rescue_P0X2_tampered.json",
                self._manifest_payload(
                    offline, evidence, runner.file_sha256(offline)
                ),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest_fingerprint"] = "0" * 64
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            patches = self._authority_patches(root, offline, evidence)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(RuntimeError, "found 0"):
                    analysis.current_manifest_authority(
                        "rescue", runner.P0_CROSS2_MODE,
                        [{"fingerprint": "job-fingerprint"}],
                    )

    def test_frontier12_authority_binds_offline_and_parent_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline = root / "offline.json"
            reject = root / "reject.json"
            offline.write_text("offline-current", encoding="utf-8")
            reject.write_text("reject-current", encoding="utf-8")
            payload = {
                "core_frontier_mode": runner.P0_MODE,
                "candidate_reward_sha256": runner.candidate_reward_sha256(
                    runner.P0_CROSS2_FRONTIER12_MODE
                ),
                "parent_candidate_mode": runner.P0_CROSS2_MODE,
                "parent_candidate_reward_sha256": (
                    runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
                ),
                "single_changed_reward_parameter": {
                    "name": "ori6_spread_frontier_bonus",
                    "P0X2_value": 0.9,
                    "candidate_value": 1.2,
                },
                "offline_saved_path_ordering_authority": str(
                    offline.resolve()
                ),
                "offline_saved_path_ordering_authority_sha256": (
                    runner.file_sha256(offline)
                ),
                "parent_candidate_reject_authority": str(reject.resolve()),
                "parent_candidate_reject_authority_sha256": (
                    runner.file_sha256(reject)
                ),
            }
            with mock.patch.object(
                runner, "P0_CROSS2_FRONTIER12_OFFLINE_AUDIT", offline
            ), mock.patch.object(
                runner, "P0_CROSS2_REJECT_REPORT", reject
            ):
                self.assertTrue(
                    analysis.manifest_uses_current_candidate_authorities(
                        payload, runner.P0_CROSS2_FRONTIER12_MODE
                    )
                )
                payload["parent_candidate_reject_authority_sha256"] = "0" * 64
                self.assertFalse(
                    analysis.manifest_uses_current_candidate_authorities(
                        payload, runner.P0_CROSS2_FRONTIER12_MODE
                    )
                )
                payload["parent_candidate_reject_authority_sha256"] = (
                    runner.file_sha256(reject)
                )
                payload["single_changed_reward_parameter"]["candidate_value"] = 1.5
                self.assertFalse(
                    analysis.manifest_uses_current_candidate_authorities(
                        payload, runner.P0_CROSS2_FRONTIER12_MODE
                    )
                )

    def test_frontier12_saved_paths_keep_all_five_orderings(self):
        orderings = frontier12_audit.build_orderings()
        self.assertEqual(set(orderings), set(runner.TASKS))
        self.assertTrue(all(row["pass"] for row in orderings.values()))
        for row in orderings.values():
            for case in row["cases"].values():
                self.assertTrue(case["single_parameter_delta_exact"])
                candidate = case["P0X2B12_frontier_bonus_1_2"]
                deltas = case["component_deltas"]
                for name, value in deltas.items():
                    if name not in frontier12_audit.ALLOWED_COMPONENT_DELTAS:
                        self.assertAlmostEqual(value, 0.0, places=10)
                if candidate["terminal_reason"] == "self_crossing":
                    self.assertAlmostEqual(
                        deltas["ori6_spread_frontier_reward"]
                        + deltas["ori6_terminal_correction_reward"],
                        0.0, places=10,
                    )
        for task, case_name in (
            ("water1", "ideal_success"),
            ("torus2", "success"),
            ("five_d3", "ideal_success"),
        ):
            case = orderings[task]["cases"][case_name]
            self.assertAlmostEqual(
                case["return_delta_P0X2B12_minus_P0X2"], 0.3, places=10
            )

    def test_entropy3_manifest_authority_binds_optimizer_and_three_evidences(self):
        mode = runner.P0_CROSS2_ENTROPY3_MODE
        profile = runner.PROFILES["B"]
        payload = {
            "core_frontier_mode": runner.P0_MODE,
            "candidate_reward_sha256": runner.candidate_reward_sha256(mode),
            "parent_candidate_mode": runner.P0_CROSS2_MODE,
            "parent_candidate_reward_sha256": runner.candidate_reward_sha256(
                runner.P0_CROSS2_MODE
            ),
            "reward_identity_reused_from_parent": True,
            "candidate_optimizer_sha256": runner.candidate_optimizer_sha256(
                mode, profile
            ),
            "parent_candidate_optimizer_sha256": runner.candidate_optimizer_sha256(
                runner.P0_CROSS2_MODE, profile
            ),
            "single_changed_training_parameter": {
                "name": "entropy_coefficient", "P0X2_value": 0.001,
                "candidate_value": 0.003,
            },
            "screen_scope": {
                "stage": "entropy_diagnostic", "tasks": ["water1"],
                "seeds": [3101, 3102, 3103], "episodes": 1024,
                "checkpoint_every": 512, "early_gate_checkpoints": [512, 1024],
            },
            "offline_saved_path_ordering_authority": str(
                runner.P0_CROSS2_OFFLINE_AUDIT.resolve()
            ),
            "offline_saved_path_ordering_authority_sha256": runner.file_sha256(
                runner.P0_CROSS2_OFFLINE_AUDIT
            ),
            "preceding_candidate_reject_authority": str(
                runner.P0_CROSS2_FRONTIER12_REJECT_REPORT.resolve()
            ),
            "preceding_candidate_reject_authority_sha256": runner.file_sha256(
                runner.P0_CROSS2_FRONTIER12_REJECT_REPORT
            ),
            "entropy_preflight_authority": str(
                runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT.resolve()
            ),
            "entropy_preflight_authority_sha256": runner.file_sha256(
                runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT
            ),
        }
        self.assertTrue(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )
        payload["single_changed_training_parameter"]["candidate_value"] = 0.004
        self.assertFalse(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )

    def test_width2_manifest_authority_binds_mechanics_and_three_evidences(self):
        mode = runner.P0_CROSS2_WIDTH2_MODE
        task = runner.TASKS["water1"]
        profile = runner.PROFILES[task.profile]
        jobs = runner.build_jobs(
            runner.PROTOCOLS["width_diagnostic"], mode, ["water1"]
        )
        authority = analysis.current_manifest_authority(
            "width_diagnostic", mode, jobs
        )
        path = Path(authority["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )
        self.assertEqual(
            payload["candidate_reward_sha256"],
            runner.candidate_reward_sha256(runner.P0_CROSS2_MODE),
        )
        self.assertEqual(
            payload["candidate_optimizer_sha256"],
            runner.candidate_optimizer_sha256(runner.P0_CROSS2_MODE, profile),
        )
        self.assertNotEqual(
            payload["candidate_mechanics_sha256"],
            payload["parent_candidate_mechanics_sha256"],
        )
        mutations = (
            ("candidate_mechanics_sha256", "0" * 64),
            ("preceding_candidate_reject_authority_sha256", "0" * 64),
            ("width_feasibility_authority_sha256", "0" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(payload))
                changed[field] = value
                self.assertFalse(
                    analysis.manifest_uses_current_candidate_authorities(
                        changed, mode
                    )
                )
        changed = json.loads(json.dumps(payload))
        changed["single_changed_mechanics_parameter"]["candidate_value"] = 0.25
        self.assertFalse(
            analysis.manifest_uses_current_candidate_authorities(changed, mode)
        )

    def test_flatten_record_includes_pairing_identity(self):
        record = {
            "task": "five_d2", "implementation": "ori6",
            "candidate_mode": runner.P0_CROSS2_MODE,
            "profile": "B", "seed": 3101, "status": "complete",
            "endpoint_episode": 2048,
            "first_hit": {"episode": 100, "mechanical_steps": 1000,
                          "policy_updates_before_hit": 12},
            "last_policy_heldout": {},
            "tail512": {
                "success_rate": 0.0, "crossing_rate": 0.0,
                "wrong_flat_rate": 0.0, "timeout_rate": 1.0,
            },
            "training": {
                "success_rate": 0.1,
                "minimum_terminal_flatness_ratio": 1.0,
                "minimum_same_state_joint_cost": 1.0,
                "maximum_spread_progress": 0.0,
                "positive_frontier_episodes": 0,
                "maximum_abs_decomposition_error": 0.0,
                "nonfinite_reward_components": 0,
            },
            "artifact_path": "artifact",
        }
        row = analysis.flatten_record(record)
        self.assertEqual(row["task"], "five_d2")
        self.assertEqual(row["implementation"], "ori6")
        self.assertEqual(row["candidate_mode"], runner.P0_CROSS2_MODE)
        self.assertEqual(row["profile"], "B")
        self.assertEqual(row["seed"], 3101)
        self.assertEqual(row["endpoint_episode"], 2048)

    def test_rejected_crossing_manifest_binds_additive_semantics_and_authorities(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE
        protocol = runner.PROTOCOLS["crossing_continuation_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        manifest = runner.manifest_path(protocol, mode, jobs)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertTrue(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )
        authority = analysis.current_manifest_authority(
            protocol.name, mode, jobs
        )
        self.assertEqual(Path(authority["path"]), manifest.resolve())
        mutations = (
            ("candidate_additive_source_sha256", {}),
            ("candidate_orchestration_source_sha256", {}),
            ("candidate_transition_semantics_sha256", "0" * 64),
            ("parent_transition_semantics_sha256", "0" * 64),
            ("early_gate_specification", {}),
            ("parent_saved_path_scope", "wrong"),
            ("preceding_candidate_reject_authority_sha256", "0" * 64),
            ("random_baseline_authority", {}),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(payload))
                changed[field] = value
                self.assertFalse(
                    analysis.manifest_uses_current_candidate_authorities(
                        changed, mode
                    )
                )

    def test_rejected_crossing_summary_recomputes_count_rate_invariants(self):
        row = {
            "steps": "10", "success": "0", "self_crossing": "0",
            "correction_failed": "0", "no_modes": "0",
            "flatness_ratio": "0.4", "spread_error": "0.05",
            "ori6_candidate": runner.REJECTED_CROSSING_CANDIDATE_NAME,
            "ori6_candidate_schema_version": (
                runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
            ),
            "ori6_crossing_policy": runner.REJECTED_CROSSING_POLICY,
            "ori6_rejected_crossing_count": "2",
            "ori6_rejected_crossing_streak_count": "1",
            "ori6_escape_after_rejected_crossing_count": "1",
            "ori6_unescaped_rejected_crossing_streak_count": "0",
            "ori6_max_consecutive_rejected_crossings": "2",
            "ori6_success_after_rejected_crossing": "0",
            "ori6_rejected_crossing_rate": "0.2",
            "ori6_rejected_crossing_escape_rate": "1",
            "ori6_total_rejected_crossing_reward": "-2",
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0",
        }
        summary = analysis.rejected_crossing_summary([row], "water1")
        self.assertTrue(summary["transition_integrity_pass"])
        self.assertAlmostEqual(summary["pooled_rejected_action_rate"], 0.2)
        changed = dict(row)
        changed["ori6_rejected_crossing_rate"] = "0.3"
        self.assertFalse(
            analysis.rejected_crossing_summary(
                [changed], "water1"
            )["transition_integrity_pass"]
        )
        changed = dict(row)
        changed["ori6_rejected_crossing_count"] = "1.5"
        self.assertFalse(
            analysis.rejected_crossing_summary(
                [changed], "water1"
            )["transition_integrity_pass"]
        )

    def test_rejected_crossing_report_requires_both_task_heldout_support(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_MODE

        def record(task, seed, *, strict=False, constructive=False,
                   heldout_support=False):
            rc = {
                "transition_integrity_pass": True,
                "constructive_proxy_episodes": int(constructive),
                "rejected_action_count": 1,
                "pooled_rejected_action_rate": 0.01,
                "rejected_streak_count": 1,
                "escape_after_rejected_streak_count": 1,
                "pooled_escape_rate": 1.0,
            }
            heldout_rc = {
                **rc,
                "constructive_proxy_episodes": int(heldout_support),
            }
            return {
                "job_id": f"{task}:{seed}", "task": task, "profile": "B",
                "seed": seed, "implementation": "ori6",
                "candidate_mode": mode, "status": "complete",
                "endpoint_episode": 2048,
                "artifact_path": str(Path("missing") / task / str(seed)),
                "artifact_validation": {"valid": True},
                "first_hit": {
                    "censored": not strict, "episode": 100 if strict else None,
                    "mechanical_steps": 1000 if strict else None,
                    "policy_updates_before_hit": 12 if strict else None,
                },
                "training": {
                    "success_rate": 0.01 if strict else 0.0,
                    "minimum_terminal_flatness_ratio": 0.1,
                    "minimum_same_state_joint_cost": 0.1,
                    "maximum_spread_progress": 0.8,
                    "positive_frontier_episodes": 1,
                    "maximum_abs_decomposition_error": 0.0,
                    "nonfinite_reward_components": 0,
                    "rejected_crossing": rc,
                },
                "tail512": {
                    "success_rate": 0.0, "crossing_rate": 0.0,
                    "wrong_flat_rate": 0.0, "timeout_rate": 1.0,
                    "rejected_crossing": rc,
                },
                "last_policy_heldout": {
                    "success_rate": 0.0,
                    "rejected_crossing": heldout_rc,
                },
                "candidate_identity": {
                    "ori6_candidate": runner.REJECTED_CROSSING_CANDIDATE_NAME,
                    "ori6_candidate_schema_version": (
                        runner.REJECTED_CROSSING_CANDIDATE_SCHEMA_VERSION
                    ),
                    "ori6_crossing_policy": runner.REJECTED_CROSSING_POLICY,
                    "candidate_entrypoint": "candidate.py",
                },
            }

        records = []
        for task in ("torus2", "water1"):
            records.extend([
                record(task, 3201, strict=True, constructive=True,
                       heldout_support=True),
                record(task, 3202, constructive=True, heldout_support=True),
                record(task, 3203),
            ])
        gates = {
            "all_three_present_and_passed": True,
            "checkpoints": [],
        }
        patches = (
            mock.patch.object(analysis, "collect", return_value=(records, [])),
            mock.patch.object(
                analysis, "crossing_continuation_gate_authorities",
                return_value=gates,
            ),
            mock.patch.object(
                analysis, "current_manifest_authority",
                return_value={"path": "manifest", "sha256": "x",
                              "manifest_fingerprint": "y"},
            ),
            mock.patch.object(analysis, "plot_learning_curves", return_value=[]),
            mock.patch.object(runner, "atomic_json"),
            mock.patch.object(analysis, "write_csv"),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = analysis.crossing_continuation_report(mode)
        self.assertTrue(result["crossing_continuation_screen_pass"])
        self.assertFalse(result["screen_pass_authorizes_full_training"])
        self.assertEqual(
            result["adoption_blockers_even_if_screen_passes"],
            ["fresh_random_baseline_all_five_tasks",
             "fresh_successful_task_regression"],
        )
        records[-2]["last_policy_heldout"]["rejected_crossing"][
            "constructive_proxy_episodes"
        ] = 0
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = analysis.crossing_continuation_report(mode)
        self.assertFalse(result["crossing_continuation_screen_pass"])


    def test_spectral_summary_checks_identity_telescope_and_rollback(self):
        row = {
            "termination_reason": "timeout",
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
            "ori6_spectral_total_progress_reward": "0.01",
            "ori6_spectral_total_rollback_reward": "0",
            "ori6_spectral_total_reward": "0.01",
            "ori6_spectral_cumulative_reward": "0.01",
            "ori6_spectral_terminal_potential": "0.2",
            "ori6_spectral_terminal_delta": "0.4",
            "ori6_spectral_terminal_rho": "0.5",
            "ori6_spectral_maximum_potential": "0.2",
            "ori6_spectral_maximum_delta": "0.4",
            "ori6_spectral_maximum_rho": "0.5",
            "ori6_spectral_reward_decomposition_error": "0",
            "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_spectral_nonfinite_component_count": "0",
        }
        self.assertTrue(
            analysis.spectral_subspace_summary([row])["integrity_pass"]
        )
        rollback = dict(row)
        rollback.update({
            "termination_reason": "wrong_flat",
            "ori6_spectral_total_progress_reward": "0.01",
            "ori6_spectral_total_rollback_reward": "-0.01",
            "ori6_spectral_total_reward": "0",
            "ori6_spectral_cumulative_reward": "0",
        })
        self.assertTrue(
            analysis.spectral_subspace_summary([rollback])["integrity_pass"]
        )
        changed = dict(rollback)
        changed["ori6_spectral_cumulative_reward"] = "0.01"
        self.assertFalse(
            analysis.spectral_subspace_summary([changed])["integrity_pass"]
        )

    def test_spectral_manifest_and_report_require_heldout_and_fresh_confirmation(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_subspace_diagnostic"]
        jobs = runner.build_jobs(protocol, mode, ["torus2", "water1"])
        manifest = runner.manifest_path(protocol, mode, jobs)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertTrue(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )
        for field, value in (
            ("spectral_formula_sha256", "0" * 64),
            ("spectral_additive_source_sha256", {}),
            ("parent_RC_reject_authority_sha256", "0" * 64),
            ("spectral_offline_design_authority_sha256", "0" * 64),
            ("early_gate_specification", {}),
        ):
            changed = json.loads(json.dumps(payload))
            changed[field] = value
            self.assertFalse(
                analysis.manifest_uses_current_candidate_authorities(
                    changed, mode
                )
            )

        def record(task, seed, *, strict=False, constructive=False,
                   heldout_support=False):
            rc = {
                "transition_integrity_pass": True,
                "constructive_proxy_episodes": int(constructive),
                "rejected_action_count": 0,
                "pooled_rejected_action_rate": 0.0,
                "rejected_streak_count": 0,
                "escape_after_rejected_streak_count": 0,
                "pooled_escape_rate": None,
            }
            spectral = {
                "integrity_pass": True,
                "maximum_potential": 0.2,
                "maximum_delta": 0.4,
                "positive_peak_episodes": 1,
                "delta_gt_1e_6_episodes": 1,
            }
            heldout_rc = {
                **rc,
                "constructive_proxy_episodes": int(heldout_support),
            }
            return {
                "job_id": f"{task}:{seed}", "task": task,
                "profile": "A" if task == "torus2" else "B",
                "seed": seed, "implementation": "ori6",
                "candidate_mode": mode, "status": "complete",
                "endpoint_episode": 2048,
                "artifact_path": str(Path("missing") / task / str(seed)),
                "artifact_validation": {"valid": True},
                "first_hit": {
                    "censored": not strict,
                    "episode": 100 if strict else None,
                    "mechanical_steps": 1000 if strict else None,
                    "policy_updates_before_hit": 12 if strict else None,
                },
                "training": {
                    "success_rate": 0.01 if strict else 0.0,
                    "minimum_terminal_flatness_ratio": 0.1,
                    "minimum_same_state_joint_cost": 0.1,
                    "maximum_spread_progress": 0.8,
                    "positive_frontier_episodes": 1,
                    "maximum_abs_decomposition_error": 0.0,
                    "nonfinite_reward_components": 0,
                    "rejected_crossing": rc,
                    "spectral_subspace": spectral,
                },
                "tail512": {
                    "success_rate": 0.0, "crossing_rate": 0.0,
                    "wrong_flat_rate": 0.0, "timeout_rate": 1.0,
                    "rejected_crossing": rc,
                    "spectral_subspace": spectral,
                },
                "last_policy_heldout": {
                    "success_rate": 0.0,
                    "rejected_crossing": heldout_rc,
                    "spectral_subspace": spectral,
                },
                "candidate_identity": {
                    "ori6_candidate": runner.SPECTRAL_SUBSPACE_CANDIDATE_NAME,
                    "ori6_candidate_schema_version": (
                        runner.SPECTRAL_SUBSPACE_CANDIDATE_SCHEMA_VERSION
                    ),
                    "ori6_crossing_policy": runner.REJECTED_CROSSING_POLICY,
                    "candidate_entrypoint": (
                        "ori_codex6_spectral_subspace_3d.py"
                    ),
                },
            }

        records = []
        for task in ("torus2", "water1"):
            records.extend([
                record(task, 3201, strict=True, constructive=True,
                       heldout_support=True),
                record(task, 3202, constructive=True,
                       heldout_support=True),
                record(task, 3203),
            ])
        gates = {"all_three_present_and_passed": True, "checkpoints": []}
        with mock.patch.object(
            analysis, "collect", return_value=(records, [])
        ), mock.patch.object(
            analysis, "_history_for_record", return_value=[]
        ), mock.patch.object(
            analysis, "spectral_subspace_gate_authorities",
            return_value=gates,
        ), mock.patch.object(
            runner, "spectral_parent_common_prefix_comparison",
            return_value={"gating": False},
        ), mock.patch.object(
            analysis, "current_manifest_authority",
            return_value={"path": "manifest", "sha256": "x",
                          "manifest_fingerprint": "y"},
        ), mock.patch.object(
            analysis, "plot_learning_curves", return_value=[]
        ), mock.patch.object(runner, "atomic_json"), mock.patch.object(
            analysis, "write_csv"
        ):
            result = analysis.spectral_subspace_report(mode)
        self.assertTrue(result["spectral_subspace_screen_pass"])
        self.assertFalse(result["screen_pass_authorizes_full_training"])
        self.assertEqual(
            result["adoption_blockers_even_if_screen_passes"],
            ["fresh_confirmation_seeds",
             "fresh_random_baseline_all_five_tasks",
             "fresh_successful_task_regression"],
        )
        records[-2]["last_policy_heldout"]["rejected_crossing"][
            "constructive_proxy_episodes"
        ] = 0
        with mock.patch.object(
            analysis, "collect", return_value=(records, [])
        ), mock.patch.object(
            analysis, "_history_for_record", return_value=[]
        ), mock.patch.object(
            analysis, "spectral_subspace_gate_authorities",
            return_value=gates,
        ), mock.patch.object(
            runner, "spectral_parent_common_prefix_comparison",
            return_value={"gating": False},
        ), mock.patch.object(
            analysis, "current_manifest_authority",
            return_value={"path": "manifest", "sha256": "x",
                          "manifest_fingerprint": "y"},
        ), mock.patch.object(
            analysis, "plot_learning_curves", return_value=[]
        ), mock.patch.object(runner, "atomic_json"), mock.patch.object(
            analysis, "write_csv"
        ):
            result = analysis.spectral_subspace_report(mode)
        self.assertFalse(result["spectral_subspace_screen_pass"])

    def test_spectral_nonregression_report_uses_one_sided_paired_gates(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE

        def record(task, seed, implementation):
            candidate = implementation == "ori6"
            updates = 0 if task == "five_d3" else 10
            heldout = {
                "episodes": 64,
                "success_rate": 0.5,
                "crossing_rate": 0.0,
                "successful_mean_spread_error": 0.01,
            }
            if candidate:
                heldout.update({
                    "rejected_crossing": {
                        "transition_integrity_pass": True,
                    },
                    "spectral_subspace": {"integrity_pass": True},
                })
            return {
                "job_id": f"{task}:{implementation}:{seed}",
                "task": task,
                "profile": "B",
                "seed": seed,
                "implementation": implementation,
                "candidate_mode": mode,
                "status": "complete",
                "artifact_path": str(Path("missing") / task / implementation / str(seed)),
                "endpoint_episode": 2048,
                "artifact_validation": {"valid": True},
                "first_hit": {
                    "censored": False,
                    "episode": 100,
                    "mechanical_steps": 1000,
                    "policy_updates_before_hit": updates,
                },
                "training": {
                    "total_mechanical_steps": 2000,
                    "maximum_abs_decomposition_error": 0.0,
                    "nonfinite_reward_components": 0,
                    "rejected_crossing": (
                        {"transition_integrity_pass": True}
                        if candidate else {}
                    ),
                    "spectral_subspace": (
                        {"integrity_pass": True} if candidate else {}
                    ),
                },
                "tail512": {
                    "success_rate": 0.25,
                    "crossing_rate": 0.0,
                    "wrong_flat_rate": 0.0,
                    "timeout_rate": 0.75,
                },
                "last_policy_heldout": heldout,
                "candidate_identity": {},
            }

        records = [
            record(task, seed, implementation)
            for task in ("five_d2", "five_d3", "watersqrt2")
            for implementation in ("ori6", "ori5_reference")
            for seed in (3301, 3302, 3303)
        ]
        reward_rows = [{
            "ori6_reward_decomposition_error": "0",
            "ori6_maximum_abs_step_reward_decomposition_error": "0",
            "ori6_nonfinite_reward_component_count": "0",
            "ori5_reward_decomposition_error": "0",
            "ori5_maximum_abs_step_reward_decomposition_error": "0",
            "ori5_nonfinite_reward_component_count": "0",
        } for _ in range(64)]
        gate = {
            "checkpoint_episode": 512,
            "present": True,
            "valid_and_passed": True,
        }

        def run_report(current_records):
            with mock.patch.object(
                runner, "validate_spectral_screen_report_authority"
            ), mock.patch.object(
                analysis, "collect", return_value=(current_records, [])
            ), mock.patch.object(
                analysis, "spectral_nonregression_gate_authority",
                return_value=gate,
            ), mock.patch.object(
                analysis, "load_csv", return_value=reward_rows
            ), mock.patch.object(
                analysis, "current_manifest_authority",
                return_value={"path": "manifest", "sha256": "x",
                              "manifest_fingerprint": "y"},
            ), mock.patch.object(
                analysis, "plot_learning_curves", return_value=[]
            ), mock.patch.object(
                analysis, "flatten_record", return_value={}
            ), mock.patch.object(
                runner, "file_sha256", return_value="bound-sha"
            ), mock.patch.object(runner, "atomic_json"), mock.patch.object(
                analysis, "write_csv"
            ):
                return analysis.spectral_nonregression_report(mode)

        result = run_report(records)
        self.assertTrue(result["successful_task_regression_gate_pass"])
        self.assertFalse(result["full_manifest_authorized"])
        for task, gate_result in result["task_results"].items():
            self.assertTrue(
                gate_result[
                    "success_one_sided_lower_bound_strictly_gt_minus_0_10"
                ]
            )
            self.assertTrue(
                gate_result[
                    "crossing_one_sided_upper_bound_strictly_lt_plus_0_10"
                ]
            )
            if task == "five_d3":
                self.assertFalse(gate_result["mechanics_gate_evaluated"])
                self.assertTrue(gate_result["zero_update_hits"])
        degraded = json.loads(json.dumps(records))
        for row in degraded:
            if row["implementation"] == "ori6":
                row["last_policy_heldout"]["success_rate"] = 0.0
            else:
                row["last_policy_heldout"]["success_rate"] = 1.0
        result = run_report(degraded)
        self.assertFalse(result["successful_task_regression_gate_pass"])

    def test_pending_spectral_nonregression_manifest_is_current_only_while_report_absent(self):
        mode = runner.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
        protocol = runner.PROTOCOLS["spectral_nonregression"]
        jobs = runner.build_jobs(protocol, mode, list(protocol.tasks))
        with mock.patch.object(
            runner, "SPECTRAL_SUBSPACE_SCREEN_REPORT",
            runner.EXPERIMENT_ROOT / "analysis" / "missing_screen_report.json",
        ), mock.patch.object(runner, "record_stale_manifests"):
            manifest = runner.manifest_path(protocol, mode, jobs)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(
                analysis.manifest_uses_current_candidate_authorities(
                    payload, mode
                )
            )
            changed = json.loads(json.dumps(payload))
            changed["early_gate_specification"] = {}
            self.assertFalse(
                analysis.manifest_uses_current_candidate_authorities(
                    changed, mode
                )
            )

        with mock.patch.object(runner, "record_stale_manifests"):
            manifest = runner.manifest_path(protocol, mode, jobs)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["spectral_screen_report_status"],
            "bound_regression_permission_authority",
        )
        self.assertTrue(
            analysis.manifest_uses_current_candidate_authorities(payload, mode)
        )
        changed = json.loads(json.dumps(payload))
        changed["spectral_screen_report_sha256"] = "0" * 64
        self.assertFalse(
            analysis.manifest_uses_current_candidate_authorities(changed, mode)
        )


if __name__ == "__main__":
    unittest.main()
