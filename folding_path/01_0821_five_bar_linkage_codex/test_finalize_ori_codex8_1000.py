import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import finalize_ori_codex8_1000 as finalizer


def history_row(episode, *, strict=False):
    return {
        "episode": str(episode), "steps": "10", "total_reward": str(episode / 100),
        "flatness_ratio": str(1 / (episode + 1)),
        "spread_error": str(.2 / (episode + 1)),
        "strict_compact_success": str(int(strict)), "self_crossing": "0",
        "correction_failed": "0", "no_modes": "0", "timeout": "1",
        "ori6_rejected_crossing_count": "0", "ori6_rejected_crossing_rate": "0",
        "ori6_spectral_terminal_potential": str(episode / 1000),
        "ori6_spectral_total_reward": str(episode / 10000),
    }


def selection_report(arm, seed, *, endpoint=2048, hit=False, proposals=None,
                     strict=0, joint=0, constructive=0, e=.2, f=.8, reject=.1,
                     exposure=200_000):
    recent = {
        "strict_count": strict, "joint_count": joint,
        "constructive_count": constructive,
        "spread_error_median": e, "flatness_median": f,
        "reject_rate_mean": reject, "invalid_count": 0,
        "episode_count": min(endpoint, 1024),
    }
    return {
        "task": "torus2", "arm": arm, "seed": seed, "endpoint": endpoint,
        "integrity_pass": True, "integrity_errors": [],
        "first_hit": {
            "hit": hit, "episode": 10 if hit else None,
            "action_proposals": proposals if hit else None,
            **({} if hit else {"censor_action_proposals": exposure}),
        },
        "full": {"action_proposals": exposure},
        "last1024": recent, "last512": recent, "previous512": recent,
        "phase_reward_hacking": {"farming_detected": False},
    }


def analysis_fixture(endpoint=2048):
    reports = {}
    for arm in finalizer.staged.ARMS:
        reports[arm] = []
        for index, seed in enumerate(finalizer.staged.SEEDS):
            if arm == "phase_v2":
                report = selection_report(
                    arm, seed, endpoint=endpoint, hit=True,
                    proposals=100 + index, strict=5, joint=8,
                    constructive=12, e=.01, f=.02, reject=.02,
                )
            elif arm == "scale":
                report = selection_report(
                    arm, seed, endpoint=endpoint, hit=True,
                    proposals=200 + index, strict=3, joint=5,
                    constructive=8, e=.02, f=.03, reject=.03,
                )
            else:
                report = selection_report(
                    arm, seed, endpoint=endpoint, hit=False,
                    strict=0, joint=1, constructive=3,
                )
            reports[arm].append(report)
    pairwise = {arm: {} for arm in finalizer.staged.ARMS}
    for candidate in finalizer.staged.ARMS:
        for opponent in finalizer.staged.ARMS:
            if candidate != opponent:
                pairwise[candidate][opponent] = finalizer.staged.paired_comparison(
                    reports[candidate], reports[opponent]
                )
    decision = {
        "task": "torus2", "endpoint": endpoint,
        "selection_rule_version": finalizer.staged.SELECTION_RULE_VERSION,
        "formal_decision_available": endpoint >= 4096,
        "strong_early_promotion": endpoint == 2048,
        "early_evidence_seed_count": 3,
        "strong_early_rule": "fixture", "paired_wins": pairwise,
        "eligible_new_arms": ["phase_v2"], "selected_arm": "phase_v2",
        "selected_seed": None, "selected_arm_min_pairwise_wins": 3,
        "all_candidate_seeds_integrity_pass": True,
        "candidate_farming_gate_pass": True, "all_task_integrity": True,
        "descriptive_winner_arm": "phase_v2", "control_non_improvement": False,
        "promote_selected_to_10000": True,
        "continuation_scope": "all three seeds of selected new arm",
        "return_used_for_selection": False,
    }
    return {
        "arms": list(finalizer.staged.ARMS),
        "reports": {"torus2": reports},
        "task_decisions": {"torus2": decision},
    }


class Codex81000FinalizerTests(unittest.TestCase):
    def approvals(self):
        return (
            finalizer.sha256_file(finalizer.SCRIPT_DIRECTORY / finalizer.FINALIZER_FILES[0]),
            finalizer.sha256_file(finalizer.SCRIPT_DIRECTORY / finalizer.FINALIZER_FILES[1]),
        )

    def test_real_random_baselines_are_officially_recomputed(self):
        torus = finalizer.load_random_baseline("torus2")
        water = finalizer.load_random_baseline("water1")
        self.assertEqual((torus["hits"], torus["n"], torus["budget"]), (0, 10, 636198))
        self.assertEqual(torus["restricted_mean_action_proposals"], 636198)
        self.assertEqual((water["hits"], water["n"], water["budget"]), (3, 10, 126319))
        self.assertAlmostEqual(water["restricted_mean_action_proposals"], 99848.0)
        self.assertIn("aggregate_results", water["authority"])
        self.assertIn("reward identity", water["equivalence_scope"])
        self.assertGreaterEqual(len(water["input_sha256"]), 50)

    def test_random_trial_summary_mutation_fails_against_history(self):
        task, seed, budget = "water1", 6101, 126319
        source = finalizer.RANDOM_ROOT / task / f"budget{budget}" / f"seed{seed}"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / task / f"budget{budget}" / f"seed{seed}"
            trial_source = source / "trials" / "uniform" / f"seed{seed}"
            trial_target = target / "trials" / "uniform" / f"seed{seed}"
            trial_target.mkdir(parents=True)
            for name in ("run_config.json", "aggregate.json"):
                shutil.copy2(source / name, target / name)
            for name in ("run_config.json", "summary.json", "episode_history.csv"):
                shutil.copy2(trial_source / name, trial_target / name)
            finalizer._official_random_seed(task, seed, budget, base)
            path = trial_target / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["mechanical_steps"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "history/summary"):
                finalizer._official_random_seed(task, seed, budget, base)

    def test_self_consistent_random_action_law_mutation_fails_exact_identity(self):
        task, seed, budget = "water1", 6101, 126319
        source = finalizer.RANDOM_ROOT / task / f"budget{budget}" / f"seed{seed}"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / task / f"budget{budget}" / f"seed{seed}"
            trial_source = source / "trials" / "uniform" / f"seed{seed}"
            trial_target = target / "trials" / "uniform" / f"seed{seed}"
            trial_target.mkdir(parents=True)
            for name in ("run_config.json", "aggregate.json"):
                shutil.copy2(source / name, target / name)
            for name in ("run_config.json", "summary.json", "episode_history.csv"):
                shutil.copy2(trial_source / name, trial_target / name)
            path = target / "run_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["fingerprint_identity"]["action_law"] = "tampered law"
            config["action_law"] = "tampered law"
            config["root_fingerprint"] = finalizer.random_baseline.canonical_digest(
                config["fingerprint_identity"]
            )
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact identity"):
                finalizer._official_random_seed(task, seed, budget, base)

    def test_fractional_random_mechanics_and_nonbinary_success_fail(self):
        root = (finalizer.RANDOM_ROOT / "water1" / "budget126319" / "seed6101"
                / "trials" / "uniform" / "seed6101")
        rows = finalizer.load_csv(root / "episode_history.csv")
        summary = finalizer.load_json(root / "summary.json")
        fractional = [dict(row) for row in rows]
        fractional[0]["mechanical_steps"] = "400.5"
        with self.assertRaisesRegex(RuntimeError, "exact integer"):
            finalizer._validate_random_history(fractional, summary, 126319)
        nonbinary = [dict(row) for row in rows]
        nonbinary[0]["strict_compact_success"] = "2"
        with self.assertRaisesRegex(RuntimeError, "binary"):
            finalizer._validate_random_history(nonbinary, summary, 126319)

    def test_contiguous_history_atomic_cap_and_restricted_checkpoint_loader(self):
        rows = [history_row(1), history_row(2)]
        self.assertEqual(finalizer.contiguous_history(rows), 2)
        rows[0]["episode"] = "1.5"
        with self.assertRaisesRegex(RuntimeError, "exact integer"):
            finalizer.contiguous_history(rows)
        rows[0]["episode"] = "1"
        rows[1]["episode"] = "3"
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            finalizer.contiguous_history(rows)
        rows[1]["episode"] = "2"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with (directory / "training_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            (directory / "checkpoint.pt").write_bytes(b"mock")
            with mock.patch.object(finalizer, "checkpoint_episode", return_value=2):
                endpoint, audit, loaded = finalizer.authoritative_endpoint(directory)
            self.assertEqual((endpoint, audit["authority"], len(loaded)),
                             (2, "checkpoint_and_history", 2))
            import torch
            checkpoint = directory / "safe_checkpoint.pt"
            torch.save({"episode": 512, "numpy_rng_state": finalizer.np.random.get_state()}, checkpoint)
            self.assertEqual(finalizer.checkpoint_episode(checkpoint), 512)
            torch.save({"episode": True}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "exact integer"):
                finalizer.checkpoint_episode(checkpoint)

    def test_runner_endpoint_integer_and_status_authority_fail_closed(self):
        base = {
            "task": "torus2", "arm": "legacy", "seed": 8101,
            "launched": True, "status": "screened", "selected": False,
            "raw_endpoint": 512, "formal_analyzed_endpoint": 512,
            "overrun_episode_count": 0,
            "overrun_excluded_from_formal_analysis": False,
        }
        key = finalizer._report_endpoint_identity(base)
        self.assertEqual(key, ("torus2", "legacy", 8101))
        self.assertEqual(finalizer.validate_runner_endpoint_record(base, key), (512, 512))
        for field in (
            "seed", "raw_endpoint", "formal_analyzed_endpoint",
            "overrun_episode_count",
        ):
            with self.subTest(field=field):
                row = dict(base); row[field] = float(base[field]) + .5
                if field == "seed":
                    with self.assertRaisesRegex(RuntimeError, "exact integer"):
                        finalizer._report_endpoint_identity(row)
                else:
                    with self.assertRaisesRegex(RuntimeError, "exact integer"):
                        finalizer.validate_runner_endpoint_record(row, key)
        bad_status = dict(base); bad_status["status"] = "selected"
        with self.assertRaisesRegex(RuntimeError, "Selected endpoint status"):
            finalizer.validate_runner_endpoint_record(bad_status, key)
        bad_launch = dict(base); bad_launch["launched"] = False
        with self.assertRaisesRegex(RuntimeError, "Unlaunched endpoint"):
            finalizer.validate_runner_endpoint_record(bad_launch, key)

    def test_rolling_and_episode_curves_include_joint(self):
        self.assertEqual(finalizer.rolling([1, 2, 3], 2), [1, 1.5, 2.5])
        self.assertEqual(finalizer.rolling([1, 9, 2], 2, "median"), [1, 5, 5.5])
        values = finalizer.rolling([math.nan, 2], 2)
        self.assertTrue(math.isnan(values[0])); self.assertEqual(values[1], 2)
        rows = [history_row(index, strict=index >= 3) for index in range(1, 5)]
        curves = finalizer.episode_curves(
            rows, 4, {"task": "torus2", "arm": "legacy", "seed": 8101}
        )
        for field in (
            "terminal_flatness_median200", "spread_error_median200",
            "strict_rate200", "joint_rate200", "constructive_rate200",
            "spectral_potential_ma200", "reject_rate200",
        ):
            self.assertIn(field, curves[-1])

    def baseline_fixture(self, cap=100):
        trials = [
            {"seed": seed, "hit": False, "censored": True,
             "first_hit_action_proposals": None,
             "observed_action_proposals": cap,
             "assigned_rmst_cost": cap,
             "censor_reason": "common_action_proposal_cap"}
            for seed in range(6101, 6111)
        ]
        return {
            "task": "water1", "budget": cap, "seeds": list(range(6101, 6111)),
            "n": 10, "hits": 0, "hit_fraction": 0.0,
            "first_hit_action_proposals": [],
            "restricted_mean_action_proposals": float(cap),
            "trials": trials, "input_sha256": {},
        }

    def test_random_comparison_uses_same_cap_rmst_not_hit_only_median(self):
        reports = [
            selection_report("phase_v2", 8101, endpoint=10000, hit=True,
                             proposals=20, exposure=200),
            selection_report("phase_v2", 8102, endpoint=10000, hit=False,
                             exposure=150),
            selection_report("phase_v2", 8103, endpoint=10000, hit=True,
                             proposals=120, exposure=180),
        ]
        with mock.patch.object(finalizer, "load_random_baseline",
                               return_value=self.baseline_fixture(100)):
            result = finalizer.random_comparison("water1", "phase_v2", reports)
        self.assertEqual(result["learning_hits"], 1)
        self.assertAlmostEqual(result["learning_restricted_mean_action_proposals"], 220 / 3)
        self.assertAlmostEqual(
            result["descriptive_random_rmst_over_learning_rmst"], 100 / (220 / 3)
        )
        self.assertEqual(result["learning_median_first_hit_action_proposals"], 20)

    def test_random_comparison_marks_insufficient_learning_exposure(self):
        reports = [
            selection_report("phase_v2", seed, endpoint=10000, hit=False,
                             exposure=50 if seed == 8103 else 150)
            for seed in finalizer.staged.SEEDS
        ]
        with mock.patch.object(finalizer, "load_random_baseline",
                               return_value=self.baseline_fixture(100)):
            result = finalizer.random_comparison("water1", "phase_v2", reports)
        self.assertFalse(result["learning_common_cap_observable"])
        self.assertIsNone(result["learning_restricted_mean_action_proposals"])
        self.assertIsNone(result["descriptive_random_rmst_over_learning_rmst"])
        self.assertIn("insufficient", result["ratio_qualifier"])

    def test_external_approval_is_required_and_non_self_referential(self):
        source, tests = self.approvals()
        result = finalizer.verify_external_approval(source, tests)
        self.assertFalse(result["self_referential_hash_claim"])
        self.assertIn("external", result["authority"])
        with self.assertRaisesRegex(RuntimeError, "approved SHA256 mismatch"):
            finalizer.verify_external_approval("0" * 64, tests)
        with self.assertRaisesRegex(RuntimeError, "Missing/invalid"):
            finalizer.verify_external_approval("bad", tests)

    def test_exact_replay_manifest_commands_and_binding_fail_closed(self):
        manifest_path = finalizer.MANIFEST_DIRECTORY / finalizer.EXPECTED_MANIFEST_NAME
        manifest = finalizer.campaign_runner.validate_manifest(manifest_path)
        self.assertEqual(manifest["manifest_fingerprint"],
                         finalizer.EXPECTED_MANIFEST_FINGERPRINT)
        self.assertEqual(len(manifest["jobs"]), 18)
        binding = finalizer.load_json(finalizer.DEFAULT_ROOT / "execution_binding.json")
        finalizer.validate_execution_binding(binding, manifest)
        bad_binding = dict(binding); bad_binding["max_parallel"] = 11
        with self.assertRaisesRegex(RuntimeError, "wrong=.*max_parallel"):
            finalizer.validate_execution_binding(bad_binding, manifest)
        bad_manifest = json.loads(json.dumps(manifest))
        bad_manifest["jobs"][0]["command"].append("--not-authorized")
        content = dict(bad_manifest); content.pop("manifest_fingerprint")
        bad_manifest["manifest_fingerprint"] = finalizer.canonical_digest(content)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated_manifest.json"
            path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                finalizer.campaign_runner.validate_manifest(path)

    def test_forensic_parent_is_verified_but_contributes_zero_primary_rows(self):
        manifest = finalizer.load_json(
            finalizer.MANIFEST_DIRECTORY / finalizer.EXPECTED_MANIFEST_NAME
        )
        provenance, hashes = finalizer.verify_forensic_parent(manifest)
        self.assertEqual(provenance["status"], "audited_abort_provenance_only")
        self.assertEqual(provenance["scientific_rows_in_primary_analysis"], 0)
        self.assertEqual(provenance["verified_file_count"], 146)
        self.assertEqual(len(hashes), 147)

    def test_paired_seed_rows_expose_every_key_at_selection_endpoint(self):
        analysis = analysis_fixture()
        rows = finalizer.paired_seed_rows("torus2", 2048, analysis)
        self.assertEqual(len(rows), 18)
        row = next(row for row in rows if row["candidate"] == "phase_v2"
                   and row["opponent"] == "scale" and row["seed"] == 8101)
        self.assertEqual(row["endpoint"], 2048)
        self.assertEqual(row["winner"], "phase_v2")
        self.assertEqual(row["first_differing_component"],
                         "conditional_negative_first_hit_proposals")
        for field in ("joint_delta", "constructive_delta",
                      "spread_error_delta_candidate_minus_opponent",
                      "flatness_delta_candidate_minus_opponent",
                      "reject_rate_delta_candidate_minus_opponent"):
            self.assertIn(field, row)

    def test_early_2048_winner_4096_revocation_reconstructs(self):
        analysis = analysis_fixture(2048)
        decision = analysis["task_decisions"]["torus2"]
        recorded = {
            **decision, "selection_endpoint": 2048,
            "comparative_winner_arm": "phase_v2", "selected_arm": None,
            "promote_selected_to_10000": False,
            "continued_seed_policy": "revoked_at_4096_integrity_or_catastrophic_gate",
            "continued_seeds": [], "promotion_revoked_endpoint": 4096,
            "promotion_revoked_kind": "catastrophic",
        }
        reports_by_key = {
            ("torus2", arm, seed): selection_report(arm, seed, endpoint=4096)
            for arm in finalizer.staged.ARMS for seed in finalizer.staged.SEEDS
        }
        with mock.patch.object(finalizer.staged, "analyze", return_value=analysis), \
                mock.patch.object(finalizer.staged, "seed_report",
                                  side_effect=lambda root, task, arm, seed, endpoint:
                                  selection_report(arm, seed, endpoint=endpoint)), \
                mock.patch.object(finalizer.staged, "catastrophic",
                                  return_value=(True, "catastrophic", ["gate"])):
            verified, pairs = finalizer.reconstruct_task_selection(
                Path("unused"), "torus2", recorded, reports_by_key
            )
        self.assertIsNone(verified["selected_arm"])
        self.assertEqual(verified["final_decision_status"],
                         "promotion_revoked_no_final_selected_arm")
        self.assertEqual(len(pairs), 18)

    def test_causal_censor_reasons_distinguish_deadline_and_nonselection(self):
        jobs = [
            {"task": "torus2", "arm": "legacy", "seed": 8101,
             "launched": True, "atomic_endpoint": 2048, "first_hit": False},
            {"task": "torus2", "arm": "phase_v2", "seed": 8101,
             "launched": True, "atomic_endpoint": 5120, "first_hit": False},
            {"task": "water1", "arm": "legacy", "seed": 8101,
             "launched": False, "atomic_endpoint": 0,
             "status": "queued_censored_at_training_deadline"},
        ]
        selection = {
            "torus2": {"selected_arm": "phase_v2", "selection_endpoint": 2048}
        }
        with mock.patch.object(finalizer, "_arm_screen_stop", return_value=None):
            finalizer.assign_causal_censor_reasons(
                Path("unused"), jobs, {}, selection, {"deadline_stop": True},
            )
        self.assertIn("not_selected_by_same_seed", jobs[0]["training_censor_reason"])
        self.assertEqual(jobs[1]["training_censor_reason"],
                         "training_deadline_during_selected_continuation")
        self.assertEqual(jobs[2]["training_censor_reason"],
                         "queued_censored_at_training_deadline")

    def test_no_hit_at_10000_has_natural_completion_censor(self):
        jobs = [{
            "task": "torus2", "arm": "phase_v2", "seed": 8101,
            "launched": True, "atomic_endpoint": 10000, "first_hit": False,
        }]
        with mock.patch.object(finalizer, "_arm_screen_stop", return_value=None):
            finalizer.assign_causal_censor_reasons(
                Path("unused"), jobs, {}, {}, {"deadline_stop": False},
            )
        self.assertIsNone(jobs[0]["training_censor_reason"])
        self.assertEqual(
            jobs[0]["first_hit_censor_reason"],
            "natural_completion_at_episode_10000_without_strict_hit",
        )

    def test_screen_stop_uses_formal_gate_despite_different_raw_overruns(self):
        jobs, reports = [], {}
        for raw, seed in zip((4096, 4608, 5120), finalizer.staged.SEEDS):
            jobs.append({
                "task": "torus2", "arm": "scale", "seed": seed,
                "atomic_endpoint": raw, "formal_analyzed_endpoint": 4096,
            })
            reports[("torus2", "scale", seed)] = selection_report(
                "scale", seed, endpoint=4096
            )
        with mock.patch.object(
            finalizer.staged, "catastrophic",
            return_value=(True, "catastrophic", ["formal gate"]),
        ):
            stop = finalizer._arm_screen_stop(
                "torus2", "scale", jobs, reports
            )
        self.assertEqual(stop["endpoint"], 4096)
        self.assertEqual(stop["stop_kind"], "catastrophic")

    def synthetic_payload(self):
        episode, optimizer = [], []
        for task in finalizer.staged.TASKS:
            identity = {"task": task, "arm": "phase_v2", "seed": 8101}
            source = [history_row(index, strict=index > 45) for index in range(1, 65)]
            episode.extend(finalizer.episode_curves(source, 64, identity))
            optimizer.extend({
                **identity, "episode": index, "critic_loss": 1 / index,
                "critic_gradient_norm": 2 / index,
                "actor_gradient_norm": 1 / index, "entropy": .1,
                "actor_learning_rate": .001, "critic_learning_rate": .0003,
            } for index in range(1, 65))
        source_hash, test_hash = self.approvals()
        approval = finalizer.verify_external_approval(source_hash, test_hash)
        return {
            "schema_version": 1, "analysis": "test", "created_utc": "now",
            "root": "test", "manifest_path": "test",
            "manifest_fingerprint": "a" * 64, "protocol": finalizer.PROTOCOL,
            "selection_rule_version": finalizer.staged.SELECTION_RULE_VERSION,
            "reward_only_change": True,
            "observation_contract": "exact_Codex6_parent_observation_unchanged",
            "hidden_ledger_pomdp_caveat": "hidden",
            "forensic_parent_provenance": {"scientific_rows_in_primary_analysis": 0},
            "runner_report": {"task_selections": {}},
            "verified_task_selections": {},
            "jobs": [{"task": "torus2", "arm": "phase_v2", "seed": 8101,
                      "launched": False, "atomic_endpoint": 0,
                      "training_censor_reason": "test"}],
            "episode_curves": episode, "optimizer_curves": optimizer,
            "paired_seed_comparisons": [], "random_comparisons": {},
            "source_sha256": {}, "finalizer_source_sha256": approval["verified_sha256"],
            "external_hash_approval": approval, "input_sha256": {},
            "comparison_contract": {},
        }

    def test_bundle_has_paired_csv_utf8_plots_and_hashes(self):
        payload = self.synthetic_payload()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            with mock.patch.object(finalizer, "verify_payload_inputs_unchanged"):
                result = finalizer.write_bundle(
                    payload, output, enforce_authority=False,
                    publish_active_check=False,
                )
            self.assertIn("paired_seed_comparisons.csv", result["files"])
            self.assertIn("Codex8", (output / "report_ja.md").read_text(encoding="utf-8"))
            qa = json.loads((output / "visual_qa.json").read_text(encoding="utf-8"))
            self.assertEqual(len(qa), 4)
            self.assertTrue(all(row["pixel_std"] > 1 for row in qa))
            self.assertTrue((output / "output_hashes.json").is_file())

    def test_japanese_report_marks_unconfirmed_2048_evidence(self):
        payload = self.synthetic_payload()
        payload["verified_task_selections"]["torus2"] = {
            "selected_arm": "phase_v2", "selection_endpoint": 2048,
            "confirmation_gate_4096": {
                "reached": False,
                "reason": "right_censored_before_4096_confirmation_gate",
            },
        }
        report = finalizer.japanese_report(payload)
        self.assertIn("early evidence / formal4096未到達", report)

    def test_pre_publish_revalidation_detects_consumed_input_mutation(self):
        payload = self.synthetic_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("original", encoding="utf-8")
            payload["input_sha256"] = {str(path): finalizer.sha256_file(path)}
            manifest = {"manifest_fingerprint": payload["manifest_fingerprint"]}
            binding = {"manifest_fingerprint": payload["manifest_fingerprint"]}
            with mock.patch.object(
                finalizer, "verify_control_plane",
                return_value=(manifest, binding, payload["runner_report"], Path("manifest")),
            ):
                finalizer.verify_payload_inputs_unchanged(payload)
                path.write_text("mutated", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Consumed input changed"):
                    finalizer.verify_payload_inputs_unchanged(payload)

    def test_output_authority_and_active_publish_toctou(self):
        self.assertEqual(
            finalizer.validate_output_authority(finalizer.DEFAULT_OUTPUT,
                                                finalizer.DEFAULT_ROOT),
            finalizer.DEFAULT_OUTPUT.resolve(),
        )
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "final"
            payload = self.synthetic_payload()
            with mock.patch.object(finalizer, "render_plots", return_value=[]), \
                    mock.patch.object(finalizer, "visual_qa", return_value=[]), \
                    mock.patch.object(finalizer, "active_codex8_pids", return_value=[{
                        "pid": 9, "command": "trainer",
                    }]), mock.patch.object(finalizer, "verify_payload_inputs_unchanged"):
                with self.assertRaisesRegex(RuntimeError, "before atomic publish"):
                    finalizer.write_bundle(
                        payload, forbidden, enforce_authority=False,
                        publish_active_check=True,
                    )
            self.assertFalse(forbidden.exists())

    def test_pid_appearance_during_input_revalidation_blocks_publish(self):
        payload = self.synthetic_payload()
        events = []

        def active():
            events.append("pid")
            return [] if events.count("pid") == 1 else [{
                "pid": 19, "command": "trainer appeared during revalidation",
            }]

        def revalidate(_payload):
            events.append("verify")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            with mock.patch.object(finalizer, "render_plots", return_value=[]), \
                    mock.patch.object(finalizer, "visual_qa", return_value=[]), \
                    mock.patch.object(finalizer, "active_codex8_pids",
                                      side_effect=active), \
                    mock.patch.object(finalizer, "verify_payload_inputs_unchanged",
                                      side_effect=revalidate):
                with self.assertRaisesRegex(RuntimeError, "before atomic publish"):
                    finalizer.write_bundle(
                        payload, output, enforce_authority=False,
                        publish_active_check=True,
                    )
            self.assertEqual(events, ["pid", "verify", "pid"])
            self.assertFalse(output.exists())

    def test_main_refuses_active_pid_before_reading_campaign(self):
        source, tests = self.approvals()
        args = ["--approved-finalizer-sha256", source,
                "--approved-test-sha256", tests]
        with mock.patch.object(finalizer, "active_codex8_pids", return_value=[{
            "pid": 7, "command": "trainer",
        }]):
            with self.assertRaisesRegex(RuntimeError, "Active Codex8"):
                finalizer.main(args)


if __name__ == "__main__":
    unittest.main()
