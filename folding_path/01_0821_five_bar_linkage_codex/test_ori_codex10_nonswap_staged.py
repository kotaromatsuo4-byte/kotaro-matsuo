"""Focused pre-launch tests for the additive Codex6 non-swap harness."""

from __future__ import annotations

import math
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ori_codex2
import ori_codex6_spectral_subspace as planar
import ori_codex6_spectral_subspace_3d as spatial

import analyze_ori_codex10_nonswap_staged as analyzer
import run_ori_codex10_nonswap_staged as runner


class Codex10NonSwapTests(unittest.TestCase):
    def test_parallelism_below_one_cohort_fails_fast(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "6..18"):
                runner.manifest_payload(Path(temporary) / "root", 5)

    def test_manifest_has_exact_fresh_18_job_factorial_and_core_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "C10NS-test"
            payload = runner.manifest_payload(root, 18)
        self.assertEqual(len(payload["jobs"]), 18)
        self.assertEqual(payload["max_parallel"], 18)
        keys = {
            (job["task"]["name"], job["arm"], job["seed"])
            for job in payload["jobs"]
        }
        expected = {
            (task, arm, seed)
            for task in runner.TASK_ORDER
            for arm in runner.TASKS[task].arms
            for seed in runner.SEEDS
        }
        self.assertEqual(keys, expected)
        self.assertEqual(len({job["job_fingerprint"] for job in payload["jobs"]}), 18)
        for name, digest in runner.EXPECTED_CORE_HASHES.items():
            self.assertEqual(payload["source_sha256"][name], digest)

    def test_manifest_validation_rejects_bound_command_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "C10NS-test"
            payload = runner.manifest_payload(root, 18)
            valid_path = runner.atomic_json(directory / "valid.json", payload)
            self.assertEqual(
                runner.validate_manifest(valid_path, root, 18)["manifest_fingerprint"],
                payload["manifest_fingerprint"],
            )
            mutated = json.loads(json.dumps(payload))
            mutated["jobs"][0]["command"][4] = "9999"
            mutation_path = runner.atomic_json(directory / "mutated.json", mutated)
            with self.assertRaisesRegex(RuntimeError, "binding is stale"):
                runner.validate_manifest(mutation_path, root, 18)

    def test_every_command_parses_as_literal_p0x2rcs05(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = runner.build_jobs(Path(temporary) / "root")
        for job in jobs:
            argv = job["command"][3:]
            module = spatial if job["task"]["flatness_subspace_dimension"] == 2 else planar
            args = module.parse_args(argv)
            expected = job["expected_config"]
            for key, value in expected.items():
                observed = getattr(args, key)
                if isinstance(value, float):
                    self.assertTrue(math.isclose(observed, value, rel_tol=1e-12, abs_tol=1e-12), key)
                else:
                    self.assertEqual(observed, value, key)
            self.assertFalse(hasattr(args, "ori6_zero_reject_reward"))
            self.assertEqual(args.ori6_crossing_policy, "reject_candidate_and_continue")
            self.assertEqual(args.ori5_crossing_penalty, 2.0)
            self.assertEqual(args.ori6_spectral_weight, 0.05)
            self.assertGreaterEqual(len(expected), 80)
            self.assertTrue({
                "data_dir", "output_dir", "selection_evaluation_seed",
                "final_evaluation_seed", "min_step_width", "projection_shrink",
                "ori5_quality_weight", "ori5_timeout_penalty",
                "ori6_spectral_relative_epsilon",
            }.issubset(expected))

    def test_optimizer_contracts_and_warm_boundary(self):
        warm = runner.OPTIMIZERS["W"]
        self.assertEqual(
            ori_codex2.actor_learning_rate_at_episode(
                warm.actor_learning_rate, 512,
                warm.actor_lr_milestones, warm.actor_lr_factors,
            ),
            1.0e-3,
        )
        self.assertEqual(
            ori_codex2.actor_learning_rate_at_episode(
                warm.actor_learning_rate, 513,
                warm.actor_lr_milestones, warm.actor_lr_factors,
            ),
            5.0e-4,
        )
        self.assertEqual(
            ori_codex2.actor_learning_rate_at_episode(
                warm.actor_learning_rate, 2001,
                warm.actor_lr_milestones, warm.actor_lr_factors,
            ),
            1.5e-4,
        )
        easy = runner.OPTIMIZERS["E"]
        self.assertEqual((easy.batch_size, easy.actor_learning_rate, easy.critic_learning_rate), (4, 2.5e-4, 7.5e-5))

    def test_analyzer_integer_parser_fails_closed(self):
        with self.assertRaises(ValueError):
            analyzer._int({"episode": "1.5"}, ("episode",))
        self.assertFalse(analyzer._same(1.0, True))
        self.assertFalse(analyzer._same(1.0, "1.0"))
        self.assertFalse(analyzer._same(False, 0))
        self.assertFalse(analyzer._same(1, True))
        self.assertFalse(analyzer._same(400, 400.0))

    def test_real_partial_config_uses_nested_candidate_authority(self):
        fixture = (
            runner.SCRIPT_DIRECTORY / "experiments_ori6" / "screen"
            / "spectral_nonregression" / "P0X2RCS05" / "ori6"
            / "five_d2" / "B" / "seed3301" / "run_config.json"
        )
        self.assertTrue(fixture.is_file())
        self.assertEqual(analyzer.validate_config(fixture, {}), [])
        with open(fixture, encoding="utf-8") as handle:
            mutated = json.load(handle)
        mutated["ori5_reward_parameters"]["spectral_subspace"]["weight"] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_config.json"
            path.write_text(json.dumps(mutated), encoding="utf-8")
            errors = analyzer.validate_config(path, {})
        self.assertTrue(any("spectral_weight" in error for error in errors))

        with open(fixture, encoding="utf-8") as handle:
            mechanics_mutation = json.load(handle)
        mechanics_mutation["max_steps"] = 399
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_config.json"
            path.write_text(json.dumps(mechanics_mutation), encoding="utf-8")
            errors = analyzer.validate_config(path, {"max_steps": 400})
        self.assertTrue(any("config max_steps" in error for error in errors))

    def test_early_selection_and_selected_revocation_paths_are_distinct(self):
        jobs = []
        for arm in ("B", "W"):
            for seed in analyzer.SEEDS:
                jobs.append({
                    "task": {"name": "five_d2"}, "arm": arm, "seed": seed,
                    "output_directory": "unused", "expected_config": {},
                })

        def report(job, endpoint):
            candidate = job["arm"] == "W"
            seed = int(job["seed"])
            return {
                "task": "five_d2", "arm": job["arm"], "seed": seed,
                "endpoint": endpoint, "integrity_pass": True, "integrity_errors": [],
                "first_hit_episode": 100 if candidate else 200,
                "first_hit_proposals": (10000 + seed) if candidate else (20000 + seed),
                "first_hit_before_or_at_first_update": False,
                "proposal_censor": 100000, "strict_hits": 20,
                "tail_window": 512,
                "tail_strict_rate": 0.60 if candidate else 0.10,
                "tail512_strict_rate": 0.60 if candidate else 0.10,
                "tail1024_strict_rate": 0.60 if candidate else 0.10,
                "tail256_strict_rate": 0.60 if candidate else 0.10,
                "fixed_stochastic_strict_rate": 0.625 if candidate else 0.125,
                "total_rejected_actions": 0, "invalid_episode_count": 0,
                "mean_tail_return": 1.0, "update_count": 64,
                "last_actor_learning_rate": 5e-4,
                "last_critic_learning_rate": 1.5e-4,
            }

        with mock.patch.object(analyzer, "seed_report", side_effect=report):
            screen = analyzer.analyse_task(jobs, "five_d2", 512)
            self.assertEqual(screen["decision"]["selected_arm"], "W")
            selected = analyzer.analyse_selected(jobs, "five_d2", "W", 4096)
            self.assertFalse(selected["decision"]["revoke"])

    def test_arm_selection_stops_only_loser_three_and_keeps_selected_three(self):
        members = []
        for arm in ("B", "W"):
            for seed in runner.SEEDS:
                process = mock.Mock()
                process.poll.return_value = None
                members.append({
                    "job": {"arm": arm, "seed": seed},
                    "process": process,
                    "stdout": None,
                    "stderr": None,
                    "selected": False,
                    "stopped": False,
                    "status": "running",
                })
        active = list(members)
        runner.apply_arm_selection(members, active, "W")
        self.assertEqual([(x["job"]["arm"], x["job"]["seed"]) for x in active], [
            ("W", seed) for seed in runner.SEEDS
        ])
        for item in members:
            process = item["process"]
            if item["job"]["arm"] == "W":
                self.assertTrue(item["selected"])
                self.assertFalse(item["stopped"])
                process.terminate.assert_not_called()
            else:
                self.assertFalse(item["selected"])
                self.assertTrue(item["stopped"])
                self.assertEqual(item["status"], "not_selected")
                process.terminate.assert_called_once_with()

    def test_training_and_fixed_eval_integrity_fail_closed(self):
        fixture = (
            runner.SCRIPT_DIRECTORY / "experiments_ori6" / "screen"
            / "spectral_nonregression" / "P0X2RCS05" / "ori6"
            / "five_d2" / "B" / "seed3301"
        )
        with open(
            fixture / "training_history.csv", newline="", encoding="utf-8-sig"
        ) as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(analyzer._integrity_errors([row], 400), [])
        mismatch = dict(row)
        mismatch["success"] = "1" if row["strict_compact_success"] == "0" else "0"
        self.assertTrue(any(
            "success != strict_compact_success" in error
            for error in analyzer._integrity_errors([mismatch], 400)
        ))
        fractional = dict(row)
        fractional["steps"] = "1.5"
        self.assertTrue(analyzer._integrity_errors([fractional], 400))
        for field in (
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
            "ori6_spectral_reward_decomposition_error",
            "ori6_nonfinite_reward_component_count",
            "ori6_spectral_total_reward",
        ):
            blank = dict(row)
            blank[field] = ""
            self.assertTrue(
                analyzer._integrity_errors([blank], 400),
                f"blank {field} must fail closed",
            )
        with open(
            fixture / "stochastic_evaluation_history.csv",
            newline="", encoding="utf-8-sig",
        ) as handle:
            fixed = next(csv.DictReader(handle))
        self.assertEqual(analyzer._fixed_evaluation_errors(fixed, 1), [])
        bad_fixed = dict(fixed)
        bad_fixed["evaluation_episodes"] = "15"
        self.assertTrue(analyzer._fixed_evaluation_errors(bad_fixed, 1))

    def test_update_clock_and_learning_rates_are_exact(self):
        config = {
            "episodes": 10000,
            "batch_size": 8,
            "actor_learning_rate": 5e-4,
            "critic_learning_rate": 1.5e-4,
            "actor_lr_milestones": [2000, 4000, 7000],
            "actor_lr_factors": [0.3, 0.1, 0.03],
        }
        rows = [
            {"episode": "8", "actor_learning_rate": "0.0005", "critic_learning_rate": "0.00015"},
            {"episode": "16", "actor_learning_rate": "0.0005", "critic_learning_rate": "0.00015"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "updates.csv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            _, errors = analyzer.read_updates(path, 16, config)
            self.assertEqual(errors, [])
            rows[1]["episode"] = "15"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            _, errors = analyzer.read_updates(path, 16, config)
            self.assertTrue(any("missing update" in error for error in errors))
            self.assertTrue(any("unexpected update" in error for error in errors))

    def test_common_proposal_cap_excludes_late_hit(self):
        reports = []
        for seed, hit in zip(analyzer.SEEDS, (90, 110, None)):
            reports.append({
                "arm": "B", "seed": seed, "integrity_pass": True,
                "first_hit_proposals": hit,
                "first_hit_before_or_at_first_update": False,
                "strict_hits": int(hit is not None), "tail_strict_rate": 0.0,
                "tail256_strict_rate": 0.0,
                "fixed_stochastic_strict_rate": 0.0,
            })
        summary = analyzer._arm_summary(reports, 100)
        self.assertEqual(summary["uncapped_hit_seed_count"], 2)
        self.assertEqual(summary["hit_seed_count"], 1)
        candidate = [{
            "seed": seed, "first_hit_proposals": 100,
        } for seed in analyzer.SEEDS]
        control = [{
            "seed": seed, "first_hit_proposals": None,
        } for seed in analyzer.SEEDS]
        paired = analyzer._paired(candidate, control, 100)
        self.assertEqual((paired["wins"], paired["losses"], paired["ties"]), (3, 0, 0))
        self.assertTrue(all(row["candidate_event_by_cap"] for row in paired["per_seed"]))

    def test_4096_gate_and_two_warning_production_stop(self):
        jobs = [{
            "task": {"name": "five_d2"}, "arm": "W", "seed": seed,
            "output_directory": "unused", "expected_config": {},
        } for seed in analyzer.SEEDS]

        def weak_report(job, endpoint):
            return {
                "task": "five_d2", "arm": "W", "seed": job["seed"],
                "endpoint": endpoint, "integrity_pass": True, "integrity_errors": [],
                "first_hit_episode": 100, "first_hit_proposals": 10000,
                "first_hit_before_or_at_first_update": False,
                "proposal_censor": 100000, "strict_hits": 100,
                "tail_window": 512, "tail_strict_rate": 0.40,
                "tail512_strict_rate": 0.40, "tail1024_strict_rate": 0.60,
                "tail256_strict_rate": 0.40,
                "fixed_stochastic_strict_rate": 0.625,
                "total_rejected_actions": 0, "invalid_episode_count": 0,
                "mean_tail_return": 1.0, "update_count": 1,
                "last_actor_learning_rate": 1e-4, "last_critic_learning_rate": 1e-4,
            }

        with mock.patch.object(analyzer, "seed_report", side_effect=weak_report):
            gate = analyzer.analyse_selected(jobs, "five_d2", "W", 4096)
            self.assertTrue(gate["decision"]["revoke"])
            first = analyzer.analyse_selected(jobs, "five_d2", "W", 5120, 0)
            self.assertFalse(first["decision"]["revoke"])
            self.assertEqual(first["production_monitor"]["consecutive_warning_count"], 1)
            second = analyzer.analyse_selected(jobs, "five_d2", "W", 6144, 1)
            self.assertTrue(second["decision"]["revoke"])
            self.assertEqual(second["production_monitor"]["consecutive_warning_count"], 2)

    def test_d3_initial_policy_hits_alone_are_not_learning_evidence(self):
        jobs = []
        for arm in ("B", "W"):
            for seed in analyzer.SEEDS:
                jobs.append({
                    "task": {"name": "five_d3"}, "arm": arm, "seed": seed,
                    "output_directory": "unused", "expected_config": {},
                })

        def initial_only(job, endpoint):
            candidate = job["arm"] == "W"
            return {
                "task": "five_d3", "arm": job["arm"], "seed": job["seed"],
                "endpoint": endpoint, "integrity_pass": True, "integrity_errors": [],
                "first_hit_episode": 1 if candidate else None,
                "first_hit_proposals": 100 if candidate else None,
                "first_hit_before_or_at_first_update": candidate,
                "proposal_censor": 100000, "strict_hits": int(candidate),
                "tail_window": 512, "tail_strict_rate": 0.0,
                "tail512_strict_rate": 0.0, "tail1024_strict_rate": 0.0,
                "tail256_strict_rate": 0.0, "fixed_stochastic_strict_rate": 0.0,
                "total_rejected_actions": 0, "invalid_episode_count": 0,
                "mean_tail_return": 0.0, "update_count": 64,
                "last_actor_learning_rate": 1e-4, "last_critic_learning_rate": 1e-4,
            }

        with mock.patch.object(analyzer, "seed_report", side_effect=initial_only):
            report = analyzer.analyse_task(jobs, "five_d3", 512)
        self.assertEqual(report["arm_summaries"]["W"]["pre_update_hit_seed_count"], 3)
        self.assertFalse(report["learning_evidence_beyond_initial_policy"]["W"])
        self.assertIsNone(report["decision"]["selected_arm"])

    def test_duplicate_scan_excludes_self_and_unrelated_python(self):
        records = [
            {"ProcessId": os.getpid(), "CommandLine": runner.RUNNER_NAME},
            {"ProcessId": 22222, "CommandLine": "python " + str(
                runner.SCRIPT_DIRECTORY / "ori_codex6_spectral_subspace.py"
            )},
            {"ProcessId": 33333, "CommandLine": "python unrelated.py"},
        ]
        completed = mock.Mock(stdout=json.dumps(records))
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            self.assertEqual(runner.active_duplicates(Path("C:/tmp/C10NS")), [22222])

    def test_delayed_cohort_rehash_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = runner.manifest_payload(Path(temporary) / "root", 12)
        with mock.patch.object(runner, "source_hashes", return_value={"changed": "yes"}):
            with self.assertRaisesRegex(RuntimeError, "source binding changed"):
                runner.verify_cohort_binding(payload, "five_d2")

    def test_exact_checkpoint_episode_parser_rejects_fraction(self):
        with self.assertRaises(ValueError):
            runner.exact_nonnegative_integer("512.5", "checkpoint episode")

        fixture = (
            runner.SCRIPT_DIRECTORY / "experiments_ori6" / "screen"
            / "spectral_nonregression" / "P0X2RCS05" / "ori6"
            / "five_d2" / "B" / "seed3301"
        )
        self.assertEqual(runner.checkpoint_episode(fixture), 512)
        with tempfile.TemporaryDirectory() as temporary:
            corrupt = Path(temporary)
            (corrupt / "checkpoint.pt").write_bytes(b"not a checkpoint")
            self.assertEqual(runner.checkpoint_episode(corrupt), 0)

    def test_natural_completion_requires_process_exit_commit_marker(self):
        item = {
            "job": {"output_directory": "unused"},
            "process": mock.Mock(),
        }
        item["process"].poll.return_value = None
        with mock.patch.object(runner, "completed_endpoint", return_value=10000):
            self.assertFalse(runner._ready([item], 10000))
            item["process"].poll.return_value = 0
            self.assertTrue(runner._ready([item], 10000))


if __name__ == "__main__":
    unittest.main()
