import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_ori_codex8_spectral_staged as runner


def _nested_wins(selected):
    return {
        arm: {
            opponent: {
                "wins": 2 if arm == selected else 1,
                "losses": 1 if arm == selected else 2,
                "ties": 0,
                "per_seed": [],
            }
            for opponent in runner.ARMS if opponent != arm
        }
        for arm in runner.ARMS
    }


def _selection(arm):
    return {
        "selected_arm": arm,
        "selected_seed": None,
        "promote_selected_to_10000": arm is not None,
        "selection_rule_version": runner.SELECTION_RULE_VERSION,
        "paired_wins": _nested_wins(arm),
    }


class _DoneProcess:
    def poll(self):
        return 0


class _Handle:
    closed = False

    def close(self):
        self.closed = True


class _FakeAnalysis:
    def __init__(self):
        self.production_calls = []

    def seed_report(self, root, task, arm, seed, endpoint):
        self.production_calls.append((task, arm, seed, endpoint))
        return {
            "task": task,
            "arm": arm,
            "seed": seed,
            "endpoint": endpoint,
        }

    @staticmethod
    def production_decision(report, baseline, prior_warning_count):
        return {"stop": False, "consecutive_warning_count": 0}

    @staticmethod
    def write_production_artifacts(report, production, output):
        return str(output)


class Codex8SpectralStagedRunnerTests(unittest.TestCase):
    def test_fixed_cohort_paths_and_arm_weights(self):
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "fresh"
            jobs = runner.build_jobs(root)
        self.assertEqual(len(jobs), 18)
        identities = {
            (job["task"]["name"], job["arm"], job["seed"])
            for job in jobs
        }
        self.assertEqual(len(identities), 18)
        expected_parent = {
            ("torus2", "legacy"): 0.05,
            ("torus2", "scale"): 0.5,
            ("torus2", "phase_v2"): 0.0,
            ("water1", "legacy"): 0.05,
            ("water1", "scale"): 0.25,
            ("water1", "phase_v2"): 0.0,
        }
        expected_lambda = {
            ("torus2", "legacy"): 0.05,
            ("torus2", "scale"): 0.05,
            ("torus2", "phase_v2"): 1.0,
            ("water1", "legacy"): 0.05,
            ("water1", "scale"): 0.025,
            ("water1", "phase_v2"): 0.5,
        }
        for job in jobs:
            task, arm, seed = (
                job["task"]["name"], job["arm"], job["seed"]
            )
            self.assertEqual(
                Path(job["output_directory"]),
                (root / task / arm / f"seed{seed}").resolve(),
            )
            self.assertAlmostEqual(
                job["effective_parent_ori6_spectral_weight"],
                expected_parent[(task, arm)],
            )
            self.assertAlmostEqual(
                job["ori8_lambda"], expected_lambda[(task, arm)]
            )

    def test_commands_bind_reward_only_and_final_parent_profiles(self):
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "fresh"
            jobs = runner.build_jobs(root)
        for job in jobs:
            command = job["command"]
            task = job["task"]["name"]
            self.assertEqual(
                command[command.index("--ori8-spectral-arm") + 1], job["arm"]
            )
            self.assertAlmostEqual(
                float(command[command.index("--ori8-lambda") + 1]),
                job["ori8_lambda"],
            )
            self.assertEqual(
                command[command.index("--ori6-spectral-weight") + 1], "0"
            )
            self.assertEqual(
                float(command[command.index("--near-miss-spread-error-far") + 1]),
                1.0 if task == "torus2" else 0.12,
            )
            self.assertEqual(
                int(command[command.index("--max-steps") + 1]),
                200 if task == "torus2" else 400,
            )
            self.assertIn("--no-random-search", command)
            self.assertIn("--ori6-zero-reject-reward", command)
            self.assertIn("--data-dir", command)

    def test_all_eighteen_commands_parse_to_exact_runtime_weights(self):
        import ori_codex8_3d as core3d

        expected = {
            ("torus2", "legacy"): 0.05,
            ("torus2", "scale"): 0.5,
            ("torus2", "phase_v2"): 0.0,
            ("water1", "legacy"): 0.05,
            ("water1", "scale"): 0.25,
            ("water1", "phase_v2"): 0.0,
        }
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            jobs = runner.build_jobs(Path(directory) / "parse_only")
            parsed = [
                (job, core3d.parse_args(job["command"][3:])) for job in jobs
            ]
        self.assertEqual(len(parsed), 18)
        for job, args in parsed:
            key = (job["task"]["name"], job["arm"])
            self.assertEqual(args.ori8_spectral_arm, job["arm"])
            self.assertAlmostEqual(args.ori8_lambda, job["ori8_lambda"])
            self.assertAlmostEqual(args.ori6_spectral_weight, expected[key])
            self.assertEqual(args.update_method, "actor_critic")
            self.assertEqual(args.episodes, 10000)
            self.assertFalse(args.random_search)

    def test_parent_architecture_and_source_closure_are_explicit(self):
        for name in (
            "ori_codex.py",
            "ori_codex2.py",
            "ori_codex3.py",
            "ori_codex4.py",
            "ori_codex4_3d.py",
            "ori_codex6_zero_reject.py",
            "ori_codex6_zero_reject_3d.py",
        ):
            self.assertIn(name, runner.SOURCE_FILES)
            self.assertIn(name, runner.PARENT_ARCHITECTURE_FILES)

    def test_thread_and_checkpoint_contracts(self):
        result = runner.verify_thread_contract()
        self.assertEqual(result["torch_intraop"], 1)
        self.assertEqual(set(result["env"].values()), {"1"})
        self.assertEqual(
            runner.reached_unchecked(runner.SCREEN_CHECKPOINTS, set(), 10000),
            [512, 1024, 2048, 4096],
        )
        self.assertEqual(
            runner.reached_unchecked(
                runner.PRODUCTION_CHECKPOINTS, set(), 10000
            ),
            [5120, 6144, 7168, 8192, 9216, 10000],
        )

    def test_task_screen_scheduler_preserves_complete_cohorts(self):
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            jobs = runner.build_jobs(Path(directory) / "fresh")
        torus, water = jobs[0], jobs[9]
        self.assertTrue(runner.task_screen_launchable(torus, {}))
        self.assertFalse(runner.task_screen_launchable(water, {}))
        self.assertTrue(
            runner.task_screen_launchable(water, {"torus2": {}})
        )

    def test_selection_requires_new_arm_and_two_wins_against_each_arm(self):
        decision = _selection("scale")
        self.assertEqual(runner.selected_arm_from_decision(decision), "scale")
        weak = json.loads(json.dumps(decision))
        weak["paired_wins"]["scale"]["phase_v2"]["wins"] = 1
        with self.assertRaisesRegex(RuntimeError, "phase_v2"):
            runner.selected_arm_from_decision(weak)
        wrong_rule = json.loads(json.dumps(decision))
        wrong_rule["selection_rule_version"] = "mutated"
        with self.assertRaisesRegex(RuntimeError, "authority"):
            runner.selected_arm_from_decision(wrong_rule)
        best_seed = json.loads(json.dumps(decision))
        best_seed["selected_seed"] = 8101
        with self.assertRaisesRegex(RuntimeError, "all three"):
            runner.selected_arm_from_decision(best_seed)
        with self.assertRaisesRegex(RuntimeError, "Legacy"):
            runner.selected_arm_from_decision(_selection("legacy"))
        self.assertIsNone(runner.selected_arm_from_decision(_selection(None)))

    def test_strong_early_promotion_is_independently_fail_closed(self):
        decision = _selection("phase_v2")
        decision.update({
            "strong_early_promotion": True,
            "all_task_integrity": True,
            "all_candidate_seeds_integrity_pass": True,
            "candidate_farming_gate_pass": True,
            "early_evidence_seed_count": 2,
        })
        self.assertEqual(
            runner.strong_early_arm_from_decision(decision), "phase_v2"
        )
        for field in (
            "all_task_integrity",
            "all_candidate_seeds_integrity_pass",
            "candidate_farming_gate_pass",
        ):
            mutation = dict(decision)
            mutation[field] = False
            with self.assertRaises(RuntimeError):
                runner.strong_early_arm_from_decision(mutation)
        weak = dict(decision)
        weak["early_evidence_seed_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "evidence"):
            runner.strong_early_arm_from_decision(weak)
        unqualified = _selection("scale")
        unqualified["strong_early_promotion"] = False
        with self.assertRaisesRegex(RuntimeError, "Unqualified"):
            runner.strong_early_arm_from_decision(unqualified)

    def test_manifest_dry_build_is_side_effect_free_and_short(self):
        synthetic_hashes = {name: "a" * 64 for name in runner.SOURCE_FILES}
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "must_not_exist"
            with mock.patch.object(runner, "analysis", object()), mock.patch.object(
                runner, "source_hashes", return_value=synthetic_hashes
            ):
                payload = runner.manifest_payload(
                    root, "a" * 64, "a" * 64
                )
            self.assertFalse(root.exists())
        self.assertEqual(len(payload["jobs"]), 18)
        self.assertEqual(payload["max_parallel"], 12)
        self.assertEqual(payload["training_stop_local"], "09:20")
        self.assertEqual(payload["report_deadline_local"], "10:00")
        self.assertEqual(
            payload["protocol"],
            "codex8_spectral_reward_only_three_arm_staged_replay1",
        )
        self.assertTrue(payload["fresh_replay_from_episode_zero"])
        self.assertIn("no checkpoint", payload["replay_policy"])
        self.assertEqual(
            payload["selection_rule_version"], runner.SELECTION_RULE_VERSION
        )
        self.assertLess(payload["path_budget"]["maximum_known_characters"], 240)
        content = dict(payload)
        fingerprint = content.pop("manifest_fingerprint")
        self.assertEqual(fingerprint, runner.canonical_digest(content))

    def test_replay_root_is_fresh_and_forensic_parent_is_fully_bound(self):
        self.assertEqual(runner.DEFAULT_ROOT.name, "C8SP2R1")
        self.assertEqual(runner.FORENSIC_PARENT_ROOT.name, "C8SP2")
        self.assertFalse(runner.DEFAULT_ROOT.exists())
        binding = runner.forensic_parent_binding()
        self.assertEqual(
            binding["manifest_fingerprint"],
            runner.CONTAMINATED_MANIFEST_FINGERPRINT,
        )
        self.assertFalse(binding["runner_report_present"])
        self.assertGreaterEqual(binding["file_count"], 100)
        self.assertTrue(Path(binding["manifest_file"]["path"]).is_file())
        self.assertEqual(len(binding["manifest_file"]["sha256"]), 64)
        for arm in runner.ARMS:
            for seed in runner.SEEDS:
                self.assertIn(
                    f"torus2/{arm}/seed{seed}/checkpoint.pt",
                    binding["files"],
                )
        jobs = runner.build_jobs(runner.DEFAULT_ROOT)
        parent = runner.FORENSIC_PARENT_ROOT.resolve()
        for job in jobs:
            destination = Path(job["output_directory"])
            self.assertNotEqual(destination, parent)
            self.assertNotIn(parent, destination.parents)
            self.assertTrue(
                job["output_directory"].startswith(str(runner.DEFAULT_ROOT.resolve()))
            )

    def test_real_manifest_roundtrip_and_protocol_mutation_fail_closed(self):
        hashes = runner.source_hashes()
        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            directory = Path(directory)
            root = directory / "fresh"
            payload = runner.manifest_payload(
                root,
                hashes[runner.CORE_PLANAR],
                hashes[runner.CORE_SPATIAL],
            )
            path = directory / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                runner.validate_manifest(path)["manifest_fingerprint"],
                payload["manifest_fingerprint"],
            )
            mutation = json.loads(json.dumps(payload))
            mutation["promotion_rule"] = "mutated"
            content = dict(mutation)
            content.pop("manifest_fingerprint")
            mutation["manifest_fingerprint"] = runner.canonical_digest(content)
            path.write_text(json.dumps(mutation), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
                runner.validate_manifest(path)

    def test_fast_finish_drains_and_revokes_early_selection_at_4096(self):
        fake_analysis = _FakeAnalysis()
        analysis_calls = []
        launch_order = []
        processes = []
        state = {"water_launched": False, "campaign_done": False}

        class ControlledProcess:
            def __init__(self, job):
                self.job = job
                self.terminated = False

            def poll(self):
                if self.terminated:
                    return -15
                return 0 if state["campaign_done"] else None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return -15 if self.terminated else 0

            def kill(self):
                self.terminated = True

        def fake_launch(root, job):
            launch_order.append(job["task"]["name"])
            if job["task"]["name"] == "water1":
                state["water_launched"] = True
            process = ControlledProcess(job)
            processes.append(process)
            return {
                "job": job,
                "process": process,
                "stdout": _Handle(),
                "stderr": _Handle(),
                "checked": set(),
                "selected": False,
                "stopped": False,
                "production_warning_count": 0,
            }

        def fake_generate(root, endpoint, task, arms):
            analysis_calls.append((task, endpoint, tuple(arms)))
            revoke = bool(
                task == "torus2"
                and endpoint == 4096
                and tuple(arms) == ("scale",)
            )
            payload = {
                "arm_decisions": {
                    task: {
                        arm: {
                            "stop": revoke and arm == "scale",
                            "stop_kind": (
                                "integrity"
                                if revoke and arm == "scale" else None
                            ),
                        }
                        for arm in arms
                    }
                },
                "task_decisions": {},
            }
            if endpoint == 2048 and len(arms) > 1:
                if task == "torus2":
                    payload["task_decisions"][task] = _selection("scale")
                    payload["task_decisions"][task].update({
                        "strong_early_promotion": True,
                        "all_task_integrity": True,
                        "all_candidate_seeds_integrity_pass": True,
                        "candidate_farming_gate_pass": True,
                        "early_evidence_seed_count": 2,
                    })
                else:
                    payload["task_decisions"][task] = {
                        "selected_arm": None,
                        "selected_seed": None,
                        "promote_selected_to_10000": False,
                        "strong_early_promotion": False,
                    }
            if endpoint == 4096 and len(arms) > 1:
                payload["task_decisions"][task] = _selection(
                    "scale" if task == "torus2" else "phase_v2"
                )
                if task == "water1":
                    state["campaign_done"] = True
            return payload, f"{task}:{endpoint}:{'_'.join(arms)}"

        def fake_checkpoint(path):
            if "torus2" in Path(path).parts and not state["water_launched"]:
                return 2048
            return 10000

        with tempfile.TemporaryDirectory(dir=runner.SCRIPT_DIRECTORY) as directory:
            root = Path(directory) / "fresh"
            jobs = runner.build_jobs(root)
            payload = {
                "experiment_root": str(root),
                "jobs": jobs,
                "protocol": "test",
                "manifest_fingerprint": "f" * 64,
                "source_sha256": {},
                "parent_architecture_sha256": {},
                "job_order": list(runner.TASKS),
                "max_parallel": 12,
                "training_stop_local": "23:59",
                "report_deadline_local": "23:59",
                "hidden_ledger_pomdp_caveat": "test",
            }
            with mock.patch.object(runner, "analysis", fake_analysis), mock.patch.object(
                runner, "active_duplicates", return_value=[]
            ), mock.patch.object(
                runner, "verify_thread_contract", return_value={"ok": True}
            ), mock.patch.object(
                runner, "launch_job", side_effect=fake_launch
            ), mock.patch.object(
                runner, "checkpoint_episode", side_effect=fake_checkpoint
            ), mock.patch.object(
                runner, "generate_analysis", side_effect=fake_generate
            ):
                report = runner.execute(
                    payload, 12, 0.0, "23:59", "23:59"
                )

        self.assertEqual(launch_order[:9], ["torus2"] * 9)
        self.assertEqual(launch_order[9:], ["water1"] * 9)
        for task in runner.TASKS:
            combined_4096 = [
                call for call in analysis_calls
                if call[0] == task and call[1] == 4096
                and len(call[2]) > 1
            ]
            self.assertEqual(
                len(combined_4096), 0 if task == "torus2" else 1
            )
        self.assertEqual(
            report["task_selections"]["torus2"]["selection_endpoint"], 2048
        )
        self.assertEqual(
            report["task_selections"]["water1"]["selection_endpoint"], 4096
        )
        selection = report["task_selections"]["torus2"]
        self.assertIsNone(selection["selected_arm"])
        self.assertFalse(selection["promote_selected_to_10000"])
        self.assertEqual(selection["comparative_winner_arm"], "scale")
        self.assertEqual(selection["promotion_revoked_endpoint"], 4096)
        self.assertEqual(selection["promotion_revoked_kind"], "integrity")
        self.assertEqual(selection["continued_seeds"], [])
        selected = {
            (row["task"], row["arm"], row["seed"])
            for row in report["endpoints"] if row["selected"]
        }
        self.assertEqual(selected, {
            ("water1", "phase_v2", seed) for seed in runner.SEEDS
        })
        production_endpoints = [
            call for call in fake_analysis.production_calls if call[3] != 4096
        ]
        self.assertEqual(len(production_endpoints), 18)
        self.assertFalse(any(call[0] == "torus2" for call in production_endpoints))
        self.assertTrue(all(
            process.terminated
            for process in processes
            if process.job["task"]["name"] == "torus2"
        ))
        self.assertTrue(all(
            row["formal_analyzed_endpoint"] == 10000
            for row in report["endpoints"] if row["selected"]
        ))


if __name__ == "__main__":
    unittest.main()
