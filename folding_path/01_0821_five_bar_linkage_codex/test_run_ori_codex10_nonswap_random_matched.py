import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_ori_codex10_nonswap_random_matched as runner


class RandomMatchedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name) / "formal_root"
        cls.manifest = runner.build_manifest(root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_manifest_exact_matrix_and_commands(self):
        manifest = self.manifest
        runner.validate_manifest(manifest)
        self.assertEqual(len(manifest["jobs"]), 30)
        self.assertEqual(
            {(j["task"], j["seed"]) for j in manifest["jobs"]},
            {(task, seed) for task in runner.TASK_BUDGETS for seed in runner.SEEDS},
        )
        for job in manifest["jobs"]:
            budget = runner.TASK_BUDGETS[job["task"]]
            self.assertIn("--no-resume", job["command"])
            self.assertEqual(job["checkpoint_interval"], budget + 1)
            index = job["command"].index("--checkpoint-every-episodes")
            self.assertEqual(job["command"][index + 1], str(budget + 1))
            self.assertEqual(job["thread_environment"], runner.THREAD_ENV)
            self.assertEqual(
                job["c10_environment"]["ori6_crossing_policy"],
                "reject_candidate_and_continue",
            )

    def test_manifest_mutations_fail(self):
        for mutation in (
            "source", "budget", "resume", "command", "model",
            "environment", "physical_cap",
        ):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.manifest)
                identity = bad["fingerprint_identity"]
                if mutation == "source":
                    identity["source_sha256"]["ori_codex3.py"] = "0" * 64
                elif mutation == "budget":
                    identity["jobs"][0]["mechanics_budget"] += 1
                elif mutation == "resume":
                    identity["jobs"][0]["resume"] = True
                elif mutation == "model":
                    identity["jobs"][0]["model_sha256"]["node"] = "0" * 64
                elif mutation == "environment":
                    identity["jobs"][0]["c10_environment"]["max_steps"] = 399
                elif mutation == "physical_cap":
                    identity["max_total_trainers"] = 13
                else:
                    identity["jobs"][0]["command"].remove("--no-resume")
                if mutation not in ("source", "physical_cap"):
                    job = identity["jobs"][0]
                    body = {
                        key: value for key, value in job.items()
                        if key != "job_fingerprint"
                    }
                    job["job_fingerprint"] = runner.canonical_digest(body)
                bad.update(identity)
                bad["fingerprint_identity"] = identity
                bad["manifest_fingerprint"] = runner.canonical_digest(identity)
                with self.assertRaises(RuntimeError):
                    runner.validate_manifest(bad)

    def synthetic_completed_job(self, parent: Path):
        job = copy.deepcopy(self.manifest["jobs"][0])
        output = parent / "job"
        job["output_root"] = str(output.resolve())
        job["mechanics_budget"] = 800
        root = runner.baseline.make_root_config(job["task"], 800, [job["seed"]])
        job["random_fingerprint_identity"] = root["fingerprint_identity"]
        job["random_root_fingerprint"] = root["root_fingerprint"]
        trial = output / "trials" / "uniform" / f"seed{job['seed']}"
        trial.mkdir(parents=True)
        runner.atomic_json(output / "run_config.json", root)
        trial_config = {
            "schema_version": runner.baseline.SCHEMA_VERSION,
            "harness_version": runner.baseline.HARNESS_VERSION,
            "root_fingerprint": root["root_fingerprint"],
            "method": runner.baseline.METHOD,
            "seed": job["seed"], "task": job["task"],
            "mechanics_budget": 800,
            "episode_cap": None,
            "max_steps_per_episode": 400,
            "candidate": runner.baseline.spectral.CANDIDATE_NAME,
            "crossing_policy": runner.baseline.rejected.CROSSING_POLICY,
            "spectral_formula_sha256": (
                runner.baseline.spectral.SPECTRAL_FORMULA_SHA256
            ),
            "budget_stop_reward": (
                "unchanged candidate reward from final ordinary valid transition"
            ),
        }
        runner.atomic_json(trial / "run_config.json", trial_config)
        audit = {
            "candidate_identity_all_rows": True,
            "committed_self_crossing_count": 0,
            "ori6_nonfinite_reward_component_count": 0,
            "spectral_nonfinite_component_count": 0,
            "maximum_abs_ori6_reward_decomposition_error": 0.0,
            "maximum_abs_ori6_step_reward_decomposition_error": 0.0,
            "maximum_abs_rejected_crossing_decomposition_error": 0.0,
            "maximum_abs_spectral_reward_decomposition_error": 0.0,
            "maximum_abs_spectral_step_reward_decomposition_error": 0.0,
        }
        summary = {
            **trial_config, "complete": True, "reward_integrity_pass": True,
            "reward_integrity": audit, "success": 0, "censored": 1,
            "censor_reason": "mechanics_budget", "mechanical_steps": 800,
            "projector_calls": 800,
            "first_success_episode": None,
            "first_success_mechanical_steps": None,
            "first_success_projector_calls": None,
            "first_hit_path": None,
            "terminal_flatness_ratio": None, "terminal_spread": None,
            "episodes": 2,
            "action_trace_chain": runner.baseline.ACTION_TRACE_CHAIN,
            "action_chain_sha256": "a" * 64,
            "action_prefix_sha256_by_mechanics": {"800": "a" * 64},
        }
        runner.atomic_json(trial / "summary.json", summary)
        runner.atomic_json(output / "aggregate.json", {
            "schema_version": runner.baseline.SCHEMA_VERSION,
            "harness_version": runner.baseline.HARNESS_VERSION,
            "candidate": runner.baseline.spectral.CANDIDATE_NAME,
            "task": job["task"], "method": runner.baseline.METHOD,
            "root_fingerprint": root["root_fingerprint"],
            "mechanics_budget": 800,
            "complete": True, "seeds": [job["seed"]], "missing_seeds": [],
            "expected_trials": 1, "completed_trials": 1, "successes": 0,
            "censored": 1, "success_at_budget": 0.0,
            "first_success_mechanical_steps": [],
            "restricted_mean_first_hit_mechanical_steps": 800.0,
            "rmst_censor_assignment": (
                "all censored trials assigned the common mechanics budget"
            ),
            "trial_summaries": [summary],
            "all_reward_integrity_pass": True,
        })
        fields = [
            "method", "seed", "episode", "success",
            "mechanical_steps", "cumulative_mechanical_steps",
            "projector_calls", "cumulative_projector_calls",
            "strict_compact_success", "self_crossing", "correction_failed",
            "timeout", "no_modes", "wrong_flat", "ori6_crossing_policy",
            "ori6_candidate", "ori6_spectral_candidate",
            "ori6_spectral_parent_candidate", "ori6_spectral_formula_sha256",
            "action_trace_sha256", "ori6_rejected_crossing_count",
            "ori6_rejected_crossing_rate",
            "ori6_nonfinite_reward_component_count",
            "ori6_spectral_nonfinite_component_count",
            "ori6_reward_decomposition_error",
            "ori6_maximum_abs_step_reward_decomposition_error",
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
            "ori6_spectral_reward_decomposition_error",
            "ori6_spectral_maximum_abs_step_reward_decomposition_error",
            "ori6_spectral_weight", "ori6_spectral_relative_epsilon",
        ]
        def row(episode, cumulative):
            return {
                "method": "uniform", "seed": str(job["seed"]),
                "episode": str(episode), "success": "0",
                "mechanical_steps": "400",
                "cumulative_mechanical_steps": str(cumulative),
                "projector_calls": "400",
                "cumulative_projector_calls": str(cumulative),
                "strict_compact_success": "0", "self_crossing": "0",
                "correction_failed": "0", "timeout": "1", "no_modes": "0",
                "wrong_flat": "0",
                "ori6_crossing_policy": "reject_candidate_and_continue",
                "ori6_candidate": runner.baseline.rejected.CANDIDATE_NAME,
                "ori6_spectral_candidate": runner.baseline.spectral.CANDIDATE_NAME,
                "ori6_spectral_parent_candidate": (
                    runner.baseline.rejected.CANDIDATE_NAME
                ),
                "ori6_spectral_formula_sha256": (
                    runner.baseline.spectral.SPECTRAL_FORMULA_SHA256
                ),
                "action_trace_sha256": "b" * 64,
                "ori6_rejected_crossing_count": "0",
                "ori6_rejected_crossing_rate": "0.0",
                "ori6_nonfinite_reward_component_count": "0",
                "ori6_spectral_nonfinite_component_count": "0",
                "ori6_reward_decomposition_error": "0.0",
                "ori6_maximum_abs_step_reward_decomposition_error": "0.0",
                "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0.0",
                "ori6_spectral_reward_decomposition_error": "0.0",
                "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0.0",
                "ori6_spectral_weight": "0.05",
                "ori6_spectral_relative_epsilon": "1e-12",
            }
        rows = [row(1, 400), row(2, 800)]
        with open(trial / "episode_history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return job, trial

    def test_completed_artifact_positive_and_partial_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job, trial = self.synthetic_completed_job(Path(directory))
            result = runner.validate_completed_job(job)
            self.assertEqual(result["censored"], 1)
            summary_path = trial / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["complete"] = False
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partial"):
                runner.validate_completed_job(job)

    def test_success_artifact_and_first_hit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            job, trial = self.synthetic_completed_job(Path(directory))
            summary_path = trial / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({
                "success": 1, "censored": 0, "censor_reason": None,
                "first_success_episode": 2,
                "first_success_mechanical_steps": 800,
                "first_success_projector_calls": 800,
                "first_hit_path": "first_hit_path",
                "terminal_flatness_ratio": 0.0005,
                "terminal_spread": 2.0,
            })
            runner.atomic_json(summary_path, summary)
            history_path = trial / "episode_history.csv"
            with open(history_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[-1]["success"] = "1"
            rows[-1]["strict_compact_success"] = "1"
            rows[-1]["timeout"] = "0"
            with open(history_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate_path = Path(job["output_root"]) / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            aggregate.update({
                "successes": 1, "censored": 0, "success_at_budget": 1.0,
                "first_success_mechanical_steps": [800],
                "trial_summaries": [summary],
            })
            runner.atomic_json(aggregate_path, aggregate)
            hit = trial / "first_hit_path"
            hit.mkdir()
            runner.atomic_json(
                hit / "actions.json",
                [{"step": index} for index in range(1, 401)],
            )
            for index in range(401):
                (hit / f"step{index}.dat").write_bytes(b"0\n")
            result = runner.validate_completed_job(job)
            self.assertEqual(result["success"], 1)

    def test_history_mutations_fail(self):
        mutations = (
            ("episode", "1.5"),
            ("cumulative_mechanical_steps", "799"),
            ("strict_compact_success", "2"),
            ("self_crossing", "1"),
            ("action_trace_sha256", "not-a-hash"),
            ("ori6_rejected_crossing_rate", "nan"),
            ("ori6_reward_decomposition_error", "garbage"),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                job, trial = self.synthetic_completed_job(Path(directory))
                path = trial / "episode_history.csv"
                with open(path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                    fields = list(rows[0])
                rows[-1][field] = value
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(RuntimeError):
                    runner.validate_completed_job(job)

    def test_checkpoint_is_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            job, trial = self.synthetic_completed_job(Path(directory))
            (trial / "checkpoint.pt").write_bytes(b"not-authority")
            with self.assertRaisesRegex(RuntimeError, "Checkpoint"):
                runner.validate_completed_job(job)

    def test_capacity_constants_and_deadline(self):
        self.assertEqual(runner.MAX_RANDOM_PARALLEL, 10)
        self.assertEqual(runner.MAX_TOTAL_TRAINERS, 12)
        self.assertGreater(runner.deadline_epoch(runner.DEADLINE_LOCAL), 0)

    def test_bound_manifest_mutation_and_deadline_crossing_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            runner.atomic_json(path, self.manifest)
            loaded, digest = runner.load_hashed_json(path)
            self.assertEqual(loaded, self.manifest)
            runner.verify_manifest_file(path, digest)
            mutated = copy.deepcopy(self.manifest)
            mutated["generated_utc"] = "mutated-after-binding"
            runner.atomic_json(path, mutated)
            with self.assertRaisesRegex(RuntimeError, "changed after binding"):
                runner.verify_manifest_file(path, digest)
        deadline = 100.0
        with mock.patch.object(runner.time, "time", side_effect=[99.9, 100.0]):
            self.assertTrue(runner.launch_before_deadline(deadline))
            self.assertFalse(runner.launch_before_deadline(deadline))

    def test_exact_process_count_and_malformed_marker(self):
        c10 = runner.load_json(runner.C10_MANIFEST)
        c10_output = c10["jobs"][0]["output_directory"]
        random_output = self.manifest["jobs"][0]["output_root"]
        records = [
            {"pid": 101, "name": "python.exe", "command_line": (
                f'python ori_codex6_spectral_subspace.py --output-dir "{c10_output}"'
            )},
            {"pid": 102, "name": "python.exe", "command_line": (
                f'python "{runner.BASELINE}" --output-root "{random_output}"'
            )},
            {"pid": 103, "name": "python.exe", "command_line": (
                "python pylsp.py --stdio"
            )},
        ]
        with mock.patch.object(runner, "process_records", return_value=records):
            count, workers, unknown = runner.active_trainer_count(self.manifest)
        self.assertEqual(count, 2)
        self.assertEqual({row["pid"] for row in workers}, {101, 102})
        self.assertEqual(unknown, [])
        records[1]["command_line"] = f'python "{runner.BASELINE}" --no-resume'
        with mock.patch.object(runner, "process_records", return_value=records):
            count, workers, unknown = runner.active_trainer_count(self.manifest)
        self.assertIn(102, unknown)

    def test_owned_child_transition_is_not_external_unknown(self):
        class FakeProcess:
            def __init__(self, pid, returncode):
                self.pid = pid
                self.returncode = returncode

            def poll(self):
                return self.returncode

        workers = [{"pid": pid, "kind": "c10", "output": str(pid)}
                   for pid in range(101, 108)]
        owned = {25036: FakeProcess(25036, None)}
        total, external = runner.reconcile_process_authority(
            workers, [25036], owned
        )
        self.assertEqual(total, 8)
        self.assertEqual(external, [])
        owned[25036].returncode = 0
        total, external = runner.reconcile_process_authority(
            workers + [{"pid": 25036, "kind": "random", "output": "owned"}],
            [25036], owned,
        )
        self.assertEqual(total, 7)
        self.assertEqual(external, [])
        total, external = runner.reconcile_process_authority(
            workers, [25036, 99999], owned
        )
        self.assertEqual(total, 7)
        self.assertEqual(external, [99999])
        self.assertEqual(
            runner.final_pid_zero_errors(
                workers + [{"pid": 25036, "kind": "random", "output": "owned"}],
                [25036], owned,
            ),
            {"owned_live": [], "external_unknown": [], "external_random": []},
        )
        errors = runner.final_pid_zero_errors(
            workers + [
                {"pid": 25036, "kind": "random", "output": "owned"},
                {"pid": 99998, "kind": "random", "output": "external"},
            ],
            [25036, 99999], owned,
        )
        self.assertEqual(errors["owned_live"], [])
        self.assertEqual(errors["external_unknown"], [99999])
        self.assertEqual(errors["external_random"], [99998])
        owned[25036].returncode = None
        self.assertEqual(
            runner.final_pid_zero_errors([], [], owned)["owned_live"],
            [25036],
        )

    def test_quoted_option_parser(self):
        value = runner.option_from_command_line(
            'python x.py --output-root "C:\\a path\\seed6101" --no-resume',
            "--output-root",
        )
        self.assertEqual(value, "C:\\a path\\seed6101")


if __name__ == "__main__":
    unittest.main()
