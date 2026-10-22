from __future__ import annotations

import csv
import json
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest import mock

import finalize_ori_codex10_nonswap_staged as finalizer


def history_row(episode: int, strict=0, flatness=.4, error=.05, steps=10):
    return {
        "episode": str(episode), "steps": str(steps),
        "success": str(strict), "strict_compact_success": str(strict),
        "self_crossing": "0",
        "correction_failed": "0", "no_modes": "0",
        "flatness_ratio": str(flatness), "spread_error": str(error),
        "spread": str(1.0 + error), "target_spread": "1.0",
        "evaluation_flatness_tolerance": "0.1", "spread_tolerance": "0.1",
        "total_reward": str(float(episode)),
        "ori6_rejected_crossing_count": "0",
        "ori6_rejected_crossing_rate": "0.0",
        "ori6_spectral_terminal_potential": "0.2",
        "ori6_spectral_total_reward": "0.01",
    }


def minimal_payload(root: Path) -> dict:
    comparisons = []
    for task in finalizer.TASKS:
        for arm in finalizer.TASK_ARMS[task]:
            comparisons.append({
                "task": task, "arm": arm, "screen_endpoint": 4096,
                "selected_at_screen": False,
                "selection_reason": "no_promising_arm_at_4096",
                "common_proposal_cap": 1000, "hit_seed_count": 0,
                "pre_update_hit_seed_count": 0,
                "post_update_hit_seed_count": 0, "stable_seed_count": 0,
                "common_cap_proposal_rmst": 1000.0,
                "mean_tail_strict_rate": 0.0,
                "mean_fixed_stochastic_strict_rate": 0.0,
                "learning_evidence_beyond_initial_policy": False,
                "production_ready_at_4096": False,
            })
    return {
        "root": str(root), "created_utc": "2026-08-28T00:00:00+00:00",
        "manifest_fingerprint": "abc", "source_sha256": {},
        "model_sha256": {}, "finalizer_source_sha256": {},
        "externally_approved_finalizer_source_sha256": {},
        "input_sha256": {}, "runner_report": {"deadline_stop": False},
        "ever_selected_arms": {},
        "final_runner_selections": {task: None for task in finalizer.TASKS},
        "jobs": [], "screen_comparison": comparisons,
        "paired_seed_comparisons": [],
        "historical_c5_anchors": [], "selected_episode_curves": [],
        "selected_fixed_evaluations": [], "all_episode_curves": [],
        "all_fixed_evaluations": [],
    }


class Codex10FinalizerTests(unittest.TestCase):
    def test_manifest_digest_excludes_generated_timestamp(self):
        left = {"schema_version": 1, "generated_utc": "a", "value": 3}
        right = {"schema_version": 1, "generated_utc": "b", "value": 3}
        self.assertEqual(
            finalizer.canonical_digest(finalizer._manifest_content(left)),
            finalizer.canonical_digest(finalizer._manifest_content(right)),
        )

    def test_atomic_endpoint_records_post_checkpoint_history_overrun(self):
        rows = [history_row(index) for index in range(1, 601)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch.object(finalizer, "checkpoint_episode", return_value=512):
                endpoint, authority = finalizer.atomic_endpoint(directory, rows)
        self.assertEqual(endpoint, 512)
        self.assertEqual(authority["history_endpoint"], 600)
        self.assertEqual(authority["post_atomic_history_overrun_rows"], 88)

    def test_curve_first_hit_proposals_preupdate_and_C_J(self):
        rows = [
            history_row(1, flatness=.8, error=.2, steps=7),
            history_row(2, strict=1, flatness=.05, error=.03, steps=11),
            history_row(3, flatness=.3, error=.04, steps=13),
        ]
        curves, metrics = finalizer.build_episode_curves(
            rows, 3, {"task": "five_d2", "arm": "B", "seed": 9101},
            8, 0.1, 0.1, 1.0,
        )
        self.assertEqual(metrics["first_hit_episode"], 2)
        self.assertEqual(metrics["first_hit_proposals"], 18)
        self.assertTrue(metrics["first_hit_before_or_at_first_update"])
        self.assertEqual(curves[0]["constructive_C"], 0)
        self.assertEqual(curves[1]["constructive_C"], 1)
        self.assertEqual(curves[1]["joint_J"], 1)
        self.assertEqual(curves[2]["constructive_C"], 1)
        self.assertEqual(curves[2]["joint_J"], 0)
        self.assertAlmostEqual(curves[-1]["strict_rate_256"], 1 / 3)
        self.assertAlmostEqual(curves[-1]["strict_rate_512"], 1 / 3)

    def test_gate_recomputation_derives_selection_then_revocation(self):
        screen = {
            "schema_version": 1, "generated_utc": "old", "task": "five_d2",
            "endpoint": 512, "arm_summaries": {},
            "decision": {"selected_arm": "W", "stop_task": False,
                         "continue_screen": False, "reason": "selected"},
        }
        revoke = {
            "schema_version": 1, "generated_utc": "old", "task": "five_d2",
            "arm": "W", "endpoint": 1024, "arm_summary": {},
            "production_monitor": {"consecutive_warning_count": 0},
            "decision": {"revoke": True, "reason": "revoked"},
        }
        stopped = {}
        task_results = {("five_d2", 512): {**screen, "generated_utc": "new"}}
        gates = [screen, revoke]
        for task in ("five_d3", "watersqrt2"):
            for endpoint in finalizer.SCREEN_CHECKPOINTS:
                stop = endpoint == 4096
                gate = {
                    "schema_version": 1, "generated_utc": "old", "task": task,
                    "endpoint": endpoint, "arm_summaries": {},
                    "decision": {
                        "selected_arm": None, "stop_task": stop,
                        "continue_screen": not stop,
                        "reason": "stopped" if stop else "continue_screen",
                    },
                }
                gates.append(gate)
                task_results[(task, endpoint)] = {
                    **gate, "generated_utc": "new"
                }
                if stop:
                    stopped[task] = gate
        report = {
            "gate_history": gates,
            "selections": {"five_d2": None, "five_d3": None, "watersqrt2": None},
            "production_warning_streak": {
                "five_d2": 0, "five_d3": 0, "watersqrt2": 0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for gate in gates:
                path = finalizer.gate_artifact_path(root, gate)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(gate), encoding="utf-8")
            selected_result = {**revoke, "generated_utc": "new"}
            with mock.patch.object(
                finalizer.staged, "analyse_task",
                side_effect=lambda jobs, task, endpoint: task_results[(task, endpoint)],
            ), mock.patch.object(
                finalizer.staged, "analyse_selected", return_value=selected_result
            ):
                formal, selected, science, rows, latest = finalizer.recompute_gates(
                    {"jobs": []}, report, root, {}
                )
                duplicate_after_revoke = {
                    **report, "gate_history": [*gates, revoke],
                }
                with self.assertRaisesRegex(
                    RuntimeError, "without matching arm selection"
                ):
                    finalizer.recompute_gates(
                        {"jobs": []}, duplicate_after_revoke, root, {}
                    )
        self.assertEqual(selected, {"five_d2": "W"})
        self.assertEqual(formal[("five_d2", "B", 9101)], 512)
        self.assertEqual(formal[("five_d2", "W", 9101)], 1024)
        self.assertEqual(science["five_d3"]["endpoint"], 4096)
        self.assertEqual(latest["five_d2"]["endpoint"], 1024)
        self.assertEqual(len(rows), 10)

    def test_bundle_is_atomic_and_hashes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "bundle"
            payload = minimal_payload(base / "campaign")
            result = finalizer.write_bundle(
                payload, output, enforce_authority=False,
                publish_active_check=False, render=False,
            )
            self.assertTrue(output.is_dir())
            self.assertIn("output_hashes.json", result["files"])
            hashes = json.loads((output / "output_hashes.json").read_text("utf-8"))
            self.assertEqual(
                hashes["report_ja.md"], finalizer.sha256_file(output / "report_ja.md")
            )
            text = (output / "report_ja.md").read_text("utf-8")
            self.assertIn("結論", text)
            self.assertIn("同一 seed の proposal-clock 比較", text)
            self.assertNotIn("\ufffd", text)
            for mojibake_marker in ("縺", "繧", "蜿"):
                self.assertNotIn(mojibake_marker, text)

    def test_active_pid_aborts_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "bundle"
            payload = minimal_payload(base / "campaign")
            with mock.patch.object(
                finalizer, "active_c10_pids", return_value=[{"pid": 42}]
            ):
                with self.assertRaisesRegex(RuntimeError, "appeared before publish"):
                    finalizer.write_bundle(
                        payload, output, enforce_authority=False,
                        publish_active_check=True, render=False,
                    )
            self.assertFalse(output.exists())

    def test_selected_plot_contains_episode_and_proposal_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            payload = minimal_payload(output / "campaign")
            payload["ever_selected_arms"] = {"five_d2": "B"}
            payload["final_runner_selections"]["five_d2"] = "B"
            payload["jobs"] = []
            payload["selected_episode_curves"] = []
            payload["selected_fixed_evaluations"] = []
            for seed in finalizer.SEEDS:
                has_hit = seed != 9103
                rows, metrics = finalizer.build_episode_curves(
                    [history_row(
                        index, strict=int(has_hit and index >= 10),
                        flatness=.05 if has_hit and index >= 10 else .4, steps=5,
                    )
                     for index in range(1, 33)],
                    32, {"task": "five_d2", "arm": "B", "seed": seed,
                         "selected": True}, 8, 0.1, 0.1, 1.0,
                )
                payload["selected_episode_curves"].extend(rows)
                payload["selected_fixed_evaluations"].append({
                    "task": "five_d2", "arm": "B", "seed": seed,
                    "selected": True, "episode": 32,
                    "cumulative_proposals": 160, "evaluation_episodes": 16,
                    "strict_compact_success_rate": .5,
                })
                payload["jobs"].append({
                    "task": "five_d2", "arm": "B", "seed": seed,
                    "formal_endpoint": 32,
                    "first_hit_episode": metrics["first_hit_episode"],
                    "first_hit_proposals": metrics["first_hit_proposals"],
                    "first_hit_before_or_at_first_update": False,
                })
            episode = output / "episode.png"
            proposals = output / "proposals.png"
            finalizer.render_selected_plot(payload, "five_d2", episode, False)
            finalizer.render_selected_plot(payload, "five_d2", proposals, True)
            qa = finalizer.visual_qa([episode, proposals])
            self.assertEqual(len(qa), 2)
            self.assertNotEqual(qa[0]["sha256"], qa[1]["sha256"])
            payload["selected_episode_curves"] = [
                row for row in payload["selected_episode_curves"]
                if row["seed"] != 9103
            ]
            with self.assertRaisesRegex(RuntimeError, "missing seed"):
                finalizer.render_selected_plot(
                    payload, "five_d2", output / "missing.png", False
                )

    def test_external_approval_is_not_self_authorized(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            digests = {}
            for index, name in enumerate(finalizer.FINALIZER_NAMES):
                data = f"approved-source-{index}".encode()
                (directory / name).write_bytes(data)
                digests[name] = hashlib.sha256(data).hexdigest()
            inputs = {}
            with mock.patch.object(finalizer, "SCRIPT_DIRECTORY", directory):
                observed = finalizer.validate_finalizer_approval(
                    digests[finalizer.FINALIZER_NAMES[0]],
                    digests[finalizer.FINALIZER_NAMES[1]], inputs,
                )
                self.assertEqual(observed, digests)
                self.assertEqual(len(inputs), 2)
                with self.assertRaisesRegex(RuntimeError, "approved SHA-256 mismatch"):
                    finalizer.validate_finalizer_approval(
                        "0" * 64, digests[finalizer.FINALIZER_NAMES[1]], {}
                    )

    def test_control_plane_binds_exact_report_types_and_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, manifests = base / "campaign", base / "manifests"
            root.mkdir()
            manifests.mkdir()
            source = base / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            source_hashes = {"source.py": finalizer.sha256_file(source)}
            jobs = []
            runner_jobs = []
            for task in finalizer.TASKS:
                for arm in finalizer.TASK_ARMS[task]:
                    for seed in finalizer.SEEDS:
                        identity = {
                            "task": {"name": task}, "arm": arm, "seed": seed,
                        }
                        identity["job_fingerprint"] = finalizer.canonical_digest(identity)
                        jobs.append(identity)
                        runner_jobs.append({
                            "task": task, "arm": arm, "seed": seed,
                            "status": "not_promising", "selected": False,
                            "endpoint": 512, "returncode": 1,
                            "output_directory": str(root / task / arm / f"s{seed}"),
                        })
            manifest = {
                "schema_version": 1, "protocol": finalizer.EXPECTED_PROTOCOL,
                "experiment_root": str(root.resolve()),
                "seeds": list(finalizer.SEEDS),
                "task_order": list(finalizer.TASKS),
                "screen_checkpoints": list(finalizer.SCREEN_CHECKPOINTS),
                "production_checkpoints": list(finalizer.PRODUCTION_CHECKPOINTS),
                "max_parallel": 18, "training_deadline_local": "22:30",
                "report_deadline_local": "23:00",
                "all_18_jobs_fresh_from_episode_zero": True,
                "selected_arm_all_three_seeds_continue": True,
                "jobs": jobs, "source_sha256": source_hashes,
                "generated_utc": "2026-08-28T07:00:00+00:00",
            }
            fingerprint = finalizer.canonical_digest(
                finalizer._manifest_content(manifest)
            )
            manifest["manifest_fingerprint"] = fingerprint
            manifest_path = manifests / f"codex10_nonswap_{fingerprint[:12]}.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            binding = {
                "manifest_fingerprint": fingerprint,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": finalizer.sha256_file(manifest_path),
                "source_sha256": source_hashes,
                "started_utc": "2026-08-28T07:01:00+00:00",
                "thread_contract": {
                    "cpu": 20,
                    "env": {
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                    },
                    "torch": 1,
                },
            }
            report = {
                "schema_version": 1, "protocol": finalizer.EXPECTED_PROTOCOL,
                "manifest_fingerprint": fingerprint,
                "finished_utc": "2026-08-28T10:00:00+00:00",
                "deadline_stop": False, "failure": None,
                "selections": {task: None for task in finalizer.TASKS},
                "production_warning_streak": {
                    task: 0 for task in finalizer.TASKS
                },
                "gate_history": [], "jobs": runner_jobs,
                "report_deadline_met": True,
            }
            (root / "execution_binding.json").write_text(
                json.dumps(binding), encoding="utf-8"
            )
            report_path = root / "runner_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            patches = (
                mock.patch.object(finalizer, "SCRIPT_DIRECTORY", base),
                mock.patch.object(finalizer, "MANIFEST_DIRECTORY", manifests),
                mock.patch.object(finalizer, "DEFAULT_ROOT", root),
                mock.patch.object(finalizer, "SOURCE_NAMES", ("source.py",)),
                mock.patch.object(
                    finalizer, "EXPECTED_MANIFEST_FINGERPRINT", fingerprint
                ),
                mock.patch.object(
                    finalizer.bound_runner, "validate_manifest", return_value=manifest
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                observed = finalizer.validate_control_plane(root)
                self.assertEqual(len(observed[3]), 18)
                report["jobs"][0]["seed"] = str(report["jobs"][0]["seed"])
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not a JSON integer"):
                    finalizer.validate_control_plane(root)
                report["jobs"][0]["seed"] = finalizer.SEEDS[0]
                report["report_deadline_met"] = False
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "inconsistent with finish time"):
                    finalizer.validate_control_plane(root)

    def test_publish_rehash_precedes_pid_check_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign = base / "campaign"
            campaign.mkdir()
            (campaign / "runner_report.json").write_text("{}", encoding="utf-8")
            authority = base / "authority.bin"
            authority.write_bytes(b"frozen")
            payload = minimal_payload(campaign)
            payload["input_sha256"] = {
                str(authority.resolve()): finalizer.sha256_file(authority)
            }
            output = base / "bundle"
            events = []
            original_replace = finalizer.os.replace

            def checked_replace(source, destination):
                events.append("replace")
                return original_replace(source, destination)

            with mock.patch.object(
                finalizer, "verify_inputs_unchanged",
                side_effect=lambda inputs: events.append("rehash"),
            ), mock.patch.object(
                finalizer, "active_c10_pids",
                side_effect=lambda root: events.append("pid0") or [],
            ), mock.patch.object(
                finalizer.os, "replace", side_effect=checked_replace,
            ):
                finalizer.write_bundle(
                    payload, output, enforce_authority=False,
                    publish_active_check=True, render=False,
                )
            self.assertEqual(events, ["rehash", "pid0", "replace"])

    def test_job_inputs_are_hashed_before_gate_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, model_dir = base / "campaign", base / "model"
            output = root / "five_d2" / "B" / "s9101"
            output.mkdir(parents=True)
            model_dir.mkdir()
            model = model_dir / "node_initial.dat"
            model.write_bytes(b"model")
            for name in finalizer.REQUIRED_JOB_INPUT_NAMES:
                (output / name).write_bytes((name + "-frozen").encode())
            job = {
                "task": {"name": "five_d2", "model_directory": "model"},
                "arm": "B", "seed": 9101,
                "output_directory": str(output),
                "model_sha256": {
                    model.name: finalizer.sha256_file(model)
                },
            }
            inputs = {}
            with mock.patch.object(finalizer, "SCRIPT_DIRECTORY", base):
                authority = finalizer.prehash_job_and_model_inputs(
                    {"jobs": [job]}, root, inputs
                )
                self.assertEqual(authority["model"][model.name], finalizer.sha256_file(model))
                self.assertEqual(
                    len(inputs), 1 + len(finalizer.REQUIRED_JOB_INPUT_NAMES)
                )
                (output / "training_history.csv").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "changed between reads"):
                    finalizer.prehash_job_and_model_inputs(
                        {"jobs": [job]}, root, inputs
                    )

    def test_mutated_consumed_input_aborts_before_pid_or_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            authority = base / "authority.bin"
            authority.write_bytes(b"frozen")
            payload = minimal_payload(base / "campaign")
            payload["input_sha256"] = {
                str(authority.resolve()): finalizer.sha256_file(authority)
            }
            authority.write_bytes(b"mutated")
            active = mock.Mock(return_value=[])
            with mock.patch.object(finalizer, "active_c10_pids", active):
                with self.assertRaisesRegex(RuntimeError, "Consumed input changed"):
                    finalizer.write_bundle(
                        payload, base / "bundle", enforce_authority=False,
                        publish_active_check=True, render=False,
                    )
            active.assert_not_called()
            self.assertFalse((base / "bundle").exists())

    def test_curve_geometry_and_reject_authorities_fail_closed(self):
        mutations = {
            "strict_geometry": {"success": "1", "strict_compact_success": "1",
                                "flatness_ratio": "0.2"},
            "self_consistent_but_unbound_tolerance": {
                "success": "1", "strict_compact_success": "1",
                "flatness_ratio": "0.2", "evaluation_flatness_tolerance": "0.2",
            },
            "success_mismatch": {"success": "1"},
            "nonbinary_invalid": {"self_crossing": "2"},
            "reject_rate": {"ori6_rejected_crossing_count": "1"},
            "nonfinite_spectral": {"ori6_spectral_terminal_potential": "nan"},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                row = history_row(1)
                row.update(mutation)
                with self.assertRaises(RuntimeError):
                    finalizer.build_episode_curves(
                        [row], 1,
                        {"task": "five_d2", "arm": "B", "seed": 9101},
                        8, 0.1, 0.1, 1.0,
                    )

    def test_final_64_is_recomputed_from_raw_strict_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = {
                "last_policy_held_out_evaluation": {
                    "evaluation_episodes": 64,
                    "strict_compact_success_rate": 0.5,
                }
            }
            (directory / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            rows = []
            for index in range(64):
                strict = int(index < 32)
                rows.append({
                    "success": strict, "strict_compact_success": strict,
                    "self_crossing": 0, "correction_failed": 0, "no_modes": 0,
                    "flatness_ratio": 0.05 if strict else 0.2,
                    "spread_error": 0.05,
                    "spread": 1.05, "target_spread": 1.0,
                    "evaluation_flatness_tolerance": 0.1,
                    "spread_tolerance": 0.1,
                })

            def write_rows():
                with (directory / "evaluation_summary.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows()
            self.assertEqual(
                finalizer.validate_final_summary(
                    directory, 10000, 0.1, 0.1, 1.0, {}
                )[
                    "strict_compact_success_rate"
                ],
                0.5,
            )
            summary["last_policy_held_out_evaluation"][
                "strict_compact_success_rate"
            ] = 26 / 64
            (directory / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "raw/summary mismatch"):
                finalizer.validate_final_summary(
                    directory, 10000, 0.1, 0.1, 1.0, {}
                )
            summary["last_policy_held_out_evaluation"][
                "strict_compact_success_rate"
            ] = 0.5
            (directory / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            rows[0]["flatness_ratio"] = 0.2
            write_rows()
            with self.assertRaisesRegex(RuntimeError, "strict/geometry mismatch"):
                finalizer.validate_final_summary(
                    directory, 10000, 0.1, 0.1, 1.0, {}
                )

    def test_runner_revocation_status_and_selected_flag_are_causal(self):
        row = {
            "seed": 9101, "selected": True,
            "status": "selected_arm_revoked", "returncode": 1,
        }
        latest = {"endpoint": 2048, "decision": {"revoke": True}}
        selected, returncode = finalizer.validate_runner_job_state(
            row, "five_d2", "W", {"five_d2": "W"}, {"five_d2": None},
            False, 2048, latest,
        )
        self.assertTrue(selected)
        self.assertEqual(returncode, 1)
        wrong_phase = {**row, "status": "production_stability_revoked"}
        with self.assertRaisesRegex(RuntimeError, "Revocation status mismatch"):
            finalizer.validate_runner_job_state(
                wrong_phase, "five_d2", "W", {"five_d2": "W"},
                {"five_d2": None}, False, 2048, latest,
            )
        nonboolean = {**row, "selected": "true"}
        with self.assertRaisesRegex(RuntimeError, "not boolean"):
            finalizer.validate_runner_job_state(
                nonboolean, "five_d2", "W", {"five_d2": "W"},
                {"five_d2": None}, False, 2048, latest,
            )
        string_returncode = {**row, "returncode": "1"}
        with self.assertRaisesRegex(RuntimeError, "not a JSON integer"):
            finalizer.validate_runner_job_state(
                string_returncode, "five_d2", "W", {"five_d2": "W"},
                {"five_d2": None}, False, 2048, latest,
            )

    def test_causal_first_hit_censor_reasons(self):
        metrics = {"first_hit_episode": None}
        self.assertEqual(
            finalizer.first_hit_censor_reason(
                metrics, "five_d2", "B", 10000, "complete",
                {"five_d2": "B"}, {"five_d2": "B"}, None, False,
            ),
            "natural_completion_at_episode_10000_without_strict_hit",
        )
        self.assertEqual(
            finalizer.first_hit_censor_reason(
                metrics, "five_d2", "W", 512, "not_selected",
                {"five_d2": "B"}, {"five_d2": "B"}, None, False,
            ),
            "competing_arm_not_selected_at_episode_512",
        )
        self.assertIn(
            "selected_arm_revoked_at_episode_2048",
            finalizer.first_hit_censor_reason(
                metrics, "five_d2", "W", 2048, "selected_arm_revoked",
                {"five_d2": "W"}, {"five_d2": None},
                {"decision": {"reason": "selected_all_seed_no_persistence"}},
                False,
            ),
        )
        self.assertEqual(
            finalizer.first_hit_censor_reason(
                metrics, "five_d2", "B", 9216, "complete",
                {"five_d2": "B"}, {"five_d2": "B"}, None, True,
            ),
            "deadline_right_censor_at_episode_9216",
        )

    def test_paired_seed_export_is_exact_three_by_three(self):
        science = {}
        for task in finalizer.TASKS:
            candidate = next(arm for arm in finalizer.TASK_ARMS[task] if arm != "B")
            summaries = {}
            for arm in ("B", candidate):
                summaries[arm] = {"reports": [{
                    "seed": seed,
                    "first_hit_episode": 5 if arm == candidate else None,
                    "first_hit_before_or_at_first_update": arm == candidate,
                    "tail_strict_rate": 0.2 if arm == candidate else 0.1,
                    "fixed_stochastic_strict_rate": 0.25 if arm == candidate else 0.125,
                } for seed in finalizer.SEEDS]}
            science[task] = {
                "endpoint": 512, "common_proposal_cap": 1000,
                "arm_summaries": summaries,
                "paired_candidate_vs_B": {
                    "candidate": candidate,
                    "per_seed": [{
                        "seed": seed, "candidate_event_by_cap": True,
                        "candidate": 500, "control_event_by_cap": False,
                        "control": 1000, "outcome": "win",
                    } for seed in finalizer.SEEDS],
                },
            }
        rows = finalizer._paired_seed_rows(science)
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["seed"] for row in rows}, set(finalizer.SEEDS))
        self.assertTrue(all(row["proposal_delta_candidate_minus_control"] == -500
                            for row in rows))

    def test_historical_anchor_hashes_and_metrics_are_live(self):
        rows = finalizer.historical_c5_anchors({})
        self.assertEqual(len(rows), 3)
        water = next(row for row in rows if row["task"] == "watersqrt2")
        self.assertEqual(water["first_hit_episode"], 47)
        self.assertEqual(water["first_hit_proposals"], 17324)


if __name__ == "__main__":
    unittest.main()
