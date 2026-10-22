import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_ori_codex8_spectral_staged as staged


def authority_row(episode=1, task="torus2", arm="legacy", *, strict=False,
                  steps=10, reward=0.0, flatness=.2, spread_error=.05,
                  timeout=False, psi=.0, phase_reward=.0, rollback=False):
    row = {
        "episode": str(episode), "steps": str(steps), "total_reward": str(reward),
        "flatness_ratio": str(flatness), "spread_error": str(spread_error),
        "strict_compact_success": str(int(strict)), "self_crossing": "0",
        "correction_failed": str(int(rollback)), "no_modes": "0", "timeout": str(int(timeout)),
        "termination_reason": "correction_failed" if rollback else ("timeout" if timeout else "running"),
        "ori6_rejected_crossing_count": "0", "ori6_rejected_crossing_rate": "0",
        "ori6_spectral_weight": str(staged.expected_parent_weight(task, arm)),
        "ori6_spectral_formula_sha256": staged.ori6_spectral.SPECTRAL_FORMULA_SHA256,
        "ori6_zero_reject_formula_sha256": staged.zero_reject.ZERO_REJECT_FORMULA_SHA256,
        "ori6_zero_reject_count": "0",
        "ori6_zero_reject_total_reward": "0",
        "ori6_zero_reject_timeout_reward": "0",
        "ori6_total_rejected_crossing_reward": "0",
        "ori5_reward_decomposition_error": "0",
        "ori5_maximum_abs_step_reward_decomposition_error": "0",
        "ori6_reward_decomposition_error": "0",
        "ori6_maximum_abs_step_reward_decomposition_error": "0",
        "ori6_spectral_reward_decomposition_error": "0",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error": "0",
        "ori6_maximum_abs_rejected_crossing_reward_decomposition_error": "0",
        "ori6_maximum_abs_zero_reject_reward_decomposition_error": "0",
        "ori5_nonfinite_reward_component_count": "0",
        "ori6_nonfinite_reward_component_count": "0",
        "ori6_spectral_nonfinite_component_count": "0",
    }
    if arm == "phase_v2":
        lam = staged.PHASE_LAMBDA[task]
        row.update({
            "ori8_candidate": staged.core.CANDIDATE_NAME,
            "ori8_candidate_schema_version": staged.core.CANDIDATE_SCHEMA_VERSION,
            "ori8_reward_formula_sha256": staged.core.REWARD_FORMULA_SHA256,
            "ori8_transition_formula_sha256": staged.core.TRANSITION_FORMULA_SHA256,
            "ori8_spectral_arm": arm, "ori8_lambda": str(lam),
            "ori8_total_reward": str(0.0 if rollback else phase_reward),
            "ori8_total_progress_reward": str(phase_reward),
            "ori8_total_rollback_reward": str(-phase_reward if rollback else 0.0),
            "ori8_cumulative_reward": str(0.0 if rollback else phase_reward),
            "ori8_initial_psi": "0",
            "ori8_terminal_psi": str(psi),
            "ori8_reward_decomposition_error": "0",
            "ori8_maximum_abs_step_reward_decomposition_error": "0",
            "ori8_nonfinite_component_count": "0",
            "ori8_close_latched": "1", "ori8_z_latched": "1",
            "ori8_close_event_count": "1", "ori8_turn_event_count": "1",
            "ori8_rollback_count": str(int(rollback)),
            "ori8_noncommit_count": "0",
        })
        assert abs(phase_reward - lam * psi) < 1e-12
    return row


def config(task="torus2", arm="legacy", seed=8101):
    return {
        **staged.expected_runtime_config(task, arm, seed),
        "ori8_candidate": staged.core.CANDIDATE_NAME,
        "ori8_candidate_schema_version": staged.core.CANDIDATE_SCHEMA_VERSION,
        "ori8_reward_formula_sha256": staged.core.REWARD_FORMULA_SHA256,
        "ori8_transition_formula_sha256": staged.core.TRANSITION_FORMULA_SHA256,
        "ori6_spectral_weight": staged.expected_parent_weight(task, arm),
        "observation_contract": "Codex6 coordinates-only observation unchanged",
        "pomdp_caveat": "phase ledger omitted from unchanged observation",
    }


def synthetic_report(arm, seed, *, hit=False, hit_steps=None, strict=0,
                     joint=0, constructive=0, e=.2, f=.8, reject=.1,
                     integrity=True, farming=False):
    recent = {
        "strict_count": strict, "joint_count": joint,
        "constructive_count": constructive, "spread_error_median": e,
        "flatness_median": f, "reject_rate_mean": reject,
        "invalid_count": 0, "episode_count": 512,
    }
    return {
        "task": "torus2", "arm": arm, "seed": seed, "endpoint": 4096,
        "integrity_pass": integrity, "integrity_errors": [],
        "first_hit": {"hit": hit, "episode": 10 if hit else None,
                      "action_proposals": hit_steps if hit else None},
        "last1024": recent, "last512": recent, "previous512": recent,
        "phase_reward_hacking": {"farming_detected": farming},
    }


class Codex8StagedAnalysisTests(unittest.TestCase):
    def test_overrun_is_capped_and_missing_prefix_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode"])
                writer.writeheader()
                writer.writerows([{"episode": 1}, {"episode": 2}, {"episode": 3}])
            rows, audit, errors = staged.read_history(path, 2)
            self.assertEqual([row["episode"] for row in rows], ["1", "2"])
            self.assertTrue(audit["overrun_capped"])
            self.assertEqual(audit["overrun_row_count"], 1)
            self.assertFalse(errors)
            rows, _, errors = staged.read_history(path, 4)
            self.assertTrue(any("contiguous" in error for error in errors))

    def test_control_and_phase_have_arm_specific_authority(self):
        control = authority_row(arm="scale")
        self.assertEqual(staged.validate_history_row(control, "torus2", "scale"), [])
        phase = authority_row(
            arm="phase_v2", psi=.4,
            phase_reward=staged.PHASE_LAMBDA["torus2"] * .4,
        )
        self.assertEqual(staged.validate_history_row(phase, "torus2", "phase_v2"), [])
        phase["ori8_cumulative_reward"] = ".3"
        self.assertIn(
            "phase_telescoping_total_vs_cumulative",
            staged.validate_history_row(phase, "torus2", "phase_v2"),
        )
        control["ori8_total_reward"] = ".1"
        self.assertIn(
            "stacked_ori8_control_reward",
            staged.validate_history_row(control, "torus2", "scale"),
        )

    def test_actual_horizon_crossing_keeps_ordinary_total_zero(self):
        # Exact authority values observed in preserved C8SP2
        # torus2/legacy/seed8102 episode 447.  Four ordinary rejects cost zero;
        # the fifth reject is at H=200 and carries only the separate timeout.
        row = authority_row(arm="legacy", timeout=True, steps=200)
        row.update({
            "episode": "447",
            "ori6_rejected_crossing_count": "5",
            "ori6_rejected_crossing_rate": ".025",
            "ori6_zero_reject_count": "4",
            "ori6_zero_reject_total_reward": "0.0",
            "ori6_zero_reject_timeout_reward": "-0.9201630438547277",
            "ori6_total_rejected_crossing_reward": "-0.9201630438547277",
            "ori6_terminal_quality_source": "last_safe_timeout_after_rejected_crossing",
            "ori6_reward_decomposition_error": "-4.653395724307785e-16",
        })
        self.assertEqual(staged.validate_history_row(row, "torus2", "legacy"), [])

    def test_horizon_crossing_zero_reject_negative_mutations_fail(self):
        row = authority_row(arm="legacy", timeout=True, steps=200)
        row.update({
            "ori6_rejected_crossing_count": "5",
            "ori6_rejected_crossing_rate": ".025",
            "ori6_zero_reject_count": "4",
            "ori6_zero_reject_timeout_reward": "-.73",
            "ori6_total_rejected_crossing_reward": "-.73",
            "ori6_terminal_quality_source": "last_safe_timeout_after_rejected_crossing",
        })
        mutations = (
            ("ori6_zero_reject_total_reward", "-.73", "zero_reject_ordinary_total_not_zero"),
            ("ori6_zero_reject_count", "5", "zero_reject_timeout_without_horizon_reject"),
            ("ori6_total_rejected_crossing_reward", "0", "zero_reject_horizon_timeout_decomposition"),
            ("termination_reason", "running", "zero_reject_horizon_termination"),
            ("steps", "199", "zero_reject_horizon_steps"),
            ("ori6_terminal_quality_source", "last_safe_timeout", "zero_reject_horizon_quality_source"),
            ("ori6_zero_reject_formula_sha256", "0" * 64, "ori6_zero_reject_formula_sha256"),
        )
        for field, value, expected in mutations:
            candidate = dict(row)
            candidate[field] = value
            self.assertIn(
                expected,
                staged.validate_history_row(candidate, "torus2", "legacy"),
                field,
            )

    def test_horizon_reject_with_zero_timeout_value_is_still_detected_by_authority(self):
        # q(last_safe)==1 gives -timeout_penalty+q == 0.  The boundary is
        # identified by count and quality-source identity, never magnitude.
        row = authority_row(arm="legacy", timeout=True, steps=200)
        row.update({
            "ori6_rejected_crossing_count": "1",
            "ori6_rejected_crossing_rate": ".005",
            "ori6_zero_reject_count": "0",
            "ori6_zero_reject_timeout_reward": "0",
            "ori6_total_rejected_crossing_reward": "0",
            "ori6_terminal_quality_source": "last_safe_timeout_after_rejected_crossing",
        })
        self.assertEqual(staged.validate_history_row(row, "torus2", "legacy"), [])
        row["ori6_zero_reject_count"] = "1"
        self.assertIn(
            "zero_reject_horizon_quality_source_without_horizon_reject",
            staged.validate_history_row(row, "torus2", "legacy"),
        )

    def test_phase_rollback_zeroes_prefix_without_terminal_psi_telescope(self):
        phase = authority_row(
            arm="phase_v2", psi=.4,
            phase_reward=staged.PHASE_LAMBDA["torus2"] * .4,
            rollback=True,
        )
        self.assertEqual(staged.validate_history_row(phase, "torus2", "phase_v2"), [])
        phase["ori8_total_rollback_reward"] = "-.3"
        errors = staged.validate_history_row(phase, "torus2", "phase_v2")
        self.assertIn("phase_progress_rollback_decomposition", errors)
        self.assertIn("phase_rollback_not_negative_prefix", errors)

    def test_config_binds_formula_arm_weight_observation_and_pomdp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps(config(arm="phase_v2")), encoding="utf-8")
            result = staged.validate_config(path, "torus2", "phase_v2", 8101)
            self.assertTrue(result["pass"], result["errors"])
            payload = config(arm="phase_v2")
            payload["ori8_reward_formula_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = staged.validate_config(path, "torus2", "phase_v2", 8101)
            self.assertIn("reward_formula", result["errors"])

    def test_every_runtime_profile_field_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            for field, bad in (
                ("max_steps", 999), ("actor_learning_rate", .2),
                ("actor_lr_milestones", [1, 2, 3]), ("data_dir", "wrong"),
                ("random_search", True), ("ori8_lambda", .9),
                ("ori8_scale_multiplier", 9), ("ori8_turn_delta_threshold", .6),
                ("ori8_pomdp_caveat", "wrong"),
                ("target_spread", 7), ("ori5_success_bonus", 9),
            ):
                payload = config(arm="scale")
                payload[field] = bad
                path.write_text(json.dumps(payload), encoding="utf-8")
                result = staged.validate_config(path, "torus2", "scale", 8101)
                self.assertIn(f"runtime_{field}", result["errors"], field)

    def test_all_eighteen_parse_time_configs_pass_before_train_returns(self):
        import ori_codex8_3d as spatial
        import run_ori_codex8_spectral_staged as runner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            root = Path(directory) / "campaign"
            for task in runner.TASKS.values():
                for arm in runner.ARMS:
                    for seed in runner.SEEDS:
                        command = runner.build_command(root, arm, task, seed)
                        args = spatial.parse_args(command[3:])
                        path.write_text(
                            json.dumps(vars(args), default=str), encoding="utf-8"
                        )
                        result = staged.validate_config(
                            path, task.name, arm, seed
                        )
                        self.assertTrue(
                            result["pass"],
                            f"{task.name}/{arm}/{seed}: {result['errors']}",
                        )
            payload = vars(spatial.parse_args(
                runner.build_command(root, "phase_v2", runner.TASKS["torus2"], 8101)[3:]
            ))
            payload.pop("ori8_pomdp_caveat")
            path.write_text(json.dumps(payload, default=str), encoding="utf-8")
            result = staged.validate_config(path, "torus2", "phase_v2", 8101)
            self.assertIn("runtime_ori8_pomdp_caveat", result["errors"])
            self.assertIn("pomdp_caveat", result["errors"])

    def test_selection_authority_fields_cannot_default_or_escape_bounds(self):
        base = authority_row()
        mutations = (
            ("steps", None, "missing_steps"),
            ("steps", "201", "selection_steps_bounds"),
            ("strict_compact_success", "2", "selection_binary_strict_compact_success"),
            ("flatness_ratio", "nan", "nonfinite_flatness_ratio"),
            ("spread_error", "-1", "selection_negative_spread_error"),
            ("ori6_rejected_crossing_count", "11", "selection_reject_count_bounds"),
            ("ori6_rejected_crossing_rate", ".2", "selection_reject_rate_consistency"),
        )
        for field, value, expected in mutations:
            row = dict(base)
            if value is None:
                row.pop(field)
            else:
                row[field] = value
            self.assertIn(expected, staged.validate_history_row(row, "torus2", "legacy"), field)

    def test_first_hit_uses_raw_action_proposals_including_hit_episode(self):
        records = [staged.classify(authority_row(1, steps=7)),
                   staged.classify(authority_row(2, steps=5, strict=True))]
        hit = staged.first_hit(records)
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["episode"], 2)
        self.assertEqual(hit["action_proposals"], 12)

    def test_lexicographic_selection_never_uses_return(self):
        fast = synthetic_report("phase_v2", 8101, hit=True, hit_steps=100,
                                strict=1, e=.5, f=2)
        slow = synthetic_report("legacy", 8101, hit=True, hit_steps=101,
                                strict=100, e=.001, f=.001)
        fast["last1024"]["return_mean_diagnostic_only"] = -999
        slow["last1024"]["return_mean_diagnostic_only"] = 999
        self.assertGreater(staged.selection_key(fast), staged.selection_key(slow))
        no_hit = synthetic_report("scale", 8101, hit=False, strict=100,
                                  joint=100, constructive=100)
        self.assertGreater(staged.selection_key(fast), staged.selection_key(no_hit))

    def test_formal_selection_requires_two_of_three_against_both_arms(self):
        reports = {}
        for arm in staged.ARMS:
            reports[arm] = []
            for seed in staged.SEEDS:
                # phase wins seeds 8101/8102 by earlier hit, loses 8103.
                rank = 100 if arm == "phase_v2" and seed != 8103 else (
                    200 if arm == "scale" else 300
                )
                if seed == 8103 and arm == "phase_v2":
                    rank = 400
                reports[arm].append(synthetic_report(
                    arm, seed, hit=True, hit_steps=rank, strict=1,
                    joint=1, constructive=1,
                ))

        def fake_seed_report(root, task, arm, seed, endpoint):
            return next(row for row in reports[arm] if row["seed"] == seed)

        with mock.patch.object(staged, "seed_report", side_effect=fake_seed_report):
            payload = staged.analyze(Path("unused"), 4096, tasks=("torus2",))
        decision = payload["task_decisions"]["torus2"]
        self.assertEqual(decision["selected_arm"], "phase_v2")
        self.assertIsNone(decision["selected_seed"])
        self.assertEqual(decision["selected_arm_min_pairwise_wins"], 2)
        self.assertTrue(decision["promote_selected_to_10000"])
        self.assertFalse(decision["return_used_for_selection"])

    def test_phase_hacking_metric_detects_high_psi_bad_timeout_farming(self):
        records = [staged.classify(authority_row(
            episode=index, arm="phase_v2", timeout=True, flatness=1.0,
            spread_error=.4, psi=.4,
            phase_reward=staged.PHASE_LAMBDA["torus2"] * .4,
        )) for index in range(1, 6)]
        summary = staged.window(records, 1, 5)
        self.assertEqual(summary["phase_high_psi_bad_timeout_rate"], 1.0)
        self.assertEqual(summary["phase_z_bad_timeout_count"], 5)

    def test_two_consecutive_production_warnings_are_required(self):
        baseline = [synthetic_report("phase_v2", seed, constructive=50)
                    for seed in staged.SEEDS]
        collapsed = [synthetic_report("phase_v2", seed, constructive=0)
                     for seed in staged.SEEDS]
        first = staged.production_decision(collapsed, baseline, 0)
        second = staged.production_decision(collapsed, baseline, 1)
        self.assertTrue(first["warning"])
        self.assertFalse(first["stop"])
        self.assertTrue(second["stop"])
        self.assertEqual(second["stop_kind"], "production_collapse")

    def test_checkpoint_bundle_is_explicitly_preliminary_utf8(self):
        payload = {
            "endpoint": 512, "tasks": ["torus2"], "arms": ["legacy"],
            "reports": {"torus2": {"legacy": [{
                **synthetic_report("legacy", 8101),
                "history_audit": {}, "config_validation": {"config": {}},
                "optimizer_records": [], "records": [],
            }]}},
            "task_decisions": {"torus2": {
                "selected_arm": None, "promote_selected_to_10000": False,
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = staged.write_artifacts(payload, Path(directory))
            report = Path(result["markdown"]).read_text(encoding="utf-8")
            self.assertIn("暫定", report)
            self.assertIn("return", report)


if __name__ == "__main__":
    unittest.main()
