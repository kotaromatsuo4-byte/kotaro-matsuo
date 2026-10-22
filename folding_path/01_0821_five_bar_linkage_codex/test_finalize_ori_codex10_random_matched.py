import copy
from pathlib import Path
import tempfile
import unittest

import finalize_ori_codex10_random_matched as finalizer


class RandomMatchedFinalizerTests(unittest.TestCase):
    def synthetic_c10_payload(self):
        arms = {"five_d2": "B", "five_d3": "W", "watersqrt2": "B"}
        curves, jobs = [], []
        for task, arm in arms.items():
            cap = finalizer.CAPS[task]
            for seed, first, preupdate in (
                (9101, 100, True), (9102, None, None), (9103, 300, False)
            ):
                curves.append({
                    "task": task, "arm": arm, "seed": seed, "episode": 1,
                    "strict": int(first is not None),
                    "cumulative_proposals": first if first is not None else cap + 10,
                })
                jobs.append({
                    "task": task, "arm": arm, "seed": seed,
                    "first_hit_proposals": first,
                    "first_hit_before_or_at_first_update": preupdate,
                    "formal_endpoint": 1,
                    "job_fingerprint": f"{task}-{arm}-{seed}",
                })
        return {
            "selected_episode_curves": curves, "jobs": jobs,
            "ever_selected_arms": arms,
            "final_runner_selections": dict(arms),
        }

    def random_observations(self):
        rows = []
        for task in finalizer.TASKS:
            cap = finalizer.CAPS[task]
            for seed in finalizer.RANDOM_SEEDS:
                rows.append({
                    "task": task, "group": "uniform_random", "arm": "uniform",
                    "seed": seed, "common_proposal_cap": cap,
                    "event_by_cap": False, "first_hit_proposals": None,
                    "assigned_proposals_at_cap": cap,
                    "observed_proposal_exposure": cap,
                    "usable_at_common_cap": True,
                    "censor_reason": "mechanics_budget",
                    "first_hit_before_or_at_first_update": None,
                    "formal_job_fingerprint": f"random-{task}-{seed}",
                })
        return rows

    def test_learning_common_cap_recensor_and_preupdate(self):
        rows = finalizer.learning_observations(self.synthetic_c10_payload())
        self.assertEqual(len(rows), 9)
        for task in finalizer.TASKS:
            task_rows = [row for row in rows if row["task"] == task]
            self.assertEqual([row["event_by_cap"] for row in task_rows], [True, False, True])
            self.assertEqual(task_rows[1]["assigned_proposals_at_cap"], finalizer.CAPS[task])
            self.assertEqual(
                [row["first_hit_before_or_at_first_update"] for row in task_rows],
                [True, None, False],
            )

    def test_primary_rmst_speedup_and_revocation(self):
        c10 = self.synthetic_c10_payload()
        observations = finalizer.learning_observations(c10) + self.random_observations()
        rows = finalizer.comparison_rows(observations)
        self.assertEqual(len(rows), 3)
        for row in rows:
            cap = row["common_proposal_cap"]
            expected_learning = (100 + cap + 300) / 3
            self.assertTrue(row["primary_comparison_available"])
            self.assertEqual(row["learning_events"], 2)
            self.assertEqual(row["random_events"], 0)
            self.assertAlmostEqual(row["learning_rmst"], expected_learning)
            self.assertAlmostEqual(row["random_rmst"], cap)
            self.assertAlmostEqual(
                row["random_over_learning_rmst"], cap / expected_learning
            )
        revoked = self.synthetic_c10_payload()
        revoked["final_runner_selections"]["five_d2"] = None
        observations = (
            finalizer.learning_observations(revoked) + self.random_observations()
        )
        row = finalizer.comparison_rows(observations)[0]
        self.assertFalse(row["primary_comparison_available"])
        self.assertNotIn("random_over_learning_rmst", row)

    def test_insufficient_exposure_is_not_assigned_cap(self):
        payload = self.synthetic_c10_payload()
        target = next(
            row for row in payload["selected_episode_curves"]
            if row["task"] == "five_d2" and row["seed"] == 9102
        )
        target["cumulative_proposals"] = finalizer.CAPS["five_d2"] - 1
        rows = finalizer.learning_observations(payload)
        row = next(
            item for item in rows
            if item["task"] == "five_d2" and item["seed"] == 9102
        )
        self.assertFalse(row["usable_at_common_cap"])
        self.assertIsNone(row["assigned_proposals_at_cap"])
        summary = finalizer.comparison_rows(rows + self.random_observations())[0]
        self.assertFalse(summary["primary_comparison_available"])
        self.assertEqual(summary["learning_n_usable"], 2)

    def test_bootstrap_is_deterministic_and_bounded(self):
        one = finalizer.bootstrap_group([10, 20, 30], [1, 0, 1], 123)
        two = finalizer.bootstrap_group([10, 20, 30], [1, 0, 1], 123)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["hit_rate"], 2 / 3)
        self.assertAlmostEqual(one["rmst"], 20)
        self.assertLessEqual(one["hit_rate_ci_low"], one["hit_rate"])
        self.assertGreaterEqual(one["hit_rate_ci_high"], one["hit_rate"])

    def test_plot_and_japanese_report_utf8(self):
        payload = self.synthetic_c10_payload()
        observations = (
            finalizer.learning_observations(payload) + self.random_observations()
        )
        comparisons = finalizer.comparison_rows(observations)
        report_payload = {"observations": observations, "comparisons": comparisons}
        text = finalizer.japanese_report(report_payload)
        self.assertIn("初回発見", text)
        self.assertNotIn("\ufffd", text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            finalizer.render_plot(observations, path)
            qa = finalizer.visual_qa(path)
            self.assertTrue(qa["pass"])
            self.assertGreaterEqual(qa["top_content_margin_px"], 20)

    def test_exact_authority_constants_match_disk(self):
        expected = {
            "run_ori_codex10_nonswap_random_matched.py": (
                finalizer.EXPECTED_RANDOM_RUNNER_SHA256
            ),
            "test_run_ori_codex10_nonswap_random_matched.py": (
                finalizer.EXPECTED_RANDOM_TEST_SHA256
            ),
            "finalize_ori_codex10_nonswap_staged.py": (
                finalizer.EXPECTED_C10_FINALIZER_SHA256
            ),
            "test_finalize_ori_codex10_nonswap_staged.py": (
                finalizer.EXPECTED_C10_FINALIZER_TEST_SHA256
            ),
        }
        for name, digest in expected.items():
            self.assertEqual(
                finalizer.sha256_bytes((finalizer.SCRIPT_DIR / name).read_bytes()),
                digest,
            )
        manifest = finalizer.RANDOM_MANIFEST.read_bytes()
        self.assertEqual(
            finalizer.sha256_bytes(manifest),
            finalizer.EXPECTED_RANDOM_MANIFEST_SHA256,
        )

    def test_exact_integer_and_sha_reject_lossy_values(self):
        for value in (True, 1.0, "1"):
            with self.assertRaises(RuntimeError):
                finalizer.exact_int(value, "x")
        for value in ("0" * 63, "g" * 64, None):
            with self.assertRaises(RuntimeError):
                finalizer.require_sha(value, "x")


if __name__ == "__main__":
    unittest.main()
