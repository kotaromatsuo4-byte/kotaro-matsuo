from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import finalize_ori_codex6_water_training_only_supplement as supplement


class WaterTrainingOnlySupplementTests(unittest.TestCase):
    def test_report_is_explicitly_non_authority(self):
        payload = {
            "training_only_10000": {
                "last_1024": {
                    "mean_return": 8.25, "median_flatness": 0.001,
                    "median_spread_error": 0.00004, "strict_count": 516,
                    "strict_rate": 516 / 1024,
                },
                "last_512": {"strict_count": 265, "strict_rate": 265 / 512},
                "evaluations": [
                    {
                        "sample_count": 16, "success_rate": 0.625,
                        "self_crossing_rate": 0.0, "mean_return": 9.7,
                    },
                    {
                        "success_rate": 0.0, "termination_reason": "timeout",
                        "mean_flatness": 1.67, "mean_spread_error": 0.116,
                    },
                ],
            },
            "integrity": {"pass": True},
            "input_hash_set_sha256": "a" * 64,
            "formal_authority": {"main_artifact_manifest_sha256": "b" * 64},
        }
        text = supplement.report_text(payload)
        self.assertIn("非authority", text)
        self.assertIn("正式authorityは", text)
        self.assertIn("final held-outではない", text)

    def test_verify_main_artifacts_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            manifest = root / "manifest.json"
            supplement.combined.atomic_json(manifest, {
                "artifact_sha256": {
                    str(artifact.resolve()): supplement.combined.file_sha256(artifact)
                }
            })
            with mock.patch.object(supplement, "MAIN_MANIFEST", manifest):
                supplement.verify_main_artifacts()
                artifact.write_text("changed", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    supplement.verify_main_artifacts()

    def test_csv_rows_keep_formal_and_training_scopes_separate(self):
        window = {"start_episode": 1, "end_episode": 2, "episode_count": 2}
        payload = {
            "formal_9728_last_1024": window,
            "training_only_10000": {
                "last_1024": window, "last_512": window,
                "evaluations": [{"evaluation_mode": "periodic_stochastic"}],
            },
            "integrity": {"checks": [{"check": "fixture", "pass": True}]},
        }
        rows = supplement.csv_rows(payload)
        self.assertEqual(rows[0]["label"], "formal_authority_9728_last1024")
        self.assertEqual(rows[1]["label"], "training_only_10000_last1024")
        self.assertEqual(rows[3]["row_type"], "evaluation")


if __name__ == "__main__":
    unittest.main()
