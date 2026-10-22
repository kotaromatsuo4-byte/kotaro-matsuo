import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import run_ori_codex6_zero_reject_water_fallback as fallback


class WaterFallbackTests(unittest.TestCase):
    def valid_gate(self):
        return {
            "checkpoint_episode": 4096,
            "candidate": fallback.staged.CANDIDATE,
            "overall_integrity_pass": True,
            "tasks": {
                "water1": {
                    "decision": {
                        "checkpoint_episode": 4096,
                        "rule": "formal_2_of_3_persistent_target_plane_gate",
                        "stop": True,
                        "promote": False,
                        "selected_seed": None,
                        "promising_seed_count": 1,
                        "promising_seeds": [4201],
                        "selection_order": [4201],
                    },
                    "seeds": [
                        {
                            "seed": seed,
                            "endpoint": 4096,
                            "integrity_pass": True,
                            "promising_at_4096": seed == 4201,
                            "promising_components": {
                                "evidence": seed == 4201,
                                "persistence": seed == 4201,
                                "quality": seed == 4201,
                                "trend": seed == 4201,
                            },
                        }
                        for seed in (4201, 4202, 4203)
                    ],
                }
            },
        }

    def test_formal_reject_is_accepted(self):
        result = fallback.validate_gate_payload(self.valid_gate())
        self.assertTrue(result["pass"], result["errors"])

    def test_pass_or_integrity_failure_is_rejected(self):
        passing = self.valid_gate()
        passing["tasks"]["water1"]["decision"].update({
            "stop": False, "promote": True, "selected_seed": 4201,
            "promising_seed_count": 2,
        })
        self.assertFalse(fallback.validate_gate_payload(passing)["pass"])
        broken = self.valid_gate()
        broken["overall_integrity_pass"] = False
        self.assertFalse(fallback.validate_gate_payload(broken)["pass"])

    def test_promising_count_and_flags_must_be_internally_consistent(self):
        payload = self.valid_gate()
        payload["tasks"]["water1"]["decision"]["promising_seed_count"] = 0
        self.assertFalse(fallback.validate_gate_payload(payload)["pass"])
        payload = self.valid_gate()
        payload["tasks"]["water1"]["seeds"][0]["promising_at_4096"] = False
        self.assertFalse(fallback.validate_gate_payload(payload)["pass"])

    def test_semantic_checkpoint_comparison_drops_only_generation_time(self):
        persisted = self.valid_gate()
        persisted["generated_utc"] = "old"
        recomputed = deepcopy(persisted)
        recomputed["generated_utc"] = "new"
        self.assertTrue(
            fallback.compare_recomputed_checkpoint(persisted, recomputed)["pass"]
        )
        recomputed["experiment_root"] = "wrong-root"
        self.assertFalse(
            fallback.compare_recomputed_checkpoint(persisted, recomputed)["pass"]
        )

    def test_counterfactual_artifacts_and_all_source_prefixes_are_bound(self):
        authority = fallback.counterfactual_authority()
        self.assertEqual(authority["recommended_cutoff"], 0.12)
        self.assertEqual(len(authority["artifact_and_generator_sha256"]), 5)
        self.assertEqual(len(authority["source_prefixes"]), 7)
        for record in authority["source_prefixes"].values():
            self.assertEqual(record["row_count"], record["endpoint"])
            self.assertEqual(record["algorithm"], "ori6_csv_prefix_sha256_v1")
            self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(
            authority["saved_success_path"]["sha256"],
            fallback.EXPECTED_SAVED_SUCCESS_SHA256,
        )

    def test_all_r1_manifest_sources_not_only_analyser_and_core_are_frozen(self):
        sources = fallback.verify_frozen_parent_sources()
        self.assertEqual(set(sources), set(fallback.staged.SOURCE_FILES))
        self.assertEqual(len(sources), 13)

    def test_fallback_config_accepts_only_cutoff_0p12(self):
        source = (
            fallback.ACTIVE_ROOT
            / "water1/B/seed4201/run_config.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["seed"] = 4301
        payload["near_miss_spread_error_far"] = 0.12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            accepted = fallback.validate_fallback_config(path, 4301)
            self.assertTrue(accepted["pass"], accepted["errors"])
            payload["near_miss_spread_error_far"] = 0.15
            path.write_text(json.dumps(payload), encoding="utf-8")
            rejected = fallback.validate_fallback_config(path, 4301)
            self.assertFalse(rejected["pass"])
            self.assertIn("near_miss_spread_error_far", rejected["errors"])

    def test_draft_is_three_fresh_water_jobs_and_no_root_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "absent_fallback_root"
            payload = fallback.draft(root)
            self.assertFalse(root.exists())
            self.assertEqual(payload["seeds"], [4301, 4302, 4303])
            self.assertEqual(len(payload["commands"]), 3)
            for seed, command in zip(payload["seeds"], payload["commands"]):
                self.assertIn("--ori6-zero-reject-reward", command)
                cutoff_index = command.index("--near-miss-spread-error-far")
                self.assertEqual(command[cutoff_index + 1], "0.12")
                seed_index = command.index("--seed")
                self.assertEqual(command[seed_index + 1], str(seed))
                output_index = command.index("--output-dir")
                self.assertIn("water1", command[output_index + 1])

    def test_real_512_prefix_survives_fallback_analysis_and_artifact_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fallback_fixture"
            for source_seed, target_seed in zip((4201, 4202, 4203), fallback.SEEDS):
                source = fallback.ACTIVE_ROOT / f"water1/B/seed{source_seed}"
                target = root / f"water1/B/seed{target_seed}"
                target.mkdir(parents=True)
                shutil.copyfile(source / "training_history.csv", target / "training_history.csv")
                shutil.copyfile(source / "update_history.csv", target / "update_history.csv")
                config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
                config["seed"] = target_seed
                config["near_miss_spread_error_far"] = 0.12
                (target / "run_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
            payload = fallback.analyse_checkpoint(root, 512, fallback.SEEDS)
            self.assertTrue(payload["overall_integrity_pass"])
            self.assertEqual(
                [item["seed"] for item in payload["tasks"]["water1"]["seeds"]],
                [4301, 4302, 4303],
            )
            output = root / "artifact_fixture"
            artifacts = fallback.analysis.write_artifacts(payload, output)
            for key in ("json", "csv", "report"):
                self.assertGreater(Path(artifacts[key]).stat().st_size, 0)
            self.assertEqual(len(artifacts["plots"]), 1)
            for path in artifacts["plots"]:
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
