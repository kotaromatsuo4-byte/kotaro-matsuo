import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import render_ori_codex10_proposal_plots_fixed as renderer


class ProposalPlotVisualFixTests(unittest.TestCase):
    def synthetic_payload(self):
        task, arm = "five_d2", "B"
        curves, fixed, jobs = [], [], []
        for seed in renderer.SEEDS:
            curves.append({
                "task": task, "arm": arm, "seed": seed, "selected": True,
                "episode": 1, "cumulative_proposals": 10, "steps": 10,
                "strict": 0, "constructive_C": 0, "joint_J": 0,
                "terminal_flatness_F": .5, "spread_error_e": .4,
                "return_diagnostic": -.5, "rejected_crossing_rate": 0.0,
                "spectral_terminal_potential": .1, "spectral_reward": .01,
                "strict_rate_256": 0.0, "constructive_C_rate_256": 0.0,
                "joint_J_rate_256": 0.0,
                "terminal_flatness_F_median_256": .5,
                "spread_error_e_median_256": .4,
                "return_diagnostic_mean_256": -.5,
                "rejected_crossing_rate_mean_256": 0.0,
                "spectral_potential_mean_256": .1,
                "spectral_reward_mean_256": .01,
                "strict_rate_512": 0.0, "constructive_C_rate_512": 0.0,
                "joint_J_rate_512": 0.0,
                "terminal_flatness_F_median_512": .5,
                "spread_error_e_median_512": .4,
                "return_diagnostic_mean_512": -.5,
                "rejected_crossing_rate_mean_512": 0.0,
                "spectral_potential_mean_512": .1,
                "spectral_reward_mean_512": .01,
            })
            fixed.append({
                "task": task, "arm": arm, "seed": seed, "selected": True,
                "episode": 1, "cumulative_proposals": 10,
                "evaluation_episodes": 16,
                "strict_compact_success_rate": 0.0,
            })
            jobs.append({
                "task": task, "arm": arm, "seed": seed,
                "formal_endpoint": 1, "first_hit_episode": None,
                "first_hit_proposals": None,
                "first_hit_before_or_at_first_update": None,
            })
        return {
            "ever_selected_arms": {task: arm},
            "final_runner_selections": {task: arm},
            "selected_episode_curves": curves,
            "selected_fixed_evaluations": fixed,
            "jobs": jobs,
        }

    def test_padded_render_has_measured_top_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.png"
            renderer.render_padded_proposal_plot(
                self.synthetic_payload(), "five_d2", path
            )
            qa = renderer.visual_qa(path)
            self.assertTrue(qa["pass"])
            self.assertGreaterEqual(qa["top_content_margin_px"], 20)

    def test_formal_inventory_and_frozen_authority_match_disk(self):
        inputs = {}
        inventory, blobs = renderer.validate_formal_inventory(
            renderer.FORMAL_ROOT,
            renderer.EXPECTED_FORMAL_OUTPUT_HASHES_SHA256,
            renderer.EXPECTED_FORMAL_FILES,
            inputs,
        )
        self.assertEqual(
            set(inventory), renderer.EXPECTED_FORMAL_FILES - {"output_hashes.json"}
        )
        self.assertIn("selected_learning_curves.csv", blobs)
        payload = renderer.load_plot_payload(blobs)
        self.assertEqual(len(payload["selected_episode_curves"]), 90_000)
        self.assertEqual(len(payload["selected_fixed_evaluations"]), 189)
        self.assertFalse(renderer.OUTPUT.exists())
        for name, expected in {
            "finalize_ori_codex10_nonswap_staged.py":
                renderer.EXPECTED_C10_FINALIZER_SHA256,
            "test_finalize_ori_codex10_nonswap_staged.py":
                renderer.EXPECTED_C10_FINALIZER_TEST_SHA256,
        }.items():
            self.assertEqual(renderer.sha256_file(renderer.SCRIPT_DIR / name), expected)

    def test_inventory_mutation_and_extra_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.txt").write_text("bound", encoding="utf-8")
            digest = hashlib.sha256((root / "data.txt").read_bytes()).hexdigest()
            inventory = json.dumps(
                {"data.txt": digest}, indent=2, sort_keys=True
            ).encode() + b"\n"
            (root / "output_hashes.json").write_bytes(inventory)
            inventory_digest = hashlib.sha256(inventory).hexdigest()
            renderer.validate_formal_inventory(
                root, inventory_digest, {"data.txt", "output_hashes.json"}, {}
            )
            (root / "data.txt").write_text("mutated", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                renderer.validate_formal_inventory(
                    root, inventory_digest,
                    {"data.txt", "output_hashes.json"}, {},
                )
            (root / "data.txt").write_text("bound", encoding="utf-8")
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                renderer.validate_formal_inventory(
                    root, inventory_digest,
                    {"data.txt", "output_hashes.json"}, {},
                )

    def test_input_rehash_precedes_final_pid_check(self):
        order = []
        with mock.patch.object(
            renderer, "verify_inputs", side_effect=lambda inputs: order.append("rehash")
        ):
            renderer.verify_before_publish(
                {}, lambda: order.append("pid") or []
            )
        self.assertEqual(order, ["rehash", "pid"])
        with mock.patch.object(renderer, "verify_inputs", return_value=None):
            with self.assertRaises(RuntimeError):
                renderer.verify_before_publish({}, lambda: [{"pid": 1}])

    def test_exact_sha_and_external_approval_fail_closed(self):
        for value in ("0" * 63, "g" * 64, None):
            with self.assertRaises(RuntimeError):
                renderer.require_sha256(value, "x")
        with self.assertRaises(RuntimeError):
            renderer.validate_self_approval("0" * 64, "0" * 64, {})


if __name__ == "__main__":
    unittest.main()
