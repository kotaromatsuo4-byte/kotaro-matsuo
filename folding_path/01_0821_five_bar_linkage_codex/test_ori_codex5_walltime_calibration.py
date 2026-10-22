"""Tiny tests for the standalone ori5 uniform wall-time calibrator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments_ori5.analysis.tools import calibrate_ori5_uniform_walltime as calibration


class Ori5UniformWalltimeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calibration.torch.set_num_threads(1)

    def _tiny_protocol(self, directory, task_ids=("five_d2",)):
        protocol = copy.deepcopy(calibration._load_json(
            calibration.DEFAULT_PROTOCOL
        ))
        protocol["protocol_id"] = "unit_test_fixed_prefix"
        protocol["bootstrap_replicates"] = 64
        protocol["bootstrap_seed"] = 77001
        protocol["tasks"] = [
            task for task in protocol["tasks"] if task["task"] in task_ids
        ]
        for index, task in enumerate(protocol["tasks"]):
            task["seeds"] = [17 + 2 * index, 18 + 2 * index]
            task["prefix_mechanics_per_seed"] = 1
            task["warmup_seed"] = 90000 + index
            task["warmup_mechanics"] = 1
        path = Path(directory) / "tiny_protocol.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(protocol, file, indent=2)
        return path

    def test_bootstrap_is_deterministic_and_uses_pooled_time(self):
        blocks = [
            {"mechanical_steps": 10, "rollout_wall_seconds": 1.0},
            {"mechanical_steps": 10, "rollout_wall_seconds": 2.0},
        ]
        first = calibration.bootstrap_task(
            blocks, [5.0, 15.0], replicates=128,
            confidence_level=0.95, seed=71,
        )
        second = calibration.bootstrap_task(
            blocks, [5.0, 15.0], replicates=128,
            confidence_level=0.95, seed=71,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first["mechanical_steps_per_wall_second"]["estimate"],
            20.0 / 3.0,
        )
        self.assertAlmostEqual(
            first["baseline_mechanics_rmst"]["estimate"], 10.0
        )
        self.assertAlmostEqual(
            first["joint_calibrated_wall_rmst_seconds"]["estimate"], 1.5
        )

    def test_protocol_rejects_reward_override_and_duplicate_seeds(self):
        protocol = copy.deepcopy(calibration._load_json(
            calibration.DEFAULT_PROTOCOL
        ))
        protocol["tasks"] = [protocol["tasks"][0]]
        protocol["tasks"][0]["core_args"].extend([
            "--ori5-success-bonus", "11"
        ])
        with self.assertRaisesRegex(ValueError, "fixed"):
            calibration.validate_protocol(protocol)

        protocol = copy.deepcopy(calibration._load_json(
            calibration.DEFAULT_PROTOCOL
        ))
        protocol["tasks"] = [protocol["tasks"][0]]
        protocol["tasks"][0]["seeds"] = [17, 17]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calibration.validate_protocol(protocol)

    def test_exact_prefix_trace_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = self._tiny_protocol(directory)
            plan = calibration.build_plan(protocol_path)
            prepared = plan["prepared"][0]
            with prepared["adapter"].patched_three_dimensional_runtime(
                prepared["args"], prepared["member"]
            ):
                first = calibration.run_fixed_prefix(prepared, 17, 2)
                second = calibration.run_fixed_prefix(prepared, 17, 2)
            self.assertEqual(first["mechanical_steps"], 2)
            self.assertEqual(second["mechanical_steps"], 2)
            self.assertEqual(
                first["prefix_trace_sha256"], second["prefix_trace_sha256"]
            )
            self.assertEqual(len(first["prefix_trace_sha256"]), 64)
            self.assertEqual(first["nonfinite_reward_component_count"], 0)

    def test_tiny_planar_and_spatial_end_to_end(self):
        task_ids = ("five_d2", "torus2", "water1")
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = self._tiny_protocol(directory, task_ids)
            output = Path(directory) / "output"
            output_path, result = calibration.run_calibration(
                protocol_path=protocol_path,
                output_directory=output,
                progress=False,
            )
            self.assertEqual(output_path, output.resolve())
            self.assertTrue(result["complete"])
            self.assertFalse(
                result["measurement_definition"]["direct_rl_wall_measurement"]
            )
            self.assertEqual(
                result["system_identity"]["process_count_used_by_calibrator"],
                1,
            )
            self.assertEqual(result["totals"]["measured_mechanics"], 6)
            self.assertEqual(result["totals"]["excluded_warmup_mechanics"], 3)
            self.assertEqual(len(result["tasks"]), 3)
            self.assertEqual(
                {task["problem"] for task in result["tasks"]},
                {"five_bar", "4unit_torus", "waterbomb"},
            )
            for task in result["tasks"]:
                self.assertEqual(len(task["blocks"]), 2)
                self.assertTrue(task["model_file_sha256"])
                self.assertGreater(
                    task["statistics"]["baseline_mechanics_rmst"][
                        "estimate"
                    ], 0.0,
                )
                for block in task["blocks"]:
                    self.assertEqual(block["mechanical_steps"], 1)
                    self.assertEqual(
                        block["nonfinite_reward_component_count"], 0
                    )
            relative_script = str(
                Path(calibration.__file__).resolve().relative_to(
                    calibration._REPOSITORY_ROOT
                )
            )
            self.assertIn(relative_script, result["source_sha256"])
            self.assertIn("ori_codex5.py", result["source_sha256"])
            self.assertEqual(
                result["reward_constants_sha256"],
                calibration.discovery.FIXED_REWARD_SHA256,
            )
            for name in (
                "run_config.json", "calibration_results.json",
                "timing_blocks.csv", "task_summary.csv", "REPORT.md",
            ):
                self.assertTrue((output / name).is_file(), name)
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("not** contain directly measured Actor--Critic", report)


if __name__ == "__main__":
    unittest.main()
