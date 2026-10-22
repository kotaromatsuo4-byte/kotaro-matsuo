import tempfile
import unittest
from pathlib import Path

import torch

import run_ori_codex6_torus_uniform_1m_replay as replay


class TorusUniformOneMillionReplayTests(unittest.TestCase):
    def test_atomic_torch_save_works_with_parent_failure_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            value = {"tensor": torch.arange(5), "episode": 512}
            replay.robust_atomic_torch_save(path, value)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["episode"], 512)
            self.assertTrue(torch.equal(loaded["tensor"], value["tensor"]))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_frozen_protocol_constants(self):
        self.assertEqual(replay.SEEDS, tuple(range(6101, 6111)))
        self.assertEqual(replay.PARENT_BUDGET, 636_198)
        self.assertEqual(replay.TARGET_BUDGET, 1_000_000)
        self.assertEqual(replay.FULL_PREFIX_EPISODES, 3_180)
        self.assertEqual(replay.FULL_PREFIX_MECHANICS, 636_000)
        self.assertEqual(replay.CHAIN_PREFIX_MECHANICS, 102_400)
        self.assertEqual(replay.CHECKPOINT_EVERY_EPISODES, 512)

    def test_expected_target_identity_is_single_seed_exact_budget(self):
        config = replay.expected_target_config(6101)
        self.assertEqual(config["task"], "torus2")
        self.assertEqual(config["seeds"], [6101])
        self.assertEqual(config["mechanics_budget"], 1_000_000)
        self.assertEqual(
            config["source_sha256"][replay.PARENT_SOURCE.name],
            replay.PARENT_SHA256,
        )

    def test_comparison_excludes_only_wall_timing(self):
        row = {
            "episode": "1",
            "action_trace_sha256": "abc",
            "rollout_seconds": "1.2",
            "cumulative_rollout_seconds": "1.2",
            "total_reward": "3.0",
        }
        self.assertEqual(
            replay.comparable_row(row),
            {
                "episode": "1",
                "action_trace_sha256": "abc",
                "total_reward": "3.0",
            },
        )

    def test_execute_requires_audited_source_hash(self):
        with self.assertRaises(SystemExit):
            replay.parse_args(["--execute"])


if __name__ == "__main__":
    unittest.main()
