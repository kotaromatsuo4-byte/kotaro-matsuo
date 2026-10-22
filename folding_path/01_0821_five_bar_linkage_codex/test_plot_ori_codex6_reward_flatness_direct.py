from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import plot_ori_codex6_reward_flatness_direct as direct


class DirectLearningPlotTests(unittest.TestCase):
    def test_cumulative_mean_and_rolling_statistics(self):
        np.testing.assert_allclose(direct.cumulative_mean([1, 3, 5]), [1, 2, 3])
        np.testing.assert_allclose(direct.rolling_mean([1, 3, 5], 2), [2, 4])
        np.testing.assert_allclose(direct.rolling_median([1, 9, 3], 2), [5, 6])

    def test_synthetic_render(self):
        episode = list(range(1, 401))

        def series(task, dataset, seed, endpoint, label):
            return {
                "task": task, "dataset": dataset, "seed": seed,
                "endpoint": endpoint, "label": label,
                "episode": episode[:endpoint],
                "reward": [value / endpoint for value in episode[:endpoint]],
                "flatness": [max(1e-4, 1 - value / endpoint) for value in episode[:endpoint]],
            }

        payload = {
            "torus2": [
                series("torus2", "r1_zero_reject", 4203, 400, "selected"),
                series("torus2", "old_P0X2RCS05", 4101, 300, "old"),
            ],
            "water1": [
                series("water1", "water_fallback_far_0p12", 4301, 400, "selected"),
                *[
                    series("water1", "r1_zero_reject", seed, 300, "r1")
                    for seed in (4201, 4202, 4203)
                ],
                series("water1", "old_P0X2RCS05", 4101, 300, "old"),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.png"
            direct.render(payload, path)
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
