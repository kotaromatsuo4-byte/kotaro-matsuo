"""Independent reward-ledger checks using the saved waterbomb trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import ori_codex3 as ori3
import ori_codex5 as ori5


ROOT = Path(__file__).resolve().parent
INITIAL_WATERBOMB_SPREAD = 1.34728491587901


def _metrics(flatness, spread, target=1.0):
    error = abs(float(spread) - float(target))
    return {
        "flatness_ratio": float(flatness),
        "spread": float(spread),
        "spread_error": error,
        "target_deviation_energy": error * error,
    }


class _ScriptedRewardEnv(ori5.CommonRewardTargetFoldingEnv):
    def compactness_metrics(self, node=None):
        return dict(self.scripted_metrics[self.scripted_index])


def _scripted_environment(
    states, reasons, validity, max_steps=400, spread_error_far=1.0
):
    env = object.__new__(_ScriptedRewardEnv)
    env.scripted_metrics = list(states)
    env.scripted_index = 0
    env.scripted_reasons = list(reasons)
    env.scripted_validity = list(validity)
    env.target_spread = 1.0
    env.spread_tolerance = 0.05
    env.evaluation_flatness_tolerance = 0.001
    env.near_miss_flatness_far = 1.0
    env.near_miss_spread_error_far = float(spread_error_far)
    env.ori5_flatness_clip = 2.0
    env.ori5_spread_milestone_initial_factor = 2.0
    env.ori5_spread_milestone_bonus = 0.9
    env.ori5_joint_flatness_thresholds = (0.5, 0.1, 0.01)
    env.ori5_joint_milestone_bonuses = (0.2, 0.3, 0.5)
    env.ori5_crossing_penalty = 1.25
    env.ori5_wrong_flat_penalty = 2.0
    env.ori5_timeout_penalty = 1.0
    env.ori5_success_bonus = 10.0
    env.ori5_quality_weight = 1.0
    env.ori5_success_speed_bonus_weight = 0.5
    env.numerical_failure_penalty = 10.0
    env.no_mode_penalty = 5.0
    env.max_steps = int(max_steps)
    env.step_count = 0
    env._reset_ori5_episode_state()
    return env


def _scripted_parent_step(
    env, physical_direction, requested_width, mode_coefficients
):
    index = env.scripted_index
    reason = env.scripted_reasons[index]
    is_valid = env.scripted_validity[index]
    env.scripted_index += 1
    env.step_count += 1
    return None, 0.0, reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(is_valid),
    }


def _load_actions(relative_path):
    with open(ROOT / relative_path, encoding="utf-8") as file:
        return json.load(file)


def _replay_saved_actions(relative_path, spread_error_far=1.0):
    actions = _load_actions(relative_path)
    states = [_metrics(1.0, INITIAL_WATERBOMB_SPREAD)]
    states.extend(
        _metrics(action["flatness_ratio"], action["spread"])
        for action in actions
    )
    reasons = [str(action["termination_reason"]) for action in actions]
    validity = [bool(action["after_state_valid"]) for action in actions]
    env = _scripted_environment(
        states, reasons, validity, spread_error_far=spread_error_far
    )
    rewards = []
    infos = []
    with mock.patch.object(
        ori3.CompactTargetFoldingEnv,
        "step",
        _scripted_parent_step,
    ):
        for _ in actions:
            _, reward, _, info = ori5.CommonRewardTargetFoldingEnv.step(
                env, None, 0.0, []
            )
            rewards.append(float(reward))
            infos.append(info)
    return env, actions, np.asarray(rewards), infos


def _component_sum(info):
    return sum(float(info[name]) for name in (
        "ori5_flatness_progress_reward",
        "ori5_target_progress_reward",
        "ori5_milestone_reward",
        "ori5_terminal_fixed_reward",
        "ori5_terminal_quality_reward",
        "ori5_terminal_speed_reward",
        "ori5_terminal_correction_reward",
    ))


class OriCodex5SavedPathAuditTests(unittest.TestCase):
    def test_saved_waterbomb_paths_reproduce_counterfactual_ranking(self):
        cases = {
            "ideal": (
                Path(
                    "experiments_3d/feasibility/"
                    "waterbomb_spread1_collision_free_path/actions.json"
                ),
                14.495606782652154,
            ),
            "near_cross": (
                Path(
                    "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                    "final_training_path/actions.json"
                ),
                0.7735206299776328,
            ),
            "wrong_xy": (
                Path(
                    "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                    "best_flatness_path/actions.json"
                ),
                -1.4343393978513934,
            ),
        }
        totals = {}
        replays = {}
        for name, (path, expected) in cases.items():
            replay = _replay_saved_actions(path)
            replays[name] = replay
            total = float(np.sum(replay[2]))
            totals[name] = total
            self.assertAlmostEqual(total, expected, places=10)
            for reward, info in zip(replay[2], replay[3]):
                self.assertAlmostEqual(
                    float(reward), _component_sum(info), places=12
                )
                self.assertEqual(
                    info["ori5_nonfinite_reward_component_count"], 0
                )
                self.assertAlmostEqual(
                    info["ori5_reward_decomposition_error"], 0.0, places=12
                )

        self.assertGreater(totals["ideal"], totals["near_cross"])
        self.assertGreater(totals["near_cross"], totals["wrong_xy"])

        annotated_cross = _replay_saved_actions(
            cases["near_cross"][0], spread_error_far=0.3
        )[2].sum()
        annotated_wrong = _replay_saved_actions(
            cases["wrong_xy"][0], spread_error_far=0.3
        )[2].sum()
        self.assertAlmostEqual(annotated_cross, 0.7735206299776328)
        self.assertAlmostEqual(annotated_wrong, -2.050961939472019)
        self.assertGreater(annotated_cross, annotated_wrong)

        _, _, _, ideal_infos = replays["ideal"]
        self.assertAlmostEqual(
            ideal_infos[-1]["ori5_terminal_speed_reward"], 0.475
        )
        self.assertTrue(all(
            info["ori5_terminal_speed_reward"] == 0.0
            for info in replays["near_cross"][3]
        ))
        self.assertTrue(all(
            info["ori5_terminal_speed_reward"] == 0.0
            for info in replays["wrong_xy"][3]
        ))

    def test_crossing_and_wrong_flat_use_the_agreed_ledgers(self):
        cross_env, _, cross_rewards, cross_infos = _replay_saved_actions(
            Path(
                "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                "final_training_path/actions.json"
            )
        )
        terminal_cross = cross_infos[-1]
        self.assertEqual(
            terminal_cross["ori5_quality_source"],
            "last_valid_before_crossing",
        )
        self.assertAlmostEqual(
            terminal_cross["ori5_q"], 0.1522603828316028, places=12
        )
        self.assertAlmostEqual(
            cross_env.ori5_last_valid_metrics["flatness_ratio"],
            0.34931630007516723,
            places=12,
        )
        self.assertAlmostEqual(float(np.sum(cross_rewards)), 0.7735206299776328)

        _, _, wrong_rewards, wrong_infos = _replay_saved_actions(
            Path(
                "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                "best_flatness_path/actions.json"
            )
        )
        total_flatness = sum(
            info["ori5_flatness_progress_reward"] for info in wrong_infos
        )
        total_milestones = sum(
            info["ori5_milestone_reward"] for info in wrong_infos
        )
        terminal_correction = wrong_infos[-1][
            "ori5_terminal_correction_reward"
        ]
        self.assertAlmostEqual(
            total_flatness + total_milestones + terminal_correction,
            0.0,
            places=12,
        )
        self.assertAlmostEqual(float(np.sum(wrong_rewards)), -1.4343393978513934)

    def test_training_wrapper_records_schema_and_writes_both_curves(self):
        args = ori5.parse_args([
            "--episodes", "1", "--moving-average-window", "1",
            "--no-random-search",
        ])
        history_row = {
            "episode": 1,
            "near_miss_q": 0.25,
            "near_miss_bonus": 0.25,
            "ori5_total_spread_milestone_reward": 0.9,
            "ori5_total_joint_milestone_reward": 0.2,
            "ori5_total_milestone_reward": 1.1,
            "ori5_total_terminal_correction_reward": 0.0,
            "ori5_reward_decomposition_error": 0.0,
            "ori5_maximum_abs_step_reward_decomposition_error": 0.0,
            "ori5_nonfinite_reward_component_count": 0,
            "success": 0,
            "self_crossing": 1,
            "wrong_flat": 0,
            "timeout": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def fake_train(_):
                with open(
                    output / "training_history.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(
                        file, fieldnames=list(history_row)
                    )
                    writer.writeheader()
                    writer.writerow(history_row)
                with open(
                    output / "run_config.json", "w", encoding="utf-8"
                ) as file:
                    json.dump({}, file)
                return output, {}

            with mock.patch.object(ori5.ori3, "train", fake_train):
                returned, summary = ori5.train(args)

            self.assertEqual(Path(returned), output)
            with open(
                output / "run_config.json", encoding="utf-8"
            ) as file:
                config = json.load(file)
            self.assertEqual(config["reward_design"], ori5.REWARD_DESIGN_NAME)
            self.assertEqual(
                config["reward_schema_version"], ori5.REWARD_SCHEMA_VERSION
            )
            self.assertEqual(
                config["ori5_reward_parameters"][
                    "path_integrated_cost_weight"
                ],
                0.0,
            )
            self.assertEqual(summary["ori5_nonfinite_reward_component_count"], 0)
            self.assertTrue(
                (output / "ori5_milestone_learning_curve.png").is_file()
            )
            self.assertTrue(
                (output / "near_miss_learning_curve.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
