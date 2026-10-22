"""Counterfactual ori6 ledger checks on archived reachable trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import ori_codex3 as ori3
import ori_codex6 as ori6
import test_ori_codex5_saved_paths as ori5_saved


ROOT = Path(__file__).resolve().parent
INITIAL_WATERBOMB_SPREAD = 1.34728491587901
INITIAL_TORUS_SPREAD = 2.414213685053052


def _metrics(flatness, spread, target):
    error = abs(float(spread) - float(target))
    return {
        "flatness_ratio": float(flatness),
        "spread": float(spread),
        "spread_error": error,
        "target_deviation_energy": error * error,
    }


class _ScriptedRewardEnv(ori6.CommonRewardTargetFoldingEnv):
    def compactness_metrics(self, node=None):
        return dict(self.scripted_metrics[self.scripted_index])


def _scripted_environment(
    states,
    reasons,
    validity,
    target,
    max_steps,
    frontier_mode="state_potential",
):
    env = object.__new__(_ScriptedRewardEnv)
    env.scripted_metrics = list(states)
    env.scripted_index = 0
    env.scripted_reasons = list(reasons)
    env.scripted_validity = list(validity)
    env.target_spread = float(target)
    env.spread_tolerance = 0.05
    env.evaluation_flatness_tolerance = 0.001
    env.near_miss_flatness_far = 1.0
    env.near_miss_spread_error_far = 1.0
    env.ori5_flatness_clip = 2.0
    env.ori5_spread_milestone_initial_factor = 2.0
    env.ori5_spread_milestone_bonus = 0.0
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
    env.ori6_spread_frontier_bonus = 0.9
    env.ori6_spread_frontier_mode = str(frontier_mode)
    env.ori6_spread_frontier_flatness_cap = 2.0
    env._reset_ori5_episode_state()
    env._validate_ori6_parameters()
    env._reset_ori6_episode_state()
    return env


def _scripted_parent_step(
    env, physical_direction, requested_width, mode_coefficients
):
    index = env.scripted_index
    reason = env.scripted_reasons[index]
    valid = env.scripted_validity[index]
    env.scripted_index += 1
    env.step_count += 1
    return None, 0.0, reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(valid),
    }


def _load_actions(relative_path):
    with open(ROOT / relative_path, encoding="utf-8") as file:
        return json.load(file)


def _replay(
    relative_path,
    *,
    target,
    initial_spread,
    max_steps,
    stop_after=None,
    final_reason=None,
    frontier_mode="state_potential",
):
    actions = _load_actions(relative_path)
    if stop_after is not None:
        actions = actions[:int(stop_after)]
    states = [_metrics(1.0, initial_spread, target)]
    states.extend(
        _metrics(action["flatness_ratio"], action["spread"], target)
        for action in actions
    )
    reasons = [str(action["termination_reason"]) for action in actions]
    if reasons and final_reason is not None:
        reasons[-1] = str(final_reason)
    validity = [bool(action["after_state_valid"]) for action in actions]
    env = _scripted_environment(
        states,
        reasons,
        validity,
        target,
        max_steps,
        frontier_mode=frontier_mode,
    )
    rewards = []
    infos = []
    with mock.patch.object(
        ori3.CompactTargetFoldingEnv,
        "step",
        _scripted_parent_step,
    ):
        for _ in actions:
            _, reward, _, info = ori6.CommonRewardTargetFoldingEnv.step(
                env, None, 0.0, []
            )
            rewards.append(float(reward))
            infos.append(info)
    return env, actions, np.asarray(rewards), infos


def _component_sum(info):
    return sum(float(info[name]) for name in (
        "ori6_flatness_progress_reward",
        "ori6_target_progress_reward",
        "ori6_spread_frontier_reward",
        "ori6_joint_milestone_reward",
        "ori6_terminal_fixed_reward",
        "ori6_terminal_quality_reward",
        "ori6_terminal_speed_reward",
        "ori6_terminal_correction_reward",
    ))


class OriCodex6SavedPathTests(unittest.TestCase):
    def test_torus_target_hill_prefix_gets_positive_a0_signal(self):
        path = Path(
            "experiments_3d/feasibility/"
            "torus_spread2_collision_free_path/actions.json"
        )
        _, actions, rewards, infos = _replay(
            path,
            target=2.0,
            initial_spread=INITIAL_TORUS_SPREAD,
            max_steps=200,
        )
        first_target = next(
            index for index, action in enumerate(actions)
            if abs(float(action["spread"]) - 2.0) <= 0.05
        )
        self.assertEqual(first_target + 1, 42)
        base_prefix = sum(
            info["ori6_base_reward"] for info in infos[:first_target + 1]
        )
        frontier_prefix = sum(
            info["ori6_spread_frontier_reward"]
            for info in infos[:first_target + 1]
        )
        reward_prefix = float(np.sum(rewards[:first_target + 1]))
        self.assertAlmostEqual(base_prefix, -0.3110218338544917, places=12)
        self.assertAlmostEqual(frontier_prefix, 0.9, places=12)
        self.assertAlmostEqual(reward_prefix, 0.5889781661455082, places=12)
        self.assertGreater(reward_prefix, 0.0)
        self.assertGreater(
            reward_prefix,
            base_prefix,
            msg="A0 must remove the torus F-hill spread dead zone.",
        )

        # If that state were the timeout endpoint, it is still better than an
        # ordinary -1 timeout even though it has not yet flattened.
        _, _, timeout_rewards, timeout_infos = _replay(
            path,
            target=2.0,
            initial_spread=INITIAL_TORUS_SPREAD,
            max_steps=42,
            stop_after=42,
            final_reason="timeout",
        )
        self.assertAlmostEqual(
            float(np.sum(timeout_rewards)), -0.4110218338544918, places=12
        )
        self.assertGreater(float(np.sum(timeout_rewards)), -1.0)
        self.assertEqual(
            timeout_infos[-1]["ori6_terminal_correction_reward"], 0.0
        )

        self.assertEqual(actions[-1]["termination_reason"], "success")
        self.assertAlmostEqual(float(np.sum(rewards)), 14.165665074938653)

    def test_waterbomb_saved_path_ranking_and_crossing_rollback(self):
        cases = {
            "ideal": Path(
                "experiments_3d/feasibility/"
                "waterbomb_spread1_collision_free_path/actions.json"
            ),
            "near_cross": Path(
                "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                "final_training_path/actions.json"
            ),
            "wrong_xy": Path(
                "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
                "best_flatness_path/actions.json"
            ),
        }
        expected = {
            "ideal": 14.495606782652153,
            "near_cross": -0.32647937002236815,
            "wrong_xy": -1.4343393978513936,
        }
        replays = {}
        totals = {}
        for name, path in cases.items():
            replay = _replay(
                path,
                target=1.0,
                initial_spread=INITIAL_WATERBOMB_SPREAD,
                max_steps=400,
            )
            replays[name] = replay
            totals[name] = float(np.sum(replay[2]))
            self.assertAlmostEqual(totals[name], expected[name], places=10)
            for reward, info in zip(replay[2], replay[3]):
                self.assertAlmostEqual(reward, _component_sum(info), places=12)
                self.assertAlmostEqual(
                    info["ori6_reward_decomposition_error"], 0.0, places=12
                )
                self.assertEqual(
                    info["ori6_nonfinite_reward_component_count"], 0
                )

        self.assertGreater(totals["ideal"], totals["near_cross"])
        self.assertGreater(totals["near_cross"], totals["wrong_xy"])

        terminal = replays["near_cross"][3][-1]
        self.assertEqual(
            terminal["ori6_quality_source"],
            "last_valid_before_crossing",
        )
        self.assertAlmostEqual(
            terminal["ori6_frontier_rollback_reward"], -0.9, places=12
        )
        self.assertAlmostEqual(
            terminal["ori6_joint_milestone_rollback_reward"], -0.2,
            places=12,
        )

    def test_ori5_saved_path_totals_remain_unchanged_after_ori6_replays(self):
        # This calls the independent ori5 replay helper after ori6 has already
        # been imported and exercised, detecting accidental global mutation.
        path = Path(
            "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
            "final_training_path/actions.json"
        )
        _, _, ori5_rewards, infos = ori5_saved._replay_saved_actions(path)
        self.assertAlmostEqual(
            float(np.sum(ori5_rewards)), 0.7735206299776328, places=12
        )
        self.assertAlmostEqual(
            sum(info["ori5_spread_milestone_reward"] for info in infos),
            0.9,
            places=12,
        )
        self.assertEqual(
            infos[-1]["ori5_terminal_correction_reward"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
