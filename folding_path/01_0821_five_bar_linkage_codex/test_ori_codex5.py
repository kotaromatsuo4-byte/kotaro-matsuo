import math
import unittest
from unittest import mock

import numpy as np

import ori_codex3 as ori3
import ori_codex5 as ori5


def metrics(flatness, spread, target=2.0):
    error = abs(float(spread) - float(target))
    return {
        "flatness_ratio": float(flatness),
        "spread": float(spread),
        "spread_error": error,
        "target_deviation_energy": error * error,
    }


class ScriptedRewardEnv(ori5.CommonRewardTargetFoldingEnv):
    def compactness_metrics(self, node=None):
        return dict(self.scripted_metrics[self.scripted_index])


def scripted_environment(states, reasons, valid=None, max_steps=10):
    env = object.__new__(ScriptedRewardEnv)
    env.scripted_metrics = list(states)
    env.scripted_index = 0
    env.scripted_reasons = list(reasons)
    env.scripted_valid = (
        [True] * len(reasons) if valid is None else list(valid)
    )
    env.target_spread = 2.0
    env.spread_tolerance = 0.05
    env.evaluation_flatness_tolerance = 0.001
    env.near_miss_flatness_far = 1.0
    env.near_miss_spread_error_far = 1.0
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


def scripted_parent_step(
    env, physical_direction, requested_width, mode_coefficients
):
    index = env.scripted_index
    reason = env.scripted_reasons[index]
    is_valid = env.scripted_valid[index]
    env.scripted_index += 1
    env.step_count += 1
    return None, 12345.0, reason != "running", {
        "termination_reason": reason,
        "after_state_valid": int(is_valid),
    }


def take_scripted_step(env):
    with mock.patch.object(
        ori3.CompactTargetFoldingEnv,
        "step",
        scripted_parent_step,
    ):
        return ori5.CommonRewardTargetFoldingEnv.step(env, None, 0.0, [])


def test_endpoint_potential_telescopes_across_a_detour():
    segments = [
        ori5.endpoint_reward_components(1.0, 1.6, 2.0, 2.4, 2.0),
        ori5.endpoint_reward_components(1.6, 0.2, 2.4, 2.1, 2.0),
    ]
    total = sum(row["base_reward"] for row in segments)
    direct = ori5.endpoint_reward_components(1.0, 0.2, 2.0, 2.1, 2.0)
    np.testing.assert_allclose(total, direct["base_reward"])
    assert all("integrated" not in key for key in direct)


def test_success_step_can_claim_all_newly_crossed_nested_milestones():
    env = scripted_environment(
        [metrics(1.0, 2.0), metrics(0.0005, 2.0)],
        ["success"],
        max_steps=10,
    )
    _, reward, done, info = take_scripted_step(env)
    assert done
    assert info["ori5_joint_milestone_rewards"] == [0.2, 0.3, 0.5]
    np.testing.assert_allclose(info["ori5_milestone_reward"], 1.0)
    np.testing.assert_allclose(info["ori5_terminal_speed_reward"], 0.45)
    np.testing.assert_allclose(
        reward, 0.9995 + 1.0 + 10.0 + 1.0 + 0.45
    )


def test_spread_milestone_is_once_only_after_a_far_initial_state():
    env = scripted_environment(
        [
            metrics(1.0, 0.0),
            metrics(1.0, 2.0),
            metrics(1.0, 2.3),
            metrics(1.0, 2.0),
        ],
        ["running", "running", "running"],
    )
    rewards = []
    for _ in range(3):
        _, _, _, info = take_scripted_step(env)
        rewards.append(info["ori5_spread_milestone_reward"])
    assert rewards == [0.9, 0.0, 0.0]


def test_crossing_uses_last_valid_quality_and_ignores_candidate_progress():
    env = scripted_environment(
        [metrics(0.01, 2.0), metrics(0.0001, 3.0)],
        ["self_crossing"],
        valid=[False],
    )
    _, reward, done, info = take_scripted_step(env)
    expected_q = math.log(100.0) / math.log(1000.0)
    assert done
    assert info["ori5_quality_source"] == "last_valid_before_crossing"
    np.testing.assert_allclose(info["ori5_q"], expected_q)
    assert info["ori5_base_reward"] == 0.0
    assert info["ori5_milestone_reward"] == 0.0
    np.testing.assert_allclose(reward, -1.25 + expected_q)


def test_wrong_flat_cancels_all_episode_flatness_and_milestone_credit():
    env = scripted_environment(
        [metrics(1.0, 2.0), metrics(0.05, 2.0), metrics(0.0005, 3.0)],
        ["running", "wrong_flat"],
    )
    _, first_reward, _, first = take_scripted_step(env)
    _, second_reward, done, second = take_scripted_step(env)
    assert done
    np.testing.assert_allclose(first["ori5_milestone_reward"], 0.5)
    np.testing.assert_allclose(
        second["ori5_terminal_correction_reward"], -(0.9995 + 0.5)
    )
    # Net episode reward contains target potential only, then -Cw + qD.
    np.testing.assert_allclose(first_reward + second_reward, -1.0 - 2.0)


def test_timeout_step_can_claim_milestones_and_terminal_near_miss():
    env = scripted_environment(
        [metrics(1.0, 2.0), metrics(0.005, 2.0)],
        ["timeout"],
        max_steps=1,
    )
    _, reward, done, info = take_scripted_step(env)
    q = math.log(1.0 / 0.005) / math.log(1000.0)
    assert done
    assert info["ori5_joint_milestone_rewards"] == [0.2, 0.3, 0.5]
    np.testing.assert_allclose(info["ori5_terminal_quality_reward"], q)
    np.testing.assert_allclose(reward, 0.995 + 1.0 - 1.0 + q)


def test_parse_defaults_and_gamma_validation():
    args = ori5.parse_args(["--episodes", "1", "--no-random-search"])
    assert args.gamma == 1.0
    assert args.ori5_crossing_penalty == 1.25
    assert args.ori5_joint_flatness_thresholds == [0.5, 0.1, 0.01]
    bad = ori5.parse_args([
        "--episodes", "1", "--no-random-search", "--gamma", "0.99"
    ])
    try:
        ori5._validate_args(bad)
    except ValueError as error:
        assert "requires --gamma 1" in str(error)
    else:
        raise AssertionError("gamma != 1 must be rejected")


class OriCodex5RewardTests(unittest.TestCase):
    test_endpoint_potential_telescopes_across_a_detour = staticmethod(
        test_endpoint_potential_telescopes_across_a_detour
    )
    test_success_step_can_claim_all_newly_crossed_nested_milestones = (
        staticmethod(
            test_success_step_can_claim_all_newly_crossed_nested_milestones
        )
    )
    test_spread_milestone_is_once_only_after_a_far_initial_state = (
        staticmethod(
            test_spread_milestone_is_once_only_after_a_far_initial_state
        )
    )
    test_crossing_uses_last_valid_quality_and_ignores_candidate_progress = (
        staticmethod(
            test_crossing_uses_last_valid_quality_and_ignores_candidate_progress
        )
    )
    test_wrong_flat_cancels_all_episode_flatness_and_milestone_credit = (
        staticmethod(
            test_wrong_flat_cancels_all_episode_flatness_and_milestone_credit
        )
    )
    test_timeout_step_can_claim_milestones_and_terminal_near_miss = (
        staticmethod(
            test_timeout_step_can_claim_milestones_and_terminal_near_miss
        )
    )
    test_parse_defaults_and_gamma_validation = staticmethod(
        test_parse_defaults_and_gamma_validation
    )


if __name__ == "__main__":
    unittest.main()
