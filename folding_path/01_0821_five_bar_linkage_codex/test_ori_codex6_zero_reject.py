"""Focused mechanics and reward tests for the ori6 zero-reject candidate."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import ori_codex2 as mechanics
import ori_codex3 as ori3
import ori_codex6_rejected_crossing as rejected
import ori_codex6_spectral_subspace as spectral
import ori_codex6_spectral_subspace_3d as spectral3d
import ori_codex6_zero_reject as zero_reject
import ori_codex6_zero_reject_3d as adapter
import test_ori_codex6_rejected_crossing as rejected_tests


ROOT = Path(__file__).resolve().parent
WATER_ACTIONS = ROOT / (
    "experiments_3d/pilot/water1_B_b8_lr5e4_s107/"
    "final_training_path/actions.json"
)


def _actions(path=WATER_ACTIONS):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _water_args(max_steps=206, weight=0.05):
    return adapter.parse_args([
        zero_reject.CLI_FLAG,
        "--model", "waterbomb",
        "--target-spread", "1",
        "--max-steps", str(int(max_steps)),
        "--ori6-spectral-weight", str(float(weight)),
        "--episodes", "1",
        "--eval-episodes", "1",
        "--periodic-stochastic-eval-episodes", "0",
        "--no-random-search",
        "--no-rigid-diagnostics",
    ])


def _assert_nested_equal(test, left, right):
    if isinstance(left, np.ndarray):
        test.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def _mechanical_snapshot(env):
    return copy.deepcopy({
        name: getattr(env, name)
        for name in (
            "node",
            "previous_potential",
            "previous_ratio",
            "previous_compact_potential",
            "_ori5_flatness_metrics_cache",
            "_ori5_compactness_metrics_cache",
            "_ori5_mode_cache_key",
            "_ori5_mode_cache",
        )
    })


def _ledger_snapshot(env):
    return copy.deepcopy({
        name: getattr(env, name)
        for name in (
            "ori5_cumulative_flatness_reward",
            "ori5_cumulative_target_reward",
            "ori5_cumulative_milestone_reward",
            "ori5_joint_milestones_awarded",
            "ori6_cumulative_spread_frontier_reward",
            "ori6_cumulative_joint_milestone_reward",
            "ori6_current_spread_progress",
            "ori6_current_flatness_gate",
            "ori6_current_spread_potential",
            "ori6_maximum_spread_progress",
            "ori6_maximum_spread_potential",
            "ori6_spectral_current_potential",
            "ori6_spectral_cumulative_reward",
            "ori6_spectral_maximum_potential",
            "ori6_spectral_maximum_delta",
            "ori6_spectral_maximum_rho",
            "ori6_spectral_rollback_count",
            "_ori6_spectral_metrics_cache",
        )
    })


def _make_water(max_steps=206, weight=0.05):
    args = _water_args(max_steps=max_steps, weight=weight)
    node, member, fix, _ = adapter.load_problem_data(args.data_dir)
    return args, node, member, fix


class _TraceAgent(rejected_tests._SavedActionAgent):
    def __init__(self, actions):
        super().__init__(actions)
        self.transitions = []
        self.finished = False

    def get_action(self, state, modes, deterministic=False):
        action = super().get_action(state, modes, deterministic=deterministic)
        marker = torch.tensor(1000.0 + self.index, dtype=torch.float64)
        return mechanics.PolicyAction(
            physical_direction=action.physical_direction,
            mode_coefficients=action.mode_coefficients,
            requested_width=action.requested_width,
            log_probability=marker,
            value=action.value,
            entropy=action.entropy,
            mean_mixture_variance=action.mean_mixture_variance,
            width_mean=action.width_mean,
            width_variance=action.width_variance,
        )

    def add_transition(self, transition):
        self.transitions.append(transition)

    def finish_episode(self):
        self.finished = True


class OriCodex6ZeroRejectTests(unittest.TestCase):
    def test_cli_is_additive_and_preserves_parent_training_arguments(self):
        parent_before = spectral.parse_args([
            spectral.CLI_FLAG,
            "--episodes", "1",
        ])
        args = zero_reject.parse_args([
            zero_reject.CLI_FLAG,
            "--episodes", "1",
            "--actor-learning-rate", "0.00023",
            "--critic-learning-rate", "0.00017",
            "--actor-lr-milestones", "1536", "4000", "7000",
            "--actor-lr-factors", "0.3", "0.1", "0.03",
        ])
        parent_after = spectral.parse_args([
            spectral.CLI_FLAG,
            "--episodes", "1",
        ])

        self.assertTrue(args.ori6_zero_reject_reward)
        self.assertTrue(args.ori6_spectral_subspace)
        self.assertTrue(args.ori6_rejected_crossing_continuation)
        self.assertEqual(args.ori6_crossing_policy, zero_reject.CROSSING_POLICY)
        self.assertEqual(args.actor_learning_rate, 0.00023)
        self.assertEqual(args.critic_learning_rate, 0.00017)
        self.assertEqual(args.actor_lr_milestones, [1536, 4000, 7000])
        self.assertEqual(args.actor_lr_factors, [0.3, 0.1, 0.03])
        self.assertEqual(
            vars(parent_before),
            vars(parent_after),
            "the additive parser must not mutate its parent parser",
        )
        parameters = zero_reject.reward_parameters(args)
        crossing = parameters["crossing_transition"]
        self.assertEqual(parameters["candidate"], zero_reject.CANDIDATE_NAME)
        self.assertEqual(crossing["ordinary_rejected_reward_formula"], "0")
        self.assertFalse(crossing["rejected_crossing_penalty_applied"])
        self.assertTrue(crossing["timeout_reward_applied_exactly_once"])

    def test_repeated_nonterminal_reject_is_exact_zero_noop(self):
        actions = _actions()
        args, node, member, fix = _make_water(max_steps=400)
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            env.reset()
            for action in actions[:-1]:
                _, _, done, info = env.step(
                    np.asarray(action["physical_direction"], dtype=float),
                    float(action["requested_width"]),
                    np.asarray(action["mode_coefficients"], dtype=float),
                )
                self.assertFalse(done)
                self.assertFalse(info["ori6_rejected_crossing"])

            # Materialise the current spectral cache before the snapshot; a
            # rejected step must not add or alter even a cache entry.
            env.spectral_subspace_metrics()
            safe_mechanics = _mechanical_snapshot(env)
            safe_ledgers = _ledger_snapshot(env)
            last = actions[-1]
            original_direction = np.asarray(
                last["physical_direction"], dtype=float
            )
            original_width = float(last["requested_width"])
            original_coefficients = np.asarray(
                last["mode_coefficients"], dtype=float
            )

            for expected_consecutive in (1, 2):
                step_before = env.step_count
                _, reward, done, info = env.step(
                    original_direction,
                    original_width,
                    original_coefficients,
                )
                self.assertFalse(done)
                self.assertEqual(reward.hex(), 0.0.hex())
                self.assertEqual(env.step_count, step_before + 1)
                self.assertEqual(info["ori6_rejected_crossing"], 1)
                self.assertEqual(
                    info["ori6_consecutive_rejected_crossings"],
                    expected_consecutive,
                )
                self.assertEqual(info["accepted_width"], 0.0)
                self.assertEqual(info["actual_displacement"], 0.0)
                self.assertEqual(info["requested_width"], original_width)
                self.assertEqual(info["ori6_spectral_reward"], 0.0)
                self.assertEqual(info["ori6_zero_reject_reward"], 0.0)
                self.assertEqual(info["near_miss_terminal_eligible"], 0)
                self.assertEqual(info["ori6_quality_source"], "none")
                for field in (
                    "ori6_base_reward",
                    "ori6_spread_frontier_reward",
                    "ori6_joint_milestone_reward",
                    "ori6_terminal_fixed_reward",
                    "ori6_terminal_quality_reward",
                    "ori6_flatness_rollback_reward",
                    "ori6_frontier_rollback_reward",
                    "ori6_joint_milestone_rollback_reward",
                    "ori6_terminal_correction_reward",
                    "ori6_rejected_crossing_fixed_reward",
                    "ori6_rejected_crossing_quality_reward",
                    "ori6_rejected_crossing_timeout_reward",
                ):
                    self.assertEqual(info[field], 0.0, field)
                _assert_nested_equal(
                    self, safe_mechanics, _mechanical_snapshot(env)
                )
                _assert_nested_equal(self, safe_ledgers, _ledger_snapshot(env))

    def test_horizon_reject_is_one_normal_timeout_reward(self):
        actions = _actions()
        args, node, member, fix = _make_water(max_steps=len(actions))
        rewards = []
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            env.reset()
            for action in actions:
                _, reward, done, info = env.step(
                    np.asarray(action["physical_direction"], dtype=float),
                    float(action["requested_width"]),
                    np.asarray(action["mode_coefficients"], dtype=float),
                )
                rewards.append(float(reward))
                if done:
                    break

        q = info["ori6_zero_reject_last_safe_q"]
        expected = -env.ori5_timeout_penalty + env.ori5_quality_weight * q
        self.assertEqual(env.step_count, len(actions))
        self.assertEqual(info["termination_reason"], "timeout")
        self.assertEqual(info["ori6_rejected_crossing"], 1)
        self.assertEqual(rewards[-1].hex(), float(expected).hex())
        self.assertEqual(info["ori6_terminal_fixed_reward"], -1.0)
        self.assertEqual(info["ori6_terminal_quality_reward"], q)
        self.assertEqual(info["ori6_rejected_crossing_fixed_reward"], 0.0)
        self.assertEqual(info["ori6_rejected_crossing_quality_reward"], 0.0)
        self.assertEqual(
            info["ori6_rejected_crossing_timeout_reward"].hex(),
            float(expected).hex(),
        )
        self.assertEqual(info["ori6_spectral_reward"], 0.0)
        self.assertEqual(info["near_miss_terminal_eligible"], 1)
        self.assertEqual(
            info["ori6_zero_reject_reward_decomposition_error"], 0.0
        )

    def test_training_trace_retains_sampled_action_and_log_probability(self):
        actions = _actions()
        args, node, member, fix = _make_water(max_steps=len(actions))
        agent = _TraceAgent(actions)
        with adapter.patched_three_dimensional_runtime(args, member):
            env = adapter.make_environment(args, node, member, fix)
            summary, nodes, trace = adapter.run_episode(
                env,
                agent,
                deterministic=True,
                training=True,
            )

        original = actions[-1]
        self.assertTrue(agent.finished)
        self.assertEqual(len(agent.transitions), len(actions))
        self.assertEqual(
            float(agent.transitions[-1].log_probability),
            1000.0 + len(actions),
        )
        self.assertEqual(
            agent.transitions[-1].reward.hex(), trace[-1]["reward"].hex()
        )
        np.testing.assert_array_equal(
            trace[-1]["physical_direction"],
            original["physical_direction"],
        )
        np.testing.assert_array_equal(
            trace[-1]["mode_coefficients"],
            original["mode_coefficients"],
        )
        self.assertEqual(
            trace[-1]["requested_width"], original["requested_width"]
        )
        self.assertEqual(trace[-1]["accepted_width"], 0.0)
        self.assertEqual(trace[-1]["ori6_rejected_crossing"], 1)
        self.assertEqual(len(nodes), len(trace) + 1)
        np.testing.assert_array_equal(nodes[-1], nodes[-2])
        self.assertEqual(summary["termination_reason"], "timeout")

    def test_lambda_zero_and_normal_accepted_transitions_are_parent_exact(self):
        actions = _actions()[:32]
        for weight in (0.0, 0.05):
            with self.subTest(weight=weight):
                parent_args = spectral3d.parse_args([
                    spectral.CLI_FLAG,
                    "--model", "waterbomb",
                    "--target-spread", "1",
                    "--max-steps", "400",
                    "--ori6-spectral-weight", str(weight),
                    "--episodes", "1",
                    "--periodic-stochastic-eval-episodes", "0",
                    "--no-random-search",
                    "--no-rigid-diagnostics",
                ])
                node, member, fix, _ = spectral3d.load_problem_data(
                    parent_args.data_dir
                )
                with spectral3d.patched_three_dimensional_runtime(
                    parent_args, member
                ):
                    parent = spectral3d.make_environment(
                        parent_args, node, member, fix
                    )
                    parent.reset()
                    parent_rows = []
                    for action in actions:
                        state, reward, done, info = parent.step(
                            np.asarray(action["physical_direction"]),
                            float(action["requested_width"]),
                            np.asarray(action["mode_coefficients"]),
                        )
                        self.assertFalse(done)
                        self.assertFalse(info["ori6_rejected_crossing"])
                        parent_rows.append((state.copy(), reward, info))

                args, _, _, _ = _make_water(max_steps=400, weight=weight)
                with adapter.patched_three_dimensional_runtime(args, member):
                    candidate = adapter.make_environment(
                        args, node, member, fix
                    )
                    candidate.reset()
                    for action, (state, reward, parent_info) in zip(
                        actions, parent_rows
                    ):
                        observed_state, observed_reward, done, info = (
                            candidate.step(
                                np.asarray(action["physical_direction"]),
                                float(action["requested_width"]),
                                np.asarray(action["mode_coefficients"]),
                            )
                        )
                        self.assertFalse(done)
                        np.testing.assert_array_equal(observed_state, state)
                        self.assertEqual(observed_reward.hex(), reward.hex())
                        self.assertEqual(
                            info["ori6_spectral_reward"].hex(),
                            parent_info["ori6_spectral_reward"].hex(),
                        )
                        self.assertEqual(
                            info["ori6_spectral_potential_difference"].hex(),
                            parent_info[
                                "ori6_spectral_potential_difference"
                            ].hex(),
                        )

    def test_all_five_saved_success_paths_are_transition_bit_exact(self):
        cases = (
            (
                "five_d2", False, None, 2.0,
                ROOT / "experiments_ori5/full/five_d2/B/seed2101/"
                "first_success_path/actions.json",
            ),
            (
                "five_d3", False, None, 3.0,
                ROOT / "experiments_ori5/full/five_d3/B/seed2101/"
                "first_success_path/actions.json",
            ),
            (
                "torus2", True, "4unit_torus", 2.0,
                ROOT / "experiments_ori6/screen/rescue/P0X2/torus2/A/"
                "seed3103/first_success_path/actions.json",
            ),
            (
                "water1", True, "waterbomb", 1.0,
                ROOT / "experiments_ori5/tuning/r1/water1/B/seed1101/"
                "first_success_path/actions.json",
            ),
            (
                "watersqrt2", True, "waterbomb", float(np.sqrt(2.0)),
                ROOT / "experiments_ori5/full/watersqrt2/B/seed2101/"
                "first_success_path/actions.json",
            ),
        )
        for name, spatial, model, target, path in cases:
            with self.subTest(task=name):
                actions = _actions(path)
                self.assertEqual(actions[-1]["termination_reason"], "success")
                if spatial:
                    parent_args = spectral3d.parse_args([
                        spectral.CLI_FLAG,
                        "--model", model,
                        "--target-spread", str(target),
                        "--max-steps", str(len(actions) + 10),
                        "--episodes", "1",
                        "--periodic-stochastic-eval-episodes", "0",
                        "--no-random-search",
                        "--no-rigid-diagnostics",
                    ])
                    zero_args = adapter.parse_args([
                        zero_reject.CLI_FLAG,
                        "--model", model,
                        "--target-spread", str(target),
                        "--max-steps", str(len(actions) + 10),
                        "--episodes", "1",
                        "--periodic-stochastic-eval-episodes", "0",
                        "--no-random-search",
                        "--no-rigid-diagnostics",
                    ])
                    node, member, fix, _ = adapter.load_problem_data(
                        zero_args.data_dir
                    )
                    parent_context = spectral3d.patched_three_dimensional_runtime(
                        parent_args, member
                    )
                    zero_context = adapter.patched_three_dimensional_runtime(
                        zero_args, member
                    )
                    parent_factory = spectral3d.make_environment
                    zero_factory = adapter.make_environment
                else:
                    parent_args = spectral.parse_args([
                        spectral.CLI_FLAG,
                        "--data-dir", str(ROOT / "model/five_bar_linkage"),
                        "--target-spread", str(target),
                        "--max-steps", str(len(actions) + 10),
                        "--episodes", "1",
                        "--no-random-search",
                    ])
                    zero_args = zero_reject.parse_args([
                        zero_reject.CLI_FLAG,
                        "--data-dir", str(ROOT / "model/five_bar_linkage"),
                        "--target-spread", str(target),
                        "--max-steps", str(len(actions) + 10),
                        "--episodes", "1",
                        "--no-random-search",
                    ])
                    node, member, fix = ori3._load_problem_data(zero_args)
                    from contextlib import nullcontext
                    parent_context = nullcontext()
                    zero_context = nullcontext()
                    parent_factory = spectral.make_environment
                    zero_factory = zero_reject.make_environment

                with parent_context:
                    parent = parent_factory(
                        parent_args, node, member, fix
                    )
                    parent.reset()
                    parent_rows = []
                    for action in actions:
                        state, reward, done, info = parent.step(
                            np.asarray(action["physical_direction"]),
                            float(action["requested_width"]),
                            np.asarray(action["mode_coefficients"]),
                        )
                        parent_rows.append((state.copy(), reward, done, info))
                        if done:
                            break

                with zero_context:
                    candidate = zero_factory(
                        zero_args, node, member, fix
                    )
                    candidate.reset()
                    candidate_rows = []
                    for action in actions:
                        state, reward, done, info = candidate.step(
                            np.asarray(action["physical_direction"]),
                            float(action["requested_width"]),
                            np.asarray(action["mode_coefficients"]),
                        )
                        candidate_rows.append((state.copy(), reward, done, info))
                        if done:
                            break

                self.assertEqual(len(parent_rows), len(candidate_rows))
                for parent_row, candidate_row in zip(
                    parent_rows, candidate_rows
                ):
                    np.testing.assert_array_equal(
                        parent_row[0], candidate_row[0]
                    )
                    self.assertEqual(
                        parent_row[1].hex(), candidate_row[1].hex()
                    )
                    self.assertEqual(parent_row[2], candidate_row[2])
                    self.assertEqual(
                        parent_row[3]["termination_reason"],
                        candidate_row[3]["termination_reason"],
                    )
                    self.assertFalse(
                        parent_row[3].get("ori6_rejected_crossing", 0)
                    )
                self.assertEqual(
                    candidate_rows[-1][3]["termination_reason"], "success"
                )

    def test_planar_and_3d_training_restore_all_runtime_hooks(self):
        spectral_original = (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
            spectral.reward_parameters,
        )
        zero_original = (
            zero_reject.make_environment,
            zero_reject.run_episode,
            zero_reject.aggregate_stochastic_evaluations,
        )
        collision_original = mechanics.has_proper_self_crossing
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planar_args = zero_reject.parse_args([
                zero_reject.CLI_FLAG,
                "--data-dir", str(ROOT / "model/five_bar_linkage"),
                "--target-spread", "2",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--max-steps", "1",
                "--batch-size", "1",
                "--moving-average-window", "1",
                "--seed", "71",
                "--no-random-search",
                "--output-dir", str(root / "planar"),
            ])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                planar_output, planar_summary = zero_reject.train(planar_args)

            spatial_args = adapter.parse_args([
                zero_reject.CLI_FLAG,
                "--model", "waterbomb",
                "--target-spread", "1",
                "--episodes", "1",
                "--eval-episodes", "1",
                "--periodic-stochastic-eval-episodes", "0",
                "--max-steps", "1",
                "--batch-size", "1",
                "--moving-average-window", "1",
                "--seed", "73",
                "--no-random-search",
                "--no-rigid-diagnostics",
                "--output-dir", str(root / "spatial"),
            ])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                spatial_output, spatial_summary = adapter.train(spatial_args)

            for output, summary, entrypoint in (
                (
                    planar_output,
                    planar_summary,
                    "ori_codex6_zero_reject.py",
                ),
                (
                    spatial_output,
                    spatial_summary,
                    "ori_codex6_zero_reject_3d.py",
                ),
            ):
                with open(
                    Path(output) / "run_config.json", encoding="utf-8"
                ) as file:
                    config = json.load(file)
                self.assertEqual(
                    summary["ori6_candidate"], zero_reject.CANDIDATE_NAME
                )
                self.assertEqual(config["candidate_entrypoint"], entrypoint)
                self.assertEqual(
                    config["zero_reject_formula_sha256"],
                    zero_reject.ZERO_REJECT_FORMULA_SHA256,
                )

        self.assertEqual(spectral_original, (
            spectral.make_environment,
            spectral.run_episode,
            spectral.aggregate_stochastic_evaluations,
            spectral.reward_parameters,
        ))
        self.assertEqual(zero_original, (
            zero_reject.make_environment,
            zero_reject.run_episode,
            zero_reject.aggregate_stochastic_evaluations,
        ))
        self.assertIs(mechanics.has_proper_self_crossing, collision_original)


if __name__ == "__main__":
    unittest.main()
