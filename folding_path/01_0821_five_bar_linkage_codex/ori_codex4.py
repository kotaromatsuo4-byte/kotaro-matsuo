"""Valid-terminal near-miss reward derived from :mod:`ori_codex3`.

The mechanics, action space, dense path reward, and Actor--Critic update are
unchanged.  A bounded joint near-miss bonus is added exactly once, and only
when an admissible episode terminates by success, wrong-flat, or timeout.
Self-crossing, correction failure, and no-mode termination cannot claim the
bonus (nor quality computed from an inadmissible candidate).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

import ori_codex2 as mechanics
import ori_codex3 as ori3


_ORI3_MAKE_ENVIRONMENT = ori3.make_environment
_ORI3_RUN_EPISODE = ori3.run_episode
_ORI3_AGGREGATE_EVALUATIONS = ori3.aggregate_stochastic_evaluations


class NearMissTargetFoldingEnv(ori3.CompactTargetFoldingEnv):
    """ori3 environment with a bounded bonus on valid terminal states."""

    def __init__(
        self,
        *args,
        near_miss_bonus_weight=1.0,
        near_miss_flatness_far=1.0,
        near_miss_spread_error_far=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.near_miss_bonus_weight = float(near_miss_bonus_weight)
        self.near_miss_flatness_far = float(near_miss_flatness_far)
        self.near_miss_spread_error_far = float(
            near_miss_spread_error_far
        )
        if (
            not np.isfinite(self.near_miss_bonus_weight)
            or self.near_miss_bonus_weight < 0.0
        ):
            raise ValueError(
                "near_miss_bonus_weight must be finite and non-negative."
            )
        if (
            not np.isfinite(self.near_miss_flatness_far)
            or self.near_miss_flatness_far
            <= self.evaluation_flatness_tolerance
        ):
            raise ValueError(
                "near_miss_flatness_far must be finite and greater than "
                "flatness_tolerance."
            )
        if (
            not np.isfinite(self.near_miss_spread_error_far)
            or self.near_miss_spread_error_far <= self.spread_tolerance
        ):
            raise ValueError(
                "near_miss_spread_error_far must be finite and greater than "
                "spread_tolerance."
            )
        self.near_miss_bonus_awarded = False

    def reset(self):
        state = super().reset()
        self.near_miss_bonus_awarded = False
        return state

    def near_miss_quality(self, flatness_ratio, spread_error):
        """Return (qF, qD, qF*qD), each clipped to [0, 1]."""
        flatness_ratio = float(flatness_ratio)
        spread_error = abs(float(spread_error))
        if not np.isfinite(flatness_ratio) or not np.isfinite(spread_error):
            return 0.0, 0.0, 0.0

        flatness_tolerance = self.evaluation_flatness_tolerance
        denominator = np.log(
            self.near_miss_flatness_far / flatness_tolerance
        )
        q_flatness = np.log(
            self.near_miss_flatness_far
            / max(flatness_ratio, flatness_tolerance)
        ) / denominator
        q_flatness = float(np.clip(q_flatness, 0.0, 1.0))

        q_spread = (
            self.near_miss_spread_error_far - spread_error
        ) / (
            self.near_miss_spread_error_far - self.spread_tolerance
        )
        q_spread = float(np.clip(q_spread, 0.0, 1.0))
        return q_flatness, q_spread, float(q_flatness * q_spread)

    def step(self, physical_direction, requested_width, mode_coefficients):
        next_state, reward, done, info = super().step(
            physical_direction, requested_width, mode_coefficients
        )
        valid_after_state = bool(info.get("after_state_valid", 0))
        if valid_after_state:
            q_flatness, q_spread, quality = self.near_miss_quality(
                info["flatness_ratio"], info["spread_error"]
            )
        else:
            # In particular, never score the geometry of a crossed candidate.
            q_flatness, q_spread, quality = 0.0, 0.0, 0.0

        reason = info["termination_reason"]
        eligible_terminal = bool(
            done
            and valid_after_state
            and reason in ("success", "wrong_flat", "timeout")
        )
        bonus = 0.0
        if eligible_terminal and not self.near_miss_bonus_awarded:
            bonus = float(self.near_miss_bonus_weight * quality)
            reward = float(reward + bonus)
            info["terminal_reward_component"] = float(
                info.get("terminal_reward_component", 0.0) + bonus
            )
            self.near_miss_bonus_awarded = True

        info.update({
            "near_miss_qF": q_flatness,
            "near_miss_qD": q_spread,
            "near_miss_q": quality,
            "near_miss_bonus": bonus,
            "near_miss_terminal_eligible": int(eligible_terminal),
            "near_miss_bonus_awarded": int(self.near_miss_bonus_awarded),
        })
        return next_state, float(reward), done, info


def make_environment(args, node, member, fix):
    """Construct ori3 mechanics with the ori4 terminal bonus parameters."""
    return NearMissTargetFoldingEnv(
        node=node,
        member=member,
        fix=fix,
        max_steps=args.max_steps,
        max_step_width=args.max_step_width,
        min_step_width=args.min_step_width,
        projection_shrink=args.projection_shrink,
        max_correction_ratio=args.max_correction_ratio,
        max_actual_step_ratio=args.max_actual_step_ratio,
        flatness_subspace_dimension=args.flatness_subspace_dimension,
        flatness_tolerance=args.flatness_tolerance,
        strain_tolerance=args.strain_tolerance,
        crossing_tolerance=args.crossing_tolerance,
        flatness_reward_weight=0.0,
        flatness_reward_clip=args.flatness_reward_clip,
        success_bonus=0.0,
        crossing_penalty=args.crossing_penalty,
        numerical_failure_penalty=args.numerical_failure_penalty,
        no_mode_penalty=args.no_mode_penalty,
        timeout_penalty=args.timeout_penalty,
        target_spread=args.target_spread,
        spread_sigma=args.spread_sigma,
        spread_tolerance=args.spread_tolerance,
        flat_quality_clip=args.flat_quality_clip,
        quality_combination=args.quality_combination,
        flat_quality_weight=args.flat_quality_weight,
        flat_signal_floor=args.flat_signal_floor,
        compact_potential_weight=args.compact_potential_weight,
        flatness_progress_weight=args.flatness_progress_weight,
        target_progress_weight=args.target_progress_weight,
        target_deviation_cost_weight=args.target_deviation_cost_weight,
        shaping_discount=args.gamma,
        flat_terminal_bonus=args.flat_terminal_bonus,
        joint_success_bonus=args.joint_success_bonus,
        wrong_flat_penalty=args.wrong_flat_penalty,
        terminate_wrong_flat=args.terminate_wrong_flat,
        training_flatness_tolerances=args.training_flatness_tolerances,
        training_flatness_milestones=args.training_flatness_milestones,
        curriculum_waypoint_bonus=args.curriculum_waypoint_bonus,
        curriculum_waypoint_start_episode=(
            args.curriculum_waypoint_start_episode
        ),
        gated_compact_cost_weight=args.gated_compact_cost_weight,
        compact_gate_scale=args.compact_gate_scale,
        near_miss_bonus_weight=args.near_miss_bonus_weight,
        near_miss_flatness_far=args.near_miss_flatness_far,
        near_miss_spread_error_far=args.near_miss_spread_error_far,
    )


def run_episode(env, agent, deterministic=False, training=False):
    """Run ori3 and add explicit near-miss fields to summaries/actions."""
    summary, nodes, actions = _ORI3_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    terminal_index = len(actions) - 1
    for action_index, action in enumerate(actions):
        valid_after_state = bool(action.get("after_state_valid", 1))
        if valid_after_state:
            q_flatness, q_spread, quality = env.near_miss_quality(
                action["flatness_ratio"], action["spread_error"]
            )
        else:
            q_flatness, q_spread, quality = 0.0, 0.0, 0.0
        eligible = bool(
            action_index == terminal_index
            and valid_after_state
            and summary["termination_reason"]
            in ("success", "wrong_flat", "timeout")
        )
        bonus = float(
            env.near_miss_bonus_weight * quality if eligible else 0.0
        )
        action.update({
            "near_miss_qF": q_flatness,
            "near_miss_qD": q_spread,
            "near_miss_q": quality,
            "near_miss_bonus": bonus,
            "near_miss_terminal_eligible": int(eligible),
        })

    if actions:
        terminal_action = actions[-1]
        q_flatness = float(terminal_action["near_miss_qF"])
        q_spread = float(terminal_action["near_miss_qD"])
        quality = float(terminal_action["near_miss_q"])
        bonus = float(terminal_action["near_miss_bonus"])
        eligible = int(terminal_action["near_miss_terminal_eligible"])
    else:
        q_flatness = q_spread = quality = bonus = 0.0
        eligible = 0
    summary.update({
        "near_miss_qF": q_flatness,
        "near_miss_qD": q_spread,
        "near_miss_q": quality,
        "near_miss_bonus": bonus,
        "near_miss_terminal_eligible": eligible,
        "near_miss_bonus_awarded": int(eligible),
    })
    return summary, nodes, actions


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _ORI3_AGGREGATE_EVALUATIONS(evaluations, episode=episode)
    if result is None:
        return None
    result.update({
        "mean_near_miss_qF": float(np.mean([
            row["near_miss_qF"] for row in evaluations
        ])),
        "mean_near_miss_qD": float(np.mean([
            row["near_miss_qD"] for row in evaluations
        ])),
        "mean_near_miss_q": float(np.mean([
            row["near_miss_q"] for row in evaluations
        ])),
        "mean_near_miss_bonus": float(np.mean([
            row["near_miss_bonus"] for row in evaluations
        ])),
        "near_miss_terminal_eligible_rate": float(np.mean([
            row["near_miss_terminal_eligible"] for row in evaluations
        ])),
    })
    return result


def save_near_miss_learning_curve(output_directory, rows, window):
    """Save the near-miss signal separately from ori3's core curves."""
    if not rows:
        return
    episodes = np.asarray([float(row["episode"]) for row in rows])
    quality = np.asarray([float(row["near_miss_q"]) for row in rows])
    bonus = np.asarray([float(row["near_miss_bonus"]) for row in rows])
    figure, axes = ori3.plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, values, title in (
        (axes[0], quality, "Terminal joint near-miss quality"),
        (axes[1], bonus, "Terminal near-miss bonus"),
    ):
        axis.plot(episodes, values, color="0.75", linewidth=0.5)
        x_average, averaged = mechanics.moving_average(values, window)
        if len(x_average):
            axis.plot(
                x_average, averaged, color="tab:blue", linewidth=1.8,
                label=f"{window}-episode average",
            )
            axis.legend()
        axis.set_xlabel("Episode")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(
        -0.02,
        max(1.02, float(np.nanmax(bonus)) * 1.05),
    )
    figure.tight_layout()
    figure.savefig(
        Path(output_directory) / "near_miss_learning_curve.png", dpi=180
    )
    ori3.plt.close(figure)


def _validate_args(args):
    finite_nonnegative = (
        "near_miss_bonus_weight",
    )
    for name in finite_nonnegative:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if args.episodes < 1:
        raise ValueError("episodes must be positive.")
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if not np.isfinite(args.max_step_width) or args.max_step_width <= 0.0:
        raise ValueError("max_step_width must be finite and positive.")
    if args.max_step_width < args.min_step_width:
        raise ValueError("max_step_width cannot be smaller than min_step_width.")
    if (
        not np.isfinite(args.flatness_tolerance)
        or args.flatness_tolerance <= 0.0
    ):
        raise ValueError("flatness_tolerance must be finite and positive.")
    if (
        not np.isfinite(args.spread_tolerance)
        or args.spread_tolerance < 0.0
    ):
        raise ValueError(
            "spread_tolerance must be finite and non-negative."
        )
    if (
        not np.isfinite(args.near_miss_flatness_far)
        or args.near_miss_flatness_far <= args.flatness_tolerance
    ):
        raise ValueError(
            "near_miss_flatness_far must exceed flatness_tolerance."
        )
    if (
        not np.isfinite(args.near_miss_spread_error_far)
        or args.near_miss_spread_error_far <= args.spread_tolerance
    ):
        raise ValueError(
            "near_miss_spread_error_far must exceed spread_tolerance."
        )


def train(args):
    """Run ori3 with narrowly scoped, exception-safe ori4 hooks."""
    _validate_args(args)
    original_ori3_components = (
        ori3.make_environment,
        ori3.run_episode,
        ori3.aggregate_stochastic_evaluations,
    )
    original_mechanics_train = mechanics.train

    def train_with_ori4_architecture(selected_args):
        objective = (
            "path_integrated_target_deviation"
            if selected_args.quality_combination
            == "path_integrated_target_deviation"
            else "compact_target_combined_potential"
        )
        update = (
            "actor_critic"
            if selected_args.update_method == "actor_critic"
            else "ppo_actor_critic"
        )
        selected_args.architecture_name = (
            f"variable_mode_{objective}_{update}_valid_terminal_near_miss"
        )
        return original_mechanics_train(selected_args)

    ori3.make_environment = make_environment
    ori3.run_episode = run_episode
    ori3.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    mechanics.train = train_with_ori4_architecture
    try:
        output_directory, summary = ori3.train(args)
    finally:
        (
            ori3.make_environment,
            ori3.run_episode,
            ori3.aggregate_stochastic_evaluations,
        ) = original_ori3_components
        mechanics.train = original_mechanics_train

    history_path = Path(output_directory) / "training_history.csv"
    with open(history_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    recent = rows[-min(1000, len(rows)):]
    summary.update({
        "near_miss_bonus_weight": float(args.near_miss_bonus_weight),
        "near_miss_flatness_far": float(args.near_miss_flatness_far),
        "near_miss_spread_error_far": float(
            args.near_miss_spread_error_far
        ),
        "last_1000_mean_near_miss_q": float(np.mean([
            float(row["near_miss_q"]) for row in recent
        ])),
        "last_1000_mean_near_miss_bonus": float(np.mean([
            float(row["near_miss_bonus"]) for row in recent
        ])),
        "last_1000_near_miss_terminal_eligible_rate": float(np.mean([
            float(row["near_miss_terminal_eligible"]) for row in recent
        ])),
        "total_near_miss_bonus": float(sum(
            float(row["near_miss_bonus"]) for row in rows
        )),
    })
    save_near_miss_learning_curve(
        output_directory, rows, args.moving_average_window
    )
    with open(
        Path(output_directory) / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    user_arguments = sys.argv[1:] if argv is None else list(argv)
    return ori3.parse_args([
        "--batch-size", "32",
        "--max-step-width", "0.1",
        "--max-steps", "400",
        "--actor-learning-rate", "1e-3",
        "--critic-learning-rate", "3e-4",
        "--near-miss-bonus-weight", "1.0",
        "--near-miss-flatness-far", "1.0",
        "--near-miss-spread-error-far", "1.0",
        "--output-dir",
        (
            "ori_codex4_h010_valid_terminal_near_miss_"
            "actor_critic_10000_results"
        ),
        *user_arguments,
    ])


if __name__ == "__main__":
    train(parse_args())
