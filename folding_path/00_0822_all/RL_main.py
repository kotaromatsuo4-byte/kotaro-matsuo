"""One-task Codex6-style Actor--Critic training and requested outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import torch

from agent import (
    ActorCriticAgent,
    Transition,
    actor_learning_rate_at_episode,
)
from env import OrigamiEnv
from policy import VariableModeActorCritic


def load_problem(model_directory: Path):
    node = np.atleast_2d(np.loadtxt(
        model_directory / "node_initial.dat", delimiter=","
    ))
    member = np.atleast_2d(np.loadtxt(
        model_directory / "member.dat", delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(
        model_directory / "fix.dat", delimiter=",", dtype=int
    ))
    return node, member, fix


def save_node_path(directory: Path, nodes: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old_path in directory.glob("step*.dat"):
        old_path.unlink()
    for step, node in enumerate(nodes):
        np.savetxt(
            directory / f"step{step}.dat",
            node,
            fmt="%.10f",
        )


def clear_rl_outputs(directory: Path) -> None:
    """Remove only artifacts produced by an earlier RL run of this task."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "learning_curve.png").unlink(missing_ok=True)
    for name in (
        "episode_metrics.csv",
        "update_metrics.csv",
        "concentration_step_traces.csv",
        "concentration_step0.csv",
        "concentration_by_step.png",
        "concentration_step0_by_episode.png",
        "policy_checkpoint_index.csv",
        "summary.json",
    ):
        (directory / name).unlink(missing_ok=True)
    for path in directory.glob("best_episode*"):
        if path.is_dir():
            shutil.rmtree(path)
    checkpoint_directory = directory / "policy_checkpoints"
    if checkpoint_directory.is_dir():
        shutil.rmtree(checkpoint_directory)


def save_policy_checkpoint(
    directory: Path,
    model: VariableModeActorCritic,
    *,
    update: int,
    training_episode: int,
    cumulative_mechanical_steps: int,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"update{update:04d}.pt"
    torch.save({
        "update": int(update),
        "training_episode": int(training_episode),
        "cumulative_mechanical_steps": int(cumulative_mechanical_steps),
        "model_state_dict": model.state_dict(),
    }, directory / filename)
    return {
        "update": int(update),
        "training_episode": int(training_episode),
        "cumulative_mechanical_steps": int(cumulative_mechanical_steps),
        "filename": filename,
    }


def rollout(
    env: OrigamiEnv,
    agent,
    *,
    training: bool,
    deterministic: bool = False,
    collect_concentration: bool = False,
) -> tuple[dict, list[np.ndarray]]:
    env.configure_episode_tolerance(training)
    state = env.reset()
    total_reward = 0.0
    nodes = [env.node.copy()]
    done = False
    concentration_trace = []
    info = {
        **env.compactness_metrics(),
        "strict_success": False,
        "termination_reason": "not_started",
    }

    while not done:
        modes = env.get_modes()
        if modes.shape[1] == 0:
            total_reward -= env.no_mode_penalty
            info = {
                **info,
                **env.compactness_metrics(),
                "strict_success": False,
                "no_modes": True,
                "termination_reason": "no_modes",
            }
            break

        action = agent.get_action(
            state, modes, deterministic=deterministic
        )
        if collect_concentration:
            concentration_trace.append({
                "step": int(env.step_count),
                "concentration": float(action.mixture_concentration),
                "mean_norm": float(action.mixture_mean_norm),
                "standard_deviation_norm": float(
                    action.mixture_standard_deviation_norm
                ),
                "mean_variance": float(action.mean_mixture_variance),
                "mode_count": int(action.mode_count),
            })
        next_state, reward, done, info = env.step(
            action.physical_direction,
            action.requested_width,
            action.mode_coefficients,
        )
        if training:
            agent.add_transition(Transition(
                reward=reward,
                log_probability=action.log_probability,
                value=action.value,
                entropy=action.entropy,
            ))

        total_reward += reward
        state = next_state
        nodes.append(env.node.copy())

    if training:
        agent.finish_episode()

    metrics = env.compactness_metrics()
    summary = {
        "total_reward": float(total_reward),
        "terminal_flatness": float(metrics["flatness_ratio"]),
        "terminal_spread": float(metrics["spread"]),
        "spread_error": float(metrics["spread_error"]),
        "strict_success": int(info.get("strict_success", False)),
        "termination_reason": str(info.get("termination_reason", "running")),
        "steps": int(env.step_count),
        "rejected_crossings": int(sum(
            bool(item.get("rejected_crossing", False))
            for item in env.reward_trace
        )),
    }
    if collect_concentration:
        summary["_concentration_trace"] = concentration_trace
    return summary, nodes


def moving_average(values: list[float], window: int):
    array = np.asarray(values, dtype=float)
    if array.size < window:
        return np.empty(0), np.empty(0)
    average = np.convolve(
        array, np.ones(window, dtype=float) / window, mode="valid"
    )
    return np.arange(window, array.size + 1), average


def save_learning_curve(
    output_directory: Path,
    rewards: list[float],
    terminal_flatness: list[float],
    window: int,
) -> None:
    episodes = np.arange(1, len(rewards) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for axis, values, label in (
        (axes[0], rewards, "Episode reward"),
        (axes[1], terminal_flatness, "Terminal flatness F"),
    ):
        axis.plot(
            episodes, values, color="0.72", linewidth=0.6, label="raw"
        )
        x_average, average = moving_average(values, window)
        if average.size:
            axis.plot(
                x_average,
                average,
                color="black",
                linewidth=1.8,
                label=f"moving average ({window})",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Episode")
    figure.tight_layout()
    figure.savefig(output_directory / "learning_curve.png", dpi=180)
    plt.close(figure)


def save_concentration_figures(
    output_directory: Path,
    task_name: str,
    seed: int,
    step_trace_rows: list[dict],
    step0_rows: list[dict],
    first_success_episode: int | None,
) -> None:
    """Save the two requested latent-policy concentration figures."""

    if not step_trace_rows or not step0_rows:
        return

    completed_episode = int(step0_rows[-1]["episode"])
    normalise = colors.Normalize(
        vmin=1,
        vmax=max(2, completed_episode),
    )
    colour_map = plt.get_cmap("viridis")
    scalar_map = cm.ScalarMappable(norm=normalise, cmap=colour_map)

    rows_by_episode: dict[int, list[dict]] = {}
    for row in step_trace_rows:
        rows_by_episode.setdefault(int(row["episode"]), []).append(row)

    figure, axis = plt.subplots(figsize=(9.2, 5.7))
    for episode in sorted(rows_by_episode):
        rows = rows_by_episode[episode]
        steps = [int(row["step"]) for row in rows]
        values = [float(row["concentration"]) for row in rows]
        axis.plot(
            steps,
            values,
            color=colour_map(normalise(episode)),
            linewidth=0.65,
            alpha=0.42,
            rasterized=True,
        )

    if first_success_episode is not None:
        success_rows = rows_by_episode[int(first_success_episode)]
        axis.plot(
            [int(row["step"]) for row in success_rows],
            [float(row["concentration"]) for row in success_rows],
            color="black",
            linewidth=2.0,
            alpha=0.95,
            label=f"first strict success (E={first_success_episode})",
        )
        axis.legend(loc="best")

    axis.set_xlabel("Step")
    axis.set_ylabel(
        r"Relative concentration $C=\|\mu\|_2/\sqrt{\sum_m\sigma_m^2}$"
    )
    axis.set_title(
        f"{task_name}, seed {seed}: initial/post-update and success traces"
    )
    axis.grid(alpha=0.22)
    colour_bar = figure.colorbar(scalar_map, ax=axis, pad=0.02)
    colour_bar.set_label("Episode")
    figure.tight_layout()
    figure.savefig(
        output_directory / "concentration_by_step.png", dpi=220
    )
    plt.close(figure)

    episodes = np.asarray(
        [int(row["episode"]) for row in step0_rows], dtype=int
    )
    concentrations = np.asarray(
        [float(row["concentration"]) for row in step0_rows], dtype=float
    )
    update_starts = np.asarray([
        int(row["is_initial_baseline"]) != 0
        or int(row["is_post_update_episode"]) != 0
        for row in step0_rows
    ])

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.plot(
        episodes,
        concentrations,
        color="0.35",
        linewidth=1.0,
        drawstyle="steps-post",
    )
    axis.scatter(
        episodes[update_starts],
        concentrations[update_starts],
        c=episodes[update_starts],
        cmap=colour_map,
        norm=normalise,
        s=17,
        linewidths=0.0,
        zorder=3,
        label="first episode under each policy",
    )
    if first_success_episode is not None:
        axis.axvline(
            first_success_episode,
            color="black",
            linewidth=1.4,
            linestyle="--",
            label=f"first strict success (E={first_success_episode})",
        )
    axis.set_xlabel("Episode")
    axis.set_ylabel(
        r"Step-0 concentration $C=\|\mu\|_2/\sqrt{\sum_m\sigma_m^2}$"
    )
    axis.set_title(f"{task_name}, seed {seed}: fixed initial state")
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(
        output_directory / "concentration_step0_by_episode.png",
        dpi=220,
    )
    plt.close(figure)


def save_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def maximum_consecutive_failures(successes: list[int]) -> int:
    longest = current = 0
    for success in successes:
        if success:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def strict_success_retention_statistics(successes: list[int]) -> dict:
    """Compute retention thresholds with integer window counts."""

    values = np.asarray(successes, dtype=np.int64)
    episodes = int(values.size)
    rolling_window = 200
    long_tail = min(1000, episodes)

    def rolling_counts(array: np.ndarray) -> np.ndarray:
        if array.size < rolling_window:
            return np.empty(0, dtype=np.int64)
        cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(array)))
        return cumulative[rolling_window:] - cumulative[:-rolling_window]

    counts = rolling_counts(values)
    confirmation_episode = None
    minimum_after_confirmation = None
    dropped_below_80pct = False
    observed_for_1000 = False
    if counts.size:
        confirmed = np.flatnonzero(10 * counts >= 9 * rolling_window)
        if confirmed.size:
            confirmation_index = int(confirmed[0])
            confirmation_episode = confirmation_index + rolling_window
            after_confirmation = counts[confirmation_index:]
            minimum_after_confirmation = (
                int(np.min(after_confirmation)) / rolling_window
            )
            dropped_below_80pct = bool(
                np.any(5 * after_confirmation < 4 * rolling_window)
            )
            observed_for_1000 = bool(
                episodes - confirmation_episode >= 1000
            )

    final_values = values[-long_tail:]
    final_counts = rolling_counts(final_values)
    final_minimum = (
        int(np.min(final_counts)) / rolling_window
        if final_counts.size else None
    )
    final_window_stable = bool(
        long_tail > 0
        and 20 * int(np.sum(final_values)) >= 19 * long_tail
        and final_counts.size
        and np.all(10 * final_counts >= 9 * rolling_window)
    )
    retention_confirmed = bool(
        confirmation_episode is not None
        and observed_for_1000
        and not dropped_below_80pct
        and final_window_stable
    )
    tail = min(200, episodes)
    return {
        "last_200_strict_success_rate": (
            float(np.sum(values[-tail:]) / tail) if tail else 0.0
        ),
        "last_1000_strict_success_rate": (
            float(np.sum(final_values) / long_tail) if long_tail else 0.0
        ),
        "rolling_200_confirmation_episode_90pct": confirmation_episode,
        "minimum_rolling_200_after_confirmation": minimum_after_confirmation,
        "dropped_below_80pct_after_confirmation": dropped_below_80pct,
        "confirmation_observed_for_1000_episodes": observed_for_1000,
        "last_1000_minimum_rolling_200_strict_rate": final_minimum,
        "final_window_stable": final_window_stable,
        "retention_confirmed": retention_confirmed,
    }


def run_rl(config: dict) -> dict:
    torch.set_num_threads(1)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    node, member, fix = load_problem(Path(config["model_directory"]))
    env = OrigamiEnv(config, data=(node, member, fix))
    model = VariableModeActorCritic(
        coordinate_size=env.coordinate_size,
        hidden_size=int(config["hidden_size"]),
        variance_bias=float(config["variance_bias"]),
        width_mean_limit=float(config["width_mean_limit"]),
    )
    agent = ActorCriticAgent(
        model=model,
        max_step_width=float(config["max_step_width"]),
        gamma=float(config["gamma"]),
        actor_learning_rate=float(config["actor_learning_rate"]),
        critic_learning_rate=float(config["critic_learning_rate"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        value_coefficient=float(config["value_coefficient"]),
        actor_gradient_clip=float(config["actor_gradient_clip"]),
        critic_gradient_clip=float(config["critic_gradient_clip"]),
        actor_reference_horizon=config.get("actor_reference_horizon"),
    )

    output_directory = Path(config.get(
        "rl_output_directory",
        Path(config["result_directory"]) / "rl",
    ))
    save_outputs = bool(config.get("save_outputs", True))
    stop_on_first_success = bool(
        config.get("stop_on_first_strict_success", False)
    )
    print_progress = bool(config.get("print_progress", True))
    print_result = bool(config.get("print_result", True))
    save_diagnostics = bool(config.get("save_diagnostics", False))
    save_concentration = bool(
        config.get("save_concentration_diagnostics", False)
    )
    save_policy_checkpoints = bool(
        config.get("save_policy_checkpoints", False)
    )
    adaptive_after_success = bool(
        config.get("adaptive_after_first_success", False)
    )
    initial_batch_size = int(config["batch_size"])
    current_batch_size = initial_batch_size
    post_success_batch_size = int(config.get(
        "post_success_batch_size", initial_batch_size
    ))
    initial_actor_learning_rate = float(config["actor_learning_rate"])
    post_success_actor_lr_factor = float(config.get(
        "post_success_actor_lr_factor", 1.0
    ))
    post_success_actor_learning_rate = (
        initial_actor_learning_rate * post_success_actor_lr_factor
    )
    training_phase = "discovery"
    adaptation_episode = None
    if save_outputs:
        clear_rl_outputs(output_directory)
    rewards: list[float] = []
    terminal_flatness: list[float] = []
    strict_success: list[int] = []
    episode_steps: list[int] = []
    termination_reasons: list[str] = []
    episode_rows: list[dict] = []
    update_rows: list[dict] = []
    concentration_step_trace_rows: list[dict] = []
    concentration_step0_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    best_reward = -np.inf
    first_success_episode = None
    first_success_episode_steps = None
    first_success_mechanical_steps = None
    optimizer_updates_before_first_success = None
    optimizer_updates_completed = 0
    cumulative_mechanical_steps = 0
    episodes_completed = 0
    last_summary = None
    last_update_episode = 0

    if save_outputs and save_policy_checkpoints:
        checkpoint_rows.append(save_policy_checkpoint(
            output_directory / "policy_checkpoints",
            model,
            update=0,
            training_episode=0,
            cumulative_mechanical_steps=0,
        ))

    episodes_requested = int(config["episodes"])
    for episode in range(1, episodes_requested + 1):
        rollout_phase = training_phase
        rollout_batch_size = current_batch_size
        rollout_actor_learning_rate = agent.actor_learning_rate
        rollout_update_index = optimizer_updates_completed
        is_initial_policy_episode = episode == 1
        is_post_update_episode = bool(
            optimizer_updates_completed > 0
            and episode == last_update_episode + 1
        )
        model.train()
        summary, nodes = rollout(
            env,
            agent,
            training=True,
            collect_concentration=save_concentration,
        )
        episode_concentration_trace = summary.pop(
            "_concentration_trace", []
        )
        episodes_completed = episode
        last_summary = summary
        cumulative_mechanical_steps += int(summary["steps"])
        rewards.append(summary["total_reward"])
        terminal_flatness.append(summary["terminal_flatness"])
        strict_success.append(summary["strict_success"])
        episode_steps.append(int(summary["steps"]))
        termination_reasons.append(summary["termination_reason"])

        if summary["total_reward"] > best_reward:
            best_reward = summary["total_reward"]
            if save_outputs:
                save_node_path(
                    output_directory / f"best_episode{episode:05d}", nodes
                )

        first_success_now = bool(
            summary["strict_success"] and first_success_episode is None
        )
        if first_success_now:
            first_success_episode = episode
            first_success_episode_steps = int(summary["steps"])
            first_success_mechanical_steps = cumulative_mechanical_steps
            optimizer_updates_before_first_success = (
                optimizer_updates_completed
            )

        if save_concentration and episode_concentration_trace:
            common = {
                "task": config["task_name"],
                "seed": int(config["seed"]),
                "episode": int(episode),
                "optimizer_updates_before_rollout": int(
                    rollout_update_index
                ),
                "is_initial_baseline": int(is_initial_policy_episode),
                "is_post_update_episode": int(is_post_update_episode),
                "is_first_strict_success": int(first_success_now),
            }
            concentration_step0_rows.append({
                **common,
                **episode_concentration_trace[0],
                "strict_success": int(summary["strict_success"]),
            })
            if (
                is_initial_policy_episode
                or is_post_update_episode
                or first_success_now
            ):
                concentration_step_trace_rows.extend(
                    {**common, **row}
                    for row in episode_concentration_trace
                )

        stop_now = bool(stop_on_first_success and summary["strict_success"])
        if first_success_now and adaptive_after_success and not stop_now:
            adaptation_episode = episode
            training_phase = "stabilization"
            current_batch_size = post_success_batch_size
            agent.set_actor_learning_rate(
                post_success_actor_learning_rate
            )

        episode_rows.append({
            "episode": int(episode),
            "reward": float(summary["total_reward"]),
            "terminal_flatness": float(summary["terminal_flatness"]),
            "terminal_spread": float(summary["terminal_spread"]),
            "spread_error": float(summary["spread_error"]),
            "strict_success": int(summary["strict_success"]),
            "termination_reason": summary["termination_reason"],
            "steps": int(summary["steps"]),
            "cumulative_mechanical_steps": int(cumulative_mechanical_steps),
            "rejected_crossings": int(summary["rejected_crossings"]),
            "rollout_phase": rollout_phase,
            "phase_after_episode": training_phase,
            "actor_lr_used": float(rollout_actor_learning_rate),
            "actor_lr_after_episode": float(agent.actor_learning_rate),
            "batch_target_used": int(rollout_batch_size),
            "batch_target_after_episode": int(current_batch_size),
            "buffer_episode_count": int(len(agent.finished_episodes)),
            "optimizer_updates_before_episode_update": int(
                optimizer_updates_completed
            ),
            "adaptation_triggered": int(
                first_success_now and adaptive_after_success and not stop_now
            ),
        })

        if stop_now:
            break

        if (
            len(agent.finished_episodes) >= current_batch_size
            or episode == episodes_requested
        ):
            if not adaptive_after_success:
                learning_rate = actor_learning_rate_at_episode(
                    config["actor_learning_rate"],
                    episode,
                    config["actor_lr_milestones"],
                    config["actor_lr_factors"],
                )
                agent.set_actor_learning_rate(learning_rate)
            update_result = agent.update()
            if update_result is not None:
                optimizer_updates_completed += 1
                update_rows.append({
                    "update": int(optimizer_updates_completed),
                    "start_episode": int(last_update_episode + 1),
                    "end_episode": int(episode),
                    "phase": training_phase,
                    "target_batch_size": int(current_batch_size),
                    **update_result,
                })
                last_update_episode = episode
                if save_outputs and save_policy_checkpoints:
                    checkpoint_rows.append(save_policy_checkpoint(
                        output_directory / "policy_checkpoints",
                        model,
                        update=optimizer_updates_completed,
                        training_episode=episode,
                        cumulative_mechanical_steps=(
                            cumulative_mechanical_steps
                        ),
                    ))

        if print_progress and (
            episode == 1 or episode % int(config["print_every"]) == 0
        ):
            recent = min(int(config["moving_average_window"]), episode)
            print(
                f"RL {config['task_name']} episode {episode}: "
                f"reward={np.mean(rewards[-recent:]):.4f}, "
                f"F={np.mean(terminal_flatness[-recent:]):.6g}, "
                f"strict={np.mean(strict_success[-recent:]):.2%}, "
                f"phase={training_phase}, "
                f"batch={current_batch_size}, "
                f"actor_lr={agent.actor_learning_rate:.3g}",
                flush=True,
            )

    if save_outputs:
        save_learning_curve(
            output_directory,
            rewards,
            terminal_flatness,
            int(config["moving_average_window"]),
        )
        if save_concentration:
            save_csv_rows(
                output_directory / "concentration_step_traces.csv",
                concentration_step_trace_rows,
            )
            save_csv_rows(
                output_directory / "concentration_step0.csv",
                concentration_step0_rows,
            )
            save_concentration_figures(
                output_directory=output_directory,
                task_name=str(config["task_name"]),
                seed=int(config["seed"]),
                step_trace_rows=concentration_step_trace_rows,
                step0_rows=concentration_step0_rows,
                first_success_episode=first_success_episode,
            )
    tail = min(200, episodes_completed)
    long_tail = min(1000, episodes_completed)
    block_size = 1000
    strict_success_rate_by_1000 = []
    mean_reward_by_1000 = []
    mean_terminal_flatness_by_1000 = []
    mean_steps_by_1000 = []
    termination_counts_by_1000 = []
    for start in range(0, episodes_completed, block_size):
        stop = min(start + block_size, episodes_completed)
        strict_success_rate_by_1000.append(
            float(np.mean(strict_success[start:stop]))
        )
        mean_reward_by_1000.append(float(np.mean(rewards[start:stop])))
        mean_terminal_flatness_by_1000.append(
            float(np.mean(terminal_flatness[start:stop]))
        )
        mean_steps_by_1000.append(
            float(np.mean(episode_steps[start:stop]))
        )
        reasons = termination_reasons[start:stop]
        termination_counts_by_1000.append({
            reason: int(reasons.count(reason))
            for reason in sorted(set(reasons))
        })

    retention = strict_success_retention_statistics(strict_success)
    result = {
        "task": config["task_name"],
        "seed": int(config["seed"]),
        "episodes": episodes_completed,
        "episodes_requested": episodes_requested,
        "stopped_on_first_strict_success": bool(
            stop_on_first_success and first_success_episode is not None
        ),
        "first_strict_success_episode": first_success_episode,
        "first_strict_success_episode_steps": first_success_episode_steps,
        "first_strict_success_mechanical_steps": (
            first_success_mechanical_steps
        ),
        "cumulative_mechanical_steps": cumulative_mechanical_steps,
        "optimizer_updates_before_first_success": (
            optimizer_updates_before_first_success
        ),
        "optimizer_updates_completed": optimizer_updates_completed,
        "actor_normalization": (
            "episode_sum_fixed_horizon"
            if config.get("actor_reference_horizon") is not None
            else "transition_mean"
        ),
        "actor_reference_horizon": config.get(
            "actor_reference_horizon"
        ),
        "adaptive_after_first_success": adaptive_after_success,
        "adaptation_episode": adaptation_episode,
        "initial_batch_size": initial_batch_size,
        "final_batch_size": current_batch_size,
        "initial_actor_learning_rate": initial_actor_learning_rate,
        "final_actor_learning_rate": agent.actor_learning_rate,
        "post_success_actor_lr_factor": post_success_actor_lr_factor,
        "last_200_strict_success_rate": retention[
            "last_200_strict_success_rate"
        ],
        "last_200_mean_reward": float(np.mean(rewards[-tail:])),
        "last_200_mean_terminal_flatness": float(
            np.mean(terminal_flatness[-tail:])
        ),
        "last_1000_strict_success_rate": retention[
            "last_1000_strict_success_rate"
        ],
        "last_1000_mean_reward": float(np.mean(rewards[-long_tail:])),
        "last_1000_mean_terminal_flatness": float(
            np.mean(terminal_flatness[-long_tail:])
        ),
        "strict_success_rate_by_1000": strict_success_rate_by_1000,
        "mean_reward_by_1000": mean_reward_by_1000,
        "mean_terminal_flatness_by_1000": (
            mean_terminal_flatness_by_1000
        ),
        "mean_steps_by_1000": mean_steps_by_1000,
        "termination_counts_by_1000": termination_counts_by_1000,
        "rolling_200_confirmation_episode_90pct": retention[
            "rolling_200_confirmation_episode_90pct"
        ],
        "minimum_rolling_200_after_confirmation": (
            retention["minimum_rolling_200_after_confirmation"]
        ),
        "dropped_below_80pct_after_confirmation": (
            retention["dropped_below_80pct_after_confirmation"]
        ),
        "confirmation_observed_for_1000_episodes": (
            retention["confirmation_observed_for_1000_episodes"]
        ),
        "last_1000_minimum_rolling_200_strict_rate": (
            retention["last_1000_minimum_rolling_200_strict_rate"]
        ),
        "final_window_stable": retention["final_window_stable"],
        "retention_confirmed": retention["retention_confirmed"],
        "maximum_consecutive_failures": maximum_consecutive_failures(
            strict_success
        ),
        "last_terminal_flatness": float(
            last_summary["terminal_flatness"]
        ),
        "last_terminal_spread": float(last_summary["terminal_spread"]),
        "last_spread_error": float(last_summary["spread_error"]),
        "best_reward": float(best_reward),
        "concentration_diagnostics_saved": bool(
            save_outputs and save_concentration
        ),
        "concentration_policy_trace_count": int(sum(
            bool(row["is_initial_baseline"])
            or bool(row["is_post_update_episode"])
            for row in concentration_step0_rows
        )),
        "policy_checkpoints_saved": int(len(checkpoint_rows)),
        "output_directory": str(output_directory) if save_outputs else None,
    }
    if save_outputs and save_policy_checkpoints:
        save_csv_rows(
            output_directory / "policy_checkpoint_index.csv",
            checkpoint_rows,
        )
    if save_outputs and save_diagnostics:
        save_csv_rows(
            output_directory / "episode_metrics.csv", episode_rows
        )
        save_csv_rows(
            output_directory / "update_metrics.csv", update_rows
        )
        with (output_directory / "summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
    if print_result:
        print(result, flush=True)
    return result


if __name__ == "__main__":
    from main import main

    main()
