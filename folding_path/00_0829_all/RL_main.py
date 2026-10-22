"""One-task Codex6-style Actor--Critic training and requested outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import (
    ActorCriticAgent,
    Transition,
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


def rollout(
    env: OrigamiEnv,
    agent,
    *,
    training: bool,
    deterministic: bool = False,
) -> tuple[dict, list[np.ndarray]]:
    env.configure_episode_tolerance(training)
    state = env.reset()
    total_reward = 0.0
    nodes = [env.node.copy()]
    done = False
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
        "rejected_crossings": sum(bool(item.get("rejected_crossing")) for item in env.reward_trace),
    }
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


def make_agent(config: dict):
    """Environment -> variable-mode policy -> Monte-Carlo actor-critic."""
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
        actor_reference_horizon=config["actor_reference_horizon"],
    )
    return env, model, agent


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(directory, model, update, episode, steps):
    """Save parameters without sampling or advancing the training RNG."""
    filename = f"update{update:05d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "update": update,
        "training_episode": episode,
        "cumulative_mechanical_steps": steps,
    }, directory / filename)
    return {
        "update": update, "training_episode": episode,
        "cumulative_mechanical_steps": steps, "filename": filename,
    }


def run_rl(config: dict) -> dict:
    torch.set_num_threads(1)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    episodes = min(int(config["episodes"]), 10_000)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = dict(config, episodes=episodes)
    env, model, agent = make_agent(config)

    # Each invocation has its own directory so earlier research results survive.
    output_directory = Path(config.get(
        "rl_output_directory",
        Path(config["result_directory"]) / "rl" /
        f"seed{config['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}",
    ))
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_directory = output_directory / "policy_checkpoints"
    checkpoint_directory.mkdir()
    (output_directory / "config.json").write_text(
        json.dumps(config, default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Output: {output_directory}", flush=True)

    rewards, flatness, checkpoint_rows = [], [], []
    updates = total_steps = 0
    best_reward = -np.inf
    first_success = None
    checkpoint_rows.append(save_checkpoint(checkpoint_directory, model, 0, 0, 0))
    write_rows(output_directory / "policy_checkpoint_index.csv", checkpoint_rows)
    with (output_directory / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        writer = None
        for episode in range(1, episodes + 1):
            model.train()
            summary, nodes = rollout(env, agent, training=True)
            total_steps += summary["steps"]
            rewards.append(summary["total_reward"])
            flatness.append(summary["terminal_flatness"])
            row = {
                "episode": episode, "optimizer_updates_before_episode": updates,
                "cumulative_mechanical_steps": total_steps, **summary,
            }
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()

            if summary["total_reward"] > best_reward:
                best_reward = summary["total_reward"]
                save_node_path(output_directory / f"best_episode{episode:05d}", nodes)

            if summary["strict_success"]:
                first_success = episode
                save_node_path(output_directory / "first_strict_success", nodes)
                # The successful episode was generated by update 'updates'.
                # Do not train on it or flush its incomplete batch.
                break

            if len(agent.finished_episodes) >= int(config["batch_size"]) or episode == episodes:
                # Final discovery experiments use a constant pre-success LR.
                agent.update()
                updates += 1
                checkpoint_rows.append(save_checkpoint(
                    checkpoint_directory, model, updates, episode, total_steps,
                ))
                write_rows(output_directory / "policy_checkpoint_index.csv", checkpoint_rows)

            if episode == 1 or episode % int(config["print_every"]) == 0:
                print(
                    f"{config['task_name']} E={episode} updates={updates} "
                    f"steps={total_steps} F={flatness[-1]:.6g}", flush=True,
                )

    # Release the unconsumed batch before independent evaluation.
    agent.finished_episodes.clear()
    agent.current_episode.clear()
    save_learning_curve(
        output_directory, rewards, flatness,
        int(config["moving_average_window"]),
    )
    result = {
        "task": config["task_name"], "seed": config["seed"],
        "episodes": episode, "success": first_success is not None,
        "first_strict_success_episode": first_success,
        "first_strict_success_episode_steps": summary["steps"] if first_success else None,
        "first_strict_success_mechanical_steps": total_steps if first_success else None,
        "optimizer_updates_before_first_success": updates if first_success else None,
        "optimizer_updates_completed": updates,
        "cumulative_mechanical_steps": total_steps,
        "output_directory": str(output_directory),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(result, flush=True)
    if first_success is not None and config.get("evaluate_after_success", True):
        del env, model, agent
        evaluate_checkpoints(output_directory)
    return result


def wilson_interval(count: int, total: int) -> tuple[float, float]:
    p = count / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return (
        0.0 if count == 0 else max(0.0, centre - radius),
        1.0 if count == total else min(1.0, centre + radius),
    )


def evaluate_checkpoints(output_directory: Path) -> list[dict]:
    """100 independent stochastic rollouts for initial and each updated policy.

    Seeds are reused between checkpoints for paired comparisons. A replicate
    reaches near-miss if ANY visited post-action state satisfies both bounds.
    Q* is max_t(qF_t*qD_t), evaluated at the same state t.
    """
    output_directory = Path(output_directory)
    config = json.loads((output_directory / "config.json").read_text(encoding="utf-8"))
    result = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    if not result["success"]:
        print("No first strict success: checkpoint evaluation skipped.", flush=True)
        return []
    with (output_directory / "policy_checkpoint_index.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        checkpoints = list(csv.DictReader(file))
    count = int(config["evaluation_rollouts"])
    if count < 1:
        raise ValueError("evaluation_rollouts must be positive")
    evaluation_directory = output_directory / "checkpoint_evaluation"
    evaluation_directory.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    env, model, agent = make_agent(config)
    model.eval()
    summaries = []
    for checkpoint in checkpoints:
        update = int(checkpoint["update"])
        raw_path = evaluation_directory / f"update{update:05d}.csv"
        # Completed checkpoints can be reused after interruption.
        if raw_path.exists():
            with raw_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if len(rows) != count:
                raise ValueError(f"Incomplete evaluation: {raw_path}")
        else:
            saved = torch.load(
                output_directory / "policy_checkpoints" / checkpoint["filename"],
                map_location="cpu", weights_only=True,
            )
            model.load_state_dict(saved["model_state_dict"])
            rows = []
            for replicate in range(count):
                seed = int(config["evaluation_seed_base"]) + replicate
                np.random.seed(seed)
                torch.manual_seed(seed)
                with torch.inference_mode():
                    summary, _ = rollout(env, agent, training=False)
                near = False
                best_quality = 0.0
                for state in env.reward_trace:
                    _, _, quality = env.near_miss_quality(
                        state["flatness_ratio"], state["spread_error"],
                    )
                    best_quality = max(best_quality, quality)
                    near |= (
                        state["flatness_ratio"] <= 0.1
                        and state["spread_error"] <= env.spread_tolerance
                    )
                rows.append({
                    "update": update, "replicate": replicate + 1,
                    "evaluation_seed": seed, "reach_near_miss": int(near),
                    "best_quality": best_quality, "steps": summary["steps"],
                    "strict_success": summary["strict_success"],
                })
                if (replicate + 1) % 10 == 0:
                    print(f"Evaluate update {update}: {replicate + 1}/{count}", flush=True)
            write_rows(raw_path, rows)

        near_count = sum(int(row["reach_near_miss"]) for row in rows)
        low, high = wilson_interval(near_count, count)
        q25, median, q75 = np.quantile(
            [float(row["best_quality"]) for row in rows], [0.25, 0.5, 0.75],
        )
        summaries.append({
            "update": update,
            "training_episode": int(checkpoint["training_episode"]),
            "training_steps": int(checkpoint["cumulative_mechanical_steps"]),
            "rollouts": count, "near_miss_rate": near_count / count,
            "near_miss_ci_low": low, "near_miss_ci_high": high,
            "best_quality_q25": q25, "best_quality_median": median,
            "best_quality_q75": q75,
        })
        write_rows(evaluation_directory / "checkpoint_summary.csv", summaries)
        plot_checkpoint_evaluation(evaluation_directory, summaries)
    return summaries


def plot_checkpoint_evaluation(directory: Path, rows: list[dict]) -> None:
    """Two black research figures; reach probability and Q* median/IQR."""
    with plt.rc_context({
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
    }):
        updates = [row["update"] for row in rows]
        for name, middle, lower, upper, ylabel in (
            ("near_miss_reach_probability", "near_miss_rate",
             "near_miss_ci_low", "near_miss_ci_high",
             "Near-miss-region reach probability"),
            ("best_quality", "best_quality_median",
             "best_quality_q25", "best_quality_q75", r"$Q^*$"),
        ):
            figure, axis = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
            axis.fill_between(
                updates, [row[lower] for row in rows], [row[upper] for row in rows],
                color="black", alpha=0.15, linewidth=0,
            )
            axis.plot(updates, [row[middle] for row in rows], color="black",
                      marker="o" if len(rows) == 1 else None, linewidth=1.25)
            axis.set(xlabel="Optimizer update", ylabel=ylabel, ylim=(0, 1))
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            figure.savefig(directory / f"{name}.png", dpi=220)
            plt.close(figure)


if __name__ == "__main__":
    from main import main
    main()
