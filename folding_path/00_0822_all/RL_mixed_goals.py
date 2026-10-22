"""Balanced D2/D3 goal-conditioned training and frozen-policy evaluation.

python RL_mixed_goals.py --episodes 128 --eval-every 128 --seed 9201
python RL_mixed_goals.py --evaluate-only result/experiments/<run>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

from main import task_config  # Set single-thread BLAS variables before torch.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import ActorCriticAgent
from env import OrigamiEnv
from policy import VariableModeActorCritic
from Random_main import UniformRandomAgent
from RL_main import rollout, save_node_path

BASE = Path(__file__).resolve().parent
TASKS = ("five_d2", "five_d3")


class GoalConditionedEnv(OrigamiEnv):
    """Append the dimensionless target spread to the coordinate observation.

Each goal has its own environment, so initial spread references and reward
potentials are computed for that goal rather than mutating an existing env.
"""

    def get_state(self):
        coordinates = super().get_state()
        goal = self.target_spread if self.config.get("condition_goal", True) else 0.0
        return np.concatenate((coordinates, np.array([goal], dtype=np.float32)))


def configurations(design):
    result = {}
    for task in design.get("tasks", TASKS):
        config = task_config(task, design["episodes"])
        config.update({
            "actor_learning_rate": design["actor_lr"],
            "critic_learning_rate": design["critic_lr"],
            "actor_reference_horizon": 400,
            "spectral_weight": 0.05,
            "condition_goal": design["condition_goal"],
            "critic_replay_steps": design.get("critic_replay_steps", 0),
        })
        result[task] = config
    return result


def make_model_agent(config):
    env = GoalConditionedEnv(config)
    model = VariableModeActorCritic(
        coordinate_size=env.coordinate_size, goal_size=1,
        hidden_size=config["hidden_size"], variance_bias=config["variance_bias"],
        width_mean_limit=config["width_mean_limit"],
    )
    agent = ActorCriticAgent(
        model, max_step_width=config["max_step_width"], gamma=config["gamma"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        entropy_coefficient=config["entropy_coefficient"],
        value_coefficient=config["value_coefficient"],
        actor_gradient_clip=config["actor_gradient_clip"],
        critic_gradient_clip=config["critic_gradient_clip"],
        actor_reference_horizon=config["actor_reference_horizon"],
    )
    if config.get("critic_replay_steps", 0):
        agent.replay_observations = []
        original_action = agent.get_action
        def capture(state, modes, **kwargs):
            if model.training:
                agent.replay_observations.append((state.copy(), modes.T.copy()))
            return original_action(state, modes, **kwargs)
        agent.get_action = capture
    return model, agent


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # OneDrive can briefly lock the destination while syncing a completed CSV.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def exploration_controls(design, rows):
    """Training-history-only controller, shared rule for every goal."""
    history = []
    if design.get("pretraining_run"):
        history = read_csv(Path(design["pretraining_run"]) / "episode_metrics.csv")
    history = history + rows
    result = {}
    for task in design.get("tasks", TASKS):
        outcomes = [int(r["strict_success"]) for r in history if r["task"] == task][-32:]
        rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
        strength = max(0.0, 1.0 - rate / 0.2) if design.get("adaptive_exploration") else 0.0
        result[task] = (rate, strength)
    return result


def save_checkpoint(directory, model, agent, schedule_rng, episode, update, steps):
    name = f"update{update:05d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "schedule_rng": schedule_rng.bit_generator.state,
        "episode": episode, "update": update, "steps": steps,
    }, directory / "checkpoints" / name)
    return {"episode": episode, "update": update, "steps": steps, "filename": name}


def train(design, resume_directory=None):
    torch.set_num_threads(1)
    torch.manual_seed(design["seed"])
    np.random.seed(design["seed"])
    schedule_rng = np.random.default_rng(design["seed"] + 1_000_000)
    critic_generator = torch.Generator().manual_seed(design["seed"] + 2_000_000)
    tasks = tuple(design.get("tasks", TASKS))
    run_label = "mixed_goals" if len(tasks) > 1 else f"single_goal_{tasks[0]}"
    directory = BASE / "result" / "experiments" / (
        f"{run_label}_seed{design['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    directory.mkdir(parents=True)
    (directory / "checkpoints").mkdir()
    configs = design.get("configs") or configurations(design)
    configs = {task: dict(config, episodes=design["episodes"]) for task, config in configs.items()}
    design = dict(design, configs=configs, source_sha256={
        name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        for name in ("RL_mixed_goals.py", "main.py", "env.py", "mechanics.py",
                     "policy.py", "agent.py", "RL_main.py", "Random_main.py",
                     "mixed_diagnostics.py", "critic_replay.py")
    })
    (directory / "design.json").write_text(json.dumps(design, default=str, indent=2))
    environments = {task: GoalConditionedEnv(c) for task, c in configs.items()}
    model, agent = make_model_agent(configs[tasks[0]])
    if design.get("warm_start_checkpoint"):
        saved = torch.load(design["warm_start_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        agent.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        schedule_rng.bit_generator.state = saved["schedule_rng"]
    steps = update = 0
    rows, update_rows = [], []
    checkpoints = [save_checkpoint(directory, model, agent, schedule_rng, 0, 0, 0)]
    first_hits = {}
    diagnostic_rows = []
    best = {task: -np.inf for task in tasks}
    start_episode = 1
    if resume_directory is not None:
        source = Path(resume_directory).resolve()
        checkpoints = read_csv(source / "checkpoint_index.csv")
        saved = torch.load(source / "checkpoints" / checkpoints[-1]["filename"],
                           map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        agent.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        schedule_rng.bit_generator.state = saved["schedule_rng"]
        steps, update = saved["steps"], saved["update"]
        start_episode = saved["episode"] + 1
        rows = read_csv(source / "episode_metrics.csv")
        integer_fields = {"episode", "updates_before_episode", "cumulative_steps",
                          "strict_success", "steps", "rejected_crossings"}
        for row in rows:
            for key in row:
                if key not in {"task", "termination_reason"}:
                    row[key] = int(row[key]) if key in integer_fields else float(row[key])
        update_rows = read_csv(source / "updates.csv")
        if (source / "advantage_diagnostics.csv").exists():
            diagnostic_rows = read_csv(source / "advantage_diagnostics.csv")
        first_hits = json.loads((source / "training_summary.json").read_text())["first_hits"]
        best = {task: max(r["total_reward"] for r in rows if r["task"] == task) for task in tasks}
        for name in ("checkpoints", "evaluation", "exploration_probe", *tasks):
            if (source / name).exists():
                shutil.copytree(source / name, directory / name, dirs_exist_ok=True)
    batch_size = design["batch_size"]
    budget = design.get("d2_step_budget")
    task_steps = {task: sum(r["steps"] for r in rows if r["task"] == task) for task in tasks}
    print(f"RUN_DIRECTORY={directory}", flush=True)
    for batch_start in range(start_episode, design["episodes"] + 1, batch_size):
        if budget is not None and task_steps.get("five_d2", 0) >= budget:
            break
        goals = [task for task in tasks for _ in range(batch_size // len(tasks))]
        schedule_rng.shuffle(goals)
        controls = exploration_controls(design, rows) if design.get("adaptive_exploration") else None
        batch_metadata = []
        for offset, task in enumerate(goals):
            episode = batch_start + offset
            model.train()
            if controls is not None:
                agent.exploration_strength = controls[task][1]
            summary, nodes = rollout(environments[task], agent, training=True)
            if controls is not None:
                summary.update(prior_success_rate=controls[task][0],
                               exploration_strength=controls[task][1])
            if design.get("diagnostics", False):
                from mixed_diagnostics import episode_metadata
                batch_metadata.append(episode_metadata(environments[task], summary, episode, update))
            steps += summary["steps"]
            task_steps[task] += summary["steps"]
            rows.append({"episode": episode, "task": task,
                         "target_spread": configs[task]["target_spread"],
                         "updates_before_episode": update,
                         "cumulative_steps": steps, **summary})
            if summary["total_reward"] > best[task]:
                best[task] = summary["total_reward"]
                save_node_path(directory / task / f"best_episode{episode:05d}", nodes)
            if summary["strict_success"] and task not in first_hits:
                first_hits[task] = {
                    "mixed_episode": episode,
                    "task_episodes": sum(row["task"] == task for row in rows),
                    "updates": update, "total_steps": steps,
                    "task_steps": sum(row["steps"] for row in rows if row["task"] == task),
                }
                save_node_path(directory / task / "first_strict_success", nodes)
        if design.get("diagnostics", False):
            from mixed_diagnostics import analyze_batch
            diagnostic, _ = analyze_batch(agent, goals, batch_metadata)
            diagnostic_rows.extend(diagnostic)
            write_csv(directory / "advantage_diagnostics.csv", diagnostic_rows)
        extra_critic = design.get("critic_replay_steps", 0)
        if extra_critic:
            from critic_replay import prepare, fit
            replay_data = prepare(agent)
        update_result = agent.update(goals, design.get("advantage_centering", "batch"))
        if extra_critic:
            fit(agent, replay_data, extra_critic, critic_generator)
        update += 1
        update_rows.append({"update": update, "episode": episode, **update_result})
        checkpoints.append(save_checkpoint(
            directory, model, agent, schedule_rng, episode, update, steps,
        ))
        write_csv(directory / "episode_metrics.csv", rows)
        write_csv(directory / "updates.csv", update_rows)
        write_csv(directory / "checkpoint_index.csv", checkpoints)
        print(f"Train E={episode} U={update} steps={steps} task_steps={task_steps} first_hits={list(first_hits)}", flush=True)
    summary = {
        "episodes": int(rows[-1]["episode"]), "episode_cap": design["episodes"],
        "updates": update, "steps": steps,
        "stop_reason": "d2_step_budget" if budget is not None and task_steps.get("five_d2", 0) >= budget else "episode_cap",
        "d2_step_budget": budget,
        "first_hits": first_hits,
        "per_task": {task: {
            "episodes": sum(r["task"] == task for r in rows),
            "steps": sum(r["steps"] for r in rows if r["task"] == task),
            "successes": sum(r["strict_success"] for r in rows if r["task"] == task),
        } for task in tasks},
    }
    (directory / "training_summary.json").write_text(json.dumps(summary, indent=2))
    plot_learning(directory, rows)
    return directory


def evaluate_job(job):
    directory, task, checkpoint, method = job
    directory = Path(directory)
    design = json.loads((directory / "design.json").read_text())
    config = design["configs"][task]
    update = int(checkpoint["update"]) if checkpoint else -1
    name = f"{task}_{method}_update{update:05d}.csv"
    output = directory / "evaluation" / name
    if output.exists():
        rows = read_csv(output)
        if len(rows) != design["rollouts"]:
            raise ValueError(f"Incomplete evaluation: {output}")
        return rows
    torch.set_num_threads(1)
    torch.manual_seed(0)
    env = GoalConditionedEnv(config)
    if method == "policy":
        model, agent = make_model_agent(config)
        saved = torch.load(directory / "checkpoints" / checkpoint["filename"],
                           map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
    rows = []
    for replicate in range(design["rollouts"]):
        # Identical evaluation seeds across checkpoints and both target goals.
        seed = design["evaluation_seed_base"] + replicate
        np.random.seed(seed)
        torch.manual_seed(seed)
        if method == "random":
            agent = UniformRandomAgent(config["max_step_width"], np.random.default_rng(seed))
        with torch.inference_mode():
            summary, _ = rollout(env, agent, training=False)
        quality = [env.near_miss_quality(r["flatness_ratio"], r["spread_error"])[2]
                   for r in env.reward_trace]
        near = any(r["flatness_ratio"] <= .1 and r["spread_error"] <= .05
                   for r in env.reward_trace)
        rows.append({
            "task": task, "method": method, "update": update,
            "training_episode": int(checkpoint["episode"]) if checkpoint else 0,
            "replicate": replicate + 1, "evaluation_seed": seed,
            "reach_near_miss": int(near), "best_quality": max(quality, default=0.),
            **summary,
        })
        if (replicate + 1) % 25 == 0:
            print(f"Evaluate {task} {method} U={update} {replicate + 1}/{design['rollouts']}", flush=True)
    write_csv(output, rows)
    return rows


def wilson(count, total):
    p, z = count / total, 1.959963984540054
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return (0. if count == 0 else mid-radius, 1. if count == total else mid+radius)


def evaluate(directory, workers=2):
    directory = Path(directory).resolve()
    design = json.loads((directory / "design.json").read_text())
    tasks = design.get("tasks", TASKS)
    checkpoints = read_csv(directory / "checkpoint_index.csv")
    selected = [c for c in checkpoints if int(c["episode"]) % design["eval_every"] == 0
                or int(c["episode"]) == design.get("resumed_from_episode")
                or c == checkpoints[-1]]
    (directory / "evaluation").mkdir(exist_ok=True)
    jobs = [(str(directory), task, c, "policy") for c in selected for task in tasks]
    jobs += [(str(directory), task, None, "random") for task in tasks]
    if workers == 1:
        results = list(map(evaluate_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate_job, jobs))
    summaries = []
    for rows in results:
        row = rows[0]
        count = len(rows)
        successes = sum(int(r["strict_success"]) for r in rows)
        near = sum(int(r["reach_near_miss"]) for r in rows)
        slo, shi = wilson(successes, count)
        nlo, nhi = wilson(near, count)
        q25, median, q75 = np.quantile([float(r["best_quality"]) for r in rows], [.25,.5,.75])
        summaries.append({
            **{k: row[k] for k in ("task", "method", "update", "training_episode")},
            "rollouts": count, "strict_success_rate": successes/count,
            "strict_ci_low": slo, "strict_ci_high": shi,
            "near_miss_rate": near/count, "near_ci_low": nlo, "near_ci_high": nhi,
            "q25": q25, "q_median": median, "q75": q75,
            "mean_steps": np.mean([float(r["steps"]) for r in rows]),
        })
    write_csv(directory / "evaluation_summary.csv", summaries)
    plot_evaluation(directory, summaries, results)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def plot_learning(directory, rows):
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, axes = plt.subplots(2, len(tasks), figsize=(5*len(tasks), 6), constrained_layout=True, squeeze=False)
        for col, task in enumerate(tasks):
            selected = [r for r in rows if r["task"] == task]
            x = np.array([r["episode"] for r in selected])
            for ax, key, ylabel in ((axes[0,col], "total_reward", "Episode reward"),
                                    (axes[1,col], "terminal_flatness", "Terminal flatness")):
                y = np.array([r[key] for r in selected])
                ax.plot(x, y, color="0.7", lw=.6)
                window = min(20, len(y))
                ax.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode="valid"), color="black")
                ax.set(xlabel="Training episode", ylabel=ylabel)
            axes[0,col].set_title(task)
            axes[1,col].set_yscale("log")
        fig.savefig(directory / "learning_curve.png", dpi=180)
        plt.close(fig)


def plot_evaluation(directory, summaries, results):
    tasks = list(dict.fromkeys(r["task"] for r in summaries))
    with plt.rc_context({"font.family": "Times New Roman", "mathtext.fontset": "stix"}):
        for task in tasks:
            selected = [r for r in summaries if r["task"] == task and r["method"] == "policy"]
            random = next(r for r in summaries if r["task"] == task and r["method"] == "random")
            x = [int(r["update"]) for r in selected]
            for name, centre, low, high, ylabel in (
                ("strict_success", "strict_success_rate", "strict_ci_low", "strict_ci_high", "Strict success probability"),
                ("near_miss", "near_miss_rate", "near_ci_low", "near_ci_high", "Near-miss-region reach probability"),
                ("quality", "q_median", "q25", "q75", r"$Q^*$"),
            ):
                fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
                ax.plot(x, [r[centre] for r in selected], color="black", marker="o")
                ax.fill_between(x, [r[low] for r in selected], [r[high] for r in selected], color="black", alpha=.15)
                ax.axhline(random[centre], color="black", ls="--", lw=1)
                ax.set(xlabel="Actor update", ylabel=ylabel, ylim=(0,1))
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                fig.savefig(directory / f"{task}_{name}.png", dpi=220)
                plt.close(fig)
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.5*len(tasks), 3.6), constrained_layout=True,
                                 sharex=True, sharey=True, squeeze=False)
        for ax, task in zip(axes[0], tasks):
            sets = [rows for rows in results if rows[0]["task"] == task and rows[0]["method"] == "policy"]
            rows = max(sets, key=lambda rs: int(rs[0]["update"]))
            ax.scatter([float(r["terminal_spread"]) for r in rows],
                       [max(float(r["terminal_flatness"]), 1e-12) for r in rows],
                       facecolors="none", edgecolors="black", s=20)
            target = 2 if task == "five_d2" else 3
            ax.axvspan(target-.05, target+.05, color="black", alpha=.1)
            ax.axhline(.001, color="black", ls="--")
            ax.set(xlabel="Terminal spread D", ylabel="Terminal flatness F", yscale="log", title=task)
        fig.savefig(directory / "terminal_distribution.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=9201)
    parser.add_argument("--task", choices=("mixed", *TASKS), default="mixed")
    parser.add_argument("--advantage-centering", choices=("batch", "goal"), default="batch")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--adaptive-exploration", action="store_true",
                        help="Broaden behavior distributions for goals below 20%% success in their last 32 training episodes")
    parser.add_argument("--critic-replay-steps", type=int, default=0)
    parser.add_argument("--d2-step-budget", type=int, help="Stop after the first complete batch reaching this cumulative D2 step count")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=1.5e-4)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-goal-input", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue a completed run in a new directory; inherit its training settings")
    parser.add_argument("--warm-start", type=Path, help="Switch a completed run to the requested tasks; preserve NN, optimizers and RNG, reset phase counters")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate(args.evaluate_only, args.workers)
        return
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("batch-size must be a positive even number")
    if args.episodes < 1 or args.episodes > 10_000 or args.episodes % args.batch_size:
        parser.error("episodes must be <=10000 and a positive multiple of batch-size")
    if args.eval_every < 1 or args.rollouts < 1 or args.workers < 1:
        parser.error("eval-every, rollouts and workers must be positive")
    if args.d2_step_budget is not None and args.d2_step_budget < 1:
        parser.error("d2-step-budget must be positive")
    design = {
        "episodes": args.episodes, "seed": args.seed, "batch_size": args.batch_size,
        "actor_lr": args.actor_lr, "critic_lr": args.critic_lr,
        "eval_every": args.eval_every, "rollouts": args.rollouts,
        "condition_goal": not args.no_goal_input, "evaluation_seed_base": 950_000,
        "tasks": list(TASKS) if args.task == "mixed" else [args.task],
        "advantage_centering": args.advantage_centering, "diagnostics": args.diagnostics,
        "critic_replay_steps": args.critic_replay_steps,
        "adaptive_exploration": args.adaptive_exploration,
        "evaluation_policy": "unmodified NN; exploration controller disabled",
    }
    if args.warm_start:
        if args.resume:
            parser.error("warm-start and resume are mutually exclusive")
        source = args.warm_start.resolve()
        previous_design = json.loads((source / "design.json").read_text())
        if previous_design.get("critic_replay_steps", 0) or args.critic_replay_steps:
            parser.error("warm-start currently supports standard critic updates only")
        for key, default in (("seed", 9201), ("batch_size", 8), ("actor_lr", 5e-4),
                             ("critic_lr", 1.5e-4), ("condition_goal", True),
                             ("advantage_centering", "batch")):
            if design[key] != previous_design.get(key, default):
                parser.error(f"warm-start must preserve {key}")
        last = read_csv(source / "checkpoint_index.csv")[-1]
        checkpoint = source / "checkpoints" / last["filename"]
        configs = configurations(design)
        for task in design["tasks"]:
            if task in previous_design["configs"]:
                configs[task] = previous_design["configs"][task]
        design.update(configs=configs, warm_start_checkpoint=str(checkpoint),
                      warm_start_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      pretraining_run=str(source), pretraining_episodes=int(last["episode"]),
                      pretraining_updates=int(last["update"]), pretraining_steps=int(last["steps"]))
    if args.resume:
        source = args.resume.resolve()
        design = json.loads((source / "design.json").read_text())
        if design.get("critic_replay_steps", 0):
            parser.error("Critic-replay experiments currently support fresh training only")
        previous = json.loads((source / "training_summary.json").read_text())["episodes"]
        if args.episodes <= previous or args.episodes % design["batch_size"]:
            parser.error("resume episodes must exceed the previous total and be a multiple of the saved batch-size")
        design.update(episodes=args.episodes, eval_every=args.eval_every,
                      parent_run=str(source), resumed_from_episode=previous)
    if args.d2_step_budget is not None:
        if "five_d2" not in design.get("tasks", TASKS):
            parser.error("d2-step-budget requires D2 training")
        design["d2_step_budget"] = args.d2_step_budget
    directory = train(design, args.resume)
    if not args.train_only:
        evaluate(directory, args.workers)


if __name__ == "__main__":
    main()
