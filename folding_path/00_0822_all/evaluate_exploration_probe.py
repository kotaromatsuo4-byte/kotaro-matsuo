"""Frozen-checkpoint evaluation WITH the training exploration controller.

Stored separately from ordinary NN-only checkpoint evaluation. No learning.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from RL_mixed_goals import (
    GoalConditionedEnv, make_model_agent, exploration_controls, read_csv,
    rollout, write_csv,
)
import numpy as np
import torch


def probe(job):
    directory, task, update, override = job
    directory = Path(directory)
    design = json.loads((directory / "design.json").read_text())
    ck = torch.load(directory / "checkpoints" / f"update{update:05d}.pt", weights_only=True)
    history = read_csv(directory / "episode_metrics.csv")[:ck["episode"]]
    rate, strength = exploration_controls(design, history)[task]
    if override is not None:
        strength = override
    torch.set_num_threads(1)
    torch.manual_seed(0)
    env = GoalConditionedEnv(design["configs"][task])
    model, agent = make_model_agent(design["configs"][task])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    agent.exploration_strength = strength
    rows = []
    for replicate in range(design["rollouts"]):
        seed = design["evaluation_seed_base"] + replicate
        torch.manual_seed(seed)
        np.random.seed(seed)
        with torch.inference_mode():
            summary, _ = rollout(env, agent, training=False)
        rows.append(dict(task=task, update=update, episode=ck["episode"],
                         evaluation_seed=seed, exploration_strength=strength,
                         prior_success_rate=rate,
                         reach_near_miss=int(any(r["flatness_ratio"] <= .1 and r["spread_error"] <= .05 for r in env.reward_trace)),
                         best_quality=max((env.near_miss_quality(r["flatness_ratio"], r["spread_error"])[2] for r in env.reward_trace), default=0.),
                         **summary))
    output = directory / "exploration_probe"
    output.mkdir(exist_ok=True)
    suffix = "" if override is None else f"_strength{override:g}"
    write_csv(output / f"{task}_update{update:05d}{suffix}.csv", rows)
    result = dict(task=task, update=update, strength=strength,
                  success=sum(r["strict_success"] for r in rows),
                  near=sum(r["reach_near_miss"] for r in rows),
                  q_median=float(np.median([r["best_quality"] for r in rows])))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--updates", type=int, nargs="+", default=[0])
    parser.add_argument("--task", choices=("five_d2", "five_d3"))
    parser.add_argument("--strength", type=float, help="Fixed counterfactual strength for matched pre/post comparisons")
    args = parser.parse_args()
    design = json.loads((args.run / "design.json").read_text())
    if args.strength is not None and not 0 <= args.strength <= 1:
        parser.error("strength must be between 0 and 1")
    tasks = [args.task] if args.task else design["tasks"]
    jobs = [(str(args.run), task, update, args.strength) for update in args.updates for task in tasks]
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(probe, jobs))
