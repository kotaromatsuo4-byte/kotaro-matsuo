"""Replay one saved single- or mixed-goal batch without changing the run."""
import argparse
import json
from pathlib import Path

import RL_mixed_goals as m
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--update", type=int, default=0)
    args = parser.parse_args()
    design = json.loads((args.run / "design.json").read_text())
    if design.get("critic_replay_steps", 0):
        parser.error("Extra critic updates are not supported by this verifier")
    torch.set_num_threads(1)
    tasks = design.get("tasks", m.TASKS)
    config = design["configs"]
    model, agent = m.make_model_agent(config[tasks[0]])
    envs = {t: m.GoalConditionedEnv(config[t]) for t in tasks}
    saved = torch.load(args.run / "checkpoints" / f"update{args.update:05d}.pt", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    agent.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved["schedule_rng"]
    goals = [t for t in tasks for _ in range(design["batch_size"] // len(tasks))]
    rng.shuffle(goals)
    original = m.read_csv(args.run / "episode_metrics.csv")
    controls = m.exploration_controls(design, original[:saved["episode"]])
    for offset, task in enumerate(goals):
        expected = original[saved["episode"] + offset]
        assert expected["task"] == task
        agent.exploration_strength = controls[task][1]
        if design.get("adaptive_exploration"):
            assert float(expected["exploration_strength"]) == controls[task][1]
            assert float(expected["prior_success_rate"]) == controls[task][0]
        summary, _ = m.rollout(envs[task], agent, training=True)
        assert all(str(v) == expected[k] for k, v in summary.items())
    agent.update(goals, design.get("advantage_centering", "batch"))
    expected = torch.load(args.run / "checkpoints" / f"update{args.update+1:05d}.pt", weights_only=True)
    assert all(torch.equal(v, expected["model_state_dict"][k]) for k, v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected["torch_rng"])
    assert rng.bit_generator.state == expected["schedule_rng"]
    print(f"PASS U={args.update}: controller, all episode results, next NN and both RNG states match exactly")


if __name__ == "__main__":
    main()
