"""Read-only replay of saved training batches and per-goal advantage diagnostics."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np


def batch_tensors(agent):
    returns, values, logs, lengths = [], [], [], []
    for episode in agent.finished_episodes:
        g = 0.0
        episode_returns = []
        for tr in reversed(episode):
            g = tr.reward + agent.gamma * g
            episode_returns.append(g)
        returns.extend(reversed(episode_returns))
        values.extend(tr.value for tr in episode)
        logs.extend(tr.log_probability for tr in episode)
        lengths.append(len(episode))
    return (torch.tensor(returns, dtype=torch.float32), torch.stack(values),
            torch.stack(logs), lengths)


def analyze_batch(agent, groups, metadata, gradients=False):
    returns, values, logs, lengths = batch_tensors(agent)
    raw = returns - values.detach()
    scale = raw.std(unbiased=False) + 1e-8
    normalized = (raw - raw.mean()) / scale
    ids = torch.repeat_interleave(torch.arange(len(groups)), torch.tensor(lengths))
    tasks = list(dict.fromkeys(groups))
    masks = {task: torch.tensor([groups[int(i)] == task for i in ids]) for task in tasks}
    centered = raw.clone()
    for mask in masks.values():
        centered[mask] = (raw[mask] - raw[mask].mean()) / scale
    rows, offset = [], 0
    for i, length in enumerate(lengths):
        s = slice(offset, offset + length)
        a, n, c = raw[s], normalized[s], centered[s]
        rows.append(dict(metadata[i], task=groups[i], transition_count=length,
            return_mean=float(returns[s].mean()), value_mean=float(values[s].detach().mean()),
            initial_return=float(returns[offset]), initial_value=float(values[offset].detach()),
            raw_adv_mean=float(a.mean()), batch_adv_mean=float(n.mean()),
            goal_adv_mean=float(c.mean()), raw_negative=float((a < 0).float().mean()),
            batch_negative=float((n < 0).float().mean()), goal_negative=float((c < 0).float().mean()),
            positive_to_negative=float(((a > 0) & (n < 0)).float().mean()),
            critic_mse=float(((returns[s]-values[s].detach())**2).mean()),
            batch_raw_mean=float(raw.mean()), batch_raw_std=float(scale)))
        offset += length
    gradient_row = {}
    if gradients and len(tasks) == 2:
        normalizer = len(groups) * agent.actor_reference_horizon
        for name, advantages in (("batch", normalized), ("goal", centered)):
            gs = []
            for task in tasks:
                loss = -(logs[masks[task]] * advantages[masks[task]]).sum() / normalizer
                grad = torch.autograd.grad(loss, agent.actor_parameter_list, retain_graph=True, allow_unused=True)
                gs.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                     for g, p in zip(grad, agent.actor_parameter_list)]).detach())
            norms = [float(torch.linalg.vector_norm(g)) for g in gs]
            gradient_row.update({f"{name}_{tasks[i]}_norm": norms[i] for i in range(2)})
            gradient_row[f"{name}_cosine"] = float(torch.dot(*gs) / (norms[0]*norms[1]+1e-12))
    return rows, gradient_row


def episode_metadata(env, summary, episode, update):
    return dict(episode=episode, update=update, reward=summary["total_reward"],
                strict_success=summary["strict_success"], termination_reason=summary["termination_reason"],
                near_miss=int(any(x["flatness_ratio"] <= .1 and x["spread_error"] <= .05 for x in env.reward_trace)),
                best_quality=max((env.near_miss_quality(x["flatness_ratio"], x["spread_error"])[2]
                                  for x in env.reward_trace), default=0.0))


def main():
    import RL_mixed_goals as m
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--updates", type=int, nargs="+", default=[0, 3, 11, 16, 32, 128, 384])
    args = parser.parse_args()
    design = json.loads((args.run / "design.json").read_text())
    configs = design["configs"]
    original = m.read_csv(args.run / "episode_metrics.csv")
    output = args.run / "advantage_diagnostic"
    output.mkdir(exist_ok=True)
    all_rows, gradient_rows = [], []
    torch.set_num_threads(1)
    for update in args.updates:
        model, agent = m.make_model_agent(configs["five_d2"])
        ck = torch.load(args.run / "checkpoints" / f"update{update:05d}.pt", weights_only=True)
        model.load_state_dict(ck["model_state_dict"])
        agent.actor_optimizer.load_state_dict(ck["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(ck["critic_optimizer"])
        envs = {t: m.GoalConditionedEnv(c) for t, c in configs.items()}
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["schedule_rng"]
        torch.set_rng_state(ck["torch_rng"])
        goals = [t for t in m.TASKS for _ in range(design["batch_size"]//2)]
        rng.shuffle(goals)
        controls = m.exploration_controls(design, original[:ck["episode"]]) if design.get("adaptive_exploration") else None
        metadata = []
        for offset, task in enumerate(goals, 1):
            episode = ck["episode"] + offset
            if controls is not None:
                agent.exploration_strength = controls[task][1]
            summary, _ = m.rollout(envs[task], agent, training=True)
            expected = original[episode-1]
            if controls is not None:
                assert agent.exploration_strength == float(expected["exploration_strength"])
            assert expected["task"] == task
            for key, value in summary.items():
                assert str(value) == expected[key] or (key != "termination_reason" and float(value) == float(expected[key])), (episode, key)
            metadata.append(episode_metadata(envs[task], summary, episode, update))
        rows, grad = analyze_batch(agent, goals, metadata, gradients=True)
        # Diagnostics must not alter the next real update or consume action RNG.
        agent.update()
        expected = torch.load(args.run / "checkpoints" / f"update{update+1:05d}.pt", weights_only=True)
        assert all(torch.equal(v, expected["model_state_dict"][k]) for k,v in model.state_dict().items())
        assert torch.equal(torch.get_rng_state(), expected["torch_rng"])
        all_rows.extend(rows)
        gradient_rows.append(dict(update=update, **grad))
        m.write_csv(output / "episodes.csv", all_rows)
        m.write_csv(output / "gradients.csv", gradient_rows)
        print(f"Replayed U={update}: all episode results and next NN exactly match", flush=True)
    print(f"DIAGNOSTIC_DIRECTORY={output}", flush=True)


if __name__ == "__main__":
    main()
