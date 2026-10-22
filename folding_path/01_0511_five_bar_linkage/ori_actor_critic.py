# -*- coding: cp932 -*-
import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i] ** 2)
    return g


def index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])


def dedx(node, num_node, member, num_member, free_index):
    dedx_all = np.zeros((num_member, num_node * 3))
    node_diff = node[member[:, 0]] - node[member[:, 1]]
    for i in range(num_member):
        i0, i1 = member[i]
        dedx_all[i, i0 * 3:(i0 + 1) * 3] = node_diff[i]
        dedx_all[i, i1 * 3:(i1 + 1) * 3] = -node_diff[i]
    return dedx_all[:, free_index]


def mode(free_index, dedx_matrix, num_node, tol=1e-8):
    _, sigma, vh = np.linalg.svd(dedx_matrix)
    rank = np.count_nonzero(sigma > tol)
    dof = dedx_matrix.shape[1] - rank
    displace_mode = np.zeros((num_node * 3, dof))
    displace_mode[free_index, :] = vh[rank:].T
    return displace_mode


def corrector(node, num_node, member, num_member, L0, free_index,
              beta0=1.0, n_iter=20, tau=0.5):
    g = incompatibility(node, member, L0)
    gamma = dedx(node, num_node, member, num_member, free_index)
    dd_free = -np.linalg.pinv(gamma) @ g
    dd = np.zeros(num_node * 3)
    dd[free_index] = dd_free

    beta = beta0
    for _ in range(n_iter):
        node_new = node + beta * dd.reshape(-1, 3)
        g_new = incompatibility(node_new, member, L0)
        if np.linalg.norm(g_new) < (1 - 1e-4 * beta) * np.linalg.norm(g):
            break
        beta *= tau
    return beta * dd.reshape(-1, 3)


def compute_flatness_z(node):
    return np.max(node[:, 2]) - np.min(node[:, 2])


def make_noisy_node(node, fix, noise_std, rng):
    noisy = node.copy()
    if noise_std <= 0.0:
        return noisy

    fixed_dof = set((int(n), int(axis)) for n, axis in fix)
    for n in range(node.shape[0]):
        for axis in range(3):
            if (n, axis) not in fixed_dof:
                noisy[n, axis] += rng.normal(0.0, noise_std)
    return noisy


class OrigamiEnv:
    def __init__(self, node, member, fix, max_steps=24, action_scale=0.20):
        self.node0 = node.copy()
        self.node = node.copy()
        self.member = member
        self.fix = fix
        self.num_node = node.shape[0]
        self.num_member = member.shape[0]
        self.max_steps = max_steps
        self.action_scale = action_scale

        self.L0 = np.array([
            np.linalg.norm(node[i0] - node[i1])
            for i0, i1 in member
        ])
        self.free = index_free(self.num_node, self.fix)
        self.initial_flatness = compute_flatness_z(self.node0)
        self.step_count = 0
        self.prev_flatness = None

    def reset(self):
        self.node = self.node0.copy()
        self.step_count = 0
        self.prev_flatness = compute_flatness_z(self.node)
        return self.get_state()

    def step(self, action):
        modes = self.get_modes()
        k = modes.shape[1]
        bounded_action = self.action_scale * np.clip(np.asarray(action[:k]), -1.0, 1.0)

        node_tmp = self.node + (modes @ bounded_action).reshape(self.num_node, 3)
        for _ in range(10):
            dc = corrector(
                node_tmp, self.num_node, self.member, self.num_member,
                self.L0, self.free
            )
            node_tmp += dc
            if np.linalg.norm(dc) < 1e-7:
                break

        self.node = node_tmp
        flatness = compute_flatness_z(self.node)
        progress = (self.prev_flatness - flatness) / max(self.initial_flatness, 1e-8)
        flatness_ratio = flatness / max(self.initial_flatness, 1e-8)
        reward = 12.0 * progress - 0.20 * flatness_ratio
        if k > 0:
            reward -= 0.03 * float(np.mean(np.square(bounded_action)))

        self.prev_flatness = flatness
        self.step_count += 1

        done = False
        success = flatness < 1e-3
        if success:
            reward += 8.0
            done = True

        strain = self.compute_strain()
        if strain > 1e-4:
            reward -= 10.0 * min(strain / 1e-4, 10.0)
            done = True

        if self.step_count >= self.max_steps:
            reward -= 2.0 * flatness_ratio
            done = True

        return self.get_state(), reward, done, {
            "flatness": flatness,
            "strain": strain,
            "success": success,
        }

    def get_state(self):
        state = self.node.flatten().copy()
        state[2::3] /= max(self.initial_flatness, 1e-8)
        return state

    def get_modes(self):
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        return mode(self.free, d, self.num_node)

    def compute_strain(self):
        strain = 0.0
        for i in range(self.num_member):
            n0, n1 = self.member[i]
            l = np.linalg.norm(self.node[n0] - self.node[n1])
            strain += (l - self.L0[i]) ** 2
        return strain


class ActorCritic(nn.Module):
    def __init__(self, state_size, mode_size):
        super().__init__()
        self.actor_fc1 = nn.Linear(state_size + mode_size, 128)
        self.actor_fc2 = nn.Linear(128, 128)
        self.actor_mu = nn.Linear(128, 1)
        self.actor_logstd = nn.Linear(128, 1)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.actor_logstd.bias, -1.0)

    def forward_actor(self, state, modes):
        k = modes.shape[0]
        if k == 0:
            empty = state.new_empty((0,))
            return empty, empty
        state_expand = state.repeat(k, 1)
        x = torch.cat([state_expand, modes], dim=1)
        h = torch.tanh(self.actor_fc1(x))
        h = torch.tanh(self.actor_fc2(h))
        mu = 1.5 * torch.tanh(self.actor_mu(h).squeeze(-1))
        log_std = torch.clamp(self.actor_logstd(h).squeeze(-1), -2.5, 0.2)
        return mu, torch.exp(log_std)

    def value(self, state):
        return self.critic(state).squeeze(-1)


@dataclass
class Transition:
    reward: float
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class Agent:
    def __init__(self, state_size, mode_size, gamma=0.97, lr=3e-4,
                 entropy_coef=0.03, value_coef=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.model = ActorCritic(state_size, mode_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.episodes = []
        self.current_episode = []

    def get_action(self, state, modes, deterministic=False):
        state_t = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        modes_t = torch.tensor(modes.T, dtype=torch.float32)
        mu, std = self.model.forward_actor(state_t, modes_t)
        value = self.model.value(state_t)

        if mu.numel() == 0:
            return np.zeros(0, dtype=np.float32), torch.tensor(0.0), value, torch.tensor(0.0)

        dist = Normal(mu, std)
        raw_action = mu if deterministic else dist.sample()
        log_prob = dist.log_prob(raw_action).sum()
        entropy = dist.entropy().sum()
        action = torch.tanh(raw_action)
        return action.detach().numpy(), log_prob, value.squeeze(0), entropy

    def add(self, reward, log_prob, value, entropy):
        self.current_episode.append(Transition(reward, log_prob, value, entropy))

    def finish_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
        self.current_episode = []

    def update(self):
        returns = []
        log_probs = []
        values = []
        entropies = []
        for episode in self.episodes:
            G = 0.0
            episode_returns = []
            for transition in reversed(episode):
                G = transition.reward + self.gamma * G
                episode_returns.insert(0, G)
            for transition, G in zip(episode, episode_returns):
                returns.append(G)
                log_probs.append(transition.log_prob)
                values.append(transition.value)
                entropies.append(transition.entropy)

        if not returns:
            return 0.0

        returns_t = torch.tensor(returns, dtype=torch.float32)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        entropies_t = torch.stack(entropies)
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.episodes = []
        return float(loss.detach())


def moving_average(data, window):
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode="valid")


def save_node_history(directory, node_history):
    os.makedirs(directory, exist_ok=True)
    for i, node_step in enumerate(node_history):
        np.savetxt(os.path.join(directory, f"{i + 1}.dat"), node_step, fmt="%.6f")


def pairwise_rmsd(shapes):
    if len(shapes) < 2:
        return np.zeros((len(shapes), len(shapes)))
    flat = shapes.reshape(len(shapes), -1)
    diff = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.mean(diff ** 2, axis=2))


def greedy_clusters(distance_matrix, threshold):
    labels = -np.ones(distance_matrix.shape[0], dtype=int)
    for i in range(distance_matrix.shape[0]):
        if labels[i] >= 0:
            continue
        label = labels.max() + 1
        labels[distance_matrix[i] <= threshold] = label
    return labels


def fold_pattern(node):
    return "".join("+" if d >= 0 else "-" for d in np.diff(node[:, 0]))


def plot_learning_curve(reward_history, flatness_history, batch_size, path):
    fig, ax1 = plt.subplots()
    ax1.plot(reward_history, color="0.5", alpha=0.35, label="Raw reward")
    ma = moving_average(reward_history, batch_size)
    if len(ma) > 0:
        x = np.arange(len(ma)) + batch_size - 1
        ax1.plot(x, ma, color="black", linewidth=2, label="Reward avg")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.grid(True)

    ax2 = ax1.twinx()
    fma = moving_average(flatness_history, batch_size)
    if len(fma) > 0:
        x = np.arange(len(fma)) + batch_size - 1
        ax2.plot(x, fma, color="tab:red", linewidth=2, label="Final flatness avg")
    ax2.set_ylabel("Final flatness", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_one_model(model_id, base_node, member, fix, args):
    seed = args.seed + model_id * 1009
    np_rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    node0 = make_noisy_node(base_node, fix, args.node_noise_std, np_rng)
    env = OrigamiEnv(
        node0, member, fix,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
    )
    agent = Agent(
        env.get_state().shape[0],
        env.num_node * 3,
        gamma=args.gamma,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )

    model_dir = os.path.join(args.output_dir, f"model_{model_id:02d}")
    best_dir = os.path.join(model_dir, "best")
    os.makedirs(model_dir, exist_ok=True)
    np.savetxt(os.path.join(model_dir, "node_initial_noisy.dat"), node0, fmt="%.6f")

    rewards = []
    flatnesses = []
    best_flatness = np.inf
    best_reward = -np.inf
    best_history = []
    best_episode = -1
    node_history = []

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        node_history = []
        info = {"flatness": env.initial_flatness, "success": False}

        while not done:
            action, log_prob, value, entropy = agent.get_action(state, env.get_modes())
            next_state, reward, done, info = env.step(action)
            agent.add(reward, log_prob, value, entropy)
            state = next_state
            total_reward += reward
            node_history.append(env.node.copy())

        agent.finish_episode()
        rewards.append(total_reward)
        flatnesses.append(info["flatness"])

        if (episode + 1) % args.batch_size == 0:
            agent.update()

        if info["flatness"] < best_flatness:
            best_flatness = info["flatness"]
            best_reward = total_reward
            best_episode = episode
            best_history = [n.copy() for n in node_history]

        if args.print_every > 0 and episode % args.print_every == 0:
            start = max(0, len(rewards) - args.batch_size)
            print(
                f"model {model_id:02d} ep {episode:5d} | reward {total_reward:8.3f} "
                f"| avg_reward {np.mean(rewards[start:]):8.3f} "
                f"| flatness {info['flatness']:.6f} "
                f"| avg_flatness {np.mean(flatnesses[start:]):.6f}"
            )

    if agent.episodes:
        agent.update()

    if best_history:
        save_node_history(best_dir, best_history)
        np.savetxt(os.path.join(model_dir, "best_final.dat"), best_history[-1], fmt="%.6f")
    torch.save(agent.model.state_dict(), os.path.join(model_dir, "policy.pt"))
    np.savetxt(os.path.join(model_dir, "reward_history.csv"), np.array(rewards), delimiter=",")
    np.savetxt(os.path.join(model_dir, "flatness_history.csv"), np.array(flatnesses), delimiter=",")
    plot_learning_curve(
        rewards,
        flatnesses,
        args.batch_size,
        os.path.join(model_dir, "learning_curve.png"),
    )

    eval_rows, eval_histories = evaluate_model(agent, env, args.eval_episodes, args.eval_deterministic)
    np.savetxt(
        os.path.join(model_dir, "eval_final_shapes.csv"),
        eval_rows,
        delimiter=",",
        header=eval_header(env.num_node),
        comments="",
    )
    for i, history in enumerate(eval_histories[:args.save_eval_histories]):
        save_node_history(os.path.join(model_dir, "eval_histories", f"episode_{i:03d}"), history)

    final_nodes = eval_rows[:, 5:].reshape(len(eval_rows), env.num_node, 3)
    best_eval = eval_rows[np.argmin(eval_rows[:, 2])]
    return {
        "model_id": model_id,
        "seed": seed,
        "best_episode": best_episode,
        "best_reward": best_reward,
        "best_flatness": best_flatness,
        "eval_mean_flatness": float(eval_rows[:, 2].mean()),
        "eval_min_flatness": float(eval_rows[:, 2].min()),
        "eval_best_pattern": fold_pattern(best_eval[5:].reshape(env.num_node, 3)),
        "eval_best_node": best_eval[5:].reshape(env.num_node, 3),
        "eval_final_nodes": final_nodes,
    }


def evaluate_model(agent, env, episodes, deterministic):
    rows = []
    histories = []
    was_training = agent.model.training
    agent.model.eval()
    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            history = []
            info = {"flatness": env.initial_flatness, "strain": 0.0, "success": False}
            while not done:
                action, _, _, _ = agent.get_action(state, env.get_modes(), deterministic=deterministic)
                state, reward, done, info = env.step(action)
                total_reward += reward
                history.append(env.node.copy())
            final = env.node.copy()
            rows.append([
                episode,
                total_reward,
                info["flatness"],
                info["strain"],
                float(info["success"]),
                *final.flatten(),
            ])
            histories.append(history)
    if was_training:
        agent.model.train()
    return np.array(rows, dtype=float), histories


def eval_header(num_node):
    return (
        "episode,total_reward,final_flatness,strain,success,"
        + ",".join([f"node{n}_{axis}" for n in range(num_node) for axis in "xyz"])
    )


def save_ensemble_summary(results, args):
    rows = []
    all_final_nodes = []
    model_ids = []
    for result in results:
        rows.append([
            result["model_id"],
            result["seed"],
            result["best_episode"],
            result["best_reward"],
            result["best_flatness"],
            result["eval_mean_flatness"],
            result["eval_min_flatness"],
        ])
        all_final_nodes.extend(result["eval_final_nodes"])
        model_ids.extend([result["model_id"]] * len(result["eval_final_nodes"]))

    rows = np.array(rows, dtype=float)
    np.savetxt(
        os.path.join(args.output_dir, "ensemble_summary.csv"),
        rows,
        delimiter=",",
        header="model_id,seed,best_episode,best_reward,best_flatness,eval_mean_flatness,eval_min_flatness",
        comments="",
    )

    all_final_nodes = np.array(all_final_nodes)
    distances = pairwise_rmsd(all_final_nodes)
    labels = greedy_clusters(distances, args.cluster_threshold)
    np.savetxt(os.path.join(args.output_dir, "ensemble_pairwise_rmsd.csv"), distances, delimiter=",")
    np.savetxt(os.path.join(args.output_dir, "ensemble_cluster_labels.csv"), labels, delimiter=",", fmt="%d")

    cluster_rows = []
    flatness = np.array([
        compute_flatness_z(node)
        for node in all_final_nodes
    ])
    model_ids = np.array(model_ids, dtype=int)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        best_idx = idx[np.argmin(flatness[idx])]
        cluster_rows.append([
            label,
            len(idx),
            best_idx,
            model_ids[best_idx],
            flatness[idx].mean(),
            flatness[idx].min(),
            flatness[idx].max(),
            fold_pattern(all_final_nodes[best_idx]),
        ])
        np.savetxt(
            os.path.join(args.output_dir, f"cluster_{int(label)}_best_final.dat"),
            all_final_nodes[best_idx],
            fmt="%.6f",
        )

    with open(os.path.join(args.output_dir, "ensemble_cluster_summary.csv"), "w", encoding="utf-8") as f:
        f.write("cluster,size,best_global_eval,best_model,mean_flatness,min_flatness,max_flatness,best_pattern\n")
        for row in cluster_rows:
            f.write(",".join(map(str, row)) + "\n")

    print("===================================")
    print("Ensemble summary")
    print(f"Models          : {len(results)}")
    print(f"Eval episodes   : {args.eval_episodes} each")
    print(f"Clusters        : {len(np.unique(labels))} (threshold RMSD={args.cluster_threshold})")
    for result in results:
        print(
            f"model {result['model_id']:02d} | best_flat {result['best_flatness']:.6f} "
            f"| eval_min {result['eval_min_flatness']:.6f} "
            f"| best_pattern {result['eval_best_pattern']}"
        )
    print(f"Saved directory : {args.output_dir}")
    print("===================================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--node-noise-std", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--save-eval-histories", type=int, default=3)
    parser.add_argument("--cluster-threshold", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="ac_noise_ensemble_results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base_node = np.loadtxt("node_initial.dat", delimiter=",", dtype=float)
    member = np.loadtxt("member.dat", delimiter=",", dtype=int)
    fix = np.loadtxt("fix.dat", delimiter=",", dtype=int)

    results = []
    for model_id in range(args.models):
        results.append(train_one_model(model_id, base_node, member, fix, args))
    save_ensemble_summary(results, args)


if __name__ == "__main__":
    main()
