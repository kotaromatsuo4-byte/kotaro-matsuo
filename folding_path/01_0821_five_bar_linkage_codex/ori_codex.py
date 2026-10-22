# -*- coding: utf-8 -*-
"""Learn mode-mixture histories that fold randomized five-bar linkages to z=0."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def incompatibility(node, member, lengths):
    """Squared bar-length constraint residuals."""
    differences = node[member[:, 0]] - node[member[:, 1]]
    return 0.5 * (np.sum(differences ** 2, axis=1) - lengths ** 2)


def index_free(num_node, fix):
    all_indices = np.arange(num_node * 3)
    fixed_indices = 3 * fix[:, 0] + fix[:, 1]
    return np.delete(all_indices, fixed_indices)


def dedx(node, member, free_index):
    num_node = node.shape[0]
    jacobian = np.zeros((len(member), num_node * 3))
    differences = node[member[:, 0]] - node[member[:, 1]]
    for row, (node0, node1) in enumerate(member):
        jacobian[row, 3 * node0:3 * (node0 + 1)] = differences[row]
        jacobian[row, 3 * node1:3 * (node1 + 1)] = -differences[row]
    return jacobian[:, free_index]


def mode(free_index, dedx_matrix, num_node, tol=1e-8):
    """Return first-order mechanism modes as columns of a matrix."""
    _, singular_values, vh = np.linalg.svd(dedx_matrix, full_matrices=True)
    rank = np.count_nonzero(singular_values > tol)
    num_modes = dedx_matrix.shape[1] - rank
    displacement_modes = np.zeros((num_node * 3, num_modes))
    displacement_modes[free_index, :] = vh[rank:].T
    return displacement_modes


def project_to_constraints(node, member, lengths, free_index,
                           tolerance=1e-10, max_iterations=30):
    """Project a predicted configuration back onto the bar-length constraints."""
    corrected = node.copy()
    for _ in range(max_iterations):
        residual = incompatibility(corrected, member, lengths)
        residual_norm = np.linalg.norm(residual)
        if residual_norm <= tolerance:
            break

        jacobian = dedx(corrected, member, free_index)
        delta_free = -np.linalg.pinv(jacobian) @ residual
        delta = np.zeros(corrected.size)
        delta[free_index] = delta_free
        delta = delta.reshape(corrected.shape)

        step_size = 1.0
        accepted = False
        for _ in range(20):
            candidate = corrected + step_size * delta
            candidate_norm = np.linalg.norm(
                incompatibility(candidate, member, lengths)
            )
            if candidate_norm < residual_norm:
                corrected = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
    return corrected


def compute_gravity_potential(node, reference_point):
    z_values = node[:, 2]
    potential = np.sum(np.abs(z_values - reference_point))
    return float(potential)


def segment_distance(point0, point1, point2, point3):
    """Shortest distance between two three-dimensional line segments."""
    direction0 = point1 - point0
    direction1 = point3 - point2
    offset = point0 - point2
    a = float(np.dot(direction0, direction0))
    b = float(np.dot(direction0, direction1))
    c = float(np.dot(direction1, direction1))
    d = float(np.dot(direction0, offset))
    e = float(np.dot(direction1, offset))
    denominator = a * c - b * b
    epsilon = 1e-14

    if a <= epsilon and c <= epsilon:
        return float(np.linalg.norm(point0 - point2))
    if a <= epsilon:
        parameter0 = 0.0
        parameter1 = max(0.0, min(1.0, e / c))
    elif c <= epsilon:
        parameter1 = 0.0
        parameter0 = max(0.0, min(1.0, -d / a))
    else:
        if denominator > epsilon:
            parameter0 = max(
                0.0, min(1.0, (b * e - c * d) / denominator)
            )
        else:
            parameter0 = 0.0
        parameter1 = (b * parameter0 + e) / c
        if parameter1 < 0.0:
            parameter1 = 0.0
            parameter0 = max(0.0, min(1.0, -d / a))
        elif parameter1 > 1.0:
            parameter1 = 1.0
            parameter0 = max(0.0, min(1.0, (b - d) / a))

    closest0 = point0 + parameter0 * direction0
    closest1 = point2 + parameter1 * direction1
    return float(np.linalg.norm(closest0 - closest1))


def nonadjacent_bar_pairs(member):
    bar_pairs = []
    for first in range(len(member)):
        first_nodes = set(map(int, member[first]))
        for second in range(first + 1, len(member)):
            if first_nodes.isdisjoint(map(int, member[second])):
                bar_pairs.append((first, second))
    return bar_pairs


def has_collision(node_before, node_after, member, bar_pairs, tolerance, substeps):
    """Check non-adjacent bars throughout one linearly interpolated motion step."""
    if tolerance < 0.0:
        return False

    for interpolation_step in range(substeps + 1):
        fraction = interpolation_step / max(substeps, 1)
        node = node_before + fraction * (node_after - node_before)
        for first, second in bar_pairs:
            first0, first1 = member[first]
            second0, second1 = member[second]
            distance = segment_distance(
                node[first0], node[first1], node[second0], node[second1]
            )
            if distance <= tolerance:
                return True
    return False


def randomise_node(base_node, member, lengths, free_index, scale, rng):
    """Add a random initial imperfection in the mechanism-mode subspace."""
    jacobian = dedx(base_node, member, free_index)
    displacement_modes = mode(free_index, jacobian, len(base_node))
    num_modes = displacement_modes.shape[1]
    if scale <= 0.0 or num_modes == 0:
        return base_node.copy()

    coefficients = rng.normal(size=num_modes)
    coefficients *= scale / max(np.linalg.norm(coefficients), 1e-12)
    randomized = base_node + (displacement_modes @ coefficients).reshape(-1, 3)
    return project_to_constraints(randomized, member, lengths, free_index)


class FoldingEnv:
    def __init__(self, base_node, member, fix, rng, max_steps=40,
                 action_scale=0.20, imperfection_scale=0.10,
                 reference_point=0.0, potential_tolerance=0.05,
                 strain_tolerance=1e-8, collision_tolerance=0.02,
                 collision_substeps=10, collision_penalty=20.0):
        self.base_node = base_node.copy()
        self.member = member
        self.fix = fix
        self.rng = rng
        self.num_node = len(base_node)
        self.free = index_free(self.num_node, fix)
        self.lengths = np.linalg.norm(
            base_node[member[:, 0]] - base_node[member[:, 1]], axis=1
        )
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.imperfection_scale = imperfection_scale
        self.reference_point = reference_point
        self.potential_tolerance = potential_tolerance
        self.strain_tolerance = strain_tolerance
        self.collision_tolerance = collision_tolerance
        self.collision_substeps = max(int(collision_substeps), 1)
        self.collision_penalty = abs(float(collision_penalty))
        self.collision_bar_pairs = nonadjacent_bar_pairs(member)
        self.coordinate_scale = max(float(np.max(self.lengths)), 1e-8)
        self.base_potential = max(
            compute_gravity_potential(base_node, reference_point), 1e-8
        )

        self.node = base_node.copy()
        self.initial_node = base_node.copy()
        self.initial_potential = self.base_potential
        self.previous_potential = self.base_potential
        self.step_count = 0

    def reset(self):
        self.node = randomise_node(
            self.base_node,
            self.member,
            self.lengths,
            self.free,
            self.imperfection_scale,
            self.rng,
        )
        self.initial_node = self.node.copy()
        self.initial_potential = max(
            compute_gravity_potential(self.node, self.reference_point), 1e-8
        )
        self.previous_potential = self.initial_potential
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return (self.node / self.coordinate_scale).flatten().astype(np.float32)

    def get_modes(self):
        jacobian = dedx(self.node, self.member, self.free)
        return mode(self.free, jacobian, self.num_node)

    def compute_strain(self):
        current_lengths = np.linalg.norm(
            self.node[self.member[:, 0]] - self.node[self.member[:, 1]], axis=1
        )
        return float(np.sum((current_lengths - self.lengths) ** 2))

    def step(self, action):
        node_before = self.node.copy()
        displacement_modes = self.get_modes()
        num_modes = displacement_modes.shape[1]
        coefficients = self.action_scale * np.clip(
            np.asarray(action[:num_modes], dtype=float), -1.0, 1.0
        )

        predicted = self.node + (
            displacement_modes @ coefficients
        ).reshape(self.num_node, 3)
        node_after = project_to_constraints(
            predicted, self.member, self.lengths, self.free
        )
        collision = has_collision(
            node_before,
            node_after,
            self.member,
            self.collision_bar_pairs,
            self.collision_tolerance,
            self.collision_substeps,
        )
        self.node = node_after

        potential = compute_gravity_potential(self.node, self.reference_point)
        progress = (
            self.previous_potential - potential
        ) / self.initial_potential
        potential_ratio = potential / self.initial_potential

        reward = 12.0 * progress - 0.10 * potential_ratio
        if num_modes:
            reward -= 0.02 * float(np.mean(coefficients ** 2))

        self.previous_potential = potential
        self.step_count += 1
        strain = self.compute_strain()
        success = potential <= self.potential_tolerance and not collision
        done = collision or success or strain > self.strain_tolerance

        if collision:
            # Make the collision step unambiguously negative and stop immediately.
            reward = -self.collision_penalty
        elif success:
            reward += 10.0
        elif strain > self.strain_tolerance:
            reward -= 10.0 * min(strain / self.strain_tolerance, 10.0)
        if self.step_count >= self.max_steps and not done:
            reward -= 2.0 * potential_ratio
            done = True

        info = {
            "gravity_potential": potential,
            "strain": strain,
            "success": success,
            "collision": collision,
            "mode_coefficients": coefficients.copy(),
        }
        return self.get_state(), float(reward), done, info


class ActorCritic(nn.Module):
    def __init__(self, state_size, mode_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size + mode_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mu = nn.Linear(64, 1)
        self.actor_log_std = nn.Linear(64, 1)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.actor_log_std.bias, -1.0)

    def action_distribution(self, state, modes):
        num_modes = modes.shape[0]
        if num_modes == 0:
            empty = state.new_empty((0,))
            return empty, empty
        repeated_state = state.repeat(num_modes, 1)
        hidden = self.actor(torch.cat([repeated_state, modes], dim=1))
        mean = 1.5 * torch.tanh(self.actor_mu(hidden).squeeze(-1))
        log_std = torch.clamp(
            self.actor_log_std(hidden).squeeze(-1), -2.5, 0.2
        )
        return mean, torch.exp(log_std)

    def value(self, state):
        return self.critic(state).squeeze(-1)


@dataclass
class Transition:
    reward: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class Agent:
    def __init__(self, state_size, mode_size, gamma=0.97, learning_rate=3e-4,
                 entropy_coefficient=0.02, value_coefficient=0.5):
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.model = ActorCritic(state_size, mode_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.finished_episodes = []
        self.current_episode = []

    def get_action(self, state, modes, deterministic=False):
        state_tensor = torch.as_tensor(
            state[np.newaxis, :], dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        mean, std = self.model.action_distribution(state_tensor, mode_tensor)
        value = self.model.value(state_tensor).squeeze(0)
        if mean.numel() == 0:
            zero = torch.tensor(0.0)
            return np.zeros(0, dtype=np.float32), zero, value, zero

        distribution = Normal(mean, std)
        raw_action = mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(raw_action).sum()
        entropy = distribution.entropy().sum()
        action = torch.tanh(raw_action)
        return (
            action.detach().cpu().numpy(),
            log_probability,
            value,
            entropy,
        )

    def add_transition(self, reward, log_probability, value, entropy):
        self.current_episode.append(
            Transition(reward, log_probability, value, entropy)
        )

    def finish_episode(self):
        if self.current_episode:
            self.finished_episodes.append(self.current_episode)
        self.current_episode = []

    def add_terminal_bonuses(self, bonuses):
        """Add a set-level bonus to the terminal transition of recent episodes."""
        bonuses = np.asarray(bonuses, dtype=float)
        if len(bonuses) > len(self.finished_episodes):
            raise ValueError("More bonuses than finished episodes.")
        start = len(self.finished_episodes) - len(bonuses)
        for offset, bonus in enumerate(bonuses):
            episode = self.finished_episodes[start + offset]
            if not episode:
                continue
            terminal = episode[-1]
            episode[-1] = Transition(
                terminal.reward + float(bonus),
                terminal.log_probability,
                terminal.value,
                terminal.entropy,
            )

    def update(self):
        if not self.finished_episodes:
            return 0.0

        returns = []
        log_probabilities = []
        values = []
        entropies = []
        for episode in self.finished_episodes:
            discounted_return = 0.0
            episode_returns = []
            for transition in reversed(episode):
                discounted_return = (
                    transition.reward + self.gamma * discounted_return
                )
                episode_returns.append(discounted_return)
            episode_returns.reverse()
            for transition, discounted_return in zip(episode, episode_returns):
                returns.append(discounted_return)
                log_probabilities.append(transition.log_probability)
                values.append(transition.value)
                entropies.append(transition.entropy)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        log_probability_tensor = torch.stack(log_probabilities)
        value_tensor = torch.stack(values)
        entropy_tensor = torch.stack(entropies)

        advantages = returns_tensor - value_tensor.detach()
        if advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss = -(log_probability_tensor * advantages).mean()
        critic_loss = nn.functional.mse_loss(value_tensor, returns_tensor)
        entropy_loss = -entropy_tensor.mean()
        loss = (
            actor_loss
            + self.value_coefficient * critic_loss
            + self.entropy_coefficient * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.finished_episodes = []
        return float(loss.detach())


def moving_average(values, window):
    if len(values) < window:
        return np.array([])
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def with_diversity_code(state, set_member_id, episodes_per_set):
    """Append an episode-persistent, target-free code on the unit circle."""
    angle = 2.0 * np.pi * set_member_id / episodes_per_set
    code = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    return np.concatenate([state, code])


def load_policy(agent, policy_path, base_state_size, mode_size):
    """Load a policy, expanding pre-diversity input layers when necessary."""
    saved = torch.load(policy_path, weights_only=True)
    current = agent.model.state_dict()
    actor_key = "actor.0.weight"
    critic_key = "critic.0.weight"
    if saved[actor_key].shape != current[actor_key].shape:
        old_actor = saved[actor_key]
        new_actor = current[actor_key].clone()
        old_state_size = old_actor.shape[1] - mode_size
        new_state_size = new_actor.shape[1] - mode_size
        if old_state_size != base_state_size or new_state_size != base_state_size + 2:
            raise ValueError("Initial policy has incompatible actor input dimensions.")
        new_actor[:, :base_state_size] = old_actor[:, :base_state_size]
        new_actor[:, new_state_size:] = old_actor[:, old_state_size:]
        new_actor[:, base_state_size:new_state_size] = 0.0
        saved[actor_key] = new_actor

        old_critic = saved[critic_key]
        new_critic = current[critic_key].clone()
        if old_critic.shape[1] != base_state_size:
            raise ValueError("Initial policy has incompatible critic input dimensions.")
        new_critic[:, :base_state_size] = old_critic
        new_critic[:, base_state_size:] = 0.0
        saved[critic_key] = new_critic
    agent.model.load_state_dict(saved)


def plot_series_with_average(values, window, ylabel, title, path):
    values = np.asarray(values, dtype=float)
    figure, axis = plt.subplots()
    axis.plot(values, color="0.65", alpha=0.45, label="Episode value")
    average = moving_average(values, window)
    if len(average):
        x = np.arange(len(average)) + window - 1
        axis.plot(x, average, color="black", linewidth=2, label=f"MA ({window})")
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_node_history(directory, node_history):
    directory.mkdir(parents=True, exist_ok=True)
    for old_path in directory.glob("*.dat"):
        if old_path.stem.isdigit():
            old_path.unlink()
    for step, node in enumerate(node_history):
        np.savetxt(directory / f"{step}.dat", node, fmt="%.8f")


def save_mode_history(path, coefficient_history):
    max_modes = max((len(item) for item in coefficient_history), default=0)
    table = np.full((len(coefficient_history), max_modes + 1), np.nan)
    if len(table):
        table[:, 0] = np.arange(len(coefficient_history))
        for step, coefficients in enumerate(coefficient_history):
            table[step, 1:1 + len(coefficients)] = coefficients
    header = "step" + "".join(f",mode_{i}" for i in range(max_modes))
    np.savetxt(path, table, delimiter=",", header=header, comments="")


def pairwise_rmsd(shapes):
    if len(shapes) == 0:
        return np.empty((0, 0))
    flattened = np.asarray(shapes).reshape(len(shapes), -1)
    differences = flattened[:, None, :] - flattened[None, :, :]
    return np.sqrt(np.mean(differences ** 2, axis=2))


def connected_component_labels(distance_matrix, threshold):
    """Cluster shapes as connected components of the RMSD threshold graph."""
    num_shapes = len(distance_matrix)
    labels = -np.ones(num_shapes, dtype=int)
    next_label = 0
    for start in range(num_shapes):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            current = stack.pop()
            neighbors = np.where(distance_matrix[current] <= threshold)[0]
            for neighbor in neighbors:
                if labels[neighbor] < 0:
                    labels[neighbor] = next_label
                    stack.append(int(neighbor))
        next_label += 1
    return labels


def greedy_cluster_labels(distance_matrix, threshold):
    """Assign shapes to threshold balls around stable, first-seen representatives."""
    labels = -np.ones(len(distance_matrix), dtype=int)
    representatives = []
    for shape_index in range(len(distance_matrix)):
        assigned_label = None
        for label, representative in enumerate(representatives):
            if distance_matrix[shape_index, representative] <= threshold:
                assigned_label = label
                break
        if assigned_label is None:
            representatives.append(shape_index)
            assigned_label = len(representatives) - 1
        labels[shape_index] = assigned_label
    return labels


def set_diversity_bonuses(final_nodes, final_potentials, collision_flags,
                          potential_threshold, cluster_threshold,
                          novelty_clip, novelty_weight, coverage_weight):
    """Reward distinct, collision-free, nearly flat outcomes within one set."""
    final_nodes = np.asarray(final_nodes)
    final_potentials = np.asarray(final_potentials, dtype=float)
    collision_flags = np.asarray(collision_flags, dtype=bool)
    bonuses = np.zeros(len(final_nodes), dtype=float)
    eligible = (
        (final_potentials <= potential_threshold)
        & ~collision_flags
    )
    eligible_indices = np.where(eligible)[0]
    metrics = {
        "eligible_count": int(len(eligible_indices)),
        "cluster_count": 0,
        "mean_nearest_rmsd": 0.0,
        "mean_pairwise_rmsd": 0.0,
        "mean_bonus": 0.0,
        "max_bonus": 0.0,
    }
    if len(eligible_indices) < 2:
        return bonuses, metrics

    eligible_nodes = final_nodes[eligible_indices]
    distances = pairwise_rmsd(eligible_nodes)
    labels = greedy_cluster_labels(distances, cluster_threshold)
    cluster_count = len(np.unique(labels))
    nearest_distances = distances.copy()
    np.fill_diagonal(nearest_distances, np.inf)
    nearest_distances = np.min(nearest_distances, axis=1)
    mean_distances = np.sum(distances, axis=1) / (len(distances) - 1)

    quality = np.sqrt(np.clip(
        1.0 - final_potentials[eligible_indices] / potential_threshold,
        0.0, 1.0,
    ))
    diversity_score = np.minimum(
        0.5 * (nearest_distances + mean_distances) / novelty_clip,
        1.0,
    )
    coverage_bonus = coverage_weight * max(cluster_count - 1, 0)
    eligible_bonuses = quality * (
        novelty_weight * diversity_score + coverage_bonus
    )
    bonuses[eligible_indices] = eligible_bonuses

    metrics.update({
        "cluster_count": int(cluster_count),
        "mean_nearest_rmsd": float(np.mean(nearest_distances)),
        "mean_pairwise_rmsd": float(np.mean(mean_distances)),
        "mean_bonus": float(np.mean(eligible_bonuses)),
        "max_bonus": float(np.max(eligible_bonuses)),
    })
    return bonuses, metrics


def plot_set_diversity(set_history, path):
    if not set_history:
        return
    history = np.asarray(set_history, dtype=float)
    figure, axis = plt.subplots()
    axis.plot(history[:, 2], color="tab:blue", label="Eligible flat outcomes")
    axis.plot(history[:, 3], color="tab:orange", label="Shape clusters")
    axis.set_xlabel("Set")
    axis.set_ylabel("Count per set")
    axis.set_title("Set-level Shape Diversity")
    axis.grid(True)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_training_checkpoint(path, completed_episodes, agent, training_rng,
                             reward_history, potential_history,
                             diversity_bonus_history, set_history, loss_history):
    torch.save({
        "completed_episodes": int(completed_episodes),
        "model": agent.model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "numpy_rng_state": training_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "reward_history": reward_history,
        "potential_history": potential_history,
        "diversity_bonus_history": diversity_bonus_history,
        "set_history": set_history,
        "loss_history": loss_history,
    }, path)


def evaluate_frozen_policy(agent, env, num_episodes, output_directory,
                           cluster_threshold, episodes_per_set,
                           deterministic=False):
    """Evaluate without optimizer updates and save every randomized folding path."""
    evaluation_directory = output_directory / "evaluation_paths"
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    agent.model.eval()

    if num_episodes <= 0:
        return {
            "success_count": 0,
            "num_clusters": 0,
            "cluster_sizes": np.array([], dtype=int),
            "flat_count": 0,
            "num_flat_clusters": 0,
            "flat_cluster_sizes": np.array([], dtype=int),
            "final_potentials": np.array([], dtype=float),
            "collision_count": 0,
        }

    final_nodes = []
    final_potentials = []
    success_flags = []
    collision_flags = []
    with torch.no_grad():
        for episode in range(num_episodes):
            set_member_id = episode % episodes_per_set
            diversity_code = with_diversity_code(
                np.empty(0, dtype=np.float32),
                set_member_id,
                episodes_per_set,
            )
            state = with_diversity_code(
                env.reset(), set_member_id, episodes_per_set
            )
            done = False
            node_history = [env.node.copy()]
            coefficient_history = []
            info = {
                "gravity_potential": env.initial_potential,
                "success": False,
            }
            while not done:
                modes = env.get_modes()
                action, _, _, _ = agent.get_action(
                    state, modes, deterministic=deterministic
                )
                next_state, _, done, info = env.step(action)
                state = with_diversity_code(
                    next_state, set_member_id, episodes_per_set
                )
                node_history.append(env.node.copy())
                coefficient_history.append(info["mode_coefficients"])

            episode_directory = evaluation_directory / f"episode_{episode:03d}"
            save_node_history(episode_directory, node_history)
            save_mode_history(
                episode_directory / "mode_coefficients.csv",
                coefficient_history,
            )
            np.savetxt(
                episode_directory / "diversity_code.csv",
                diversity_code[np.newaxis, :],
                delimiter=",",
                header="code_cos,code_sin",
                comments="",
            )
            final_nodes.append(env.node.copy())
            final_potentials.append(info["gravity_potential"])
            success_flags.append(info["success"])
            collision_flags.append(info["collision"])

    final_nodes = np.asarray(final_nodes)
    final_potentials = np.asarray(final_potentials)
    success_flags = np.asarray(success_flags, dtype=bool)
    collision_flags = np.asarray(collision_flags, dtype=bool)
    flat_flags = final_potentials <= env.potential_tolerance
    flat_nodes = final_nodes[flat_flags]
    successful_nodes = final_nodes[success_flags]
    if len(flat_nodes):
        flat_distances = pairwise_rmsd(flat_nodes)
        flat_labels = connected_component_labels(
            flat_distances, cluster_threshold
        )
        flat_cluster_sizes = np.bincount(flat_labels)
        num_flat_clusters = len(flat_cluster_sizes)
    else:
        flat_cluster_sizes = np.array([], dtype=int)
        num_flat_clusters = 0
    if len(successful_nodes):
        distances = pairwise_rmsd(successful_nodes)
        labels = connected_component_labels(distances, cluster_threshold)
        cluster_sizes = np.bincount(labels)
        num_clusters = len(cluster_sizes)
    else:
        cluster_sizes = np.array([], dtype=int)
        num_clusters = 0

    print("===================================")
    print("Frozen-policy evaluation")
    print(f"Randomized cases       : {num_episodes}")
    print(f"Diversity codes        : {episodes_per_set}")
    print(f"Deterministic actions  : {deterministic}")
    print(f"Potential <= tolerance : {flat_flags.sum()} / {num_episodes} "
          f"(tolerance={env.potential_tolerance:g})")
    print(f"Flat shape types        : {num_flat_clusters} "
          f"(RMSD <= {cluster_threshold:g})")
    if len(flat_cluster_sizes):
        print(f"Flat cluster sizes      : {flat_cluster_sizes.tolist()}")
    print(f"Collision-free flat     : {success_flags.sum()} / {num_episodes}")
    print(f"Collision terminations  : {collision_flags.sum()} / {num_episodes}")
    print(f"Valid flat shape types  : {num_clusters} (RMSD <= {cluster_threshold:g})")
    if len(cluster_sizes):
        print(f"Cluster sizes          : {cluster_sizes.tolist()}")
    print(
        f"Final potential        : mean={final_potentials.mean():.6g}, "
        f"min={final_potentials.min():.6g}, max={final_potentials.max():.6g}"
    )
    print(f"Saved paths            : {evaluation_directory}")
    print("===================================")
    return {
        "success_count": int(success_flags.sum()),
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "flat_count": int(flat_flags.sum()),
        "num_flat_clusters": num_flat_clusters,
        "flat_cluster_sizes": flat_cluster_sizes,
        "final_potentials": final_potentials,
        "collision_count": int(collision_flags.sum()),
    }


def train(args):
    if args.episodes_per_set < 2:
        raise ValueError("--episodes-per-set must be at least 2.")
    if args.sets_per_batch < 1:
        raise ValueError("--sets-per-batch must be at least 1.")
    if args.diversity_potential_threshold <= 0.0:
        raise ValueError("--diversity-potential-threshold must be positive.")
    if args.diversity_potential_start < args.diversity_potential_threshold:
        raise ValueError(
            "--diversity-potential-start must be at least the final threshold."
        )
    if args.novelty_clip <= 0.0:
        raise ValueError("--novelty-clip must be positive.")

    script_directory = Path(__file__).resolve().parent
    data_directory = (
        Path(args.data_dir).resolve() if args.data_dir else script_directory
    )
    output_directory = Path(args.output_dir)
    if not output_directory.is_absolute():
        output_directory = script_directory / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Small networks are considerably faster without a large CPU thread pool.
    torch.set_num_threads(1)
    training_rng = np.random.default_rng(args.seed)

    base_node = np.atleast_2d(np.loadtxt(
        data_directory / "node_initial.dat", delimiter=",", dtype=float
    ))
    member = np.atleast_2d(np.loadtxt(
        data_directory / "member.dat", delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(
        data_directory / "fix.dat", delimiter=",", dtype=int
    ))

    environment = FoldingEnv(
        base_node,
        member,
        fix,
        training_rng,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        imperfection_scale=args.imperfection_scale,
        reference_point=args.reference_point,
        potential_tolerance=args.potential_tolerance,
        strain_tolerance=args.strain_tolerance,
        collision_tolerance=args.collision_tolerance,
        collision_substeps=args.collision_substeps,
        collision_penalty=args.collision_penalty,
    )
    initial_state = environment.reset()
    coded_initial_state = with_diversity_code(
        initial_state, 0, args.episodes_per_set
    )
    agent = Agent(
        state_size=len(coded_initial_state),
        mode_size=base_node.size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy_coefficient,
    )
    if args.initial_policy and args.resume_checkpoint:
        raise ValueError("Use either --initial-policy or --resume-checkpoint, not both.")
    if args.initial_policy:
        policy_path = Path(args.initial_policy)
        if not policy_path.is_absolute():
            policy_path = script_directory / policy_path
        load_policy(agent, policy_path, len(initial_state), base_node.size)
        print(f"Loaded initial policy: {policy_path}")

    reward_history = []
    terminal_potential_history = []
    diversity_bonus_history = []
    set_history = []
    loss_history = []
    start_episode = 0
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = script_directory / checkpoint_path
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.model.load_state_dict(checkpoint["model"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        training_rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        reward_history = list(checkpoint["reward_history"])
        terminal_potential_history = list(checkpoint["potential_history"])
        diversity_bonus_history = list(checkpoint["diversity_bonus_history"])
        set_history = list(checkpoint["set_history"])
        loss_history = list(checkpoint["loss_history"])
        start_episode = int(checkpoint["completed_episodes"])
        print(f"Resumed checkpoint: {checkpoint_path} at episode {start_episode}")
    set_final_nodes = []
    set_final_potentials = []
    set_collision_flags = []
    sets_in_batch = 0
    latest_set_metrics = {
        "eligible_count": 0,
        "cluster_count": 0,
        "mean_bonus": 0.0,
    }
    diversity_threshold = args.diversity_potential_start
    print(
        f"Training with {args.episodes_per_set} episodes/set, "
        f"{args.sets_per_batch} sets/minibatch "
        f"({args.episodes_per_set * args.sets_per_batch} episodes/update)"
    )
    checkpoint_path = output_directory / "training_checkpoint.pt"
    for episode in range(start_episode, args.episodes):
        set_member_id = episode % args.episodes_per_set
        state = with_diversity_code(
            environment.reset(), set_member_id, args.episodes_per_set
        )
        done = False
        total_reward = 0.0
        info = {
            "gravity_potential": environment.initial_potential,
            "success": False,
        }
        while not done:
            modes = environment.get_modes()
            action, log_probability, value, entropy = agent.get_action(
                state, modes
            )
            next_state, reward, done, info = environment.step(action)
            agent.add_transition(reward, log_probability, value, entropy)
            state = with_diversity_code(
                next_state, set_member_id, args.episodes_per_set
            )
            total_reward += reward

        agent.finish_episode()
        reward_history.append(total_reward)
        terminal_potential_history.append(info["gravity_potential"])
        diversity_bonus_history.append(0.0)
        set_final_nodes.append(environment.node.copy())
        set_final_potentials.append(info["gravity_potential"])
        set_collision_flags.append(info["collision"])

        set_finished = (
            len(set_final_nodes) == args.episodes_per_set
            or episode + 1 == args.episodes
        )
        if set_finished:
            training_fraction = episode / max(args.episodes - 1, 1)
            diversity_threshold = (
                args.diversity_potential_start
                + training_fraction * (
                    args.diversity_potential_threshold
                    - args.diversity_potential_start
                )
            )
            bonuses, latest_set_metrics = set_diversity_bonuses(
                set_final_nodes,
                set_final_potentials,
                set_collision_flags,
                diversity_threshold,
                args.diversity_cluster_threshold,
                args.novelty_clip,
                args.diversity_weight,
                args.set_coverage_weight,
            )
            agent.add_terminal_bonuses(bonuses)
            history_start = len(reward_history) - len(bonuses)
            for offset, bonus in enumerate(bonuses):
                reward_history[history_start + offset] += float(bonus)
                diversity_bonus_history[history_start + offset] = float(bonus)

            set_history.append([
                len(set_history),
                len(set_final_nodes),
                latest_set_metrics["eligible_count"],
                latest_set_metrics["cluster_count"],
                latest_set_metrics["mean_nearest_rmsd"],
                latest_set_metrics["mean_pairwise_rmsd"],
                latest_set_metrics["mean_bonus"],
                latest_set_metrics["max_bonus"],
                float(np.mean(set_final_potentials)),
                diversity_threshold,
            ])
            set_final_nodes = []
            set_final_potentials = []
            set_collision_flags = []
            sets_in_batch += 1

            if sets_in_batch == args.sets_per_batch:
                loss_history.append(agent.update())
                sets_in_batch = 0

                if (
                    args.checkpoint_every > 0
                    and (episode + 1) % args.checkpoint_every == 0
                ):
                    save_training_checkpoint(
                        checkpoint_path,
                        episode + 1,
                        agent,
                        training_rng,
                        reward_history,
                        terminal_potential_history,
                        diversity_bonus_history,
                        set_history,
                        loss_history,
                    )

        if args.print_every > 0 and (
            episode == 0 or (episode + 1) % args.print_every == 0
        ):
            start = max(0, len(reward_history) - args.moving_average_window)
            recent_rewards = reward_history[start:]
            recent_potentials = terminal_potential_history[start:]
            print(
                f"Episode {episode + 1:5d}/{args.episodes} "
                f"| reward_MA {np.mean(recent_rewards):8.3f} "
                f"| terminal_potential_MA {np.mean(recent_potentials):.6f} "
                f"| set_flat {latest_set_metrics['eligible_count']} "
                f"| set_clusters {latest_set_metrics['cluster_count']} "
                f"| set_bonus {latest_set_metrics['mean_bonus']:.3f} "
                f"| diversity_limit {diversity_threshold:.3f}"
            )

    if agent.finished_episodes:
        loss_history.append(agent.update())

    torch.save(agent.model.state_dict(), output_directory / "policy.pt")
    np.savetxt(
        output_directory / "reward_history.csv",
        np.asarray(reward_history),
        delimiter=",",
        header="total_reward",
        comments="",
    )
    np.savetxt(
        output_directory / "terminal_gravity_potential_history.csv",
        np.asarray(terminal_potential_history),
        delimiter=",",
        header="terminal_gravity_potential",
        comments="",
    )
    np.savetxt(
        output_directory / "loss_history.csv",
        np.asarray(loss_history),
        delimiter=",",
        header="actor_critic_loss",
        comments="",
    )
    np.savetxt(
        output_directory / "diversity_bonus_history.csv",
        np.asarray(diversity_bonus_history),
        delimiter=",",
        header="terminal_diversity_bonus",
        comments="",
    )
    np.savetxt(
        output_directory / "set_diversity_history.csv",
        np.asarray(set_history),
        delimiter=",",
        header=(
            "set_id,set_size,eligible_flat_count,shape_cluster_count,"
            "mean_nearest_rmsd,mean_pairwise_rmsd,mean_bonus,max_bonus,"
            "mean_terminal_potential,diversity_potential_limit"
        ),
        comments="",
    )
    plot_series_with_average(
        reward_history,
        args.moving_average_window,
        "Cumulative reward",
        "Learning Curve",
        output_directory / "learning_curve.png",
    )
    plot_series_with_average(
        terminal_potential_history,
        args.moving_average_window,
        "Terminal gravity potential",
        "Terminal Gravity Potential",
        output_directory / "terminal_gravity_potential.png",
    )
    plot_series_with_average(
        diversity_bonus_history,
        args.moving_average_window,
        "Terminal diversity bonus",
        "Diversity Bonus",
        output_directory / "diversity_bonus.png",
    )
    plot_set_diversity(
        set_history,
        output_directory / "set_diversity.png",
    )

    # Use an independent random stream and keep the optimizer frozen for testing.
    evaluation_rng = np.random.default_rng(args.seed + 1_000_003)
    torch.manual_seed(args.seed + 2_000_003)
    evaluation_environment = FoldingEnv(
        base_node,
        member,
        fix,
        evaluation_rng,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        imperfection_scale=args.imperfection_scale,
        reference_point=args.reference_point,
        potential_tolerance=args.potential_tolerance,
        strain_tolerance=args.strain_tolerance,
        collision_tolerance=args.collision_tolerance,
        collision_substeps=args.collision_substeps,
        collision_penalty=args.collision_penalty,
    )
    return evaluate_frozen_policy(
        agent,
        evaluation_environment,
        args.eval_episodes,
        output_directory,
        args.cluster_threshold,
        args.episodes_per_set,
        args.eval_deterministic,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--episodes-per-set", type=int, default=10)
    parser.add_argument("--sets-per-batch", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--imperfection-scale", type=float, default=0.10)
    parser.add_argument("--reference-point", type=float, default=0.0)
    parser.add_argument("--potential-tolerance", type=float, default=0.05)
    parser.add_argument("--strain-tolerance", type=float, default=1e-8)
    parser.add_argument("--collision-tolerance", type=float, default=0.02)
    parser.add_argument("--collision-substeps", type=int, default=10)
    parser.add_argument("--collision-penalty", type=float, default=20.0)
    parser.add_argument("--cluster-threshold", type=float, default=0.05)
    parser.add_argument("--diversity-potential-start", type=float, default=2.0)
    parser.add_argument("--diversity-potential-threshold", type=float, default=0.50)
    parser.add_argument("--diversity-cluster-threshold", type=float, default=0.08)
    parser.add_argument("--novelty-clip", type=float, default=0.60)
    parser.add_argument("--diversity-weight", type=float, default=20.0)
    parser.add_argument("--set-coverage-weight", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.02)
    parser.add_argument("--moving-average-window", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=str, default="gravity_diversity_actor_critic_results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
