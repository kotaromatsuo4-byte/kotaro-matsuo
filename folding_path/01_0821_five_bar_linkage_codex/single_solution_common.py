"""Shared fixed-start actor-critic training for the two step-width policies."""

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

from ori_codex import (
    connected_component_labels,
    dedx,
    index_free,
    mode,
    nonadjacent_bar_pairs,
    pairwise_rmsd,
    project_to_constraints,
)


def compute_flatness_potential(node, subspace_dimension):
    """Squared distance to the best affine line (1) or plane (2)."""
    coordinates = np.asarray(node, dtype=float)
    if subspace_dimension not in (1, 2):
        raise ValueError("subspace_dimension must be 1 (line) or 2 (plane).")
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return float(np.sum(singular_values[subspace_dimension:] ** 2))


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation_polynomial(a0, a1, b0, b1, c0, c1):
    """Coefficients q2, q1, q0 for orient(a(t), b(t), c(t))."""
    ab0 = b0 - a0
    ac0 = c0 - a0
    delta_ab = (b1 - b0) - (a1 - a0)
    delta_ac = (c1 - c0) - (a1 - a0)
    return np.array([
        _cross_2d(delta_ab, delta_ac),
        _cross_2d(delta_ab, ac0) + _cross_2d(ab0, delta_ac),
        _cross_2d(ab0, ac0),
    ])


def _roots_in_open_unit_interval(coefficients, tolerance):
    first = 0
    while first < len(coefficients) - 1 and abs(coefficients[first]) <= tolerance:
        first += 1
    polynomial = coefficients[first:]
    if len(polynomial) <= 1:
        return []
    roots = np.roots(polynomial)
    return [
        float(root.real) for root in roots
        if abs(root.imag) <= tolerance
        and tolerance < root.real < 1.0 - tolerance
    ]


def _proper_segment_intersection(a, b, c, d, tolerance):
    orient_ab_c = _cross_2d(b - a, c - a)
    orient_ab_d = _cross_2d(b - a, d - a)
    orient_cd_a = _cross_2d(d - c, a - c)
    orient_cd_b = _cross_2d(d - c, b - c)

    def opposite(first, second):
        return (
            (first > tolerance and second < -tolerance)
            or (first < -tolerance and second > tolerance)
        )

    return opposite(orient_ab_c, orient_ab_d) and opposite(
        orient_cd_a, orient_cd_b
    )


def has_proper_self_crossing(node_before, node_after, member, bar_pairs,
                             tolerance=1e-10):
    """Detect a proper x-z segment crossing during an open motion interval.

    Collinear overlap and endpoint contact are intentionally not crossings.
    Node motion between the corrected endpoint configurations is linear.
    """
    node_before = np.asarray(node_before, dtype=float)
    node_after = np.asarray(node_after, dtype=float)
    y_values = np.concatenate([node_before[:, 1], node_after[:, 1]])
    if np.ptp(y_values) > tolerance:
        raise ValueError(
            "Proper self-crossing detection is implemented for the current "
            "x-z planar linkage only."
        )
    before = node_before[:, [0, 2]]
    after = node_after[:, [0, 2]]
    for first, second in bar_pairs:
        a_index, b_index = member[first]
        c_index, d_index = member[second]
        point_indices = (a_index, b_index, c_index, d_index)
        polynomials = (
            _orientation_polynomial(
                before[a_index], after[a_index],
                before[b_index], after[b_index],
                before[c_index], after[c_index],
            ),
            _orientation_polynomial(
                before[a_index], after[a_index],
                before[b_index], after[b_index],
                before[d_index], after[d_index],
            ),
            _orientation_polynomial(
                before[c_index], after[c_index],
                before[d_index], after[d_index],
                before[a_index], after[a_index],
            ),
            _orientation_polynomial(
                before[c_index], after[c_index],
                before[d_index], after[d_index],
                before[b_index], after[b_index],
            ),
        )
        breakpoints = [0.0, 1.0]
        for polynomial in polynomials:
            breakpoints.extend(
                _roots_in_open_unit_interval(polynomial, tolerance)
            )
        sorted_breakpoints = sorted(breakpoints)
        breakpoints = [sorted_breakpoints[0]]
        for value in sorted_breakpoints[1:]:
            if value - breakpoints[-1] > tolerance:
                breakpoints.append(value)
        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            if right - left <= tolerance:
                continue
            fraction = 0.5 * (left + right)
            points = [
                before[index] + fraction * (after[index] - before[index])
                for index in point_indices
            ]
            if _proper_segment_intersection(*points, tolerance=tolerance):
                return True
    return False


def current_strain(node, member, lengths):
    current_lengths = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    return float(np.sum((current_lengths - lengths) ** 2))


def canonical_modes(node, member, free_index):
    """Return SVD modes with deterministic column signs."""
    modes = mode(free_index, dedx(node, member, free_index), len(node))
    for column in range(modes.shape[1]):
        pivot = int(np.argmax(np.abs(modes[:, column])))
        if modes[pivot, column] < 0.0:
            modes[:, column] *= -1.0
    return modes


class FixedStartEnv:
    def __init__(self, node, member, fix, method, max_steps, max_step_width,
                 flatness_subspace_dimension, flatness_tolerance,
                 strain_tolerance, crossing_tolerance, line_search_shrink,
                 line_search_min_width):
        self.node0 = node.copy()
        self.node = node.copy()
        self.member = member
        self.fix = fix
        self.method = method
        self.num_node = len(node)
        self.free = index_free(self.num_node, fix)
        self.lengths = np.linalg.norm(
            node[member[:, 0]] - node[member[:, 1]], axis=1
        )
        self.mode_size = canonical_modes(node, member, self.free).shape[1]
        self.max_steps = max_steps
        self.max_step_width = max_step_width
        self.flatness_subspace_dimension = flatness_subspace_dimension
        self.flatness_tolerance = flatness_tolerance
        self.strain_tolerance = strain_tolerance
        self.crossing_tolerance = crossing_tolerance
        self.line_search_shrink = line_search_shrink
        self.line_search_min_width = line_search_min_width
        self.crossing_pairs = nonadjacent_bar_pairs(member)
        self.coordinate_scale = max(float(np.max(self.lengths)), 1e-8)
        self.initial_potential = compute_flatness_potential(
            node, flatness_subspace_dimension
        )
        if self.initial_potential <= 1e-12:
            raise ValueError(
                "The initial flatness potential is zero. Choose a lower "
                "fitting-subspace dimension for this structure."
            )
        self.previous_potential = self.initial_potential
        self.step_count = 0

    def reset(self):
        self.node = self.node0.copy()
        self.previous_potential = self.initial_potential
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        coordinates = (self.node / self.coordinate_scale).flatten()
        potential_ratio = self.previous_potential / self.initial_potential
        time_ratio = self.step_count / self.max_steps
        return np.concatenate([
            coordinates,
            np.array([potential_ratio, time_ratio]),
        ]).astype(np.float32)

    def get_modes(self):
        return canonical_modes(self.node, self.member, self.free)

    def _candidate(self, modes, direction, width):
        used = np.zeros(modes.shape[1], dtype=float)
        count = min(len(direction), len(used))
        used[:count] = direction[:count]
        norm = np.linalg.norm(used)
        if norm > 1e-12:
            used /= norm
        predicted = self.node + (modes @ (width * used)).reshape(
            self.num_node, 3
        )
        corrected = project_to_constraints(
            predicted, self.member, self.lengths, self.free
        )
        return corrected, used

    def _line_search(self, modes, direction, initial_width):
        width = min(float(initial_width), self.max_step_width)
        best = self.node.copy()
        used = np.zeros(modes.shape[1], dtype=float)
        failed = True
        while width >= self.line_search_min_width:
            candidate, used = self._candidate(modes, direction, width)
            strain = current_strain(candidate, self.member, self.lengths)
            potential = compute_flatness_potential(
                candidate, self.flatness_subspace_dimension
            )
            if (
                strain <= self.strain_tolerance
                and potential < self.previous_potential - 1e-10
            ):
                best = candidate
                failed = False
                break
            width *= self.line_search_shrink
        if failed:
            width = 0.0
        return best, used, width, failed

    def step(self, direction, requested_width=None):
        node_before = self.node.copy()
        modes = self.get_modes()
        if self.method == "line_search":
            node_after, used, width, search_failed = self._line_search(
                modes, direction, self.max_step_width
            )
        else:
            requested_width = float(np.clip(
                requested_width, 0.0, self.max_step_width
            ))
            node_after, used, width, search_failed = self._line_search(
                modes, direction, requested_width
            )

        strain = current_strain(node_after, self.member, self.lengths)
        correction_failed = strain > self.strain_tolerance
        self_crossing = has_proper_self_crossing(
            node_before,
            node_after,
            self.member,
            self.crossing_pairs,
            self.crossing_tolerance,
        ) if width > 0.0 else False
        self.node = node_after
        potential = compute_flatness_potential(
            self.node, self.flatness_subspace_dimension
        )
        progress = (
            self.previous_potential - potential
        ) / self.initial_potential
        potential_ratio = potential / self.initial_potential
        reward = 12.0 * progress - 0.10 * potential_ratio

        self.previous_potential = potential
        self.step_count += 1
        success = (
            potential_ratio <= self.flatness_tolerance
            and not self_crossing
            and not correction_failed
        )
        done = (
            self_crossing
            or correction_failed
            or success
            or self.step_count >= self.max_steps
        )

        info = {
            "flatness_potential": potential,
            "flatness_ratio": potential_ratio,
            "strain": strain,
            "success": success,
            "self_crossing": self_crossing,
            "correction_failed": correction_failed,
            "line_search_failed": search_failed,
            "step_width": width,
            "mode_coefficients": used.copy(),
        }
        return self.get_state(), float(reward), done, info


class LineSearchActorCritic(nn.Module):
    def __init__(self, state_size, mode_size):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, mode_size)
        self.actor_variance = nn.Linear(64, mode_size)
        nn.init.zeros_(self.actor_variance.weight)
        nn.init.constant_(self.actor_variance.bias, -2.0)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def distribution_parameters(self, state):
        hidden = self.actor_body(state)
        mean = torch.tanh(self.actor_mean(hidden)).squeeze(0)
        variance = (2.0 * torch.sigmoid(
            self.actor_variance(hidden)
        )).squeeze(0).clamp(min=1e-4, max=2.0)
        return mean, variance

    def value(self, state):
        return self.critic(state).squeeze()


class LearnedStepActorCritic(nn.Module):
    """Actor: Linear(state, modes) -> Tanh -> Linear(modes, 2)."""
    def __init__(self, state_size, mode_size):
        super().__init__()
        if mode_size != 2:
            raise ValueError(
                "The learned-step actor expects exactly two mechanism modes."
            )
        self.actor_hidden = nn.Sequential(
            nn.Linear(state_size, mode_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(mode_size, 2)
        self.actor_variance = nn.Linear(mode_size, 2)
        nn.init.zeros_(self.actor_variance.weight)
        nn.init.constant_(self.actor_variance.bias, -2.0)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def distribution_parameters(self, state):
        hidden = self.actor_hidden(state)
        mean = torch.tanh(self.actor_mean(hidden)).squeeze(0)
        variance = (2.0 * torch.sigmoid(
            self.actor_variance(hidden)
        )).squeeze(0).clamp(min=1e-4, max=2.0)
        return mean, variance

    def value(self, state):
        return self.critic(state).squeeze()


@dataclass
class Transition:
    reward: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class Agent:
    def __init__(self, model, method, mode_size, max_step_width, gamma,
                 learning_rate, entropy_coefficient, value_coefficient=0.5):
        self.model = model
        self.method = method
        self.mode_size = mode_size
        self.max_step_width = max_step_width
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.episodes = []
        self.current_episode = []
        self.last_mean = None
        self.last_variance = None

    def get_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(
            state[np.newaxis, :], dtype=torch.float32
        )
        mean, variance = self.model.distribution_parameters(state_tensor)
        self.last_mean = mean.detach().cpu().numpy().copy()
        self.last_variance = variance.detach().cpu().numpy().copy()
        std = torch.sqrt(variance)
        distribution = Normal(mean, std)
        raw = mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(raw).sum()
        entropy = distribution.entropy().sum()
        value = self.model.value(state_tensor)

        if self.method == "line_search":
            direction = torch.tanh(raw).detach().numpy()
            proposed_width = None
        else:
            angle = np.pi * float(torch.tanh(raw[0]).detach())
            direction = np.array([np.cos(angle), np.sin(angle)])
            proposed_width = self.max_step_width * float(
                torch.sigmoid(raw[1]).detach()
            )
        return direction, proposed_width, log_probability, value, entropy

    def add(self, reward, log_probability, value, entropy):
        self.current_episode.append(
            Transition(reward, log_probability, value, entropy)
        )

    def finish_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = []

    def update(self):
        if not self.episodes:
            return 0.0
        returns = []
        log_probabilities = []
        values = []
        entropies = []
        for episode in self.episodes:
            discounted = 0.0
            episode_returns = []
            for transition in reversed(episode):
                discounted = transition.reward + self.gamma * discounted
                episode_returns.append(discounted)
            episode_returns.reverse()
            for transition, episode_return in zip(episode, episode_returns):
                returns.append(episode_return)
                log_probabilities.append(transition.log_probability)
                values.append(transition.value)
                entropies.append(transition.entropy)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probabilities = torch.stack(log_probabilities)
        values = torch.stack(values)
        entropies = torch.stack(entropies)
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)
        actor_loss = -(log_probabilities * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = (
            actor_loss
            + self.value_coefficient * critic_loss
            + self.entropy_coefficient * entropy_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.episodes = []
        return float(loss.detach())


class FlatArchive:
    def __init__(self, threshold):
        self.threshold = threshold
        self.shapes = []
        self.rows = []

    def add(self, node, episode, steps, potential):
        if self.shapes:
            novelty = min(
                float(np.sqrt(np.mean((node - shape) ** 2)))
                for shape in self.shapes
            )
            if novelty <= self.threshold:
                return False
        self.shapes.append(node.copy())
        self.rows.append([episode, steps, potential])
        return True

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for index, shape in enumerate(self.shapes):
            np.savetxt(directory / f"shape_{index:03d}.dat", shape, fmt="%.8f")
        rows = np.asarray(self.rows, dtype=float).reshape(-1, 3)
        np.savetxt(
            directory / "archive_summary.csv", rows, delimiter=",",
            header="episode,steps,flatness_potential", comments="",
        )


def moving_average(values, window):
    if len(values) < window:
        return np.array([])
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_history(values, window, ylabel, title, path):
    figure, axis = plt.subplots()
    axis.plot(values, color="0.65", alpha=0.4, label="Episode")
    average = moving_average(values, window)
    if len(average):
        x = np.arange(len(average)) + window - 1
        axis.plot(x, average, color="black", linewidth=2, label=f"MA ({window})")
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_path(directory, nodes, action_rows, mode_size):
    directory.mkdir(parents=True, exist_ok=True)
    for step, node in enumerate(nodes):
        np.savetxt(directory / f"{step}.dat", node, fmt="%.8f")
    header = (
        "step," + ",".join(f"mode_{i}" for i in range(mode_size))
        + ",step_width,flatness_potential,flatness_ratio"
    )
    np.savetxt(
        directory / "actions.csv", np.asarray(action_rows), delimiter=",",
        header=header, comments="",
    )


def evaluate(agent, env, episodes, output_directory, deterministic,
             cluster_threshold):
    agent.model.eval()
    final_nodes = []
    success = []
    self_crossings = []
    potentials = []
    potential_ratios = []
    steps = []
    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            done = False
            nodes = [env.node.copy()]
            actions = []
            info = {
                "success": False,
                "self_crossing": False,
                "flatness_potential": env.initial_potential,
                "flatness_ratio": 1.0,
            }
            while not done:
                direction, requested_width, _, _, _ = agent.get_action(
                    state, deterministic=deterministic
                )
                state, _, done, info = env.step(direction, requested_width)
                nodes.append(env.node.copy())
                row = [
                    len(actions), *info["mode_coefficients"],
                    info["step_width"], info["flatness_potential"],
                    info["flatness_ratio"],
                ]
                actions.append(row)
            save_path(
                output_directory / "evaluation_paths" / f"episode_{episode:03d}",
                nodes, actions, env.mode_size,
            )
            final_nodes.append(env.node.copy())
            success.append(info["success"])
            self_crossings.append(info["self_crossing"])
            potentials.append(info["flatness_potential"])
            potential_ratios.append(info["flatness_ratio"])
            steps.append(env.step_count)
    final_nodes = np.asarray(final_nodes)
    success = np.asarray(success, dtype=bool)
    self_crossings = np.asarray(self_crossings, dtype=bool)
    potentials = np.asarray(potentials)
    potential_ratios = np.asarray(potential_ratios)
    steps = np.asarray(steps)
    successful_nodes = final_nodes[success]
    if len(successful_nodes):
        distances = pairwise_rmsd(successful_nodes)
        labels = connected_component_labels(distances, cluster_threshold)
        cluster_count = len(np.unique(labels))
        cluster_sizes = np.bincount(labels)
    else:
        cluster_count = 0
        cluster_sizes = np.array([], dtype=int)
    rows = np.column_stack([
        np.arange(episodes), potentials, potential_ratios,
        success.astype(int), self_crossings.astype(int), steps,
    ])
    np.savetxt(
        output_directory / "evaluation_summary.csv", rows, delimiter=",",
        header=(
            "episode,flatness_potential,flatness_ratio,success,"
            "self_crossing,steps"
        ),
        comments="",
    )
    print("===================================")
    print(f"Method                 : {env.method}")
    print(f"Evaluation cases       : {episodes}")
    print(f"Deterministic          : {deterministic}")
    print(f"Flat cases             : {success.sum()} / {episodes}")
    print(f"Flat shape types       : {cluster_count} (RMSD <= {cluster_threshold:g})")
    if len(cluster_sizes):
        print(f"Cluster sizes          : {cluster_sizes.tolist()}")
    print(f"Self-crossing endings  : {self_crossings.sum()} / {episodes}")
    print(
        f"Flatness potential     : mean={potentials.mean():.6g}, "
        f"min={potentials.min():.6g}, max={potentials.max():.6g}"
    )
    print(
        f"Normalized potential  : mean={potential_ratios.mean():.6g}, "
        f"min={potential_ratios.min():.6g}, max={potential_ratios.max():.6g}"
    )
    if success.any():
        print(f"Successful steps       : mean={steps[success].mean():.2f}, min={steps[success].min()}")
    print("===================================")
    return {
        "success_count": int(success.sum()),
        "cluster_count": cluster_count,
        "self_crossing_count": int(self_crossings.sum()),
        "potentials": potentials,
        "potential_ratios": potential_ratios,
    }


def run(args, method):
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    script_directory = Path(__file__).resolve().parent
    output_directory = Path(args.output_dir)
    if not output_directory.is_absolute():
        output_directory = script_directory / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "run_config.json", "w", encoding="utf-8") as file:
        config = vars(args).copy()
        config["method"] = method
        json.dump(config, file, indent=2)

    node = np.atleast_2d(np.loadtxt(script_directory / "node_initial.dat", delimiter=","))
    member = np.atleast_2d(np.loadtxt(script_directory / "member.dat", delimiter=",", dtype=int))
    fix = np.atleast_2d(np.loadtxt(script_directory / "fix.dat", delimiter=",", dtype=int))
    env = FixedStartEnv(
        node, member, fix, method, args.max_steps, args.max_step_width,
        args.flatness_subspace_dimension, args.flatness_tolerance,
        args.strain_tolerance, args.crossing_tolerance, args.line_search_shrink,
        args.line_search_min_width,
    )
    state_size = len(env.reset())
    if method == "line_search":
        model = LineSearchActorCritic(state_size, env.mode_size)
    else:
        model = LearnedStepActorCritic(state_size, env.mode_size)
    agent = Agent(
        model, method, env.mode_size, args.max_step_width, args.gamma,
        args.learning_rate, args.entropy_coefficient,
    )
    archive = FlatArchive(args.archive_threshold)
    rewards = []
    potentials = []
    potential_ratios = []
    success_history = []
    self_crossing_history = []
    step_history = []
    policy_mean_history = []
    policy_variance_history = []
    losses = []
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        nodes = [env.node.copy()]
        episode_means = []
        episode_variances = []
        info = {
            "success": False,
            "self_crossing": False,
            "flatness_potential": env.initial_potential,
            "flatness_ratio": 1.0,
        }
        while not done:
            direction, requested_width, log_probability, value, entropy = agent.get_action(state)
            episode_means.append(agent.last_mean)
            episode_variances.append(agent.last_variance)
            state, reward, done, info = env.step(direction, requested_width)
            agent.add(reward, log_probability, value, entropy)
            total_reward += reward
            nodes.append(env.node.copy())
        agent.finish_episode()
        rewards.append(total_reward)
        potentials.append(info["flatness_potential"])
        potential_ratios.append(info["flatness_ratio"])
        success_history.append(int(info["success"]))
        self_crossing_history.append(int(info["self_crossing"]))
        step_history.append(env.step_count)
        policy_mean_history.append(np.mean(episode_means, axis=0))
        policy_variance_history.append(np.mean(episode_variances, axis=0))
        if info["success"]:
            archive.add(
                env.node, episode, env.step_count, info["flatness_potential"]
            )
        if (episode + 1) % args.batch_size == 0:
            losses.append(agent.update())
        if args.print_every > 0 and (episode == 0 or (episode + 1) % args.print_every == 0):
            start = max(0, len(rewards) - args.moving_average_window)
            print(
                f"Episode {episode + 1:5d}/{args.episodes} "
                f"| reward_MA {np.mean(rewards[start:]):8.3f} "
                f"| potential_MA {np.mean(potentials[start:]):.6f} "
                f"| success_rate {np.mean(success_history[start:]):.3f} "
                f"| crossing_rate {np.mean(self_crossing_history[start:]):.3f} "
                f"| archive {len(archive.shapes)}"
            )
    if agent.episodes:
        losses.append(agent.update())

    torch.save(agent.model.state_dict(), output_directory / "policy.pt")
    np.savetxt(output_directory / "reward_history.csv", rewards, delimiter=",", header="reward", comments="")
    np.savetxt(output_directory / "terminal_potential_history.csv", potentials, delimiter=",", header="potential", comments="")
    np.savetxt(
        output_directory / "terminal_flatness_ratio_history.csv",
        potential_ratios, delimiter=",", header="flatness_ratio", comments="",
    )
    np.savetxt(output_directory / "success_history.csv", success_history, delimiter=",", header="success", comments="")
    np.savetxt(
        output_directory / "self_crossing_history.csv", self_crossing_history,
        delimiter=",", header="self_crossing", comments="",
    )
    np.savetxt(output_directory / "step_history.csv", step_history, delimiter=",", header="steps", comments="")
    parameter_header = ",".join(
        f"action_{index}" for index in range(len(policy_mean_history[0]))
    )
    np.savetxt(
        output_directory / "policy_mean_history.csv", policy_mean_history,
        delimiter=",", header=parameter_header, comments="",
    )
    np.savetxt(
        output_directory / "policy_variance_history.csv",
        policy_variance_history, delimiter=",", header=parameter_header,
        comments="",
    )
    np.savetxt(output_directory / "loss_history.csv", losses, delimiter=",", header="loss", comments="")
    archive.save(output_directory / "training_flat_archive")
    plot_history(rewards, args.moving_average_window, "Cumulative reward", "Learning Curve", output_directory / "learning_curve.png")
    plot_history(
        potentials, args.moving_average_window, "Terminal flatness potential",
        "Terminal Flatness Potential", output_directory / "terminal_potential.png",
    )
    plot_history(
        potential_ratios, args.moving_average_window,
        "Normalized terminal flatness potential", "Terminal Flatness Ratio",
        output_directory / "terminal_flatness_ratio.png",
    )
    return evaluate(
        agent, env, args.eval_episodes, output_directory,
        args.eval_deterministic, args.cluster_threshold,
    )


def add_common_arguments(parser, default_output):
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-step-width", type=float, default=0.25)
    parser.add_argument("--line-search-shrink", type=float, default=0.5)
    parser.add_argument("--line-search-min-width", type=float, default=1e-3)
    parser.add_argument(
        "--flatness-subspace-dimension", type=int, choices=(1, 2), default=1,
    )
    parser.add_argument("--flatness-tolerance", type=float, default=1e-3)
    parser.add_argument("--strain-tolerance", type=float, default=1e-8)
    parser.add_argument("--crossing-tolerance", type=float, default=1e-10)
    parser.add_argument("--archive-threshold", type=float, default=0.05)
    parser.add_argument("--cluster-threshold", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--moving-average-window", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=default_output)
    return parser


__all__ = [
    "add_common_arguments",
    "run",
]
