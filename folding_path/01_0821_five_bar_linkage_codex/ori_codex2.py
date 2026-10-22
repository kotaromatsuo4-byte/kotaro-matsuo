"""Actor--Critic search for a flat folding path from one fixed start.

The policy supports a changing number of instantaneous mechanism modes. It
first samples one coefficient for every current mode, normalises the modal
coefficient vector to represent direction only, and then samples a signed
step width conditioned on the resulting physical displacement direction.

Flatness is never used as a step-acceptance condition. Predictor widths are
reduced only when projection onto the bar-length constraint manifold fails.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from ori_codex import dedx, index_free, mode, project_to_constraints
from single_solution_common import (
    compute_flatness_potential,
    current_strain,
    has_proper_self_crossing,
    nonadjacent_bar_pairs,
)


def bounded_variance(logit, minimum=1e-4, maximum=2.0):
    """Map an unconstrained tensor to a strictly positive bounded variance."""
    return minimum + (maximum - minimum) * torch.sigmoid(logit)


def canonical_physical_direction(normalised_mixture, mode_tensor):
    """Map modal coefficients to a unit direction in one fixed hemisphere."""
    physical_direction = normalised_mixture @ mode_tensor
    physical_norm = torch.linalg.vector_norm(physical_direction)
    if float(physical_norm.detach()) <= 1e-12:
        return normalised_mixture, torch.zeros_like(physical_direction)
    physical_direction = physical_direction / physical_norm
    pivot = torch.argmax(torch.abs(physical_direction))
    if float(physical_direction[pivot].detach()) < 0.0:
        physical_direction = -physical_direction
        normalised_mixture = -normalised_mixture
    return normalised_mixture, physical_direction


class FixedStartFoldingEnv:
    """Fixed-start linkage environment with variable mechanism dimension."""

    def __init__(
        self,
        node,
        member,
        fix,
        max_steps=80,
        max_step_width=0.5,
        min_step_width=1e-4,
        projection_shrink=0.5,
        max_correction_ratio=1.0,
        max_actual_step_ratio=1.25,
        flatness_subspace_dimension=1,
        flatness_tolerance=1e-3,
        strain_tolerance=1e-8,
        crossing_tolerance=1e-10,
        flatness_reward_weight=1.0,
        flatness_reward_clip=2.0,
        success_bonus=10.0,
        crossing_penalty=10.0,
        numerical_failure_penalty=10.0,
        no_mode_penalty=5.0,
        timeout_penalty=1.0,
    ):
        self.node0 = np.asarray(node, dtype=float).copy()
        self.node = self.node0.copy()
        self.member = np.asarray(member, dtype=int)
        self.fix = np.asarray(fix, dtype=int)
        self.num_node = len(self.node0)
        self.coordinate_size = self.node0.size
        self.free = index_free(self.num_node, self.fix)
        self.lengths = np.linalg.norm(
            self.node0[self.member[:, 0]] - self.node0[self.member[:, 1]],
            axis=1,
        )
        self.coordinate_scale = max(float(np.max(self.lengths)), 1e-8)
        self.max_steps = int(max_steps)
        self.max_step_width = float(max_step_width)
        self.min_step_width = float(min_step_width)
        self.projection_shrink = float(projection_shrink)
        self.max_correction_ratio = float(max_correction_ratio)
        self.max_actual_step_ratio = float(max_actual_step_ratio)
        if not 0.0 < self.projection_shrink < 1.0:
            raise ValueError("projection_shrink must lie strictly between 0 and 1.")
        if self.min_step_width <= 0.0:
            raise ValueError("min_step_width must be positive.")
        if self.max_correction_ratio <= 0.0:
            raise ValueError("max_correction_ratio must be positive.")
        if self.max_actual_step_ratio <= 0.0:
            raise ValueError("max_actual_step_ratio must be positive.")
        self.flatness_subspace_dimension = int(flatness_subspace_dimension)
        self.flatness_tolerance = float(flatness_tolerance)
        self.strain_tolerance = float(strain_tolerance)
        self.crossing_tolerance = float(crossing_tolerance)
        self.flatness_reward_weight = float(flatness_reward_weight)
        self.flatness_reward_clip = float(flatness_reward_clip)
        self.success_bonus = float(success_bonus)
        self.crossing_penalty = abs(float(crossing_penalty))
        self.numerical_failure_penalty = abs(
            float(numerical_failure_penalty)
        )
        self.no_mode_penalty = abs(float(no_mode_penalty))
        self.timeout_penalty = abs(float(timeout_penalty))
        self.crossing_pairs = nonadjacent_bar_pairs(self.member)

        self.initial_potential = compute_flatness_potential(
            self.node0, self.flatness_subspace_dimension
        )
        if self.initial_potential <= 1e-12:
            raise ValueError(
                "Initial flatness is zero. Use subspace dimension 1 for the "
                "planar five-bar linkage."
            )
        self.previous_potential = self.initial_potential
        self.previous_ratio = 1.0
        self.step_count = 0

    def reset(self):
        self.node = self.node0.copy()
        self.previous_potential = self.initial_potential
        self.previous_ratio = 1.0
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        """The NN observation contains coordinates only, as specified."""
        return (self.node / self.coordinate_scale).flatten().astype(np.float32)

    def get_modes(self):
        jacobian = dedx(self.node, self.member, self.free)
        return mode(self.free, jacobian, self.num_node)

    def flatness_metrics(self, node=None):
        coordinates = self.node if node is None else node
        potential = compute_flatness_potential(
            coordinates, self.flatness_subspace_dimension
        )
        ratio = potential / self.initial_potential
        rms_distance = (
            np.sqrt(max(potential, 0.0) / self.num_node)
            / self.coordinate_scale
        )
        return float(potential), float(ratio), float(rms_distance)

    def _project_candidate(self, direction, requested_width):
        direction = np.asarray(direction, dtype=float).reshape(self.node.shape)
        direction_norm = np.linalg.norm(direction)
        requested_width = float(
            np.clip(
                requested_width,
                -self.max_step_width,
                self.max_step_width,
            )
        )
        if direction_norm <= 1e-12 or abs(requested_width) < self.min_step_width:
            return self.node.copy(), 0.0, False, 0, 0.0, 0.0, 0.0

        direction = direction / direction_norm
        width = requested_width
        attempts = 0
        last_strain = np.inf
        last_correction_norm = np.inf
        last_actual_displacement = np.inf
        while abs(width) >= self.min_step_width:
            attempts += 1
            predicted = self.node + width * direction
            corrected = project_to_constraints(
                predicted,
                self.member,
                self.lengths,
                self.free,
            )
            if np.all(np.isfinite(corrected)):
                last_strain = current_strain(
                    corrected, self.member, self.lengths
                )
                last_correction_norm = float(
                    np.linalg.norm(corrected - predicted)
                )
                last_actual_displacement = float(
                    np.linalg.norm(corrected - self.node)
                )
                correction_is_local = (
                    last_correction_norm
                    <= self.max_correction_ratio * abs(width)
                )
                actual_step_is_local = (
                    last_actual_displacement
                    <= self.max_actual_step_ratio * abs(width)
                )
                if (
                    last_strain <= self.strain_tolerance
                    and correction_is_local
                    and actual_step_is_local
                ):
                    return (
                        corrected,
                        float(width),
                        False,
                        attempts,
                        last_strain,
                        last_correction_norm,
                        last_actual_displacement,
                    )
            width *= self.projection_shrink

        return (
            self.node.copy(),
            0.0,
            True,
            attempts,
            float(last_strain),
            float(last_correction_norm),
            float(last_actual_displacement),
        )

    def step(self, physical_direction, requested_width, mode_coefficients):
        node_before = self.node.copy()
        ratio_before = self.previous_ratio
        (
            node_after,
            accepted_width,
            correction_failed,
            projection_attempts,
            strain,
            correction_norm,
            actual_displacement,
        ) = self._project_candidate(physical_direction, requested_width)

        self_crossing = False
        if not correction_failed and abs(accepted_width) > 0.0:
            self_crossing = has_proper_self_crossing(
                node_before,
                node_after,
                self.member,
                self.crossing_pairs,
                self.crossing_tolerance,
            )

        self.node = node_after
        potential, ratio, rms_distance = self.flatness_metrics()
        bounded_before = min(
            max(ratio_before, 0.0), self.flatness_reward_clip
        )
        bounded_after = min(max(ratio, 0.0), self.flatness_reward_clip)
        reward = self.flatness_reward_weight * (
            bounded_before - bounded_after
        )

        self.previous_potential = potential
        self.previous_ratio = ratio
        self.step_count += 1

        # The tolerance is applied to the normalised squared residual
        # P(node) / P(initial), not to a world-coordinate distance directly.
        success = (
            ratio <= self.flatness_tolerance
            and not self_crossing
            and not correction_failed
        )
        no_modes = False
        if not success and not self_crossing and not correction_failed:
            no_modes = self.get_modes().shape[1] == 0
        timeout = self.step_count >= self.max_steps

        if correction_failed:
            reward -= self.numerical_failure_penalty
            termination_reason = "correction_failed"
        elif self_crossing:
            reward -= self.crossing_penalty
            termination_reason = "self_crossing"
        elif success:
            reward += self.success_bonus
            termination_reason = "success"
        elif no_modes:
            reward -= self.no_mode_penalty
            termination_reason = "no_modes"
        elif timeout:
            reward -= self.timeout_penalty
            termination_reason = "timeout"
        else:
            termination_reason = "running"

        done = termination_reason != "running"
        info = {
            "flatness_potential": potential,
            "flatness_ratio": ratio,
            "flatness_rms": rms_distance,
            "strain": float(strain),
            "success": success,
            "self_crossing": self_crossing,
            "correction_failed": correction_failed,
            "no_modes": no_modes,
            "timeout": termination_reason == "timeout",
            "termination_reason": termination_reason,
            "requested_width": float(requested_width),
            "accepted_width": float(accepted_width),
            "projection_attempts": int(projection_attempts),
            "correction_norm": float(correction_norm),
            "actual_displacement": float(actual_displacement),
            "correction_to_width_ratio": float(
                correction_norm / max(abs(accepted_width), self.min_step_width)
            ),
            "actual_to_width_ratio": float(
                actual_displacement
                / max(abs(accepted_width), self.min_step_width)
            ),
            "mode_count": int(len(mode_coefficients)),
            "mode_coefficients": np.asarray(
                mode_coefficients, dtype=float
            ).copy(),
        }
        return self.get_state(), float(reward), done, info


class VariableModeActorCritic(nn.Module):
    """Actor--Critic whose actor emits one distribution per current mode."""

    def __init__(
        self,
        coordinate_size,
        hidden_size=96,
        variance_bias=-2.5,
        width_mean_limit=3.0,
    ):
        super().__init__()
        self.coordinate_size = int(coordinate_size)
        self.hidden_size = int(hidden_size)
        self.width_mean_limit = float(width_mean_limit)
        if self.width_mean_limit <= 0.0:
            raise ValueError("width_mean_limit must be positive.")

        self.mode_encoder = nn.Sequential(
            nn.Linear(2 * coordinate_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mixture_mean = nn.Linear(hidden_size, 1)
        self.mixture_variance = nn.Linear(hidden_size, 1)

        self.width_body = nn.Sequential(
            nn.Linear(2 * coordinate_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.width_mean = nn.Linear(hidden_size, 1)
        self.width_variance = nn.Linear(hidden_size, 1)

        self.critic_state = nn.Sequential(
            nn.Linear(coordinate_size, hidden_size),
            nn.Tanh(),
        )
        self.critic_mode = nn.Sequential(
            nn.Linear(2 * coordinate_size, hidden_size),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        nn.init.zeros_(self.mixture_variance.weight)
        nn.init.constant_(self.mixture_variance.bias, variance_bias)
        nn.init.zeros_(self.width_variance.weight)
        nn.init.constant_(self.width_variance.bias, variance_bias)

    def modal_distribution_parameters(self, state, modes):
        mode_count = modes.shape[0]
        if mode_count == 0:
            empty = state.new_empty((0,))
            features = state.new_empty((0, self.hidden_size))
            return empty, empty, features
        repeated_state = state.repeat(mode_count, 1)
        features = self.mode_encoder(
            torch.cat([repeated_state, modes], dim=1)
        )
        mean = torch.tanh(self.mixture_mean(features).squeeze(-1))
        variance = bounded_variance(
            self.mixture_variance(features).squeeze(-1)
        )
        return mean, variance, features

    def width_distribution_parameters(
        self, state, physical_direction, modal_features
    ):
        if modal_features.shape[0] == 0:
            pooled_modes = state.new_zeros((1, self.hidden_size))
        else:
            pooled_modes = modal_features.mean(dim=0, keepdim=True)
        width_input = torch.cat(
            [state, physical_direction.unsqueeze(0), pooled_modes], dim=1
        )
        hidden = self.width_body(width_input)
        raw_mean = self.width_mean(hidden).squeeze()
        mean = self.width_mean_limit * torch.tanh(
            raw_mean / self.width_mean_limit
        )
        variance = bounded_variance(
            self.width_variance(hidden).squeeze()
        )
        return mean, variance

    def value(self, state, modes):
        state_feature = self.critic_state(state)
        if modes.shape[0] == 0:
            mode_feature = state.new_zeros((1, self.hidden_size))
        else:
            repeated_state = state.repeat(modes.shape[0], 1)
            mode_feature = self.critic_mode(
                torch.cat([repeated_state, modes], dim=1)
            ).mean(dim=0, keepdim=True)
        return self.critic(
            torch.cat([state_feature, mode_feature], dim=1)
        ).squeeze()

    def actor_parameters(self):
        modules = (
            self.mode_encoder,
            self.mixture_mean,
            self.mixture_variance,
            self.width_body,
            self.width_mean,
            self.width_variance,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def critic_parameters(self):
        modules = (self.critic_state, self.critic_mode, self.critic)
        return [parameter for module in modules for parameter in module.parameters()]


@dataclass
class Transition:
    reward: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


@dataclass
class PolicyAction:
    physical_direction: np.ndarray
    mode_coefficients: np.ndarray
    requested_width: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    mean_mixture_variance: float
    width_mean: float
    width_variance: float


class ActorCriticAgent:
    """Monte-Carlo actor--critic with episode-separated returns."""

    def __init__(
        self,
        model,
        max_step_width,
        gamma=1.0,
        actor_learning_rate=1e-3,
        critic_learning_rate=3e-4,
        entropy_coefficient=0.001,
        value_coefficient=0.5,
        actor_gradient_clip=0.5,
        critic_gradient_clip=1.0,
    ):
        self.model = model
        self.max_step_width = float(max_step_width)
        self.gamma = float(gamma)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.actor_gradient_clip = float(actor_gradient_clip)
        self.critic_gradient_clip = float(critic_gradient_clip)
        self.actor_parameter_list = self.model.actor_parameters()
        self.critic_parameter_list = self.model.critic_parameters()
        actor_ids = {id(parameter) for parameter in self.actor_parameter_list}
        critic_ids = {id(parameter) for parameter in self.critic_parameter_list}
        model_ids = {id(parameter) for parameter in self.model.parameters()}
        if actor_ids & critic_ids or actor_ids | critic_ids != model_ids:
            raise RuntimeError("Actor and critic parameter partitions are invalid.")
        self.actor_optimizer = optim.Adam(
            self.actor_parameter_list, lr=float(actor_learning_rate)
        )
        self.critic_optimizer = optim.Adam(
            self.critic_parameter_list, lr=float(critic_learning_rate)
        )
        self.finished_episodes = []
        self.current_episode = []

    def get_action(self, state, modes, deterministic=False):
        state_tensor = torch.as_tensor(
            state[np.newaxis, :], dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        mean, variance, modal_features = (
            self.model.modal_distribution_parameters(state_tensor, mode_tensor)
        )
        if mean.numel() == 0:
            raise ValueError("Cannot sample an action without mechanism modes.")

        mixture_distribution = Normal(mean, torch.sqrt(variance))
        raw_mixture = mean if deterministic else mixture_distribution.sample()
        mixture_norm = torch.linalg.vector_norm(raw_mixture)
        if float(mixture_norm.detach()) <= 1e-12:
            normalised_mixture = torch.zeros_like(raw_mixture)
            physical_direction = state_tensor.new_zeros(state_tensor.shape[1])
        else:
            normalised_mixture = raw_mixture / mixture_norm
            # A modal direction and its negative span the same line. Put the
            # physical direction in one deterministic hemisphere so that the
            # signed width alone selects the travel direction.
            normalised_mixture, physical_direction = (
                canonical_physical_direction(normalised_mixture, mode_tensor)
            )

        width_mean, width_variance = (
            self.model.width_distribution_parameters(
                state_tensor, physical_direction, modal_features
            )
        )
        width_distribution = Normal(
            width_mean, torch.sqrt(width_variance)
        )
        raw_width = (
            width_mean if deterministic else width_distribution.sample()
        )
        signed_width = self.max_step_width * torch.tanh(raw_width)

        # Gaussian samples are latent actions. The environment receives a
        # deterministic bounded transform, so latent log probabilities give a
        # valid score-function gradient despite the direction normalisation.
        log_probability = (
            mixture_distribution.log_prob(raw_mixture).sum()
            + width_distribution.log_prob(raw_width).sum()
        )
        # Keep entropy regularisation approximately independent of the current
        # number of modes. The joint action log probability remains a sum.
        entropy = (
            mixture_distribution.entropy().mean()
            + width_distribution.entropy().sum()
        )
        value = self.model.value(state_tensor, mode_tensor)

        return PolicyAction(
            physical_direction=physical_direction.detach().cpu().numpy(),
            mode_coefficients=normalised_mixture.detach().cpu().numpy(),
            requested_width=float(signed_width.detach()),
            log_probability=log_probability,
            value=value,
            entropy=entropy,
            mean_mixture_variance=float(variance.mean().detach()),
            width_mean=float(width_mean.detach()),
            width_variance=float(width_variance.detach()),
        )

    def add_transition(self, transition):
        self.current_episode.append(transition)

    def finish_episode(self):
        if self.current_episode:
            self.finished_episodes.append(self.current_episode)
        self.current_episode = []

    def set_actor_learning_rate(self, learning_rate):
        learning_rate = float(learning_rate)
        for parameter_group in self.actor_optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    @property
    def actor_learning_rate(self):
        return float(self.actor_optimizer.param_groups[0]["lr"])

    @property
    def critic_learning_rate(self):
        return float(self.critic_optimizer.param_groups[0]["lr"])

    def update(self):
        if not self.finished_episodes:
            return None

        returns = []
        log_probabilities = []
        values = []
        entropies = []
        for episode in self.finished_episodes:
            discounted_return = 0.0
            episode_returns = []
            for transition in reversed(episode):
                discounted_return = (
                    transition.reward
                    + self.gamma * discounted_return
                )
                episode_returns.append(discounted_return)
            episode_returns.reverse()
            for transition, discounted_return in zip(
                episode, episode_returns
            ):
                returns.append(discounted_return)
                log_probabilities.append(transition.log_probability)
                values.append(transition.value)
                entropies.append(transition.entropy)

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
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

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(
            self.actor_parameter_list, self.actor_gradient_clip
        )
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            self.critic_parameter_list, self.critic_gradient_clip
        )
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.finished_episodes = []
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy_tensor.mean().detach()),
            "actor_gradient_norm": float(actor_gradient_norm),
            "critic_gradient_norm": float(critic_gradient_norm),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }


def run_episode(env, agent, deterministic=False, training=False):
    state = env.reset()
    done = False
    total_reward = 0.0
    nodes = [env.node.copy()]
    actions = []
    mixture_variances = []
    width_means = []
    width_variances = []
    requested_widths = []
    accepted_widths = []
    projection_attempts = 0
    correction_ratios = []
    actual_step_ratios = []
    mode_counts = []
    minimum_ratio = 1.0
    info = {
        "flatness_potential": env.initial_potential,
        "flatness_ratio": 1.0,
        "flatness_rms": env.flatness_metrics(env.node0)[2],
        "success": False,
        "self_crossing": False,
        "correction_failed": False,
        "no_modes": False,
        "timeout": False,
        "termination_reason": "not_started",
    }

    while not done:
        modes = env.get_modes()
        if modes.shape[1] == 0:
            info = dict(info)
            info.update({
                "no_modes": True,
                "termination_reason": "no_modes",
            })
            total_reward -= env.no_mode_penalty
            done = True
            break

        action = agent.get_action(state, modes, deterministic=deterministic)
        next_state, reward, done, info = env.step(
            action.physical_direction,
            action.requested_width,
            action.mode_coefficients,
        )
        if training:
            agent.add_transition(
                Transition(
                    reward,
                    action.log_probability,
                    action.value,
                    action.entropy,
                )
            )
        total_reward += reward
        state = next_state
        nodes.append(env.node.copy())
        mixture_variances.append(action.mean_mixture_variance)
        width_means.append(action.width_mean)
        width_variances.append(action.width_variance)
        requested_widths.append(info["requested_width"])
        accepted_widths.append(info["accepted_width"])
        projection_attempts += info["projection_attempts"]
        correction_ratios.append(info["correction_to_width_ratio"])
        actual_step_ratios.append(info["actual_to_width_ratio"])
        mode_counts.append(info["mode_count"])
        minimum_ratio = min(minimum_ratio, info["flatness_ratio"])
        actions.append({
            "step": env.step_count,
            "mode_basis": modes.tolist(),
            "mode_coefficients": info["mode_coefficients"].tolist(),
            "physical_direction": action.physical_direction.tolist(),
            "requested_width": info["requested_width"],
            "accepted_width": info["accepted_width"],
            "flatness_ratio": info["flatness_ratio"],
            "reward": reward,
            "projection_attempts": info["projection_attempts"],
            "correction_norm": info["correction_norm"],
            "actual_displacement": info["actual_displacement"],
        })

    if training:
        agent.finish_episode()

    summary = {
        "total_reward": float(total_reward),
        "flatness_potential": float(info["flatness_potential"]),
        "flatness_ratio": float(info["flatness_ratio"]),
        "flatness_rms": float(info["flatness_rms"]),
        "minimum_flatness_ratio": float(minimum_ratio),
        "success": int(info["success"]),
        "self_crossing": int(info["self_crossing"]),
        "correction_failed": int(info["correction_failed"]),
        "no_modes": int(info["no_modes"]),
        "timeout": int(info["timeout"]),
        "termination_reason": info["termination_reason"],
        "steps": int(env.step_count),
        "mean_mixture_variance": float(
            np.mean(mixture_variances) if mixture_variances else np.nan
        ),
        "mean_width_mean": float(
            np.mean(width_means) if width_means else np.nan
        ),
        "mean_width_variance": float(
            np.mean(width_variances) if width_variances else np.nan
        ),
        "mean_abs_requested_width": float(
            np.mean(np.abs(requested_widths)) if requested_widths else 0.0
        ),
        "mean_abs_accepted_width": float(
            np.mean(np.abs(accepted_widths)) if accepted_widths else 0.0
        ),
        "projection_attempts": int(projection_attempts),
        "mean_correction_to_width_ratio": float(
            np.mean(correction_ratios) if correction_ratios else 0.0
        ),
        "max_correction_to_width_ratio": float(
            np.max(correction_ratios) if correction_ratios else 0.0
        ),
        "mean_actual_to_width_ratio": float(
            np.mean(actual_step_ratios) if actual_step_ratios else 0.0
        ),
        "max_actual_to_width_ratio": float(
            np.max(actual_step_ratios) if actual_step_ratios else 0.0
        ),
        "minimum_mode_count": int(min(mode_counts) if mode_counts else 0),
        "maximum_mode_count": int(max(mode_counts) if mode_counts else 0),
    }
    return summary, nodes, actions


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.array([]), np.array([])
    averaged = np.convolve(
        values, np.ones(window, dtype=float) / window, mode="valid"
    )
    x = np.arange(window, len(values) + 1)
    return x, averaged


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_rollout(directory, nodes, actions):
    directory.mkdir(parents=True, exist_ok=True)
    # Keep rollout filenames compact and human-readable.  The broad pattern
    # also removes legacy ``step_0000.dat`` files when an existing rollout is
    # overwritten with the new ``step0.dat`` convention.
    for stale_step in directory.glob("step*.dat"):
        stale_step.unlink()
    for index, node in enumerate(nodes):
        np.savetxt(
            directory / f"step{index}.dat", node, fmt="%.10f"
        )
    with open(directory / "actions.json", "w", encoding="utf-8") as file:
        json.dump(actions, file, indent=2)


def save_checkpoint(
    path,
    model,
    agent,
    config,
    episode,
    evaluation_metrics=None,
    evaluation_score=None,
):
    torch.save(
        {
            "episode": int(episode),
            "model_state_dict": model.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            "actor_learning_rate": agent.actor_learning_rate,
            "critic_learning_rate": agent.critic_learning_rate,
            "evaluation_metrics": evaluation_metrics,
            "evaluation_score": evaluation_score,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "config": config,
        },
        path,
    )


def actor_learning_rate_at_episode(
    initial_learning_rate, episode, milestones, factors
):
    if len(milestones) != len(factors):
        raise ValueError("actor_lr_milestones and actor_lr_factors must match.")
    learning_rate = float(initial_learning_rate)
    for milestone, factor in zip(milestones, factors):
        if episode > int(milestone):
            learning_rate = float(initial_learning_rate) * float(factor)
    return learning_rate


def fixed_seed_stochastic_evaluation(
    environment, agent, evaluation_episodes, evaluation_seed
):
    """Evaluate comparable cases without advancing the training RNG streams."""
    torch_rng_state = torch.get_rng_state()
    numpy_rng_state = np.random.get_state()
    evaluations = []
    try:
        for evaluation_index in range(int(evaluation_episodes)):
            case_seed = int(evaluation_seed) + evaluation_index
            torch.manual_seed(case_seed)
            np.random.seed(case_seed)
            with torch.no_grad():
                summary, _, _ = run_episode(
                    environment,
                    agent,
                    deterministic=False,
                    training=False,
                )
            summary["evaluation_episode"] = evaluation_index
            evaluations.append(summary)
    finally:
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
    return evaluations


def aggregate_stochastic_evaluations(evaluations, episode=None):
    if not evaluations:
        return None
    ratios = [row["flatness_ratio"] for row in evaluations]
    result = {
        "evaluation_episodes": len(evaluations),
        "success_rate": float(np.mean([
            row["success"] for row in evaluations
        ])),
        "self_crossing_rate": float(np.mean([
            row["self_crossing"] for row in evaluations
        ])),
        "mean_terminal_ratio": float(np.mean(ratios)),
        "median_terminal_ratio": float(np.median(ratios)),
        "mean_steps": float(np.mean([row["steps"] for row in evaluations])),
    }
    if episode is not None:
        result = {"episode": int(episode), **result}
    return result


def policy_selection_score(evaluation):
    return (
        float(evaluation["success_rate"]),
        -float(evaluation["self_crossing_rate"]),
        -float(evaluation["mean_terminal_ratio"]),
        -float(evaluation["mean_steps"]),
    )


def clone_model_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_training_plots(
    output_directory,
    history,
    evaluation_history,
    stochastic_evaluation_history,
    window,
):
    episodes = np.asarray([row["episode"] for row in history])
    rewards = np.asarray([row["total_reward"] for row in history])
    ratios = np.asarray([row["flatness_ratio"] for row in history])
    successes = np.asarray([row["success"] for row in history])
    crossings = np.asarray([row["self_crossing"] for row in history])
    steps = np.asarray([row["steps"] for row in history])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(episodes, rewards, color="0.75", linewidth=0.5)
    x, averaged = moving_average(rewards, window)
    if len(x):
        axes[0, 0].plot(x, averaged, color="black", linewidth=1.8)
    axes[0, 0].set_title("Episode return")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(
        episodes,
        np.maximum(ratios, 1e-12),
        color="0.75",
        linewidth=0.5,
    )
    x, averaged = moving_average(ratios, window)
    if len(x):
        axes[0, 1].plot(
            x, np.maximum(averaged, 1e-12), color="black", linewidth=1.8
        )
    if evaluation_history:
        axes[0, 1].plot(
            [row["episode"] for row in evaluation_history],
            [max(row["flatness_ratio"], 1e-12) for row in evaluation_history],
            color="tab:red",
            marker="o",
            markersize=2,
            linewidth=1,
            label="deterministic evaluation",
        )
        axes[0, 1].legend()
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Terminal flatness ratio")
    axes[0, 1].set_xlabel("Episode")

    x, success_rate = moving_average(successes, window)
    if len(x):
        axes[1, 0].plot(x, success_rate, label="success", color="tab:green")
    x, crossing_rate = moving_average(crossings, window)
    if len(x):
        axes[1, 0].plot(x, crossing_rate, label="crossing", color="tab:red")
    if stochastic_evaluation_history:
        axes[1, 0].plot(
            [row["episode"] for row in stochastic_evaluation_history],
            [row["success_rate"] for row in stochastic_evaluation_history],
            marker="o",
            markersize=3,
            linewidth=1,
            color="tab:blue",
            label="frozen stochastic evaluation",
        )
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Moving terminal rates")
    axes[1, 0].set_xlabel("Episode")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        axes[1, 0].legend(handles, labels)

    axes[1, 1].plot(episodes, steps, color="0.75", linewidth=0.5)
    x, averaged = moving_average(steps, window)
    if len(x):
        axes[1, 1].plot(x, averaged, color="black", linewidth=1.8)
    axes[1, 1].set_title("Episode length")
    axes[1, 1].set_xlabel("Episode")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "training_curves.png", dpi=180)
    plt.close(figure)


def make_environment(args, node, member, fix):
    return FixedStartFoldingEnv(
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
        flatness_reward_weight=args.flatness_reward_weight,
        flatness_reward_clip=args.flatness_reward_clip,
        success_bonus=args.success_bonus,
        crossing_penalty=args.crossing_penalty,
        numerical_failure_penalty=args.numerical_failure_penalty,
        no_mode_penalty=args.no_mode_penalty,
        timeout_penalty=args.timeout_penalty,
    )


def resolve_output_directory(script_directory, requested):
    output_directory = Path(requested)
    if not output_directory.is_absolute():
        output_directory = script_directory / output_directory
    if output_directory.exists() and any(output_directory.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = output_directory.with_name(
            f"{output_directory.name}_{timestamp}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def train(args):
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.actor_lr_milestones) != len(args.actor_lr_factors):
        raise ValueError(
            "actor_lr_milestones and actor_lr_factors must have equal length."
        )
    if list(args.actor_lr_milestones) != sorted(args.actor_lr_milestones):
        raise ValueError("actor_lr_milestones must be sorted.")
    if any(factor <= 0.0 for factor in args.actor_lr_factors):
        raise ValueError("actor_lr_factors must be positive.")

    script_directory = Path(__file__).resolve().parent
    data_directory = (
        Path(args.data_dir).resolve()
        if args.data_dir is not None
        else script_directory
    )
    output_directory = resolve_output_directory(
        script_directory, args.output_dir
    )

    node = np.atleast_2d(
        np.loadtxt(data_directory / "node_initial.dat", delimiter=",")
    )
    member = np.atleast_2d(
        np.loadtxt(
            data_directory / "member.dat", delimiter=",", dtype=int
        )
    )
    fix = np.atleast_2d(
        np.loadtxt(data_directory / "fix.dat", delimiter=",", dtype=int)
    )
    environment = make_environment(args, node, member, fix)
    evaluation_environment = make_environment(args, node, member, fix)
    initial_modes = environment.get_modes()

    model = VariableModeActorCritic(
        coordinate_size=environment.coordinate_size,
        hidden_size=args.hidden_size,
        variance_bias=args.variance_bias,
        width_mean_limit=args.width_mean_limit,
    )
    agent = ActorCriticAgent(
        model=model,
        max_step_width=args.max_step_width,
        gamma=args.gamma,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        actor_gradient_clip=args.actor_gradient_clip,
        critic_gradient_clip=args.critic_gradient_clip,
    )

    config = vars(args).copy()
    config.update({
        "initial_mode_count": int(initial_modes.shape[1]),
        "coordinate_size": int(environment.coordinate_size),
        "initial_flatness_potential": environment.initial_potential,
        "architecture": getattr(
            args,
            "architecture_name",
            (
                "variable_mode_canonical_direction_conditional_signed_width_"
                "actor_critic"
            ),
        ),
        "output_directory": str(output_directory),
    })
    with open(output_directory / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    history = []
    evaluation_history = []
    stochastic_evaluation_history = []
    update_history = []
    first_success_saved = False
    best_ratio = np.inf
    best_rollout = None
    best_policy_score = None
    best_policy_metrics = None
    best_policy_state = None
    final_rollout = None

    for episode in range(1, args.episodes + 1):
        model.train()
        summary, nodes, actions = run_episode(
            environment, agent, deterministic=False, training=True
        )
        summary["episode"] = episode
        history.append(summary)
        final_rollout = (nodes, actions)

        valid_terminal = not (
            summary["self_crossing"]
            or summary["correction_failed"]
            or summary["no_modes"]
        )
        if valid_terminal and summary["flatness_ratio"] < best_ratio:
            best_ratio = summary["flatness_ratio"]
            best_rollout = (nodes, actions)
        if summary["success"] and not first_success_saved:
            save_rollout(
                output_directory / "first_success_path", nodes, actions
            )
            first_success_saved = True

        if episode % args.batch_size == 0 or episode == args.episodes:
            actor_learning_rate = actor_learning_rate_at_episode(
                args.actor_learning_rate,
                episode,
                args.actor_lr_milestones,
                args.actor_lr_factors,
            )
            agent.set_actor_learning_rate(actor_learning_rate)
            update_result = agent.update()
            if update_result is not None:
                update_result["episode"] = episode
                update_history.append(update_result)

        if (
            episode % args.eval_interval == 0
            or episode == 1
            or episode == args.episodes
        ):
            model.eval()
            with torch.no_grad():
                evaluation, eval_nodes, eval_actions = run_episode(
                    evaluation_environment,
                    agent,
                    deterministic=True,
                    training=False,
                )
            evaluation["episode"] = episode
            evaluation_history.append(evaluation)
            if evaluation["success"]:
                save_rollout(
                    output_directory / "latest_deterministic_success_path",
                    eval_nodes,
                    eval_actions,
                )
            if args.periodic_stochastic_eval_episodes > 0:
                frozen_evaluations = fixed_seed_stochastic_evaluation(
                    evaluation_environment,
                    agent,
                    args.periodic_stochastic_eval_episodes,
                    args.selection_evaluation_seed,
                )
                frozen_evaluation = aggregate_stochastic_evaluations(
                    frozen_evaluations, episode=episode
                )
                stochastic_evaluation_history.append(frozen_evaluation)
                candidate_score = policy_selection_score(frozen_evaluation)
                if (
                    best_policy_score is None
                    or candidate_score > best_policy_score
                ):
                    best_policy_score = candidate_score
                    best_policy_metrics = frozen_evaluation.copy()
                    best_policy_state = clone_model_state(model)
                    save_checkpoint(
                        output_directory / "best_policy.pt",
                        model,
                        agent,
                        config,
                        episode,
                        evaluation_metrics=best_policy_metrics,
                        evaluation_score=best_policy_score,
                    )
                    with open(
                        output_directory / "best_policy_metrics.json",
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(
                            {
                                "episode": episode,
                                "selection_score": list(best_policy_score),
                                "selection_evaluation": best_policy_metrics,
                            },
                            file,
                            indent=2,
                        )

        if episode % args.print_every == 0 or episode == 1:
            recent = history[-min(200, len(history)):]
            mean_reward = np.mean([row["total_reward"] for row in recent])
            mean_ratio = np.mean([row["flatness_ratio"] for row in recent])
            success_rate = np.mean([row["success"] for row in recent])
            crossing_rate = np.mean(
                [row["self_crossing"] for row in recent]
            )
            print(
                f"Episode {episode:6d} | reward {mean_reward:8.4f} | "
                f"terminal ratio {mean_ratio:10.6g} | "
                f"success {success_rate:6.2%} | crossing {crossing_rate:6.2%}",
                flush=True,
            )

        if episode % args.checkpoint_every == 0:
            save_csv(output_directory / "training_history.csv", history)
            save_csv(
                output_directory / "deterministic_evaluation_history.csv",
                evaluation_history,
            )
            save_csv(
                output_directory / "stochastic_evaluation_history.csv",
                stochastic_evaluation_history,
            )
            save_csv(output_directory / "update_history.csv", update_history)
            save_checkpoint(
                output_directory / "checkpoint.pt",
                model,
                agent,
                config,
                episode,
            )

    save_csv(output_directory / "training_history.csv", history)
    save_csv(
        output_directory / "deterministic_evaluation_history.csv",
        evaluation_history,
    )
    save_csv(
        output_directory / "stochastic_evaluation_history.csv",
        stochastic_evaluation_history,
    )
    save_csv(output_directory / "update_history.csv", update_history)
    if best_rollout is not None:
        save_rollout(
            output_directory / "best_flatness_path", *best_rollout
        )
    if final_rollout is not None:
        save_rollout(output_directory / "final_training_path", *final_rollout)

    save_checkpoint(
        output_directory / "policy.pt",
        model,
        agent,
        config,
        args.episodes,
    )

    model.eval()
    last_policy_state = clone_model_state(model)
    final_evaluations = fixed_seed_stochastic_evaluation(
        evaluation_environment,
        agent,
        args.eval_episodes,
        args.final_evaluation_seed,
    )
    save_csv(output_directory / "evaluation_summary.csv", final_evaluations)

    with torch.no_grad():
        last_deterministic, last_nodes, last_actions = run_episode(
            evaluation_environment,
            agent,
            deterministic=True,
            training=False,
        )
    save_rollout(
        output_directory / "last_policy_deterministic_path",
        last_nodes,
        last_actions,
    )

    if best_policy_state is None:
        best_policy_state = last_policy_state
        best_policy_metrics = aggregate_stochastic_evaluations(
            final_evaluations, episode=args.episodes
        )
        best_policy_score = policy_selection_score(best_policy_metrics)
        save_checkpoint(
            output_directory / "best_policy.pt",
            model,
            agent,
            config,
            args.episodes,
            evaluation_metrics=best_policy_metrics,
            evaluation_score=best_policy_score,
        )

    model.load_state_dict(best_policy_state)
    model.eval()
    best_policy_evaluations = fixed_seed_stochastic_evaluation(
        evaluation_environment,
        agent,
        args.eval_episodes,
        args.final_evaluation_seed,
    )
    save_csv(
        output_directory / "best_policy_evaluation_summary.csv",
        best_policy_evaluations,
    )
    with torch.no_grad():
        best_deterministic, best_nodes, best_actions = run_episode(
            evaluation_environment,
            agent,
            deterministic=True,
            training=False,
        )
    save_rollout(
        output_directory / "best_policy_deterministic_path",
        best_nodes,
        best_actions,
    )
    model.load_state_dict(last_policy_state)

    save_training_plots(
        output_directory,
        history,
        evaluation_history,
        stochastic_evaluation_history,
        args.moving_average_window,
    )

    last_evaluation_metrics = aggregate_stochastic_evaluations(
        final_evaluations, episode=args.episodes
    )
    best_held_out_metrics = aggregate_stochastic_evaluations(
        best_policy_evaluations,
        episode=(
            best_policy_metrics["episode"]
            if best_policy_metrics is not None
            else args.episodes
        ),
    )
    training_success_count = sum(row["success"] for row in history)
    final_recent = history[-min(1000, len(history)):]
    summary = {
        "training_episodes": args.episodes,
        "training_successes": training_success_count,
        "last_1000_success_rate": float(
            np.mean([row["success"] for row in final_recent])
        ),
        "last_1000_mean_terminal_ratio": float(
            np.mean([row["flatness_ratio"] for row in final_recent])
        ),
        "last_policy_held_out_evaluation": last_evaluation_metrics,
        "last_policy_deterministic_evaluation": last_deterministic,
        "best_policy_selection_evaluation": best_policy_metrics,
        "best_policy_held_out_evaluation": best_held_out_metrics,
        "best_policy_deterministic_evaluation": best_deterministic,
        "best_training_flatness_ratio": float(best_ratio),
    }
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Training complete")
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_directory}")
    return output_directory, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Variable-mode actor--critic flat-folding experiment."
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument(
        "--periodic-stochastic-eval-episodes", type=int, default=50
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-step-width", type=float, default=0.5)
    parser.add_argument("--min-step-width", type=float, default=1e-4)
    parser.add_argument("--projection-shrink", type=float, default=0.5)
    parser.add_argument("--max-correction-ratio", type=float, default=1.0)
    parser.add_argument("--max-actual-step-ratio", type=float, default=1.25)
    parser.add_argument(
        "--flatness-subspace-dimension",
        type=int,
        choices=(1, 2),
        default=1,
    )
    parser.add_argument(
        "--flatness-tolerance",
        type=float,
        default=1e-3,
        help=(
            "Success threshold for the normalised best-fit subspace squared "
            "residual P(node)/P(initial)."
        ),
    )
    parser.add_argument("--strain-tolerance", type=float, default=1e-8)
    parser.add_argument("--crossing-tolerance", type=float, default=1e-10)
    parser.add_argument("--flatness-reward-weight", type=float, default=1.0)
    parser.add_argument("--flatness-reward-clip", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--crossing-penalty", type=float, default=10.0)
    parser.add_argument(
        "--numerical-failure-penalty", type=float, default=10.0
    )
    parser.add_argument("--no-mode-penalty", type=float, default=5.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--actor-learning-rate",
        "--learning-rate",
        dest="actor_learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--actor-lr-milestones",
        type=int,
        nargs="*",
        default=[2000, 4000, 7000],
    )
    parser.add_argument(
        "--actor-lr-factors",
        type=float,
        nargs="*",
        default=[0.3, 0.1, 0.03],
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--actor-gradient-clip", type=float, default=0.5)
    parser.add_argument("--critic-gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument(
        "--variance-bias", type=float, default=-1.0986122886681098
    )
    parser.add_argument("--width-mean-limit", type=float, default=3.0)
    parser.add_argument("--moving-average-window", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--selection-evaluation-seed", type=int, default=1000003
    )
    parser.add_argument(
        "--final-evaluation-seed", type=int, default=2000003
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ori_codex2_actor_critic_stabilized_10000_results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
