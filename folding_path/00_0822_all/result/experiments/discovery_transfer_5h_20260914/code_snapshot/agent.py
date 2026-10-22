"""Episode-separated Monte-Carlo Actor--Critic for variable-mode folding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from policy import (
    PolicyAction,
    VariableModeActorCritic,
    canonical_physical_direction,
)


@dataclass
class Transition:
    reward: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class ActorCriticAgent:
    """Monte-Carlo actor--critic with returns kept separate by episode."""

    def __init__(
        self,
        model: VariableModeActorCritic,
        max_step_width: float,
        gamma: float = 1.0,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.001,
        value_coefficient: float = 0.5,
        actor_gradient_clip: float = 0.5,
        critic_gradient_clip: float = 1.0,
        actor_reference_horizon: float | None = None,
    ) -> None:
        self.model = model
        self.max_step_width = float(max_step_width)
        self.gamma = float(gamma)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.actor_gradient_clip = float(actor_gradient_clip)
        self.critic_gradient_clip = float(critic_gradient_clip)
        self.actor_reference_horizon = (
            None
            if actor_reference_horizon is None
            else float(actor_reference_horizon)
        )
        if (
            self.actor_reference_horizon is not None
            and self.actor_reference_horizon <= 0.0
        ):
            raise ValueError("actor_reference_horizon must be positive.")

        self.actor_parameter_list = self.model.actor_parameters()
        self.critic_parameter_list = self.model.critic_parameters()
        actor_ids = {id(parameter) for parameter in self.actor_parameter_list}
        critic_ids = {
            id(parameter) for parameter in self.critic_parameter_list
        }
        model_ids = {id(parameter) for parameter in self.model.parameters()}
        if actor_ids & critic_ids or actor_ids | critic_ids != model_ids:
            raise RuntimeError("Actor and critic parameter partitions are invalid.")

        self.actor_optimizer = optim.Adam(
            self.actor_parameter_list,
            lr=float(actor_learning_rate),
        )
        self.critic_optimizer = optim.Adam(
            self.critic_parameter_list,
            lr=float(critic_learning_rate),
        )
        self.finished_episodes: list[list[Transition]] = []
        self.current_episode: list[Transition] = []
        self.exploration_strength = 0.0

    def get_action(
        self,
        state: np.ndarray,
        modes: np.ndarray,
        deterministic: bool = False,
    ) -> PolicyAction:
        state_tensor = torch.as_tensor(
            state[np.newaxis, :], dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        mean, variance, modal_features = (
            self.model.modal_distribution_parameters(
                state_tensor, mode_tensor
            )
        )
        if mean.numel() == 0:
            raise ValueError("Cannot sample an action without mechanism modes.")

        # Use the SAME transformed Gaussian for sampling and its log density.
        # The controller is fixed for a batch; zero preserves legacy numerics.
        if self.exploration_strength:
            mean = mean / (1.0 + self.exploration_strength)
            variance = variance + self.exploration_strength
        mixture_distribution = Normal(mean, torch.sqrt(variance))
        raw_mixture = (
            mean if deterministic else mixture_distribution.sample()
        )
        mixture_norm = torch.linalg.vector_norm(raw_mixture)
        if float(mixture_norm.detach()) <= 1e-12:
            normalised_mixture = torch.zeros_like(raw_mixture)
            physical_direction = state_tensor.new_zeros(
                mode_tensor.shape[1]
            )
        else:
            normalised_mixture = raw_mixture / mixture_norm
            normalised_mixture, physical_direction = (
                canonical_physical_direction(
                    normalised_mixture, mode_tensor
                )
            )

        width_mean, width_variance = (
            self.model.width_distribution_parameters(
                state_tensor,
                physical_direction,
                modal_features,
            )
        )
        if self.exploration_strength:
            width_mean = width_mean / (1.0 + self.exploration_strength)
            width_variance = width_variance + self.exploration_strength
        width_distribution = Normal(
            width_mean, torch.sqrt(width_variance)
        )
        raw_width = (
            width_mean if deterministic else width_distribution.sample()
        )
        signed_width = self.max_step_width * torch.tanh(raw_width)

        # These diagnostics reuse the distribution parameters that generated
        # the action.  They do not perform another forward pass or consume
        # random numbers, so enabling their logging leaves the rollout
        # sequence unchanged.
        mixture_mean_norm = torch.linalg.vector_norm(mean)
        mixture_standard_deviation_norm = torch.sqrt(variance.sum())
        mixture_concentration = (
            mixture_mean_norm / mixture_standard_deviation_norm
        )

        # The Gaussians are latent actions. The direction normalisation and
        # bounded-width transform passed to the environment are deterministic.
        log_probability = (
            mixture_distribution.log_prob(raw_mixture).sum()
            + width_distribution.log_prob(raw_width).sum()
        )
        # Mean modal entropy avoids changing the regularisation scale merely
        # because the instantaneous mechanism dimension changed.
        entropy = (
            mixture_distribution.entropy().mean()
            + width_distribution.entropy().sum()
        )
        value = self.model.value(state_tensor, mode_tensor)

        return PolicyAction(
            physical_direction=physical_direction.detach().cpu().numpy(),
            mode_coefficients=(
                normalised_mixture.detach().cpu().numpy()
            ),
            requested_width=float(signed_width.detach()),
            log_probability=log_probability,
            value=value,
            entropy=entropy,
            mean_mixture_variance=float(variance.mean().detach()),
            mixture_mean_norm=float(mixture_mean_norm.detach()),
            mixture_standard_deviation_norm=float(
                mixture_standard_deviation_norm.detach()
            ),
            mixture_concentration=float(mixture_concentration.detach()),
            mode_count=int(mean.numel()),
            width_mean=float(width_mean.detach()),
            width_variance=float(width_variance.detach()),
        )

    def add_transition(self, transition: Transition) -> None:
        self.current_episode.append(transition)

    def add(
        self,
        reward: float,
        log_probability: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:
        """Convenience wrapper for constructing a transition."""

        self.add_transition(
            Transition(reward, log_probability, value, entropy)
        )

    def finish_episode(self) -> None:
        if self.current_episode:
            self.finished_episodes.append(self.current_episode)
        self.current_episode = []

    @staticmethod
    def _set_optimizer_learning_rate(
        optimizer: optim.Optimizer,
        learning_rate: float,
    ) -> None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def set_actor_learning_rate(self, learning_rate: float) -> None:
        self._set_optimizer_learning_rate(
            self.actor_optimizer, learning_rate
        )

    def set_critic_learning_rate(self, learning_rate: float) -> None:
        self._set_optimizer_learning_rate(
            self.critic_optimizer, learning_rate
        )

    def set_learning_rates(
        self,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
    ) -> None:
        if actor_learning_rate is not None:
            self.set_actor_learning_rate(actor_learning_rate)
        if critic_learning_rate is not None:
            self.set_critic_learning_rate(critic_learning_rate)

    @property
    def actor_learning_rate(self) -> float:
        return float(self.actor_optimizer.param_groups[0]["lr"])

    @property
    def critic_learning_rate(self) -> float:
        return float(self.critic_optimizer.param_groups[0]["lr"])

    def update(self, episode_groups=None, advantage_centering="batch") -> dict[str, float] | None:
        if not self.finished_episodes:
            return None

        episode_count = len(self.finished_episodes)
        returns: list[float] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

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
            if advantage_centering == "goal":
                if episode_groups is None or len(episode_groups) != episode_count:
                    raise ValueError("Goal centering needs one group per episode")
                scale = advantages.std(unbiased=False) + 1e-8
                centered = advantages.clone()
                labels = [g for g, episode in zip(episode_groups, self.finished_episodes) for _ in episode]
                for group in dict.fromkeys(episode_groups):
                    mask = torch.tensor([g == group for g in labels], device=advantages.device)
                    centered[mask] = advantages[mask] - advantages[mask].mean()
                advantages = centered / scale
            elif advantage_centering == "batch":
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std(unbiased=False) + 1e-8)
            else:
                raise ValueError("Unknown advantage centering")

        if self.actor_reference_horizon is None:
            actor_normalizer = float(advantages.numel())
            actor_normalization = "transition_mean"
        else:
            actor_normalizer = float(
                episode_count * self.actor_reference_horizon
            )
            actor_normalization = "episode_sum_fixed_horizon"

        actor_loss = -(
            log_probability_tensor * advantages
        ).sum() / actor_normalizer
        critic_loss = nn.functional.mse_loss(
            value_tensor, returns_tensor
        )
        entropy_loss = -entropy_tensor.sum() / actor_normalizer
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
            "episode_count": int(episode_count),
            "transition_count": int(returns_tensor.numel()),
            "actor_normalizer": float(actor_normalizer),
            "actor_normalization": actor_normalization,
        }


def learning_rate_at_episode(
    initial_learning_rate: float,
    episode: int,
    milestones: list[int] | tuple[int, ...],
    factors: list[float] | tuple[float, ...],
) -> float:
    """Return the Codex6 piecewise-constant learning-rate schedule."""

    if len(milestones) != len(factors):
        raise ValueError("learning-rate milestones and factors must match.")
    learning_rate = float(initial_learning_rate)
    for milestone, factor in zip(milestones, factors):
        if episode > int(milestone):
            learning_rate = float(initial_learning_rate) * float(factor)
    return learning_rate


actor_learning_rate_at_episode = learning_rate_at_episode

# Short alias retained for the simple folder-level API.
Agent = ActorCriticAgent
