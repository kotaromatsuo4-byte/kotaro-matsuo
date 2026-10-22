"""Variable-mode actor--critic network used by the Codex6 experiments.

The actor first samples a Gaussian coefficient for every instantaneous mode.
The resulting physical direction is canonicalised to one hemisphere; a
second Gaussian then supplies the signed and bounded step width. Mean pooling
keeps both actor and critic compatible with a changing number of modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def bounded_variance(
    logit: torch.Tensor,
    minimum: float = 1e-4,
    maximum: float = 2.0,
) -> torch.Tensor:
    """Map an unconstrained tensor to a positive, bounded variance."""

    return minimum + (maximum - minimum) * torch.sigmoid(logit)


def canonical_physical_direction(
    normalised_mixture: torch.Tensor,
    mode_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map modal coefficients to a unit direction in one fixed hemisphere.

    ``mode_tensor`` has shape ``(number_of_modes, coordinate_size)``. A modal
    direction and its negative span the same line, so fixing the sign here
    leaves the signed width as the sole selector of travel direction.
    """

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


class VariableModeActorCritic(nn.Module):
    """Actor--critic whose actor emits one distribution per current mode."""

    def __init__(
        self,
        coordinate_size: int,
        hidden_size: int = 96,
        variance_bias: float = -2.5,
        width_mean_limit: float = 3.0,
        goal_size: int = 0,
    ) -> None:
        super().__init__()
        self.coordinate_size = int(coordinate_size)
        self.state_size = self.coordinate_size + int(goal_size)
        self.hidden_size = int(hidden_size)
        self.width_mean_limit = float(width_mean_limit)
        if self.width_mean_limit <= 0.0:
            raise ValueError("width_mean_limit must be positive.")

        self.mode_encoder = nn.Sequential(
            nn.Linear(self.state_size + self.coordinate_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )
        self.mixture_mean = nn.Linear(self.hidden_size, 1)
        self.mixture_variance = nn.Linear(self.hidden_size, 1)

        self.width_body = nn.Sequential(
            nn.Linear(
                self.state_size + self.coordinate_size + self.hidden_size,
                self.hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )
        self.width_mean = nn.Linear(self.hidden_size, 1)
        self.width_variance = nn.Linear(self.hidden_size, 1)

        self.critic_state = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.Tanh(),
        )
        self.critic_mode = nn.Sequential(
            nn.Linear(self.state_size + self.coordinate_size, self.hidden_size),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

        nn.init.zeros_(self.mixture_variance.weight)
        nn.init.constant_(self.mixture_variance.bias, variance_bias)
        nn.init.zeros_(self.width_variance.weight)
        nn.init.constant_(self.width_variance.bias, variance_bias)

    def modal_distribution_parameters(
        self,
        state: torch.Tensor,
        modes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self,
        state: torch.Tensor,
        physical_direction: torch.Tensor,
        modal_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        variance = bounded_variance(self.width_variance(hidden).squeeze())
        return mean, variance

    def value(self, state: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
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

    def actor_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.mode_encoder,
            self.mixture_mean,
            self.mixture_variance,
            self.width_body,
            self.width_mean,
            self.width_variance,
        )
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]

    def critic_parameters(self) -> list[nn.Parameter]:
        modules = (self.critic_state, self.critic_mode, self.critic)
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]


@dataclass
class PolicyAction:
    physical_direction: np.ndarray
    mode_coefficients: np.ndarray
    requested_width: float
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    mean_mixture_variance: float
    mixture_mean_norm: float
    mixture_standard_deviation_norm: float
    mixture_concentration: float
    mode_count: int
    width_mean: float
    width_variance: float


# Short alias retained for the simple folder-level API.
Policy = VariableModeActorCritic
