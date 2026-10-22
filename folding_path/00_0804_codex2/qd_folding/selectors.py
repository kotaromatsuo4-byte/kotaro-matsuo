"""Perturbation-origin selectors for the second-stage comparison.

All selectors receive the same frozen candidate list and the same binary
coverage reward.  Ordinary UCB stores a lifetime mean for each immutable state
ID.  SM-UCB instead estimates the *current marginal gain* as a function of the
state feature and the current terminal-archive occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

try:
    from scipy.linalg import solve_triangular as _solve_triangular
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    def _solve_triangular(matrix, right, *, lower):
        del lower
        return np.linalg.solve(matrix, right)

from .candidate_sets import CandidateState


SELECTOR_KINDS = ("random", "ucb", "smucb")


ArchiveContext = frozenset[tuple[str, str]]


@dataclass(slots=True)
class SelectionDiagnostic:
    state_id: int
    score: float | None
    posterior_mean: float | None = None
    posterior_std: float | None = None
    visits: int | None = None


def _random_tie(
    values: np.ndarray,
    rng: np.random.Generator,
) -> int:
    maximum = float(np.max(values))
    if math.isinf(maximum):
        tied = np.flatnonzero(np.isinf(values))
    else:
        tied = np.flatnonzero(
            np.isclose(values, maximum, rtol=1.0e-12, atol=1.0e-14)
        )
    return int(rng.choice(tied))


class RandomSelector:
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.last_diagnostic: SelectionDiagnostic | None = None

    def select(
        self,
        candidates: Iterable[CandidateState],
        archive_context: ArchiveContext,
    ) -> CandidateState:
        del archive_context
        values = list(candidates)
        if not values:
            raise LookupError("no candidate state is available")
        selected = values[int(self.rng.integers(len(values)))]
        self.last_diagnostic = SelectionDiagnostic(selected.state_id, None)
        return selected

    def update(
        self,
        state: CandidateState,
        archive_context: ArchiveContext,
        reward: float,
    ) -> None:
        del state, archive_context, reward


@dataclass(slots=True)
class _LifetimeStatistics:
    visits: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0


class LifetimeUCBSelector:
    """Standard UCB applied heuristically to archive-relative binary reward."""

    name = "ucb"

    def __init__(self, rng: np.random.Generator, *, exploration: float) -> None:
        if exploration < 0.0 or not math.isfinite(exploration):
            raise ValueError("UCB exploration must be finite and non-negative")
        self.rng = rng
        self.exploration = float(exploration)
        self.statistics: dict[int, _LifetimeStatistics] = {}
        self.total_updates = 0
        self.last_diagnostic: SelectionDiagnostic | None = None

    def _statistics(self, state_id: int) -> _LifetimeStatistics:
        return self.statistics.setdefault(int(state_id), _LifetimeStatistics())

    def select(
        self,
        candidates: Iterable[CandidateState],
        archive_context: ArchiveContext,
    ) -> CandidateState:
        del archive_context
        values = list(candidates)
        if not values:
            raise LookupError("no candidate state is available")
        scores: list[float] = []
        for state in values:
            statistics = self._statistics(state.state_id)
            if statistics.visits == 0:
                scores.append(math.inf)
                continue
            bonus = self.exploration * math.sqrt(
                math.log(max(self.total_updates, 2)) / statistics.visits
            )
            scores.append(statistics.mean + bonus)
        index = _random_tie(np.asarray(scores, dtype=float), self.rng)
        selected = values[index]
        statistics = self._statistics(selected.state_id)
        self.last_diagnostic = SelectionDiagnostic(
            selected.state_id,
            scores[index],
            posterior_mean=statistics.mean,
            posterior_std=None,
            visits=statistics.visits,
        )
        return selected

    def update(
        self,
        state: CandidateState,
        archive_context: ArchiveContext,
        reward: float,
    ) -> None:
        del archive_context
        value = float(reward)
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError("coverage reward must be finite and lie in [0, 1]")
        statistics = self._statistics(state.state_id)
        statistics.visits += 1
        statistics.reward_sum += value
        self.total_updates += 1


@dataclass(slots=True)
class _SMObservation:
    feature: np.ndarray
    archive: ArchiveContext
    reward: float


class SMUCBSelector:
    """Kernel SM-UCB over state and current archive context.

    The kernel is a product of an RBF state kernel and an archive kernel.  The
    latter is the product of a cosine kernel on exact cell-occupancy vectors
    (implemented using set intersections) and an RBF kernel on archive size.
    This is positive semidefinite and explicitly lets the predicted payoff of
    one state change as terminal cells are added.
    """

    name = "smucb"

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        state_lengthscale: float,
        archive_lengthscale: float,
        noise_variance: float,
        delta: float,
        beta_scale: float,
    ) -> None:
        positive = {
            "state_lengthscale": state_lengthscale,
            "archive_lengthscale": archive_lengthscale,
            "noise_variance": noise_variance,
            "beta_scale": beta_scale,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("SM-UCB values must be positive: " + ", ".join(invalid))
        if not 0.0 < delta < 1.0:
            raise ValueError("SM-UCB delta must lie strictly between zero and one")
        self.rng = rng
        self.state_lengthscale = float(state_lengthscale)
        self.archive_lengthscale = float(archive_lengthscale)
        self.noise_variance = float(noise_variance)
        self.delta = float(delta)
        self.beta_scale = float(beta_scale)
        self.observations: list[_SMObservation] = []
        self._cholesky: np.ndarray | None = None
        self.last_diagnostic: SelectionDiagnostic | None = None

    def _state_kernel(self, first: np.ndarray, second: np.ndarray) -> float:
        difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
        squared_rms = float(np.mean(difference * difference))
        return math.exp(
            -0.5 * squared_rms / (self.state_lengthscale**2)
        )

    def _archive_kernel(
        self,
        first: ArchiveContext,
        second: ArchiveContext,
    ) -> float:
        # Adding a constant feature avoids the undefined empty-set cosine and
        # allows early observations to transfer to later archive contexts.
        cosine = (1.0 + len(first & second)) / math.sqrt(
            (1.0 + len(first)) * (1.0 + len(second))
        )
        difference = float(len(first) - len(second))
        count_kernel = math.exp(
            -0.5 * difference * difference / (self.archive_lengthscale**2)
        )
        return float(cosine * count_kernel)

    def _kernel(
        self,
        first_feature: np.ndarray,
        first_archive: ArchiveContext,
        second_feature: np.ndarray,
        second_archive: ArchiveContext,
    ) -> float:
        return self._state_kernel(first_feature, second_feature) * (
            0.05 + 0.95 * self._archive_kernel(first_archive, second_archive)
        )

    def _beta(self, next_round: int) -> float:
        # A finite-domain GP-UCB style schedule is used in the executable
        # experiment.  The much larger theorem-level beta from SM-UCB is
        # reported separately in the manuscript and is not claimed here.
        time_value = max(int(next_round), 1)
        argument = ((time_value + 1) ** 2) * math.pi**2 / (6.0 * self.delta)
        return max(2.0 * self.beta_scale * math.log(argument), 1.0e-12)

    def _posterior(
        self,
        candidates: list[CandidateState],
        archive_context: ArchiveContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.observations)
        if count == 0:
            return np.zeros(len(candidates)), np.ones(len(candidates))
        if self._cholesky is None or self._cholesky.shape != (count, count):
            raise RuntimeError("SM-UCB Cholesky cache is inconsistent")
        rewards = np.asarray(
            [value.reward for value in self.observations], dtype=float
        )
        alpha = _solve_triangular(
            self._cholesky.T,
            _solve_triangular(self._cholesky, rewards, lower=True),
            lower=False,
        )
        cross_kernel = np.asarray(
            [
                [
                    self._kernel(
                        observation.feature,
                        observation.archive,
                        candidate.feature,
                        archive_context,
                    )
                    for candidate in candidates
                ]
                for observation in self.observations
            ],
            dtype=float,
        )
        means = cross_kernel.T @ alpha
        projected = _solve_triangular(
            self._cholesky,
            cross_kernel,
            lower=True,
        )
        variances = np.maximum(1.0 - np.sum(projected * projected, axis=0), 0.0)
        deviations = np.sqrt(variances)
        return means, deviations

    def select(
        self,
        candidates: Iterable[CandidateState],
        archive_context: ArchiveContext,
    ) -> CandidateState:
        values = list(candidates)
        if not values:
            raise LookupError("no candidate state is available")
        means, deviations = self._posterior(values, archive_context)
        beta = self._beta(len(self.observations) + 1)
        scores = means + math.sqrt(beta) * deviations
        index = _random_tie(scores, self.rng)
        selected = values[index]
        self.last_diagnostic = SelectionDiagnostic(
            selected.state_id,
            float(scores[index]),
            posterior_mean=float(means[index]),
            posterior_std=float(deviations[index]),
            visits=None,
        )
        return selected

    def update(
        self,
        state: CandidateState,
        archive_context: ArchiveContext,
        reward: float,
    ) -> None:
        value = float(reward)
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError("coverage reward must be finite and lie in [0, 1]")
        observation = _SMObservation(
            feature=np.asarray(state.feature, dtype=float).copy(),
            archive=frozenset(archive_context),
            reward=value,
        )
        old_count = len(self.observations)
        if old_count == 0:
            diagonal = math.sqrt(1.0 + self.noise_variance)
            self._cholesky = np.asarray([[diagonal]], dtype=float)
        else:
            if self._cholesky is None:
                raise RuntimeError("SM-UCB Cholesky cache is missing")
            covariance = np.asarray(
                [
                    self._kernel(
                        previous.feature,
                        previous.archive,
                        observation.feature,
                        observation.archive,
                    )
                    for previous in self.observations
                ],
                dtype=float,
            )
            projected = _solve_triangular(
                self._cholesky,
                covariance,
                lower=True,
            )
            diagonal_square = (
                1.0
                + self.noise_variance
                - float(projected @ projected)
            )
            diagonal = math.sqrt(max(diagonal_square, 1.0e-12))
            expanded = np.zeros((old_count + 1, old_count + 1), dtype=float)
            expanded[:old_count, :old_count] = self._cholesky
            expanded[old_count, :old_count] = projected
            expanded[old_count, old_count] = diagonal
            self._cholesky = expanded
        self.observations.append(observation)


def make_selector(kind: str, config, rng: np.random.Generator):
    """Construct one of the three selectors from the common configuration."""

    normalized = str(kind).strip().lower()
    if normalized == "random":
        return RandomSelector(rng)
    if normalized == "ucb":
        return LifetimeUCBSelector(
            rng,
            exploration=config.ucb_exploration,
        )
    if normalized == "smucb":
        return SMUCBSelector(
            rng,
            state_lengthscale=config.smucb_state_lengthscale,
            archive_lengthscale=config.smucb_archive_lengthscale,
            noise_variance=config.smucb_noise_variance,
            delta=config.smucb_delta,
            beta_scale=config.smucb_beta_scale,
        )
    raise ValueError(f"unknown selector: {kind}")


__all__ = [
    "ArchiveContext",
    "LifetimeUCBSelector",
    "RandomSelector",
    "SELECTOR_KINDS",
    "SMUCBSelector",
    "SelectionDiagnostic",
    "make_selector",
]
