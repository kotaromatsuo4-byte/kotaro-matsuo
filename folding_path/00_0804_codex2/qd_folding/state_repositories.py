"""Active state repositories and UCB accounting.

The complete trajectory history is owned by :mod:`qd_folding.archives`.
This module only keeps references to states which may be selected as future
perturbation starting points.

Trajectories are plain dictionaries.  The only mandatory entry is
``trajectory["path"]``, a ``(state_dimension, number_of_states)`` NumPy
array.  The following optional entries make validity and intervention
accounting unambiguous:

``terminal_reason``
    ``"flat"``, ``"collision"``, ``"budget"``, ``"failed"``,
    ``"stalled"``, ``"max_steps"``, or ``"running"``.
``valid_state_mask``
    Boolean array with one value per path column.
``intervention_counts``
    Cumulative intervention count at every path column.
``intervention_steps``
    Alternative list of path-column indices at which interventions occurred.

Configuration descriptors are expected to be dimensionless.  A model-aware
``descriptor_fn(state)`` should normally be supplied.  The generic fallback
flattens the state and optionally divides it by ``descriptor_normalization``.
Gravity, mechanism modes, and state validity are also callbacks so that this
module does not import mechanics and cannot create a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import ExperimentConfig


DescriptorFunction = Callable[[np.ndarray], np.ndarray]
ScalarStateFunction = Callable[[np.ndarray], float]
ModesFunction = Callable[[np.ndarray], np.ndarray]
ValidityFunction = Callable[[dict, int], bool]


# ---------------------------------------------------------------------------
# Generic descriptors and UCB statistics
# ---------------------------------------------------------------------------


def normalized_configuration_descriptor(
    state: np.ndarray,
    descriptor_fn: DescriptorFunction | None = None,
    normalization: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return a finite one-dimensional normalized descriptor.

    ``descriptor_fn`` takes precedence.  If it is omitted, the state is
    flattened and divided componentwise by ``normalization`` when supplied.
    The function intentionally does not unit-normalize the vector: doing so
    would erase physically meaningful scale differences between states.
    """

    raw = descriptor_fn(state) if descriptor_fn is not None else state
    descriptor = np.asarray(raw, dtype=float).reshape(-1)
    if descriptor.size == 0:
        raise ValueError("A configuration descriptor must not be empty")
    if normalization is not None:
        scale = np.asarray(normalization, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("descriptor normalization must be finite and positive")
        descriptor = descriptor / scale
    if np.any(~np.isfinite(descriptor)):
        raise ValueError("A configuration descriptor must be finite")
    return descriptor


def descriptor_rms_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Root-mean-square distance between two descriptor vectors."""

    first_array = np.asarray(first, dtype=float).reshape(-1)
    second_array = np.asarray(second, dtype=float).reshape(-1)
    if first_array.shape != second_array.shape:
        raise ValueError("Descriptor dimensions do not match")
    return float(np.sqrt(np.mean((first_array - second_array) ** 2)))


@dataclass(slots=True)
class ArmStatistics:
    """Statistics shared by event states and bounded descriptor cells."""

    visits: int = 0
    reward_sum: float = 0.0
    reward_square_sum: float = 0.0
    simulated_steps: int = 0
    successes: int = 0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0

    @property
    def reward_variance(self) -> float:
        if self.visits < 2:
            return 0.0
        second_moment = self.reward_square_sum / self.visits
        return max(second_moment - self.mean_reward**2, 0.0)


def update_arm_statistics(
    statistics: ArmStatistics,
    reward: float,
    *,
    simulated_steps: int = 0,
    success: bool = False,
) -> None:
    """Update one arm after a completed allocation.

    The caller decides whether ``reward`` is raw or already cost-normalized.
    ``simulated_steps`` is retained separately for diagnostics.
    """

    reward = float(reward)
    simulated_steps = int(simulated_steps)
    if not math.isfinite(reward):
        raise ValueError("UCB reward must be finite")
    if simulated_steps < 0:
        raise ValueError("simulated_steps must be non-negative")
    statistics.visits += 1
    statistics.reward_sum += reward
    statistics.reward_square_sum += reward * reward
    statistics.simulated_steps += simulated_steps
    statistics.successes += int(bool(success))


def ucb_score(
    statistics: ArmStatistics,
    total_visits: int,
    exploration: float,
) -> float:
    """Standard UCB1 score for one arm.

    An unvisited arm receives ``+inf``.  Random tie-breaking is handled by
    :func:`select_ucb_arm`.
    """

    if statistics.visits <= 0:
        return math.inf
    total_visits = max(int(total_visits), statistics.visits, 1)
    exploration = float(exploration)
    if exploration < 0.0 or not math.isfinite(exploration):
        raise ValueError("UCB exploration coefficient must be finite and non-negative")
    bonus = exploration * math.sqrt(
        math.log(max(total_visits, 2)) / statistics.visits
    )
    return statistics.mean_reward + bonus


def _statistics_of(arm: Any) -> ArmStatistics:
    if isinstance(arm, ArmStatistics):
        return arm
    if isinstance(arm, Mapping):
        statistics = arm.get("statistics", arm.get("stats"))
    else:
        statistics = getattr(arm, "statistics", getattr(arm, "stats", None))
    if not isinstance(statistics, ArmStatistics):
        raise TypeError("Each UCB arm must expose ArmStatistics as stats/statistics")
    return statistics


def select_ucb_arm(
    arms: Sequence[Any] | Iterable[Any],
    *,
    total_visits: int,
    exploration: float,
    rng: np.random.Generator,
) -> Any:
    """Select the maximum-UCB arm with unbiased random tie-breaking."""

    candidates = list(arms)
    if not candidates:
        raise LookupError("No eligible UCB arms")
    scores = np.asarray(
        [
            ucb_score(_statistics_of(arm), total_visits, exploration)
            for arm in candidates
        ],
        dtype=float,
    )
    maximum = float(np.max(scores))
    if math.isinf(maximum):
        tied = np.flatnonzero(np.isinf(scores))
    else:
        tied = np.flatnonzero(
            np.isclose(scores, maximum, rtol=1.0e-12, atol=1.0e-14)
        )
    return candidates[int(rng.choice(tied))]


# ---------------------------------------------------------------------------
# Shared state records and trajectory helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StateRecord:
    """Reference to one path column retained by an active repository."""

    state_id: int
    trajectory: dict
    step: int
    descriptor: np.ndarray
    intervention_count: int
    created_order: int
    source_rollout: Any = None
    event_mask: int = 0
    event_severity: float = 0.0
    event_metrics: dict[str, float] = field(default_factory=dict)
    statistics: ArmStatistics = field(default_factory=ArmStatistics)
    active: bool = True
    merged_count: int = 0

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self.trajectory["path"])[:, self.step]

    @property
    def stats(self) -> ArmStatistics:
        """Short compatibility alias used by generic UCB utilities."""

        return self.statistics

    @property
    def visits(self) -> int:
        return self.statistics.visits

    @property
    def reward_sum(self) -> float:
        return self.statistics.reward_sum


def _trajectory_path(trajectory: dict) -> np.ndarray:
    if "path" not in trajectory:
        raise KeyError("trajectory must contain a 'path' matrix")
    path = np.asarray(trajectory["path"], dtype=float)
    if path.ndim != 2 or path.shape[1] == 0:
        raise ValueError("trajectory['path'] must be a non-empty 2-D array")
    if np.any(~np.isfinite(path)):
        raise ValueError("trajectory path must contain only finite values")
    return path


def _terminal_reason(trajectory: dict) -> str:
    value = trajectory.get("terminal_reason", "running")
    return str(value).strip().lower()


_INVALID_LAST_STATE_REASONS = {
    "flat",
    "collision",
    "collided",
    "budget",
    "budget_exhausted",
    "failed",
    "failure",
    "invalid",
    "numerical_failure",
    "correction_failed",
    "max_path_steps",
}


def _valid_nonterminal_mask(
    trajectory: dict,
    path: np.ndarray,
    validity_fn: ValidityFunction | None,
) -> np.ndarray:
    count = path.shape[1]
    if "valid_state_mask" in trajectory:
        mask = np.asarray(trajectory["valid_state_mask"], dtype=bool).reshape(-1)
        if len(mask) != count:
            raise ValueError("valid_state_mask length must equal path columns")
        mask = mask.copy()
    else:
        mask = np.ones(count, dtype=bool)
    if _terminal_reason(trajectory) in _INVALID_LAST_STATE_REASONS:
        mask[-1] = False
    if validity_fn is not None:
        for step in range(count):
            mask[step] = mask[step] and bool(validity_fn(trajectory, step))
    return mask


def _intervention_count_at(trajectory: dict, step: int) -> int:
    if "intervention_counts" in trajectory:
        counts = np.asarray(trajectory["intervention_counts"], dtype=int).reshape(-1)
        if len(counts) != np.asarray(trajectory["path"]).shape[1]:
            raise ValueError("intervention_counts length must equal path columns")
        return max(int(counts[step]), 0)
    if "intervention_steps" in trajectory:
        return sum(int(value) <= step for value in trajectory["intervention_steps"])
    if "interventions" in trajectory:
        interventions = trajectory["interventions"]
        if isinstance(interventions, Sequence):
            result = 0
            for value in interventions:
                if isinstance(value, Mapping):
                    location = value.get("step", value.get("path_step"))
                    if location is not None and int(location) <= step:
                        result += 1
            if result or len(interventions) == 0:
                return result
    return max(int(trajectory.get("intervention_count", 0)), 0)


def _state_is_eligible(
    record: StateRecord,
    config: ExperimentConfig,
) -> bool:
    return (
        record.active
        and record.step < config.max_path_steps
        and record.intervention_count < config.max_interventions
    )


def _mode_projector(modes: np.ndarray) -> np.ndarray:
    matrix = np.asarray(modes, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("Mechanism modes must be a one- or two-dimensional array")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    threshold = max(float(singular_values[0]) * 1.0e-10, 1.0e-14)
    basis = left[:, singular_values > threshold]
    return basis @ basis.T


def _projector_change(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("Mechanism projectors have incompatible dimensions")
    return float(np.linalg.norm(first - second, ord="fro") / math.sqrt(first.shape[0]))


# ---------------------------------------------------------------------------
# Event frontier
# ---------------------------------------------------------------------------


class EventType(IntFlag):
    NONE = 0
    STRIDE = 1 << 0
    POTENTIAL_EXTREMUM = 1 << 1
    BASIS_CHANGE = 1 << 2
    DESCRIPTOR_CURVATURE = 1 << 3
    COLLISION_PREDECESSOR = 1 << 4
    RANDOM = 1 << 5
    LAST_NONFLAT = 1 << 6
    NOVELTY = 1 << 7


@dataclass(slots=True)
class _EventCandidate:
    step: int
    descriptor: np.ndarray
    mask: EventType = EventType.NONE
    severity: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def add(
        self,
        event: EventType,
        severity: float,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> None:
        self.mask |= event
        self.severity += max(float(severity), 0.0)
        if metric_name is not None and metric_value is not None:
            self.metrics[metric_name] = max(
                self.metrics.get(metric_name, -math.inf),
                float(metric_value),
            )


@dataclass(slots=True)
class RepositoryInsertionReport:
    considered: int = 0
    triggered: int = 0
    added: int = 0
    merged: int = 0
    replaced: int = 0
    rejected_novelty: int = 0
    rejected_capacity: int = 0
    rejected_rollout_limit: int = 0
    state_ids: list[int] = field(default_factory=list)
    evicted_state_ids: list[int] = field(default_factory=list)

    @property
    def exploration_gain(self) -> int:
        return self.added


class EventFrontier:
    """Bounded event-driven collection of state-level UCB arms.

    Event extraction is performed only at and after ``new_start_step``.
    Novelty is normally an admission/deduplication criterion.  Setting
    ``config.event_novelty_as_trigger`` additionally makes every sufficiently
    novel valid state an event candidate.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        descriptor_fn: DescriptorFunction | None = None,
        gravity_fn: ScalarStateFunction | None = None,
        modes_fn: ModesFunction | None = None,
        validity_fn: ValidityFunction | None = None,
        descriptor_normalization: float | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        config.validate()
        if config.event_frontier_capacity <= 0:
            raise ValueError("event_frontier_capacity must be positive")
        if config.event_max_add_per_rollout <= 0:
            raise ValueError("event_max_add_per_rollout must be positive")
        self.config = config
        self.descriptor_fn = descriptor_fn
        self.gravity_fn = gravity_fn
        self.modes_fn = modes_fn
        self.validity_fn = validity_fn
        self.descriptor_normalization = descriptor_normalization
        self.rng = rng or np.random.default_rng(config.seed)
        self._records: dict[int, StateRecord] = {}
        self._active_ids: list[int] = []
        self._next_state_id = 0
        self._created_order = 0

    def __len__(self) -> int:
        return len(self._active_ids)

    @property
    def records(self) -> list[StateRecord]:
        return [self._records[value] for value in self._active_ids]

    @property
    def all_records(self) -> list[StateRecord]:
        return [self._records[key] for key in sorted(self._records)]

    def get(self, state_id: int) -> StateRecord:
        return self._records[int(state_id)]

    @property
    def total_visits(self) -> int:
        return sum(record.statistics.visits for record in self.records)

    def _descriptor(self, state: np.ndarray) -> np.ndarray:
        return normalized_configuration_descriptor(
            state, self.descriptor_fn, self.descriptor_normalization
        )

    def _nearest_active(
        self, descriptor: np.ndarray
    ) -> tuple[float, StateRecord | None]:
        if not self._active_ids:
            return math.inf, None
        records = self.records
        distances = np.asarray(
            [
                descriptor_rms_distance(record.descriptor, descriptor)
                for record in records
            ]
        )
        nearest = int(np.argmin(distances))
        return float(distances[nearest]), records[nearest]

    def _descriptor_sequence(
        self, trajectory: dict, path: np.ndarray
    ) -> list[np.ndarray]:
        supplied = trajectory.get("descriptors")
        if supplied is not None:
            values = np.asarray(supplied, dtype=float)
            if values.ndim != 2 or values.shape[0] != path.shape[1]:
                raise ValueError("trajectory descriptors must have shape (states, dim)")
            return [
                normalized_configuration_descriptor(
                    value, None, self.descriptor_normalization
                )
                for value in values
            ]
        return [self._descriptor(path[:, step]) for step in range(path.shape[1])]

    def _potential_sequence(
        self, trajectory: dict, path: np.ndarray
    ) -> np.ndarray | None:
        supplied = trajectory.get("potentials")
        if supplied is not None:
            values = np.asarray(supplied, dtype=float).reshape(-1)
            if len(values) != path.shape[1]:
                raise ValueError("trajectory potentials must match path columns")
            return values
        if self.gravity_fn is None:
            return None
        return np.asarray(
            [self.gravity_fn(path[:, step]) for step in range(path.shape[1])],
            dtype=float,
        )

    def _projector_sequence(
        self, trajectory: dict, path: np.ndarray
    ) -> list[np.ndarray] | None:
        supplied_projectors = trajectory.get("mode_projectors")
        if supplied_projectors is not None:
            values = [np.asarray(value, dtype=float) for value in supplied_projectors]
            if len(values) != path.shape[1]:
                raise ValueError("mode_projectors must match path columns")
            return values
        supplied_modes = trajectory.get("modes")
        if supplied_modes is not None:
            values = list(supplied_modes)
            if len(values) != path.shape[1]:
                raise ValueError("trajectory modes must match path columns")
            return [_mode_projector(value) for value in values]
        if self.modes_fn is None:
            return None
        return [
            _mode_projector(self.modes_fn(path[:, step]))
            for step in range(path.shape[1])
        ]

    def _event_candidates(
        self,
        trajectory: dict,
        new_start_step: int,
    ) -> tuple[list[_EventCandidate], int]:
        path = _trajectory_path(trajectory)
        count = path.shape[1]
        first = max(int(new_start_step), 0)
        if first >= count:
            return [], 0
        valid_mask = _valid_nonterminal_mask(
            trajectory, path, self.validity_fn
        )
        valid_steps = [
            step
            for step in range(first, count)
            if valid_mask[step]
            and step < self.config.max_path_steps
            and _intervention_count_at(trajectory, step)
            < self.config.max_interventions
        ]
        if not valid_steps:
            return [], 0

        descriptors = self._descriptor_sequence(trajectory, path)
        potentials = self._potential_sequence(trajectory, path)
        candidates: dict[int, _EventCandidate] = {}

        def candidate(step: int) -> _EventCandidate:
            if step not in candidates:
                candidates[step] = _EventCandidate(step, descriptors[step])
            return candidates[step]

        stride = max(int(self.config.event_stride), 1)
        for step in range(first, count, stride):
            if step in valid_steps:
                candidate(step).add(EventType.STRIDE, 1.0)

        window = max(int(self.config.event_peak_window), 1)
        prominence_threshold = max(
            float(self.config.event_potential_prominence), 1.0e-15
        )
        if potentials is not None:
            for step in valid_steps:
                if step - window < 0 or step + window >= count:
                    continue
                left = potentials[step - window : step]
                right = potentials[step + 1 : step + window + 1]
                peak = min(
                    float(potentials[step] - np.max(left)),
                    float(potentials[step] - np.max(right)),
                )
                valley = min(
                    float(np.min(left) - potentials[step]),
                    float(np.min(right) - potentials[step]),
                )
                prominence = max(peak, valley)
                if prominence >= self.config.event_potential_prominence:
                    candidate(step).add(
                        EventType.POTENTIAL_EXTREMUM,
                        1.0 + prominence / prominence_threshold,
                        "potential_prominence",
                        prominence,
                    )

        projectors = self._projector_sequence(trajectory, path)
        basis_threshold = max(
            float(self.config.event_basis_change_threshold), 1.0e-15
        )
        if projectors is not None:
            for step in valid_steps:
                changes: list[float] = []
                if step > 0:
                    changes.append(
                        _projector_change(projectors[step - 1], projectors[step])
                    )
                if step + 1 < count:
                    changes.append(
                        _projector_change(projectors[step], projectors[step + 1])
                    )
                change = max(changes, default=0.0)
                if change >= self.config.event_basis_change_threshold:
                    candidate(step).add(
                        EventType.BASIS_CHANGE,
                        1.0 + change / basis_threshold,
                        "basis_projector_change",
                        change,
                    )

        curvature_threshold = max(
            float(self.config.event_curvature_threshold), 1.0e-15
        )
        for step in valid_steps:
            if step <= 0 or step + 1 >= count:
                continue
            before = descriptors[step] - descriptors[step - 1]
            after = descriptors[step + 1] - descriptors[step]
            denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
            if denominator <= 1.0e-14:
                curvature = 0.0
            else:
                cosine = float(np.dot(before, after) / denominator)
                curvature = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            if curvature >= self.config.event_curvature_threshold:
                candidate(step).add(
                    EventType.DESCRIPTOR_CURVATURE,
                    1.0 + curvature / curvature_threshold,
                    "descriptor_curvature",
                    curvature,
                )

        if _terminal_reason(trajectory) in {"collision", "collided"}:
            predecessors = [step for step in valid_steps if step < count - 1]
            if predecessors:
                step = max(predecessors)
                candidate(step).add(EventType.COLLISION_PREDECESSOR, 4.0)

        if potentials is not None and (
            _terminal_reason(trajectory) == "flat"
            or float(potentials[-1]) <= self.config.flatness_tolerance
        ):
            for step in range(count - 2, first - 1, -1):
                if (
                    valid_mask[step]
                    and float(potentials[step]) > self.config.layer_threshold
                ):
                    candidate(step).add(EventType.LAST_NONFLAT, 3.0)
                    break

        random_count = min(
            max(int(self.config.event_random_points), 0), len(valid_steps)
        )
        if random_count:
            for step in self.rng.choice(
                np.asarray(valid_steps), size=random_count, replace=False
            ):
                candidate(int(step)).add(EventType.RANDOM, 0.75)

        if self.config.event_novelty_as_trigger:
            novelty_threshold = max(
                float(self.config.event_novelty_tolerance), 1.0e-15
            )
            for step in valid_steps:
                distance, _ = self._nearest_active(descriptors[step])
                if distance > self.config.event_novelty_tolerance:
                    finite_distance = (
                        distance if math.isfinite(distance) else 3.0 * novelty_threshold
                    )
                    candidate(step).add(
                        EventType.NOVELTY,
                        1.0 + min(finite_distance / novelty_threshold, 3.0),
                        "novelty_distance",
                        finite_distance,
                    )

        ordered = sorted(
            candidates.values(), key=lambda value: (-value.severity, value.step)
        )
        selected: list[_EventCandidate] = []
        nms_radius = max(int(self.config.event_peak_window), 0)
        for value in ordered:
            if any(abs(value.step - old.step) <= nms_radius for old in selected):
                continue
            selected.append(value)
        return selected, len(valid_steps)

    def _evict_for(self, severity: float) -> StateRecord | None:
        if len(self._active_ids) < self.config.event_frontier_capacity:
            return None
        active = self.records
        victim = min(
            active,
            key=lambda value: (
                value.event_severity,
                -value.created_order,
                value.state_id,
            ),
        )
        if severity <= victim.event_severity:
            raise OverflowError
        victim.active = False
        self._active_ids.remove(victim.state_id)
        return victim

    def add_trajectory(
        self,
        trajectory: dict,
        *,
        new_start_step: int | None = None,
        source_rollout: Any = None,
    ) -> RepositoryInsertionReport:
        """Extract events and update the bounded active frontier."""

        if new_start_step is None:
            new_start_step = int(trajectory.get("new_start_step", 0))
        if source_rollout is None:
            source_rollout = trajectory.get(
                "history_id", trajectory.get("source")
            )
        candidates, considered = self._event_candidates(
            trajectory, new_start_step
        )
        report = RepositoryInsertionReport(
            considered=considered, triggered=len(candidates)
        )
        for value in candidates:
            distance, nearest = self._nearest_active(value.descriptor)
            if (
                nearest is not None
                and distance <= self.config.event_merge_tolerance
            ):
                nearest.event_mask |= int(value.mask)
                nearest.event_severity = max(
                    nearest.event_severity, value.severity
                )
                for name, metric in value.metrics.items():
                    nearest.event_metrics[name] = max(
                        nearest.event_metrics.get(name, -math.inf), metric
                    )
                nearest.merged_count += 1
                report.merged += 1
                report.state_ids.append(nearest.state_id)
                continue
            if distance <= self.config.event_novelty_tolerance:
                report.rejected_novelty += 1
                continue
            if report.added >= self.config.event_max_add_per_rollout:
                report.rejected_rollout_limit += 1
                continue

            novelty_bonus = (
                min(
                    distance / max(self.config.event_novelty_tolerance, 1.0e-15),
                    3.0,
                )
                if math.isfinite(distance)
                else 3.0
            )
            admission_severity = value.severity + 0.1 * novelty_bonus
            try:
                victim = self._evict_for(admission_severity)
            except OverflowError:
                report.rejected_capacity += 1
                continue
            if victim is not None:
                report.replaced += 1
                report.evicted_state_ids.append(victim.state_id)

            state_id = self._next_state_id
            self._next_state_id += 1
            record = StateRecord(
                state_id=state_id,
                trajectory=trajectory,
                step=value.step,
                descriptor=value.descriptor.copy(),
                intervention_count=_intervention_count_at(
                    trajectory, value.step
                ),
                created_order=self._created_order,
                source_rollout=source_rollout,
                event_mask=int(value.mask),
                event_severity=admission_severity,
                event_metrics=dict(value.metrics),
            )
            self._created_order += 1
            self._records[state_id] = record
            self._active_ids.append(state_id)
            report.added += 1
            report.state_ids.append(state_id)
        return report

    ingest_rollout = add_trajectory

    def eligible_records(self) -> list[StateRecord]:
        return [
            record for record in self.records if _state_is_eligible(record, self.config)
        ]

    def select(
        self, *, total_visits: int | None = None
    ) -> StateRecord:
        eligible = self.eligible_records()
        return select_ucb_arm(
            eligible,
            total_visits=self.total_visits if total_visits is None else total_visits,
            exploration=self.config.ucb_exploration,
            rng=self.rng,
        )

    def update(
        self,
        state_or_id: StateRecord | int,
        reward: float,
        *,
        simulated_steps: int = 0,
        success: bool = False,
    ) -> None:
        record = (
            state_or_id
            if isinstance(state_or_id, StateRecord)
            else self._records[int(state_or_id)]
        )
        update_arm_statistics(
            record.statistics,
            reward,
            simulated_steps=simulated_steps,
            success=success,
        )


# ---------------------------------------------------------------------------
# Fixed-bin bounded state archive
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StateCell:
    """A fixed descriptor bin acting as one UCB arm."""

    cell_id: int
    key: tuple[int, ...]
    created_order: int
    statistics: ArmStatistics = field(default_factory=ArmStatistics)
    representative_ids: list[int] = field(default_factory=list)
    states_seen: int = 0
    active: bool = True

    @property
    def stats(self) -> ArmStatistics:
        return self.statistics

    @property
    def visits(self) -> int:
        return self.statistics.visits

    @property
    def reward_sum(self) -> float:
        return self.statistics.reward_sum


@dataclass(slots=True)
class BoundedArchiveInsertionReport:
    considered: int = 0
    new_cells_seen: int = 0
    active_cells_added: int = 0
    active_cells_evicted: int = 0
    representatives_added: int = 0
    representatives_replaced: int = 0
    states_rejected_inactive_cell: int = 0
    state_ids: list[int] = field(default_factory=list)
    cell_ids: list[int] = field(default_factory=list)

    @property
    def exploration_gain(self) -> int:
        return self.active_cells_added + self.representatives_replaced


class BoundedStateArchive:
    """Fixed-bin active archive with reservoir-sampled representatives.

    Each active descriptor cell is one UCB arm.  Arm statistics survive
    representative replacement.  When the cell capacity is reached, distinct
    cells themselves are maintained by reservoir sampling; evicted cells keep
    their stable IDs and statistics but are no longer eligible.  Neither cell
    nor representative retention consults path quality.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        descriptor_fn: DescriptorFunction | None = None,
        validity_fn: ValidityFunction | None = None,
        descriptor_normalization: float | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        config.validate()
        if config.state_archive_max_cells <= 0:
            raise ValueError("state_archive_max_cells must be positive")
        self.config = config
        self.descriptor_fn = descriptor_fn
        self.validity_fn = validity_fn
        self.descriptor_normalization = descriptor_normalization
        self.rng = rng or np.random.default_rng(config.seed)
        self._states: dict[int, StateRecord] = {}
        self._cells: dict[int, StateCell] = {}
        self._bin_to_cell_id: dict[tuple[int, ...], int] = {}
        self._active_cell_ids: list[int] = []
        self._next_state_id = 0
        self._next_cell_id = 0
        self._created_order = 0
        self._distinct_cells_seen = 0

    def __len__(self) -> int:
        return len(self._active_cell_ids)

    @property
    def cells(self) -> list[StateCell]:
        return [self._cells[value] for value in self._active_cell_ids]

    @property
    def all_cells(self) -> list[StateCell]:
        return [self._cells[key] for key in sorted(self._cells)]

    def get_cell(self, cell_id: int) -> StateCell:
        return self._cells[int(cell_id)]

    def get_state(self, state_id: int) -> StateRecord:
        return self._states[int(state_id)]

    @property
    def records(self) -> list[StateRecord]:
        result: list[StateRecord] = []
        for cell in self.cells:
            result.extend(
                self._states[state_id] for state_id in cell.representative_ids
            )
        return result

    @property
    def total_visits(self) -> int:
        return sum(cell.statistics.visits for cell in self.cells)

    def _descriptor(self, state: np.ndarray) -> np.ndarray:
        return normalized_configuration_descriptor(
            state, self.descriptor_fn, self.descriptor_normalization
        )

    def bin_key(self, descriptor: np.ndarray) -> tuple[int, ...]:
        """Quantize a normalized descriptor into a fixed-width centered bin."""

        width = float(self.config.state_bin_width)
        values = np.floor(np.asarray(descriptor) / width + 0.5).astype(np.int64)
        return tuple(int(value) for value in values)

    def _new_cell(self, key: tuple[int, ...]) -> tuple[StateCell, int | None]:
        cell_id = self._next_cell_id
        self._next_cell_id += 1
        cell = StateCell(
            cell_id=cell_id,
            key=key,
            created_order=self._created_order,
            active=False,
        )
        self._created_order += 1
        self._cells[cell_id] = cell
        self._bin_to_cell_id[key] = cell_id
        self._distinct_cells_seen += 1

        evicted: int | None = None
        capacity = self.config.state_archive_max_cells
        if len(self._active_cell_ids) < capacity:
            cell.active = True
            self._active_cell_ids.append(cell_id)
        else:
            slot = int(self.rng.integers(0, self._distinct_cells_seen))
            if slot < capacity:
                evicted = self._active_cell_ids[slot]
                old = self._cells[evicted]
                old.active = False
                for state_id in old.representative_ids:
                    self._states[state_id].active = False
                self._active_cell_ids[slot] = cell_id
                cell.active = True
        return cell, evicted

    def _store_representative(
        self,
        cell: StateCell,
        trajectory: dict,
        step: int,
        descriptor: np.ndarray,
        source_rollout: Any,
    ) -> tuple[str, int | None]:
        maximum = self.config.state_representatives_per_cell
        if len(cell.representative_ids) < maximum:
            slot = len(cell.representative_ids)
            action = "added"
        else:
            slot = int(self.rng.integers(0, cell.states_seen))
            if slot >= maximum:
                return "ignored", None
            old_id = cell.representative_ids[slot]
            self._states[old_id].active = False
            action = "replaced"

        state_id = self._next_state_id
        self._next_state_id += 1
        record = StateRecord(
            state_id=state_id,
            trajectory=trajectory,
            step=step,
            descriptor=descriptor.copy(),
            intervention_count=_intervention_count_at(trajectory, step),
            created_order=self._created_order,
            source_rollout=source_rollout,
        )
        self._created_order += 1
        self._states[state_id] = record
        if action == "added":
            cell.representative_ids.append(state_id)
        else:
            cell.representative_ids[slot] = state_id
        return action, state_id

    def add_trajectory(
        self,
        trajectory: dict,
        *,
        new_start_step: int | None = None,
        source_rollout: Any = None,
    ) -> BoundedArchiveInsertionReport:
        """Offer every valid nonterminal state after ``new_start_step``."""

        path = _trajectory_path(trajectory)
        if new_start_step is None:
            new_start_step = int(trajectory.get("new_start_step", 0))
        if source_rollout is None:
            source_rollout = trajectory.get(
                "history_id", trajectory.get("source")
            )
        valid_mask = _valid_nonterminal_mask(
            trajectory, path, self.validity_fn
        )
        report = BoundedArchiveInsertionReport()
        for step in range(max(int(new_start_step), 0), path.shape[1]):
            if not valid_mask[step]:
                continue
            report.considered += 1
            descriptor = self._descriptor(path[:, step])
            key = self.bin_key(descriptor)
            cell_id = self._bin_to_cell_id.get(key)
            if cell_id is None:
                cell, evicted = self._new_cell(key)
                report.new_cells_seen += 1
                if cell.active:
                    report.active_cells_added += 1
                if evicted is not None:
                    report.active_cells_evicted += 1
            else:
                cell = self._cells[cell_id]
            report.cell_ids.append(cell.cell_id)
            # Count every observation of the bin, including observations made
            # after a global cell-reservoir eviction.
            cell.states_seen += 1
            if not cell.active:
                report.states_rejected_inactive_cell += 1
                continue
            action, state_id = self._store_representative(
                cell, trajectory, step, descriptor, source_rollout
            )
            if action == "added":
                report.representatives_added += 1
            elif action == "replaced":
                report.representatives_replaced += 1
            if state_id is not None:
                report.state_ids.append(state_id)
        return report

    ingest_rollout = add_trajectory

    def eligible_representatives(self, cell: StateCell) -> list[StateRecord]:
        if not cell.active:
            return []
        return [
            self._states[state_id]
            for state_id in cell.representative_ids
            if _state_is_eligible(self._states[state_id], self.config)
        ]

    def eligible_cells(self) -> list[StateCell]:
        return [
            cell for cell in self.cells if self.eligible_representatives(cell)
        ]

    def select_cell(
        self, *, total_visits: int | None = None
    ) -> StateCell:
        return select_ucb_arm(
            self.eligible_cells(),
            total_visits=self.total_visits if total_visits is None else total_visits,
            exploration=self.config.ucb_exploration,
            rng=self.rng,
        )

    def select(
        self, *, total_visits: int | None = None
    ) -> tuple[StateCell, StateRecord]:
        """Select a UCB cell and then one eligible representative uniformly."""

        cell = self.select_cell(total_visits=total_visits)
        representatives = self.eligible_representatives(cell)
        state = representatives[int(self.rng.integers(len(representatives)))]
        return cell, state

    def update(
        self,
        cell_or_id: StateCell | int,
        reward: float,
        *,
        simulated_steps: int = 0,
        success: bool = False,
    ) -> None:
        cell = (
            cell_or_id
            if isinstance(cell_or_id, StateCell)
            else self._cells[int(cell_or_id)]
        )
        update_arm_statistics(
            cell.statistics,
            reward,
            simulated_steps=simulated_steps,
            success=success,
        )


__all__ = [
    "ArmStatistics",
    "BoundedArchiveInsertionReport",
    "BoundedStateArchive",
    "EventFrontier",
    "EventType",
    "RepositoryInsertionReport",
    "StateCell",
    "StateRecord",
    "descriptor_rms_distance",
    "normalized_configuration_descriptor",
    "select_ucb_arm",
    "ucb_score",
    "update_arm_statistics",
]
