"""Equal-cardinality state sets for the nonstationarity experiments.

The current protocol compares *where* a perturbation may be applied before it
compares selection scores.  Every generated state is assigned one immutable
``state_id`` and one immutable random priority.  Three raw universes are kept:

``history``
    Every newly simulated, valid, non-flat state.
``frontier``
    The deepest restartable state of every completed rollout.
``event``
    Every frontier state plus states selected by deterministic geometric/path
    events (stride, potential extremum, mechanism-basis change, path curvature,
    collision predecessor, and last non-flat state).

The active sets are bottom-k priority samples.  At every refresh their common
size is

    min(capacity, |frontier|, |event|, |history|).

Thus candidate count cannot explain a difference between the three rules.  No
event severity, novelty rejection, or UCB statistic is used for admission.
Newly generated states are visible only after the next search epoch begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
import hashlib
import math
from typing import Any

import numpy as np

from .descriptors import configuration_descriptor
from .mechanics import (
    Model,
    gravity_potential,
    mechanism_modes,
    vector_to_node,
)


POOL_KINDS = ("frontier", "event", "history")


class EventType(IntFlag):
    """Deterministic events used only to define the event universe."""

    NONE = 0
    STRIDE = 1 << 0
    POTENTIAL_EXTREMUM = 1 << 1
    BASIS_CHANGE = 1 << 2
    DESCRIPTOR_CURVATURE = 1 << 3
    COLLISION_PREDECESSOR = 1 << 4
    LAST_NONFLAT = 1 << 5
    ROLLOUT_BOUNDARY = 1 << 6


@dataclass(slots=True)
class CandidateState:
    """One immutable restart state and the full prefix needed to reproduce it."""

    state_id: int
    trajectory: dict
    step: int
    descriptor: np.ndarray
    feature: np.ndarray
    depth: int
    intervention_count: int
    potential: float
    mechanism_dimension: int
    priority: float
    created_order: int
    source_rollout: Any = None
    event_mask: int = 0
    is_frontier: bool = False

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self.trajectory["path"], dtype=float)[:, self.step]


@dataclass(slots=True)
class CandidateInsertionReport:
    """Diagnostics for one bootstrap path or one newly simulated rollout."""

    considered: int = 0
    eligible_added: int = 0
    event_added: int = 0
    frontier_added: int = 0
    state_ids: list[int] = field(default_factory=list)
    event_state_ids: list[int] = field(default_factory=list)
    frontier_state_id: int | None = None


def _stable_priority(seed: int, state_id: int) -> float:
    """Map ``(seed, state_id)`` to a reproducible value in the open unit interval."""

    payload = f"{int(seed)}:{int(state_id)}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(2**64)


def _intervention_count_at(trajectory: dict, step: int) -> int:
    """Return interventions already present at a path column."""

    if "intervention_counts" in trajectory:
        values = np.asarray(trajectory["intervention_counts"], dtype=int).reshape(-1)
        if len(values) != np.asarray(trajectory["path"]).shape[1]:
            raise ValueError("intervention_counts must match the trajectory length")
        return max(int(values[step]), 0)
    intervention_steps = trajectory.get("intervention_steps", [])
    return sum(int(value) <= int(step) for value in intervention_steps)


def _mode_projector(modes: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto a mechanism subspace."""

    matrix = np.asarray(modes, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("mechanism modes must be a matrix")
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
        raise ValueError("mechanism projectors have incompatible shapes")
    scale = math.sqrt(max(first.shape[0], 1))
    return float(np.linalg.norm(first - second, ord="fro") / scale)


class CandidatePools:
    """Maintain nested raw universes and equal-cardinality active samples.

    The three universes are all updated in every run, even though only one of
    them is used to select perturbation origins.  This makes the cardinality
    equality directly testable in the log of every condition.
    """

    def __init__(self, model: Model, config, *, priority_seed: int) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.priority_seed = int(priority_seed)
        self.records: dict[int, CandidateState] = {}
        self.universes: dict[str, set[int]] = {
            kind: set() for kind in POOL_KINDS
        }
        self._next_state_id = 0
        self._created_order = 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def raw_sizes(self) -> dict[str, int]:
        return {kind: len(self.universes[kind]) for kind in POOL_KINDS}

    @property
    def common_size(self) -> int:
        sizes = self.raw_sizes
        return min(
            int(self.config.candidate_pool_capacity),
            *(sizes[kind] for kind in POOL_KINDS),
        )

    def _bottom_k(self, kind: str, count: int) -> list[CandidateState]:
        if kind not in POOL_KINDS:
            raise ValueError(f"unknown candidate-pool kind: {kind}")
        ordered = sorted(
            self.universes[kind],
            key=lambda state_id: (
                self.records[state_id].priority,
                state_id,
            ),
        )
        return [self.records[state_id] for state_id in ordered[:count]]

    def active_sets(self) -> dict[str, list[CandidateState]]:
        """Return a frozen, equal-size snapshot for the next search epoch."""

        count = self.common_size
        if count <= 0:
            raise LookupError("candidate pools are empty")
        result = {
            kind: self._bottom_k(kind, count) for kind in POOL_KINDS
        }
        sizes = {len(values) for values in result.values()}
        if sizes != {count}:
            raise RuntimeError("candidate pools lost the equal-cardinality invariant")
        return result

    def _state_feature(
        self,
        descriptor: np.ndarray,
        *,
        depth: int,
        potential: float,
        mechanism_dimension: int,
        intervention_count: int,
    ) -> np.ndarray:
        """Feature used by SM-UCB; it includes depth because rollout uses it."""

        length_scale = max(float(self.model.length_scale), 1.0e-12)
        potential_scale = max(
            float(self.model.num_node) * length_scale,
            1.0e-12,
        )
        path_scale = max(int(self.config.max_path_steps), 1)
        return np.concatenate(
            [
                np.asarray(descriptor, dtype=float).reshape(-1),
                np.asarray(
                    [
                        depth / path_scale,
                        (path_scale - depth) / path_scale,
                        potential / potential_scale,
                        mechanism_dimension / max(self.model.vector_size, 1),
                        intervention_count / max(depth, 1),
                    ],
                    dtype=float,
                ),
            ]
        )

    def ingest_trajectory(
        self,
        trajectory: dict,
        *,
        first_new_column: int,
        source_rollout: Any = None,
    ) -> CandidateInsertionReport:
        """Add only newly simulated suffix states from one complete rollout.

        Prefix columns are references to already existing states and must never
        be counted again.  The caller therefore passes ``new_start_step + 1``
        for ordinary restart rollouts and ``1`` for each bootstrap path (the
        common root is deliberately excluded from every candidate universe).
        """

        path = np.asarray(trajectory["path"], dtype=float)
        if path.ndim != 2 or path.shape[0] != self.model.vector_size:
            raise ValueError("trajectory path has an incompatible shape")
        if np.any(~np.isfinite(path)):
            raise ValueError("trajectory path must be finite")
        first = max(int(first_new_column), 0)
        report = CandidateInsertionReport()
        if first >= path.shape[1]:
            return report

        descriptor_cache: dict[int, np.ndarray] = {}
        potential_cache: dict[int, float] = {}
        modes_cache: dict[int, np.ndarray] = {}
        projector_cache: dict[int, np.ndarray] = {}

        def descriptor_at(step: int) -> np.ndarray:
            if step not in descriptor_cache:
                descriptor_cache[step] = configuration_descriptor(
                    path[:, step], self.model
                )
            return descriptor_cache[step]

        def potential_at(step: int) -> float:
            if step not in potential_cache:
                potential_cache[step] = gravity_potential(
                    vector_to_node(path[:, step], self.model)
                )
            return potential_cache[step]

        def modes_at(step: int) -> np.ndarray:
            if step not in modes_cache:
                modes_cache[step] = mechanism_modes(
                    vector_to_node(path[:, step], self.model), self.model
                )
            return modes_cache[step]

        def projector_at(step: int) -> np.ndarray:
            if step not in projector_cache:
                projector_cache[step] = _mode_projector(modes_at(step))
            return projector_cache[step]

        reason = str(trajectory.get("terminal_reason", "running")).lower()
        invalid_final_reasons = {
            "flat",
            "collision",
            "collided",
            "correction_failed",
            "no_mechanism",
            "no_projected_perturbation",
            "optimizer_failed",
            "invalid_sequence",
        }
        final_step = path.shape[1] - 1
        new_records: dict[int, CandidateState] = {}
        for step in range(first, path.shape[1]):
            report.considered += 1
            if step <= 0 or step >= self.config.max_path_steps:
                continue
            if step == final_step and reason in invalid_final_reasons:
                continue
            potential = potential_at(step)
            if potential <= self.config.flatness_tolerance:
                continue
            try:
                modes = modes_at(step)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            mechanism_dimension = int(modes.shape[1])
            if mechanism_dimension <= 0:
                continue

            descriptor = descriptor_at(step)
            intervention_count = _intervention_count_at(trajectory, step)
            state_id = self._next_state_id
            self._next_state_id += 1
            record = CandidateState(
                state_id=state_id,
                trajectory=trajectory,
                step=step,
                descriptor=descriptor.copy(),
                feature=self._state_feature(
                    descriptor,
                    depth=step,
                    potential=potential,
                    mechanism_dimension=mechanism_dimension,
                    intervention_count=intervention_count,
                ),
                depth=step,
                intervention_count=intervention_count,
                potential=float(potential),
                mechanism_dimension=mechanism_dimension,
                priority=_stable_priority(self.priority_seed, state_id),
                created_order=self._created_order,
                source_rollout=source_rollout,
            )
            self._created_order += 1
            self.records[state_id] = record
            self.universes["history"].add(state_id)
            new_records[step] = record
            report.eligible_added += 1
            report.state_ids.append(state_id)

        if not new_records:
            return report

        # Event decisions are retrospective but use only this completed
        # trajectory.  They do not inspect the terminal outcome archive.
        stride = max(int(self.config.event_stride), 1)
        prominence_threshold = max(
            float(self.config.event_potential_prominence), 1.0e-15
        )
        basis_threshold = max(
            float(self.config.event_basis_change_threshold), 1.0e-15
        )
        curvature_threshold = max(
            float(self.config.event_curvature_threshold), 1.0e-15
        )
        window = max(int(self.config.event_peak_window), 1)

        for step, record in new_records.items():
            mask = EventType.NONE
            if step % stride == 0:
                mask |= EventType.STRIDE

            if step - window >= 0 and step + window < path.shape[1]:
                center = potential_at(step)
                left = [potential_at(value) for value in range(step - window, step)]
                right = [
                    potential_at(value)
                    for value in range(step + 1, step + window + 1)
                ]
                peak = min(center - max(left), center - max(right))
                valley = min(min(left) - center, min(right) - center)
                if max(peak, valley) >= prominence_threshold:
                    mask |= EventType.POTENTIAL_EXTREMUM

            changes: list[float] = []
            try:
                if step > 0:
                    changes.append(
                        _projector_change(projector_at(step - 1), projector_at(step))
                    )
                if step + 1 < path.shape[1]:
                    changes.append(
                        _projector_change(projector_at(step), projector_at(step + 1))
                    )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                changes = []
            if max(changes, default=0.0) >= basis_threshold:
                mask |= EventType.BASIS_CHANGE

            if 0 < step < path.shape[1] - 1:
                before = descriptor_at(step) - descriptor_at(step - 1)
                after = descriptor_at(step + 1) - descriptor_at(step)
                denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
                if denominator > 1.0e-14:
                    cosine = float(np.dot(before, after) / denominator)
                    curvature = math.acos(float(np.clip(cosine, -1.0, 1.0)))
                    if curvature >= curvature_threshold:
                        mask |= EventType.DESCRIPTOR_CURVATURE
            record.event_mask |= int(mask)

        boundary = new_records[max(new_records)]
        boundary.is_frontier = True
        boundary.event_mask |= int(EventType.ROLLOUT_BOUNDARY)
        self.universes["frontier"].add(boundary.state_id)
        report.frontier_added = 1
        report.frontier_state_id = boundary.state_id

        if reason in {"collision", "collided"}:
            boundary.event_mask |= int(EventType.COLLISION_PREDECESSOR)

        if reason == "flat":
            for step in sorted(new_records, reverse=True):
                if potential_at(step) > self.config.layer_threshold:
                    new_records[step].event_mask |= int(EventType.LAST_NONFLAT)
                    break

        for record in new_records.values():
            if EventType(record.event_mask) != EventType.NONE:
                self.universes["event"].add(record.state_id)
                report.event_added += 1
                report.event_state_ids.append(record.state_id)

        # Boundary insertion above makes frontier a subset of event, while all
        # event states were already inserted into history.
        if not self.universes["frontier"] <= self.universes["event"]:
            raise RuntimeError("frontier must be a subset of event")
        if not self.universes["event"] <= self.universes["history"]:
            raise RuntimeError("event must be a subset of history")
        return report


__all__ = [
    "CandidateInsertionReport",
    "CandidatePools",
    "CandidateState",
    "EventType",
    "POOL_KINDS",
]
