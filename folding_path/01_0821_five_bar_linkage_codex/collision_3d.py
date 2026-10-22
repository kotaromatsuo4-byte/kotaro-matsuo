"""Continuous proper-crossing detection for linearly moving 3-D bars.

The linkage model uses zero-thickness bars.  Consequently this module treats
only a *proper* intersection of the interiors of two non-adjacent bars during
the open motion interval ``0 < t < 1`` as a collision.  Adjacent bars,
endpoint-only contact, parallel contact, and collinear overlap are allowed.
Those semantics are deliberate: a mathematically flat terminal state can
contain coincident bars, while a physical thickness/layer model is outside
the information contained in ``member.dat``.

Every node is assumed to move linearly between two corrected configurations.
For a non-coplanar bar pair, an intersection can occur only at a real root of
the cubic scalar-triple-product (coplanarity) polynomial; a sign change is
required so tangential contact is not mistaken for pass-through.  A pair that
is coplanar for the whole transition is handled by partitioning time at all
projected orientation events and detecting entry from a non-intersecting into
a properly intersecting interval.  Thus a persistent/static contact is not a
new crossing.  Swept AABBs reject the vast majority of member pairs before
either operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


COLLISION_BACKEND_NAME = "continuous_proper_crossing_3d"
COLLISION_BACKEND_VERSION = "1.1.0"
COLLISION_SEMANTICS = (
    "new entry into a nonadjacent zero-thickness bar-interior proper crossing "
    "in the open linear-motion interval; persistent/static intersection, "
    "endpoint contact, and parallel/collinear overlap are allowed"
)


@dataclass(frozen=True)
class CollisionEvent:
    """The first detected proper crossing in member-index coordinates."""

    first_member: int
    second_member: int
    time: float
    first_parameter: float
    second_parameter: float
    distance: float
    coplanar_transition: bool


@dataclass(frozen=True)
class CollisionCheckResult:
    """Collision result plus inexpensive broad/narrow-phase diagnostics."""

    collision: bool
    event: Optional[CollisionEvent]
    total_nonadjacent_pairs: int
    broad_phase_candidates: int
    narrow_phase_checks: int


def nonadjacent_member_pairs(member):
    """Return unordered member-index pairs that share no node."""

    members = _validate_member(member)
    pairs = []
    for first in range(len(members)):
        first_nodes = {int(members[first, 0]), int(members[first, 1])}
        for second in range(first + 1, len(members)):
            if first_nodes.isdisjoint(
                (int(members[second, 0]), int(members[second, 1]))
            ):
                pairs.append((first, second))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _validate_member(member):
    array = np.asarray(member)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("member must have shape (number_of_members, 2).")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
            raise ValueError("member indices must be finite integers.")
    array = array.astype(int, copy=False)
    if np.any(array < 0):
        raise ValueError("member indices cannot be negative.")
    return array


def _validate_nodes(node_before, node_after, member):
    before = np.asarray(node_before, dtype=float)
    after = np.asarray(node_after, dtype=float)
    if before.ndim != 2 or before.shape[1] != 3:
        raise ValueError("node_before must have shape (number_of_nodes, 3).")
    if after.shape != before.shape:
        raise ValueError("node_after must have the same shape as node_before.")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("node coordinates must be finite.")
    if len(member) and int(np.max(member)) >= len(before):
        raise ValueError("member contains a node index outside the node array.")
    return before, after


def _triple(first, second, third):
    return float(np.dot(first, np.cross(second, third)))


def _coplanarity_coefficients(a0, a1, b0, b1, c0, c1, d0, d1):
    """Ascending coefficients of det(B-A, C-A, D-A) over time."""

    p0 = b0 - a0
    p1 = (b1 - b0) - (a1 - a0)
    q0 = c0 - a0
    q1 = (c1 - c0) - (a1 - a0)
    r0 = d0 - a0
    r1 = (d1 - d0) - (a1 - a0)
    return np.asarray(
        [
            _triple(p0, q0, r0),
            _triple(p1, q0, r0)
            + _triple(p0, q1, r0)
            + _triple(p0, q0, r1),
            _triple(p1, q1, r0)
            + _triple(p1, q0, r1)
            + _triple(p0, q1, r1),
            _triple(p1, q1, r1),
        ],
        dtype=float,
    )


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation_coefficients_2d(a0, a1, b0, b1, c0, c1):
    """Ascending coefficients of orient(A(t), B(t), C(t))."""

    u0 = b0 - a0
    u1 = (b1 - b0) - (a1 - a0)
    v0 = c0 - a0
    v1 = (c1 - c0) - (a1 - a0)
    return np.asarray(
        [
            _cross_2d(u0, v0),
            _cross_2d(u1, v0) + _cross_2d(u0, v1),
            _cross_2d(u1, v1),
        ],
        dtype=float,
    )


def _real_roots_open(coefficients, coefficient_tolerance, time_tolerance):
    """Return unique real polynomial roots strictly inside (0, 1)."""

    coefficients = np.asarray(coefficients, dtype=float)
    scale = float(np.max(np.abs(coefficients))) if len(coefficients) else 0.0
    if scale <= coefficient_tolerance:
        return []
    normalised = coefficients / scale
    highest = len(normalised) - 1
    while highest > 0 and abs(normalised[highest]) <= coefficient_tolerance:
        highest -= 1
    if highest == 0:
        return []
    roots = []
    for root in np.roots(normalised[: highest + 1][::-1]):
        if abs(float(root.imag)) <= 1.0e-8:
            time = float(root.real)
            if time_tolerance < time < 1.0 - time_tolerance:
                roots.append(time)
    roots.sort()
    unique = []
    for value in roots:
        if not unique or value - unique[-1] > time_tolerance:
            unique.append(value)
    return unique


def _polynomial_value(coefficients, time):
    """Evaluate an ascending-order polynomial without reversing allocations."""

    value = 0.0
    for coefficient in np.asarray(coefficients, dtype=float)[::-1]:
        value = value * float(time) + float(coefficient)
    return float(value)


def _sign_changing_roots(
    coefficients,
    coefficient_tolerance,
    time_tolerance,
):
    """Return open roots across which the polynomial actually changes sign."""

    roots = _real_roots_open(
        coefficients, coefficient_tolerance, time_tolerance
    )
    if not roots:
        return []
    boundaries = [0.0, *roots, 1.0]
    result = []
    scale = max(float(np.max(np.abs(coefficients))), 1.0e-300)
    value_tolerance = coefficient_tolerance * scale
    for index, root in enumerate(roots, start=1):
        left = 0.5 * (boundaries[index - 1] + root)
        right = 0.5 * (root + boundaries[index + 1])
        left_value = _polynomial_value(coefficients, left)
        right_value = _polynomial_value(coefficients, right)
        if (
            abs(left_value) > value_tolerance
            and abs(right_value) > value_tolerance
            and left_value * right_value < 0.0
        ):
            result.append(root)
    return result


def _geometry_scale(points):
    points = np.asarray(points, dtype=float)
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return max(float(np.max(np.linalg.norm(differences, axis=2))), 1.0e-12)


def _static_proper_intersection(
    first_start,
    first_end,
    second_start,
    second_end,
    distance_tolerance,
    parameter_tolerance,
    parallel_sine_tolerance,
):
    """Return closest data only for a transversal interior intersection."""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    first_length = float(np.linalg.norm(first_direction))
    second_length = float(np.linalg.norm(second_direction))
    if first_length <= distance_tolerance or second_length <= distance_tolerance:
        return None
    cross_norm = float(np.linalg.norm(np.cross(
        first_direction, second_direction
    )))
    if cross_norm <= (
        parallel_sine_tolerance * first_length * second_length
    ):
        # Parallel and collinear contact/overlap are deliberately admissible.
        return None

    offset = first_start - second_start
    aa = float(np.dot(first_direction, first_direction))
    bb = float(np.dot(first_direction, second_direction))
    cc = float(np.dot(second_direction, second_direction))
    dd = float(np.dot(first_direction, offset))
    ee = float(np.dot(second_direction, offset))
    denominator = aa * cc - bb * bb
    if denominator <= (
        parallel_sine_tolerance * first_length * second_length
    ) ** 2:
        return None
    first_parameter = (bb * ee - cc * dd) / denominator
    second_parameter = (aa * ee - bb * dd) / denominator
    if not (
        parameter_tolerance < first_parameter < 1.0 - parameter_tolerance
        and parameter_tolerance
        < second_parameter
        < 1.0 - parameter_tolerance
    ):
        # An endpoint-only touch is not a proper bar-interior crossing.
        return None
    first_point = first_start + first_parameter * first_direction
    second_point = second_start + second_parameter * second_direction
    distance = float(np.linalg.norm(first_point - second_point))
    if distance > distance_tolerance:
        return None
    return first_parameter, second_parameter, distance


def _node_at_time(start, end, time):
    return start + float(time) * (end - start)


def _intersection_at_time(
    endpoints,
    time,
    distance_tolerance,
    parameter_tolerance,
    parallel_sine_tolerance,
):
    a0, a1, b0, b1, c0, c1, d0, d1 = endpoints
    return _static_proper_intersection(
        _node_at_time(a0, a1, time),
        _node_at_time(b0, b1, time),
        _node_at_time(c0, c1, time),
        _node_at_time(d0, d1, time),
        distance_tolerance,
        parameter_tolerance,
        parallel_sine_tolerance,
    )


def _coplanar_event_times(endpoints, coefficient_tolerance, time_tolerance):
    """Partition a coplanar transition at every projected orientation event."""

    a0, a1, b0, b1, c0, c1, d0, d1 = endpoints
    triples = (
        (a0, a1, b0, b1, c0, c1),
        (a0, a1, b0, b1, d0, d1),
        (c0, c1, d0, d1, a0, a1),
        (c0, c1, d0, d1, b0, b1),
    )
    roots = []
    # Taking roots in all three coordinate projections avoids depending on a
    # single plane normal that can rotate or become edge-on during the step.
    for dimensions in ((0, 1), (0, 2), (1, 2)):
        for triple in triples:
            projected = tuple(point[list(dimensions)] for point in triple)
            roots.extend(_real_roots_open(
                _orientation_coefficients_2d(*projected),
                coefficient_tolerance,
                time_tolerance,
            ))
    roots.sort()
    unique = []
    for value in roots:
        if not unique or value - unique[-1] > time_tolerance:
            unique.append(value)
    return unique


class ContinuousBarCollisionChecker:
    """Reusable collision checker with cached topology and pair arrays.

    ``parameter_tolerance`` is a dimensionless fraction of each bar length.
    The default therefore gives a scale-aware endpoint neighbourhood: an
    apparent intersection within 0.1 percent of either endpoint is treated as
    endpoint contact.  This absorbs the residual coordinate error of solved
    flat configurations without weakening crossings through bar interiors.
    """

    def __init__(
        self,
        member,
        *,
        distance_tolerance=1.0e-9,
        time_tolerance=1.0e-9,
        parameter_tolerance=1.0e-3,
        parallel_sine_tolerance=1.0e-10,
        coefficient_tolerance=1.0e-10,
        pair_indices=None,
    ):
        self.member = _validate_member(member).copy()
        self.distance_tolerance = float(distance_tolerance)
        self.time_tolerance = float(time_tolerance)
        self.parameter_tolerance = float(parameter_tolerance)
        self.parallel_sine_tolerance = float(parallel_sine_tolerance)
        self.coefficient_tolerance = float(coefficient_tolerance)
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance < 0:
            raise ValueError("distance_tolerance must be finite and non-negative.")
        for name in (
            "time_tolerance",
            "parameter_tolerance",
            "parallel_sine_tolerance",
            "coefficient_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.time_tolerance >= 0.5 or self.parameter_tolerance >= 0.5:
            raise ValueError("time and parameter tolerances must be below 0.5.")

        if pair_indices is None:
            pairs = nonadjacent_member_pairs(self.member)
        else:
            pairs = np.asarray(pair_indices, dtype=int).reshape(-1, 2)
            if len(pairs) and (
                np.min(pairs) < 0 or np.max(pairs) >= len(self.member)
            ):
                raise ValueError("pair_indices contains an invalid member index.")
            for first, second in pairs:
                if first >= second:
                    raise ValueError("pair_indices must be ordered as first < second.")
                if set(self.member[first]) & set(self.member[second]):
                    raise ValueError("pair_indices must contain non-adjacent bars only.")
        self.pair_indices = pairs.copy()

    def _broad_phase_pairs(self, before, after):
        if not len(self.pair_indices):
            return self.pair_indices
        first_nodes = self.member[:, 0]
        second_nodes = self.member[:, 1]
        swept_points = np.stack(
            (
                before[first_nodes],
                before[second_nodes],
                after[first_nodes],
                after[second_nodes],
            ),
            axis=1,
        )
        lower = np.min(swept_points, axis=1) - self.distance_tolerance
        upper = np.max(swept_points, axis=1) + self.distance_tolerance
        first = self.pair_indices[:, 0]
        second = self.pair_indices[:, 1]
        overlaps = np.all(upper[first] >= lower[second], axis=1) & np.all(
            upper[second] >= lower[first], axis=1
        )
        return self.pair_indices[overlaps]

    def check(self, node_before, node_after):
        """Return collision diagnostics for one corrected linear transition."""

        before, after = _validate_nodes(
            node_before, node_after, self.member
        )
        candidates = self._broad_phase_pairs(before, after)
        narrow_checks = 0
        for first_member, second_member in candidates:
            narrow_checks += 1
            ia, ib = self.member[first_member]
            ja, jb = self.member[second_member]
            endpoints = (
                before[ia], after[ia],
                before[ib], after[ib],
                before[ja], after[ja],
                before[jb], after[jb],
            )
            coefficients = _coplanarity_coefficients(*endpoints)
            scale = _geometry_scale((
                before[ia], before[ib], before[ja], before[jb],
                after[ia], after[ib], after[ja], after[jb],
            ))
            coplanarity_threshold = self.coefficient_tolerance * scale ** 3
            coplanar_transition = bool(
                np.max(np.abs(coefficients)) <= coplanarity_threshold
            )
            if coplanar_transition:
                roots = _coplanar_event_times(
                    endpoints,
                    self.coefficient_tolerance,
                    self.time_tolerance,
                )
                boundaries = [0.0, *roots, 1.0]
                representative_times = [
                    0.5 * (left + right)
                    for left, right in zip(boundaries[:-1], boundaries[1:])
                    if right - left > self.time_tolerance
                ]
                # A proper intersection already present immediately after
                # t=0 is pre-existing contact, not a new pass-through.  Only
                # a false -> true interval transition is a collision event.
                if not representative_times:
                    times = []
                    continue
                first_time, *later_times = representative_times
                previous_is_intersecting = _intersection_at_time(
                    endpoints,
                    first_time,
                    self.distance_tolerance,
                    self.parameter_tolerance,
                    self.parallel_sine_tolerance,
                ) is not None
                for time in later_times:
                    intersection = _intersection_at_time(
                        endpoints,
                        time,
                        self.distance_tolerance,
                        self.parameter_tolerance,
                        self.parallel_sine_tolerance,
                    )
                    is_intersecting = intersection is not None
                    if is_intersecting and not previous_is_intersecting:
                        first_parameter, second_parameter, distance = intersection
                        event = CollisionEvent(
                            first_member=int(first_member),
                            second_member=int(second_member),
                            time=float(time),
                            first_parameter=float(first_parameter),
                            second_parameter=float(second_parameter),
                            distance=float(distance),
                            coplanar_transition=True,
                        )
                        return CollisionCheckResult(
                            collision=True,
                            event=event,
                            total_nonadjacent_pairs=len(self.pair_indices),
                            broad_phase_candidates=len(candidates),
                            narrow_phase_checks=narrow_checks,
                        )
                    previous_is_intersecting = is_intersecting
                times = []
            else:
                times = _sign_changing_roots(
                    coefficients,
                    self.coefficient_tolerance,
                    self.time_tolerance,
                )

            for time in sorted(set(times)):
                intersection = _intersection_at_time(
                    endpoints,
                    time,
                    self.distance_tolerance,
                    self.parameter_tolerance,
                    self.parallel_sine_tolerance,
                )
                if intersection is not None:
                    first_parameter, second_parameter, distance = intersection
                    event = CollisionEvent(
                        first_member=int(first_member),
                        second_member=int(second_member),
                        time=float(time),
                        first_parameter=float(first_parameter),
                        second_parameter=float(second_parameter),
                        distance=float(distance),
                        coplanar_transition=coplanar_transition,
                    )
                    return CollisionCheckResult(
                        collision=True,
                        event=event,
                        total_nonadjacent_pairs=len(self.pair_indices),
                        broad_phase_candidates=len(candidates),
                        narrow_phase_checks=narrow_checks,
                    )
        return CollisionCheckResult(
            collision=False,
            event=None,
            total_nonadjacent_pairs=len(self.pair_indices),
            broad_phase_candidates=len(candidates),
            narrow_phase_checks=narrow_checks,
        )

    def __call__(self, node_before, node_after):
        return bool(self.check(node_before, node_after).collision)


def make_legacy_signature_checker(member, **kwargs):
    """Return a cached callback compatible with ori_codex2's call signature.

    The callback accepts ``(node_before, node_after, member, bar_pairs,
    tolerance)`` and returns ``True`` for a collision, matching
    ``ori_codex2.has_proper_self_crossing``.  Its topology and tolerance are
    intentionally fixed when the callback is created; later arguments are
    validated only to prevent accidentally using it with another model.
    """

    configured_member = _validate_member(member)
    checker = ContinuousBarCollisionChecker(configured_member, **kwargs)

    def callback(node_before, node_after, runtime_member, bar_pairs, tolerance):
        del bar_pairs, tolerance
        runtime_member = _validate_member(runtime_member)
        if not np.array_equal(runtime_member, configured_member):
            raise ValueError("The cached collision checker received another topology.")
        return checker(node_before, node_after)

    callback.checker = checker
    return callback


def has_proper_self_crossing(
    node_before,
    node_after,
    member,
    bar_pairs,
    tolerance,
):
    """Drop-in, uncached ori_codex2-compatible collision function.

    Prefer :func:`make_legacy_signature_checker` in long training runs so the
    topology and broad-phase pair arrays are validated and cached only once.
    """

    checker = ContinuousBarCollisionChecker(
        member,
        distance_tolerance=max(float(tolerance), 0.0),
        pair_indices=bar_pairs,
    )
    return checker(node_before, node_after)


__all__ = [
    "COLLISION_BACKEND_NAME",
    "COLLISION_BACKEND_VERSION",
    "COLLISION_SEMANTICS",
    "CollisionCheckResult",
    "CollisionEvent",
    "ContinuousBarCollisionChecker",
    "has_proper_self_crossing",
    "make_legacy_signature_checker",
    "nonadjacent_member_pairs",
]
