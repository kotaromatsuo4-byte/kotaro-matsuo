"""Geometry and constraint mechanics shared by all five folding tasks.

Nodes have shape ``(n, 3)`` and each member stores two zero-based node
indices.  The numerical order and tolerances follow the Codex6 authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COLLISION_BACKEND_NAME = "continuous_proper_crossing_3d"
COLLISION_BACKEND_VERSION = "1.1.0"
COLLISION_SEMANTICS = (
    "new entry into a nonadjacent zero-thickness bar-interior proper crossing "
    "in the open linear-motion interval; persistent/static intersection, "
    "endpoint contact, and parallel/collinear overlap are allowed"
)


def index_free(num_node, fix):
    """Flattened coordinate indices not listed in ``fix``."""
    fixed = np.asarray(fix, dtype=int).reshape(-1, 2)
    return np.delete(np.arange(int(num_node) * 3), 3 * fixed[:, 0] + fixed[:, 1])


def incompatibility(node, member, lengths):
    """Squared bar-length residuals, ``(||xi-xj||^2-L^2)/2``."""
    coordinates = np.asarray(node, dtype=float)
    members = np.asarray(member, dtype=int).reshape(-1, 2)
    differences = coordinates[members[:, 0]] - coordinates[members[:, 1]]
    return 0.5 * (
        np.sum(differences**2, axis=1) - np.asarray(lengths, dtype=float) ** 2
    )


def dedx(node, member, free_index):
    """Constraint Jacobian restricted to the free coordinates."""
    coordinates = np.asarray(node, dtype=float)
    members = np.asarray(member, dtype=int).reshape(-1, 2)
    jacobian = np.zeros((len(members), coordinates.size), dtype=float)
    differences = coordinates[members[:, 0]] - coordinates[members[:, 1]]
    for row, (node0, node1) in enumerate(members):
        jacobian[row, 3 * node0 : 3 * (node0 + 1)] = differences[row]
        jacobian[row, 3 * node1 : 3 * (node1 + 1)] = -differences[row]
    return jacobian[:, np.asarray(free_index, dtype=int)]


def mode(free_index, dedx_matrix, num_node, tol=1.0e-8):
    """Right-nullspace mechanism modes as full-coordinate columns."""
    jacobian = np.asarray(dedx_matrix, dtype=float)
    _, singular_values, right = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > float(tol)))
    modes = np.zeros((int(num_node) * 3, jacobian.shape[1] - rank), dtype=float)
    modes[np.asarray(free_index, dtype=int), :] = right[rank:].T
    return modes


def canonical_modes(node, member, free_index, tol=1.0e-8):
    """SVD modes with the deterministic column signs used by Codex6."""
    modes = mode(
        free_index, dedx(node, member, free_index), len(np.asarray(node)), tol=tol
    )
    for column in range(modes.shape[1]):
        pivot = int(np.argmax(np.abs(modes[:, column])))
        if modes[pivot, column] < 0.0:
            modes[:, column] *= -1.0
    return modes


def project_to_constraints(
    node,
    member,
    lengths,
    free_index,
    tolerance=1.0e-10,
    max_iterations=30,
):
    """Newton/pseudoinverse projection used by every Codex6 task."""
    corrected = np.asarray(node, dtype=float).copy()
    for _ in range(int(max_iterations)):
        residual = incompatibility(corrected, member, lengths)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= float(tolerance):
            break
        jacobian = dedx(corrected, member, free_index)
        delta_free = -np.linalg.pinv(jacobian) @ residual
        delta = np.zeros(corrected.size, dtype=float)
        delta[np.asarray(free_index, dtype=int)] = delta_free
        delta = delta.reshape(corrected.shape)
        step_size = 1.0
        accepted = False
        for _ in range(20):
            candidate = corrected + step_size * delta
            if np.linalg.norm(incompatibility(candidate, member, lengths)) < residual_norm:
                corrected = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
    return corrected


def current_strain(node, member, lengths):
    """Sum of squared bar-length errors."""
    coordinates = np.asarray(node, dtype=float)
    members = np.asarray(member, dtype=int).reshape(-1, 2)
    current_lengths = np.linalg.norm(
        coordinates[members[:, 0]] - coordinates[members[:, 1]], axis=1
    )
    return float(np.sum((current_lengths - np.asarray(lengths, dtype=float)) ** 2))


def project_candidate(
    node,
    direction,
    requested_width,
    member,
    lengths,
    free_index,
    *,
    max_step_width,
    min_step_width=1.0e-4,
    projection_shrink=0.5,
    strain_tolerance=1.0e-8,
    max_correction_ratio=1.0,
    max_actual_step_ratio=1.25,
):
    """Project a proposal, shrinking width until its correction is local.

    Returns ``(node, width, failed, attempts, strain, correction_norm,
    actual_displacement)``.
    """
    coordinates = np.asarray(node, dtype=float)
    direction = np.asarray(direction, dtype=float).reshape(coordinates.shape)
    direction_norm = float(np.linalg.norm(direction))
    width = float(np.clip(requested_width, -max_step_width, max_step_width))
    if direction_norm <= 1.0e-12 or abs(width) < min_step_width:
        return coordinates.copy(), 0.0, False, 0, 0.0, 0.0, 0.0

    direction = direction / direction_norm
    attempts = 0
    last_strain = np.inf
    last_correction_norm = np.inf
    last_actual_displacement = np.inf
    while abs(width) >= min_step_width:
        attempts += 1
        predicted = coordinates + width * direction
        corrected = project_to_constraints(predicted, member, lengths, free_index)
        if np.all(np.isfinite(corrected)):
            last_strain = current_strain(corrected, member, lengths)
            last_correction_norm = float(np.linalg.norm(corrected - predicted))
            last_actual_displacement = float(np.linalg.norm(corrected - coordinates))
            if (
                last_strain <= strain_tolerance
                and last_correction_norm <= max_correction_ratio * abs(width)
                and last_actual_displacement <= max_actual_step_ratio * abs(width)
            ):
                return (
                    corrected, float(width), False, attempts,
                    float(last_strain), float(last_correction_norm),
                    float(last_actual_displacement),
                )
        width *= projection_shrink
    return (
        coordinates.copy(), 0.0, True, attempts, float(last_strain),
        float(last_correction_norm), float(last_actual_displacement),
    )


def compute_flatness_potential(node, subspace_dimension):
    """Squared distance to the best affine line (1) or plane (2)."""
    coordinates = np.asarray(node, dtype=float)
    dimension = int(subspace_dimension)
    if dimension not in (1, 2):
        raise ValueError("subspace_dimension must be 1 (line) or 2 (plane)")
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return float(np.sum(singular_values[dimension:] ** 2))


def flatness_metrics(node, subspace_dimension, initial_potential, coordinate_scale):
    """Return ``(potential, initial-normalised ratio, normalised RMS)``."""
    potential = compute_flatness_potential(node, subspace_dimension)
    ratio = potential / float(initial_potential)
    rms = (
        np.sqrt(max(potential, 0.0) / len(np.asarray(node)))
        / max(float(coordinate_scale), 1.0e-12)
    )
    return float(potential), float(ratio), float(rms)


def projected_spread(node, subspace_dimension, coordinate_scale=1.0):
    """Diameter after projection into the best-fit affine subspace."""
    coordinates = np.asarray(node, dtype=float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    dimension = min(int(subspace_dimension), right.shape[0])
    projected = centered @ right[:dimension].T
    differences = projected[:, np.newaxis, :] - projected[np.newaxis, :, :]
    diameter = np.max(np.linalg.norm(differences, axis=2))
    return float(diameter / max(float(coordinate_scale), 1.0e-12))


def target_deviation_energy(spread, target_spread):
    deviation = float(spread) - float(target_spread)
    if not np.isfinite(deviation):
        return 1.0e12
    deviation = float(np.clip(deviation, -1.0e6, 1.0e6))
    return float(deviation * deviation)


def subspace_projector(basis, k):
    selected = np.asarray(basis, dtype=float)[:, : int(k)]
    return selected @ selected.T


def spectral_subspace_geometry(
    node, k, reference_projector=None, relative_epsilon=1.0e-12
):
    """Codex6 spectral plane/line-swap potential geometry."""
    coordinates = np.asarray(node, dtype=float)
    k = int(k)
    epsilon = float(relative_epsilon)
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if k < 1 or k >= len(singular_values):
        raise ValueError("k must leave sigma[k] available")
    projector = subspace_projector(left, k)
    a = float(singular_values[k - 1])
    b = float(singular_values[k])
    singular_sum = float(a + b)
    gap = float(max(a - b, 0.0))
    threshold = float(epsilon * max(singular_sum, float(singular_values[0])))
    degenerate = bool(singular_sum == 0.0 or gap <= threshold)
    rho = 0.0 if degenerate else float(gap / (singular_sum * (1.0 + epsilon)))
    if reference_projector is None:
        delta = 0.0
    else:
        reference = np.asarray(reference_projector, dtype=float)
        if reference.shape != projector.shape:
            raise ValueError("reference projector shape does not match")
        delta = float(np.clip(
            np.linalg.norm(projector - reference, ord="fro") / np.sqrt(2.0 * k),
            0.0, 1.0,
        ))
    return {
        "projector": np.array(projector, copy=True),
        "singular_values": np.array(singular_values, copy=True),
        "sigma_k": a,
        "sigma_k_plus_1": b,
        "singular_sum": singular_sum,
        "spectral_gap": gap,
        "degeneracy_threshold": threshold,
        "gap_degenerate": int(degenerate),
        "rho": rho,
        "delta": delta,
        "potential": float(rho * delta),
    }


def nonadjacent_bar_pairs(member):
    """Unordered member-index pairs that share no node."""
    members = np.asarray(member, dtype=int).reshape(-1, 2)
    pairs = []
    for first in range(len(members)):
        first_nodes = set(map(int, members[first]))
        for second in range(first + 1, len(members)):
            if first_nodes.isdisjoint(map(int, members[second])):
                pairs.append((first, second))
    return pairs


def _validate_member(member):
    array = np.asarray(member)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("member must have shape (number_of_members, 2)")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
            raise ValueError("member indices must be finite integers")
    array = array.astype(int, copy=False)
    if np.any(array < 0):
        raise ValueError("member indices cannot be negative")
    return array


def _validate_nodes(node_before, node_after, member):
    before = np.asarray(node_before, dtype=float)
    after = np.asarray(node_after, dtype=float)
    if before.ndim != 2 or before.shape[1] != 3:
        raise ValueError("node_before must have shape (number_of_nodes, 3)")
    if after.shape != before.shape:
        raise ValueError("node_after must have the same shape as node_before")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("node coordinates must be finite")
    if len(member) and int(np.max(member)) >= len(before):
        raise ValueError("member contains a node index outside the node array")
    return before, after


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation_polynomial(a0, a1, b0, b1, c0, c1):
    ab0 = b0 - a0
    ac0 = c0 - a0
    delta_ab = (b1 - b0) - (a1 - a0)
    delta_ac = (c1 - c0) - (a1 - a0)
    return np.array([
        _cross_2d(delta_ab, delta_ac),
        _cross_2d(delta_ab, ac0) + _cross_2d(ab0, delta_ac),
        _cross_2d(ab0, ac0),
    ])


def _roots_open_descending(coefficients, tolerance):
    first = 0
    while first < len(coefficients) - 1 and abs(coefficients[first]) <= tolerance:
        first += 1
    polynomial = coefficients[first:]
    if len(polynomial) <= 1:
        return []
    return [
        float(root.real) for root in np.roots(polynomial)
        if abs(root.imag) <= tolerance and tolerance < root.real < 1.0 - tolerance
    ]


def _proper_segment_intersection_2d(a, b, c, d, tolerance):
    values = (
        _cross_2d(b - a, c - a), _cross_2d(b - a, d - a),
        _cross_2d(d - c, a - c), _cross_2d(d - c, b - c),
    )
    def opposite(first, second):
        return (first > tolerance and second < -tolerance) or (
            first < -tolerance and second > tolerance
        )
    return opposite(values[0], values[1]) and opposite(values[2], values[3])


def has_proper_self_crossing_planar(
    node_before, node_after, member, bar_pairs=None, tolerance=1.0e-10
):
    """Detect a proper crossing during planar x-z linear motion."""
    before3 = np.asarray(node_before, dtype=float)
    after3 = np.asarray(node_after, dtype=float)
    if np.ptp(np.concatenate([before3[:, 1], after3[:, 1]])) > tolerance:
        raise ValueError("planar crossing detection requires constant y")
    before = before3[:, [0, 2]]
    after = after3[:, [0, 2]]
    members = np.asarray(member, dtype=int).reshape(-1, 2)
    pairs = nonadjacent_bar_pairs(members) if bar_pairs is None else bar_pairs
    for first, second in pairs:
        a_index, b_index = members[first]
        c_index, d_index = members[second]
        indices = (a_index, b_index, c_index, d_index)
        polynomials = (
            _orientation_polynomial(before[a_index], after[a_index], before[b_index], after[b_index], before[c_index], after[c_index]),
            _orientation_polynomial(before[a_index], after[a_index], before[b_index], after[b_index], before[d_index], after[d_index]),
            _orientation_polynomial(before[c_index], after[c_index], before[d_index], after[d_index], before[a_index], after[a_index]),
            _orientation_polynomial(before[c_index], after[c_index], before[d_index], after[d_index], before[b_index], after[b_index]),
        )
        breakpoints = [0.0, 1.0]
        for polynomial in polynomials:
            breakpoints.extend(_roots_open_descending(polynomial, tolerance))
        ordered = sorted(breakpoints)
        unique = [ordered[0]]
        for value in ordered[1:]:
            if value - unique[-1] > tolerance:
                unique.append(value)
        for left, right in zip(unique[:-1], unique[1:]):
            if right - left <= tolerance:
                continue
            fraction = 0.5 * (left + right)
            points = [
                before[index] + fraction * (after[index] - before[index])
                for index in indices
            ]
            if _proper_segment_intersection_2d(*points, tolerance=tolerance):
                return True
    return False


@dataclass(frozen=True)
class CollisionEvent:
    first_member: int
    second_member: int
    time: float
    first_parameter: float
    second_parameter: float
    distance: float
    coplanar_transition: bool


@dataclass(frozen=True)
class CollisionCheckResult:
    collision: bool
    event: CollisionEvent | None
    total_nonadjacent_pairs: int
    broad_phase_candidates: int
    narrow_phase_checks: int


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
    return np.asarray([
        _triple(p0, q0, r0),
        _triple(p1, q0, r0) + _triple(p0, q1, r0) + _triple(p0, q0, r1),
        _triple(p1, q1, r0) + _triple(p1, q0, r1) + _triple(p0, q1, r1),
        _triple(p1, q1, r1),
    ], dtype=float)


def _orientation_coefficients_2d(a0, a1, b0, b1, c0, c1):
    """Ascending coefficients of orient(A(t), B(t), C(t))."""
    u0 = b0 - a0
    u1 = (b1 - b0) - (a1 - a0)
    v0 = c0 - a0
    v1 = (c1 - c0) - (a1 - a0)
    return np.asarray([
        _cross_2d(u0, v0),
        _cross_2d(u1, v0) + _cross_2d(u0, v1),
        _cross_2d(u1, v1),
    ], dtype=float)


def _real_roots_open(coefficients, coefficient_tolerance, time_tolerance):
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
    value = 0.0
    for coefficient in np.asarray(coefficients, dtype=float)[::-1]:
        value = value * float(time) + float(coefficient)
    return float(value)


def _sign_changing_roots(coefficients, coefficient_tolerance, time_tolerance):
    roots = _real_roots_open(coefficients, coefficient_tolerance, time_tolerance)
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
    """Closest-point data only for a transversal interior intersection."""
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    first_length = float(np.linalg.norm(first_direction))
    second_length = float(np.linalg.norm(second_direction))
    if first_length <= distance_tolerance or second_length <= distance_tolerance:
        return None
    cross_norm = float(np.linalg.norm(np.cross(first_direction, second_direction)))
    if cross_norm <= parallel_sine_tolerance * first_length * second_length:
        return None

    offset = first_start - second_start
    aa = float(np.dot(first_direction, first_direction))
    bb = float(np.dot(first_direction, second_direction))
    cc = float(np.dot(second_direction, second_direction))
    dd = float(np.dot(first_direction, offset))
    ee = float(np.dot(second_direction, offset))
    denominator = aa * cc - bb * bb
    if denominator <= (parallel_sine_tolerance * first_length * second_length) ** 2:
        return None
    first_parameter = (bb * ee - cc * dd) / denominator
    second_parameter = (aa * ee - bb * dd) / denominator
    if not (
        parameter_tolerance < first_parameter < 1.0 - parameter_tolerance
        and parameter_tolerance < second_parameter < 1.0 - parameter_tolerance
    ):
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
    a0, a1, b0, b1, c0, c1, d0, d1 = endpoints
    triples = (
        (a0, a1, b0, b1, c0, c1),
        (a0, a1, b0, b1, d0, d1),
        (c0, c1, d0, d1, a0, a1),
        (c0, c1, d0, d1, b0, b1),
    )
    roots = []
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
    """Cached Codex6 continuous 3-D crossing checker for one topology."""

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
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance < 0.0:
            raise ValueError("distance_tolerance must be finite and non-negative")
        for name in (
            "time_tolerance", "parameter_tolerance",
            "parallel_sine_tolerance", "coefficient_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.time_tolerance >= 0.5 or self.parameter_tolerance >= 0.5:
            raise ValueError("time and parameter tolerances must be below 0.5")
        pairs = nonadjacent_bar_pairs(self.member) if pair_indices is None else pair_indices
        self.pair_indices = np.asarray(pairs, dtype=int).reshape(-1, 2).copy()

    def _broad_phase_pairs(self, before, after):
        if not len(self.pair_indices):
            return self.pair_indices
        first_nodes = self.member[:, 0]
        second_nodes = self.member[:, 1]
        swept_points = np.stack((
            before[first_nodes], before[second_nodes],
            after[first_nodes], after[second_nodes],
        ), axis=1)
        lower = np.min(swept_points, axis=1) - self.distance_tolerance
        upper = np.max(swept_points, axis=1) + self.distance_tolerance
        first = self.pair_indices[:, 0]
        second = self.pair_indices[:, 1]
        overlaps = np.all(upper[first] >= lower[second], axis=1) & np.all(
            upper[second] >= lower[first], axis=1
        )
        return self.pair_indices[overlaps]

    def check(self, node_before, node_after):
        before, after = _validate_nodes(node_before, node_after, self.member)
        candidates = self._broad_phase_pairs(before, after)
        narrow_checks = 0
        for first_member, second_member in candidates:
            narrow_checks += 1
            ia, ib = self.member[first_member]
            ja, jb = self.member[second_member]
            endpoints = (
                before[ia], after[ia], before[ib], after[ib],
                before[ja], after[ja], before[jb], after[jb],
            )
            coefficients = _coplanarity_coefficients(*endpoints)
            scale = _geometry_scale((
                before[ia], before[ib], before[ja], before[jb],
                after[ia], after[ib], after[ja], after[jb],
            ))
            coplanar = bool(
                np.max(np.abs(coefficients))
                <= self.coefficient_tolerance * scale**3
            )
            if coplanar:
                roots = _coplanar_event_times(
                    endpoints, self.coefficient_tolerance, self.time_tolerance
                )
                boundaries = [0.0, *roots, 1.0]
                times = [
                    0.5 * (left + right)
                    for left, right in zip(boundaries[:-1], boundaries[1:])
                    if right - left > self.time_tolerance
                ]
                if not times:
                    continue
                first_time, *later_times = times
                previous = _intersection_at_time(
                    endpoints, first_time, self.distance_tolerance,
                    self.parameter_tolerance, self.parallel_sine_tolerance,
                ) is not None
                for time in later_times:
                    intersection = _intersection_at_time(
                        endpoints, time, self.distance_tolerance,
                        self.parameter_tolerance, self.parallel_sine_tolerance,
                    )
                    intersecting = intersection is not None
                    if intersecting and not previous:
                        return self._collision_result(
                            first_member, second_member, time, intersection,
                            True, candidates, narrow_checks,
                        )
                    previous = intersecting
                event_times = []
            else:
                event_times = _sign_changing_roots(
                    coefficients, self.coefficient_tolerance, self.time_tolerance
                )
            for time in sorted(set(event_times)):
                intersection = _intersection_at_time(
                    endpoints, time, self.distance_tolerance,
                    self.parameter_tolerance, self.parallel_sine_tolerance,
                )
                if intersection is not None:
                    return self._collision_result(
                        first_member, second_member, time, intersection,
                        coplanar, candidates, narrow_checks,
                    )
        return CollisionCheckResult(
            False, None, len(self.pair_indices), len(candidates), narrow_checks
        )

    def _collision_result(
        self, first_member, second_member, time, intersection,
        coplanar, candidates, narrow_checks,
    ):
        first_parameter, second_parameter, distance = intersection
        event = CollisionEvent(
            int(first_member), int(second_member), float(time),
            float(first_parameter), float(second_parameter), float(distance),
            bool(coplanar),
        )
        return CollisionCheckResult(
            True, event, len(self.pair_indices), len(candidates), int(narrow_checks)
        )

    def __call__(self, node_before, node_after):
        return bool(self.check(node_before, node_after).collision)


def make_legacy_signature_checker(member, **kwargs):
    """Cached callback with Codex6's five-argument crossing signature."""
    configured_member = _validate_member(member)
    checker = ContinuousBarCollisionChecker(configured_member, **kwargs)

    def callback(node_before, node_after, runtime_member, bar_pairs, tolerance):
        del bar_pairs, tolerance
        runtime_member = _validate_member(runtime_member)
        if not np.array_equal(runtime_member, configured_member):
            raise ValueError("cached collision checker received another topology")
        return checker(node_before, node_after)

    callback.checker = checker
    return callback


def has_proper_self_crossing(
    node_before,
    node_after,
    member,
    bar_pairs=None,
    tolerance=1.0e-10,
    *,
    spatial=False,
):
    """Unified planar/spatial continuous proper-crossing predicate."""
    if not spatial:
        return has_proper_self_crossing_planar(
            node_before, node_after, member, bar_pairs, tolerance
        )
    return ContinuousBarCollisionChecker(
        member,
        distance_tolerance=max(float(tolerance), 0.0),
        pair_indices=bar_pairs,
    )(node_before, node_after)


def has_self_crossing(
    node_before,
    node_after,
    member,
    spatial,
    tolerance=1.0e-10,
    bar_pairs=None,
):
    """Short environment-facing alias for the unified checker."""
    return has_proper_self_crossing(
        node_before, node_after, member, bar_pairs, tolerance,
        spatial=bool(spatial),
    )


# Legacy names used by the original skeleton scripts.
def calc_dedx(node, num_node, member, num_member, free_index):
    del num_node, num_member
    return dedx(node, member, free_index)


calc_mode = mode


__all__ = [
    "COLLISION_BACKEND_NAME", "COLLISION_BACKEND_VERSION", "COLLISION_SEMANTICS",
    "CollisionCheckResult", "CollisionEvent", "ContinuousBarCollisionChecker",
    "calc_dedx", "calc_mode", "canonical_modes", "compute_flatness_potential",
    "current_strain", "dedx", "flatness_metrics", "has_proper_self_crossing",
    "has_proper_self_crossing_planar", "has_self_crossing", "incompatibility",
    "index_free", "make_legacy_signature_checker", "mode",
    "nonadjacent_bar_pairs", "project_candidate",
    "project_to_constraints", "projected_spread", "spectral_subspace_geometry",
    "subspace_projector", "target_deviation_energy",
]
