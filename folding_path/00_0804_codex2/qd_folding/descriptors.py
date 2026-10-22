"""State and terminal descriptors for the five-bar model."""

from __future__ import annotations

import math

import numpy as np

from .mechanics import Model, cross2, gravity_potential, vector_to_node


def configuration_descriptor(
    state_vector: np.ndarray, model: Model
) -> np.ndarray:
    """Return a cheap, dimensionless descriptor of a configuration.

    The end nodes of this model are fixed, so its global translation and
    rotation are already removed by the boundary conditions.  The descriptor
    is therefore the flattened x-z coordinates, normalized by mean member
    length.
    """

    node = vector_to_node(state_vector, model)
    return node[:, [0, 2]].reshape(-1) / max(model.length_scale, 1.0e-12)


def descriptor_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Root-mean-square descriptor distance."""

    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def descriptor_bin(
    descriptor: np.ndarray, width: float
) -> tuple[int, ...]:
    """Map a state to a fixed grid cell.

    Rounding rather than truncation makes zero the center of a cell.
    """

    values = np.floor(np.asarray(descriptor) / width + 0.5).astype(np.int64)
    return tuple(int(value) for value in values)


def descriptor_bin_center(key: tuple[int, ...], width: float) -> np.ndarray:
    return width * np.asarray(key, dtype=float)


def fold_shape_signature(
    state_vector: np.ndarray, model: Model, tolerance: float = 1.0e-7
) -> str:
    """Discrete left/right orientation of every member at a flat terminal."""

    node = vector_to_node(state_vector, model)
    labels: list[str] = []
    for first, second in model.member:
        dx = float(node[second, 0] - node[first, 0])
        if dx > tolerance:
            labels.append("+")
        elif dx < -tolerance:
            labels.append("-")
        else:
            labels.append("0")
    return "".join(labels)


def overlapping_member_pairs(
    state_vector: np.ndarray,
    model: Model,
    tolerance: float,
) -> list[tuple[int, int, np.ndarray]]:
    """Return nonadjacent member pairs that overlap in the final projection."""

    node = vector_to_node(state_vector, model)
    points = node[:, [0, 2]]
    pairs: list[tuple[int, int, np.ndarray]] = []
    for i, (i0, i1) in enumerate(model.member):
        p0, p1 = points[i0], points[i1]
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm <= tolerance:
            continue
        direction = direction / norm
        if direction[0] < 0.0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        for j in range(i + 1, model.num_member):
            j0, j1 = model.member[j]
            if len({int(i0), int(i1), int(j0), int(j1)}) < 4:
                continue
            q0, q1 = points[j0], points[j1]
            q_direction = q1 - q0
            if abs(cross2(p1 - p0, q_direction)) > tolerance * norm:
                continue
            transverse = max(
                abs(float(np.dot(q0 - p0, normal))),
                abs(float(np.dot(q1 - p0, normal))),
            )
            if transverse > tolerance:
                continue
            p_interval = sorted(
                (float(np.dot(p0, direction)), float(np.dot(p1, direction)))
            )
            q_interval = sorted(
                (float(np.dot(q0, direction)), float(np.dot(q1, direction)))
            )
            overlap = min(p_interval[1], q_interval[1]) - max(
                p_interval[0], q_interval[0]
            )
            if overlap > tolerance:
                pairs.append((i, j, normal))
    return pairs


def last_nonflat_index(
    path: np.ndarray, model: Model, layer_threshold: float
) -> int:
    """Index used for a common layer-order observation frame."""

    if path.shape[1] <= 1:
        return 0
    for step in range(path.shape[1] - 2, -1, -1):
        node = vector_to_node(path[:, step], model)
        if gravity_potential(node) > layer_threshold:
            return step
    return 0


def layer_order_signature(
    path: np.ndarray,
    model: Model,
    layer_threshold: float,
    overlap_tolerance: float,
    separation_tolerance: float,
) -> tuple[str, int]:
    """Layer-order labels at the last common nonflat configuration."""

    observation_step = last_nonflat_index(path, model, layer_threshold)
    observation = vector_to_node(path[:, observation_step], model)[:, [0, 2]]
    labels: list[str] = []
    for i, j, normal in overlapping_member_pairs(
        path[:, -1], model, overlap_tolerance
    ):
        i0, i1 = model.member[i]
        j0, j1 = model.member[j]
        midpoint_i = 0.5 * (observation[i0] + observation[i1])
        midpoint_j = 0.5 * (observation[j0] + observation[j1])
        separation = float(np.dot(midpoint_i - midpoint_j, normal))
        if separation > separation_tolerance:
            labels.append(f"{i}>{j}")
        elif separation < -separation_tolerance:
            labels.append(f"{j}>{i}")
        else:
            labels.append(f"{i}?{j}")
    return ("|".join(sorted(labels)) if labels else "none", observation_step)


def terminal_cell(
    trajectory: dict, model: Model, config
) -> tuple[tuple[str, str], dict]:
    """Return the categorical terminal cell and diagnostic metadata."""

    path = trajectory["path"]
    shape = fold_shape_signature(path[:, -1], model)
    layer, layer_step = layer_order_signature(
        path,
        model,
        config.layer_threshold,
        config.overlap_tolerance,
        config.layer_separation_tolerance,
    )
    return (shape, layer), {
        "shape": shape,
        "layer_order": layer,
        "layer_observation_step": layer_step,
        "terminal_descriptor": configuration_descriptor(path[:, -1], model),
    }


def path_curvature(descriptors: np.ndarray) -> np.ndarray:
    """Turning angle of a descriptor-space path at every interior state."""

    count = len(descriptors)
    result = np.zeros(count, dtype=float)
    for index in range(1, count - 1):
        before = descriptors[index] - descriptors[index - 1]
        after = descriptors[index + 1] - descriptors[index]
        norm = float(np.linalg.norm(before) * np.linalg.norm(after))
        if norm <= 1.0e-14:
            continue
        cosine = float(np.dot(before, after) / norm)
        result[index] = math.acos(float(np.clip(cosine, -1.0, 1.0)))
    return result
