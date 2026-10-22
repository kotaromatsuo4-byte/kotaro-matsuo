"""Geometry helpers for a two-perturbation angle macro.

The functions use physical nodal displacement vectors rather than SVD mode
coefficients.  Consequently, sign flips or orthogonal rotations of a null
space basis do not change the transported directions.

For a first displacement ``v1`` executed from ``q0`` to ``q1``, define

``p = normalize(P_T(q1) v1)``

and construct an explicit physical direction ``u`` by projecting a physical
reference ``w`` into the new tangent space and applying Gram--Schmidt against
``p``.  The second displacement is

``v2(theta) = rho * (cos(theta) p + sin(theta) u)``.

For mechanisms with more than two infinitesimal modes, ``u`` remains an
explicit physical direction; no arbitrary pair of SVD columns is assumed.
"""

from __future__ import annotations

import numpy as np

from mechanics import calc_dedx, calc_mode


def tangent_basis(node, model, mode_tolerance=None):
    """Return a full-coordinate null-space basis at ``node``."""
    jacobian = calc_dedx(
        np.asarray(node, dtype=float),
        model["num_node"],
        model["member"],
        model["num_member"],
        model["index_free"],
    )
    return calc_mode(
        model["index_free"],
        jacobian,
        model["num_node"],
        tol=mode_tolerance,
    )


def project_to_tangent(node, physical_vector, model, mode_tolerance=None):
    """Project a physical nodal vector into the current tangent space."""
    basis = tangent_basis(node, model, mode_tolerance)
    vector = np.asarray(physical_vector, dtype=float).reshape(-1)
    if basis.shape[1] == 0:
        return np.zeros_like(vector), basis
    projection = basis @ (np.linalg.pinv(basis, rcond=1.0e-12) @ vector)
    return projection, basis


def _normalized(vector, norm=1.0):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1.0e-12:
        return None
    return float(norm) * vector / magnitude


def sample_tangent_displacement(
    node,
    model,
    displacement_norm,
    rng,
    mode_tolerance=None,
):
    """Sample an isotropic coefficient direction and return its physical vector.

    The Gaussian coefficient distribution is rotationally invariant.  Thus
    the resulting physical sample is invariant in distribution to an
    orthogonal rotation or sign change of the SVD null-space basis.
    """
    basis = tangent_basis(node, model, mode_tolerance)
    if basis.shape[1] == 0:
        return None
    coefficient = rng.normal(size=basis.shape[1])
    coefficient /= max(float(np.linalg.norm(coefficient)), 1.0e-12)
    vector = basis @ coefficient
    normalized = _normalized(vector, displacement_norm)
    if normalized is None:
        return None
    return normalized.reshape(model["num_node"], 3)


def sample_orthogonal_reference(
    node,
    first_displacement,
    model,
    reference_norm,
    rng,
    mode_tolerance=None,
    maximum_attempts=50,
):
    """Sample a physical tangent reference orthogonal to ``first_displacement``.

    A one-dimensional tangent has no independent orthogonal reference and
    returns ``None``.  Higher-dimensional tangents return a full nodal vector.
    """
    basis = tangent_basis(node, model, mode_tolerance)
    if basis.shape[1] < 2:
        return None
    first = _normalized(first_displacement, 1.0)
    if first is None:
        return None
    for _ in range(int(maximum_attempts)):
        coefficient = rng.normal(size=basis.shape[1])
        trial = basis @ coefficient
        trial -= np.dot(trial, first) * first
        normalized = _normalized(trial, reference_norm)
        if normalized is not None:
            return normalized.reshape(model["num_node"], 3)
    return None


def transported_frame(
    node_after_first,
    first_displacement,
    physical_reference,
    model,
    displacement_norm,
    minimum_projection_ratio=0.10,
    rng=None,
    mode_tolerance=None,
):
    """Construct transported unit directions ``p`` and ``u`` at ``q1``.

    ``u`` is obtained from the projected physical reference by
    Gram--Schmidt.  If that reference becomes almost parallel to ``p`` and
    the tangent dimension is at least two, an optional random physical
    fallback is generated.  The fallback is reported explicitly.

    The returned ``p`` and ``u`` are unit vectors.  ``p_displacement`` and
    ``u_displacement`` have the requested physical norm and are convenient GP
    features.
    """
    node = np.asarray(node_after_first, dtype=float)
    basis = tangent_basis(node, model, mode_tolerance)
    if basis.shape[1] == 0:
        return None

    first = np.asarray(first_displacement, dtype=float).reshape(-1)
    first_norm = float(np.linalg.norm(first))
    projected_first = basis @ (np.linalg.pinv(basis, rcond=1.0e-12) @ first)
    projected_first_norm = float(np.linalg.norm(projected_first))
    projection_ratio = projected_first_norm / max(first_norm, 1.0e-12)
    if (
        projected_first_norm <= 1.0e-12
        or projection_ratio < float(minimum_projection_ratio)
    ):
        return None
    p = projected_first / projected_first_norm

    reference_projection_ratio = 0.0
    reference_fallback = False
    u = None
    if physical_reference is not None:
        reference = np.asarray(physical_reference, dtype=float).reshape(-1)
        reference_norm = float(np.linalg.norm(reference))
        projected_reference = basis @ (
            np.linalg.pinv(basis, rcond=1.0e-12) @ reference
        )
        reference_projection_ratio = float(
            np.linalg.norm(projected_reference) / max(reference_norm, 1.0e-12)
        )
        orthogonal = projected_reference - np.dot(projected_reference, p) * p
        u = _normalized(orthogonal, 1.0)

    if u is None and basis.shape[1] >= 2 and rng is not None:
        for _ in range(50):
            trial = basis @ rng.normal(size=basis.shape[1])
            trial -= np.dot(trial, p) * p
            u = _normalized(trial, 1.0)
            if u is not None:
                reference_fallback = True
                break

    return {
        "p": p.reshape(model["num_node"], 3),
        "u": None if u is None else u.reshape(model["num_node"], 3),
        "p_displacement": (
            float(displacement_norm) * p
        ).reshape(model["num_node"], 3),
        "u_displacement": (
            None
            if u is None
            else (float(displacement_norm) * u).reshape(model["num_node"], 3)
        ),
        "projection_ratio": float(projection_ratio),
        "reference_projection_ratio": float(reference_projection_ratio),
        "reference_fallback": bool(reference_fallback),
        "tangent_dimension": int(basis.shape[1]),
    }


def second_displacement(frame, theta, displacement_norm):
    """Evaluate ``rho (cos(theta) p + sin(theta) u)``.

    In a one-dimensional tangent, only angles with zero sine are valid.
    """
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    p = np.asarray(frame["p"], dtype=float).reshape(-1)
    if frame["u"] is None:
        if abs(sine) > 1.0e-10:
            return None
        direction = np.sign(cosine) * p
    else:
        u = np.asarray(frame["u"], dtype=float).reshape(-1)
        direction = cosine * p + sine * u
    normalized = _normalized(direction, displacement_norm)
    if normalized is None:
        return None
    return normalized.reshape(frame["p"].shape)


def angle_grid(num_angles, tangent_dimension):
    """Return a balanced deterministic angle grid that always contains zero."""
    if int(tangent_dimension) <= 1:
        return np.asarray([0.0, np.pi], dtype=float)
    count = max(int(num_angles), 2)
    return np.linspace(-np.pi, np.pi, count, endpoint=False, dtype=float)


def circular_distance(angle1, angle2):
    """Smallest absolute distance between two angles."""
    difference = float(angle1) - float(angle2)
    return abs((difference + np.pi) % (2.0 * np.pi) - np.pi)


def macro_gp_input(
    node_after_first,
    frame,
    candidate_second_displacement,
    theta,
    index_free,
    reference_point,
    length_scale,
):
    """Return ``[q1, transported v1, u, v2, cos(theta), sin(theta), ratios]``.

    All physical coordinate blocks are restricted to free coordinates and
    normalized by the model length.  A zero ``u`` block plus an availability
    mask keeps the dimension fixed for one-degree-of-freedom mechanisms.
    """
    node = np.asarray(node_after_first, dtype=float).reshape(-1, 3).copy()
    node[:, 2] -= float(reference_point)
    free = np.asarray(index_free, dtype=int).reshape(-1)
    scale = float(length_scale)

    q_part = node.reshape(-1)[free] / scale
    p_part = np.asarray(frame["p_displacement"]).reshape(-1)[free] / scale
    if frame["u_displacement"] is None:
        u_part = np.zeros(len(free), dtype=float)
        u_available = 0.0
    else:
        u_part = (
            np.asarray(frame["u_displacement"]).reshape(-1)[free] / scale
        )
        u_available = 1.0
    v2_part = (
        np.asarray(candidate_second_displacement).reshape(-1)[free] / scale
    )
    scalar_part = np.asarray(
        [
            np.cos(theta),
            np.sin(theta),
            frame["projection_ratio"],
            frame["reference_projection_ratio"],
            u_available,
        ],
        dtype=float,
    )
    return np.concatenate((q_part, p_part, u_part, v2_part, scalar_part))


def make_angle_candidates(
    node_after_first,
    frame,
    num_angles,
    displacement_norm,
    index_free,
    reference_point,
    length_scale,
):
    """Create all valid second-angle candidates for one executed ``q1``."""
    candidates = []
    for angle_index, theta in enumerate(
        angle_grid(num_angles, frame["tangent_dimension"])
    ):
        displacement = second_displacement(frame, theta, displacement_norm)
        if displacement is None:
            continue
        candidates.append(
            {
                "angle_index": int(angle_index),
                "theta": float(theta),
                "cos_theta": float(np.cos(theta)),
                "sin_theta": float(np.sin(theta)),
                "displacement": displacement,
                "input": macro_gp_input(
                    node_after_first,
                    frame,
                    displacement,
                    theta,
                    index_free,
                    reference_point,
                    length_scale,
                ),
            }
        )
    return candidates
