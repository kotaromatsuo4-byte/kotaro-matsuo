"""Gaussian-process surrogates used by ``main5``.

The functions in this module intentionally do not assume a five-bar linkage.
The number of nodes, the number of free coordinates, and the instantaneous
mechanism dimension are obtained from the model passed by the caller.

The surrogate input is an *executed transition*,

    x = [q_free / L, displacement_free / L],

where ``displacement`` is the tangent predictor displacement before the
corrector.  Using ``corrected_child - parent`` only for training would make
the training and candidate inputs inconsistent.

Three separate statistical quantities are intended to be fitted:

1. the probability that a rollout stops in a flat state;
2. a continuous flatness measure available for every completed rollout;
3. the terminal descriptor conditioned on the rollout being flat.

Collision descriptors are therefore never averaged with flat descriptors.
Only NumPy and SciPy are required.  Models are plain dictionaries rather than
custom classes so that the numerical steps remain easy to inspect.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.special import expit
from scipy.stats import norm

from descriptor import (
    find_last_nonflat_state_id,
    find_overlapping_member_pairs,
)
from mechanics import calc_dedx, calc_mode


def transition_input(
    node,
    displacement,
    index_free,
    reference_point,
    length_scale,
    layer_context_node=None,
):
    """Return the GP input for one proposed or executed transition.

    ``node`` and ``displacement`` both have shape ``(num_node, 3)``.
    Only free coordinates are retained.  The reference flat plane is parallel
    to the xy plane, hence ``reference_point`` is subtracted only from z.

    The state and action blocks both use the same physical length unit here.
    Their different characteristic scales are handled by the two kernel
    length scales, ``state_length_scale`` and ``action_length_scale``.
    """
    node_array = np.asarray(node, dtype=float).reshape(-1, 3)
    displacement_array = np.asarray(displacement, dtype=float).reshape(-1, 3)
    free = np.asarray(index_free, dtype=int).reshape(-1)

    shifted_node = node_array.copy()
    shifted_node[:, 2] -= float(reference_point)
    scale = float(length_scale)

    state_part = shifted_node.reshape(-1)[free] / scale
    # 重なり順descriptorは、候補状態より前の「最後の非平坦状態」に依存する
    # 場合がある。その既知のpath memoryを加え、(q, action)だけでは非Markov
    # になる状況を避ける。Noneなら従来のstate+action入力になる。
    if layer_context_node is not None:
        context = np.asarray(layer_context_node, dtype=float)
        free_z = free[free % 3 == 2]
        context_z = (
            context.reshape(-1)[free_z] - float(reference_point)
        ) / scale
        state_part = np.concatenate((state_part, context_z))
    action_part = displacement_array.reshape(-1)[free] / scale
    return np.concatenate((state_part, action_part))


def make_terminal_output_spec(
    initial_node,
    index_free,
    reference_point,
    length_scale,
):
    """Describe how to compress and reconstruct a terminal descriptor.

    The full continuous descriptor is

    ``[terminal xy / L, (last_nonflat_z - reference_point) / L]``.

    Every Cartesian degree of freedom appears exactly once in that vector:
    x and y in the terminal-shape part, and z in the last-nonflat part.
    Fixed coordinates can therefore be reconstructed from the initial model,
    and only entries corresponding to ``index_free`` need to be regressed.
    For a structure with ``n_free`` free Cartesian coordinates, the active
    GP output dimension is consequently ``n_free``.
    """
    initial = np.asarray(initial_node, dtype=float).reshape(-1, 3)
    free = np.asarray(index_free, dtype=int).reshape(-1)
    num_node = initial.shape[0]
    scale = float(length_scale)

    full_reference = np.concatenate(
        (
            (initial[:, :2] / scale).reshape(-1),
            (initial[:, 2] - float(reference_point)) / scale,
        )
    )

    active_positions = []
    for global_index in free:
        node_index, axis = divmod(int(global_index), 3)
        if axis < 2:
            descriptor_index = 2 * node_index + axis
        else:
            descriptor_index = 2 * num_node + node_index
        active_positions.append(descriptor_index)

    return {
        "num_node": int(num_node),
        "active_positions": np.asarray(active_positions, dtype=int),
        "full_reference": full_reference,
        "reference_point": float(reference_point),
        "length_scale": scale,
    }


def encode_terminal_target(terminal_node, observation_node, output_spec):
    """Compress a flat terminal and its last non-flat observation state."""
    terminal = np.asarray(terminal_node, dtype=float).reshape(-1, 3)
    observation = np.asarray(observation_node, dtype=float).reshape(-1, 3)
    scale = output_spec["length_scale"]
    reference = output_spec["reference_point"]

    full_descriptor = np.concatenate(
        (
            (terminal[:, :2] / scale).reshape(-1),
            (observation[:, 2] - reference) / scale,
        )
    )
    return full_descriptor[output_spec["active_positions"]]


def terminal_target_from_history(
    history,
    terminal_state_id,
    layer_threshold,
    output_spec,
):
    """Create the active continuous target for one flat rollout."""
    observation_state_id = find_last_nonflat_state_id(
        history,
        int(terminal_state_id),
        layer_threshold,
    )
    return encode_terminal_target(
        history[int(terminal_state_id)].node,
        history[observation_state_id].node,
        output_spec,
    )


def decode_terminal_target(active_descriptor, output_spec):
    """Reconstruct the full continuous descriptor from active GP outputs.

    A vector input returns a vector.  An array with arbitrary leading axes
    returns an array with the same leading axes and a final axis of length
    ``3 * num_node``.
    """
    active = np.asarray(active_descriptor, dtype=float)
    positions = output_spec["active_positions"]
    if active.shape[-1] != len(positions):
        raise ValueError("terminal descriptor has an unexpected dimension")

    full_shape = active.shape[:-1] + (len(output_spec["full_reference"]),)
    full = np.broadcast_to(output_spec["full_reference"], full_shape).copy()
    full[..., positions] = active
    return full


def _rms_squared_distance(block1, block2):
    """Pairwise squared RMS distance without a 3-D difference tensor."""
    first = np.asarray(block1, dtype=float)
    second = np.asarray(block2, dtype=float)
    dimension = first.shape[1]
    if dimension == 0:
        return np.zeros((len(first), len(second)), dtype=float)

    first_norm = np.sum(first * first, axis=1)[:, None]
    second_norm = np.sum(second * second, axis=1)[None, :]
    squared = first_norm + second_norm - 2.0 * first @ second.T
    return np.maximum(squared, 0.0) / dimension


def block_matern32_kernel(
    inputs1,
    inputs2,
    state_dimension,
    state_length_scale,
    action_length_scale,
    signal_variance=1.0,
):
    r"""Matérn-3/2 kernel with separate state and action length scales.

    The dimension-independent block distance is

    .. math::

        r^2 = \frac{\operatorname{mean}(q-q')^2}{\ell_q^2}
            + \frac{\operatorname{mean}(v-v')^2}{\ell_v^2}.

    RMS rather than a raw coordinate sum prevents the kernel scale from
    changing merely because a structure has more nodes.
    """
    first = np.atleast_2d(np.asarray(inputs1, dtype=float))
    second = np.atleast_2d(np.asarray(inputs2, dtype=float))
    state_dimension = int(state_dimension)

    if first.shape[1] != second.shape[1]:
        raise ValueError("kernel inputs must have the same dimension")
    if not 0 < state_dimension < first.shape[1]:
        raise ValueError("state_dimension must split state and action blocks")

    state_distance = _rms_squared_distance(
        first[:, :state_dimension],
        second[:, :state_dimension],
    ) / float(state_length_scale) ** 2
    action_distance = _rms_squared_distance(
        first[:, state_dimension:],
        second[:, state_dimension:],
    ) / float(action_length_scale) ** 2

    radius = np.sqrt(np.maximum(state_distance + action_distance, 0.0))
    scaled_radius = np.sqrt(3.0) * radius
    return float(signal_variance) * (
        1.0 + scaled_radius
    ) * np.exp(-scaled_radius)


def fit_shared_kernel_gp(
    inputs,
    targets,
    state_dimension,
    state_length_scale,
    action_length_scale,
    signal_variance=1.0,
    noise_std=0.05,
    jitter=1.0e-8,
    output_scale_floor=1.0e-2,
    output_mean=None,
):
    """Fit exact GPR for one or many outputs using one shared kernel.

    All output coordinates share one input kernel and one Cholesky factor, but
    have separate means and physical output scales.  Conditional output
    correlations are ignored in this first, transparent implementation.  The
    returned variance is consequently diagonal in output space.

    ``output_scale_floor`` is essential when the observed flat rollouts all
    have the same terminal descriptor.  Without the floor, normalization
    would turn the posterior uncertainty into exactly zero and prevent
    uncertainty-driven exploration.

    Exact fitting costs O(N^3) in the number of transition samples.  This
    function deliberately uses every supplied sample.  For several thousand
    transitions it should be called in batches rather than after every edge,
    or later replaced behind the same API by a sparse/RFF approximation.
    """
    x = np.asarray(inputs, dtype=float)
    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("GP inputs and targets must be aligned 2-D arrays")
    if len(x) == 0:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("GP training data must be finite")

    if output_mean is None:
        y_mean = np.mean(y, axis=0)
    else:
        y_mean = np.broadcast_to(
            np.asarray(output_mean, dtype=float),
            (y.shape[1],),
        ).copy()

    floor = np.broadcast_to(
        np.asarray(output_scale_floor, dtype=float),
        (y.shape[1],),
    )
    y_scale = np.maximum(np.std(y, axis=0), floor)
    normalized_target = (y - y_mean) / y_scale

    kernel_parameters = {
        "state_dimension": int(state_dimension),
        "state_length_scale": float(state_length_scale),
        "action_length_scale": float(action_length_scale),
        "signal_variance": float(signal_variance),
    }
    covariance = block_matern32_kernel(x, x, **kernel_parameters)
    diagonal_noise = float(noise_std) ** 2

    used_jitter = float(jitter)
    identity = np.eye(len(x))
    for _ in range(9):
        try:
            lower = cholesky(
                covariance + (diagonal_noise + used_jitter) * identity,
                lower=True,
                check_finite=False,
            )
            break
        except np.linalg.LinAlgError:
            used_jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("GP covariance remained non-positive definite")

    alpha = cho_solve(
        (lower, True),
        normalized_target,
        check_finite=False,
    )
    return {
        "kind": "shared_exact_gpr",
        "training_inputs": x.copy(),
        "target_mean": y_mean,
        "target_scale": y_scale,
        "lower_cholesky": lower,
        "alpha": alpha,
        "kernel_parameters": kernel_parameters,
        "noise_std": float(noise_std),
        "jitter": used_jitter,
        "output_dimension": int(y.shape[1]),
    }


def predict_shared_kernel_gp(
    model,
    query_inputs,
    include_noise=False,
    chunk_size=2048,
):
    """Return posterior mean and diagonal output variance.

    Both returned arrays have shape ``(num_query, output_dimension)``.
    ``include_noise=False`` gives epistemic/latent variance, which is normally
    the appropriate quantity for an acquisition function.
    """
    if model is None:
        raise ValueError("a fitted GP model is required")

    query = np.atleast_2d(np.asarray(query_inputs, dtype=float))
    means = []
    variances = []
    prior_variance = model["kernel_parameters"]["signal_variance"]

    for start in range(0, len(query), int(chunk_size)):
        query_chunk = query[start:start + int(chunk_size)]
        cross_covariance = block_matern32_kernel(
            model["training_inputs"],
            query_chunk,
            **model["kernel_parameters"],
        )
        normalized_mean = cross_covariance.T @ model["alpha"]
        mean = (
            model["target_mean"]
            + normalized_mean * model["target_scale"]
        )

        solved = solve_triangular(
            model["lower_cholesky"],
            cross_covariance,
            lower=True,
            check_finite=False,
        )
        latent_variance = prior_variance - np.sum(solved * solved, axis=0)
        latent_variance = np.maximum(latent_variance, 0.0)
        if include_noise:
            latent_variance += model["noise_std"] ** 2
        variance = (
            latent_variance[:, None]
            * model["target_scale"][None, :] ** 2
        )

        means.append(mean)
        variances.append(variance)

    return np.vstack(means), np.vstack(variances)


def sample_shared_kernel_gp(
    model,
    query_inputs,
    num_samples,
    rng,
):
    """Draw independent marginal output samples for every query candidate.

    The result has shape ``(num_samples, num_query, output_dimension)``.
    Cross-candidate and cross-output correlations are not represented; this is
    sufficient for estimating each candidate's marginal new-cell probability,
    but not for selecting a joint batch by q-EI or q-UCB.
    """
    mean, variance = predict_shared_kernel_gp(model, query_inputs)
    standard_normal = rng.normal(
        size=(int(num_samples),) + mean.shape,
    )
    return mean[None, :, :] + np.sqrt(variance)[None, :, :] * standard_normal


def fit_binary_probability_gp(
    inputs,
    binary_labels,
    state_dimension,
    state_length_scale,
    action_length_scale,
    signal_variance=1.0,
    noise_std=0.20,
    jitter=1.0e-8,
    prior_probability=0.5,
    label_probability=0.90,
):
    """Fit a cold-start-safe approximation to binary GP classification.

    Labels are mapped to finite logits and those logits are fitted by Gaussian
    process regression.  At prediction time the logistic-normal integral is
    approximated by ``sigmoid(mu/sqrt(1 + pi*variance/8))``.

    This is *not* an exact Bernoulli-likelihood GP classifier, so its output
    must not be interpreted as a calibrated physical probability without a
    held-out, rollout-level calibration check.  It is a stable and transparent
    ranking model for small data.  The prior latent mean is the logit of
    ``prior_probability``; therefore a data-free region returns to that prior
    even when the initial observations all belong to one class.
    """
    x = np.asarray(inputs, dtype=float)
    labels = np.asarray(binary_labels, dtype=float).reshape(-1)
    prior_probability = float(np.clip(prior_probability, 1.0e-6, 1.0 - 1.0e-6))
    prior_logit = float(np.log(prior_probability / (1.0 - prior_probability)))

    if len(x) == 0:
        return {
            "kind": "constant_binary_probability",
            "probability": prior_probability,
            "latent_variance": 1.0,
        }
    if x.ndim != 2 or len(x) != len(labels):
        raise ValueError("binary GP inputs and labels must be aligned")

    positive_probability = float(
        np.clip(label_probability, 0.5 + 1.0e-6, 1.0 - 1.0e-6)
    )
    positive_logit = np.log(
        positive_probability / (1.0 - positive_probability)
    )
    latent_target = np.where(labels > 0.5, positive_logit, -positive_logit)

    regression_model = fit_shared_kernel_gp(
        x,
        latent_target[:, None],
        state_dimension,
        state_length_scale,
        action_length_scale,
        signal_variance=signal_variance,
        noise_std=noise_std,
        jitter=jitter,
        output_scale_floor=positive_logit,
        output_mean=np.asarray([prior_logit]),
    )
    return {
        "kind": "binary_probability_gpr_approximation",
        "regression_model": regression_model,
        "prior_probability": prior_probability,
    }


def predict_binary_probability(model, query_inputs):
    """Return approximate positive-class probability and latent variance."""
    query = np.atleast_2d(np.asarray(query_inputs, dtype=float))
    if model["kind"] == "constant_binary_probability":
        probability = np.full(len(query), model["probability"], dtype=float)
        variance = np.full(len(query), model["latent_variance"], dtype=float)
        return probability, variance

    latent_mean, latent_variance = predict_shared_kernel_gp(
        model["regression_model"],
        query,
    )
    latent_mean = latent_mean[:, 0]
    latent_variance = latent_variance[:, 0]
    denominator = np.sqrt(1.0 + np.pi * latent_variance / 8.0)
    probability = expit(latent_mean / denominator)
    return probability, latent_variance


def fit_surrogate_models(
    inputs,
    flat_labels,
    flatness_targets,
    descriptor_targets,
    state_dimension,
    state_length_scale,
    action_length_scale,
    signal_variance=1.0,
    regression_noise_std=0.05,
    classifier_noise_std=0.20,
    jitter=1.0e-8,
    flatness_scale_floor=0.10,
    descriptor_scale_floor=0.05,
    flatness_output_mean=None,
    prior_flat_probability=0.5,
):
    """Fit stop, flatness, and flat-conditional descriptor surrogates.

    ``descriptor_targets`` must have one row per input.  Rows corresponding to
    collision or timeout may be filled with NaNs; only finite rows whose
    ``flat_labels`` are true are used for the descriptor GP.
    """
    x = np.asarray(inputs, dtype=float)
    flat = np.asarray(flat_labels, dtype=bool).reshape(-1)
    flatness = np.asarray(flatness_targets, dtype=float).reshape(-1)
    descriptor = np.asarray(descriptor_targets, dtype=float)
    if descriptor.ndim == 1:
        descriptor = descriptor[:, None]

    if len(x) == 0:
        flat_probability_model = fit_binary_probability_gp(
            x,
            np.empty(0),
            state_dimension,
            state_length_scale,
            action_length_scale,
            prior_probability=prior_flat_probability,
        )
        return {
            "flat_probability": flat_probability_model,
            "flatness": None,
            "descriptor": None,
            "num_samples": 0,
            "num_flat_samples": 0,
        }

    if not (len(x) == len(flat) == len(flatness) == len(descriptor)):
        raise ValueError("all surrogate training arrays must have equal length")

    flat_probability_model = fit_binary_probability_gp(
        x,
        flat,
        state_dimension,
        state_length_scale,
        action_length_scale,
        signal_variance=signal_variance,
        noise_std=classifier_noise_std,
        jitter=jitter,
        prior_probability=prior_flat_probability,
    )
    flatness_model = fit_shared_kernel_gp(
        x,
        flatness[:, None],
        state_dimension,
        state_length_scale,
        action_length_scale,
        signal_variance=signal_variance,
        noise_std=regression_noise_std,
        jitter=jitter,
        output_scale_floor=flatness_scale_floor,
        output_mean=flatness_output_mean,
    )

    descriptor_mask = flat & np.all(np.isfinite(descriptor), axis=1)
    if np.any(descriptor_mask):
        descriptor_model = fit_shared_kernel_gp(
            x[descriptor_mask],
            descriptor[descriptor_mask],
            state_dimension,
            state_length_scale,
            action_length_scale,
            signal_variance=signal_variance,
            noise_std=regression_noise_std,
            jitter=jitter,
            output_scale_floor=descriptor_scale_floor,
        )
    else:
        descriptor_model = None

    return {
        "flat_probability": flat_probability_model,
        "flatness": flatness_model,
        "descriptor": descriptor_model,
        "num_samples": int(len(x)),
        "num_flat_samples": int(np.count_nonzero(descriptor_mask)),
    }


def unit_mode_directions(num_mode, num_directions, rng):
    """Generate fixed-norm coefficient directions for an arbitrary DOF.

    One-dimensional mechanisms have only the two unique directions.  A
    two-dimensional mechanism uses an evenly spaced angular grid with a small
    random phase rotation.  Higher-dimensional mechanisms use antipodal pairs
    of random unit vectors.
    """
    num_mode = int(num_mode)
    num_directions = int(num_directions)
    if num_mode <= 0 or num_directions <= 0:
        return np.empty((0, max(num_mode, 0)), dtype=float)

    if num_mode == 1:
        if num_directions == 1:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            return np.asarray([[sign]])
        return np.asarray([[1.0], [-1.0]])

    if num_mode == 2:
        phase = rng.uniform(0.0, 2.0 * np.pi / num_directions)
        angle = phase + 2.0 * np.pi * np.arange(num_directions) / num_directions
        return np.column_stack((np.cos(angle), np.sin(angle)))

    num_base = (num_directions + 1) // 2
    base = rng.normal(size=(num_base, num_mode))
    norms = np.linalg.norm(base, axis=1)
    while np.any(norms < 1.0e-12):
        invalid = norms < 1.0e-12
        base[invalid] = rng.normal(size=(np.count_nonzero(invalid), num_mode))
        norms = np.linalg.norm(base, axis=1)
    base /= norms[:, None]

    directions = []
    for direction in base:
        directions.append(direction)
        if len(directions) < num_directions:
            directions.append(-direction)
    return np.asarray(directions[:num_directions])


def generate_tangent_candidates(
    history,
    parent_ids,
    member,
    index_free,
    num_directions,
    perturbation_scale,
    reference_point,
    length_scale,
    rng,
    mode_cache=None,
    mode_tolerance=None,
    layer_threshold=None,
    include_transported_direction=False,
    transport_min_projection_ratio=0.10,
):
    r"""Generate tangent-space perturbations without running a corrector.

    The returned list contains dictionaries with ``parent_id``, the *scaled*
    ``coefficient``, physical ``displacement``, and the GP ``input``.  The
    mechanism dimension may differ between states; the physical GP input size
    remains fixed because it is determined only by ``index_free``.

    If ``include_transported_direction`` is true and the parent was itself
    created by a perturbation, the nearest grid direction is replaced by a
    candidate that transports the previous *physical* predictor displacement
    to the current tangent space,

    .. math::

        \bm v_{\mathrm{tr}}
        = \rho\,
          \frac{\bm B(\bm q)\bm B(\bm q)^T\bm v_{\mathrm{prev}}}
          {\|\bm B(\bm q)\bm B(\bm q)^T\bm v_{\mathrm{prev}}\|}.

    The construction uses the tangent-space projector, so it is invariant to
    sign changes and orthogonal rotations of the SVD basis.
    """
    member_array = np.asarray(member, dtype=int).reshape(-1, 2)
    free = np.asarray(index_free, dtype=int)
    num_member = len(member_array)
    candidates = []

    for parent_id in parent_ids:
        parent_id = int(parent_id)
        state = history[parent_id]
        if state.terminal_reason is not None:
            continue
        node = np.asarray(state.node, dtype=float)
        num_node = len(node)

        if mode_cache is not None and parent_id in mode_cache:
            mode = mode_cache[parent_id]
        else:
            jacobian = calc_dedx(
                node,
                num_node,
                member_array,
                num_member,
                free,
            )
            mode = calc_mode(
                free,
                jacobian,
                num_node,
                tol=mode_tolerance,
            )
            if mode_cache is not None:
                mode_cache[parent_id] = mode

        if layer_threshold is None:
            layer_context_node = None
        else:
            context_state_id = find_last_nonflat_state_id(
                history,
                parent_id,
                layer_threshold,
            )
            layer_context_node = history[context_state_id].node

        previous_projection = None
        previous_projection_norm = 0.0
        if (
            state.incoming_step_type == "perturbation"
            and state.incoming_displacement is not None
            and mode.shape[1] > 0
        ):
            previous = np.asarray(
                state.incoming_displacement,
                dtype=float,
            ).reshape(-1)
            previous_norm = float(np.linalg.norm(previous))
            # calc_mode currently returns orthonormal SVD vectors.  Using the
            # pseudoinverse keeps this projector valid even if another
            # mechanics backend later returns a non-orthonormal basis.
            previous_projection = mode @ (
                np.linalg.pinv(mode, rcond=1.0e-12) @ previous
            )
            previous_projection_norm = float(np.linalg.norm(previous_projection))
            projection_ratio = previous_projection_norm / max(
                previous_norm,
                1.0e-12,
            )
            if (
                previous_projection_norm <= 1.0e-12
                or projection_ratio < float(transport_min_projection_ratio)
            ):
                previous_projection = None
                previous_projection_norm = 0.0
                projection_ratio = 0.0
        else:
            projection_ratio = 0.0

        def append_candidate(
            coefficient,
            displacement,
            direction_index,
            candidate_kind,
        ):
            displacement_vector = np.asarray(displacement, dtype=float).reshape(-1)
            alignment = 0.0
            if previous_projection is not None:
                displacement_norm = float(np.linalg.norm(displacement_vector))
                if displacement_norm > 0.0:
                    alignment = float(
                        np.clip(
                            np.dot(displacement_vector, previous_projection)
                            / (displacement_norm * previous_projection_norm),
                            0.0,
                            1.0,
                        )
                    )
            candidates.append(
                {
                    "parent_id": parent_id,
                    "direction_index": int(direction_index),
                    "coefficient": np.asarray(coefficient, dtype=float).copy(),
                    "displacement": np.asarray(displacement, dtype=float).reshape(
                        num_node,
                        3,
                    ),
                    "input": transition_input(
                        node,
                        displacement,
                        free,
                        reference_point,
                        length_scale,
                        layer_context_node,
                    ),
                    "candidate_kind": candidate_kind,
                    "is_transported_direction": candidate_kind == "transported",
                    "transport_alignment": alignment,
                    "transport_projection_ratio": float(projection_ratio),
                }
            )

        parent_candidate_start = len(candidates)
        directions = unit_mode_directions(
            mode.shape[1],
            num_directions,
            rng,
        )
        for direction_index, direction in enumerate(directions):
            coefficient = float(perturbation_scale) * direction
            displacement = (mode @ coefficient).reshape(num_node, 3)
            append_candidate(
                coefficient,
                displacement,
                direction_index,
                "tangent_grid",
            )

        if include_transported_direction and previous_projection is not None:
            transported_vector = (
                float(perturbation_scale)
                * previous_projection
                / previous_projection_norm
            )
            transported_coefficient = mode.T @ transported_vector
            append_candidate(
                transported_coefficient,
                transported_vector.reshape(num_node, 3),
                -1,
                "transported",
            )
            # Replace the nearest grid direction instead of adding a new arm.
            # Thus every eligible parent keeps the same candidate count and a
            # random selector is not biased merely by list multiplicity.
            transported_candidate = candidates.pop()
            local_candidates = candidates[parent_candidate_start:]
            if local_candidates:
                closest = int(
                    np.argmax(
                        [
                            candidate["transport_alignment"]
                            for candidate in local_candidates
                        ]
                    )
                )
                candidates[parent_candidate_start + closest] = (
                    transported_candidate
                )
                # If this exact transported action has already been evaluated,
                # main5 can restore the grid arm that was replaced here.
                transported_candidate["replaced_grid_candidate"] = (
                    local_candidates[closest]
                )

    return candidates


def decode_shape_and_layer(
    active_descriptor,
    output_spec,
    member,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """Decode a continuous GP output into archive shape and layer key.

    Tolerances are dimensionless because the reconstructed continuous
    descriptor is already divided by ``length_scale``.
    """
    full = decode_terminal_target(active_descriptor, output_spec)
    if full.ndim != 1:
        raise ValueError("decode_shape_and_layer accepts one descriptor")
    if not np.all(np.isfinite(full)):
        return full[:2 * output_spec["num_node"]], (), False

    num_node = output_spec["num_node"]
    shape_descriptor = full[:2 * num_node]
    observation_z = full[2 * num_node:]

    normalized_terminal_node = np.zeros((num_node, 3), dtype=float)
    normalized_terminal_node[:, :2] = shape_descriptor.reshape(num_node, 2)
    overlaps = find_overlapping_member_pairs(
        normalized_terminal_node,
        np.asarray(member, dtype=int).reshape(-1, 2),
        1.0,
        overlap_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )

    layer_key = []
    valid = True
    member_array = np.asarray(member, dtype=int).reshape(-1, 2)
    for overlap in overlaps:
        member_i = overlap["member_i"]
        member_j = overlap["member_j"]
        i0, i1 = member_array[member_i]
        j0, j1 = member_array[member_j]
        parameter_i = overlap["parameter_i"]
        parameter_j = overlap["parameter_j"]

        z_i = (
            (1.0 - parameter_i) * observation_z[i0]
            + parameter_i * observation_z[i1]
        )
        z_j = (
            (1.0 - parameter_j) * observation_z[j0]
            + parameter_j * observation_z[j1]
        )
        separation = z_i - z_j

        if separation > separation_tolerance:
            upper_member = int(member_i)
        elif separation < -separation_tolerance:
            upper_member = int(member_j)
        else:
            upper_member = -1
            valid = False
        layer_key.append((int(member_i), int(member_j), upper_member))

    return shape_descriptor, tuple(layer_key), valid


def normalized_archive_distance(
    shape_descriptor,
    layer_key,
    elite_archive,
    num_node,
    rms_tolerance,
    max_tolerance,
    layer_mismatch_distance=2.0,
):
    """Return distance to the nearest existing cell in archive units.

    A value no larger than one means that the descriptor belongs to an
    existing cell.  A different layer key is itself a new cell, represented by
    the finite value ``layer_mismatch_distance`` for acquisition ranking.
    """
    if len(elite_archive) == 0:
        return 1.0

    shape = np.asarray(shape_descriptor, dtype=float).reshape(num_node, 2)
    nearest = np.inf
    found_same_layer = False
    for cell in elite_archive:
        if layer_key != cell["layer_key"]:
            continue
        found_same_layer = True
        existing = np.asarray(cell["shape_descriptor"], dtype=float).reshape(
            num_node,
            2,
        )
        node_distance = np.linalg.norm(shape - existing, axis=1)
        rms_distance = np.sqrt(np.mean(node_distance ** 2))
        maximum_distance = np.max(node_distance)
        distance = max(
            rms_distance / max(float(rms_tolerance), 1.0e-12),
            maximum_distance / max(float(max_tolerance), 1.0e-12),
        )
        nearest = min(nearest, distance)

    if not found_same_layer:
        return float(layer_mismatch_distance)
    return float(nearest)


def probability_below_gp(model, query_inputs, threshold):
    """Return P(Y <= threshold) for a scalar Gaussian-process output."""
    if model is None:
        return np.full(len(np.atleast_2d(query_inputs)), 0.5, dtype=float)
    mean, variance = predict_shared_kernel_gp(model, query_inputs)
    mean = mean[:, 0]
    standard_deviation = np.sqrt(np.maximum(variance[:, 0], 0.0))
    deterministic = standard_deviation < 1.0e-12
    probability = np.empty(len(mean), dtype=float)
    probability[deterministic] = (
        mean[deterministic] <= float(threshold)
    ).astype(float)
    probability[~deterministic] = norm.cdf(
        (float(threshold) - mean[~deterministic])
        / standard_deviation[~deterministic]
    )
    return probability


def descriptor_novelty_statistics(
    descriptor_model,
    query_inputs,
    elite_archive,
    output_spec,
    member,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
    uncertainty_beta=2.0,
    layer_uncertainty_scale=0.10,
    num_posterior_samples=0,
    rng=None,
):
    """Predict mean novelty, confidence radius, and new-cell probability."""
    query = np.atleast_2d(np.asarray(query_inputs, dtype=float))
    if descriptor_model is None:
        zeros = np.zeros(len(query), dtype=float)
        return {
            "mean_distance": zeros.copy(),
            "uncertainty_radius": zeros.copy(),
            "posterior_new_probability": zeros.copy(),
        }

    mean, variance = predict_shared_kernel_gp(descriptor_model, query)
    positions = output_spec["active_positions"]
    shape_dimension = 2 * output_spec["num_node"]
    output_scales = np.where(
        positions < shape_dimension,
        max(float(rms_tolerance), 1.0e-12),
        max(float(layer_uncertainty_scale), 1.0e-12),
    )

    # UCB側は平均と半径に同一のweighted RMS normを用いる。これにより
    # continuous descriptor集合への距離については1-Lipschitz上界として
    # 説明できる。実archiveのmax-node閾値と離散layer判定そのものに対する
    # 厳密上界ではないため、最終的なcell判定は必ず実rolloutで行う。
    mean_valid = np.all(np.isfinite(mean), axis=1)
    if len(elite_archive) == 0:
        mean_distance = np.ones(len(query), dtype=float)
    else:
        elite_targets = []
        for cell in elite_archive:
            continuous = cell.get("continuous_descriptor")
            if continuous is None:
                continue
            elite_targets.append(
                np.asarray(continuous, dtype=float)[positions]
            )
        if elite_targets:
            elite_targets = np.asarray(elite_targets, dtype=float)
            normalized_difference = (
                mean[:, None, :] - elite_targets[None, :, :]
            ) / output_scales[None, None, :]
            mean_distance = np.min(
                np.sqrt(np.mean(normalized_difference ** 2, axis=2)),
                axis=1,
            )
        else:
            mean_distance = np.ones(len(query), dtype=float)

    normalized_variance = variance / output_scales[None, :] ** 2
    uncertainty_radius = np.sqrt(
        float(uncertainty_beta)
        * np.mean(normalized_variance, axis=1)
    )

    posterior_new_probability = np.zeros(len(query), dtype=float)
    if int(num_posterior_samples) > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        samples = sample_shared_kernel_gp(
            descriptor_model,
            query,
            int(num_posterior_samples),
            rng,
        )
        new_count = np.zeros(len(query), dtype=float)
        for sample in samples:
            for index, prediction in enumerate(sample):
                shape, layer_key, valid = decode_shape_and_layer(
                    prediction,
                    output_spec,
                    member,
                    overlap_tolerance,
                    separation_tolerance,
                    collinearity_tolerance,
                    parallel_tolerance,
                )
                if not valid:
                    continue
                distance = normalized_archive_distance(
                    shape,
                    layer_key,
                    elite_archive,
                    output_spec["num_node"],
                    rms_tolerance,
                    max_tolerance,
                )
                if len(elite_archive) == 0 or distance > 1.0:
                    new_count[index] += 1.0
        posterior_new_probability = new_count / int(num_posterior_samples)

    return {
        "mean_distance": mean_distance,
        "mean_valid": mean_valid,
        "uncertainty_radius": uncertainty_radius,
        "posterior_new_probability": posterior_new_probability,
    }


def score_acquisition(
    surrogate_models,
    query_inputs,
    elite_archive,
    output_spec,
    member,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
    flatness_target_threshold,
    acquisition_kind="ucb",
    novelty_kappa=1.0,
    reach_weight=0.25,
    cold_uncertainty_weight=0.10,
    layer_uncertainty_scale=0.10,
    num_posterior_samples=64,
    rng=None,
):
    r"""Score state-action candidates for discovery of new flat cells.

    With a flat-conditional descriptor model, ``acquisition_kind='ucb'`` uses

    .. math::

        p_F(x) [d_A(\mu_D(x)) + \kappa r_D(x)]
        + \lambda_h p_{\mathrm{near}}(x).

    ``acquisition_kind='posterior_new'`` replaces the bracket by the Monte
    Carlo probability that a flat-conditional descriptor is outside the
    current archive.  Before any flat rollout exists, the score falls back to
    flat probability, continuous reach probability, and stop-model latent
    uncertainty.  A random selection fraction should still be applied by the
    caller so that model error cannot permanently hide a basin.
    """
    query = np.atleast_2d(np.asarray(query_inputs, dtype=float))
    flat_probability, stop_latent_variance = predict_binary_probability(
        surrogate_models["flat_probability"],
        query,
    )
    reach_probability = probability_below_gp(
        surrogate_models.get("flatness"),
        query,
        flatness_target_threshold,
    )

    descriptor_model = surrogate_models.get("descriptor")
    if descriptor_model is None:
        score = (
            flat_probability * reach_probability
            + float(cold_uncertainty_weight)
            * np.sqrt(np.maximum(stop_latent_variance, 0.0))
        )
        zeros = np.zeros(len(query), dtype=float)
        return {
            "score": score,
            "flat_probability": flat_probability,
            "reach_probability": reach_probability,
            "mean_novelty": zeros.copy(),
            "descriptor_uncertainty": zeros.copy(),
            "posterior_new_probability": zeros.copy(),
            "stop_latent_variance": stop_latent_variance,
        }

    posterior_count = (
        int(num_posterior_samples)
        if acquisition_kind == "posterior_new"
        else 0
    )
    novelty = descriptor_novelty_statistics(
        descriptor_model,
        query,
        elite_archive,
        output_spec,
        member,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
        rms_tolerance,
        max_tolerance,
        uncertainty_beta=novelty_kappa ** 2,
        layer_uncertainty_scale=layer_uncertainty_scale,
        num_posterior_samples=posterior_count,
        rng=rng,
    )

    if acquisition_kind == "ucb":
        discovery_value = (
            novelty["mean_distance"]
            + novelty["uncertainty_radius"]
        )
    elif acquisition_kind == "posterior_new":
        discovery_value = novelty["posterior_new_probability"]
    else:
        raise ValueError("acquisition_kind must be 'ucb' or 'posterior_new'")

    score = (
        flat_probability * discovery_value
        + float(reach_weight) * reach_probability
    )
    return {
        "score": score,
        "flat_probability": flat_probability,
        "reach_probability": reach_probability,
        "mean_novelty": novelty["mean_distance"],
        "descriptor_uncertainty": novelty["uncertainty_radius"],
        "posterior_new_probability": novelty["posterior_new_probability"],
        "stop_latent_variance": stop_latent_variance,
    }


def select_candidate_indices(
    candidates,
    scores,
    num_guided,
    num_random,
    rng,
    unique_guided_parents=False,
):
    """Select high-score candidates and an explicit random exploration set."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(candidates) != len(values):
        raise ValueError("candidate and score counts differ")

    order = np.argsort(-values, kind="stable")
    guided = []
    guided_parents = set()
    for index in order:
        if len(guided) >= int(num_guided):
            break
        parent_id = int(candidates[int(index)]["parent_id"])
        if unique_guided_parents and parent_id in guided_parents:
            continue
        guided.append(int(index))
        guided_parents.add(parent_id)

    remaining = np.asarray(
        [index for index in range(len(candidates)) if index not in guided],
        dtype=int,
    )
    random_count = min(int(num_random), len(remaining))
    if random_count > 0:
        random_indices = rng.choice(
            remaining,
            size=random_count,
            replace=False,
        ).astype(int).tolist()
    else:
        random_indices = []
    return guided + random_indices
