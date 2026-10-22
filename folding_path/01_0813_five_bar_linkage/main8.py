"""Variable-length transported-perturbation macro search.

The decision variable of one search rollout is the complete, unexecuted
macro

    (parent StateNode, first physical tangent displacement v1, integer K).

After executing ``v1``, the previous *physical predictor displacement* is
projected into the tangent space of every corrected child and rescaled to the
same norm.  At most ``K`` perturbation edges are therefore chained before a
deterministic suffix is run.  This is a basis-invariant way to restrict the
otherwise rapidly growing multi-step action space to a persistent direction.

The four selectors use exactly the same candidate-pool construction:

``variable_k_random``
    Uniformly sample ``(parent, v1, K)`` from the common pool.

``variable_k_gp_argmax``
    Select the largest novelty-UCB score (with an explicit epsilon-random
    quota).  This intentionally tests strong GP selection pressure.

``variable_k_gp_argmax_shuffled``
    Apply the same argmax after permuting scores among candidates.

``variable_k_gp_filter``
    Keep the top score fraction and sample uniformly inside it.  The GP is a
    filter rather than a brittle global optimizer.

``variable_k_gp_filter_shuffled``
    Permute scores among the same candidates before applying the same filter.
    This negative control separates GP ranking information from the effects
    of the warmup, pool, epsilon quota, and filter cardinality.

``variable_k_gp_group_argmax`` / ``variable_k_gp_group_filter``
    Uniformly choose one ``(parent, v1)`` group first, then use the GP only to
    rank K inside that group.  Their shuffled controls permute score-to-K
    correspondence independently inside every group and never move a score
    between parents or first directions.

One completed macro gives one teacher.  Flat, collision, timeout,
corrector-failure, and transport-projection-failure outcomes train the flat
probability and flatness models.  The continuous terminal descriptor is
defined only for flat outcomes.  Cost is ``log(1 + mechanics edges)``; a
branch cut only by the global budget is right-censored and excluded.

All selectors start with a paired warmup.  For the same ``(parent, v1)``, the
default anchors ``K={1,2,4,8,16,24}`` are executed as counterfactual branches.
Every repeated mechanics edge is charged to the common budget.
``warmup_prefix_sha256`` and ``warmup_training_sha256`` make exact matching
auditable across selectors.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from angle_macro import project_to_tangent, sample_tangent_displacement, tangent_basis
from gp import (
    fit_shared_kernel_gp,
    fit_surrogate_models,
    make_terminal_output_spec,
    predict_shared_kernel_gp,
    score_acquisition,
    terminal_target_from_history,
)
from main5 import (
    _configuration,
    _deterministic_roll_once,
    _discovery_auc,
    _discovery_event,
    _initial_deterministic_trunk,
    _load_model,
    _update_archive,
)
from main6 import _perturbation_step, _prefix_hash
from main7 import _centered_free_coordinates, _last_nonflat_context
from mechanics import compute_gravity_potential
from result import reset_output_directory, save_discovery_curve, save_elite_paths
from state import create_history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int_list(name, default):
    """Read a comma-separated integer catalogue without hiding duplicates."""
    text = os.environ.get(name)
    if text is None or not text.strip():
        return [int(value) for value in default]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _configuration8():
    """Return physical defaults plus main8-only search parameters."""
    selector = os.environ.get("FOLDING_MAIN8_SELECTOR", "variable_k_gp_filter")
    allowed = (
        "variable_k_random",
        "variable_k_gp_argmax",
        "variable_k_gp_argmax_shuffled",
        "variable_k_gp_filter",
        "variable_k_gp_filter_shuffled",
        "variable_k_gp_group_argmax",
        "variable_k_gp_group_argmax_shuffled",
        "variable_k_gp_group_filter",
        "variable_k_gp_group_filter_shuffled",
    )
    if selector not in allowed:
        raise ValueError(
            "FOLDING_MAIN8_SELECTOR must be variable_k_random, "
            "variable_k_gp_argmax, variable_k_gp_argmax_shuffled, "
            "variable_k_gp_filter, variable_k_gp_filter_shuffled, "
            "variable_k_gp_group_argmax, "
            "variable_k_gp_group_argmax_shuffled, "
            "variable_k_gp_group_filter, or "
            "variable_k_gp_group_filter_shuffled"
        )

    config = _configuration()
    k_min = _env_int("FOLDING_MAIN8_K_MIN", 1)
    k_max = _env_int("FOLDING_MAIN8_K_MAX", 24)
    explicit_k_values = _env_int_list(
        "FOLDING_MAIN8_K_VALUES", range(k_min, k_max + 1)
    )
    warmup_k_values = _env_int_list(
        "FOLDING_MAIN8_WARMUP_K_VALUES", (1, 2, 4, 8, 16, 24)
    )
    config.update(
        {
            "algorithm_version": (
                "main8_variable_k_transport_macro_gp_v3_group_k"
                if selector.startswith("variable_k_gp_group_")
                else "main8_variable_k_transport_macro_gp_v2"
            ),
            "selector": selector,
            "candidate_parent_limit": _env_int("FOLDING_MAIN8_PARENT_LIMIT", 16),
            "first_directions_per_parent": _env_int(
                "FOLDING_MAIN8_FIRST_DIRECTIONS", 2
            ),
            "k_min": k_min,
            "k_max": k_max,
            "k_values": sorted(set(explicit_k_values)),
            "paired_warmup_groups": _env_int(
                "FOLDING_MAIN8_PAIRED_WARMUP_GROUPS", 2
            ),
            "warmup_k_values": warmup_k_values,
            "epsilon_random": _env_float("FOLDING_MAIN8_EPSILON", 0.20),
            "filter_keep_fraction": _env_float(
                "FOLDING_MAIN8_FILTER_KEEP_FRACTION", 0.25
            ),
            "refit_rollout_interval": _env_int(
                "FOLDING_MAIN8_REFIT_INTERVAL", 3
            ),
            "transport_min_projection_ratio": _env_float(
                "FOLDING_MAIN8_MIN_PROJECTION_RATIO", 0.10
            ),
            "state_length_scale": _env_float(
                "FOLDING_MAIN8_GP_STATE_LENGTH", 0.50
            ),
            # The local feature transform below makes the existing two-block
            # GP kernel exactly equivalent to three distance blocks:
            # state RMS/l_q, v1 RMS/l_v, and log(K) Euclidean/l_K.
            "first_displacement_length_scale": _env_float(
                "FOLDING_MAIN8_GP_V_LENGTH", 0.50
            ),
            "k_length_scale": _env_float(
                "FOLDING_MAIN8_GP_K_LENGTH", 0.35
            ),
            "action_length_scale": 1.0,
            "cost_output_scale_floor": _env_float(
                "FOLDING_MAIN8_COST_SCALE_FLOOR", 0.05
            ),
            "result_name": os.environ.get(
                "FOLDING_RESULT_NAME", f"main8_{selector}"
            ),
        }
    )

    if config["candidate_parent_limit"] <= 0:
        raise ValueError("FOLDING_MAIN8_PARENT_LIMIT must be positive")
    if config["first_directions_per_parent"] <= 0:
        raise ValueError("FOLDING_MAIN8_FIRST_DIRECTIONS must be positive")
    if config["k_min"] <= 0 or config["k_max"] < config["k_min"]:
        raise ValueError("main8 requires 1 <= K_MIN <= K_MAX")
    if not config["k_values"]:
        raise ValueError("FOLDING_MAIN8_K_VALUES must not be empty")
    if min(config["k_values"]) <= 0:
        raise ValueError("all FOLDING_MAIN8_K_VALUES must be positive")
    # An explicit catalogue defines the effective bounds.  K_MIN/K_MAX are
    # retained in configuration.json to document how the default was made.
    config["k_min"] = int(min(config["k_values"]))
    config["k_max"] = int(max(config["k_values"]))
    if config["paired_warmup_groups"] < 0:
        raise ValueError("FOLDING_MAIN8_PAIRED_WARMUP_GROUPS must be nonnegative")
    if not 0.0 <= config["epsilon_random"] <= 1.0:
        raise ValueError("FOLDING_MAIN8_EPSILON must be in [0, 1]")
    if not 0.0 < config["filter_keep_fraction"] <= 1.0:
        raise ValueError("FOLDING_MAIN8_FILTER_KEEP_FRACTION must be in (0, 1]")
    if config["refit_rollout_interval"] <= 0:
        raise ValueError("FOLDING_MAIN8_REFIT_INTERVAL must be positive")
    if not 0.0 <= config["transport_min_projection_ratio"] <= 1.0:
        raise ValueError("FOLDING_MAIN8_MIN_PROJECTION_RATIO must be in [0, 1]")
    if not config["warmup_k_values"]:
        raise ValueError("FOLDING_MAIN8_WARMUP_K_VALUES must not be empty")
    if config["first_displacement_length_scale"] <= 0.0:
        raise ValueError("FOLDING_MAIN8_GP_V_LENGTH must be positive")
    if config["k_length_scale"] <= 0.0:
        raise ValueError("FOLDING_MAIN8_GP_K_LENGTH must be positive")
    return config


def _make_model8(base_directory, config):
    node, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    free_dimension = int(len(index_free))
    length_scale = float(np.mean(member_length))
    # State kernel block: q + last-nonflat path context.
    # Action block: physical v1 + log-K.  Its features are pre-scaled so the
    # shared GP implementation evaluates a separate v1 and K distance.
    return node, {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": int(len(node)),
        "num_member": int(len(member)),
        "length_scale": length_scale,
        "free_dimension": free_dimension,
        "input_state_dimension": 2 * free_dimension,
        "input_dimension": 3 * free_dimension + 1,
    }


def _k_values(config):
    return np.asarray(config["k_values"], dtype=int)


def _macro_input(parent_node, context_node, first_displacement, k, model, config):
    """Encode ``(q, path context, v1, K)`` without SVD coefficients."""
    q_part = _centered_free_coordinates(parent_node, model, config)
    context_part = _centered_free_coordinates(context_node, model, config)
    free = model["index_free"]
    v1_raw = (
        np.asarray(first_displacement, dtype=float).reshape(-1)[free]
        / model["length_scale"]
    )
    log_k_raw = np.log1p(float(k)) / np.log1p(float(config["k_max"]))

    # gp.block_matern32_kernel averages squared distance over the whole action
    # block.  With d physical v1 coordinates and one K coordinate, the two
    # factors below cancel that averaging exactly:
    #
    #   mean((v-v')^2)/l_v^2 + (logK-logK')^2/l_K^2.
    #
    # Thus K is not diluted merely because a larger structure has more nodes.
    dimension = model["free_dimension"]
    v1_part = (
        np.sqrt((dimension + 1.0) / dimension)
        * v1_raw
        / config["first_displacement_length_scale"]
    )
    log_k = (
        np.sqrt(dimension + 1.0) * log_k_raw / config["k_length_scale"]
    )
    return np.concatenate((q_part, context_part, v1_part, [log_k]))


def _candidate_pool(history, model, config, rng):
    """Generate one bounded common pool before running candidate mechanics."""
    valid = np.asarray(
        [state.state_id for state in history if state.terminal_reason is None],
        dtype=int,
    )
    if len(valid) == 0:
        return []
    parent_count = min(len(valid), config["candidate_parent_limit"])
    if parent_count < len(valid):
        parent_ids = rng.choice(valid, size=parent_count, replace=False)
    else:
        parent_ids = rng.permutation(valid)

    candidates = []
    group_id = 0
    values = _k_values(config)
    for parent_value in parent_ids:
        parent_id = int(parent_value)
        parent_node = history[parent_id].node
        context_node = _last_nonflat_context(
            history, parent_id, config["layer_threshold"]
        )
        tangent_dimension = int(tangent_basis(parent_node, model).shape[1])
        if tangent_dimension == 0:
            continue
        for direction_index in range(config["first_directions_per_parent"]):
            v1 = sample_tangent_displacement(
                parent_node,
                model,
                config["perturbation_scale"],
                rng,
            )
            if v1 is None:
                continue
            for k_index, k in enumerate(values):
                candidates.append(
                    {
                        "group_id": int(group_id),
                        "parent_id": parent_id,
                        "direction_index": int(direction_index),
                        "v1": np.asarray(v1, dtype=float).copy(),
                        "k": int(k),
                        "k_index": int(k_index),
                        "tangent_dimension": tangent_dimension,
                        "input": _macro_input(
                            parent_node,
                            context_node,
                            v1,
                            int(k),
                            model,
                            config,
                        ),
                    }
                )
            group_id += 1
    return candidates


def _nearest_k_in_group(candidates, group_id, target_k):
    indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["group_id"] == int(group_id)
    ]
    if not indices:
        return None
    distances = [abs(candidates[index]["k"] - int(target_k)) for index in indices]
    return int(indices[int(np.argmin(distances))])


def _transport_previous(node, previous_displacement, model, config):
    """Project the preceding physical action and preserve its physical norm."""
    previous = np.asarray(previous_displacement, dtype=float).reshape(-1)
    previous_norm = float(np.linalg.norm(previous))
    projection, _ = project_to_tangent(node, previous, model)
    projection_norm = float(np.linalg.norm(projection))
    ratio = projection_norm / max(previous_norm, 1.0e-12)
    if (
        projection_norm <= 1.0e-12
        or ratio < config["transport_min_projection_ratio"]
    ):
        return None, float(ratio)
    displacement = previous_norm * projection / projection_norm
    return displacement.reshape(model["num_node"], 3), float(ratio)


def _complete_independent_suffix(
    history, current, new_state_ids, remaining_budget, model, config
):
    """Run up to ``max_rollout_steps`` deterministic edges after the chain.

    The suffix horizon is deliberately independent of K.  Consequently the
    mechanics cost of one macro can be as large as ``K + max_rollout_steps``;
    increasing K never silently steals deterministic convergence steps.
    """
    suffix_steps = 0
    while (
        current.terminal_reason is None
        and suffix_steps < config["max_rollout_steps"]
        and suffix_steps < int(remaining_budget)
    ):
        current = _deterministic_roll_once(
            history, current.state_id, model, config
        )
        new_state_ids.append(int(current.state_id))
        suffix_steps += 1
    if current.terminal_reason is not None:
        reason = current.terminal_reason
    elif suffix_steps >= config["max_rollout_steps"]:
        reason = "timeout"
    else:
        # Only the global mechanics budget can cause this censored outcome.
        reason = None
    return current, int(suffix_steps), reason


def _execute_macro(history, candidate, remaining_budget, model, config):
    """Execute K persistent perturbations and one deterministic suffix."""
    current = history[int(candidate["parent_id"])]
    displacement = np.asarray(candidate["v1"], dtype=float).copy()
    new_ids = []
    perturbation_ids = []
    projection_ratios = []
    proposed_norms = []
    corrected_norms = []
    used_steps = 0
    stop_reason = None

    for chain_index in range(int(candidate["k"])):
        if current.terminal_reason is not None:
            stop_reason = current.terminal_reason
            break
        if used_steps >= int(remaining_budget):
            break
        if chain_index > 0:
            displacement, ratio = _transport_previous(
                current.node, displacement, model, config
            )
            projection_ratios.append(float(ratio))
            if displacement is None:
                stop_reason = "projection_failure"
                break

        parent_node = np.asarray(current.node, dtype=float).copy()
        proposed_norms.append(float(np.linalg.norm(displacement)))
        current = _perturbation_step(
            history, current.state_id, displacement, model, config
        )
        corrected_norms.append(float(np.linalg.norm(current.node - parent_node)))
        new_ids.append(int(current.state_id))
        perturbation_ids.append(int(current.state_id))
        used_steps += 1
        if current.terminal_reason is not None:
            stop_reason = current.terminal_reason
            break

    chain_completed = len(perturbation_ids) == int(candidate["k"])
    suffix_steps = 0
    if stop_reason is None and chain_completed:
        current, suffix_steps, stop_reason = _complete_independent_suffix(
            history,
            current,
            new_ids,
            int(remaining_budget) - used_steps,
            model,
            config,
        )
        used_steps += suffix_steps

    return {
        "terminal": current,
        "new_ids": new_ids,
        "perturbation_ids": perturbation_ids,
        "used_steps": int(used_steps),
        "suffix_steps": int(suffix_steps),
        "stop_reason": stop_reason,
        "requested_k": int(candidate["k"]),
        "executed_k": int(len(perturbation_ids)),
        "chain_completed": bool(chain_completed),
        "projection_ratios": projection_ratios,
        "proposed_norms": proposed_norms,
        "corrected_norms": corrected_norms,
    }


def _new_training_data(output_dimension):
    return {
        "inputs": [],
        "flat_labels": [],
        "flatness_targets": [],
        "descriptor_targets": [],
        "requested_k": [],
        "executed_k": [],
        "rollout_ids": [],
        "stop_reasons": [],
        "cost_inputs": [],
        "log_cost_targets": [],
        "cost_rollout_ids": [],
        "output_dimension": int(output_dimension),
    }


def _append_macro_teacher(
    data,
    history,
    selected_input,
    requested_k,
    executed_k,
    all_macro_state_ids,
    stop_reason,
    rollout_id,
    config,
    output_spec,
):
    """Append one endpoint teacher and one finite-cost teacher per macro."""
    completed_reasons = {
        "flat",
        "collision",
        "timeout",
        "corrector_failure",
        "projection_failure",
    }
    if stop_reason not in completed_reasons or not all_macro_state_ids:
        # ``None`` means that the global budget right-censored this macro.
        return False

    data["cost_inputs"].append(np.asarray(selected_input, dtype=float).copy())
    data["log_cost_targets"].append(
        float(np.log1p(len(all_macro_state_ids)))
    )
    data["cost_rollout_ids"].append(int(rollout_id))

    potentials = np.asarray(
        [history[state_id].potential for state_id in all_macro_state_ids],
        dtype=float,
    )
    flatness = float(
        np.log1p(
            np.min(potentials)
            / max(float(config["flatness_tolerance"]), 1.0e-12)
        )
    )
    if stop_reason == "flat":
        descriptor = terminal_target_from_history(
            history,
            all_macro_state_ids[-1],
            config["layer_threshold"],
            output_spec,
        )
    else:
        descriptor = np.full(data["output_dimension"], np.nan, dtype=float)

    data["inputs"].append(np.asarray(selected_input, dtype=float).copy())
    data["flat_labels"].append(stop_reason == "flat")
    data["flatness_targets"].append(flatness)
    data["descriptor_targets"].append(descriptor)
    data["requested_k"].append(int(requested_k))
    data["executed_k"].append(int(executed_k))
    data["rollout_ids"].append(int(rollout_id))
    data["stop_reasons"].append(str(stop_reason))
    return True


def _uses_gp(config):
    return config["selector"].startswith("variable_k_gp")


def _fit_models(data, model, config):
    inputs = np.asarray(data["inputs"], dtype=float).reshape(
        -1, model["input_dimension"]
    )
    descriptors = np.asarray(data["descriptor_targets"], dtype=float).reshape(
        -1, data["output_dimension"]
    )
    surrogate = fit_surrogate_models(
        inputs,
        np.asarray(data["flat_labels"], dtype=bool),
        np.asarray(data["flatness_targets"], dtype=float),
        descriptors,
        state_dimension=model["input_state_dimension"],
        state_length_scale=config["state_length_scale"],
        action_length_scale=config["action_length_scale"],
        signal_variance=config["signal_variance"],
        regression_noise_std=config["regression_noise_std"],
        classifier_noise_std=config["classifier_noise_std"],
        descriptor_scale_floor=config["descriptor_scale_floor"],
        prior_flat_probability=0.5,
    )
    cost_inputs = np.asarray(data["cost_inputs"], dtype=float).reshape(
        -1, model["input_dimension"]
    )
    if len(cost_inputs) > 0:
        surrogate["cost"] = fit_shared_kernel_gp(
            cost_inputs,
            np.asarray(data["log_cost_targets"], dtype=float).reshape(-1, 1),
            state_dimension=model["input_state_dimension"],
            state_length_scale=config["state_length_scale"],
            action_length_scale=config["action_length_scale"],
            signal_variance=config["signal_variance"],
            noise_std=config["regression_noise_std"],
            output_scale_floor=config["cost_output_scale_floor"],
        )
    else:
        surrogate["cost"] = None
    return surrogate


def _select_candidate(
    candidates,
    surrogate,
    elite_archive,
    output_spec,
    model,
    config,
    selection_rng,
    shuffle_rng,
    posterior_rng,
):
    """Select one candidate and return acquisition diagnostics."""
    if config["selector"] == "variable_k_random":
        return int(selection_rng.integers(len(candidates))), None, 1, len(candidates)
    if surrogate is None:
        return int(selection_rng.integers(len(candidates))), None, 5, len(candidates)

    query = np.asarray([candidate["input"] for candidate in candidates])
    diagnostics = score_acquisition(
        surrogate,
        query,
        elite_archive,
        output_spec,
        model["member"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
        config["rms_tolerance"],
        config["max_tolerance"],
        flatness_target_threshold=np.log(2.0),
        acquisition_kind="ucb",
        novelty_kappa=config["novelty_kappa"],
        reach_weight=config["reach_weight"],
        cold_uncertainty_weight=config["cold_uncertainty_weight"],
        layer_uncertainty_scale=config["layer_uncertainty_scale"],
        num_posterior_samples=config["posterior_samples"],
        rng=posterior_rng,
    )
    diagnostics["selection_score"] = np.asarray(
        diagnostics["score"], dtype=float
    ).copy()
    if surrogate.get("cost") is not None:
        cost_mean, cost_variance = predict_shared_kernel_gp(
            surrogate["cost"], query
        )
        diagnostics["cost_log_mean"] = cost_mean[:, 0]
        diagnostics["cost_log_std"] = np.sqrt(
            np.maximum(cost_variance[:, 0], 0.0)
        )
        diagnostics["expected_cost_steps"] = np.maximum(
            np.exp(cost_mean[:, 0] + 0.5 * cost_variance[:, 0]) - 1.0,
            0.0,
        )

    if config["selector"].endswith("_shuffled"):
        if config["selector"].startswith("variable_k_gp_group_"):
            # Destroy only the score-to-K correspondence.  Scores never move
            # between different (parent, v1) groups, so the negative control
            # preserves exactly the same hierarchical search restriction.
            shuffled_score = diagnostics["selection_score"].copy()
            for group_id in sorted(
                {candidate["group_id"] for candidate in candidates}
            ):
                group_indices = np.asarray(
                    [
                        index
                        for index, candidate in enumerate(candidates)
                        if candidate["group_id"] == group_id
                    ],
                    dtype=int,
                )
                shuffled_score[group_indices] = shuffle_rng.permutation(
                    shuffled_score[group_indices]
                )
            diagnostics["selection_score"] = shuffled_score
        else:
            # Keep the original global shuffled controls byte-for-byte
            # equivalent to main8 v2.
            diagnostics["selection_score"] = shuffle_rng.permutation(
                diagnostics["selection_score"]
            )

    if selection_rng.random() < config["epsilon_random"]:
        return (
            int(selection_rng.integers(len(candidates))),
            diagnostics,
            5,
            len(candidates),
        )

    scores = np.nan_to_num(
        diagnostics["selection_score"], nan=-np.inf, neginf=-np.inf
    )
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) == 0:
        return (
            int(selection_rng.integers(len(candidates))),
            diagnostics,
            5,
            len(candidates),
        )

    if config["selector"].startswith("variable_k_gp_group_"):
        # Parent/v1 exploration remains exactly uniform.  GP selection
        # pressure acts only on K after one group has been chosen.
        group_ids = np.asarray(
            sorted({candidate["group_id"] for candidate in candidates}),
            dtype=int,
        )
        chosen_group = int(selection_rng.choice(group_ids))
        group_indices = np.asarray(
            [
                index
                for index, candidate in enumerate(candidates)
                if candidate["group_id"] == chosen_group
            ],
            dtype=int,
        )
        finite_group = group_indices[np.isfinite(scores[group_indices])]
        if len(finite_group) == 0:
            return (
                int(selection_rng.choice(group_indices)),
                diagnostics,
                5,
                len(group_indices),
            )

        if "group_argmax" in config["selector"]:
            maximum = float(np.max(scores[finite_group]))
            tied = finite_group[
                np.isclose(
                    scores[finite_group],
                    maximum,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ]
            code = 11 if config["selector"].endswith("_shuffled") else 10
            return int(selection_rng.choice(tied)), diagnostics, code, 1

        keep_count = max(
            1,
            int(
                np.ceil(
                    config["filter_keep_fraction"] * len(group_indices)
                )
            ),
        )
        group_order = np.argsort(
            -scores[group_indices], kind="stable"
        )[:keep_count]
        retained = group_indices[group_order]
        code = 13 if config["selector"].endswith("_shuffled") else 12
        return (
            int(selection_rng.choice(retained)),
            diagnostics,
            code,
            keep_count,
        )

    if config["selector"] in (
        "variable_k_gp_argmax",
        "variable_k_gp_argmax_shuffled",
    ):
        maximum = float(np.max(scores[finite]))
        tied = finite[np.isclose(scores[finite], maximum, rtol=1.0e-12, atol=1.0e-12)]
        code = 9 if config["selector"].endswith("_shuffled") else 6
        return int(selection_rng.choice(tied)), diagnostics, code, 1

    keep_count = max(
        1, int(np.ceil(config["filter_keep_fraction"] * len(candidates)))
    )
    top_indices = np.argsort(-scores, kind="stable")[:keep_count]
    code = 8 if config["selector"].endswith("_shuffled") else 7
    return int(selection_rng.choice(top_indices)), diagnostics, code, keep_count


def _diagnostic_values(diagnostics, selected_index):
    names = {
        "acquisition": "selection_score",
        "raw_acquisition": "score",
        "p_flat": "flat_probability",
        "p_reach": "reach_probability",
        "mean_novelty": "mean_novelty",
        "descriptor_uncertainty": "descriptor_uncertainty",
        "cost_log_mean": "cost_log_mean",
        "cost_log_std": "cost_log_std",
        "expected_cost_steps": "expected_cost_steps",
    }
    result = {}
    for output_name, source_name in names.items():
        if diagnostics is None or diagnostics.get(source_name) is None:
            result[output_name] = np.nan
        else:
            result[output_name] = float(diagnostics[source_name][selected_index])
    return result


def _mean_or_nan(values):
    return float(np.mean(values)) if values else np.nan


def _score_rank_fraction(diagnostics, selected_index):
    """Return 1/N for the best score and 1 for the worst score."""
    if diagnostics is None:
        return np.nan
    score = diagnostics.get("selection_score")
    if score is None:
        return np.nan
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    order = np.argsort(-score, kind="stable")
    position = np.flatnonzero(order == int(selected_index))
    if len(position) == 0:
        return np.nan
    return float((int(position[0]) + 1) / len(score))


def _rollout_row(
    rollout_id,
    rollout_start,
    used_steps,
    selected,
    result,
    is_new,
    selection_code,
    pool,
    retained_count,
    diagnostics,
    selected_index,
    training_size,
    warmup_group,
    warmup_role,
):
    return {
        "rollout_id": int(rollout_id),
        "start_step": int(rollout_start),
        "end_step": int(used_steps),
        "parent_id": int(selected["parent_id"]),
        "first_child_state_id": (
            -1 if not result["perturbation_ids"] else result["perturbation_ids"][0]
        ),
        "terminal_state_id": int(result["terminal"].state_id),
        "spent_steps": int(result["used_steps"]),
        "perturbation_steps": int(result["executed_k"]),
        "suffix_steps": int(result["suffix_steps"]),
        "requested_k": int(result["requested_k"]),
        "executed_k": int(result["executed_k"]),
        "chain_completed": int(result["chain_completed"]),
        "stop_reason": result["stop_reason"],
        "new_cell": int(is_new),
        "selection_code": int(selection_code),
        "group_id": int(selected["group_id"]),
        "direction_index": int(selected["direction_index"]),
        "candidate_pool_size": int(len(pool)),
        "candidate_parent_count": int(
            len({candidate["parent_id"] for candidate in pool})
        ),
        "candidate_group_count": int(
            len({candidate["group_id"] for candidate in pool})
        ),
        "candidate_k_count": int(len({candidate["k"] for candidate in pool})),
        "retained_candidate_count": int(retained_count),
        "selected_score_rank_fraction": _score_rank_fraction(
            diagnostics, selected_index
        ),
        "tangent_dimension": int(selected["tangent_dimension"]),
        "minimum_projection_ratio": (
            float(min(result["projection_ratios"]))
            if result["projection_ratios"]
            else np.nan
        ),
        "mean_projection_ratio": _mean_or_nan(result["projection_ratios"]),
        "mean_proposed_norm": _mean_or_nan(result["proposed_norms"]),
        "mean_corrected_norm": _mean_or_nan(result["corrected_norms"]),
        "training_rollouts": int(training_size),
        "warmup_group": int(warmup_group),
        "warmup_k_role": str(warmup_role),
        **_diagnostic_values(diagnostics, selected_index),
    }


def _training_hash(data):
    digest = hashlib.sha256()
    for key in (
        "inputs",
        "flat_labels",
        "flatness_targets",
        "descriptor_targets",
        "requested_k",
        "executed_k",
        "stop_reasons",
        "log_cost_targets",
    ):
        values = data[key]
        if key == "stop_reasons":
            digest.update("\0".join(map(str, values)).encode("utf-8"))
        else:
            digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_outputs(
    output_directory,
    history,
    elite_archive,
    discovery_curve,
    discovery_events,
    training_data,
    rollout_rows,
    config,
    model,
    summary,
):
    reset_output_directory(output_directory)
    save_elite_paths(history, elite_archive, output_directory)
    curve = list(discovery_curve)
    if curve[-1][0] != config["total_step_budget"]:
        curve.append((config["total_step_budget"], len(elite_archive)))
    save_discovery_curve(curve, output_directory)
    _write_csv(output_directory / "rollout_diagnostics.csv", rollout_rows)
    (output_directory / "discovery_events.json").write_text(
        json.dumps(discovery_events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_directory / "training_data.npz",
        inputs=np.asarray(training_data["inputs"], dtype=float).reshape(
            -1, model["input_dimension"]
        ),
        flat_labels=np.asarray(training_data["flat_labels"], dtype=np.int8),
        flatness_targets=np.asarray(
            training_data["flatness_targets"], dtype=float
        ),
        descriptor_targets=np.asarray(
            training_data["descriptor_targets"], dtype=float
        ).reshape(-1, training_data["output_dimension"]),
        requested_k=np.asarray(training_data["requested_k"], dtype=int),
        executed_k=np.asarray(training_data["executed_k"], dtype=int),
        rollout_ids=np.asarray(training_data["rollout_ids"], dtype=int),
        stop_reasons=np.asarray(training_data["stop_reasons"], dtype="U24"),
        cost_inputs=np.asarray(training_data["cost_inputs"], dtype=float).reshape(
            -1, model["input_dimension"]
        ),
        log_cost_targets=np.asarray(
            training_data["log_cost_targets"], dtype=float
        ),
        cost_rollout_ids=np.asarray(training_data["cost_rollout_ids"], dtype=int),
    )


def main():
    base_directory = Path(__file__).resolve().parent
    config = _configuration8()
    initial_node, model = _make_model8(base_directory, config)
    output_spec = make_terminal_output_spec(
        initial_node,
        model["index_free"],
        config["reference_point"],
        model["length_scale"],
    )
    history = create_history(
        initial_node,
        compute_gravity_potential(initial_node, config["reference_point"]),
    )

    seed = int(config["seed"])
    candidate_rng = np.random.default_rng(seed + 8101)
    selection_rng = np.random.default_rng(seed + 8202)
    warmup_rng = np.random.default_rng(seed + 8303)
    posterior_rng = np.random.default_rng(seed + 8404)
    shuffle_rng = np.random.default_rng(seed + 8505)
    np.random.seed(seed)

    elite_archive = []
    discovery_curve = [(0, 0)]
    discovery_events = []
    rollout_rows = []
    training_data = _new_training_data(len(output_spec["active_positions"]))
    surrogate = None
    completed_since_fit = 0
    completed_warmup_groups = 0
    completed_warmup_branches = 0
    used_steps = 0
    rollout_id = 0
    warmup_end_step = None
    warmup_end_cells = None
    warmup_prefix_hash = None
    warmup_training_hash = None
    fit_time = 0.0
    score_time = 0.0
    start_time = time.perf_counter()

    trunk_terminal, _, spent, trunk_reason = _initial_deterministic_trunk(
        history, config["total_step_budget"], model, config
    )
    used_steps += spent
    if trunk_terminal is not None and trunk_reason == "flat":
        if _update_archive(
            elite_archive,
            history,
            trunk_terminal.state_id,
            model,
            config,
        ):
            discovery_curve.append((used_steps, len(elite_archive)))
            discovery_events.append(
                _discovery_event(
                    history,
                    trunk_terminal.state_id,
                    used_steps,
                    len(elite_archive) - 1,
                    0,
                )
            )
    trunk_end_step = int(used_steps)
    trunk_cell_count = int(len(elite_archive))
    if config["paired_warmup_groups"] == 0:
        warmup_end_step = int(used_steps)
        warmup_end_cells = int(len(elite_archive))
        warmup_prefix_hash = _prefix_hash(history, elite_archive)
        warmup_training_hash = _training_hash(training_data)

    while used_steps < config["total_step_budget"]:
        pool = _candidate_pool(history, model, config, candidate_rng)
        if not pool:
            break

        if completed_warmup_groups < config["paired_warmup_groups"]:
            group_ids = sorted({candidate["group_id"] for candidate in pool})
            group_id = int(warmup_rng.choice(group_ids))
            targets = [
                (target_k, f"anchor_{target_k}", 30 + index)
                for index, target_k in enumerate(config["warmup_k_values"])
            ]
            branch_indices = []
            for target_k, role, code in targets:
                index = _nearest_k_in_group(pool, group_id, target_k)
                if index is not None and index not in [item[0] for item in branch_indices]:
                    branch_indices.append((index, role, code))

            completed_all = True
            for selected_index, warmup_role, selection_code in branch_indices:
                if used_steps >= config["total_step_budget"]:
                    completed_all = False
                    break
                selected = pool[selected_index]
                rollout_start = int(used_steps)
                result = _execute_macro(
                    history,
                    selected,
                    config["total_step_budget"] - used_steps,
                    model,
                    config,
                )
                used_steps += result["used_steps"]
                if result["stop_reason"] is None:
                    completed_all = False
                completed_warmup_branches += 1
                teacher_added = _append_macro_teacher(
                    training_data,
                    history,
                    selected["input"],
                    result["requested_k"],
                    result["executed_k"],
                    result["new_ids"],
                    result["stop_reason"],
                    rollout_id,
                    config,
                    output_spec,
                )
                completed_since_fit += int(teacher_added)

                is_new = False
                if result["stop_reason"] == "flat":
                    is_new = _update_archive(
                        elite_archive,
                        history,
                        result["terminal"].state_id,
                        model,
                        config,
                    )
                    if is_new:
                        discovery_curve.append((used_steps, len(elite_archive)))
                        discovery_events.append(
                            _discovery_event(
                                history,
                                result["terminal"].state_id,
                                used_steps,
                                len(elite_archive) - 1,
                                selection_code,
                            )
                        )
                rollout_rows.append(
                    _rollout_row(
                        rollout_id,
                        rollout_start,
                        used_steps,
                        selected,
                        result,
                        is_new,
                        selection_code,
                        pool,
                        1,
                        None,
                        selected_index,
                        len(training_data["inputs"]),
                        completed_warmup_groups,
                        warmup_role,
                    )
                )
                rollout_id += 1

            if completed_all and branch_indices:
                completed_warmup_groups += 1
                if completed_warmup_groups >= config["paired_warmup_groups"]:
                    warmup_end_step = int(used_steps)
                    warmup_end_cells = int(len(elite_archive))
                    warmup_prefix_hash = _prefix_hash(history, elite_archive)
                    warmup_training_hash = _training_hash(training_data)
            continue

        if (
            _uses_gp(config)
            and training_data["inputs"]
            and (
                surrogate is None
                or completed_since_fit >= config["refit_rollout_interval"]
            )
        ):
            fit_start = time.perf_counter()
            surrogate = _fit_models(training_data, model, config)
            fit_time += time.perf_counter() - fit_start
            completed_since_fit = 0

        score_start = time.perf_counter()
        (
            selected_index,
            diagnostics,
            selection_code,
            retained_count,
        ) = _select_candidate(
            pool,
            surrogate,
            elite_archive,
            output_spec,
            model,
            config,
            selection_rng,
            shuffle_rng,
            posterior_rng,
        )
        score_time += time.perf_counter() - score_start
        selected = pool[selected_index]
        rollout_start = int(used_steps)
        result = _execute_macro(
            history,
            selected,
            config["total_step_budget"] - used_steps,
            model,
            config,
        )
        used_steps += result["used_steps"]
        teacher_added = _append_macro_teacher(
            training_data,
            history,
            selected["input"],
            result["requested_k"],
            result["executed_k"],
            result["new_ids"],
            result["stop_reason"],
            rollout_id,
            config,
            output_spec,
        )
        completed_since_fit += int(teacher_added)

        is_new = False
        if result["stop_reason"] == "flat":
            is_new = _update_archive(
                elite_archive,
                history,
                result["terminal"].state_id,
                model,
                config,
            )
            if is_new:
                discovery_curve.append((used_steps, len(elite_archive)))
                discovery_events.append(
                    _discovery_event(
                        history,
                        result["terminal"].state_id,
                        used_steps,
                        len(elite_archive) - 1,
                        selection_code,
                    )
                )

        rollout_rows.append(
            _rollout_row(
                rollout_id,
                rollout_start,
                used_steps,
                selected,
                result,
                is_new,
                selection_code,
                pool,
                retained_count,
                diagnostics,
                selected_index,
                len(training_data["inputs"]),
                -1,
                "none",
            )
        )
        rollout_id += 1
        if config["verbose"] and (
            is_new or used_steps >= config["total_step_budget"]
        ):
            print(
                f"steps={used_steps}, cells={len(elite_archive)}, "
                f"selector={config['selector']}, parent={selected['parent_id']}, "
                f"K={selected['k']}"
            )

    if warmup_end_step is None:
        warmup_end_step = int(used_steps)
        warmup_end_cells = int(len(elite_archive))
        warmup_prefix_hash = _prefix_hash(history, elite_archive)
        warmup_training_hash = _training_hash(training_data)

    wall_time = time.perf_counter() - start_time
    stop_reasons = [row["stop_reason"] for row in rollout_rows]
    selection_counts = Counter(row["selection_code"] for row in rollout_rows)
    k_requested = [row["requested_k"] for row in rollout_rows]
    k_executed = [row["executed_k"] for row in rollout_rows]
    guided_rows = [row for row in rollout_rows if row["warmup_group"] < 0]
    summary = {
        "algorithm_version": config["algorithm_version"],
        "selector": config["selector"],
        "seed": seed,
        "step_budget": int(config["total_step_budget"]),
        "used_steps": int(used_steps),
        "elite_cells": int(len(elite_archive)),
        "discovery_auc": float(
            _discovery_auc(discovery_curve, config["total_step_budget"])
        ),
        "trunk_end_step": trunk_end_step,
        "trunk_cell_count": trunk_cell_count,
        "trunk_stop_reason": trunk_reason,
        "num_states": int(len(history)),
        "num_rollouts": int(len(rollout_rows)),
        "training_rollouts": int(len(training_data["inputs"])),
        "cost_training_rollouts": int(len(training_data["cost_inputs"])),
        "flat_rollouts": int(sum(reason == "flat" for reason in stop_reasons)),
        "collision_rollouts": int(
            sum(reason == "collision" for reason in stop_reasons)
        ),
        "timeout_rollouts": int(sum(reason == "timeout" for reason in stop_reasons)),
        "corrector_failure_rollouts": int(
            sum(reason == "corrector_failure" for reason in stop_reasons)
        ),
        "projection_failure_rollouts": int(
            sum(reason == "projection_failure" for reason in stop_reasons)
        ),
        "censored_rollouts": int(sum(reason is None for reason in stop_reasons)),
        "mean_requested_k": _mean_or_nan(k_requested),
        "mean_executed_k": _mean_or_nan(k_executed),
        "mean_guided_requested_k": _mean_or_nan(
            [row["requested_k"] for row in guided_rows]
        ),
        "mean_guided_executed_k": _mean_or_nan(
            [row["executed_k"] for row in guided_rows]
        ),
        "requested_k_counts": {
            str(k): int(count) for k, count in sorted(Counter(k_requested).items())
        },
        "discovery_requested_k": [
            int(row["requested_k"]) for row in rollout_rows if row["new_cell"]
        ],
        "selection_code_counts": {
            str(code): int(count)
            for code, count in sorted(selection_counts.items())
        },
        "selection_code_legend": {
            "0": "deterministic_trunk",
            "1": "variable_k_uniform_random",
            "5": "gp_epsilon_or_cold_random",
            "6": "gp_global_argmax",
            "7": "gp_top_fraction_filter",
            "8": "gp_shuffled_top_fraction_filter",
            "9": "gp_shuffled_global_argmax",
            "10": "gp_uniform_group_then_k_argmax",
            "11": "gp_uniform_group_then_shuffled_k_argmax",
            "12": "gp_uniform_group_then_k_filter",
            "13": "gp_uniform_group_then_shuffled_k_filter",
            **{
                str(30 + index): f"paired_warmup_anchor_{target_k}"
                for index, target_k in enumerate(config["warmup_k_values"])
            },
        },
        "paired_warmup_requested_groups": int(config["paired_warmup_groups"]),
        "paired_warmup_completed_groups": int(completed_warmup_groups),
        "paired_warmup_completed_branches": int(completed_warmup_branches),
        "warmup_completed": bool(
            completed_warmup_groups >= config["paired_warmup_groups"]
        ),
        "warmup_end_step": int(warmup_end_step),
        "warmup_end_cells": int(warmup_end_cells),
        "warmup_prefix_sha256": warmup_prefix_hash,
        "warmup_training_sha256": warmup_training_hash,
        "input_dimension": int(model["input_dimension"]),
        "input_state_dimension": int(model["input_state_dimension"]),
        "k_candidate_values": [int(value) for value in _k_values(config)],
        "fit_time_seconds": float(fit_time),
        "score_time_seconds": float(score_time),
        "wall_time_seconds": float(wall_time),
    }
    output_directory = (
        base_directory / "result" / config["result_name"] / f"seed_{seed:03d}"
    )
    _save_outputs(
        output_directory,
        history,
        elite_archive,
        discovery_curve,
        discovery_events,
        training_data,
        rollout_rows,
        config,
        model,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
