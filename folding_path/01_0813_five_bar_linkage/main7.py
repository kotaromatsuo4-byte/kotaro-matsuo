"""Joint K=2 macro selection before either perturbation is executed.

``main6.py`` can learn only the second angle after the first mechanics step
has already been paid.  This file moves the decision boundary one level
earlier.  Every candidate is a complete, still-unexecuted macro

    (parent state, first physical displacement v1, physical reference w,
     second angle theta).

The GP input contains only quantities available at that decision boundary:
the parent coordinates, a path-context state, ``v1``, ``w``, and
``cos(theta), sin(theta)``.  It contains no SVD coefficients.  The corrected
state q1, the transported frame (p, u), and v2 are generated deterministically
only after one candidate is selected.  Thus sign flips or rotations of an SVD
null-space basis do not change the stored representation.

Selectors share one joint candidate-pool construction:

``independent_random``
    Uniform sampling from all parent/v1/theta candidates.

``fixed_transport``
    Random parent/v1 and theta=0 with probability 0.8; uniform sampling from
    the whole pool with probability 0.2.

``joint_macro_gp`` / ``joint_macro_gp_shuffled``
    Novelty-UCB selection, or its score-permutation negative control.  The
    selected candidate is sampled from the top score fraction, and 20 percent
    of decisions remain uniformly random.

``joint_macro_gp_cost_aware`` and its shuffled control
    Divide the discovery acquisition by the predicted remaining number of
    mechanics steps.  Cost is learned as log(1+C), where C includes q1, q2,
    and the deterministic suffix.

All selectors use the same paired warmup by default.  For a common
(parent,v1,w) group, theta in {0, pi/2, pi, -pi/2} is evaluated.  All q1, q2,
and suffix steps are counted against the mechanics budget.  In mechanisms
with more than two tangent modes, each v1 receives one random physical
orthogonal reference w, so one random two-dimensional tangent slice is
searched per group; this approximation is explicitly reported.
"""

from __future__ import annotations

import csv
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from angle_macro import (
    angle_grid,
    circular_distance,
    sample_orthogonal_reference,
    sample_tangent_displacement,
    second_displacement,
    tangent_basis,
    transported_frame,
)
from gp import (
    fit_shared_kernel_gp,
    fit_surrogate_models,
    make_terminal_output_spec,
    score_acquisition,
    terminal_target_from_history,
)
from main5 import (
    _configuration,
    _discovery_auc,
    _discovery_event,
    _initial_deterministic_trunk,
    _load_model,
    _update_archive,
)
from main6 import (
    _complete_deterministic_suffix,
    _cost_adjusted_acquisition,
    _perturbation_step,
    _prefix_hash,
)
from mechanics import compute_gravity_potential
from result import reset_output_directory, save_discovery_curve, save_elite_paths
from state import create_history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _configuration7():
    selector = os.environ.get("FOLDING_MAIN7_SELECTOR", "joint_macro_gp")
    allowed = (
        "independent_random",
        "fixed_transport",
        "joint_macro_gp",
        "joint_macro_gp_shuffled",
        "joint_macro_gp_cost_aware",
        "joint_macro_gp_cost_aware_shuffled",
    )
    if selector not in allowed:
        raise ValueError(
            "FOLDING_MAIN7_SELECTOR must be independent_random, "
            "fixed_transport, joint_macro_gp, joint_macro_gp_shuffled, "
            "joint_macro_gp_cost_aware, or "
            "joint_macro_gp_cost_aware_shuffled"
        )

    # Reuse the physical, archive, and GP defaults of main5.  main7-specific
    # environment names below avoid changing main5/main6 behavior.
    config = _configuration()
    config.update(
        {
            "algorithm_version": "main7_joint_k2_macro_gp_v1",
            "selector": selector,
            "num_angles": _env_int("FOLDING_MAIN7_NUM_ANGLES", 16),
            "candidate_parent_limit": _env_int(
                "FOLDING_MAIN7_PARENT_LIMIT", 16
            ),
            "first_directions_per_parent": _env_int(
                "FOLDING_MAIN7_FIRST_DIRECTIONS", 2
            ),
            "paired_warmup_groups": _env_int(
                "FOLDING_MAIN7_PAIRED_WARMUP_GROUPS", 4
            ),
            "epsilon_random": _env_float("FOLDING_MAIN7_EPSILON", 0.20),
            "top_fraction": _env_float(
                "FOLDING_MAIN7_TOP_FRACTION", 0.25
            ),
            "refit_rollout_interval": _env_int(
                "FOLDING_MAIN7_REFIT_INTERVAL", 3
            ),
            "minimum_projection_ratio": _env_float(
                "FOLDING_MAIN7_MIN_PROJECTION_RATIO", 0.10
            ),
            "state_length_scale": _env_float(
                "FOLDING_MAIN7_GP_STATE_LENGTH", 0.50
            ),
            "action_length_scale": _env_float(
                "FOLDING_MAIN7_GP_ACTION_LENGTH", 0.50
            ),
            "cost_denominator": os.environ.get(
                "FOLDING_MAIN7_COST_DENOMINATOR", "expected"
            ).lower(),
            "cost_lcb_kappa": _env_float(
                "FOLDING_MAIN7_COST_LCB_KAPPA", 1.0
            ),
            "cost_floor_steps": _env_float(
                "FOLDING_MAIN7_COST_FLOOR_STEPS", 2.0
            ),
            "cost_output_scale_floor": _env_float(
                "FOLDING_MAIN7_COST_SCALE_FLOOR", 0.05
            ),
            "result_name": os.environ.get(
                "FOLDING_RESULT_NAME", f"main7_{selector}"
            ),
        }
    )

    if config["num_angles"] < 2:
        raise ValueError("FOLDING_MAIN7_NUM_ANGLES must be at least 2")
    if config["candidate_parent_limit"] <= 0:
        raise ValueError("FOLDING_MAIN7_PARENT_LIMIT must be positive")
    if config["first_directions_per_parent"] <= 0:
        raise ValueError("FOLDING_MAIN7_FIRST_DIRECTIONS must be positive")
    if config["paired_warmup_groups"] < 0:
        raise ValueError("FOLDING_MAIN7_PAIRED_WARMUP_GROUPS must be nonnegative")
    if not 0.0 <= config["epsilon_random"] <= 1.0:
        raise ValueError("FOLDING_MAIN7_EPSILON must be in [0, 1]")
    if not 0.0 < config["top_fraction"] <= 1.0:
        raise ValueError("FOLDING_MAIN7_TOP_FRACTION must be in (0, 1]")
    if config["refit_rollout_interval"] <= 0:
        raise ValueError("FOLDING_MAIN7_REFIT_INTERVAL must be positive")
    if config["cost_denominator"] not in ("expected", "lcb"):
        raise ValueError("FOLDING_MAIN7_COST_DENOMINATOR must be expected or lcb")
    if config["cost_lcb_kappa"] < 0.0:
        raise ValueError("FOLDING_MAIN7_COST_LCB_KAPPA must be nonnegative")
    if config["cost_floor_steps"] <= 0.0:
        raise ValueError("FOLDING_MAIN7_COST_FLOOR_STEPS must be positive")
    return config


def _make_model7(base_directory, config):
    node, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    free_dimension = int(len(index_free))
    length_scale = float(np.mean(member_length))
    # State block: parent q + last-nonflat path context.
    # Action block: physical v1 + physical w + cos/sin + w mask.
    input_dimension = 4 * free_dimension + 3
    return node, {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": int(len(node)),
        "num_member": int(len(member)),
        "length_scale": length_scale,
        "input_state_dimension": 2 * free_dimension,
        "input_dimension": input_dimension,
        "free_dimension": free_dimension,
    }


def _centered_free_coordinates(node, model, config):
    centered = np.asarray(node, dtype=float).reshape(-1, 3).copy()
    centered[:, 2] -= float(config["reference_point"])
    return (
        centered.reshape(-1)[model["index_free"]] / model["length_scale"]
    )


def _last_nonflat_context(history, parent_id, threshold):
    """Return the latest ancestor whose potential is above ``threshold``.

    A parent already above the threshold is its own context.  For a near-flat
    parent, the walk supplies the last clearly non-flat geometry on the same
    tree path.  This is path context, not a future terminal observation.
    """
    state = history[int(parent_id)]
    while state.potential < float(threshold) and state.parent_id is not None:
        state = history[int(state.parent_id)]
    return state.node


def _joint_macro_input(
    parent_node,
    context_node,
    first_displacement,
    physical_reference,
    theta,
    model,
    config,
):
    """Encode an unexecuted macro using only physical, basis-free features."""
    q_part = _centered_free_coordinates(parent_node, model, config)
    context_part = _centered_free_coordinates(context_node, model, config)
    free = model["index_free"]
    scale = model["length_scale"]
    v1_part = np.asarray(first_displacement).reshape(-1)[free] / scale
    if physical_reference is None:
        w_part = np.zeros(model["free_dimension"], dtype=float)
        w_available = 0.0
    else:
        w_part = np.asarray(physical_reference).reshape(-1)[free] / scale
        w_available = 1.0
    return np.concatenate(
        (
            q_part,
            context_part,
            v1_part,
            w_part,
            np.asarray([np.cos(theta), np.sin(theta), w_available]),
        )
    )


def _joint_candidate_pool(history, model, config, rng):
    """Generate one common bounded pool before any candidate mechanics runs."""
    valid = np.asarray(
        [state.state_id for state in history if state.terminal_reason is None],
        dtype=int,
    )
    if len(valid) == 0:
        return []
    count = min(len(valid), config["candidate_parent_limit"])
    if count < len(valid):
        parents = rng.choice(valid, size=count, replace=False)
    else:
        # Random permutation prevents the history insertion order from being
        # a hidden tie breaker when the pool is not yet full.
        parents = rng.permutation(valid)

    candidates = []
    group_id = 0
    for parent_id_value in parents:
        parent_id = int(parent_id_value)
        parent_node = history[parent_id].node
        context_node = _last_nonflat_context(
            history, parent_id, config["layer_threshold"]
        )
        tangent_dimension = int(tangent_basis(parent_node, model).shape[1])
        if tangent_dimension == 0:
            continue
        for _ in range(config["first_directions_per_parent"]):
            v1 = sample_tangent_displacement(
                parent_node,
                model,
                config["perturbation_scale"],
                rng,
            )
            if v1 is None:
                continue
            w = sample_orthogonal_reference(
                parent_node,
                v1,
                model,
                config["perturbation_scale"],
                rng,
            )
            # If constructing an independent physical reference fails, only
            # the two valid collinear directions are admitted.  This avoids
            # filling the common pool with actions that cannot define v2.
            angle_dimension = tangent_dimension if w is not None else 1
            for angle_index, theta in enumerate(
                angle_grid(config["num_angles"], angle_dimension)
            ):
                candidates.append(
                    {
                        "group_id": int(group_id),
                        "parent_id": parent_id,
                        "v1": np.asarray(v1, dtype=float).copy(),
                        "w": None if w is None else np.asarray(w, dtype=float).copy(),
                        "theta": float(theta),
                        "angle_index": int(angle_index),
                        "tangent_dimension": tangent_dimension,
                        "input": _joint_macro_input(
                            parent_node,
                            context_node,
                            v1,
                            w,
                            theta,
                            model,
                            config,
                        ),
                    }
                )
            group_id += 1
    return candidates


def _nearest_in_group(candidates, group_id, target_theta):
    indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["group_id"] == int(group_id)
    ]
    if not indices:
        return None
    distances = [
        circular_distance(candidates[index]["theta"], target_theta)
        for index in indices
    ]
    return int(indices[int(np.argmin(distances))])


def _transport_indices(candidates):
    groups = sorted({candidate["group_id"] for candidate in candidates})
    return np.asarray(
        [_nearest_in_group(candidates, group, 0.0) for group in groups],
        dtype=int,
    )


def _new_training_data(output_dimension):
    return {
        "inputs": [],
        "flat_labels": [],
        "flatness_targets": [],
        "descriptor_targets": [],
        "q2_executed": [],
        "rollout_ids": [],
        "stop_reasons": [],
        "theta": [],
        "cost_inputs": [],
        "log_cost_targets": [],
        "cost_q2_executed": [],
        "cost_rollout_ids": [],
        "output_dimension": int(output_dimension),
    }


def _append_macro_teacher(
    data,
    history,
    selected_input,
    theta,
    all_macro_state_ids,
    q2_executed,
    stop_reason,
    rollout_id,
    config,
    output_spec,
):
    """Append at most one row for one selected macro decision.

    A first perturbation can itself reach flat/collision, and an unavailable
    transported frame can end in a deterministic flat/collision suffix.  Both
    are deterministic outcomes of the selected ``(parent,v1,w,theta)`` and
    must teach the joint selector even though q2 was not executed.  Otherwise
    an invalid first action remains invisible and may be repeatedly selected.

    Flat/collision rows train the endpoint models. A known finite cost trains
    the cost GP. A rollout cut only by the global budget is censored and is
    not mislabeled as cheap. ``q2_executed`` is saved as an audit label rather
    than folded into the flat/collision label.
    """
    if not all_macro_state_ids:
        return False, False
    cost_added = False
    if stop_reason in ("flat", "collision", "timeout", "corrector_failure"):
        data["cost_inputs"].append(np.asarray(selected_input).copy())
        data["log_cost_targets"].append(float(np.log1p(len(all_macro_state_ids))))
        data["cost_q2_executed"].append(bool(q2_executed))
        data["cost_rollout_ids"].append(int(rollout_id))
        cost_added = True
    if stop_reason not in ("flat", "collision"):
        return False, cost_added

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

    data["inputs"].append(np.asarray(selected_input).copy())
    data["flat_labels"].append(stop_reason == "flat")
    data["flatness_targets"].append(flatness)
    data["descriptor_targets"].append(descriptor)
    data["q2_executed"].append(bool(q2_executed))
    data["rollout_ids"].append(int(rollout_id))
    data["stop_reasons"].append(stop_reason)
    data["theta"].append(float(theta))
    return True, cost_added


def _uses_gp(config):
    return config["selector"].startswith("joint_macro_gp")


def _uses_cost_model(config):
    return "cost_aware" in config["selector"]


def _fit_joint_gp(data, model, config):
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
    if _uses_cost_model(config):
        cost_inputs = np.asarray(data["cost_inputs"], dtype=float).reshape(
            -1, model["input_dimension"]
        )
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
    """Select from the common pool and return diagnostics plus a code."""
    if config["selector"] == "independent_random":
        return int(selection_rng.integers(len(candidates))), None, 1

    if config["selector"] == "fixed_transport":
        if selection_rng.random() < config["epsilon_random"]:
            return int(selection_rng.integers(len(candidates))), None, 3
        choices = _transport_indices(candidates)
        return int(selection_rng.choice(choices)), None, 2

    if surrogate is None:
        return int(selection_rng.integers(len(candidates))), None, 5

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
    diagnostics["selection_score"] = diagnostics["score"].copy()
    if _uses_cost_model(config):
        diagnostics["selection_score"], cost_diagnostics = (
            _cost_adjusted_acquisition(
                surrogate,
                query,
                diagnostics["selection_score"],
                config,
            )
        )
        diagnostics.update(cost_diagnostics)

    if config["selector"].endswith("_shuffled"):
        diagnostics["selection_score"] = shuffle_rng.permutation(
            diagnostics["selection_score"]
        )

    if selection_rng.random() < config["epsilon_random"]:
        return int(selection_rng.integers(len(candidates))), diagnostics, 5

    finite_scores = np.nan_to_num(
        diagnostics["selection_score"], nan=-np.inf
    )
    keep_count = max(
        1, int(np.ceil(config["top_fraction"] * len(candidates)))
    )
    top_indices = np.argsort(-finite_scores, kind="stable")[:keep_count]
    selected_index = int(selection_rng.choice(top_indices))
    if config["selector"] == "joint_macro_gp":
        code = 6
    elif config["selector"] == "joint_macro_gp_shuffled":
        code = 7
    elif config["selector"] == "joint_macro_gp_cost_aware":
        code = 8
    else:
        code = 9
    return selected_index, diagnostics, code


def _execute_macro(history, candidate, remaining_budget, model, config):
    """Execute q1, deterministically construct q2, then finish the suffix."""
    parent_id = int(candidate["parent_id"])
    q1 = _perturbation_step(
        history, parent_id, candidate["v1"], model, config
    )
    new_ids = [int(q1.state_id)]
    used_steps = 1
    frame = None
    q2 = None

    if q1.terminal_reason is not None:
        return {
            "terminal": q1,
            "new_ids": new_ids,
            "used_steps": used_steps,
            "stop_reason": q1.terminal_reason,
            "q1": q1,
            "q2": None,
            "frame": None,
            "q2_executed": False,
        }
    if used_steps >= remaining_budget:
        return {
            "terminal": q1,
            "new_ids": new_ids,
            "used_steps": used_steps,
            "stop_reason": None,
            "q1": q1,
            "q2": None,
            "frame": None,
            "q2_executed": False,
        }

    # No random fallback is passed: (q, v1, w, theta) uniquely determines
    # q1, the transported frame, and v2.
    frame = transported_frame(
        q1.node,
        candidate["v1"],
        candidate["w"],
        model,
        config["perturbation_scale"],
        config["minimum_projection_ratio"],
        rng=None,
    )
    v2 = None if frame is None else second_displacement(
        frame, candidate["theta"], config["perturbation_scale"]
    )
    if v2 is None:
        terminal, suffix_spent, stop_reason = _complete_deterministic_suffix(
            history,
            q1,
            new_ids,
            remaining_budget - used_steps,
            model,
            config,
        )
        used_steps += suffix_spent
        return {
            "terminal": terminal,
            "new_ids": new_ids,
            "used_steps": used_steps,
            "stop_reason": stop_reason,
            "q1": q1,
            "q2": None,
            "frame": frame,
            "q2_executed": False,
        }

    q2 = _perturbation_step(history, q1.state_id, v2, model, config)
    new_ids.append(int(q2.state_id))
    used_steps += 1
    terminal, suffix_spent, stop_reason = _complete_deterministic_suffix(
        history,
        q2,
        new_ids,
        remaining_budget - used_steps,
        model,
        config,
    )
    used_steps += suffix_spent
    return {
        "terminal": terminal,
        "new_ids": new_ids,
        "used_steps": used_steps,
        "stop_reason": stop_reason,
        "q1": q1,
        "q2": q2,
        "frame": frame,
        "q2_executed": True,
    }


def _diagnostic_values(diagnostics, selected_index, pool_size):
    if diagnostics is None:
        return {
            "acquisition": np.nan,
            "p_flat": np.nan,
            "p_reach": np.nan,
            "mean_novelty": np.nan,
            "descriptor_uncertainty": np.nan,
            "cost_log_mean": np.nan,
            "cost_log_std": np.nan,
            "cost_denominator_steps": np.nan,
        }
    def value(name, fallback=np.nan):
        array = diagnostics.get(name)
        if array is None:
            return float(fallback)
        return float(array[selected_index])
    return {
        "acquisition": value("selection_score"),
        "p_flat": value("flat_probability"),
        "p_reach": value("reach_probability"),
        "mean_novelty": value("mean_novelty"),
        "descriptor_uncertainty": value("descriptor_uncertainty"),
        "cost_log_mean": value("cost_log_mean"),
        "cost_log_std": value("cost_log_std"),
        "cost_denominator_steps": value("cost_denominator_steps"),
    }


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
        q2_executed=np.asarray(training_data["q2_executed"], dtype=np.int8),
        rollout_ids=np.asarray(training_data["rollout_ids"], dtype=int),
        theta=np.asarray(training_data["theta"], dtype=float),
        stop_reasons=np.asarray(training_data["stop_reasons"], dtype="U24"),
        cost_inputs=np.asarray(
            training_data["cost_inputs"], dtype=float
        ).reshape(-1, model["input_dimension"]),
        log_cost_targets=np.asarray(
            training_data["log_cost_targets"], dtype=float
        ),
        cost_q2_executed=np.asarray(
            training_data["cost_q2_executed"], dtype=np.int8
        ),
        cost_rollout_ids=np.asarray(
            training_data["cost_rollout_ids"], dtype=int
        ),
    )


def main():
    base_directory = Path(__file__).resolve().parent
    config = _configuration7()
    initial_node, model = _make_model7(base_directory, config)
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
    candidate_rng = np.random.default_rng(seed + 7101)
    selection_rng = np.random.default_rng(seed + 7202)
    warmup_rng = np.random.default_rng(seed + 7303)
    posterior_rng = np.random.default_rng(seed + 7404)
    shuffle_rng = np.random.default_rng(seed + 7505)
    np.random.seed(seed)

    elite_archive = []
    discovery_curve = [(0, 0)]
    discovery_events = []
    rollout_rows = []
    training_data = _new_training_data(len(output_spec["active_positions"]))
    surrogate = None
    completed_since_fit = 0
    completed_k2_macros = 0
    completed_warmup_groups = 0
    completed_warmup_branches = 0
    completed_warmup_k2_branches = 0
    used_steps = 0
    rollout_id = 0
    warmup_end_step = None
    warmup_end_cells = None
    warmup_prefix_hash = None
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
        warmup_end_step = used_steps
        warmup_end_cells = len(elite_archive)
        warmup_prefix_hash = _prefix_hash(history, elite_archive)

    while used_steps < config["total_step_budget"]:
        pool = _joint_candidate_pool(history, model, config, candidate_rng)
        if not pool:
            break

        # A paired group is an exact common prefix for every selector.  The
        # four branches repeat the deterministic q1 mechanics from one parent;
        # this is intentional counterfactual data and every repeated step is
        # charged to the budget.
        if completed_warmup_groups < config["paired_warmup_groups"]:
            group_ids = sorted({candidate["group_id"] for candidate in pool})
            group_id = int(warmup_rng.choice(group_ids))
            targets = (
                (0.0, "theta_zero", 20),
                (0.5 * np.pi, "theta_plus_half_pi", 21),
                (np.pi, "theta_pi", 22),
                (-0.5 * np.pi, "theta_minus_half_pi", 23),
            )
            branch_indices = []
            for target, role, code in targets:
                index = _nearest_in_group(pool, group_id, target)
                if index is None:
                    continue
                # A 1-DOF grid maps orthogonal targets onto 0 or pi.  Keep
                # only distinct actual candidate indices in that case.
                if index not in [item[0] for item in branch_indices]:
                    branch_indices.append((index, role, code))

            completed_all = True
            for selected_index, angle_role, selection_code in branch_indices:
                if used_steps >= config["total_step_budget"]:
                    completed_all = False
                    break
                selected = pool[selected_index]
                rollout_start = used_steps
                result = _execute_macro(
                    history,
                    selected,
                    config["total_step_budget"] - used_steps,
                    model,
                    config,
                )
                used_steps += result["used_steps"]
                if result["stop_reason"] is None:
                    # The global budget censored this branch.  Its mechanics
                    # remain in the common prefix, but the paired group is not
                    # claimed as a completed calibration group.
                    completed_all = False
                completed_warmup_branches += 1
                if result["q2_executed"]:
                    completed_k2_macros += 1
                    completed_warmup_k2_branches += 1
                teacher_added, cost_added = _append_macro_teacher(
                    training_data,
                    history,
                    selected["input"],
                    selected["theta"],
                    result["new_ids"],
                    result["q2_executed"],
                    result["stop_reason"],
                    rollout_id,
                    config,
                    output_spec,
                )
                if teacher_added or (_uses_cost_model(config) and cost_added):
                    completed_since_fit += 1

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
                frame = result["frame"]
                rollout_rows.append(
                    {
                        "rollout_id": rollout_id,
                        "start_step": rollout_start,
                        "end_step": used_steps,
                        "parent_id": selected["parent_id"],
                        "q1_state_id": result["q1"].state_id,
                        "q2_state_id": (
                            -1 if result["q2"] is None else result["q2"].state_id
                        ),
                        "spent_steps": result["used_steps"],
                        "stop_reason": result["stop_reason"],
                        "new_cell": int(is_new),
                        "selection_code": selection_code,
                        "theta": selected["theta"],
                        "angle_index": selected["angle_index"],
                        "group_id": selected["group_id"],
                        "candidate_pool_size": len(pool),
                        "candidate_parent_count": len(
                            {candidate["parent_id"] for candidate in pool}
                        ),
                        "q2_executed": int(result["q2_executed"]),
                        "tangent_dimension": selected["tangent_dimension"],
                        "projection_ratio": (
                            np.nan if frame is None else frame["projection_ratio"]
                        ),
                        "training_rollouts": len(training_data["inputs"]),
                        "warmup_group": completed_warmup_groups,
                        "warmup_angle_role": angle_role,
                        **_diagnostic_values(None, selected_index, len(pool)),
                    }
                )
                rollout_id += 1

            if completed_all and len(branch_indices) > 0:
                completed_warmup_groups += 1
                if completed_warmup_groups >= config["paired_warmup_groups"]:
                    warmup_end_step = int(used_steps)
                    warmup_end_cells = int(len(elite_archive))
                    warmup_prefix_hash = _prefix_hash(history, elite_archive)
            continue

        if (
            _uses_gp(config)
            and len(training_data["inputs"]) > 0
            and (
                surrogate is None
                or completed_since_fit >= config["refit_rollout_interval"]
            )
        ):
            fit_start = time.perf_counter()
            surrogate = _fit_joint_gp(training_data, model, config)
            fit_time += time.perf_counter() - fit_start
            completed_since_fit = 0

        score_start = time.perf_counter()
        selected_index, diagnostics, selection_code = _select_candidate(
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
        rollout_start = used_steps
        result = _execute_macro(
            history,
            selected,
            config["total_step_budget"] - used_steps,
            model,
            config,
        )
        used_steps += result["used_steps"]
        if result["q2_executed"]:
            completed_k2_macros += 1
        teacher_added, cost_added = _append_macro_teacher(
            training_data,
            history,
            selected["input"],
            selected["theta"],
            result["new_ids"],
            result["q2_executed"],
            result["stop_reason"],
            rollout_id,
            config,
            output_spec,
        )
        if teacher_added or (_uses_cost_model(config) and cost_added):
            completed_since_fit += 1

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

        frame = result["frame"]
        row = {
            "rollout_id": rollout_id,
            "start_step": rollout_start,
            "end_step": used_steps,
            "parent_id": selected["parent_id"],
            "q1_state_id": result["q1"].state_id,
            "q2_state_id": -1 if result["q2"] is None else result["q2"].state_id,
            "spent_steps": result["used_steps"],
            "stop_reason": result["stop_reason"],
            "new_cell": int(is_new),
            "selection_code": selection_code,
            "theta": selected["theta"],
            "angle_index": selected["angle_index"],
            "group_id": selected["group_id"],
            "candidate_pool_size": len(pool),
            "candidate_parent_count": len(
                {candidate["parent_id"] for candidate in pool}
            ),
            "q2_executed": int(result["q2_executed"]),
            "tangent_dimension": selected["tangent_dimension"],
            "projection_ratio": (
                np.nan if frame is None else frame["projection_ratio"]
            ),
            "training_rollouts": len(training_data["inputs"]),
            "warmup_group": -1,
            "warmup_angle_role": "none",
            **_diagnostic_values(diagnostics, selected_index, len(pool)),
        }
        rollout_rows.append(row)
        rollout_id += 1
        if config["verbose"] and (is_new or used_steps >= config["total_step_budget"]):
            print(
                f"steps={used_steps}, cells={len(elite_archive)}, "
                f"selector={config['selector']}, parent={selected['parent_id']}, "
                f"theta={selected['theta']:.3f}"
            )

    if warmup_end_step is None:
        warmup_end_step = int(used_steps)
        warmup_end_cells = int(len(elite_archive))
        warmup_prefix_hash = _prefix_hash(history, elite_archive)

    wall_time = time.perf_counter() - start_time
    stop_reasons = [row["stop_reason"] for row in rollout_rows]
    selection_counts = Counter(row["selection_code"] for row in rollout_rows)
    maximum_tangent_dimension = max(
        (row["tangent_dimension"] for row in rollout_rows), default=0
    )
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
        "completed_k2_macros": int(completed_k2_macros),
        "training_rollouts": int(len(training_data["inputs"])),
        "cost_training_rollouts": int(len(training_data["cost_inputs"])),
        "flat_rollouts": int(sum(reason == "flat" for reason in stop_reasons)),
        "collision_rollouts": int(
            sum(reason == "collision" for reason in stop_reasons)
        ),
        "selection_code_counts": {
            str(code): int(count) for code, count in sorted(selection_counts.items())
        },
        "selection_code_legend": {
            "0": "deterministic_trunk",
            "1": "independent_random",
            "2": "fixed_theta_zero",
            "3": "fixed_epsilon_random",
            "5": "gp_epsilon_or_cold_random",
            "6": "joint_gp_top_fraction",
            "7": "joint_gp_shuffled_top_fraction",
            "8": "joint_gp_cost_aware_top_fraction",
            "9": "joint_gp_cost_aware_shuffled_top_fraction",
            "20": "paired_warmup_theta_zero",
            "21": "paired_warmup_theta_plus_half_pi",
            "22": "paired_warmup_theta_pi",
            "23": "paired_warmup_theta_minus_half_pi",
        },
        "paired_warmup_requested_groups": int(config["paired_warmup_groups"]),
        "paired_warmup_completed_groups": int(completed_warmup_groups),
        "paired_warmup_completed_branches": int(completed_warmup_branches),
        "paired_warmup_completed_k2_branches": int(
            completed_warmup_k2_branches
        ),
        "warmup_completed": bool(
            completed_warmup_groups >= config["paired_warmup_groups"]
        ),
        "warmup_end_step": int(warmup_end_step),
        "warmup_end_cells": int(warmup_end_cells),
        "warmup_prefix_sha256": warmup_prefix_hash,
        "input_dimension": int(model["input_dimension"]),
        "input_state_dimension": int(model["input_state_dimension"]),
        "maximum_tangent_dimension": int(maximum_tangent_dimension),
        "higher_dof_random_2d_slice_used": bool(maximum_tangent_dimension > 2),
        "fit_time_seconds": float(fit_time),
        "score_time_seconds": float(score_time),
        "wall_time_seconds": float(wall_time),
    }
    output_directory = (
        base_directory
        / "result"
        / config["result_name"]
        / f"seed_{seed:03d}"
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
