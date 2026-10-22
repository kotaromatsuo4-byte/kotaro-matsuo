"""K=2 angle-macro comparison with an identical balanced warmup prefix.

Every search rollout first applies a common random ``(parent, v1)`` rule.
After the corrector has produced ``q1``, only the second angle is selected:

``independent_random``
    Uniformly select one angle from the common balanced grid.

``fixed_transport``
    Select ``theta = 0`` with probability 0.8 and a uniform random angle with
    probability 0.2.

``fixed_reverse``
    Select ``theta = pi`` with probability 0.8 and a uniform random angle with
    probability 0.2.  This diagnostic separates the value of making two
    consecutive perturbations from the value of preserving their direction.

``learned_angle_gp``
    Fit a GP to one target per completed K=2 rollout and select an angle with
    the existing novelty UCB-like acquisition, retaining 20 percent explicit
    random exploration.

``learned_angle_gp_shuffled``
    Negative control: permute the same GP scores among the angle candidates
    before applying the otherwise identical selection rule.

``learned_angle_gp_cost_aware``
    Divide the novelty acquisition by a GP prediction of the remaining
    mechanics cost after ``q1``.  The cost target is ``log(1 + C)``, where
    ``C`` counts the second perturbation step and its deterministic suffix.

``learned_angle_gp_cost_aware_shuffled``
    Negative control: permute the final cost-adjusted scores among candidates.

By default all selectors execute the same balanced 16-angle warmup.
With the same seed their tree, archive, and mechanics-step count must be
identical until handoff.  The handoff hash saved in ``summary.json`` makes
that property directly auditable.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from angle_macro import (
    angle_grid,
    circular_distance,
    macro_gp_input,
    make_angle_candidates,
    sample_orthogonal_reference,
    sample_tangent_displacement,
    second_displacement,
    transported_frame,
)
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
from mechanics import compute_gravity_potential
from result import (
    reset_output_directory,
    save_discovery_curve,
    save_elite_paths,
)
from roll import displacement_roll
from state import create_history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _configuration6():
    selector = os.environ.get("FOLDING_MAIN6_SELECTOR", "learned_angle_gp")
    allowed = (
        "independent_random",
        "fixed_transport",
        "fixed_reverse",
        "learned_angle_gp",
        "learned_angle_gp_shuffled",
        "learned_angle_gp_cost_aware",
        "learned_angle_gp_cost_aware_shuffled",
    )
    if selector not in allowed:
        raise ValueError(
            "FOLDING_MAIN6_SELECTOR must be independent_random, "
            "fixed_transport, fixed_reverse, learned_angle_gp, "
            "learned_angle_gp_shuffled, learned_angle_gp_cost_aware, or "
            "learned_angle_gp_cost_aware_shuffled"
        )

    config = _configuration()
    config.update(
        {
            "algorithm_version": (
                "main6_k2_angle_gp_v1_2_cost_aware"
                if selector.startswith("learned_angle_gp_cost_aware")
                else "main6_k2_angle_gp_v1_1_top_fraction"
            ),
            "selector": selector,
            "num_angles": _env_int("FOLDING_MAIN6_NUM_ANGLES", 16),
            "warmup_rollouts": _env_int(
                "FOLDING_MAIN6_WARMUP_ROLLOUTS", 16
            ),
            "common_warmup": bool(
                _env_int("FOLDING_MAIN6_COMMON_WARMUP", 1)
            ),
            # Optional counterfactual warmup.  When enabled, one executed q1
            # is shared by three fully evaluated q2 branches (0, +/-pi/2, pi).
            # It is disabled by default so legacy selector trajectories and
            # results remain unchanged.
            "paired_warmup": bool(
                _env_int("FOLDING_MAIN6_PAIRED_WARMUP", 0)
            ),
            "paired_warmup_groups": _env_int(
                "FOLDING_MAIN6_PAIRED_WARMUP_GROUPS", 4
            ),
            "epsilon_random": _env_float("FOLDING_MAIN6_EPSILON", 0.20),
            "top_fraction": _env_float(
                "FOLDING_MAIN6_TOP_FRACTION", 0.25
            ),
            "refit_rollout_interval": _env_int(
                "FOLDING_MAIN6_REFIT_INTERVAL", 3
            ),
            "minimum_projection_ratio": _env_float(
                "FOLDING_MAIN6_MIN_PROJECTION_RATIO", 0.10
            ),
            "state_length_scale": _env_float(
                "FOLDING_MAIN6_GP_STATE_LENGTH", 0.50
            ),
            "action_length_scale": _env_float(
                "FOLDING_MAIN6_GP_ACTION_LENGTH", 0.50
            ),
            # The default expected-cost denominator is conservative.  The
            # optional ``lcb`` denominator is optimistic in the standard
            # reward-UCB / cost-LCB sense, but may over-select candidates whose
            # cost is merely uncertain.
            "cost_denominator": os.environ.get(
                "FOLDING_MAIN6_COST_DENOMINATOR", "expected"
            ).lower(),
            "cost_lcb_kappa": _env_float(
                "FOLDING_MAIN6_COST_LCB_KAPPA", 1.0
            ),
            "cost_floor_steps": _env_float(
                "FOLDING_MAIN6_COST_FLOOR_STEPS", 1.0
            ),
            "cost_output_scale_floor": _env_float(
                "FOLDING_MAIN6_COST_SCALE_FLOOR", 0.05
            ),
            "max_first_action_attempts": _env_int(
                "FOLDING_MAIN6_FIRST_ACTION_ATTEMPTS", 100
            ),
            "result_name": os.environ.get(
                "FOLDING_RESULT_NAME", f"main6_{selector}"
            ),
        }
    )
    if config["num_angles"] < 2:
        raise ValueError("FOLDING_MAIN6_NUM_ANGLES must be at least 2")
    if config["warmup_rollouts"] < 0:
        raise ValueError("FOLDING_MAIN6_WARMUP_ROLLOUTS must be nonnegative")
    if config["paired_warmup_groups"] < 0:
        raise ValueError("FOLDING_MAIN6_PAIRED_WARMUP_GROUPS must be nonnegative")
    if not 0.0 <= config["epsilon_random"] <= 1.0:
        raise ValueError("FOLDING_MAIN6_EPSILON must be in [0, 1]")
    if not 0.0 < config["top_fraction"] <= 1.0:
        raise ValueError("FOLDING_MAIN6_TOP_FRACTION must be in (0, 1]")
    if config["refit_rollout_interval"] <= 0:
        raise ValueError("FOLDING_MAIN6_REFIT_INTERVAL must be positive")
    if config["cost_denominator"] not in ("expected", "lcb"):
        raise ValueError(
            "FOLDING_MAIN6_COST_DENOMINATOR must be expected or lcb"
        )
    if config["cost_lcb_kappa"] < 0.0:
        raise ValueError("FOLDING_MAIN6_COST_LCB_KAPPA must be nonnegative")
    if config["cost_floor_steps"] <= 0.0:
        raise ValueError("FOLDING_MAIN6_COST_FLOOR_STEPS must be positive")
    if config["paired_warmup"]:
        config["algorithm_version"] += "_paired_warmup"
    return config


def _make_model(base_directory, config):
    node, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    length_scale = float(np.mean(member_length))
    free_dimension = int(len(index_free))
    # q1 + transported-v1 + u + v2 + five scalar context features
    input_dimension = 4 * free_dimension + 5
    return node, {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": len(node),
        "num_member": len(member),
        "length_scale": length_scale,
        # Kernel context block = q1 + transported-v1 + explicit u.
        # Action block = v2 + angle/projection scalar features.
        "input_state_dimension": 3 * free_dimension,
        "input_dimension": int(input_dimension),
    }


def _perturbation_step(history, parent_id, displacement, model, config):
    return displacement_roll(
        history,
        int(parent_id),
        displacement,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        config["constraint_tolerance"],
        config["beta0"],
        config["corrector_iterations"],
        config["line_search_iterations"],
        config["line_search_tau"],
        config["reference_point"],
        config["flatness_tolerance"],
        config["dist_tol"],
        config["normal_tol"],
        config["cos_tol"],
        config["collision_model"],
    )


def _execute_common_first_action(history, model, config, parent_rng, action_rng):
    """Randomly choose a nonterminal parent and execute a random ``v1``."""
    for _ in range(config["max_first_action_attempts"]):
        valid = [
            state.state_id for state in history if state.terminal_reason is None
        ]
        if not valid:
            return None
        parent_id = int(parent_rng.choice(valid))
        v1 = sample_tangent_displacement(
            history[parent_id].node,
            model,
            config["perturbation_scale"],
            action_rng,
        )
        if v1 is None:
            continue
        w = sample_orthogonal_reference(
            history[parent_id].node,
            v1,
            model,
            config["perturbation_scale"],
            action_rng,
        )
        q1 = _perturbation_step(
            history,
            parent_id,
            v1,
            model,
            config,
        )
        return {
            "parent_id": parent_id,
            "q1_state_id": int(q1.state_id),
            "q1": q1,
            "v1": v1,
            "w": w,
        }
    return None


def _complete_deterministic_suffix(
    history,
    current,
    new_state_ids,
    remaining_budget,
    model,
    config,
):
    used = 0
    while (
        current.terminal_reason is None
        and len(new_state_ids) < config["max_rollout_steps"]
        and used < remaining_budget
    ):
        current = _deterministic_roll_once(
            history,
            current.state_id,
            model,
            config,
        )
        new_state_ids.append(current.state_id)
        used += 1
    if current.terminal_reason is not None:
        reason = current.terminal_reason
    elif len(new_state_ids) >= config["max_rollout_steps"]:
        reason = "timeout"
    else:
        reason = None
    return current, used, reason


def _new_training_data(output_dimension):
    return {
        "inputs": [],
        "flat_labels": [],
        "flatness_targets": [],
        "descriptor_targets": [],
        "rollout_ids": [],
        "stop_reasons": [],
        "theta": [],
        "cost_inputs": [],
        "log_cost_targets": [],
        "cost_rollout_ids": [],
        "output_dimension": int(output_dimension),
    }


def _append_macro_teacher(
    data,
    history,
    selected_input,
    theta,
    state_ids_after_second,
    stop_reason,
    rollout_id,
    config,
    output_spec,
):
    """Append exactly one teacher row for one completed second-step rollout."""
    if not state_ids_after_second:
        return False
    # Cost is known for a physical terminal or for the configured rollout
    # horizon.  A run cut only by the global budget has right-censored cost and
    # is deliberately omitted instead of being mislabeled as cheap.
    if stop_reason in ("flat", "collision", "timeout"):
        data["cost_inputs"].append(
            np.asarray(selected_input, dtype=float).copy()
        )
        data["log_cost_targets"].append(
            float(np.log1p(len(state_ids_after_second)))
        )
        data["cost_rollout_ids"].append(int(rollout_id))
    if stop_reason not in ("flat", "collision"):
        return False
    potentials = np.asarray(
        [history[state_id].potential for state_id in state_ids_after_second],
        dtype=float,
    )
    flatness = float(
        np.log1p(
            np.min(potentials)
            / max(config["flatness_tolerance"], 1.0e-12)
        )
    )
    if stop_reason == "flat":
        descriptor = terminal_target_from_history(
            history,
            state_ids_after_second[-1],
            config["layer_threshold"],
            output_spec,
        )
    else:
        descriptor = np.full(data["output_dimension"], np.nan, dtype=float)

    data["inputs"].append(np.asarray(selected_input, dtype=float).copy())
    data["flat_labels"].append(stop_reason == "flat")
    data["flatness_targets"].append(flatness)
    data["descriptor_targets"].append(descriptor)
    data["rollout_ids"].append(int(rollout_id))
    data["stop_reasons"].append(stop_reason)
    data["theta"].append(float(theta))
    return True


def _uses_cost_model(config):
    return config["selector"].startswith("learned_angle_gp_cost_aware")


def _fit_angle_gp(data, model, config):
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
    # Preserve the fitting work and numerical behavior of the pre-existing
    # selectors: the fourth GP is fitted only for an explicitly cost-aware run.
    if _uses_cost_model(config):
        cost_inputs = np.asarray(data["cost_inputs"], dtype=float).reshape(
            -1, model["input_dimension"]
        )
        surrogate["cost"] = fit_shared_kernel_gp(
            cost_inputs,
            np.asarray(data["log_cost_targets"], dtype=float)[:, None],
            state_dimension=model["input_state_dimension"],
            state_length_scale=config["state_length_scale"],
            action_length_scale=config["action_length_scale"],
            signal_variance=config["signal_variance"],
            noise_std=config["regression_noise_std"],
            output_scale_floor=config["cost_output_scale_floor"],
        )
    return surrogate


def _cost_adjusted_acquisition(surrogate, query, numerator, config):
    """Return discovery value per predicted future mechanics step.

    The cost GP output is ``Y = log(1 + C)``.  If ``Y`` is Gaussian, the
    expected raw cost is ``exp(mu + variance / 2) - 1``.  Alternatively, an
    optimistic cost lower confidence bound uses
    ``exp(mu - kappa * sigma) - 1``.  Both are clipped to a positive physical
    minimum because a valid K=2 macro always pays at least the q2 step.
    """
    cost_model = surrogate.get("cost")
    if cost_model is None:
        ones = np.ones(len(query), dtype=float)
        return np.asarray(numerator, dtype=float).copy(), {
            "cost_log_mean": np.full(len(query), np.nan),
            "cost_log_std": np.full(len(query), np.nan),
            "cost_denominator_steps": ones,
        }

    log_mean, log_variance = predict_shared_kernel_gp(cost_model, query)
    log_mean = log_mean[:, 0]
    log_variance = np.maximum(log_variance[:, 0], 0.0)
    log_std = np.sqrt(log_variance)
    if config["cost_denominator"] == "expected":
        log_one_plus_cost = log_mean + 0.5 * log_variance
    else:
        log_one_plus_cost = (
            log_mean - config["cost_lcb_kappa"] * log_std
        )
    predicted_cost = np.expm1(np.clip(log_one_plus_cost, -50.0, 50.0))
    denominator = np.maximum(predicted_cost, config["cost_floor_steps"])
    adjusted = np.asarray(numerator, dtype=float) / denominator
    return adjusted, {
        "cost_log_mean": log_mean,
        "cost_log_std": log_std,
        "cost_denominator_steps": denominator,
    }


def _balanced_warmup_active(config, completed_second_actions):
    # Paired warmup replaces the legacy one-angle-per-q1 warmup.  The main
    # loop records its handoff by completed paired groups instead.
    if config["paired_warmup"]:
        return False
    if config["common_warmup"]:
        return completed_second_actions < config["warmup_rollouts"]
    return (
        config["selector"].startswith("learned_angle_gp")
        and completed_second_actions < config["warmup_rollouts"]
    )


def _nearest_angle_candidate(candidates, target):
    distances = [circular_distance(candidate["theta"], target) for candidate in candidates]
    return int(np.argmin(distances))


def _exact_angle_candidate(q1, frame, theta, model, config):
    """Create an exact paired-warmup angle, independent of the online grid."""
    displacement = second_displacement(
        frame,
        theta,
        config["perturbation_scale"],
    )
    if displacement is None:
        return None
    return {
        "angle_index": -1,
        "theta": float(theta),
        "cos_theta": float(np.cos(theta)),
        "sin_theta": float(np.sin(theta)),
        "displacement": displacement,
        "input": macro_gp_input(
            q1.node,
            frame,
            displacement,
            theta,
            model["index_free"],
            config["reference_point"],
            model["length_scale"],
        ),
    }


def _select_second_angle(
    candidates,
    completed_second_actions,
    warmup_order,
    surrogate,
    elite_archive,
    output_spec,
    model,
    config,
    selection_rng,
    shuffle_rng,
    posterior_rng,
):
    """Return candidate index, diagnostics, and a numeric selection code."""
    if _balanced_warmup_active(config, completed_second_actions):
        target = warmup_order[completed_second_actions % len(warmup_order)]
        return _nearest_angle_candidate(candidates, target), None, 5

    if config["selector"] == "independent_random":
        return int(selection_rng.integers(len(candidates))), None, 1
    if config["selector"] == "fixed_transport":
        if selection_rng.random() < config["epsilon_random"]:
            return int(selection_rng.integers(len(candidates))), None, 8
        return _nearest_angle_candidate(candidates, 0.0), None, 2
    if config["selector"] == "fixed_reverse":
        if selection_rng.random() < config["epsilon_random"]:
            return int(selection_rng.integers(len(candidates))), None, 11
        return _nearest_angle_candidate(candidates, np.pi), None, 10

    if surrogate is None:
        return int(selection_rng.integers(len(candidates))), None, 4

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
    if config["selector"] in (
        "learned_angle_gp_shuffled",
        "learned_angle_gp_cost_aware_shuffled",
    ):
        diagnostics["selection_score"] = shuffle_rng.permutation(
            diagnostics["selection_score"]
        )

    if selection_rng.random() < config["epsilon_random"]:
        return int(selection_rng.integers(len(candidates))), diagnostics, 4
    finite = np.nan_to_num(diagnostics["selection_score"], nan=-np.inf)
    keep_count = max(
        1,
        int(np.ceil(config["top_fraction"] * len(candidates))),
    )
    top_indices = np.argsort(-finite, kind="stable")[:keep_count]
    selected = int(selection_rng.choice(top_indices))
    if config["selector"] == "learned_angle_gp_shuffled":
        code = 9
    elif config["selector"] == "learned_angle_gp_cost_aware":
        code = 12
    elif config["selector"] == "learned_angle_gp_cost_aware_shuffled":
        code = 13
    else:
        code = 3
    return selected, diagnostics, code


def _prefix_hash(history, elite_archive):
    """Hash the exact warmup tree geometry, genealogy, and archive cells."""
    digest = hashlib.sha256()
    for state in history:
        digest.update(np.asarray(state.node, dtype="<f8").tobytes())
        digest.update(str(state.parent_id).encode("ascii"))
        digest.update(str(state.terminal_reason).encode("utf-8"))
        digest.update(str(state.incoming_step_type).encode("utf-8"))
    for cell in elite_archive:
        digest.update(np.asarray(cell["shape_descriptor"], dtype="<f8").tobytes())
        digest.update(repr(cell["layer_key"]).encode("utf-8"))
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
        rollout_ids=np.asarray(training_data["rollout_ids"], dtype=int),
        theta=np.asarray(training_data["theta"], dtype=float),
        stop_reasons=np.asarray(training_data["stop_reasons"], dtype="U24"),
        cost_inputs=np.asarray(training_data["cost_inputs"], dtype=float).reshape(
            -1, model["input_dimension"]
        ),
        log_cost_targets=np.asarray(
            training_data["log_cost_targets"], dtype=float
        ),
        cost_rollout_ids=np.asarray(
            training_data["cost_rollout_ids"], dtype=int
        ),
    )


def main():
    base_directory = Path(__file__).resolve().parent
    config = _configuration6()
    initial_node, model = _make_model(base_directory, config)
    output_spec = make_terminal_output_spec(
        initial_node,
        model["index_free"],
        config["reference_point"],
        model["length_scale"],
    )
    initial_potential = compute_gravity_potential(
        initial_node, config["reference_point"]
    )
    history = create_history(initial_node, initial_potential)

    seed = config["seed"]
    parent_rng = np.random.default_rng(seed + 6101)
    first_action_rng = np.random.default_rng(seed + 6202)
    frame_rng = np.random.default_rng(seed + 6303)
    selection_rng = np.random.default_rng(seed + 6404)
    posterior_rng = np.random.default_rng(seed + 6505)
    warmup_rng = np.random.default_rng(seed + 6606)
    shuffle_rng = np.random.default_rng(seed + 6707)
    np.random.seed(seed)

    standard_angles = angle_grid(config["num_angles"], 2)
    warmup_order = warmup_rng.permutation(standard_angles)
    training_data = _new_training_data(len(output_spec["active_positions"]))
    elite_archive = []
    discovery_curve = [(0, 0)]
    discovery_events = []
    rollout_rows = []
    surrogate = None
    completed_since_fit = 0
    completed_second_actions = 0
    completed_paired_groups = 0
    completed_paired_branches = 0
    used_steps = 0
    rollout_id = 0
    warmup_end_step = None
    warmup_end_cells = None
    warmup_prefix_hash = None
    fit_time = 0.0
    score_time = 0.0
    start_time = time.perf_counter()

    trunk_terminal, trunk_ids, spent, trunk_reason = _initial_deterministic_trunk(
        history,
        config["total_step_budget"],
        model,
        config,
    )
    used_steps += spent
    trunk_new = False
    if trunk_terminal is not None and trunk_reason == "flat":
        trunk_new = _update_archive(
            elite_archive,
            history,
            trunk_terminal.state_id,
            model,
            config,
        )
        if trunk_new:
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
    trunk_end_step = used_steps
    trunk_cell_count = len(elite_archive)
    warmup_required_for_run = (
        config["paired_warmup"]
        or config["common_warmup"]
        or config["selector"].startswith("learned_angle_gp")
    )
    requested_warmup_units = (
        config["paired_warmup_groups"]
        if config["paired_warmup"]
        else config["warmup_rollouts"]
    )
    if not warmup_required_for_run or requested_warmup_units == 0:
        warmup_end_step = int(used_steps)
        warmup_end_cells = int(len(elite_archive))
        warmup_prefix_hash = _prefix_hash(history, elite_archive)

    while used_steps < config["total_step_budget"]:
        rollout_start = used_steps
        first = _execute_common_first_action(
            history,
            model,
            config,
            parent_rng,
            first_action_rng,
        )
        if first is None:
            break
        used_steps += 1
        q1 = first["q1"]
        all_new_ids = [q1.state_id]
        state_ids_after_second = []
        q2_state_id = None
        theta = np.nan
        angle_index = -1
        selection_code = 6
        diagnostics = None
        selected_index = None
        frame = None
        candidates = []

        if q1.terminal_reason is not None:
            terminal = q1
            stop_reason = q1.terminal_reason
        elif used_steps >= config["total_step_budget"]:
            terminal = q1
            stop_reason = None
        else:
            frame = transported_frame(
                q1.node,
                first["v1"],
                first["w"],
                model,
                config["perturbation_scale"],
                config["minimum_projection_ratio"],
                rng=frame_rng,
            )
            if frame is None:
                terminal, suffix_spent, stop_reason = (
                    _complete_deterministic_suffix(
                        history,
                        q1,
                        all_new_ids,
                        config["total_step_budget"] - used_steps,
                        model,
                        config,
                    )
                )
                used_steps += suffix_spent
                selection_code = 7
            else:
                candidates = make_angle_candidates(
                    q1.node,
                    frame,
                    config["num_angles"],
                    config["perturbation_scale"],
                    model["index_free"],
                    config["reference_point"],
                    model["length_scale"],
                )
                if not candidates:
                    terminal, suffix_spent, stop_reason = (
                        _complete_deterministic_suffix(
                            history,
                            q1,
                            all_new_ids,
                            config["total_step_budget"] - used_steps,
                            model,
                            config,
                        )
                    )
                    used_steps += suffix_spent
                    selection_code = 7
                else:
                    if (
                        config["paired_warmup"]
                        and completed_paired_groups
                        < config["paired_warmup_groups"]
                    ):
                        group_index = int(completed_paired_groups)
                        middle_angle = (
                            0.5 * np.pi
                            if group_index % 2 == 0
                            else -0.5 * np.pi
                        )
                        branch_specs = (
                            ("transport", 0.0, 14),
                            ("orthogonal", middle_angle, 15),
                            ("reverse", np.pi, 16),
                        )
                        branches_this_group = 0
                        for branch_index, (
                            branch_role,
                            target_angle,
                            branch_code,
                        ) in enumerate(branch_specs):
                            if used_steps >= config["total_step_budget"]:
                                break
                            branch_candidate = _exact_angle_candidate(
                                q1,
                                frame,
                                target_angle,
                                model,
                                config,
                            )
                            # A one-dimensional tangent has no orthogonal u.
                            # Only in that degenerate case do we fall back to
                            # the nearest valid grid direction, which remains
                            # explicit in the saved actual theta.
                            if branch_candidate is None:
                                candidate_index = _nearest_angle_candidate(
                                    candidates, target_angle
                                )
                                branch_candidate = candidates[
                                    candidate_index
                                ]
                            branch_theta = float(branch_candidate["theta"])
                            branch_start = (
                                rollout_start
                                if branch_index == 0
                                else used_steps
                            )
                            branch_new_ids = [q1.state_id]
                            branch_after_second = []
                            q2_branch = _perturbation_step(
                                history,
                                q1.state_id,
                                branch_candidate["displacement"],
                                model,
                                config,
                            )
                            branch_new_ids.append(q2_branch.state_id)
                            branch_after_second.append(q2_branch.state_id)
                            used_steps += 1
                            completed_second_actions += 1
                            completed_paired_branches += 1

                            (
                                branch_terminal,
                                suffix_spent,
                                branch_stop_reason,
                            ) = _complete_deterministic_suffix(
                                history,
                                q2_branch,
                                branch_new_ids,
                                config["total_step_budget"] - used_steps,
                                model,
                                config,
                            )
                            branch_after_second.extend(branch_new_ids[2:])
                            used_steps += suffix_spent

                            cost_count_before = len(
                                training_data["cost_inputs"]
                            )
                            added = _append_macro_teacher(
                                training_data,
                                history,
                                branch_candidate["input"],
                                branch_theta,
                                branch_after_second,
                                branch_stop_reason,
                                rollout_id,
                                config,
                                output_spec,
                            )
                            cost_added = (
                                len(training_data["cost_inputs"])
                                > cost_count_before
                            )
                            if added or (
                                _uses_cost_model(config) and cost_added
                            ):
                                completed_since_fit += 1

                            branch_new_cell = False
                            if branch_stop_reason == "flat":
                                branch_new_cell = _update_archive(
                                    elite_archive,
                                    history,
                                    branch_terminal.state_id,
                                    model,
                                    config,
                                )
                                if branch_new_cell:
                                    discovery_curve.append(
                                        (used_steps, len(elite_archive))
                                    )
                                    discovery_events.append(
                                        _discovery_event(
                                            history,
                                            branch_terminal.state_id,
                                            used_steps,
                                            len(elite_archive) - 1,
                                            branch_code,
                                        )
                                    )

                            rollout_rows.append(
                                {
                                    "rollout_id": int(rollout_id),
                                    "start_step": int(branch_start),
                                    "end_step": int(used_steps),
                                    "parent_id": int(first["parent_id"]),
                                    "q1_state_id": int(first["q1_state_id"]),
                                    "q2_state_id": int(q2_branch.state_id),
                                    "spent_steps": int(
                                        used_steps - branch_start
                                    ),
                                    "stop_reason": branch_stop_reason,
                                    "new_cell": int(branch_new_cell),
                                    "selection_code": int(branch_code),
                                    "theta": branch_theta,
                                    "angle_index": int(
                                        branch_candidate["angle_index"]
                                    ),
                                    "num_angle_candidates": int(
                                        len(candidates)
                                    ),
                                    "projection_ratio": frame[
                                        "projection_ratio"
                                    ],
                                    "reference_projection_ratio": frame[
                                        "reference_projection_ratio"
                                    ],
                                    "reference_fallback": int(
                                        frame["reference_fallback"]
                                    ),
                                    "training_rollouts": int(
                                        len(training_data["inputs"])
                                    ),
                                    "paired_warmup_group": group_index,
                                    "paired_warmup_branch": int(branch_index),
                                    "paired_angle_role": branch_role,
                                    "shared_q1_cost_included": int(
                                        branch_index == 0
                                    ),
                                    "acquisition": np.nan,
                                    "p_flat": np.nan,
                                    "p_reach": np.nan,
                                    "mean_novelty": np.nan,
                                    "descriptor_uncertainty": np.nan,
                                    "cost_log_mean": np.nan,
                                    "cost_log_std": np.nan,
                                    "cost_denominator_steps": np.nan,
                                }
                            )
                            rollout_id += 1
                            branches_this_group += 1

                        if branches_this_group == len(branch_specs):
                            completed_paired_groups += 1
                            if (
                                warmup_end_step is None
                                and completed_paired_groups
                                >= config["paired_warmup_groups"]
                            ):
                                warmup_end_step = int(used_steps)
                                warmup_end_cells = int(len(elite_archive))
                                warmup_prefix_hash = _prefix_hash(
                                    history, elite_archive
                                )
                        if config["verbose"] and (
                            branches_this_group > 0
                        ):
                            print(
                                f"paired warmup group={group_index}, "
                                f"branches={branches_this_group}, "
                                f"steps={used_steps}, cells={len(elite_archive)}"
                            )
                        # The q1 and every evaluated branch have already been
                        # accounted for.  Do not execute a fourth selected
                        # angle in the ordinary single-branch path below.
                        continue

                    warmup_now = _balanced_warmup_active(
                        config, completed_second_actions
                    )
                    has_training_data = len(training_data["inputs"]) > 0 or (
                        _uses_cost_model(config)
                        and len(training_data["cost_inputs"]) > 0
                    )
                    if (
                        config["selector"].startswith("learned_angle_gp")
                        and not warmup_now
                        and has_training_data
                        and (
                            surrogate is None
                            or completed_since_fit
                            >= config["refit_rollout_interval"]
                        )
                    ):
                        fit_start = time.perf_counter()
                        surrogate = _fit_angle_gp(
                            training_data, model, config
                        )
                        fit_time += time.perf_counter() - fit_start
                        completed_since_fit = 0

                    score_start = time.perf_counter()
                    selected_index, diagnostics, selection_code = (
                        _select_second_angle(
                            candidates,
                            completed_second_actions,
                            warmup_order,
                            surrogate,
                            elite_archive,
                            output_spec,
                            model,
                            config,
                            selection_rng,
                            shuffle_rng,
                            posterior_rng,
                        )
                    )
                    score_time += time.perf_counter() - score_start
                    selected = candidates[selected_index]
                    theta = float(selected["theta"])
                    angle_index = int(selected["angle_index"])
                    q2 = _perturbation_step(
                        history,
                        q1.state_id,
                        selected["displacement"],
                        model,
                        config,
                    )
                    q2_state_id = int(q2.state_id)
                    all_new_ids.append(q2.state_id)
                    state_ids_after_second.append(q2.state_id)
                    used_steps += 1
                    completed_second_actions += 1

                    terminal, suffix_spent, stop_reason = (
                        _complete_deterministic_suffix(
                            history,
                            q2,
                            all_new_ids,
                            config["total_step_budget"] - used_steps,
                            model,
                            config,
                        )
                    )
                    state_ids_after_second.extend(all_new_ids[2:])
                    used_steps += suffix_spent
                    cost_count_before = len(training_data["cost_inputs"])
                    added = _append_macro_teacher(
                        training_data,
                        history,
                        selected["input"],
                        theta,
                        state_ids_after_second,
                        stop_reason,
                        rollout_id,
                        config,
                        output_spec,
                    )
                    cost_added = (
                        len(training_data["cost_inputs"]) > cost_count_before
                    )
                    if added or (_uses_cost_model(config) and cost_added):
                        completed_since_fit += 1

        is_new_cell = False
        if terminal is not None and stop_reason == "flat":
            is_new_cell = _update_archive(
                elite_archive,
                history,
                terminal.state_id,
                model,
                config,
            )
            if is_new_cell:
                discovery_curve.append((used_steps, len(elite_archive)))
                discovery_events.append(
                    _discovery_event(
                        history,
                        terminal.state_id,
                        used_steps,
                        len(elite_archive) - 1,
                        selection_code,
                    )
                )

        if (
            warmup_end_step is None
            and (
                (
                    config["paired_warmup"]
                    and completed_paired_groups
                    >= config["paired_warmup_groups"]
                )
                or (
                    not config["paired_warmup"]
                    and completed_second_actions
                    >= config["warmup_rollouts"]
                )
            )
        ):
            warmup_end_step = int(used_steps)
            warmup_end_cells = int(len(elite_archive))
            warmup_prefix_hash = _prefix_hash(history, elite_archive)

        if diagnostics is not None and selected_index is not None:
            diagnostic_values = {
                "acquisition": float(
                    diagnostics.get("selection_score", diagnostics["score"])[
                        selected_index
                    ]
                ),
                "p_flat": float(
                    diagnostics["flat_probability"][selected_index]
                ),
                "p_reach": float(
                    diagnostics["reach_probability"][selected_index]
                ),
                "mean_novelty": float(
                    diagnostics["mean_novelty"][selected_index]
                ),
                "descriptor_uncertainty": float(
                    diagnostics["descriptor_uncertainty"][selected_index]
                ),
                "cost_log_mean": float(
                    diagnostics.get(
                        "cost_log_mean",
                        np.full(len(candidates), np.nan),
                    )[selected_index]
                ),
                "cost_log_std": float(
                    diagnostics.get(
                        "cost_log_std",
                        np.full(len(candidates), np.nan),
                    )[selected_index]
                ),
                "cost_denominator_steps": float(
                    diagnostics.get(
                        "cost_denominator_steps",
                        np.full(len(candidates), np.nan),
                    )[selected_index]
                ),
            }
        else:
            diagnostic_values = {
                "acquisition": np.nan,
                "p_flat": np.nan,
                "p_reach": np.nan,
                "mean_novelty": np.nan,
                "descriptor_uncertainty": np.nan,
                "cost_log_mean": np.nan,
                "cost_log_std": np.nan,
                "cost_denominator_steps": np.nan,
            }
        rollout_rows.append(
            {
                "rollout_id": int(rollout_id),
                "start_step": int(rollout_start),
                "end_step": int(used_steps),
                "parent_id": int(first["parent_id"]),
                "q1_state_id": int(first["q1_state_id"]),
                "q2_state_id": -1 if q2_state_id is None else q2_state_id,
                "spent_steps": int(used_steps - rollout_start),
                "stop_reason": stop_reason,
                "new_cell": int(is_new_cell),
                "selection_code": int(selection_code),
                "theta": theta,
                "angle_index": int(angle_index),
                "num_angle_candidates": int(len(candidates)),
                "projection_ratio": (
                    np.nan if frame is None else frame["projection_ratio"]
                ),
                "reference_projection_ratio": (
                    np.nan
                    if frame is None
                    else frame["reference_projection_ratio"]
                ),
                "reference_fallback": (
                    0 if frame is None else int(frame["reference_fallback"])
                ),
                "training_rollouts": int(len(training_data["inputs"])),
                "paired_warmup_group": -1,
                "paired_warmup_branch": -1,
                "paired_angle_role": "none",
                "shared_q1_cost_included": 1,
                **diagnostic_values,
            }
        )
        rollout_id += 1
        if config["verbose"] and (is_new_cell or used_steps >= config["total_step_budget"]):
            print(
                f"steps={used_steps}, cells={len(elite_archive)}, "
                f"selector={config['selector']}, theta={theta}"
            )

    if warmup_end_step is None:
        warmup_end_step = int(used_steps)
        warmup_end_cells = int(len(elite_archive))
        warmup_prefix_hash = _prefix_hash(history, elite_archive)

    wall_time = time.perf_counter() - start_time
    stop_reasons = [row["stop_reason"] for row in rollout_rows]
    selection_counts = Counter(
        int(row["selection_code"]) for row in rollout_rows
    )
    summary = {
        "algorithm_version": config["algorithm_version"],
        "selector": config["selector"],
        "seed": int(seed),
        "step_budget": int(config["total_step_budget"]),
        "used_steps": int(used_steps),
        "elite_cells": int(len(elite_archive)),
        "discovery_auc": float(
            _discovery_auc(discovery_curve, config["total_step_budget"])
        ),
        "trunk_end_step": int(trunk_end_step),
        "trunk_cell_count": int(trunk_cell_count),
        "trunk_stop_reason": trunk_reason,
        "num_states": int(len(history)),
        "num_rollouts": int(len(rollout_rows)),
        "completed_second_actions": int(completed_second_actions),
        "training_rollouts": int(len(training_data["inputs"])),
        "cost_training_rollouts": int(
            len(training_data["cost_inputs"])
        ),
        "flat_rollouts": int(sum(reason == "flat" for reason in stop_reasons)),
        "collision_rollouts": int(
            sum(reason == "collision" for reason in stop_reasons)
        ),
        "selection_code_counts": {
            str(code): int(count)
            for code, count in sorted(selection_counts.items())
        },
        "selection_code_legend": {
            "1": "independent_random",
            "2": "fixed_theta_zero",
            "3": "learned_top_fraction",
            "4": "learned_epsilon_random",
            "5": "common_balanced_warmup",
            "6": "first_step_terminal_or_budget_end",
            "7": "transport_frame_unavailable",
            "8": "fixed_epsilon_random",
            "9": "shuffled_score_top_fraction",
            "10": "fixed_theta_pi",
            "11": "fixed_reverse_epsilon_random",
            "12": "cost_aware_top_fraction",
            "13": "cost_aware_shuffled_top_fraction",
            "14": "paired_warmup_theta_zero",
            "15": "paired_warmup_theta_orthogonal",
            "16": "paired_warmup_theta_pi",
        },
        "common_warmup": bool(config["common_warmup"]),
        "warmup_requested_rollouts": int(config["warmup_rollouts"]),
        "paired_warmup": bool(config["paired_warmup"]),
        "paired_warmup_requested_groups": int(
            config["paired_warmup_groups"]
        ),
        "paired_warmup_completed_groups": int(completed_paired_groups),
        "paired_warmup_completed_branches": int(completed_paired_branches),
        "warmup_completed": bool(
            completed_paired_groups >= config["paired_warmup_groups"]
            if config["paired_warmup"]
            else completed_second_actions >= config["warmup_rollouts"]
        ),
        "warmup_end_step": int(warmup_end_step),
        "warmup_end_cells": int(warmup_end_cells),
        "warmup_prefix_sha256": warmup_prefix_hash,
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
