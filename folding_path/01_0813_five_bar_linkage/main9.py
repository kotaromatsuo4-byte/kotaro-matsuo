"""Collision-continuing variable-K search with a conservative forest filter.

One expensive decision is ``(parent state q, physical tangent displacement
v1, integer K)``.  The first displacement is corrected, then its physical
direction is projected into every new tangent space and kept at the same
norm for at most K perturbation edges.  An independent deterministic suffix
then continues to a flat state.  Collision is recorded on every edge but never
stops integration; only flatness, corrector failure, or the global mechanics
budget completes/censors a macro.  A positive ``max_suffix_steps`` can be set
as an explicit diagnostic timeout; the default zero disables this local cap.

The three selectors use the same candidate-pool generator:

``random``
    Uniform sampling from the full pool.
``forest_filter``
    Conservatively reject candidates confidently predicted to return to a
    cell which already has a collision-free representative, then sample
    uniformly from the retained pool.  An epsilon fraction bypasses filtering.
``forest_filter_shuffled``
    Permute only the final reject mask.  It preserves the rejection count and
    is therefore a negative control for information in the learned filter.

Each macro contributes residual/hindsight rows for its persistent prefixes.
Rows from the same macro share ``macro_id`` and their weights sum to one, so
they cannot masquerade as independent simulations during forest bootstrap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from angle_macro import project_to_tangent, sample_tangent_displacement, tangent_basis
from collision_continue import (
    archive_layer_view,
    deterministic_step_continue,
    displacement_step_continue,
    dual_archive_counts,
    initialize_path_metrics,
    new_dual_archive,
    update_dual_archive_from_metrics,
)
from forest_cell_model import (
    DuplicateFilterThresholds,
    conservative_duplicate_mask,
    fit_cell_filter,
    make_macro_feature,
    permute_reject_mask,
    predict_cell_filter,
)
from main5 import _configuration, _discovery_auc, _load_model
from main7 import _last_nonflat_context
from mechanics import compute_gravity_potential
from potential_tree import save_potential_tree
from result import reset_output_directory, save_discovery_curve, save_elite_paths
from state import create_history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int_list(name, default):
    text = os.environ.get(name)
    if text is None or not text.strip():
        return [int(value) for value in default]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _configuration9():
    """Physical defaults plus main9-only, environment-overridable settings."""
    # The held-out study found no safe learned-filter advantage.  Keep the
    # validated variable-K random search as the zero-configuration default;
    # learned filters remain explicit experimental selectors.
    selector = os.environ.get("FOLDING_MAIN9_SELECTOR", "random")
    allowed = ("random", "forest_filter", "forest_filter_shuffled")
    if selector not in allowed:
        raise ValueError(f"FOLDING_MAIN9_SELECTOR must be one of {allowed}")

    config = _configuration()
    k_values = sorted(
        set(_env_int_list("FOLDING_MAIN9_K_VALUES", range(1, 25)))
    )
    warmup_values = _env_int_list(
        "FOLDING_MAIN9_WARMUP_K_VALUES", (1, 2, 4, 8, 16, 24)
    )
    config.update(
        {
            "algorithm_version": (
                "main9_collision_continue_forest_filter_v1_1_unbounded_suffix"
            ),
            "selector": selector,
            "k_values": k_values,
            "k_max": max(k_values) if k_values else 0,
            "warmup_k_values": warmup_values,
            "paired_warmup_groups": _env_int(
                "FOLDING_MAIN9_PAIRED_WARMUP_GROUPS", 0
            ),
            "candidate_parent_limit": _env_int(
                "FOLDING_MAIN9_PARENT_LIMIT", 16
            ),
            "directions_per_parent": _env_int(
                "FOLDING_MAIN9_DIRECTIONS", 2
            ),
            "max_suffix_steps": _env_int(
                "FOLDING_MAIN9_MAX_SUFFIX_STEPS", 0
            ),
            "transport_min_projection_ratio": _env_float(
                "FOLDING_MAIN9_MIN_PROJECTION_RATIO", 0.10
            ),
            "epsilon_random": _env_float("FOLDING_MAIN9_EPSILON", 0.20),
            "refit_macro_interval": _env_int(
                "FOLDING_MAIN9_REFIT_INTERVAL", 3
            ),
            "forest_kind": os.environ.get(
                "FOLDING_MAIN9_FOREST_KIND", "random_forest"
            ),
            "forest_estimators": _env_int(
                "FOLDING_MAIN9_FOREST_ESTIMATORS", 64
            ),
            "forest_max_depth": _env_int(
                "FOLDING_MAIN9_FOREST_MAX_DEPTH", 10
            ),
            "forest_min_leaf": _env_int(
                "FOLDING_MAIN9_FOREST_MIN_LEAF", 2
            ),
            "filter_mean_probability": _env_float(
                "FOLDING_MAIN9_FILTER_MEAN_PROBABILITY", 0.90
            ),
            "filter_lower_probability": _env_float(
                "FOLDING_MAIN9_FILTER_LOWER_PROBABILITY", 0.70
            ),
            "filter_vote_fraction": _env_float(
                "FOLDING_MAIN9_FILTER_VOTE_FRACTION", 0.80
            ),
            "filter_max_disagreement": _env_float(
                "FOLDING_MAIN9_FILTER_MAX_DISAGREEMENT", 0.08
            ),
            "filter_max_support_ratio": _env_float(
                "FOLDING_MAIN9_FILTER_MAX_SUPPORT_RATIO", 1.0
            ),
            "filter_min_groups_per_class": _env_int(
                "FOLDING_MAIN9_FILTER_MIN_GROUPS_PER_CLASS", 4
            ),
            "max_reject_fraction": _env_float(
                "FOLDING_MAIN9_MAX_REJECT_FRACTION", 0.50
            ),
            "result_name": os.environ.get(
                "FOLDING_RESULT_NAME", f"main9_{selector}"
            ),
            "save_tree": bool(_env_int("FOLDING_SAVE_TREE", 0)),
            "verbose": bool(_env_int("FOLDING_VERBOSE", 1)),
        }
    )
    if not k_values or min(k_values) <= 0:
        raise ValueError("FOLDING_MAIN9_K_VALUES must contain positive integers")
    if config["candidate_parent_limit"] <= 0:
        raise ValueError("FOLDING_MAIN9_PARENT_LIMIT must be positive")
    if config["directions_per_parent"] <= 0:
        raise ValueError("FOLDING_MAIN9_DIRECTIONS must be positive")
    if config["max_suffix_steps"] < 0:
        raise ValueError(
            "FOLDING_MAIN9_MAX_SUFFIX_STEPS must be nonnegative "
            "(zero means no local suffix timeout)"
        )
    if config["paired_warmup_groups"] < 0:
        raise ValueError("paired warmup group count must be nonnegative")
    if not 0.0 <= config["epsilon_random"] <= 1.0:
        raise ValueError("FOLDING_MAIN9_EPSILON must be in [0, 1]")
    if not 0.0 <= config["max_reject_fraction"] < 1.0:
        raise ValueError("FOLDING_MAIN9_MAX_REJECT_FRACTION must be in [0, 1)")
    if config["refit_macro_interval"] <= 0:
        raise ValueError("FOLDING_MAIN9_REFIT_INTERVAL must be positive")
    return config


def _make_model(base_directory, config):
    node, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    length_scale = float(np.mean(member_length))
    free_dimension = int(len(index_free))
    return node, {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": int(len(node)),
        "num_member": int(len(member)),
        "length_scale": length_scale,
        "free_dimension": free_dimension,
        "input_dimension": 3 * free_dimension + 1,
    }


def _feature(history, parent_id, displacement, k, model, config):
    parent = history[int(parent_id)]
    context = _last_nonflat_context(
        history, int(parent_id), config["layer_threshold"]
    )
    return make_macro_feature(
        parent.node,
        context,
        displacement,
        int(k),
        length_scale=model["length_scale"],
        free_indices=model["index_free"],
        reference_point=config["reference_point"],
        k_max=config["k_max"],
    )


def _candidate_pool(history, path_metrics, model, config, rng):
    """Build a bounded pool using collision-free, nonterminal parents only."""
    eligible = np.asarray(
        [
            state.state_id
            for state in history
            if state.terminal_reason is None
            and path_metrics[state.state_id].collision_free
        ],
        dtype=int,
    )
    if len(eligible) == 0:
        return []
    parent_count = min(len(eligible), config["candidate_parent_limit"])
    parents = rng.choice(eligible, size=parent_count, replace=False)
    candidates = []
    group_id = 0
    for parent_value in parents:
        parent_id = int(parent_value)
        node = history[parent_id].node
        tangent_dimension = int(tangent_basis(node, model).shape[1])
        if tangent_dimension == 0:
            continue
        for direction_index in range(config["directions_per_parent"]):
            displacement = sample_tangent_displacement(
                node,
                model,
                config["perturbation_scale"],
                rng,
            )
            if displacement is None:
                continue
            for k in config["k_values"]:
                candidates.append(
                    {
                        "group_id": int(group_id),
                        "parent_id": parent_id,
                        "direction_index": int(direction_index),
                        "v1": np.asarray(displacement, dtype=float).copy(),
                        "k": int(k),
                        "input": _feature(
                            history,
                            parent_id,
                            displacement,
                            int(k),
                            model,
                            config,
                        ),
                    }
                )
            group_id += 1
    return candidates


def _nearest_group_k(pool, group_id, target_k):
    indices = [
        index
        for index, item in enumerate(pool)
        if item["group_id"] == int(group_id)
    ]
    if not indices:
        return None
    distances = [abs(pool[index]["k"] - int(target_k)) for index in indices]
    return int(indices[int(np.argmin(distances))])


def _transport(node, previous_displacement, model, config):
    previous = np.asarray(previous_displacement, dtype=float).reshape(-1)
    previous_norm = float(np.linalg.norm(previous))
    projection, _ = project_to_tangent(node, previous, model)
    projection_norm = float(np.linalg.norm(projection))
    ratio = projection_norm / max(previous_norm, 1.0e-12)
    if (
        projection_norm <= 1.0e-12
        or ratio < config["transport_min_projection_ratio"]
    ):
        return None, ratio
    transported = previous_norm * projection / projection_norm
    return transported.reshape(model["num_node"], 3), ratio


def _perturbation_step(
    history, path_metrics, parent_id, displacement, model, config
):
    return displacement_step_continue(
        history,
        parent_id,
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
        path_metrics,
        config["collision_model"],
        model["length_scale"],
    )


def _deterministic_step(history, path_metrics, parent_id, model, config):
    return deterministic_step_continue(
        history,
        parent_id,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        config["target_norm_max"],
        config["constraint_tolerance"],
        config["beta0"],
        config["corrector_iterations"],
        config["line_search_iterations"],
        config["line_search_tau"],
        config["cma_population"],
        config["cma_generations"],
        config["cma_seed"],
        config["reference_point"],
        config["flatness_tolerance"],
        config["dist_tol"],
        config["normal_tol"],
        config["cos_tol"],
        path_metrics,
        config["collision_model"],
        model["length_scale"],
    )


def _execute_macro(history, path_metrics, candidate, remaining_budget, model, config):
    """Execute persistent edges, then an independent deterministic suffix."""
    current = history[int(candidate["parent_id"])]
    displacement = np.asarray(candidate["v1"], dtype=float).copy()
    perturbation_ids = []
    new_ids = []
    projection_ratios = []
    edge_collisions = []
    used = 0
    projection_failure = False

    for chain_index in range(int(candidate["k"])):
        if current.terminal_reason is not None or used >= int(remaining_budget):
            break
        if chain_index > 0:
            displacement, ratio = _transport(
                current.node, displacement, model, config
            )
            projection_ratios.append(float(ratio))
            if displacement is None:
                projection_failure = True
                break
        current, collided = _perturbation_step(
            history,
            path_metrics,
            current.state_id,
            displacement,
            model,
            config,
        )
        perturbation_ids.append(int(current.state_id))
        new_ids.append(int(current.state_id))
        edge_collisions.append(bool(collided))
        used += 1

    suffix_steps = 0
    suffix_limit = int(config["max_suffix_steps"])
    while (
        current.terminal_reason is None
        and (suffix_limit == 0 or suffix_steps < suffix_limit)
        and used < int(remaining_budget)
    ):
        current, collided = _deterministic_step(
            history, path_metrics, current.state_id, model, config
        )
        new_ids.append(int(current.state_id))
        edge_collisions.append(bool(collided))
        used += 1
        suffix_steps += 1

    if current.terminal_reason is not None:
        stop_reason = current.terminal_reason
    elif used >= int(remaining_budget):
        stop_reason = "global_budget"
    elif suffix_limit > 0 and suffix_steps >= suffix_limit:
        stop_reason = "timeout"
    else:
        raise RuntimeError("deterministic suffix ended without a stop condition")
    return {
        "terminal": current,
        "new_ids": new_ids,
        "perturbation_ids": perturbation_ids,
        "used_steps": int(used),
        "suffix_steps": int(suffix_steps),
        "stop_reason": stop_reason,
        "requested_k": int(candidate["k"]),
        "executed_k": int(len(perturbation_ids)),
        "projection_failure": bool(projection_failure),
        "projection_ratios": projection_ratios,
        "edge_collisions": edge_collisions,
    }


def _initial_trunk(history, path_metrics, remaining_budget, model, config):
    current = history[0]
    if current.potential <= config["flatness_tolerance"]:
        current.terminal_reason = "flat"
        return current, 0, "flat"
    steps = 0
    suffix_limit = int(config["max_suffix_steps"])
    while (
        current.terminal_reason is None
        and (suffix_limit == 0 or steps < suffix_limit)
        and steps < int(remaining_budget)
    ):
        current, _ = _deterministic_step(
            history, path_metrics, current.state_id, model, config
        )
        steps += 1
    if current.terminal_reason is not None:
        reason = current.terminal_reason
    elif steps >= int(remaining_budget):
        reason = "global_budget"
    elif suffix_limit > 0 and steps >= suffix_limit:
        reason = "timeout"
    else:
        raise RuntimeError("initial trunk ended without a stop condition")
    return current, int(steps), reason


def _archive_update(archive, history, path_metrics, terminal_id, model, config):
    return update_dual_archive_from_metrics(
        archive,
        history,
        terminal_id,
        path_metrics,
        model["member"],
        model["length_scale"],
        config["layer_threshold"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
        config["rms_tolerance"],
        config["max_tolerance"],
        config["reference_point"],
    )


def _new_training_data():
    keys = (
        "inputs",
        "cell_labels",
        "rollout_ids",
        "macro_ids",
        "sample_weights",
        "stop_reasons",
        "path_collision_edges",
        "path_geometric_length",
        "residual_collision_edges",
        "residual_geometric_length",
        "requested_k",
        "executed_k",
        "source_requested_k",
        "source_executed_k",
        "is_hindsight",
        "projection_failure",
        "source_macro_id",
        "macro_start_step",
        "macro_end_step",
    )
    return {key: [] for key in keys}


def _append_teachers(
    data,
    history,
    path_metrics,
    result,
    macro_id,
    macro_start_step,
    macro_end_step,
    cell_label,
    model,
    config,
):
    """Add one unit-weight macro group, including valid persistent prefixes."""
    perturbation_ids = result["perturbation_ids"]
    prefixes = []
    for prefix_index, child_id in enumerate(perturbation_ids):
        child = history[child_id]
        parent_id = int(child.parent_id)
        if not path_metrics[parent_id].collision_free:
            continue
        remaining_k = max(1, int(result["requested_k"]) - prefix_index)
        prefixes.append(
            (
                parent_id,
                np.asarray(child.incoming_displacement, dtype=float),
                remaining_k,
                max(0, int(result["executed_k"]) - prefix_index),
                prefix_index > 0,
            )
        )
    if not prefixes:
        return 0

    terminal_quality = path_metrics[result["terminal"].state_id]
    valid_label = int(cell_label) if result["stop_reason"] == "flat" else -1
    weight = 1.0 / len(prefixes)
    for parent_id, displacement, requested_k, executed_k, hindsight in prefixes:
        parent_quality = path_metrics[parent_id]
        data["inputs"].append(
            _feature(
                history,
                parent_id,
                displacement,
                requested_k,
                model,
                config,
            )
        )
        data["cell_labels"].append(valid_label)
        data["rollout_ids"].append(int(macro_id))
        data["macro_ids"].append(int(macro_id))
        data["sample_weights"].append(weight)
        data["stop_reasons"].append(str(result["stop_reason"]))
        data["path_collision_edges"].append(
            int(terminal_quality.collision_edge_count)
        )
        data["path_geometric_length"].append(
            float(terminal_quality.geometric_path_length)
        )
        data["residual_collision_edges"].append(
            int(
                terminal_quality.collision_edge_count
                - parent_quality.collision_edge_count
            )
        )
        data["residual_geometric_length"].append(
            float(
                terminal_quality.geometric_path_length
                - parent_quality.geometric_path_length
            )
        )
        data["requested_k"].append(int(requested_k))
        data["executed_k"].append(int(executed_k))
        data["source_requested_k"].append(int(result["requested_k"]))
        data["source_executed_k"].append(int(result["executed_k"]))
        data["is_hindsight"].append(bool(hindsight))
        data["projection_failure"].append(bool(result["projection_failure"]))
        data["source_macro_id"].append(int(macro_id))
        data["macro_start_step"].append(int(macro_start_step))
        data["macro_end_step"].append(int(macro_end_step))
    return len(prefixes)


def _fit_filter(data, config, seed):
    labels = np.asarray(data["cell_labels"], dtype=int)
    mask = labels >= 0
    if not np.any(mask):
        return None
    inputs = np.asarray(data["inputs"], dtype=float)[mask]
    groups = np.asarray(data["macro_ids"], dtype=int)[mask]
    weights = np.asarray(data["sample_weights"], dtype=float)[mask]
    return fit_cell_filter(
        inputs,
        labels[mask],
        groups,
        weights,
        model_kind=config["forest_kind"],
        rng_seed=int(seed),
        n_estimators=config["forest_estimators"],
        max_depth=config["forest_max_depth"],
        min_samples_leaf=config["forest_min_leaf"],
    )


def _rejectable_classes(archive):
    return [
        index
        for index, cell in enumerate(archive)
        if cell.get("feasible_elite") is not None
    ]


def _thresholds(config):
    return DuplicateFilterThresholds(
        mean_probability=config["filter_mean_probability"],
        lower_probability=config["filter_lower_probability"],
        vote_fraction=config["filter_vote_fraction"],
        max_disagreement=config["filter_max_disagreement"],
        max_support_ratio=config["filter_max_support_ratio"],
        min_groups_per_class=config["filter_min_groups_per_class"],
    )


def _pool_hash(pool):
    digest = hashlib.sha256()
    for item in pool:
        digest.update(
            np.asarray(
                [item["parent_id"], item["group_id"], item["k"]],
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(np.asarray(item["v1"], dtype=np.float64).tobytes())
        digest.update(np.asarray(item["input"], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _select(pool, selector, forest, archive, selection_rng, shuffle_rng, config):
    """Choose uniformly after a one-sided duplicate filter."""
    count = len(pool)
    # Every selector consumes this draw, including random and cold-start
    # controls.  Therefore their candidate trajectory remains bitwise common
    # until the learned filter actually rejects at least one candidate.
    epsilon_draw = float(selection_rng.random())
    base = {
        "selection_mode": "random",
        "selection_code": 0,
        "candidate_count": int(count),
        "candidate_pool_sha256": _pool_hash(pool),
        "rejected_count": 0,
        "raw_rejected_count": 0,
        "retained_count": int(count),
        "reject_cap": int(np.floor(config["max_reject_fraction"] * count)),
        "reject_cap_applied": False,
        "epsilon_bypass": False,
        "epsilon_draw": epsilon_draw,
        "forest_ready": bool(forest is not None and forest.filter_ready),
        "predicted_cell": -1,
        "max_probability": float("nan"),
        "support_ratio": float("nan"),
    }
    if selector == "random" or forest is None or not forest.filter_ready:
        index = int(selection_rng.integers(count))
        base["selection_mode"] = "random" if selector == "random" else "cold_random"
        base["selection_code"] = 0 if selector == "random" else 1
        return index, base
    if epsilon_draw < config["epsilon_random"]:
        index = int(selection_rng.integers(count))
        base.update(
            {"selection_mode": "epsilon_random", "selection_code": 2,
             "epsilon_bypass": True}
        )
        return index, base

    query = np.asarray([item["input"] for item in pool], dtype=float)
    prediction = predict_cell_filter(forest, query)
    reject = conservative_duplicate_mask(
        prediction, _rejectable_classes(archive), _thresholds(config)
    )
    raw_rejected_count = int(np.sum(reject))
    reject_cap = int(np.floor(config["max_reject_fraction"] * count))
    if raw_rejected_count > reject_cap:
        # Keep only the strongest duplicate decisions.  This one-sided safety
        # cap guarantees that an imperfect online model cannot suppress most
        # of the action pool, while retaining a deterministic ranking rule.
        rejected_indices = np.flatnonzero(reject)
        confidence = np.asarray(prediction["max_probability"], dtype=float)
        strongest = rejected_indices[
            np.argsort(confidence[rejected_indices], kind="stable")[::-1]
            [:reject_cap]
        ]
        reject = np.zeros(count, dtype=bool)
        reject[strongest] = True
    if selector == "forest_filter_shuffled":
        reject, _ = permute_reject_mask(reject, shuffle_rng)
        mode = "forest_filter_shuffled"
        code = 4
    else:
        mode = "forest_filter"
        code = 3
    retained = np.flatnonzero(~reject)
    if len(retained) == 0:
        retained = np.arange(count)
        mode += "_all_rejected_fallback"
        code = 5
    if len(retained) == count and np.array_equal(retained, np.arange(count)):
        # Match the full-random selector exactly when filtering changes
        # nothing; Generator.choice on an explicit array is not part of that
        # fairness contract.
        index = int(selection_rng.integers(count))
        if mode in ("forest_filter", "forest_filter_shuffled"):
            mode += "_no_rejection"
    else:
        index = int(selection_rng.choice(retained))
    labels = np.asarray(prediction["predicted_labels"])
    base.update(
        {
            "selection_mode": mode,
            "selection_code": code,
            "rejected_count": int(np.sum(reject)),
            "raw_rejected_count": raw_rejected_count,
            "retained_count": int(len(retained)),
            "reject_cap": reject_cap,
            "reject_cap_applied": bool(raw_rejected_count > reject_cap),
            "predicted_cell": int(labels[index]),
            "max_probability": float(prediction["max_probability"][index]),
            "support_ratio": float(prediction["support_ratio"][index]),
        }
    )
    return index, base


def _hash_state(history, path_metrics, archive):
    digest = hashlib.sha256()
    for state in history:
        digest.update(np.asarray(state.node, dtype=np.float64).tobytes())
        digest.update(
            f"{state.state_id}:{state.parent_id}:{state.terminal_reason}:"
            f"{state.incoming_step_type}".encode("utf-8")
        )
        quality = path_metrics[state.state_id]
        digest.update(
            np.asarray(
                [quality.collision_edge_count, quality.geometric_path_length,
                 quality.edge_count, quality.perturbation_count],
                dtype=np.float64,
            ).tobytes()
        )
    for cell in archive:
        digest.update(np.asarray(cell["shape_descriptor"]).tobytes())
        digest.update(repr(cell["layer_key"]).encode("utf-8"))
        digest.update(str(cell.get("feasible_elite") is not None).encode("ascii"))
    return digest.hexdigest()


def _hash_training(data):
    digest = hashlib.sha256()
    for key in data:
        if key == "stop_reasons":
            digest.update("\0".join(data[key]).encode("utf-8"))
        else:
            digest.update(np.asarray(data[key]).tobytes())
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_finished(curve, budget, cells):
    result = list(curve)
    if result[-1][0] != int(budget):
        result.append((int(budget), int(cells)))
    return result


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _append_discovery_events(
    events, archive_update, terminal_state_id, step, macro_id, selection_code
):
    if not archive_update["accepted"]:
        return
    common = {
        "cell_index": int(archive_update["cell_index"]),
        "terminal_state_id": int(terminal_state_id),
        "mechanics_step": int(step),
        "macro_id": int(macro_id),
        "selection_code": int(selection_code),
    }
    if archive_update["new_kinematic"]:
        events.append({"archive_layer": "kinematic", **common})
    if archive_update["new_feasible"]:
        events.append({"archive_layer": "feasible", **common})


def _save_outputs(
    output_directory,
    history,
    archive,
    kinematic_curve,
    feasible_curve,
    diagnostics,
    discovery_events,
    training,
    config,
    model,
    summary,
):
    reset_output_directory(output_directory)
    feasible_directory = output_directory / "feasible"
    kinematic_directory = output_directory / "kinematic"
    save_elite_paths(
        history, archive_layer_view(archive, "feasible"), feasible_directory
    )
    save_elite_paths(
        history, archive_layer_view(archive, "kinematic"), kinematic_directory
    )
    save_discovery_curve(feasible_curve, output_directory)
    save_discovery_curve(kinematic_curve, kinematic_directory)
    _write_csv(output_directory / "rollout_diagnostics.csv", diagnostics)
    (output_directory / "configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_directory / "cell_archive.json").write_text(
        json.dumps(_jsonable(archive), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "discovery_events.json").write_text(
        json.dumps(_jsonable(discovery_events), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    arrays = {}
    integer_keys = {
        "cell_labels", "rollout_ids", "macro_ids", "path_collision_edges",
        "residual_collision_edges", "requested_k", "executed_k",
        "source_requested_k", "source_executed_k", "source_macro_id",
        "macro_start_step", "macro_end_step",
    }
    for key, values in training.items():
        if key == "inputs":
            arrays[key] = np.asarray(values, dtype=float).reshape(
                -1, model["input_dimension"]
            )
        elif key == "stop_reasons":
            arrays[key] = np.asarray(values, dtype="U32")
        elif key in ("is_hindsight", "projection_failure"):
            arrays[key] = np.asarray(values, dtype=bool)
        elif key in integer_keys:
            arrays[key] = np.asarray(values, dtype=int)
        else:
            arrays[key] = np.asarray(values, dtype=float)
    np.savez_compressed(output_directory / "training_data.npz", **arrays)
    if config["save_tree"]:
        try:
            save_potential_tree(
                history=history,
                output_directory=output_directory,
                member=model["member"],
                reference_point=config["reference_point"],
            )
        except ValueError as error:
            (output_directory / "potential_tree_skipped.txt").write_text(
                str(error), encoding="utf-8"
            )


def _record_macro(
    diagnostics,
    macro_id,
    selected,
    selection,
    result,
    start_step,
    end_step,
    path_metrics,
    archive_update,
    teacher_rows,
):
    quality = path_metrics[result["terminal"].state_id]
    ratios = result["projection_ratios"]
    diagnostics.append(
        {
            "macro_id": int(macro_id),
            "start_step": int(start_step),
            "end_step": int(end_step),
            "mechanics_cost": int(result["used_steps"]),
            "parent_id": int(selected["parent_id"]),
            "terminal_state_id": int(result["terminal"].state_id),
            "requested_K": int(result["requested_k"]),
            "executed_K": int(result["executed_k"]),
            "suffix_steps": int(result["suffix_steps"]),
            "stop_reason": result["stop_reason"],
            "projection_failure": int(result["projection_failure"]),
            "minimum_projection_ratio": (
                float(min(ratios)) if ratios else float("nan")
            ),
            "collision_edges_in_macro": int(sum(result["edge_collisions"])),
            "path_collision_edges": int(quality.collision_edge_count),
            "path_geometric_length": float(quality.geometric_path_length),
            "path_collision_free": int(quality.collision_free),
            "kinematic_cell_index": (
                -1 if archive_update["cell_index"] is None
                else int(archive_update["cell_index"])
            ),
            "new_kinematic": int(archive_update["new_kinematic"]),
            "new_feasible": int(archive_update["new_feasible"]),
            "teacher_rows": int(teacher_rows),
            **selection,
        }
    )


def main():
    base_directory = Path(__file__).resolve().parent
    config = _configuration9()
    initial_node, model = _make_model(base_directory, config)
    history = create_history(
        initial_node,
        compute_gravity_potential(initial_node, config["reference_point"]),
    )
    path_metrics = initialize_path_metrics(0)
    archive = new_dual_archive()
    training = _new_training_data()
    diagnostics = []
    discovery_events = []
    kinematic_curve = [(0, 0)]
    feasible_curve = [(0, 0)]

    seed = int(config["seed"])
    candidate_rng = np.random.default_rng(seed + 9101)
    selection_rng = np.random.default_rng(seed + 9202)
    warmup_rng = np.random.default_rng(seed + 9303)
    shuffle_rng = np.random.default_rng(seed + 9404)
    np.random.seed(seed)
    used_steps = 0
    macro_id = 0
    forest = None
    macros_since_fit = 0
    fit_time = 0.0
    score_time = 0.0
    warmup_groups_completed = 0
    warmup_prefix_hash = None
    warmup_training_hash = None
    warmup_end_step = None
    start_time = time.perf_counter()

    trunk, spent, trunk_reason = _initial_trunk(
        history,
        path_metrics,
        config["total_step_budget"],
        model,
        config,
    )
    used_steps += spent
    if trunk_reason == "flat":
        update = _archive_update(
            archive, history, path_metrics, trunk.state_id, model, config
        )
        _append_discovery_events(
            discovery_events, update, trunk.state_id, used_steps, -1, -1
        )
        if update["new_kinematic"]:
            kinematic_curve.append((used_steps, len(archive)))
        if update["new_feasible"]:
            feasible_curve.append((used_steps, dual_archive_counts(archive)[1]))

    if config["paired_warmup_groups"] == 0:
        warmup_end_step = int(used_steps)
        warmup_prefix_hash = _hash_state(history, path_metrics, archive)
        warmup_training_hash = _hash_training(training)

    while used_steps < config["total_step_budget"]:
        pool = _candidate_pool(
            history, path_metrics, model, config, candidate_rng
        )
        if not pool:
            break

        if warmup_groups_completed < config["paired_warmup_groups"]:
            group_ids = sorted({item["group_id"] for item in pool})
            group_id = int(warmup_rng.choice(group_ids))
            indices = []
            for target_k in config["warmup_k_values"]:
                index = _nearest_group_k(pool, group_id, target_k)
                if index is not None and index not in indices:
                    indices.append(index)
            complete_group = True
            for branch_index, selected_index in enumerate(indices):
                if used_steps >= config["total_step_budget"]:
                    complete_group = False
                    break
                selected = pool[selected_index]
                selection = {
                    "selection_mode": "paired_warmup",
                    "selection_code": 10 + branch_index,
                    "candidate_count": len(pool),
                    "candidate_pool_sha256": _pool_hash(pool),
                    "rejected_count": 0,
                    "raw_rejected_count": 0,
                    "retained_count": len(pool),
                    "reject_cap": int(
                        np.floor(config["max_reject_fraction"] * len(pool))
                    ),
                    "reject_cap_applied": False,
                    "epsilon_bypass": False,
                    "epsilon_draw": float("nan"),
                    "forest_ready": False,
                    "predicted_cell": -1,
                    "max_probability": float("nan"),
                    "support_ratio": float("nan"),
                }
                macro_start = int(used_steps)
                result = _execute_macro(
                    history,
                    path_metrics,
                    selected,
                    config["total_step_budget"] - used_steps,
                    model,
                    config,
                )
                used_steps += result["used_steps"]
                update = _archive_update(
                    archive,
                    history,
                    path_metrics,
                    result["terminal"].state_id,
                    model,
                    config,
                )
                _append_discovery_events(
                    discovery_events,
                    update,
                    result["terminal"].state_id,
                    used_steps,
                    macro_id,
                    selection["selection_code"],
                )
                label = update["cell_index"] if update["accepted"] else -1
                rows = _append_teachers(
                    training,
                    history,
                    path_metrics,
                    result,
                    macro_id,
                    macro_start,
                    used_steps,
                    label,
                    model,
                    config,
                )
                _record_macro(
                    diagnostics, macro_id, selected, selection, result,
                    macro_start, used_steps, path_metrics, update, rows
                )
                if update["new_kinematic"]:
                    kinematic_curve.append((used_steps, len(archive)))
                if update["new_feasible"]:
                    feasible_curve.append(
                        (used_steps, dual_archive_counts(archive)[1])
                    )
                macro_id += 1
            if complete_group:
                warmup_groups_completed += 1
            if warmup_groups_completed >= config["paired_warmup_groups"]:
                warmup_end_step = int(used_steps)
                warmup_prefix_hash = _hash_state(history, path_metrics, archive)
                warmup_training_hash = _hash_training(training)
            continue

        if (
            config["selector"] != "random"
            and (forest is None or macros_since_fit >= config["refit_macro_interval"])
        ):
            fit_start = time.perf_counter()
            forest = _fit_filter(training, config, seed + 9505 + macro_id)
            fit_time += time.perf_counter() - fit_start
            macros_since_fit = 0

        score_start = time.perf_counter()
        selected_index, selection = _select(
            pool,
            config["selector"],
            forest,
            archive,
            selection_rng,
            shuffle_rng,
            config,
        )
        score_time += time.perf_counter() - score_start
        selected = pool[selected_index]
        macro_start = int(used_steps)
        result = _execute_macro(
            history,
            path_metrics,
            selected,
            config["total_step_budget"] - used_steps,
            model,
            config,
        )
        used_steps += result["used_steps"]
        update = _archive_update(
            archive,
            history,
            path_metrics,
            result["terminal"].state_id,
            model,
            config,
        )
        _append_discovery_events(
            discovery_events,
            update,
            result["terminal"].state_id,
            used_steps,
            macro_id,
            selection["selection_code"],
        )
        label = update["cell_index"] if update["accepted"] else -1
        rows = _append_teachers(
            training,
            history,
            path_metrics,
            result,
            macro_id,
            macro_start,
            used_steps,
            label,
            model,
            config,
        )
        _record_macro(
            diagnostics, macro_id, selected, selection, result,
            macro_start, used_steps, path_metrics, update, rows
        )
        if update["new_kinematic"]:
            kinematic_curve.append((used_steps, len(archive)))
        if update["new_feasible"]:
            feasible_curve.append((used_steps, dual_archive_counts(archive)[1]))
        macro_id += 1
        macros_since_fit += 1
        if config["verbose"]:
            kinematic_count, feasible_count = dual_archive_counts(archive)
            print(
                f"step={used_steps} macro={macro_id} "
                f"kinematic={kinematic_count} feasible={feasible_count} "
                f"stop={result['stop_reason']}"
            )

    if warmup_end_step is None:
        warmup_end_step = int(used_steps)
        warmup_prefix_hash = _hash_state(history, path_metrics, archive)
        warmup_training_hash = _hash_training(training)
    kinematic_count, feasible_count = dual_archive_counts(archive)
    kinematic_curve = _curve_finished(
        kinematic_curve, config["total_step_budget"], kinematic_count
    )
    feasible_curve = _curve_finished(
        feasible_curve, config["total_step_budget"], feasible_count
    )
    summary = {
        "algorithm_version": config["algorithm_version"],
        "selector": config["selector"],
        "seed": seed,
        "budget": int(config["total_step_budget"]),
        "used_steps": int(used_steps),
        "state_count": int(len(history)),
        "macro_count": int(macro_id),
        "kinematic_cells": int(kinematic_count),
        "feasible_cells": int(feasible_count),
        "kinematic_auc": float(
            _discovery_auc(kinematic_curve, config["total_step_budget"])
        ),
        "feasible_auc": float(
            _discovery_auc(feasible_curve, config["total_step_budget"])
        ),
        "training_rows": int(len(training["cell_labels"])),
        "valid_cell_training_rows": int(
            np.sum(np.asarray(training["cell_labels"], dtype=int) >= 0)
        ),
        "warmup_groups_completed": int(warmup_groups_completed),
        "warmup_end_step": int(warmup_end_step),
        "warmup_prefix_sha256": warmup_prefix_hash,
        "warmup_training_sha256": warmup_training_hash,
        "final_state_sha256": _hash_state(history, path_metrics, archive),
        "final_training_sha256": _hash_training(training),
        "fit_time_seconds": float(fit_time),
        "score_time_seconds": float(score_time),
        "wall_time_seconds": float(time.perf_counter() - start_time),
    }
    output_directory = (
        base_directory / "result" / config["result_name"] / f"seed_{seed:03d}"
    )
    _save_outputs(
        output_directory,
        history,
        archive,
        kinematic_curve,
        feasible_curve,
        diagnostics,
        discovery_events,
        training,
        config,
        model,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
