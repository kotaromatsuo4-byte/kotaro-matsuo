"""Offline paired panel for the variable-K persistent macro.

For every panel group, one parent state ``q`` and one first physical tangent
displacement ``v1`` are fixed.  All requested values of ``K`` are then run as
independent branches from that same ``(q, v1)``.  This answers two questions
without mixing them with the changing online search tree:

1. Does changing only K actually change flat/collision outcomes or terminal
   archive cells?
2. Can a GP trained on other complete ``(q, v1)`` groups rank the K values of
   a held-out group better than uniform selection?

Two representations are compared.

``raw_coefficient_gp``
    Parent geometry, path context, the SVD coefficient vector used to make
    ``v1``, and K.  It is low dimensional but depends on the returned SVD
    basis, so it is only a reference model.

``basis_free_gp``
    The exact physical main8 representation: parent geometry, path context,
    physical nodal displacement ``v1``, and log-scaled K.  It is invariant to
    sign changes and orthogonal rotations of the SVD null-space basis.

All K rows from one ``(q, v1)`` group are kept in the same cross-validation
fold.  Score-shuffled controls permute held-out scores *within each group*, so
they retain the same score multiset and shortlist size while destroying only
the correspondence between score and K.

This file imports helpers from main8 but does not modify main8, gp, or any
other existing source file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

from angle_macro import tangent_basis
from archive import find_elite_cell
from descriptor import terminal_descriptor
from gp import (
    fit_surrogate_models,
    make_terminal_output_spec,
    score_acquisition,
    terminal_target_from_history,
)
from main5 import _initial_deterministic_trunk, _update_archive
from main7 import _centered_free_coordinates, _last_nonflat_context
from main8 import _configuration8, _execute_macro, _macro_input, _make_model8
from mechanics import compute_gravity_potential
from state import create_history


METHODS = (
    "raw_coefficient_gp",
    "basis_free_gp",
    "raw_coefficient_gp_score_shuffled",
    "basis_free_gp_score_shuffled",
)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _choose_evenly_spaced(values, maximum):
    values = np.asarray(values, dtype=int)
    if len(values) <= int(maximum):
        return values.tolist()
    positions = np.rint(
        np.linspace(0, len(values) - 1, int(maximum))
    ).astype(int)
    return values[np.unique(positions)].tolist()


def _panel_configuration(args):
    config = _configuration8()
    k_values = sorted(set(int(value) for value in args.k_values))
    if not k_values or min(k_values) <= 0:
        raise ValueError("all panel K values must be positive")
    config.update(
        {
            "selector": "variable_k_random",
            "perturbation_scale": float(args.perturbation_scale),
            "max_rollout_steps": int(args.max_suffix_steps),
            "k_values": k_values,
            "k_min": int(min(k_values)),
            "k_max": int(max(k_values)),
            "state_length_scale": float(args.state_length_scale),
            "first_displacement_length_scale": float(
                args.physical_action_length_scale
            ),
            "k_length_scale": float(args.k_length_scale),
            "action_length_scale": 1.0,
        }
    )
    return config


def _raw_coefficient_input(
    parent_node,
    context_node,
    coefficient,
    mode_dimension,
    maximum_mode_dimension,
    k,
    model,
    config,
    coefficient_length_scale,
):
    """Return a fixed-size, deliberately SVD-basis-dependent input."""
    q_part = _centered_free_coordinates(parent_node, model, config)
    context_part = _centered_free_coordinates(context_node, model, config)
    padded = np.zeros(int(maximum_mode_dimension), dtype=float)
    padded[: int(mode_dimension)] = np.asarray(coefficient, dtype=float)

    action_dimension = int(maximum_mode_dimension) + 1
    coefficient_part = (
        np.sqrt(action_dimension / max(int(maximum_mode_dimension), 1))
        * padded
        / float(coefficient_length_scale)
    )
    log_k_raw = np.log1p(float(k)) / np.log1p(float(config["k_max"]))
    k_part = (
        np.sqrt(action_dimension)
        * log_k_raw
        / float(config["k_length_scale"])
    )
    return np.concatenate((q_part, context_part, coefficient_part, [k_part]))


def _terminal_cell(
    archive,
    history,
    terminal_state_id,
    model,
    config,
    add_if_missing,
):
    shape, layer_key = terminal_descriptor(
        history,
        int(terminal_state_id),
        model["member"],
        model["length_scale"],
        config["layer_threshold"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
    )
    if any(upper_member == -1 for _, _, upper_member in layer_key):
        return -1
    cell = find_elite_cell(
        archive,
        shape,
        layer_key,
        config["rms_tolerance"],
        config["max_tolerance"],
    )
    if cell is not None or not add_if_missing:
        return -1 if cell is None else int(cell)
    _update_archive(
        archive,
        history,
        int(terminal_state_id),
        model,
        config,
    )
    cell = find_elite_cell(
        archive,
        shape,
        layer_key,
        config["rms_tolerance"],
        config["max_tolerance"],
    )
    return -1 if cell is None else int(cell)


def _make_group_specs(history, parent_ids, model, config, groups_per_parent, rng):
    specs = []
    for parent_slot, parent_id_value in enumerate(parent_ids):
        parent_id = int(parent_id_value)
        parent = history[parent_id]
        basis = tangent_basis(parent.node, model)
        mode_dimension = int(basis.shape[1])
        if mode_dimension == 0:
            continue
        for direction_index in range(int(groups_per_parent)):
            coefficient = rng.normal(size=mode_dimension)
            coefficient /= max(float(np.linalg.norm(coefficient)), 1.0e-12)
            physical = basis @ coefficient
            physical *= config["perturbation_scale"] / max(
                float(np.linalg.norm(physical)), 1.0e-12
            )
            specs.append(
                {
                    "group_id": len(specs),
                    "parent_slot": int(parent_slot),
                    "parent_id": parent_id,
                    "parent_depth": int(parent.depth),
                    "parent_potential": float(parent.potential),
                    "direction_index": int(direction_index),
                    "mode_dimension": mode_dimension,
                    "coefficient": coefficient.copy(),
                    "v1": physical.reshape(model["num_node"], 3),
                }
            )
    return specs


def generate_panel(seed, base_directory, args):
    config = _panel_configuration(args)
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
    trunk_terminal, trunk_ids, trunk_steps, trunk_reason = (
        _initial_deterministic_trunk(
            history,
            int(args.trunk_budget),
            model,
            config,
        )
    )

    baseline_archive = []
    if trunk_terminal is not None and trunk_reason == "flat":
        _update_archive(
            baseline_archive,
            history,
            trunk_terminal.state_id,
            model,
            config,
        )
    full_archive = [dict(cell) for cell in baseline_archive]

    eligible_parent_ids = [
        state.state_id for state in history if state.terminal_reason is None
    ]
    parent_ids = _choose_evenly_spaced(eligible_parent_ids, args.parents)
    rng = np.random.default_rng(int(seed) + 8801)
    specs = _make_group_specs(
        history,
        parent_ids,
        model,
        config,
        args.v1_per_parent,
        rng,
    )
    if len(specs) < 2:
        raise RuntimeError("at least two feasible (q, v1) groups are required")
    maximum_mode_dimension = max(spec["mode_dimension"] for spec in specs)

    rows = []
    total_panel_mechanics = 0
    for spec in specs:
        parent = history[spec["parent_id"]]
        context_node = _last_nonflat_context(
            history,
            spec["parent_id"],
            config["layer_threshold"],
        )
        for k in config["k_values"]:
            basis_free_input = _macro_input(
                parent.node,
                context_node,
                spec["v1"],
                int(k),
                model,
                config,
            )
            raw_input = _raw_coefficient_input(
                parent.node,
                context_node,
                spec["coefficient"],
                spec["mode_dimension"],
                maximum_mode_dimension,
                int(k),
                model,
                config,
                args.raw_coefficient_length_scale,
            )
            candidate = {
                "parent_id": spec["parent_id"],
                "v1": spec["v1"],
                "k": int(k),
            }
            result = _execute_macro(
                history,
                candidate,
                int(k) + config["max_rollout_steps"],
                model,
                config,
            )
            total_panel_mechanics += int(result["used_steps"])
            state_ids = result["new_ids"]
            if state_ids:
                minimum_potential = min(
                    float(history[state_id].potential) for state_id in state_ids
                )
                flatness_target = float(
                    np.log1p(
                        minimum_potential
                        / max(config["flatness_tolerance"], 1.0e-12)
                    )
                )
            else:
                flatness_target = float("nan")

            flat = result["stop_reason"] == "flat"
            terminal_cell = -1
            existing_cell = -1
            if flat:
                existing_cell = _terminal_cell(
                    baseline_archive,
                    history,
                    result["terminal"].state_id,
                    model,
                    config,
                    False,
                )
                terminal_cell = _terminal_cell(
                    full_archive,
                    history,
                    result["terminal"].state_id,
                    model,
                    config,
                    True,
                )
                descriptor = terminal_target_from_history(
                    history,
                    result["terminal"].state_id,
                    config["layer_threshold"],
                    output_spec,
                )
            else:
                descriptor = np.full(
                    len(output_spec["active_positions"]), np.nan, dtype=float
                )

            rows.append(
                {
                    "seed": int(seed),
                    **{
                        key: spec[key]
                        for key in (
                            "group_id",
                            "parent_slot",
                            "parent_id",
                            "parent_depth",
                            "parent_potential",
                            "direction_index",
                            "mode_dimension",
                        )
                    },
                    "k": int(k),
                    "requested_k": int(result["requested_k"]),
                    "executed_k": int(result["executed_k"]),
                    "chain_completed": int(result["chain_completed"]),
                    "suffix_steps": int(result["suffix_steps"]),
                    "mechanics_cost": int(result["used_steps"]),
                    "stop_reason": result["stop_reason"],
                    "flat": int(flat),
                    "collision": int(result["stop_reason"] == "collision"),
                    "projection_failure": int(
                        result["stop_reason"] == "projection_failure"
                    ),
                    "archive_valid": int(terminal_cell >= 0),
                    "existing_baseline_cell": int(existing_cell >= 0),
                    "true_new": int(flat and terminal_cell >= 0 and existing_cell < 0),
                    "terminal_cell": int(terminal_cell),
                    "terminal_state_id": int(result["terminal"].state_id),
                    "flatness_target": flatness_target,
                    "minimum_projection_ratio": (
                        float(min(result["projection_ratios"]))
                        if result["projection_ratios"]
                        else np.nan
                    ),
                    "raw_feature": raw_input,
                    "basis_free_feature": basis_free_input,
                    "terminal_descriptor": descriptor,
                }
            )

    return {
        "config": config,
        "model": model,
        "output_spec": output_spec,
        "history": history,
        "rows": rows,
        "baseline_archive": baseline_archive,
        "full_archive": full_archive,
        "trunk_steps": int(trunk_steps),
        "trunk_reason": trunk_reason,
        "num_groups": int(len(specs)),
        "num_parents": int(len(parent_ids)),
        "maximum_mode_dimension": int(maximum_mode_dimension),
        "total_panel_mechanics": int(total_panel_mechanics),
    }


def _group_folds(rows, number_folds, seed):
    groups = np.asarray(sorted({row["group_id"] for row in rows}), dtype=int)
    rng = np.random.default_rng(int(seed) + 8903)
    groups = rng.permutation(groups)
    pieces = np.array_split(groups, min(int(number_folds), len(groups)))
    row_groups = np.asarray([row["group_id"] for row in rows], dtype=int)
    return [
        (
            np.flatnonzero(~np.isin(row_groups, test_groups)),
            np.flatnonzero(np.isin(row_groups, test_groups)),
        )
        for test_groups in pieces
        if len(test_groups)
    ]


def _fit_and_score(train_rows, test_rows, feature_name, panel, args, rng):
    train_x = np.asarray([row[feature_name] for row in train_rows], dtype=float)
    test_x = np.asarray([row[feature_name] for row in test_rows], dtype=float)
    descriptors = np.asarray(
        [row["terminal_descriptor"] for row in train_rows], dtype=float
    )
    surrogate = fit_surrogate_models(
        train_x,
        np.asarray([row["flat"] for row in train_rows], dtype=bool),
        np.asarray([row["flatness_target"] for row in train_rows], dtype=float),
        descriptors,
        state_dimension=panel["model"]["input_state_dimension"],
        state_length_scale=panel["config"]["state_length_scale"],
        action_length_scale=1.0,
        signal_variance=panel["config"]["signal_variance"],
        regression_noise_std=panel["config"]["regression_noise_std"],
        classifier_noise_std=panel["config"]["classifier_noise_std"],
        descriptor_scale_floor=panel["config"]["descriptor_scale_floor"],
        prior_flat_probability=0.5,
    )
    diagnostics = score_acquisition(
        surrogate,
        test_x,
        panel["baseline_archive"],
        panel["output_spec"],
        panel["model"]["member"],
        panel["config"]["overlap_tolerance"],
        panel["config"]["separation_tolerance"],
        panel["config"]["collinearity_tolerance"],
        panel["config"]["parallel_tolerance"],
        panel["config"]["rms_tolerance"],
        panel["config"]["max_tolerance"],
        flatness_target_threshold=np.log(2.0),
        acquisition_kind="ucb",
        novelty_kappa=float(args.kappa),
        reach_weight=panel["config"]["reach_weight"],
        cold_uncertainty_weight=panel["config"]["cold_uncertainty_weight"],
        layer_uncertainty_scale=panel["config"]["layer_uncertainty_scale"],
        num_posterior_samples=0,
        rng=rng,
    )
    return diagnostics


def cross_validated_scores(panel, args, seed):
    rows = panel["rows"]
    predictions = [
        {
            "row_index": int(index),
            "group_id": int(row["group_id"]),
            "k": int(row["k"]),
            "true_new": int(row["true_new"]),
            "flat": int(row["flat"]),
            "mechanics_cost": int(row["mechanics_cost"]),
        }
        for index, row in enumerate(rows)
    ]
    folds = _group_folds(rows, args.folds, seed)
    if len(folds) < 2:
        raise RuntimeError("group-wise cross-validation requires at least two folds")

    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        for method, feature_name in (
            ("raw_coefficient_gp", "raw_feature"),
            ("basis_free_gp", "basis_free_feature"),
        ):
            diagnostics = _fit_and_score(
                train_rows,
                test_rows,
                feature_name,
                panel,
                args,
                np.random.default_rng(
                    int(seed) + 9001 + 101 * fold_index + (0 if method.startswith("raw") else 1)
                ),
            )
            scores = np.asarray(diagnostics["score"], dtype=float)
            shuffled = scores.copy()
            test_groups = np.asarray(
                [row["group_id"] for row in test_rows], dtype=int
            )
            for group_id in np.unique(test_groups):
                local = np.flatnonzero(test_groups == group_id)
                shuffle_rng = np.random.default_rng(
                    int(seed)
                    + 9109
                    + 1009 * fold_index
                    + 17 * int(group_id)
                    + (0 if method.startswith("raw") else 1)
                )
                shuffled[local] = shuffle_rng.permutation(scores[local])

            for local_index, row_index in enumerate(test_indices):
                prediction = predictions[int(row_index)]
                prediction["fold"] = int(fold_index)
                prediction[f"{method}_score"] = float(scores[local_index])
                prediction[f"{method}_score_shuffled"] = float(
                    shuffled[local_index]
                )
                prediction[f"{method}_p_flat"] = float(
                    diagnostics["flat_probability"][local_index]
                )
                prediction[f"{method}_p_reach"] = float(
                    diagnostics["reach_probability"][local_index]
                )
                prediction[f"{method}_mean_novelty"] = float(
                    diagnostics["mean_novelty"][local_index]
                )
                prediction[f"{method}_uncertainty"] = float(
                    diagnostics["descriptor_uncertainty"][local_index]
                )
    return predictions


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    ranks = rankdata(scores, method="average")
    count_positive = int(np.count_nonzero(positive))
    count_negative = int(np.count_nonzero(negative))
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)


def _selection_metrics(rows, predictions, score_key):
    groups = sorted({row["group_id"] for row in rows})
    top1_new = []
    top1_cost = []
    top25_new = []
    top25_any = []
    top25_cost = []
    eligible_success = []
    reciprocal_rank = []
    for group_id in groups:
        indices = [
            index for index, row in enumerate(rows) if row["group_id"] == group_id
        ]
        scores = np.asarray([predictions[index][score_key] for index in indices])
        utilities = np.asarray([rows[index]["true_new"] for index in indices])
        costs = np.asarray([rows[index]["mechanics_cost"] for index in indices])
        maximum = np.max(scores)
        tied = np.flatnonzero(
            np.isclose(scores, maximum, rtol=1.0e-10, atol=1.0e-12)
        )
        top1_new.append(float(np.mean(utilities[tied])))
        top1_cost.append(float(np.mean(costs[tied])))
        keep = max(1, int(np.ceil(0.25 * len(indices))))
        top = np.argsort(-scores, kind="stable")[:keep]
        top25_new.append(float(np.mean(utilities[top])))
        top25_any.append(float(np.any(utilities[top])))
        top25_cost.append(float(np.mean(costs[top])))
        positive = np.flatnonzero(utilities > 0)
        if len(positive):
            ranks = rankdata(-scores, method="average")
            best_rank = float(np.min(ranks[positive]))
            reciprocal_rank.append(1.0 / best_rank)
            eligible_success.append(float(np.any(np.isin(tied, positive))))

    labels = [row["true_new"] for row in rows]
    scores = [prediction[score_key] for prediction in predictions]
    return {
        "row_new_cell_auc": _binary_auc(labels, scores),
        "group_top1_new_rate": float(np.mean(top1_new)),
        "group_top1_mean_cost": float(np.mean(top1_cost)),
        "group_top25_mean_new_rate": float(np.mean(top25_new)),
        "group_top25_any_new_rate": float(np.mean(top25_any)),
        "group_top25_mean_cost": float(np.mean(top25_cost)),
        "eligible_group_top1_success_rate": (
            float(np.mean(eligible_success)) if eligible_success else float("nan")
        ),
        "eligible_group_mean_reciprocal_rank": (
            float(np.mean(reciprocal_rank)) if reciprocal_rank else float("nan")
        ),
        "eligible_groups": int(len(eligible_success)),
    }


def summarize_panel(seed, panel, predictions, args):
    rows = panel["rows"]
    groups = sorted({row["group_id"] for row in rows})
    uniform_new = []
    uniform_cost = []
    oracle_top1 = []
    oracle_top25 = []
    cell_changes = []
    outcome_changes = []
    flat_changes = []
    distinct_flat_cells = []
    for group_id in groups:
        group = [row for row in rows if row["group_id"] == group_id]
        utility = np.asarray([row["true_new"] for row in group], dtype=float)
        uniform_new.append(float(np.mean(utility)))
        uniform_cost.append(float(np.mean([row["mechanics_cost"] for row in group])))
        oracle_top1.append(float(np.max(utility)))
        keep = max(1, int(np.ceil(0.25 * len(group))))
        oracle_top25.append(float(np.mean(np.sort(utility)[-keep:])))
        cells = {row["terminal_cell"] for row in group if row["terminal_cell"] >= 0}
        labels = {
            f"cell:{row['terminal_cell']}"
            if row["terminal_cell"] >= 0
            else f"stop:{row['stop_reason']}"
            for row in group
        }
        cell_changes.append(float(len(cells) >= 2))
        outcome_changes.append(float(len(labels) >= 2))
        flat_changes.append(float(len({row["flat"] for row in group}) >= 2))
        distinct_flat_cells.append(float(len(cells)))

    method_scores = {
        "raw_coefficient_gp": "raw_coefficient_gp_score",
        "basis_free_gp": "basis_free_gp_score",
        "raw_coefficient_gp_score_shuffled": (
            "raw_coefficient_gp_score_shuffled"
        ),
        "basis_free_gp_score_shuffled": (
            "basis_free_gp_score_shuffled"
        ),
    }
    model_statistics = {
        method: _selection_metrics(rows, predictions, score_key)
        for method, score_key in method_scores.items()
    }

    k_statistics = []
    for k in panel["config"]["k_values"]:
        selected = [row for row in rows if row["k"] == int(k)]
        k_statistics.append(
            {
                "k": int(k),
                "count": int(len(selected)),
                "flat_rate": float(np.mean([row["flat"] for row in selected])),
                "collision_rate": float(
                    np.mean([row["collision"] for row in selected])
                ),
                "projection_failure_rate": float(
                    np.mean([row["projection_failure"] for row in selected])
                ),
                "new_rate": float(
                    np.mean([row["true_new"] for row in selected])
                ),
                "mean_executed_k": float(
                    np.mean([row["executed_k"] for row in selected])
                ),
                "mean_mechanics_cost": float(
                    np.mean([row["mechanics_cost"] for row in selected])
                ),
            }
        )

    return {
        "seed": int(seed),
        "num_groups": int(len(groups)),
        "num_rows": int(len(rows)),
        "num_parents": int(panel["num_parents"]),
        "k_values": [int(value) for value in panel["config"]["k_values"]],
        "trunk_steps": int(panel["trunk_steps"]),
        "trunk_stop_reason": panel["trunk_reason"],
        "baseline_cell_count": int(len(panel["baseline_archive"])),
        "full_panel_cell_count": int(len(panel["full_archive"])),
        "total_panel_mechanics": int(panel["total_panel_mechanics"]),
        "flat_rate": float(np.mean([row["flat"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in rows)),
        "group_truth": {
            "uniform_random_new_rate": float(np.mean(uniform_new)),
            "uniform_random_mean_cost": float(np.mean(uniform_cost)),
            "oracle_top1_new_rate": float(np.mean(oracle_top1)),
            "oracle_top25_mean_new_rate": float(np.mean(oracle_top25)),
            "terminal_cell_changes_with_k_group_fraction": float(
                np.mean(cell_changes)
            ),
            "any_outcome_changes_with_k_group_fraction": float(
                np.mean(outcome_changes)
            ),
            "flat_status_changes_with_k_group_fraction": float(
                np.mean(flat_changes)
            ),
            "mean_distinct_flat_cells_per_group": float(
                np.mean(distinct_flat_cells)
            ),
            "max_distinct_flat_cells_in_group": int(max(distinct_flat_cells)),
        },
        "model_statistics": model_statistics,
        "k_statistics": k_statistics,
        "configuration": {
            "parents": int(args.parents),
            "v1_per_parent": int(args.v1_per_parent),
            "folds": int(args.folds),
            "perturbation_scale": float(args.perturbation_scale),
            "max_suffix_steps": int(args.max_suffix_steps),
            "state_length_scale": float(args.state_length_scale),
            "physical_action_length_scale": float(
                args.physical_action_length_scale
            ),
            "raw_coefficient_length_scale": float(
                args.raw_coefficient_length_scale
            ),
            "k_length_scale": float(args.k_length_scale),
            "ucb_kappa": float(args.kappa),
            "cv_split": "all K of one (q,v1) group stay in one fold",
            "primary_model": "basis_free_gp",
        },
    }


def run_seed(seed, base_directory, result_directory, args):
    start = time.perf_counter()
    panel = generate_panel(seed, base_directory, args)
    predictions = cross_validated_scores(panel, args, seed)
    summary = summarize_panel(seed, panel, predictions, args)
    summary["wall_time_seconds"] = float(time.perf_counter() - start)

    seed_directory = result_directory / f"seed_{int(seed):03d}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    compact_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in (
                "raw_feature",
                "basis_free_feature",
                "terminal_descriptor",
            )
        }
        for row in panel["rows"]
    ]
    _write_csv(seed_directory / "panel_outcomes.csv", compact_rows)
    _write_csv(seed_directory / "cv_predictions.csv", predictions)
    np.savez_compressed(
        seed_directory / "panel_arrays.npz",
        raw_features=np.asarray(
            [row["raw_feature"] for row in panel["rows"]], dtype=float
        ),
        basis_free_features=np.asarray(
            [row["basis_free_feature"] for row in panel["rows"]], dtype=float
        ),
        terminal_descriptors=np.asarray(
            [row["terminal_descriptor"] for row in panel["rows"]], dtype=float
        ),
        group_ids=np.asarray(
            [row["group_id"] for row in panel["rows"]], dtype=int
        ),
        k=np.asarray([row["k"] for row in panel["rows"]], dtype=int),
    )
    (seed_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def aggregate_summaries(summaries):
    def nanmean_or_nan(values):
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if len(finite) else float("nan")

    aggregate = {
        "num_seeds": int(len(summaries)),
        "seeds": [int(summary["seed"]) for summary in summaries],
        "total_groups": int(sum(summary["num_groups"] for summary in summaries)),
        "mean_panel_cells": float(
            np.mean([summary["full_panel_cell_count"] for summary in summaries])
        ),
        "mean_total_panel_mechanics": float(
            np.mean([summary["total_panel_mechanics"] for summary in summaries])
        ),
        "group_truth": {},
        "models": {},
    }
    for key in summaries[0]["group_truth"]:
        aggregate["group_truth"][key] = nanmean_or_nan(
            [summary["group_truth"][key] for summary in summaries]
        )
    for method in METHODS:
        aggregate["models"][method] = {}
        for key in summaries[0]["model_statistics"][method]:
            values = [
                summary["model_statistics"][method][key]
                for summary in summaries
            ]
            if key == "eligible_groups":
                aggregate["models"][method][key] = int(sum(values))
            else:
                aggregate["models"][method][key] = nanmean_or_nan(values)

    aggregate["k_statistics"] = []
    for k in summaries[0]["k_values"]:
        per_seed = [
            next(row for row in summary["k_statistics"] if row["k"] == k)
            for summary in summaries
        ]
        aggregate["k_statistics"].append(
            {
                "k": int(k),
                "total_count": int(sum(row["count"] for row in per_seed)),
                "mean_flat_rate": float(
                    np.mean([row["flat_rate"] for row in per_seed])
                ),
                "mean_collision_rate": float(
                    np.mean([row["collision_rate"] for row in per_seed])
                ),
                "mean_projection_failure_rate": float(
                    np.mean(
                        [row["projection_failure_rate"] for row in per_seed]
                    )
                ),
                "mean_new_rate": float(
                    np.mean([row["new_rate"] for row in per_seed])
                ),
                "mean_executed_k": float(
                    np.mean([row["mean_executed_k"] for row in per_seed])
                ),
                "mean_mechanics_cost": float(
                    np.mean([row["mean_mechanics_cost"] for row in per_seed])
                ),
            }
        )
    return aggregate


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[700, 701, 702])
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--v1-per-parent", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(range(1, 25)))
    parser.add_argument("--perturbation-scale", type=float, default=0.20)
    parser.add_argument("--max-suffix-steps", type=int, default=100)
    parser.add_argument("--trunk-budget", type=int, default=100)
    parser.add_argument("--state-length-scale", type=float, default=0.50)
    parser.add_argument("--physical-action-length-scale", type=float, default=0.50)
    parser.add_argument("--raw-coefficient-length-scale", type=float, default=0.75)
    parser.add_argument("--k-length-scale", type=float, default=0.35)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--result-name", default="persistent_k_panel_pilot")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.parents <= 0 or args.v1_per_parent <= 0:
        raise ValueError("parents and v1-per-parent must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if args.max_suffix_steps <= 0:
        raise ValueError("max-suffix-steps must be positive")
    base_directory = Path(__file__).resolve().parent
    result_directory = base_directory / "result" / args.result_name
    result_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        summary = run_seed(seed, base_directory, result_directory, args)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    aggregate = aggregate_summaries(summaries)
    (result_directory / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
