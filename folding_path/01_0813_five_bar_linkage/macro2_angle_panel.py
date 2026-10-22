"""Offline paired panel for the second action of a fixed-K=2 macro.

This file is deliberately independent of ``main5.py``.  It asks a narrow
diagnostic question before changing the online search algorithm:

    For the same parent state q and the same first perturbation v1, does a
    second perturbation that continues v1 produce useful terminal cells more
    often than a rotated or reversed second perturbation?

The first corrector is shared inside a panel group.  A physical reference is
sampled orthogonally to v1 at q, transported to the corrected child q1, and
Gram--Schmidt orthogonalized against the transported continuation direction.
The second action is

    v2(theta) = rho (cos(theta) p + sin(theta) u).

All theta values are then rolled out independently.  Cross-validation holds
out each corrected q1 group, never individual theta rows.  Three GPs are
compared:

``raw``
    [q1, v2], matching the one-step physical state/action representation.
``angle_aware``
    [q1, projected-v1, physical-perpendicular, v2, cos(theta), sin(theta),
    projection ratios, availability].  This matches the online K=2 angle
    selector's explicit physical context.
``feature_shuffled``
    The angle-aware representation, but its direction-memory context training
    rows are permuted while q1 and v2 remain correct.  Test inputs remain
    physically correct.  This is a negative control for merely increasing the
    input dimension.

The GP-UCB-like diagnostic is

    p_flat(x) [d_A(mu_D(x)) + kappa sqrt(mean(var_D(x)))],

where A is the frozen archive produced by the initial deterministic trunk.
It is an offline ranking diagnostic, not a replacement for the cell-exact
archive check.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

from archive import find_elite_cell, update_elite_archive
from angle_macro import (
    macro_gp_input,
    sample_orthogonal_reference,
    second_displacement,
    transported_frame,
)
from descriptor import terminal_continuous_descriptor, terminal_descriptor
from gp import (
    fit_binary_probability_gp,
    fit_shared_kernel_gp,
    predict_binary_probability,
    predict_shared_kernel_gp,
)
from mechanics import calc_dedx, calc_mode, compute_gravity_potential
from roll import deterministic_roll, displacement_roll
from state import create_history


DEFAULT_ANGLES_DEG = tuple(
    np.rad2deg(np.linspace(-np.pi, np.pi, 16, endpoint=False))
)


def load_model(model_directory):
    """Read the same three model files as the main search."""
    node = np.atleast_2d(
        np.loadtxt(model_directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(model_directory / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(model_directory / "fix.dat", delimiter=",", dtype=int)
    )
    index_all = np.arange(3 * len(node))
    index_fixed = 3 * fix[:, 0] + fix[:, 1]
    index_free = np.delete(index_all, index_fixed)
    member_length = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    return {
        "node_initial": node,
        "member": member,
        "index_free": index_free,
        "member_length": member_length,
        "num_node": len(node),
        "num_member": len(member),
        "length_scale": float(np.mean(member_length)),
    }


def default_config(perturbation_scale, max_suffix_steps):
    """Numerical settings matched to the current small-step experiments."""
    return {
        "reference_point": 0.0,
        "constraint_tolerance": 1.0e-10,
        "beta0": 1.0,
        "corrector_iterations": 10,
        "line_search_iterations": 10,
        "line_search_tau": 0.5,
        "cma_population": 6,
        "cma_generations": 10,
        "cma_seed": 0,
        "target_norm_max": 0.10,
        "perturbation_scale": float(perturbation_scale),
        "max_suffix_steps": int(max_suffix_steps),
        "flatness_tolerance": 0.05,
        "dist_tol": 0.05,
        "normal_tol": 0.10,
        "cos_tol": 0.99,
        "collision_model": "planar",
        "layer_threshold": 0.10,
        "overlap_tolerance": 0.05,
        "separation_tolerance": 1.0e-5,
        "collinearity_tolerance": 0.02,
        "parallel_tolerance": 0.02,
        "rms_tolerance": 0.02,
        "max_tolerance": 0.05,
    }


def mechanism_mode(node, model):
    jacobian = calc_dedx(
        node,
        model["num_node"],
        model["member"],
        model["num_member"],
        model["index_free"],
    )
    return calc_mode(
        model["index_free"],
        jacobian,
        model["num_node"],
    )


def deterministic_once(history, parent_id, model, config):
    return deterministic_roll(
        history,
        int(parent_id),
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
        config["collision_model"],
    )


def displacement_once(history, parent_id, displacement, model, config):
    return displacement_roll(
        history,
        int(parent_id),
        np.asarray(displacement, dtype=float),
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


def initial_trunk(history, model, config):
    """Generate one deterministic root-to-terminal trunk."""
    current = history[0]
    nonterminal_ids = [0]
    steps = 0
    while current.terminal_reason is None and steps < config["max_suffix_steps"]:
        current = deterministic_once(history, current.state_id, model, config)
        steps += 1
        if current.terminal_reason is None:
            nonterminal_ids.append(current.state_id)
    return current, nonterminal_ids, steps


def rollout_second_angle(
    history,
    first_child_id,
    second_displacement_value,
    model,
    config,
):
    """Execute v2(theta), then one deterministic suffix."""
    second = np.asarray(second_displacement_value, dtype=float).reshape(
        model["num_node"], 3
    )
    current = displacement_once(
        history, first_child_id, second, model, config
    )
    steps_after_first = 1
    while (
        current.terminal_reason is None
        and steps_after_first < config["max_suffix_steps"]
    ):
        current = deterministic_once(history, current.state_id, model, config)
        steps_after_first += 1
    stop_reason = current.terminal_reason
    if stop_reason is None:
        stop_reason = "timeout"
    return current, second, steps_after_first, stop_reason


def archive_cell(elite_archive, history, terminal_state_id, model, config):
    """Return the exact archive cell, adding it if required."""
    shape, layer = terminal_descriptor(
        history,
        terminal_state_id,
        model["member"],
        model["length_scale"],
        config["layer_threshold"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
    )
    if any(upper == -1 for _, _, upper in layer):
        return -1
    cell = find_elite_cell(
        elite_archive,
        shape,
        layer,
        config["rms_tolerance"],
        config["max_tolerance"],
    )
    if cell is not None:
        return int(cell)
    update_elite_archive(
        elite_archive,
        history,
        terminal_state_id,
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
    return len(elite_archive) - 1


def normalized_state_action(node, displacement, model, config):
    """Return q_free/L and v_free/L."""
    shifted = np.asarray(node, dtype=float).copy()
    shifted[:, 2] -= config["reference_point"]
    free = model["index_free"]
    state = shifted.reshape(-1)[free] / model["length_scale"]
    action = np.asarray(displacement).reshape(-1)[free] / model["length_scale"]
    return state, action


def choose_parent_ids(nonterminal_ids, number):
    """Spread panel parents over potential progress on the trunk."""
    ids = np.asarray(nonterminal_ids, dtype=int)
    if len(ids) <= int(number):
        return ids.tolist()
    positions = np.rint(np.linspace(0, len(ids) - 1, int(number))).astype(int)
    return ids[np.unique(positions)].tolist()


def generate_panel(seed, model, config, num_parents, v1_per_parent, angles):
    """Generate one exhaustive paired theta panel and its frozen baseline."""
    rng = np.random.default_rng(int(seed) + 1701)
    initial_potential = compute_gravity_potential(
        model["node_initial"], config["reference_point"]
    )
    history = create_history(model["node_initial"], initial_potential)
    trunk_terminal, nonterminal_ids, trunk_steps = initial_trunk(
        history, model, config
    )

    baseline_archive = []
    if trunk_terminal.terminal_reason == "flat":
        update_elite_archive(
            baseline_archive,
            history,
            trunk_terminal.state_id,
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
    baseline_count = len(baseline_archive)
    full_archive = [dict(cell) for cell in baseline_archive]
    baseline_continuous = [
        np.asarray(cell["continuous_descriptor"], dtype=float)
        for cell in baseline_archive
    ]

    rows = []
    pair_id = 0
    parent_ids = choose_parent_ids(nonterminal_ids, num_parents)
    for parent_slot, parent_id in enumerate(parent_ids):
        parent = history[int(parent_id)]
        mode = mechanism_mode(parent.node, model)
        if mode.shape[1] < 2:
            continue

        made = 0
        attempts = 0
        while made < int(v1_per_parent) and attempts < 20 * int(v1_per_parent):
            attempts += 1
            coefficient = rng.normal(size=mode.shape[1])
            coefficient /= max(float(np.linalg.norm(coefficient)), 1.0e-12)
            first_displacement = (
                config["perturbation_scale"] * mode @ coefficient
            ).reshape(model["num_node"], 3)
            physical_reference = sample_orthogonal_reference(
                parent.node,
                first_displacement,
                model,
                config["perturbation_scale"],
                rng,
            )
            if physical_reference is None:
                continue
            first_child = displacement_once(
                history, parent_id, first_displacement, model, config
            )
            if first_child.terminal_reason is not None:
                continue
            frame = transported_frame(
                first_child.node,
                first_displacement,
                physical_reference,
                model,
                config["perturbation_scale"],
                minimum_projection_ratio=0.10,
                rng=rng,
            )
            if frame is None:
                continue

            for theta_deg in angles:
                theta_rad = np.deg2rad(float(theta_deg))
                second_candidate = second_displacement(
                    frame, theta_rad, config["perturbation_scale"]
                )
                if second_candidate is None:
                    continue
                terminal, second, after_first_steps, stop_reason = (
                    rollout_second_angle(
                        history,
                        first_child.state_id,
                        second_candidate,
                        model,
                        config,
                    )
                )
                is_flat = stop_reason == "flat"
                if is_flat:
                    continuous = terminal_continuous_descriptor(
                        history,
                        terminal.state_id,
                        model["length_scale"],
                        config["layer_threshold"],
                        config["reference_point"],
                    )
                    cell_id = archive_cell(
                        full_archive,
                        history,
                        terminal.state_id,
                        model,
                        config,
                    )
                else:
                    continuous = np.full(3 * model["num_node"], np.nan)
                    cell_id = -1
                true_new = bool(is_flat and cell_id >= baseline_count and cell_id >= 0)
                q1_feature, second_feature = normalized_state_action(
                    first_child.node, second, model, config
                )
                raw_feature = np.concatenate((q1_feature, second_feature))
                angle_feature = macro_gp_input(
                    first_child.node,
                    frame,
                    second,
                    theta_rad,
                    model["index_free"],
                    config["reference_point"],
                    model["length_scale"],
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "parent_slot": int(parent_slot),
                        "parent_id": int(parent_id),
                        "q1_state_id": int(first_child.state_id),
                        "pair_id": int(pair_id),
                        "theta_deg": float(theta_deg),
                        "cos_theta": float(np.cos(theta_rad)),
                        "sin_theta": float(np.sin(theta_rad)),
                        "projection_ratio": float(frame["projection_ratio"]),
                        "reference_projection_ratio": float(
                            frame["reference_projection_ratio"]
                        ),
                        "reference_fallback": int(frame["reference_fallback"]),
                        "stop_reason": stop_reason,
                        "flat": int(is_flat),
                        "archive_valid": int(cell_id >= 0),
                        "true_new": int(true_new),
                        "terminal_cell": int(cell_id),
                        # One first step is included because each macro would
                        # have to pay it in an online, unshared evaluation.
                        "equivalent_macro_steps": int(1 + after_first_steps),
                        "terminal_state_id": int(terminal.state_id),
                        "raw_feature": raw_feature,
                        "angle_feature": angle_feature,
                        "terminal_descriptor": continuous,
                    }
                )
            pair_id += 1
            made += 1

    return {
        "history": history,
        "rows": rows,
        "baseline_archive": baseline_archive,
        "full_archive": full_archive,
        "baseline_continuous": baseline_continuous,
        "trunk_steps": int(trunk_steps),
        "trunk_stop_reason": trunk_terminal.terminal_reason,
        "num_requested_parents": int(num_parents),
        "num_actual_parents": int(len({row["parent_id"] for row in rows})),
        "num_pairs": int(len({row["pair_id"] for row in rows})),
    }


def fit_scaler(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.maximum(scale, 0.05)
    return mean, scale


def transform(values, mean, scale):
    return (np.asarray(values, dtype=float) - mean) / scale


def fit_panel_gp(train_x, flat, descriptors, state_dimension):
    """Fit flat ranking and flat-conditional descriptor GPs."""
    flat_model = fit_binary_probability_gp(
        train_x,
        flat,
        state_dimension,
        1.0,
        1.0,
        signal_variance=1.0,
        noise_std=0.25,
        prior_probability=0.5,
    )
    descriptor_mask = np.asarray(flat, dtype=bool) & np.all(
        np.isfinite(descriptors), axis=1
    )
    descriptor_model = None
    if np.count_nonzero(descriptor_mask) >= 2:
        descriptor_model = fit_shared_kernel_gp(
            train_x[descriptor_mask],
            descriptors[descriptor_mask],
            state_dimension,
            1.0,
            1.0,
            signal_variance=1.0,
            noise_std=0.08,
            output_scale_floor=0.05,
        )
    return flat_model, descriptor_model


def gp_ucb_scores(flat_model, descriptor_model, test_x, baseline, kappa):
    p_flat, _ = predict_binary_probability(flat_model, test_x)
    if descriptor_model is None:
        zeros = np.zeros(len(test_x), dtype=float)
        return p_flat * zeros, p_flat, zeros, zeros
    mean, variance = predict_shared_kernel_gp(descriptor_model, test_x)
    if baseline:
        references = np.asarray(baseline, dtype=float)
        distance = np.min(
            np.sqrt(np.mean((mean[:, None, :] - references[None, :, :]) ** 2, axis=2)),
            axis=1,
        )
    else:
        distance = np.sqrt(np.mean(mean * mean, axis=1))
    uncertainty = np.sqrt(np.mean(np.maximum(variance, 0.0), axis=1))
    score = p_flat * (distance + float(kappa) * uncertainty)
    return score, p_flat, distance, uncertainty


def parent_group_folds(rows, folds, seed):
    # pair_id uniquely identifies one corrected q1.  All sixteen theta rows
    # from that q1 are held out together.
    groups = np.asarray([row["pair_id"] for row in rows], dtype=int)
    unique = np.unique(groups)
    if len(unique) < 2:
        return []
    rng = np.random.default_rng(int(seed) + 2903)
    rng.shuffle(unique)
    pieces = np.array_split(unique, min(int(folds), len(unique)))
    return [
        (np.flatnonzero(~np.isin(groups, test_groups)), np.flatnonzero(np.isin(groups, test_groups)))
        for test_groups in pieces
        if len(test_groups)
    ]


def cross_validated_predictions(rows, baseline, num_free, folds, seed, kappa):
    """Return held-out scores for raw, angle-aware, and shuffled inputs."""
    raw = np.asarray([row["raw_feature"] for row in rows], dtype=float)
    angle = np.asarray([row["angle_feature"] for row in rows], dtype=float)
    flat = np.asarray([row["flat"] for row in rows], dtype=bool)
    descriptors = np.asarray(
        [row["terminal_descriptor"] for row in rows], dtype=float
    )
    results = []
    for fold_index, (train, test) in enumerate(
        parent_group_folds(rows, folds, seed)
    ):
        for method in ("raw", "angle_aware", "feature_shuffled"):
            source = raw if method == "raw" else angle
            train_source = source[train].copy()
            if method == "feature_shuffled":
                # Keep q1 and candidate v2 physically correct, but jointly
                # permute projected-v1, the perpendicular reference, and all
                # scalar angle context.  Thus the shuffled model has the same
                # dimension and marginal feature distribution as angle_aware.
                nf = int(num_free)
                context_columns = np.concatenate(
                    (
                        np.arange(nf, 3 * nf),
                        np.arange(4 * nf, train_source.shape[1]),
                    )
                )
                permutation_rng = np.random.default_rng(
                    int(seed) + 10007 * (fold_index + 1)
                )
                permutation = permutation_rng.permutation(len(train_source))
                shuffled_context = train_source[
                    permutation[:, None], context_columns[None, :]
                ]
                train_source[:, context_columns] = shuffled_context

            mean, scale = fit_scaler(train_source)
            train_x = transform(train_source, mean, scale)
            test_x = transform(source[test], mean, scale)
            flat_model, descriptor_model = fit_panel_gp(
                train_x,
                flat[train],
                descriptors[train],
                # raw=[q1 | v2], while angle-aware follows the online
                # kernel split [q1,p,u | v2,angle/scalar context].
                state_dimension=(
                    int(num_free) if method == "raw" else 3 * int(num_free)
                ),
            )
            score, p_flat, distance, uncertainty = gp_ucb_scores(
                flat_model,
                descriptor_model,
                test_x,
                baseline,
                kappa,
            )
            for local_index, row_index in enumerate(test):
                results.append(
                    {
                        "row_index": int(row_index),
                        "fold": int(fold_index),
                        "method": method,
                        "score": float(score[local_index]),
                        "predicted_flat_probability": float(p_flat[local_index]),
                        "predicted_archive_distance": float(distance[local_index]),
                        "descriptor_uncertainty": float(uncertainty[local_index]),
                    }
                )
    return results


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    ranks = rankdata(scores, method="average")
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(np.count_nonzero(negative))
    return float(
        (np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def safe_spearman(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(spearmanr(first, second).statistic)


def summarize(seed, rows, predictions, panel):
    true_new = np.asarray([row["true_new"] for row in rows], dtype=int)
    flat = np.asarray([row["flat"] for row in rows], dtype=int)
    cosine = np.asarray([row["cos_theta"] for row in rows], dtype=float)
    pair_ids = sorted({row["pair_id"] for row in rows})
    pair_indices = {
        pair: [
            index for index, row in enumerate(rows) if row["pair_id"] == pair
        ]
        for pair in pair_ids
    }

    # These group-conditioned quantities match the actual decision: after q1
    # is fixed, choose one (or a shortlist) from its sixteen theta candidates.
    uniform_group_values = []
    fixed_zero_values = []
    fixed_pi_values = []
    oracle_values = []
    oracle_top_quarter_values = []
    for pair in pair_ids:
        indices = pair_indices[pair]
        utilities = np.asarray(
            [rows[index]["true_new"] for index in indices], dtype=float
        )
        uniform_group_values.append(float(np.mean(utilities)))
        fixed_zero_values.append(
            float(
                next(
                    rows[index]["true_new"]
                    for index in indices
                    if abs(rows[index]["theta_deg"]) <= 1.0e-10
                )
            )
        )
        fixed_pi_values.append(
            float(
                next(
                    rows[index]["true_new"]
                    for index in indices
                    if abs(abs(rows[index]["theta_deg"]) - 180.0) <= 1.0e-10
                )
            )
        )
        oracle_values.append(float(np.max(utilities)))
        top_count = max(int(np.ceil(0.25 * len(indices))), 1)
        oracle_top_quarter_values.append(
            float(np.mean(np.sort(utilities)[-top_count:]))
        )

    group_conditioned_truth = {
        "num_q1_groups": int(len(pair_ids)),
        "uniform_random_new_rate": float(np.mean(uniform_group_values)),
        "fixed_theta_zero_new_rate": float(np.mean(fixed_zero_values)),
        "fixed_theta_pi_new_rate": float(np.mean(fixed_pi_values)),
        "oracle_top1_new_rate": float(np.mean(oracle_values)),
        "oracle_top25_mean_new_rate": float(
            np.mean(oracle_top_quarter_values)
        ),
    }

    prediction_maps = {
        method: {
            prediction["row_index"]: prediction
            for prediction in predictions
            if prediction["method"] == method
        }
        for method in ("raw", "angle_aware", "feature_shuffled")
    }
    angle_rows = []
    for theta in DEFAULT_ANGLES_DEG:
        selected_indices = [
            index
            for index, row in enumerate(rows)
            if row["theta_deg"] == float(theta)
        ]
        if not selected_indices:
            continue
        selected = [rows[index] for index in selected_indices]
        mean_scores = {
            f"{method}_mean_score": float(
                np.mean(
                    [
                        prediction_maps[method][index]["score"]
                        for index in selected_indices
                    ]
                )
            )
            for method in prediction_maps
        }
        angle_rows.append(
            {
                "theta_deg": float(theta),
                "count": len(selected),
                "flat_rate": float(np.mean([row["flat"] for row in selected])),
                "new_rate": float(np.mean([row["true_new"] for row in selected])),
                "unique_valid_cells": len(
                    {row["terminal_cell"] for row in selected if row["terminal_cell"] >= 0}
                ),
                "mean_equivalent_macro_steps": float(
                    np.mean([row["equivalent_macro_steps"] for row in selected])
                ),
                **mean_scores,
            }
        )

    model_summary = {}
    for method in ("raw", "angle_aware", "feature_shuffled"):
        method_predictions = [p for p in predictions if p["method"] == method]
        by_row = prediction_maps[method]
        ordered_scores = np.asarray(
            [by_row[index]["score"] for index in range(len(rows))], dtype=float
        )
        selected_utilities = []
        top_quarter_utilities = []
        top_quarter_any = []
        theta_zero_top_tied = []
        theta_zero_ranks = []
        for pair in pair_ids:
            indices = pair_indices[pair]
            scores = ordered_scores[indices]
            maximum = np.max(scores)
            tied_local = np.flatnonzero(
                np.isclose(scores, maximum, rtol=1.0e-10, atol=1.0e-12)
            )
            selected_utilities.append(
                float(np.mean([rows[indices[j]]["true_new"] for j in tied_local]))
            )
            top_count = max(int(np.ceil(0.25 * len(indices))), 1)
            top_local = np.argsort(-scores, kind="stable")[:top_count]
            top_utilities = np.asarray(
                [rows[indices[j]]["true_new"] for j in top_local], dtype=float
            )
            top_quarter_utilities.append(float(np.mean(top_utilities)))
            top_quarter_any.append(float(np.any(top_utilities > 0.5)))
            zero_local = next(
                j for j, index in enumerate(indices) if rows[index]["theta_deg"] == 0.0
            )
            theta_zero_top_tied.append(int(zero_local in tied_local))
            descending_rank = rankdata(-scores, method="average")
            theta_zero_ranks.append(float(descending_rank[zero_local]))

        model_summary[method] = {
            "new_cell_auc": binary_auc(true_new, ordered_scores),
            "flat_auc": binary_auc(flat, ordered_scores),
            "score_cosine_spearman": safe_spearman(cosine, ordered_scores),
            "expected_new_rate_at_top_tied_score": float(np.mean(selected_utilities)),
            "group_top1_new_rate": float(np.mean(selected_utilities)),
            "group_top25_mean_new_rate": float(
                np.mean(top_quarter_utilities)
            ),
            "group_top25_any_new_rate": float(np.mean(top_quarter_any)),
            "theta_zero_top_tied_rate": float(np.mean(theta_zero_top_tied)),
            "theta_zero_mean_rank": float(np.mean(theta_zero_ranks)),
            "mean_score": float(np.mean(ordered_scores)),
        }

    return {
        "seed": int(seed),
        "num_rows": int(len(rows)),
        "num_pairs": int(panel["num_pairs"]),
        "num_parent_groups": int(panel["num_actual_parents"]),
        "trunk_steps": int(panel["trunk_steps"]),
        "trunk_stop_reason": panel["trunk_stop_reason"],
        "baseline_cell_count": int(len(panel["baseline_archive"])),
        "primary_frozen_archive": "A1_trunk_only",
        "total_panel_cell_count": int(len(panel["full_archive"])),
        "flat_rate": float(np.mean(flat)) if len(flat) else float("nan"),
        "new_rate": float(np.mean(true_new)) if len(true_new) else float("nan"),
        "truth_cosine_new_spearman": safe_spearman(cosine, true_new),
        "truth_cosine_flat_spearman": safe_spearman(cosine, flat),
        "group_conditioned_truth": group_conditioned_truth,
        "angle_statistics": angle_rows,
        "model_statistics": model_summary,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(seed, base_directory, result_directory, args):
    model = load_model(base_directory / "model")
    config = default_config(args.perturbation_scale, args.max_suffix_steps)
    start = time.perf_counter()
    panel = generate_panel(
        seed,
        model,
        config,
        args.parents,
        args.v1_per_parent,
        DEFAULT_ANGLES_DEG,
    )
    rows = panel["rows"]
    if panel["num_actual_parents"] < 2:
        raise RuntimeError("at least two feasible parent groups are required")
    predictions = cross_validated_predictions(
        rows,
        panel["baseline_continuous"],
        len(model["index_free"]),
        args.folds,
        seed,
        args.kappa,
    )
    summary = summarize(seed, rows, predictions, panel)
    summary["wall_time_seconds"] = float(time.perf_counter() - start)
    summary["configuration"] = {
        "angles_deg": list(DEFAULT_ANGLES_DEG),
        "requested_parents": int(args.parents),
        "v1_per_parent": int(args.v1_per_parent),
        "folds": int(args.folds),
        "perturbation_scale": float(args.perturbation_scale),
        "max_suffix_steps": int(args.max_suffix_steps),
        "ucb_kappa": float(args.kappa),
    }

    seed_directory = result_directory / f"seed_{int(seed):03d}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    panel_csv_rows = []
    for row in rows:
        panel_csv_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in ("raw_feature", "angle_feature", "terminal_descriptor")
            }
        )
    write_csv(
        seed_directory / "panel.csv",
        panel_csv_rows,
        list(panel_csv_rows[0]),
    )
    write_csv(
        seed_directory / "cv_predictions.csv",
        predictions,
        list(predictions[0]),
    )
    np.savez_compressed(
        seed_directory / "panel_arrays.npz",
        raw_features=np.asarray([row["raw_feature"] for row in rows]),
        angle_features=np.asarray([row["angle_feature"] for row in rows]),
        terminal_descriptors=np.asarray(
            [row["terminal_descriptor"] for row in rows]
        ),
    )
    (seed_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def aggregate_summaries(summaries):
    methods = ("raw", "angle_aware", "feature_shuffled")
    group_counts = [
        int(summary["group_conditioned_truth"]["num_q1_groups"])
        for summary in summaries
    ]
    aggregate = {
        "num_seeds": len(summaries),
        "seeds": [summary["seed"] for summary in summaries],
        "total_q1_groups": int(sum(group_counts)),
        "mean_q1_groups_per_seed": float(np.mean(group_counts)),
        "mean_panel_cells": float(
            np.mean([summary["total_panel_cell_count"] for summary in summaries])
        ),
        "mean_truth_cosine_new_spearman": float(
            np.nanmean([summary["truth_cosine_new_spearman"] for summary in summaries])
        ),
        "mean_truth_cosine_flat_spearman": float(
            np.nanmean([summary["truth_cosine_flat_spearman"] for summary in summaries])
        ),
        "models": {},
        "group_conditioned_truth": {
            key: float(
                np.mean(
                    [summary["group_conditioned_truth"][key] for summary in summaries]
                )
            )
            for key in (
                "uniform_random_new_rate",
                "fixed_theta_zero_new_rate",
                "fixed_theta_pi_new_rate",
                "oracle_top1_new_rate",
                "oracle_top25_mean_new_rate",
            )
        },
    }
    for method in methods:
        keys = summaries[0]["model_statistics"][method]
        aggregate["models"][method] = {
            key: float(
                np.nanmean(
                    [summary["model_statistics"][method][key] for summary in summaries]
                )
            )
            for key in keys
        }
    aggregate["angle_statistics"] = []
    for theta in DEFAULT_ANGLES_DEG:
        per_seed = [
            next(
                row
                for row in summary["angle_statistics"]
                if abs(row["theta_deg"] - float(theta)) <= 1.0e-10
            )
            for summary in summaries
        ]
        aggregate["angle_statistics"].append(
            {
                "theta_deg": float(theta),
                "total_count": int(sum(row["count"] for row in per_seed)),
                "mean_flat_rate": float(
                    np.mean([row["flat_rate"] for row in per_seed])
                ),
                "mean_new_rate": float(
                    np.mean([row["new_rate"] for row in per_seed])
                ),
                "mean_unique_valid_cells_per_seed": float(
                    np.mean([row["unique_valid_cells"] for row in per_seed])
                ),
                "raw_mean_score": float(
                    np.mean([row["raw_mean_score"] for row in per_seed])
                ),
                "angle_aware_mean_score": float(
                    np.mean(
                        [row["angle_aware_mean_score"] for row in per_seed]
                    )
                ),
                "feature_shuffled_mean_score": float(
                    np.mean(
                        [row["feature_shuffled_mean_score"] for row in per_seed]
                    )
                ),
            }
        )
    return aggregate


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--v1-per-parent", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--perturbation-scale", type=float, default=0.20)
    parser.add_argument("--max-suffix-steps", type=int, default=100)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument(
        "--result-name", default="macro2_angle_panel_pilot"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    base_directory = Path(__file__).resolve().parent
    result_directory = base_directory / "result" / args.result_name
    result_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        summary = run_seed(
            seed, base_directory, result_directory, args
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    aggregate = aggregate_summaries(summaries)
    (result_directory / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
