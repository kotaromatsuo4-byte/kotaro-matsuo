"""Paired offline test of raw and transport-context Gaussian processes.

This script deliberately leaves ``main5.py`` and ``gp.py`` unchanged.  It
constructs several *direct perturbation children*.  At each such state it
evaluates the same small panel of actions:

* exact tangent-space transport of the preceding physical displacement;
* positive and negative angular offsets from that transport direction;
* its reverse;
* independent random tangent directions.

Every panel action is followed by the same deterministic rollout used by
``main5``.  Models are fitted and tested with group-wise cross validation:
all actions sharing one parent stay in the same fold.  This avoids the very
optimistic leakage that would arise if near-identical actions at one state
were split between training and test rows.

The raw GP input is the existing ``[state, proposed displacement]`` vector.
The context GP additionally receives the projected preceding displacement,
whether the parent is a direct perturbation child, its projection ratio, the
consecutive perturbation count, and the signed action/transport alignment.
Only one row per completed perturbation rollout is used for fitting; the many
deterministic suffix edges therefore cannot dominate this diagnostic.

Environment variables useful for a quick run are:

``FOLDING_PANEL_GROUPS`` (default 12), ``FOLDING_PANEL_FOLDS`` (default 3),
``FOLDING_PANEL_RANDOM_DIRECTIONS`` (default 2), ``FOLDING_SEED`` and
``FOLDING_PANEL_RESULT_NAME``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from archive import find_elite_cell
from descriptor import (
    find_last_nonflat_state_id,
    terminal_continuous_descriptor,
    terminal_descriptor,
)
from gp import (
    fit_surrogate_models,
    make_terminal_output_spec,
    score_acquisition,
    terminal_target_from_history,
    transition_input,
    unit_mode_directions,
)
from main5 import (
    _configuration,
    _deterministic_roll_once,
    _initial_deterministic_trunk,
    _load_model,
)
from mechanics import calc_dedx, calc_mode, compute_gravity_potential
from roll import displacement_roll
from state import create_history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _panel_configuration():
    config = _configuration()
    config.update(
        {
            "num_groups": _env_int("FOLDING_PANEL_GROUPS", 12),
            "num_folds": _env_int("FOLDING_PANEL_FOLDS", 3),
            "random_directions": _env_int(
                "FOLDING_PANEL_RANDOM_DIRECTIONS", 2
            ),
            "source_attempt_factor": _env_int(
                "FOLDING_PANEL_SOURCE_ATTEMPT_FACTOR", 20
            ),
            "result_name": os.environ.get(
                "FOLDING_PANEL_RESULT_NAME", "gp_paired_panel"
            ),
            "offset_degrees": tuple(
                float(value)
                for value in os.environ.get(
                    "FOLDING_PANEL_OFFSETS_DEG", "15,45"
                ).split(",")
                if value.strip()
            ),
        }
    )
    if config["num_groups"] < 3:
        raise ValueError("FOLDING_PANEL_GROUPS must be at least 3")
    if config["num_folds"] < 2:
        raise ValueError("FOLDING_PANEL_FOLDS must be at least 2")
    return config


def _make_model(base_directory, config):
    node, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    length_scale = float(np.mean(member_length))
    state_dimension = int(
        len(index_free) + np.count_nonzero(index_free % 3 == 2)
    )
    return node, {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": len(node),
        "num_member": len(member),
        "length_scale": length_scale,
        "input_state_dimension": state_dimension,
        "input_dimension": int(state_dimension + len(index_free)),
    }


def _mode(node, model):
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


def _add_source_child(history, parent_id, displacement, model, config):
    """Execute only the source perturbation, not its deterministic suffix."""
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


def _single_action_rollout(history, candidate, model, config):
    """One panel perturbation followed by a deterministic terminal suffix.

    This is intentionally local.  ``main5._candidate_rollout`` may implement
    a multi-step macro action, whereas this diagnostic must keep the tested
    action definition fixed at exactly one perturbation.
    """
    current = _add_source_child(
        history,
        int(candidate["parent_id"]),
        candidate["displacement"],
        model,
        config,
    )
    perturbation_child_id = current.state_id
    new_ids = [current.state_id]
    used_steps = 1
    while (
        current.terminal_reason is None
        and used_steps < config["max_rollout_steps"]
    ):
        current = _deterministic_roll_once(
            history,
            current.state_id,
            model,
            config,
        )
        new_ids.append(current.state_id)
        used_steps += 1
    if current.terminal_reason is not None:
        reason = current.terminal_reason
    elif used_steps >= config["max_rollout_steps"]:
        reason = "timeout"
    else:
        reason = None
    return current, new_ids, used_steps, reason, perturbation_child_id


def _consecutive_perturbation_count(history, state_id):
    count = 0
    current_id = int(state_id)
    while current_id is not None:
        state = history[current_id]
        if state.incoming_step_type != "perturbation":
            break
        count += 1
        current_id = state.parent_id
    return count


def _transport_frame(history, parent_id, mode):
    """Return a unit coefficient direction and physical transport context."""
    state = history[int(parent_id)]
    previous = np.asarray(state.incoming_displacement, dtype=float).reshape(-1)
    previous_norm = float(np.linalg.norm(previous))
    projected = mode @ (np.linalg.pinv(mode, rcond=1.0e-12) @ previous)
    projected_norm = float(np.linalg.norm(projected))
    ratio = projected_norm / max(previous_norm, 1.0e-12)
    coefficient = np.linalg.pinv(mode, rcond=1.0e-12) @ projected
    coefficient_norm = float(np.linalg.norm(coefficient))
    if coefficient_norm <= 1.0e-12 or projected_norm <= 1.0e-12:
        return None
    return coefficient / coefficient_norm, projected, ratio


def _orthogonal_direction(direction, rng):
    """Choose one reproducible unit direction orthogonal to ``direction``."""
    direction = np.asarray(direction, dtype=float)
    if len(direction) == 1:
        return None
    for _ in range(100):
        trial = rng.normal(size=len(direction))
        trial -= np.dot(trial, direction) * direction
        norm = float(np.linalg.norm(trial))
        if norm > 1.0e-12:
            return trial / norm
    return None


def _action_panel(history, parent_id, model, config, rng):
    """Create matched directions around the transported physical direction."""
    state = history[int(parent_id)]
    mode = _mode(state.node, model)
    if mode.shape[1] == 0:
        return []
    frame = _transport_frame(history, parent_id, mode)
    if frame is None:
        return []
    transport_direction, projected_previous, projection_ratio = frame
    orthogonal = _orthogonal_direction(transport_direction, rng)

    directions = [("transport", 0.0, transport_direction)]
    if orthogonal is not None:
        for degrees in config["offset_degrees"]:
            angle = np.deg2rad(float(degrees))
            for sign in (-1.0, 1.0):
                direction = (
                    np.cos(angle) * transport_direction
                    + sign * np.sin(angle) * orthogonal
                )
                directions.append(
                    (f"offset_{sign * degrees:+g}", sign * degrees, direction)
                )
    directions.append(("reverse", 180.0, -transport_direction))

    for random_index in range(config["random_directions"]):
        direction = rng.normal(size=mode.shape[1])
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        angle = np.degrees(
            np.arccos(np.clip(np.dot(direction, transport_direction), -1.0, 1.0))
        )
        directions.append((f"random_{random_index}", float(angle), direction))

    context_state_id = find_last_nonflat_state_id(
        history,
        parent_id,
        config["layer_threshold"],
    )
    candidates = []
    for kind, offset, direction in directions:
        coefficient = float(config["perturbation_scale"]) * direction
        displacement = (mode @ coefficient).reshape(model["num_node"], 3)
        signed_alignment = float(
            np.dot(displacement.reshape(-1), projected_previous)
            / max(
                np.linalg.norm(displacement) * np.linalg.norm(projected_previous),
                1.0e-12,
            )
        )
        raw_input = transition_input(
            state.node,
            displacement,
            model["index_free"],
            config["reference_point"],
            model["length_scale"],
            history[context_state_id].node,
        )
        raw_state_dimension = model["input_state_dimension"]
        previous_part = (
            projected_previous[model["index_free"]] / model["length_scale"]
        )
        state_context = np.concatenate(
            (
                raw_input[:raw_state_dimension],
                previous_part,
                np.asarray(
                    [
                        1.0,
                        projection_ratio,
                        min(_consecutive_perturbation_count(history, parent_id), 4)
                        / 4.0,
                    ]
                ),
            )
        )
        action_context = np.concatenate(
            (raw_input[raw_state_dimension:], [signed_alignment])
        )
        candidates.append(
            {
                "parent_id": int(parent_id),
                "kind": kind,
                "offset_degrees": float(offset),
                "coefficient": coefficient,
                "displacement": displacement,
                "raw_input": raw_input,
                "context_input": np.concatenate((state_context, action_context)),
                "context_state_dimension": len(state_context),
                "alignment": signed_alignment,
                "projection_ratio": float(projection_ratio),
            }
        )
    return candidates


def _create_source_parents(history, trunk_ids, model, config, rng):
    """Create direct perturbation children distributed over trunk progress."""
    bases = [0] + [
        int(state_id)
        for state_id in trunk_ids
        if history[int(state_id)].terminal_reason is None
    ]
    if not bases:
        return []

    source_ids = []
    maximum_attempts = config["num_groups"] * config["source_attempt_factor"]
    for attempt in range(maximum_attempts):
        if len(source_ids) >= config["num_groups"]:
            break
        # A cyclic progress stratum gives every part of the deterministic trunk
        # a chance before random repeats are introduced.
        position = (attempt % max(config["num_groups"], 1) + 0.5) / max(
            config["num_groups"], 1
        )
        base_index = min(int(position * len(bases)), len(bases) - 1)
        if attempt >= config["num_groups"]:
            base_index = int(rng.integers(len(bases)))
        base_id = bases[base_index]
        mode = _mode(history[base_id].node, model)
        if mode.shape[1] == 0:
            continue
        direction = unit_mode_directions(mode.shape[1], 1, rng)[0]
        displacement = (
            mode @ (float(config["perturbation_scale"]) * direction)
        ).reshape(model["num_node"], 3)
        child = _add_source_child(
            history,
            base_id,
            displacement,
            model,
            config,
        )
        if child.terminal_reason is None:
            source_ids.append(child.state_id)
    return source_ids


def _valid_terminal_descriptor(history, terminal_state_id, model, config):
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
    valid = not any(upper == -1 for _, _, upper in layer)
    return shape, layer, valid


def _evaluate_panel(history, source_ids, model, config, output_spec, rng):
    rows = []
    rollout_id = 0
    for group_id, parent_id in enumerate(source_ids):
        candidates = _action_panel(history, parent_id, model, config, rng)
        for candidate_index, candidate in enumerate(candidates):
            terminal, new_ids, spent, reason, perturbation_child_id = (
                _single_action_rollout(
                    history,
                    candidate,
                    model,
                    config,
                )
            )
            if terminal is None or reason not in ("flat", "collision"):
                continue
            potentials = np.asarray(
                [history[state_id].potential for state_id in new_ids], dtype=float
            )
            flatness_target = float(
                np.log1p(
                    np.min(potentials)
                    / max(config["flatness_tolerance"], 1.0e-12)
                )
            )
            if reason == "flat":
                descriptor_target = terminal_target_from_history(
                    history,
                    terminal.state_id,
                    config["layer_threshold"],
                    output_spec,
                )
                shape, layer, valid_flat = _valid_terminal_descriptor(
                    history,
                    terminal.state_id,
                    model,
                    config,
                )
                continuous_descriptor = terminal_continuous_descriptor(
                    history,
                    terminal.state_id,
                    model["length_scale"],
                    config["layer_threshold"],
                    config["reference_point"],
                )
            else:
                descriptor_target = np.full(
                    len(output_spec["active_positions"]), np.nan
                )
                shape, layer, valid_flat = None, None, False
                continuous_descriptor = None
            rows.append(
                {
                    "row_id": len(rows),
                    "rollout_id": rollout_id,
                    "group_id": int(group_id),
                    "candidate_index": int(candidate_index),
                    "parent_id": int(parent_id),
                    "perturbation_child_id": int(perturbation_child_id),
                    "terminal_state_id": int(terminal.state_id),
                    "kind": candidate["kind"],
                    "offset_degrees": candidate["offset_degrees"],
                    "alignment": candidate["alignment"],
                    "projection_ratio": candidate["projection_ratio"],
                    "spent_steps": int(spent),
                    "stop_reason": reason,
                    "flat": reason == "flat",
                    "valid_flat": bool(valid_flat),
                    "flatness_target": flatness_target,
                    "descriptor_target": descriptor_target,
                    "shape_descriptor": shape,
                    "layer_key": layer,
                    "continuous_descriptor": continuous_descriptor,
                    "raw_input": candidate["raw_input"],
                    "context_input": candidate["context_input"],
                    "context_state_dimension": candidate[
                        "context_state_dimension"
                    ],
                }
            )
            rollout_id += 1
    return rows


def _archive_from_rows(rows, config):
    archive = []
    for row in rows:
        if not row["valid_flat"]:
            continue
        cell_index = find_elite_cell(
            archive,
            row["shape_descriptor"],
            row["layer_key"],
            config["rms_tolerance"],
            config["max_tolerance"],
        )
        if cell_index is None:
            archive.append(
                {
                    "shape_descriptor": row["shape_descriptor"],
                    "layer_key": row["layer_key"],
                    "continuous_descriptor": row["continuous_descriptor"],
                    "terminal_state_id": row["terminal_state_id"],
                }
            )
    return archive


def _is_new_cell(row, archive, config):
    if not row["valid_flat"]:
        return False
    return find_elite_cell(
        archive,
        row["shape_descriptor"],
        row["layer_key"],
        config["rms_tolerance"],
        config["max_tolerance"],
    ) is None


def _fit(rows, input_name, state_dimension, model, config):
    inputs = np.asarray([row[input_name] for row in rows], dtype=float)
    labels = np.asarray([row["flat"] for row in rows], dtype=bool)
    flatness = np.asarray([row["flatness_target"] for row in rows], dtype=float)
    descriptors = np.asarray(
        [row["descriptor_target"] for row in rows], dtype=float
    )
    return fit_surrogate_models(
        inputs,
        labels,
        flatness,
        descriptors,
        state_dimension=state_dimension,
        state_length_scale=config["state_length_scale"],
        action_length_scale=config["action_length_scale"],
        signal_variance=config["signal_variance"],
        regression_noise_std=config["regression_noise_std"],
        classifier_noise_std=config["classifier_noise_std"],
        descriptor_scale_floor=config["descriptor_scale_floor"],
        prior_flat_probability=0.5,
    )


def _scores(surrogate, rows, input_name, archive, output_spec, model, config, rng):
    inputs = np.asarray([row[input_name] for row in rows], dtype=float)
    return score_acquisition(
        surrogate,
        inputs,
        archive,
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
        rng=rng,
    )["score"]


def _roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(np.count_nonzero(labels))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    value = (
        np.sum(ranks[labels]) - positive * (positive + 1) / 2.0
    ) / (positive * negative)
    return float(value)


def _average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(np.count_nonzero(labels))
    if positive == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered]) / positive)


def _rank_metrics(scored_rows, score_name):
    groups = sorted(
        {
            (row["fold"], row["archive_variant"], row["group_id"])
            for row in scored_rows
        }
    )
    transport_ranks = []
    top1_new = []
    top2_new = []
    eligible_new_groups = 0
    for fold, archive_variant, group_id in groups:
        group = [
            row
            for row in scored_rows
            if row["fold"] == fold
            and row["archive_variant"] == archive_variant
            and row["group_id"] == group_id
        ]
        ordered = sorted(group, key=lambda row: -row[score_name])
        for rank, row in enumerate(ordered, start=1):
            if row["kind"] == "transport":
                transport_ranks.append(rank)
                break
        if any(row["new_cell"] for row in group):
            eligible_new_groups += 1
            top1_new.append(bool(ordered[0]["new_cell"]))
            top2_new.append(any(row["new_cell"] for row in ordered[:2]))
    return {
        "mean_transport_rank": (
            float(np.mean(transport_ranks)) if transport_ranks else None
        ),
        "transport_top1_rate": (
            float(np.mean(np.asarray(transport_ranks) == 1))
            if transport_ranks
            else None
        ),
        "transport_mrr": (
            float(np.mean(1.0 / np.asarray(transport_ranks)))
            if transport_ranks
            else None
        ),
        "eligible_new_groups": int(eligible_new_groups),
        "new_cell_success_at_1": (
            float(np.mean(top1_new)) if top1_new else None
        ),
        "new_cell_success_at_2": (
            float(np.mean(top2_new)) if top2_new else None
        ),
    }


def _random_baseline(scored_rows):
    """Analytic expectation for uniform selection within every test panel."""
    group_metrics = []
    keys = sorted(
        {
            (row["fold"], row["archive_variant"], row["group_id"])
            for row in scored_rows
        }
    )
    for fold, archive_variant, group_id in keys:
        group = [
            row
            for row in scored_rows
            if row["fold"] == fold
            and row["archive_variant"] == archive_variant
            and row["group_id"] == group_id
        ]
        count = len(group)
        positive = int(np.count_nonzero([row["new_cell"] for row in group]))
        if count == 0:
            continue
        harmonic = float(np.sum(1.0 / np.arange(1, count + 1)))
        if positive > 0:
            success_at_1 = positive / count
            if count >= 2:
                failure_at_2 = (
                    (count - positive) * (count - positive - 1)
                    / (count * (count - 1))
                )
                success_at_2 = 1.0 - failure_at_2
            else:
                success_at_2 = success_at_1
        else:
            success_at_1 = np.nan
            success_at_2 = np.nan
        group_metrics.append(
            (count, harmonic / count, success_at_1, success_at_2, positive > 0)
        )
    eligible = [metric for metric in group_metrics if metric[4]]
    return {
        "mean_transport_rank": float(
            np.mean([(metric[0] + 1.0) / 2.0 for metric in group_metrics])
        ),
        "transport_top1_rate": float(
            np.mean([1.0 / metric[0] for metric in group_metrics])
        ),
        "transport_mrr": float(np.mean([metric[1] for metric in group_metrics])),
        "eligible_new_groups": int(len(eligible)),
        "new_cell_success_at_1": (
            float(np.mean([metric[2] for metric in eligible]))
            if eligible
            else None
        ),
        "new_cell_success_at_2": (
            float(np.mean([metric[3] for metric in eligible]))
            if eligible
            else None
        ),
        "new_cell_roc_auc": 0.5,
        "new_cell_average_precision": None,
    }


def _cross_validate(rows, model, config, output_spec, rng):
    group_ids = np.asarray(sorted({row["group_id"] for row in rows}), dtype=int)
    shuffled = rng.permutation(group_ids)
    folds = np.array_split(shuffled, min(config["num_folds"], len(shuffled)))
    scored_rows = []
    fold_summaries = []

    for fold_index, test_groups in enumerate(folds):
        test_set = set(int(value) for value in test_groups)
        train = [row for row in rows if row["group_id"] not in test_set]
        test = [row for row in rows if row["group_id"] in test_set]
        full_archive = _archive_from_rows(train, config)
        raw_model = _fit(
            train,
            "raw_input",
            model["input_state_dimension"],
            model,
            config,
        )
        context_state_dimension = rows[0]["context_state_dimension"]
        context_model = _fit(
            train,
            "context_input",
            context_state_dimension,
            model,
            config,
        )
        archive_variants = [
            ("early_one_cell", f"cell_{index}", [cell])
            for index, cell in enumerate(full_archive)
        ]
        archive_variants.append(("saturated", "all", full_archive))
        for archive_regime, archive_variant, archive in archive_variants:
            raw_score = _scores(
                raw_model,
                test,
                "raw_input",
                archive,
                output_spec,
                model,
                config,
                rng,
            )
            context_score = _scores(
                context_model,
                test,
                "context_input",
                archive,
                output_spec,
                model,
                config,
                rng,
            )
            fold_labels = []
            for row, raw_value, context_value in zip(
                test, raw_score, context_score
            ):
                scored = {
                    "fold": int(fold_index),
                    "archive_regime": archive_regime,
                    "archive_variant": archive_variant,
                    "row_id": row["row_id"],
                    "group_id": row["group_id"],
                    "parent_id": row["parent_id"],
                    "kind": row["kind"],
                    "offset_degrees": row["offset_degrees"],
                    "alignment": row["alignment"],
                    "projection_ratio": row["projection_ratio"],
                    "stop_reason": row["stop_reason"],
                    "valid_flat": int(row["valid_flat"]),
                    "new_cell": int(_is_new_cell(row, archive, config)),
                    "raw_score": float(raw_value),
                    "context_score": float(context_value),
                    "transport_score": float(row["kind"] == "transport"),
                }
                fold_labels.append(scored["new_cell"])
                scored_rows.append(scored)
            fold_summaries.append(
                {
                    "fold": int(fold_index),
                    "archive_regime": archive_regime,
                    "archive_variant": archive_variant,
                    "train_groups": int(len(group_ids) - len(test_groups)),
                    "test_groups": int(len(test_groups)),
                    "train_archive_cells": int(len(full_archive)),
                    "known_archive_cells": int(len(archive)),
                    "test_new_candidates": int(np.sum(fold_labels)),
                }
            )

    summary = {"folds": fold_summaries, "regimes": {}}
    for regime in ("early_one_cell", "saturated"):
        regime_rows = [
            row for row in scored_rows if row["archive_regime"] == regime
        ]
        labels = np.asarray(
            [row["new_cell"] for row in regime_rows], dtype=bool
        )
        regime_summary = {
            "num_test_rows": int(len(regime_rows)),
            "num_new_cell_labels": int(np.count_nonzero(labels)),
            "models": {},
        }
        for model_name, score_name in (
            ("raw_gp", "raw_score"),
            ("context_gp", "context_score"),
            ("transport_heuristic", "transport_score"),
        ):
            scores = np.asarray(
                [row[score_name] for row in regime_rows], dtype=float
            )
            metrics = _rank_metrics(regime_rows, score_name)
            metrics.update(
                {
                    "new_cell_roc_auc": _roc_auc(labels, scores),
                    "new_cell_average_precision": _average_precision(
                        labels, scores
                    ),
                }
            )
            regime_summary["models"][model_name] = metrics
        regime_summary["models"]["uniform_random_expected"] = (
            _random_baseline(regime_rows)
        )
        summary["regimes"][regime] = regime_summary
    summary["num_test_rows"] = int(len(scored_rows))
    return scored_rows, summary


def _write_rows(path, rows):
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main():
    base_directory = Path(__file__).resolve().parent
    config = _panel_configuration()
    initial_node, model = _make_model(base_directory, config)
    initial_potential = compute_gravity_potential(
        initial_node, config["reference_point"]
    )
    history = create_history(initial_node, initial_potential)
    output_spec = make_terminal_output_spec(
        initial_node,
        model["index_free"],
        config["reference_point"],
        model["length_scale"],
    )
    rng = np.random.default_rng(config["seed"] + 8101)
    np.random.seed(config["seed"])

    _, trunk_ids, _, trunk_reason = _initial_deterministic_trunk(
        history,
        config["max_rollout_steps"],
        model,
        config,
    )
    source_ids = _create_source_parents(
        history, trunk_ids, model, config, rng
    )
    if len(source_ids) < 3:
        raise RuntimeError(
            f"only {len(source_ids)} valid direct parents could be generated"
        )
    rows = _evaluate_panel(
        history, source_ids, model, config, output_spec, rng
    )
    complete_groups = {
        group_id
        for group_id in {row["group_id"] for row in rows}
        if any(
            row["group_id"] == group_id and row["kind"] == "transport"
            for row in rows
        )
    }
    rows = [row for row in rows if row["group_id"] in complete_groups]
    if len(complete_groups) < 3:
        raise RuntimeError("fewer than three complete paired groups were obtained")

    scored_rows, cv_summary = _cross_validate(
        rows, model, config, output_spec, rng
    )
    output_directory = (
        base_directory
        / "result"
        / config["result_name"]
        / f"seed_{config['seed']:03d}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_rows(output_directory / "paired_panel_scores.csv", scored_rows)

    compact_rows = []
    for row in rows:
        compact_rows.append(
            {
                key: row[key]
                for key in (
                    "row_id",
                    "group_id",
                    "candidate_index",
                    "parent_id",
                    "terminal_state_id",
                    "kind",
                    "offset_degrees",
                    "alignment",
                    "projection_ratio",
                    "spent_steps",
                    "stop_reason",
                    "flat",
                    "valid_flat",
                    "flatness_target",
                )
            }
        )
    _write_rows(output_directory / "paired_panel_outcomes.csv", compact_rows)
    np.savez_compressed(
        output_directory / "paired_panel_arrays.npz",
        raw_inputs=np.asarray([row["raw_input"] for row in rows]),
        context_inputs=np.asarray([row["context_input"] for row in rows]),
        descriptor_targets=np.asarray(
            [row["descriptor_target"] for row in rows]
        ),
        group_ids=np.asarray([row["group_id"] for row in rows], dtype=int),
        flat_labels=np.asarray([row["flat"] for row in rows], dtype=bool),
    )

    summary = {
        "experiment": "paired_transport_context_gp_v1",
        "seed": int(config["seed"]),
        "trunk_stop_reason": trunk_reason,
        "requested_groups": int(config["num_groups"]),
        "complete_groups": int(len(complete_groups)),
        "panel_rollouts": int(len(rows)),
        "physical_flat_rollouts": int(
            np.count_nonzero([row["flat"] for row in rows])
        ),
        "valid_flat_rollouts": int(
            np.count_nonzero([row["valid_flat"] for row in rows])
        ),
        "raw_input_dimension": int(len(rows[0]["raw_input"])),
        "context_input_dimension": int(len(rows[0]["context_input"])),
        "group_split": "group-wise cross validation by common parent",
        **cv_summary,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
