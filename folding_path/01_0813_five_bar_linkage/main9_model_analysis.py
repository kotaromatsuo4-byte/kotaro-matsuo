"""Diagnose terminal-cell learnability in the saved main9 random corpus.

This analysis is read-only with respect to mechanics results.  It writes only
derived JSON/CSV files below ``result/main9_model_analysis`` and never imports
or modifies the search loop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CORPUS = SCRIPT_DIRECTORY / "result" / "main9_random_corpus_b20000"
REFERENCE = SCRIPT_DIRECTORY / "result" / "reachability_audit" / "reachability_report.json"
CHRONOLOGICAL_EVALUATION = SCRIPT_DIRECTORY / "result" / "main9_forest_eval_b20000"
RANDOM_ORDER_EVALUATION = SCRIPT_DIRECTORY / "result" / "main9_forest_eval_b20000_random_order"
LOOSE_FILTER_EVALUATION = SCRIPT_DIRECTORY / "result" / "main9_forest_eval_b20000_loose_loco"
OUTPUT = SCRIPT_DIRECTORY / "result" / "main9_model_analysis"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _shape_matches(cell: dict[str, Any], target: dict[str, Any], parameters: dict[str, float]) -> bool:
    if cell["layer_key"] != target["layer_key"]:
        return False
    shape = np.asarray(cell["shape_descriptor"], dtype=float).reshape(-1, 2)
    prototype = np.asarray(target["prototype_shape_descriptor"], dtype=float).reshape(-1, 2)
    distance = np.linalg.norm(shape - prototype, axis=1)
    return bool(
        np.sqrt(np.mean(distance * distance)) <= parameters["rms_tolerance"]
        and np.max(distance) <= parameters["max_tolerance"]
    )


def _cell_maps(seed_directories: list[Path], reference: dict[str, Any]) -> dict[int, dict[int, int]]:
    unmatched = set()
    maps: dict[int, dict[int, int | tuple[Any, ...]]] = {}
    for seed_directory in seed_directories:
        seed = int(seed_directory.name.rsplit("_", 1)[-1])
        cells = _read_json(seed_directory / "cell_archive.json")
        mapping: dict[int, int | tuple[Any, ...]] = {}
        for cell_index, cell in enumerate(cells):
            target_id = None
            for target in reference["targets"]:
                if _shape_matches(cell, target, reference["descriptor_parameters"]):
                    target_id = int(target["target_id"])
                    break
            if target_id is None:
                signature = tuple(tuple(part) for part in cell["layer_key"])
                unmatched.add(signature)
                mapping[cell_index] = signature
            else:
                mapping[cell_index] = target_id
        maps[seed] = mapping
    extra_lookup = {
        signature: 6 + index for index, signature in enumerate(sorted(unmatched))
    }
    return {
        seed: {
            cell: value if isinstance(value, int) else extra_lookup[value]
            for cell, value in mapping.items()
        }
        for seed, mapping in maps.items()
    }


def _canonical_t6(seed_directory: Path, cell_map: dict[int, int]) -> tuple[int, dict[int, int]]:
    events = _read_json(seed_directory / "discovery_events.json")
    first_steps: dict[int, int] = {}
    for event in events:
        if event["archive_layer"] != "kinematic":
            continue
        mapped = cell_map[int(event["cell_index"])]
        if mapped < 6 and mapped not in first_steps:
            first_steps[mapped] = int(event["mechanics_step"])
    if len(first_steps) != 6:
        raise RuntimeError(f"{seed_directory.name} did not reach all canonical targets")
    return max(first_steps.values()), first_steps


def _primary_indices(data: dict[str, np.ndarray], valid_only: bool = True) -> np.ndarray:
    mask = ~np.asarray(data["is_hindsight"], dtype=bool)
    if valid_only:
        mask &= np.asarray(data["cell_labels"], dtype=int) >= 0
    return np.flatnonzero(mask)


def _macro_step_map(data: dict[str, np.ndarray]) -> dict[int, int]:
    result = {}
    for index in _primary_indices(data, valid_only=False):
        result[int(data["macro_ids"][index])] = int(data["macro_end_step"][index])
    return result


def _normalized_entropy(labels: np.ndarray, class_count: int) -> float:
    if len(labels) == 0 or class_count <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / np.sum(counts)
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(class_count))


def _neighbor_records(
    seed: int,
    data: dict[str, np.ndarray],
    mode: str,
) -> list[dict[str, Any]]:
    valid = np.asarray(data["cell_labels"], dtype=int) >= 0
    primary = (~np.asarray(data["is_hindsight"], dtype=bool)) & valid
    primary_indices = np.flatnonzero(primary)
    query_x = np.asarray(data["inputs"][primary_indices], dtype=float)
    mean = np.mean(query_x, axis=0)
    scale = np.maximum(np.std(query_x, axis=0), 1.0e-6)
    all_x = (np.asarray(data["inputs"], dtype=float) - mean) / scale
    query_x = all_x[primary_indices]
    query_groups = np.asarray(data["macro_ids"][primary_indices], dtype=int)
    query_labels = np.asarray(data["cell_labels"][primary_indices], dtype=int)
    pool_mask = primary if mode == "primary" else valid
    pool_groups = np.unique(np.asarray(data["macro_ids"][pool_mask], dtype=int))
    class_count = len(np.unique(query_labels))
    records = []
    for query_index, group in enumerate(query_groups):
        group_distances = []
        group_labels = []
        for other_group in pool_groups:
            if other_group == group:
                continue
            indices = np.flatnonzero(
                pool_mask & (np.asarray(data["macro_ids"], dtype=int) == other_group)
            )
            difference = all_x[indices] - query_x[query_index]
            distance = float(np.min(np.sqrt(np.mean(difference * difference, axis=1))))
            group_distances.append(distance)
            group_labels.append(int(data["cell_labels"][indices[0]]))
        order = np.argsort(group_distances)
        distances = np.asarray(group_distances, dtype=float)[order]
        labels = np.asarray(group_labels, dtype=int)[order]
        neighbors = labels[: min(10, len(labels))]
        records.append(
            {
                "seed": seed,
                "mode": mode,
                "nearest_distance": float(distances[0]),
                "nearest_same_cell": int(labels[0] == query_labels[query_index]),
                "neighbor10_entropy": _normalized_entropy(neighbors, class_count),
            }
        )
    return records


def _distance_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for mode in ("primary", "hindsight"):
        selected = [record for record in records if record["mode"] == mode]
        distance = np.asarray([record["nearest_distance"] for record in selected])
        same = np.asarray([record["nearest_same_cell"] for record in selected])
        entropy = np.asarray([record["neighbor10_entropy"] for record in selected])
        edges = np.quantile(distance, np.linspace(0.0, 1.0, 6))
        for bin_index in range(5):
            mask = (distance >= edges[bin_index]) & (
                (distance <= edges[bin_index + 1])
                if bin_index == 4
                else (distance < edges[bin_index + 1])
            )
            rows.append(
                {
                    "mode": mode,
                    "distance_quantile_bin": bin_index,
                    "lower_distance": float(edges[bin_index]),
                    "upper_distance": float(edges[bin_index + 1]),
                    "queries": int(np.count_nonzero(mask)),
                    "nearest_same_cell_rate": float(np.mean(same[mask])),
                    "mean_neighbor10_normalized_entropy": float(np.mean(entropy[mask])),
                }
            )
    return rows


def _evaluation_rows(directory: Path, train_order: str) -> list[dict[str, Any]]:
    with (directory / "aggregate_metrics.csv").open(encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    fields = (
        "pooled_known_accuracy",
        "known_macro_f1_mean",
        "known_nll_mean",
        "known_brier_mean",
        "pooled_duplicate_rejection_rate",
        "pooled_reject_precision",
        "pooled_pseudo_unknown_retention",
        "pooled_loco_unknown_retention",
    )
    rows = []
    for source_row in source:
        row: dict[str, Any] = {
            "train_order": train_order,
            "model_kind": source_row["model_kind"],
            "train_macros": int(source_row["train_macros"]),
        }
        for field in fields:
            text = source_row.get(field, "")
            row[field] = float(text) if text not in (None, "") else None
        rows.append(row)
    return rows


def _first_sustained_prevalence_gain(rows: list[dict[str, Any]], model: str) -> int | None:
    selected = sorted(
        [row for row in rows if row["train_order"] == "chronological"],
        key=lambda row: row["train_macros"],
    )
    prevalence = {
        row["train_macros"]: row["pooled_known_accuracy"]
        for row in selected
        if row["model_kind"] == "prevalence"
    }
    model_rows = [row for row in selected if row["model_kind"] == model]
    for start_index, row in enumerate(model_rows):
        if all(
            future["pooled_known_accuracy"] > prevalence[future["train_macros"]]
            for future in model_rows[start_index:]
        ):
            return int(row["train_macros"])
    return None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_directories = sorted(CORPUS.glob("seed_*"))
    reference = _read_json(REFERENCE)
    cell_maps = _cell_maps(seed_directories, reference)

    seed_summaries = []
    macro_timeline = []
    class_rows = []
    k_rows = []
    neighbor_records = []
    step_at_macro_count: dict[int, list[int]] = {}
    pooled_joint = np.zeros((24, 8), dtype=int)

    for seed_directory in seed_directories:
        seed = int(seed_directory.name.rsplit("_", 1)[-1])
        data = _load_npz(seed_directory / "training_data.npz")
        cell_map = cell_maps[seed]
        t6, target_steps = _canonical_t6(seed_directory, cell_map)
        valid = np.asarray(data["cell_labels"], dtype=int) >= 0
        primary = (~np.asarray(data["is_hindsight"], dtype=bool)) & valid
        all_primary = ~np.asarray(data["is_hindsight"], dtype=bool)
        macro_steps = _macro_step_map(data)
        valid_macros = sorted(
            np.unique(np.asarray(data["macro_ids"][primary], dtype=int)),
            key=lambda macro: macro_steps[int(macro)],
        )
        all_macros = sorted(macro_steps, key=macro_steps.get)

        for macro_count in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320):
            if macro_count <= len(valid_macros):
                step_at_macro_count.setdefault(macro_count, []).append(
                    int(macro_steps[int(valid_macros[macro_count - 1])])
                )

        for macro_count, macro in enumerate(all_macros, start=1):
            step = int(macro_steps[macro])
            completed = np.asarray(data["macro_end_step"], dtype=int) <= step
            completed_primary = completed & primary
            completed_valid = completed & valid
            discovered = {
                target
                for target, target_step in target_steps.items()
                if target_step <= step
            }
            macro_timeline.append(
                {
                    "seed": seed,
                    "mechanics_step": step,
                    "completed_macros": macro_count,
                    "valid_independent_macros": int(
                        len(np.unique(data["macro_ids"][completed_primary]))
                    ),
                    "valid_training_rows": int(np.count_nonzero(completed_valid)),
                    "observed_raw_classes": int(
                        len(np.unique(data["cell_labels"][completed_primary]))
                    ),
                    "canonical_targets_discovered": len(discovered),
                }
            )

        t6_mask = primary & (np.asarray(data["macro_end_step"], dtype=int) <= t6)
        t6_rows = valid & (np.asarray(data["macro_end_step"], dtype=int) <= t6)
        final_labels = np.asarray(data["cell_labels"][primary], dtype=int)
        t6_labels = np.asarray(data["cell_labels"][t6_mask], dtype=int)
        final_counts = np.bincount(final_labels, minlength=len(cell_map))
        t6_counts = np.bincount(t6_labels, minlength=len(cell_map))
        row_counts = np.bincount(
            np.asarray(data["cell_labels"][valid], dtype=int), minlength=len(cell_map)
        )
        for raw_cell in range(len(cell_map)):
            class_rows.append(
                {
                    "seed": seed,
                    "raw_cell": raw_cell,
                    "canonical_or_extra_cell": cell_map[raw_cell],
                    "macros_at_t6": int(t6_counts[raw_cell]),
                    "macros_final": int(final_counts[raw_cell]),
                    "rows_final": int(row_counts[raw_cell]),
                }
            )

        macro_costs = []
        macro_row_counts = []
        for index in np.flatnonzero(all_primary):
            macro = int(data["macro_ids"][index])
            mask = np.asarray(data["macro_ids"], dtype=int) == macro
            macro_costs.append(
                int(data["macro_end_step"][index] - data["macro_start_step"][index])
            )
            macro_row_counts.append(int(np.count_nonzero(mask)))
        positive_final = final_counts[final_counts > 0]
        seed_summaries.append(
            {
                "seed": seed,
                "canonical_t6_step": t6,
                "independent_macros_at_t6": int(
                    len(np.unique(data["macro_ids"][t6_mask]))
                ),
                "training_rows_at_t6": int(np.count_nonzero(t6_rows)),
                "raw_classes_at_t6": int(len(np.unique(t6_labels))),
                "final_independent_valid_macros": len(valid_macros),
                "final_training_rows": int(np.count_nonzero(valid)),
                "final_raw_classes": int(len(np.unique(final_labels))),
                "final_macro_class_max_min_ratio": float(
                    np.max(positive_final) / np.min(positive_final)
                ),
                "final_majority_prevalence": float(np.max(final_counts) / np.sum(final_counts)),
                "mean_mechanics_steps_per_macro": float(np.mean(macro_costs)),
                "median_mechanics_steps_per_macro": float(np.median(macro_costs)),
                "mean_rows_per_macro": float(np.mean(macro_row_counts)),
                "hindsight_rows": int(np.count_nonzero(data["is_hindsight"] & valid)),
                "censored_macros": int(len(all_macros) - len(valid_macros)),
            }
        )

        canonical_labels = np.asarray(
            [cell_map[int(label)] for label in data["cell_labels"][primary]], dtype=int
        )
        requested_k = np.asarray(data["source_requested_k"][primary], dtype=int)
        for k in range(1, 25):
            counts = np.bincount(canonical_labels[requested_k == k], minlength=8)
            pooled_joint[k - 1] += counts
            k_rows.append(
                {
                    "seed": seed,
                    "requested_k": k,
                    "macros": int(np.sum(counts)),
                    "observed_cells": int(np.count_nonzero(counts)),
                    "majority_probability": float(np.max(counts) / np.sum(counts)),
                    "normalized_cell_entropy": _normalized_entropy(
                        np.repeat(np.arange(8), counts), 8
                    ),
                    **{f"cell_{cell}": int(counts[cell]) for cell in range(8)},
                }
            )

        neighbor_records.extend(_neighbor_records(seed, data, "primary"))
        neighbor_records.extend(_neighbor_records(seed, data, "hindsight"))

    distance_bins = _distance_bins(neighbor_records)
    neighbor_summary = []
    for mode in ("primary", "hindsight"):
        rows = [row for row in neighbor_records if row["mode"] == mode]
        neighbor_summary.append(
            {
                "mode": mode,
                "queries": len(rows),
                "median_nearest_distance": float(
                    np.median([row["nearest_distance"] for row in rows])
                ),
                "nearest_same_cell_rate": float(
                    np.mean([row["nearest_same_cell"] for row in rows])
                ),
                "mean_neighbor10_normalized_entropy": float(
                    np.mean([row["neighbor10_entropy"] for row in rows])
                ),
            }
        )

    probability = pooled_joint / np.sum(pooled_joint)
    p_k = np.sum(probability, axis=1)
    p_cell = np.sum(probability, axis=0)
    mutual_information = 0.0
    for k, cell in zip(*np.nonzero(probability)):
        mutual_information += probability[k, cell] * np.log(
            probability[k, cell] / (p_k[k] * p_cell[cell])
        )
    cell_entropy = -np.sum(p_cell[p_cell > 0] * np.log(p_cell[p_cell > 0]))
    for k in range(24):
        counts = pooled_joint[k]
        k_rows.append(
            {
                "seed": "pooled",
                "requested_k": k + 1,
                "macros": int(np.sum(counts)),
                "observed_cells": int(np.count_nonzero(counts)),
                "majority_probability": float(np.max(counts) / np.sum(counts)),
                "normalized_cell_entropy": _normalized_entropy(
                    np.repeat(np.arange(8), counts), 8
                ),
                **{f"cell_{cell}": int(counts[cell]) for cell in range(8)},
            }
        )

    evaluation_rows = _evaluation_rows(CHRONOLOGICAL_EVALUATION, "chronological")
    evaluation_rows += _evaluation_rows(RANDOM_ORDER_EVALUATION, "random")
    loose_rows = _evaluation_rows(LOOSE_FILTER_EVALUATION, "chronological_loose")
    evaluation_rows += loose_rows
    sustained = {
        model: _first_sustained_prevalence_gain(evaluation_rows, model)
        for model in ("random_forest", "extra_trees", "knn")
    }
    strict_first_rejection = {}
    for model in ("random_forest", "extra_trees", "knn"):
        rows = sorted(
            [
                row
                for row in evaluation_rows
                if row["train_order"] == "chronological"
                and row["model_kind"] == model
            ],
            key=lambda row: row["train_macros"],
        )
        strict_first_rejection[model] = next(
            (
                row["train_macros"]
                for row in rows
                if (row["pooled_duplicate_rejection_rate"] or 0.0) > 0.0
            ),
            None,
        )

    order_effect = {}
    for model in ("random_forest", "extra_trees", "knn"):
        chrono = {
            row["train_macros"]: row["pooled_known_accuracy"]
            for row in evaluation_rows
            if row["train_order"] == "chronological" and row["model_kind"] == model
        }
        random = {
            row["train_macros"]: row["pooled_known_accuracy"]
            for row in evaluation_rows
            if row["train_order"] == "random" and row["model_kind"] == model
        }
        common = sorted(set(chrono) & set(random))
        differences = np.asarray([random[size] - chrono[size] for size in common])
        order_effect[model] = {
            "mean_random_minus_chronological_accuracy": float(np.mean(differences)),
            "mean_absolute_difference": float(np.mean(np.abs(differences))),
        }

    analysis = {
        "corpus": str(CORPUS),
        "seed_summaries": seed_summaries,
        "aggregate": {
            "median_canonical_t6_step": float(
                np.median([row["canonical_t6_step"] for row in seed_summaries])
            ),
            "median_independent_macros_at_t6": float(
                np.median([row["independent_macros_at_t6"] for row in seed_summaries])
            ),
            "median_training_rows_at_t6": float(
                np.median([row["training_rows_at_t6"] for row in seed_summaries])
            ),
            "neighbor_summary": neighbor_summary,
            "k_cell_mutual_information_nats": float(mutual_information),
            "cell_entropy_nats": float(cell_entropy),
            "k_fraction_of_cell_entropy": float(mutual_information / cell_entropy),
            "first_sustained_prevalence_gain_macros": sustained,
            "strict_first_rejection_macros": strict_first_rejection,
            "loose_filter_useful_gate": _read_json(
                LOOSE_FILTER_EVALUATION / "report.json"
            )["recommendations_min_train_macros"],
            "median_step_at_macro_count": {
                str(count): float(np.median(steps))
                for count, steps in step_at_macro_count.items()
            },
            "train_order_effect": order_effect,
        },
    }

    _write_csv(OUTPUT / "macro_timeline.csv", macro_timeline)
    _write_csv(OUTPUT / "class_balance.csv", class_rows)
    _write_csv(OUTPUT / "distance_bins.csv", distance_bins)
    _write_csv(OUTPUT / "k_cell_distribution.csv", k_rows)
    _write_csv(OUTPUT / "model_comparison.csv", evaluation_rows)
    with (OUTPUT / "analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(_finite(analysis), stream, ensure_ascii=False, indent=2)
    print(f"wrote main9 model analysis to {OUTPUT}")


if __name__ == "__main__":
    main()
