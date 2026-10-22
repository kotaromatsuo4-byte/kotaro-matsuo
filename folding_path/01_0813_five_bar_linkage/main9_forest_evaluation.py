"""Evaluate main9 terminal-cell models and their online search results.

This is the stricter companion to :mod:`forest_cell_learning_curve`.  It
keeps one *fixed chronological holdout* for every point on a nested learning
curve, and it never splits rows belonging to the same perturbation macro.
That matters when one expensive ``(q, v, K)`` rollout is relabelled into
several hindsight/residual training rows.

Expected ``training_data.npz`` schema
-------------------------------------

Required fields:

``inputs``
    Float array ``(n_rows, n_features)`` containing the encoded ``(q,v,K)``.
``cell_labels``
    Scalar terminal-cell label for each row.  All rows in one macro must have
    the same label.

Optional fields (safe fallbacks are used when absent):

``macro_ids`` (or ``rollout_ids``)
    Independent rollout/macro id.  Hindsight rows must reuse the source id.
    If absent, every row is treated as an independent macro.
``sample_weights``
    Non-negative row weights.  The forest additionally normalizes the total
    weight of every macro to one.  Defaults to one.
``macro_start_step`` (legacy alias ``start_step``)
    Mechanics step at which the macro began.  Used only for chronology;
    defaults to row order.
``is_hindsight``
    Boolean.  The earliest non-hindsight row is the single evaluation row for
    a macro.  Defaults to false.

main9 uses ``cell_labels == -1`` for corrector failures, timeouts, global
budget censoring, and ambiguous terminal descriptors.  Those rows are
reported and excluded before any split or fit; ``-1`` is never learned as a
terminal cell.

The script reports closed-set accuracy/macro-F1/Brier/NLL, but the primary
quantities for a one-sided duplicate filter are rejection precision,
duplicate rejection rate, pseudo-unknown retention, leave-one-cell-out
(LOCO) retention, and novelty enrichment among survivors.

Outputs in ``--output-directory`` are ``seed_metrics.csv``,
``aggregate_metrics.csv``, ``report.json``, and ``REPORT.md``.  When
``--online-results`` is supplied, scalar fields from each ``seed_*/summary.json``
are also aggregated without assuming a particular main9 summary revision.
Use ``--synthetic-smoke`` to generate small grouped corpora and exercise the
whole pipeline without running mechanics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from forest_cell_learning_curve import (
    _evaluation_row,
    _label_key,
    _load_corpus,
    _ordered_macros,
    _parse_sizes,
)
from forest_cell_model import (
    DuplicateFilterThresholds,
    conservative_duplicate_mask,
    fit_cell_filter,
    predict_cell_filter,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_TRAINING_SIZES = "8,16,32,64,128,256,512,1024"
MODEL_KINDS = ("random_forest", "extra_trees", "knn", "prevalence")


def _object_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _stable_classes(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    classes: list[Any] = []
    lookup: dict[Any, int] = {}
    codes = []
    for raw in values:
        value = _label_key(raw)
        if value not in lookup:
            lookup[value] = len(classes)
            classes.append(value)
        codes.append(lookup[value])
    return _object_array(classes), np.asarray(codes, dtype=int)


def _resolve_input_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()
    local = SCRIPT_DIRECTORY / "result" / path
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(f"input path was not found: {raw_path}")


def discover_corpora(arguments: Sequence[str | Path]) -> list[Path]:
    """Find at most one preferred corpus per seed directory.

    ``training_data.npz`` is preferred over the transitional
    ``cell_training_data.npz`` name, so passing a result root cannot silently
    double-count one seed.
    """

    candidates: list[Path] = []
    for argument in arguments:
        path = _resolve_input_path(argument)
        if path.is_file():
            candidates.append(path)
            continue
        direct = [path / "training_data.npz", path / "cell_training_data.npz"]
        selected = next((item for item in direct if item.is_file()), None)
        if selected is not None:
            candidates.append(selected)
            continue
        seed_directories = sorted(
            item for item in path.rglob("seed_*") if item.is_dir()
        )
        for seed_directory in seed_directories:
            preferred = seed_directory / "training_data.npz"
            legacy = seed_directory / "cell_training_data.npz"
            if preferred.is_file():
                candidates.append(preferred)
            elif legacy.is_file():
                candidates.append(legacy)
        if not seed_directories:
            candidates.extend(sorted(path.rglob("training_data.npz")))
            candidates.extend(sorted(path.rglob("cell_training_data.npz")))

    result: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = resolved.parent
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    if not result:
        raise FileNotFoundError("no training_data.npz corpus was found")
    return result


def _seed_name(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed[_-]?(.*)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1) or part
    return path.parent.name


def _macro_labels(
    data: Mapping[str, np.ndarray],
    macros: Sequence[Any],
    rows_by_macro: Mapping[Any, np.ndarray],
) -> dict[Any, Any]:
    labels: dict[Any, Any] = {}
    for macro in macros:
        rows = rows_by_macro[macro]
        keys = {_label_key(data["cell_labels"][row]) for row in rows}
        if len(keys) != 1:
            raise ValueError(
                f"macro {macro!r} has {len(keys)} terminal-cell labels; "
                "hindsight rows must share one outcome"
            )
        labels[macro] = next(iter(keys))
    return labels


def _rows_for_macros(
    macro_list: Sequence[Any], rows_by_macro: Mapping[Any, np.ndarray]
) -> np.ndarray:
    if not macro_list:
        return np.empty(0, dtype=int)
    return np.concatenate([rows_by_macro[macro] for macro in macro_list])


def _primary_rows(
    data: Mapping[str, np.ndarray],
    macro_list: Sequence[Any],
    rows_by_macro: Mapping[Any, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [_evaluation_row(data, rows_by_macro[macro]) for macro in macro_list],
        dtype=int,
    )


def _rms_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left - right
    return np.sqrt(np.mean(difference * difference, axis=-1))


class _GroupedKNN:
    """Small group-aware kNN baseline used only by the offline evaluator."""

    def __init__(self, neighbors: int = 5, support_quantile: float = 0.95):
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        self.neighbors = int(neighbors)
        self.support_quantile = float(support_quantile)

    def fit(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> "_GroupedKNN":
        x = np.asarray(inputs, dtype=float)
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1.0e-12] = 1.0
        self.x_ = (x - self.mean_) / self.scale_
        self.classes_, self.y_ = _stable_classes(labels)
        _, self.groups_ = _stable_classes(groups)
        self.group_values_ = np.unique(self.groups_)
        self.group_labels_ = np.empty(len(self.group_values_), dtype=int)
        for group in self.group_values_:
            group_labels = np.unique(self.y_[self.groups_ == group])
            if len(group_labels) != 1:
                raise ValueError("each macro group needs one terminal-cell label")
            self.group_labels_[int(group)] = int(group_labels[0])
        self.class_group_counts_ = np.bincount(
            self.group_labels_, minlength=len(self.classes_)
        )
        self.support_radii_ = self._support_radii()
        return self

    @property
    def filter_ready(self) -> bool:
        return len(self.classes_) >= 2

    def _group_distances(self, query: np.ndarray) -> np.ndarray:
        row_distances = _rms_distance(self.x_, query.reshape(1, -1))
        return np.asarray(
            [np.min(row_distances[self.groups_ == group]) for group in self.group_values_]
        )

    def _between_groups(self, left: int, right: int) -> float:
        x_left = self.x_[self.groups_ == left]
        x_right = self.x_[self.groups_ == right]
        difference = x_left[:, None, :] - x_right[None, :, :]
        return float(np.min(_rms_distance(difference, np.zeros_like(difference))))

    def _support_radii(self) -> np.ndarray:
        radii = np.full(len(self.classes_), np.nan, dtype=float)
        support_neighbor = max(1, min(3, self.neighbors))
        for cell in range(len(self.classes_)):
            groups = self.group_values_[self.group_labels_ == cell]
            if len(groups) < 2:
                continue
            distances = []
            for group in groups:
                other = [
                    self._between_groups(int(group), int(candidate))
                    for candidate in groups
                    if candidate != group
                ]
                other.sort()
                distances.append(other[min(support_neighbor, len(other)) - 1])
            radii[cell] = max(
                float(np.quantile(distances, self.support_quantile)), 1.0e-12
            )
        return radii

    def predict(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        query = (np.asarray(inputs, dtype=float) - self.mean_) / self.scale_
        probabilities = np.zeros((len(query), len(self.classes_)), dtype=float)
        agreement = np.zeros(len(query), dtype=float)
        support_distance = np.full(len(query), np.inf, dtype=float)
        predicted = np.zeros(len(query), dtype=int)
        for row, point in enumerate(query):
            distances = self._group_distances(point)
            count = min(self.neighbors, len(distances))
            selected = np.argsort(distances, kind="stable")[:count]
            selected_distance = distances[selected]
            selected_labels = self.group_labels_[selected]
            weights = 1.0 / np.maximum(selected_distance, 1.0e-9)
            scores = np.bincount(
                selected_labels, weights=weights, minlength=len(self.classes_)
            ).astype(float)
            # A tiny group-prevalence prior avoids literal zero probabilities.
            scores += 1.0e-6 * self.class_group_counts_
            probabilities[row] = scores / np.sum(scores)
            predicted[row] = int(np.argmax(probabilities[row]))
            agreement[row] = float(np.mean(selected_labels == predicted[row]))
            class_distances = np.sort(
                distances[self.group_labels_ == predicted[row]]
            )
            support_index = min(max(1, min(3, self.neighbors)), len(class_distances)) - 1
            support_distance[row] = class_distances[support_index]

        indices = np.arange(len(query))
        radii = self.support_radii_[predicted]
        support_ratio = np.divide(
            support_distance,
            radii,
            out=np.full(len(query), np.inf, dtype=float),
            where=np.isfinite(radii) & (radii > 0.0),
        )
        labels = _object_array([self.classes_[cell] for cell in predicted])
        return {
            "classes": self.classes_.copy(),
            "probabilities": probabilities,
            "predicted_class_indices": predicted,
            "predicted_labels": labels,
            "max_probability": probabilities[indices, predicted],
            "lower_probability": probabilities[indices, predicted],
            "committee_agreement": agreement,
            "epistemic": 1.0 - agreement,
            "support_distance": support_distance,
            "support_ratio": support_ratio,
            "class_group_count": self.class_group_counts_[predicted],
        }


class _Prevalence:
    """Feature-blind group-prevalence baseline; it is never in support."""

    def fit(self, labels: np.ndarray, groups: np.ndarray) -> "_Prevalence":
        classes, y = _stable_classes(labels)
        _, group_codes = _stable_classes(groups)
        group_labels = []
        for group in np.unique(group_codes):
            values = np.unique(y[group_codes == group])
            if len(values) != 1:
                raise ValueError("each macro group needs one terminal-cell label")
            group_labels.append(int(values[0]))
        counts = np.bincount(group_labels, minlength=len(classes)).astype(float)
        self.classes_ = classes
        self.counts_ = counts.astype(int)
        self.probability_ = counts / np.sum(counts)
        return self

    @property
    def filter_ready(self) -> bool:
        # It can classify, but cannot justify a local duplicate rejection.
        return False

    def predict(self, row_count: int) -> dict[str, np.ndarray]:
        probability = np.repeat(self.probability_.reshape(1, -1), row_count, axis=0)
        predicted = np.argmax(probability, axis=1)
        indices = np.arange(row_count)
        labels = _object_array([self.classes_[cell] for cell in predicted])
        return {
            "classes": self.classes_.copy(),
            "probabilities": probability,
            "predicted_class_indices": predicted,
            "predicted_labels": labels,
            "max_probability": probability[indices, predicted],
            "lower_probability": probability[indices, predicted],
            "committee_agreement": probability[indices, predicted],
            "epistemic": 1.0 - probability[indices, predicted],
            "support_distance": np.full(row_count, np.inf),
            "support_ratio": np.full(row_count, np.inf),
            "class_group_count": self.counts_[predicted],
        }


def _fit_and_predict(
    model_kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    train_weights: np.ndarray,
    test_x: np.ndarray,
    *,
    n_estimators: int,
    knn_neighbors: int,
    rng_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    if model_kind in ("random_forest", "extra_trees"):
        model = fit_cell_filter(
            train_x,
            train_y,
            train_groups,
            train_weights,
            model_kind=model_kind,
            rng_seed=rng_seed,
            n_estimators=n_estimators,
        )
        prediction = predict_cell_filter(model, test_x)
        return model.classes_, prediction, bool(model.filter_ready)
    if model_kind == "knn":
        model = _GroupedKNN(neighbors=knn_neighbors).fit(
            train_x, train_y, train_groups
        )
        return model.classes_, model.predict(test_x), model.filter_ready
    if model_kind == "prevalence":
        model = _Prevalence().fit(train_y, train_groups)
        return model.classes_, model.predict(len(test_x)), model.filter_ready
    raise ValueError(f"unknown model kind: {model_kind}")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | int]:
    lookup = {_label_key(value): index for index, value in enumerate(classes)}
    rows = [row for row, value in enumerate(labels) if _label_key(value) in lookup]
    if not rows:
        return {
            "known_test_count": 0,
            "known_correct_count": 0,
            "known_accuracy": float("nan"),
            "known_macro_f1": float("nan"),
            "known_nll": float("nan"),
            "known_brier": float("nan"),
        }
    rows_array = np.asarray(rows, dtype=int)
    target = np.asarray([lookup[_label_key(labels[row])] for row in rows], dtype=int)
    selected_probability = np.maximum(
        probabilities[rows_array, target], 1.0e-15
    )
    predicted = np.argmax(probabilities[rows_array], axis=1)
    one_hot = np.zeros_like(probabilities[rows_array])
    one_hot[np.arange(len(rows_array)), target] = 1.0
    f1_values = []
    for cell in np.unique(target):
        true_positive = int(np.count_nonzero((target == cell) & (predicted == cell)))
        false_positive = int(np.count_nonzero((target != cell) & (predicted == cell)))
        false_negative = int(np.count_nonzero((target == cell) & (predicted != cell)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(_safe_ratio(2 * true_positive, denominator))
    correct = int(np.count_nonzero(predicted == target))
    return {
        "known_test_count": int(len(rows_array)),
        "known_correct_count": correct,
        "known_accuracy": _safe_ratio(correct, len(rows_array)),
        "known_macro_f1": float(np.nanmean(f1_values)),
        "known_nll": float(-np.mean(np.log(selected_probability))),
        "known_brier": float(
            np.mean(np.sum((probabilities[rows_array] - one_hot) ** 2, axis=1))
        ),
    }


def _fixed_universe_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Score a fixed evaluation universe, counting unseen classes as errors.

    ``_classification_metrics`` deliberately conditions on classes already
    present in training.  That conditional score is useful diagnostically,
    but its denominator changes while new cells are first observed.  These
    metrics keep every supplied macro in the denominator: a true class absent
    from ``classes`` has probability zero and is therefore an incorrect
    prediction.  This is the appropriate open-world learning-curve score.
    """

    labels_array = np.asarray(labels)
    conditional = _classification_metrics(labels_array, probabilities, classes)
    total = int(len(labels_array))
    known = int(conditional["known_test_count"])
    correct = int(conditional["known_correct_count"])
    return {
        f"{prefix}_count": total,
        f"{prefix}_known_count": known,
        f"{prefix}_correct_count": correct,
        f"{prefix}_class_coverage": _safe_ratio(known, total),
        f"{prefix}_accuracy": _safe_ratio(correct, total),
        f"{prefix}_known_only_accuracy": conditional["known_accuracy"],
    }


def _filter_metrics(
    labels: np.ndarray,
    reject: np.ndarray,
    known_classes: Iterable[Any],
) -> dict[str, float | int]:
    known = {_label_key(value) for value in known_classes}
    duplicate = np.asarray(
        [_label_key(value) in known for value in labels], dtype=bool
    )
    unknown = ~duplicate
    rejected = np.asarray(reject, dtype=bool)
    survivors = ~rejected
    duplicate_rejected = int(np.count_nonzero(rejected & duplicate))
    unknown_retained = int(np.count_nonzero(survivors & unknown))
    survivor_unknown = int(np.count_nonzero(survivors & unknown))
    baseline_novelty = float(np.mean(unknown)) if len(unknown) else float("nan")
    survivor_novelty = (
        float(np.mean(unknown[survivors])) if np.any(survivors) else float("nan")
    )
    return {
        "test_macro_count": int(len(labels)),
        "duplicate_count": int(np.count_nonzero(duplicate)),
        "pseudo_unknown_count": int(np.count_nonzero(unknown)),
        "rejected_count": int(np.count_nonzero(rejected)),
        "survivor_count": int(np.count_nonzero(survivors)),
        "duplicate_rejected_count": duplicate_rejected,
        "pseudo_unknown_retained_count": unknown_retained,
        "survivor_unknown_count": survivor_unknown,
        "reject_precision": _safe_ratio(duplicate_rejected, np.count_nonzero(rejected)),
        "duplicate_rejection_rate": _safe_ratio(
            duplicate_rejected, np.count_nonzero(duplicate)
        ),
        "pseudo_unknown_retention": _safe_ratio(
            unknown_retained, np.count_nonzero(unknown)
        ),
        "baseline_novelty_rate": baseline_novelty,
        "survivor_novelty_rate": survivor_novelty,
        "survivor_novelty_enrichment": _safe_ratio(
            survivor_novelty, baseline_novelty
        ),
    }


def _loco_retention(
    *,
    model_kind: str,
    data: Mapping[str, np.ndarray],
    train_macros: Sequence[Any],
    holdout_macros: Sequence[Any],
    macro_labels: Mapping[Any, Any],
    rows_by_macro: Mapping[Any, np.ndarray],
    thresholds: DuplicateFilterThresholds,
    n_estimators: int,
    knn_neighbors: int,
    rng_seed: int,
) -> dict[str, float | int]:
    """Hide each training cell in turn and measure whether it survives.

    A target cell is removed at *macro* granularity.  Only primary rows from
    the fixed holdout are queried.  This emulates a genuinely unseen cell
    without pretending that a closed-set classifier has an unknown class.
    """

    holdout_by_label: dict[Any, list[Any]] = defaultdict(list)
    for macro in holdout_macros:
        holdout_by_label[macro_labels[macro]].append(macro)
    train_labels = []
    for macro in train_macros:
        label = macro_labels[macro]
        if label not in train_labels:
            train_labels.append(label)

    retained_total = 0
    queried_total = 0
    evaluated_cells = 0
    for cell_index, held_out_cell in enumerate(train_labels):
        query_macros = holdout_by_label.get(held_out_cell, [])
        if not query_macros:
            continue
        remaining = [
            macro
            for macro in train_macros
            if macro_labels[macro] != held_out_cell
        ]
        query_rows = _primary_rows(data, query_macros, rows_by_macro)
        queried_total += len(query_rows)
        evaluated_cells += 1
        if not remaining:
            retained_total += len(query_rows)
            continue
        train_rows = _rows_for_macros(remaining, rows_by_macro)
        classes, prediction, filter_ready = _fit_and_predict(
            model_kind,
            data["inputs"][train_rows],
            data["cell_labels"][train_rows],
            data["macro_ids"][train_rows],
            data["sample_weights"][train_rows],
            data["inputs"][query_rows],
            n_estimators=n_estimators,
            knn_neighbors=knn_neighbors,
            rng_seed=rng_seed + 104729 * (cell_index + 1),
        )
        if filter_ready:
            reject = conservative_duplicate_mask(
                prediction, classes, thresholds=thresholds
            )
        else:
            reject = np.zeros(len(query_rows), dtype=bool)
        retained_total += int(np.count_nonzero(~reject))
    return {
        "loco_cells_evaluated": int(evaluated_cells),
        "loco_unknown_count": int(queried_total),
        "loco_unknown_retained_count": int(retained_total),
        "loco_unknown_retention": _safe_ratio(retained_total, queried_total),
    }


def _missing_schema_fields(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as saved:
        keys = set(saved.files)
    missing = []
    if not ({"macro_ids", "rollout_ids"} & keys):
        missing.append("macro_ids/rollout_ids")
    if "sample_weights" not in keys:
        missing.append("sample_weights")
    if not ({"macro_start_step", "start_step"} & keys):
        missing.append("macro_start_step/start_step")
    if "is_hindsight" not in keys:
        missing.append("is_hindsight")
    return missing


def _valid_terminal_label(value: Any) -> bool:
    """Recognize main9's negative integer censoring sentinel."""

    value = _label_key(value)
    if value is None or value == "":
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and value >= 0)
    # Hashable non-numeric labels remain supported for other structures.
    return True


def _load_main9_corpus(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load shared fields and aliases, then remove censored macro rows."""

    data = _load_corpus(path)
    with np.load(path, allow_pickle=True) as saved:
        keys = set(saved.files)
        if "macro_ids" not in keys and "rollout_ids" in keys:
            data["macro_ids"] = np.asarray(
                saved["rollout_ids"], dtype=object
            ).reshape(-1)
        if "macro_start_step" in keys:
            data["start_step"] = np.asarray(
                saved["macro_start_step"], dtype=float
            ).reshape(-1)

    row_count = len(data["inputs"])
    for name in ("macro_ids", "start_step"):
        if len(data[name]) != row_count:
            raise ValueError(f"unaligned {name} alias in {path}")
    valid = np.asarray(
        [_valid_terminal_label(value) for value in data["cell_labels"]],
        dtype=bool,
    )
    ignored_rows = int(np.count_nonzero(~valid))
    invalid_macros = {
        _label_key(data["macro_ids"][row]) for row in np.flatnonzero(~valid)
    }
    valid_macros = {
        _label_key(data["macro_ids"][row]) for row in np.flatnonzero(valid)
    }
    overlap = invalid_macros & valid_macros
    if overlap:
        raise ValueError(
            f"{len(overlap)} macro id(s) mix valid and censored terminal labels"
        )
    if not np.any(valid):
        raise ValueError("corpus has no completed, unambiguous terminal cell")
    filtered = {name: np.asarray(values)[valid] for name, values in data.items()}
    return filtered, {
        "source_rows": int(row_count),
        "ignored_censored_rows": ignored_rows,
        "ignored_censored_macros": int(len(invalid_macros)),
    }


def evaluate_corpus(
    path: Path,
    *,
    training_sizes: Sequence[int],
    holdout_macros: int,
    holdout_fraction: float,
    train_order: str,
    model_kinds: Sequence[str],
    n_estimators: int,
    knn_neighbors: int,
    rng_seed: int,
    thresholds: DuplicateFilterThresholds,
    skip_loco: bool,
    target_reject_precision: float,
    target_unknown_retention: float,
    min_duplicate_rejection: float,
) -> list[dict[str, Any]]:
    data, censoring = _load_main9_corpus(path)
    macros, rows_by_macro = _ordered_macros(data)
    if len(macros) < 3:
        raise ValueError("at least three independent macros are required")
    macro_labels = _macro_labels(data, macros, rows_by_macro)

    if holdout_macros > 0:
        test_count = int(holdout_macros)
    else:
        test_count = max(1, int(math.ceil(len(macros) * holdout_fraction)))
    if test_count >= len(macros):
        raise ValueError(
            f"fixed holdout ({test_count}) leaves no train macro from {len(macros)}"
        )
    holdout = list(macros[-test_count:])
    train_pool = list(macros[:-test_count])
    if train_order == "random":
        # One permutation is reused by all sizes, hence prefixes remain nested.
        permutation = np.random.default_rng(rng_seed).permutation(len(train_pool))
        train_pool = [train_pool[index] for index in permutation]
    elif train_order != "chronological":
        raise ValueError("train_order must be chronological or random")

    holdout_rows = _primary_rows(data, holdout, rows_by_macro)
    # A second, fixed universe contains every completed independent macro.
    # It is intentionally reported only as an *apparent* all-corpus score,
    # because the selected training macros are part of this query set.
    all_primary_rows = _primary_rows(data, macros, rows_by_macro)
    missing = _missing_schema_fields(path)
    output: list[dict[str, Any]] = []
    for requested_size in training_sizes:
        if requested_size > len(train_pool):
            continue
        selected_macros = train_pool[:requested_size]
        if set(selected_macros) & set(holdout):
            raise RuntimeError("internal error: a macro leaked into fixed holdout")
        train_rows = _rows_for_macros(selected_macros, rows_by_macro)
        for model_index, model_kind in enumerate(model_kinds):
            fit_seed = int(
                rng_seed + requested_size * 7919 + model_index * 1009
            )
            query_rows = np.concatenate((holdout_rows, all_primary_rows))
            classes, combined_prediction, filter_ready = _fit_and_predict(
                model_kind,
                data["inputs"][train_rows],
                data["cell_labels"][train_rows],
                data["macro_ids"][train_rows],
                data["sample_weights"][train_rows],
                data["inputs"][query_rows],
                n_estimators=n_estimators,
                knn_neighbors=knn_neighbors,
                rng_seed=fit_seed,
            )
            holdout_count = len(holdout_rows)
            prediction = {
                name: values[:holdout_count]
                if isinstance(values, np.ndarray) and len(values) == len(query_rows)
                else values
                for name, values in combined_prediction.items()
            }
            all_corpus_prediction = {
                name: values[holdout_count:]
                if isinstance(values, np.ndarray) and len(values) == len(query_rows)
                else values
                for name, values in combined_prediction.items()
            }
            if filter_ready:
                reject = conservative_duplicate_mask(
                    prediction, classes, thresholds=thresholds
                )
            else:
                reject = np.zeros(len(holdout_rows), dtype=bool)
            row: dict[str, Any] = {
                "corpus": str(path),
                "result_name": path.parent.parent.name,
                "seed": _seed_name(path),
                "model_kind": model_kind,
                "train_order": train_order,
                "train_macros": int(requested_size),
                "train_rows": int(len(train_rows)),
                "train_classes": int(len(classes)),
                "available_train_macros": int(len(train_pool)),
                "fixed_holdout_macros": int(len(holdout)),
                "filter_ready": bool(filter_ready),
                "missing_optional_fields": ",".join(missing),
                **censoring,
            }
            row.update(
                _classification_metrics(
                    data["cell_labels"][holdout_rows],
                    prediction["probabilities"],
                    classes,
                )
            )
            row.update(
                _fixed_universe_metrics(
                    data["cell_labels"][holdout_rows],
                    prediction["probabilities"],
                    classes,
                    prefix="fixed_universe",
                )
            )
            row.update(
                _fixed_universe_metrics(
                    data["cell_labels"][all_primary_rows],
                    all_corpus_prediction["probabilities"],
                    classes,
                    prefix="all_corpus_apparent",
                )
            )
            row.update(
                _filter_metrics(
                    data["cell_labels"][holdout_rows], reject, classes
                )
            )
            if skip_loco:
                row.update(
                    {
                        "loco_cells_evaluated": 0,
                        "loco_unknown_count": 0,
                        "loco_unknown_retained_count": 0,
                        "loco_unknown_retention": float("nan"),
                    }
                )
            else:
                row.update(
                    _loco_retention(
                        model_kind=model_kind,
                        data=data,
                        train_macros=selected_macros,
                        holdout_macros=holdout,
                        macro_labels=macro_labels,
                        rows_by_macro=rows_by_macro,
                        thresholds=thresholds,
                        n_estimators=n_estimators,
                        knn_neighbors=knn_neighbors,
                        rng_seed=fit_seed,
                    )
                )
            retention = row["loco_unknown_retention"]
            if not np.isfinite(retention):
                retention = row["pseudo_unknown_retention"]
            row["useful_filter"] = bool(
                np.isfinite(row["reject_precision"])
                and row["reject_precision"] >= target_reject_precision
                and np.isfinite(retention)
                and retention >= target_unknown_retention
                and np.isfinite(row["duplicate_rejection_rate"])
                and row["duplicate_rejection_rate"] >= min_duplicate_rejection
            )
            output.append(row)
    if not output:
        raise ValueError(
            f"no requested training size fits the {len(train_pool)}-macro train pool"
        )
    return output


COUNT_FIELDS = (
    "known_test_count",
    "known_correct_count",
    "test_macro_count",
    "duplicate_count",
    "pseudo_unknown_count",
    "rejected_count",
    "survivor_count",
    "duplicate_rejected_count",
    "pseudo_unknown_retained_count",
    "survivor_unknown_count",
    "loco_unknown_count",
    "loco_unknown_retained_count",
    "fixed_universe_count",
    "fixed_universe_known_count",
    "fixed_universe_correct_count",
    "all_corpus_apparent_count",
    "all_corpus_apparent_known_count",
    "all_corpus_apparent_correct_count",
)

MEAN_FIELDS = (
    "known_accuracy",
    "known_macro_f1",
    "known_nll",
    "known_brier",
    "reject_precision",
    "duplicate_rejection_rate",
    "pseudo_unknown_retention",
    "baseline_novelty_rate",
    "survivor_novelty_rate",
    "survivor_novelty_enrichment",
    "loco_unknown_retention",
    "fixed_universe_class_coverage",
    "fixed_universe_accuracy",
    "fixed_universe_known_only_accuracy",
    "all_corpus_apparent_class_coverage",
    "all_corpus_apparent_accuracy",
    "all_corpus_apparent_known_only_accuracy",
)


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def aggregate_learning_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["result_name"],
                row["model_kind"],
                int(row["train_macros"]),
                row["train_order"],
            )
        ].append(row)
    output = []
    for (
        result_name,
        model_kind,
        train_macros,
        train_order,
    ), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "result_name": result_name,
            "model_kind": model_kind,
            "train_order": train_order,
            "train_macros": train_macros,
            "seeds": len(group),
            "useful_seed_fraction": float(
                np.mean([bool(row["useful_filter"]) for row in group])
            ),
        }
        for field in COUNT_FIELDS:
            item[f"{field}_sum"] = int(sum(int(row[field]) for row in group))
        for field in MEAN_FIELDS:
            values = _finite(row[field] for row in group)
            item[f"{field}_mean"] = statistics.fmean(values) if values else float("nan")
            item[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
            )
            item[f"{field}_n"] = len(values)

        item["pooled_known_accuracy"] = _safe_ratio(
            item["known_correct_count_sum"], item["known_test_count_sum"]
        )
        item["pooled_fixed_universe_accuracy"] = _safe_ratio(
            item["fixed_universe_correct_count_sum"],
            item["fixed_universe_count_sum"],
        )
        item["pooled_fixed_universe_class_coverage"] = _safe_ratio(
            item["fixed_universe_known_count_sum"],
            item["fixed_universe_count_sum"],
        )
        item["pooled_all_corpus_apparent_accuracy"] = _safe_ratio(
            item["all_corpus_apparent_correct_count_sum"],
            item["all_corpus_apparent_count_sum"],
        )
        item["pooled_all_corpus_apparent_class_coverage"] = _safe_ratio(
            item["all_corpus_apparent_known_count_sum"],
            item["all_corpus_apparent_count_sum"],
        )
        item["pooled_reject_precision"] = _safe_ratio(
            item["duplicate_rejected_count_sum"], item["rejected_count_sum"]
        )
        item["pooled_duplicate_rejection_rate"] = _safe_ratio(
            item["duplicate_rejected_count_sum"], item["duplicate_count_sum"]
        )
        item["pooled_pseudo_unknown_retention"] = _safe_ratio(
            item["pseudo_unknown_retained_count_sum"],
            item["pseudo_unknown_count_sum"],
        )
        item["pooled_loco_unknown_retention"] = _safe_ratio(
            item["loco_unknown_retained_count_sum"], item["loco_unknown_count_sum"]
        )
        baseline = _safe_ratio(
            item["pseudo_unknown_count_sum"], item["test_macro_count_sum"]
        )
        survivor = _safe_ratio(
            item["survivor_unknown_count_sum"], item["survivor_count_sum"]
        )
        item["pooled_survivor_novelty_enrichment"] = _safe_ratio(survivor, baseline)
        output.append(item)
    return output


def recommend_training_sizes(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    target_reject_precision: float,
    target_unknown_retention: float,
    min_duplicate_rejection: float,
    min_useful_seed_fraction: float,
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    result_models = sorted(
        {
            (str(row["result_name"]), str(row["model_kind"]))
            for row in aggregate_rows
        }
    )
    for result_name, model_kind in result_models:
        candidates = []
        for row in aggregate_rows:
            if (
                row["result_name"] != result_name
                or row["model_kind"] != model_kind
            ):
                continue
            retention = row["pooled_loco_unknown_retention"]
            if not np.isfinite(retention):
                retention = row["pooled_pseudo_unknown_retention"]
            if (
                np.isfinite(row["pooled_reject_precision"])
                and row["pooled_reject_precision"] >= target_reject_precision
                and np.isfinite(retention)
                and retention >= target_unknown_retention
                and np.isfinite(row["pooled_duplicate_rejection_rate"])
                and row["pooled_duplicate_rejection_rate"] >= min_duplicate_rejection
                and row["useful_seed_fraction"] >= min_useful_seed_fraction
            ):
                candidates.append(int(row["train_macros"]))
        recommendations[f"{result_name}/{model_kind}"] = (
            min(candidates) if candidates else None
        )
    return recommendations


def discover_online_summaries(arguments: Sequence[str | Path]) -> list[Path]:
    result = []
    seen = set()
    for argument in arguments:
        path = _resolve_input_path(argument)
        candidates = [path] if path.is_file() else list(path.rglob("summary.json"))
        for candidate in candidates:
            if candidate.name != "summary.json":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return sorted(result)


def load_online_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row: dict[str, Any] = {"source": str(path)}
        for key, value in payload.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                row[key] = value
        row.setdefault("seed", _seed_name(path))
        row.setdefault("selector", payload.get("algorithm_version", path.parent.parent.name))
        row.setdefault("result_name", path.parent.parent.name)
        rows.append(row)
    return rows


def aggregate_online_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("selector", "unknown"))].append(row)
    output = []
    excluded = {"seed", "source", "selector"}
    for selector, group in sorted(groups.items()):
        item: dict[str, Any] = {"selector": selector, "seeds": len(group)}
        keys = sorted(set().union(*(row.keys() for row in group)) - excluded)
        for key in keys:
            values = _finite(row.get(key) for row in group)
            if not values:
                continue
            item[f"{key}_mean"] = statistics.fmean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            item[f"{key}_n"] = len(values)
        output.append(item)
    return output


def pair_online_rows(
    rows: Sequence[Mapping[str, Any]], baseline_selector: str | None
) -> list[dict[str, Any]]:
    selectors = sorted({str(row.get("selector", "unknown")) for row in rows})
    if baseline_selector is None:
        baseline_selector = next(
            (selector for selector in selectors if "random" in selector.lower()), None
        )
    if baseline_selector is None or baseline_selector not in selectors:
        return []
    by_selector_seed = {
        (str(row.get("selector")), str(row.get("seed"))): row for row in rows
    }
    excluded = {"seed", "source", "selector"}
    output = []
    for selector in selectors:
        if selector == baseline_selector:
            continue
        paired_seeds = sorted(
            {str(row.get("seed")) for row in rows if row.get("selector") == selector}
            & {
                str(row.get("seed"))
                for row in rows
                if row.get("selector") == baseline_selector
            }
        )
        keys = set()
        for seed in paired_seeds:
            keys |= set(by_selector_seed[(selector, seed)])
            keys |= set(by_selector_seed[(baseline_selector, seed)])
        for key in sorted(keys - excluded):
            differences = []
            for seed in paired_seeds:
                left = by_selector_seed[(selector, seed)].get(key)
                right = by_selector_seed[(baseline_selector, seed)].get(key)
                left_values = _finite([left])
                right_values = _finite([right])
                if left_values and right_values:
                    differences.append(left_values[0] - right_values[0])
            if not differences:
                continue
            output.append(
                {
                    "selector": selector,
                    "baseline_selector": baseline_selector,
                    "metric": key,
                    "paired_seeds": len(differences),
                    "delta_mean": statistics.fmean(differences),
                    "delta_std": statistics.stdev(differences) if len(differences) > 1 else 0.0,
                }
            )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(value) for key, value in row.items()})


def _format_metric(value: Any, digits: int = 3) -> str:
    values = _finite([value])
    return f"{values[0]:.{digits}f}" if values else "--"


def _markdown_report(
    aggregate_rows: Sequence[Mapping[str, Any]],
    recommendations: Mapping[str, Any],
    online_aggregate: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> str:
    lines = [
        "# main9 terminal-cell filter evaluation",
        "",
        "All learning-curve points use the same final chronological holdout. "
        "Rows from one macro are never divided between train and test.",
        "",
        "## Offline learning curve",
        "",
        "| result | model | train macros | seeds | accuracy | macro F1 | NLL | Brier | reject precision | duplicate rejection | LOCO retention | pseudo-unknown retention | novelty lift | useful seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {result} | {model} | {train} | {seeds} | {accuracy} | {f1} | {nll} | "
            "{brier} | {precision} | {duplicate} | {loco} | {unknown} | "
            "{lift} | {useful} |".format(
                result=row["result_name"],
                model=row["model_kind"],
                train=row["train_macros"],
                seeds=row["seeds"],
                accuracy=_format_metric(row["pooled_known_accuracy"]),
                f1=_format_metric(row["known_macro_f1_mean"]),
                nll=_format_metric(row["known_nll_mean"]),
                brier=_format_metric(row["known_brier_mean"]),
                precision=_format_metric(row["pooled_reject_precision"]),
                duplicate=_format_metric(row["pooled_duplicate_rejection_rate"]),
                loco=_format_metric(row["pooled_loco_unknown_retention"]),
                unknown=_format_metric(row["pooled_pseudo_unknown_retention"]),
                lift=_format_metric(row["pooled_survivor_novelty_enrichment"]),
                useful=_format_metric(row["useful_seed_fraction"]),
            )
        )
    lines += ["", "## Earliest useful-filter size", ""]
    for model, size in recommendations.items():
        value = f"{size} independent macros" if size is not None else "not demonstrated"
        lines.append(f"- `{model}`: {value}")
    lines += [
        "",
        "A point is called useful only when its rejection precision, unknown-cell "
        "retention, duplicate rejection rate, and useful-seed fraction pass the "
        "configured thresholds. This is an operational gate, not a claim of "
        "statistical significance.",
        "",
        "## Expected NPZ schema",
        "",
        "- Required: `inputs`, `cell_labels`.",
        "- Recommended: `macro_ids` (or `rollout_ids`), `sample_weights`, `macro_start_step` (or `start_step`), `is_hindsight`.",
        "- Every row with the same `macro_ids` value must have the same terminal-cell label.",
        "- `cell_labels == -1` rows are censored and excluded, never treated as a terminal class.",
        "- Missing `macro_ids` is accepted, but then row leakage cannot be detected because every row is treated as independent.",
        "",
        "Configured gate: reject precision >= {p:.3f}, unknown retention >= {u:.3f}, "
        "duplicate rejection >= {d:.3f}, useful-seed fraction >= {s:.3f}.".format(
            p=config["target_reject_precision"],
            u=config["target_unknown_retention"],
            d=config["min_duplicate_rejection"],
            s=config["min_useful_seed_fraction"],
        ),
    ]
    if online_aggregate:
        priority = (
            "feasible_elite_cells_mean",
            "kinematic_cells_mean",
            "elite_cells_mean",
            "discovery_auc_mean",
            "feasible_discovery_auc_mean",
            "kinematic_discovery_auc_mean",
            "feasible_auc_mean",
            "kinematic_auc_mean",
            "used_steps_mean",
            "wall_time_seconds_mean",
        )
        metrics = [
            metric
            for metric in priority
            if any(metric in row for row in online_aggregate)
        ]
        lines += ["", "## Online result summaries", ""]
        header = "| selector | seeds | " + " | ".join(metrics) + " |"
        separator = "|---|---:|" + "---:|" * len(metrics)
        lines += [header, separator]
        for row in online_aggregate:
            lines.append(
                "| {selector} | {seeds} | {values} |".format(
                    selector=row["selector"],
                    seeds=row["seeds"],
                    values=" | ".join(_format_metric(row.get(metric)) for metric in metrics),
                )
            )
    if skipped:
        lines += ["", "## Skipped inputs", ""]
        lines.extend(f"- `{row['path']}`: {row['reason']}" for row in skipped)
    return "\n".join(lines) + "\n"


def generate_synthetic_corpora(
    output_directory: Path, *, seeds: int, macros: int, rng_seed: int
) -> list[Path]:
    """Create a small nonlinear grouped corpus for a reproducible smoke test."""

    if seeds <= 0 or macros < 16:
        raise ValueError("synthetic seeds must be positive and macros >= 16")
    paths = []
    feature_count = 9
    class_count = 6
    holdout_start = int(0.75 * macros)
    root = output_directory / "synthetic_inputs"
    for seed_index in range(seeds):
        rng = np.random.default_rng(rng_seed + seed_index)
        centers = rng.normal(scale=2.0, size=(class_count, feature_count))
        inputs = []
        labels = []
        macro_ids = []
        weights = []
        steps = []
        hindsight = []
        for macro in range(macros):
            # Cell 5 exists only in the fixed tail, giving a natural unknown.
            if macro == 1:
                # Exercise main9's censored-macro sentinel in every smoke run.
                cell = -1
            elif macro >= holdout_start and macro % 4 == 0:
                cell = 5
            else:
                cell = int((macro * 3 + macro // 5 + seed_index) % 5)
            center = centers[cell] if cell >= 0 else rng.normal(size=feature_count)
            base = center + 0.25 * rng.normal(size=feature_count)
            base[0] += 0.35 * math.sin(base[1] * base[2])
            for suffix in range(2):
                inputs.append(base + 0.08 * suffix * rng.normal(size=feature_count))
                labels.append(cell)
                macro_ids.append(macro)
                weights.append(0.5)
                steps.append(float(macro) + 0.01 * suffix)
                hindsight.append(bool(suffix))
        seed_directory = root / f"seed_{rng_seed + seed_index}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        path = seed_directory / "training_data.npz"
        np.savez_compressed(
            path,
            inputs=np.asarray(inputs, dtype=float),
            cell_labels=np.asarray(labels, dtype=int),
            rollout_ids=np.asarray(macro_ids, dtype=int),
            sample_weights=np.asarray(weights, dtype=float),
            macro_start_step=np.asarray(steps, dtype=float),
            is_hindsight=np.asarray(hindsight, dtype=bool),
        )
        paths.append(path.resolve())
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpora",
        nargs="*",
        help="training_data.npz, seed directory, or result root",
    )
    parser.add_argument(
        "--training-sizes",
        nargs="+",
        default=[DEFAULT_TRAINING_SIZES],
        help="nested independent-macro counts (comma or space separated)",
    )
    parser.add_argument(
        "--holdout-macros",
        type=int,
        default=64,
        help="fixed final chronological macros; 0 uses --holdout-fraction",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--train-order",
        choices=("chronological", "random"),
        default="chronological",
        help="random uses one seeded permutation shared by all nested sizes",
    )
    parser.add_argument(
        "--model-kinds", nargs="+", choices=MODEL_KINDS, default=MODEL_KINDS
    )
    parser.add_argument("--n-estimators", type=int, default=64)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=20260817)
    parser.add_argument("--skip-loco", action="store_true")
    parser.add_argument("--target-reject-precision", type=float, default=0.90)
    parser.add_argument("--target-unknown-retention", type=float, default=0.95)
    parser.add_argument("--min-duplicate-rejection", type=float, default=0.05)
    parser.add_argument("--min-useful-seed-fraction", type=float, default=0.50)
    parser.add_argument("--mean-probability", type=float, default=0.90)
    parser.add_argument("--lower-probability", type=float, default=0.70)
    parser.add_argument("--vote-fraction", type=float, default=0.80)
    parser.add_argument("--max-disagreement", type=float, default=0.08)
    parser.add_argument("--max-support-ratio", type=float, default=1.00)
    parser.add_argument("--min-groups-per-class", type=int, default=4)
    parser.add_argument(
        "--online-results",
        nargs="*",
        default=[],
        help="optional main9 result roots containing seed_*/summary.json",
    )
    parser.add_argument("--online-baseline-selector", default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-seeds", type=int, default=2)
    parser.add_argument("--synthetic-macros", type=int, default=96)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "main9_forest_evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.holdout_macros < 0:
        raise ValueError("holdout-macros must be nonnegative")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("holdout-fraction must be in (0, 1)")
    if args.n_estimators <= 0 or args.knn_neighbors <= 0:
        raise ValueError("estimator and neighbor counts must be positive")
    for value in (
        args.target_reject_precision,
        args.target_unknown_retention,
        args.min_duplicate_rejection,
        args.min_useful_seed_fraction,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("operational gate thresholds must be in [0, 1]")

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.synthetic_smoke:
        corpus_paths = generate_synthetic_corpora(
            output_directory,
            seeds=args.synthetic_seeds,
            macros=args.synthetic_macros,
            rng_seed=args.rng_seed,
        )
    else:
        if not args.corpora:
            raise ValueError("pass a corpus/result root or use --synthetic-smoke")
        corpus_paths = discover_corpora(args.corpora)

    training_sizes = _parse_sizes(args.training_sizes)
    thresholds = DuplicateFilterThresholds(
        mean_probability=args.mean_probability,
        lower_probability=args.lower_probability,
        vote_fraction=args.vote_fraction,
        max_disagreement=args.max_disagreement,
        max_support_ratio=args.max_support_ratio,
        min_groups_per_class=args.min_groups_per_class,
    )
    seed_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for corpus_index, path in enumerate(corpus_paths):
        try:
            seed_rows.extend(
                evaluate_corpus(
                    path,
                    training_sizes=training_sizes,
                    holdout_macros=args.holdout_macros,
                    holdout_fraction=args.holdout_fraction,
                    train_order=args.train_order,
                    model_kinds=args.model_kinds,
                    n_estimators=args.n_estimators,
                    knn_neighbors=args.knn_neighbors,
                    rng_seed=args.rng_seed + corpus_index * 1_000_003,
                    thresholds=thresholds,
                    skip_loco=args.skip_loco,
                    target_reject_precision=args.target_reject_precision,
                    target_unknown_retention=args.target_unknown_retention,
                    min_duplicate_rejection=args.min_duplicate_rejection,
                )
            )
        except (ValueError, RuntimeError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    if not seed_rows:
        detail = "; ".join(f"{row['path']}: {row['reason']}" for row in skipped)
        raise RuntimeError(f"no compatible corpus could be evaluated. {detail}")

    aggregate_rows = aggregate_learning_rows(seed_rows)
    recommendations = recommend_training_sizes(
        aggregate_rows,
        target_reject_precision=args.target_reject_precision,
        target_unknown_retention=args.target_unknown_retention,
        min_duplicate_rejection=args.min_duplicate_rejection,
        min_useful_seed_fraction=args.min_useful_seed_fraction,
    )

    online_paths = discover_online_summaries(args.online_results)
    online_rows = load_online_rows(online_paths)
    online_aggregate = aggregate_online_rows(online_rows)
    online_pairs = pair_online_rows(online_rows, args.online_baseline_selector)

    config = {
        "training_sizes": training_sizes,
        "holdout_macros": args.holdout_macros,
        "holdout_fraction": args.holdout_fraction,
        "train_order": args.train_order,
        "model_kinds": list(args.model_kinds),
        "n_estimators": args.n_estimators,
        "knn_neighbors": args.knn_neighbors,
        "rng_seed": args.rng_seed,
        "skip_loco": args.skip_loco,
        "thresholds": asdict(thresholds),
        "target_reject_precision": args.target_reject_precision,
        "target_unknown_retention": args.target_unknown_retention,
        "min_duplicate_rejection": args.min_duplicate_rejection,
        "min_useful_seed_fraction": args.min_useful_seed_fraction,
    }
    report = {
        "schema_version": "main9_forest_evaluation_v1",
        "configuration": config,
        "corpora": [str(path) for path in corpus_paths],
        "skipped": skipped,
        "recommendations_min_train_macros": recommendations,
        "seed_metrics": seed_rows,
        "aggregate_metrics": aggregate_rows,
        "online_seed_metrics": online_rows,
        "online_aggregate_metrics": online_aggregate,
        "online_paired_metrics": online_pairs,
    }
    _write_csv(output_directory / "seed_metrics.csv", seed_rows)
    _write_csv(output_directory / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(output_directory / "online_seed_metrics.csv", online_rows)
    _write_csv(output_directory / "online_aggregate_metrics.csv", online_aggregate)
    _write_csv(output_directory / "online_paired_metrics.csv", online_pairs)
    (output_directory / "report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "REPORT.md").write_text(
        _markdown_report(
            aggregate_rows, recommendations, online_aggregate, skipped, config
        ),
        encoding="utf-8",
    )
    print(
        f"evaluated {len(corpus_paths) - len(skipped)} corpus/corpora; "
        f"wrote {len(seed_rows)} seed rows to {output_directory}"
    )
    return report


if __name__ == "__main__":
    main()
