"""Offline learning curves for terminal-cell duplicate filters.

The script expects an ``.npz`` corpus with these main9-facing fields:

``inputs``
    Candidate features with shape ``(rows, features)``.
``cell_labels``
    Actual terminal-cell label for every row.
``macro_ids``
    Independent macro group.  Residual/hindsight rows share the original id.

Optional fields are ``sample_weights``, ``macro_start_step`` (or legacy
``start_step``), and ``is_hindsight``.
When optional fields are absent, rows receive unit weight, macro ids determine
chronology, and the first row of every macro is the evaluation row.  A file
without ``macro_ids`` is still accepted, but every row is conservatively
treated as an independent macro.

Each curve point trains on a chronological prefix and evaluates the next
macros.  Outcomes whose cell is absent from the training prefix are treated as
pseudo-unknown.  Thus the primary filter metrics are duplicate rejection and
pseudo-unknown retention, not ordinary closed-set accuracy alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from forest_cell_model import (
    DuplicateFilterThresholds,
    conservative_duplicate_mask,
    fit_cell_filter,
    predict_cell_filter,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _label_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _parse_sizes(values: Sequence[str]) -> list[int]:
    result = []
    for value in values:
        result.extend(int(part) for part in str(value).split(",") if part.strip())
    if any(size <= 0 for size in result):
        raise ValueError("training sizes must be positive")
    return sorted(set(result))


def _discover_corpora(arguments: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument).expanduser()
        if not path.exists():
            local = SCRIPT_DIRECTORY / "result" / argument
            if local.exists():
                path = local
        if path.is_file():
            paths.append(path.resolve())
        elif path.is_dir():
            candidates = sorted(path.glob("seed_*/cell_training_data.npz"))
            candidates += sorted(path.glob("seed_*/training_data.npz"))
            if not candidates:
                candidates = sorted(path.rglob("*.npz"))
            paths.extend(candidate.resolve() for candidate in candidates)
        else:
            raise FileNotFoundError(f"corpus path was not found: {argument}")
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError("no corpus .npz files were found")
    return unique


def _load_corpus(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        keys = set(saved.files)
        required = {"inputs", "cell_labels"}
        if not required.issubset(keys):
            missing = ", ".join(sorted(required - keys))
            raise ValueError(f"{path} lacks required fields: {missing}")
        x = np.asarray(saved["inputs"], dtype=float)
        labels = np.asarray(saved["cell_labels"], dtype=object).reshape(-1)
        row_count = len(x)
        if x.ndim != 2 or len(labels) != row_count:
            raise ValueError(f"unaligned inputs/cell_labels in {path}")

        if "macro_ids" in keys:
            macro_ids = np.asarray(saved["macro_ids"], dtype=object).reshape(-1)
        else:
            macro_ids = np.arange(row_count, dtype=int)
        if "sample_weights" in keys:
            weights = np.asarray(saved["sample_weights"], dtype=float).reshape(-1)
        else:
            weights = np.ones(row_count, dtype=float)
        if "macro_start_step" in keys:
            start_step = np.asarray(
                saved["macro_start_step"], dtype=float
            ).reshape(-1)
        elif "start_step" in keys:
            start_step = np.asarray(saved["start_step"], dtype=float).reshape(-1)
        else:
            start_step = np.arange(row_count, dtype=float)
        if "is_hindsight" in keys:
            is_hindsight = np.asarray(saved["is_hindsight"], dtype=bool).reshape(-1)
        else:
            is_hindsight = np.zeros(row_count, dtype=bool)

    for name, values in (
        ("macro_ids", macro_ids),
        ("sample_weights", weights),
        ("start_step", start_step),
        ("is_hindsight", is_hindsight),
    ):
        if len(values) != row_count:
            raise ValueError(f"unaligned {name} in {path}")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(weights)):
        raise ValueError(f"non-finite input or weight in {path}")
    if np.any(weights < 0.0):
        raise ValueError(f"negative sample weight in {path}")
    # main9 uses -1 for numerical failure, timeout, or a globally censored
    # macro.  It is not a terminal cell class.  Remove the complete group so
    # hindsight rows from a censored macro cannot leak into either split.
    invalid_macros = {
        _label_key(macro_ids[row])
        for row, label in enumerate(labels)
        if _label_key(label) == -1
    }
    if invalid_macros:
        mixed = {
            _label_key(macro_ids[row])
            for row, label in enumerate(labels)
            if _label_key(macro_ids[row]) in invalid_macros
            and _label_key(label) != -1
        }
        if mixed:
            raise ValueError(
                f"{path} contains macro groups with both terminal and "
                "censored labels"
            )
        keep = np.asarray(
            [_label_key(value) not in invalid_macros for value in macro_ids],
            dtype=bool,
        )
        x = x[keep]
        labels = labels[keep]
        macro_ids = macro_ids[keep]
        weights = weights[keep]
        start_step = start_step[keep]
        is_hindsight = is_hindsight[keep]

    return {
        "inputs": x,
        "cell_labels": labels,
        "macro_ids": macro_ids,
        "sample_weights": weights,
        "start_step": start_step,
        "is_hindsight": is_hindsight,
    }


def _ordered_macros(data: dict[str, np.ndarray]) -> tuple[list[Any], dict[Any, np.ndarray]]:
    rows_by_macro: dict[Any, list[int]] = {}
    for row_index, raw_macro in enumerate(data["macro_ids"]):
        macro = _label_key(raw_macro)
        rows_by_macro.setdefault(macro, []).append(row_index)
    arrays = {
        macro: np.asarray(rows, dtype=int) for macro, rows in rows_by_macro.items()
    }
    ordered = sorted(
        arrays,
        key=lambda macro: (
            float(np.min(data["start_step"][arrays[macro]])),
            str(macro),
        ),
    )
    return ordered, arrays


def _evaluation_row(data: dict[str, np.ndarray], rows: np.ndarray) -> int:
    primary = rows[~data["is_hindsight"][rows]]
    candidates = primary if len(primary) else rows
    steps = data["start_step"][candidates]
    return int(candidates[int(np.argmin(steps))])


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _known_class_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    lookup = {_label_key(value): index for index, value in enumerate(classes)}
    known_rows = [
        row for row, value in enumerate(labels) if _label_key(value) in lookup
    ]
    if not known_rows:
        return {
            "known_accuracy": float("nan"),
            "known_log_loss": float("nan"),
            "known_brier": float("nan"),
        }
    indices = np.asarray(known_rows, dtype=int)
    target = np.asarray([lookup[_label_key(labels[row])] for row in indices])
    selected = np.maximum(probabilities[indices, target], 1.0e-15)
    predicted = np.argmax(probabilities[indices], axis=1)
    one_hot = np.zeros_like(probabilities[indices])
    one_hot[np.arange(len(indices)), target] = 1.0
    return {
        "known_accuracy": float(np.mean(predicted == target)),
        "known_log_loss": float(-np.mean(np.log(selected))),
        "known_brier": float(np.mean(np.sum((probabilities[indices] - one_hot) ** 2, axis=1))),
    }


def _filter_metrics(
    labels: np.ndarray,
    reject: np.ndarray,
    known_classes: Iterable[Any],
) -> dict[str, float]:
    known = {_label_key(value) for value in known_classes}
    duplicate = np.asarray([_label_key(value) in known for value in labels], dtype=bool)
    unknown = ~duplicate
    rejected = np.asarray(reject, dtype=bool)
    survivors = ~rejected
    baseline_new_rate = float(np.mean(unknown)) if len(unknown) else float("nan")
    survivor_new_rate = (
        float(np.mean(unknown[survivors])) if np.any(survivors) else float("nan")
    )
    return {
        "duplicate_count": int(np.count_nonzero(duplicate)),
        "pseudo_unknown_count": int(np.count_nonzero(unknown)),
        "rejected_count": int(np.count_nonzero(rejected)),
        "duplicate_rejection_rate": _safe_ratio(
            np.count_nonzero(rejected & duplicate), np.count_nonzero(duplicate)
        ),
        "reject_precision": _safe_ratio(
            np.count_nonzero(rejected & duplicate), np.count_nonzero(rejected)
        ),
        "pseudo_unknown_retention": _safe_ratio(
            np.count_nonzero(survivors & unknown), np.count_nonzero(unknown)
        ),
        "baseline_new_rate": baseline_new_rate,
        "survivor_new_rate": survivor_new_rate,
        "new_rate_lift": _safe_ratio(survivor_new_rate, baseline_new_rate),
    }


def evaluate_corpus(
    path: Path,
    *,
    training_sizes: Sequence[int],
    test_macros: int,
    model_kinds: Sequence[str],
    n_estimators: int,
    rng_seed: int,
    thresholds: DuplicateFilterThresholds,
) -> list[dict[str, Any]]:
    data = _load_corpus(path)
    macros, rows_by_macro = _ordered_macros(data)
    results: list[dict[str, Any]] = []
    for train_count in training_sizes:
        if train_count >= len(macros):
            continue
        test_group_list = macros[train_count : train_count + int(test_macros)]
        if not test_group_list:
            continue
        train_group_list = macros[:train_count]
        train_rows = np.concatenate([rows_by_macro[macro] for macro in train_group_list])
        test_rows = np.asarray(
            [_evaluation_row(data, rows_by_macro[macro]) for macro in test_group_list],
            dtype=int,
        )
        for model_index, model_kind in enumerate(model_kinds):
            model = fit_cell_filter(
                data["inputs"][train_rows],
                data["cell_labels"][train_rows],
                data["macro_ids"][train_rows],
                data["sample_weights"][train_rows],
                model_kind=model_kind,
                rng_seed=int(rng_seed + 1009 * model_index + 7919 * train_count),
                n_estimators=n_estimators,
            )
            prediction = predict_cell_filter(model, data["inputs"][test_rows])
            reject = conservative_duplicate_mask(
                prediction, model.classes_, thresholds=thresholds
            )
            row: dict[str, Any] = {
                "corpus": str(path),
                "model_kind": model_kind,
                "train_macros": int(train_count),
                "train_rows": int(len(train_rows)),
                "train_classes": int(len(model.classes_)),
                "test_macros": int(len(test_rows)),
                "filter_ready": bool(model.filter_ready),
            }
            row.update(
                _known_class_metrics(
                    data["cell_labels"][test_rows],
                    prediction["probabilities"],
                    model.classes_,
                )
            )
            row.update(
                _filter_metrics(
                    data["cell_labels"][test_rows], reject, model.classes_
                )
            )
            results.append(row)
    return results


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_results(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        keys = list(rows[0]) if rows else []
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        serializable = [
            {key: _json_value(value) for key, value in row.items()} for row in rows
        ]
        with output.open("w", encoding="utf-8") as stream:
            json.dump(serializable, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", help="NPZ file or result directory")
    parser.add_argument(
        "--training-sizes",
        nargs="+",
        default=["4,8,16,32,64,128"],
        help="chronological macro-prefix sizes",
    )
    parser.add_argument("--test-macros", type=int, default=32)
    parser.add_argument(
        "--model-kinds",
        nargs="+",
        choices=("random_forest", "extra_trees"),
        default=("random_forest", "extra_trees"),
    )
    parser.add_argument("--n-estimators", type=int, default=64)
    parser.add_argument("--rng-seed", type=int, default=20260817)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "forest_cell_learning_curve.json",
    )
    args = parser.parse_args()
    if args.test_macros <= 0 or args.n_estimators <= 0:
        raise ValueError("test-macros and n-estimators must be positive")

    sizes = _parse_sizes(args.training_sizes)
    paths = _discover_corpora(args.corpora)
    rows: list[dict[str, Any]] = []
    skipped = []
    for path in paths:
        try:
            rows.extend(
                evaluate_corpus(
                    path,
                    training_sizes=sizes,
                    test_macros=args.test_macros,
                    model_kinds=args.model_kinds,
                    n_estimators=args.n_estimators,
                    rng_seed=args.rng_seed,
                    thresholds=DuplicateFilterThresholds(),
                )
            )
        except ValueError as error:
            skipped.append((str(path), str(error)))
    if not rows:
        messages = "; ".join(f"{path}: {error}" for path, error in skipped)
        raise RuntimeError(f"no compatible corpus could be evaluated. {messages}")
    _write_results(rows, args.output.resolve())
    print(f"wrote {len(rows)} curve rows to {args.output.resolve()}")
    if skipped:
        print(f"skipped {len(skipped)} incompatible NPZ files")


if __name__ == "__main__":
    main()
