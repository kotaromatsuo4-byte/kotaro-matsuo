"""Offline feature ablation for the main9 terminal-cell duplicate filter.

The saved five-bar feature layout is::

    [q_free(6), last_nonflat_context_free(6), v_free(6), logK(1)]

This script compares the literal ``(q, v, K)`` input against main9's current
``(q, context, v, K)`` input without rerunning or changing any mechanics.  A
fixed final chronological holdout is reused at every nested training size,
and all hindsight rows from one rollout remain in the same macro group.

Closed-set predictions and leave-one-cell-out (LOCO) predictions are fitted
once and then replayed through several conservative-to-relaxed duplicate
threshold profiles.  Thus threshold trade-offs do not receive different
model fits.  Outputs are CSV, JSON, and a compact Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from forest_cell_learning_curve import _load_corpus, _ordered_macros, _parse_sizes
from forest_cell_model import DuplicateFilterThresholds, conservative_duplicate_mask
from main9_forest_evaluation import (
    _classification_metrics,
    _evaluation_row,
    _filter_metrics,
    _fit_and_predict,
    _label_key,
    _macro_labels,
    _primary_rows,
    _rows_for_macros,
    _safe_ratio,
    discover_corpora,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
FEATURE_LAYOUT_DIMENSION = 19
FEATURE_SETS: dict[str, tuple[int, ...]] = {
    "q_v_k": tuple(list(range(0, 6)) + list(range(12, 19))),
    "q_context_v_k": tuple(range(19)),
}
DEFAULT_SIZES = "8,12,16,32,64,96,128,192,320"
MODEL_KINDS = ("random_forest", "extra_trees", "knn", "prevalence")


# These are not fitted hyperparameters.  They are an offline operating-point
# sweep over the same cached predictions.  The named `relative` point is the
# aggressive profile used in the existing main9 LOCO investigation.
THRESHOLD_PROFILES: dict[str, DuplicateFilterThresholds] = {
    "strict": DuplicateFilterThresholds(),
    "p70": DuplicateFilterThresholds(0.70, 0.50, 0.70, 0.10, 1.00, 4),
    "p55": DuplicateFilterThresholds(0.55, 0.25, 0.60, 0.12, 1.00, 4),
    "p40": DuplicateFilterThresholds(0.40, 0.15, 0.55, 0.12, 1.00, 4),
    "relative": DuplicateFilterThresholds(0.25, 0.08, 0.50, 0.12, 1.00, 4),
    "relative_support075": DuplicateFilterThresholds(
        0.25, 0.08, 0.50, 0.12, 0.75, 4
    ),
    "relative_support15": DuplicateFilterThresholds(
        0.25, 0.08, 0.50, 0.20, 1.50, 4
    ),
    "loose_groups2": DuplicateFilterThresholds(0.25, 0.05, 0.40, 0.25, 2.00, 2),
}

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
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_mean(values: Iterable[Any]) -> float:
    selected = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            selected.append(number)
    return float(np.mean(selected)) if selected else float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _project_inputs(data: Mapping[str, np.ndarray], feature_set: str) -> dict[str, np.ndarray]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    inputs = np.asarray(data["inputs"], dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != FEATURE_LAYOUT_DIMENSION:
        raise ValueError(
            f"expected saved main9 input dimension 19, found {inputs.shape}"
        )
    projected = {name: np.asarray(values) for name, values in data.items()}
    projected["inputs"] = inputs[:, FEATURE_SETS[feature_set]]
    return projected


def _loco_prediction_cases(
    *,
    model_kind: str,
    data: Mapping[str, np.ndarray],
    train_macros: Sequence[Any],
    holdout_macros: Sequence[Any],
    macro_labels: Mapping[Any, Any],
    rows_by_macro: Mapping[Any, np.ndarray],
    n_estimators: int,
    knn_neighbors: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    holdout_by_label: dict[Any, list[Any]] = defaultdict(list)
    for macro in holdout_macros:
        holdout_by_label[macro_labels[macro]].append(macro)
    train_labels: list[Any] = []
    for macro in train_macros:
        label = macro_labels[macro]
        if label not in train_labels:
            train_labels.append(label)

    cases: list[dict[str, Any]] = []
    for cell_index, held_out_cell in enumerate(train_labels):
        query_macros = holdout_by_label.get(held_out_cell, [])
        if not query_macros:
            continue
        query_rows = _primary_rows(data, query_macros, rows_by_macro)
        remaining = [
            macro for macro in train_macros if macro_labels[macro] != held_out_cell
        ]
        if not remaining:
            cases.append({"query_count": len(query_rows), "filter_ready": False})
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
        cases.append(
            {
                "query_count": len(query_rows),
                "classes": classes,
                "prediction": prediction,
                "filter_ready": bool(filter_ready),
            }
        )
    return cases


def _loco_metrics(
    cases: Sequence[Mapping[str, Any]], thresholds: DuplicateFilterThresholds
) -> dict[str, Any]:
    total = 0
    retained = 0
    for case in cases:
        count = int(case["query_count"])
        total += count
        if not case["filter_ready"]:
            retained += count
            continue
        rejected = conservative_duplicate_mask(
            case["prediction"], case["classes"], thresholds=thresholds
        )
        retained += int(np.count_nonzero(~rejected))
    return {
        "loco_cells_evaluated": int(len(cases)),
        "loco_unknown_count": int(total),
        "loco_unknown_retained_count": int(retained),
        "loco_unknown_retention": _safe_ratio(retained, total),
    }


def _evaluate_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(job["path"]))
    feature_set = str(job["feature_set"])
    # `_load_corpus` already removes complete macro groups labelled with the
    # main9 censoring sentinel -1.  Count them from the immutable source NPZ
    # first; do not overwrite its already-filtered aliases afterward.
    with np.load(path, allow_pickle=True) as saved:
        source_labels = np.asarray(saved["cell_labels"], dtype=object).reshape(-1)
        if "macro_ids" in saved.files:
            source_macros = np.asarray(saved["macro_ids"], dtype=object).reshape(-1)
        else:
            source_macros = np.arange(len(source_labels), dtype=int)
    invalid_rows = np.asarray(
        [_label_key(value) == -1 for value in source_labels], dtype=bool
    )
    censoring = {
        "source_rows": int(len(source_labels)),
        "ignored_censored_rows": int(np.count_nonzero(invalid_rows)),
        "ignored_censored_macros": int(
            len({_label_key(value) for value in source_macros[invalid_rows]})
        ),
    }
    data = _load_corpus(path)
    data = _project_inputs(data, feature_set)
    macros, rows_by_macro = _ordered_macros(data)
    macro_labels = _macro_labels(data, macros, rows_by_macro)
    holdout_count = int(job["holdout_macros"])
    if holdout_count >= len(macros):
        raise ValueError(f"holdout {holdout_count} leaves no training macros in {path}")
    holdout = list(macros[-holdout_count:])
    train_pool = list(macros[:-holdout_count])
    holdout_rows = _primary_rows(data, holdout, rows_by_macro)
    seed = path.parent.name.replace("seed_", "")
    output: list[dict[str, Any]] = []

    for requested_size in job["training_sizes"]:
        if int(requested_size) > len(train_pool):
            continue
        selected = train_pool[: int(requested_size)]
        train_rows = _rows_for_macros(selected, rows_by_macro)
        for model_index, model_kind in enumerate(job["model_kinds"]):
            fit_seed = int(job["rng_seed"]) + int(requested_size) * 7919 + model_index * 1009
            classes, prediction, filter_ready = _fit_and_predict(
                model_kind,
                data["inputs"][train_rows],
                data["cell_labels"][train_rows],
                data["macro_ids"][train_rows],
                data["sample_weights"][train_rows],
                data["inputs"][holdout_rows],
                n_estimators=int(job["n_estimators"]),
                knn_neighbors=int(job["knn_neighbors"]),
                rng_seed=fit_seed,
            )
            classification = _classification_metrics(
                data["cell_labels"][holdout_rows], prediction["probabilities"], classes
            )
            if bool(job.get("skip_loco", False)):
                loco_cases = []
            else:
                loco_cases = _loco_prediction_cases(
                    model_kind=model_kind,
                    data=data,
                    train_macros=selected,
                    holdout_macros=holdout,
                    macro_labels=macro_labels,
                    rows_by_macro=rows_by_macro,
                    n_estimators=int(job["n_estimators"]),
                    knn_neighbors=int(job["knn_neighbors"]),
                    rng_seed=fit_seed,
                )
            for profile_name, thresholds in THRESHOLD_PROFILES.items():
                if filter_ready:
                    rejected = conservative_duplicate_mask(
                        prediction, classes, thresholds=thresholds
                    )
                else:
                    rejected = np.zeros(len(holdout_rows), dtype=bool)
                metrics = _filter_metrics(
                    data["cell_labels"][holdout_rows], rejected, classes
                )
                metrics.update(_loco_metrics(loco_cases, thresholds))
                retention = metrics["loco_unknown_retention"]
                useful = bool(
                    np.isfinite(metrics["reject_precision"])
                    and metrics["reject_precision"] >= float(job["target_precision"])
                    and np.isfinite(retention)
                    and retention >= float(job["target_retention"])
                    and np.isfinite(metrics["duplicate_rejection_rate"])
                    and metrics["duplicate_rejection_rate"]
                    >= float(job["minimum_rejection"])
                )
                output.append(
                    {
                        "corpus": str(path),
                        "seed": seed,
                        "feature_set": feature_set,
                        "feature_dimension": int(data["inputs"].shape[1]),
                        "threshold_profile": profile_name,
                        "model_kind": model_kind,
                        "train_macros": int(requested_size),
                        "train_rows": int(len(train_rows)),
                        "train_classes": int(len(classes)),
                        "fixed_holdout_macros": len(holdout),
                        "filter_ready": bool(filter_ready),
                        "useful_filter": useful,
                        **censoring,
                        **classification,
                        **metrics,
                    }
                )
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["feature_set"]),
                str(row["threshold_profile"]),
                str(row["model_kind"]),
                int(row["train_macros"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        feature_set, profile, model_kind, train_macros = key
        item: dict[str, Any] = {
            "feature_set": feature_set,
            "feature_dimension": int(group[0]["feature_dimension"]),
            "threshold_profile": profile,
            "model_kind": model_kind,
            "train_macros": train_macros,
            "seeds": len(group),
            "useful_seed_fraction": float(
                np.mean([bool(row["useful_filter"]) for row in group])
            ),
            "known_accuracy_mean": _finite_mean(row["known_accuracy"] for row in group),
            "known_macro_f1_mean": _finite_mean(row["known_macro_f1"] for row in group),
            "known_nll_mean": _finite_mean(row["known_nll"] for row in group),
            "known_brier_mean": _finite_mean(row["known_brier"] for row in group),
        }
        for field in COUNT_FIELDS:
            item[f"{field}_sum"] = int(sum(int(row[field]) for row in group))
        item["pooled_known_accuracy"] = _safe_ratio(
            item["known_correct_count_sum"], item["known_test_count_sum"]
        )
        item["pooled_reject_precision"] = _safe_ratio(
            item["duplicate_rejected_count_sum"], item["rejected_count_sum"]
        )
        item["pooled_duplicate_rejection_rate"] = _safe_ratio(
            item["duplicate_rejected_count_sum"], item["duplicate_count_sum"]
        )
        item["pooled_pseudo_unknown_retention"] = _safe_ratio(
            item["pseudo_unknown_retained_count_sum"], item["pseudo_unknown_count_sum"]
        )
        item["pooled_loco_unknown_retention"] = _safe_ratio(
            item["loco_unknown_retained_count_sum"], item["loco_unknown_count_sum"]
        )
        output.append(item)
    return output


def _passes_gate(row: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        np.isfinite(row["pooled_reject_precision"])
        and row["pooled_reject_precision"] >= args.target_precision
        and np.isfinite(row["pooled_loco_unknown_retention"])
        and row["pooled_loco_unknown_retention"] >= args.target_retention
        and np.isfinite(row["pooled_duplicate_rejection_rate"])
        and row["pooled_duplicate_rejection_rate"] >= args.minimum_rejection
        and row["useful_seed_fraction"] >= args.minimum_useful_seed_fraction
    )


def _paired_classification(aggregate: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Classification does not depend on the threshold profile; use strict once.
    rows = [row for row in aggregate if row["threshold_profile"] == "strict"]
    lookup = {
        (row["feature_set"], row["model_kind"], row["train_macros"]): row
        for row in rows
    }
    output = []
    for model in MODEL_KINDS:
        for size in sorted({int(row["train_macros"]) for row in rows}):
            reduced = lookup.get(("q_v_k", model, size))
            full = lookup.get(("q_context_v_k", model, size))
            if reduced is None or full is None:
                continue
            output.append(
                {
                    "model_kind": model,
                    "train_macros": size,
                    "q_v_k_accuracy": reduced["pooled_known_accuracy"],
                    "q_context_v_k_accuracy": full["pooled_known_accuracy"],
                    "accuracy_delta_q_v_k_minus_full": reduced["pooled_known_accuracy"]
                    - full["pooled_known_accuracy"],
                    "q_v_k_macro_f1": reduced["known_macro_f1_mean"],
                    "q_context_v_k_macro_f1": full["known_macro_f1_mean"],
                    "macro_f1_delta_q_v_k_minus_full": reduced["known_macro_f1_mean"]
                    - full["known_macro_f1_mean"],
                }
            )
    return output


def _recommendations(
    aggregate: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for feature_set in FEATURE_SETS:
        for model in MODEL_KINDS:
            candidates = [
                row
                for row in aggregate
                if row["feature_set"] == feature_set
                and row["model_kind"] == model
                and _passes_gate(row, args)
            ]
            candidates.sort(key=lambda row: (row["train_macros"], row["threshold_profile"]))
            recommendations[f"{feature_set}/{model}"] = (
                {
                    "train_macros": int(candidates[0]["train_macros"]),
                    "threshold_profile": candidates[0]["threshold_profile"],
                }
                if candidates
                else None
            )
    return recommendations


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.3f}" if math.isfinite(number) else "--"


def _markdown(
    paired: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    recommendations: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# main9 feature ablation: `(q,v,K)` vs `(q,context,v,K)`",
        "",
        "The physics and saved labels are unchanged. The final 64 chronological macros "
        "per seed are the fixed holdout, and hindsight rows remain grouped by source macro.",
        "",
        "The earlier reachability audit found canonical T6 at 11 / 14 / 17 completed "
        "macros (seeds 800--802). Consequently 12 and 16 macros are the relevant "
        "online-data regime; 32+ macros are already generally after T6.",
        "",
        "## Closed-set classification",
        "",
        "Values are pooled known-class accuracy. `delta` is `(q,v,K) - full`.",
        "",
        "| model | macros | q,v,K | q,context,v,K | delta | F1 delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {model} | {size} | {reduced} | {full} | {delta} | {f1delta} |".format(
                model=row["model_kind"],
                size=row["train_macros"],
                reduced=_fmt(row["q_v_k_accuracy"]),
                full=_fmt(row["q_context_v_k_accuracy"]),
                delta=_fmt(row["accuracy_delta_q_v_k_minus_full"]),
                f1delta=_fmt(row["macro_f1_delta_q_v_k_minus_full"]),
            )
        )

    lines.extend(
        [
            "",
            "## Safe duplicate-filter gate",
            "",
            f"Gate: pooled rejection precision >= {args.target_precision:.2f}, "
            f"LOCO unknown retention >= {args.target_retention:.2f}, duplicate rejection "
            f">= {args.minimum_rejection:.2f}, and useful in >= "
            f"{args.minimum_useful_seed_fraction:.2f} of seeds.",
            "",
            "| feature | model | first demonstrated point |",
            "|---|---|---|",
        ]
    )
    for key, value in recommendations.items():
        feature, model = key.split("/", 1)
        demonstrated = (
            f"{value['train_macros']} macros / {value['threshold_profile']}"
            if value is not None
            else "not demonstrated"
        )
        lines.append(f"| {feature} | {model} | {demonstrated} |")

    lines.extend(
        [
            "",
            "## Strict and relaxed trade-off at 16 and 320 macros",
            "",
            "`dup` is pooled known-duplicate rejection; `LOCO` is retained withheld-cell "
            "fraction. Precision is omitted when nothing is rejected.",
            "",
            "| feature | model | macros | profile | accuracy | precision | dup | LOCO | useful seeds |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected = [
        row
        for row in aggregate
        if row["train_macros"] in (16, 320)
        and row["threshold_profile"] in ("strict", "relative", "loose_groups2")
        and row["model_kind"] != "prevalence"
    ]
    for row in selected:
        lines.append(
            "| {feature} | {model} | {size} | {profile} | {accuracy} | {precision} | "
            "{duplicate} | {loco} | {useful} |".format(
                feature=row["feature_set"],
                model=row["model_kind"],
                size=row["train_macros"],
                profile=row["threshold_profile"],
                accuracy=_fmt(row["pooled_known_accuracy"]),
                precision=_fmt(row["pooled_reject_precision"]),
                duplicate=_fmt(row["pooled_duplicate_rejection_rate"]),
                loco=_fmt(row["pooled_loco_unknown_retention"]),
                useful=_fmt(row["useful_seed_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "All threshold profiles and per-seed counts are retained in "
            "`aggregate_metrics.csv` and `seed_metrics.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus",
        nargs="?",
        default="result/main9_random_corpus_b20000",
        help="main9 random corpus result root",
    )
    parser.add_argument("--training-sizes", nargs="+", default=[DEFAULT_SIZES])
    parser.add_argument("--holdout-macros", type=int, default=64)
    parser.add_argument("--model-kinds", nargs="+", choices=MODEL_KINDS, default=MODEL_KINDS)
    parser.add_argument("--n-estimators", type=int, default=16)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=20260817)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--skip-loco",
        action="store_true",
        help="fit only the fixed-holdout learning curve; useful for a fast ablation",
    )
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--target-retention", type=float, default=0.95)
    parser.add_argument("--minimum-rejection", type=float, default=0.05)
    parser.add_argument("--minimum-useful-seed-fraction", type=float, default=0.50)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SCRIPT_DIRECTORY / "result" / "main9_feature_ablation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.holdout_macros <= 0 or args.n_estimators <= 0 or args.workers <= 0:
        raise ValueError("holdout, estimators, and workers must be positive")
    sizes = _parse_sizes(args.training_sizes)
    corpora = discover_corpora([args.corpus])
    jobs = []
    for corpus_index, path in enumerate(corpora):
        for feature_set in FEATURE_SETS:
            jobs.append(
                {
                    "path": str(path),
                    "feature_set": feature_set,
                    "training_sizes": sizes,
                    "holdout_macros": args.holdout_macros,
                    "model_kinds": tuple(args.model_kinds),
                    "n_estimators": args.n_estimators,
                    "knn_neighbors": args.knn_neighbors,
                    "rng_seed": args.rng_seed + corpus_index * 1_000_003,
                    "skip_loco": args.skip_loco,
                    "target_precision": args.target_precision,
                    "target_retention": args.target_retention,
                    "minimum_rejection": args.minimum_rejection,
                }
            )
    if args.workers == 1:
        result_groups = [_evaluate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            result_groups = list(executor.map(_evaluate_job, jobs))
    seed_rows = [row for group in result_groups for row in group]
    aggregate = _aggregate(seed_rows)
    paired = _paired_classification(aggregate)
    recommendations = _recommendations(aggregate, args)

    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": [str(path) for path in corpora],
        "feature_sets": {name: list(indices) for name, indices in FEATURE_SETS.items()},
        "saved_layout": "q6,context6,v6,logK1",
        "training_sizes": sizes,
        "holdout_macros": args.holdout_macros,
        "model_kinds": list(args.model_kinds),
        "n_estimators": args.n_estimators,
        "knn_neighbors": args.knn_neighbors,
        "rng_seed": args.rng_seed,
        "skip_loco": args.skip_loco,
        "threshold_profiles": {
            name: asdict(value) for name, value in THRESHOLD_PROFILES.items()
        },
        "gate": {
            "target_precision": args.target_precision,
            "target_loco_retention": args.target_retention,
            "minimum_duplicate_rejection": args.minimum_rejection,
            "minimum_useful_seed_fraction": args.minimum_useful_seed_fraction,
        },
        "canonical_t6_independent_macros_from_main9_model_analysis": [11, 14, 17],
    }
    report = {
        "schema_version": "main9_feature_ablation_v1",
        "configuration": config,
        "recommendations": recommendations,
        "paired_classification": paired,
        "aggregate_metrics": aggregate,
        "seed_metrics": seed_rows,
    }
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    _write_csv(output / "paired_classification.csv", paired)
    (output / "report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        _markdown(paired, aggregate, recommendations, args), encoding="utf-8"
    )
    print(
        f"evaluated {len(corpora)} corpora x {len(FEATURE_SETS)} feature sets; "
        f"wrote {len(seed_rows)} seed rows to {output}"
    )
    return report


if __name__ == "__main__":
    main()
