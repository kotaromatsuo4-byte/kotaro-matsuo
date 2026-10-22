"""Cost-aware reanalysis of a saved persistent-K panel.

This script performs no rollout or mechanics calculation.  It reads the
paired panel produced by ``persistent_k_panel.py`` and uses exactly the saved
group-wise cross-validation folds.  In every fold it fits a basis-free GP to

    log(1 + mechanics_cost)

on complete training groups.  The held-out expected mechanics cost is the
posterior mean in the original cost domain,

    E[C | x, D] = exp(mu_log + variance_log / 2) - 1.

The cost-aware score divides the already saved basis-free discovery
acquisition by this expected cost.  It deliberately does not use a cost LCB:
an uncertain candidate must not look artificially cheap.  A within-group
score shuffle is included as a negative control.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from gp import fit_shared_kernel_gp, predict_shared_kernel_gp


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_seed(seed_directory):
    outcomes = _read_csv(seed_directory / "panel_outcomes.csv")
    saved_predictions = _read_csv(seed_directory / "cv_predictions.csv")
    arrays = np.load(seed_directory / "panel_arrays.npz")
    summary = json.loads(
        (seed_directory / "summary.json").read_text(encoding="utf-8")
    )

    count = len(outcomes)
    if len(saved_predictions) != count:
        raise ValueError(f"row count mismatch in {seed_directory}")
    features = np.asarray(arrays["basis_free_features"], dtype=float)
    if len(features) != count:
        raise ValueError(f"feature count mismatch in {seed_directory}")

    records = []
    for index, (outcome, prediction) in enumerate(
        zip(outcomes, saved_predictions)
    ):
        if int(prediction["row_index"]) != index:
            raise ValueError(f"row order mismatch in {seed_directory}")
        if (
            int(outcome["group_id"]) != int(prediction["group_id"])
            or int(outcome["k"]) != int(prediction["k"])
        ):
            raise ValueError(f"panel/prediction alignment mismatch in {seed_directory}")
        records.append(
            {
                "seed": int(outcome["seed"]),
                "row_index": index,
                "group_id": int(outcome["group_id"]),
                "k": int(outcome["k"]),
                "fold": int(prediction["fold"]),
                "true_new": int(outcome["true_new"]),
                "mechanics_cost": float(outcome["mechanics_cost"]),
                "basis_free_score": float(
                    prediction["basis_free_gp_score"]
                ),
            }
        )

    # Strict group CV audit: no (q,v1) group may occur in two folds.
    for group_id in sorted({record["group_id"] for record in records}):
        folds = {
            record["fold"]
            for record in records
            if record["group_id"] == group_id
        }
        if len(folds) != 1:
            raise ValueError(
                f"group {group_id} crosses folds in {seed_directory}"
            )

    dimension = features.shape[1]
    if (dimension - 1) % 3:
        raise ValueError("basis-free feature does not have 3*d+1 entries")
    free_dimension = (dimension - 1) // 3
    state_dimension = 2 * free_dimension
    return records, features, summary, state_dimension


def _fit_cost_predictions(records, features, summary, state_dimension, args):
    folds = np.asarray([record["fold"] for record in records], dtype=int)
    costs = np.asarray(
        [record["mechanics_cost"] for record in records], dtype=float
    )
    expected = np.full(len(records), np.nan, dtype=float)
    log_mean = np.full(len(records), np.nan, dtype=float)
    log_variance = np.full(len(records), np.nan, dtype=float)

    state_length_scale = float(
        summary["configuration"]["state_length_scale"]
    )
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if not len(train) or not len(test):
            raise ValueError("every CV fold needs training and test rows")
        model = fit_shared_kernel_gp(
            features[train],
            np.log1p(costs[train])[:, None],
            state_dimension=state_dimension,
            state_length_scale=state_length_scale,
            action_length_scale=float(args.action_length_scale),
            signal_variance=float(args.signal_variance),
            noise_std=float(args.noise_std),
            output_scale_floor=float(args.output_scale_floor),
        )
        mean, variance = predict_shared_kernel_gp(model, features[test])
        local_mean = mean[:, 0]
        local_variance = np.maximum(variance[:, 0], 0.0)
        # This is the log-normal posterior mean, not a cost LCB.  Including
        # epistemic variance prevents an unknown candidate from being treated
        # as optimistically cheap.  Every completed branch costs >= 1 step.
        with np.errstate(over="ignore"):
            local_expected = np.exp(
                local_mean + 0.5 * local_variance
            ) - 1.0
        expected[test] = np.maximum(local_expected, 1.0)
        log_mean[test] = local_mean
        log_variance[test] = local_variance

    if not np.all(np.isfinite(expected)):
        raise ValueError("non-finite expected cost from held-out GP")
    return expected, log_mean, log_variance


def _within_group_shuffle(records, scores, seed):
    shuffled = np.asarray(scores, dtype=float).copy()
    for group_id in sorted({record["group_id"] for record in records}):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record["group_id"] == group_id
            ],
            dtype=int,
        )
        rng = np.random.default_rng(
            int(seed) + 11801 + 37 * int(group_id)
        )
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    ranks = rankdata(scores, method="average")
    num_positive = int(np.count_nonzero(positive))
    num_negative = int(np.count_nonzero(negative))
    return float(
        (
            np.sum(ranks[positive])
            - num_positive * (num_positive + 1) / 2.0
        )
        / (num_positive * num_negative)
    )


def _ratio(new_rate, mean_cost):
    return float(new_rate / mean_cost) if mean_cost > 0.0 else float("nan")


def _selection_metrics(records, score_name):
    group_keys = sorted(
        {(record["seed"], record["group_id"]) for record in records}
    )
    top1_new = []
    top1_cost = []
    top25_new = []
    top25_cost = []
    top25_any = []
    for seed, group_id in group_keys:
        group = [
            record
            for record in records
            if record["seed"] == seed and record["group_id"] == group_id
        ]
        scores = np.asarray([record[score_name] for record in group])
        utility = np.asarray([record["true_new"] for record in group])
        costs = np.asarray([record["mechanics_cost"] for record in group])

        maximum = np.max(scores)
        tied = np.flatnonzero(
            np.isclose(scores, maximum, rtol=1.0e-10, atol=1.0e-12)
        )
        top1_new.append(float(np.mean(utility[tied])))
        top1_cost.append(float(np.mean(costs[tied])))

        keep = max(1, int(np.ceil(0.25 * len(group))))
        selected = np.argsort(-scores, kind="stable")[:keep]
        top25_new.append(float(np.mean(utility[selected])))
        top25_cost.append(float(np.mean(costs[selected])))
        top25_any.append(float(np.any(utility[selected])))

    top1_new_rate = float(np.mean(top1_new))
    top1_mean_cost = float(np.mean(top1_cost))
    top25_new_rate = float(np.mean(top25_new))
    top25_mean_cost = float(np.mean(top25_cost))
    return {
        "row_new_cell_auc": _binary_auc(
            [record["true_new"] for record in records],
            [record[score_name] for record in records],
        ),
        "group_top1_new_rate": top1_new_rate,
        "group_top1_mean_cost": top1_mean_cost,
        "group_top1_new_per_step_proxy": _ratio(
            top1_new_rate, top1_mean_cost
        ),
        "group_top25_mean_new_rate": top25_new_rate,
        "group_top25_any_new_rate": float(np.mean(top25_any)),
        "group_top25_mean_cost": top25_mean_cost,
        "group_top25_new_per_step_proxy": _ratio(
            top25_new_rate, top25_mean_cost
        ),
    }


def _uniform_metrics(records):
    new_rate = float(np.mean([record["true_new"] for record in records]))
    mean_cost = float(
        np.mean([record["mechanics_cost"] for record in records])
    )
    return {
        "new_rate": new_rate,
        "mean_cost": mean_cost,
        "new_per_step_proxy": _ratio(new_rate, mean_cost),
    }


def _cost_prediction_metrics(records):
    actual = np.asarray(
        [record["mechanics_cost"] for record in records], dtype=float
    )
    predicted = np.asarray(
        [record["expected_cost"] for record in records], dtype=float
    )
    correlation = spearmanr(actual, predicted).statistic
    within_group_correlations = []
    for seed, group_id in sorted(
        {(record["seed"], record["group_id"]) for record in records}
    ):
        group = [
            record
            for record in records
            if record["seed"] == seed and record["group_id"] == group_id
        ]
        group_actual = np.asarray(
            [record["mechanics_cost"] for record in group], dtype=float
        )
        group_predicted = np.asarray(
            [record["expected_cost"] for record in group], dtype=float
        )
        # A constant realized cost contains no K-ranking information.  It is
        # excluded rather than turning SciPy's undefined correlation into 0.
        if np.ptp(group_actual) <= 0.0 or np.ptp(group_predicted) <= 0.0:
            continue
        within_group_correlations.append(
            float(spearmanr(group_actual, group_predicted).statistic)
        )
    return {
        "mae_steps": float(np.mean(np.abs(predicted - actual))),
        "rmse_steps": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mean_actual_steps": float(np.mean(actual)),
        "mean_predicted_steps": float(np.mean(predicted)),
        "spearman_rank_correlation": float(correlation),
        "mean_within_group_spearman": (
            float(np.mean(within_group_correlations))
            if within_group_correlations
            else float("nan")
        ),
        "median_within_group_spearman": (
            float(np.median(within_group_correlations))
            if within_group_correlations
            else float("nan")
        ),
        "within_group_spearman_groups": int(len(within_group_correlations)),
    }


def _run_seed(seed_directory, args):
    records, features, summary, state_dimension = _load_seed(seed_directory)
    expected, log_mean, log_variance = _fit_cost_predictions(
        records, features, summary, state_dimension, args
    )
    raw_score = np.asarray(
        [record["basis_free_score"] for record in records], dtype=float
    )
    cost_aware_score = raw_score / expected
    shuffled_score = _within_group_shuffle(
        records, cost_aware_score, records[0]["seed"]
    )

    for index, record in enumerate(records):
        record["expected_cost"] = float(expected[index])
        record["cost_log_mean"] = float(log_mean[index])
        record["cost_log_std"] = float(np.sqrt(log_variance[index]))
        record["cost_aware_score"] = float(cost_aware_score[index])
        record["cost_aware_score_shuffled"] = float(shuffled_score[index])

    _write_csv(seed_directory / "cost_aware_cv_predictions.csv", records)
    return records, {
        "seed": int(records[0]["seed"]),
        "num_groups": int(len({record["group_id"] for record in records})),
        "num_rows": int(len(records)),
        "cost_prediction": _cost_prediction_metrics(records),
        "uniform_random": _uniform_metrics(records),
        "basis_free_gp": _selection_metrics(records, "basis_free_score"),
        "basis_free_gp_per_expected_cost": _selection_metrics(
            records, "cost_aware_score"
        ),
        "basis_free_gp_per_expected_cost_score_shuffled": (
            _selection_metrics(records, "cost_aware_score_shuffled")
        ),
    }


def _aggregate(all_records, seed_summaries, args):
    metrics = {
        "uniform_random": _uniform_metrics(all_records),
        "basis_free_gp": _selection_metrics(all_records, "basis_free_score"),
        "basis_free_gp_per_expected_cost": _selection_metrics(
            all_records, "cost_aware_score"
        ),
        "basis_free_gp_per_expected_cost_score_shuffled": (
            _selection_metrics(all_records, "cost_aware_score_shuffled")
        ),
    }
    uniform_proxy = metrics["uniform_random"]["new_per_step_proxy"]
    for name in (
        "basis_free_gp",
        "basis_free_gp_per_expected_cost",
        "basis_free_gp_per_expected_cost_score_shuffled",
    ):
        metrics[name]["top1_proxy_relative_to_uniform"] = float(
            metrics[name]["group_top1_new_per_step_proxy"] / uniform_proxy
        )
        metrics[name]["top25_proxy_relative_to_uniform"] = float(
            metrics[name]["group_top25_new_per_step_proxy"] / uniform_proxy
        )

    return {
        "evaluation": "saved paired panel; no mechanics re-executed",
        "mechanics_reexecuted": False,
        "seeds": [int(summary["seed"]) for summary in seed_summaries],
        "total_groups": int(
            sum(summary["num_groups"] for summary in seed_summaries)
        ),
        "total_rows": int(len(all_records)),
        "cv_split": "saved folds; every complete (q,v1) group held out together",
        "cost_target": "log1p(mechanics_cost)",
        "expected_cost_formula": "exp(mu_log + 0.5 * variance_log) - 1",
        "cost_uncertainty_policy": (
            "posterior mean in cost space; no lower confidence bound"
        ),
        "cost_floor_steps": 1.0,
        "new_per_step_proxy_definition": (
            "mean(true_new among selected) / mean(actual mechanics_cost among selected)"
        ),
        "cost_gp_configuration": {
            "kernel": "same basis-free block Matern-3/2 as outcome GP",
            "action_length_scale": float(args.action_length_scale),
            "signal_variance": float(args.signal_variance),
            "noise_std": float(args.noise_std),
            "output_scale_floor": float(args.output_scale_floor),
        },
        "cost_prediction": _cost_prediction_metrics(all_records),
        "selection": metrics,
        "per_seed": seed_summaries,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path(__file__).resolve().parent
        / "result"
        / "persistent_k_panel_seed700_702",
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--action-length-scale", type=float, default=1.0)
    parser.add_argument("--signal-variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--output-scale-floor", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_arguments()
    result_directory = args.result_directory.resolve()
    aggregate_path = result_directory / "aggregate_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else aggregate["seeds"]

    all_records = []
    seed_summaries = []
    for seed in seeds:
        records, summary = _run_seed(
            result_directory / f"seed_{int(seed):03d}", args
        )
        all_records.extend(records)
        seed_summaries.append(summary)

    report = _aggregate(all_records, seed_summaries, args)
    (result_directory / "cost_aware_offline_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate["cost_aware_offline"] = report
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
