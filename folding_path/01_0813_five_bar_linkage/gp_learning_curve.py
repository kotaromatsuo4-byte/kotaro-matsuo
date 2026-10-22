"""Evaluate how the main5 surrogates improve with completed rollouts.

This script reads existing ``result/<name>/seed_*/training_data.npz`` files.
For every seed, the final perturbation rollouts are fixed once as a holdout
set.  A learning curve is then made from the trunk and the first ``n``
perturbation rollouts that precede that holdout.  Test inputs are only the
perturbation edge of each holdout rollout, so no edge from a test rollout is
allowed to enter the training set.

Three training modes are available:

``all_edges``
    Use every transition on each selected training rollout.
``perturbation_only``
    Use only the perturbation transition.  The deterministic trunk has no
    perturbation edge and therefore contributes no row in this mode.
``rollout_balanced``
    Deterministically retain at most the same number of well-spaced edges from
    each rollout, always retaining its perturbation edge when one exists.
    This is a subsampling approximation to equal rollout weighting; the exact
    GP API currently has no observation-weight argument.

The binary model is the GP-regression proxy implemented in ``gp.py``.  Its
probabilities are useful for ranking but are not assumed to be calibrated.
All validation metrics are computed at rollout level rather than edge level.
No scikit-learn dependency is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp import (
    fit_surrogate_models,
    make_terminal_output_spec,
    predict_binary_probability,
    predict_shared_kernel_gp,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
TRAINING_MODES = ("all_edges", "perturbation_only", "rollout_balanced")


def parse_integer_list(values, minimum=1):
    """Accept both ``--arg 4 8`` and ``--arg 4,8`` CLI notation."""
    integers = []
    for value in values:
        integers.extend(int(part) for part in str(value).split(",") if part)
    if not integers or any(value < minimum for value in integers):
        raise ValueError(f"integer lists must contain values >= {minimum}")
    return sorted(set(integers))


def resolve_result_directory(argument):
    """Resolve either a directory path or a name below the local result dir."""
    path = Path(argument).expanduser()
    if path.is_dir():
        return path.resolve()
    local_path = SCRIPT_DIRECTORY / "result" / argument
    if local_path.is_dir():
        return local_path.resolve()
    raise FileNotFoundError(f"result directory was not found: {argument}")


def seed_number(seed_directory):
    """Return the numeric suffix of a ``seed_<number>`` directory."""
    return int(seed_directory.name.rsplit("_", 1)[-1])


def find_seed_directories(result_directory, selected_seeds=None):
    """Find deterministic, numerically sorted seed directories."""
    directories = [
        path
        for path in Path(result_directory).glob("seed_*")
        if path.is_dir()
        and (path / "training_data.npz").is_file()
        and (path / "config.json").is_file()
    ]
    directories.sort(key=seed_number)
    if selected_seeds is not None:
        selected = set(int(seed) for seed in selected_seeds)
        directories = [path for path in directories if seed_number(path) in selected]
    if not directories:
        raise FileNotFoundError("no seed directories with saved training data were found")
    return directories


def load_seed_data(seed_directory):
    """Load one saved dataset and validate its row alignment."""
    with (Path(seed_directory) / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with np.load(Path(seed_directory) / "training_data.npz") as saved:
        data = {
            "inputs": np.asarray(saved["inputs"], dtype=float),
            "flat_labels": np.asarray(saved["flat_labels"], dtype=bool),
            "flatness_targets": np.asarray(
                saved["flatness_targets"], dtype=float
            ),
            "descriptor_targets": np.asarray(
                saved["descriptor_targets"], dtype=float
            ),
            "rollout_ids": np.asarray(saved["rollout_ids"], dtype=int),
            "edge_kinds": np.asarray(saved["edge_kinds"]).astype(str),
        }

    row_count = len(data["inputs"])
    if any(len(array) != row_count for array in data.values()):
        raise ValueError(f"unaligned arrays in {seed_directory}")
    if data["inputs"].ndim != 2 or data["descriptor_targets"].ndim != 2:
        raise ValueError(f"unexpected saved array shape in {seed_directory}")
    return data, config


def load_output_spec(model_directory, reference_point):
    """Reconstruct the active terminal-output positions used by main5."""
    directory = Path(model_directory)
    node = np.atleast_2d(
        np.loadtxt(directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(directory / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(directory / "fix.dat", delimiter=",", dtype=int)
    )
    fixed_indices = 3 * fix[:, 0] + fix[:, 1]
    index_free = np.delete(np.arange(3 * len(node)), fixed_indices)
    member_lengths = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    length_scale = float(np.mean(member_lengths))
    output_spec = make_terminal_output_spec(
        node,
        index_free,
        reference_point,
        length_scale,
    )
    return output_spec, index_free


def perturbation_rollout_ids(data):
    """Return rollout ids that contain an executed perturbation edge."""
    mask = data["edge_kinds"] == "perturbation"
    return np.unique(data["rollout_ids"][mask])


def fixed_split(data, holdout_rollouts):
    """Fix the final perturbation rollouts as test for every curve point."""
    perturbation_ids = perturbation_rollout_ids(data)
    holdout_count = int(holdout_rollouts)
    if len(perturbation_ids) <= holdout_count:
        raise ValueError(
            f"need more than {holdout_count} completed perturbation rollouts"
        )
    test_ids = perturbation_ids[-holdout_count:]
    available_train_ids = perturbation_ids[:-holdout_count]
    all_ids = np.unique(data["rollout_ids"])
    trunk_id = int(np.min(all_ids))
    if trunk_id in perturbation_ids:
        raise ValueError("the earliest saved rollout unexpectedly has a perturbation")
    return trunk_id, available_train_ids, test_ids


def one_perturbation_edge_per_rollout(data, rollout_ids):
    """Select one candidate-like test row from each requested rollout."""
    selected = []
    for rollout_id in rollout_ids:
        matches = np.flatnonzero(
            (data["rollout_ids"] == rollout_id)
            & (data["edge_kinds"] == "perturbation")
        )
        if len(matches) != 1:
            raise ValueError(
                f"rollout {rollout_id} has {len(matches)} perturbation edges"
            )
        selected.append(int(matches[0]))
    return np.asarray(selected, dtype=int)


def evenly_spaced_indices(indices, count):
    """Choose at most ``count`` deterministic, well-spaced entries."""
    values = np.asarray(indices, dtype=int)
    if len(values) <= count:
        return values
    positions = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
    return values[np.unique(positions)]


def balanced_rollout_indices(data, rollout_id, edges_per_rollout):
    """Subsample one rollout while always retaining its perturbation edge."""
    all_indices = np.flatnonzero(data["rollout_ids"] == rollout_id)
    limit = int(edges_per_rollout)
    if len(all_indices) <= limit:
        return all_indices

    perturbation = all_indices[
        data["edge_kinds"][all_indices] == "perturbation"
    ]
    if len(perturbation) == 0:
        return evenly_spaced_indices(all_indices, limit)

    required = int(perturbation[0])
    remaining = all_indices[all_indices != required]
    other = evenly_spaced_indices(remaining, max(limit - 1, 0))
    return np.sort(np.concatenate(([required], other)))


def training_indices(
    data,
    trunk_id,
    perturbation_ids,
    mode,
    balanced_edges_per_rollout,
):
    """Create a training-row index without admitting any holdout rollout."""
    selected_rollouts = np.concatenate(
        (np.asarray([trunk_id], dtype=int), np.asarray(perturbation_ids, dtype=int))
    )
    if mode == "all_edges":
        return np.flatnonzero(np.isin(data["rollout_ids"], selected_rollouts))

    if mode == "perturbation_only":
        return np.flatnonzero(
            np.isin(data["rollout_ids"], perturbation_ids)
            & (data["edge_kinds"] == "perturbation")
        )

    if mode == "rollout_balanced":
        pieces = [
            balanced_rollout_indices(
                data,
                rollout_id,
                balanced_edges_per_rollout,
            )
            for rollout_id in selected_rollouts
        ]
        return np.sort(np.concatenate(pieces))

    raise ValueError(f"unknown training mode: {mode}")


def binary_auc(labels, probabilities):
    """Compute tie-aware ROC AUC; return NaN when one class is absent."""
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    positive_count = int(np.count_nonzero(labels))
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return np.nan

    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(probabilities), dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1, dtype=float)
    for value in np.unique(probabilities):
        tied = np.flatnonzero(probabilities == value)
        ranks[tied] = np.mean(ranks[tied])
    rank_sum = np.sum(ranks[labels])
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def safe_correlation(first, second):
    """Return Pearson correlation only when both arrays vary."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def descriptor_archive_scales(output_spec, config):
    """Return the continuous W-distance scales used by main5 acquisition."""
    positions = np.asarray(output_spec["active_positions"], dtype=int)
    shape_dimension = 2 * int(output_spec["num_node"])
    return np.where(
        positions < shape_dimension,
        float(config["rms_tolerance"]),
        float(config["layer_uncertainty_scale"]),
    )


def fit_models(data, indices, state_dimension, config):
    """Fit the three existing surrogate models to selected saved rows."""
    return fit_surrogate_models(
        data["inputs"][indices],
        data["flat_labels"][indices],
        data["flatness_targets"][indices],
        data["descriptor_targets"][indices],
        state_dimension=state_dimension,
        state_length_scale=float(config["state_length_scale"]),
        action_length_scale=float(config["action_length_scale"]),
        signal_variance=float(config["signal_variance"]),
        regression_noise_std=float(config["regression_noise_std"]),
        classifier_noise_std=float(config["classifier_noise_std"]),
        descriptor_scale_floor=float(config["descriptor_scale_floor"]),
        prior_flat_probability=0.5,
    )


def evaluate_models(
    models,
    data,
    test_indices,
    archive_scales,
    root_descriptor,
    novelty_kappa,
):
    """Evaluate only one perturbation input per held-out rollout.

    ``root_novelty_*`` freezes the archive to the initial deterministic
    terminal.  It therefore asks whether the surrogate can rank actions that
    reach the other observed basin, without changing the task as train size
    grows.  The random corpus contains only two archive cells, so this is not
    evidence that an entirely unseen third basin can be predicted.
    """
    query = data["inputs"][test_indices]
    labels = data["flat_labels"][test_indices]
    flatness_target = data["flatness_targets"][test_indices]

    start = time.perf_counter()
    probability, _ = predict_binary_probability(models["flat_probability"], query)
    flatness_mean, _ = predict_shared_kernel_gp(models["flatness"], query)
    flatness_mean = flatness_mean[:, 0]

    flat_mask = labels & np.all(
        np.isfinite(data["descriptor_targets"][test_indices]), axis=1
    )
    descriptor_errors = np.empty(0, dtype=float)
    root_novelty_auc = np.nan
    root_novelty_precision_at_5 = np.nan
    root_novelty_positive_rate = np.nan
    if models["descriptor"] is not None:
        descriptor_mean_all, descriptor_variance_all = (
            predict_shared_kernel_gp(models["descriptor"], query)
        )
        descriptor_mean = descriptor_mean_all[flat_mask]
        target = data["descriptor_targets"][test_indices[flat_mask]]
        descriptor_errors = np.sqrt(
            np.mean(((descriptor_mean - target) / archive_scales) ** 2, axis=1)
        )

        true_new = np.zeros(len(query), dtype=bool)
        if np.any(flat_mask):
            true_distance = np.sqrt(
                np.mean(
                    (
                        (target - root_descriptor[None, :])
                        / archive_scales[None, :]
                    )
                    ** 2,
                    axis=1,
                )
            )
            true_new[flat_mask] = true_distance > 1.0

        predicted_distance = np.sqrt(
            np.mean(
                (
                    (descriptor_mean_all - root_descriptor[None, :])
                    / archive_scales[None, :]
                )
                ** 2,
                axis=1,
            )
        )
        confidence_radius = float(novelty_kappa) * np.sqrt(
            np.mean(
                descriptor_variance_all / archive_scales[None, :] ** 2,
                axis=1,
            )
        )
        novelty_score = probability * (
            predicted_distance + confidence_radius
        )
        root_novelty_auc = binary_auc(true_new, novelty_score)
        root_novelty_positive_rate = float(np.mean(true_new))
        top_count = min(5, len(novelty_score))
        top_indices = np.argsort(-novelty_score, kind="stable")[:top_count]
        root_novelty_precision_at_5 = float(np.mean(true_new[top_indices]))
    predict_seconds = time.perf_counter() - start

    brier = float(np.mean((probability - labels.astype(float)) ** 2))
    flatness_rmse = float(
        np.sqrt(np.mean((flatness_mean - flatness_target) ** 2))
    )
    return {
        "test_rollout_count": int(len(test_indices)),
        "test_flat_count": int(np.count_nonzero(labels)),
        "test_collision_count": int(np.count_nonzero(~labels)),
        "flat_auc": binary_auc(labels, probability),
        "flat_brier": brier,
        "flatness_rmse": flatness_rmse,
        "flatness_correlation": safe_correlation(
            flatness_mean, flatness_target
        ),
        "descriptor_flat_test_count": int(len(descriptor_errors)),
        "descriptor_median_archive_error": (
            float(np.median(descriptor_errors))
            if len(descriptor_errors)
            else np.nan
        ),
        "descriptor_within_one_rate": (
            float(np.mean(descriptor_errors <= 1.0))
            if len(descriptor_errors)
            else np.nan
        ),
        "root_novelty_auc": root_novelty_auc,
        "root_novelty_precision_at_5": root_novelty_precision_at_5,
        "root_novelty_positive_rate": root_novelty_positive_rate,
        "predict_seconds": float(predict_seconds),
    }


def evaluate_seed(
    seed_directory,
    train_sizes,
    modes,
    holdout_rollouts,
    balanced_edges_per_rollout,
    output_spec,
    index_free,
):
    """Evaluate all valid curve points for one seed."""
    data, config = load_seed_data(seed_directory)
    if data["descriptor_targets"].shape[1] != len(output_spec["active_positions"]):
        raise ValueError(
            f"descriptor/model dimension mismatch in {seed_directory}"
        )
    state_dimension = data["inputs"].shape[1] - len(index_free)
    if state_dimension <= 0:
        raise ValueError(f"invalid state/action split in {seed_directory}")

    trunk_id, available_train_ids, test_ids = fixed_split(
        data, holdout_rollouts
    )
    test_indices = one_perturbation_edge_per_rollout(data, test_ids)
    scales = descriptor_archive_scales(output_spec, config)
    trunk_descriptor_indices = np.flatnonzero(
        (data["rollout_ids"] == trunk_id)
        & data["flat_labels"]
        & np.all(np.isfinite(data["descriptor_targets"]), axis=1)
    )
    if len(trunk_descriptor_indices) == 0:
        raise ValueError("the initial trunk has no finite flat descriptor")
    root_descriptor = data["descriptor_targets"][trunk_descriptor_indices[0]]
    rows = []

    for requested_size in train_sizes:
        if requested_size > len(available_train_ids):
            print(
                f"skip seed {seed_number(seed_directory)} size {requested_size}: "
                f"only {len(available_train_ids)} pre-holdout rollouts"
            )
            continue
        selected_ids = available_train_ids[:requested_size]
        for mode in modes:
            indices = training_indices(
                data,
                trunk_id,
                selected_ids,
                mode,
                balanced_edges_per_rollout,
            )
            if len(indices) == 0:
                continue
            if np.any(np.isin(data["rollout_ids"][indices], test_ids)):
                raise RuntimeError("holdout leakage detected")

            fit_start = time.perf_counter()
            models = fit_models(data, indices, state_dimension, config)
            fit_seconds = time.perf_counter() - fit_start
            metrics = evaluate_models(
                models,
                data,
                test_indices,
                scales,
                root_descriptor,
                float(config["novelty_kappa"]),
            )
            rows.append(
                {
                    "result_name": Path(seed_directory).parent.name,
                    "seed": seed_number(seed_directory),
                    "mode": mode,
                    "requested_train_rollouts": int(requested_size),
                    "train_edge_count": int(len(indices)),
                    "train_rollout_count": int(
                        len(np.unique(data["rollout_ids"][indices]))
                    ),
                    "fit_seconds": float(fit_seconds),
                    **metrics,
                }
            )
    return rows


def write_rows_csv(path, rows):
    """Write dictionaries with a stable column order."""
    if not rows:
        raise ValueError("there are no rows to write")
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_rows(rows):
    """Compute seed means, standard deviations, and valid seed counts."""
    identifier_columns = {
        "result_name",
        "seed",
        "mode",
        "requested_train_rollouts",
    }
    metric_columns = [
        column for column in rows[0] if column not in identifier_columns
    ]
    groups = sorted(
        {
            (row["mode"], int(row["requested_train_rollouts"]))
            for row in rows
        },
        key=lambda item: (item[0], item[1]),
    )
    aggregated = []
    for mode, train_size in groups:
        selected = [
            row
            for row in rows
            if row["mode"] == mode
            and int(row["requested_train_rollouts"]) == train_size
        ]
        result = {
            "result_name": selected[0]["result_name"],
            "mode": mode,
            "requested_train_rollouts": train_size,
            "seed_count": len(selected),
        }
        for column in metric_columns:
            values = np.asarray([row[column] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{column}_mean"] = (
                float(np.mean(finite)) if len(finite) else np.nan
            )
            result[f"{column}_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            )
            result[f"{column}_valid_seed_count"] = int(len(finite))
        aggregated.append(result)
    return aggregated


def plot_learning_curves(rows, output_path):
    """Save seed-mean learning curves and one-standard-deviation bands."""
    panels = [
        ("flat_auc", "Flat ROC AUC", (0.0, 1.0)),
        ("flat_brier", "Flat Brier score", (0.0, None)),
        ("flatness_rmse", "Flatness RMSE", (0.0, None)),
        ("root_novelty_auc", "Root-archive novelty ROC AUC", (0.0, 1.0)),
        (
            "descriptor_median_archive_error",
            "Descriptor median W-error",
            (0.0, None),
        ),
        (
            "descriptor_within_one_rate",
            "Descriptor error <= 1",
            (0.0, 1.0),
        ),
        ("fit_seconds", "GP fit time [s]", (0.0, None)),
        ("train_edge_count", "Training edges", (0.0, None)),
    ]
    modes = [mode for mode in TRAINING_MODES if any(r["mode"] == mode for r in rows)]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for axis, (metric, title, limits) in zip(axes.flat, panels):
        for mode in modes:
            mode_rows = [row for row in rows if row["mode"] == mode]
            sizes = sorted({int(row["requested_train_rollouts"]) for row in mode_rows})
            means = []
            deviations = []
            for size in sizes:
                values = np.asarray(
                    [
                        row[metric]
                        for row in mode_rows
                        if int(row["requested_train_rollouts"]) == size
                    ],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                means.append(np.mean(finite) if len(finite) else np.nan)
                deviations.append(
                    np.std(finite, ddof=1) if len(finite) > 1 else 0.0
                )
            sizes = np.asarray(sizes, dtype=float)
            means = np.asarray(means, dtype=float)
            deviations = np.asarray(deviations, dtype=float)
            axis.plot(sizes, means, marker="o", label=mode)
            axis.fill_between(
                sizes,
                means - deviations,
                means + deviations,
                alpha=0.15,
            )
        axis.set_title(title)
        axis.set_xlabel("Training perturbation rollouts")
        axis.grid(alpha=0.25)
        # main5の現在のcold start (n=4) と、1200-stepで実際に集まった
        # 摂動rollout数に近いn=32を全panelで示す。
        axis.axvline(4, color="0.45", linestyle=":", linewidth=1.0)
        axis.axvline(32, color="0.45", linestyle="--", linewidth=1.0)
        if metric == "descriptor_median_archive_error":
            # archive尺度では1以下が同一cell判定に必要な精度である。
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        if limits[0] is not None:
            axis.set_ylim(bottom=limits[0])
        if limits[1] is not None:
            axis.set_ylim(top=limits[1])

    axes.flat[0].legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        help="result directory path, or a name below 01_0813.../result",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=["4", "8", "12", "16"],
        help="numbers of pre-holdout perturbation rollouts (space/comma separated)",
    )
    parser.add_argument(
        "--holdout-rollouts",
        type=int,
        default=8,
        help="fixed number of final perturbation rollouts used for testing",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=TRAINING_MODES,
        default=list(TRAINING_MODES),
    )
    parser.add_argument(
        "--balanced-edges-per-rollout",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="optional seed numbers (space/comma separated)",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=SCRIPT_DIRECTORY / "model",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    return parser


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    if args.holdout_rollouts <= 0 or args.balanced_edges_per_rollout <= 0:
        raise ValueError("holdout and balanced edge counts must be positive")
    train_sizes = parse_integer_list(args.train_sizes)
    seeds = (
        parse_integer_list(args.seeds, minimum=0)
        if args.seeds is not None
        else None
    )
    result_directory = resolve_result_directory(args.result)
    seed_directories = find_seed_directories(result_directory, seeds)

    _, first_config = load_seed_data(seed_directories[0])
    output_spec, index_free = load_output_spec(
        args.model_directory,
        float(first_config["reference_point"]),
    )
    rows = []
    for seed_directory in seed_directories:
        rows.extend(
            evaluate_seed(
                seed_directory,
                train_sizes,
                args.modes,
                args.holdout_rollouts,
                args.balanced_edges_per_rollout,
                output_spec,
                index_free,
            )
        )
    if not rows:
        raise ValueError("no requested training size was available")

    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else result_directory / "gp_learning_curve"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_rows_csv(output_directory / "seed_metrics.csv", rows)
    mean_rows = aggregate_seed_rows(rows)
    write_rows_csv(output_directory / "seed_mean_metrics.csv", mean_rows)
    plot_learning_curves(rows, output_directory / "learning_curve.png")

    evaluation_config = {
        "source_result_directory": str(result_directory),
        "train_sizes": train_sizes,
        "holdout_rollouts": args.holdout_rollouts,
        "modes": list(args.modes),
        "balanced_edges_per_rollout": args.balanced_edges_per_rollout,
        "seeds": [seed_number(path) for path in seed_directories],
    }
    with (output_directory / "evaluation_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(evaluation_config, stream, indent=2)
    print(f"saved {len(rows)} seed-level evaluations to {output_directory}")


if __name__ == "__main__":
    main()
