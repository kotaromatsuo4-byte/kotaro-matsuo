"""Promotion, random-baseline, and 10000-episode spectral analysis.

This module is additive: it consumes the immutable outputs of the existing
ori6 analyzer and the spectral production runner.  It never rewrites the
existing runner, analyzer, candidate, or their historical artifacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

import analyze_ori_codex6_experiments as base_analyzer
import run_ori_codex6_experiments as base_runner
import run_ori_codex6_spectral_production as production


SCHEMA_VERSION = 1
ANALYZER_VERSION = "1.1.0"
ANALYSIS_DIRECTORY = production.ANALYSIS_DIRECTORY
TWO_TASK_REPORT_PATH = (
    ANALYSIS_DIRECTORY / "spectral_two_task_full_diagnostic_report.json"
)
TWO_TASK_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY / "spectral_two_task_full_diagnostic_summary.csv"
)
TWO_TASK_CURVE_DIRECTORY = (
    ANALYSIS_DIRECTORY / "spectral_two_task_full_diagnostic_curves"
)
TWO_TASK_ROLLING_WINDOW = 200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_once(path: Path, payload: dict, fingerprint_field: str) -> dict:
    path = Path(path)
    payload = dict(payload)
    payload[fingerprint_field] = production.canonical_self_fingerprint(
        payload, fingerprint_field
    )
    if path.is_file():
        existing = load_json(path)
        if existing != payload:
            raise RuntimeError(f"Refusing to overwrite immutable authority: {path}")
        return existing
    base_runner.atomic_json(path, payload)
    return payload


def finalize_regression_authority() -> dict:
    """Bind the exact post-run report SHA after independently validating it."""

    if production.REGRESSION_AUTHORITY_PATH.is_file():
        return production.validate_regression_authority()
    report_path = production.REGRESSION_REPORT_PATH
    if not report_path.is_file():
        raise FileNotFoundError(
            "Regression report has not been generated; promotion remains "
            f"pending: {report_path}"
        )
    report = load_json(report_path)
    report_sha = production.file_sha256(report_path)
    validation = production.validate_regression_report_payload(
        report, report_path=report_path, report_sha256=report_sha
    )
    manifest = production.validate_fixed_regression_manifest()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "authority_type": "ori6_spectral_production_regression_promotion",
        "status": "pass",
        "candidate_mode": production.MODE,
        "candidate_mode_slug": production.MODE_SLUG,
        "regression_report": {
            "path": str(report_path.resolve()),
            "sha256": report_sha,
            "canonical_sha256": production.canonical_digest(report),
        },
        "regression_manifest": {
            "path": str(production.REGRESSION_MANIFEST_PATH.resolve()),
            "sha256": production.REGRESSION_MANIFEST_SHA256,
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        },
        "validated_gate": validation,
        "all_three_task_gates_pass": True,
        "complete_endpoint_count": 18,
        "task_names": list(production.REGRESSION_TASKS),
        "base_source_sha256": base_runner.source_hashes(),
        "parent_additive_source_sha256": (
            base_runner.rejected_crossing_additive_source_hashes()
        ),
        "spectral_additive_source_sha256": (
            base_runner.spectral_subspace_additive_source_hashes()
        ),
        "created_utc": utc_now(),
    }
    authority = write_once(
        production.REGRESSION_AUTHORITY_PATH,
        payload,
        "authority_fingerprint",
    )
    return production.validate_regression_authority(
        production.REGRESSION_AUTHORITY_PATH
    )


def bootstrap_rmst(assigned: list[int], seed: int,
                   samples: int = 20000) -> dict:
    values = np.asarray(assigned, dtype=float)
    if not len(values):
        return {"estimate": None, "ci95_lower": None, "ci95_upper": None}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(samples),
    }


def random_task_summary(validation: dict, task_index: int) -> dict:
    aggregate = validation["aggregate"]
    budget = int(aggregate["mechanics_budget"])
    assigned = [
        int(row["first_success_mechanical_steps"])
        if row["success"] else budget
        for row in aggregate["trial_summaries"]
    ]
    per_seed = {
        str(int(row["seed"])): {
            "success": bool(row["success"]),
            "mechanical_steps": int(row["mechanical_steps"]),
            "first_success_mechanical_steps": (
                int(row["first_success_mechanical_steps"])
                if row["first_success_mechanical_steps"] is not None
                else None
            ),
            "action_chain_sha256": row["action_chain_sha256"],
            "action_prefix_sha256_by_mechanics": row[
                "action_prefix_sha256_by_mechanics"
            ],
        }
        for row in aggregate["trial_summaries"]
    }
    return {
        "pass": True,
        "task": aggregate["task"],
        "mechanics_budget": budget,
        "seeds": aggregate["seeds"],
        "trials": aggregate["completed_trials"],
        "successes": aggregate["successes"],
        "censored": aggregate["censored"],
        "success_at_budget": aggregate["success_at_budget"],
        "first_success_mechanical_steps": aggregate[
            "first_success_mechanical_steps"
        ],
        "assigned_first_hit_or_censor_mechanics": assigned,
        "mechanics_rmst": bootstrap_rmst(
            assigned, 9510000 + task_index
        ),
        "per_seed": per_seed,
        "all_reward_integrity_pass": aggregate[
            "all_reward_integrity_pass"
        ],
        "output_root": validation["output_root"],
        "root_fingerprint": validation["root_fingerprint"],
        "run_config_sha256": validation["run_config_sha256"],
        "aggregate_sha256": validation["aggregate_sha256"],
    }


def random_screen_report(require_complete: bool = False) -> dict:
    regression = production.prerequisite_status()
    if not regression["pass"]:
        raise RuntimeError(f"Regression authority is not valid: {regression}")
    jobs = production.random_screen_jobs(regression)
    task_results = {}
    errors = []
    for index, job in enumerate(jobs):
        try:
            validation = production.validate_random_artifact(job)
            task_results[job["task"]] = random_task_summary(validation, index)
        except Exception as error:
            errors.append({"task": job["task"], "error": repr(error)})
    complete = bool(
        not errors
        and set(task_results) == set(production.TASK_NAMES)
        and all(row["pass"] for row in task_results.values())
    )
    if require_complete and not complete:
        raise RuntimeError(f"Random screen is incomplete: {errors}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "report_type": "ori6_spectral_production_random_screen",
        "status": "pass" if complete else "pending",
        "candidate_mode": production.MODE,
        "candidate_mode_slug": production.MODE_SLUG,
        "method": "uniform",
        "action_law": production.random_harness.ACTION_LAW,
        "tasks": list(production.TASK_NAMES),
        "seeds": list(production.RANDOM_SEEDS),
        "mechanics_budget": production.INITIAL_RANDOM_MECHANICS,
        "total_preregistered_censor_mechanics": (
            len(production.TASK_NAMES)
            * len(production.RANDOM_SEEDS)
            * production.INITIAL_RANDOM_MECHANICS
        ),
        "all_five_tasks_complete": complete,
        "task_results": task_results,
        "errors": errors,
        "regression_authority_path": str(
            production.REGRESSION_AUTHORITY_PATH.resolve()
        ),
        "regression_authority_sha256": production.file_sha256(
            production.REGRESSION_AUTHORITY_PATH
        ),
        "regression_authority_fingerprint": regression[
            "authority_fingerprint"
        ],
        "candidate_reward_sha256": base_runner.candidate_reward_sha256(
            production.MODE
        ),
        "transition_semantics_sha256": (
            base_runner.candidate_transition_semantics_sha256(
                production.MODE
            )
        ),
        "base_source_sha256": base_runner.source_hashes(),
        "candidate_additive_source_sha256": {
            **base_runner.rejected_crossing_additive_source_hashes(),
            **base_runner.spectral_subspace_additive_source_hashes(),
        },
        "random_harness_sha256": production.file_sha256(
            production.SCRIPT_DIRECTORY
            / "ori_codex6_spectral_random_baseline.py"
        ),
        "extension_rule": (
            "for each task whose RL first hit exceeds 102400 mechanics, rerun "
            "the same seeds 5101--5110 from their deterministic initial states "
            "at the exact RL first-hit mechanics; if RL is censored, use its "
            "total consumed mechanics"
        ),
        "generated_utc": utc_now(),
    }
    if complete:
        if production.RANDOM_SCREEN_REPORT_PATH.is_file():
            return production.validate_random_screen_report()
        return write_once(
            production.RANDOM_SCREEN_REPORT_PATH,
            payload,
            "report_fingerprint",
        )
    payload["report_fingerprint"] = production.canonical_self_fingerprint(
        payload, "report_fingerprint"
    )
    return payload


def random_extension_results(require_complete: bool = False) -> dict:
    regression = production.prerequisite_status()
    if not regression["pass"]:
        raise RuntimeError("Regression authority is not valid")
    jobs = production.random_extension_jobs(regression)
    initial = production.validate_random_screen_report()
    results = {}
    errors = []
    for index, job in enumerate(jobs):
        try:
            validation = production.validate_random_artifact(job)
            summary = random_task_summary(
                validation, 100 + index
            )
            initial_seeds = initial["task_results"][job["task"]][
                "per_seed"
            ]
            extension_seeds = summary["per_seed"]
            prefix_errors = []
            checkpoint = str(production.INITIAL_RANDOM_MECHANICS)
            for seed in map(str, production.RANDOM_SEEDS):
                initial_row = initial_seeds[seed]
                extension_row = extension_seeds[seed]
                if initial_row["success"]:
                    replay_pass = bool(
                        extension_row["success"]
                        and extension_row["first_success_mechanical_steps"]
                        == initial_row["first_success_mechanical_steps"]
                        and extension_row["action_chain_sha256"]
                        == initial_row["action_chain_sha256"]
                    )
                else:
                    replay_pass = bool(
                        extension_row[
                            "action_prefix_sha256_by_mechanics"
                        ].get(checkpoint)
                        == initial_row[
                            "action_prefix_sha256_by_mechanics"
                        ].get(checkpoint)
                    )
                if not replay_pass:
                    prefix_errors.append(int(seed))
            summary["initial_prefix_replay_pass"] = (
                not prefix_errors
            )
            summary["initial_prefix_replay_failed_seeds"] = (
                prefix_errors
            )
            if prefix_errors:
                raise RuntimeError(
                    f"Extension prefix mismatch for seeds {prefix_errors}"
                )
            results[job["task"]] = summary
        except Exception as error:
            errors.append({"task": job["task"], "error": repr(error)})
    complete = not errors and len(results) == len(jobs)
    if require_complete and not complete:
        raise RuntimeError(f"Random extension is incomplete: {errors}")
    return {
        "status": "pass" if complete else "pending",
        "complete": complete,
        "required_tasks": [job["task"] for job in jobs],
        "task_results": results,
        "errors": errors,
    }


def selected_random_summary(task_name: str, rl_budget: int,
                            initial_report: dict,
                            extension: dict) -> tuple[dict, str]:
    initial = initial_report["task_results"][task_name]
    if rl_budget <= production.INITIAL_RANDOM_MECHANICS:
        return initial, "initial_102400_mechanics_screen"
    candidate = extension.get("task_results", {}).get(task_name)
    if candidate is None or int(candidate["mechanics_budget"]) != rl_budget:
        raise RuntimeError(
            f"Task {task_name} requires random extension to {rl_budget} mechanics"
        )
    return candidate, "preregistered_same_seed_extension"


def random_comparison(record: dict, random_summary: dict,
                      source: str) -> dict:
    hit = record["first_hit"]
    rl_mechanics = (
        int(hit["mechanical_steps"])
        if not hit["censored"]
        else int(record["training"]["total_mechanical_steps"])
    )
    rmst = float(random_summary["mechanics_rmst"]["estimate"])
    return {
        "task": record["task"],
        "random_source": source,
        "common_mechanics_horizon": int(random_summary["mechanics_budget"]),
        "rl_strict_success": not hit["censored"],
        "rl_first_hit_episode": hit.get("episode"),
        "rl_first_hit_or_censor_mechanics": rl_mechanics,
        "random_trials": random_summary["trials"],
        "random_successes": random_summary["successes"],
        "random_success_at_horizon": random_summary["success_at_budget"],
        "random_mechanics_rmst": random_summary["mechanics_rmst"],
        "random_rmst_over_rl_first_hit_mechanics": (
            float(rmst / rl_mechanics)
            if not hit["censored"] and rl_mechanics > 0 else None
        ),
        "interpretation": (
            "descriptive first-discovery mechanics ratio"
            if not hit["censored"]
            else "both methods are compared only as censored outcomes at the "
            "RL-consumed mechanics horizon"
        ),
    }


def plot_random_comparisons(comparisons: list[dict],
                            random_by_task: dict[str, dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    directory = ANALYSIS_DIRECTORY / "spectral_production_random_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    for comparison in comparisons:
        task = comparison["task"]
        summary = random_by_task[task]
        values = sorted(summary["first_success_mechanical_steps"])
        trials = int(summary["trials"])
        cap = int(summary["mechanics_budget"])
        figure, axis = plt.subplots(figsize=(7.4, 4.6))
        if values:
            axis.step(
                values, np.arange(1, len(values) + 1) / trials,
                where="post", color="tab:blue", linewidth=2,
                label="uniform first-hit ECDF",
            )
        censored = trials - len(values)
        if censored:
            axis.scatter(
                [cap], [len(values) / trials], marker="|", s=190,
                color="tab:blue", label=f"uniform censored x{censored}",
            )
        axis.axvline(
            comparison["rl_first_hit_or_censor_mechanics"],
            color="tab:orange", linewidth=2, linestyle="--",
            label=(
                "RL first strict hit" if comparison["rl_strict_success"]
                else "RL censor"
            ),
        )
        axis.axvline(
            summary["mechanics_rmst"]["estimate"], color="tab:blue",
            linewidth=1.5, linestyle=":", label="uniform mechanics RMST",
        )
        axis.set_xlim(left=0)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Cumulative mechanical steps")
        axis.set_ylabel("First strict-success probability")
        axis.set_title(f"{task}: spectral RL vs matched uniform")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        output = directory / f"{task}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(str(output.resolve()))
    return outputs


def full_primary_report(require_complete: bool = False) -> dict:
    records = production.full_artifact_records(
        require_complete=require_complete
    )
    initial = production.validate_random_screen_report()
    try:
        extension = random_extension_results(require_complete=False)
    except Exception as error:
        extension = {
            "status": "pending", "complete": False,
            "required_tasks": [], "task_results": {},
            "errors": [{"error": repr(error)}],
        }
    comparisons = []
    random_by_task = {}
    extension_requirements = []
    for record in records:
        hit = record["first_hit"]
        rl_budget = (
            int(hit["mechanical_steps"])
            if not hit["censored"]
            else int(record["training"]["total_mechanical_steps"])
        )
        try:
            random_summary, source = selected_random_summary(
                record["task"], rl_budget, initial, extension
            )
        except RuntimeError:
            extension_requirements.append({
                "task": record["task"],
                "required_mechanics_budget": rl_budget,
            })
            continue
        random_by_task[record["task"]] = random_summary
        comparisons.append(random_comparison(record, random_summary, source))
    curves = base_analyzer.plot_learning_curves(
        records, "spectral_production_P0X2RCS05_seed4101"
    ) if records else []
    random_plots = plot_random_comparisons(
        comparisons, random_by_task
    ) if comparisons else []
    complete = len(records) == len(production.TASK_NAMES)
    comparison_complete = complete and not extension_requirements
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "report_type": "ori6_spectral_production_full_primary",
        "status": (
            "complete" if comparison_complete
            else "rl_complete_random_extension_pending"
            if complete else "rl_incomplete"
        ),
        "candidate_mode": production.MODE,
        "candidate_mode_slug": production.MODE_SLUG,
        "seed": production.PRIMARY_SEED,
        "episodes": 10000,
        "all_five_rl_complete": complete,
        "random_comparison_complete": comparison_complete,
        "records": records,
        "random_comparisons": comparisons,
        "random_extension_requirements": extension_requirements,
        "random_extension_command": (
            f"{Path(base_runner.sys.executable)} "
            "run_ori_codex6_spectral_production.py --stage "
            "random_extension --max-parallel 5 --execute"
            if extension_requirements else None
        ),
        "learning_curve_pngs": curves,
        "random_comparison_pngs": random_plots,
        "regression_authority_sha256": production.file_sha256(
            production.REGRESSION_AUTHORITY_PATH
        ),
        "random_screen_report_sha256": production.file_sha256(
            production.RANDOM_SCREEN_REPORT_PATH
        ),
        "generated_utc": utc_now(),
    }
    payload["report_fingerprint"] = production.canonical_self_fingerprint(
        payload, "report_fingerprint"
    )
    output = ANALYSIS_DIRECTORY / "spectral_production_full_primary_report.json"
    base_runner.atomic_json(output, payload)
    if records:
        base_analyzer.write_csv(
            ANALYSIS_DIRECTORY / "spectral_production_full_primary_summary.csv",
            [base_analyzer.flatten_record(record) for record in records],
        )
    if comparisons:
        base_analyzer.write_csv(
            ANALYSIS_DIRECTORY
            / "spectral_production_random_comparison_summary.csv",
            comparisons,
        )
    return payload


def rolling_mean(values, window: int = TWO_TASK_ROLLING_WINDOW) -> np.ndarray:
    """Trailing mean with a partial window at the beginning."""

    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("rolling window must be positive")
    if not len(values):
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    counts = indices - starts + 1
    return (cumulative[indices + 1] - cumulative[starts]) / counts


def rolling_median(values,
                   window: int = TWO_TASK_ROLLING_WINDOW) -> np.ndarray:
    """Trailing median with a partial window at the beginning."""

    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("rolling window must be positive")
    if not len(values):
        return values
    result = np.empty(len(values), dtype=float)
    prefix = min(len(values), window - 1)
    for index in range(prefix):
        result[index] = float(np.median(values[:index + 1]))
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        result[window - 1:] = np.median(windows, axis=1)
    return result


def numeric_series(history: list[dict], name: str,
                   default: float = 0.0) -> np.ndarray:
    result = []
    for row in history:
        try:
            value = float(row.get(name, default))
        except (TypeError, ValueError):
            value = float(default)
        result.append(value)
    return np.asarray(result, dtype=float)


def _plot_two_task_axis_set(history: list[dict], x: np.ndarray,
                            x_label: str, task_name: str,
                            output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = TWO_TASK_ROLLING_WINDOW
    returns = numeric_series(history, "total_reward")
    terminal_f = numeric_series(history, "flatness_ratio")
    spread_error = numeric_series(history, "spread_error")
    cumulative_mean_return = np.cumsum(returns) / np.arange(
        1, len(returns) + 1
    )

    figure, axes = plt.subplots(4, 2, figsize=(14.0, 14.5))
    axes = axes.ravel()

    axes[0].plot(x, returns, color="tab:blue", alpha=0.10, linewidth=0.55,
                 label="episodic return")
    axes[0].plot(x, rolling_mean(returns, window), color="tab:blue",
                 linewidth=1.4, label=f"{window}-episode moving mean")
    axes[0].set_title("Moving episodic return")
    axes[0].set_ylabel("Episodic return")

    axes[1].plot(x, cumulative_mean_return, color="tab:purple",
                 linewidth=1.45, label="cumulative mean return")
    axes[1].set_title("Cumulative mean episodic return (not cumulative sum)")
    axes[1].set_ylabel("Mean return through current episode")

    positive_floor = 1.0e-12
    axes[2].plot(
        x, np.maximum(rolling_mean(terminal_f, window), positive_floor),
        color="tab:green", linewidth=1.35,
        label=f"{window}-episode rolling mean",
    )
    axes[2].plot(
        x, np.maximum(rolling_median(terminal_f, window), positive_floor),
        color="tab:olive", linewidth=1.25, linestyle="--",
        label=f"{window}-episode rolling median",
    )
    axes[2].axhline(0.5, color="0.35", linewidth=0.75, linestyle=":",
                    label="F=0.5 diagnostic threshold")
    axes[2].set_yscale("log")
    axes[2].set_title("Terminal F (flatness ratio), log scale")
    axes[2].set_ylabel("Terminal F")

    axes[3].plot(
        x, np.maximum(rolling_mean(spread_error, window), positive_floor),
        color="tab:orange", linewidth=1.35,
        label=f"{window}-episode rolling mean",
    )
    axes[3].plot(
        x, np.maximum(rolling_median(spread_error, window), positive_floor),
        color="tab:red", linewidth=1.25, linestyle="--",
        label=f"{window}-episode rolling median",
    )
    axes[3].axhline(0.1, color="0.35", linewidth=0.75, linestyle=":",
                    label="spread error=0.1 threshold")
    axes[3].set_yscale("log")
    axes[3].set_title("Terminal spread error, log scale")
    axes[3].set_ylabel("Absolute spread error")

    termination_fields = (
        ("success", "success", "tab:green", "-"),
        ("timeout", "timeout", "tab:gray", "--"),
        ("wrong_flat", "wrong-flat", "tab:orange", ":"),
        ("correction_failed", "correction-failed", "tab:red", "-."),
        ("no_modes", "no-modes", "tab:brown", (0, (3, 1, 1, 1))),
        ("self_crossing", "self-crossing", "tab:pink", (0, (1, 1))),
    )
    for field, label, color, style in termination_fields:
        axes[4].plot(
            x, rolling_mean(numeric_series(history, field), window),
            color=color, linestyle=style, linewidth=1.15, label=label,
        )
    axes[4].set_ylim(-0.03, 1.03)
    axes[4].set_title(f"{window}-episode success / termination rates")
    axes[4].set_ylabel("Episode fraction")

    axes[5].plot(
        x, rolling_mean(numeric_series(
            history, "ori6_rejected_crossing_rate"
        ), window),
        color="tab:red", linewidth=1.3, label="rejected action rate",
    )
    axes[5].plot(
        x, rolling_mean(numeric_series(
            history, "ori6_rejected_crossing_escape_rate"
        ), window),
        color="tab:green", linewidth=1.25, linestyle="--",
        label="escape-after-rejected-streak rate",
    )
    axes[5].set_ylim(-0.03, 1.03)
    axes[5].set_title("Rejected-crossing continuation signal")
    axes[5].set_ylabel("Rolling rate")

    axes[6].plot(
        x, rolling_mean(numeric_series(
            history, "ori6_spectral_total_reward"
        ), window),
        color="tab:blue", linewidth=1.3,
        label="spectral episodic reward, rolling mean",
    )
    axes[6].plot(
        x, rolling_mean(numeric_series(
            history, "ori6_spectral_maximum_potential"
        ), window),
        color="tab:cyan", linewidth=1.25, linestyle="--",
        label="spectral maximum potential, rolling mean",
    )
    axes[6].axhline(0.0, color="0.35", linewidth=0.7)
    axes[6].set_title("Spectral learning signal")
    axes[6].set_ylabel("Reward / normalized potential")

    for field, label, color, style in (
        ("near_miss_qF", "near-miss flatness qF", "tab:green", "-"),
        ("near_miss_qD", "near-miss spread qD", "tab:orange", "--"),
        ("near_miss_q", "joint near-miss q", "tab:purple", ":"),
    ):
        axes[7].plot(
            x, rolling_mean(numeric_series(history, field), window),
            color=color, linestyle=style, linewidth=1.2, label=label,
        )
    axes[7].set_ylim(-0.03, 1.03)
    axes[7].set_title("Near-miss component signals")
    axes[7].set_ylabel("Rolling quality")

    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=7.2, loc="best")
    figure.suptitle(
        f"P0X2RCS05 long-horizon diagnostic: {task_name}, seed4101",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_two_task_diagnostic_curves(
    records: list[dict], output_directory: Path = TWO_TASK_CURVE_DIRECTORY,
) -> list[dict]:
    outputs = []
    output_directory = Path(output_directory)
    for record in sorted(records, key=lambda row: row["task"]):
        history = base_analyzer.load_csv(
            Path(record["artifact_path"]) / "training_history.csv"
        )
        episodes = numeric_series(history, "episode")
        mechanics = np.cumsum(numeric_series(history, "steps"))
        episode_path = output_directory / (
            f"{record['task']}_seed{record['seed']}_by_episode.png"
        )
        mechanics_path = output_directory / (
            f"{record['task']}_seed{record['seed']}_by_mechanics.png"
        )
        _plot_two_task_axis_set(
            history, episodes, "Episode", record["task"], episode_path
        )
        _plot_two_task_axis_set(
            history, mechanics, "Cumulative mechanical steps",
            record["task"], mechanics_path,
        )
        outputs.append({
            "task": record["task"],
            "episode_axis_png": str(episode_path.resolve()),
            "cumulative_mechanics_axis_png": str(mechanics_path.resolve()),
            "rolling_window_episodes": TWO_TASK_ROLLING_WINDOW,
        })
    return outputs


def two_task_record_integrity(record: dict, job: dict) -> dict:
    errors = []
    validation = record.get("artifact_validation") or {}
    if not (
        validation.get("valid") is True
        and validation.get("complete") is True
        and validation.get("endpoint_episode")
        == production.TWO_TASK_DIAGNOSTIC_EPISODES
        and validation.get("training_rows")
        == production.TWO_TASK_DIAGNOSTIC_EPISODES
        and record.get("endpoint_episode")
        == production.TWO_TASK_DIAGNOSTIC_EPISODES
    ):
        errors.append("complete_10000_artifact_validation")
    if (
        record.get("task") != job["task"]["name"]
        or record.get("seed") != production.PRIMARY_SEED
        or record.get("profile") != job["profile"]["name"]
        or record.get("implementation") != "ori6"
        or record.get("candidate_mode") != production.MODE
    ):
        errors.append("record_cohort_identity")
    if (
        record.get("source_sha256") != job["source_sha256"]
        or record.get("model_sha256") != job["model_sha256"]
    ):
        errors.append("record_source_or_model_binding")
    training = record.get("training") or {}
    rejected = training.get("rejected_crossing") or {}
    spectral = training.get("spectral_subspace") or {}
    if not (
        training.get("maximum_abs_decomposition_error", math.inf)
        <= 1.0e-10
        and training.get("nonfinite_reward_components") == 0
        and rejected.get("transition_integrity_pass") is True
        and rejected.get("episodes")
        == production.TWO_TASK_DIAGNOSTIC_EPISODES
        and spectral.get("integrity_pass") is True
        and spectral.get("episodes")
        == production.TWO_TASK_DIAGNOSTIC_EPISODES
        and spectral.get("maximum_abs_reward_decomposition_error", math.inf)
        <= 1.0e-10
        and spectral.get("nonfinite_component_count") == 0
    ):
        errors.append("training_reward_transition_spectral_integrity")
    for name in ("last_policy_heldout", "best_policy_heldout"):
        heldout = record.get(name) or {}
        heldout_rejected = heldout.get("rejected_crossing") or {}
        heldout_spectral = heldout.get("spectral_subspace") or {}
        if not (
            heldout.get("episodes") == 100
            and heldout_rejected.get("transition_integrity_pass") is True
            and heldout_rejected.get("episodes") == 100
            and heldout_spectral.get("integrity_pass") is True
            and heldout_spectral.get("episodes") == 100
            and heldout_spectral.get("nonfinite_component_count") == 0
        ):
            errors.append(f"{name}_100_episode_integrity")
    identity = record.get("candidate_identity") or {}
    if not (
        identity.get("ori6_candidate")
        == base_runner.SPECTRAL_SUBSPACE_CANDIDATE_NAME
        and identity.get("ori6_parent_candidate")
        == base_runner.REJECTED_CROSSING_CANDIDATE_NAME
        and identity.get("ori6_crossing_policy")
        == base_runner.REJECTED_CROSSING_POLICY
        and identity.get("spectral_formula_sha256")
        == base_runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
        and base_runner.values_equal(
            identity.get("spectral_weight"),
            base_runner.SPECTRAL_SUBSPACE_WEIGHT,
        )
    ):
        errors.append("candidate_formula_weight_identity")
    artifact = Path(str(record.get("artifact_path", "")))
    if not (
        artifact.is_dir()
        and production.file_sha256(artifact / "run_config.json")
        == record.get("run_config_sha256")
        and production.file_sha256(artifact / "training_history.csv")
        == record.get("training_history_sha256")
    ):
        errors.append("artifact_file_sha256_binding")
    return {
        "pass": not errors,
        "errors": errors,
        "task": job["task"]["name"],
        "seed": job["seed"],
        "endpoint_episode": record.get("endpoint_episode"),
    }


def two_task_full_diagnostic_report(
    require_complete: bool = False,
) -> dict:
    screen_gate = production.validate_two_task_screen_authority()
    gates = {"spectral_screen_user_scope_override": screen_gate}
    jobs = production.build_two_task_diagnostic_jobs(screen_gate)
    errors = []
    try:
        manifest = production.validate_immutable_manifest(
            production.TWO_TASK_DIAGNOSTIC_STAGE, jobs, gates
        )
    except Exception as error:
        manifest = None
        errors.append({"scope": "manifest", "error": repr(error)})

    records = []
    integrity = {}
    for job in jobs:
        destination = Path(job["output_directory"])
        effective, validation = base_runner.discover_output(job, destination)
        if effective is None:
            errors.append({
                "job_id": job["job_id"],
                "error": "complete artifact not found",
                "validation": validation,
            })
            continue
        record = base_analyzer.summarize_job(
            job, effective, validation, "complete"
        )
        records.append(record)
        integrity[job["task"]["name"]] = two_task_record_integrity(
            record, job
        )

    expected_keys = {
        (task, production.PRIMARY_SEED)
        for task in production.TWO_TASK_DIAGNOSTIC_TASKS
    }
    observed_keys = {(row["task"], row["seed"]) for row in records}
    exact_cohort_complete = bool(
        manifest is not None
        and len(records) == 2
        and observed_keys == expected_keys
        and all(
            row.get("endpoint_episode")
            == production.TWO_TASK_DIAGNOSTIC_EPISODES
            for row in records
        )
    )
    integrity_pass = bool(
        exact_cohort_complete
        and set(integrity) == set(production.TWO_TASK_DIAGNOSTIC_TASKS)
        and all(row["pass"] for row in integrity.values())
    )
    curves = plot_two_task_diagnostic_curves(records) if records else []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "report_type": "ori6_spectral_two_task_full_diagnostic",
        "stage": production.TWO_TASK_DIAGNOSTIC_STAGE,
        "status": (
            "complete_integrity_pass" if integrity_pass
            else "integrity_rejected" if exact_cohort_complete
            else "incomplete"
        ),
        "candidate_mode": production.MODE,
        "candidate_mode_slug": production.MODE_SLUG,
        "tasks": list(production.TWO_TASK_DIAGNOSTIC_TASKS),
        "seed": production.PRIMARY_SEED,
        "episodes": production.TWO_TASK_DIAGNOSTIC_EPISODES,
        "spectral_weight": base_runner.SPECTRAL_SUBSPACE_WEIGHT,
        "exact_two_job_cohort_complete": exact_cohort_complete,
        "all_integrity_pass": integrity_pass,
        "manifest_authority": manifest,
        "screen_and_user_scope_authority": screen_gate,
        "records": records,
        "integrity_by_task": integrity,
        "errors": errors,
        "learning_curve_pngs": curves,
        "learning_curve_definition": {
            "axis_sets": ["episode", "cumulative mechanical steps"],
            "rolling_window_episodes": TWO_TASK_ROLLING_WINDOW,
            "return_panels": [
                "moving episodic return",
                "cumulative mean episodic return (not cumulative sum)",
            ],
            "terminal_F_panel": (
                "rolling mean and median of flatness_ratio; log y-axis"
            ),
            "spread_panel": (
                "rolling mean and median of spread_error; log y-axis"
            ),
            "event_panels": [
                "success and termination rates",
                "rejected-action and escape rates",
                "spectral episodic reward and maximum potential",
                "near-miss qF/qD/joint q",
            ],
        },
        "source_sha256": base_runner.source_hashes(),
        "parent_additive_source_sha256": (
            base_runner.rejected_crossing_additive_source_hashes()
        ),
        "spectral_additive_source_sha256": (
            base_runner.spectral_subspace_additive_source_hashes()
        ),
        "model_sha256_by_task": {
            task: base_runner.model_hashes(base_runner.TASKS[task])
            for task in production.TWO_TASK_DIAGNOSTIC_TASKS
        },
        "candidate_reward_sha256": base_runner.candidate_reward_sha256(
            production.MODE
        ),
        "candidate_mechanics_sha256_by_task": {
            task: base_runner.candidate_mechanics_sha256(
                production.MODE, base_runner.TASKS[task]
            ) for task in production.TWO_TASK_DIAGNOSTIC_TASKS
        },
        "spectral_formula_sha256": (
            base_runner.SPECTRAL_SUBSPACE_FORMULA_SHA256
        ),
        "generated_utc": utc_now(),
    }
    payload["report_fingerprint"] = production.canonical_self_fingerprint(
        payload, "report_fingerprint"
    )
    base_runner.atomic_json(TWO_TASK_REPORT_PATH, payload)
    if records:
        base_analyzer.write_csv(
            TWO_TASK_SUMMARY_PATH,
            [base_analyzer.flatten_record(record) for record in records],
        )
    if require_complete and not integrity_pass:
        raise RuntimeError(
            "two_task_full_diagnostic requires exactly two complete 10000-"
            f"episode integrity-passing artifacts; errors={errors}; "
            f"integrity={integrity}"
        )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "regression_authority", "random_screen", "full_primary",
            "random_extension", production.TWO_TASK_DIAGNOSTIC_STAGE,
        ),
        required=True,
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    if options.stage == "regression_authority":
        result = finalize_regression_authority()
    elif options.stage == "random_screen":
        result = random_screen_report(options.require_complete)
    elif options.stage == "random_extension":
        result = random_extension_results(options.require_complete)
    elif options.stage == production.TWO_TASK_DIAGNOSTIC_STAGE:
        result = two_task_full_diagnostic_report(options.require_complete)
    else:
        result = full_primary_report(options.require_complete)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
