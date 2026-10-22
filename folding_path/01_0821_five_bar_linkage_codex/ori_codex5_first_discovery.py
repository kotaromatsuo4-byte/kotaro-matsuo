"""Budget-exact first-discovery benchmark for all ori_codex5 problems.

This harness compares a learning Actor--Critic, an identically initialised
but frozen neural sampler, and the existing uniform modal sampler.  It runs no
evaluation rollouts and has no episode cap: every unsuccessful trial is
administratively censored at exactly the same cumulative mechanics-step
budget.  Both the planar five-bar problem and the spatial torus/waterbomb
problems use the same scheduling, checkpoint, and aggregation code.

The ori5 reward is fixed by this benchmark.  Task and optimiser options such
as target spread, batch size, learning rates, horizon, and seed remain
configurable, but reward coefficients and tolerances cannot be overridden.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch

import collision_3d
import ori_codex3 as ori3
import ori_codex4_3d_first_discovery as legacy
import ori_codex5 as ori5
import ori_codex5_3d as ori5_3d


SCHEMA_VERSION = 2
HARNESS_VERSION = "2.0.0"
METHODS = ("learning", "frozen", "uniform")
PROBLEMS = ("five_bar", "4unit_torus", "waterbomb")
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_FIVE_BAR_DATA = _SCRIPT_DIRECTORY / "model" / "five_bar_linkage"

# The tested atomic I/O and exact-budget wrapper are deliberately reused.
BudgetedEnvironment = legacy.BudgetedEnvironment
_NoUpdateAgent = legacy._NoUpdateAgent
_atomic_json = legacy._atomic_json
_atomic_csv = legacy._atomic_csv
_canonical_digest = legacy._canonical_digest
_file_sha256 = legacy._file_sha256
_jsonable = legacy._jsonable

_LEGACY_HISTORY_ROW = legacy._history_row
_LEGACY_HISTORY_FIELDS = legacy._HISTORY_FIELDS


FIXED_REWARD_SPECIFICATION = {
    "reward_design": ori5.REWARD_DESIGN_NAME,
    "reward_schema_version": ori5.REWARD_SCHEMA_VERSION,
    "gamma": 1.0,
    "flatness_tolerance": 0.001,
    "spread_tolerance": 0.05,
    "flatness_clip": 2.0,
    "spread_milestone_initial_factor": 2.0,
    "spread_milestone_bonus": 0.9,
    "joint_flatness_thresholds": [0.5, 0.1, 0.01],
    "joint_milestone_bonuses": [0.2, 0.3, 0.5],
    "crossing_penalty": 1.25,
    "wrong_flat_penalty": 2.0,
    "timeout_penalty": 1.0,
    "success_bonus": 10.0,
    "quality_weight": 1.0,
    "success_speed_bonus_weight": 0.5,
    "near_miss_flatness_far": 1.0,
    "near_miss_spread_error_far": 1.0,
    "numerical_failure_penalty": 10.0,
    "no_mode_penalty": 5.0,
    "path_integrated_cost_weight": 0.0,
}
FIXED_REWARD_SHA256 = _canonical_digest(FIXED_REWARD_SPECIFICATION)

_FORBIDDEN_EXACT_REWARD_OPTIONS = {
    "--gamma",
    "--flatness-tolerance",
    "--spread-tolerance",
    "--spread-sigma",
    "--flat-quality-clip",
    "--flat-quality-weight",
    "--flat-signal-floor",
    "--compact-potential-weight",
    "--flatness-reward-clip",
    "--near-miss-bonus-weight",
    "--near-miss-flatness-far",
    "--near-miss-spread-error-far",
    "--crossing-penalty",
    "--numerical-failure-penalty",
    "--no-mode-penalty",
    "--timeout-penalty",
    "--quality-combination",
    "--flatness-progress-weight",
    "--target-progress-weight",
    "--target-deviation-cost-weight",
    "--flat-terminal-bonus",
    "--joint-success-bonus",
    "--wrong-flat-penalty",
    "--terminate-wrong-flat",
    "--training-flatness-tolerances",
    "--training-flatness-milestones",
    "--curriculum-waypoint-bonus",
    "--curriculum-waypoint-start-episode",
    "--gated-compact-cost-weight",
    "--compact-gate-scale",
    "--update-method",
    "--reset-critic-at-waypoint-start",
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _option_was_supplied(arguments, option):
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _reject_reward_overrides(arguments):
    for argument in arguments:
        option = str(argument).split("=", 1)[0]
        if option.startswith("--ori5-") or option in (
            _FORBIDDEN_EXACT_REWARD_OPTIONS
        ):
            raise ValueError(
                f"{option} is fixed by the ori5 first-discovery benchmark."
            )
        if option == "--episode-cap":
            raise ValueError(
                "ori5 first-discovery has no episode cap; use only the common "
                "--mechanics-budget censor."
            )


def _load_problem_data(data_directory):
    # The validated ori5 3-D loader is also valid for the five-bar's N x 3
    # coordinate representation; flatness dimension and collision backend are
    # selected independently below.
    return ori5_3d.load_problem_data(data_directory)


class _PlanarAdapter:
    load_problem_data = staticmethod(_load_problem_data)
    make_environment = staticmethod(ori5.make_environment)
    run_episode = staticmethod(ori5.run_episode)

    @staticmethod
    def patched_three_dimensional_runtime(args, member):
        del args, member
        return nullcontext(None)


_PLANAR_ADAPTER = _PlanarAdapter()


def _adapter_for_problem(problem):
    if problem == "five_bar":
        return _PLANAR_ADAPTER
    if problem in ("4unit_torus", "waterbomb"):
        return ori5_3d
    raise ValueError(f"Unknown benchmark problem: {problem}")


def _effective_reward_specification(args):
    parameters = ori5.reward_parameters(args)
    return {
        "reward_design": ori5.REWARD_DESIGN_NAME,
        "reward_schema_version": ori5.REWARD_SCHEMA_VERSION,
        "gamma": float(args.gamma),
        "flatness_tolerance": float(args.flatness_tolerance),
        "spread_tolerance": float(args.spread_tolerance),
        "flatness_clip": parameters["flatness_clip"],
        "spread_milestone_initial_factor": parameters[
            "spread_milestone_initial_factor"
        ],
        "spread_milestone_bonus": parameters["spread_milestone_bonus"],
        "joint_flatness_thresholds": parameters[
            "joint_flatness_thresholds"
        ],
        "joint_milestone_bonuses": parameters[
            "joint_milestone_bonuses"
        ],
        "crossing_penalty": parameters["crossing_penalty"],
        "wrong_flat_penalty": parameters["wrong_flat_penalty"],
        "timeout_penalty": parameters["timeout_penalty"],
        "success_bonus": parameters["success_bonus"],
        "quality_weight": parameters["quality_weight"],
        "success_speed_bonus_weight": parameters[
            "success_speed_bonus_weight"
        ],
        "near_miss_flatness_far": parameters["near_miss_flatness_far"],
        "near_miss_spread_error_far": parameters[
            "near_miss_spread_error_far"
        ],
        "numerical_failure_penalty": float(
            args.numerical_failure_penalty
        ),
        "no_mode_penalty": float(args.no_mode_penalty),
        "path_integrated_cost_weight": parameters[
            "path_integrated_cost_weight"
        ],
    }


def _normalise_and_validate(args):
    """Validate the fixed reward and the common learning protocol."""

    ori5._validate_args(args)
    problem = str(args.benchmark_problem)
    expected_dimension = 1 if problem == "five_bar" else 2
    if int(args.flatness_subspace_dimension) != expected_dimension:
        raise ValueError(
            f"{problem} requires flatness subspace dimension "
            f"{expected_dimension}."
        )
    if args.update_method != "actor_critic":
        raise ValueError("First-discovery compares the ori5 Actor--Critic.")
    if _canonical_digest(_effective_reward_specification(args)) != (
        FIXED_REWARD_SHA256
    ):
        raise ValueError(
            "Effective reward differs from the fixed ori5 benchmark reward."
        )
    if len(args.actor_lr_milestones) != len(args.actor_lr_factors):
        raise ValueError("actor_lr_milestones and factors must have equal length.")
    if list(args.actor_lr_milestones) != sorted(args.actor_lr_milestones):
        raise ValueError("actor_lr_milestones must be sorted.")
    if any(float(factor) <= 0.0 for factor in args.actor_lr_factors):
        raise ValueError("actor_lr_factors must be positive.")
    args.random_search = False
    args.reward_design = ori5.REWARD_DESIGN_NAME
    args.reward_schema_version = ori5.REWARD_SCHEMA_VERSION
    args.ori5_reward_parameters = ori5.reward_parameters(args)
    return args


_HISTORY_FIELDS = _LEGACY_HISTORY_FIELDS + (
    "reward_design",
    "reward_schema_version",
    "ori5_total_flatness_progress_reward",
    "ori5_total_target_progress_reward",
    "ori5_total_spread_milestone_reward",
    "ori5_total_joint_milestone_reward",
    "ori5_total_milestone_reward",
    "ori5_total_terminal_fixed_reward",
    "ori5_total_terminal_quality_reward",
    "ori5_total_terminal_speed_reward",
    "ori5_total_terminal_correction_reward",
    "ori5_reward_decomposition_error",
    "ori5_nonfinite_reward_component_count",
    "ori5_last_valid_flatness_ratio",
    "ori5_last_valid_spread",
)


def _history_row(method, seed, episode, summary, cumulative_steps,
                 cumulative_calls, episode_seconds, rollout_seconds, actions):
    row = _LEGACY_HISTORY_ROW(
        method, seed, episode, summary, cumulative_steps, cumulative_calls,
        episode_seconds, rollout_seconds, actions,
    )
    row.update({
        "reward_design": summary.get("reward_design"),
        "reward_schema_version": summary.get("reward_schema_version"),
        "ori5_total_flatness_progress_reward": summary.get(
            "ori5_total_flatness_progress_reward", 0.0
        ),
        "ori5_total_target_progress_reward": summary.get(
            "ori5_total_target_progress_reward", 0.0
        ),
        "ori5_total_spread_milestone_reward": summary.get(
            "ori5_total_spread_milestone_reward", 0.0
        ),
        "ori5_total_joint_milestone_reward": summary.get(
            "ori5_total_joint_milestone_reward", 0.0
        ),
        "ori5_total_milestone_reward": summary.get(
            "ori5_total_milestone_reward", 0.0
        ),
        "ori5_total_terminal_fixed_reward": summary.get(
            "ori5_total_terminal_fixed_reward", 0.0
        ),
        "ori5_total_terminal_quality_reward": summary.get(
            "ori5_total_terminal_quality_reward", 0.0
        ),
        "ori5_total_terminal_speed_reward": summary.get(
            "ori5_total_terminal_speed_reward", 0.0
        ),
        "ori5_total_terminal_correction_reward": summary.get(
            "ori5_total_terminal_correction_reward", 0.0
        ),
        "ori5_reward_decomposition_error": summary.get(
            "ori5_reward_decomposition_error", 0.0
        ),
        "ori5_nonfinite_reward_component_count": summary.get(
            "ori5_nonfinite_reward_component_count", 0
        ),
        "ori5_last_valid_flatness_ratio": summary.get(
            "ori5_last_valid_flatness_ratio"
        ),
        "ori5_last_valid_spread": summary.get("ori5_last_valid_spread"),
    })
    return row


@contextmanager
def _bound_legacy_runtime(selected_adapter):
    """Bind the tested scheduler to ori5 without changing legacy files."""

    original = (
        legacy.adapter,
        legacy._HISTORY_FIELDS,
        legacy._history_row,
    )
    legacy.adapter = selected_adapter
    legacy._HISTORY_FIELDS = _HISTORY_FIELDS
    legacy._history_row = _history_row
    try:
        yield
    finally:
        (
            legacy.adapter,
            legacy._HISTORY_FIELDS,
            legacy._history_row,
        ) = original


def _trial_paths(output_root, method, seed):
    directory = Path(output_root) / "trials" / method / f"seed{int(seed)}"
    return directory, directory / "summary.json", directory / "run_config.json"


def _read_history_audit(path):
    if not Path(path).is_file():
        return {
            "maximum_abs_reward_decomposition_error": 0.0,
            "nonfinite_reward_component_count": 0,
        }
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        "maximum_abs_reward_decomposition_error": float(max(
            [abs(float(row["ori5_reward_decomposition_error"]))
             for row in rows]
            or [0.0]
        )),
        "nonfinite_reward_component_count": int(sum(
            int(float(row["ori5_nonfinite_reward_component_count"]))
            for row in rows
        )),
    }


def run_trial(
    args,
    method,
    seed,
    mechanics_budget,
    output_root,
    root_fingerprint,
    resume=True,
    checkpoint_every_batches=1,
    progress_every=0,
):
    """Run one budget-exact ori5 method/seed trial with no episode cap."""

    selected_adapter = _adapter_for_problem(args.benchmark_problem)
    trial_directory, summary_path, config_path = _trial_paths(
        output_root, method, seed
    )
    preexisting = summary_path.is_file()
    if preexisting:
        with open(summary_path, encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("reward_constants_sha256") != FIXED_REWARD_SHA256:
            raise RuntimeError(
                f"Completed trial has missing/foreign reward identity: "
                f"{summary_path}"
            )

    with _bound_legacy_runtime(selected_adapter):
        result = legacy.run_trial(
            args=args,
            method=method,
            seed=seed,
            mechanics_budget=mechanics_budget,
            episode_cap=None,
            output_root=output_root,
            root_fingerprint=root_fingerprint,
            resume=resume,
            checkpoint_every_batches=checkpoint_every_batches,
            progress_every=progress_every,
        )

    audit = _read_history_audit(trial_directory / "episode_history.csv")
    result.update({
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "benchmark_problem": args.benchmark_problem,
        "spatial_dimension": int(args.benchmark_spatial_dimension),
        "reward_design": ori5.REWARD_DESIGN_NAME,
        "reward_schema_version": ori5.REWARD_SCHEMA_VERSION,
        "reward_constants_sha256": FIXED_REWARD_SHA256,
        "episode_cap": None,
        "episode_cap_supported": False,
        "budget_stop_reward": (
            "unchanged ori5 reward from the final ordinary valid transition"
        ),
        "reward_audit": audit,
    })
    _atomic_json(summary_path, result)

    with open(config_path, encoding="utf-8") as file:
        trial_config = json.load(file)
    trial_config.update({
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "benchmark_problem": args.benchmark_problem,
        "spatial_dimension": int(args.benchmark_spatial_dimension),
        "reward_design": ori5.REWARD_DESIGN_NAME,
        "reward_schema_version": ori5.REWARD_SCHEMA_VERSION,
        "reward_constants": FIXED_REWARD_SPECIFICATION,
        "reward_constants_sha256": FIXED_REWARD_SHA256,
        "episode_cap": None,
        "episode_cap_supported": False,
        "budget_stop_reward": result["budget_stop_reward"],
    })
    _atomic_json(config_path, trial_config)
    return result


def _verify_root_config(root_config):
    if int(root_config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("Unsupported or missing first-discovery schema.")
    identity = root_config.get("fingerprint_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Root configuration lacks fingerprint_identity.")
    recomputed = _canonical_digest(identity)
    if recomputed != root_config.get("root_fingerprint"):
        raise RuntimeError("Root fingerprint does not match its identity payload.")
    for key, identity_value in identity.items():
        if root_config.get(key) != identity_value:
            raise RuntimeError(
                f"Root configuration field {key!r} differs from its "
                "fingerprinted identity payload."
            )
    if root_config.get("effective_ori4_args") != identity.get(
        "effective_ori5_args"
    ):
        raise RuntimeError(
            "Legacy aggregate argument alias differs from fingerprinted "
            "ori5 arguments."
        )
    if root_config.get("reward_constants_sha256") != FIXED_REWARD_SHA256:
        raise RuntimeError("Root reward identity differs from fixed ori5 reward.")
    if root_config.get("reward_constants") != FIXED_REWARD_SPECIFICATION:
        raise RuntimeError("Root reward constants differ from fixed ori5 reward.")
    effective_args = identity.get("effective_ori5_args")
    if not isinstance(effective_args, dict):
        raise RuntimeError("Root identity lacks effective ori5 arguments.")
    try:
        effective_reward = _effective_reward_specification(
            argparse.Namespace(**effective_args)
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Root effective arguments cannot reconstruct the reward identity."
        ) from error
    if _canonical_digest(effective_reward) != FIXED_REWARD_SHA256:
        raise RuntimeError(
            "Root effective arguments differ from the fixed ori5 reward."
        )
    problem = root_config.get("benchmark_problem")
    if problem not in PROBLEMS:
        raise RuntimeError(f"Unknown benchmark problem in root: {problem!r}")
    expected_dimension = 2 if problem == "five_bar" else 3
    expected_flatness_dimension = 1 if problem == "five_bar" else 2
    if int(root_config.get("spatial_dimension", -1)) != expected_dimension:
        raise RuntimeError("Root problem/spatial-dimension mismatch.")
    if int(effective_args.get("flatness_subspace_dimension", -1)) != (
        expected_flatness_dimension
    ):
        raise RuntimeError("Root problem/flatness-dimension mismatch.")
    if root_config.get("episode_cap") is not None:
        raise RuntimeError("Episode-capped ori5 root configuration is forbidden.")
    if int(root_config.get("mechanics_budget", 0)) < 1:
        raise RuntimeError("Root mechanics budget must be positive.")
    methods = list(root_config.get("methods", []))
    seeds = [int(seed) for seed in root_config.get("seeds", [])]
    if not methods or any(method not in METHODS for method in methods):
        raise RuntimeError("Root methods must be a non-empty ori5 method list.")
    if not seeds:
        raise RuntimeError("Root seeds must be non-empty.")
    if len(methods) != len(set(methods)):
        raise RuntimeError("Root configuration contains duplicate methods.")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Root configuration contains duplicate seeds.")
    return methods, seeds


def _validate_trial_tree(output_root, root_config, require_complete):
    methods, seeds = _verify_root_config(root_config)
    expected = {(method, seed) for seed in seeds for method in methods}
    seen = {}
    trial_root = Path(output_root) / "trials"
    for path in sorted(trial_root.glob("**/summary.json")):
        relative = path.relative_to(trial_root)
        if len(relative.parts) != 3 or not relative.parts[1].startswith("seed"):
            raise RuntimeError(f"Unexpected trial summary location: {path}")
        path_method = relative.parts[0]
        try:
            path_seed = int(relative.parts[1][4:])
        except ValueError as error:
            raise RuntimeError(f"Invalid trial seed directory: {path}") from error
        with open(path, encoding="utf-8") as file:
            row = json.load(file)
        if not row.get("complete"):
            raise RuntimeError(f"Non-atomic incomplete trial summary: {path}")
        content_pair = (str(row.get("method")), int(row.get("seed", -1)))
        path_pair = (path_method, path_seed)
        if content_pair != path_pair:
            raise RuntimeError(
                f"Trial path/content identity mismatch: {path_pair} != "
                f"{content_pair}"
            )
        if content_pair in seen:
            raise RuntimeError(
                f"Duplicate completed trial {content_pair}: {seen[content_pair]} "
                f"and {path}"
            )
        if content_pair not in expected:
            raise RuntimeError(f"Unexpected completed trial: {content_pair}")
        if row.get("root_fingerprint") != root_config["root_fingerprint"]:
            raise RuntimeError(f"Trial fingerprint mismatch: {path}")
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(f"Trial schema mismatch: {path}")
        if row.get("benchmark_problem") != root_config["benchmark_problem"]:
            raise RuntimeError(f"Trial problem mismatch: {path}")
        if int(row.get("spatial_dimension", -1)) != int(
            root_config["spatial_dimension"]
        ):
            raise RuntimeError(f"Trial dimension mismatch: {path}")
        if row.get("reward_constants_sha256") != FIXED_REWARD_SHA256:
            raise RuntimeError(f"Trial reward identity mismatch: {path}")
        if row.get("episode_cap") is not None:
            raise RuntimeError(f"Episode-capped trial is forbidden: {path}")
        budget = int(root_config["mechanics_budget"])
        if int(row.get("mechanics_budget", -1)) != budget:
            raise RuntimeError(f"Trial mechanics budget mismatch: {path}")
        steps = int(row.get("mechanical_steps", -1))
        if steps < 1 or steps > budget:
            raise RuntimeError(f"Invalid mechanics count in {path}")
        if not row.get("success") and steps != budget:
            raise RuntimeError(
                f"Censored trial did not consume exact mechanics budget: {path}"
            )
        if not row.get("success") and row.get("censor_reason") != (
            "mechanics_budget"
        ):
            raise RuntimeError(f"Invalid censor reason in {path}")
        if bool(row.get("success")) == bool(row.get("censored")):
            raise RuntimeError(f"Inconsistent success/censor flags in {path}")
        audit = row.get("reward_audit")
        if not isinstance(audit, dict):
            raise RuntimeError(f"Missing reward audit in {path}")
        try:
            decomposition_error = float(
                audit["maximum_abs_reward_decomposition_error"]
            )
            nonfinite_count = int(audit["nonfinite_reward_component_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid reward audit in {path}") from error
        if not np.isfinite(decomposition_error) or decomposition_error < 0.0:
            raise RuntimeError(f"Invalid decomposition audit in {path}")
        if nonfinite_count < 0:
            raise RuntimeError(f"Invalid nonfinite audit in {path}")
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Missing trial run_config: {config_path}")
        with open(config_path, encoding="utf-8") as file:
            trial_config = json.load(file)
        for key, expected_value in (
            ("root_fingerprint", root_config["root_fingerprint"]),
            ("method", content_pair[0]),
            ("seed", content_pair[1]),
            ("mechanics_budget", budget),
            ("reward_constants_sha256", FIXED_REWARD_SHA256),
            ("benchmark_problem", root_config["benchmark_problem"]),
            ("spatial_dimension", root_config["spatial_dimension"]),
            ("episode_cap", None),
        ):
            if trial_config.get(key) != expected_value:
                raise RuntimeError(
                    f"Trial run_config mismatch for {key}: {config_path}"
                )
        seen[content_pair] = path
    missing = sorted(expected - set(seen))
    if require_complete and missing:
        raise RuntimeError(
            "Incomplete benchmark; missing method/seed pairs: "
            + ", ".join(f"{method}:{seed}" for method, seed in missing)
        )
    return expected, seen, missing


def aggregate_results(
    output_root, mechanics_budget=None, require_complete=False
):
    """Strictly validate and aggregate every visible method/seed trial."""

    output_root = Path(output_root)
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    with open(config_path, encoding="utf-8") as file:
        root_config = json.load(file)
    expected, seen, missing = _validate_trial_tree(
        output_root, root_config, require_complete
    )
    expected_budget = int(root_config["mechanics_budget"])
    if mechanics_budget is not None and int(mechanics_budget) != expected_budget:
        raise ValueError(
            f"Requested budget {mechanics_budget} does not match run budget "
            f"{expected_budget}."
        )
    aggregate = legacy.aggregate_results(
        output_root,
        mechanics_budget=expected_budget,
        episode_cap=None,
        require_complete=require_complete,
    )
    reward_audits = []
    for path in seen.values():
        with open(path, encoding="utf-8") as file:
            reward_audits.append(json.load(file)["reward_audit"])
    aggregate.update({
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "benchmark_problem": root_config["benchmark_problem"],
        "spatial_dimension": root_config["spatial_dimension"],
        "root_fingerprint": root_config["root_fingerprint"],
        "reward_design": ori5.REWARD_DESIGN_NAME,
        "reward_schema_version": ori5.REWARD_SCHEMA_VERSION,
        "reward_constants_sha256": FIXED_REWARD_SHA256,
        "episode_cap": None,
        "episode_cap_supported": False,
        "expected_trials": len(expected),
        "completed_trials": len(seen),
        "complete": not missing,
        "missing_method_seed_pairs": [
            {"method": method, "seed": seed} for method, seed in missing
        ],
        "reward_audit": {
            "maximum_abs_reward_decomposition_error": float(max(
                [float(row["maximum_abs_reward_decomposition_error"])
                 for row in reward_audits]
                or [0.0]
            )),
            "nonfinite_reward_component_count": int(sum(
                int(row["nonfinite_reward_component_count"])
                for row in reward_audits
            )),
        },
        "aggregate_validation": {
            "root_fingerprint_recomputed": True,
            "path_content_identity_checked": True,
            "duplicate_pairs_rejected": True,
            "trial_run_configs_checked": True,
            "censored_trials_exact_budget_checked": True,
        },
    })
    _atomic_json(output_root / "aggregate.json", aggregate)
    return aggregate


def _parse_harness_args(argv):
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Budget-only first discovery for ori_codex5.",
    )
    parser.add_argument("--problem", choices=PROBLEMS, default="five_bar")
    parser.add_argument("--mechanics-budget", type=int, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--trial-seed-start", type=int, default=None)
    parser.add_argument("--trial-count", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--benchmark-print-every", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("-h", "--help", action="store_true", dest="harness_help")
    options, remaining = parser.parse_known_args(argv)
    if options.harness_help:
        parser.print_help()
        print("\nRemaining non-reward options are forwarded to ori_codex5.")
        raise SystemExit(0)
    return options, remaining


def _source_provenance():
    names = (
        "ori_codex2.py",
        "ori_codex3.py",
        "ori_codex4.py",
        "ori_codex4_3d_first_discovery.py",
        "ori_codex5.py",
        "ori_codex5_3d.py",
        "collision_3d.py",
        Path(__file__).name,
    )
    return {
        name: _file_sha256(_SCRIPT_DIRECTORY / name)
        for name in names if (_SCRIPT_DIRECTORY / name).is_file()
    }


def _parse_core_args(problem, core_argv):
    if problem == "five_bar":
        args = ori5.parse_args([
            "--data-dir", str(_FIVE_BAR_DATA),
            "--flatness-subspace-dimension", "1",
            "--output-dir", "__unused_by_first_discovery__",
            *core_argv,
        ])
        args.rigid_motion_diagnostics = False
        spatial_dimension = 2
    else:
        args = ori5_3d.parse_args([
            "--model", problem,
            "--output-dir", "__unused_by_first_discovery__",
            *core_argv,
        ])
        spatial_dimension = 3
    args.benchmark_problem = problem
    args.benchmark_spatial_dimension = spatial_dimension
    return _normalise_and_validate(args)


def _prepare_run(argv=None):
    raw_arguments = sys.argv[1:] if argv is None else list(argv)
    if _option_was_supplied(raw_arguments, "--episode-cap"):
        raise ValueError(
            "ori5 first-discovery has no episode cap; use --mechanics-budget."
        )
    harness, core_argv = _parse_harness_args(raw_arguments)
    _reject_reward_overrides(core_argv)
    if harness.aggregate_only:
        if harness.output_root is None:
            raise ValueError("--aggregate-only requires --output-root.")
        output_root = Path(harness.output_root)
        if not output_root.is_absolute():
            output_root = _SCRIPT_DIRECTORY / output_root
        return harness, None, output_root, [], None
    if harness.mechanics_budget is None or harness.mechanics_budget < 1:
        raise ValueError("A positive --mechanics-budget is required.")
    if harness.trial_count < 1:
        raise ValueError("trial_count must be positive.")
    if harness.shard_count < 1 or not 0 <= harness.shard_index < (
        harness.shard_count
    ):
        raise ValueError("Require 0 <= shard-index < shard-count.")
    if harness.checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive.")
    if len(harness.methods) != len(set(harness.methods)):
        raise ValueError("Duplicate methods are not allowed.")

    core_args = _parse_core_args(harness.problem, core_argv)
    if harness.trial_seeds is not None:
        seeds = [int(seed) for seed in harness.trial_seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate trial seeds are not allowed.")
        if harness.trial_seed_start is not None:
            raise ValueError(
                "Do not combine --trial-seeds and --trial-seed-start."
            )
    else:
        first_seed = (
            int(core_args.seed)
            if harness.trial_seed_start is None
            else int(harness.trial_seed_start)
        )
        seeds = list(range(first_seed, first_seed + int(harness.trial_count)))

    selected_adapter = _adapter_for_problem(harness.problem)
    node, member, fix, files = selected_adapter.load_problem_data(
        core_args.data_dir
    )
    model_hashes = {name: _file_sha256(path) for name, path in files.items()}
    effective_args = {
        key: _jsonable(value) for key, value in vars(core_args).items()
        if key not in ("output_dir", "three_dimensional_adapter")
    }
    collision_identity = (
        {
            "backend": "single_solution_common.planar_proper_crossing",
            "dimension": 2,
        }
        if harness.problem == "five_bar"
        else {
            "backend": collision_3d.COLLISION_BACKEND_NAME,
            "version": collision_3d.COLLISION_BACKEND_VERSION,
            "dimension": 3,
        }
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "benchmark_problem": harness.problem,
        "spatial_dimension": int(core_args.benchmark_spatial_dimension),
        "effective_ori5_args": effective_args,
        "reward_constants": FIXED_REWARD_SPECIFICATION,
        "reward_constants_sha256": FIXED_REWARD_SHA256,
        "methods": list(harness.methods),
        "seeds": seeds,
        "mechanics_budget": int(harness.mechanics_budget),
        "episode_cap": None,
        "model_file_sha256": model_hashes,
        "source_sha256": _source_provenance(),
        "collision_identity": collision_identity,
    }
    root_fingerprint = _canonical_digest(identity)
    if harness.output_root is None:
        slug = (
            f"{harness.problem}_spread{float(core_args.target_spread):.8g}_"
            f"budget{int(harness.mechanics_budget)}_"
            f"{root_fingerprint[:12]}"
        ).replace(".", "p")
        output_root = _SCRIPT_DIRECTORY / "ori5_first_discovery_results" / slug
    else:
        output_root = Path(harness.output_root)
        if not output_root.is_absolute():
            output_root = _SCRIPT_DIRECTORY / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        **identity,
        # Alias retained only for the tested legacy aggregate implementation.
        "effective_ori4_args": effective_args,
        "fingerprint_identity": identity,
        "root_fingerprint": root_fingerprint,
        "created_utc": _utc_now(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "episode_cap_supported": False,
        "comparison_rules": {
            "strict_success": "strict_compact_success == 1",
            "stop": "first strict success or exact mechanics budget",
            "evaluation_rollouts": 0,
            "learning_update": (
                "one Actor--Critic update after each complete batch that "
                "finishes before discovery/censoring"
            ),
            "final_partial_batch_update": False,
            "administratively_truncated_episode_update": False,
            "frozen_actor": "same seed/model/action law, no updates",
            "uniform_random": (
                "isotropic current-mode direction and uniform signed width"
            ),
            "episode_cap": "forbidden",
        },
        "node_count": int(len(node)),
        "member_count": int(len(member)),
        "fixed_scalar_dof_count": int(len(fix)),
    }
    run_config_path = output_root / "run_config.json"
    if run_config_path.is_file():
        with open(run_config_path, encoding="utf-8") as file:
            existing = json.load(file)
        _verify_root_config(existing)
        if existing["root_fingerprint"] != root_fingerprint:
            raise RuntimeError(
                f"Output root contains a different experiment: {output_root}"
            )
    else:
        _atomic_json(run_config_path, run_config)

    tasks = [
        (method, seed) for seed in seeds for method in harness.methods
    ]
    tasks = [
        task for index, task in enumerate(tasks)
        if index % harness.shard_count == harness.shard_index
    ]
    return harness, core_args, output_root, tasks, root_fingerprint


def main(argv=None):
    torch.set_num_threads(1)
    harness, args, output_root, tasks, fingerprint = _prepare_run(argv)
    if harness.aggregate_only:
        aggregate = aggregate_results(output_root, require_complete=True)
        print(json.dumps(aggregate, indent=2))
        return aggregate

    for method, seed in tasks:
        result = run_trial(
            args=args,
            method=method,
            seed=seed,
            mechanics_budget=harness.mechanics_budget,
            output_root=output_root,
            root_fingerprint=fingerprint,
            resume=harness.resume,
            checkpoint_every_batches=harness.checkpoint_every_batches,
            progress_every=harness.benchmark_print_every,
        )
        print(
            f"{method} seed={seed}: success={bool(result['success'])} "
            f"episodes={result['episodes']} steps={result['mechanical_steps']} "
            f"projector_calls={result['projector_calls']}",
            flush=True,
        )
    aggregate = aggregate_results(
        output_root,
        mechanics_budget=harness.mechanics_budget,
        require_complete=(harness.shard_count == 1),
    )
    print(json.dumps(aggregate, indent=2))
    print(f"Results: {output_root}")
    return aggregate


if __name__ == "__main__":
    main()
