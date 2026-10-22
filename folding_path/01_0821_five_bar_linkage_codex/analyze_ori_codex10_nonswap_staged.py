"""Checkpoint-safe analysis for the Codex6 non-swap hyperparameter screen.

The runner passes manifest-bound job records to this module.  Only the exact
CSV prefix ending at an atomic checkpoint is used; any child-process overrun
is deliberately ignored.  Discovery is measured in mechanical proposals
(cumulative ``steps``), rather than episodes, because rejected-and-continued
episodes can be much longer than the old terminate-on-crossing episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SEEDS = (9101, 9102, 9103)
TASK_ORDER = ("five_d2", "five_d3", "watersqrt2")
SCREEN_CHECKPOINTS = (512, 1024, 2048, 4096)
PRODUCTION_CHECKPOINTS = (5120, 6144, 7168, 8192, 9216, 10000)
PRODUCTION_READY_SEED_TAIL512 = 0.20
PRODUCTION_READY_SEED_FIXED = 0.20
PRODUCTION_READY_MEDIAN_TAIL512 = 0.50
PRODUCTION_MEDIAN_TAIL1024 = 0.70
PRODUCTION_MIN_SEED_TAIL1024 = 0.50
PRODUCTION_MEDIAN_FIXED = 0.60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(row: dict, names: Sequence[str], default=math.nan) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return float(default)


def _int(row: dict, names: Sequence[str], default=0) -> int:
    value = _float(row, names, default)
    if not math.isfinite(value):
        return int(default)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"non-integral value {value!r} in {tuple(names)!r}")
    return int(rounded)


def _required_int(row: dict, name: str) -> int:
    raw = row.get(name)
    if raw in (None, ""):
        raise ValueError(f"missing integral field {name}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric value {raw!r} in {name}") from error
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"non-integral value {raw!r} in {name}")
    return int(value)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else math.nan


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else math.nan


def _same(expected, observed) -> bool:
    if isinstance(expected, list):
        return isinstance(observed, list) and len(expected) == len(observed) and all(
            _same(a, b) for a, b in zip(expected, observed)
        )
    if isinstance(expected, bool):
        return isinstance(observed, bool) and expected is observed
    if isinstance(expected, int):
        return isinstance(observed, int) and not isinstance(observed, bool) and expected == observed
    if isinstance(expected, float):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            return False
        try:
            value = float(observed)
            return math.isfinite(value) and math.isclose(
                expected, value, rel_tol=1e-12, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return type(expected) is type(observed) and expected == observed


def read_prefix(path: Path, endpoint: int) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], [f"missing {path.name}"]
    rows: dict[int, dict] = {}
    errors: list[str] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                episode = _int(row, ("episode",), -1)
                if 1 <= episode <= endpoint:
                    if episode in rows:
                        errors.append(f"duplicate episode {episode}")
                    rows[episode] = row
    except (OSError, csv.Error, ValueError) as error:
        return [], [f"read failed: {error}"]
    missing = sorted(set(range(1, endpoint + 1)) - set(rows))
    if missing:
        errors.append(
            f"missing episode prefix {missing[:5]}"
            + (f" ({len(missing)} total)" if len(missing) > 5 else "")
        )
    return [rows[key] for key in sorted(rows)], errors


def read_updates(
    path: Path, endpoint: int, expected_config: dict,
) -> tuple[list[dict], list[str]]:
    rows, errors = read_prefix_sparse(path, endpoint)
    batch_size = int(expected_config["batch_size"])
    expected = set(range(batch_size, endpoint + 1, batch_size))
    if endpoint == int(expected_config["episodes"]):
        expected.add(endpoint)
    try:
        observed = {_int(row, ("episode",), -1) for row in rows}
    except ValueError as error:
        return [], errors + [f"invalid update episode: {error}"]
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        errors.append(f"missing update episodes {missing[:5]}")
    if unexpected:
        errors.append(f"unexpected update episodes {unexpected[:5]}")
    initial_actor = float(expected_config["actor_learning_rate"])
    critic = float(expected_config["critic_learning_rate"])
    milestones = expected_config["actor_lr_milestones"]
    factors = expected_config["actor_lr_factors"]
    for row in rows:
        episode = _int(row, ("episode",), -1)
        expected_actor = initial_actor
        for milestone, factor in zip(milestones, factors):
            if episode > int(milestone):
                expected_actor = initial_actor * float(factor)
        observed_actor = _float(row, ("actor_learning_rate",))
        observed_critic = _float(row, ("critic_learning_rate",))
        if not _same(expected_actor, observed_actor):
            errors.append(
                f"actor LR at {episode}: {observed_actor!r} != {expected_actor!r}"
            )
        if not _same(critic, observed_critic):
            errors.append(f"critic LR at {episode}: {observed_critic!r} != {critic!r}")
    return rows, errors


def read_prefix_sparse(path: Path, endpoint: int) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], [f"missing {path.name}"]
    result: list[dict] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            result = [
                row for row in csv.DictReader(handle)
                if 1 <= _int(row, ("episode",), -1) <= endpoint
            ]
    except (OSError, csv.Error, ValueError) as error:
        return [], [f"read failed: {error}"]
    episodes = [_int(row, ("episode",), -1) for row in result]
    errors = ["duplicate sparse episodes"] if len(episodes) != len(set(episodes)) else []
    return result, errors


def validate_config(path: Path, expected: dict) -> list[str]:
    if not path.is_file():
        return ["missing run_config.json"]
    try:
        with open(path, encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [f"config read failed: {error}"]
    errors = []
    for key, value in expected.items():
        if not _same(value, observed.get(key)):
            errors.append(f"config {key}: {observed.get(key)!r} != {value!r}")
    if "output_dir" in expected and not _same(
        expected["output_dir"], observed.get("output_directory")
    ):
        errors.append(
            "config output_directory: "
            f"{observed.get('output_directory')!r} != {expected['output_dir']!r}"
        )
    expected_architecture = (
        "variable_mode_ori6_continuous_spread_state_potential_rollback_"
        "actor_critic"
    )
    for key in ("architecture_name", "architecture"):
        if observed.get(key) != expected_architecture:
            errors.append(
                f"config {key}: {observed.get(key)!r} != {expected_architecture!r}"
            )
    # The frozen spectral trainer writes these identities inside the inherited
    # ``ori5_reward_parameters`` object before training starts.  Top-level
    # candidate names are added only after a natural train() return, so an
    # atomic 512 checkpoint must be validated against the nested authority.
    parameters = observed.get("ori6_reward_parameters")
    if not isinstance(parameters, dict):
        parameters = observed.get("ori5_reward_parameters")
    if not isinstance(parameters, dict):
        errors.append("missing nested effective reward parameters")
        parameters = {}
    transition = parameters.get("crossing_transition", {})
    spectral = parameters.get("spectral_subspace", {})
    frontier = parameters.get("spread_frontier", {})
    nested_expected = {
        "reward_design": (
            observed.get("reward_design"),
            "ori6_continuous_spread_state_potential_rollback",
        ),
        "candidate": (
            parameters.get("candidate"),
            "ori6_RC_spectral_subspace_potential",
        ),
        "parent_candidate": (
            parameters.get("parent_candidate"),
            "ori6_P0X2_rejected_crossing_continuation",
        ),
        "crossing_policy": (
            transition.get("policy"), "reject_candidate_and_continue"
        ),
        "candidate_committed": (transition.get("candidate_committed"), False),
        "ordinary_reject_formula": (
            transition.get("ordinary_rejected_reward_formula"),
            "-crossing_penalty + quality_weight * q(last_safe)",
        ),
        "horizon_reject_formula": (
            transition.get("horizon_rejected_reward_formula"),
            "-crossing_penalty + quality_weight * q(last_safe) - timeout_penalty",
        ),
        "reject_step_count_increments": (
            transition.get("step_count_increments"), True
        ),
        "rejected_endpoint_reward": (
            transition.get("rejected_candidate_endpoint_reward"), 0.0
        ),
        "rejected_frontier_reward": (
            transition.get("rejected_candidate_spread_frontier_reward"), 0.0
        ),
        "rejected_milestone_reward": (
            transition.get("rejected_candidate_new_milestone_reward"), 0.0
        ),
        "crossing_penalty": (parameters.get("crossing_penalty"), 2.0),
        "quality_weight": (parameters.get("quality_weight"), 1.0),
        "timeout_penalty": (parameters.get("timeout_penalty"), 1.0),
        "spectral_weight": (spectral.get("weight"), 0.05),
        "spectral_relative_epsilon": (spectral.get("relative_epsilon"), 1e-12),
        "spectral_accepted_valid_only": (
            spectral.get("accepted_valid_transitions_only"), True
        ),
        "spectral_formula_sha256": (
            spectral.get("formula_sha256"),
            "552ae8ad0d3ff5cc6d9499b84bb44351a24e7b0eb61113a0e429645dc2fa06c6",
        ),
        "frontier_mode": (frontier.get("mode"), "state_potential"),
        "frontier_weight": (frontier.get("maximum_bonus"), 0.9),
        "frontier_flatness_gate_cap": (frontier.get("flatness_gate_cap"), 2.0),
    }
    for label, (actual, required) in nested_expected.items():
        if not _same(required, actual):
            errors.append(f"nested {label}: {actual!r} != {required!r}")
    for key, required in (
        ("ori6_candidate", "ori6_RC_spectral_subspace_potential"),
        ("ori6_parent_candidate", "ori6_P0X2_rejected_crossing_continuation"),
    ):
        # Natural completions may add these redundant top-level fields.  If
        # present, they must agree; absence at an intermediate checkpoint is
        # the frozen trainer's expected behavior.
        if key in observed and observed[key] != required:
            errors.append(f"top-level {key}: {observed[key]!r} != {required!r}")
    return errors


def _integrity_errors(rows: Sequence[dict], max_steps: int) -> list[str]:
    errors: list[str] = []
    required = (
        "success",
        "strict_compact_success",
        "steps",
        "self_crossing",
        "correction_failed",
        "no_modes",
        "ori6_rejected_crossing_count",
        "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
        "ori6_spectral_total_reward",
        "ori6_spectral_reward_decomposition_error",
        "ori6_candidate",
        "ori6_crossing_policy",
        "ori6_spectral_candidate",
        "ori6_spectral_parent_candidate",
        "ori6_spectral_weight",
        "ori5_reward_decomposition_error",
        "ori5_maximum_abs_step_reward_decomposition_error",
        "ori5_nonfinite_reward_component_count",
        "ori6_reward_decomposition_error",
        "ori6_maximum_abs_step_reward_decomposition_error",
        "ori6_nonfinite_reward_component_count",
        "ori6_spectral_maximum_abs_step_reward_decomposition_error",
        "ori6_spectral_nonfinite_component_count",
    )
    if rows:
        for name in required:
            if name not in rows[0]:
                errors.append(f"missing integrity field {name}")
    decomposition = 0.0
    nonfinite = 0
    for row_index, row in enumerate(rows, start=1):
        episode = _int(row, ("episode",), row_index)
        for name in required:
            if row.get(name) in (None, "") and len(errors) < 50:
                errors.append(f"episode {episode}: missing {name}")
        exact_values = {}
        for name, lower, upper in (
            ("success", 0, 1),
            ("strict_compact_success", 0, 1),
            ("self_crossing", 0, 1),
            ("correction_failed", 0, 1),
            ("no_modes", 0, 1),
            ("steps", 1, max_steps),
            ("ori6_rejected_crossing_count", 0, max_steps),
        ):
            if row.get(name) in (None, ""):
                if len(errors) < 50:
                    errors.append(f"episode {episode}: missing {name}")
                continue
            try:
                value = _required_int(row, name)
            except ValueError as error:
                if len(errors) < 50:
                    errors.append(f"episode {episode}: {error}")
                continue
            exact_values[name] = value
            if not lower <= value <= upper and len(errors) < 50:
                errors.append(
                    f"episode {episode}: {name}={value} outside [{lower},{upper}]"
                )
        if (
            "success" in exact_values and "strict_compact_success" in exact_values
            and exact_values["success"] != exact_values["strict_compact_success"]
            and len(errors) < 50
        ):
            errors.append(f"episode {episode}: success != strict_compact_success")
        if (
            "steps" in exact_values
            and "ori6_rejected_crossing_count" in exact_values
            and exact_values["ori6_rejected_crossing_count"] > exact_values["steps"]
            and len(errors) < 50
        ):
            errors.append(f"episode {episode}: rejected crossings exceed steps")
        identities = {
            "ori6_candidate": "ori6_P0X2_rejected_crossing_continuation",
            "ori6_crossing_policy": "reject_candidate_and_continue",
            "ori6_spectral_candidate": "ori6_RC_spectral_subspace_potential",
            "ori6_spectral_parent_candidate": "ori6_P0X2_rejected_crossing_continuation",
        }
        for name, required_value in identities.items():
            if row.get(name) != required_value and len(errors) < 50:
                errors.append(
                    f"episode {episode}: {name}={row.get(name)!r} != {required_value!r}"
                )
        spectral_weight = _float(row, ("ori6_spectral_weight",))
        if (
            not math.isfinite(spectral_weight)
            or not math.isclose(spectral_weight, 0.05, rel_tol=1e-12, abs_tol=1e-12)
        ) and len(errors) < 50:
            errors.append(f"episode {episode}: invalid spectral weight {spectral_weight!r}")
        for name in (
            "ori6_spectral_total_reward",
            "ori6_maximum_abs_rejected_crossing_reward_decomposition_error",
            "ori6_spectral_reward_decomposition_error",
        ):
            numeric = _float(row, (name,))
            if not math.isfinite(numeric) and len(errors) < 50:
                errors.append(f"episode {episode}: nonfinite or missing {name}")
        for name, value in row.items():
            if name.endswith("reward_decomposition_error") or name.endswith(
                "abs_step_reward_decomposition_error"
            ):
                numeric = _float(row, (name,))
                if not math.isfinite(numeric):
                    if len(errors) < 50:
                        errors.append(f"episode {episode}: nonfinite or missing {name}")
                else:
                    decomposition = max(decomposition, abs(numeric))
            if "nonfinite" in name and "count" in name:
                try:
                    count = _required_int(row, name)
                except ValueError as error:
                    count = 0
                    if len(errors) < 50:
                        errors.append(f"episode {episode}: {error}")
                if row.get(name) in (None, ""):
                    count = 0
                elif count < 0 and len(errors) < 50:
                    errors.append(f"episode {episode}: negative {name}")
                nonfinite += max(count, 0)
    if decomposition > 1e-8:
        errors.append(f"reward decomposition {decomposition:.3g} > 1e-8")
    if nonfinite:
        errors.append(f"nonfinite reward count {nonfinite}")
    return errors


def _fixed_evaluation_errors(row: dict | None, endpoint: int) -> list[str]:
    if row is None:
        return [f"missing fixed stochastic evaluation at {endpoint}"]
    errors = []
    try:
        episodes = _int(row, ("evaluation_episodes",), -1)
    except ValueError as error:
        episodes = -1
        errors.append(f"fixed evaluation episodes invalid: {error}")
    if episodes != 16:
        errors.append(f"fixed evaluation episodes {episodes} != 16")
    if row.get("strict_compact_success_rate") in (None, ""):
        errors.append("missing fixed strict_compact_success_rate")
    rate = _float(row, ("strict_compact_success_rate",))
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        errors.append(f"fixed strict rate outside [0,1]: {rate!r}")
    elif episodes == 16 and not math.isclose(
        rate * episodes, round(rate * episodes), rel_tol=0.0, abs_tol=1e-9
    ):
        errors.append(f"fixed strict rate {rate!r} is not a 16-trial proportion")
    return errors


def seed_report(job: dict, endpoint: int) -> dict:
    output = Path(job["output_directory"])
    rows, errors = read_prefix(output / "training_history.csv", endpoint)
    batch_size = int(job["expected_config"]["batch_size"])
    updates, update_errors = read_updates(
        output / "update_history.csv", endpoint, job["expected_config"]
    )
    evaluations, evaluation_errors = read_prefix_sparse(
        output / "stochastic_evaluation_history.csv", endpoint
    )
    errors.extend(update_errors)
    errors.extend(evaluation_errors)
    errors.extend(validate_config(output / "run_config.json", job["expected_config"]))
    errors.extend(_integrity_errors(rows, int(job["expected_config"]["max_steps"])))

    cumulative = 0
    first_episode = None
    first_proposals = None
    strict_flags: list[int] = []
    rejects = 0
    invalids = 0
    rewards: list[float] = []
    for row in rows:
        cumulative += _int(row, ("steps",), 0)
        # ``strict_compact_success`` is the frozen scientific endpoint;
        # ``success`` is checked for equality but never used as a fallback.
        strict = int(_int(row, ("strict_compact_success",), 0) != 0)
        strict_flags.append(strict)
        if strict and first_episode is None:
            first_episode = _int(row, ("episode",), 0)
            first_proposals = cumulative
        rejects += _int(row, ("ori6_rejected_crossing_count",), 0)
        invalids += int(any(
            _int(row, (name,), 0) != 0
            for name in ("self_crossing", "correction_failed", "no_modes")
        ))
        rewards.append(_float(row, ("total_reward",)))

    window = min(512, len(strict_flags))
    tail = strict_flags[-window:] if window else []
    window1024 = min(1024, len(strict_flags))
    tail1024 = strict_flags[-window1024:] if window1024 else []
    tail256 = strict_flags[-min(256, len(strict_flags)):] if strict_flags else []
    fixed = next(
        (row for row in evaluations if _int(row, ("episode",), -1) == endpoint),
        None,
    )
    errors.extend(_fixed_evaluation_errors(fixed, endpoint))
    fixed_rate = _float(fixed or {}, ("strict_compact_success_rate",), 0.0)
    actor_lr = _float(updates[-1], ("actor_learning_rate",)) if updates else math.nan
    critic_lr = _float(updates[-1], ("critic_learning_rate",)) if updates else math.nan
    return {
        "task": job["task"]["name"],
        "arm": job["arm"],
        "seed": int(job["seed"]),
        "endpoint": endpoint,
        "integrity_pass": not errors,
        "integrity_errors": errors,
        "first_hit_episode": first_episode,
        "first_hit_proposals": first_proposals,
        "first_hit_before_or_at_first_update": bool(
            first_episode is not None and first_episode <= batch_size
        ),
        "proposal_censor": cumulative,
        "strict_hits": sum(strict_flags),
        "tail_window": window,
        "tail_strict_rate": sum(tail) / max(len(tail), 1),
        "tail512_strict_rate": sum(tail) / max(len(tail), 1),
        "tail1024_window": window1024,
        "tail1024_strict_rate": sum(tail1024) / max(len(tail1024), 1),
        "tail256_strict_rate": sum(tail256) / max(len(tail256), 1),
        "fixed_stochastic_strict_rate": fixed_rate,
        "total_rejected_actions": rejects,
        "invalid_episode_count": invalids,
        "mean_tail_return": _mean(rewards[-window:]),
        "update_count": len(updates),
        "last_actor_learning_rate": actor_lr,
        "last_critic_learning_rate": critic_lr,
    }


def _arm_summary(reports: Sequence[dict], common_cap: int) -> dict:
    times = [
        min(
            int(report["first_hit_proposals"])
            if report["first_hit_proposals"] is not None else common_cap,
            common_cap,
        )
        for report in reports
    ]
    stable = [
        report for report in reports
        if report["strict_hits"] > 0
        and (
            report["tail256_strict_rate"] >= 0.05
            or report["fixed_stochastic_strict_rate"] >= 1.0 / 16.0
        )
    ]
    tail512 = [
        float(report.get("tail512_strict_rate", report["tail_strict_rate"]))
        for report in reports
    ]
    tail1024 = [
        float(report.get("tail1024_strict_rate", report["tail_strict_rate"]))
        for report in reports
    ]
    fixed = [float(report["fixed_stochastic_strict_rate"]) for report in reports]
    production_ready_seeds = [
        report for report in reports
        if float(report.get("tail512_strict_rate", report["tail_strict_rate"]))
        >= PRODUCTION_READY_SEED_TAIL512
        and float(report["fixed_stochastic_strict_rate"])
        >= PRODUCTION_READY_SEED_FIXED
    ]
    uncapped_hit_seed_count = sum(
        report["first_hit_proposals"] is not None for report in reports
    )
    at_cap_reports = [
        report for report in reports
        if report["first_hit_proposals"] is not None
        and int(report["first_hit_proposals"]) <= common_cap
    ]
    hit_seed_count = len(at_cap_reports)
    post_update_hit_seed_count = sum(
        report in at_cap_reports
        and not report["first_hit_before_or_at_first_update"]
        for report in reports
    )
    return {
        "arm": reports[0]["arm"],
        "integrity_pass": all(report["integrity_pass"] for report in reports),
        "hit_seed_count": hit_seed_count,
        "uncapped_hit_seed_count": uncapped_hit_seed_count,
        "pre_update_hit_seed_count": hit_seed_count - post_update_hit_seed_count,
        "post_update_hit_seed_count": post_update_hit_seed_count,
        "only_pre_update_discoveries": bool(
            hit_seed_count > 0 and post_update_hit_seed_count == 0
        ),
        "stable_seed_count": len(stable),
        "common_cap_proposal_rmst": _mean(times),
        "mean_tail_strict_rate": _mean(report["tail_strict_rate"] for report in reports),
        "median_tail512_strict_rate": _median(tail512),
        "median_tail1024_strict_rate": _median(tail1024),
        "minimum_seed_tail1024_strict_rate": min(tail1024),
        "mean_tail256_strict_rate": _mean(report["tail256_strict_rate"] for report in reports),
        "mean_fixed_stochastic_strict_rate": _mean(
            report["fixed_stochastic_strict_rate"] for report in reports
        ),
        "median_fixed_stochastic_strict_rate": _median(fixed),
        "production_ready_seed_count": len(production_ready_seeds),
        "production_ready_at_4096": bool(
            len(production_ready_seeds) >= 2
            and _median(tail512) >= PRODUCTION_READY_MEDIAN_TAIL512
        ),
        "strict_hits": sum(report["strict_hits"] for report in reports),
        "reports": list(reports),
    }


def _learning_evidence(summary: dict) -> bool:
    """Exclude initial-policy discovery alone from claims about learning speed."""
    return bool(
        summary["post_update_hit_seed_count"] > 0
        or summary["stable_seed_count"] >= 2
        or summary["median_tail512_strict_rate"] >= 0.05
    )


def _production_warning(summary: dict) -> dict:
    reasons = []
    if summary["median_tail1024_strict_rate"] < PRODUCTION_MEDIAN_TAIL1024:
        reasons.append("median_last1024_below_0.70")
    if summary["minimum_seed_tail1024_strict_rate"] < PRODUCTION_MIN_SEED_TAIL1024:
        reasons.append("minimum_seed_last1024_below_0.50")
    if summary["median_fixed_stochastic_strict_rate"] < PRODUCTION_MEDIAN_FIXED:
        reasons.append("median_fixed_stochastic_below_0.60")
    return {
        "warning": bool(reasons),
        "warning_reasons": reasons,
        "thresholds": {
            "median_last1024": PRODUCTION_MEDIAN_TAIL1024,
            "minimum_seed_last1024": PRODUCTION_MIN_SEED_TAIL1024,
            "median_fixed_stochastic": PRODUCTION_MEDIAN_FIXED,
        },
    }


def _paired(candidate: Sequence[dict], control: Sequence[dict], cap: int) -> dict:
    by_seed = {int(report["seed"]): report for report in control}
    wins = losses = ties = 0
    rows = []
    for report in candidate:
        baseline = by_seed[int(report["seed"])]
        a_event = bool(
            report["first_hit_proposals"] is not None
            and int(report["first_hit_proposals"]) <= cap
        )
        b_event = bool(
            baseline["first_hit_proposals"] is not None
            and int(baseline["first_hit_proposals"]) <= cap
        )
        a = int(report["first_hit_proposals"]) if a_event else cap
        b = int(baseline["first_hit_proposals"]) if b_event else cap
        a_key = (int(a_event), -a)
        b_key = (int(b_event), -b)
        outcome = "win" if a_key > b_key else "loss" if a_key < b_key else "tie"
        wins += outcome == "win"
        losses += outcome == "loss"
        ties += outcome == "tie"
        rows.append({
            "seed": report["seed"],
            "candidate_event_by_cap": a_event, "candidate": a,
            "control_event_by_cap": b_event, "control": b,
            "outcome": outcome,
        })
    return {"wins": wins, "losses": losses, "ties": ties, "per_seed": rows}


def _promising(summary: dict, endpoint: int) -> bool:
    if not summary["integrity_pass"]:
        return False
    if endpoint <= 512:
        return summary["hit_seed_count"] >= 2 and summary["stable_seed_count"] >= 2
    if endpoint <= 1024:
        return summary["hit_seed_count"] >= 2 and summary["stable_seed_count"] >= 1
    return summary["hit_seed_count"] >= 1 and (
        summary["stable_seed_count"] >= 1 or summary["mean_tail_strict_rate"] >= 0.01
    )


def _better(left: dict, right: dict) -> bool:
    """Dual-objective comparison with stability before discovery tie-breaks."""
    left_noninferior = (
        left["hit_seed_count"] >= right["hit_seed_count"]
        and left["common_cap_proposal_rmst"] <= 1.10 * right["common_cap_proposal_rmst"]
        and left["mean_tail_strict_rate"] + 0.05 >= right["mean_tail_strict_rate"]
        and left["mean_fixed_stochastic_strict_rate"] + 1.0 / 16.0
        >= right["mean_fixed_stochastic_strict_rate"]
    )
    improvement = (
        left["hit_seed_count"] > right["hit_seed_count"]
        or left["stable_seed_count"] > right["stable_seed_count"]
        or left["mean_tail_strict_rate"] >= right["mean_tail_strict_rate"] + 0.05
        or left["common_cap_proposal_rmst"] <= 0.95 * right["common_cap_proposal_rmst"]
    )
    return bool(left_noninferior and improvement)


def analyse_task(jobs: Sequence[dict], task: str, endpoint: int) -> dict:
    selected = [job for job in jobs if job["task"]["name"] == task]
    if len(selected) != 6:
        raise ValueError(f"{task}: expected six jobs, found {len(selected)}")
    reports = [seed_report(job, endpoint) for job in selected]
    arms = sorted({job["arm"] for job in selected})
    if len(arms) != 2 or "B" not in arms:
        raise ValueError(f"{task}: expected B plus one candidate, got {arms}")
    common_cap = min(report["proposal_censor"] for report in reports)
    summaries = {
        arm: _arm_summary([report for report in reports if report["arm"] == arm], common_cap)
        for arm in arms
    }
    candidate = next(arm for arm in arms if arm != "B")
    paired = _paired(summaries[candidate]["reports"], summaries["B"]["reports"], common_cap)
    all_integrity = all(report["integrity_pass"] for report in reports)
    promising = {arm: _promising(summary, endpoint) for arm, summary in summaries.items()}
    learning_evidence = {
        arm: _learning_evidence(summary) for arm, summary in summaries.items()
    }
    production_ready = {
        arm: bool(summary["integrity_pass"] and summary["production_ready_at_4096"])
        for arm, summary in summaries.items()
    }
    eligible = production_ready if endpoint >= 4096 else promising
    selected_arm = None
    reason = "continue_screen"
    stop_task = False
    if not all_integrity:
        stop_task = True
        reason = "integrity_failure"
    else:
        candidate_better = _better(summaries[candidate], summaries["B"])
        baseline_better = _better(summaries["B"], summaries[candidate])
        if endpoint == 512:
            if (
                eligible[candidate] and learning_evidence[candidate]
                and candidate_better and paired["wins"] >= 2
            ):
                selected_arm, reason = candidate, "strong_candidate_at_512"
            elif eligible["B"] and learning_evidence["B"] and baseline_better:
                selected_arm, reason = "B", "strong_control_at_512"
            elif (
                eligible["B"] and eligible[candidate]
                and learning_evidence["B"] and learning_evidence[candidate]
                and not candidate_better
            ):
                # Parsimony tie-break prevents an easy task from consuming the
                # whole deadline merely to distinguish negligible differences.
                selected_arm, reason = "B", "both_strong_control_tiebreak_at_512"
        elif endpoint >= 1024 and any(eligible.values()):
            if eligible[candidate] and learning_evidence[candidate] and candidate_better:
                selected_arm, reason = candidate, f"candidate_selected_at_{endpoint}"
            elif (
                eligible["B"] and learning_evidence["B"]
                and (baseline_better or not eligible[candidate])
            ):
                selected_arm, reason = "B", f"control_selected_at_{endpoint}"
            elif endpoint >= 2048:
                # Descriptive lexicographic choice, still requiring the arm to
                # be absolutely promising.  Ties prefer the simpler control.
                keys = {
                    arm: (
                        summaries[arm]["stable_seed_count"],
                        summaries[arm]["mean_tail_strict_rate"],
                        summaries[arm]["mean_fixed_stochastic_strict_rate"],
                        -summaries[arm]["common_cap_proposal_rmst"],
                        arm == "B",
                    )
                    for arm in arms if eligible[arm] and learning_evidence[arm]
                }
                if keys:
                    selected_arm = max(keys, key=keys.get)
                    reason = f"promising_descriptive_selection_at_{endpoint}"
        if endpoint >= 4096 and selected_arm is None:
            stop_task = True
            reason = "no_promising_arm_at_4096"
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "task": task,
        "endpoint": endpoint,
        "common_proposal_cap": common_cap,
        "all_integrity_pass": all_integrity,
        "arm_summaries": summaries,
        "paired_candidate_vs_B": {"candidate": candidate, **paired},
        "promising": promising,
        "learning_evidence_beyond_initial_policy": learning_evidence,
        "production_ready_at_4096": production_ready,
        "decision": {
            "selected_arm": selected_arm,
            "stop_task": stop_task,
            "continue_screen": selected_arm is None and not stop_task,
            "reason": reason,
        },
    }


def analyse_selected(
    jobs: Sequence[dict], task: str, arm: str, endpoint: int,
    prior_production_warning_streak: int = 0,
) -> dict:
    """Revalidate an early-selected three-seed arm without resurrecting losers."""
    selected = [
        job for job in jobs
        if job["task"]["name"] == task and job["arm"] == arm
    ]
    if len(selected) != 3:
        raise ValueError(f"{task}/{arm}: expected three selected jobs")
    reports = [seed_report(job, endpoint) for job in selected]
    cap = min(report["proposal_censor"] for report in reports)
    summary = _arm_summary(reports, cap)
    production_monitor = _production_warning(summary)
    warning_streak = (
        prior_production_warning_streak + 1
        if endpoint in PRODUCTION_CHECKPOINTS and production_monitor["warning"]
        else 0
    )
    production_monitor["consecutive_warning_count"] = warning_streak
    revoke = not summary["integrity_pass"]
    reason = "selected_integrity_failure" if revoke else "selected_arm_continues"
    if not revoke and 1024 <= endpoint <= 4096:
        if summary["hit_seed_count"] == 0:
            revoke, reason = True, "selected_all_seed_no_discovery"
        elif endpoint >= 2048 and summary["stable_seed_count"] == 0:
            revoke, reason = True, "selected_all_seed_no_persistence"
        elif endpoint == 4096 and not summary["production_ready_at_4096"]:
            revoke, reason = True, "selected_arm_not_production_ready_at_4096"
    if (
        not revoke and endpoint in PRODUCTION_CHECKPOINTS
        and warning_streak >= 2
    ):
        revoke, reason = True, "two_consecutive_production_stability_warnings"
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "task": task,
        "arm": arm,
        "endpoint": endpoint,
        "arm_summary": summary,
        "production_monitor": production_monitor,
        "decision": {"revoke": revoke, "reason": reason},
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def serialisable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: serialisable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialisable(item) for item in value]
        return value

    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(serialisable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
    return path.resolve()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=TASK_ORDER, required=True)
    parser.add_argument("--endpoint", type=int, choices=SCREEN_CHECKPOINTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = analyse_task(manifest["jobs"], args.task, args.endpoint)
    write_json(args.output, payload)
    print(json.dumps(payload["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
