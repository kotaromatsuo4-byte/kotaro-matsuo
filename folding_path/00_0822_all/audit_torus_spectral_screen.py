"""Read-only audit of the torus spectral-weight screening results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260830_torus_spectral_screen_v1"
)

FIRST_RESULT_FIELDS = (
    "first_strict_success_episode",
    "first_strict_success_episode_steps",
    "first_strict_success_mechanical_steps",
    "optimizer_updates_before_first_success",
)


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def expected_worker_path(alpha: str, seed: int) -> Path:
    label = alpha.replace(".", "p")
    return (
        EXPERIMENT_DIRECTORY
        / "screening"
        / "workers"
        / f"alpha_{label}_seed{seed}.csv"
    )


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"expected one data row, found {len(rows)}")
    return rows[0]


def audit_row(
    row: dict[str, str],
    *,
    alpha: str,
    seed: int,
    experiment_max_episodes: int,
    controlled: dict,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    expected_values = {
        "task": "torus_d2",
        "spectral_alpha": alpha,
        "seed": str(seed),
        "max_steps_per_episode": str(
            controlled["max_steps_per_episode"]
        ),
        "actor_reference_horizon": str(
            controlled["actor_reference_horizon"]
        ),
        "batch_size": str(controlled["batch_size"]),
    }
    for field, expected in expected_values.items():
        if row.get(field) != expected:
            errors.append(
                f"{field}={row.get(field)!r}, expected {expected!r}"
            )

    decimal_values = {
        "max_step_width": controlled["max_step_width"],
        "actor_learning_rate": controlled["actor_learning_rate"],
        "critic_learning_rate": controlled["critic_learning_rate"],
    }
    for field, expected in decimal_values.items():
        try:
            matches = Decimal(row[field]) == Decimal(str(expected))
        except (KeyError, ArithmeticError):
            matches = False
        if not matches:
            errors.append(
                f"{field}={row.get(field)!r}, expected {expected!r}"
            )

    try:
        declared_cap = as_int(row, "max_episodes")
        episodes = as_int(row, "episodes_completed")
        success = as_int(row, "success")
        censored = as_int(row, "censored_at_max_episodes")
        updates_at_stop = as_int(row, "optimizer_updates_at_stop")
        steps_at_stop = as_int(row, "mechanical_steps_at_stop")
        horizon = as_int(row, "max_steps_per_episode")
        batch_size = as_int(row, "batch_size")
    except (KeyError, TypeError, ValueError) as error:
        return "invalid", errors + [f"integer parsing failed: {error}"]

    if declared_cap != experiment_max_episodes:
        # This catches the 8-episode smoke-test row without treating it as a
        # legitimate 10,000-episode right-censored observation.
        return "diagnostic_partial", errors

    if success == 1:
        if censored != 0:
            errors.append("successful row must have censored flag 0")
        try:
            first_episode = as_int(row, "first_strict_success_episode")
            success_steps = as_int(
                row, "first_strict_success_episode_steps"
            )
            first_mechanical = as_int(
                row, "first_strict_success_mechanical_steps"
            )
            updates_before = as_int(
                row, "optimizer_updates_before_first_success"
            )
        except (KeyError, TypeError, ValueError) as error:
            return "invalid", errors + [
                f"success-field parsing failed: {error}"
            ]

        if episodes != first_episode:
            errors.append(
                "episodes_completed must equal first-success episode"
            )
        if not 1 <= first_episode <= experiment_max_episodes:
            errors.append("first-success episode is outside the run cap")
        if not 1 <= success_steps <= horizon:
            errors.append("success-episode step count is outside horizon")
        if first_mechanical != steps_at_stop:
            errors.append(
                "first-success mechanical steps must equal steps at stop"
            )
        if not success_steps <= first_mechanical <= episodes * horizon:
            errors.append("cumulative mechanical-step count is impossible")
        expected_updates = (first_episode - 1) // batch_size
        if updates_before != expected_updates:
            errors.append(
                f"updates before success={updates_before}, expected "
                f"{expected_updates}"
            )
        if updates_at_stop != updates_before:
            errors.append(
                "stop-on-success run performed an update after success"
            )
        return ("success" if not errors else "invalid"), errors

    if success != 0:
        errors.append(f"success flag must be 0 or 1, found {success}")
    if censored != 1:
        errors.append("unsuccessful capped row must have censored flag 1")
    if episodes != experiment_max_episodes:
        errors.append("censored row did not reach the experiment cap")
    if any(row.get(field, "") != "" for field in FIRST_RESULT_FIELDS):
        errors.append("censored row contains first-success values")
    expected_updates = (episodes + batch_size - 1) // batch_size
    if updates_at_stop != expected_updates:
        errors.append(
            f"updates at censoring={updates_at_stop}, expected "
            f"{expected_updates}"
        )
    if not 0 <= steps_at_stop <= episodes * horizon:
        errors.append("mechanical steps at censoring are outside bounds")
    return ("right_censored" if not errors else "invalid"), errors


def display_value(row: dict[str, str] | None, status: str, field: str) -> str:
    if status == "success" and row is not None:
        return row[field]
    if status == "right_censored" and row is not None:
        censor_field = {
            "first_strict_success_episode": "episodes_completed",
            "optimizer_updates_before_first_success": (
                "optimizer_updates_at_stop"
            ),
            "first_strict_success_mechanical_steps": (
                "mechanical_steps_at_stop"
            ),
        }[field]
        return f"censored@{row[censor_field]}"
    return "-"


def main() -> None:
    manifest_path = EXPERIMENT_DIRECTORY / "manifest.json"
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    alphas = [str(value) for value in manifest["spectral_alphas"]]
    seeds = [int(value) for value in manifest["seeds"]]
    cap = int(manifest["max_episodes"])
    controlled = manifest["controlled_settings"]
    outcomes: list[tuple[str, int, str, dict[str, str] | None]] = []
    audit_errors: list[str] = []

    print("alpha,seed,status,first_episode,updates_before,mechanical_steps")
    for alpha in alphas:
        for seed in seeds:
            path = expected_worker_path(alpha, seed)
            if not path.exists():
                status = "incomplete_missing"
                row = None
                errors: list[str] = []
            else:
                try:
                    row = read_single_row(path)
                    status, errors = audit_row(
                        row,
                        alpha=alpha,
                        seed=seed,
                        experiment_max_episodes=cap,
                        controlled=controlled,
                    )
                except (OSError, ValueError) as error:
                    row = None
                    status = "invalid"
                    errors = [str(error)]

            outcomes.append((alpha, seed, status, row))
            print(
                ",".join((
                    alpha,
                    str(seed),
                    status,
                    display_value(
                        row, status, "first_strict_success_episode"
                    ),
                    display_value(
                        row,
                        status,
                        "optimizer_updates_before_first_success",
                    ),
                    display_value(
                        row,
                        status,
                        "first_strict_success_mechanical_steps",
                    ),
                ))
            )
            audit_errors.extend(
                f"alpha={alpha}, seed={seed}: {error}"
                for error in errors
            )

    print("\nalpha,success,right_censored,incomplete_or_diagnostic,invalid")
    for alpha in alphas:
        counts = Counter(
            status
            for outcome_alpha, _, status, _ in outcomes
            if outcome_alpha == alpha
        )
        incomplete = (
            counts["incomplete_missing"] + counts["diagnostic_partial"]
        )
        print(
            f"{alpha},{counts['success']},{counts['right_censored']},"
            f"{incomplete},{counts['invalid']}"
        )

    print(f"\naudit_errors={len(audit_errors)}")
    for error in audit_errors:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
