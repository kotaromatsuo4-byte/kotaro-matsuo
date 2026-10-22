"""Create the no-training P0X2E3 entropy-screen preflight authority."""

from __future__ import annotations

import csv
from dataclasses import asdict
import math
from pathlib import Path

import run_ori_codex6_experiments as runner


STAGE2_SUMMARY = (
    runner.SCRIPT_DIRECTORY / "experiments_ori5" / "analysis"
    / "stage2_run_summary.csv"
)


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recursive_differences(left, right, prefix="") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            result.extend(recursive_differences(left.get(key), right.get(key), path))
        return result
    if left != right:
        return [{"path": prefix, "P0X2_value": left, "candidate_value": right}]
    return []


def stage2_prior() -> dict:
    selected = {
        row["profile"]: row for row in rows(STAGE2_SUMMARY)
        if row["task"] == "water1" and row["profile"] in ("B", "E", "F")
    }
    if set(selected) != {"B", "E", "F"}:
        raise RuntimeError("ori5 Stage2 water1 B/E/F rows are incomplete")

    def summarize(name: str) -> dict:
        row = selected[name]
        first = row.get("first_success_episode")
        return {
            "profile": name,
            "batch_size": int(row["batch_size"]),
            "actor_learning_rate": float(row["actor_learning_rate"]),
            "critic_learning_rate": float(row["critic_learning_rate"]),
            "entropy_coefficient": float(row["entropy_coefficient"]),
            "episodes": int(row["episodes"]),
            "training_successes": int(row["training_successes"]),
            "first_success_episode": int(first) if first else None,
            "tail512_success_rate": float(row["tail_success_rate"]),
            "tail512_crossing_rate": float(row["tail_crossing_rate"]),
            "heldout_strict_success_rate": float(
                row["heldout_strict_success_rate"]
            ),
        }

    result = {name: summarize(name) for name in ("B", "E", "F")}
    entropy_only = recursive_differences(
        {
            key: result["B"][key] for key in (
                "batch_size", "actor_learning_rate",
                "critic_learning_rate", "entropy_coefficient",
            )
        },
        {
            key: result["F"][key] for key in (
                "batch_size", "actor_learning_rate",
                "critic_learning_rate", "entropy_coefficient",
            )
        },
    )
    if [row["path"] for row in entropy_only] != ["entropy_coefficient"]:
        raise RuntimeError("ori5 profile F is no longer entropy-only relative to B")
    return {
        "authority_path": str(STAGE2_SUMMARY.resolve()),
        "authority_sha256": runner.file_sha256(STAGE2_SUMMARY),
        "profile_correction": (
            "The entropy-only 0.001->0.003 comparison is B versus F. "
            "Profile E keeps entropy=0.001 and changes batch size and both learning rates."
        ),
        "profiles": result,
        "B_to_F_optimizer_differences": entropy_only,
        "negative_prior": (
            "At the single Stage2 water1 seed1102, B found one training success "
            "at episode1771 while F found none; both final held-out success rates "
            "were zero and F tail512 crossing rate was 1.0."
        ),
    }


def P0X2_entropy_prior() -> dict:
    result = {}
    for seed in runner.PROTOCOLS["rescue"].seeds:
        path = (
            runner.EXPERIMENT_ROOT / "screen" / "rescue" / "P0X2"
            / "water1" / "B" / f"seed{seed}" / "update_history.csv"
        )
        history = rows(path)
        values = [float(row["entropy"]) for row in history]
        if len(values) < 64 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"P0X2 seed{seed} entropy history is incomplete")
        result[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": runner.file_sha256(path),
            "update_count": len(values),
            "first32_mean": sum(values[:32]) / 32,
            "last32_mean": sum(values[-32:]) / 32,
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "per_seed": result,
        "negative_prior": (
            "P0X2 policy entropy remains near 2.1 through training in all three "
            "water1 seeds, so coefficient 0.003 is not motivated by measured "
            "entropy collapse; it is only a minimal exploration-pressure diagnostic."
        ),
    }


def execution_identity() -> dict:
    protocol = runner.PROTOCOLS["entropy_diagnostic"]
    task = runner.TASKS["water1"]
    profile = runner.PROFILES[task.profile]
    seed = protocol.seeds[0]
    parent_config = runner.expected_config(
        task, profile, protocol, seed, "ori6", runner.P0_CROSS2_MODE
    )
    candidate_config = runner.expected_config(
        task, profile, protocol, seed, "ori6", runner.P0_CROSS2_ENTROPY3_MODE
    )
    config_differences = recursive_differences(parent_config, candidate_config)

    destination = runner.EXPERIMENT_ROOT / "tmp" / "P0X2E3_preflight_command"
    base = {
        "task": asdict(task),
        "profile": asdict(profile),
        "protocol": asdict(protocol),
        "seed": seed,
        "implementation": "ori6",
    }
    parent_command = runner.build_command(
        {**base, "candidate_mode": runner.P0_CROSS2_MODE}, destination
    )
    candidate_command = runner.build_command(
        {**base, "candidate_mode": runner.P0_CROSS2_ENTROPY3_MODE}, destination
    )
    command_differences = [
        {
            "index": index,
            "flag": parent_command[index - 1] if index else None,
            "P0X2_value": left,
            "candidate_value": right,
        }
        for index, (left, right) in enumerate(zip(parent_command, candidate_command))
        if left != right
    ]
    duplicate_flags = sorted({
        token for token in candidate_command
        if token.startswith("--") and candidate_command.count(token) > 1
    })
    return {
        "parent_candidate_mode": runner.P0_CROSS2_MODE,
        "candidate_mode": runner.P0_CROSS2_ENTROPY3_MODE,
        "parent_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ),
        "candidate_reward_sha256": runner.candidate_reward_sha256(
            runner.P0_CROSS2_ENTROPY3_MODE
        ),
        "reward_hash_unchanged": runner.candidate_reward_sha256(
            runner.P0_CROSS2_MODE
        ) == runner.candidate_reward_sha256(runner.P0_CROSS2_ENTROPY3_MODE),
        "parent_optimizer": runner.expected_optimizer(
            runner.P0_CROSS2_MODE, profile
        ),
        "candidate_optimizer": runner.expected_optimizer(
            runner.P0_CROSS2_ENTROPY3_MODE, profile
        ),
        "expected_config_differences": config_differences,
        "command_differences": command_differences,
        "duplicate_long_flags": duplicate_flags,
        "execution_difference_is_entropy_only": (
            [row["path"] for row in config_differences] == ["entropy_coefficient"]
            and len(command_differences) == 1
            and command_differences[0]["flag"] == "--entropy-coefficient"
            and float(command_differences[0]["P0X2_value"]) == 1.0e-3
            and float(command_differences[0]["candidate_value"]) == 3.0e-3
            and not duplicate_flags
        ),
    }


def build_report() -> dict:
    identity = execution_identity()
    B12 = runner.P0_CROSS2_FRONTIER12_REJECT_REPORT
    offline = runner.P0_CROSS2_OFFLINE_AUDIT
    report = {
        "schema_version": 1,
        "status": "preflight_complete_no_training",
        "candidate_mode": runner.P0_CROSS2_ENTROPY3_MODE,
        "candidate_mode_slug": runner.MODE_SLUGS[
            runner.P0_CROSS2_ENTROPY3_MODE
        ],
        "recommendation": "bounded_1024_diagnostic_only_negative_prior",
        "training_launched": False,
        "screen_scope": {
            "task": "water1",
            "seeds": list(runner.PROTOCOLS["entropy_diagnostic"].seeds),
            "episodes": 1024,
            "checkpoints": [512, 1024],
        },
        "execution_identity": identity,
        "saved_path_reward_authority": {
            "path": str(offline.resolve()),
            "sha256": runner.file_sha256(offline),
            "candidate_reward_sha256": identity["candidate_reward_sha256"],
            "reused_unchanged_from": runner.P0_CROSS2_MODE,
        },
        "preceding_B12_reject_authority": {
            "path": str(B12.resolve()),
            "sha256": runner.file_sha256(B12),
        },
        "ori5_stage2_prior": stage2_prior(),
        "P0X2_policy_entropy_prior": P0X2_entropy_prior(),
        "formal_1024_gate": {
            "pass_requires": {
                "tail256_median_spread_error_le_0_20_seed_count_min": 2,
                "constructive_rate_seed_count_min": 2,
                "constructive_rate": (
                    "tail256 spread_error<=0.1 rate>=0.05 OR "
                    "ori6_maximum_spread_progress>=0.8 rate>=0.25"
                ),
                "tail256_crossing_rate_le_0_75_seed_count_min": 2,
                "reward_integrity": True,
            },
            "reject_if": (
                "at least 2 seeds have zero strict success through episode1024 "
                "and tail256 median spread_error>=0.30"
            ),
        },
        "source_sha256": runner.source_hashes(),
        "model_sha256_by_task": {
            name: runner.model_hashes(task) for name, task in runner.TASKS.items()
        },
    }
    report["preflight_pass"] = (
        identity["reward_hash_unchanged"] is True
        and identity["execution_difference_is_entropy_only"] is True
        and runner.candidate_reward_sha256(runner.P0_CROSS2_MODE)
        == "ba48b5e037202cab32962ae45b8083d459e89c521815907597e5bcc10cef6892"
    )
    return report


def main() -> int:
    report = build_report()
    if report["preflight_pass"] is not True:
        raise RuntimeError("P0X2E3 preflight failed")
    runner.atomic_json(runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT, report)
    print(runner.P0_CROSS2_ENTROPY3_PREFLIGHT_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
