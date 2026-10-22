"""Freeze the historical P0X2W2 episode-1024 rejection evidence.

This utility intentionally does not import the live ori6 runner or learner.  The
next candidate changes those files, whereas this authority must remain bound to
the source hashes recorded in the immutable historical W2 manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "experiments_ori6/manifests/width_diagnostic_P0X2W2_d1dc9b798e14.json"
GATE_512 = ROOT / "experiments_ori6/early_gates/width_diagnostic/P0X2W2/water1/checkpoint_512.json"
GATE_1024 = ROOT / "experiments_ori6/early_gates/width_diagnostic/P0X2W2/water1/checkpoint_1024.json"
OUTPUT = ROOT / "experiments_ori6/analysis/width_diagnostic_P0X2W2_reject_report.json"

EXPECTED_MANIFEST_SHA256 = "2e47fc17def1a1b6e8ff7d45792759d912cd09434ba598704383a0225c371410"
EXPECTED_MANIFEST_FINGERPRINT = "d1dc9b798e14a21cdc0abe9129ff3d42abe045170598b16be7eea02ec93fcdfc"
EXPECTED_GATE_512_SHA256 = "57267afac6df0864e4664dd06336c8b17e5662c3007bb11f4816f0d5a5269a6d"
EXPECTED_GATE_1024_SHA256 = "ca512b5507099a30962e98be8fae1aae4ef2400c0ab8b715c135c09e86e5ddd7"
EXPECTED_MODE = "state_potential_crossing_penalty_2_max_step_width_0p2"
EXPECTED_SLUG = "P0X2W2"
EXPECTED_SEEDS = (3101, 3102, 3103)
EXPECTED_EPISODE = 1024
EXPECTED_UPDATE_ROWS = 128

ARTIFACT_NAMES = (
    "checkpoint.pt",
    "training_history.csv",
    "update_history.csv",
    "run_config.json",
    "best_policy.pt",
    "best_policy_metrics.json",
    "deterministic_evaluation_history.csv",
    "stochastic_evaluation_history.csv",
)

REWARD_COMPONENT_FIELDS = (
    "ori6_total_flatness_progress_reward",
    "ori6_total_target_progress_reward",
    "ori6_total_spread_frontier_reward",
    "ori6_total_hard_spread_milestone_reward",
    "ori6_total_joint_milestone_reward",
    "ori6_total_terminal_fixed_reward",
    "ori6_total_terminal_quality_reward",
    "ori6_total_terminal_speed_reward",
    # The three rollback fields are the decomposition of this correction, so
    # including both would double count it.
    "ori6_total_terminal_correction_reward",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def equal_json(left: Any, right: Any) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def expected_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and expected_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            expected_subset(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return equal_json(actual, expected)


def verify_expected_config(actual: dict[str, Any], expected: dict[str, Any], seed: int) -> None:
    missing = sorted(set(expected) - set(actual))
    require(not missing, f"seed{seed}: run_config missing expected keys {missing}")
    mismatches = [key for key, value in expected.items() if not expected_subset(actual[key], value)]
    require(not mismatches, f"seed{seed}: run_config mismatches historical manifest keys {mismatches}")


def status_path(seed: int) -> Path:
    root = ROOT / f"experiments_ori6/_runner_logs/width_diagnostic/P0X2W2/water1/ori6/seed{seed}"
    matches = sorted(root.glob("*/status.json"))
    require(len(matches) == 1, f"seed{seed}: expected exactly one runner status, found {len(matches)}")
    return matches[0]


def finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    require(math.isfinite(value), f"nonfinite {field}")
    return value


def artifact_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        path = directory / name
        require(path.is_file(), f"missing historical artifact: {relative(path)}")
        inventory[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return inventory


def reward_audit(rows: list[dict[str, str]], seed: int) -> dict[str, Any]:
    max_saved_decomp = 0.0
    max_recomputed_decomp = 0.0
    max_step_decomp = 0.0
    nonfinite = 0
    crossing_fixed: set[float] = set()
    max_crossing_frontier_cancel = 0.0
    max_crossing_joint_cancel = 0.0
    counts = {
        "success": 0,
        "self_crossing": 0,
        "wrong_flat": 0,
        "timeout": 0,
        "correction_failed": 0,
        "no_modes": 0,
    }

    for row in rows:
        for field in REWARD_COMPONENT_FIELDS + (
            "total_reward",
            "ori6_reward_decomposition_error",
            "ori6_maximum_abs_step_reward_decomposition_error",
        ):
            try:
                value = float(row[field])
            except (KeyError, ValueError):
                nonfinite += 1
                continue
            if not math.isfinite(value):
                nonfinite += 1

        total = float(row["total_reward"])
        recomputed = sum(float(row[field]) for field in REWARD_COMPONENT_FIELDS)
        max_recomputed_decomp = max(max_recomputed_decomp, abs(total - recomputed))
        max_saved_decomp = max(max_saved_decomp, abs(float(row["ori6_reward_decomposition_error"])))
        max_step_decomp = max(
            max_step_decomp,
            abs(float(row["ori6_maximum_abs_step_reward_decomposition_error"])),
        )
        nonfinite += int(float(row["ori6_nonfinite_reward_component_count"]))

        for field in counts:
            counts[field] += int(float(row[field]))
        if int(float(row["self_crossing"])):
            crossing_fixed.add(float(row["ori6_total_terminal_fixed_reward"]))
            max_crossing_frontier_cancel = max(
                max_crossing_frontier_cancel,
                abs(
                    float(row["ori6_total_spread_frontier_reward"])
                    + float(row["ori6_total_frontier_rollback_reward"])
                ),
            )
            max_crossing_joint_cancel = max(
                max_crossing_joint_cancel,
                abs(
                    float(row["ori6_total_joint_milestone_reward"])
                    + float(row["ori6_total_joint_milestone_rollback_reward"])
                ),
            )

    audit = {
        "seed": seed,
        "episode_count": len(rows),
        "termination_counts": counts,
        "crossing_terminal_fixed_reward_unique": sorted(crossing_fixed),
        "maximum_abs_crossing_frontier_plus_rollback": max_crossing_frontier_cancel,
        "maximum_abs_crossing_joint_plus_rollback": max_crossing_joint_cancel,
        "maximum_abs_saved_episode_reward_decomposition_error": max_saved_decomp,
        "maximum_abs_recomputed_episode_reward_decomposition_error": max_recomputed_decomp,
        "maximum_abs_step_reward_decomposition_error": max_step_decomp,
        "nonfinite_reward_component_count": nonfinite,
    }
    audit["pass"] = (
        nonfinite == 0
        and crossing_fixed == {-2.0}
        and max_crossing_frontier_cancel <= 1e-12
        and max_crossing_joint_cancel <= 1e-12
        and max_saved_decomp <= 1e-12
        and max_recomputed_decomp <= 1e-12
        and max_step_decomp <= 1e-12
    )
    return audit


def checkpoint_metrics(rows: list[dict[str, str]], gate_seed: dict[str, Any], seed: int) -> dict[str, Any]:
    tail = rows[-256:]
    tail_spread_errors = sorted(float(row["spread_error"]) for row in tail)
    tail_flatness = sorted(float(row["flatness_ratio"]) for row in tail)
    direct = {
        "seed": seed,
        "termination_counts": {
            field: sum(int(float(row[field])) for row in rows)
            for field in ("success", "self_crossing", "wrong_flat", "timeout", "correction_failed")
        },
        "minimum_terminal_flatness_ratio": min(float(row["flatness_ratio"]) for row in rows),
        "minimum_same_state_joint_cost": min(
            float(row["flatness_ratio"])
            + float(row["spread_error"]) ** 2
            for row in rows
        ),
        "joint_milestone_episode_count": sum(
            float(row["ori6_total_joint_milestone_reward"]) > 1e-12 for row in rows
        ),
        "tail256": {
            "success_rate": sum(int(float(row["success"])) for row in tail) / 256.0,
            "crossing_rate": sum(int(float(row["self_crossing"])) for row in tail) / 256.0,
            "wrong_flat_rate": sum(int(float(row["wrong_flat"])) for row in tail) / 256.0,
            "timeout_rate": sum(int(float(row["timeout"])) for row in tail) / 256.0,
            # Preserve the historical gate's two deliberately different
            # median conventions exactly.
            "median_flatness_ratio": tail_flatness[len(tail_flatness) // 2],
            "median_exact_spread_error": 0.5
            * (
                tail_spread_errors[(len(tail_spread_errors) - 1) // 2]
                + tail_spread_errors[len(tail_spread_errors) // 2]
            ),
            "plane_switch_endpoint_proxy_count": sum(
                float(row["flatness_ratio"]) <= 0.5
                and float(row["spread_error"]) <= 0.1
                for row in tail
            ),
            "valid_joint_target_or_strict_count": sum(
                (
                    float(row["flatness_ratio"]) <= 0.1
                    and float(row["spread_error"]) <= 0.05
                    and int(float(row["self_crossing"])) == 0
                    and int(float(row["correction_failed"])) == 0
                    and int(float(row["no_modes"])) == 0
                )
                for row in tail
            ),
        },
    }

    checks = {
        "success_rate": "tail_success_rate",
        "crossing_rate": "tail_crossing_rate",
        "wrong_flat_rate": "tail_wrong_flat_rate",
        "timeout_rate": "tail_timeout_rate",
        "median_flatness_ratio": "tail_median_flatness_ratio",
        "median_exact_spread_error": "tail_exact_median_spread_error",
        "plane_switch_endpoint_proxy_count": "tail_plane_switch_endpoint_proxy_episodes",
        "valid_joint_target_or_strict_count": "tail_valid_joint_target_episodes",
    }
    for direct_key, gate_key in checks.items():
        direct_value = direct["tail256"][direct_key]
        gate_value = gate_seed[gate_key]
        if isinstance(direct_value, float):
            require(abs(direct_value - float(gate_value)) <= 1e-12, f"seed{seed}: gate mismatch {gate_key}")
        else:
            require(direct_value == int(gate_value), f"seed{seed}: gate mismatch {gate_key}")
    require(
        abs(direct["minimum_terminal_flatness_ratio"] - float(gate_seed["minimum_terminal_flatness_ratio"]))
        <= 1e-12,
        f"seed{seed}: minimum flatness mismatch",
    )
    require(
        direct["joint_milestone_episode_count"] == int(gate_seed["joint_milestone_episodes"]),
        f"seed{seed}: joint milestone mismatch",
    )
    return direct


def build_report() -> dict[str, Any]:
    require(file_sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256, "historical W2 manifest SHA mismatch")
    require(file_sha256(GATE_512) == EXPECTED_GATE_512_SHA256, "historical W2 512 gate SHA mismatch")
    require(file_sha256(GATE_1024) == EXPECTED_GATE_1024_SHA256, "historical W2 1024 gate SHA mismatch")

    manifest = load_json(MANIFEST)
    manifest_content = dict(manifest)
    recorded_manifest_digest = manifest_content.pop("manifest_fingerprint", None)
    manifest_content.pop("created_utc", None)
    require(recorded_manifest_digest == EXPECTED_MANIFEST_FINGERPRINT, "historical manifest fingerprint mismatch")
    require(canonical_digest(manifest_content) == recorded_manifest_digest, "historical manifest self-digest mismatch")
    require(manifest["candidate_mode"] == EXPECTED_MODE, "historical candidate mode mismatch")
    require(manifest["candidate_mode_slug"] == EXPECTED_SLUG, "historical candidate slug mismatch")
    require(manifest["protocol"]["episodes"] == 2048, "historical intended budget mismatch")
    require(manifest["protocol"]["seeds"] == list(EXPECTED_SEEDS), "historical seed set mismatch")
    require(manifest["protocol"]["tasks"] == ["water1"], "historical task set mismatch")

    for path_key, sha_key in (
        ("offline_saved_path_ordering_authority", "offline_saved_path_ordering_authority_sha256"),
        ("preceding_candidate_reject_authority", "preceding_candidate_reject_authority_sha256"),
        ("width_feasibility_authority", "width_feasibility_authority_sha256"),
    ):
        authority_path = Path(manifest[path_key])
        require(authority_path.resolve().is_file(), f"missing historical authority {path_key}")
        require(file_sha256(authority_path) == manifest[sha_key], f"historical authority SHA mismatch {path_key}")

    gate512 = load_json(GATE_512)
    gate1024 = load_json(GATE_1024)
    require(gate512["checkpoint_episode"] == 512 and gate512["stop"] is False, "512 gate was not continue")
    require(gate512["reason"] == "continue", "512 gate reason mismatch")
    require(gate1024["checkpoint_episode"] == EXPECTED_EPISODE, "1024 gate endpoint mismatch")
    require(gate1024["stop"] is True, "1024 gate did not stop")
    require(gate1024["reason"] == "width_diagnostic_wide_spread_reject", "1024 gate reason mismatch")
    formal = gate1024["formal_width_diagnostic_gate"]
    require(formal["pass"] is False and formal["reject"] is True, "1024 formal reject flags mismatch")
    require(formal["reward_integrity_pass"] is True, "1024 reward integrity gate failed")
    require(formal["tail_exact_median_spread_error_ge_0_30_seed_count"] == 2, "1024 wide spread count mismatch")
    require(formal["tail_plane_switch_proxy_positive_seed_count"] == 0, "1024 proxy count mismatch")
    require(formal["tail_valid_F_le_0_1_e_le_0_05_or_strict_seed_count"] == 0, "1024 valid joint count mismatch")

    jobs_by_seed = {int(job["seed"]): job for job in manifest["jobs"]}
    require(set(jobs_by_seed) == set(EXPECTED_SEEDS), "historical job seed set mismatch")

    integrity_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pooled_training: list[dict[str, str]] = []
    for seed in EXPECTED_SEEDS:
        job = jobs_by_seed[seed]
        directory = Path(job["output_directory"])
        require(directory.resolve().is_dir(), f"seed{seed}: historical artifact directory absent")
        rows = read_csv(directory / "training_history.csv")
        updates = read_csv(directory / "update_history.csv")
        require(len(rows) == EXPECTED_EPISODE, f"seed{seed}: training row count mismatch")
        require(len(updates) == EXPECTED_UPDATE_ROWS, f"seed{seed}: update row count mismatch")
        require([int(float(row["episode"])) for row in rows] == list(range(1, EXPECTED_EPISODE + 1)), f"seed{seed}: training episode sequence mismatch")
        require(int(float(updates[-1]["episode"])) == EXPECTED_EPISODE, f"seed{seed}: update endpoint mismatch")

        config = load_json(directory / "run_config.json")
        verify_expected_config(config, job["expected_config"], seed)
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        require(isinstance(checkpoint, dict), f"seed{seed}: checkpoint payload is not a mapping")
        require(int(checkpoint.get("episode", -1)) == EXPECTED_EPISODE, f"seed{seed}: checkpoint episode mismatch")
        require(equal_json(checkpoint.get("config"), config), f"seed{seed}: checkpoint/run_config mismatch")

        status_file = status_path(seed)
        status = load_json(status_file)
        require(status["job_id"] == job["job_id"], f"seed{seed}: status job id mismatch")
        require(status["job_fingerprint"] == job["fingerprint"], f"seed{seed}: status fingerprint mismatch")
        require(status["command"] == job["command"], f"seed{seed}: status command mismatch")
        require(status["status"] == "early_stopped_at_checkpoint", f"seed{seed}: status is not early stopped")
        require(status["return_code"] == 1, f"seed{seed}: deliberate stop return code mismatch")
        require(status["checkpoint_episode"] == EXPECTED_EPISODE, f"seed{seed}: status endpoint mismatch")
        require(status["cohort_decision_checkpoint_episode"] == EXPECTED_EPISODE, f"seed{seed}: cohort endpoint mismatch")
        require(status["early_gate_decision"] == gate1024, f"seed{seed}: embedded gate differs from authority")
        partial = status["partial_artifact_validation"]
        require(partial["valid"] is True and partial["errors"] == [], f"seed{seed}: partial validation failed")
        require(partial["endpoint_episode"] == EXPECTED_EPISODE, f"seed{seed}: partial endpoint mismatch")
        require(partial["training_rows"] == EXPECTED_EPISODE, f"seed{seed}: partial training rows mismatch")
        require(partial["update_rows"] == EXPECTED_UPDATE_ROWS, f"seed{seed}: partial update rows mismatch")
        require(not (directory / "summary.json").exists(), f"seed{seed}: unexpected full summary")

        inventory = artifact_inventory(directory)
        reward = reward_audit(rows, seed)
        require(reward["pass"], f"seed{seed}: reward integrity failed")
        metrics = checkpoint_metrics(rows, gate1024["per_seed"][str(seed)], seed)
        pooled_training.extend(rows)
        reward_rows.append(reward)
        metric_rows.append(metrics)
        integrity_rows.append(
            {
                "seed": seed,
                "job_id": job["job_id"],
                "job_fingerprint": job["fingerprint"],
                "artifact_path": relative(directory),
                "runner_status_path": relative(status_file),
                "runner_status_sha256": file_sha256(status_file),
                "checkpoint_validation": {
                    "valid": True,
                    "endpoint_episode": EXPECTED_EPISODE,
                    "checkpoint_payload_episode": int(checkpoint["episode"]),
                    "training_rows": len(rows),
                    "update_rows": len(updates),
                    "config_matches_historical_manifest": True,
                    "checkpoint_config_matches_run_config": True,
                },
                "full_finalization_validation": {
                    "claimed": False,
                    "summary_json_absent": True,
                    "reason": "preregistered_checkpoint_rejection_before_heldout_finalization",
                },
                "artifact_inventory": inventory,
            }
        )

    pooled_reward = reward_audit(pooled_training, seed=-1)
    require(pooled_reward["pass"], "pooled reward integrity failed")
    pooled_reward.pop("seed")

    report: dict[str, Any] = {
        "schema_version": 1,
        "authority_type": "ori6_width_diagnostic_atomic_checkpoint_reject",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_mode": EXPECTED_MODE,
        "candidate_mode_slug": EXPECTED_SLUG,
        "parent_candidate_mode": manifest["parent_candidate_mode"],
        "task": "water1",
        "decision_endpoint_episode": EXPECTED_EPISODE,
        "decision": "reject_candidate",
        "decision_reason": "width_diagnostic_wide_spread_reject",
        "scope": {
            "protocol": "width_diagnostic",
            "historically_preregistered_episode_cap": 2048,
            "actual_atomic_decision_endpoint": EXPECTED_EPISODE,
            "checkpoint_every": 512,
            "seeds": list(EXPECTED_SEEDS),
            "tasks": ["water1"],
            "is_atomic_checkpoint_authority": True,
            "episode_2048_claimed": False,
            "heldout_claimed": False,
            "random_baseline_claimed": False,
        },
        "historical_binding": {
            "manifest_path": relative(MANIFEST),
            "manifest_sha256": file_sha256(MANIFEST),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "checkpoint_512_gate_path": relative(GATE_512),
            "checkpoint_512_gate_sha256": file_sha256(GATE_512),
            "checkpoint_1024_gate_path": relative(GATE_1024),
            "checkpoint_1024_gate_sha256": file_sha256(GATE_1024),
            "offline_saved_path_ordering_authority_path": relative(Path(manifest["offline_saved_path_ordering_authority"])),
            "offline_saved_path_ordering_authority_sha256": manifest["offline_saved_path_ordering_authority_sha256"],
            "preceding_E3_reject_authority_path": relative(Path(manifest["preceding_candidate_reject_authority"])),
            "preceding_E3_reject_authority_sha256": manifest["preceding_candidate_reject_authority_sha256"],
            "width_feasibility_authority_path": relative(Path(manifest["width_feasibility_authority"])),
            "width_feasibility_authority_sha256": manifest["width_feasibility_authority_sha256"],
            "source_sha256": manifest["source_sha256"],
            "model_sha256_by_task": manifest["model_sha256_by_task"],
            "candidate_reward_sha256": manifest["candidate_reward_sha256"],
            "parent_candidate_reward_sha256": manifest["parent_candidate_reward_sha256"],
            "reward_identity_reused_from_parent": manifest["reward_identity_reused_from_parent"],
            "candidate_optimizer_sha256": manifest["candidate_optimizer_sha256"],
            "parent_candidate_optimizer_sha256": manifest["parent_candidate_optimizer_sha256"],
            "optimizer_identity_reused_from_parent": manifest["optimizer_identity_reused_from_parent"],
            "candidate_mechanics_sha256": manifest["candidate_mechanics_sha256"],
            "parent_candidate_mechanics_sha256": manifest["parent_candidate_mechanics_sha256"],
            "single_changed_mechanics_parameter": manifest["single_changed_mechanics_parameter"],
        },
        "execution_disposition": {
            "checkpoint_512_decision": "continue",
            "checkpoint_1024_decision": "stop_and_reject",
            "all_three_runs_stopped_at_same_atomic_episode": True,
            "all_three_runner_statuses": "early_stopped_at_checkpoint",
            "episode_2048_was_not_executed": True,
            "interpretation": (
                "The nonzero process return codes record deliberate runner termination after the "
                "common episode-1024 atomic checkpoint and do not invalidate the saved checkpoints."
            ),
        },
        "atomic_checkpoint_integrity": {
            "all_three_valid_at_episode_1024": True,
            "all_three_config_match_historical_manifest": True,
            "all_three_training_rows": EXPECTED_EPISODE,
            "all_three_update_rows": EXPECTED_UPDATE_ROWS,
            "per_seed": integrity_rows,
        },
        "preregistered_gate": {
            "checkpoint_512": gate512,
            "checkpoint_1024": gate1024,
            "exact_reject_predicates": {
                "formal_pass": formal["pass"],
                "formal_reject": formal["reject"],
                "reward_integrity_pass": formal["reward_integrity_pass"],
                "wide_spread_seed_count": formal["tail_exact_median_spread_error_ge_0_30_seed_count"],
                "wide_spread_seed_threshold": 2,
                "tail_plane_switch_proxy_positive_seed_count": formal["tail_plane_switch_proxy_positive_seed_count"],
                "tail_valid_joint_or_strict_seed_count": formal["tail_valid_F_le_0_1_e_le_0_05_or_strict_seed_count"],
                "all_seed_tail_crossing_rate_ge_0_95": formal["all_seed_tail_crossing_rate_ge_0_95"],
            },
        },
        "reward_integrity": {
            "reward_identity_is_historical_P0X2": True,
            "candidate_reward_sha256": manifest["candidate_reward_sha256"],
            "per_seed": reward_rows,
            "pooled": pooled_reward,
        },
        "checkpoint_metrics": {"per_seed": metric_rows},
        "conclusion": {
            "adopt_width_0p2": False,
            "summary": (
                "Reducing water1 max_step_width from 0.3 to 0.2 did not produce a strict success "
                "or a tail orientation-inferred constructive endpoint by episode 1024; two of "
                "three seeds instead had tail median exact spread error at least 0.30."
            ),
        },
        "limitations": [
            "The preregistered checkpoint-1024 gate stopped all runs, so no episode-2048 or episode-10000 claim is made.",
            "No held-out final evaluation is claimed because the cohort was stopped before finalization.",
            "The plane-switch signal is an orientation-inferred endpoint proxy, not a direct PCA-plane-normal measurement.",
            "This authority binds historical source hashes from the immutable W2 manifest and intentionally does not require subsequently edited live source to match them.",
        ],
        "self_digest_scheme": {
            "field": "authority_fingerprint",
            "algorithm": "sha256",
            "payload": "canonical JSON of the report with authority_fingerprint omitted",
            "canonical_json": "UTF-8, sorted keys, separators comma/colon, ensure_ascii=false, allow_nan=false",
        },
    }
    report["authority_fingerprint"] = canonical_digest(report)
    return report


def verify_report(report: dict[str, Any]) -> None:
    content = dict(report)
    recorded = content.pop("authority_fingerprint", None)
    require(isinstance(recorded, str) and len(recorded) == 64, "authority fingerprint absent")
    require(canonical_digest(content) == recorded, "authority self-digest mismatch")


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        require(OUTPUT.is_file(), f"authority absent: {OUTPUT}")
        report = load_json(OUTPUT)
        verify_report(report)
        rebuilt = build_report()
        # Generation time is provenance, not a semantic difference during a
        # later integrity audit.
        rebuilt["generated_utc"] = report["generated_utc"]
        rebuilt["authority_fingerprint"] = report["authority_fingerprint"]
        require(equal_json(rebuilt, report), "authority content no longer reproduces from bound evidence")
        print(json.dumps({"path": relative(OUTPUT), "sha256": file_sha256(OUTPUT), "authority_fingerprint": report["authority_fingerprint"], "valid": True}, indent=2))
        return

    require(not OUTPUT.exists(), f"refusing to overwrite immutable authority: {OUTPUT}")
    report = build_report()
    verify_report(report)
    atomic_write(OUTPUT, report)
    persisted = load_json(OUTPUT)
    verify_report(persisted)
    print(json.dumps({"path": relative(OUTPUT), "sha256": file_sha256(OUTPUT), "authority_fingerprint": persisted["authority_fingerprint"], "valid": True}, indent=2))


if __name__ == "__main__":
    main()
