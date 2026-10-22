"""Fail-closed production orchestration for the ori6 spectral candidate.

The existing ori6 runner and analyzer remain byte-identical.  This additive
runner permits only the preregistered sequence:

1. a source/manifest/report-bound successful-task regression authority;
2. uniform-only random screening on all five tasks, seeds 5101--5110, with an
   exact 102400-mechanics censor per trial;
3. one 10000-episode RL seed (4101) on all five tasks;
4. if needed, a same-seed random extension to the RL first-hit mechanics, or
   to total consumed RL mechanics when RL is censored.

An independent ``two_task_full_diagnostic`` stage records the later explicit
user override for torus2 and water1 only.  It validates and binds the immutable
bounded spectral screen, but deliberately does not relabel that screen as an
adoption pass or require the separate nonregression/random sequence.

Dry-run is the default.  ``--execute`` never launches through a missing,
failed, stale, or only partially completed prerequisite.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import analyze_ori_codex6_experiments as base_analyzer
import ori_codex6_spectral_random_baseline as random_harness
import run_ori_codex6_experiments as base


SCHEMA_VERSION = 1
PRODUCTION_VERSION = "1.0.0"
MODE = base.P0_CROSS2_REJECTED_CONTINUATION_SPECTRAL_MODE
MODE_SLUG = base.MODE_SLUGS[MODE]
TASK_NAMES = tuple(base.TASKS)
PRIMARY_SEED = 4101
RANDOM_SEEDS = tuple(range(5101, 5111))
INITIAL_RANDOM_MECHANICS = 102400
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_ROOT = base.EXPERIMENT_ROOT
ANALYSIS_DIRECTORY = EXPERIMENT_ROOT / "analysis"
MANIFEST_DIRECTORY = EXPERIMENT_ROOT / "manifests"

REGRESSION_MANIFEST_PATH = (
    EXPERIMENT_ROOT / "manifests"
    / "spectral_nonregression_P0X2RCS05_22ed530e0ba3.json"
)
REGRESSION_MANIFEST_SHA256 = (
    "ba9f015c04e97c75c37d5447456d2984a4e669d9daeccab7d953cf79e2288198"
)
REGRESSION_MANIFEST_FINGERPRINT = (
    "22ed530e0ba3473990f5fd149579bc120288b8794103dedf2968fa64fa217e2a"
)
REGRESSION_REPORT_PATH = (
    ANALYSIS_DIRECTORY / "spectral_nonregression_P0X2RCS05_report.json"
)
REGRESSION_AUTHORITY_PATH = (
    ANALYSIS_DIRECTORY / "spectral_production_regression_authority.json"
)
RANDOM_SCREEN_REPORT_PATH = (
    ANALYSIS_DIRECTORY / "spectral_production_random_screen_report.json"
)

TWO_TASK_DIAGNOSTIC_STAGE = "two_task_full_diagnostic"
TWO_TASK_DIAGNOSTIC_TASKS = ("torus2", "water1")
TWO_TASK_DIAGNOSTIC_EPISODES = 10000
TWO_TASK_DIAGNOSTIC_MAX_PARALLEL = 2
TWO_TASK_SCREEN_REPORT_PATH = base.SPECTRAL_SUBSPACE_SCREEN_REPORT
TWO_TASK_SCREEN_REPORT_SHA256 = (
    "fa68e83201dd2028141c12f90d200dcdb10b080c7a02ccf5ebf8bc5e3a02af30"
)
TWO_TASK_SCREEN_REPORT_FINGERPRINT = (
    "fc30d7fb675d188b3d8ba69bb74f9f2369d1a4c63315d88b2bd1f87188de01fd"
)
TWO_TASK_SCOPE_OVERRIDE = {
    "decision_type": "explicit_user_diagnostic_scope_override",
    "scope": "torus2_and_water1_only",
    "tasks": list(TWO_TASK_DIAGNOSTIC_TASKS),
    "seed": PRIMARY_SEED,
    "episodes": TWO_TASK_DIAGNOSTIC_EPISODES,
    "spectral_weight": 0.05,
    "five_bar_regression_risk_accepted_for_this_diagnostic": True,
    "nonregression_gate_required": False,
    "random_screen_gate_required": False,
    "candidate_adoption_claim_authorized": False,
    "purpose": (
        "fresh-seed long-horizon diagnostic of whether the immutable "
        "bounded-screen signal develops into learning"
    ),
}

REGRESSION_TASKS = ("five_d2", "five_d3", "watersqrt2")
REGRESSION_REQUIRED_TASK_FLAGS = (
    "pass",
    "success_one_sided_lower_bound_strictly_gt_minus_0_10",
    "crossing_one_sided_upper_bound_strictly_lt_plus_0_10",
    "spread_precision_pass",
    "tail_collapse_gate_pass",
    "mechanics_gate_pass",
    "candidate_reward_transition_spectral_integrity_pass",
    "reference_reward_integrity_pass",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return base.file_sha256(Path(path))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def production_source_hashes() -> dict[str, str]:
    names = (
        "ori_codex6_spectral_random_baseline.py",
        "run_ori_codex6_spectral_production.py",
        "analyze_ori_codex6_spectral_production.py",
    )
    return {
        name: file_sha256(SCRIPT_DIRECTORY / name)
        for name in names if (SCRIPT_DIRECTORY / name).is_file()
    }


def canonical_self_fingerprint(payload: dict, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return canonical_digest(value)


def validate_fixed_regression_manifest(path: Path = REGRESSION_MANIFEST_PATH,
                                       expected_sha256: str =
                                       REGRESSION_MANIFEST_SHA256) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed regression manifest: {path}")
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("Fixed regression manifest SHA-256 changed")
    payload = load_json(path)
    content = dict(payload)
    recorded = content.pop("manifest_fingerprint", None)
    content.pop("created_utc", None)
    if recorded != canonical_digest(content):
        raise RuntimeError("Regression manifest self-fingerprint is invalid")
    if expected_sha256 == REGRESSION_MANIFEST_SHA256 and recorded != (
        REGRESSION_MANIFEST_FINGERPRINT
    ):
        raise RuntimeError("Regression manifest historical fingerprint changed")
    protocol = payload.get("protocol", {})
    if (
        protocol.get("name") != "spectral_nonregression"
        or protocol.get("episodes") != 2048
        or protocol.get("seeds") != [3301, 3302, 3303]
        or protocol.get("tasks") != list(REGRESSION_TASKS)
        or protocol.get("implementations") != ["ori6", "ori5_reference"]
        or payload.get("candidate_mode") != MODE
        or len(payload.get("jobs", [])) != 18
    ):
        raise RuntimeError("Regression manifest protocol identity changed")
    return payload


def validate_regression_report_payload(
    payload: dict, *, report_path: Path = REGRESSION_REPORT_PATH,
    report_sha256: str | None = None,
    manifest_path: Path = REGRESSION_MANIFEST_PATH,
    manifest_sha256: str = REGRESSION_MANIFEST_SHA256,
) -> dict:
    """Validate every promotion-relevant regression result, not just status."""

    manifest = validate_fixed_regression_manifest(
        manifest_path, manifest_sha256
    )
    errors = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version")
    if payload.get("stage") != "spectral_nonregression":
        errors.append("stage")
    if payload.get("candidate_mode") != MODE:
        errors.append("candidate_mode")
    if payload.get("candidate_mode_slug") != MODE_SLUG:
        errors.append("candidate_mode_slug")
    if payload.get("status") != "pass":
        errors.append("status")
    if payload.get("successful_task_regression_gate_pass") is not True:
        errors.append("overall_gate")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 18:
        errors.append("records_18")
    else:
        observed = {
            (
                row.get("task"), row.get("implementation"),
                int(row.get("seed", -1)), int(row.get("endpoint_episode", -1)),
            ) for row in records
        }
        expected = {
            (task, implementation, seed, 2048)
            for task in REGRESSION_TASKS
            for implementation in ("ori6", "ori5_reference")
            for seed in (3301, 3302, 3303)
        }
        if observed != expected:
            errors.append("record_cohort")
    task_results = payload.get("task_results")
    if not isinstance(task_results, dict) or set(task_results) != set(
        REGRESSION_TASKS
    ):
        errors.append("task_result_scope")
    else:
        for task_name in REGRESSION_TASKS:
            result = task_results[task_name]
            for flag in REGRESSION_REQUIRED_TASK_FLAGS:
                if result.get(flag) is not True:
                    errors.append(f"{task_name}.{flag}")
            checkpoint = result.get("checkpoint_512_authority", {})
            if checkpoint.get("valid_and_passed") is not True:
                errors.append(f"{task_name}.checkpoint_512")
    manifest_authority = payload.get("manifest_authority", {})
    if (
        Path(str(manifest_authority.get("path", ""))).resolve()
        != Path(manifest_path).resolve()
        or manifest_authority.get("sha256") != manifest_sha256
        or manifest_authority.get("manifest_fingerprint")
        != manifest.get("manifest_fingerprint")
    ):
        errors.append("manifest_authority")
    if payload.get("candidate_reward_sha256") != (
        base.candidate_reward_sha256(MODE)
    ):
        errors.append("candidate_reward")
    if payload.get("spectral_additive_source_sha256") != (
        base.spectral_subspace_additive_source_hashes()
    ):
        errors.append("spectral_sources")
    if payload.get("parent_additive_source_sha256") != (
        base.rejected_crossing_additive_source_hashes()
    ):
        errors.append("parent_sources")
    if payload.get("source_sha256") != base.source_hashes():
        errors.append("base_sources")
    if report_sha256 is not None and file_sha256(report_path) != report_sha256:
        errors.append("report_sha256")
    if errors:
        raise RuntimeError(
            "Regression report is not a promotion authority: "
            + ", ".join(errors)
        )
    return {
        "valid": True,
        "tasks": list(REGRESSION_TASKS),
        "records": 18,
        "manifest_sha256": manifest_sha256,
        "report_sha256": (
            report_sha256
            if report_sha256 is not None else
            file_sha256(report_path) if Path(report_path).is_file() else None
        ),
    }


def validate_regression_authority(
    authority_path: Path = REGRESSION_AUTHORITY_PATH,
) -> dict:
    authority_path = Path(authority_path)
    if not authority_path.is_file():
        raise FileNotFoundError(
            f"Regression promotion authority is pending: {authority_path}"
        )
    authority = load_json(authority_path)
    if authority.get("authority_fingerprint") != canonical_self_fingerprint(
        authority, "authority_fingerprint"
    ):
        raise RuntimeError("Regression authority self-fingerprint mismatch")
    report = authority.get("regression_report", {})
    manifest = authority.get("regression_manifest", {})
    if (
        authority.get("authority_type")
        != "ori6_spectral_production_regression_promotion"
        or authority.get("status") != "pass"
        or authority.get("candidate_mode") != MODE
        or Path(str(report.get("path", ""))).resolve()
        != REGRESSION_REPORT_PATH.resolve()
        or Path(str(manifest.get("path", ""))).resolve()
        != REGRESSION_MANIFEST_PATH.resolve()
        or manifest.get("sha256") != REGRESSION_MANIFEST_SHA256
        or manifest.get("manifest_fingerprint")
        != REGRESSION_MANIFEST_FINGERPRINT
    ):
        raise RuntimeError("Regression authority identity mismatch")
    report_sha = report.get("sha256")
    if not isinstance(report_sha, str) or len(report_sha) != 64:
        raise RuntimeError("Regression report historical SHA is not bound")
    if file_sha256(REGRESSION_REPORT_PATH) != report_sha:
        raise RuntimeError("Regression report changed after promotion")
    payload = load_json(REGRESSION_REPORT_PATH)
    validate_regression_report_payload(
        payload, report_path=REGRESSION_REPORT_PATH,
        report_sha256=report_sha,
    )
    if report.get("canonical_sha256") != canonical_digest(payload):
        raise RuntimeError("Regression report canonical digest mismatch")
    return authority


def prerequisite_status() -> dict:
    """Return a non-throwing status suitable for dry-run planning."""

    try:
        authority = validate_regression_authority()
    except FileNotFoundError as error:
        return {
            "status": "pending_not_launchable",
            "pass": False,
            "reason": str(error),
        }
    except Exception as error:
        return {
            "status": "rejected_not_launchable",
            "pass": False,
            "reason": repr(error),
        }
    return {
        "status": "pass",
        "pass": True,
        "authority_path": str(REGRESSION_AUTHORITY_PATH.resolve()),
        "authority_sha256": file_sha256(REGRESSION_AUTHORITY_PATH),
        "authority_fingerprint": authority["authority_fingerprint"],
        "regression_report_sha256": authority["regression_report"]["sha256"],
    }


def validate_two_task_screen_authority(
    path: Path = TWO_TASK_SCREEN_REPORT_PATH,
) -> dict:
    """Bind the immutable screen plus the separate user scope override.

    The historical report remains a formal adoption rejection and is never
    relabelled as a 10000-episode authorization.  The new diagnostic is
    launchable only because the user supplied a narrower, explicit decision
    after that report: exactly torus2 and water1, seed 4101, 10000 episodes.
    """

    path = Path(path)
    if path.resolve() != Path(TWO_TASK_SCREEN_REPORT_PATH).resolve():
        raise RuntimeError("Two-task spectral screen report path mismatch")
    if not path.is_file():
        raise FileNotFoundError(
            f"Two-task spectral screen report is missing: {path}"
        )
    if file_sha256(path) != TWO_TASK_SCREEN_REPORT_SHA256:
        raise RuntimeError("Two-task spectral screen report SHA-256 changed")
    if TWO_TASK_SCREEN_REPORT_SHA256 != (
        base.SPECTRAL_SUBSPACE_SCREEN_REPORT_SHA256
    ):
        raise RuntimeError("Runner/base spectral screen SHA binding disagrees")
    if TWO_TASK_SCREEN_REPORT_FINGERPRINT != (
        base.SPECTRAL_SUBSPACE_SCREEN_REPORT_FINGERPRINT
    ):
        raise RuntimeError(
            "Runner/base spectral screen fingerprint binding disagrees"
        )
    payload = base.validate_spectral_screen_report_authority(path)
    if payload.get("authority_fingerprint") != (
        TWO_TASK_SCREEN_REPORT_FINGERPRINT
    ):
        raise RuntimeError("Two-task spectral screen self-digest changed")
    scope = payload.get("scope", {})
    if (
        scope.get("tasks") != list(TWO_TASK_DIAGNOSTIC_TASKS)
        or payload.get("spectral_subspace_screen_pass") is not False
        or payload.get("candidate_adopted") is not False
        or payload.get("full_10000_episode_training_authorized_by_this_report")
        is not False
    ):
        raise RuntimeError("Two-task screen disposition/scope changed")
    if (
        TWO_TASK_SCOPE_OVERRIDE["tasks"]
        != list(TWO_TASK_DIAGNOSTIC_TASKS)
        or TWO_TASK_SCOPE_OVERRIDE["seed"] != PRIMARY_SEED
        or TWO_TASK_SCOPE_OVERRIDE["episodes"]
        != TWO_TASK_DIAGNOSTIC_EPISODES
        or not base.values_equal(
            TWO_TASK_SCOPE_OVERRIDE["spectral_weight"],
            base.SPECTRAL_SUBSPACE_WEIGHT,
        )
    ):
        raise RuntimeError("Two-task user override constants are inconsistent")
    historical = payload.get("historical_binding", {})
    return {
        "status": "pass_user_scope_override",
        "pass": True,
        "screen_report_path": str(path.resolve()),
        "screen_report_sha256": TWO_TASK_SCREEN_REPORT_SHA256,
        "screen_report_authority_fingerprint": (
            TWO_TASK_SCREEN_REPORT_FINGERPRINT
        ),
        "screen_report_canonical_sha256": canonical_digest(payload),
        "screen_report_formal_decision": payload.get("decision"),
        "screen_report_authorizes_10000_episodes": False,
        "screen_historical_binding_sha256": canonical_digest(historical),
        "user_scope_override": dict(TWO_TASK_SCOPE_OVERRIDE),
        "user_scope_override_sha256": canonical_digest(
            TWO_TASK_SCOPE_OVERRIDE
        ),
    }


def random_output_root(stage: str, task_name: str, budget: int) -> Path:
    if stage not in ("screen", "extension"):
        raise ValueError(stage)
    return (
        EXPERIMENT_ROOT / "random" / MODE_SLUG / stage / task_name
        / f"budget{int(budget)}"
    ).resolve()


def build_random_job(stage: str, task_name: str, budget: int,
                     regression_gate: dict) -> dict:
    task = base.TASKS[task_name]
    destination = random_output_root(stage, task_name, budget)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "production_version": PRODUCTION_VERSION,
        "stage": f"random_{stage}",
        "candidate_mode": MODE,
        "candidate_mode_slug": MODE_SLUG,
        "task": task_name,
        "task_specification": asdict(task),
        "profile": asdict(base.PROFILES[task.profile]),
        "method": "uniform",
        "seeds": list(RANDOM_SEEDS),
        "mechanics_budget": int(budget),
        "output_root": str(destination),
        "candidate_reward_sha256": base.candidate_reward_sha256(MODE),
        "transition_semantics_sha256": (
            base.candidate_transition_semantics_sha256(MODE)
        ),
        "model_sha256": base.model_hashes(task),
        "base_source_sha256": base.source_hashes(),
        "candidate_additive_source_sha256": {
            **base.rejected_crossing_additive_source_hashes(),
            **base.spectral_subspace_additive_source_hashes(),
        },
        "random_harness_sha256": file_sha256(
            SCRIPT_DIRECTORY / "ori_codex6_spectral_random_baseline.py"
        ),
        "regression_authority_fingerprint": regression_gate[
            "authority_fingerprint"
        ],
    }
    fingerprint = canonical_digest(identity)
    command = [
        sys.executable,
        str(SCRIPT_DIRECTORY / "ori_codex6_spectral_random_baseline.py"),
        "--task", task_name,
        "--mechanics-budget", str(int(budget)),
        "--trial-seeds", *[str(seed) for seed in RANDOM_SEEDS],
        "--output-root", str(destination),
        "--checkpoint-every-episodes", "128",
        "--print-every", "100",
    ]
    return {
        **identity,
        "job_id": f"random_{stage}:{MODE_SLUG}:{task_name}:budget{budget}",
        "fingerprint": fingerprint,
        "command": command,
    }


def random_screen_jobs(regression_gate: dict) -> list[dict]:
    return [
        build_random_job(
            "screen", task_name, INITIAL_RANDOM_MECHANICS,
            regression_gate,
        ) for task_name in TASK_NAMES
    ]


def validate_random_artifact(job: dict, require_complete: bool = True) -> dict:
    root = Path(job["output_root"])
    config = random_harness.load_root(root)
    if (
        config.get("task") != job["task"]
        or config.get("seeds") != list(RANDOM_SEEDS)
        or config.get("mechanics_budget") != job["mechanics_budget"]
        or config.get("candidate") != base.SPECTRAL_SUBSPACE_CANDIDATE_NAME
        or config.get("model_sha256") != job["model_sha256"]
        or config.get("reward_sha256") != job["candidate_reward_sha256"]
        or config.get("source_sha256", {}).get(
            "ori_codex6_spectral_random_baseline.py"
        ) != job["random_harness_sha256"]
    ):
        raise RuntimeError(f"Random artifact identity mismatch: {root}")
    aggregate_path = root / "aggregate.json"
    if not aggregate_path.is_file():
        if require_complete:
            raise RuntimeError(f"Missing random aggregate: {aggregate_path}")
        return {"valid": False, "complete": False, "output_root": str(root)}
    aggregate = load_json(aggregate_path)
    if (
        aggregate.get("root_fingerprint") != config["root_fingerprint"]
        or aggregate.get("task") != job["task"]
        or aggregate.get("mechanics_budget") != job["mechanics_budget"]
        or aggregate.get("seeds") != list(RANDOM_SEEDS)
        or aggregate.get("expected_trials") != len(RANDOM_SEEDS)
        or aggregate.get("completed_trials") != len(RANDOM_SEEDS)
        or aggregate.get("complete") is not True
        or aggregate.get("all_reward_integrity_pass") is not True
    ):
        raise RuntimeError(f"Random aggregate failed validation: {root}")
    return {
        "valid": True,
        "complete": True,
        "output_root": str(root),
        "run_config_sha256": file_sha256(root / "run_config.json"),
        "aggregate_sha256": file_sha256(aggregate_path),
        "root_fingerprint": config["root_fingerprint"],
        "aggregate": aggregate,
    }


def validate_random_screen_report(
    path: Path = RANDOM_SCREEN_REPORT_PATH,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Random screen report is pending: {path}")
    payload = load_json(path)
    if payload.get("report_fingerprint") != canonical_self_fingerprint(
        payload, "report_fingerprint"
    ):
        raise RuntimeError("Random screen report self-fingerprint mismatch")
    if (
        payload.get("report_type")
        != "ori6_spectral_production_random_screen"
        or payload.get("status") != "pass"
        or payload.get("all_five_tasks_complete") is not True
        or payload.get("candidate_mode") != MODE
        or payload.get("seeds") != list(RANDOM_SEEDS)
        or payload.get("mechanics_budget") != INITIAL_RANDOM_MECHANICS
        or set(payload.get("task_results", {})) != set(TASK_NAMES)
    ):
        raise RuntimeError("Random screen report does not authorize full RL")
    if (
        payload.get("method") != "uniform"
        or payload.get("action_law") != random_harness.ACTION_LAW
        or payload.get("candidate_reward_sha256")
        != base.candidate_reward_sha256(MODE)
        or payload.get("transition_semantics_sha256")
        != base.candidate_transition_semantics_sha256(MODE)
        or payload.get("base_source_sha256") != base.source_hashes()
        or payload.get("candidate_additive_source_sha256") != {
            **base.rejected_crossing_additive_source_hashes(),
            **base.spectral_subspace_additive_source_hashes(),
        }
        or payload.get("random_harness_sha256") != file_sha256(
            SCRIPT_DIRECTORY / "ori_codex6_spectral_random_baseline.py"
        )
    ):
        raise RuntimeError("Random screen report source/protocol binding changed")
    regression = validate_regression_authority()
    if payload.get("regression_authority_fingerprint") != regression.get(
        "authority_fingerprint"
    ):
        raise RuntimeError("Random report regression authority is stale")
    gate = prerequisite_status()
    jobs = random_screen_jobs(gate)
    for job in jobs:
        bound = payload["task_results"][job["task"]]
        validation = validate_random_artifact(job)
        aggregate = validation["aggregate"]
        budget = int(aggregate["mechanics_budget"])
        assigned = [
            int(row["first_success_mechanical_steps"])
            if row["success"] else budget
            for row in aggregate["trial_summaries"]
        ]
        rmst = sum(assigned) / len(assigned)
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
        if (
            bound.get("pass") is not True
            or bound.get("aggregate_sha256")
            != validation["aggregate_sha256"]
            or bound.get("run_config_sha256")
            != validation["run_config_sha256"]
            or bound.get("root_fingerprint")
            != validation["root_fingerprint"]
            or bound.get("mechanics_budget") != budget
            or bound.get("seeds") != list(RANDOM_SEEDS)
            or bound.get("trials") != len(RANDOM_SEEDS)
            or bound.get("successes") != aggregate["successes"]
            or bound.get("censored") != aggregate["censored"]
            or bound.get("success_at_budget")
            != aggregate["success_at_budget"]
            or bound.get("first_success_mechanical_steps")
            != aggregate["first_success_mechanical_steps"]
            or bound.get("assigned_first_hit_or_censor_mechanics")
            != assigned
            or bound.get("per_seed") != per_seed
            or bound.get("all_reward_integrity_pass") is not True
            or not isinstance(bound.get("mechanics_rmst"), dict)
            or not base.values_equal(
                bound["mechanics_rmst"].get("estimate"), rmst
            )
        ):
            raise RuntimeError(
                f"Random screen artifact changed for {job['task']}"
            )
    return payload


def random_screen_status() -> dict:
    try:
        report = validate_random_screen_report()
    except FileNotFoundError as error:
        return {
            "status": "pending_not_launchable",
            "pass": False,
            "reason": str(error),
        }
    except Exception as error:
        return {
            "status": "rejected_not_launchable",
            "pass": False,
            "reason": repr(error),
        }
    return {
        "status": "pass",
        "pass": True,
        "report_path": str(RANDOM_SCREEN_REPORT_PATH.resolve()),
        "report_sha256": file_sha256(RANDOM_SCREEN_REPORT_PATH),
        "report_fingerprint": report["report_fingerprint"],
    }


def full_protocol() -> base.Protocol:
    return base.Protocol(
        "full_primary", 10000, (PRIMARY_SEED,), TASK_NAMES, ("ori6",),
        500, 25, 100, 1024, 100, 8310000, 8410000,
    )


def build_full_jobs(regression_gate: dict,
                    random_gate: dict) -> list[dict]:
    protocol = full_protocol()
    hashes = base.source_hashes()
    jobs = []
    for task_name in TASK_NAMES:
        task = base.TASKS[task_name]
        profile = base.PROFILES[task.profile]
        destination = base.output_directory(
            protocol, MODE, "ori6", task, PRIMARY_SEED
        )
        identity = {
            "schema_version": 1,
            "protocol": asdict(protocol),
            "task": asdict(task),
            "profile": asdict(profile),
            "seed": PRIMARY_SEED,
            "implementation": "ori6",
            "candidate_mode": MODE,
            "output_directory": str(destination),
            "expected_config": base.expected_config(
                task, profile, protocol, PRIMARY_SEED, "ori6", MODE
            ),
            "source_sha256": hashes,
            "model_sha256": base.model_hashes(task),
            "core_frontier_mode": base.core_frontier_mode(MODE),
            "candidate_reward_sha256": base.candidate_reward_sha256(MODE),
            "candidate_optimizer_sha256": (
                base.candidate_optimizer_sha256(MODE, profile)
            ),
            "candidate_mechanics_sha256": (
                base.candidate_mechanics_sha256(MODE, task)
            ),
            "candidate_transition_semantics_sha256": (
                base.candidate_transition_semantics_sha256(MODE)
            ),
            "candidate_additive_source_sha256": {
                **base.rejected_crossing_additive_source_hashes(),
                **base.spectral_subspace_additive_source_hashes(),
            },
            "spectral_additive_source_sha256": (
                base.spectral_subspace_additive_source_hashes()
            ),
            "ori6_candidate": base.SPECTRAL_SUBSPACE_CANDIDATE_NAME,
            "ori6_parent_candidate": base.REJECTED_CROSSING_CANDIDATE_NAME,
            "ori6_crossing_policy": base.REJECTED_CROSSING_POLICY,
            "spectral_formula_sha256": (
                base.SPECTRAL_SUBSPACE_FORMULA_SHA256
            ),
            "spectral_weight": base.SPECTRAL_SUBSPACE_WEIGHT,
            "regression_authority_fingerprint": regression_gate[
                "authority_fingerprint"
            ],
            "random_screen_report_fingerprint": random_gate[
                "report_fingerprint"
            ],
        }
        fingerprint = canonical_digest(identity)
        identity.update({
            "job_id": (
                f"spectral_full_primary:{MODE_SLUG}:ori6:{task_name}:"
                f"{profile.name}:seed{PRIMARY_SEED}"
            ),
            "fingerprint": fingerprint,
        })
        identity["command"] = base.build_command(identity, destination)
        jobs.append(identity)
    return jobs


def two_task_diagnostic_protocol() -> base.Protocol:
    return base.Protocol(
        TWO_TASK_DIAGNOSTIC_STAGE,
        TWO_TASK_DIAGNOSTIC_EPISODES,
        (PRIMARY_SEED,),
        TWO_TASK_DIAGNOSTIC_TASKS,
        ("ori6",),
        500, 25, 100, 1024, 100, 8310000, 8410000,
    )


def two_task_diagnostic_output_directory(
    task: base.Task, profile: base.Profile,
) -> Path:
    return base.ensure_below_root(
        EXPERIMENT_ROOT / "full" / TWO_TASK_DIAGNOSTIC_STAGE
        / MODE_SLUG / task.name / profile.name / f"seed{PRIMARY_SEED}"
    ).resolve()


def build_two_task_diagnostic_jobs(screen_gate: dict) -> list[dict]:
    """Build the exact two-job fresh-seed long-horizon diagnostic."""

    if (
        screen_gate.get("pass") is not True
        or screen_gate.get("screen_report_sha256")
        != TWO_TASK_SCREEN_REPORT_SHA256
        or screen_gate.get("screen_report_authority_fingerprint")
        != TWO_TASK_SCREEN_REPORT_FINGERPRINT
        or screen_gate.get("user_scope_override_sha256")
        != canonical_digest(TWO_TASK_SCOPE_OVERRIDE)
    ):
        raise RuntimeError("Invalid two-task diagnostic screen/override gate")
    protocol = two_task_diagnostic_protocol()
    hashes = base.source_hashes()
    jobs = []
    for task_name in TWO_TASK_DIAGNOSTIC_TASKS:
        task = base.TASKS[task_name]
        profile = base.PROFILES[task.profile]
        destination = two_task_diagnostic_output_directory(task, profile)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "production_version": PRODUCTION_VERSION,
            "stage": TWO_TASK_DIAGNOSTIC_STAGE,
            "protocol": asdict(protocol),
            "task": asdict(task),
            "profile": asdict(profile),
            "seed": PRIMARY_SEED,
            "implementation": "ori6",
            "candidate_mode": MODE,
            "candidate_mode_slug": MODE_SLUG,
            "output_directory": str(destination),
            "expected_config": base.expected_config(
                task, profile, protocol, PRIMARY_SEED, "ori6", MODE
            ),
            "source_sha256": hashes,
            "model_sha256": base.model_hashes(task),
            "core_frontier_mode": base.core_frontier_mode(MODE),
            "candidate_reward_sha256": base.candidate_reward_sha256(MODE),
            "candidate_optimizer_sha256": (
                base.candidate_optimizer_sha256(MODE, profile)
            ),
            "candidate_mechanics_sha256": (
                base.candidate_mechanics_sha256(MODE, task)
            ),
            "candidate_transition_semantics_sha256": (
                base.candidate_transition_semantics_sha256(MODE)
            ),
            "candidate_additive_source_sha256": {
                **base.rejected_crossing_additive_source_hashes(),
                **base.spectral_subspace_additive_source_hashes(),
            },
            "parent_additive_source_sha256": (
                base.rejected_crossing_additive_source_hashes()
            ),
            "spectral_additive_source_sha256": (
                base.spectral_subspace_additive_source_hashes()
            ),
            "production_runner_sha256": file_sha256(Path(__file__)),
            "ori6_candidate": base.SPECTRAL_SUBSPACE_CANDIDATE_NAME,
            "ori6_parent_candidate": base.REJECTED_CROSSING_CANDIDATE_NAME,
            "ori6_crossing_policy": base.REJECTED_CROSSING_POLICY,
            "spectral_formula_sha256": (
                base.SPECTRAL_SUBSPACE_FORMULA_SHA256
            ),
            "spectral_weight": base.SPECTRAL_SUBSPACE_WEIGHT,
            "spectral_relative_epsilon": (
                base.SPECTRAL_SUBSPACE_RELATIVE_EPSILON
            ),
            "screen_report_path": screen_gate["screen_report_path"],
            "screen_report_sha256": screen_gate["screen_report_sha256"],
            "screen_report_authority_fingerprint": screen_gate[
                "screen_report_authority_fingerprint"
            ],
            "screen_report_canonical_sha256": screen_gate[
                "screen_report_canonical_sha256"
            ],
            "screen_historical_binding_sha256": screen_gate[
                "screen_historical_binding_sha256"
            ],
            "user_scope_override": dict(TWO_TASK_SCOPE_OVERRIDE),
            "user_scope_override_sha256": screen_gate[
                "user_scope_override_sha256"
            ],
            "fresh_output_policy": (
                "dedicated two_task_full_diagnostic root; never reuse "
                "full_primary or bounded-screen outputs"
            ),
        }
        fingerprint = canonical_digest(identity)
        identity.update({
            "job_id": (
                f"{TWO_TASK_DIAGNOSTIC_STAGE}:{MODE_SLUG}:ori6:"
                f"{task.name}:{profile.name}:seed{PRIMARY_SEED}"
            ),
            "fingerprint": fingerprint,
        })
        identity["command"] = base.build_command(identity, destination)
        jobs.append(identity)
    if (
        len(jobs) != 2
        or [job["task"]["name"] for job in jobs]
        != list(TWO_TASK_DIAGNOSTIC_TASKS)
    ):
        raise RuntimeError("Two-task diagnostic cohort invariant failed")
    return jobs


def two_task_diagnostic_artifact_records(
    require_complete: bool = True,
) -> list[dict]:
    screen_gate = validate_two_task_screen_authority()
    records = []
    for job in build_two_task_diagnostic_jobs(screen_gate):
        destination = Path(job["output_directory"])
        effective, validation = base.discover_output(job, destination)
        if effective is None:
            if require_complete:
                raise RuntimeError(
                    f"Incomplete two-task diagnostic job: {job['job_id']}; "
                    f"validation={validation}"
                )
            continue
        records.append(base_analyzer.summarize_job(
            job, effective, validation, "complete"
        ))
    if require_complete and len(records) != len(TWO_TASK_DIAGNOSTIC_TASKS):
        raise RuntimeError(
            "Two-task diagnostic requires exactly two complete artifacts"
        )
    return records


def full_artifact_records(require_complete: bool = True) -> list[dict]:
    regression = prerequisite_status()
    random_gate = random_screen_status()
    if not regression["pass"] or not random_gate["pass"]:
        raise RuntimeError("Full artifact identity prerequisites are absent")
    records = []
    for job in build_full_jobs(regression, random_gate):
        destination = Path(job["output_directory"])
        effective, validation = base.discover_output(job, destination)
        if effective is None:
            if require_complete:
                raise RuntimeError(f"Incomplete full job: {job['job_id']}")
            continue
        records.append(base_analyzer.summarize_job(
            job, effective, validation, "complete"
        ))
    if require_complete and len(records) != len(TASK_NAMES):
        raise RuntimeError("Full primary requires five complete artifacts")
    return records


def extension_budget_by_task() -> dict[str, int]:
    records = full_artifact_records(require_complete=True)
    budgets = {}
    for record in records:
        hit = record["first_hit"]
        budget = (
            int(hit["mechanical_steps"])
            if not hit["censored"]
            else int(record["training"]["total_mechanical_steps"])
        )
        if budget > INITIAL_RANDOM_MECHANICS:
            budgets[record["task"]] = budget
    return budgets


def random_extension_jobs(regression_gate: dict) -> list[dict]:
    # Full completion already implies a valid random-screen authority.
    validate_random_screen_report()
    return [
        build_random_job("extension", task_name, budget, regression_gate)
        for task_name, budget in sorted(extension_budget_by_task().items())
    ]


def manifest_payload(stage: str, jobs: list[dict], gates: dict) -> dict:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "production_version": PRODUCTION_VERSION,
        "stage": stage,
        "candidate_mode": MODE,
        "candidate_mode_slug": MODE_SLUG,
        "jobs": jobs,
        "gates": gates,
        "base_source_sha256": base.source_hashes(),
        "candidate_additive_source_sha256": {
            **base.rejected_crossing_additive_source_hashes(),
            **base.spectral_subspace_additive_source_hashes(),
        },
        "production_source_sha256": production_source_hashes(),
        "model_sha256_by_task": {
            name: base.model_hashes(task)
            for name, task in base.TASKS.items()
        },
        "recovery_semantics": (
            "uniform exact checkpoint resume; RL completed-skip or "
            "preserved-incomplete fresh retry"
        ),
    }
    fingerprint = canonical_digest(identity)
    return {
        **identity,
        "manifest_fingerprint": fingerprint,
        "created_utc": utc_now(),
    }


def write_immutable_manifest(stage: str, jobs: list[dict], gates: dict) -> Path:
    payload = manifest_payload(stage, jobs, gates)
    path = MANIFEST_DIRECTORY / (
        f"spectral_production_{stage}_{payload['manifest_fingerprint'][:12]}.json"
    )
    if path.is_file():
        existing = load_json(path)
        left = dict(existing)
        right = dict(payload)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        # JSON serialisation turns dataclass tuple fields into arrays.  Compare
        # their canonical JSON identities so rerunning the same dry-run does
        # not mistake that representation-only change for a collision.
        if canonical_digest(left) != canonical_digest(right):
            raise RuntimeError(f"Immutable manifest collision: {path}")
    else:
        base.atomic_json(path, payload)
    return path


def validate_immutable_manifest(stage: str, jobs: list[dict],
                                gates: dict) -> dict:
    """Validate the exact current immutable production manifest."""

    expected = manifest_payload(stage, jobs, gates)
    path = MANIFEST_DIRECTORY / (
        f"spectral_production_{stage}_"
        f"{expected['manifest_fingerprint'][:12]}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing production manifest: {path}")
    observed = load_json(path)
    if observed.get("manifest_fingerprint") != expected[
        "manifest_fingerprint"
    ]:
        raise RuntimeError("Production manifest fingerprint mismatch")
    observed_identity = dict(observed)
    expected_identity = dict(expected)
    observed_identity.pop("created_utc", None)
    expected_identity.pop("created_utc", None)
    if canonical_digest(observed_identity) != canonical_digest(
        expected_identity
    ):
        raise RuntimeError("Production manifest content/source binding changed")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "manifest_fingerprint": observed["manifest_fingerprint"],
    }


def random_status_path(job: dict) -> Path:
    return (
        EXPERIMENT_ROOT / "_runner_logs" / "spectral_production"
        / job["stage"] / job["task"] / "status.json"
    )


def load_random_status(job: dict) -> dict | None:
    path = random_status_path(job)
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def classify_random_job(job: dict) -> tuple[str, dict | None]:
    try:
        validation = validate_random_artifact(job)
        return "complete", validation
    except FileNotFoundError:
        pass
    except RuntimeError as error:
        # Only a correctly identified root that has not emitted its final
        # aggregate is resumable.  Configuration/source mismatches fail closed.
        if "Missing random aggregate:" not in str(error):
            return "invalid", {"error": repr(error)}
    status = load_random_status(job)
    if status and status.get("status") == "running":
        try:
            if base.process_is_alive(int(status.get("pid", -1))):
                return "running", status
        except (TypeError, ValueError):
            pass
    return (
        "resumable" if Path(job["output_root"]).exists() else "pending",
        status,
    )


def launch_random(job: dict) -> dict:
    status_file = random_status_path(job)
    log_directory = status_file.parent
    log_directory.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "job_fingerprint": job["fingerprint"],
        "status": "launching",
        "started_utc": utc_now(),
        "command": job["command"],
        "output_root": job["output_root"],
    }
    base.atomic_json(status_file, status)
    stdout = open(log_directory / "stdout.log", "a", encoding="utf-8")
    stderr = open(log_directory / "stderr.log", "a", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(RANDOM_SEEDS[0]),
    })
    process = subprocess.Popen(
        job["command"], cwd=SCRIPT_DIRECTORY,
        stdout=stdout, stderr=stderr, env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    status.update({"status": "running", "pid": int(process.pid)})
    base.atomic_json(status_file, status)
    return {
        "job": job, "process": process, "status": status,
        "status_file": status_file, "stdout": stdout, "stderr": stderr,
        "started": time.monotonic(),
    }


def execute_random(jobs: list[dict], max_parallel: int) -> int:
    queue = list(jobs)
    active = []
    failed = 0
    while queue or active:
        while queue and len(active) < max_parallel:
            job = queue.pop(0)
            print(f"LAUNCH {job['job_id']} -> {job['output_root']}", flush=True)
            active.append(launch_random(job))
        for item in list(active):
            process = item["process"]
            if process.poll() is None:
                continue
            item["stdout"].close()
            item["stderr"].close()
            try:
                validation = validate_random_artifact(item["job"])
                successful = process.returncode == 0 and validation["valid"]
            except Exception as error:
                validation = {"valid": False, "error": repr(error)}
                successful = False
            status = dict(item["status"])
            status.update({
                "status": "complete" if successful else "failed",
                "finished_utc": utc_now(),
                "return_code": int(process.returncode),
                "runner_wall_seconds": time.monotonic() - item["started"],
                "artifact_validation": validation,
            })
            base.atomic_json(item["status_file"], status)
            active.remove(item)
            if not successful:
                failed += 1
            print(
                f"{status['status'].upper()} {item['job']['job_id']} "
                f"rc={process.returncode}", flush=True,
            )
        if active:
            time.sleep(1)
    return failed


def classify_full_jobs(jobs: list[dict], rerun_incomplete: bool):
    processes = base.active_python_processes()
    rows = []
    pending = []
    for job in jobs:
        state, destination, details = base.classify(
            job, rerun_incomplete, processes
        )
        rows.append({
            "job_id": job["job_id"], "state": state,
            "destination": str(destination), "details": details,
        })
        if state in ("pending", "pending_retry"):
            pending.append((job, destination))
    return rows, pending


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "random_screen", "full_primary", "random_extension",
            TWO_TASK_DIAGNOSTIC_STAGE,
        ),
        required=True,
    )
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    if not 1 <= options.max_parallel <= 12:
        raise ValueError("--max-parallel must be between 1 and 12")

    # This independent path must be handled before the original production
    # prerequisite chain.  Its only authority is the exact immutable screen
    # plus the explicit, narrowly recorded user override above.
    if options.stage == TWO_TASK_DIAGNOSTIC_STAGE:
        if options.max_parallel != TWO_TASK_DIAGNOSTIC_MAX_PARALLEL:
            raise ValueError(
                "two_task_full_diagnostic requires --max-parallel 2"
            )
        screen_gate = validate_two_task_screen_authority()
        gates = {"spectral_screen_user_scope_override": screen_gate}
        jobs = build_two_task_diagnostic_jobs(screen_gate)
        manifest = write_immutable_manifest(options.stage, jobs, gates)
        states, pending = classify_full_jobs(
            jobs, options.rerun_incomplete
        )
        print(f"Manifest: {manifest}")
        print(json.dumps({
            "stage": options.stage,
            "exact_job_count": len(jobs),
            "max_parallel": options.max_parallel,
            "regression_gate_required": False,
            "random_screen_gate_required": False,
            "states": states,
        }, indent=2))
        blocking = [
            row for row in states
            if row["state"] not in ("complete", "pending", "pending_retry")
        ]
        if blocking:
            raise RuntimeError(
                f"Two-task diagnostic jobs have blocking states: {blocking}"
            )
        if options.execute:
            return base.execute(
                pending, early_stop=False,
                max_parallel=TWO_TASK_DIAGNOSTIC_MAX_PARALLEL,
            )
        for job, destination in pending:
            print(
                "DRY RUN "
                + subprocess.list2cmdline(base.build_command(job, destination))
            )
        return 0

    regression_gate = prerequisite_status()
    gates = {"regression": regression_gate}
    if not regression_gate["pass"]:
        print(json.dumps({
            "stage": options.stage,
            "status": "pending_not_launchable",
            "gates": gates,
            "execute_requested": options.execute,
        }, indent=2))
        if options.execute:
            raise RuntimeError(regression_gate["reason"])
        return 0

    if options.stage == "random_screen":
        jobs = random_screen_jobs(regression_gate)
        manifest = write_immutable_manifest(options.stage, jobs, gates)
        states = []
        pending = []
        for job in jobs:
            state, details = classify_random_job(job)
            states.append({
                "job_id": job["job_id"], "state": state,
                "output_root": job["output_root"], "details": details,
            })
            if state in ("pending", "resumable"):
                pending.append(job)
        print(f"Manifest: {manifest}")
        print(json.dumps({"states": states}, indent=2))
        if any(row["state"] == "invalid" for row in states):
            raise RuntimeError("Invalid completed random artifact exists")
        if options.execute:
            return execute_random(pending, options.max_parallel)
        for job in pending:
            print("DRY RUN " + subprocess.list2cmdline(job["command"]))
        return 0

    random_gate = random_screen_status()
    gates["random_screen"] = random_gate
    if not random_gate["pass"]:
        print(json.dumps({
            "stage": options.stage,
            "status": "pending_not_launchable",
            "gates": gates,
            "execute_requested": options.execute,
        }, indent=2))
        if options.execute:
            raise RuntimeError(random_gate["reason"])
        return 0

    if options.stage == "full_primary":
        jobs = build_full_jobs(regression_gate, random_gate)
        manifest = write_immutable_manifest(options.stage, jobs, gates)
        states, pending = classify_full_jobs(jobs, options.rerun_incomplete)
        print(f"Manifest: {manifest}")
        print(json.dumps({"states": states}, indent=2))
        blocking = [
            row for row in states
            if row["state"] not in ("complete", "pending", "pending_retry")
        ]
        if blocking:
            raise RuntimeError(f"Full jobs have blocking states: {blocking}")
        if options.execute:
            return base.execute(
                pending, early_stop=False,
                max_parallel=options.max_parallel,
            )
        for job, destination in pending:
            print(
                "DRY RUN "
                + subprocess.list2cmdline(base.build_command(job, destination))
            )
        return 0

    jobs = random_extension_jobs(regression_gate)
    manifest = write_immutable_manifest(options.stage, jobs, gates)
    states = []
    pending = []
    for job in jobs:
        state, details = classify_random_job(job)
        states.append({
            "job_id": job["job_id"], "state": state,
            "output_root": job["output_root"], "details": details,
        })
        if state in ("pending", "resumable"):
            pending.append(job)
    print(f"Manifest: {manifest}")
    print(json.dumps({
        "states": states,
        "no_extension_required": not jobs,
    }, indent=2))
    if any(row["state"] == "invalid" for row in states):
        raise RuntimeError("Invalid completed extension artifact exists")
    if options.execute:
        return execute_random(pending, options.max_parallel)
    for job in pending:
        print("DRY RUN " + subprocess.list2cmdline(job["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
