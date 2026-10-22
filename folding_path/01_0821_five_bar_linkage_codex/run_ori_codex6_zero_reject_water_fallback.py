"""Fail-closed water-only staged fallback for near-miss spread cutoff 0.12.

The fallback is additive and cannot be authorized unless the immutable R1
experiment has produced a valid formal water checkpoint-4096 rejection.  Use
``--draft`` before that authority exists, ``--authorize-manifest`` after the
rejection, then a separate ``--execute-manifest PATH`` invocation to launch.
No mode resumes or reuses an existing experiment root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import analyze_ori_codex6_zero_reject_staged as analysis
import run_ori_codex6_zero_reject_staged as staged


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ACTIVE_ROOT = (
    SCRIPT_DIRECTORY / "experiments_ori6/staged/zero_reject_reward/P0X2RCS05ZR0R1"
)
ACTIVE_GATE = ACTIVE_ROOT / "analysis/checkpoint_4096_water1/checkpoint_analysis.json"
ACTIVE_BINDING = ACTIVE_ROOT / "execution_manifest_binding.json"
ACTIVE_MANIFEST = (
    SCRIPT_DIRECTORY
    / "experiments_ori6/manifests/zero_reject_staged_dfc53efe4698.json"
)
DEFAULT_ROOT = (
    SCRIPT_DIRECTORY
    / "experiments_ori6/staged/zero_reject_water_fallback/P0X2RCS05ZR0WF12"
)
MANIFEST_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori6/manifests"
EXPECTED_ACTIVE_FINGERPRINT = (
    "dfc53efe46983844dc6cd28676c4389ccff79679cc7d66a002000fbfd76a8703"
)
EXPECTED_ACTIVE_MANIFEST_SHA256 = (
    "108588a09a1b5a2af7f4fb9eefaf9b3b162da52ff1ea2596a208ed9ed56c14d2"
)
EXPECTED_FROZEN_HASHES = {
    "run_ori_codex6_zero_reject_staged.py": (
        "d698ad4c713d0c6fdf156a5f6e586fc982cf659c5a420815fb0e39327e9586cd"
    ),
    "analyze_ori_codex6_zero_reject_staged.py": (
        "85b0e08becc9d89bab95ba6ffa475d38ee1deedaa8f7600bc5a52fd4284fe5ac"
    ),
    **staged.EXPECTED_CORE_HASHES,
}
SEEDS = (4301, 4302, 4303)
SCREEN_CHECKPOINTS = staged.SCREEN_CHECKPOINTS
PRODUCTION_CHECKPOINTS = staged.PRODUCTION_CHECKPOINTS
STOP_AT = datetime(2026, 8, 27, 8, 25, tzinfo=timezone(timedelta(hours=9)))
CUTOFF = 0.12
CANDIDATE_SLUG = "P0X2RCS05ZR0WF12"
TASK = staged.Task(
    "water1", "B", "model/waterbomb", 1.0, 400, 0.3,
    5.0e-4, 1.5e-4, (2000, 4000, 7000), CUTOFF,
)

COUNTERFACTUAL_ROOT = (
    SCRIPT_DIRECTORY / "experiments_ori6/analysis/water_cutoff_counterfactual_20260827"
)
EXPECTED_COUNTERFACTUAL_HASHES = {
    SCRIPT_DIRECTORY / "analyze_ori_codex6_water_cutoff_counterfactual.py": (
        "28d6955b0b1ef9f23909e59c4352983de71542ecce4dc507d22b2ba88a441d63"
    ),
    COUNTERFACTUAL_ROOT / "water_cutoff_counterfactual.json": (
        "86d87ee0e2d254217b86abbaa5bd65fc36f46fce9c73e843a932a0a4850e1715"
    ),
    COUNTERFACTUAL_ROOT / "water_cutoff_episode_counterfactual.csv": (
        "f1caf982948a25fdd00958017199e3eb3bec9d01ec8569f8033a71be4b5c4dbe"
    ),
    COUNTERFACTUAL_ROOT / "water_cutoff_summary.csv": (
        "3bbda19c32c2b126f9f687603935fbe76f89e3e58b1969fdebe1bb971e1d9dfc"
    ),
    COUNTERFACTUAL_ROOT / "water_cutoff_report.md": (
        "527eea18e82508dcd8b54e956524f25ff35cc6a79430b98ccdc21734770957ea"
    ),
}
SOURCE_PREFIX_SPECS = {
    "current_seed4201_ep2048": (
        ACTIVE_ROOT / "water1/B/seed4201/training_history.csv", 2048,
        "ae61fdbd33c675d83201f1b1330165ce5a0b6922bbd73d1b27b5c76ba8b47f8f",
    ),
    "current_seed4202_ep2048": (
        ACTIVE_ROOT / "water1/B/seed4202/training_history.csv", 2048,
        "1257c357017639aa1c0fd3115989bfb8c119d9a0ccb3ac717d5ad994650def44",
    ),
    "current_seed4203_ep2048": (
        ACTIVE_ROOT / "water1/B/seed4203/training_history.csv", 2048,
        "324a69ba681a8b5899268895976fb469bef8dd87603fb9a1aa8694cc5fb4efd9",
    ),
    "old_two_task_seed4101_ep6144": (
        SCRIPT_DIRECTORY / (
            "experiments_ori6/full/two_task_full_diagnostic/P0X2RCS05/"
            "water1/B/seed4101/training_history.csv"
        ), 6144,
        "838954ac896f872818958c54da560b780b7ccc3c9d45de70c5219a99b42dc3fd",
    ),
    "saved_spectral_seed3201_ep1024": (
        SCRIPT_DIRECTORY / (
            "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05/"
            "water1/B/seed3201/training_history.csv"
        ), 1024,
        "f051640f8c6d714c5d285b50715b901ad94f6fe1067a38c75cf0e8388bdbea62",
    ),
    "saved_spectral_seed3202_ep1024": (
        SCRIPT_DIRECTORY / (
            "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05/"
            "water1/B/seed3202/training_history.csv"
        ), 1024,
        "562d980f8341150186b6045ca84e9c09be52afbbfeb47bb4220f2bf34e555594",
    ),
    "saved_spectral_seed3203_ep1024": (
        SCRIPT_DIRECTORY / (
            "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05/"
            "water1/B/seed3203/training_history.csv"
        ), 1024,
        "9a29ba9eda1417ed9eb11af4892c8ba263c87a45cc1a07f0f20f7543251ba35e",
    ),
}
SAVED_SUCCESS_PATH = SCRIPT_DIRECTORY / (
    "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05/"
    "water1/B/seed3203/first_success_path/actions.json"
)
EXPECTED_SAVED_SUCCESS_SHA256 = (
    "eaa17acbe1b4db0255861277f249b86772ff0a16bc5787610e4c11153193ea2b"
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def csv_prefix_sha256(path: Path, endpoint: int) -> dict:
    """Hash a contiguous CSV prefix independently of later appended rows."""
    digest = hashlib.sha256()
    digest.update(b"ori6_csv_prefix_sha256_v1\0")
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        digest.update(json.dumps(
            fields, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"))
        digest.update(b"\n")
        for row in reader:
            episode = int(row["episode"])
            if episode > endpoint:
                break
            if episode != count + 1:
                raise RuntimeError(
                    f"non-contiguous counterfactual source {path}: "
                    f"episode {episode}, expected {count + 1}"
                )
            values = [row.get(field, "") for field in fields]
            digest.update(json.dumps(
                values, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    if count != endpoint:
        raise RuntimeError(
            f"counterfactual source {path} ends at {count}, expected {endpoint}"
        )
    return {
        "path": str(path.resolve()),
        "endpoint": endpoint,
        "row_count": count,
        "field_count": len(fields),
        "algorithm": "ori6_csv_prefix_sha256_v1",
        "sha256": digest.hexdigest(),
    }


def counterfactual_authority() -> dict:
    files = {}
    for path, expected in EXPECTED_COUNTERFACTUAL_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"missing bound counterfactual artifact: {path}")
        actual = staged.file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"counterfactual artifact changed: {path}: {actual} != {expected}"
            )
        files[str(path.resolve())] = actual
    prefixes = {}
    for label, (path, endpoint, expected) in SOURCE_PREFIX_SPECS.items():
        record = csv_prefix_sha256(path, endpoint)
        if record["sha256"] != expected:
            raise RuntimeError(
                f"counterfactual source prefix changed: {label}: "
                f"{record['sha256']} != {expected}"
            )
        prefixes[label] = record
    success_sha = staged.file_sha256(SAVED_SUCCESS_PATH)
    if success_sha != EXPECTED_SAVED_SUCCESS_SHA256:
        raise RuntimeError(
            f"saved success path changed: {success_sha} != "
            f"{EXPECTED_SAVED_SUCCESS_SHA256}"
        )
    return {
        "interpretation": (
            "recorded-trajectory terminal reward counterfactual; not retraining evidence"
        ),
        "recommended_cutoff": CUTOFF,
        "artifact_and_generator_sha256": files,
        "source_prefixes": prefixes,
        "saved_success_path": {
            "path": str(SAVED_SUCCESS_PATH.resolve()),
            "sha256": success_sha,
        },
    }


def validate_gate_payload(payload: dict) -> dict:
    errors: list[str] = []
    if payload.get("checkpoint_episode") != 4096:
        errors.append("authority checkpoint is not 4096")
    if payload.get("candidate") != staged.CANDIDATE:
        errors.append("authority candidate mismatch")
    if payload.get("overall_integrity_pass") is not True:
        errors.append("authority integrity did not pass")
    water = payload.get("tasks", {}).get("water1")
    if not isinstance(water, dict):
        errors.append("missing water1 authority")
        decision, reports = {}, []
    else:
        decision = water.get("decision", {})
        reports = water.get("seeds", [])
    expected_decision = {
        "checkpoint_episode": 4096,
        "rule": "formal_2_of_3_persistent_target_plane_gate",
        "stop": True,
        "promote": False,
        "selected_seed": None,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            errors.append(f"water decision {key}={decision.get(key)!r}, expected {expected!r}")
    count = decision.get("promising_seed_count")
    if not isinstance(count, int) or count >= 2:
        errors.append("water rejection is not fewer than 2/3 promising seeds")
    if sorted(report.get("seed") for report in reports) != [4201, 4202, 4203]:
        errors.append("authority does not contain all three preregistered seeds")
    if any(
        report.get("endpoint") != 4096 or not report.get("integrity_pass")
        for report in reports
    ):
        errors.append("authority seed endpoint/integrity mismatch")
    promising_from_reports = [
        report.get("seed") for report in reports
        if report.get("promising_at_4096") is True
    ]
    for report in reports:
        components = report.get("promising_components", {})
        expected_promising = bool(
            report.get("integrity_pass")
            and isinstance(components, dict)
            and set(components) == {"evidence", "persistence", "quality", "trend"}
            and all(value is True for value in components.values())
        )
        if report.get("promising_at_4096") is not expected_promising:
            errors.append(
                f"seed {report.get('seed')} promising flag/components inconsistent"
            )
    if count != len(promising_from_reports):
        errors.append("decision promising count disagrees with seed reports")
    if decision.get("promising_seeds") != promising_from_reports:
        errors.append("decision promising seed list disagrees with seed reports")
    selection_order = decision.get("selection_order")
    if (
        not isinstance(selection_order, list)
        or sorted(selection_order) != sorted(promising_from_reports)
    ):
        errors.append("decision selection order disagrees with promising seeds")
    return {
        "pass": not errors,
        "errors": errors,
        "decision": decision,
        "promising_seed_count": count,
    }


def _semantic_checkpoint(payload: dict) -> dict:
    serialised = analysis._serialisable(payload)
    if not isinstance(serialised, dict):
        raise RuntimeError("checkpoint payload is not an object")
    result = dict(serialised)
    result.pop("generated_utc", None)
    return result


def compare_recomputed_checkpoint(persisted: dict, recomputed: dict) -> dict:
    left = _semantic_checkpoint(persisted)
    right = _semantic_checkpoint(recomputed)
    left_digest = staged.canonical_digest(left)
    right_digest = staged.canonical_digest(right)
    return {
        "pass": left == right,
        "persisted_semantic_sha256": left_digest,
        "recomputed_semantic_sha256": right_digest,
    }


def verify_frozen_parent_sources() -> dict[str, str]:
    if not ACTIVE_MANIFEST.is_file():
        raise RuntimeError(f"bound R1 manifest is absent: {ACTIVE_MANIFEST}")
    if staged.file_sha256(ACTIVE_MANIFEST) != EXPECTED_ACTIVE_MANIFEST_SHA256:
        raise RuntimeError("bound R1 manifest bytes changed")
    active_manifest = _read_json(ACTIVE_MANIFEST)
    if active_manifest.get("manifest_fingerprint") != EXPECTED_ACTIVE_FINGERPRINT:
        raise RuntimeError("bound R1 manifest fingerprint changed")
    source_maps = [job.get("source_sha256") for job in active_manifest.get("jobs", [])]
    if len(source_maps) != 6 or any(item != source_maps[0] for item in source_maps):
        raise RuntimeError("R1 jobs do not share one complete source hash map")
    expected_sources = source_maps[0]
    if not isinstance(expected_sources, dict) or set(expected_sources) != set(staged.SOURCE_FILES):
        raise RuntimeError("R1 source hash map is incomplete")
    result = {}
    for name, expected in expected_sources.items():
        actual = staged.file_sha256(SCRIPT_DIRECTORY / name)
        if actual != expected:
            raise RuntimeError(
                f"R1-bound source changed: {name}: {actual} != {expected}"
            )
        result[name] = actual
    for name, expected in EXPECTED_FROZEN_HASHES.items():
        actual = result.get(name)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {name}: {actual} != {expected}")
    return result


def validate_formal_authority() -> dict:
    if not ACTIVE_GATE.is_file():
        raise RuntimeError(
            "Formal R1 water checkpoint-4096 authority is absent; fallback is locked"
        )
    if not ACTIVE_BINDING.is_file():
        raise RuntimeError("R1 execution binding is absent")
    # The recomputation is meaningful only under the byte-frozen analyser/core.
    verify_frozen_parent_sources()
    binding = _read_json(ACTIVE_BINDING)
    if binding.get("manifest_fingerprint") != EXPECTED_ACTIVE_FINGERPRINT:
        raise RuntimeError("R1 manifest fingerprint mismatch")
    if binding.get("manifest_sha256") != EXPECTED_ACTIVE_MANIFEST_SHA256:
        raise RuntimeError("R1 manifest SHA-256 mismatch")
    if staged.file_sha256(ACTIVE_BINDING) != (
        "e7ca8d36307ceababd5d3b98330665d49015cfccc2934db5955340fa8586be72"
    ):
        raise RuntimeError("R1 execution binding content changed")
    gate = _read_json(ACTIVE_GATE)
    validation = validate_gate_payload(gate)
    if not validation["pass"]:
        raise RuntimeError("Invalid formal water rejection: " + "; ".join(validation["errors"]))
    recomputed = analysis.analyse_checkpoint(
        ACTIVE_ROOT, 4096, tasks=("water1",), seeds=(4201, 4202, 4203)
    )
    recomputed_validation = validate_gate_payload(recomputed)
    if not recomputed_validation["pass"]:
        raise RuntimeError(
            "CSV-recomputed water gate is not a valid formal rejection: "
            + "; ".join(recomputed_validation["errors"])
        )
    semantic_match = compare_recomputed_checkpoint(gate, recomputed)
    if not semantic_match["pass"]:
        raise RuntimeError(
            "Persisted formal water gate does not exactly match frozen-analyser "
            "recomputation from CSV"
        )
    return {
        "active_root": str(ACTIVE_ROOT.resolve()),
        "gate_path": str(ACTIVE_GATE.resolve()),
        "gate_sha256": staged.file_sha256(ACTIVE_GATE),
        "binding_path": str(ACTIVE_BINDING.resolve()),
        "binding_sha256": staged.file_sha256(ACTIVE_BINDING),
        "active_manifest_fingerprint": EXPECTED_ACTIVE_FINGERPRINT,
        "active_manifest_sha256": EXPECTED_ACTIVE_MANIFEST_SHA256,
        "validated_decision": validation,
        "csv_recomputed_decision": recomputed_validation,
        "persisted_vs_csv_recomputed": semantic_match,
    }


def frozen_source_hashes() -> dict[str, str]:
    result = verify_frozen_parent_sources()
    result[Path(__file__).name] = staged.file_sha256(Path(__file__).resolve())
    return result


def build_jobs(root: Path) -> list[dict]:
    source_hashes = frozen_source_hashes()
    models = staged.model_hashes(TASK)
    task_identity = asdict(TASK)
    task_identity["actor_lr_milestones"] = list(TASK.actor_lr_milestones)
    result = []
    for seed in SEEDS:
        command = staged.build_command(root, TASK, seed)
        identity = {
            "task": task_identity,
            "seed": seed,
            "candidate": staged.CANDIDATE,
            "candidate_slug": CANDIDATE_SLUG,
            "output_directory": str(staged.output_directory(root, TASK, seed)),
            "command": command,
            "source_sha256": source_hashes,
            "model_sha256": models,
        }
        identity["job_fingerprint"] = staged.canonical_digest(identity)
        result.append(identity)
    return result


def manifest_payload(root: Path) -> dict:
    authority = validate_formal_authority()
    payload = {
        "schema_version": 1,
        "protocol": "ori6_zero_reject_water_cutoff_0p12_fail_closed_fallback",
        "generated_utc": staged.utc_now(),
        "candidate": staged.CANDIDATE,
        "candidate_slug": CANDIDATE_SLUG,
        "experiment_root": str(root.resolve()),
        "seeds": list(SEEDS),
        "near_miss_spread_error_far": CUTOFF,
        "deadline_stop_at": STOP_AT.isoformat(),
        "formal_reject_authority": authority,
        "counterfactual_authority": counterfactual_authority(),
        "screen_checkpoints": list(SCREEN_CHECKPOINTS),
        "production_checkpoints": list(PRODUCTION_CHECKPOINTS),
        "screen_rule": "same water thresholds; at least 2/3 promising at 4096",
        "post_4096_scope": "one selected-seed diagnostic, not three-seed confirmation",
        "jobs": build_jobs(root),
    }
    payload["manifest_fingerprint"] = staged.canonical_digest(payload)
    return payload


def authorize_manifest(root: Path) -> Path:
    if root.exists():
        raise FileExistsError(f"fallback root already exists: {root}")
    payload = manifest_payload(root)
    path = MANIFEST_DIRECTORY / (
        f"zero_reject_water_fallback_{payload['manifest_fingerprint'][:12]}.json"
    )
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite manifest: {path}")
        return path
    staged.atomic_json(path, payload)
    return path


def validate_manifest(path: Path, root: Path) -> dict:
    payload = _read_json(path)
    claimed = payload.get("manifest_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("manifest_fingerprint", None)
    if staged.canonical_digest(unsigned) != claimed:
        raise RuntimeError("fallback manifest fingerprint is invalid")
    rebuilt = manifest_payload(root)
    left, right = dict(payload), dict(rebuilt)
    left.pop("generated_utc", None)
    right.pop("generated_utc", None)
    left.pop("manifest_fingerprint", None)
    right.pop("manifest_fingerprint", None)
    if left != right:
        raise RuntimeError("fallback manifest/source/authority binding changed")
    if payload.get("experiment_root") != str(root.resolve()):
        raise RuntimeError("fallback manifest root mismatch")
    return payload


def validate_fallback_config(path: Path, seed: int) -> dict:
    base = analysis.validate_config(path, "water1", seed)
    checks = dict(base.get("checks", {}))
    try:
        config = _read_json(path)
        cutoff_ok = math.isclose(
            float(config.get("near_miss_spread_error_far", math.nan)),
            CUTOFF, abs_tol=1e-12,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        cutoff_ok = False
    checks["near_miss_spread_error_far"] = cutoff_ok
    errors = [
        error for error in base.get("errors", [])
        if error != "near_miss_spread_error_far"
    ]
    if not cutoff_ok:
        errors.append("near_miss_spread_error_far")
    return {"pass": not errors, "checks": checks, "errors": errors}


def seed_report(root: Path, seed: int, endpoint: int) -> dict:
    report = analysis.seed_report(root, "water1", seed, endpoint)
    run_dir = root / "water1/B" / f"seed{seed}"
    config = validate_fallback_config(run_dir / "run_config.json", seed)
    integrity = [
        error for error in report["integrity_errors"]
        if not error.startswith("config:")
    ]
    integrity.extend(f"config:{error}" for error in config["errors"])
    report["config_validation"] = config
    report["integrity_errors"] = integrity
    report["integrity_pass"] = not integrity
    report["promising_at_4096"] = (
        not integrity and all(report["promising_components"].values())
    )
    return report


def analyse_checkpoint(
    root: Path,
    endpoint: int,
    seeds: Sequence[int],
    selected_seed: int | None = None,
    baseline_4096: dict | None = None,
    previous_warning: bool = False,
) -> dict:
    reports = [seed_report(root, seed, endpoint) for seed in seeds]
    decision = analysis.task_decision("water1", reports, endpoint)
    if endpoint > 4096:
        if selected_seed is None or baseline_4096 is None or len(reports) != 1:
            raise ValueError("production analysis requires selected seed and 4096 baseline")
        warning = analysis.production_warning("water1", reports[0], baseline_4096)
        sustained = warning["warning"] and previous_warning
        decision = {
            "task": "water1",
            "checkpoint_episode": endpoint,
            "rule": "selected_seed_two_consecutive_checkpoint_collapse_gate",
            "stop": sustained or not reports[0]["integrity_pass"],
            "promote": True,
            "selected_seed": selected_seed,
            "collapse_warning": warning,
            "previous_checkpoint_warning": previous_warning,
            "sustained_collapse": sustained,
            "reasons": (
                ["two consecutive collapse warnings"] if sustained else
                ["single warning; require confirmation"] if warning["warning"] else
                ["no collapse warning"]
            ),
        }
    return {
        "schema_version": 1,
        "analysis_type": "ori6_zero_reject_water_fallback_checkpoint",
        "generated_utc": staged.utc_now(),
        "experiment_root": str(root.resolve()),
        "checkpoint_episode": endpoint,
        "candidate": staged.CANDIDATE,
        "fallback_near_miss_spread_error_far": CUTOFF,
        "tasks": {"water1": {"seeds": reports, "decision": decision}},
        "overall_integrity_pass": all(report["integrity_pass"] for report in reports),
    }


def execute(root: Path, manifest: Path, payload: dict, poll_seconds: float) -> int:
    if root.exists():
        raise FileExistsError(f"fresh fallback root already exists: {root}")
    if datetime.now(STOP_AT.tzinfo) >= STOP_AT:
        raise RuntimeError("refusing to launch after fallback safety stop")
    # Revalidate formal rejection and every byte-bound source immediately before launch.
    validate_manifest(manifest, root)
    root.mkdir(parents=True, exist_ok=False)
    staged.atomic_json(root / "execution_manifest_binding.json", {
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": staged.file_sha256(manifest),
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "formal_reject_gate_sha256": payload["formal_reject_authority"]["gate_sha256"],
        "started_utc": staged.utc_now(),
    })
    items = []
    try:
        for job in payload["jobs"]:
            items.append(staged.launch(root, job))
    except Exception:
        for item in items:
            if item["active"]:
                staged.stop_item(root, item, "partial_launch_failure")
        raise
    checked: set[int] = set()
    selected_seed: int | None = None
    baseline: dict | None = None
    previous_warning = False
    decisions = []
    active = True
    try:
        while active and datetime.now(STOP_AT.tzinfo) < STOP_AT:
            for item in items:
                if item["active"] and item["process"].poll() is not None:
                    staged.finish_item(root, item)
            members = [
                item for item in items
                if selected_seed is None or item["job"]["seed"] == selected_seed
            ]
            active_members = [item for item in members if item["active"]]
            schedule = SCREEN_CHECKPOINTS if selected_seed is None else PRODUCTION_CHECKPOINTS
            processed = False
            for endpoint in schedule:
                if endpoint in checked or not staged.checkpoint_ready(members, endpoint):
                    continue
                task_payload = analyse_checkpoint(
                    root, endpoint,
                    SEEDS if selected_seed is None else (selected_seed,),
                    selected_seed=selected_seed,
                    baseline_4096=baseline,
                    previous_warning=previous_warning,
                )
                decision = task_payload["tasks"]["water1"]["decision"]
                if endpoint > 4096:
                    previous_warning = bool(
                        decision.get("collapse_warning", {}).get("warning", False)
                    )
                artifact = staged.write_checkpoint_artifacts(
                    root, task_payload, endpoint, "water1"
                )
                decision["artifact_generation"] = artifact
                checked.add(endpoint)
                decisions.append(decision)
                staged.atomic_json(root / "analysis/decision_history.json", {
                    "updated_utc": staged.utc_now(), "decisions": decisions,
                })
                processed = True
                if endpoint == 1024 and decision["stop"]:
                    for item in active_members:
                        staged.stop_item(root, item, "catastrophic_1024_gate", decision)
                    active = False
                elif endpoint == 4096:
                    if not decision["promote"]:
                        for item in active_members:
                            staged.stop_item(root, item, "not_promising_at_4096", decision)
                        active = False
                    else:
                        selected_seed = int(decision["selected_seed"])
                        baseline = next(
                            report for report in task_payload["tasks"]["water1"]["seeds"]
                            if report["seed"] == selected_seed
                        )
                        for item in active_members:
                            if item["job"]["seed"] != selected_seed:
                                staged.stop_item(root, item, "not_selected_after_4096", decision)
                elif endpoint > 4096 and decision["stop"]:
                    for item in active_members:
                        staged.stop_item(root, item, "sustained_production_collapse", decision)
                    active = False
                break
            if not active_members and not processed:
                active = False
            if active:
                time.sleep(min(30.0, max(1.0, poll_seconds)))
    finally:
        for item in items:
            if item["active"]:
                staged.stop_item(root, item, "deadline_or_runner_finalization")
    relevant = [
        item for item in items
        if selected_seed is None or item["job"]["seed"] == selected_seed
    ]
    endpoints = [
        staged.checkpoint_episode(Path(item["job"]["output_directory"])) or 0
        for item in relevant
    ]
    endpoint = min(endpoints) if endpoints else 0
    final_analysis = None
    final_artifacts = None
    if endpoint:
        final_analysis = analyse_checkpoint(
            root, endpoint,
            (selected_seed,) if selected_seed is not None else SEEDS,
            selected_seed=selected_seed,
            baseline_4096=baseline if endpoint > 4096 else None,
            previous_warning=previous_warning,
        )
        final_artifacts = staged.write_checkpoint_artifacts(
            root, final_analysis, endpoint, "final_water1"
        )
    staged.atomic_json(root / "analysis/final_summary.json", {
        "schema_version": 1,
        "protocol": "ori6_zero_reject_water_cutoff_0p12_fail_closed_fallback",
        "candidate": staged.CANDIDATE,
        "generated_utc": staged.utc_now(),
        "deadline_stop_at": STOP_AT.isoformat(),
        "formal_reject_authority": payload["formal_reject_authority"],
        "selected_seed": selected_seed,
        "endpoint": endpoint,
        "decisions": decisions,
        "analysis": analysis._serialisable(final_analysis),
        "artifacts": final_artifacts,
        "scope_note": "4096 three-seed screen; post-4096 selected-seed diagnostic",
    })
    return 0


def draft(root: Path) -> dict:
    jobs = build_jobs(root)
    authority_status = {"ready": False, "path": str(ACTIVE_GATE.resolve())}
    if ACTIVE_GATE.is_file():
        try:
            authority_status = {"ready": True, **validate_formal_authority()}
        except RuntimeError as error:
            authority_status["error"] = str(error)
    return {
        "mode": "draft_no_launch_no_manifest",
        "candidate_slug": CANDIDATE_SLUG,
        "experiment_root": str(root.resolve()),
        "cutoff": CUTOFF,
        "seeds": list(SEEDS),
        "stop_at": STOP_AT.isoformat(),
        "formal_reject_authority": authority_status,
        "commands": [job["command"] for job in jobs],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--authorize-manifest", action="store_true")
    mode.add_argument("--execute-manifest", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    root = options.experiment_root.resolve()
    if options.draft:
        print(json.dumps(draft(root), indent=2, ensure_ascii=False))
        return 0
    if options.authorize_manifest:
        manifest = authorize_manifest(root)
        print(json.dumps({
            "authorized_manifest": str(manifest.resolve()),
            "manifest_sha256": staged.file_sha256(manifest),
            "execute_command": subprocess.list2cmdline([
                sys.executable, str(Path(__file__).resolve()),
                "--execute-manifest", str(manifest.resolve()),
                "--experiment-root", str(root),
            ]),
        }, indent=2))
        return 0
    manifest = options.execute_manifest.resolve()
    payload = validate_manifest(manifest, root)
    return execute(root, manifest, payload, options.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
