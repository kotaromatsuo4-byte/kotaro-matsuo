"""Fail-closed 10:00 finalizer for the Codex8 spectral campaign.

Despite the filename suffix, ``1000`` means the 10:00 report deadline, not
one thousand episodes.  The finalizer never launches or terminates training.
It refuses publication while a Codex8 trainer/runner is active, verifies the
immutable manifest and source closure, caps every comparison at an atomic
endpoint, and publishes a UTF-8 JSON/CSV/PNG/Japanese-Markdown bundle by one
atomic directory rename.
"""

from __future__ import annotations

import argparse
import bisect
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Iterable, Sequence

import numpy as np
import analyze_ori_codex8_spectral_staged as staged
import ori_codex6_spectral_random_baseline as random_baseline
import run_ori_codex8_spectral_staged as campaign_runner


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY / "experiments_ori8" / "C8SP2R1"
DEFAULT_OUTPUT = SCRIPT_DIRECTORY / "experiments_ori8" / "analysis" / "C8SP2R1_1000"
FORENSIC_PARENT_ROOT = SCRIPT_DIRECTORY / "experiments_ori8" / "C8SP2"
MANIFEST_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori8" / "manifests"
PROTOCOL = "codex8_spectral_reward_only_three_arm_staged_replay1"
EXPECTED_MANIFEST_FINGERPRINT = (
    "f6c0dcfdbee9d2b3f6f1df8a6cacead775a5a8f01196335def32bfef754894d3"
)
EXPECTED_MANIFEST_NAME = "codex8_spectral_staged_replay1_f6c0dcfdbee9.json"
EXPECTED_STARTED_UTC = "2026-08-27T16:52:02.026125+00:00"
FORENSIC_PARENT_FINGERPRINT = (
    "56a0da899663735a1be5768c56c4e6a8f0138389492992a51d71516909fd3620"
)
FORENSIC_ABORT_REASON = (
    "C8SP2 torus cohorts were valid through atomic episode 512, but frozen "
    "analyzer 23cdf444 incorrectly classified a separate horizon-boundary "
    "timeout reward as the ordinary zero-reject total and eliminated all "
    "three arms; the supervisor was stopped before water reached a formal "
    "checkpoint"
)
EXPECTED_BINDING_THREADS = {
    "env": {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    },
    "logical_cpu_count": 20,
    "torch_interop_descriptive": 12,
    "torch_intraop": 1,
}
RANDOM_ROOT = (
    SCRIPT_DIRECTORY / "experiments_ori6" / "random"
    / "first_discovery_matched_20260827"
)
RANDOM_BUDGETS = {"torus2": 636_198, "water1": 126_319}
RANDOM_SEEDS = tuple(range(6101, 6111))
RANDOM_ROOT_FINGERPRINTS = {
    "torus2": {
        6101: "6500e5ee04ea9cdf64e4a72870017c2e7e7f43151c4174ec7b7598274989d8db",
        6102: "4aa267880b9467c6405ae22504a85d5ff331ea143848cac76938ca860137cd76",
        6103: "5fefc20fa630daf6338dc0b76e6f2f8e1a5a2b404866897fc86701b648a4bdcd",
        6104: "e5bd6e31e32fb7424876e6e1bd0677b2dbf8df764fa0098cbe986cc871868174",
        6105: "eebb64ce92de490500f68451c1393e00a8dc0a3bdc442d02f735bed25580adbf",
        6106: "5b9866bce500df7d1d5fdc08942f060aae15ac30aa79265be0fc3a8804794cc1",
        6107: "46f97b1ee0064123f90ab64229ecfc499eadf74eb07a1a254c421e7557accbee",
        6108: "4e5bf1f2dad3914162a3d2aa28d4619a2a4391bd1d25ce357bfe30a19c64a905",
        6109: "6280a99f5860c5e7b83157431b7a269929e349f36c605d7aed395687457fb592",
        6110: "e2276abaee58d7b6099b3a53a9bdb9a1ed3b3d4e7d9d8ebb44d62dcd59d2b013",
    },
    "water1": {
        6101: "db0ccb9927c7cc39da0a2861824faadad2f6ea02cd2dea7d3857ce96d7a70876",
        6102: "82b6f9c3efebbdeea077264cef74be32f7512d60c38456620c73c0e42f5e2a15",
        6103: "a721e5b52622091c197ae3d8ab8f5636125279d6e166385f471a2c20a16de020",
        6104: "fda97390030e35532b3b1d0b54a365d8dcf12e70a95c344a8c17dfcb37ffd199",
        6105: "ea494bc87ec37ad32f77246759e0052112aaf720308d48d81e8539aafcf7815a",
        6106: "8367c66b58881cc885a93d39e1d33e5b6c5a46bcc31591bfb74a85cabba1a8f1",
        6107: "5e04cd70210235018ec8e0aac4ba27c7ed08614b4aa7f6d6769bdccfc2aef7dd",
        6108: "170fc2cd4974487aedc5d7705d4fe4dfe3584e5333e3b5a443f041e6325b5727",
        6109: "665bd0fa385bd9e6c6d8bef9c06011a911fa263e609039973c7fb7b7bd59fdf6",
        6110: "6c5eaa8adb528281d723f9c00ea5502e8c9c4f576222b186e9a41d99cefcd806",
    },
}
ROLLING_WINDOW = 200
FINALIZER_FILES = (
    "finalize_ori_codex8_1000.py",
    "test_finalize_ori_codex8_1000.py",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being hashed: {path}")
    return digest.hexdigest()


def verify_external_approval(
    approved_finalizer_sha256: str,
    approved_test_sha256: str,
) -> dict:
    """Bind publication to hashes supplied by an authority outside this file.

    A source file cannot safely contain its own approved digest: inserting that
    digest changes the source and therefore changes the digest.  Production
    invocation must instead supply both frozen values on the command line.
    """

    expected = {
        "finalize_ori_codex8_1000.py": approved_finalizer_sha256,
        "test_finalize_ori_codex8_1000.py": approved_test_sha256,
    }
    actual = {}
    for name, value in expected.items():
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"Missing/invalid external approval SHA256: {name}")
        path = SCRIPT_DIRECTORY / name
        actual[name] = sha256_file(path)
        if actual[name] != value:
            raise RuntimeError(
                f"Externally approved SHA256 mismatch: {name}: "
                f"expected={value} actual={actual[name]}"
            )
    return {
        "authority": "required_cli_values_external_to_the_hashed_sources",
        "approved_sha256": expected,
        "verified_sha256": actual,
        "self_referential_hash_claim": False,
    }


def canonical_digest(value) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being read: {path}")
    return data


def load_json(path: Path) -> dict:
    return json.loads(stable_bytes(path).decode("utf-8-sig"))


def load_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(
        stable_bytes(path).decode("utf-8-sig")
    )))


def active_codex8_pids() -> list[dict]:
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and ((($_.CommandLine -like '*ori_codex8.py*') "
        "-or ($_.CommandLine -like '*ori_codex8_3d.py*')) -or "
        "($_.CommandLine -like '*run_ori_codex8_spectral_staged.py*'))}; "
        "$p | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    return [{"pid": _exact_int(row["ProcessId"], "active process id"),
             "command": row.get("CommandLine", "")}
            for row in payload]


def require_below(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Path escaped authority root: {path}")
    return path


def find_manifest(fingerprint: str) -> Path:
    if fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise RuntimeError("Campaign manifest fingerprint is not the approved replay FP")
    path = (MANIFEST_DIRECTORY / EXPECTED_MANIFEST_NAME).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing exact approved manifest: {path}")
    return path


def _job_identity(job: dict) -> tuple[str, str, int]:
    task = job.get("task", {})
    task = task.get("name") if isinstance(task, dict) else task
    return str(task), str(job.get("arm")), _exact_int(job.get("seed"), "job seed")


def _report_endpoint_identity(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("task")), str(row.get("arm")),
        _exact_int(row.get("seed"), "runner endpoint seed"),
    )


def validate_runner_endpoint_record(row: dict, key: tuple[str, str, int]) -> tuple[int, int]:
    allowed_status = {
        "selected", "screened", "queued_censored_at_training_deadline",
        "not_launched_by_staged_protocol",
    }
    if row.get("status") not in allowed_status:
        raise RuntimeError(f"Unknown runner endpoint status: {key}")
    if type(row.get("launched")) is not bool or type(row.get("selected")) is not bool:
        raise RuntimeError(f"Runner endpoint booleans invalid: {key}")
    raw = _exact_int(row.get("raw_endpoint"), "runner raw_endpoint")
    formal = _exact_int(
        row.get("formal_analyzed_endpoint"), "runner formal_analyzed_endpoint"
    )
    overrun = _exact_int(row.get("overrun_episode_count"), "runner overrun_episode_count")
    if not 0 <= formal <= raw <= 10000 or formal not in (0, *staged.CHECKPOINTS):
        raise RuntimeError(f"Runner endpoint bounds invalid: {key}")
    if overrun != raw - formal:
        raise RuntimeError(f"Runner endpoint overrun arithmetic invalid: {key}")
    if type(row.get("overrun_excluded_from_formal_analysis")) is not bool:
        raise RuntimeError(f"Runner endpoint overrun flag is not boolean: {key}")
    if row["overrun_excluded_from_formal_analysis"] != (raw > formal):
        raise RuntimeError(f"Runner endpoint overrun flag invalid: {key}")
    if not row["launched"] and (raw != 0 or formal != 0 or row["selected"]):
        raise RuntimeError(f"Unlaunched endpoint contains run state: {key}")
    if (row["status"] == "selected") != row["selected"]:
        raise RuntimeError(f"Selected endpoint status mismatch: {key}")
    if row["launched"] != (row["status"] in {"selected", "screened"}):
        raise RuntimeError(f"Runner endpoint launch/status mismatch: {key}")
    return raw, formal


def expected_execution_binding(manifest: dict) -> dict:
    return {
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
        "source_sha256": manifest["source_sha256"],
        "parent_architecture_sha256": manifest["parent_architecture_sha256"],
        "started_utc": EXPECTED_STARTED_UTC,
        "job_order": ["torus2", "water1"],
        "max_parallel": 12,
        "training_stop_local": "09:20",
        "report_deadline_local": "10:00",
        "reward_only_change": True,
        "hidden_ledger_pomdp_caveat": manifest["hidden_ledger_pomdp_caveat"],
        "child_thread_contract": EXPECTED_BINDING_THREADS,
        "fresh_replay_from_episode_zero": True,
        "forensic_parent_summary": {
            "binding_sha256": (
                "25df830c0fd3ed65ab8f7d16bb2c279cd0757b5442282c4b6dcf02d752fe63bb"
            ),
            "manifest_fingerprint": FORENSIC_PARENT_FINGERPRINT,
            "reason": FORENSIC_ABORT_REASON,
            "root": str(FORENSIC_PARENT_ROOT.resolve()),
        },
    }


def validate_execution_binding(binding: dict, manifest: dict) -> None:
    binding_exact = expected_execution_binding(manifest)
    if binding != binding_exact:
        missing = sorted(set(binding_exact) - set(binding))
        extra = sorted(set(binding) - set(binding_exact))
        wrong = sorted(
            field for field in set(binding_exact) & set(binding)
            if binding[field] != binding_exact[field]
        )
        raise RuntimeError(
            f"Execution binding differs from exact launched authority: "
            f"missing={missing} extra={extra} wrong={wrong}"
        )


def verify_control_plane(root: Path) -> tuple[dict, dict, dict, Path]:
    root = Path(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise RuntimeError("Production campaign root is fixed exactly to experiments_ori8/C8SP2R1")
    binding_path = root / "execution_binding.json"
    report_path = root / "runner_report.json"
    if not binding_path.is_file() or not report_path.is_file():
        raise RuntimeError("Missing execution_binding.json or runner_report.json")
    binding, report = load_json(binding_path), load_json(report_path)
    fingerprint = binding.get("manifest_fingerprint")
    if fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise RuntimeError("Execution binding is not the exact approved replay FP")
    manifest_path = find_manifest(fingerprint)
    manifest = load_json(manifest_path)
    # This is the campaign runner's own complete validator.  It rebuilds all
    # 18 jobs and checks exact commands, model hashes, job fingerprints,
    # source closure, architecture closure, paths and immutable protocol.
    runner_validated = campaign_runner.validate_manifest(manifest_path)
    if runner_validated != manifest:
        raise RuntimeError("Frozen runner validation did not return the exact manifest")
    exact = {
        "protocol": PROTOCOL,
        "experiment_root": str(root),
        "tasks": list(staged.TASKS),
        "arms": list(staged.ARMS),
        "seeds": list(staged.SEEDS),
        "episodes_cap": 10000,
        "screen_checkpoints": list(staged.SCREEN_CHECKPOINTS),
        "production_checkpoints": list(staged.PRODUCTION_CHECKPOINTS),
        "selection_rule_version": staged.SELECTION_RULE_VERSION,
        "reward_only_change": True,
        "observation_contract": "exact_Codex6_parent_observation_unchanged",
        "training_stop_local": "09:20",
        "report_deadline_local": "10:00",
        "fresh_replay_from_episode_zero": True,
        "replay_policy": (
            "same seeds, frozen core and commands restart from episode zero in a "
            "new root; no checkpoint, history, RNG, model, or optimizer state is "
            "copied from the forensic parent"
        ),
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"Manifest protocol mismatch: {field}")
    forensic = manifest.get("forensic_parent", {})
    forensic_exact = {
        "root": str(FORENSIC_PARENT_ROOT.resolve()),
        "manifest_fingerprint": FORENSIC_PARENT_FINGERPRINT,
        "reason": FORENSIC_ABORT_REASON,
        "reuse_policy": "preserve read-only; copy/resume/reuse no files or optimizer state",
        "runner_report_present": False,
    }
    for field, expected in forensic_exact.items():
        if forensic.get(field) != expected:
            raise RuntimeError(f"Forensic parent authority mismatch: {field}")
    validate_execution_binding(binding, manifest)
    if report.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError("Runner report fingerprint mismatch")
    if report.get("protocol") != PROTOCOL or report.get("schema_version") != 1:
        raise RuntimeError("Runner report protocol/schema mismatch")
    if report.get("training_deadline_local") != "09:20" or report.get("report_deadline_local") != "10:00":
        raise RuntimeError("Runner report deadline mismatch")
    if len(manifest.get("jobs", [])) != 18 or len(report.get("endpoints", [])) != 18:
        raise RuntimeError("Campaign must contain exactly 18 jobs/endpoints")
    for field in ("deadline_stop", "report_deadline_met"):
        if type(report.get(field)) is not bool:
            raise RuntimeError(f"Runner report field must be boolean: {field}")
    try:
        datetime.fromisoformat(str(report["finished_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Runner report has invalid finished_utc") from error
    if not isinstance(report.get("task_selections"), dict):
        raise RuntimeError("Runner report task_selections must be a mapping")
    if not set(report["task_selections"]).issubset(set(staged.TASKS)):
        raise RuntimeError("Runner report contains an unknown task selection")
    expected_matrix = {
        (task, arm, seed) for task in staged.TASKS for arm in staged.ARMS
        for seed in staged.SEEDS
    }
    if {_job_identity(job) for job in manifest["jobs"]} != expected_matrix:
        raise RuntimeError("Manifest job matrix mismatch")
    if {_report_endpoint_identity(row) for row in report["endpoints"]} != expected_matrix:
        raise RuntimeError("Runner endpoint matrix mismatch")
    manifest_by_key = {_job_identity(job): job for job in manifest["jobs"]}
    for row in report["endpoints"]:
        key = _report_endpoint_identity(row)
        job = manifest_by_key[key]
        validate_runner_endpoint_record(row, key)
        if Path(job["output_directory"]).resolve() != staged.run_directory(
            root, *key
        ).resolve():
            raise RuntimeError(f"Manifest output directory mismatch: {key}")
    artifact_names = {
        "checkpoint_analysis.json", "seed_metrics.csv",
        "production_checkpoint.json", "selected_seed_window.csv",
        "checkpoint_report_ja.md",
    }
    seen_artifacts = set()
    if not isinstance(report.get("artifacts"), list):
        raise RuntimeError("Runner report artifacts must be a list")
    for group in report["artifacts"]:
        if not isinstance(group, dict) or set(group) != {"json", "csv", "markdown"}:
            raise RuntimeError("Runner artifact group schema mismatch")
        for raw_path in group.values():
            path = require_below(Path(raw_path), root / "analysis")
            if path.name not in artifact_names or not path.is_file():
                raise RuntimeError(f"Runner artifact path/name invalid: {path}")
            if path in seen_artifacts:
                raise RuntimeError(f"Runner artifact path duplicated: {path}")
            seen_artifacts.add(path)
    return manifest, binding, report, manifest_path


def _float(row: dict, *names: str, default=math.nan) -> float:
    return staged._float(row, *names, default=default)


def _int(row: dict, *names: str, default=0) -> int:
    return staged._int(row, *names, default=default)


def contiguous_history(rows: Sequence[dict]) -> int:
    episodes = [
        _exact_int(row.get("episode"), "training history episode")
        for row in rows
    ]
    if episodes != list(range(1, len(rows) + 1)):
        raise RuntimeError("training_history.csv is not unique/contiguous from episode 1")
    return len(rows)


def checkpoint_episode(path: Path) -> int:
    if not path.is_file():
        return 0
    import torch
    # The trainer includes NumPy RNG state beside tensors.  Keep the restricted
    # weights-only unpickler and allowlist only those concrete NumPy containers
    # needed by that frozen checkpoint schema.
    from numpy.core.multiarray import _reconstruct
    numpy_safe = [
        _reconstruct, np.ndarray, np.dtype,
        type(np.dtype(np.float64)), type(np.dtype(np.uint32)),
    ]
    checkpoint_bytes = stable_bytes(path)
    with torch.serialization.safe_globals(numpy_safe):
        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    raw = payload.get("episode", 0)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(f"Checkpoint episode is not an exact integer: {path}")
    value = raw
    if value < 0 or value > 10000:
        raise RuntimeError(f"Invalid checkpoint episode {value}: {path}")
    return value


def authoritative_endpoint(directory: Path) -> tuple[int, dict, list[dict]]:
    history_path = directory / "training_history.csv"
    if not history_path.is_file():
        return 0, {"authority": "none"}, []
    rows = load_csv(history_path)
    history_endpoint = contiguous_history(rows)
    summary_endpoint = 0
    if (directory / "summary.json").is_file():
        summary = load_json(directory / "summary.json")
        raw_summary_endpoint = summary.get(
            "training_episodes", summary.get("episodes", summary.get("episode"))
        )
        summary_endpoint = (
            _exact_int(raw_summary_endpoint, "training summary episode")
            if raw_summary_endpoint is not None else 0
        )
    checkpoint = checkpoint_episode(directory / "checkpoint.pt")
    if summary_endpoint == 10000:
        if history_endpoint < 10000:
            raise RuntimeError(f"Final summary without 10000 history rows: {directory}")
        endpoint, authority = 10000, "summary_and_history"
    elif checkpoint > 0:
        endpoint, authority = min(checkpoint, history_endpoint), "checkpoint_and_history"
    else:
        return 0, {
            "authority": "no_atomic_checkpoint", "history_endpoint": history_endpoint,
            "checkpoint_endpoint": 0, "summary_endpoint": summary_endpoint,
        }, rows
    return endpoint, {
        "authority": authority, "history_endpoint": history_endpoint,
        "checkpoint_endpoint": checkpoint, "summary_endpoint": summary_endpoint,
        "overrun_history_rows": max(history_endpoint - endpoint, 0),
    }, rows


def rolling(values: Sequence[float], window: int, kind="mean") -> list[float]:
    if kind == "median":
        queue, ordered, result = deque(), [], []
        for raw in values:
            value = float(raw)
            finite = math.isfinite(value)
            queue.append(value if finite else None)
            if finite:
                bisect.insort(ordered, value)
            if len(queue) > window:
                expired = queue.popleft()
                if expired is not None:
                    ordered.pop(bisect.bisect_left(ordered, expired))
            n = len(ordered)
            result.append(
                math.nan if not n else ordered[n // 2] if n % 2
                else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
            )
        return [float(value) for value in result]
    queue, total, count, result = deque(), 0.0, 0, []
    for raw in values:
        value = float(raw)
        finite = math.isfinite(value)
        queue.append(value if finite else None)
        if finite:
            total += value; count += 1
        if len(queue) > window:
            expired = queue.popleft()
            if expired is not None:
                total -= expired; count -= 1
        result.append(float(total / count) if count else math.nan)
    return result


def episode_curves(rows: Sequence[dict], endpoint: int, identity: dict) -> list[dict]:
    rows = list(rows[:endpoint])
    records = [staged.classify(row) for row in rows]
    curves = []
    for row, record in zip(rows, records):
        curves.append({
            **identity, "episode": record["episode"],
            "reward": record["reward"], "terminal_flatness": record["flatness"],
            "spread_error": record["spread_error"], "strict": int(record["strict"]),
            "joint": int(record["joint"]), "constructive": int(record["constructive"]),
            "reject_rate": record["reject_rate"],
            "spectral_potential": _float(
                row, "ori8_terminal_psi", "ori6_spectral_terminal_potential", default=0.0
            ),
            "spectral_reward": _float(
                row, "ori8_total_reward", "ori6_spectral_total_reward", default=0.0
            ),
            "phase_close": _int(row, "ori8_close_latched"),
            "phase_z": _int(row, "ori8_z_latched"),
            "phase_rollback": _int(row, "ori8_rollback_count"),
        })
    for field, kind, destination in (
        ("reward", "mean", "reward_ma200"),
        ("terminal_flatness", "median", "terminal_flatness_median200"),
        ("spread_error", "median", "spread_error_median200"),
        ("strict", "mean", "strict_rate200"),
        ("joint", "mean", "joint_rate200"),
        ("constructive", "mean", "constructive_rate200"),
        ("reject_rate", "mean", "reject_rate200"),
        ("spectral_potential", "mean", "spectral_potential_ma200"),
        ("spectral_reward", "mean", "spectral_reward_ma200"),
        ("phase_z", "mean", "phase_z_rate200"),
    ):
        values = rolling([row[field] for row in curves], ROLLING_WINDOW, kind)
        for row, value in zip(curves, values):
            row[destination] = value
    return curves


def optimizer_curves(rows: Sequence[dict], endpoint: int, identity: dict) -> list[dict]:
    result = []
    for row in rows:
        episode = _exact_int(row.get("episode"), "optimizer episode")
        if not 0 < episode <= endpoint:
            continue
        values = {
            "critic_loss": _float(row, "critic_loss"),
            "critic_gradient_norm": _float(row, "critic_gradient_norm"),
            "actor_gradient_norm": _float(row, "actor_gradient_norm"),
            "entropy": _float(row, "entropy"),
            "actor_learning_rate": _float(row, "actor_learning_rate"),
            "critic_learning_rate": _float(row, "critic_learning_rate"),
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Nonfinite optimizer row {identity} episode {episode}")
        result.append({**identity, "episode": episode, **values})
    return result


def _report_endpoint_map(report: dict) -> dict[tuple[str, str, int], dict]:
    return {
        _report_endpoint_identity(row): row
        for row in report["endpoints"]
    }


def verify_forensic_parent(manifest: dict) -> tuple[dict, dict]:
    """Verify the aborted run as provenance only; return no scientific rows."""

    forensic = manifest["forensic_parent"]
    root = FORENSIC_PARENT_ROOT.resolve()
    ledger = forensic.get("files")
    if not isinstance(ledger, dict) or len(ledger) != _exact_int(
        forensic.get("file_count"), "forensic file_count"
    ):
        raise RuntimeError("Forensic parent file ledger schema/count mismatch")
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if actual_names != set(ledger):
        raise RuntimeError("Forensic parent file set changed after abort")
    hashes = {}
    for relative, authority in ledger.items():
        path = require_below(root / Path(relative), root)
        if path.stat().st_size != _exact_int(
            authority.get("size"), f"forensic file size {relative}"
        ):
            raise RuntimeError(f"Forensic parent file size changed: {relative}")
        digest = sha256_file(path)
        if digest != authority.get("sha256"):
            raise RuntimeError(f"Forensic parent file digest changed: {relative}")
        hashes[str(path)] = digest
    parent_manifest = forensic.get("manifest_file", {})
    parent_manifest_path = Path(parent_manifest.get("path", "")).resolve()
    if (
        parent_manifest_path.name != "codex8_spectral_staged_56a0da899663.json"
        or not parent_manifest_path.is_file()
        or parent_manifest_path.stat().st_size != _exact_int(
            parent_manifest.get("size"), "forensic parent manifest size"
        )
        or sha256_file(parent_manifest_path) != parent_manifest.get("sha256")
    ):
        raise RuntimeError("Forensic parent manifest changed")
    hashes[str(parent_manifest_path)] = parent_manifest["sha256"]
    return ({
        "root": str(root),
        "manifest_fingerprint": forensic["manifest_fingerprint"],
        "reason": forensic["reason"],
        "reuse_policy": forensic["reuse_policy"],
        "verified_file_count": len(ledger),
        "scientific_rows_in_primary_analysis": 0,
        "status": "audited_abort_provenance_only",
    }, hashes)


def _without_generated_utc(payload: dict) -> dict:
    result = dict(payload)
    result.pop("generated_utc", None)
    return result


def _exact_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} is boolean, not an exact integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} is not numeric") from error
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise RuntimeError(f"{field} is not an exact integer")
    return int(numeric)


def _validate_random_history(rows: Sequence[dict], summary: dict, budget: int) -> None:
    if not rows:
        raise RuntimeError("Random trial history is empty")
    episodes = [_exact_int(row.get("episode"), "random episode") for row in rows]
    if episodes != list(range(1, len(rows) + 1)):
        raise RuntimeError("Random trial history is not contiguous")
    mechanics = 0
    strict_rows = []
    for row in rows:
        if row.get("method") != random_baseline.METHOD:
            raise RuntimeError("Random history method mismatch")
        mechanics += _exact_int(row.get("mechanical_steps"), "random mechanical_steps")
        if mechanics != _exact_int(
            row.get("cumulative_mechanical_steps"),
            "random cumulative_mechanical_steps",
        ):
            raise RuntimeError("Random cumulative mechanical steps mismatch")
        strict = _exact_int(
            row.get("strict_compact_success"), "random strict_compact_success"
        )
        success_row = _exact_int(row.get("success"), "random success")
        if strict not in (0, 1) or success_row not in (0, 1) or strict != success_row:
            raise RuntimeError("Random success/strict field is not exact binary identity")
        if strict:
            strict_rows.append(row)
    if mechanics != _exact_int(summary.get("mechanical_steps"), "summary mechanical_steps"):
        raise RuntimeError("Random history/summary mechanical steps mismatch")
    summary_success = _exact_int(summary.get("success"), "summary success")
    if summary_success not in (0, 1):
        raise RuntimeError("Random summary success is not binary")
    success = bool(summary_success)
    if success != bool(strict_rows) or len(strict_rows) > 1:
        raise RuntimeError("Random strict-event identity mismatch")
    if success:
        if strict_rows[-1] is not rows[-1]:
            raise RuntimeError("Random baseline continued after first strict hit")
        if _exact_int(
            summary.get("first_success_mechanical_steps"),
            "summary first_success_mechanical_steps",
        ) != mechanics:
            raise RuntimeError("Random first-hit mechanics mismatch")
    elif mechanics != budget or summary.get("censor_reason") != "mechanics_budget":
        raise RuntimeError("Random censor is not at the exact mechanics budget")
    recomputed_audit = random_baseline.reward_integrity(list(rows))
    if recomputed_audit != summary.get("reward_integrity"):
        raise RuntimeError("Random reward-integrity summary differs from history")
    if not random_baseline.validate_reward_integrity(recomputed_audit):
        raise RuntimeError("Random reward-integrity recomputation failed")


def _official_random_seed(
    task: str, seed: int, budget: int, base_root: Path
) -> tuple[dict, dict, str]:
    root = Path(base_root) / task / f"budget{budget}" / f"seed{seed}"
    root_config_path = root / "run_config.json"
    aggregate_path = root / "aggregate.json"
    trial = root / "trials" / random_baseline.METHOD / f"seed{seed}"
    trial_config_path = trial / "run_config.json"
    summary_path = trial / "summary.json"
    history_path = trial / "episode_history.csv"
    required = (
        root_config_path, aggregate_path, trial_config_path, summary_path,
        history_path,
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"Missing official random authority under {root}")
    hashes = {str(path.resolve()): sha256_file(path) for path in required}
    root_config = random_baseline.load_root(root)
    expected_identity = random_baseline.root_identity(task, budget, [seed])
    if (
        root_config.get("task") != task
        or root_config.get("seeds") != [seed]
        or _exact_int(
            root_config.get("mechanics_budget"), "random root mechanics_budget"
        ) != budget
        or root_config.get("fingerprint_identity") != expected_identity
        or root_config.get("root_fingerprint")
        != RANDOM_ROOT_FINGERPRINTS[task][seed]
        or root_config.get("root_fingerprint")
        != random_baseline.canonical_digest(expected_identity)
    ):
        raise RuntimeError(f"Random root exact identity mismatch: {root}")
    for name, digest in expected_identity["source_sha256"].items():
        source_path = SCRIPT_DIRECTORY / name
        if sha256_file(source_path) != digest:
            raise RuntimeError(f"Random authority source changed: {name}")
        hashes[str(source_path.resolve())] = digest
    model_names = {"node": "node_initial.dat", "member": "member.dat", "fix": "fix.dat"}
    model_root = Path(expected_identity["effective_candidate_args"]["data_dir"])
    for key, digest in expected_identity["model_sha256"].items():
        model_path = model_root / model_names[key]
        if sha256_file(model_path) != digest:
            raise RuntimeError(f"Random authority model changed: {model_path}")
        hashes[str(model_path.resolve())] = digest
    stored = load_json(aggregate_path)
    trial_config = load_json(trial_config_path)
    summary = load_json(summary_path)
    for field, value in trial_config.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Random trial config/summary mismatch: {field}")
    _validate_random_history(load_csv(history_path), summary, budget)

    # aggregate_results is the official recomputation, but it writes its
    # aggregate.  Stage only authority inputs in a temporary root so the
    # historical baseline remains byte-for-byte read-only.
    with tempfile.TemporaryDirectory(prefix="codex8_random_recompute_") as tmp:
        staged_root = Path(tmp) / "root"
        staged_trial = (
            staged_root / "trials" / random_baseline.METHOD / f"seed{seed}"
        )
        staged_trial.mkdir(parents=True)
        shutil.copy2(root_config_path, staged_root / "run_config.json")
        shutil.copy2(trial_config_path, staged_trial / "run_config.json")
        shutil.copy2(summary_path, staged_trial / "summary.json")
        shutil.copy2(history_path, staged_trial / "episode_history.csv")
        recomputed = random_baseline.aggregate_results(
            staged_root, require_complete=True
        )
    if _without_generated_utc(recomputed) != _without_generated_utc(stored):
        raise RuntimeError(f"Stored random aggregate differs from official recompute: {root}")
    return summary, hashes, str(root_config["root_fingerprint"])


def load_random_baseline(task: str, *, base_root: Path = RANDOM_ROOT) -> dict:
    budget = RANDOM_BUDGETS[task]
    trials, hashes, fingerprints = [], {}, {}
    for seed in RANDOM_SEEDS:
        trial, trial_hashes, fingerprint = _official_random_seed(
            task, seed, budget, base_root
        )
        hashes.update(trial_hashes)
        fingerprints[seed] = fingerprint
        hit = bool(trial.get("success"))
        trials.append({
            "seed": seed, "hit": hit, "censored": not hit,
            "first_hit_action_proposals": trial.get("first_success_mechanical_steps"),
            "observed_action_proposals": _exact_int(
                trial["mechanical_steps"], "random trial mechanical_steps"
            ),
            "assigned_rmst_cost": (
                _exact_int(
                    trial["first_success_mechanical_steps"],
                    "random trial first_success_mechanical_steps",
                ) if hit else budget
            ),
            "censor_reason": None if hit else "common_action_proposal_cap",
        })
    costs = [row["assigned_rmst_cost"] for row in trials]
    return {
        "task": task, "budget": budget, "seeds": list(RANDOM_SEEDS),
        "n": len(trials), "hits": sum(row["hit"] for row in trials),
        "hit_fraction": sum(row["hit"] for row in trials) / len(trials),
        "first_hit_action_proposals": [
            row["first_hit_action_proposals"] for row in trials if row["hit"]
        ],
        "restricted_mean_action_proposals": float(statistics.fmean(costs)),
        "censor_assignment": "no-hit seed assigned the exact common action-proposal budget",
        "trials": trials, "input_sha256": hashes,
        "root_fingerprint_by_seed": fingerprints,
        "authority": (
            "ori_codex6_spectral_random_baseline.load_root plus "
            "aggregate_results official recomputation on a read-only temporary copy; "
            "trial reward audit recomputed from episode_history.csv"
        ),
        "equivalence_scope": (
            "geometry, uniform action proposals, max-step horizon and zero-reject "
            "crossing continuation are pathwise applicable; reward identity is excluded"
        ),
    }


def _median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(values)) if values else None


def random_comparison(task: str, selected_arm: str, reports: Sequence[dict]) -> dict:
    baseline = load_random_baseline(task)
    cap = baseline["budget"]
    learning_trials = []
    for row in reports:
        hit = row["first_hit"]
        exposure = _exact_int(
            row["full"]["action_proposals"], "learning observed action proposals"
        )
        hit_cost = (
            _exact_int(hit["action_proposals"], "learning first-hit action proposals")
            if hit["hit"] else None
        )
        event = hit_cost is not None and hit_cost <= cap
        observable = bool(event or exposure >= cap)
        learning_trials.append({
            "seed": _exact_int(row["seed"], "learning seed"),
            "hit_at_common_cap": event,
            "first_hit_action_proposals": hit_cost if event else None,
            "uncapped_first_hit_action_proposals": hit_cost,
            "observed_action_proposals": exposure,
            "common_cap_observable": observable,
            "assigned_rmst_cost": hit_cost if event else cap if observable else None,
            "censor_reason": (
                None if event else "common_action_proposal_cap"
                if observable else "insufficient_learning_exposure_for_common_cap"
            ),
        })
    hits = [
        row["first_hit_action_proposals"] for row in learning_trials
        if row["hit_at_common_cap"]
    ]
    complete = all(row["endpoint"] == 10000 for row in reports)
    common_cap_observable = all(
        row["common_cap_observable"] for row in learning_trials
    )
    median_hit = _median(hits)
    learning_rmst = (
        float(statistics.fmean(
            row["assigned_rmst_cost"] for row in learning_trials
        )) if common_cap_observable else None
    )
    ratio = (
        baseline["restricted_mean_action_proposals"] / learning_rmst
        if complete and common_cap_observable and learning_rmst else None
    )
    return {
        "task": task, "selected_arm": selected_arm,
        "learning_seed_count": len(reports), "learning_complete_10000": complete,
        "common_action_proposal_cap": cap,
        "learning_hits": len(hits),
        "learning_hits_at_common_cap": len(hits),
        "learning_uncapped_hits_by_episode_10000": sum(
            row["first_hit"]["hit"] for row in reports
        ),
        "learning_hit_fraction": len(hits) / len(reports),
        "learning_first_hit_action_proposals": hits,
        "learning_median_first_hit_action_proposals": median_hit,
        "learning_trials": learning_trials,
        "learning_common_cap_observable": common_cap_observable,
        "learning_restricted_mean_action_proposals": learning_rmst,
        "random": baseline,
        "descriptive_random_rmst_over_learning_rmst": ratio,
        "ratio_qualifier": (
            "descriptive_common-cap_RMST_ratio_cross-cohort_not_paired"
            if ratio is not None else
            "unavailable_insufficient_learning_exposure_at_common_cap"
            if not common_cap_observable else
            "unavailable_until_all_three_learning_seeds_complete_10000"
        ),
        "primary_measure": "cumulative raw action proposals to first strict hit",
        "reward_identity_matched": False,
    }


SELECTION_COMPONENTS = (
    "hit_indicator", "conditional_negative_first_hit_proposals",
    "strict_last1024", "joint_last1024", "constructive_last1024",
    "negative_spread_error_median", "negative_flatness_median",
    "negative_reject_rate",
)


def paired_seed_rows(task: str, endpoint: int, analysis: dict) -> list[dict]:
    """Expose every exact same-seed lexicographic component at its endpoint."""

    decision = analysis["task_decisions"][task]
    reports = analysis["reports"][task]
    result = []
    for candidate in analysis["arms"]:
        candidate_by_seed = {row["seed"]: row for row in reports[candidate]}
        for opponent in analysis["arms"]:
            if opponent == candidate:
                continue
            opponent_by_seed = {row["seed"]: row for row in reports[opponent]}
            comparison = decision["paired_wins"][candidate][opponent]
            evidence = {row["seed"]: row for row in comparison["per_seed"]}
            for seed in staged.SEEDS:
                left = candidate_by_seed[seed]
                right = opponent_by_seed[seed]
                left_key = staged.selection_key(left)
                right_key = staged.selection_key(right)
                first_difference = next((
                    SELECTION_COMPONENTS[index]
                    for index, (a, b) in enumerate(zip(left_key, right_key))
                    if a != b
                ), "tie")
                lh, rh = left["first_hit"], right["first_hit"]
                lr, rr = left["last1024"], right["last1024"]
                result.append({
                    "task": task, "endpoint": endpoint,
                    "candidate": candidate, "opponent": opponent,
                    "seed": seed, "winner": evidence[seed]["winner"],
                    "first_differing_component": first_difference,
                    "candidate_hit": int(lh["hit"]),
                    "opponent_hit": int(rh["hit"]),
                    "hit_indicator_delta": int(lh["hit"]) - int(rh["hit"]),
                    "candidate_first_hit_proposals": lh.get("action_proposals"),
                    "opponent_first_hit_proposals": rh.get("action_proposals"),
                    "candidate_censor_proposals": lh.get("censor_action_proposals"),
                    "opponent_censor_proposals": rh.get("censor_action_proposals"),
                    "first_hit_proposals_delta_if_both_hit": (
                        int(lh["action_proposals"]) - int(rh["action_proposals"])
                        if lh["hit"] and rh["hit"] else None
                    ),
                    "candidate_strict_last1024": lr["strict_count"],
                    "opponent_strict_last1024": rr["strict_count"],
                    "strict_delta": lr["strict_count"] - rr["strict_count"],
                    "candidate_joint_last1024": lr["joint_count"],
                    "opponent_joint_last1024": rr["joint_count"],
                    "joint_delta": lr["joint_count"] - rr["joint_count"],
                    "candidate_constructive_last1024": lr["constructive_count"],
                    "opponent_constructive_last1024": rr["constructive_count"],
                    "constructive_delta": (
                        lr["constructive_count"] - rr["constructive_count"]
                    ),
                    "candidate_spread_error_median": lr["spread_error_median"],
                    "opponent_spread_error_median": rr["spread_error_median"],
                    "spread_error_delta_candidate_minus_opponent": (
                        lr["spread_error_median"] - rr["spread_error_median"]
                    ),
                    "candidate_flatness_median": lr["flatness_median"],
                    "opponent_flatness_median": rr["flatness_median"],
                    "flatness_delta_candidate_minus_opponent": (
                        lr["flatness_median"] - rr["flatness_median"]
                    ),
                    "candidate_reject_rate": lr["reject_rate_mean"],
                    "opponent_reject_rate": rr["reject_rate_mean"],
                    "reject_rate_delta_candidate_minus_opponent": (
                        lr["reject_rate_mean"] - rr["reject_rate_mean"]
                    ),
                })
    return result


def _require_runner_decision_matches(independent: dict, recorded: dict, task: str) -> None:
    for field, value in independent.items():
        if recorded.get(field) != value:
            raise RuntimeError(
                f"Runner selection disagrees with independent analysis: {task}/{field}"
            )


def reconstruct_task_selection(
    root: Path, task: str, recorded: dict, reports_by_key: dict
) -> tuple[dict, list[dict]]:
    selection_endpoint = _exact_int(
        recorded.get("selection_endpoint"), f"{task} selection_endpoint"
    )
    if selection_endpoint not in (2048, 4096):
        raise RuntimeError(f"Invalid selection endpoint for {task}: {selection_endpoint}")
    complete_inputs = all(
        (task, arm, seed) in reports_by_key
        and reports_by_key[(task, arm, seed)]["endpoint"] >= selection_endpoint
        for arm in staged.ARMS for seed in staged.SEEDS
    )
    if not complete_inputs:
        if (
            recorded.get("selected_arm") is not None
            or recorded.get("promote_selected_to_10000", False)
            or recorded.get("promotion_revoked_endpoint") is not None
        ):
            raise RuntimeError(f"Positive/revoked selection lacks atomic inputs: {task}")
        return ({
            **recorded,
            "independent_reconstruction": (
                "unavailable_after_preselection_integrity_or_catastrophic_stop"
            ),
            "selected_arm": None,
            "promote_selected_to_10000": False,
        }, [])

    provisional = staged.analyze(root, selection_endpoint, tasks=(task,))
    provisional_decision = provisional["task_decisions"][task]
    rows = paired_seed_rows(task, selection_endpoint, provisional)
    revoked_endpoint = recorded.get("promotion_revoked_endpoint")
    if revoked_endpoint is not None:
        if (
            selection_endpoint != 2048
            or _exact_int(
                revoked_endpoint, f"{task} promotion_revoked_endpoint"
            ) != 4096
            or recorded.get("selected_arm") is not None
            or recorded.get("promote_selected_to_10000") is not False
            or recorded.get("continued_seeds") != []
        ):
            raise RuntimeError(f"Malformed early-promotion revocation: {task}")
        winner = recorded.get("comparative_winner_arm")
        if (
            winner not in staged.NEW_ARMS
            or provisional_decision.get("selected_arm") != winner
            or provisional_decision.get("strong_early_promotion") is not True
        ):
            raise RuntimeError(f"Revoked arm was not the qualified 2048 winner: {task}")
        mutable = {
            "selected_arm", "promote_selected_to_10000",
            "continued_seed_policy", "continued_seeds",
        }
        for field, value in provisional_decision.items():
            if field not in mutable and recorded.get(field) != value:
                raise RuntimeError(f"Revoked selection history mismatch: {task}/{field}")
        cohort = [
            staged.seed_report(root, task, winner, seed, 4096)
            for seed in staged.SEEDS
        ]
        stop, stop_kind, reasons = staged.catastrophic(cohort, 4096)
        if not stop or stop_kind != recorded.get("promotion_revoked_kind"):
            raise RuntimeError(f"4096 revocation gate does not reconstruct: {task}")
        return ({
            **recorded,
            "independent_reconstruction": "2048_winner_then_4096_revocation",
            "provisional_decision_2048": provisional_decision,
            "revocation_gate_4096": {
                "stop": stop, "stop_kind": stop_kind, "reasons": reasons,
            },
            "final_decision_status": "promotion_revoked_no_final_selected_arm",
        }, rows)

    _require_runner_decision_matches(provisional_decision, recorded, task)
    verified = dict(provisional_decision)
    verified["selection_endpoint"] = selection_endpoint
    verified["continued_seed_policy"] = recorded.get("continued_seed_policy")
    verified["continued_seeds"] = recorded.get("continued_seeds")
    if selection_endpoint == 2048 and verified.get("selected_arm") is not None:
        winner = verified["selected_arm"]
        reached_4096 = all(
            reports_by_key[(task, winner, seed)]["endpoint"] >= 4096
            for seed in staged.SEEDS
        )
        if reached_4096:
            cohort = [
                staged.seed_report(root, task, winner, seed, 4096)
                for seed in staged.SEEDS
            ]
            stop, stop_kind, reasons = staged.catastrophic(cohort, 4096)
            if stop:
                raise RuntimeError(
                    f"Runner retained an early winner that failed its 4096 gate: {task}"
                )
            verified["confirmation_gate_4096"] = {
                "reached": True, "stop": False,
                "stop_kind": stop_kind, "reasons": reasons,
            }
        else:
            verified["confirmation_gate_4096"] = {
                "reached": False,
                "reason": "right_censored_before_4096_confirmation_gate",
            }
    return verified, rows


def _arm_screen_stop(
    task: str, arm: str, jobs: Sequence[dict], reports_by_key: dict
) -> dict | None:
    members = [row for row in jobs if row["task"] == task and row["arm"] == arm]
    # Raw checkpoints can overrun by different amounts while the runner kills
    # a cohort.  The causal gate is the common formal analyzed endpoint only.
    endpoints = {
        _exact_int(
            row.get("formal_analyzed_endpoint", row.get("atomic_endpoint")),
            f"{task}/{arm} formal analyzed endpoint",
        )
        for row in members
    }
    if len(members) != 3 or len(endpoints) != 1:
        return None
    endpoint = endpoints.pop()
    if endpoint not in staged.SCREEN_CHECKPOINTS:
        return None
    reports = [reports_by_key.get((task, arm, seed)) for seed in staged.SEEDS]
    if any(row is None for row in reports):
        return None
    stop, stop_kind, reasons = staged.catastrophic(reports, endpoint)
    return ({"stop_kind": stop_kind, "reasons": reasons, "endpoint": endpoint}
            if stop else None)


def assign_causal_censor_reasons(
    root: Path, jobs: list[dict], reports_by_key: dict,
    selections: dict, runner_report: dict,
) -> None:
    stops = {
        (task, arm): _arm_screen_stop(task, arm, jobs, reports_by_key)
        for task in staged.TASKS for arm in staged.ARMS
    }
    for row in jobs:
        if not row.get("launched"):
            row["training_censor_reason"] = row["status"]
            continue
        endpoint = _exact_int(row.get("atomic_endpoint"), "job atomic endpoint")
        if endpoint == 0:
            row["training_censor_reason"] = "deadline_before_first_atomic_checkpoint"
            continue
        if endpoint == 10000:
            row["training_censor_reason"] = None
        else:
            task, arm = row["task"], row["arm"]
            selection = selections.get(task)
            stop = stops.get((task, arm))
            if selection and selection.get("promotion_revoked_endpoint"):
                if arm == selection.get("comparative_winner_arm"):
                    row["training_censor_reason"] = (
                        "strong_early_winner_revoked_at_4096_"
                        f"{selection.get('promotion_revoked_kind')}"
                    )
                else:
                    row["training_censor_reason"] = (
                        "not_continued_after_provisional_2048_selection"
                    )
            elif stop is not None:
                row["training_censor_reason"] = (
                    f"screen_{stop['stop_kind']}_stop_at_{stop['endpoint']}"
                )
                row["training_censor_details"] = stop["reasons"]
            elif selection and selection.get("selected_arm") is not None:
                selected = selection["selected_arm"]
                if arm != selected:
                    row["training_censor_reason"] = (
                        f"not_selected_by_same_seed_rule_at_"
                        f"{selection['selection_endpoint']}"
                    )
                elif runner_report["deadline_stop"]:
                    row["training_censor_reason"] = (
                        "training_deadline_during_selected_continuation"
                    )
                else:
                    # A selected seed can stop only by the predeclared
                    # production integrity/collapse monitor.
                    baseline = staged.seed_report(
                        root, task, arm, row["seed"], 4096
                    )
                    warnings = 0
                    found = None
                    for checkpoint in staged.PRODUCTION_CHECKPOINTS:
                        if checkpoint > endpoint:
                            break
                        current = staged.seed_report(
                            root, task, arm, row["seed"], checkpoint
                        )
                        decision = staged.production_decision(
                            current, baseline, warnings
                        )
                        warnings = _exact_int(
                            decision["consecutive_warning_count"],
                            "production consecutive_warning_count",
                        )
                        if decision["stop"]:
                            found = (checkpoint, decision)
                            break
                    if found is None or found[0] != endpoint:
                        raise RuntimeError(
                            f"Selected seed has no causal production stop: {row}"
                        )
                    row["training_censor_reason"] = (
                        f"{found[1]['stop_kind']}_at_{found[0]}"
                    )
                    row["training_censor_details"] = found[1]["reasons"]
            elif selection and selection.get("control_non_improvement"):
                row["training_censor_reason"] = (
                    "legacy_control_winner_no_new_arm_promotion"
                    if arm == "legacy" else
                    "new_arm_lost_to_legacy_control_at_4096"
                )
            elif selection:
                row["training_censor_reason"] = (
                    "formal_same_seed_comparison_inconclusive_no_promotion"
                )
            elif runner_report["deadline_stop"]:
                row["training_censor_reason"] = (
                    "training_deadline_before_task_level_selection"
                )
            else:
                row["training_censor_reason"] = "staged_protocol_no_continuation"
        if row.get("first_hit"):
            row["first_hit_censor_reason"] = None
        elif endpoint == 10000:
            row["first_hit_censor_reason"] = (
                "natural_completion_at_episode_10000_without_strict_hit"
            )
        else:
            row["first_hit_censor_reason"] = row["training_censor_reason"]


def build_payload(
    root: Path,
    approved_finalizer_sha256: str,
    approved_test_sha256: str,
    active_check=True,
) -> dict:
    if active_check:
        active = active_codex8_pids()
        if active:
            raise RuntimeError(f"Active Codex8 PID(s); refusing finalization: {active}")
    approval = verify_external_approval(
        approved_finalizer_sha256, approved_test_sha256
    )
    manifest, binding, runner_report, manifest_path = verify_control_plane(root)
    forensic_provenance, forensic_hashes = verify_forensic_parent(manifest)
    endpoint_map = _report_endpoint_map(runner_report)
    jobs, reports_by_key, episode_rows, optimizer_rows, input_hashes = [], {}, [], [], {}
    for path in (
        manifest_path, Path(root) / "execution_binding.json",
        Path(root) / "runner_report.json",
    ):
        input_hashes[str(path.resolve())] = sha256_file(path)
    for group in runner_report["artifacts"]:
        for path in group.values():
            path = Path(path).resolve()
            input_hashes[str(path)] = sha256_file(path)
    input_hashes.update(forensic_hashes)
    for job in manifest["jobs"]:
        task, arm, seed = _job_identity(job)
        identity = {"task": task, "arm": arm, "seed": seed}
        endpoint_record = endpoint_map[(task, arm, seed)]
        directory = staged.run_directory(root, task, arm, seed)
        launched = bool(endpoint_record.get("launched"))
        if not launched:
            jobs.append({
                **identity, "launched": False, "atomic_endpoint": 0,
                "status": endpoint_record.get("status"),
                "censor_reason": endpoint_record.get("status"),
                "integrity_status": "not_applicable_unlaunched",
            })
            continue
        for name in (
            "run_config.json", "training_history.csv", "update_history.csv",
            "checkpoint.pt", "summary.json",
        ):
            path = directory / name
            if path.is_file():
                input_hashes[str(path.resolve())] = sha256_file(path)
        raw_endpoint, authority, history = authoritative_endpoint(directory)
        runner_raw = _exact_int(
            endpoint_record.get("raw_endpoint"), "runner raw_endpoint"
        )
        if raw_endpoint != runner_raw:
            raise RuntimeError(
                f"Runner/finalizer raw endpoint mismatch {identity}: "
                f"{runner_raw} != {raw_endpoint}"
            )
        endpoint = _exact_int(
            endpoint_record.get("formal_analyzed_endpoint"),
            "runner formal_analyzed_endpoint",
        )
        if endpoint > raw_endpoint:
            raise RuntimeError(f"Formal endpoint exceeds raw checkpoint: {identity}")
        if endpoint == 0:
            jobs.append({
                **identity, "launched": True, "atomic_endpoint": 0,
                "raw_checkpoint_endpoint": raw_endpoint,
                "status": endpoint_record.get("status"),
                "integrity_status": "no_atomic_rows_consumed",
            })
            continue
        report = staged.seed_report(root, task, arm, seed, endpoint)
        if not report["integrity_pass"]:
            raise RuntimeError(f"Run integrity failure {identity}: {report['integrity_errors'][:8]}")
        reports_by_key[(task, arm, seed)] = report
        updates = load_csv(directory / "update_history.csv")
        episode_rows.extend(episode_curves(history, endpoint, identity))
        optimizer_rows.extend(optimizer_curves(updates, endpoint, identity))
        hit, recent = report["first_hit"], report["last1024"]
        jobs.append({
            **identity, "launched": True, "atomic_endpoint": endpoint,
            "raw_checkpoint_endpoint": raw_endpoint,
            "endpoint_authority": authority, "status": endpoint_record.get("status"),
            "selected": bool(endpoint_record.get("selected")),
            "formal_analyzed_endpoint": _exact_int(
                endpoint_record.get("formal_analyzed_endpoint"),
                "runner formal_analyzed_endpoint",
            ),
            "overrun_excluded_from_formal_analysis": bool(
                endpoint_record.get("overrun_excluded_from_formal_analysis")
            ),
            "integrity_status": "verified", "first_hit": bool(hit["hit"]),
            "first_hit_episode": hit["episode"],
            "first_hit_action_proposals": hit["action_proposals"],
            "first_hit_censor_action_proposals": hit.get("censor_action_proposals"),
            "observed_action_proposals": report["full"]["action_proposals"],
            "strict_count_last1024": recent["strict_count"],
            "joint_count_last1024": recent["joint_count"],
            "constructive_count_last1024": recent["constructive_count"],
            "flatness_median_last1024": recent["flatness_median"],
            "spread_error_median_last1024": recent["spread_error_median"],
            "reject_rate_last1024": recent["reject_rate_mean"],
            "phase_farming_detected": report["phase_reward_hacking"]["farming_detected"],
        })

    verified_selections, paired_rows = {}, []
    for task, recorded in runner_report.get("task_selections", {}).items():
        verified, task_pairs = reconstruct_task_selection(
            root, task, recorded, reports_by_key
        )
        verified_selections[task] = verified
        paired_rows.extend(task_pairs)

    assign_causal_censor_reasons(
        root, jobs, reports_by_key, verified_selections, runner_report
    )

    random_results = {}
    for task, decision in verified_selections.items():
        arm = decision.get("selected_arm")
        if arm is None:
            continue
        reports = [reports_by_key.get((task, arm, seed)) for seed in staged.SEEDS]
        if all(report is not None and report["endpoint"] == 10000 for report in reports):
            comparison = random_comparison(task, arm, reports)
            random_results[task] = comparison
            input_hashes.update(comparison["random"]["input_sha256"])

    source_hashes = dict(manifest["source_sha256"])
    finalizer_hashes = approval["verified_sha256"]
    return {
        "schema_version": 1, "analysis": "codex8_1000_final",
        "created_utc": utc_now(), "root": str(Path(root).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "protocol": PROTOCOL, "selection_rule_version": staged.SELECTION_RULE_VERSION,
        "reward_only_change": True,
        "observation_contract": manifest["observation_contract"],
        "hidden_ledger_pomdp_caveat": manifest["hidden_ledger_pomdp_caveat"],
        "forensic_parent_provenance": forensic_provenance,
        "runner_report": runner_report, "verified_task_selections": verified_selections,
        "jobs": jobs, "episode_curves": episode_rows,
        "optimizer_curves": optimizer_rows,
        "paired_seed_comparisons": paired_rows,
        "random_comparisons": random_results,
        "source_sha256": source_hashes, "finalizer_source_sha256": finalizer_hashes,
        "external_hash_approval": approval,
        "input_sha256": input_hashes,
        "comparison_contract": {
            "primary": "strict first-hit then cumulative raw action proposals",
            "return_used_for_arm_selection": False,
            "overrun_policy": "excluded beyond each declared atomic comparison endpoint",
            "random_scope": (
                "geometry/action-proposal equivalence only; reward identity excluded; "
                "random cohort seeds differ from learning seeds"
            ),
        },
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rows:
            writer.writeheader(); writer.writerows(rows)


def _plot_rows(payload: dict, task: str, arm: str, seed: int) -> list[dict]:
    return [row for row in payload["episode_curves"] if (
        row["task"], row["arm"], row["seed"]
    ) == (task, arm, seed)]


def render_plots(payload: dict, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    colours = {"legacy": "#6b7280", "scale": "#d97706", "phase_v2": "#2563eb"}
    for task in staged.TASKS:
        figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
        for arm in staged.ARMS:
            for index, seed in enumerate(staged.SEEDS):
                rows = _plot_rows(payload, task, arm, seed)
                if not rows:
                    continue
                x = [row["episode"] for row in rows]
                label = f"{arm} seed{seed}" if index == 0 else None
                style = {"color": colours[arm], "alpha": .38, "linewidth": .8}
                axes[0, 0].plot(x, [row["reward_ma200"] for row in rows], label=label, **style)
                axes[0, 1].plot(x, [max(row["terminal_flatness_median200"], 1e-8) for row in rows], **style)
                axes[1, 0].plot(x, [max(row["spread_error_median200"], 1e-8) for row in rows], **style)
                axes[1, 1].plot(x, [row["strict_rate200"] for row in rows], **style)
                axes[1, 1].plot(x, [row["joint_rate200"] for row in rows],
                                color=colours[arm], alpha=.26, linewidth=.7, linestyle=":")
                axes[1, 1].plot(x, [row["constructive_rate200"] for row in rows],
                                color=colours[arm], alpha=.20, linewidth=.6, linestyle="--")
                axes[2, 0].plot(x, [row["spectral_potential_ma200"] for row in rows], **style)
                axes[2, 1].plot(x, [row["reject_rate200"] for row in rows], **style)
        titles = (
            "Return MA200 (diagnostic only)", "Terminal flatness F (median200)",
            "Spread error e (median200)",
            "Strict (solid) / joint (dotted) / constructive (dashed)",
            "Spectral potential / phase Psi MA200", "Rejected crossing rate MA200",
        )
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title); axis.set_xlabel("episode"); axis.grid(alpha=.2)
        axes[0, 1].set_yscale("log"); axes[1, 0].set_yscale("log")
        axes[0, 0].legend(fontsize=8, ncol=3)
        path = output / f"learning_curve_{task}.png"
        figure.suptitle(f"Codex8 {task}")
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)

        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for arm in staged.ARMS:
            for seed in staged.SEEDS:
                rows = [row for row in payload["optimizer_curves"] if (
                    row["task"], row["arm"], row["seed"]
                ) == (task, arm, seed)]
                if not rows:
                    continue
                x = [row["episode"] for row in rows]
                style = {"color": colours[arm], "alpha": .30, "linewidth": .7}
                axes[0, 0].plot(x, [max(row["critic_loss"], 1e-12) for row in rows], **style)
                axes[0, 1].plot(x, [row["critic_gradient_norm"] for row in rows], **style)
                axes[1, 0].plot(x, [row["actor_gradient_norm"] for row in rows], **style)
                axes[1, 1].plot(x, [row["actor_learning_rate"] for row in rows], **style)
                axes[1, 1].plot(x, [row["critic_learning_rate"] for row in rows],
                                color=colours[arm], alpha=.22, linewidth=.7, linestyle="--")
        for axis, title in zip(axes.flat, (
            "Critic loss", "Critic gradient norm", "Actor gradient norm",
            "Actor LR (solid) / critic LR (dashed)"
        )):
            axis.set_title(title); axis.set_xlabel("episode"); axis.grid(alpha=.2)
        axes[0, 0].set_yscale("log"); axes[1, 1].set_yscale("log")
        path = output / f"optimizer_curve_{task}.png"
        figure.suptitle(f"Codex8 optimizer diagnostics: {task}")
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)

    if payload["random_comparisons"]:
        figure, axes = plt.subplots(
            1, len(payload["random_comparisons"]),
            figsize=(6 * len(payload["random_comparisons"]), 5), squeeze=False,
            constrained_layout=True,
        )
        for axis, (task, comparison) in zip(axes.flat, payload["random_comparisons"].items()):
            learning = comparison["learning_trials"]
            random = comparison["random"]["trials"]
            learning_hits = [row["first_hit_action_proposals"] for row in learning
                             if row["hit_at_common_cap"]]
            learning_censored = [row for row in learning
                                 if row["common_cap_observable"] and not row["hit_at_common_cap"]]
            learning_insufficient = [row for row in learning
                                     if not row["common_cap_observable"]]
            random_hits = [row["first_hit_action_proposals"] for row in random if row["hit"]]
            random_censored = [row for row in random if not row["hit"]]
            if learning_hits:
                axis.scatter([0] * len(learning_hits), learning_hits,
                             label="learning hit<=cap", color="#2563eb")
            if learning_censored:
                axis.scatter([0] * len(learning_censored),
                             [comparison["common_action_proposal_cap"]] * len(learning_censored),
                             label="learning censored@cap", marker="x", color="#1e3a8a")
            if learning_insufficient:
                axis.scatter([0] * len(learning_insufficient),
                             [row["observed_action_proposals"] for row in learning_insufficient],
                             label="learning exposure<cap", marker="^", color="#7c3aed")
            axis.scatter([0] * len(learning),
                         [row["observed_action_proposals"] for row in learning],
                         label="selected-arm observed exposure", marker="_",
                         s=180, color="#0f766e")
            if random_hits:
                axis.scatter([1] * len(random_hits), random_hits,
                             label="random hit<=cap", color="#d97706")
            if random_censored:
                axis.scatter([1] * len(random_censored),
                             [comparison["random"]["budget"]] * len(random_censored),
                             label="random censored@cap", marker="x", color="#991b1b")
            axis.axhline(comparison["common_action_proposal_cap"], color="#111827",
                         alpha=.25, linewidth=.8, linestyle="--", label="common cap")
            axis.set_xticks([0, 1], ["learning", "random"])
            axis.set_yscale("log"); axis.set_ylabel("raw action proposals")
            axis.set_title(task); axis.grid(alpha=.2); axis.legend(fontsize=8)
        path = output / "first_hit_vs_random.png"
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)
    return paths


def visual_qa(paths: Sequence[Path]) -> list[dict]:
    from PIL import Image, ImageStat
    result = []
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path).convert("RGB") as image:
            stats = ImageStat.Stat(image)
            pixel_std = float(sum(stats.stddev) / len(stats.stddev))
            width, height = image.size
        if path.stat().st_size < 5000 or width < 600 or height < 400 or pixel_std < 1.0:
            raise RuntimeError(f"Blank/invalid plot: {path}")
        result.append({
            "file": path.name, "bytes": path.stat().st_size,
            "width": width, "height": height, "pixel_std": pixel_std,
        })
    return result


def _fmt(value, digits=4):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{digits}g}" if math.isfinite(value) else "—"


def japanese_report(payload: dict) -> str:
    lines = [
        "# Codex8 spectral reward検証（10:00確定版）", "",
        "## 結論", "",
    ]
    jobs = payload["jobs"]
    for task in staged.TASKS:
        selection = payload["verified_task_selections"].get(task)
        if not selection:
            lines.append(f"- **{task}**: deadlineまでに同一seed比較の選択点へ到達せず、結論はcensoredです。")
            continue
        arm = selection.get("selected_arm")
        if arm is None:
            if selection.get("promotion_revoked_endpoint"):
                lines.append(
                    f"- **{task}**: episode 2048の暫定winner "
                    f"**{selection.get('comparative_winner_arm')}** は、episode 4096の"
                    f"{selection.get('promotion_revoked_kind')} gateで昇格取消。"
                    "最終selected armはありません。"
                )
            elif selection.get("control_non_improvement"):
                lines.append(
                    f"- **{task}**: legacy controlが両new armに2/3同一seed勝利。"
                    "新spectral報酬の改善は示されず、10,000 episodeへ昇格なし。"
                )
            else:
                lines.append(f"- **{task}**: scale/phase_v2の明確な2/3同一seed勝利はなく、10,000 episodeへ昇格なし。")
            continue
        endpoint = selection["selection_endpoint"]
        selected_jobs = [row for row in jobs if row["task"] == task and row["arm"] == arm]
        hit_text = ", ".join(
            f"seed{row['seed']}={row.get('first_hit_action_proposals') or 'censored'}"
            for row in selected_jobs
        )
        if endpoint == 2048:
            confirmation = selection.get("confirmation_gate_4096")
            if not isinstance(confirmation, dict) or type(
                confirmation.get("reached")
            ) is not bool:
                raise RuntimeError(
                    f"2048 selection lacks an exact 4096 confirmation state: {task}"
                )
            if confirmation["reached"]:
                conclusion = (
                    f"episode 2048のearly evidenceで **{arm}** を選択し、"
                    "formal 4096 confirmation gateにも到達・通過。"
                )
            else:
                conclusion = (
                    f"episode 2048のearly evidenceで **{arm}** を暫定選択。"
                    "**early evidence / formal4096未到達**のため、"
                    "4096での確認はright-censoredです。"
                )
        else:
            conclusion = f"episode {endpoint}で **{arm}** を選択。"
        lines.append(
            f"- **{task}**: {conclusion}"
            f"初回strict発見のraw action proposalsは {hit_text}。"
        )
    lines += [
        "", "arm選択はreturnを使わず、初回strict発見の有無→発見までのproposal数→直近1024のstrict/joint/constructive→e→F→reject率の順です。", "",
        "## Seed別の結果", "",
        "| task | arm | seed | atomic ep | first hit proposals | strict / joint / constructive (last1024) | median e | median F |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in jobs:
        if not row.get("launched") or not row.get("atomic_endpoint"):
            lines.append(
                f"| {row['task']} | {row['arm']} | {row['seed']} | 0 | "
                f"{row.get('training_censor_reason','censored')} | — | — | — |"
            )
            continue
        first_hit_text = (
            str(row["first_hit_action_proposals"])
            if row.get("first_hit") else
            f"censored@{row.get('first_hit_censor_action_proposals')} "
            f"({row.get('first_hit_censor_reason')})"
        )
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['seed']} | {row['atomic_endpoint']} | "
            f"{first_hit_text} | "
            f"{row['strict_count_last1024']} / {row['joint_count_last1024']} / {row['constructive_count_last1024']} | "
            f"{_fmt(row['spread_error_median_last1024'])} | {_fmt(row['flatness_median_last1024'])} |"
        )
    lines += ["", "## 同一seed pairwise差（選択endpointで打切り）", ""]
    pair_groups = {}
    for row in payload["paired_seed_comparisons"]:
        key = (row["task"], row["endpoint"], row["candidate"], row["opponent"])
        counts = pair_groups.setdefault(key, {"wins": 0, "losses": 0, "ties": 0})
        if row["winner"] == row["candidate"]:
            counts["wins"] += 1
        elif row["winner"] == row["opponent"]:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    if pair_groups:
        lines += [
            "| task | endpoint | candidate | opponent | W-L-T |",
            "|---|---:|---|---|---:|",
        ]
        for key, counts in sorted(pair_groups.items()):
            task, endpoint, candidate, opponent = key
            lines.append(
                f"| {task} | {endpoint} | {candidate} | {opponent} | "
                f"{counts['wins']}-{counts['losses']}-{counts['ties']} |"
            )
        lines.append(
            "各seedのhit/proposal/strict/joint/constructive/e/F/reject差は "
            "`paired_seed_comparisons.csv` に記録しています。"
        )
    else:
        lines.append("選択endpointへ到達した完全な3-arm cohortはありません。")
    lines += ["", "## ランダム探索との初回発見比較", ""]
    if not payload["random_comparisons"]:
        lines.append("10,000 episodeを完了してstrict hitを得た昇格armがないため、事前条件に従いrandom比較は確定していません。")
    for task, item in payload["random_comparisons"].items():
        random = item["random"]
        lines.append(
            f"- **{task}**: 共通cap内でlearning {item['learning_hits_at_common_cap']}/3 hit、"
            f"random {random['hits']}/10 hit（learningのepisode 10,000までのuncapped hitは"
            f"{item['learning_uncapped_hits_by_episode_10000']}/3）。"
            f"共通cap={item['common_action_proposal_cap']}、"
            f"learning cap付きRMST={_fmt(item['learning_restricted_mean_action_proposals'])}、"
            f"random cap付きRMST={_fmt(random['restricted_mean_action_proposals'])} proposals。"
            f"random/learning RMST比={_fmt(item['descriptive_random_rmst_over_learning_rmst'])}"
            f"（{item['ratio_qualifier']}）。hitのみのlearning中央値="
            f"{_fmt(item['learning_median_first_hit_action_proposals'])}は記述値です。"
        )
    lines += [
        "", "random baselineとCodex8は、幾何・uniform action law・max-step・自己交差候補をrejectしてepisode継続する遷移について比較可能です。報酬値のidentityは一致条件に含めず、learningとrandomのseed cohortも異なるため、比は記述的な目安です。", "",
        "## 報酬・観測の注意", "",
        "- legacy/scaleはCodex6 spectral potentialとrollbackをそのまま使い、重みだけが異なります。phase_v2だけがclose→turnの順序付きpotentialです。",
        "- actor/critic入力はCodex6の節点座標（および既存のmode-conditioned action処理）のままで、rho、delta、z、時刻は追加していません。したがってphase_v2はhidden-ledgerを持つ部分観測のablationです。",
        "- phase Psiが高いのにF/eが悪いtimeoutを別集計し、reward farmingをarm選択から除外しています。",
        "- 3 seedsなので不確実性は大きく、2/3規則は再現性の証明ではありません。",
        "- 旧C8SP2はanalyzer偽陽性で中止した監査来歴に限り、fresh replayの曲線・比較には1行も混ぜていません。",
        "", "## Integrity", "",
        f"- manifest: `{payload['manifest_fingerprint']}`",
        "- finalizer/test SHAはソース自身に埋め込まず、外部CLI承認値と実ファイルを開始時・公開直前に照合。",
        "- 実行中PIDなしをfinalization開始時とatomic publish直前に再確認。",
        "- run_config、source/formula hash、episode連続性、reward decomposition、nonfinite、atomic endpointをfail-closedで検証。",
        "- comparison endpoint以後に書かれたoverrun rowは選択判定から除外。",
    ]
    return "\n".join(lines) + "\n"


def validate_output_authority(output: Path, root: Path) -> Path:
    if Path(root).resolve() != DEFAULT_ROOT.resolve():
        raise RuntimeError("Campaign root is fixed exactly")
    if Path(output).resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("Final output is fixed exactly to C8SP2R1_1000")
    return Path(output).resolve()


def verify_payload_inputs_unchanged(payload: dict) -> None:
    approval = payload.get("external_hash_approval", {}).get("approved_sha256", {})
    verify_external_approval(
        approval.get("finalize_ori_codex8_1000.py"),
        approval.get("test_finalize_ori_codex8_1000.py"),
    )
    manifest, binding, report, _ = verify_control_plane(Path(payload["root"]))
    if (
        manifest.get("manifest_fingerprint") != payload.get("manifest_fingerprint")
        or binding.get("manifest_fingerprint") != payload.get("manifest_fingerprint")
        or report != payload.get("runner_report")
    ):
        raise RuntimeError("Control plane changed during finalization")
    for name, expected in payload["source_sha256"].items():
        path = SCRIPT_DIRECTORY / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Bound source changed during finalization: {name}")
    for raw_path, expected in payload["input_sha256"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Consumed input changed during finalization: {path}")


def write_bundle(payload: dict, output: Path, *, enforce_authority=True,
                 publish_active_check=True) -> dict:
    output = Path(output).resolve()
    if enforce_authority:
        output = validate_output_authority(output, Path(payload["root"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building.", dir=output.parent) as tmp:
        build = Path(tmp)
        compact = {key: value for key, value in payload.items()
                   if key not in {"episode_curves", "optimizer_curves"}}
        (build / "final_report.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_csv(build / "job_summary.csv", payload["jobs"])
        _write_csv(
            build / "paired_seed_comparisons.csv",
            payload["paired_seed_comparisons"],
        )
        _write_csv(build / "episode_curves.csv", payload["episode_curves"])
        _write_csv(build / "optimizer_curves.csv", payload["optimizer_curves"])
        (build / "report_ja.md").write_text(japanese_report(payload), encoding="utf-8")
        (build / "input_hashes.json").write_text(json.dumps({
            "manifest_fingerprint": payload["manifest_fingerprint"],
            "source_sha256": payload["source_sha256"],
            "finalizer_source_sha256": payload["finalizer_source_sha256"],
            "external_hash_approval": payload["external_hash_approval"],
            "input_sha256": payload["input_sha256"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        plots = render_plots(payload, build)
        qa = visual_qa(plots)
        (build / "visual_qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        inventory = {
            path.name: sha256_file(path) for path in sorted(build.iterdir()) if path.is_file()
        }
        (build / "output_hashes.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if publish_active_check:
            active = active_codex8_pids()
            if active:
                raise RuntimeError(f"Active Codex8 PID(s) appeared before atomic publish: {active}")
        verify_payload_inputs_unchanged(payload)
        # Recheck *after* the potentially multi-second full input/control-plane
        # revalidation.  This is the check immediately adjacent to publication;
        # the earlier check also catches a trainer active while rendering ends.
        if publish_active_check:
            active = active_codex8_pids()
            if active:
                raise RuntimeError(
                    f"Active Codex8 PID(s) appeared before atomic publish: {active}"
                )
        os.replace(build, output)
    return {"output": str(output), "files": sorted(path.name for path in output.iterdir())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approved-finalizer-sha256", required=True)
    parser.add_argument("--approved-test-sha256", required=True)
    args = parser.parse_args(argv)
    validate_output_authority(args.output, args.root)
    active = active_codex8_pids()
    if active:
        raise RuntimeError(f"Active Codex8 PID(s); refusing finalization: {active}")
    payload = build_payload(
        args.root,
        args.approved_finalizer_sha256,
        args.approved_test_sha256,
        active_check=False,
    )
    result = write_bundle(payload, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
