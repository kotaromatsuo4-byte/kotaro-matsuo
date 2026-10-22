"""Launch-ready C11 zero-reject non-swap optimizer screen.

This is an additive reward ablation of the frozen C10 screen.  The complete
optimizer matrix is retained (five_d2 B/W, five_d3 B/W, watersqrt2 B/E), as
are seeds 9101--9103, proposal-based selection, atomic 512/1024/2048/4096
gates, and continuation of all three seeds in the selected arm to episode
10000.  Only rejected-crossing reward authority and its entrypoint change.

No training can start from the default mode: a source/model/command-bound
manifest must first be authorized, then supplied explicitly to
``--execute-manifest``.  The experiment root is always fresh and immutable.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import analyze_ori_codex11_zero_nonswap_staged as analysis


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BASE_RUNNER = SCRIPT_DIRECTORY / "run_ori_codex10_nonswap_staged.py"
_BASE_MODULE_NAME = "_ori11_c10_nonswap_runner_base"


def _load_frozen_base():
    existing = sys.modules.get(_BASE_MODULE_NAME)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        _BASE_MODULE_NAME, BASE_RUNNER
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[_BASE_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


_base = _load_frozen_base()
_ORIGINAL_BUILD_JOBS = _base.build_jobs
_ORIGINAL_MANIFEST_PAYLOAD = _base.manifest_payload
_ORIGINAL_EXECUTE = _base.execute

EXPERIMENT_DIRECTORY = SCRIPT_DIRECTORY / "experiments_ori11"
DEFAULT_ROOT = EXPERIMENT_DIRECTORY / "C11ZRNS1"
MANIFEST_DIRECTORY = EXPERIMENT_DIRECTORY / "manifests"
RUNNER_NAME = "run_ori_codex11_zero_nonswap_staged.py"
ANALYZER_NAME = "analyze_ori_codex11_zero_nonswap_staged.py"
CAMPAIGN = "C11ZRNS1"
PROTOCOL = "codex6_zero_reject_nonswap_optimizer_screen_to_10000"

SEEDS = analysis.SEEDS
TASK_ORDER = analysis.TASK_ORDER
SCREEN_CHECKPOINTS = analysis.SCREEN_CHECKPOINTS
PRODUCTION_CHECKPOINTS = analysis.PRODUCTION_CHECKPOINTS
TRAINING_DATE_LOCAL = "2026-08-29"
TRAINING_DEADLINE_LOCAL = "09:00"
REPORT_DEADLINE_LOCAL = "09:30"
TRAINING_DEADLINE_LOCAL_ISO = "2026-08-29T09:00:00+09:00"
REPORT_DEADLINE_LOCAL_ISO = "2026-08-29T09:30:00+09:00"
MAX_PARALLEL = 18

EXPECTED_CORE_HASHES = {
    **_base.EXPECTED_CORE_HASHES,
    "run_ori_codex10_nonswap_staged.py": (
        "e015386c7f968b1c4b39861a88b1c206af6756d55357096b2f4294e490418282"
    ),
    "analyze_ori_codex10_nonswap_staged.py": (
        "63764b04bbea989495f5ad11eab511875f326a0e88b78b80b8175c137417899c"
    ),
    "ori_codex6_zero_reject.py": (
        "c113dfeedb8a5e7252e5bac01de9b35d2ccff4d0517392d797bbd5e0a835898d"
    ),
    "ori_codex6_zero_reject_3d.py": (
        "b995787a4764e9b464ab98b8fd3d34621a30d10f1ed58dc22027b123143ace42"
    ),
}

SOURCE_FILES = (
    RUNNER_NAME,
    ANALYZER_NAME,
    "run_ori_codex10_nonswap_staged.py",
    "analyze_ori_codex10_nonswap_staged.py",
    "ori_codex6_zero_reject.py",
    "ori_codex6_zero_reject_3d.py",
    *(
        name for name in _base.SOURCE_FILES
        if name not in {
            "run_ori_codex10_nonswap_staged.py",
            "analyze_ori_codex10_nonswap_staged.py",
        }
    ),
)

THREAD_ENVIRONMENT = dict(_base.THREAD_ENVIRONMENT)
MODEL_FILES = _base.MODEL_FILES
Task = _base.Task
Optimizer = _base.Optimizer
TASKS = _base.TASKS
OPTIMIZERS = _base.OPTIMIZERS

# All reward/geometry/training arguments remain C10-identical.  The explicit
# opt-in flag is the sole parser-level reward change.
COMMON_ARGUMENTS = (*_base.COMMON_ARGUMENTS, "--ori6-zero-reject-reward")


def entrypoint(task: Task) -> Path:
    name = "ori_codex6_zero_reject_3d.py" if task.spatial else "ori_codex6_zero_reject.py"
    return (SCRIPT_DIRECTORY / name).resolve()


def expected_config(task: Task, command: Sequence[str]) -> dict:
    module_name = (
        "ori_codex6_zero_reject_3d" if task.spatial
        else "ori_codex6_zero_reject"
    )
    module = importlib.import_module(module_name)
    parsed = vars(module.parse_args(list(command[3:])))
    json.dumps(parsed, allow_nan=False)
    return parsed


def build_jobs(root: Path) -> list[dict]:
    jobs = _ORIGINAL_BUILD_JOBS(root)
    for identity in jobs:
        identity.pop("literal_codex6_control", None)
        identity["literal_zero_reject_ablation"] = {
            "campaign": CAMPAIGN,
            "candidate": analysis.CANDIDATE,
            "parent_candidate": analysis.PARENT_CANDIDATE,
            "crossing_policy": analysis.CROSSING_POLICY,
            "candidate_committed": False,
            "ordinary_reject_reward": "0",
            "horizon_reject_reward": (
                "-timeout_penalty + quality_weight * q(last_safe)"
            ),
            "configured_crossing_penalty": 2.0,
            "crossing_penalty_applied": False,
            "spectral_weight": 0.05,
            "frontier_weight": 0.9,
            "formula_sha256": analysis.FORMULA_SHA256,
        }
        # The original fingerprint included the now-replaced reward identity.
        identity.pop("job_fingerprint", None)
        identity["job_fingerprint"] = _base.canonical_digest(identity)
    return jobs


def manifest_payload(root: Path, max_parallel: int) -> dict:
    payload = _ORIGINAL_MANIFEST_PAYLOAD(root, max_parallel)
    payload.update({
        "protocol": PROTOCOL,
        "campaign": CAMPAIGN,
        "candidate": analysis.CANDIDATE,
        "reward_ablation_only": True,
        "training_date_local": TRAINING_DATE_LOCAL,
        "training_deadline_local": TRAINING_DEADLINE_LOCAL,
        "report_deadline_local": REPORT_DEADLINE_LOCAL,
        "training_deadline_local_iso": TRAINING_DEADLINE_LOCAL_ISO,
        "report_deadline_local_iso": REPORT_DEADLINE_LOCAL_ISO,
        "paired_seed_design": True,
        "full_c10_optimizer_matrix": {
            task: list(TASKS[task].arms) for task in TASK_ORDER
        },
        "selected_arm_all_three_seeds_continue": True,
    })
    payload.pop("manifest_fingerprint", None)
    content = dict(payload)
    content.pop("generated_utc")
    payload["manifest_fingerprint"] = _base.canonical_digest(content)
    return payload


def manifest_path(payload: dict) -> Path:
    return MANIFEST_DIRECTORY / (
        "codex11_zero_nonswap_" + payload["manifest_fingerprint"][:12] + ".json"
    )


def active_duplicates(root: Path) -> list[int]:
    """Detect both the supervisor and either zero-reject child entrypoint."""
    needle = str(root.resolve()).replace("'", "''")
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -eq 'python.exe' -and $_.CommandLine -like '*"
        + needle + "*'}; @($p | Select-Object ProcessId,CommandLine)"
        "|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip() or "[]")
    records = payload if isinstance(payload, list) else [payload]
    authorities = (
        RUNNER_NAME,
        "ori_codex6_zero_reject.py",
        "ori_codex6_zero_reject_3d.py",
    )
    return [
        int(record["ProcessId"])
        for record in records
        if isinstance(record, dict)
        and int(record.get("ProcessId", -1)) != os.getpid()
        and any(name in str(record.get("CommandLine", "")) for name in authorities)
    ]


def execute(payload: dict, manifest: Path, poll_seconds: float) -> dict:
    """Fail before creating the root if the preregistered local date elapsed."""
    now = datetime.now()
    expected_date = date.fromisoformat(payload["training_date_local"])
    if now.date() != expected_date:
        raise RuntimeError(
            f"campaign is bound to {expected_date.isoformat()}, local date is "
            f"{now.date().isoformat()}"
        )
    deadline = datetime.combine(
        expected_date,
        datetime.strptime(payload["training_deadline_local"], "%H:%M").time(),
    )
    if now >= deadline:
        raise RuntimeError(
            f"training deadline already elapsed: {payload['training_deadline_local_iso']}"
        )
    return _ORIGINAL_EXECUTE(payload, manifest, poll_seconds)


# Redirect every global lookup made by the privately loaded frozen runner.
_base.analysis = analysis
_base.EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY
_base.DEFAULT_ROOT = DEFAULT_ROOT
_base.MANIFEST_DIRECTORY = MANIFEST_DIRECTORY
_base.RUNNER_NAME = RUNNER_NAME
_base.ANALYZER_NAME = ANALYZER_NAME
_base.SEEDS = SEEDS
_base.TASK_ORDER = TASK_ORDER
_base.SCREEN_CHECKPOINTS = SCREEN_CHECKPOINTS
_base.PRODUCTION_CHECKPOINTS = PRODUCTION_CHECKPOINTS
_base.TRAINING_DEADLINE_LOCAL = TRAINING_DEADLINE_LOCAL
_base.REPORT_DEADLINE_LOCAL = REPORT_DEADLINE_LOCAL
_base.MAX_PARALLEL = MAX_PARALLEL
_base.EXPECTED_CORE_HASHES = EXPECTED_CORE_HASHES
_base.SOURCE_FILES = SOURCE_FILES
_base.THREAD_ENVIRONMENT = THREAD_ENVIRONMENT
_base.TASKS = TASKS
_base.OPTIMIZERS = OPTIMIZERS
_base.COMMON_ARGUMENTS = COMMON_ARGUMENTS
_base.entrypoint = entrypoint
_base.expected_config = expected_config
_base.build_jobs = build_jobs
_base.manifest_payload = manifest_payload
_base.manifest_path = manifest_path
_base.active_duplicates = active_duplicates
_base.execute = execute

# Public aliases preserve the C10 runner API for audit/tests.
file_sha256 = _base.file_sha256
canonical_digest = _base.canonical_digest
atomic_json = _base.atomic_json
ensure_below = _base.ensure_below
source_hashes = _base.source_hashes
model_hashes = _base.model_hashes
output_directory = _base.output_directory
build_command = _base.build_command
write_manifest = _base.write_manifest
validate_manifest = _base.validate_manifest
verify_thread_contract = _base.verify_thread_contract
verify_cohort_binding = _base.verify_cohort_binding
completed_endpoint = _base.completed_endpoint
checkpoint_episode = _base.checkpoint_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--authorize-manifest", action="store_true")
    mode.add_argument("--execute-manifest", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)
    if not 6 <= args.max_parallel <= MAX_PARALLEL:
        parser.error(f"--max-parallel must be 6..{MAX_PARALLEL}")
    root = ensure_below(args.experiment_root, SCRIPT_DIRECTORY)
    if args.execute_manifest:
        payload = validate_manifest(
            args.execute_manifest.resolve(), root, args.max_parallel
        )
        result = execute(payload, args.execute_manifest.resolve(), args.poll_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    payload = manifest_payload(root, args.max_parallel)
    if args.draft:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return payload
    path = write_manifest(payload)
    print(path)
    return str(path)


if __name__ == "__main__":
    main()
