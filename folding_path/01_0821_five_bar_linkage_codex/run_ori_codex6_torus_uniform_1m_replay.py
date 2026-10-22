"""Fail-closed additive launcher for the torus uniform 1M replay.

The original random harness is deliberately left unchanged.  A worker imports
that frozen harness, verifies its SHA-256, and replaces only its atomic
``torch.save`` helper.  The replacement writes through an already-open file
handle because the installed PyTorch rejects the parent's hidden temporary
filename.

The 636,198-proposal trials cannot be continued in place: their budget is part
of the root fingerprint, their final partial episode has 198 steps, and their
summaries do not contain RNG/environment state.  Consequently this launcher
performs a fresh deterministic replay to 1,000,000 proposals in new roots.
It then checks the 102,400-action chain and every non-timing episode-history
field through episode 3,180 (636,000 proposals) against the completed parent
trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
import torch

import ori_codex6_spectral_random_baseline as parent


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PARENT_SOURCE = SCRIPT_DIRECTORY / "ori_codex6_spectral_random_baseline.py"
PARENT_SHA256 = "0a67c9f6f8ca10813d97dd5b9b77a0c009d3991bc68541e2b32ef6008e0d0525"
TASK = "torus2"
SEEDS = tuple(range(6101, 6111))
PARENT_BUDGET = 636_198
TARGET_BUDGET = 1_000_000
FULL_PREFIX_EPISODES = 3_180
FULL_PREFIX_MECHANICS = 636_000
CHAIN_PREFIX_MECHANICS = 102_400
CHECKPOINT_EVERY_EPISODES = 512
PRINT_EVERY_EPISODES = 100
DEFAULT_BASE = (
    SCRIPT_DIRECTORY
    / "experiments_ori6/random/first_discovery_matched_20260827/torus2"
)
DEFAULT_LOG_ROOT = (
    SCRIPT_DIRECTORY
    / "experiments_ori6/_runner_logs/first_discovery_matched_20260827_1m"
)
TIMING_FIELDS = {"rollout_seconds", "cumulative_rollout_seconds"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def robust_atomic_torch_save(path: Path, value) -> None:
    """Atomically save via a file object, avoiding PyTorch path validation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "xb") as handle:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return
        except FileExistsError:
            continue
        finally:
            if temporary.exists():
                temporary.unlink()
    raise RuntimeError(f"Could not allocate atomic temporary near {path}")


def ensure_below(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Path escaped expected root: {resolved}")
    return resolved


def parent_root(base: Path, seed: int) -> Path:
    return ensure_below(
        base / f"budget{PARENT_BUDGET}" / f"seed{seed}", base
    )


def target_root(base: Path, seed: int) -> Path:
    return ensure_below(
        base / f"budget{TARGET_BUDGET}" / f"seed{seed}", base
    )


def active_random_processes() -> list[dict]:
    found = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        joined = " ".join(command)
        if (
            "ori_codex6_spectral_random_baseline.py" in joined
            or Path(__file__).name in joined
        ) and "--task torus2" in joined:
            found.append({
                "pid": int(process.info["pid"]),
                "command": joined,
                "created": float(process.info.get("create_time") or 0.0),
            })
    return found


def retry_stderr_path(seed: int) -> Path:
    return (
        SCRIPT_DIRECTORY
        / "experiments_ori6/_runner_logs/first_discovery_matched_20260827"
        / f"torus2-seed{seed}.retry1.stderr.log"
    )


def validate_parent_trial(base: Path, seed: int) -> dict:
    root = parent_root(base, seed)
    config_path = root / "run_config.json"
    trial = root / "trials/uniform" / f"seed{seed}"
    summary_path = trial / "summary.json"
    history_path = trial / "episode_history.csv"
    errors = []
    for path in (config_path, summary_path, history_path):
        if not path.is_file():
            errors.append(f"missing:{path}")
    stderr = retry_stderr_path(seed)
    if not stderr.is_file() or stderr.stat().st_size != 0:
        errors.append(f"retry_stderr_not_clean:{stderr}")
    if errors:
        return {"seed": seed, "pass": False, "errors": errors}

    config = load_json(config_path)
    summary = load_json(summary_path)
    source_hashes = config.get("source_sha256", {})
    expected = {
        "task": TASK,
        "seeds": [seed],
        "mechanics_budget": PARENT_BUDGET,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            errors.append(f"config_{field}")
    if source_hashes.get(PARENT_SOURCE.name) != PARENT_SHA256:
        errors.append("config_parent_source_sha256")
    if (
        summary.get("complete") is not True
        or int(summary.get("seed", -1)) != seed
        or int(summary.get("mechanics_budget", -1)) != PARENT_BUDGET
        or summary.get("reward_integrity_pass") is not True
    ):
        errors.append("summary_identity_or_integrity")
    mechanics = int(summary.get("mechanical_steps", -1))
    if summary.get("success"):
        if not 1 <= mechanics <= PARENT_BUDGET:
            errors.append("successful_parent_mechanics")
    elif mechanics != PARENT_BUDGET:
        errors.append("censored_parent_not_exact")
    chain = summary.get("action_prefix_sha256_by_mechanics", {}).get(
        str(CHAIN_PREFIX_MECHANICS)
    )
    if not isinstance(chain, str) or len(chain) != 64:
        errors.append("missing_102400_chain")
    history = read_csv(history_path)
    if len(history) < FULL_PREFIX_EPISODES:
        errors.append("parent_history_shorter_than_3180")
    elif int(history[FULL_PREFIX_EPISODES - 1][
        "cumulative_mechanical_steps"
    ]) != FULL_PREFIX_MECHANICS:
        errors.append("parent_636000_boundary")
    return {
        "seed": seed,
        "pass": not errors,
        "errors": errors,
        "root": str(root),
        "summary": str(summary_path),
        "history": str(history_path),
        "success": int(bool(summary.get("success"))),
        "mechanical_steps": mechanics,
        "chain_102400": chain,
        "root_fingerprint": config.get("root_fingerprint"),
    }


def expected_target_config(seed: int) -> dict:
    return parent.make_root_config(TASK, TARGET_BUDGET, (seed,))


def launcher_identity(base: Path) -> dict:
    return {
        "schema_version": 1,
        "protocol": "torus_uniform_fresh_deterministic_replay_to_1m",
        "launcher": Path(__file__).name,
        "launcher_sha256": file_sha256(Path(__file__)),
        "frozen_parent": PARENT_SOURCE.name,
        "frozen_parent_sha256": file_sha256(PARENT_SOURCE),
        "expected_parent_sha256": PARENT_SHA256,
        "task": TASK,
        "seeds": list(SEEDS),
        "parent_budget": PARENT_BUDGET,
        "target_budget": TARGET_BUDGET,
        "base": str(base.resolve()),
        "checkpoint_every_episodes": CHECKPOINT_EVERY_EPISODES,
        "print_every_episodes": PRINT_EVERY_EPISODES,
        "prefix_contract": {
            "chain_mechanics": CHAIN_PREFIX_MECHANICS,
            "full_episode_prefix_episodes": FULL_PREFIX_EPISODES,
            "full_episode_prefix_mechanics": FULL_PREFIX_MECHANICS,
            "uncompared_partial_parent_episode_steps": (
                PARENT_BUDGET - FULL_PREFIX_MECHANICS
            ),
        },
        "target_root_fingerprints": {
            str(seed): expected_target_config(seed)["root_fingerprint"]
            for seed in SEEDS
        },
    }


def preflight(base: Path, *, permit_owned_resume: bool = False) -> dict:
    errors = []
    if file_sha256(PARENT_SOURCE) != PARENT_SHA256:
        errors.append("frozen_parent_source_hash_changed")
    active = active_random_processes()
    if active:
        errors.append("active_torus_random_processes")
    parents = [validate_parent_trial(base, seed) for seed in SEEDS]
    if not all(row["pass"] for row in parents):
        errors.append("parent_636198_cohort_not_clean")
    targets = []
    for seed in SEEDS:
        root = target_root(base, seed)
        state = "absent"
        if root.exists():
            state = "existing"
            if not permit_owned_resume:
                errors.append(f"target_root_exists:{seed}")
            else:
                config_path = root / "run_config.json"
                if not config_path.is_file():
                    errors.append(f"resume_root_missing_config:{seed}")
                else:
                    actual = load_json(config_path).get("root_fingerprint")
                    expected = expected_target_config(seed)["root_fingerprint"]
                    if actual != expected:
                        errors.append(f"resume_root_fingerprint:{seed}")
        targets.append({"seed": seed, "root": str(root), "state": state})
    result = {
        **launcher_identity(base),
        "ready": not errors,
        "errors": errors,
        "active_processes": active,
        "parent_trials": parents,
        "target_roots": targets,
        "permit_owned_resume": bool(permit_owned_resume),
    }
    result["preflight_sha256"] = canonical_digest(result)
    return result


def worker(seed: int, base: Path) -> int:
    if seed not in SEEDS:
        raise ValueError("Worker seed escaped the frozen cohort")
    if file_sha256(PARENT_SOURCE) != PARENT_SHA256:
        raise RuntimeError("Frozen parent source hash changed")
    parent.atomic_torch_save = robust_atomic_torch_save
    root = target_root(base, seed)
    parent.main([
        "--task", TASK,
        "--mechanics-budget", str(TARGET_BUDGET),
        "--trial-seeds", str(seed),
        "--output-root", str(root),
        "--resume",
        "--checkpoint-every-episodes", str(CHECKPOINT_EVERY_EPISODES),
        "--print-every", str(PRINT_EVERY_EPISODES),
    ])
    return 0


def comparable_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in TIMING_FIELDS}


def validate_target_trial(base: Path, seed: int) -> dict:
    old = validate_parent_trial(base, seed)
    root = target_root(base, seed)
    trial = root / "trials/uniform" / f"seed{seed}"
    paths = {
        "config": root / "run_config.json",
        "summary": trial / "summary.json",
        "history": trial / "episode_history.csv",
    }
    errors = []
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing_{name}")
    stderr = DEFAULT_LOG_ROOT / f"torus2-seed{seed}.stderr.log"
    if not stderr.is_file() or stderr.stat().st_size != 0:
        errors.append("worker_stderr_not_clean")
    if errors:
        return {"seed": seed, "pass": False, "errors": errors}
    config = load_json(paths["config"])
    summary = load_json(paths["summary"])
    expected = expected_target_config(seed)
    if config.get("root_fingerprint") != expected["root_fingerprint"]:
        errors.append("target_root_fingerprint")
    if config.get("source_sha256", {}).get(PARENT_SOURCE.name) != PARENT_SHA256:
        errors.append("target_parent_source_hash")
    if (
        summary.get("complete") is not True
        or int(summary.get("mechanics_budget", -1)) != TARGET_BUDGET
        or summary.get("reward_integrity_pass") is not True
    ):
        errors.append("target_summary_identity_or_integrity")
    mechanics = int(summary.get("mechanical_steps", -1))
    if summary.get("success"):
        if not 1 <= mechanics <= TARGET_BUDGET:
            errors.append("successful_target_mechanics")
    elif mechanics != TARGET_BUDGET:
        errors.append("censored_target_not_exact_1m")
    new_chain = summary.get("action_prefix_sha256_by_mechanics", {}).get(
        str(CHAIN_PREFIX_MECHANICS)
    )
    if new_chain != old.get("chain_102400"):
        errors.append("chain_102400_mismatch")
    old_rows = read_csv(Path(old["history"]))
    new_rows = read_csv(paths["history"])
    if len(new_rows) < FULL_PREFIX_EPISODES:
        errors.append("target_history_shorter_than_3180")
        mismatch_episode = None
    else:
        mismatch_episode = None
        for index in range(FULL_PREFIX_EPISODES):
            if comparable_row(old_rows[index]) != comparable_row(new_rows[index]):
                mismatch_episode = index + 1
                errors.append(f"full_episode_prefix_mismatch:{index + 1}")
                break
        if int(new_rows[FULL_PREFIX_EPISODES - 1][
            "cumulative_mechanical_steps"
        ]) != FULL_PREFIX_MECHANICS:
            errors.append("target_636000_boundary")
    checkpoint = trial / "checkpoint.pt"
    return {
        "seed": seed,
        "pass": not errors,
        "errors": errors,
        "success": int(bool(summary.get("success"))),
        "mechanical_steps": mechanics,
        "chain_102400_match": new_chain == old.get("chain_102400"),
        "full_episode_prefix_match": mismatch_episode is None,
        "full_episode_prefix_mechanics": FULL_PREFIX_MECHANICS,
        "checkpoint_exists": checkpoint.is_file(),
        "root_fingerprint": config.get("root_fingerprint"),
    }


def final_audit(base: Path) -> dict:
    trials = [validate_target_trial(base, seed) for seed in SEEDS]
    result = {
        **launcher_identity(base),
        "complete": all(row["pass"] for row in trials),
        "trials": trials,
        "successes": sum(row.get("success", 0) for row in trials),
        "censored": sum(1 - row.get("success", 0) for row in trials),
        "completed_utc": parent.utc_now(),
    }
    result["audit_sha256"] = canonical_digest(result)
    return result


def launch(base: Path, log_root: Path, *, approved_sha256: str,
           permit_owned_resume: bool = False) -> dict:
    current_sha256 = file_sha256(Path(__file__))
    if approved_sha256.lower() != current_sha256:
        raise RuntimeError("Independent audit did not approve this launcher hash")
    check = preflight(base, permit_owned_resume=permit_owned_resume)
    if not check["ready"]:
        raise RuntimeError("Preflight blocked: " + ",".join(check["errors"]))
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for seed in SEEDS:
        stdout_path = log_root / f"torus2-seed{seed}.stdout.log"
        stderr_path = log_root / f"torus2-seed{seed}.stderr.log"
        stdout = open(stdout_path, "a" if permit_owned_resume else "x", encoding="utf-8")
        stderr = open(stderr_path, "a" if permit_owned_resume else "x", encoding="utf-8")
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker", "--seed", str(seed), "--base", str(base.resolve()),
        ]
        process = subprocess.Popen(
            command, cwd=SCRIPT_DIRECTORY, stdout=stdout, stderr=stderr,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        processes.append((seed, process, stdout, stderr))
    failures = []
    try:
        while processes:
            remaining = []
            for seed, process, stdout, stderr in processes:
                code = process.poll()
                if code is None:
                    remaining.append((seed, process, stdout, stderr))
                    continue
                stdout.close()
                stderr.close()
                if code != 0:
                    failures.append({"seed": seed, "returncode": code})
            processes = remaining
            if processes:
                time.sleep(2.0)
    finally:
        for _, _, stdout, stderr in processes:
            stdout.close()
            stderr.close()
    if failures:
        raise RuntimeError(f"1M replay worker failures: {failures}")
    audit = final_audit(base)
    atomic_json(log_root / "final_audit.json", audit)
    if not audit["complete"]:
        raise RuntimeError("Final prefix/exact-budget audit failed")
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-launcher-sha256")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if sum((args.dry_run, args.execute, args.worker)) != 1:
        parser.error("select exactly one of --dry-run, --execute, --worker")
    if args.worker and args.seed not in SEEDS:
        parser.error("--worker requires a frozen cohort --seed")
    if args.execute and not args.approved_launcher_sha256:
        parser.error("--execute requires --approved-launcher-sha256")
    return args


def main(argv=None):
    args = parse_args(argv)
    base = ensure_below(args.base, SCRIPT_DIRECTORY)
    log_root = ensure_below(args.log_root, SCRIPT_DIRECTORY)
    if args.worker:
        return worker(args.seed, base)
    if args.dry_run:
        result = preflight(base, permit_owned_resume=args.resume_existing)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    result = launch(
        base, log_root,
        approved_sha256=args.approved_launcher_sha256,
        permit_owned_resume=args.resume_existing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
