"""Launch the bounded water-D=1 spread-alpha screen.

This runner is additive: it reads the completed Codex8 replay manifest for
the literal alpha=1 controls and launches only alpha=2/3 challengers through
``ori_codex9_spread_alpha_3d.py``.  No checkpoint or optimizer state is
reused.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parent
CONTROL_MANIFEST = (
    BASE / "experiments_ori8" / "manifests"
    / "codex8_spectral_staged_replay1_f6c0dcfdbee9.json"
)
ENTRYPOINT = BASE / "ori_codex9_spread_alpha_3d.py"
DEFAULT_ROOT = BASE / "experiments_ori9" / "C9SA1"
ALPHAS = (2.0, 3.0)
SEEDS = (8101, 8102, 8103)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_option(command, option, value):
    command = list(command)
    index = command.index(option)
    command[index + 1] = str(value)
    return command


def build_jobs(root, episodes):
    with open(CONTROL_MANIFEST, encoding="utf-8") as handle:
        control = json.load(handle)
    controls = {
        int(job["seed"]): job for job in control["jobs"]
        if job["task"]["name"] == "water1" and job["arm"] == "scale"
    }
    if tuple(sorted(controls)) != SEEDS:
        raise RuntimeError("Codex8 alpha=1 control matrix is incomplete")

    jobs = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            source = controls[seed]
            command = copy.deepcopy(source["command"])
            command[2] = str(ENTRYPOINT)
            command = replace_option(command, "--episodes", int(episodes))
            output = root / "water1" / f"alpha{int(alpha)}" / f"seed{seed}"
            command = replace_option(command, "--output-dir", output)
            command.extend([
                "--ori9-spread-alpha-ablation",
                "--ori9-spread-alpha", str(int(alpha)),
            ])
            jobs.append({
                "alpha": alpha,
                "seed": seed,
                "output_directory": str(output),
                "command": command,
                "control_output_directory": source["output_directory"],
                "control_job_fingerprint": source["job_fingerprint"],
            })
    return jobs


def parse_deadline(value):
    hour, minute = (int(part) for part in value.split(":"))
    now = datetime.now()
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--deadline", default="14:50")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to reuse experiment root: {root}")
    if args.episodes != 2048:
        raise ValueError("The preregistered small screen is fixed at 2048 episodes")
    if not ENTRYPOINT.is_file():
        raise FileNotFoundError(ENTRYPOINT)
    deadline = parse_deadline(args.deadline)
    if deadline <= datetime.now():
        raise ValueError("deadline must be later today")

    jobs = build_jobs(root, args.episodes)
    root.mkdir(parents=True)
    log_root = root / "_runner_logs"
    log_root.mkdir()
    manifest = {
        "candidate": "codex9_water1_spread_alpha_screen",
        "created_at": datetime.now().astimezone().isoformat(),
        "deadline": deadline.astimezone().isoformat(),
        "episodes": args.episodes,
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "control_alpha": 1.0,
        "control_manifest": str(CONTROL_MANIFEST),
        "control_manifest_sha256": sha256_file(CONTROL_MANIFEST),
        "entrypoint": str(ENTRYPOINT),
        "entrypoint_sha256": sha256_file(ENTRYPOINT),
        "thread_environment": THREAD_ENV,
        "jobs": jobs,
    }
    with open(root / "screen_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    processes = []
    for job in jobs:
        stem = f"alpha{int(job['alpha'])}_seed{job['seed']}"
        stdout_path = log_root / f"{stem}.stdout.log"
        stderr_path = log_root / f"{stem}.stderr.log"
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            job["command"], cwd=BASE, env=environment,
            stdout=stdout_handle, stderr=stderr_handle,
            creationflags=flags,
        )
        processes.append((job, process, stdout_handle, stderr_handle))

    execution = {
        "started_at": datetime.now().astimezone().isoformat(),
        "runner_pid": os.getpid(),
        "children": [
            {"alpha": job["alpha"], "seed": job["seed"], "pid": process.pid}
            for job, process, _, _ in processes
        ],
    }
    with open(root / "execution.json", "w", encoding="utf-8") as handle:
        json.dump(execution, handle, indent=2)

    deadline_stop = False
    try:
        while any(process.poll() is None for _, process, _, _ in processes):
            if datetime.now() >= deadline:
                deadline_stop = True
                for _, process, _, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                cutoff = time.time() + 20.0
                while time.time() < cutoff and any(
                    process.poll() is None for _, process, _, _ in processes
                ):
                    time.sleep(0.25)
                for _, process, _, _ in processes:
                    if process.poll() is None:
                        process.kill()
                break
            time.sleep(1.0)
    finally:
        for _, _, stdout_handle, stderr_handle in processes:
            stdout_handle.close()
            stderr_handle.close()

    report = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "deadline_stop": deadline_stop,
        "jobs": [{
            "alpha": job["alpha"],
            "seed": job["seed"],
            "returncode": process.poll(),
            "output_directory": job["output_directory"],
        } for job, process, _, _ in processes],
    }
    with open(root / "runner_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0 if all(row[1].returncode == 0 for row in processes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
