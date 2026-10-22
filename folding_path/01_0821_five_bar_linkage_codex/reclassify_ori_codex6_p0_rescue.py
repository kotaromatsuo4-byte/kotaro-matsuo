"""Reclassify the deliberately stopped P0 water rescue cohort.

This is a derived-artifact operation.  It never writes into a trainer output
directory and refuses to run while a matching Python child is alive.  It also
preserves the runner's original status payload inside each replacement status.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import run_ori_codex6_experiments as runner


MODE = runner.P0_MODE
PROTOCOL = runner.PROTOCOLS["rescue"]
TASK = "water1"
COMMON_GATE_EPISODE = 1024
ENDPOINTS = {3101: None, 3102: 2048, 3103: 1024}
GATE_DIRECTORY = (
    runner.EXPERIMENT_ROOT / "early_gates" / "rescue" / "P0" / TASK
)
CORRECTED_GATE_PATH = GATE_DIRECTORY / "checkpoint_1024_corrected.json"
REPORT_PATH = (
    runner.EXPERIMENT_ROOT / "analysis" / "rescue_P0_reclassification.json"
)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    ordered = sorted(values)
    return {
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "maximum": float(ordered[-1]),
    }


def return_order(rows: list[dict]) -> dict:
    by_reason: dict[str, list[float]] = {}
    for row in rows[:COMMON_GATE_EPISODE]:
        by_reason.setdefault(str(row["termination_reason"]), []).append(
            float(row["total_reward"])
        )
    crossing = by_reason.get("self_crossing", [])
    timeout = by_reason.get("timeout", [])
    crossing_mean = mean(crossing)
    timeout_mean = mean(timeout)
    estimable = crossing_mean is not None and timeout_mean is not None
    difference = (
        float(crossing_mean - timeout_mean) if estimable else None
    )
    return {
        "checkpoint_episode": COMMON_GATE_EPISODE,
        "crossing_count": len(crossing),
        "timeout_count": len(timeout),
        "crossing_mean_total_reward": crossing_mean,
        "timeout_mean_total_reward": timeout_mean,
        "crossing_minus_timeout_mean_total_reward": difference,
        "crossing_total_reward_quantiles": quantiles(crossing),
        "timeout_total_reward_quantiles": quantiles(timeout),
        "estimable": estimable,
        "required_order": "mean(self_crossing return) < mean(timeout return)",
        "pass": bool(estimable and difference < 0.0),
        "crossing_attractor": bool(estimable and difference >= 0.0),
    }


def main() -> int:
    jobs = {
        int(job["seed"]): job
        for job in runner.build_jobs(PROTOCOL, MODE, [TASK])
    }
    if set(jobs) != set(ENDPOINTS):
        raise RuntimeError(f"Unexpected water cohort seeds: {sorted(jobs)}")

    processes = runner.active_python_processes()
    active = []
    for seed, job in jobs.items():
        destination = Path(job["output_directory"])
        process = runner.external_process(destination, processes)
        if process is not None and runner.process_is_alive(process["pid"]):
            active.append({"seed": seed, **process})
    if active:
        raise RuntimeError(f"Refusing to reclassify active jobs: {active}")

    original_statuses = {}
    validations = {}
    rows_by_seed = {}
    stderr_by_seed = {}
    for seed, job in jobs.items():
        destination = Path(job["output_directory"])
        status_file = runner.status_path(job, destination)
        status = load_json(status_file)
        if status.get("job_fingerprint") != job["fingerprint"]:
            raise RuntimeError(f"seed{seed}: job fingerprint mismatch")
        log_directory = status_file.parent
        stderr_path = log_directory / "stderr.log"
        stderr_bytes = stderr_path.stat().st_size
        if stderr_bytes:
            raise RuntimeError(f"seed{seed}: stderr has {stderr_bytes} bytes")
        endpoint = ENDPOINTS[seed]
        validation = runner.validate_artifact(
            job, destination, endpoint=endpoint
        )
        if not validation["valid"]:
            raise RuntimeError(
                f"seed{seed}: artifact validation failed: {validation['errors']}"
            )
        rows = runner.read_csv(destination / "training_history.csv")
        if len(rows) < COMMON_GATE_EPISODE:
            raise RuntimeError(f"seed{seed}: fewer than 1024 atomic rows")
        original_statuses[seed] = status
        validations[seed] = validation
        rows_by_seed[seed] = rows
        stderr_by_seed[seed] = {
            "path": str(stderr_path.resolve()), "bytes": stderr_bytes,
        }

    base_decision = runner.early_gate_decision(
        "rescue", TASK, rows_by_seed, COMMON_GATE_EPISODE
    )
    orders = {
        str(seed): return_order(rows) for seed, rows in rows_by_seed.items()
    }
    pass_count = sum(row["pass"] for row in orders.values())
    attractor_count = sum(row["crossing_attractor"] for row in orders.values())
    formal_pass_possible = pass_count >= 2
    if formal_pass_possible or attractor_count < 2:
        raise RuntimeError(
            "Corrected stop condition not met: "
            f"pass_count={pass_count}, attractor_count={attractor_count}"
        )
    corrected_decision = {
        **base_decision,
        "schema_version": 2,
        "stop": True,
        "reason": "water1_crossing_return_order_formally_impossible_2_of_3_at_checkpoint_1024",
        "supersedes": str(
            (GATE_DIRECTORY / "checkpoint_1024.json").resolve()
        ),
        "formal_return_order_gate": {
            "common_checkpoint_episode": COMMON_GATE_EPISODE,
            "required_seed_pass_count": 2,
            "observed_seed_pass_count": pass_count,
            "observed_crossing_attractor_seed_count": attractor_count,
            "pass_possible_from_fixed_checkpoint_data": formal_pass_possible,
            "per_seed": orders,
        },
        "termination_scope": {
            "completed_seed_preserved": 3101,
            "stopped_atomic_training_endpoints": {
                "3102": ENDPOINTS[3102], "3103": ENDPOINTS[3103],
            },
            "all_matching_python_processes_absent_before_reclassification": True,
            "stderr_by_seed": stderr_by_seed,
        },
        "source_sha256": runner.source_hashes(),
        "model_sha256": runner.model_hashes(runner.TASKS[TASK]),
    }
    runner.atomic_json(CORRECTED_GATE_PATH, corrected_decision)
    gate_hash = runner.file_sha256(CORRECTED_GATE_PATH)

    status_records = []
    for seed, job in jobs.items():
        destination = Path(job["output_directory"])
        status_file = runner.status_path(job, destination)
        prior = original_statuses[seed]
        replacement = dict(prior)
        replacement["prior_runner_classification"] = {
            "status": prior.get("status"),
            "artifact_validation": prior.get("artifact_validation"),
            "return_code": prior.get("return_code"),
            "note": (
                "The live runner used the pre-fix ori6 curve filename gate; "
                "interrupted children also lacked post-training evaluation artifacts."
            ),
        }
        replacement["effective_output_directory"] = str(destination.resolve())
        replacement["reclassified_utc"] = runner.utc_now()
        replacement["reclassification_authority"] = str(REPORT_PATH.resolve())
        if seed == 3101:
            replacement["status"] = "complete"
            replacement["artifact_validation"] = validations[seed]
        else:
            replacement.update({
                "status": "early_stopped_at_checkpoint",
                "checkpoint_episode": ENDPOINTS[seed],
                "early_gate_decision": corrected_decision,
                "early_gate_decision_path": str(CORRECTED_GATE_PATH.resolve()),
                "early_gate_decision_sha256": gate_hash,
                "partial_artifact_validation": validations[seed],
            })
        runner.atomic_json(status_file, replacement)
        status_records.append({
            "seed": seed,
            "status_path": str(status_file.resolve()),
            "status": replacement["status"],
            "endpoint_episode": validations[seed]["endpoint_episode"],
            "artifact_validation": validations[seed],
            "stderr": stderr_by_seed[seed],
        })

    report = {
        "schema_version": 1,
        "stage": "rescue",
        "candidate_mode": MODE,
        "candidate_mode_slug": runner.MODE_SLUGS[MODE],
        "task": TASK,
        "corrected_decision_path": str(CORRECTED_GATE_PATH.resolve()),
        "corrected_decision_sha256": gate_hash,
        "corrected_decision": corrected_decision,
        "status_records": status_records,
        "canonical_manifest": str(
            runner.manifest_path(PROTOCOL, MODE, list(jobs.values())).resolve()
        ),
        "source_sha256": runner.source_hashes(),
    }
    # Write the report last so every referenced status and decision is already
    # durable.  Statuses point to this deterministic path, not its later hash.
    runner.atomic_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
