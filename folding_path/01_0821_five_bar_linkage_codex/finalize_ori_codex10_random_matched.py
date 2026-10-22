"""Fail-closed C10NS1 selected-policy versus C10R2 uniform comparison.

This additive finalizer never trains, resumes, or stops a process.  It accepts
only the immutable C10NS1 control plane and admissible rc=0 C10R2 trials.  The
primary estimand is first strict discovery on the common raw-action-proposal
clock fixed separately for each task.  Returns and reward magnitudes are not
comparison outcomes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time

import finalize_ori_codex10_nonswap_staged as c10final
import run_ori_codex10_nonswap_random_matched as random_runner


SCRIPT_DIR = Path(__file__).resolve().parent
C10_ROOT = (SCRIPT_DIR / "experiments_ori10" / "C10NS1").resolve()
RANDOM_ROOT = (SCRIPT_DIR / "experiments_ori10" / "C10R2").resolve()
RANDOM_MANIFEST = (
    SCRIPT_DIR / "experiments_ori10" / "manifests"
    / "codex10_nonswap_random_matched_71ce7d754c2a.json"
).resolve()
OUTPUT = (
    SCRIPT_DIR / "experiments_ori10" / "analysis"
    / "C10NS1_vs_C10R2_random"
).resolve()
TASKS = ("five_d2", "five_d3", "watersqrt2")
CAPS = {"five_d2": 819_200, "five_d3": 409_600, "watersqrt2": 409_600}
LEARNING_SEEDS = (9101, 9102, 9103)
RANDOM_SEEDS = tuple(range(6101, 6111))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_RANDOM_MANIFEST_SHA256 = (
    "afb3d88ec693890425d7b80bd6adfa5f1deb8b09861c452bda8826088c2989ba"
)
EXPECTED_RANDOM_MANIFEST_FINGERPRINT = (
    "71ce7d754c2a39e62e9bc8d1ac48f7e858fe7b1363cd7674f7e9cd438512cd8b"
)
EXPECTED_RANDOM_RUNNER_SHA256 = (
    "d6c4e414be42fef8666b2ecb90ae01ce8a64a79e0ae29e89c726d169a084d9e9"
)
EXPECTED_RANDOM_TEST_SHA256 = (
    "e5fb490e84983e9c8c7f08f74d5935a57152f4fef29d431ea884e0006ca73837"
)
EXPECTED_C10_FINALIZER_SHA256 = (
    "e3e6b121bdcefd62f7c720145ed49eb92fb92f65d5e728b8dd57127345197d45"
)
EXPECTED_C10_FINALIZER_TEST_SHA256 = (
    "5ced4493fa027eb9032873d8ad553110ad770e767c350092b3131b29bb96370e"
)
SELF_NAMES = (
    "finalize_ori_codex10_random_matched.py",
    "test_finalize_ori_codex10_random_matched.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_bytes(path: Path) -> bytes:
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while reading: {path}")
    return data


def record_input(path: Path, inputs: dict[str, str]) -> bytes:
    path = Path(path).resolve()
    data = stable_bytes(path)
    digest = sha256_bytes(data)
    previous = inputs.get(str(path))
    if previous is not None and previous != digest:
        raise RuntimeError(f"Input changed between reads: {path}")
    inputs[str(path)] = digest
    return data


def load_json(path: Path, inputs: dict[str, str]) -> dict:
    value = json.loads(record_input(path, inputs).decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input is not an object: {path}")
    return value


def load_csv(path: Path, inputs: dict[str, str]) -> list[dict]:
    text = record_input(path, inputs).decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def verify_inputs(inputs: dict[str, str]) -> None:
    for raw_path, expected in sorted(inputs.items()):
        path = Path(raw_path)
        if not path.is_file() or sha256_bytes(stable_bytes(path)) != expected:
            raise RuntimeError(f"Consumed input changed before publish: {path}")


def exact_int(value, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} is not an exact integer")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{label} is below {minimum}")
    return value


def exact_binary(value, label: str) -> int:
    result = exact_int(value, label)
    if result not in (0, 1):
        raise RuntimeError(f"{label} is not binary")
    return result


def require_sha(value: str, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeError(f"{label} is not SHA-256")
    return value.lower()


def parse_utc(value, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} lacks timezone")
    return parsed


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise RuntimeError("Cannot take percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_group(costs: list[int], events: list[int], seed: int) -> dict:
    if not costs or len(costs) != len(events):
        raise RuntimeError("Bootstrap group is empty or misaligned")
    rng = random.Random(seed)
    rates, rmsts = [], []
    n = len(costs)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(n) for _ in range(n)]
        rates.append(sum(events[index] for index in indices) / n)
        rmsts.append(sum(costs[index] for index in indices) / n)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "hit_rate": sum(events) / n,
        "hit_rate_ci_low": percentile(rates, alpha),
        "hit_rate_ci_high": percentile(rates, 1.0 - alpha),
        "rmst": sum(costs) / n,
        "rmst_ci_low": percentile(rmsts, alpha),
        "rmst_ci_high": percentile(rmsts, 1.0 - alpha),
    }


def bootstrap_speedup(
    learning_costs: list[int], random_costs: list[int], seed: int,
) -> dict:
    if not learning_costs or not random_costs:
        raise RuntimeError("Speedup bootstrap requires both groups")
    rng = random.Random(seed)
    ratios = []
    for _ in range(BOOTSTRAP_REPLICATES):
        learning = sum(
            learning_costs[rng.randrange(len(learning_costs))]
            for _ in learning_costs
        ) / len(learning_costs)
        control = sum(
            random_costs[rng.randrange(len(random_costs))]
            for _ in random_costs
        ) / len(random_costs)
        ratios.append(control / learning)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "random_over_learning_rmst": (
            (sum(random_costs) / len(random_costs))
            / (sum(learning_costs) / len(learning_costs))
        ),
        "random_over_learning_rmst_ci_low": percentile(ratios, alpha),
        "random_over_learning_rmst_ci_high": percentile(
            ratios, 1.0 - alpha
        ),
    }


def validate_external_source_authority(inputs: dict[str, str]) -> dict[str, str]:
    expected = {
        "run_ori_codex10_nonswap_random_matched.py": (
            EXPECTED_RANDOM_RUNNER_SHA256
        ),
        "test_run_ori_codex10_nonswap_random_matched.py": (
            EXPECTED_RANDOM_TEST_SHA256
        ),
        "finalize_ori_codex10_nonswap_staged.py": (
            EXPECTED_C10_FINALIZER_SHA256
        ),
        "test_finalize_ori_codex10_nonswap_staged.py": (
            EXPECTED_C10_FINALIZER_TEST_SHA256
        ),
    }
    for name, digest in expected.items():
        observed = sha256_bytes(record_input(SCRIPT_DIR / name, inputs))
        if observed != digest:
            raise RuntimeError(f"Approved source changed: {name}")
    return expected


def validate_self_approval(
    approved_finalizer_sha256: str, approved_test_sha256: str,
    inputs: dict[str, str],
) -> dict[str, str]:
    expected = {
        SELF_NAMES[0]: require_sha(
            approved_finalizer_sha256, "approved finalizer SHA"
        ),
        SELF_NAMES[1]: require_sha(approved_test_sha256, "approved test SHA"),
    }
    for name, digest in expected.items():
        observed = sha256_bytes(record_input(SCRIPT_DIR / name, inputs))
        if observed != digest:
            raise RuntimeError(f"External self-approval mismatch: {name}")
    return expected


def random_artifact_paths(job: dict, result: dict) -> dict[str, Path]:
    output = Path(job["output_root"])
    seed = job["seed"]
    trial = output / "trials" / "uniform" / f"seed{seed}"
    paths = {
        "root_config": output / "run_config.json",
        "aggregate": output / "aggregate.json",
        "trial_config": trial / "run_config.json",
        "summary": trial / "summary.json",
        "history": trial / "episode_history.csv",
    }
    for key in result.get("artifact_sha256", {}):
        if key.startswith("first_hit_path/"):
            paths[key] = trial / key
    return paths


def load_random_campaign(inputs: dict[str, str]) -> tuple[dict, list[dict]]:
    if sha256_bytes(record_input(RANDOM_MANIFEST, inputs)) != (
        EXPECTED_RANDOM_MANIFEST_SHA256
    ):
        raise RuntimeError("Random manifest file SHA mismatch")
    manifest = json.loads(stable_bytes(RANDOM_MANIFEST).decode("utf-8"))
    random_runner.validate_manifest(manifest, RANDOM_MANIFEST)
    if manifest.get("manifest_fingerprint") != (
        EXPECTED_RANDOM_MANIFEST_FINGERPRINT
    ) or Path(manifest.get("experiment_root", "")).resolve() != RANDOM_ROOT:
        raise RuntimeError("Random manifest fingerprint/root mismatch")
    binding = load_json(RANDOM_ROOT / "execution_binding.json", inputs)
    report = load_json(RANDOM_ROOT / "runner_report.json", inputs)
    expected_binding_keys = {
        "schema_version", "protocol", "manifest_path", "manifest_sha256",
        "manifest_fingerprint", "supervisor_pid", "started_utc",
        "thread_environment", "deadline_local", "status",
    }
    if set(binding) != expected_binding_keys or binding.get("status") != "running":
        raise RuntimeError("Random execution binding schema mismatch")
    if exact_int(binding.get("schema_version"), "random binding schema") != 1:
        raise RuntimeError("Random binding schema version mismatch")
    if exact_int(binding.get("supervisor_pid"), "random supervisor PID", 1) < 1:
        raise RuntimeError("Random supervisor PID is invalid")
    started = parse_utc(binding.get("started_utc"), "random started_utc")
    if binding.get("thread_environment") != random_runner.THREAD_ENV \
            or binding.get("deadline_local") != random_runner.DEADLINE_LOCAL:
        raise RuntimeError("Random thread/deadline binding mismatch")
    if (
        Path(binding.get("manifest_path", "")).resolve() != RANDOM_MANIFEST
        or require_sha(binding.get("manifest_sha256"), "bound manifest SHA")
        != EXPECTED_RANDOM_MANIFEST_SHA256
        or binding.get("manifest_fingerprint")
        != EXPECTED_RANDOM_MANIFEST_FINGERPRINT
        or binding.get("protocol") != random_runner.PROTOCOL
    ):
        raise RuntimeError("Random binding authority mismatch")
    for key in (
        "schema_version", "protocol", "manifest_path", "manifest_sha256",
        "manifest_fingerprint", "supervisor_pid", "started_utc",
        "thread_environment", "deadline_local",
    ):
        if report.get(key) != binding.get(key):
            raise RuntimeError(f"Random report/binding mismatch: {key}")
    expected_report_keys = expected_binding_keys | {
        "finished_utc", "expected_jobs", "admissible_jobs",
        "partial_or_failed_jobs", "all_formal_jobs_complete", "fatal_error",
        "results", "launch_audit", "pending_unlaunched",
    }
    if set(report) != expected_report_keys:
        raise RuntimeError("Random runner report schema mismatch")
    finished = parse_utc(report.get("finished_utc"), "random finished_utc")
    if finished < started:
        raise RuntimeError("Random finish precedes start")
    if not isinstance(report.get("launch_audit"), list):
        raise RuntimeError("Random launch audit is not a list")
    if report.get("status") not in {"complete", "incomplete"}:
        raise RuntimeError("Random runner report status is invalid")
    if exact_int(report.get("expected_jobs"), "random expected jobs") != 30:
        raise RuntimeError("Random expected job count mismatch")
    admissible_count = exact_int(
        report.get("admissible_jobs"), "random admissible jobs"
    )
    if exact_int(
        report.get("partial_or_failed_jobs"), "random excluded jobs"
    ) != 30 - admissible_count:
        raise RuntimeError("Random report count decomposition mismatch")
    if not isinstance(report.get("all_formal_jobs_complete"), bool):
        raise RuntimeError("Random completion flag is not boolean")
    if report.get("all_formal_jobs_complete") != (
        admissible_count == 30
    ):
        raise RuntimeError("Random report completion flag mismatch")
    if (report.get("status") == "complete") != (admissible_count == 30):
        raise RuntimeError("Random report status/count mismatch")
    if admissible_count == 30 and report.get("fatal_error") is not None:
        raise RuntimeError("Complete random report has a fatal error")
    jobs = {job["job_fingerprint"]: job for job in manifest["jobs"]}
    matrix = {(job["task"], job["seed"]) for job in manifest["jobs"]}
    results = report.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Random results are not a list")
    seen_fingerprints, seen_matrix, observations = set(), set(), []
    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("Random result row is not an object")
        fingerprint = row.get("job_fingerprint")
        if fingerprint not in jobs or fingerprint in seen_fingerprints:
            raise RuntimeError("Random result fingerprint is invalid or duplicated")
        seen_fingerprints.add(fingerprint)
        job = jobs[fingerprint]
        identity = (job["task"], job["seed"])
        if row.get("task") != identity[0] or row.get("seed") != identity[1]:
            raise RuntimeError("Random result task/seed mismatch")
        seen_matrix.add(identity)
        if not isinstance(row.get("admissible"), bool):
            raise RuntimeError("Random admissible flag is not boolean")
        admissible = row["admissible"]
        if admissible:
            if exact_int(row.get("return_code"), "random return code") != 0:
                raise RuntimeError("Admissible random result has nonzero rc")
            validated = random_runner.validate_completed_job(job)
            for key, value in validated.items():
                if row.get(key) != value:
                    raise RuntimeError(
                        f"Random report/artifact mismatch: {identity}/{key}"
                    )
            for key, path in random_artifact_paths(job, validated).items():
                digest = sha256_bytes(record_input(path, inputs))
                if digest != validated["artifact_sha256"][key]:
                    raise RuntimeError(f"Random artifact hash mismatch: {path}")
            observations.append({
                "task": job["task"], "group": "uniform_random",
                "arm": "uniform", "seed": job["seed"],
                "common_proposal_cap": job["mechanics_budget"],
                "event_by_cap": bool(validated["success"]),
                "first_hit_proposals": validated[
                    "first_success_mechanical_steps"
                ],
                "assigned_proposals_at_cap": (
                    validated["first_success_mechanical_steps"]
                    if validated["success"] else job["mechanics_budget"]
                ),
                "observed_proposal_exposure": validated["mechanical_steps"],
                "usable_at_common_cap": True,
                "censor_reason": (
                    None if validated["success"] else "mechanics_budget"
                ),
                "first_hit_before_or_at_first_update": None,
                "formal_job_fingerprint": fingerprint,
            })
    pending = report.get("pending_unlaunched")
    if not isinstance(pending, list):
        raise RuntimeError("Random pending list is invalid")
    pending_matrix = set()
    for row in pending:
        identity = (
            row.get("task"), exact_int(row.get("seed"), "pending seed")
        )
        if identity not in matrix or identity in pending_matrix:
            raise RuntimeError("Random pending identity is invalid or duplicated")
        pending_matrix.add(identity)
    if seen_matrix & pending_matrix or seen_matrix | pending_matrix != matrix:
        raise RuntimeError("Random result/pending matrix is not an exact partition")
    if len(observations) != admissible_count:
        raise RuntimeError("Random admissible observation count mismatch")
    for name, digest in manifest["source_sha256"].items():
        path = SCRIPT_DIR / name
        if sha256_bytes(record_input(path, inputs)) != digest:
            raise RuntimeError(f"Random source changed: {name}")
    for task in TASKS:
        task_spec = random_runner.baseline.base_runner.TASKS[task]
        model_dir = SCRIPT_DIR / task_spec.model_directory
        expected = random_runner.baseline.model_hashes(task)
        for short_name, digest in expected.items():
            filename = {
                "node": "node_initial.dat", "member": "member.dat", "fix": "fix.dat"
            }[short_name]
            if sha256_bytes(record_input(model_dir / filename, inputs)) != digest:
                raise RuntimeError(f"Random model changed: {task}/{filename}")
    return {
        "manifest": manifest, "binding": binding, "runner_report": report,
        "admissible_count": admissible_count,
    }, observations


def learning_observations(c10_payload: dict) -> list[dict]:
    curves = c10_payload["selected_episode_curves"]
    jobs = {
        (row["task"], row["arm"], row["seed"]): row
        for row in c10_payload["jobs"]
    }
    observations = []
    for task in TASKS:
        ever_arm = c10_payload["ever_selected_arms"].get(task)
        final_arm = c10_payload["final_runner_selections"].get(task)
        if ever_arm is None:
            continue
        for seed in LEARNING_SEEDS:
            rows = sorted(
                (
                    row for row in curves
                    if row["task"] == task and row["arm"] == ever_arm
                    and row["seed"] == seed
                ),
                key=lambda row: row["episode"],
            )
            if not rows:
                raise RuntimeError(f"Missing selected learning curve: {task}/s{seed}")
            cap = CAPS[task]
            hits = [
                row for row in rows
                if row["strict"] == 1
                and row["cumulative_proposals"] <= cap
            ]
            first = hits[0] if hits else None
            exposure = exact_int(
                rows[-1]["cumulative_proposals"], "learning proposal exposure", 1
            )
            usable = first is not None or exposure >= cap
            job = jobs[(task, ever_arm, seed)]
            if first is not None:
                first_proposals = exact_int(
                    first["cumulative_proposals"], "learning first-hit proposals", 1
                )
                if job.get("first_hit_proposals") != first_proposals:
                    raise RuntimeError(
                        f"Learning first-hit mismatch: {task}/{ever_arm}/s{seed}"
                    )
                assigned = first_proposals
                censor = None
                preupdate = bool(job["first_hit_before_or_at_first_update"])
            elif usable:
                first_proposals, assigned = None, cap
                censor, preupdate = "no_strict_hit_by_common_cap", None
            else:
                first_proposals, assigned = None, None
                censor, preupdate = "insufficient_learning_exposure_at_common_cap", None
            observations.append({
                "task": task, "group": "c10_selected_learning",
                "arm": ever_arm, "seed": seed,
                "common_proposal_cap": cap,
                "event_by_cap": first is not None,
                "first_hit_proposals": first_proposals,
                "assigned_proposals_at_cap": assigned,
                "observed_proposal_exposure": exposure,
                "usable_at_common_cap": usable,
                "censor_reason": censor,
                "first_hit_before_or_at_first_update": preupdate,
                "final_selection_retained": final_arm == ever_arm,
                "formal_endpoint": job["formal_endpoint"],
                "formal_job_fingerprint": job["job_fingerprint"],
            })
    return observations


def comparison_rows(observations: list[dict]) -> list[dict]:
    result = []
    for task_index, task in enumerate(TASKS):
        learning_all = [row for row in observations if row["task"] == task and row["group"] == "c10_selected_learning"]
        random_all = [row for row in observations if row["task"] == task and row["group"] == "uniform_random"]
        learning = [row for row in learning_all if row["usable_at_common_cap"]]
        control = [row for row in random_all if row["usable_at_common_cap"]]
        retained = bool(learning_all) and all(
            row.get("final_selection_retained") is True for row in learning_all
        )
        complete = (
            len(learning) == len(LEARNING_SEEDS)
            and len(control) == len(RANDOM_SEEDS)
            and retained
        )
        entry = {
            "task": task, "common_proposal_cap": CAPS[task],
            "learning_arm": learning_all[0]["arm"] if learning_all else None,
            "learning_n_usable": len(learning),
            "learning_n_expected": len(LEARNING_SEEDS),
            "random_n_usable": len(control),
            "random_n_expected": len(RANDOM_SEEDS),
            "final_selection_retained": retained,
            "primary_comparison_available": complete,
            "availability_reason": (
                "complete_exact_common_cap_comparison" if complete
                else "incomplete_random_or_learning_exposure_or_selection_revoked"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        }
        if learning:
            learning_costs = [row["assigned_proposals_at_cap"] for row in learning]
            learning_events = [int(row["event_by_cap"]) for row in learning]
            entry.update({
                f"learning_{key}": value for key, value in bootstrap_group(
                    learning_costs, learning_events, 10_000 + task_index
                ).items()
            })
            entry["learning_events"] = sum(learning_events)
            hits = [row["first_hit_proposals"] for row in learning if row["event_by_cap"]]
            entry["learning_successful_hit_median_descriptive"] = (
                statistics.median(hits) if hits else None
            )
            entry["learning_preupdate_events"] = sum(
                row["first_hit_before_or_at_first_update"] is True for row in learning
            )
            entry["learning_postupdate_events"] = sum(
                row["event_by_cap"] and row["first_hit_before_or_at_first_update"] is False
                for row in learning
            )
        if control:
            random_costs = [row["assigned_proposals_at_cap"] for row in control]
            random_events = [int(row["event_by_cap"]) for row in control]
            entry.update({
                f"random_{key}": value for key, value in bootstrap_group(
                    random_costs, random_events, 20_000 + task_index
                ).items()
            })
            entry["random_events"] = sum(random_events)
            hits = [row["first_hit_proposals"] for row in control if row["event_by_cap"]]
            entry["random_successful_hit_median_descriptive"] = (
                statistics.median(hits) if hits else None
            )
        if complete:
            entry.update(bootstrap_speedup(
                [row["assigned_proposals_at_cap"] for row in learning],
                [row["assigned_proposals_at_cap"] for row in control],
                30_000 + task_index,
            ))
        result.append(entry)
    return result


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_plot(observations: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    figure.get_layout_engine().set(h_pad=0.12, hspace=0.08)
    colours = {
        "c10_selected_learning": "#2563eb",
        "uniform_random": "#d97706",
    }
    offsets = {
        "c10_selected_learning": 0.12,
        "uniform_random": -0.12,
    }
    for axis, task in zip(axes, TASKS):
        rows = [row for row in observations if row["task"] == task]
        for group in ("c10_selected_learning", "uniform_random"):
            group_rows = sorted(
                (row for row in rows if row["group"] == group),
                key=lambda row: row["seed"],
            )
            for index, row in enumerate(group_rows):
                if row["usable_at_common_cap"]:
                    x = row["assigned_proposals_at_cap"]
                    marker = "o" if row["event_by_cap"] else ">"
                else:
                    x, marker = max(1, row["observed_proposal_exposure"]), "x"
                jitter = (index - (len(group_rows) - 1) / 2.0) * 0.018
                y = (1 if group == "c10_selected_learning" else 0) + offsets[group] + jitter
                axis.scatter(
                    x, y, marker=marker, s=38, color=colours[group],
                    alpha=0.86, linewidths=0.8,
                )
                axis.annotate(
                    f"s{row['seed']}", (x, y), xytext=(3, 2),
                    textcoords="offset points", fontsize=5.5,
                    color=colours[group], alpha=0.82,
                )
        axis.axvline(CAPS[task], color="#6b7280", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlim(1, CAPS[task] * 1.35)
        axis.set_ylim(-0.4, 1.4)
        axis.set_yticks([0, 1], ["Uniform random", "C10 selected"])
        axis.set_title(task)
        axis.set_xlabel("Raw action proposals to first strict hit")
        axis.grid(True, axis="x", which="both", alpha=0.2)
    axes[0].set_ylabel("Method (all seeds shown)")
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#111827",
               label="strict-hit event"),
        Line2D([], [], marker=">", linestyle="none", color="#111827",
               label="right-censored at common cap"),
        Line2D([], [], marker="x", linestyle="none", color="#111827",
               label="insufficient exposure"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("First strict discovery on the common proposal clock")
    figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)


def visual_qa(path: Path) -> dict:
    from PIL import Image, ImageChops, ImageStat
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        standard_deviation = max(ImageStat.Stat(rgb).stddev)
        difference = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, (255, 255, 255))
        ).convert("L")
        mask = difference.point(lambda value: 255 if value > 10 else 0)
        bounding_box = mask.getbbox()
    if bounding_box is None:
        raise RuntimeError("First-hit plot is blank")
    top_margin_px = bounding_box[1]
    if path.stat().st_size < 5_000 or width < 900 or height < 300 \
            or standard_deviation < 1.0 or top_margin_px < 20:
        raise RuntimeError("First-hit plot failed visual QA")
    return {
        "file": path.name, "bytes": path.stat().st_size,
        "width": width, "height": height,
        "top_content_margin_px": top_margin_px,
        "maximum_channel_standard_deviation": standard_deviation,
        "pass": True,
    }


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def japanese_report(payload: dict) -> str:
    lines = [
        "# C10 学習と exact uniform random の初回発見比較",
        "",
        "主評価は strict compact success の初回発見までに提案した raw action 数です。"
        "自己交差で reject された提案も1 proposalとして数えます。報酬値は比較に使いません。",
        "",
        "| task | 学習arm | cap | 学習 hit/n | random hit/n | 学習 RMST | random RMST | random/学習 | 95% bootstrap CI | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["comparisons"]:
        speed = row.get("random_over_learning_rmst")
        interval = (
            f"[{fmt(row.get('random_over_learning_rmst_ci_low'))}, "
            f"{fmt(row.get('random_over_learning_rmst_ci_high'))}]"
            if speed is not None else "—"
        )
        lines.append(
            f"| {row['task']} | {row.get('learning_arm') or '—'} | "
            f"{row['common_proposal_cap']:,} | "
            f"{row.get('learning_events', 0)}/{row['learning_n_usable']} | "
            f"{row.get('random_events', 0)}/{row['random_n_usable']} | "
            f"{fmt(row.get('learning_rmst'), 1)} | "
            f"{fmt(row.get('random_rmst'), 1)} | {fmt(speed)} | {interval} | "
            f"{'primary available' if row['primary_comparison_available'] else '不完全・主張不可'} |"
        )
    lines.extend([
        "",
        "`random/学習 RMST > 1` は、共通 cap において学習の初回発見が速い方向です。"
        "成功例だけの中央値ではなく、未発見 seed を cap に割り当てた RMST を主指標にしています。",
        "",
        "## 学習前後の区別",
        "",
        "| task | 初回更新以前のhit | 更新後のhit | final selection retained |",
        "|---|---:|---:|---|",
    ])
    for row in payload["comparisons"]:
        lines.append(
            f"| {row['task']} | {row.get('learning_preupdate_events', 0)} | "
            f"{row.get('learning_postupdate_events', 0)} | "
            f"{'yes' if row['final_selection_retained'] else 'no'} |"
        )
    lines.extend([
        "",
        "## 不確実性と解釈上の制約",
        "",
        f"- CI は seed を単位とする percentile bootstrap "
        f"({BOOTSTRAP_REPLICATES:,} resamples, 95%) です。学習 n=3、random n=10 の"
        "小標本なので、区間は離散的かつ不安定であり、強い一般化には使えません。",
        "- 学習3 seedすべてが共通 cap まで観測可能、random 10 seedすべてが rc=0・complete、"
        "かつ選択armが最終的に保持された場合だけ primary comparison available とします。",
        "- 初回更新以前の hit は初期化 policy の寄与を含むため、学習で形成された探索圧の証拠とは"
        "分けて表示します。",
        "- action law は一様 random、mechanics/H/幅/strict判定/crossing transition は C10 と一致します。"
        "optimizer と報酬は random の行動を変えないため、報酬値そのものは比較対象外です。",
        "- C10R1 は supervisor race で中止された診断 run であり、本表には一切混入していません。",
        "",
    ])
    return "\n".join(lines)


def active_random_pids(manifest: dict) -> list[dict]:
    _, workers, unknown = random_runner.active_trainer_count(manifest)
    if unknown:
        raise RuntimeError(f"Uninspectable Python processes: {unknown}")
    active = [row for row in workers if row["kind"] == "random"]
    active.extend({"pid": pid, "kind": "supervisor"}
                  for pid in random_runner.duplicate_supervisor())
    return active


def wait_until_ready(timeout_seconds: float, poll_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        random_ready = (RANDOM_ROOT / "runner_report.json").is_file()
        c10_ready = (C10_ROOT / "runner_report.json").is_file()
        if random_ready and c10_ready:
            manifest = json.loads(RANDOM_MANIFEST.read_text(encoding="utf-8"))
            if not active_random_pids(manifest) and not c10final.active_c10_pids(C10_ROOT):
                return
        if time.monotonic() >= deadline:
            raise RuntimeError("C10/random campaigns are not finalizable before timeout")
        time.sleep(max(1.0, min(30.0, poll_seconds)))


def analyze(
    approved_finalizer_sha256: str, approved_test_sha256: str,
) -> dict:
    inputs: dict[str, str] = {}
    self_approval = validate_self_approval(
        approved_finalizer_sha256, approved_test_sha256, inputs
    )
    external = validate_external_source_authority(inputs)
    if c10final.active_c10_pids(C10_ROOT):
        raise RuntimeError("C10 trainer/supervisor is still active")
    random_manifest_preview = json.loads(
        record_input(RANDOM_MANIFEST, inputs).decode("utf-8")
    )
    if active_random_pids(random_manifest_preview):
        raise RuntimeError("Random trainer/supervisor is still active")
    c10_payload = c10final.analyze_campaign(
        C10_ROOT,
        c10final.MANIFEST_DIRECTORY / "codex10_nonswap_7f1a9303048b.json",
        active_check=True,
        approved_finalizer_sha256=EXPECTED_C10_FINALIZER_SHA256,
        approved_test_sha256=EXPECTED_C10_FINALIZER_TEST_SHA256,
    )
    for path, digest in c10_payload["input_sha256"].items():
        current = sha256_bytes(record_input(Path(path), inputs))
        if current != digest:
            raise RuntimeError(f"C10 finalized input hash mismatch: {path}")
    random_control, random_rows = load_random_campaign(inputs)
    learning_rows = learning_observations(c10_payload)
    observations = learning_rows + random_rows
    comparisons = comparison_rows(observations)
    verify_inputs(inputs)
    return {
        "schema_version": 1,
        "analysis": "c10ns1_selected_vs_c10r2_exact_uniform_first_discovery",
        "created_utc": utc_now(),
        "estimand": (
            "first strict compact success on cumulative raw action proposals; "
            "right-censor at exact task common cap"
        ),
        "reward_identity_used_for_outcome": False,
        "rejected_crossing_proposals_counted": True,
        "common_caps": CAPS,
        "learning_seeds": list(LEARNING_SEEDS),
        "random_seeds": list(RANDOM_SEEDS),
        "c10_manifest_fingerprint": c10_payload["manifest_fingerprint"],
        "random_manifest_fingerprint": EXPECTED_RANDOM_MANIFEST_FINGERPRINT,
        "random_runner_status": random_control["runner_report"]["status"],
        "random_admissible_jobs": random_control["admissible_count"],
        "c10_ever_selected_arms": c10_payload["ever_selected_arms"],
        "c10_final_selections": c10_payload["final_runner_selections"],
        "comparisons": comparisons,
        "observations": observations,
        "bootstrap_note": (
            "Percentile bootstrap over seeds; learning n=3 and random n=10 "
            "are small, so intervals are discrete and exploratory."
        ),
        "c10r1_exclusion": (
            "C10R1 aborted after a supervisor process-enumeration race and is "
            "not consumed by this analysis."
        ),
        "approved_source_sha256": {**external, **self_approval},
        "input_sha256": dict(sorted(inputs.items())),
    }


def write_bundle(payload: dict, output: Path = OUTPUT) -> dict:
    output = Path(output).resolve()
    if output != OUTPUT or output.exists():
        raise RuntimeError(f"Output must be the fresh fixed path: {OUTPUT}")
    output.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="tmp-c10-random-", dir=output.parent))
    try:
        write_json(build / "final_report.json", payload)
        write_csv(build / "first_hit_observations.csv", payload["observations"])
        write_csv(build / "comparison_summary.csv", payload["comparisons"])
        (build / "report_ja.md").write_text(
            japanese_report(payload), encoding="utf-8", newline="\n"
        )
        render_plot(payload["observations"], build / "first_hit_proposal_clock.png")
        write_json(build / "visual_qa.json", [
            visual_qa(build / "first_hit_proposal_clock.png")
        ])
        write_json(build / "input_hashes.json", payload["input_sha256"])
        hashes = {
            path.name: sha256_bytes(stable_bytes(path))
            for path in sorted(build.iterdir()) if path.is_file()
        }
        write_json(build / "output_hashes.json", hashes)
        verify_inputs(payload["input_sha256"])
        manifest = json.loads(stable_bytes(RANDOM_MANIFEST).decode("utf-8"))
        if c10final.active_c10_pids(C10_ROOT) or active_random_pids(manifest):
            raise RuntimeError("Campaign PID appeared immediately before publish")
        os.replace(build, output)
    finally:
        if build.exists():
            shutil.rmtree(build)
    return {
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
        "output_hashes_sha256": sha256_bytes(
            stable_bytes(output / "output_hashes.json")
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-finalizer-sha256", required=True)
    parser.add_argument("--approved-test-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=8 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    options = parser.parse_args(argv)
    wait_until_ready(options.wait_seconds, options.poll_seconds)
    payload = analyze(
        options.approved_finalizer_sha256, options.approved_test_sha256
    )
    result = write_bundle(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
