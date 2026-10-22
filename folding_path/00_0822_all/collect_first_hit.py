"""Collect first strict-success trajectories without altering source runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from env import OrigamiEnv
from main import task_config


BASE_DIRECTORY = Path(__file__).resolve().parent
RESULT_DIRECTORY = BASE_DIRECTORY / "result"
OUTPUT_DIRECTORY = RESULT_DIRECTORY / "first_hit"
STEP_PATTERN = re.compile(r"step(\d+)\.dat")
FLATNESS_RATIO_TOLERANCE = 1.0e-3
SPREAD_ERROR_TOLERANCE = 0.05
STRICT_VALIDATION_SLACK = 1.0e-9
TASK_DIRECTORY = {
    "five_d2": "five_d2",
    "five_d3": "five_d3",
    "torus_d2": "torus",
}


@dataclass(frozen=True)
class Source:
    path: str
    kind: str
    note: str = ""


SOURCES: dict[tuple[str, str, str, int], Source] = {
    # Random search writes first strict success explicitly.
    ("five_d2", "random", "", 6102): Source(
        "result/five_d2/random/seed6102/first_strict_success", "existing"
    ),
    ("five_d2", "random", "", 6103): Source(
        "result/five_d2/random/seed6103/first_strict_success", "existing"
    ),
    ("five_d3", "random", "", 6101): Source(
        "result/five_d3/random/seed6101/first_strict_success", "existing"
    ),
    ("five_d3", "random", "", 6102): Source(
        "result/five_d3/random/seed6102/first_strict_success", "existing"
    ),
    ("five_d3", "random", "", 6103): Source(
        "result/five_d3/random/seed6103/first_strict_success", "existing"
    ),
    ("five_d3", "random", "", 6104): Source(
        "result/five_d3/random/seed6104/first_strict_success", "existing"
    ),
    ("five_d3", "random", "", 6105): Source(
        "result/five_d3/random/seed6105/first_strict_success", "existing"
    ),
    # Five-bar fixed-horizon/adaptive RL screen and exact reproductions.
    ("five_d2", "rl", "0.05", 9101): Source(
        "result/experiments/20260902_qf_qd_median_seed_v1/"
        "runs/five_d2/seed9101/best_episode00624",
        "existing",
    ),
    ("five_d2", "rl", "0.05", 9102): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "five/five_d2/seed9102/best_episode00482",
        "replay",
    ),
    ("five_d2", "rl", "0.05", 9103): Source(
        "result/experiments/20260901_first_discovery_concentration_v1/"
        "runs/five_d2/seed9103/best_episode00315",
        "existing",
    ),
    ("five_d2", "rl", "0.05", 9104): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "five/five_d2/seed9104/best_episode00692",
        "replay",
    ),
    ("five_d2", "rl", "0.05", 9105): Source(
        "result/experiments/20260902_slowest_discovery_evidence_v1/"
        "runs/five_d2/seed9105/best_episode00803",
        "existing",
    ),
    ("five_d3", "rl", "0.05", 9101): Source(
        "result/experiments/20260902_slowest_discovery_evidence_v1/"
        "runs/five_d3/seed9101/best_episode00189",
        "existing",
    ),
    ("five_d3", "rl", "0.05", 9102): Source(
        "result/five_d3/rl/best_episode00009",
        "legacy_equivalent",
        "Matched to the screen row by seed, episode, step count, and terminal geometry.",
    ),
    ("five_d3", "rl", "0.05", 9103): Source(
        "result/experiments/20260901_first_discovery_concentration_v1/"
        "runs/five_d3/seed9103/best_episode00009",
        "existing",
    ),
    ("five_d3", "rl", "0.05", 9104): Source(
        "result/experiments/20260902_qf_qd_median_seed_v1/"
        "runs/five_d3/seed9104/best_episode00010",
        "existing",
    ),
    ("five_d3", "rl", "0.05", 9105): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "five/five_d3/seed9105/best_episode00014",
        "replay",
    ),
    # Torus alpha=0.05 stability reruns exactly reproduce the screen first hits.
    ("torus_d2", "rl", "0.05", 8101): Source(
        "result/experiments/20260831_reward_ablation_v1/"
        "torus_alpha_0p05_stability/seed8101/best_episode02632",
        "existing",
    ),
    ("torus_d2", "rl", "0.05", 8102): Source(
        "result/experiments/20260831_reward_ablation_v1/"
        "torus_alpha_0p05_stability/seed8102/best_episode01569",
        "existing",
    ),
    ("torus_d2", "rl", "0.05", 8103): Source(
        "result/experiments/20260831_reward_ablation_v1/"
        "torus_alpha_0p05_stability/seed8103/best_episode01446",
        "existing",
    ),
    ("torus_d2", "rl", "0.05", 8105): Source(
        "result/experiments/20260831_reward_ablation_v1/"
        "torus_alpha_0p05_stability/seed8105/best_episode01259",
        "existing",
    ),
    # Torus alpha=0.25 screen reproductions.
    ("torus_d2", "rl", "0.25", 8101): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "torus/alpha_0p25/seed8101/best_episode02619",
        "replay",
    ),
    ("torus_d2", "rl", "0.25", 8102): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "torus/alpha_0p25/seed8102/best_episode03105",
        "replay",
    ),
    ("torus_d2", "rl", "0.25", 8103): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "torus/alpha_0p25/seed8103/best_episode00853",
        "replay",
    ),
    ("torus_d2", "rl", "0.25", 8104): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "torus/alpha_0p25/seed8104/best_episode01322",
        "replay",
    ),
    # Torus alpha=0.5. Seed 8102 is identical through discovery because every
    # preceding episode reaches the common 200-step horizon.
    ("torus_d2", "rl", "0.5", 8101): Source(
        "result/experiments/20260903_first_hit_replay_v1/"
        "torus/alpha_0p5/seed8101/best_episode01659",
        "replay",
    ),
    ("torus_d2", "rl", "0.5", 8102): Source(
        "result/torus_d2/rl/best_episode01126",
        "legacy_equivalent",
        "Legacy run, but discovery prefix is equivalent to the fixed-horizon screen: "
        "all preceding episodes have the full 200-step horizon and E/U/t/M match.",
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def integer(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def alpha_directory(alpha: str) -> str:
    return "alpha_" + alpha.replace(".", "p")


def trajectory_files(directory: Path) -> list[tuple[int, Path]]:
    numbered: list[tuple[int, Path]] = []
    for path in directory.glob("step*.dat"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected trajectory filename: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    if not numbered:
        raise ValueError(f"no step DAT files: {directory}")
    expected = list(range(numbered[-1][0] + 1))
    observed = [number for number, _ in numbered]
    if observed != expected:
        raise ValueError(f"non-contiguous trajectory: {directory}")
    if any(path.stat().st_size == 0 for _, path in numbered):
        raise ValueError(f"empty step DAT file: {directory}")
    return numbered


def trajectory_hash(files: list[tuple[int, Path]]) -> str:
    digest = hashlib.sha256()
    for number, path in files:
        digest.update(f"step{number}.dat\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_geometry(
    record: dict[str, object],
    files: list[tuple[int, Path]],
) -> dict[str, float]:
    environment = OrigamiEnv(task_config(str(record["task"]), 1))
    initial = np.loadtxt(files[0][1], dtype=float)
    terminal = np.loadtxt(files[-1][1], dtype=float)
    if initial.shape != environment.node0.shape or not np.allclose(
        initial, environment.node0, rtol=0.0, atol=1.0e-8
    ):
        raise ValueError(f"initial state mismatch: {files[0][1]}")
    metrics = environment.compactness_metrics(terminal)
    if (
        metrics["flatness_ratio"]
        > FLATNESS_RATIO_TOLERANCE + STRICT_VALIDATION_SLACK
        or metrics["spread_error"]
        > SPREAD_ERROR_TOLERANCE + STRICT_VALIDATION_SLACK
    ):
        raise ValueError(
            f"terminal state is not strict success: {files[-1][1]} "
            f"F={metrics['flatness_ratio']} dD={metrics['spread_error']}"
        )
    return {
        "terminal_flatness_ratio_recomputed": float(
            metrics["flatness_ratio"]
        ),
        "terminal_spread_recomputed": float(metrics["spread"]),
        "terminal_spread_error_recomputed": float(metrics["spread_error"]),
    }


def records() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    random_path = RESULT_DIRECTORY / "random_summary_all.csv"
    for row in read_rows(random_path):
        task = row["task"]
        if task not in TASK_DIRECTORY:
            continue
        output.append({
            "task": task,
            "method": "random",
            "spectral_alpha": "",
            "seed": int(row["seed"]),
            "success": int(row["success"]),
            "censored": 1 - int(row["success"]),
            "episodes_completed": (
                int(row["first_strict_success_episode"])
                if int(row["success"])
                else 10_000
            ),
            "first_strict_success_episode": integer(
                row["first_strict_success_episode"]
            ),
            "first_strict_success_episode_steps": None,
            "first_strict_success_mechanical_steps": (
                int(row["mechanical_steps"]) if int(row["success"]) else None
            ),
            "mechanical_steps_at_stop": int(row["mechanical_steps"]),
            "optimizer_updates_before_first_success": None,
            "optimizer_updates_at_stop": None,
            "max_episodes": 10_000,
            "max_steps_per_episode": int(row["max_steps_per_episode"]),
            "max_step_width": float(row["max_step_width"]),
            "target_spread": float(row["target_spread"]),
            "source_summary": random_path.relative_to(BASE_DIRECTORY).as_posix(),
            "source_result": row,
        })

    five_path = (
        RESULT_DIRECTORY
        / "experiments/20260830_fixednorm_adaptive_v1/"
        "screening/first_success_summary.csv"
    )
    for row in read_rows(five_path):
        output.append({
            "task": row["task"],
            "method": "rl",
            "spectral_alpha": "0.05",
            "seed": int(row["seed"]),
            "success": int(row["success"]),
            "censored": 1 - int(row["success"]),
            "episodes_completed": int(row["episodes_completed"]),
            "first_strict_success_episode": integer(
                row["first_strict_success_episode"]
            ),
            "first_strict_success_episode_steps": integer(
                row["first_strict_success_episode_steps"]
            ),
            "first_strict_success_mechanical_steps": integer(
                row["first_strict_success_mechanical_steps"]
            ),
            "mechanical_steps_at_stop": int(row["cumulative_mechanical_steps"]),
            "optimizer_updates_before_first_success": integer(
                row["optimizer_updates_before_first_success"]
            ),
            "optimizer_updates_at_stop": integer(
                row["optimizer_updates_before_first_success"]
            ),
            "max_episodes": 10_000,
            "max_steps_per_episode": int(row["actor_reference_horizon"]),
            "max_step_width": 0.1,
            "target_spread": 2.0 if row["task"] == "five_d2" else 3.0,
            "source_summary": five_path.relative_to(BASE_DIRECTORY).as_posix(),
            "source_result": row,
        })

    torus_paths = (
        RESULT_DIRECTORY
        / "experiments/20260830_torus_spectral_screen_v1/"
        "screening/first_success_summary.csv",
    )
    for torus_path in torus_paths:
        for row in read_rows(torus_path):
            if row["task"] != "torus_d2":
                continue
            alpha = str(float(row["spectral_alpha"])).rstrip("0").rstrip(".")
            if alpha == "":
                alpha = "0"
            output.append({
                "task": "torus_d2",
                "method": "rl",
                "spectral_alpha": alpha,
                "seed": int(row["seed"]),
                "success": int(row["success"]),
                "censored": int(row["censored_at_max_episodes"]),
                "episodes_completed": int(row["episodes_completed"]),
                "first_strict_success_episode": integer(
                    row["first_strict_success_episode"]
                ),
                "first_strict_success_episode_steps": integer(
                    row["first_strict_success_episode_steps"]
                ),
                "first_strict_success_mechanical_steps": integer(
                    row["first_strict_success_mechanical_steps"]
                ),
                "mechanical_steps_at_stop": int(row["mechanical_steps_at_stop"]),
                "optimizer_updates_before_first_success": integer(
                    row["optimizer_updates_before_first_success"]
                ),
                "optimizer_updates_at_stop": int(
                    row["optimizer_updates_at_stop"]
                ),
                "max_episodes": int(row["max_episodes"]),
                "max_steps_per_episode": int(row["max_steps_per_episode"]),
                "max_step_width": float(row["max_step_width"]),
                "target_spread": 2.0,
                "source_summary": torus_path.relative_to(BASE_DIRECTORY).as_posix(),
                "source_result": row,
            })

    return sorted(output, key=lambda row: (
        str(row["task"]),
        str(row["method"]),
        str(row["spectral_alpha"]),
        int(row["seed"]),
    ))


def validate_record_set(rows: list[dict[str, object]]) -> None:
    expected = {
        *((task, "random", "", seed)
          for task in TASK_DIRECTORY
          for seed in range(6101, 6106)),
        *((task, "rl", "0.05", seed)
          for task in ("five_d2", "five_d3")
          for seed in range(9101, 9106)),
        *(("torus_d2", "rl", alpha, seed)
          for alpha in ("0.05", "0.25", "0.5")
          for seed in range(8101, 8106)),
    }
    keys = [
        (
            str(row["task"]),
            str(row["method"]),
            str(row["spectral_alpha"]),
            int(row["seed"]),
        )
        for row in rows
    ]
    observed = set(keys)
    if len(keys) != len(observed):
        raise ValueError("duplicate condition rows in source summaries")
    if observed != expected:
        raise ValueError(
            f"condition set mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    successful = {key for key, row in zip(keys, rows) if int(row["success"])}
    if set(SOURCES) != successful:
        raise ValueError(
            "source mapping mismatch: "
            f"missing={sorted(successful - set(SOURCES))}, "
            f"extra={sorted(set(SOURCES) - successful)}"
        )


def destination_for(record: dict[str, object]) -> Path:
    directory = (
        OUTPUT_DIRECTORY
        / TASK_DIRECTORY[str(record["task"])]
        / str(record["method"])
    )
    alpha = str(record["spectral_alpha"])
    if alpha and record["task"] == "torus_d2":
        directory /= alpha_directory(alpha)
    return directory / f"seed{record['seed']}"


def collect(allow_missing: bool) -> list[dict[str, object]]:
    rows = records()
    validate_record_set(rows)
    prepared: dict[tuple[str, str, str, int], dict[str, object]] = {}
    missing: list[str] = []

    # Validate every source and any existing destination before writing.
    for record in rows:
        key = (
            str(record["task"]),
            str(record["method"]),
            str(record["spectral_alpha"]),
            int(record["seed"]),
        )
        destination = destination_for(record)
        source = SOURCES.get(key)
        status = "censored_no_first_hit"
        source_files: list[tuple[int, Path]] = []
        last_step: int | None = None
        digest = ""
        source_relative = ""
        source_kind = ""
        source_note = ""
        geometry: dict[str, float | None] = {
            "terminal_flatness_ratio_recomputed": None,
            "terminal_spread_recomputed": None,
            "terminal_spread_error_recomputed": None,
        }

        if int(record["success"]):
            if source is None:  # Also guarded by validate_record_set().
                raise ValueError(f"no source mapping for {key}")
            source_directory = BASE_DIRECTORY / source.path
            source_relative = source.path
            source_kind = source.kind
            source_note = source.note
            if not source_directory.is_dir():
                missing.append(str(source_directory))
                status = "missing_source"
            else:
                if record["method"] == "rl":
                    expected_episode = int(
                        record["first_strict_success_episode"]
                    )
                    expected_name = f"best_episode{expected_episode:05d}"
                    if source_directory.name != expected_name:
                        raise ValueError(
                            f"first-hit episode directory mismatch for {key}: "
                            f"{source_directory.name} != {expected_name}"
                        )
                source_files = trajectory_files(source_directory)
                last_step = source_files[-1][0]
                expected_steps = record["first_strict_success_episode_steps"]
                if expected_steps is not None and last_step != expected_steps:
                    raise ValueError(
                        f"step mismatch for {key}: {last_step} != {expected_steps}"
                    )
                record["first_strict_success_episode_steps"] = last_step
                digest = trajectory_hash(source_files)
                geometry = validate_geometry(record, source_files)
                status = "copied_" + source.kind

        existing_steps = list(destination.glob("step*.dat"))
        if existing_steps:
            if not source_files:
                raise ValueError(
                    f"stale trajectory in non-success destination: {destination}"
                )
            if trajectory_hash(trajectory_files(destination)) != digest:
                raise ValueError(f"destination collision: {destination}")
        prepared[key] = {
            "record": record,
            "destination": destination,
            "source_files": source_files,
            "status": status,
            "last_step": last_step,
            "digest": digest,
            "source_relative": source_relative,
            "source_kind": source_kind,
            "source_note": source_note,
            "geometry": geometry,
        }

    if missing and not allow_missing:
        raise FileNotFoundError("missing successful trajectories:\n" + "\n".join(missing))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_rows: list[dict[str, object]] = []
    for item in prepared.values():
        record = item["record"]
        destination = item["destination"]
        source_files = item["source_files"]
        destination.mkdir(parents=True, exist_ok=True)
        for _, source_path in source_files:
            destination_path = destination / source_path.name
            if not destination_path.exists():
                shutil.copy2(source_path, destination_path)
        dat_count = 0
        if source_files:
            destination_files = trajectory_files(destination)
            if trajectory_hash(destination_files) != item["digest"]:
                raise ValueError(f"copy verification failed: {destination}")
            dat_count = len(destination_files)

        source_summary_path = BASE_DIRECTORY / str(record["source_summary"])
        metadata = {
            **record,
            "status": item["status"],
            "source_trajectory": item["source_relative"],
            "source_kind": item["source_kind"],
            "source_note": item["source_note"],
            "source_summary_sha256": file_hash(source_summary_path),
            "destination": destination.relative_to(BASE_DIRECTORY).as_posix(),
            "trajectory_dat_count": dat_count,
            "trajectory_last_step": item["last_step"],
            "trajectory_sha256": item["digest"],
            "trajectory_hash_scheme": (
                "SHA-256 over numeric step order; for each file append UTF-8 "
                "canonical filename step{number}.dat, NUL byte, then exact file bytes"
            ),
            **item["geometry"],
            "flatness_ratio_tolerance": FLATNESS_RATIO_TOLERANCE,
            "spread_error_tolerance": SPREAD_ERROR_TOLERANCE,
            "strict_validation_slack": STRICT_VALIDATION_SLACK,
            "mechanical_step_definition": (
                "environment action attempt; rejected crossing no-op also consumes one"
            ),
            "crossing_policy": (
                "reject with no state change; consume one environment step; continue "
                "unless the horizon is reached, where timeout return is applied"
            ),
            "collected_at": generated_at,
        }
        metadata_path = destination / "metadata.json"
        temporary_metadata = destination / "metadata.json.tmp"
        with temporary_metadata.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        temporary_metadata.replace(metadata_path)

        manifest_rows.append({
            name: value
            for name, value in metadata.items()
            if name != "source_result"
        })

    fields = list(manifest_rows[0])
    temporary_manifest = OUTPUT_DIRECTORY / "manifest.csv.tmp"
    with temporary_manifest.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(OUTPUT_DIRECTORY / "manifest.csv")
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    result = collect(allow_missing=args.allow_missing)
    copied = sum(str(row["status"]).startswith("copied_") for row in result)
    censored = sum(row["status"] == "censored_no_first_hit" for row in result)
    missing = sum(row["status"] == "missing_source" for row in result)
    print(
        f"conditions={len(result)} copied={copied} "
        f"censored={censored} missing={missing}"
    )


if __name__ == "__main__":
    main()
