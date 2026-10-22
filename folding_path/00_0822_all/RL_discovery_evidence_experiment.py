"""Slowest-successful-seed runs used for first-discovery evidence plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from RL_adaptive_experiment import experiment_config as five_config
from RL_main import run_rl
from RL_torus_spectral_screen import experiment_config as torus_config


EXPERIMENT_NAME = "20260902_slowest_discovery_evidence_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
MAX_EPISODES = 10_000

# Slowest first discovery among the successful seeds in each five-seed screen.
# Torus seed 8104 was censored without success and is not eligible for a curve
# ending at first discovery.
SEEDS = {
    "five_d2": 9105,
    "five_d3": 9101,
    "torus_d2": 8101,
}

EXPECTED = {
    "five_d2": {
        "episode": 803,
        "episode_steps": 388,
        "mechanical_steps": 321_134,
        "updates": 100,
    },
    "five_d3": {
        "episode": 189,
        "episode_steps": 379,
        "mechanical_steps": 75_579,
        "updates": 23,
    },
    "torus_d2": {
        "episode": 2632,
        "episode_steps": 197,
        "mechanical_steps": 526_397,
        "updates": 328,
    },
}

TORUS_SOURCE_METRICS = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260831_reward_ablation_v1"
    / "torus_alpha_0p05_stability"
    / "seed8101"
    / "episode_metrics.csv"
)


def experiment_config(task_name: str) -> dict:
    seed = SEEDS[task_name]
    if task_name.startswith("five_"):
        config = five_config(task_name, MAX_EPISODES, seed)
    else:
        config = torus_config("0.05", seed, MAX_EPISODES)
    config.update({
        "save_outputs": True,
        "save_diagnostics": True,
        "save_concentration_diagnostics": False,
        "save_policy_checkpoints": task_name == "five_d2",
        "stop_on_first_strict_success": True,
        "print_progress": True,
        "print_result": False,
        "rl_output_directory": (
            EXPERIMENT_DIRECTORY / "runs" / task_name / f"seed{seed}"
        ),
    })
    return config


def serialisable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialisable(item) for item in value]
    if isinstance(value, dict):
        return {key: serialisable(item) for key, item in value.items()}
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    names = (
        "RL_discovery_evidence_experiment.py",
        "evaluate_five_d2_checkpoints.py",
        "plot_discovery_evidence.py",
        "RL_adaptive_experiment.py",
        "RL_torus_spectral_screen.py",
        "RL_main.py",
        "agent.py",
        "policy.py",
        "env.py",
        "mechanics.py",
        "main.py",
    )
    return {
        name: sha256(BASE_DIRECTORY / name)
        for name in names
        if (BASE_DIRECTORY / name).is_file()
    }


def write_manifest() -> Path:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "tasks": list(SEEDS),
        "seeds": SEEDS,
        "selection": (
            "slowest first strict discovery among successful seeds in each "
            "completed five-seed screen"
        ),
        "torus_censored_seed": 8104,
        "expected_screen_results": EXPECTED,
        "resolved_configs": {
            task_name: serialisable(experiment_config(task_name))
            for task_name in SEEDS
        },
        "torus_learning_curve_source": str(TORUS_SOURCE_METRICS),
        "torus_learning_curve_source_sha256": (
            sha256(TORUS_SOURCE_METRICS)
            if TORUS_SOURCE_METRICS.is_file() else None
        ),
        "source_sha256": source_hashes(),
    }
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = EXPERIMENT_DIRECTORY / "manifest.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return output


def verify_result(task_name: str, result: dict) -> None:
    expected = EXPECTED[task_name]
    actual = {
        "episode": result["first_strict_success_episode"],
        "episode_steps": result["first_strict_success_episode_steps"],
        "mechanical_steps": result["first_strict_success_mechanical_steps"],
        "updates": result["optimizer_updates_before_first_success"],
    }
    if actual != expected:
        raise RuntimeError(
            f"{task_name} did not reproduce screening: "
            f"expected {expected}, got {actual}"
        )


def run_task(task_name: str) -> dict:
    result = run_rl(experiment_config(task_name))
    verify_result(task_name, result)
    print(json.dumps({
        "task": task_name,
        "seed": SEEDS[task_name],
        **EXPECTED[task_name],
        "reproduced": True,
    }, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("run-needed")
    subparsers.add_parser("run-all")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", choices=tuple(SEEDS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        print(write_manifest())
    elif args.command == "run-needed":
        write_manifest()
        for task_name in ("five_d2", "five_d3"):
            run_task(task_name)
    elif args.command == "run-all":
        write_manifest()
        for task_name in SEEDS:
            run_task(task_name)
    elif args.command == "run":
        write_manifest()
        run_task(args.task)


if __name__ == "__main__":
    main()
