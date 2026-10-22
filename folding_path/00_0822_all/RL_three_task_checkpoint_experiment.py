"""Reproduce the three selected first-discovery runs with policy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from RL_adaptive_experiment import experiment_config as five_config
from RL_main import run_rl
from RL_torus_spectral_screen import experiment_config as torus_config


EXPERIMENT_NAME = "20260902_three_task_checkpoint_evidence_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
MAX_EPISODES = 10_000

SEEDS = {
    "five_d2": 9101,
    "five_d3": 9101,
    "torus_d2": 8102,
}

EXPECTED = {
    "five_d2": {
        "episode": 624,
        "episode_steps": 340,
        "mechanical_steps": 249_158,
        "updates": 77,
    },
    "five_d3": {
        "episode": 189,
        "episode_steps": 379,
        "mechanical_steps": 75_579,
        "updates": 23,
    },
    "torus_d2": {
        "episode": 1569,
        "episode_steps": 198,
        "mechanical_steps": 313_798,
        "updates": 196,
    },
}


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
        "save_policy_checkpoints": True,
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
        "RL_three_task_checkpoint_experiment.py",
        "evaluate_three_task_checkpoints.py",
        "plot_three_task_checkpoint_evaluation.py",
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
        "seed_selection": {
            "five_d2": (
                "median first-discovery speed among five successful seeds"
            ),
            "five_d3": "slowest first-discovery seed among five successes",
            "torus_d2": (
                "median rank among the five alpha=0.05 screening seeds, "
                "with censored seed 8104 ranked after all discoveries"
            ),
        },
        "torus_censored_seed": 8104,
        "expected_screen_results": EXPECTED,
        "resolved_configs": {
            task_name: serialisable(experiment_config(task_name))
            for task_name in SEEDS
        },
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
    subparsers.add_parser("run-all")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", choices=tuple(SEEDS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        print(write_manifest())
    elif args.command == "run-all":
        write_manifest()
        for task_name in SEEDS:
            run_task(task_name)
    elif args.command == "run":
        run_task(args.task)


if __name__ == "__main__":
    main()
