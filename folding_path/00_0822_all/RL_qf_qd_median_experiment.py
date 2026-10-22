"""Reproduce the median-discovery seed used by the qF--qD analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from RL_adaptive_experiment import experiment_config as five_config
from RL_main import run_rl
from RL_torus_spectral_screen import experiment_config as torus_config


EXPERIMENT_NAME = "20260902_qf_qd_median_seed_v1"
BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY / "result" / "experiments" / EXPERIMENT_NAME
)
MAX_EPISODES = 10_000

SCREEN_RESULTS = {
    "five_d2": {
        9101: 249_158,
        9102: 192_738,
        9103: 125_684,
        9104: 276_524,
        9105: 321_134,
    },
    "five_d3": {
        9101: 75_579,
        9102: 3_501,
        9103: 3_592,
        9104: 3_954,
        9105: 5_539,
    },
    "torus_d2": {
        8101: 526_397,
        8102: 313_798,
        8103: 289_199,
        8104: 2_000_000,
        8105: 251_788,
    },
}

# Median cumulative mechanical steps among the five screening seeds.  The
# censored torus run is represented by its 2,000,000-step lower bound, which
# ranks it after every seed that discovered a solution.
SEEDS = {
    task_name: sorted(results, key=results.get)[len(results) // 2]
    for task_name, results in SCREEN_RESULTS.items()
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


def source_hashes() -> dict[str, str]:
    names = (
        "RL_qf_qd_median_experiment.py",
        "plot_qf_qd_terminal.py",
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
        name: hashlib.sha256(
            (BASE_DIRECTORY / name).read_bytes()
        ).hexdigest()
        for name in names
    }


def write_manifest() -> Path:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "tasks": list(SEEDS),
        "max_episodes": MAX_EPISODES,
        "seed_selection": (
            "median cumulative mechanical steps to first strict success "
            "among five screening seeds; a censored seed is ranked after "
            "all discovered seeds"
        ),
        "seeds": SEEDS,
        "screen_mechanical_steps": SCREEN_RESULTS,
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


def run_task(task_name: str) -> dict:
    return run_rl(experiment_config(task_name))


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
        print(write_manifest())
        for task_name in SEEDS:
            run_task(task_name)
    elif args.command == "run":
        write_manifest()
        run_task(args.task)


if __name__ == "__main__":
    main()
