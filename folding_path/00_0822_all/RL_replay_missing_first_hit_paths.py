"""Replay the three five-bar screens whose first-hit paths were not saved."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from RL_adaptive_experiment import experiment_config
from RL_main import run_rl


BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260903_first_hit_replay_v1"
    / "five"
)

EXPECTED = {
    ("five_d2", 9102): {
        "episode": 482,
        "episode_steps": 399,
        "mechanical_steps": 192_738,
        "updates": 60,
    },
    ("five_d2", 9104): {
        "episode": 692,
        "episode_steps": 303,
        "mechanical_steps": 276_524,
        "updates": 86,
    },
    ("five_d3", 9105): {
        "episode": 14,
        "episode_steps": 339,
        "mechanical_steps": 5_539,
        "updates": 1,
    },
}


def output_directory(task_name: str, seed: int) -> Path:
    return EXPERIMENT_DIRECTORY / task_name / f"seed{seed}"


def validate_path(
    directory: Path,
    expected_episode: int,
    expected_steps: int,
) -> dict:
    success_directory = directory / f"best_episode{expected_episode:05d}"
    step_pattern = re.compile(r"step(\d+)\.dat")
    numbered_files = []
    for path in success_directory.glob("step*.dat"):
        match = step_pattern.fullmatch(path.name)
        if match is not None:
            numbered_files.append((int(match.group(1)), path))
    numbered_files.sort()
    actual_steps = [step for step, _ in numbered_files]
    wanted_steps = list(range(expected_steps + 1))
    if actual_steps != wanted_steps:
        missing = sorted(set(wanted_steps) - set(actual_steps))
        extra = sorted(set(actual_steps) - set(wanted_steps))
        raise RuntimeError(
            f"invalid DAT sequence in {success_directory}: "
            f"missing={missing}, extra={extra}"
        )
    if any(path.stat().st_size == 0 for _, path in numbered_files):
        raise RuntimeError(f"empty DAT file in {success_directory}")
    return {
        "success_path_directory": str(success_directory.resolve()),
        "step_file_count": len(numbered_files),
        "first_step_file": numbered_files[0][1].name,
        "last_step_file": numbered_files[-1][1].name,
        "missing_step_indices": [],
    }


def replay(task_name: str, seed: int) -> dict:
    key = (task_name, seed)
    expected = EXPECTED[key]
    directory = output_directory(task_name, seed)
    if directory.exists():
        raise FileExistsError(
            f"refusing to alter existing replay directory: {directory}"
        )

    config = experiment_config(task_name, 10_000, seed)
    config.update({
        "save_outputs": True,
        "save_diagnostics": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
        "rl_output_directory": directory,
    })
    result = run_rl(config)

    observed = {
        "episode": result["first_strict_success_episode"],
        "episode_steps": result["first_strict_success_episode_steps"],
        "mechanical_steps": result["first_strict_success_mechanical_steps"],
        "updates": result["optimizer_updates_before_first_success"],
    }
    if observed != expected:
        raise RuntimeError(
            f"screen replay mismatch for {task_name}, seed {seed}: "
            f"expected={expected}, observed={observed}"
        )

    report = {
        "task": task_name,
        "seed": seed,
        **observed,
        **validate_path(
            directory,
            expected["episode"],
            expected["episode_steps"],
        ),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("five_d2", "five_d3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if (args.task, args.seed) not in EXPECTED:
        parser.error("requested task/seed is not one of the three missing paths")
    replay(args.task, args.seed)


if __name__ == "__main__":
    main()
