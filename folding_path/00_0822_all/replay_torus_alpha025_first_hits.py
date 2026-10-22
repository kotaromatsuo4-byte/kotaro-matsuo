"""Replay the successful alpha=0.25 torus screening runs with paths saved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RL_torus_spectral_screen import experiment_config
from RL_main import load_problem, run_rl
from env import OrigamiEnv
import numpy as np


BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260903_first_hit_replay_v1"
    / "torus"
    / "alpha_0p25"
)
ALPHA = "0.25"
MAX_EPISODES = 10_000
EXPECTED = {
    8101: {
        "episode": 2619,
        "episode_steps": 196,
        "mechanical_steps": 523_796,
        "updates": 327,
    },
    8102: {
        "episode": 3105,
        "episode_steps": 199,
        "mechanical_steps": 620_999,
        "updates": 388,
    },
    8103: {
        "episode": 853,
        "episode_steps": 199,
        "mechanical_steps": 170_599,
        "updates": 106,
    },
    8104: {
        "episode": 1322,
        "episode_steps": 198,
        "mechanical_steps": 264_398,
        "updates": 165,
    },
}


def numeric_step(path: Path) -> int:
    return int(path.stem.removeprefix("step"))


def verify(seed: int, result: dict) -> Path:
    expected = EXPECTED[seed]
    actual = {
        "episode": result["first_strict_success_episode"],
        "episode_steps": result["first_strict_success_episode_steps"],
        "mechanical_steps": result["first_strict_success_mechanical_steps"],
        "updates": result["optimizer_updates_before_first_success"],
    }
    if actual != expected:
        raise RuntimeError(
            f"seed {seed} replay mismatch: expected {expected}, got {actual}"
        )

    hit_directory = (
        OUTPUT_ROOT
        / f"seed{seed}"
        / f"best_episode{expected['episode']:05d}"
    )
    paths = sorted(hit_directory.glob("step*.dat"), key=numeric_step)
    expected_steps = list(range(expected["episode_steps"] + 1))
    actual_steps = [numeric_step(path) for path in paths]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"seed {seed} DAT sequence mismatch: expected step0.."
            f"step{expected['episode_steps']}, got {len(paths)} files"
        )
    return hit_directory


def audit_saved(seed: int) -> dict:
    expected = EXPECTED[seed]
    hit_directory = (
        OUTPUT_ROOT
        / f"seed{seed}"
        / f"best_episode{expected['episode']:05d}"
    )
    paths = sorted(hit_directory.glob("step*.dat"), key=numeric_step)
    indices = [numeric_step(path) for path in paths]
    expected_indices = list(range(expected["episode_steps"] + 1))
    if indices != expected_indices:
        raise RuntimeError(
            f"seed {seed} DAT sequence mismatch: expected step0.."
            f"step{expected['episode_steps']}, got {len(paths)} files"
        )

    config = experiment_config(ALPHA, seed, MAX_EPISODES)
    initial_node, member, fix = load_problem(Path(config["model_directory"]))
    env = OrigamiEnv(config, data=(initial_node, member, fix))
    final_node = np.atleast_2d(np.loadtxt(paths[-1]))
    metrics = env.compactness_metrics(final_node)
    strict = bool(
        metrics["flatness_ratio"] <= env.evaluation_flatness_tolerance
        and metrics["spread_error"] <= env.spread_tolerance
    )
    if final_node.shape != initial_node.shape or not strict:
        raise RuntimeError(
            f"seed {seed} saved terminal state is not a strict success: "
            f"shape={final_node.shape}, metrics={metrics}"
        )
    return {
        "seed": seed,
        "episode": expected["episode"],
        "episode_steps": expected["episode_steps"],
        "mechanical_steps": expected["mechanical_steps"],
        "updates": expected["updates"],
        "hit_directory": str(hit_directory.resolve()),
        "dat_count": len(paths),
        "first_step": indices[0],
        "last_step": indices[-1],
        "terminal_flatness": metrics["flatness_ratio"],
        "terminal_spread": metrics["spread"],
        "terminal_spread_error": metrics["spread_error"],
        "strict": strict,
    }


def replay(seed: int) -> None:
    output_directory = OUTPUT_ROOT / f"seed{seed}"
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to replace existing output: {output_directory}"
        )

    config = experiment_config(ALPHA, seed, MAX_EPISODES)
    config.update({
        "save_outputs": True,
        "save_diagnostics": False,
        "print_progress": False,
        "print_result": False,
        "rl_output_directory": output_directory,
    })
    result = run_rl(config)
    hit_directory = verify(seed, result)
    expected = EXPECTED[seed]
    print(
        f"verified seed={seed} episode={expected['episode']} "
        f"episode_steps={expected['episode_steps']} "
        f"mechanical_steps={expected['mechanical_steps']} "
        f"updates={expected['updates']} path={hit_directory.resolve()}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=tuple(EXPECTED))
    group.add_argument("--audit-all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.audit_all:
        for replay_seed in EXPECTED:
            print(json.dumps(audit_saved(replay_seed)), flush=True)
    else:
        replay(arguments.seed)
