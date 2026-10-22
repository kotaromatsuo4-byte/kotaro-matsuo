"""First-discovery training, followed by frozen-checkpoint evaluation.

Change the globals below, then run main.py. Existing Python module boundaries
follow the environment / policy / agent / training structure of ori_reinfore.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


# Keep one run small and reproducible on a machine that may also be used for
# other work.  These variables must be set before NumPy/PyTorch are imported.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from RL_main import run_rl
from Random_main import run_random


BASE_DIRECTORY = Path(__file__).resolve().parent

# 実験条件はここで変更する。座標データは2Dでも (x,y,z) の3列。
TASK_NAME = "five_d2"
SEED = None  # None: task preset; or an integer such as 9101
GEOMETRY_DIMENSION = None  # None: task preset; 2: xz / best-fit line; 3: spatial / plane
SPECTRAL_WEIGHT = 0.05
MAX_EPISODES = 10_000
EVALUATION_ROLLOUTS = 100
RUN_RANDOM = False

COMMON = {
    "episodes": 10_000,
    "batch_size": 8,
    "gamma": 1.0,
    "hidden_size": 96,
    "variance_bias": -1.0986122886681098,
    "width_mean_limit": 3.0,
    "entropy_coefficient": 0.001,
    "value_coefficient": 0.5,
    "actor_gradient_clip": 0.5,
    "critic_gradient_clip": 1.0,
    "min_step_width": 1.0e-4,
    "projection_shrink": 0.5,
    "max_correction_ratio": 1.0,
    "max_actual_step_ratio": 1.25,
    "flatness_tolerance": 1.0e-3,
    "spread_tolerance": 0.05,
    "strain_tolerance": 1.0e-8,
    "crossing_tolerance": 1.0e-10,
    "flatness_clip": 2.0,
    "numerical_failure_penalty": 10.0,
    "no_mode_penalty": 5.0,
    "timeout_penalty": 1.0,
    "wrong_flat_penalty": 2.0,
    "success_bonus": 10.0,
    "quality_weight": 1.0,
    "success_speed_bonus_weight": 0.5,
    "near_miss_flatness_far": 1.0,
    "spread_frontier_bonus": 0.9,
    "spread_frontier_initial_factor": 2.0,
    "joint_flatness_thresholds": (0.5, 0.1, 0.01),
    "joint_potential_values": (0.2, 0.3, 0.5),
    "spectral_relative_epsilon": 1.0e-12,
    "moving_average_window": 200,
    "print_every": 128,
    "random_seeds": tuple(range(6101, 6106)),
    "random_max_episodes": 10_000,
}

TASKS = {
    "five_d2": {
        "model_name": "five_bar_linkage",
        "target_spread": 2.0,
        "max_steps": 400,
        "max_step_width": 0.1,
        "actor_learning_rate": 5.0e-4,
        "critic_learning_rate": 1.5e-4,
        "near_miss_spread_error_far": 1.0,
        "seed": 9101,
    },
    "five_d3": {
        "model_name": "five_bar_linkage",
        "target_spread": 3.0,
        "max_steps": 400,
        "max_step_width": 0.1,
        "actor_learning_rate": 1.0e-3,
        "critic_learning_rate": 1.5e-4,
        "near_miss_spread_error_far": 1.0,
        "seed": 9102,
    },
    "water_sqrt2": {
        "model_name": "waterbomb",
        "target_spread": math.sqrt(2.0),
        "max_steps": 400,
        "max_step_width": 0.3,
        "actor_learning_rate": 5.0e-4,
        "critic_learning_rate": 1.5e-4,
        "near_miss_spread_error_far": 1.0,
        "seed": 9102,
    },
    "torus_d2": {
        "model_name": "4unit_torus",
        "target_spread": 2.0,
        "max_steps": 200,
        "max_step_width": 0.2,
        "actor_learning_rate": 1.0e-3,
        "critic_learning_rate": 3.0e-4,
        "near_miss_spread_error_far": 1.0,
        "seed": 8102,
    },
    "water_d1": {
        "model_name": "waterbomb",
        "target_spread": 1.0,
        "max_steps": 400,
        "max_step_width": 0.3,
        "actor_learning_rate": 5.0e-4,
        "critic_learning_rate": 1.5e-4,
        "near_miss_spread_error_far": 0.12,
        "seed": 8101,
    },
}


def task_config(task_name: str, episodes: int | None = None) -> dict:
    config = dict(COMMON)
    config.update(TASKS[task_name])
    config["task_name"] = task_name
    config["model_directory"] = BASE_DIRECTORY / "model" / config["model_name"]
    config["result_directory"] = BASE_DIRECTORY / "result" / task_name
    dimension = GEOMETRY_DIMENSION
    if dimension is None:
        dimension = 2 if task_name.startswith("five_") else 3
    if dimension not in (2, 3):
        raise ValueError("GEOMETRY_DIMENSION must be 2 or 3")
    config.update({
        "geometry_dimension": dimension,
        "flatness_subspace_dimension": dimension - 1,
        "spatial_crossing": dimension == 3,
        "spectral_weight": SPECTRAL_WEIGHT,
        "actor_reference_horizon": int(config["max_steps"]),
        "evaluation_rollouts": EVALUATION_ROLLOUTS,
        "evaluation_seed_base": 750_000 + 1000 * list(TASKS).index(task_name),
        "episodes": MAX_EPISODES,
    })
    if SEED is not None:
        config["seed"] = int(SEED)
    if episodes is not None:
        config["episodes"] = int(episodes)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), default=TASK_NAME)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-random", action="store_true")
    parser.add_argument("--evaluate-only", type=Path, help="Resume evaluation of a saved RL run directory")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--random-max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluate_only is not None:
        from RL_main import evaluate_checkpoints
        evaluate_checkpoints(args.evaluate_only)
        return
    config = task_config(args.task, args.episodes)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.random_max_episodes is not None:
        config["random_max_episodes"] = int(args.random_max_episodes)

    run_rl(config)
    if (RUN_RANDOM or args.run_random) and not args.skip_random:
        run_random(config)


if __name__ == "__main__":
    main()
