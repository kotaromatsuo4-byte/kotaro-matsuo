"""Evaluate every checkpoint of the slowest successful torus run."""

from __future__ import annotations

import evaluate_three_task_checkpoints as evaluator
from RL_slowest_seed_checkpoint_experiment import (
    EXPECTED,
    EXPERIMENT_DIRECTORY,
    SEEDS,
    experiment_config,
)


def configure() -> None:
    evaluator.EXPECTED = EXPECTED
    evaluator.EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY
    evaluator.SEEDS = SEEDS
    evaluator.TASKS = ("torus_d2",)
    evaluator.experiment_config = experiment_config
    evaluator.EVALUATION_DIRECTORY = (
        EXPERIMENT_DIRECTORY / "torus_checkpoint_evaluation"
    )


def main() -> None:
    configure()
    evaluator.main()


if __name__ == "__main__":
    main()
