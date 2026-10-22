"""Command-line entry point for the nonstationarity-aware two-stage study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd_folding.config import ExperimentConfig
from qd_folding.protocol_experiment import run_two_stage_protocol


DEFAULT_STAGE1_SEEDS = [11, 29, 47, 71, 89]
DEFAULT_STAGE2_SEEDS = [101, 131, 151, 181, 211]


def _parse_assignment(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected KEY=JSON_VALUE")
    key, raw = value.split("=", 1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 compares root-restart controls and equal-size "
            "frontier/event/history sets with random selection. Stage 2 "
            "fixes the winner among frontier/event/history and compares "
            "random, lifetime UCB, and archive-context SM-UCB."
        )
    )
    parser.add_argument("--stage", choices=["all", "1", "2"], default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--candidate-capacity", type=int)
    parser.add_argument("--epoch-rollouts", type=int)
    parser.add_argument(
        "--stage1-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_STAGE1_SEEDS,
    )
    parser.add_argument(
        "--stage2-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_STAGE2_SEEDS,
    )
    parser.add_argument(
        "--winner-pool",
        choices=["frontier", "event", "history"],
        help="Required only for stage 2 when no saved stage-1 winner exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/nonstationary_protocol"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        type=_parse_assignment,
        metavar="KEY=VALUE",
        help="Override any ExperimentConfig field; VALUE may be JSON.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> ExperimentConfig:
    values: dict = {}
    if arguments.config is not None:
        values.update(
            json.loads(arguments.config.read_text(encoding="utf-8"))
        )
    if arguments.budget is not None:
        values["search_step_budget"] = arguments.budget
    if arguments.candidate_capacity is not None:
        values["candidate_pool_capacity"] = arguments.candidate_capacity
    if arguments.epoch_rollouts is not None:
        values["candidate_epoch_rollouts"] = arguments.epoch_rollouts
    for key, value in arguments.set:
        values[key] = value
    if arguments.quick:
        values.setdefault("search_step_budget", 80)
        values.setdefault("max_path_steps", 24)
        values.setdefault("local_cma_generations", 4)
        values.setdefault("local_cma_population", 4)
        values.setdefault("candidate_pool_capacity", 8)
        values.setdefault("candidate_epoch_rollouts", 3)
        values.setdefault("save_all_histories", False)
    return ExperimentConfig.from_dict(values)


def main() -> None:
    arguments = parse_arguments()
    config = build_config(arguments)
    project_root = Path(__file__).resolve().parent
    output = arguments.output
    if not output.is_absolute():
        output = project_root / output
    stages = (1, 2) if arguments.stage == "all" else (int(arguments.stage),)
    stage1_seeds = (
        [11] if arguments.quick else list(arguments.stage1_seeds)
    )
    stage2_seeds = (
        [101] if arguments.quick else list(arguments.stage2_seeds)
    )
    results = run_two_stage_protocol(
        config=config,
        stage1_seeds=stage1_seeds,
        stage2_seeds=stage2_seeds,
        project_root=project_root,
        output=output,
        stages=stages,
        winner_pool=arguments.winner_pool,
        save_outputs=True,
        clean_output=arguments.clean_output,
    )
    if "stage1" in results:
        winner = results["stage1"]["winner"]["pool_kind"]
        print(f"Stage-1 winner: {winner}")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
