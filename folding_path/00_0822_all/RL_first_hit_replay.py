"""Replay selected torus first-hit runs while saving their node paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RL_main import run_rl
from RL_reward_ablation_experiment import (
    experiment_config as ablation_config,
)
from RL_torus_spectral_screen import (
    experiment_config as spectral_config,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (
    BASE_DIRECTORY
    / "result"
    / "experiments"
    / "20260903_first_hit_replay_v1"
    / "torus"
)


def replay_config(alpha: str, seed: int) -> tuple[dict, Path]:
    if alpha == "0":
        if seed not in (8103, 8105):
            raise ValueError("alpha=0 replay seed must be 8103 or 8105")
        config = ablation_config("torus_d2", 10_000, seed, 0.0)
        output_directory = EXPERIMENT_DIRECTORY / "alpha_0" / f"seed{seed}"
    elif alpha == "0.5":
        if seed != 8101:
            raise ValueError("alpha=0.5 replay seed must be 8101")
        config = spectral_config("0.5", seed, 10_000)
        output_directory = EXPERIMENT_DIRECTORY / "alpha_0p5" / "seed8101"
    else:
        raise ValueError("alpha must be 0 or 0.5")

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to alter existing output: {output_directory}"
        )
    config.update({
        "save_outputs": True,
        "save_diagnostics": False,
        "stop_on_first_strict_success": True,
        "print_progress": False,
        "print_result": False,
        "rl_output_directory": output_directory,
    })
    return config, output_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", required=True, choices=("0", "0.5"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    config, output_directory = replay_config(args.alpha, args.seed)
    result = run_rl(config)
    print(json.dumps({
        "alpha": args.alpha,
        "output_directory": str(output_directory),
        "result": result,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
