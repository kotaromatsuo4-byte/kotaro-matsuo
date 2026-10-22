"""Matched uniform-random first-discovery search for one task."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
import torch

from env import OrigamiEnv
from policy import PolicyAction, canonical_physical_direction
from RL_main import load_problem, rollout, save_node_path


class UniformRandomAgent:
    """Isotropic current-mode direction and uniform signed step width."""

    def __init__(self, max_step_width: float, rng: np.random.Generator):
        self.max_step_width = float(max_step_width)
        self.rng = rng

    def get_action(self, state, modes, deterministic=False) -> PolicyAction:
        del state, deterministic
        mode_count = int(modes.shape[1])
        coefficients = self.rng.normal(size=mode_count)
        while np.linalg.norm(coefficients) <= 1.0e-12:
            coefficients = self.rng.normal(size=mode_count)
        coefficients /= np.linalg.norm(coefficients)

        coefficient_tensor = torch.as_tensor(
            coefficients, dtype=torch.float32
        )
        mode_tensor = torch.as_tensor(modes.T, dtype=torch.float32)
        coefficient_tensor, physical_direction = (
            canonical_physical_direction(
                coefficient_tensor, mode_tensor
            )
        )
        requested_width = float(self.rng.uniform(
            -self.max_step_width, self.max_step_width
        ))
        zero = torch.zeros(())
        return PolicyAction(
            physical_direction=physical_direction.cpu().numpy(),
            mode_coefficients=coefficient_tensor.cpu().numpy(),
            requested_width=requested_width,
            log_probability=zero,
            value=zero,
            entropy=zero,
            mean_mixture_variance=1.0,
            mixture_mean_norm=0.0,
            mixture_standard_deviation_norm=float(np.sqrt(mode_count)),
            mixture_concentration=0.0,
            mode_count=mode_count,
            width_mean=0.0,
            width_variance=self.max_step_width ** 2 / 3.0,
        )


def save_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_random(config: dict) -> list[dict]:
    node, member, fix = load_problem(Path(config["model_directory"]))
    output_directory = Path(config["result_directory"]) / "random"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "random_summary.csv").unlink(missing_ok=True)
    rows = []

    for seed in config["random_seeds"]:
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        env = OrigamiEnv(config, data=(node, member, fix))
        agent = UniformRandomAgent(config["max_step_width"], rng)
        success_directory = (
            output_directory / f"seed{seed}" / "first_strict_success"
        )
        if success_directory.is_dir():
            shutil.rmtree(success_directory)
        cumulative_steps = 0
        final_summary = None
        success_episode = None

        for episode in range(1, int(config["random_max_episodes"]) + 1):
            summary, nodes = rollout(
                env, agent, training=False, deterministic=False
            )
            cumulative_steps += summary["steps"]
            final_summary = summary
            if summary["strict_success"]:
                success_episode = episode
                save_node_path(
                    success_directory,
                    nodes,
                )
                break

        rows.append({
            "seed": seed,
            "success": int(success_episode is not None),
            "first_strict_success_episode": success_episode,
            "mechanical_steps": cumulative_steps,
            "terminal_flatness": final_summary["terminal_flatness"],
            "terminal_spread": final_summary["terminal_spread"],
            "spread_error": final_summary["spread_error"],
        })
        print(
            f"random {config['task_name']} seed {seed}: "
            f"success={success_episode is not None}, "
            f"episode={success_episode}, steps={cumulative_steps}",
            flush=True,
        )

    save_summary(output_directory / "random_summary.csv", rows)
    return rows


if __name__ == "__main__":
    from main import main

    main()
