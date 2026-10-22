"""Three-dimensional adapter for :mod:`ori_codex6`.

Only model defaults, continuous 3-D bar-collision checking, and logging-only
rigid-motion diagnostics are added here.  The ori6 reward is shared verbatim
with the planar module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex4_3d as ori4_3d
import ori_codex6 as ori6


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_ORIGINAL_ORI6_MAKE_ENVIRONMENT = ori6.make_environment
_ORIGINAL_ORI6_RUN_EPISODE = ori6.run_episode
_ORIGINAL_ORI6_AGGREGATE_EVALUATIONS = (
    ori6.aggregate_stochastic_evaluations
)
_MODEL_DEFAULT_TARGETS = {
    "4unit_torus": 2.0,
    "waterbomb": 1.0,
}

load_problem_data = ori4_3d.load_problem_data
support_compatible_rigid_basis = ori4_3d.support_compatible_rigid_basis
add_rigid_motion_diagnostics = ori4_3d.add_rigid_motion_diagnostics


def make_environment(args, node, member, fix):
    node = np.asarray(node, dtype=float)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError("ori_codex6_3d can only construct a 3-D environment.")
    environment = _ORIGINAL_ORI6_MAKE_ENVIRONMENT(
        args, node, member, fix
    )
    environment.rigid_motion_diagnostics_enabled = bool(
        getattr(args, "rigid_motion_diagnostics", False)
    )
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _ORIGINAL_ORI6_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    return add_rigid_motion_diagnostics(env, summary, nodes, actions)


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _ORIGINAL_ORI6_AGGREGATE_EVALUATIONS(
        evaluations, episode=episode
    )
    if result is None:
        return None
    result["rigid_motion_diagnostics_enabled_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_enabled", 0)
        for row in evaluations
    ]))
    result["rigid_motion_diagnostics_applicable_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_applicable", 0)
        for row in evaluations
    ]))
    aggregate_names = {
        "mean_commanded_rigid_energy_fraction": (
            "mean_commanded_rigid_energy_fraction"
        ),
        "mean_actual_rigid_energy_fraction": (
            "mean_actual_rigid_energy_fraction"
        ),
        "valid_path_rigid_energy_fraction": (
            "mean_valid_path_rigid_energy_fraction"
        ),
        "total_rigid_path_length_normalised": (
            "mean_total_rigid_path_length_normalised"
        ),
        "total_internal_path_length_normalised": (
            "mean_total_internal_path_length_normalised"
        ),
    }
    for field, aggregate_name in aggregate_names.items():
        values = [
            float(row[field]) for row in evaluations
            if row.get(field) is not None
        ]
        result[aggregate_name] = float(np.mean(values)) if values else None
    return result


def _build_metadata(args, node, member, fix, files):
    metadata = ori4_3d._build_metadata(args, node, member, fix, files)
    metadata.update({
        "adapter": "ori_codex6_3d",
        "adapter_version": "1.0.0",
        "reward_design": ori6.REWARD_DESIGN_NAME,
        "reward_schema_version": ori6.REWARD_SCHEMA_VERSION,
        "reward_parameters": ori6.reward_parameters(args),
        "path_integrated_spread_cost": False,
        "collision_quality_reference": "last_valid_before_crossing",
        "actor_critic_observation_changed_from_ori5": False,
        "frontier_history_scalar_present": False,
    })
    return metadata


@contextmanager
def patched_three_dimensional_runtime(args, member):
    """Install 3-D collision and ori6 hooks, restoring all on exit."""

    pairs = collision_3d.nonadjacent_member_pairs(member)
    collision_callback = collision_3d.make_legacy_signature_checker(
        member,
        pair_indices=pairs,
        distance_tolerance=float(args.crossing_tolerance),
    )
    original = (
        mechanics.has_proper_self_crossing,
        ori6.make_environment,
        ori6.run_episode,
        ori6.aggregate_stochastic_evaluations,
    )
    mechanics.has_proper_self_crossing = collision_callback
    ori6.make_environment = make_environment
    ori6.run_episode = run_episode
    ori6.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    try:
        yield collision_callback
    finally:
        (
            mechanics.has_proper_self_crossing,
            ori6.make_environment,
            ori6.run_episode,
            ori6.aggregate_stochastic_evaluations,
        ) = original


def train(args):
    if int(args.flatness_subspace_dimension) != 2:
        raise ValueError(
            "The 3-D adapter requires --flatness-subspace-dimension 2."
        )
    node, member, fix, files = load_problem_data(args.data_dir)
    args.data_dir = str(Path(args.data_dir).resolve())
    if not hasattr(args, "selected_model"):
        args.selected_model = Path(args.data_dir).name
    metadata = _build_metadata(args, node, member, fix, files)
    args.three_dimensional_adapter = metadata

    with patched_three_dimensional_runtime(args, member):
        output_directory, summary = ori6.train(args)

    output_directory = Path(output_directory)
    summary["three_dimensional_adapter"] = metadata
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config["three_dimensional_adapter"] = metadata
    config["effective_output_directory"] = str(output_directory.resolve())
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    adapter_parser = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False
    )
    adapter_parser.add_argument(
        "--model", choices=tuple(_MODEL_DEFAULT_TARGETS), default=None
    )
    diagnostic_group = adapter_parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--rigid-diagnostics", dest="rigid_diagnostics", action="store_true"
    )
    diagnostic_group.add_argument(
        "--no-rigid-diagnostics",
        dest="rigid_diagnostics",
        action="store_false",
    )
    adapter_parser.set_defaults(rigid_diagnostics=None)
    adapter_options, remaining = adapter_parser.parse_known_args(arguments)

    model_for_default_path = adapter_options.model or "4unit_torus"
    injected = ["--flatness-subspace-dimension", "2"]
    if not ori4_3d._option_was_supplied(remaining, "--data-dir"):
        injected.extend([
            "--data-dir",
            str(_SCRIPT_DIRECTORY / "model" / model_for_default_path),
        ])
    args = ori6.parse_args([*injected, *remaining])
    args.data_dir = str(Path(args.data_dir).resolve())
    inferred_model = Path(args.data_dir).name
    selected_model = adapter_options.model or inferred_model
    args.selected_model = selected_model

    node, member, _, _ = load_problem_data(args.data_dir)
    maximum_length = ori4_3d._maximum_initial_member_length(node, member)
    if not ori4_3d._option_was_supplied(remaining, "--max-step-width"):
        args.max_step_width = 0.1 * maximum_length
        args.step_width_rule = "0.1_times_initial_maximum_member_length"
    else:
        args.step_width_rule = "user_supplied"

    if not ori4_3d._option_was_supplied(remaining, "--target-spread"):
        args.target_spread = _MODEL_DEFAULT_TARGETS.get(selected_model, 2.0)

    if adapter_options.rigid_diagnostics is None:
        args.rigid_motion_diagnostics = selected_model == "4unit_torus"
    else:
        args.rigid_motion_diagnostics = bool(
            adapter_options.rigid_diagnostics
        )

    if not ori4_3d._option_was_supplied(remaining, "--output-dir"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        run_name = (
            f"{ori4_3d._safe_slug(selected_model)}_"
            f"spread{ori4_3d._number_slug(args.target_spread)}_"
            f"h{ori4_3d._number_slug(args.max_step_width)}_"
            f"seed{int(args.seed)}_{timestamp}_{unique_suffix}"
        )
        args.output_dir = str(Path("ori_codex6_3d_results") / run_name)
    return args


if __name__ == "__main__":
    train(parse_args())
