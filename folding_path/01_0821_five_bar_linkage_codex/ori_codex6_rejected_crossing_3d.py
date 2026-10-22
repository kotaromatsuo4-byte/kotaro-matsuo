"""Three-dimensional adapter for rejected-crossing ori6 continuation."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex6_3d as base3d
import ori_codex6_rejected_crossing as candidate


ADAPTER_NAME = "ori_codex6_rejected_crossing_3d"
ADAPTER_VERSION = "1.0.0"

load_problem_data = base3d.load_problem_data
support_compatible_rigid_basis = base3d.support_compatible_rigid_basis
add_rigid_motion_diagnostics = base3d.add_rigid_motion_diagnostics

_BASE_CANDIDATE_MAKE_ENVIRONMENT = candidate.make_environment
_BASE_CANDIDATE_RUN_EPISODE = candidate.run_episode
_BASE_CANDIDATE_AGGREGATE = candidate.aggregate_stochastic_evaluations


def make_environment(args, node, member, fix):
    node = np.asarray(node, dtype=float)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError(f"{ADAPTER_NAME} can only construct a 3-D environment.")
    environment = _BASE_CANDIDATE_MAKE_ENVIRONMENT(
        args, node, member, fix
    )
    environment.rigid_motion_diagnostics_enabled = bool(
        getattr(args, "rigid_motion_diagnostics", False)
    )
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_CANDIDATE_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    return add_rigid_motion_diagnostics(env, summary, nodes, actions)


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_CANDIDATE_AGGREGATE(
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
    for field, aggregate_name in {
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
    }.items():
        values = [
            float(row[field]) for row in evaluations
            if row.get(field) is not None
        ]
        result[aggregate_name] = float(np.mean(values)) if values else None
    return result


def _build_metadata(args, node, member, fix, files):
    metadata = base3d._build_metadata(args, node, member, fix, files)
    parameters = candidate.reward_parameters(args)
    metadata.update({
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "ori6_candidate": candidate.CANDIDATE_NAME,
        "ori6_candidate_schema_version": (
            candidate.CANDIDATE_SCHEMA_VERSION
        ),
        "reward_parameters": parameters,
        "collision_quality_reference": parameters[
            "collision_quality_reference"
        ],
        "crossing_transition_policy": parameters[
            "crossing_transition"
        ],
        "base_ori6_source_is_unmodified": True,
    })
    return metadata


@contextmanager
def patched_three_dimensional_runtime(args, member):
    """Install continuous 3-D collision and candidate hooks locally."""

    pairs = collision_3d.nonadjacent_member_pairs(member)
    collision_callback = collision_3d.make_legacy_signature_checker(
        member,
        pair_indices=pairs,
        distance_tolerance=float(args.crossing_tolerance),
    )
    original = (
        mechanics.has_proper_self_crossing,
        candidate.make_environment,
        candidate.run_episode,
        candidate.aggregate_stochastic_evaluations,
    )
    mechanics.has_proper_self_crossing = collision_callback
    candidate.make_environment = make_environment
    candidate.run_episode = run_episode
    candidate.aggregate_stochastic_evaluations = (
        aggregate_stochastic_evaluations
    )
    try:
        yield collision_callback
    finally:
        (
            mechanics.has_proper_self_crossing,
            candidate.make_environment,
            candidate.run_episode,
            candidate.aggregate_stochastic_evaluations,
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
        output_directory, summary = candidate.train(args)

    output_directory = Path(output_directory)
    summary["three_dimensional_adapter"] = metadata
    with open(
        output_directory / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "three_dimensional_adapter": metadata,
        "effective_output_directory": str(output_directory.resolve()),
        "candidate_entrypoint": Path(__file__).name,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = candidate._candidate_parser().parse_known_args(
        arguments
    )
    args = base3d.parse_args(remaining)
    args.ori6_rejected_crossing_continuation = bool(
        selected.ori6_rejected_crossing_continuation
    )
    args.ori6_crossing_policy = candidate.CROSSING_POLICY
    if not candidate._option_was_supplied(
        arguments, "--ori5-crossing-penalty"
    ):
        args.ori5_crossing_penalty = 2.0
    if not candidate._option_was_supplied(arguments, "--output-dir"):
        base_name = Path(args.output_dir).name
        args.output_dir = str(
            Path("ori_codex6_rejected_crossing_3d_results") / base_name
        )
    return args


if __name__ == "__main__":
    train(parse_args())
